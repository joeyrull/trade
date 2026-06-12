const express = require('express');
const multer  = require('multer');
const path    = require('path');
const fs      = require('fs');
const { spawn } = require('child_process');
const { v4: uuidv4 } = require('uuid');

const router = express.Router();

const UPLOAD_DIR  = path.join(__dirname, '..', 'uploads');
const JOBS_DIR    = path.join(__dirname, '..', 'jobs');
// Analysis engine is selectable: the default MediaPipe analyzer, or the
// PitchCap bridge (stronger 3D motion capture, emits the same schema). Both
// are spawned with the identical CLI/stdout contract, so this is a one-line
// swap. The motion-fps --rebuild step always uses analyze_pitcher.py, which
// operates purely on the schema-compatible series.json either engine writes.
const ENGINE         = (process.env.PITCHER_ENGINE || 'pitchcap').toLowerCase();
const ANALYZE_SCRIPT = path.join(__dirname, '..', 'scripts',
  ENGINE === 'pitchcap' ? 'pitchcap_analyze.py' : 'analyze_pitcher.py');
const REBUILD_SCRIPT = path.join(__dirname, '..', 'scripts', 'analyze_pitcher.py');
const FUSE_SCRIPT = path.join(__dirname, '..', 'scripts', 'fuse_cameras.py');
const PYTHON      = process.env.PYTHON_BIN || 'python3';
const LIMIT_MB    = parseInt(process.env.UPLOAD_LIMIT_MB || '500', 10);

const CAMERA_ANGLES = ['side', 'front', 'behind', 'three_quarter', 'other'];

[UPLOAD_DIR, JOBS_DIR].forEach(d => fs.mkdirSync(d, { recursive: true }));

const storage = multer.diskStorage({
  destination: UPLOAD_DIR,
  filename: (req, file, cb) => cb(null, `${uuidv4()}${path.extname(file.originalname)}`),
});
const upload = multer({
  storage,
  limits: { fileSize: LIMIT_MB * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const ok = /\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(file.originalname);
    cb(ok ? null : new Error('Video files only'), ok);
  },
});

// In-memory job store (survives server lifetime)
const jobs = {};

function createJob(id, cameraInputs, throwHand) {
  jobs[id] = {
    id, status: 'queued', throwHand,
    cameras: cameraInputs.map((c, i) => ({
      index: i,
      angle: c.angle,
      uploadPath: c.uploadPath,
      outputDir: path.join(JOBS_DIR, id, `cam${i}`),
      progress: { stage: 'queued', pct: 0 },
      annotatedVideo: null,
      metricsPath: null,
      metricsUrl: null,
      metricsFusedPath: null,
      metricsFusedUrl: null,
      done: false,
    })),
    progress: { stage: 'queued', pct: 0 },
    sync: null,
    createdAt: Date.now(),
    error: null,
  };
  return jobs[id];
}

// Combine each camera's individual progress into one overall progress object,
// since they're processed one after another.
function updateOverallProgress(job) {
  const total = job.cameras.length;
  const sum = job.cameras.reduce((s, c) => s + (c.done ? 100 : (c.progress.pct || 0)), 0);
  const activeIdx = job.cameras.findIndex(c => !c.done);
  const active = job.cameras[activeIdx === -1 ? total - 1 : activeIdx];
  job.progress = {
    stage: active.progress.stage || 'queued',
    pct: Math.round((sum / total) * 10) / 10,
    camera: (activeIdx === -1 ? total - 1 : activeIdx) + 1,
    totalCameras: total,
  };
}

// Build a real-world-time "motion energy" signal from a metrics.json's
// frames — combined hip/chest/arm rotation speed, sampled at
// frame/motion_fps. Used to cross-correlate the two cameras' clocks; this
// combination stays large and sharply peaked around acceleration/release
// from any camera angle, unlike a single rotation axis which can be
// foreshortened depending on the camera's viewing direction.
function motionSignal(data) {
  const motionFps = data.summary.motion_fps;
  return data.frames.map(f => ({
    t: f.frame / motionFps,
    v: Math.abs(f.metrics.hip_rotation_speed) +
       Math.abs(f.metrics.chest_rotation_speed) +
       Math.abs(f.metrics.arm_speed),
  }));
}

// Linear interpolation of a {t, v}[] series (sorted by t) at time `t`.
// Returns null outside the series' range.
function interpAt(signal, t) {
  const first = signal[0], last = signal[signal.length - 1];
  if (t < first.t || t > last.t) return null;
  let lo = 0, hi = signal.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (signal[mid].t <= t) lo = mid; else hi = mid;
  }
  const a = signal[lo], b = signal[hi];
  return b.t === a.t ? a.v : a.v + (t - a.t) / (b.t - a.t) * (b.v - a.v);
}

function pearson(a, b) {
  const n = a.length;
  const meanA = a.reduce((s, x) => s + x, 0) / n;
  const meanB = b.reduce((s, x) => s + x, 0) / n;
  let num = 0, denA = 0, denB = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - meanA, db = b[i] - meanB;
    num += da * db; denA += da * da; denB += db * db;
  }
  return (denA === 0 || denB === 0) ? 0 : num / Math.sqrt(denA * denB);
}

// Refine a release-frame-based offset estimate — a single-instant guess that
// can be thrown off by a mis-detected release frame in either camera — by
// cross-correlating the two cameras' motion-energy signals over real-world
// time. `offset` follows computeSync's convention: camera-0 time t0
// corresponds to camera-1 time t0 - offset. Searches lags within the search
// window of the initial estimate and returns the best-correlated lag, or
// null if nothing in that window correlates well enough to trust.
const SYNC_SEARCH_WINDOW_S = 2;
const SYNC_MIN_CORRELATION = 0.5;
const SYNC_GRID_POINTS     = 200;

function crossCorrelateOffset(sig0, sig1, initialOffset) {
  const dur0 = sig0[sig0.length - 1].t;
  const dur1 = sig1[sig1.length - 1].t;
  const window  = Math.min(SYNC_SEARCH_WINDOW_S, 0.5 * Math.min(dur0, dur1));
  const step    = Math.min(dur0, dur1) / SYNC_GRID_POINTS;
  const minSpan = 0.5 * Math.min(dur0, dur1);
  if (step <= 0) return null;

  let best = null;
  for (let lag = initialOffset - window; lag <= initialOffset + window; lag += step) {
    const lo = Math.max(0, lag);
    const hi = Math.min(dur0, lag + dur1);
    if (hi - lo < minSpan) continue;

    const a = [], b = [];
    for (let t = lo; t <= hi; t += step) {
      a.push(interpAt(sig0, t));
      b.push(interpAt(sig1, t - lag));
    }
    const corr = pearson(a, b);
    if (!best || corr > best.corr) best = { lag, corr };
  }
  if (!best || best.corr < SYNC_MIN_CORRELATION) return null;
  return { offset: Math.round(best.lag * 10000) / 10000, correlation: Math.round(best.corr * 1000) / 1000 };
}

// Absolute angular difference between two angles, wrapped to [0, 180] —
// used as a scale-independent "amount of rotation" proxy that doesn't care
// which direction a joint angle was wound or where it's centered.
function angDiff(a, b) {
  const d = ((a - b + 180) % 360 + 360) % 360 - 180;
  return Math.abs(d);
}

// Find the lag that maximizes the Pearson correlation between two {t, v}[]
// signals of durations dur0/dur1, requiring the overlapping span to cover at
// least 30% of the shorter signal so short, accidental overlaps near the
// edges don't win on noise alone. Returns null if no lag has a usable span.
const LAG_SEARCH_GRID = 80;

function bestLag(sig0, sig1, dur0, dur1) {
  const grid = LAG_SEARCH_GRID;
  const lagMin = -dur1, lagMax = dur0;
  const lagStep = (lagMax - lagMin) / grid;
  const minSpan = 0.3 * Math.min(dur0, dur1);
  let best = null;
  for (let lag = lagMin; lag <= lagMax; lag += lagStep) {
    const lo = Math.max(0, lag), hi = Math.min(dur0, lag + dur1);
    if (hi - lo < minSpan) continue;
    const step = (hi - lo) / grid;
    if (step <= 0) continue;
    const a = [], b = [];
    for (let t = lo; t <= hi; t += step) {
      a.push(interpAt(sig0, t));
      b.push(interpAt(sig1, t - lag));
    }
    const corr = pearson(a, b);
    if (!best || corr > best.corr) best = { lag, corr };
  }
  return best;
}

// When one camera's motion_fps is ambiguous (couldn't be confidently derived
// from container metadata, see analyze_pitcher.py's detect_frame_rates),
// search candidate motion_fps values for it and find the one whose
// motion-energy signal best cross-correlates with the confident camera's.
// Candidates are built from frame-to-frame joint-angle deltas — a
// scale-independent "amount of motion" proxy — multiplied by the candidate
// rate to turn it into a speed signal comparable to motionSignal()'s.
const SCALE_SEARCH = { kMin: 1, kMax: 12, kStep: 0.05, minCorr: 0.5 };

function findMotionFpsCorrection(refData, ambData) {
  const sigRef = motionSignal(refData);
  const durRef = sigRef[sigRef.length - 1].t;

  const playbackFpsAmb = ambData.summary.fps;
  const nAmb = ambData.frames.length;
  const rawDelta = [0];
  for (let i = 1; i < nAmb; i++) {
    const f0 = ambData.frames[i - 1].metrics, f1 = ambData.frames[i].metrics;
    rawDelta.push(
      angDiff(f1.hip_rotation, f0.hip_rotation) +
      angDiff(f1.shoulder_rotation, f0.shoulder_rotation) +
      Math.abs(f1.elbow_angle - f0.elbow_angle)
    );
  }

  let best = null;
  for (let k = SCALE_SEARCH.kMin; k <= SCALE_SEARCH.kMax; k += SCALE_SEARCH.kStep) {
    const motionFps = playbackFpsAmb * k;
    const sigCand = rawDelta.map((d, i) => ({ t: i / motionFps, v: d * motionFps }));
    const durCand = sigCand[sigCand.length - 1].t;
    const result = bestLag(sigRef, sigCand, durRef, durCand);
    if (result && (!best || result.corr > best.corr)) {
      best = { k, motionFps, lag: result.lag, corr: result.corr };
    }
  }
  if (!best || best.corr < SCALE_SEARCH.minCorr) return null;
  return best;
}

// Sync the cameras' timelines onto a shared real-world clock (frame index /
// motion_fps, since slow-motion clips advance their frame index much faster
// than real time — mixing in playback fps would silently misalign any pair
// of clips with different slow-mo factors). A first estimate comes from each
// clip's detected release frame; that estimate is then refined by
// cross-correlating a motion-energy signal between the two clips, which is
// far more robust than relying on a single detected frame in each camera.
// offsetsSeconds[i] is how much later (positive) or earlier (negative)
// camera i's clock runs relative to camera 0's, in real-world seconds:
// t_camera_0 = t_camera_i + offsetsSeconds[i].
function computeSync(job) {
  if (job.cameras.length < 2) return null;
  try {
    const data = job.cameras.map(c => JSON.parse(fs.readFileSync(c.metricsPath, 'utf8')));
    const releaseTime = d => d.summary.key_frames.release / d.summary.motion_fps;
    const ref = releaseTime(data[0]);
    const sig0 = motionSignal(data[0]);

    const offsetsSeconds = [0];
    const methods = ['reference'];
    const correlations = [null];

    for (let i = 1; i < data.length; i++) {
      const initial = ref - releaseTime(data[i]);
      const refined = crossCorrelateOffset(sig0, motionSignal(data[i]), initial);
      offsetsSeconds.push(refined ? refined.offset : Math.round(initial * 10000) / 10000);
      methods.push(refined ? 'cross-correlation' : 'release-frame');
      correlations.push(refined ? refined.correlation : null);
    }

    return { referenceCamera: 0, offsetsSeconds, methods, correlations };
  } catch (_) {
    return null;
  }
}

// After both cameras finish and a sync offset is available, fill each
// camera's low-confidence frames using the other camera's data at the
// corresponding instant (see fuse_cameras.py). Runs both directions in
// parallel; failures are non-fatal — the per-camera metrics already
// produced by runCamera() remain the result either way.
function runFusion(job) {
  if (job.cameras.length !== 2 || !job.sync) return;
  const [cam0, cam1] = job.cameras;
  const offsets = job.sync.offsetsSeconds;

  const pairs = [
    { self: cam0, other: cam1, offset: offsets[1] - offsets[0] },
    { self: cam1, other: cam0, offset: offsets[0] - offsets[1] },
  ];

  for (const { self, other, offset } of pairs) {
    const outPath = path.join(self.outputDir, 'metrics_fused.json');
    const args = [FUSE_SCRIPT, self.metricsPath, other.metricsPath, String(offset), outPath];
    const proc = spawn(PYTHON, args);

    let stderr = '';
    proc.stderr.on('data', (d) => { stderr += d.toString(); });
    proc.on('close', (code) => {
      if (code === 0 && fs.existsSync(outPath)) {
        self.metricsFusedPath = outPath;
        self.metricsFusedUrl  = `/jobs/${job.id}/${path.basename(self.outputDir)}/metrics_fused.json`;
      } else {
        console.warn(`[fusion ${job.id} ${path.basename(self.outputDir)}] failed: ${stderr.trim()}`);
      }
    });
  }
}

// Minimum scale-correction factor worth acting on. A correction this large
// (e.g. k≈4.83 turning a misdetected 49.66fps into ~240fps) only fires for
// genuinely mis-detected slow-mo rates, not normal cross-correlation noise
// around k=1.
const MOTION_FPS_CORRECTION_MIN_K = 1.15;

// Called once all cameras have finished analysis. If exactly one camera's
// motion_fps was ambiguous (see detect_frame_rates in analyze_pitcher.py) and
// the other was confidently detected, search for a better motion_fps for the
// ambiguous camera by cross-correlating against the confident camera, and if
// a clearly-better rate is found, rebuild that camera's metrics with it
// before computing sync/fusion. This runs the rebuild as a subprocess using
// the series.json sidecar saved during analysis, so pose estimation and video
// annotation don't need to be redone.
function finalizeJob(job) {
  job.progress = { stage: 'syncing', pct: 100 };

  const finish = () => {
    job.status = 'done';
    job.progress = { stage: 'done', pct: 100 };
    job.sync = computeSync(job);
    runFusion(job);
  };

  if (job.cameras.length !== 2) return finish();

  let data;
  try {
    data = job.cameras.map(c => JSON.parse(fs.readFileSync(c.metricsPath, 'utf8')));
  } catch (_) {
    return finish();
  }

  const confident = data.map(d => d.summary.motion_fps_confident !== false);
  let refIdx = -1, ambIdx = -1;
  if (confident[0] && !confident[1]) { refIdx = 0; ambIdx = 1; }
  else if (confident[1] && !confident[0]) { refIdx = 1; ambIdx = 0; }
  if (ambIdx === -1) return finish();

  const correction = findMotionFpsCorrection(data[refIdx], data[ambIdx]);
  if (!correction || correction.k < MOTION_FPS_CORRECTION_MIN_K) return finish();

  const cam = job.cameras[ambIdx];
  const seriesPath = path.join(cam.outputDir, 'series.json');
  if (!fs.existsSync(seriesPath)) return finish();

  const args = [
    REBUILD_SCRIPT, '--rebuild', seriesPath,
    '--metrics', cam.metricsPath,
    '--motion-fps', String(correction.motionFps),
    '--output', cam.metricsPath,
  ];
  const proc = spawn(PYTHON, args);
  let stderr = '';
  proc.stderr.on('data', (d) => { stderr += d.toString(); });
  proc.on('close', (code) => {
    if (code !== 0) {
      console.warn(`[motion-fps correction ${job.id} cam${ambIdx}] rebuild failed: ${stderr.trim()}`);
    } else {
      console.log(`[motion-fps correction ${job.id} cam${ambIdx}] corrected ${data[ambIdx].summary.motion_fps} -> ${correction.motionFps.toFixed(2)} (corr=${correction.corr.toFixed(2)})`);
    }
    finish();
  });
}

function runJob(job) {
  job.status = 'running';
  runCamera(job, 0);
}

function runCamera(job, camIdx) {
  const cam = job.cameras[camIdx];
  fs.mkdirSync(cam.outputDir, { recursive: true });

  const args = [
    ANALYZE_SCRIPT,
    cam.uploadPath,
    '--output-dir', cam.outputDir,
    '--throw-hand', job.throwHand,
    '--progress',
  ];

  const proc = spawn(PYTHON, args);

  proc.stdout.on('data', (data) => {
    const lines = data.toString().trim().split('\n');
    for (const line of lines) {
      try {
        const msg = JSON.parse(line.trim());
        if (msg.status === 'done') {
          cam.annotatedVideo = `/jobs/${job.id}/cam${camIdx}/annotated.mp4`;
          cam.metricsPath    = path.join(cam.outputDir, 'metrics.json');
          cam.metricsUrl     = `/jobs/${job.id}/cam${camIdx}/metrics.json`;
          cam.progress = { stage: 'done', pct: 100 };
          cam.done = true;
        } else if (msg.stage) {
          cam.progress = msg;
        }
      } catch (_) { /* non-JSON log line */ }
    }
    updateOverallProgress(job);
  });

  proc.stderr.on('data', (data) => {
    const text = data.toString();
    if (text.includes('[warn]')) {
      console.warn(`[analysis ${job.id} cam${camIdx}] ${text.trim()}`);
    }
    if (text.includes('Error') || text.includes('Traceback') ||
        text.includes('Check failed') || /^F\d{8}/m.test(text)) {
      job.error = (job.error || '') + `[camera ${camIdx + 1}] ` + text;
    }
  });

  proc.on('close', (code, signal) => {
    if (code !== 0 && !cam.done) {
      job.status = 'error';
      job.error  = job.error || `Camera ${camIdx + 1}: process exited with code ${code}` +
        (signal ? ` (signal ${signal})` : '');
      return;
    }

    if (camIdx + 1 < job.cameras.length) {
      runCamera(job, camIdx + 1);
    } else {
      finalizeJob(job);
    }
  });
}

// POST /api/analysis/upload  — upload video(s) and start analysis
router.post('/upload', upload.fields([
  { name: 'video',  maxCount: 1 },
  { name: 'video2', maxCount: 1 },
]), (req, res) => {
  const videoFile = req.files?.video?.[0];
  if (!videoFile) return res.status(400).json({ error: 'No video file uploaded' });

  const throwHand = (req.body.throwHand || 'left').toLowerCase();
  if (!['left', 'right'].includes(throwHand)) {
    return res.status(400).json({ error: 'throwHand must be "left" or "right"' });
  }

  const angle = (req.body.angle || 'side').toLowerCase();
  if (!CAMERA_ANGLES.includes(angle)) {
    return res.status(400).json({ error: 'Invalid camera angle' });
  }

  const cameraInputs = [{ uploadPath: videoFile.path, angle }];

  const video2File = req.files?.video2?.[0];
  if (video2File) {
    const angle2 = (req.body.angle2 || 'front').toLowerCase();
    if (!CAMERA_ANGLES.includes(angle2)) {
      return res.status(400).json({ error: 'Invalid second camera angle' });
    }
    cameraInputs.push({ uploadPath: video2File.path, angle: angle2 });
  }

  const id  = uuidv4();
  const job = createJob(id, cameraInputs, throwHand);
  runJob(job);

  res.json({
    jobId:     id,
    status:    'running',
    throwHand,
    cameras:   cameraInputs.map(c => ({ angle: c.angle })),
    filename:  videoFile.originalname,
  });
});

// GET /api/analysis/status/:id — poll for progress
router.get('/status/:id', (req, res) => {
  const job = jobs[req.params.id];
  if (!job) return res.status(404).json({ error: 'Job not found' });

  res.json({
    id:        job.id,
    status:    job.status,
    throwHand: job.throwHand,
    progress:  job.progress,
    cameras:   job.cameras.map(c => ({
      angle: c.angle,
      annotatedVideo: c.annotatedVideo,
      metricsUrl: c.metricsUrl,
      metricsFusedUrl: c.metricsFusedUrl,
    })),
    sync:      job.sync,
    error:     job.error,
  });
});

// GET /api/analysis/result/:id — full result with inline metrics JSON per camera
router.get('/result/:id', async (req, res) => {
  const job = jobs[req.params.id];
  if (!job) return res.status(404).json({ error: 'Job not found' });
  if (job.status !== 'done') return res.status(202).json({ status: job.status });

  try {
    const cameras = job.cameras.map(c => ({
      angle: c.angle,
      annotatedVideo: c.annotatedVideo,
      metrics: JSON.parse(fs.readFileSync(c.metricsPath, 'utf8')),
      metricsFused: c.metricsFusedPath ? JSON.parse(fs.readFileSync(c.metricsFusedPath, 'utf8')) : null,
    }));
    res.json({ status: 'done', cameras, sync: job.sync });
  } catch (e) {
    res.status(500).json({ error: 'Could not read metrics: ' + e.message });
  }
});

// GET /api/analysis/jobs — list all jobs (for debugging)
router.get('/jobs', (req, res) => {
  const list = Object.values(jobs).map(j => ({
    id: j.id, status: j.status, throwHand: j.throwHand,
    cameras: j.cameras.map(c => c.angle),
    createdAt: j.createdAt, progress: j.progress,
  }));
  res.json(list.sort((a, b) => b.createdAt - a.createdAt));
});

module.exports = router;
