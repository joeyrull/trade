const express = require('express');
const multer  = require('multer');
const path    = require('path');
const fs      = require('fs');
const { spawn } = require('child_process');
const { v4: uuidv4 } = require('uuid');

const router = express.Router();

const UPLOAD_DIR = path.join(__dirname, '..', 'uploads');
const JOBS_DIR   = path.join(__dirname, '..', 'jobs');
const SCRIPT     = path.join(__dirname, '..', 'scripts', 'analyze_pitcher.py');
const PYTHON     = process.env.PYTHON_BIN || 'python3';
const LIMIT_MB   = parseInt(process.env.UPLOAD_LIMIT_MB || '500', 10);

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

// Sync the cameras' timelines using each clip's detected release frame as the
// shared event. offsetsSeconds[i] is how much later (positive) or earlier
// (negative) camera i's clock runs relative to camera 0's.
function computeSync(job) {
  if (job.cameras.length < 2) return null;
  try {
    const summaries = job.cameras.map(c => JSON.parse(fs.readFileSync(c.metricsPath, 'utf8')).summary);
    const releaseTime = s => s.key_frames.release / s.fps;
    const ref = releaseTime(summaries[0]);
    return {
      referenceCamera: 0,
      offsetsSeconds: summaries.map(s => Math.round((ref - releaseTime(s)) * 10000) / 10000),
    };
  } catch (_) {
    return null;
  }
}

function runJob(job) {
  job.status = 'running';
  runCamera(job, 0);
}

function runCamera(job, camIdx) {
  const cam = job.cameras[camIdx];
  fs.mkdirSync(cam.outputDir, { recursive: true });

  const args = [
    SCRIPT,
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
      job.status = 'done';
      job.progress = { stage: 'done', pct: 100 };
      job.sync = computeSync(job);
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
