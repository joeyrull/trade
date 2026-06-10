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

function createJob(id, uploadPath, throwHand) {
  jobs[id] = {
    id, status: 'queued', throwHand,
    uploadPath, outputDir: path.join(JOBS_DIR, id),
    progress: { stage: 'queued', pct: 0 },
    createdAt: Date.now(),
    annotatedVideo: null,
    metrics: null,
    error: null,
  };
  return jobs[id];
}

function runJob(job) {
  job.status = 'running';
  fs.mkdirSync(job.outputDir, { recursive: true });

  const args = [
    SCRIPT,
    job.uploadPath,
    '--output-dir', job.outputDir,
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
          job.status = 'done';
          job.annotatedVideo = '/jobs/' + job.id + '/annotated.mp4';
          job.metrics        = '/jobs/' + job.id + '/metrics.json';
          job.progress = { stage: 'done', pct: 100 };
        } else if (msg.stage) {
          job.progress = msg;
        }
      } catch (_) { /* non-JSON log line */ }
    }
  });

  proc.stderr.on('data', (data) => {
    const text = data.toString();
    // Surface non-fatal warnings (e.g. a failed H.264 re-encode that leaves an
    // unplayable mpeg4 file) to the server console so they aren't lost.
    if (text.includes('[warn]')) {
      console.warn(`[analysis ${job.id}] ${text.trim()}`);
    }
    // Some MediaPipe logs go to stderr - only flag actual errors, including
    // native CHECK-failure crashes (abseil "F..." fatal logs / "Check failed")
    // which don't say "Error" but precede a SIGABRT.
    if (text.includes('Error') || text.includes('Traceback') ||
        text.includes('Check failed') || /^F\d{8}/m.test(text)) {
      job.error = (job.error || '') + text;
    }
  });

  proc.on('close', (code, signal) => {
    if (code !== 0 && job.status !== 'done') {
      job.status = 'error';
      job.error  = job.error || `Process exited with code ${code}` +
        (signal ? ` (signal ${signal})` : '');
    }
  });
}

// POST /api/analysis/upload  — upload video and start analysis
router.post('/upload', upload.single('video'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No video file uploaded' });

  const throwHand = (req.body.throwHand || 'left').toLowerCase();
  if (!['left', 'right'].includes(throwHand)) {
    return res.status(400).json({ error: 'throwHand must be "left" or "right"' });
  }

  const id  = uuidv4();
  const job = createJob(id, req.file.path, throwHand);
  runJob(job);

  res.json({
    jobId:     id,
    status:    'running',
    throwHand,
    filename:  req.file.originalname,
  });
});

// GET /api/analysis/status/:id — poll for progress
router.get('/status/:id', (req, res) => {
  const job = jobs[req.params.id];
  if (!job) return res.status(404).json({ error: 'Job not found' });

  res.json({
    id:            job.id,
    status:        job.status,
    throwHand:     job.throwHand,
    progress:      job.progress,
    annotatedVideo: job.annotatedVideo,
    metricsUrl:    job.metrics,
    error:         job.error,
  });
});

// GET /api/analysis/result/:id — full result with inline metrics JSON
router.get('/result/:id', async (req, res) => {
  const job = jobs[req.params.id];
  if (!job) return res.status(404).json({ error: 'Job not found' });
  if (job.status !== 'done') return res.status(202).json({ status: job.status });

  const metricsPath = path.join(job.outputDir, 'metrics.json');
  try {
    const metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
    res.json({
      status:        'done',
      annotatedVideo: job.annotatedVideo,
      metrics,
    });
  } catch (e) {
    res.status(500).json({ error: 'Could not read metrics: ' + e.message });
  }
});

// GET /api/analysis/jobs — list all jobs (for debugging)
router.get('/jobs', (req, res) => {
  const list = Object.values(jobs).map(j => ({
    id: j.id, status: j.status, throwHand: j.throwHand,
    createdAt: j.createdAt, progress: j.progress,
  }));
  res.json(list.sort((a, b) => b.createdAt - a.createdAt));
});

module.exports = router;
