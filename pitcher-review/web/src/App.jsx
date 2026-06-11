import { useState, useCallback, useEffect, useRef } from 'react';
import VideoUploader from './components/VideoUploader';
import AnalysisViewer from './components/AnalysisViewer';
import ProgressBar from './components/ProgressBar';

const POLL_MS = 1200;

export default function App() {
  const [job, setJob] = useState(null);       // { jobId, status, progress, ... }
  const [result, setResult] = useState(null); // { annotatedVideo, metrics }
  const pollRef = useRef(null);

  const stopPolling = () => {
    if (pollRef.current) clearInterval(pollRef.current);
  };

  const poll = useCallback(async (jobId) => {
    try {
      const r = await fetch(`/api/analysis/status/${jobId}`);
      const data = await r.json();
      setJob(data);

      if (data.status === 'done') {
        stopPolling();
        const res = await fetch(`/api/analysis/result/${jobId}`);
        const full = await res.json();
        setResult(full);
      } else if (data.status === 'error') {
        stopPolling();
      }
    } catch (_) { /* network hiccup — keep polling */ }
  }, []);

  const handleUpload = useCallback(async ({ file, throwHand }) => {
    setJob(null);
    setResult(null);
    stopPolling();

    const form = new FormData();
    form.append('video', file);
    form.append('throwHand', throwHand);

    const r = await fetch('/api/analysis/upload', { method: 'POST', body: form });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || 'Upload failed');

    setJob({ ...data, status: 'running', progress: { stage: 'queued', pct: 0 } });
    pollRef.current = setInterval(() => poll(data.jobId), POLL_MS);
  }, [poll]);

  const handleReset = () => {
    stopPolling();
    setJob(null);
    setResult(null);
  };

  useEffect(() => stopPolling, []);

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-inner">
          <span className="logo">Pitcher Review</span>
          <span className="subtitle">Side-View Biomechanics — Torso · Chest · Arm Speed</span>
          {(job || result) && (
            <button className="btn-ghost" onClick={handleReset}>New Analysis</button>
          )}
        </div>
      </header>

      <main className="app-main">
        {!job && !result && <VideoUploader onUpload={handleUpload} />}

        {job && !result && (
          <div className="progress-screen">
            <ProgressBar job={job} />
          </div>
        )}

        {result && <AnalysisViewer result={result} />}
      </main>
    </div>
  );
}
