const STAGES = ['queued', 'extracting', 'annotating', 'done'];
const STAGE_LABELS = {
  queued:     'Queued',
  extracting: 'Extracting pose landmarks…',
  annotating: 'Rendering annotated video…',
  done:       'Complete',
};

export default function ProgressBar({ job }) {
  const { progress = {}, status, error } = job;
  const pct   = progress.pct || 0;
  const stage = progress.stage || 'queued';

  if (status === 'error') {
    return (
      <div className="progress-card error">
        <span className="progress-icon">❌</span>
        <h3>Analysis Failed</h3>
        <p className="error-msg">{error || 'Unknown error'}</p>
      </div>
    );
  }

  return (
    <div className="progress-card">
      <span className="progress-icon">⚙️</span>
      <h3>Analyzing Delivery</h3>

      <div className="stage-steps">
        {STAGES.filter(s => s !== 'queued').map((s, i) => {
          const idx  = STAGES.indexOf(stage);
          const sIdx = STAGES.indexOf(s);
          const done = sIdx < idx || status === 'done';
          const active = s === stage && status !== 'done';
          return (
            <div key={s} className={`stage-step ${done ? 'done' : ''} ${active ? 'active' : ''}`}>
              <span className="step-dot">{done ? '✓' : i + 1}</span>
              <span className="step-label">{STAGE_LABELS[s]}</span>
            </div>
          );
        })}
      </div>

      <div className="progress-bar-wrap">
        <div className="progress-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <p className="progress-pct">{Math.round(pct)}%</p>
      <p className="progress-note">
        Analysis time depends on video length and frame rate. A 5-second clip takes ~30–60 seconds.
      </p>
    </div>
  );
}
