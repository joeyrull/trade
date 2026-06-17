const SPEEDS = [0.1, 0.25, 0.5, 1];

// Frame-step + slow-motion controls for a <video> element. Lets you crawl
// through a delivery one frame at a time — handy for manually checking a
// measurement when auto-tracking is uncertain (cluttered background, fast
// motion, etc).
export default function VideoControls({ videoRef, fps }) {
  const step = (delta) => {
    const vid = videoRef.current;
    if (!vid || !fps) return;
    vid.pause();
    const frame = Math.round(vid.currentTime * fps) + delta;
    vid.currentTime = Math.max(0, frame / fps);
  };

  const setSpeed = (rate) => {
    const vid = videoRef.current;
    if (vid) vid.playbackRate = rate;
  };

  return (
    <div className="video-controls">
      <div className="video-controls-group">
        <button type="button" className="video-ctrl-btn" onClick={() => step(-1)} title="Previous frame">⏮ Frame</button>
        <button type="button" className="video-ctrl-btn" onClick={() => step(1)} title="Next frame">Frame ⏭</button>
      </div>
      <div className="video-controls-group">
        <span className="video-ctrl-label">Speed</span>
        {SPEEDS.map(s => (
          <button
            key={s}
            type="button"
            className="video-ctrl-btn"
            onClick={() => setSpeed(s)}
          >
            {s}×
          </button>
        ))}
      </div>
    </div>
  );
}
