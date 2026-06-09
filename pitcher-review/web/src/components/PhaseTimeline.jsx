import { useMemo } from 'react';

const PHASE_COLORS = {
  setup: '#555',
  windup: '#c8a800',
  stride: '#0088bb',
  foot_strike: '#00aa55',
  arm_cocking: '#cc5000',
  acceleration: '#cc1500',
  release: '#9900cc',
  follow_through: '#5555bb',
};

const PHASE_SHORT = {
  setup: 'SETUP',
  windup: 'WIND',
  stride: 'STRIDE',
  foot_strike: 'FS',
  arm_cocking: 'COCK',
  acceleration: 'ACCEL',
  release: 'REL',
  follow_through: 'FLW',
};

export default function PhaseTimeline({ summary, frames, currentFrame, onSeek }) {
  const totalFrames = summary.total_frames || frames.length;
  const phases = summary.phases || {};

  const segments = useMemo(() => {
    return Object.entries(phases)
      .filter(([, { start, end }]) => end > start)
      .map(([name, { start, end }]) => ({
        name,
        start,
        end,
        width: ((end - start) / totalFrames) * 100,
        left:  (start / totalFrames) * 100,
      }))
      .sort((a, b) => a.start - b.start);
  }, [phases, totalFrames]);

  const playheadPct = (currentFrame / totalFrames) * 100;

  return (
    <div className="phase-timeline">
      <div
        className="timeline-track"
        onClick={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const pct = (e.clientX - rect.left) / rect.width;
          onSeek(Math.round(pct * totalFrames));
        }}
      >
        {segments.map(seg => (
          <div
            key={seg.name}
            className="timeline-segment"
            style={{
              left: `${seg.left}%`,
              width: `${seg.width}%`,
              background: PHASE_COLORS[seg.name] || '#666',
            }}
            title={`${seg.name.replace(/_/g, ' ')} — frames ${seg.start}–${seg.end}`}
          >
            {seg.width > 4 && (
              <span className="segment-label">{PHASE_SHORT[seg.name] || seg.name}</span>
            )}
          </div>
        ))}
        {/* Playhead */}
        <div
          className="timeline-playhead"
          style={{ left: `${playheadPct}%` }}
        />
      </div>

      {/* Legend */}
      <div className="timeline-legend">
        {segments.map(seg => (
          <button
            key={seg.name}
            className="legend-item"
            onClick={() => onSeek(seg.start)}
            title={`Jump to ${seg.name}`}
          >
            <span className="legend-dot" style={{ background: PHASE_COLORS[seg.name] }} />
            <span>{seg.name.replace(/_/g, ' ')}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
