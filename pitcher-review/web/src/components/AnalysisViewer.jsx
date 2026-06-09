import { useState, useRef, useEffect, useCallback } from 'react';
import MetricsChart from './MetricsChart';
import PhaseTimeline from './PhaseTimeline';
import SummaryReport from './SummaryReport';
import PoseOverlay from './PoseOverlay';

const TABS = ['Overview', 'Charts', 'Phases', 'Annotated Video'];

export default function AnalysisViewer({ result }) {
  const { annotatedVideo, metrics } = result;
  const { summary, frames } = metrics;

  const [tab, setTab] = useState('Overview');
  const [currentFrame, setCurrentFrame] = useState(0);
  const videoRef = useRef(null);

  // Sync video time → frame index
  useEffect(() => {
    const vid = videoRef.current;
    if (!vid) return;
    const update = () => {
      const f = Math.min(
        Math.round(vid.currentTime * summary.fps),
        frames.length - 1,
      );
      setCurrentFrame(f);
    };
    vid.addEventListener('timeupdate', update);
    return () => vid.removeEventListener('timeupdate', update);
  }, [summary.fps, frames.length]);

  const seekToFrame = useCallback((frameIdx) => {
    const vid = videoRef.current;
    if (vid) {
      vid.currentTime = frameIdx / summary.fps;
    }
    setCurrentFrame(frameIdx);
  }, [summary.fps]);

  const frame = frames[currentFrame] || frames[0];

  return (
    <div className="viewer">
      {/* Sticky video + live metrics strip */}
      <div className="viewer-top">
        <div className="video-column">
          <div className="video-wrap">
            <video
              ref={videoRef}
              src={annotatedVideo}
              controls
              playsInline
              className="main-video"
            />
          </div>
          <PhaseTimeline
            summary={summary}
            frames={frames}
            currentFrame={currentFrame}
            onSeek={seekToFrame}
          />
        </div>

        <div className="live-metrics-column">
          <LiveMetrics frame={frame} summary={summary} />
        </div>
      </div>

      {/* Tab navigation */}
      <div className="tabs">
        {TABS.map(t => (
          <button
            key={t}
            className={`tab-btn ${tab === t ? 'active' : ''}`}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </div>

      <div className="tab-content">
        {tab === 'Overview' && (
          <SummaryReport summary={summary} frames={frames} />
        )}
        {tab === 'Charts' && (
          <MetricsChart frames={frames} summary={summary} currentFrame={currentFrame} onSeek={seekToFrame} />
        )}
        {tab === 'Phases' && (
          <PhaseDetail summary={summary} frames={frames} onSeek={seekToFrame} />
        )}
        {tab === 'Annotated Video' && (
          <div className="annotated-tab">
            <p className="annotated-note">
              The annotated video shows the MediaPipe pose skeleton, live metrics HUD, and phase labels overlaid on every frame.
            </p>
            <video src={annotatedVideo} controls playsInline className="annotated-video-large" />
            <a className="btn-secondary" href={annotatedVideo} download="pitcher_annotated.mp4">
              Download Annotated Video
            </a>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Live Metrics Sidebar ───────────────────────────────────────────────────
function LiveMetrics({ frame, summary }) {
  if (!frame) return null;
  const m = frame.metrics;
  const phase = frame.phase || 'setup';

  const PHASE_COLORS = {
    setup: '#888',
    windup: '#ffd700',
    stride: '#00c8ff',
    foot_strike: '#00ff88',
    arm_cocking: '#ff6400',
    acceleration: '#ff2200',
    release: '#cc00ff',
    follow_through: '#8080ff',
  };

  const metricRows = [
    { label: 'Hip Rotation',      val: m.hip_rotation,      unit: '°',   lo: -45, hi: 45 },
    { label: 'Shoulder Rotation', val: m.shoulder_rotation, unit: '°',   lo: -45, hi: 45 },
    { label: 'Hip-Shoulder Sep',  val: m.hip_shoulder_sep,  unit: '°',   good: v => v > 20 },
    { label: 'Elbow Height',      val: m.elbow_height_pct,  unit: '%',   good: v => v > 3  },
    { label: 'Elbow Angle',       val: m.elbow_angle,       unit: '°',   neutral: true      },
    { label: 'Arm Speed',         val: Math.abs(m.arm_speed), unit: '°/s', good: v => v > 500 },
    { label: 'Trunk Tilt',        val: m.trunk_tilt,        unit: '°',   neutral: true      },
  ];

  return (
    <div className="live-metrics">
      <div className="phase-badge" style={{ background: PHASE_COLORS[phase] || '#888' }}>
        {phase.replace(/_/g, ' ').toUpperCase()}
      </div>
      <p className="frame-label">Frame {frame.frame} · t={frame.time.toFixed(3)}s</p>

      {metricRows.map(({ label, val, unit, good, neutral }) => {
        const color = neutral ? '#fff'
          : good ? (good(val) ? '#4dff88' : '#ff8844')
          : '#fff';
        return (
          <div key={label} className="metric-row">
            <span className="metric-label">{label}</span>
            <span className="metric-val" style={{ color }}>
              {typeof val === 'number' ? val.toFixed(1) : '—'}{unit}
            </span>
          </div>
        );
      })}

      <div className="arm-speed-bar-wrap">
        <div
          className="arm-speed-bar"
          style={{ width: `${Math.min(100, Math.abs(m.arm_speed) / 10)}%` }}
        />
        <span className="arm-speed-label">Arm Speed</span>
      </div>
    </div>
  );
}

// ── Phase Detail Tab ───────────────────────────────────────────────────────
function PhaseDetail({ summary, frames, onSeek }) {
  const phases = summary.phases || {};
  const PHASE_DESCRIPTIONS = {
    setup:          'Pitcher in set or wind-up stance.',
    windup:         'Initial weight shift and leg lift begins.',
    stride:         'Lead leg driving forward toward home plate.',
    foot_strike:    'Lead foot contacts the ground — energy transfer begins.',
    arm_cocking:    'Arm loads externally — elbow rises to shoulder height.',
    acceleration:   'Hip-to-shoulder energy transfer — maximum X-factor.',
    release:        'Ball leaves the hand — peak wrist velocity.',
    follow_through: 'Arm decelerates across the body, reducing arm stress.',
  };

  return (
    <div className="phase-detail">
      {Object.entries(phases).map(([name, { start, end }]) => {
        const phaseFrames = frames.slice(start, end + 1);
        if (phaseFrames.length === 0) return null;

        const avgHss = avg(phaseFrames, f => f.metrics.hip_shoulder_sep);
        const peakSpd = Math.max(...phaseFrames.map(f => Math.abs(f.metrics.arm_speed)));
        const durationMs = ((end - start) / summary.fps * 1000).toFixed(0);

        return (
          <div key={name} className="phase-card" onClick={() => onSeek(start)}>
            <div className="phase-card-header">
              <span className={`phase-dot phase-${name}`} />
              <span className="phase-name">{name.replace(/_/g, ' ')}</span>
              <span className="phase-duration">{durationMs} ms</span>
            </div>
            <p className="phase-desc">{PHASE_DESCRIPTIONS[name]}</p>
            <div className="phase-stats">
              <Stat label="Avg Hip-Shoulder Sep" val={avgHss} unit="°" />
              <Stat label="Peak Arm Speed" val={peakSpd} unit="°/s" />
              <Stat label="Frames" val={end - start} unit="" />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Stat({ label, val, unit }) {
  return (
    <div className="stat">
      <span className="stat-val">{typeof val === 'number' ? val.toFixed(1) : val}{unit}</span>
      <span className="stat-label">{label}</span>
    </div>
  );
}

function avg(arr, fn) {
  if (!arr.length) return 0;
  return arr.reduce((s, x) => s + fn(x), 0) / arr.length;
}
