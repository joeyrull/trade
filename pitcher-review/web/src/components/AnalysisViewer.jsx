import { useState, useRef, useEffect, useCallback } from 'react';
import MetricsChart from './MetricsChart';
import PhaseTimeline from './PhaseTimeline';
import SummaryReport from './SummaryReport';
import PoseOverlay from './PoseOverlay';
import LabReport from './LabReport';
import KinematicSequence from './KinematicSequence';
import VideoControls from './VideoControls';
import { PHASE_COLORS, METRIC_COLORS, STATUS_COLORS } from '../theme';

const BASE_TABS = ['Overview', 'Charts', 'Phases', 'Lab Report', 'Annotated Video'];

const ANGLE_LABELS = {
  side:          'Side view',
  front:         'Front view',
  behind:        'Behind',
  three_quarter: '3/4 angle',
  other:         'Other angle',
  multiview:     'Multi-view (2 cameras)',
};

export default function AnalysisViewer({ result }) {
  const { cameras, sync } = result;
  const [activeCamera, setActiveCamera] = useState(0);
  const camera = cameras[activeCamera];
  const { annotatedVideo, metrics, metricsFused } = camera;
  const fusionAvailable = !!(metricsFused && metricsFused.summary.fusion?.applied);
  const [useFused, setUseFused] = useState(fusionAvailable);
  const activeMetrics = (useFused && fusionAvailable) ? metricsFused : metrics;
  const { summary, frames } = activeMetrics;

  const [tab, setTab] = useState('Overview');
  const [currentFrame, setCurrentFrame] = useState(0);
  const videoRef = useRef(null);
  const annotatedVideoRef = useRef(null);

  const switchCamera = useCallback((idx) => {
    setActiveCamera(idx);
    setCurrentFrame(0);
  }, []);

  // Default to fused metrics whenever they become available for the active
  // camera (e.g. after switching cameras or once fusion finishes processing).
  useEffect(() => {
    setUseFused(fusionAvailable);
  }, [activeCamera, fusionAvailable]);

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

  // The "Kinematic Sequence" tab only exists for analyses produced by the
  // PitchCap engine (which attaches summary.pitchcap).
  const hasPitchCap = !!summary.pitchcap;
  const TABS = hasPitchCap
    ? ['Overview', 'Kinematic Sequence', 'Charts', 'Phases', 'Lab Report', 'Annotated Video']
    : BASE_TABS;

  return (
    <div className="viewer">
      {cameras.length > 1 && (
        <div className="camera-selector">
          {cameras.map((c, i) => (
            <button
              key={i}
              className={`camera-btn ${i === activeCamera ? 'active' : ''}`}
              onClick={() => switchCamera(i)}
            >
              Camera {i + 1} — {ANGLE_LABELS[c.angle] || c.angle}
            </button>
          ))}
          {sync && (
            <span className="sync-note">
              {sync.methods?.[activeCamera] === 'cross-correlation'
                ? `Auto-synced (motion match ${(sync.correlations[activeCamera] * 100).toFixed(0)}%)`
                : sync.methods?.[activeCamera] === 'release-frame'
                  ? 'Synced on release frame'
                  : 'Reference camera'}
              {sync.offsetsSeconds[activeCamera] !== 0 &&
                ` (offset ${sync.offsetsSeconds[activeCamera] > 0 ? '+' : ''}${(sync.offsetsSeconds[activeCamera] * 1000).toFixed(0)} ms vs. Camera ${sync.referenceCamera + 1})`}
            </span>
          )}
          {fusionAvailable && (
            <label className="fusion-toggle">
              <input
                type="checkbox"
                checked={useFused}
                onChange={(e) => setUseFused(e.target.checked)}
              />
              Multi-camera fusion ({metricsFused.summary.fusion.filled_frames} frames filled
              from Camera {(activeCamera === 0 ? 2 : 1)})
            </label>
          )}
        </div>
      )}

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
          <VideoControls videoRef={videoRef} fps={summary.fps} />
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
        {tab === 'Kinematic Sequence' && hasPitchCap && (
          <KinematicSequence pitchcap={summary.pitchcap} currentFrame={currentFrame} onSeek={seekToFrame} />
        )}
        {tab === 'Charts' && (
          <MetricsChart frames={frames} summary={summary} currentFrame={currentFrame} onSeek={seekToFrame} />
        )}
        {tab === 'Phases' && (
          <PhaseDetail summary={summary} frames={frames} onSeek={seekToFrame} />
        )}
        {tab === 'Lab Report' && (
          <LabReport summary={summary} frames={frames} currentFrame={currentFrame} onSeek={seekToFrame} />
        )}
        {tab === 'Annotated Video' && (
          <div className="annotated-tab">
            <p className="annotated-note">
              The annotated video shows the MediaPipe pose skeleton, live metrics HUD, and phase labels overlaid on every frame.
            </p>
            <video ref={annotatedVideoRef} src={annotatedVideo} controls playsInline className="annotated-video-large" />
            <VideoControls videoRef={annotatedVideoRef} fps={summary.fps} />
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

  const metricRows = [
    { label: 'Hip Rotation',      val: m.hip_rotation,      unit: '°',   lo: -45, hi: 45 },
    { label: 'Shoulder Rotation', val: m.shoulder_rotation, unit: '°',   lo: -45, hi: 45 },
    { label: 'Hip-Shoulder Sep',  val: m.hip_shoulder_sep,  unit: '°',   good: v => v > 20 },
    { label: 'Elbow Height',      val: m.elbow_height_pct,  unit: '%',   good: v => v > 3  },
    { label: 'Elbow Angle',       val: m.elbow_angle,       unit: '°',   neutral: true      },
    { label: 'Hip Speed',         val: Math.abs(m.hip_rotation_speed),   unit: '°/s', good: v => v > 300 },
    { label: 'Chest Speed',       val: Math.abs(m.chest_rotation_speed), unit: '°/s', good: v => v > 400 },
    { label: 'Arm Speed',         val: Math.abs(m.arm_speed), unit: '°/s', good: v => v > 500 },
    { label: 'Trunk Tilt',        val: m.trunk_tilt,        unit: '°',   neutral: true      },
  ];

  return (
    <div className="live-metrics">
      <div className="phase-badge" style={{ background: PHASE_COLORS[phase] || PHASE_COLORS.setup }}>
        {phase.replace(/_/g, ' ').toUpperCase()}
      </div>
      {m.low_confidence && (
        <div className="low-conf-indicator" title="MediaPipe pose tracking confidence is low for this frame — metrics below may be inaccurate.">
          ⚠ Low-confidence tracking
        </div>
      )}
      <p className="frame-label">Frame {frame.frame} · t={frame.time.toFixed(3)}s</p>

      {metricRows.map(({ label, val, unit, good, neutral }) => {
        const frozen = m.low_confidence;
        const color = frozen ? STATUS_COLORS.frozen
          : neutral ? 'var(--text)'
          : good ? (good(val) ? STATUS_COLORS.good : STATUS_COLORS.bad)
          : 'var(--text)';
        return (
          <div key={label} className={`metric-row ${frozen ? 'frozen' : ''}`}>
            <span className="metric-label">{label}</span>
            <span className="metric-val" style={{ color }}>
              {typeof val === 'number' ? val.toFixed(1) : '—'}{unit}
              {frozen && <span className="frozen-tag"> (held)</span>}
            </span>
          </div>
        );
      })}

      <div className="speed-bars">
        <SpeedBar label="Hip"   value={Math.abs(m.hip_rotation_speed)}   max={800} color={METRIC_COLORS.hipSpeed} />
        <SpeedBar label="Chest" value={Math.abs(m.chest_rotation_speed)} max={1000} color={METRIC_COLORS.chestSpeed} />
        <SpeedBar label="Arm"   value={Math.abs(m.arm_speed)}            max={1200} color={METRIC_COLORS.armSpeed} />
      </div>
    </div>
  );
}

function SpeedBar({ label, value, max, color }) {
  return (
    <div className="speed-bar-row">
      <span className="speed-bar-label">{label}</span>
      <div className="speed-bar-track">
        <div
          className="speed-bar-fill"
          style={{ width: `${Math.min(100, (value / max) * 100)}%`, background: color }}
        />
      </div>
      <span className="speed-bar-val">{value.toFixed(0)}°/s</span>
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
        // Real-world duration: for slow-motion clips, motion_fps (the true
        // capture rate) differs from fps (the slower playback rate), so the
        // same frame span covers far less real time than fps would suggest.
        const durationMs = ((end - start) / (summary.motion_fps || summary.fps) * 1000).toFixed(0);

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
