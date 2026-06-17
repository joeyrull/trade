import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ReferenceLine, ResponsiveContainer,
} from 'recharts';
import { useMemo } from 'react';
import { METRIC_COLORS, STATUS_COLORS } from '../theme';
import { downsample } from '../lib/chart';

// PitchCap's headline output: the kinematic sequence — pelvis, trunk,
// shoulder (upper arm), and elbow (forearm) angular velocities (deg/s) over
// time, when each segment peaks, and the order/lags between those peaks. A
// healthy delivery fires proximal→distal: pelvis → trunk → shoulder → elbow.
// This panel only renders when the analysis was produced by the PitchCap
// engine (summary.pitchcap present).

const SEGMENT_ORDER = ['pelvis', 'trunk', 'shoulder', 'elbow'];
const SEGMENT_COLORS = {
  pelvis:   METRIC_COLORS.hipSpeed,
  trunk:    METRIC_COLORS.chestSpeed,
  shoulder: METRIC_COLORS.armSpeed,
  elbow:    METRIC_COLORS.elbowAngle,
};
const SEGMENT_LABEL = {
  pelvis: 'Pelvis (hips)', trunk: 'Trunk',
  shoulder: 'Shoulder (upper arm)', elbow: 'Elbow (forearm)',
};
const LAG_PAIRS = [
  ['pelvis_to_trunk', 'pelvis', 'trunk'],
  ['trunk_to_shoulder', 'trunk', 'shoulder'],
  ['shoulder_to_elbow', 'shoulder', 'elbow'],
];

export default function KinematicSequence({ pitchcap, currentFrame = 0, onSeek }) {
  const ks = pitchcap?.kinematic_sequence;
  const segments = ks?.segments || {};

  // ks.fps is the real-world (motion) rate, so a peak time in seconds maps to a
  // decoded frame index via t * fps — the same index the video player seeks to.
  const fps = ks?.fps || 30;
  const seekToTime = (t) => onSeek && t != null && onSeek(Math.round(t * fps));
  const chartData = useMemo(() => {
    const series = SEGMENT_ORDER.map(name => segments[name]?.series_degps || []);
    const n = Math.max(0, ...series.map(s => s.length));
    const rows = [];
    for (let i = 0; i < n; i++) {
      const row = { t: parseFloat((i / fps).toFixed(3)) };
      SEGMENT_ORDER.forEach((name, j) => {
        const v = series[j][i];
        row[name] = v != null ? parseFloat(v.toFixed(1)) : null;
      });
      rows.push(row);
    }
    return downsample(rows);
  }, [segments, fps]);

  if (!ks) return null;

  const order = ks.sequence_order || [];
  const lags = ks.inter_peak_lags_ms || {};
  const properOrder = order.join(',') === SEGMENT_ORDER.join(',');
  const fmtLag = (v) => (v == null ? '—' : `${v} ms`);  // null lag (dead segment) shows '—', not '— ms'
  const warnings = [...new Set([...(ks.segment_warnings || []), ...(pitchcap.warnings || [])])];

  // Peak time markers (skip invalid segments whose peak_time_s is null).
  const peakLines = SEGMENT_ORDER
    .filter(name => segments[name]?.peak_time_s != null)
    .map(name => ({ name, t: parseFloat(segments[name].peak_time_s.toFixed(3)) }));

  const modeLabel = pitchcap.mode === 'multiview'
    ? `Multi-view triangulation (${pitchcap.n_cams} cameras)`
    : 'Single-camera (monocular 3D)';

  return (
    <div className="charts-panel kinematic-sequence">
      <div className="ks-header">
        <h3 className="ks-title">Kinematic Sequence <span className="ks-engine">PitchCap</span></h3>
        <p className="ks-sub">
          {modeLabel}
          {pitchcap.reprojection_error_px != null &&
            ` · reprojection error ${pitchcap.reprojection_error_px.toFixed(1)} px`}
        </p>
      </div>

      {/* Sequence order ribbon */}
      <div className="ks-order">
        {order.map((name, i) => (
          <span key={name} className="ks-order-step">
            <span className="ks-chip" style={{ background: SEGMENT_COLORS[name] || 'var(--border)' }}>
              {SEGMENT_LABEL[name] || name}
            </span>
            {i < order.length - 1 && <span className="ks-arrow">→</span>}
          </span>
        ))}
        <span
          className="ks-order-verdict"
          style={{ color: properOrder ? STATUS_COLORS.good : STATUS_COLORS.bad }}
          title="A healthy delivery sequences proximal→distal: pelvis → trunk → shoulder → elbow."
        >
          {properOrder ? '✓ proper proximal→distal order' : '⚠ out of sequence'}
        </span>
      </div>

      {/* Peak cards */}
      <div className="ks-peaks">
        {SEGMENT_ORDER.map(name => {
          const seg = segments[name];
          if (!seg) return null;
          const invalid = seg.peak_time_s == null;
          // tracked_frac < 0.5 means the segment's peak came mostly from
          // interpolated/held joints — flag it as low-confidence.
          const frac = seg.tracked_frac;
          const lowConf = frac != null && frac < 0.5;
          return (
            <div
              key={name}
              className={`ks-peak-card ${lowConf ? 'low-conf' : ''} ${!invalid && onSeek ? 'seekable' : ''}`}
              onClick={() => !invalid && seekToTime(seg.peak_time_s)}
              title={!invalid && onSeek ? 'Jump to this segment\'s peak in the video' : undefined}
            >
              <span className="ks-peak-seg" style={{ color: SEGMENT_COLORS[name] }}>
                {SEGMENT_LABEL[name] || name}
              </span>
              <span className="ks-peak-val">
                {invalid ? '—' : `${Math.round(seg.peak_degps)}`}
                <span className="ks-peak-unit">°/s</span>
              </span>
              <span className="ks-peak-time">
                {invalid ? 'not reconstructed' : `peak @ ${seg.peak_time_s.toFixed(3)}s`}
              </span>
              {frac != null && (
                <span className="ks-peak-track" title="Share of frames where every joint this segment needs was tracked.">
                  {lowConf ? '⚠ ' : ''}{Math.round(frac * 100)}% tracked
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Inter-peak lags */}
      <div className="ks-lags">
        {LAG_PAIRS.map(([key, a, b]) => (
          <div className="ks-lag" key={key}>
            <span className="ks-lag-val">{fmtLag(lags[key])}</span>
            <span className="ks-lag-label">{a} → {b}</span>
          </div>
        ))}
      </div>

      {/* Angular-velocity curves */}
      <div className="chart-block">
        <h4 className="chart-title">Segment Angular Velocity (°/s)</h4>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={chartData} onClick={d => d?.activePayload && seekToTime(d.activePayload[0]?.payload?.t)}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="t" stroke="var(--text-muted)" tick={{ fontSize: 11 }}
              label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -5, fill: 'var(--text-muted)', fontSize: 11 }} />
            <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} unit="°/s" />
            <Tooltip
              contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-strong)', borderRadius: 6 }}
              labelStyle={{ color: 'var(--text-secondary)' }}
              formatter={(val, name) => [`${val?.toFixed?.(0) ?? val}°/s`, SEGMENT_LABEL[name] || name]}
              labelFormatter={t => `t = ${t}s`}
            />
            <Legend formatter={name => SEGMENT_LABEL[name] || name} />
            {peakLines.map(({ name, t }) => (
              <ReferenceLine key={name} x={t} stroke={SEGMENT_COLORS[name]} strokeDasharray="4 3" strokeOpacity={0.5} />
            ))}
            <ReferenceLine x={+(currentFrame / fps).toFixed(3)} stroke="var(--text)" strokeOpacity={0.65} />
            {SEGMENT_ORDER.map(name => (
              <Line key={name} type="monotone" dataKey={name} stroke={SEGMENT_COLORS[name]}
                dot={false} strokeWidth={2} connectNulls />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {warnings.length > 0 && (
        <ul className="ks-warnings">
          {warnings.map((w, i) => <li key={i}>⚠ {w}</li>)}
        </ul>
      )}

      <p className="ks-note">
        Angular velocity is scale-invariant, so the curve <em>shape</em> and the peak
        <em> timing &amp; order</em> are the most trustworthy readings here; absolute
        °/s magnitudes are approximate and improve with multi-camera capture and 240fps footage.
      </p>
    </div>
  );
}
