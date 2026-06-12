import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ReferenceLine, ResponsiveContainer,
} from 'recharts';
import { useMemo } from 'react';
import { METRIC_COLORS, STATUS_COLORS } from '../theme';

// PitchCap's headline output: the kinematic sequence — pelvis, trunk, and
// throwing-arm angular velocities (deg/s) over time, when each segment peaks,
// and the order/lags between those peaks. A healthy delivery fires
// proximal→distal: pelvis → trunk → arm. This panel only renders when the
// analysis was produced by the PitchCap engine (summary.pitchcap present).

const SEGMENT_COLORS = {
  pelvis: METRIC_COLORS.hipSpeed,
  trunk:  METRIC_COLORS.chestSpeed,
  arm:    METRIC_COLORS.armSpeed,
};
const SEGMENT_LABEL = { pelvis: 'Pelvis (hips)', trunk: 'Trunk', arm: 'Throwing arm' };

function downsample(arr, maxPts = 300) {
  if (arr.length <= maxPts) return arr;
  const step = Math.ceil(arr.length / maxPts);
  return arr.filter((_, i) => i % step === 0);
}

export default function KinematicSequence({ pitchcap }) {
  const ks = pitchcap?.kinematic_sequence;
  const segments = ks?.segments || {};

  const fps = ks?.fps || 30;
  const chartData = useMemo(() => {
    const pelvis = segments.pelvis?.series_degps || [];
    const trunk = segments.trunk?.series_degps || [];
    const arm = segments.arm?.series_degps || [];
    const n = Math.max(pelvis.length, trunk.length, arm.length);
    const rows = [];
    for (let i = 0; i < n; i++) {
      rows.push({
        t: parseFloat((i / fps).toFixed(3)),
        pelvis: pelvis[i] != null ? parseFloat(pelvis[i].toFixed(1)) : null,
        trunk: trunk[i] != null ? parseFloat(trunk[i].toFixed(1)) : null,
        arm: arm[i] != null ? parseFloat(arm[i].toFixed(1)) : null,
      });
    }
    return downsample(rows);
  }, [segments, fps]);

  if (!ks) return null;

  const order = ks.sequence_order || [];
  const lags = ks.inter_peak_lags_ms || {};
  const properOrder = order.join(',') === 'pelvis,trunk,arm';
  const warnings = [...(ks.segment_warnings || []), ...(pitchcap.warnings || [])];

  // Peak time markers (skip invalid segments whose peak_time_s is null).
  const peakLines = ['pelvis', 'trunk', 'arm']
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
          title="A healthy delivery sequences proximal→distal: pelvis → trunk → arm."
        >
          {properOrder ? '✓ proper proximal→distal order' : '⚠ out of sequence'}
        </span>
      </div>

      {/* Peak cards */}
      <div className="ks-peaks">
        {['pelvis', 'trunk', 'arm'].map(name => {
          const seg = segments[name];
          if (!seg) return null;
          const invalid = seg.peak_time_s == null;
          return (
            <div key={name} className="ks-peak-card">
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
            </div>
          );
        })}
      </div>

      {/* Inter-peak lags */}
      <div className="ks-lags">
        <div className="ks-lag">
          <span className="ks-lag-val">{lags.pelvis_to_trunk ?? '—'} ms</span>
          <span className="ks-lag-label">pelvis → trunk</span>
        </div>
        <div className="ks-lag">
          <span className="ks-lag-val">{lags.trunk_to_arm ?? '—'} ms</span>
          <span className="ks-lag-label">trunk → arm</span>
        </div>
      </div>

      {/* Angular-velocity curves */}
      <div className="chart-block">
        <h4 className="chart-title">Segment Angular Velocity (°/s)</h4>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={chartData}>
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
            <Line type="monotone" dataKey="pelvis" stroke={SEGMENT_COLORS.pelvis} dot={false} strokeWidth={2} connectNulls />
            <Line type="monotone" dataKey="trunk" stroke={SEGMENT_COLORS.trunk} dot={false} strokeWidth={2} connectNulls />
            <Line type="monotone" dataKey="arm" stroke={SEGMENT_COLORS.arm} dot={false} strokeWidth={2} connectNulls />
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
