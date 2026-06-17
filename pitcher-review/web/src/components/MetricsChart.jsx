import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ReferenceLine, ResponsiveContainer, Brush,
} from 'recharts';
import { useMemo } from 'react';
import { PHASE_COLORS, METRIC_COLORS } from '../theme';
import { downsample } from '../lib/chart';

export default function MetricsChart({ frames, summary, currentFrame, onSeek }) {
  const chartData = useMemo(() => downsample(
    frames.map(f => ({
      t: parseFloat(f.time.toFixed(3)),
      frame: f.frame,
      hip: parseFloat(f.metrics.hip_rotation.toFixed(2)),
      shoulder: parseFloat(f.metrics.shoulder_rotation.toFixed(2)),
      hss: parseFloat(f.metrics.hip_shoulder_sep.toFixed(2)),
      hipSpeed: parseFloat(Math.abs(f.metrics.hip_rotation_speed).toFixed(1)),
      chestSpeed: parseFloat(Math.abs(f.metrics.chest_rotation_speed).toFixed(1)),
      armSpeed: parseFloat(Math.abs(f.metrics.arm_speed).toFixed(1)),
      elbowAngle: parseFloat(f.metrics.elbow_angle.toFixed(1)),
      elbowH: parseFloat(f.metrics.elbow_height_pct.toFixed(2)),
      trunkTilt: parseFloat(f.metrics.trunk_tilt.toFixed(2)),
    }))
  ), [frames]);

  const phases = summary.phases || {};

  // Key phase reference lines
  const refLines = Object.entries(phases)
    .filter(([n]) => ['foot_strike', 'release'].includes(n))
    .map(([name, { start }]) => ({
      name,
      t: parseFloat((start / summary.fps).toFixed(3)),
      color: PHASE_COLORS[name] || PHASE_COLORS.setup,
    }));

  const currentT = parseFloat((currentFrame / summary.fps).toFixed(3));

  const CustomDot = () => null;

  return (
    <div className="charts-panel">
      {/* Rotation & Hip-Shoulder Separation */}
      <ChartBlock title="Rotation Angles & X-Factor (Hip-Shoulder Separation)">
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={chartData} onClick={d => d?.activePayload && onSeek(d.activePayload[0]?.payload?.frame)}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="t" stroke="var(--text-muted)" tick={{ fontSize: 11 }} label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -5, fill: 'var(--text-muted)', fontSize: 11 }} />
            <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} unit="°" />
            <Tooltip
              contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-strong)', borderRadius: 6 }}
              labelStyle={{ color: 'var(--text-secondary)' }}
              formatter={(val, name) => [`${val.toFixed(1)}°`, name]}
              labelFormatter={t => `t = ${t}s`}
            />
            <Legend wrapperStyle={{ color: 'var(--text-secondary)', fontSize: 12 }} />
            {refLines.map(r => (
              <ReferenceLine key={r.name} x={r.t} stroke={r.color} strokeDasharray="4 2"
                label={{ value: r.name.replace('_', ' '), fill: r.color, fontSize: 10, position: 'top' }} />
            ))}
            <ReferenceLine x={currentT} stroke="var(--text)" strokeDasharray="2 2" />
            <Line type="monotone" dataKey="hip"      name="Hip Rotation"     stroke={METRIC_COLORS.hip} strokeWidth={2} dot={<CustomDot />} />
            <Line type="monotone" dataKey="shoulder" name="Shoulder Rotation" stroke={METRIC_COLORS.shoulder} strokeWidth={2} dot={<CustomDot />} />
            <Line type="monotone" dataKey="hss"      name="Hip-Shoulder Sep"  stroke={METRIC_COLORS.hss} strokeWidth={2} dot={<CustomDot />} />
          </LineChart>
        </ResponsiveContainer>
      </ChartBlock>

      {/* Kinetic Chain Speed: Hip -> Chest -> Arm */}
      <ChartBlock title="Kinetic Chain Speed — Hip → Chest → Arm (°/s)">
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={chartData} onClick={d => d?.activePayload && onSeek(d.activePayload[0]?.payload?.frame)}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="t" stroke="var(--text-muted)" tick={{ fontSize: 11 }} label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -5, fill: 'var(--text-muted)', fontSize: 11 }} />
            <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} unit="°/s" />
            <Tooltip
              contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-strong)', borderRadius: 6 }}
              labelStyle={{ color: 'var(--text-secondary)' }}
              formatter={(val, name) => [`${val.toFixed(0)}°/s`, name]}
              labelFormatter={t => `t = ${t}s`}
            />
            <Legend wrapperStyle={{ color: 'var(--text-secondary)', fontSize: 12 }} />
            {refLines.map(r => (
              <ReferenceLine key={r.name} x={r.t} stroke={r.color} strokeDasharray="4 2" />
            ))}
            <ReferenceLine x={currentT} stroke="var(--text)" strokeDasharray="2 2" />
            <Line type="monotone" dataKey="hipSpeed"   name="Hip Speed"   stroke={METRIC_COLORS.hipSpeed} strokeWidth={2} dot={<CustomDot />} />
            <Line type="monotone" dataKey="chestSpeed" name="Chest Speed" stroke={METRIC_COLORS.chestSpeed} strokeWidth={2} dot={<CustomDot />} />
            <Line type="monotone" dataKey="armSpeed"   name="Arm Speed"   stroke={METRIC_COLORS.armSpeed} strokeWidth={2} dot={<CustomDot />} />
          </LineChart>
        </ResponsiveContainer>
      </ChartBlock>

      {/* Elbow height + trunk tilt */}
      <ChartBlock title="Elbow Height (%) & Trunk Tilt (°)">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData} onClick={d => d?.activePayload && onSeek(d.activePayload[0]?.payload?.frame)}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="t" stroke="var(--text-muted)" tick={{ fontSize: 11 }} label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -5, fill: 'var(--text-muted)', fontSize: 11 }} />
            <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-strong)', borderRadius: 6 }}
              labelStyle={{ color: 'var(--text-secondary)' }}
              labelFormatter={t => `t = ${t}s`}
            />
            <Legend wrapperStyle={{ color: 'var(--text-secondary)', fontSize: 12 }} />
            {refLines.map(r => (
              <ReferenceLine key={r.name} x={r.t} stroke={r.color} strokeDasharray="4 2" />
            ))}
            <ReferenceLine x={currentT} stroke="var(--text)" strokeDasharray="2 2" />
            <ReferenceLine y={0} stroke="var(--border-strong)" />
            <Line type="monotone" dataKey="elbowH"    name="Elbow Height (%)" stroke={METRIC_COLORS.elbowH} strokeWidth={2} dot={<CustomDot />} />
            <Line type="monotone" dataKey="trunkTilt" name="Trunk Tilt (°)"   stroke={METRIC_COLORS.trunkTilt} strokeWidth={2} dot={<CustomDot />} />
          </LineChart>
        </ResponsiveContainer>
      </ChartBlock>
    </div>
  );
}

function ChartBlock({ title, children }) {
  return (
    <div className="chart-block">
      <h4 className="chart-title">{title}</h4>
      {children}
    </div>
  );
}
