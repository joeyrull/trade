import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ReferenceLine, ResponsiveContainer, Brush,
} from 'recharts';
import { useMemo } from 'react';

const PHASE_COLORS = {
  setup: '#888', windup: '#ffd700', stride: '#00c8ff',
  foot_strike: '#00ff88', arm_cocking: '#ff6400',
  acceleration: '#ff2200', release: '#cc00ff',
  follow_through: '#8080ff',
};

// Downsample to at most maxPts for chart performance
function downsample(arr, maxPts = 300) {
  if (arr.length <= maxPts) return arr;
  const step = Math.ceil(arr.length / maxPts);
  return arr.filter((_, i) => i % step === 0);
}

export default function MetricsChart({ frames, summary, currentFrame, onSeek }) {
  const chartData = useMemo(() => downsample(
    frames.map(f => ({
      t: parseFloat(f.time.toFixed(3)),
      frame: f.frame,
      hip: parseFloat(f.metrics.hip_rotation.toFixed(2)),
      shoulder: parseFloat(f.metrics.shoulder_rotation.toFixed(2)),
      hss: parseFloat(f.metrics.hip_shoulder_sep.toFixed(2)),
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
      color: PHASE_COLORS[name] || '#888',
    }));

  const currentT = parseFloat((currentFrame / summary.fps).toFixed(3));

  const CustomDot = () => null;

  return (
    <div className="charts-panel">
      {/* Rotation & Hip-Shoulder Separation */}
      <ChartBlock title="Rotation Angles & X-Factor (Hip-Shoulder Separation)">
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={chartData} onClick={d => d?.activePayload && onSeek(d.activePayload[0]?.payload?.frame)}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="t" stroke="#888" tick={{ fontSize: 11 }} label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -5, fill: '#888', fontSize: 11 }} />
            <YAxis stroke="#888" tick={{ fontSize: 11 }} unit="°" />
            <Tooltip
              contentStyle={{ background: '#1a1a1a', border: '1px solid #444', borderRadius: 6 }}
              labelStyle={{ color: '#aaa' }}
              formatter={(val, name) => [`${val.toFixed(1)}°`, name]}
              labelFormatter={t => `t = ${t}s`}
            />
            <Legend wrapperStyle={{ color: '#ccc', fontSize: 12 }} />
            {refLines.map(r => (
              <ReferenceLine key={r.name} x={r.t} stroke={r.color} strokeDasharray="4 2"
                label={{ value: r.name.replace('_', ' '), fill: r.color, fontSize: 10, position: 'top' }} />
            ))}
            <ReferenceLine x={currentT} stroke="#fff" strokeDasharray="2 2" />
            <Line type="monotone" dataKey="hip"      name="Hip Rotation"     stroke="#f0a000" strokeWidth={2} dot={<CustomDot />} />
            <Line type="monotone" dataKey="shoulder" name="Shoulder Rotation" stroke="#00aaff" strokeWidth={2} dot={<CustomDot />} />
            <Line type="monotone" dataKey="hss"      name="Hip-Shoulder Sep"  stroke="#00ff88" strokeWidth={2} dot={<CustomDot />} />
          </LineChart>
        </ResponsiveContainer>
      </ChartBlock>

      {/* Arm Speed */}
      <ChartBlock title="Arm Speed (°/s)">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData} onClick={d => d?.activePayload && onSeek(d.activePayload[0]?.payload?.frame)}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="t" stroke="#888" tick={{ fontSize: 11 }} label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -5, fill: '#888', fontSize: 11 }} />
            <YAxis stroke="#888" tick={{ fontSize: 11 }} unit="°/s" />
            <Tooltip
              contentStyle={{ background: '#1a1a1a', border: '1px solid #444', borderRadius: 6 }}
              labelStyle={{ color: '#aaa' }}
              formatter={(val, name) => [`${val.toFixed(0)}°/s`, name]}
              labelFormatter={t => `t = ${t}s`}
            />
            <Legend wrapperStyle={{ color: '#ccc', fontSize: 12 }} />
            {refLines.map(r => (
              <ReferenceLine key={r.name} x={r.t} stroke={r.color} strokeDasharray="4 2" />
            ))}
            <ReferenceLine x={currentT} stroke="#fff" strokeDasharray="2 2" />
            <Line type="monotone" dataKey="armSpeed" name="Arm Speed" stroke="#ff4444" strokeWidth={2} dot={<CustomDot />} />
          </LineChart>
        </ResponsiveContainer>
      </ChartBlock>

      {/* Elbow height + trunk tilt */}
      <ChartBlock title="Elbow Height (%) & Trunk Tilt (°)">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData} onClick={d => d?.activePayload && onSeek(d.activePayload[0]?.payload?.frame)}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="t" stroke="#888" tick={{ fontSize: 11 }} label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -5, fill: '#888', fontSize: 11 }} />
            <YAxis stroke="#888" tick={{ fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: '#1a1a1a', border: '1px solid #444', borderRadius: 6 }}
              labelStyle={{ color: '#aaa' }}
              labelFormatter={t => `t = ${t}s`}
            />
            <Legend wrapperStyle={{ color: '#ccc', fontSize: 12 }} />
            {refLines.map(r => (
              <ReferenceLine key={r.name} x={r.t} stroke={r.color} strokeDasharray="4 2" />
            ))}
            <ReferenceLine x={currentT} stroke="#fff" strokeDasharray="2 2" />
            <ReferenceLine y={0} stroke="#555" />
            <Line type="monotone" dataKey="elbowH"    name="Elbow Height (%)" stroke="#aa88ff" strokeWidth={2} dot={<CustomDot />} />
            <Line type="monotone" dataKey="trunkTilt" name="Trunk Tilt (°)"   stroke="#ffaa44" strokeWidth={2} dot={<CustomDot />} />
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
