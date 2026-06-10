import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer,
} from 'recharts';
import { useMemo } from 'react';
import { METRICS_LIBRARY } from '../data/metricsLibrary';
import LabScorecard from './LabScorecard';

const PHASE_COLORS = {
  setup: '#888', windup: '#ffd700', stride: '#00c8ff',
  foot_strike: '#00ff88', arm_cocking: '#ff6400',
  acceleration: '#ff2200', release: '#cc00ff',
  follow_through: '#8080ff',
};

function downsample(arr, maxPts = 300) {
  if (arr.length <= maxPts) return arr;
  const step = Math.ceil(arr.length / maxPts);
  return arr.filter((_, i) => i % step === 0);
}

function grade(val, good, avg) {
  if (val >= good) return { letter: 'A', color: '#1a8a4a' };
  if (val >= avg)  return { letter: 'B', color: '#9aa015' };
  return           { letter: 'C', color: '#c0501a' };
}

export default function LabReport({ summary, frames, currentFrame, onSeek }) {
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
  const refLines = Object.entries(phases)
    .filter(([n]) => ['foot_strike', 'release'].includes(n))
    .map(([name, { start }]) => ({
      name,
      t: parseFloat((start / summary.fps).toFixed(3)),
      color: PHASE_COLORS[name] || '#888',
    }));

  const currentT = parseFloat((currentFrame / summary.fps).toFixed(3));

  return (
    <div className="lab-report">
      <p className="lab-report-intro">
        A breakdown of every metric tracked during this delivery — what it measures,
        how it trended over the course of the pitch, and how to work on it. Click any
        chart to jump the video to that point in the delivery.
      </p>

      <LabScorecard summary={summary} frames={frames} />

      {METRICS_LIBRARY.map(m => (
        <LabMetricCard
          key={m.key}
          metric={m}
          chartData={chartData}
          refLines={refLines}
          currentT={currentT}
          onSeek={onSeek}
        />
      ))}
    </div>
  );
}

function LabMetricCard({ metric, chartData, refLines, currentT, onSeek }) {
  const { key, label, unit, color, definition, drills, aggregate, grade: gradeCfg } = metric;

  const values = chartData.map(d => d[key]);
  const maxVal = values.length ? Math.max(...values) : 0;
  const minVal = values.length ? Math.min(...values) : 0;

  let badge = null;
  let headlineLabel = '';
  let headlineValue = '';
  if (aggregate === 'max') {
    headlineLabel = 'Peak';
    headlineValue = `${maxVal.toFixed(1)}${unit}`;
    if (gradeCfg) badge = grade(maxVal, gradeCfg.good, gradeCfg.avg);
  } else {
    headlineLabel = 'Range';
    headlineValue = `${minVal.toFixed(1)} to ${maxVal.toFixed(1)}${unit}`;
  }

  return (
    <div className="lab-metric-card">
      <div className="lab-metric-header">
        <h3 className="lab-metric-title">{label}</h3>
        <div className="lab-metric-headline">
          <span className="lab-headline-label">{headlineLabel}</span>
          <span className="lab-headline-value">{headlineValue}</span>
          {badge && (
            <span className="lab-grade-badge" style={{ background: badge.color }}>{badge.letter}</span>
          )}
        </div>
      </div>

      <p className="lab-metric-definition">{definition}</p>

      <div className="lab-chart-panel">
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={chartData} onClick={d => d?.activePayload && onSeek(d.activePayload[0]?.payload?.frame)}>
            <CartesianGrid strokeDasharray="3 3" stroke="#ccc" />
            <XAxis dataKey="t" stroke="#555" tick={{ fontSize: 11 }}
              label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -5, fill: '#555', fontSize: 11 }} />
            <YAxis stroke="#555" tick={{ fontSize: 11 }} unit={unit} />
            <Tooltip
              contentStyle={{ background: '#fff', border: '1px solid #ccc', borderRadius: 6, color: '#222' }}
              labelStyle={{ color: '#666' }}
              formatter={val => [`${val.toFixed(1)}${unit}`, label]}
              labelFormatter={t => `t = ${t}s`}
            />
            {refLines.map(r => (
              <ReferenceLine key={r.name} x={r.t} stroke={r.color} strokeDasharray="4 2"
                label={{ value: r.name.replace('_', ' '), fill: r.color, fontSize: 10, position: 'top' }} />
            ))}
            <ReferenceLine x={currentT} stroke="#222" strokeDasharray="2 2" />
            {(aggregate === 'range') && <ReferenceLine y={0} stroke="#bbb" />}
            <Line type="monotone" dataKey={key} name={label} stroke={color} strokeWidth={2.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="lab-drills">
        <h4>Drills &amp; Cues to Improve</h4>
        <ul>
          {drills.map((d, i) => <li key={i}>{d}</li>)}
        </ul>
      </div>
    </div>
  );
}
