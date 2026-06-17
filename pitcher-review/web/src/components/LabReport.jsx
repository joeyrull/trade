import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ReferenceArea, ResponsiveContainer,
} from 'recharts';
import { useMemo } from 'react';
import { METRICS_LIBRARY } from '../data/metricsLibrary';
import LabScorecard from './LabScorecard';
import { PHASE_COLORS, GRADE_COLORS, gradeColor } from '../theme';
import { downsample } from '../lib/chart';

function grade(val, good, avg) {
  if (val >= good) return { letter: 'A', color: gradeColor('A') };
  if (val >= avg)  return { letter: 'B', color: gradeColor('B') };
  return           { letter: 'C', color: gradeColor('C') };
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
      lowConf: f.metrics.low_confidence,
    }))
  ), [frames]);

  const phases = summary.phases || {};
  const refLines = Object.entries(phases)
    .filter(([n]) => ['foot_strike', 'release'].includes(n))
    .map(([name, { start }]) => ({
      name,
      t: parseFloat((start / summary.fps).toFixed(3)),
      color: PHASE_COLORS[name] || PHASE_COLORS.setup,
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
  const { key, label, unit, color, definition, reference, drills, aggregate, grade: gradeCfg } = metric;

  let maxEntry = null, minEntry = null;
  for (const d of chartData) {
    if (maxEntry === null || d[key] > maxEntry[key]) maxEntry = d;
    if (minEntry === null || d[key] < minEntry[key]) minEntry = d;
  }
  const maxVal = maxEntry ? maxEntry[key] : 0;
  const minVal = minEntry ? minEntry[key] : 0;

  let badge = null;
  let headlineLabel = '';
  let headlineValue = '';
  let lowConfidence = false;
  if (aggregate === 'max') {
    headlineLabel = 'Peak';
    headlineValue = `${maxVal.toFixed(1)}${unit}`;
    lowConfidence = !!maxEntry?.lowConf;
    if (gradeCfg) badge = grade(maxVal, gradeCfg.good, gradeCfg.avg);
  } else {
    headlineLabel = 'Range';
    headlineValue = `${minVal.toFixed(1)} to ${maxVal.toFixed(1)}${unit}`;
    lowConfidence = !!(maxEntry?.lowConf || minEntry?.lowConf);
  }

  // For graded metrics, fix the y-axis domain so we can shade "average" and
  // "elite" benchmark bands behind the trendline (matches the scorecard bars).
  const yDomain = gradeCfg
    ? [Math.min(0, minVal), Math.ceil(Math.max(maxVal, gradeCfg.good) * 1.1 / 10) * 10]
    : ['auto', 'auto'];

  return (
    <div className="lab-metric-card">
      <div className="lab-metric-header">
        <h3 className="lab-metric-title">{label}</h3>
        <div className="lab-metric-headline">
          <span className="lab-headline-label">{headlineLabel}</span>
          <span className="lab-headline-value">
            {headlineValue}
            {lowConfidence && (
              <span className="low-conf-badge" title="Includes a low-confidence pose tracking frame — value may be inaccurate.">⚠</span>
            )}
          </span>
          {badge && (
            <span className="lab-grade-badge" style={{ background: badge.color }}>{badge.letter}</span>
          )}
        </div>
      </div>

      <p className="lab-metric-definition">{definition}</p>
      {reference && <p className="lab-metric-reference">MLB / Kinatrax reference: {reference}</p>}

      <div className="lab-chart-panel">
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={chartData} onClick={d => d?.activePayload && onSeek(d.activePayload[0]?.payload?.frame)}>
            <CartesianGrid strokeDasharray="3 3" stroke="#d6d9de" />
            <XAxis dataKey="t" stroke="#6b7280" tick={{ fontSize: 11 }}
              label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -5, fill: '#6b7280', fontSize: 11 }} />
            <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} unit={unit} domain={yDomain} />
            <Tooltip
              contentStyle={{ background: '#fff', border: '1px solid #d6d9de', borderRadius: 6, color: '#2a2f3a' }}
              labelStyle={{ color: '#6b7280' }}
              formatter={val => [`${val.toFixed(1)}${unit}`, label]}
              labelFormatter={t => `t = ${t}s`}
            />
            {gradeCfg && (
              <>
                <ReferenceArea y1={gradeCfg.avg} y2={gradeCfg.good} fill={GRADE_COLORS.B} fillOpacity={0.14}
                  label={{ value: 'Average range', position: 'insideTopLeft', fill: GRADE_COLORS.B, fontSize: 10 }} />
                <ReferenceArea y1={gradeCfg.good} y2={yDomain[1]} fill={GRADE_COLORS.A} fillOpacity={0.14}
                  label={{ value: 'Elite range', position: 'insideTopLeft', fill: GRADE_COLORS.A, fontSize: 10 }} />
              </>
            )}
            {refLines.map(r => (
              <ReferenceLine key={r.name} x={r.t} stroke={r.color} strokeDasharray="4 2"
                label={{ value: r.name.replace('_', ' '), fill: r.color, fontSize: 10, position: 'top' }} />
            ))}
            <ReferenceLine x={currentT} stroke="#2a2f3a" strokeDasharray="2 2" />
            {(aggregate === 'range') && <ReferenceLine y={0} stroke="#c4c8ce" />}
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
