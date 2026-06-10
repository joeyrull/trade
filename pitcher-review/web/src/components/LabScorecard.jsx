import { PILLARS } from '../data/pillars';

// Maps a raw value onto a 0-100 scale where `avg` ≈ 50 and `good` ≈ 90,
// extrapolating linearly beyond those anchors and clamping to [0, 100].
function scoreMetric({ avg, good, lowerIsBetter }, value) {
  if (value == null || Number.isNaN(value)) return null;
  const span = lowerIsBetter ? (avg - good) : (good - avg);
  const delta = lowerIsBetter ? (avg - value) : (value - avg);
  const pct = 50 + (delta / span) * 40;
  return Math.max(0, Math.min(100, pct));
}

function ratingFor(score) {
  if (score >= 80) return { label: 'Elite range',   color: '#4dff88' };
  if (score >= 50) return { label: 'Average range', color: '#aadd44' };
  return                  { label: 'Needs work',     color: '#ff8844' };
}

export default function LabScorecard({ summary, frames }) {
  const pillars = PILLARS.map(pillar => {
    const metrics = pillar.metrics
      .map(m => {
        const value = m.getValue(summary, frames);
        const score = scoreMetric(m, value);
        const lowConfidence = m.getLowConfidence?.(summary, frames) ?? false;
        return score == null ? null : { ...m, value, score, lowConfidence };
      })
      .filter(Boolean);

    const composite = metrics.length
      ? metrics.reduce((s, m) => s + m.score, 0) / metrics.length
      : null;

    return { ...pillar, metrics, composite };
  }).filter(p => p.metrics.length > 0);

  if (pillars.length === 0) return null;

  return (
    <div className="lab-scorecard">
      {pillars.map(pillar => {
        const rating = ratingFor(pillar.composite);
        return (
          <div className="pillar-card" key={pillar.name}>
            <div className="pillar-card-header">
              <div>
                <h3 className="pillar-name">{pillar.name}</h3>
                <p className="pillar-desc">{pillar.description}</p>
              </div>
              <div className="pillar-score" style={{ color: rating.color }}>
                <span className="pillar-score-val">{pillar.composite.toFixed(0)}</span>
                <span className="pillar-score-label">{rating.label}</span>
              </div>
            </div>

            <div className="pillar-metrics">
              {pillar.metrics.map(m => (
                <div className="pillar-metric" key={m.label}>
                  <div className="pillar-metric-top">
                    <span className="pillar-metric-label">{m.label}</span>
                    <span className="pillar-metric-value">
                      {m.value.toFixed(1)}{m.unit}
                      {m.lowConfidence && (
                        <span className="low-conf-badge" title="Based on a low-confidence pose tracking frame — value may be inaccurate.">⚠</span>
                      )}
                    </span>
                  </div>
                  <div className="pillar-bar-track">
                    <div className="pillar-bar-marker" style={{ left: `${m.score}%` }} />
                  </div>
                  <div className="pillar-bar-zones">
                    <span>Below Avg</span>
                    <span>Avg</span>
                    <span>Elite</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
