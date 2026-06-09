function grade(val, good, avg) {
  if (val >= good) return { letter: 'A', color: '#4dff88' };
  if (val >= avg)  return { letter: 'B', color: '#aadd44' };
  return           { letter: 'C', color: '#ff8844' };
}

function MetricCard({ label, value, unit, grade: g, note }) {
  return (
    <div className="summary-card">
      <div className="summary-card-top">
        <span className="summary-label">{label}</span>
        {g && (
          <span className="summary-grade" style={{ color: g.color }}>{g.letter}</span>
        )}
      </div>
      <div className="summary-value">
        {typeof value === 'number' ? value.toFixed(1) : value}
        <span className="summary-unit">{unit}</span>
      </div>
      {note && <p className="summary-note">{note}</p>}
    </div>
  );
}

export default function SummaryReport({ summary, frames }) {
  const { peak, key_frames, phases, fps, duration_s, throw_hand } = summary;

  if (!peak) return <p style={{ color: '#888', padding: 24 }}>No summary data available.</p>;

  const maxArmSpeed = peak.max_arm_speed || 0;
  const maxHss      = peak.max_hip_shoulder_sep || 0;
  const releaseF    = key_frames?.release ?? 0;
  const fsF         = key_frames?.foot_strike ?? 0;
  const merF        = key_frames?.max_ext_rot ?? 0;

  // Time from foot strike to release (should be ~0.1–0.2s for elite)
  const fsToRelease = ((releaseF - fsF) / fps * 1000).toFixed(0);
  // Time from MER to release (arm acceleration window)
  const merToRelease = ((releaseF - merF) / fps * 1000).toFixed(0);

  const releaseFrame = frames[releaseF] || {};
  const fsFrame      = frames[fsF] || {};

  const hsAtFs  = fsFrame.metrics?.hip_shoulder_sep ?? 0;
  const elbowH  = (frames[merF] || {}).metrics?.elbow_height_pct ?? 0;

  return (
    <div className="summary-report">
      <div className="summary-hero">
        <div className="hero-stat">
          <span className="hero-val">{maxArmSpeed.toFixed(0)}</span>
          <span className="hero-unit">°/s</span>
          <span className="hero-label">Peak Arm Speed</span>
        </div>
        <div className="hero-stat">
          <span className="hero-val">{maxHss.toFixed(1)}</span>
          <span className="hero-unit">°</span>
          <span className="hero-label">Max Hip-Shoulder Sep</span>
        </div>
        <div className="hero-stat">
          <span className="hero-val">{fsToRelease}</span>
          <span className="hero-unit">ms</span>
          <span className="hero-label">Foot Strike → Release</span>
        </div>
        <div className="hero-stat">
          <span className="hero-val">{merToRelease}</span>
          <span className="hero-unit">ms</span>
          <span className="hero-label">MER → Release</span>
        </div>
      </div>

      <div className="summary-grid">
        <MetricCard
          label="Max Arm Speed"
          value={maxArmSpeed}
          unit="°/s"
          grade={grade(maxArmSpeed, 700, 400)}
          note="Arm angular velocity at peak — higher = more whip through the zone."
        />
        <MetricCard
          label="Max Hip-Shoulder Sep"
          value={maxHss}
          unit="°"
          grade={grade(maxHss, 25, 12)}
          note="X-factor: hip opening ahead of shoulders. Elite range: 25–45°."
        />
        <MetricCard
          label="Hip-Shoulder Sep at Foot Strike"
          value={hsAtFs}
          unit="°"
          grade={grade(hsAtFs, 20, 8)}
          note="Separation when lead foot lands — the stored torque available for acceleration."
        />
        <MetricCard
          label="Elbow Height at MER"
          value={elbowH}
          unit="% above shoulder"
          grade={grade(elbowH, 5, 0)}
          note="Throwing elbow height relative to shoulder at max external rotation. Elbow at or above shoulder reduces stress."
        />
        <MetricCard
          label="MER → Release"
          value={parseFloat(merToRelease)}
          unit="ms"
          note="Acceleration window. Shorter time = more explosive."
        />
        <MetricCard
          label="Duration"
          value={duration_s}
          unit="s"
          note={`${summary.total_frames} frames @ ${fps.toFixed(0)} fps`}
        />
      </div>

      <div className="recommendations">
        <h3>Key Observations</h3>
        <ul>
          {maxHss < 15 && (
            <li className="obs-warn">
              <strong>Low hip-shoulder separation ({maxHss.toFixed(1)}°):</strong>{' '}
              Hips and shoulders are rotating together — the body isn't generating a whipping sequence.
              Focus on firing the hips before the shoulders.
            </li>
          )}
          {maxHss >= 25 && (
            <li className="obs-good">
              <strong>Strong hip-shoulder separation ({maxHss.toFixed(1)}°):</strong>{' '}
              Good torque sequence. Energy is loading efficiently before shoulder rotation.
            </li>
          )}
          {elbowH < 0 && (
            <li className="obs-warn">
              <strong>Elbow dropping below shoulder at MER:</strong>{' '}
              Low elbow at max external rotation increases UCL stress. Aim to keep elbow at or above shoulder line.
            </li>
          )}
          {maxArmSpeed > 700 && (
            <li className="obs-good">
              <strong>High arm speed ({maxArmSpeed.toFixed(0)}°/s):</strong>{' '}
              Efficient arm action with good acceleration through the zone.
            </li>
          )}
          {maxArmSpeed < 300 && (
            <li className="obs-warn">
              <strong>Low detected arm speed ({maxArmSpeed.toFixed(0)}°/s):</strong>{' '}
              This may indicate slow-motion footage (expected), or arm path inefficiency.
              Try filming at normal speed for more accurate arm speed readings.
            </li>
          )}
          {parseFloat(merToRelease) > 200 && (
            <li className="obs-warn">
              <strong>Long MER-to-release window ({merToRelease}ms):</strong>{' '}
              Arm lingers at max external rotation — consider a more explosive transition.
            </li>
          )}
          <li className="obs-info">
            <strong>Camera angle:</strong>{' '}
            Analysis optimized for first-base-side view ({throw_hand === 'left' ? 'lefty' : 'righty'} pitcher).
            For best accuracy, keep the full delivery in frame and film at ≥60 fps.
          </li>
        </ul>
      </div>
    </div>
  );
}
