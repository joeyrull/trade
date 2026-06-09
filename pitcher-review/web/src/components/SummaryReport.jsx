function grade(val, good, avg) {
  if (val >= good) return { letter: 'A', color: '#4dff88' };
  if (val >= avg)  return { letter: 'B', color: '#aadd44' };
  return           { letter: 'C', color: '#ff8844' };
}

function SequenceNode({ label, ms, color }) {
  return (
    <div className="seq-node">
      <span className="seq-dot" style={{ background: color }} />
      <span className="seq-label">{label}</span>
      <span className="seq-time">{ms}ms</span>
    </div>
  );
}

function SequenceArrow({ ms, ok }) {
  return (
    <div className={`seq-arrow ${ok ? 'ok' : 'bad'}`}>
      <span className="seq-arrow-line" />
      <span className="seq-arrow-gap">{ms}ms</span>
    </div>
  );
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
  const { peak, key_frames, phases, fps, duration_s, throw_hand, sequencing } = summary;

  if (!peak) return <p style={{ color: '#888', padding: 24 }}>No summary data available.</p>;

  const maxArmSpeed   = peak.max_arm_speed || 0;
  const maxHipSpeed   = peak.max_hip_rotation_speed || 0;
  const maxChestSpeed = peak.max_chest_rotation_speed || 0;
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
          <span className="hero-val">{maxHipSpeed.toFixed(0)}</span>
          <span className="hero-unit">°/s</span>
          <span className="hero-label">Peak Hip Speed</span>
        </div>
        <div className="hero-stat">
          <span className="hero-val">{maxChestSpeed.toFixed(0)}</span>
          <span className="hero-unit">°/s</span>
          <span className="hero-label">Peak Chest Speed</span>
        </div>
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
          label="Max Hip Rotation Speed"
          value={maxHipSpeed}
          unit="°/s"
          grade={grade(maxHipSpeed, 500, 250)}
          note="How fast the hips rotate — the first link in the kinetic chain."
        />
        <MetricCard
          label="Max Chest Rotation Speed"
          value={maxChestSpeed}
          unit="°/s"
          grade={grade(maxChestSpeed, 700, 350)}
          note="Shoulder/torso angular velocity — should peak after the hips."
        />
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

      {sequencing && (
        <div className="sequencing-panel">
          <h3>Kinetic Chain Sequencing</h3>
          <p className="sequencing-intro">
            Efficient deliveries fire like a whip: the hips reach peak rotation speed first,
            then the chest, then the arm — each link transferring energy to the next.
          </p>
          <div className="sequence-row">
            <SequenceNode label="Hip Peak"   ms={(sequencing.hip_peak_frame / fps * 1000).toFixed(0)} color="#00c8ff" />
            <SequenceArrow ms={sequencing.hip_to_chest_ms} ok={sequencing.hip_to_chest_ms >= 0} />
            <SequenceNode label="Chest Peak" ms={(sequencing.chest_peak_frame / fps * 1000).toFixed(0)} color="#ffaa00" />
            <SequenceArrow ms={sequencing.chest_to_arm_ms} ok={sequencing.chest_to_arm_ms >= 0} />
            <SequenceNode label="Arm Peak"   ms={(sequencing.arm_peak_frame / fps * 1000).toFixed(0)} color="#ff4444" />
          </div>
          <div className="sequence-verdict">
            {sequencing.proper_order ? (
              <span className="obs-good-inline">
                ✓ Proper sequence — hips lead, chest follows, arm finishes ({sequencing.hip_to_arm_ms}ms hip-to-arm).
              </span>
            ) : (
              <span className="obs-warn-inline">
                ⚠ Sequence out of order — peaks aren't firing hip → chest → arm. This often shows up as
                "arm-only" throwing where the body rotates together instead of sequentially.
              </span>
            )}
          </div>
        </div>
      )}

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
          {sequencing && !sequencing.proper_order && (
            <li className="obs-warn">
              <strong>Out-of-sequence kinetic chain:</strong>{' '}
              Hip, chest, and arm rotation speeds aren't peaking in the hip → chest → arm order.
              This usually means the upper body is rotating with the hips instead of being "left behind"
              to build separation, reducing the whip effect and adding arm strain.
            </li>
          )}
          {sequencing && sequencing.proper_order && sequencing.hip_to_arm_ms < 150 && (
            <li className="obs-good">
              <strong>Tight, well-sequenced delivery ({sequencing.hip_to_arm_ms}ms hip-to-arm):</strong>{' '}
              Energy transfers quickly from hips to arm — a hallmark of efficient, high-velocity mechanics.
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
