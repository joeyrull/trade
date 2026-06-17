// Pillar groupings for the Lab Report scorecard, inspired by how labs like
// Driveline/Maven group a delivery into a handful of categories, each scored
// against an "average" and "elite" benchmark band.

function elbowHeightAtMER(summary, frames) {
  const merF = summary.key_frames?.max_ext_rot ?? 0;
  return frames[merF]?.metrics?.elbow_height_pct ?? 0;
}

export const PILLARS = [
  {
    name: 'Lower Half',
    description: 'How explosively the hips initiate the delivery — the first link in the kinetic chain.',
    metrics: [
      {
        label: 'Hip Rotation Speed',
        unit: '°/s',
        avg: 250,
        good: 550,
        getValue: (summary) => summary.peak?.max_hip_rotation_speed,
        getLowConfidence: (summary) => summary.peak?.max_hip_rotation_speed_low_confidence,
      },
    ],
  },
  {
    name: 'Sequencing',
    description: 'How efficiently energy transfers from hips → chest → arm, building the "whip" through the body.',
    metrics: [
      {
        label: 'Chest Rotation Speed',
        unit: '°/s',
        avg: 350,
        good: 700,
        getValue: (summary) => summary.peak?.max_chest_rotation_speed,
        getLowConfidence: (summary) => summary.peak?.max_chest_rotation_speed_low_confidence,
      },
      {
        label: 'Hip-Shoulder Separation',
        unit: '°',
        avg: 18,
        good: 32,
        getValue: (summary) => summary.peak?.max_hip_shoulder_sep,
        getLowConfidence: (summary) => summary.peak?.max_hss_low_confidence,
      },
      {
        label: 'Hip → Arm Timing',
        unit: 'ms',
        avg: 250,
        good: 120,
        lowerIsBetter: true,
        getValue: (summary) => summary.sequencing?.hip_to_arm_ms,
        getLowConfidence: (summary) =>
          summary.peak?.max_hip_rotation_speed_low_confidence ||
          summary.peak?.max_chest_rotation_speed_low_confidence ||
          summary.peak?.max_arm_speed_low_confidence,
        // Negative timing looks "fast" numerically, but if the kinetic chain
        // fired out of order (arm before hip/chest), the timing isn't good —
        // it's backwards. Override the score so it doesn't read as "Elite".
        scoreOverride: (summary) => summary.sequencing?.proper_order === false ? 15 : null,
        note: (summary) => summary.sequencing?.proper_order === false
          ? 'Kinetic chain fired out of order (arm before hips/chest) — timing should not be read as fast.'
          : null,
      },
    ],
  },
  {
    name: 'Arm Action',
    description: 'Arm speed and elbow positioning through acceleration and release.',
    metrics: [
      {
        label: 'Arm Speed',
        unit: '°/s',
        avg: 400,
        good: 700,
        getValue: (summary) => summary.peak?.max_arm_speed,
        getLowConfidence: (summary) => summary.peak?.max_arm_speed_low_confidence,
      },
      {
        label: 'Elbow Height at Max External Rotation',
        unit: '% above shoulder',
        avg: 0,
        good: 5,
        getValue: elbowHeightAtMER,
        getLowConfidence: (summary, frames) =>
          frames[summary.key_frames?.max_ext_rot ?? 0]?.metrics?.low_confidence ?? false,
      },
    ],
  },
];
