// Lab-report metric definitions: what each metric means, how it's graded,
// and drills/cues for improving it. Mirrors the per-frame fields produced
// by MetricsChart's chartData (hip, shoulder, hss, hipSpeed, chestSpeed,
// armSpeed, elbowAngle, elbowH, trunkTilt).

import { METRIC_COLORS } from '../theme';

export const METRICS_LIBRARY = [
  {
    key: 'hipSpeed',
    label: 'Hip Rotation Speed',
    unit: '°/s',
    color: METRIC_COLORS.hipSpeed,
    aggregate: 'max',
    grade: { good: 550, avg: 250 },
    definition:
      'How fast the pelvis rotates toward home plate during the delivery. ' +
      'This is the first major rotational link in the kinetic chain — the hips should ' +
      'fire first and fastest, transferring energy up through the trunk, shoulders, and arm.',
    reference: '581° – 811°/s (MLB average peak pelvis angular velocity)',
    drills: [
      'Med ball rotational throws (standing and step-behind) to train explosive hip drive.',
      'Resisted band hip rotations to build pelvis rotation speed and strength.',
      'Stride-and-stick drills focusing on driving the lead hip open hard at foot strike.',
      'Hip-shoulder separation drills (e.g., "separation throws") that delay shoulder rotation while hips fire early.',
    ],
  },
  {
    key: 'chestSpeed',
    label: 'Chest / Shoulder Rotation Speed',
    unit: '°/s',
    color: METRIC_COLORS.chestSpeed,
    aggregate: 'max',
    grade: { good: 700, avg: 350 },
    definition:
      'The angular velocity of the upper torso (shoulder line) as it rotates toward the plate. ' +
      'In an efficient kinetic chain, the chest reaches its peak rotation speed after the hips, ' +
      'amplifying the energy passed up from the lower half before it reaches the arm. ' +
      'Pro mocap (3D marker-based) typically reads higher than this app\'s single-camera ' +
      'pose estimate, so treat the good/average bands below as calibrated to this app\'s own ' +
      'scale rather than an absolute MLB cutoff.',
    reference: '861° – 1187°/s (MLB average peak trunk angular velocity, 3D marker-based mocap)',
    drills: [
      'Rotational med ball scoop throws to build torso rotational power.',
      'Cable or band torso rotations emphasizing speed, not just resistance.',
      'Towel/wall drills that exaggerate hip-to-shoulder separation before the chest opens.',
      'Video review focused on delaying shoulder rotation until front foot plant.',
    ],
  },
  {
    key: 'armSpeed',
    label: 'Arm Speed (Elbow Extension Rate)',
    unit: '°/s',
    color: METRIC_COLORS.armSpeed,
    aggregate: 'max',
    grade: { good: 700, avg: 400 },
    definition:
      'How quickly the elbow extends during the arm-acceleration phase, just before release. ' +
      'This reflects the "whip" of the arm at the end of the kinetic chain — a product of everything ' +
      'that happened in the hips, trunk, and shoulder up to this point, not just arm strength. ' +
      'Pro 3D marker-based mocap measures elbow extension several times faster than a single-camera ' +
      'pose estimate can resolve, so the good/average bands below are calibrated to this app\'s own ' +
      'scale rather than an absolute MLB cutoff.',
    reference: '2030° – 2542°/s (MLB average peak elbow extension velocity, 3D marker-based mocap)',
    drills: [
      'Plyo-care/weighted ball drills (under qualified supervision) to train arm-speed acceleration.',
      'Long toss programs to build arm speed across distances.',
      'Wrist flip / pull-down drills emphasizing a quick, loose arm path through release.',
      'Full kinetic-chain mechanics work — arm speed often improves most by fixing hip/trunk sequencing, not arm-only drills.',
    ],
  },
  {
    key: 'hss',
    label: 'Hip-Shoulder Separation (X-Factor)',
    unit: '°',
    color: METRIC_COLORS.hss,
    aggregate: 'max',
    grade: { good: 32, avg: 18 },
    definition:
      'The angular difference between hip rotation and shoulder rotation at any instant. ' +
      'Elite pitchers create a large separation (the hips open while the shoulders stay closed), ' +
      'storing elastic energy in the trunk that whips the shoulders and arm through afterward.',
    reference: '32° – 52° (MLB average range at front-foot plant)',
    drills: [
      '"Stride and hold" drills — stride out and rotate the hips while keeping the shoulders closed for a beat.',
      'Med ball separation throws: rotate the hips first, delay the upper body turn.',
      'Resistance band drills anchoring the lead shoulder while driving the hip open.',
      'Slow-motion shadow pitching focused purely on the timing gap between hip and shoulder rotation.',
    ],
  },
  {
    key: 'hip',
    label: 'Hip Rotation Angle',
    unit: '°',
    color: METRIC_COLORS.hip,
    aggregate: 'range',
    definition:
      'The orientation of the pelvis relative to home plate over time. Watching this curve shows ' +
      'how and when the hips open through the delivery — a smooth, progressively increasing curve ' +
      'that opens up around foot strike is typical of good lower-half sequencing.',
    drills: [
      'Balance and stride drills to ensure the hips stay closed during the leg lift and open on time at foot strike.',
      'Video comparison drills — overlay your hip-rotation curve against a model pitcher to spot early or late hip opening.',
      'Hip mobility work (90/90 stretches, hip airplanes) to allow a fuller range of rotation.',
    ],
  },
  {
    key: 'shoulder',
    label: 'Shoulder Rotation Angle',
    unit: '°',
    color: METRIC_COLORS.shoulder,
    aggregate: 'range',
    definition:
      'The orientation of the shoulder line relative to home plate over time. The gap between this ' +
      'curve and the hip-rotation curve is the hip-shoulder separation (X-factor) — ideally the ' +
      'shoulder curve lags noticeably behind the hip curve through foot strike before catching up quickly.',
    drills: [
      '"Stay closed" drills that delay shoulder rotation until the front foot plants.',
      'Band-resisted shoulder turns to feel the upper body rotating independently of the hips.',
      'Video review comparing hip vs. shoulder timing frame-by-frame.',
    ],
  },
  {
    key: 'elbowH',
    label: 'Elbow Height',
    unit: '% above shoulder',
    color: METRIC_COLORS.elbowH,
    aggregate: 'max',
    grade: { good: 5, avg: 0 },
    definition:
      'The throwing elbow\'s height relative to the shoulder line, expressed as a percentage. ' +
      'Positive values mean the elbow is at or above shoulder height. Keeping the elbow up during ' +
      'arm cocking and acceleration is associated with reduced stress on the elbow (UCL) and shoulder.',
    drills: [
      '"High-elbow" wall drills — drive the throwing elbow up to shoulder height before the arm comes forward.',
      'Towel drills emphasizing an early, tall arm path rather than a dropped/short-armed action.',
      'External rotation mobility work for the throwing shoulder to make a high-elbow position easier to reach.',
      'Slow-motion video checks at max external rotation to confirm the elbow is at or above the shoulder line.',
    ],
  },
  {
    key: 'elbowAngle',
    label: 'Elbow Flexion Angle',
    unit: '°',
    color: METRIC_COLORS.elbowAngle,
    aggregate: 'range',
    definition:
      'The bend at the elbow joint over time (180° = fully extended, smaller values = more bent). ' +
      'A typical pattern shows the elbow flexed during arm cocking, then rapidly extending toward ' +
      '180° around release. A smooth, late extension pattern is generally desirable.',
    reference:
      '~56°-86° at foot plant, ~86°-106° at max external rotation, ~150°-160° at release ' +
      '(MLB average, converted from Kinatrax\'s straight-arm = 0° convention to this app\'s 180°)',
    drills: [
      'Wrist-weight or light-band extension drills to train a quick, late elbow snap toward release.',
      'Drop-and-drive arm circles to groove a natural flexion-to-extension pattern.',
      'Video review checking that full extension happens at or near release, not too early.',
    ],
  },
  {
    key: 'trunkTilt',
    label: 'Trunk Tilt',
    unit: '°',
    color: METRIC_COLORS.trunkTilt,
    aggregate: 'range',
    definition:
      'The forward/lateral lean of the torso through the delivery. Trunk tilt influences release ' +
      'height, extension toward the plate, and the plane of pitches. Excessive early tilt or a ' +
      'sudden change near release can be a sign of inefficient posture or compensation.',
    drills: [
      'Posture drills (e.g., wall drills) to maintain a stable spine angle from leg lift through foot strike.',
      'Core stability work (planks, anti-rotation presses) to control trunk position during rotation.',
      'Video review comparing trunk angle at foot strike vs. release to check for excessive late collapse.',
    ],
  },
];
