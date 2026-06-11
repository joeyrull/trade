// Shared color palette for the "refined dark analytics" theme. Keep this in
// sync with the CSS custom properties defined in :root (App.css) — CSS vars
// can't be imported into JS, so the dark-UI tones are duplicated here for use
// in inline styles (SVG/recharts strokes, badges, etc).

// Per-phase colors used in the HUD badge, timeline, and chart reference lines.
// All are dark/muted enough to host white text.
export const PHASE_COLORS = {
  setup:          '#555f6e',
  windup:         '#9c8a3f',
  stride:         '#3f7f99',
  foot_strike:    '#4f8f6b',
  arm_cocking:    '#a36a3c',
  acceleration:   '#a8453a',
  release:        '#7a5a9e',
  follow_through: '#5a649c',
};

// Metric/chart line colors
export const METRIC_COLORS = {
  hipSpeed:    '#5aa3b0', // muted cyan
  chestSpeed:  '#c99a52', // muted amber
  armSpeed:    '#c2685f', // muted red/terracotta
  hss:         '#5fae7d', // muted green
  hip:         '#bd7a48', // muted burnt orange
  shoulder:    '#5b8bc4', // muted steel blue
  elbowH:      '#8b7aa8', // muted purple
  elbowAngle:  '#a594c2', // muted lavender
  trunkTilt:   '#b3935a', // muted ochre
};

// A/B/C grade colors
export const GRADE_COLORS = {
  A: '#4f8f63',
  B: '#a3964f',
  C: '#b3624a',
};

export function gradeColor(letter) {
  return GRADE_COLORS[letter] || GRADE_COLORS.C;
}

// Composite-score rating bands (Lab Report scorecard)
export const RATING_COLORS = {
  elite:   '#5fae7d',
  average: '#a3b35a',
  needsWork: '#c2825f',
};

// Live-metrics good/bad indicator colors
export const STATUS_COLORS = {
  good: '#5fae7d',
  bad:  '#c2825f',
  frozen: '#5a6172',
};
