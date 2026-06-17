// Shared color palette for the warm "amber/copper analytics" theme. Keep this
// in sync with the CSS custom properties defined in :root (App.css) — CSS vars
// can't be imported into JS, so the dark-UI tones are duplicated here for use
// in inline styles (SVG/recharts strokes, badges, etc).

// Per-phase colors used in the HUD badge, timeline, and chart reference lines.
// All are dark/muted enough to host white text. The set spans the warm
// amber/copper/bronze family with a couple of cool counterpoints (teal, slate)
// so the eight phases stay distinguishable.
export const PHASE_COLORS = {
  setup:          '#5c5444',
  windup:         '#8a6d3a',
  stride:         '#4f7a72',
  foot_strike:    '#6f8a45',
  arm_cocking:    '#a8743c',
  acceleration:   '#aa4f3a',
  release:        '#8a5a78',
  follow_through: '#5a6480',
};

// Metric/chart line colors — warm-dominant, with sage and dusty-blue as cool
// counterpoints so nine simultaneous lines remain separable.
export const METRIC_COLORS = {
  hipSpeed:    '#6b9a8a', // sage teal
  chestSpeed:  '#d4a24e', // amber
  armSpeed:    '#c2685f', // copper red
  hss:         '#8caa4e', // warm olive green
  hip:         '#b08968', // bronze
  shoulder:    '#7a92a8', // dusty blue
  elbowH:      '#b08a92', // warm mauve
  elbowAngle:  '#c9a87e', // tan
  trunkTilt:   '#c99a52', // ochre
  armSlot:     '#9a7bb0', // muted violet
  leadKneeExt: '#5f9ac2', // steel blue
};

// A/B/C grade colors
export const GRADE_COLORS = {
  A: '#8caa4e',
  B: '#c9a04e',
  C: '#c2685f',
};

export function gradeColor(letter) {
  return GRADE_COLORS[letter] || GRADE_COLORS.C;
}

// Composite-score rating bands (Lab Report scorecard)
export const RATING_COLORS = {
  elite:   '#8caa4e',
  average: '#c9a04e',
  needsWork: '#c2825f',
};

// Live-metrics good/bad indicator colors
export const STATUS_COLORS = {
  good: '#8caa4e',
  bad:  '#c2825f',
  frozen: '#5c5444',
};
