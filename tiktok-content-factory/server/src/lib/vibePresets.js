const VIBES = {
  'live-it': {
    name: 'Live It',
    description: 'Colorful, retro-warm energy',
    cssFilter: 'saturate(1.6) sepia(0.15) brightness(1.05) contrast(1.1) hue-rotate(8deg)',
    sharp: {
      modulate: { saturation: 1.6, brightness: 1.05, hue: 8 },
      linear: { multiplier: 1.1, offset: -8 },
    },
    overlay: { r: 255, g: 180, b: 80, alpha: 0.08 },
    grain: 0.04,
  },
  'intense': {
    name: 'Intense',
    description: 'High-contrast, dramatic desaturated look',
    cssFilter: 'saturate(0.15) contrast(2) brightness(0.9)',
    sharp: {
      modulate: { saturation: 0.15, brightness: 0.9 },
      linear: { multiplier: 2.0, offset: -80 },
    },
    overlay: { r: 0, g: 0, b: 30, alpha: 0.12 },
    grain: 0.02,
  },
};

module.exports = { VIBES };
