const sharp = require('sharp');
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');

const VIBES = {
  'live-it': {
    name: 'Live It',
    modulate: { saturation: 1.6, brightness: 1.05, hue: 8 },
    linear: { multiplier: 1.1, offset: -8 },
    overlay: { r: 255, g: 180, b: 80, alpha: 20 },
  },
  'intense': {
    name: 'Intense',
    modulate: { saturation: 0.15, brightness: 0.9 },
    linear: { multiplier: 2.0, offset: -80 },
    overlay: { r: 0, g: 0, b: 30, alpha: 30 },
  },
};

async function fetchBuffer(urlOrPath) {
  if (urlOrPath.startsWith('http')) {
    const { data } = await axios.get(urlOrPath, { responseType: 'arraybuffer' });
    return Buffer.from(data);
  }
  return fs.readFile(urlOrPath);
}

function parseArgs(args) {
  const opts = { vibe: 'live-it', out: null, crop: false };
  let source = '';
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--vibe') opts.vibe = args[++i];
    else if (args[i] === '--out') opts.out = args[++i];
    else if (args[i] === '--crop') opts.crop = true;
    else if (!args[i].startsWith('--')) source = args[i];
  }
  return { source, opts };
}

module.exports = async function edit(args) {
  const { source, opts } = parseArgs(args);

  if (!source) {
    console.log('Usage: tcf edit <imageUrl|path> [--vibe live-it|intense] [--out output.jpg] [--crop]');
    console.log('\nVibes: live-it (colorful/retro), intense (high-contrast)');
    return;
  }

  const preset = VIBES[opts.vibe];
  if (!preset) {
    console.error(`Unknown vibe: ${opts.vibe}. Valid: ${Object.keys(VIBES).join(', ')}`);
    process.exit(1);
  }

  console.log(`\nApplying "${preset.name}" vibe...`);
  const inputBuffer = await fetchBuffer(source);
  const meta = await sharp(inputBuffer).metadata();

  let pipeline = sharp(inputBuffer);

  if (opts.crop) {
    const targetRatio = 9 / 16;
    const currentRatio = meta.width / meta.height;
    let cropWidth, cropHeight, left, top;
    if (currentRatio > targetRatio) {
      cropHeight = meta.height;
      cropWidth = Math.round(meta.height * targetRatio);
      left = Math.round((meta.width - cropWidth) / 2);
      top = 0;
    } else {
      cropWidth = meta.width;
      cropHeight = Math.round(meta.width / targetRatio);
      left = 0;
      top = Math.round((meta.height - cropHeight) / 3);
    }
    pipeline = pipeline.extract({ left, top, width: cropWidth, height: cropHeight }).resize(1080, 1920);
    console.log('Cropped to 9:16 TikTok format (1080x1920)');
  }

  pipeline = pipeline
    .modulate(preset.modulate)
    .linear(preset.linear.multiplier, preset.linear.offset);

  if (preset.overlay) {
    const { r, g, b, alpha } = preset.overlay;
    const w = opts.crop ? 1080 : meta.width;
    const h = opts.crop ? 1920 : meta.height;
    const overlayBuffer = await sharp({
      create: { width: w, height: h, channels: 4, background: { r, g, b, alpha } },
    }).png().toBuffer();
    pipeline = pipeline.composite([{ input: overlayBuffer, blend: 'over' }]);
  }

  const outPath = opts.out || `./output-${opts.vibe}-${Date.now()}.jpg`;
  await fs.mkdir(path.dirname(path.resolve(outPath)), { recursive: true });
  await pipeline.jpeg({ quality: 90 }).toFile(outPath);

  console.log(`Done! Saved to: ${outPath}`);
};
