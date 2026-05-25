const sharp = require('sharp');
const axios = require('axios');
const path = require('path');
const fs = require('fs').promises;
const { VIBES } = require('./vibePresets');

async function fetchImageBuffer(urlOrPath) {
  if (urlOrPath.startsWith('http')) {
    const { data } = await axios.get(urlOrPath, { responseType: 'arraybuffer' });
    return Buffer.from(data);
  }
  return fs.readFile(urlOrPath);
}

async function applyVibe(urlOrPath, vibeKey, outputPath) {
  const preset = VIBES[vibeKey];
  if (!preset) throw new Error(`Unknown vibe: ${vibeKey}`);

  const inputBuffer = await fetchImageBuffer(urlOrPath);
  const meta = await sharp(inputBuffer).metadata();

  let pipeline = sharp(inputBuffer);

  pipeline = pipeline.modulate(preset.sharp.modulate);

  const { multiplier, offset } = preset.sharp.linear;
  pipeline = pipeline.linear(multiplier, offset);

  if (preset.overlay) {
    const { r, g, b, alpha } = preset.overlay;
    const overlayBuffer = await sharp({
      create: { width: meta.width, height: meta.height, channels: 4, background: { r, g, b, alpha: Math.round(alpha * 255) } },
    }).png().toBuffer();

    pipeline = pipeline.composite([{ input: overlayBuffer, blend: 'over' }]);
  }

  if (preset.grain > 0) {
    pipeline = pipeline.convolve({
      width: 3,
      height: 3,
      kernel: [0, 0, 0, 0, 1 + preset.grain * 8, 0, 0, 0, 0],
    });
  }

  if (outputPath) {
    await fs.mkdir(path.dirname(outputPath), { recursive: true });
    await pipeline.toFile(outputPath);
    return { outputPath };
  }

  const buffer = await pipeline.toBuffer();
  return { buffer, contentType: 'image/jpeg' };
}

async function generateTikTokCrop(urlOrPath, outputPath) {
  const inputBuffer = await fetchImageBuffer(urlOrPath);
  const meta = await sharp(inputBuffer).metadata();

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

  const pipeline = sharp(inputBuffer)
    .extract({ left, top, width: cropWidth, height: cropHeight })
    .resize(1080, 1920, { fit: 'fill' });

  if (outputPath) {
    await fs.mkdir(path.dirname(outputPath), { recursive: true });
    await pipeline.toFile(outputPath);
    return { outputPath };
  }

  const buffer = await pipeline.toBuffer();
  return { buffer, contentType: 'image/jpeg' };
}

module.exports = { applyVibe, generateTikTokCrop };
