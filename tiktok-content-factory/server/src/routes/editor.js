const express = require('express');
const multer = require('multer');
const path = require('path');
const { VIBES } = require('../lib/vibePresets');
const { applyVibe, generateTikTokCrop } = require('../lib/imageProcessor');

const router = express.Router();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 20 * 1024 * 1024 } });

router.get('/vibes', (req, res) => {
  const vibes = Object.entries(VIBES).map(([key, v]) => ({
    key,
    name: v.name,
    description: v.description,
    cssFilter: v.cssFilter,
  }));
  res.json(vibes);
});

router.post('/apply-vibe', upload.single('image'), async (req, res) => {
  const { vibe, imageUrl, crop = 'false' } = req.body;

  if (!vibe) return res.status(400).json({ error: 'vibe is required' });
  if (!VIBES[vibe]) return res.status(400).json({ error: `Unknown vibe. Valid: ${Object.keys(VIBES).join(', ')}` });

  try {
    const source = req.file ? req.file.buffer : imageUrl;
    if (!source) return res.status(400).json({ error: 'Provide image file or imageUrl' });

    let inputSource = source;

    if (crop === 'true') {
      const cropResult = await generateTikTokCrop(source, null);
      inputSource = cropResult.buffer;
    }

    const result = await applyVibe(inputSource, vibe, null);

    res.set('Content-Type', result.contentType);
    res.set('Content-Disposition', `attachment; filename="vibe-${vibe}.jpg"`);
    res.send(result.buffer);
  } catch (err) {
    console.error('Editor apply-vibe error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

router.post('/crop', upload.single('image'), async (req, res) => {
  const { imageUrl } = req.body;

  try {
    const source = req.file ? req.file.buffer : imageUrl;
    if (!source) return res.status(400).json({ error: 'Provide image file or imageUrl' });

    const result = await generateTikTokCrop(source, null);
    res.set('Content-Type', result.contentType);
    res.set('Content-Disposition', 'attachment; filename="tiktok-crop.jpg"');
    res.send(result.buffer);
  } catch (err) {
    console.error('Editor crop error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
