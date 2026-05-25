const express = require('express');
const unsplash = require('../lib/unsplash');
const pexels = require('../lib/pexels');

const router = express.Router();

router.get('/search', async (req, res) => {
  const { q, source = 'both', type = 'photo', page = 1, perPage = 20 } = req.query;
  if (!q) return res.status(400).json({ error: 'Query parameter "q" is required' });

  const opts = { page: parseInt(page), perPage: parseInt(perPage) };

  try {
    let results = [];

    if (source === 'unsplash' || source === 'both') {
      const photos = await unsplash.search(q, opts);
      results = results.concat(photos);
    }

    if (source === 'pexels' || source === 'both') {
      if (type === 'video') {
        const videos = await pexels.searchVideos(q, opts);
        results = results.concat(videos);
      } else {
        const photos = await pexels.searchPhotos(q, opts);
        results = results.concat(photos);
      }
    }

    res.json({ results, total: results.length, page: parseInt(page) });
  } catch (err) {
    console.error('Discovery search error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

router.get('/trending', async (req, res) => {
  const { source = 'both', page = 1 } = req.query;
  const opts = { page: parseInt(page), perPage: 20 };

  try {
    let results = [];

    if (source === 'unsplash' || source === 'both') {
      const photos = await unsplash.trending(opts);
      results = results.concat(photos);
    }

    if (source === 'pexels' || source === 'both') {
      const photos = await pexels.trending(opts);
      results = results.concat(photos);
    }

    res.json({ results, total: results.length });
  } catch (err) {
    console.error('Discovery trending error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
