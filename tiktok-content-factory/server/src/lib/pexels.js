const axios = require('axios');

const BASE = 'https://api.pexels.com';

function client() {
  return axios.create({
    baseURL: BASE,
    headers: { Authorization: process.env.PEXELS_API_KEY },
  });
}

async function searchPhotos(query, { page = 1, perPage = 20, orientation = 'portrait' } = {}) {
  const { data } = await client().get('/v1/search', {
    params: { query, page, per_page: perPage, orientation },
  });
  return data.photos.map(normalizePhoto);
}

async function searchVideos(query, { page = 1, perPage = 20, orientation = 'portrait' } = {}) {
  const { data } = await client().get('/videos/search', {
    params: { query, page, per_page: perPage, orientation },
  });
  return data.videos.map(normalizeVideo);
}

async function trending({ page = 1, perPage = 20 } = {}) {
  const { data } = await client().get('/v1/curated', {
    params: { page, per_page: perPage },
  });
  return data.photos.map(normalizePhoto);
}

function normalizePhoto(p) {
  return {
    id: `pexels_${p.id}`,
    source: 'pexels',
    type: 'photo',
    thumb: p.src.small,
    preview: p.src.large,
    full: p.src.original,
    downloadUrl: p.src.original,
    width: p.width,
    height: p.height,
    description: p.alt || '',
    credit: { name: p.photographer, url: p.photographer_url },
    tags: [],
    color: p.avg_color,
  };
}

function normalizeVideo(v) {
  const file = v.video_files.find(f => f.quality === 'hd') || v.video_files[0];
  return {
    id: `pexels_video_${v.id}`,
    source: 'pexels',
    type: 'video',
    thumb: v.image,
    preview: v.image,
    full: file?.link,
    downloadUrl: file?.link,
    width: v.width,
    height: v.height,
    duration: v.duration,
    description: '',
    credit: { name: v.user.name, url: v.user.url },
    tags: [],
  };
}

module.exports = { searchPhotos, searchVideos, trending };
