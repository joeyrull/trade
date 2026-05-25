const axios = require('axios');

const BASE = 'https://api.unsplash.com';

function client() {
  return axios.create({
    baseURL: BASE,
    headers: { Authorization: `Client-ID ${process.env.UNSPLASH_ACCESS_KEY}` },
  });
}

async function search(query, { page = 1, perPage = 20, orientation = 'portrait' } = {}) {
  const { data } = await client().get('/search/photos', {
    params: { query, page, per_page: perPage, orientation },
  });
  return data.results.map(normalizePhoto);
}

async function trending({ page = 1, perPage = 20 } = {}) {
  const { data } = await client().get('/photos', {
    params: { order_by: 'popular', page, per_page: perPage, orientation: 'portrait' },
  });
  return data.map(normalizePhoto);
}

function normalizePhoto(p) {
  return {
    id: `unsplash_${p.id}`,
    source: 'unsplash',
    type: 'photo',
    thumb: p.urls.small,
    preview: p.urls.regular,
    full: p.urls.full,
    downloadUrl: p.links.download,
    width: p.width,
    height: p.height,
    description: p.description || p.alt_description || '',
    credit: { name: p.user.name, url: p.user.links.html },
    tags: (p.tags || []).map(t => t.title),
    color: p.color,
  };
}

module.exports = { search, trending };
