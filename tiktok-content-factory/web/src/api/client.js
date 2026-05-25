const BASE = '/api';

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function del(path) {
  const res = await fetch(`${BASE}${path}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(await res.text());
  return res.status === 204 ? null : res.json();
}

export const discovery = {
  search: (q, { source = 'both', type = 'photo', page = 1 } = {}) =>
    get(`/discovery/search?q=${encodeURIComponent(q)}&source=${source}&type=${type}&page=${page}`),
  trending: (source = 'both') =>
    get(`/discovery/trending?source=${source}`),
};

export const curation = {
  list: () => get('/curation/collections'),
  create: (name, vibe, description) => post('/curation/collections', { name, vibe, description }),
  deleteCollection: (id) => del(`/curation/collections/${id}`),
  addItem: (collectionId, item) => post(`/curation/collections/${collectionId}/items`, { item }),
  removeItem: (collectionId, itemId) => del(`/curation/collections/${collectionId}/items/${itemId}`),
};

export const editor = {
  vibes: () => get('/editor/vibes'),
  applyVibe: async (imageUrl, vibe, crop = false) => {
    const res = await fetch(`${BASE}/editor/apply-vibe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageUrl, vibe, crop: String(crop) }),
    });
    if (!res.ok) throw new Error(await res.text());
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  },
};
