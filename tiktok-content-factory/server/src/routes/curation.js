const express = require('express');
const fs = require('fs').promises;
const path = require('path');
const { randomUUID } = require('crypto');

const router = express.Router();
const STORAGE = path.join(process.env.STORAGE_PATH || './data', 'collections.json');

async function loadCollections() {
  try {
    const raw = await fs.readFile(STORAGE, 'utf8');
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

async function saveCollections(collections) {
  await fs.mkdir(path.dirname(STORAGE), { recursive: true });
  await fs.writeFile(STORAGE, JSON.stringify(collections, null, 2));
}

router.get('/collections', async (req, res) => {
  const collections = await loadCollections();
  res.json(collections);
});

router.post('/collections', async (req, res) => {
  const { name, vibe, description = '' } = req.body;
  if (!name) return res.status(400).json({ error: 'Collection name is required' });

  const collections = await loadCollections();
  const collection = {
    id: randomUUID(),
    name,
    vibe: vibe || null,
    description,
    items: [],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
  collections.push(collection);
  await saveCollections(collections);
  res.status(201).json(collection);
});

router.get('/collections/:id', async (req, res) => {
  const collections = await loadCollections();
  const collection = collections.find(c => c.id === req.params.id);
  if (!collection) return res.status(404).json({ error: 'Collection not found' });
  res.json(collection);
});

router.delete('/collections/:id', async (req, res) => {
  let collections = await loadCollections();
  const before = collections.length;
  collections = collections.filter(c => c.id !== req.params.id);
  if (collections.length === before) return res.status(404).json({ error: 'Collection not found' });
  await saveCollections(collections);
  res.status(204).end();
});

router.post('/collections/:id/items', async (req, res) => {
  const { item } = req.body;
  if (!item) return res.status(400).json({ error: 'Item data is required' });

  const collections = await loadCollections();
  const collection = collections.find(c => c.id === req.params.id);
  if (!collection) return res.status(404).json({ error: 'Collection not found' });

  if (collection.items.some(i => i.id === item.id)) {
    return res.status(409).json({ error: 'Item already in collection' });
  }

  collection.items.push({ ...item, savedAt: new Date().toISOString() });
  collection.updatedAt = new Date().toISOString();
  await saveCollections(collections);
  res.status(201).json(collection);
});

router.delete('/collections/:id/items/:itemId', async (req, res) => {
  const collections = await loadCollections();
  const collection = collections.find(c => c.id === req.params.id);
  if (!collection) return res.status(404).json({ error: 'Collection not found' });

  const before = collection.items.length;
  collection.items = collection.items.filter(i => i.id !== req.params.itemId);
  if (collection.items.length === before) return res.status(404).json({ error: 'Item not found' });

  collection.updatedAt = new Date().toISOString();
  await saveCollections(collections);
  res.json(collection);
});

module.exports = router;
