import React, { useState, useEffect } from 'react';
import MediaGrid from '../MediaGrid/index.jsx';
import { curation } from '../../api/client.js';
import './CurationDashboard.css';

export default function CurationDashboard({ onEdit }) {
  const [collections, setCollections] = useState([]);
  const [active, setActive] = useState(null);
  const [newName, setNewName] = useState('');
  const [newVibe, setNewVibe] = useState('');
  const [showCreate, setShowCreate] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => { load(); }, []);

  async function load() {
    setLoading(true);
    try {
      const cols = await curation.list();
      setCollections(cols);
      if (!active && cols.length) setActive(cols[0].id);
    } finally {
      setLoading(false);
    }
  }

  async function createCollection() {
    if (!newName.trim()) return;
    try {
      const col = await curation.create(newName.trim(), newVibe || null);
      setCollections(prev => [...prev, col]);
      setActive(col.id);
      setNewName('');
      setNewVibe('');
      setShowCreate(false);
    } catch (e) {
      alert(e.message);
    }
  }

  async function deleteCollection(id) {
    if (!confirm('Delete this collection?')) return;
    try {
      await curation.deleteCollection(id);
      const updated = collections.filter(c => c.id !== id);
      setCollections(updated);
      if (active === id) setActive(updated[0]?.id || null);
    } catch (e) {
      alert(e.message);
    }
  }

  async function removeItem(itemId) {
    if (!active) return;
    try {
      const updated = await curation.removeItem(active, itemId);
      setCollections(prev => prev.map(c => c.id === active ? updated : c));
    } catch (e) {
      alert(e.message);
    }
  }

  const activeCol = collections.find(c => c.id === active);

  return (
    <div className="panel curation">
      <div className="panel-header">
        <h1 className="panel-title">Collections</h1>
        <button className="btn" onClick={() => setShowCreate(true)}>+ New Collection</button>
      </div>

      {loading ? (
        <div className="loading">Loading collections...</div>
      ) : (
        <div className="curation-layout">
          <aside className="collection-list">
            {collections.length === 0 && (
              <p className="empty-hint">No collections yet.</p>
            )}
            {collections.map(col => (
              <div
                key={col.id}
                className={`collection-item${active === col.id ? ' active' : ''}`}
                onClick={() => setActive(col.id)}
              >
                <div className="col-info">
                  <span className="col-name">{col.name}</span>
                  <span className="col-count">{col.items.length} items</span>
                </div>
                {col.vibe && <span className={`tag ${col.vibe}`}>{col.vibe === 'live-it' ? 'Live It' : 'Intense'}</span>}
                <button className="col-del" onClick={e => { e.stopPropagation(); deleteCollection(col.id); }}>✕</button>
              </div>
            ))}
          </aside>

          <div className="collection-content">
            {activeCol ? (
              <>
                <div className="collection-header">
                  <div>
                    <h2 className="col-title">{activeCol.name}</h2>
                    {activeCol.vibe && <span className={`tag ${activeCol.vibe}`}>{activeCol.vibe === 'live-it' ? 'Live It' : 'Intense'}</span>}
                  </div>
                  <span className="col-meta">Updated {new Date(activeCol.updatedAt).toLocaleDateString()}</span>
                </div>
                {activeCol.items.length === 0
                  ? <div className="empty-collection">Save items from Discovery to see them here.</div>
                  : <MediaGrid
                      items={activeCol.items}
                      onEdit={onEdit}
                      onSave={item => removeItem(item.id)}
                      savedIds={new Set(activeCol.items.map(i => i.id))}
                    />
                }
              </>
            ) : (
              <div className="empty-collection">Select or create a collection to get started.</div>
            )}
          </div>
        </div>
      )}

      {showCreate && (
        <div className="modal-backdrop" onClick={() => setShowCreate(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <h3>New Collection</h3>
            <input
              placeholder="Collection name"
              value={newName}
              onChange={e => setNewName(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && createCollection()}
              autoFocus
            />
            <select value={newVibe} onChange={e => setNewVibe(e.target.value)}>
              <option value="">No vibe assigned</option>
              <option value="live-it">Live It (colorful/retro)</option>
              <option value="intense">Intense (high-contrast)</option>
            </select>
            <div className="modal-actions">
              <button className="btn" onClick={createCollection}>Create</button>
              <button className="btn secondary" onClick={() => setShowCreate(false)}>Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
