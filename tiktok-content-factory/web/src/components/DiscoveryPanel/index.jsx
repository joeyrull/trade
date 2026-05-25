import React, { useState, useEffect } from 'react';
import MediaGrid from '../MediaGrid/index.jsx';
import { discovery, curation } from '../../api/client.js';
import './DiscoveryPanel.css';

export default function DiscoveryPanel({ onEdit }) {
  const [query, setQuery] = useState('');
  const [source, setSource] = useState('both');
  const [type, setType] = useState('photo');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [collections, setCollections] = useState([]);
  const [savedIds, setSavedIds] = useState(new Set());
  const [saveTarget, setSaveTarget] = useState('');
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [pendingItem, setPendingItem] = useState(null);

  useEffect(() => {
    loadTrending();
    loadCollections();
  }, []);

  async function loadTrending() {
    setLoading(true);
    setError(null);
    try {
      const data = await discovery.trending();
      setResults(data.results);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadCollections() {
    try {
      const cols = await curation.list();
      setCollections(cols);
      const ids = new Set(cols.flatMap(c => c.items.map(i => i.id)));
      setSavedIds(ids);
    } catch {}
  }

  async function handleSearch(e) {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await discovery.search(query, { source, type });
      setResults(data.results);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function handleSave(item) {
    setPendingItem(item);
    setShowSaveModal(true);
    setSaveTarget(collections[0]?.id || '');
  }

  async function confirmSave() {
    if (!saveTarget || !pendingItem) return;
    try {
      await curation.addItem(saveTarget, pendingItem);
      setSavedIds(prev => new Set([...prev, pendingItem.id]));
      setShowSaveModal(false);
    } catch (e) {
      alert(e.message);
    }
  }

  return (
    <div className="panel discovery">
      <div className="panel-header">
        <h1 className="panel-title">Discover Content</h1>
      </div>

      <form className="search-bar" onSubmit={handleSearch}>
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Search photos & videos..."
          className="search-input"
        />
        <select value={source} onChange={e => setSource(e.target.value)}>
          <option value="both">All Sources</option>
          <option value="unsplash">Unsplash</option>
          <option value="pexels">Pexels</option>
        </select>
        <select value={type} onChange={e => setType(e.target.value)}>
          <option value="photo">Photos</option>
          <option value="video">Videos</option>
        </select>
        <button type="submit" className="btn">Search</button>
        <button type="button" className="btn secondary" onClick={loadTrending}>Trending</button>
      </form>

      {error && <div className="error-msg">{error}</div>}

      {loading
        ? <div className="loading">Fetching content...</div>
        : <MediaGrid items={results} onEdit={onEdit} onSave={handleSave} savedIds={savedIds} />
      }

      {showSaveModal && (
        <div className="modal-backdrop" onClick={() => setShowSaveModal(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <h3>Save to Collection</h3>
            {collections.length === 0
              ? <p className="modal-hint">No collections yet — create one in the Collections tab.</p>
              : <>
                  <select value={saveTarget} onChange={e => setSaveTarget(e.target.value)}>
                    {collections.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                  </select>
                  <div className="modal-actions">
                    <button className="btn" onClick={confirmSave}>Save</button>
                    <button className="btn secondary" onClick={() => setShowSaveModal(false)}>Cancel</button>
                  </div>
                </>
            }
          </div>
        </div>
      )}
    </div>
  );
}
