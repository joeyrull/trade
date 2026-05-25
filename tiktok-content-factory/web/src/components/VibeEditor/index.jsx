import React, { useState, useEffect, useRef } from 'react';
import { editor } from '../../api/client.js';
import './VibeEditor.css';

const VIBES_FALLBACK = [
  { key: 'live-it', name: 'Live It', description: 'Colorful, retro-warm energy', cssFilter: 'saturate(1.6) sepia(0.15) brightness(1.05) contrast(1.1) hue-rotate(8deg)' },
  { key: 'intense', name: 'Intense', description: 'High-contrast, dramatic desaturated look', cssFilter: 'saturate(0.15) contrast(2) brightness(0.9)' },
];

export default function VibeEditor({ initialImage }) {
  const [vibes, setVibes] = useState(VIBES_FALLBACK);
  const [selectedVibe, setSelectedVibe] = useState('live-it');
  const [imageUrl, setImageUrl] = useState(initialImage?.preview || '');
  const [inputUrl, setInputUrl] = useState('');
  const [crop, setCrop] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [resultUrl, setResultUrl] = useState(null);
  const [error, setError] = useState(null);
  const fileRef = useRef();

  useEffect(() => {
    editor.vibes().then(setVibes).catch(() => {});
  }, []);

  useEffect(() => {
    if (initialImage) {
      setImageUrl(initialImage.preview || initialImage.thumb || '');
      setResultUrl(null);
    }
  }, [initialImage]);

  function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    setResultUrl(null);
  }

  async function applyVibe() {
    if (!imageUrl) return;
    setProcessing(true);
    setError(null);
    try {
      const url = await editor.applyVibe(imageUrl, selectedVibe, crop);
      setResultUrl(url);
    } catch (e) {
      setError(e.message);
    } finally {
      setProcessing(false);
    }
  }

  const activeVibe = vibes.find(v => v.key === selectedVibe);
  const previewStyle = imageUrl ? { filter: activeVibe?.cssFilter || 'none' } : {};

  return (
    <div className="panel editor">
      <div className="panel-header">
        <h1 className="panel-title">Vibe Editor</h1>
      </div>

      <div className="editor-layout">
        <div className="editor-controls">
          <section className="control-section">
            <h3 className="section-title">Source Image</h3>
            <div className="source-options">
              <button className="btn secondary" onClick={() => fileRef.current?.click()}>Upload File</button>
              <input ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFileUpload} />
            </div>
            <div className="url-row">
              <input
                placeholder="Or paste image URL..."
                value={inputUrl}
                onChange={e => setInputUrl(e.target.value)}
              />
              <button
                className="btn secondary"
                onClick={() => { setImageUrl(inputUrl); setResultUrl(null); }}
                disabled={!inputUrl}
              >Use</button>
            </div>
          </section>

          <section className="control-section">
            <h3 className="section-title">Vibe</h3>
            <div className="vibe-selector">
              {vibes.map(v => (
                <button
                  key={v.key}
                  className={`vibe-card${selectedVibe === v.key ? ' active' : ''} ${v.key}`}
                  onClick={() => { setSelectedVibe(v.key); setResultUrl(null); }}
                >
                  <div className="vibe-name">{v.name}</div>
                  <div className="vibe-desc">{v.description}</div>
                </button>
              ))}
            </div>
          </section>

          <section className="control-section">
            <h3 className="section-title">Options</h3>
            <label className="checkbox-row">
              <input type="checkbox" checked={crop} onChange={e => setCrop(e.target.checked)} />
              Auto-crop to TikTok 9:16
            </label>
          </section>

          <button
            className="btn apply-btn"
            onClick={applyVibe}
            disabled={!imageUrl || processing}
          >
            {processing ? 'Processing...' : `Apply ${activeVibe?.name || 'Vibe'}`}
          </button>

          {error && <div className="error-msg">{error}</div>}
        </div>

        <div className="editor-preview">
          <div className="preview-pane">
            <div className="preview-label">Original</div>
            {imageUrl
              ? <img src={imageUrl} alt="Original" className="preview-img" />
              : <div className="preview-empty">No image selected</div>
            }
          </div>

          <div className="preview-pane">
            <div className="preview-label">
              Preview
              {activeVibe && <span className={`tag ${selectedVibe} preview-tag`}>{activeVibe.name}</span>}
            </div>
            {imageUrl
              ? <img src={imageUrl} alt="Preview" className="preview-img" style={previewStyle} />
              : <div className="preview-empty">—</div>
            }
          </div>

          <div className="preview-pane">
            <div className="preview-label">
              Rendered
              {resultUrl && (
                <a className="download-link" href={resultUrl} download={`tiktok-${selectedVibe}.jpg`}>⬇ Download</a>
              )}
            </div>
            {resultUrl
              ? <img src={resultUrl} alt="Result" className="preview-img" />
              : <div className="preview-empty">{processing ? 'Processing...' : 'Hit Apply to render'}</div>
            }
          </div>
        </div>
      </div>
    </div>
  );
}
