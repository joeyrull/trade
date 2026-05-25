import React from 'react';
import './MediaGrid.css';

export default function MediaGrid({ items, onEdit, onSave, savedIds = new Set() }) {
  if (!items.length) {
    return <div className="media-empty">No results yet. Search for content above.</div>;
  }

  return (
    <div className="media-grid">
      {items.map(item => (
        <MediaCard
          key={item.id}
          item={item}
          onEdit={onEdit}
          onSave={onSave}
          saved={savedIds.has(item.id)}
        />
      ))}
    </div>
  );
}

function MediaCard({ item, onEdit, onSave, saved }) {
  return (
    <div className="media-card">
      <div className="media-thumb">
        {item.type === 'video'
          ? <video src={item.preview} muted loop onMouseEnter={e => e.target.play()} onMouseLeave={e => e.target.pause()} />
          : <img src={item.thumb} alt={item.description} loading="lazy" />
        }
        <div className="media-overlay">
          <button className="media-btn" onClick={() => onEdit?.(item)} title="Edit in Vibe Editor">✏️</button>
          {onSave && (
            <button className={`media-btn${saved ? ' saved' : ''}`} onClick={() => onSave(item)} title={saved ? 'Saved' : 'Save to collection'}>
              {saved ? '★' : '☆'}
            </button>
          )}
        </div>
        <div className="media-source">{item.source}</div>
        {item.type === 'video' && <div className="media-type-badge">VIDEO</div>}
      </div>
      {item.credit && (
        <div className="media-credit">by {item.credit.name}</div>
      )}
    </div>
  );
}
