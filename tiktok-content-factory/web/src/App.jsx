import React, { useState } from 'react';
import DiscoveryPanel from './components/DiscoveryPanel/index.jsx';
import CurationDashboard from './components/CurationDashboard/index.jsx';
import VibeEditor from './components/VibeEditor/index.jsx';

const NAV = [
  { id: 'discover', icon: '🔍', label: 'Discover' },
  { id: 'curate', icon: '📂', label: 'Collections' },
  { id: 'editor', icon: '🎨', label: 'Vibe Editor' },
];

export default function App() {
  const [view, setView] = useState('discover');
  const [editorImage, setEditorImage] = useState(null);

  function openInEditor(item) {
    setEditorImage(item);
    setView('editor');
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-logo">TikTok <span>Factory</span></div>
        <nav className="sidebar-nav">
          {NAV.map(n => (
            <button
              key={n.id}
              className={`nav-item${view === n.id ? ' active' : ''}`}
              onClick={() => setView(n.id)}
            >
              <span className="icon">{n.icon}</span> {n.label}
            </button>
          ))}
        </nav>
      </aside>

      <main className="main">
        {view === 'discover' && <DiscoveryPanel onEdit={openInEditor} />}
        {view === 'curate' && <CurationDashboard onEdit={openInEditor} />}
        {view === 'editor' && <VibeEditor initialImage={editorImage} />}
      </main>
    </div>
  );
}
