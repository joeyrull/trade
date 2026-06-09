import { useState, useCallback, useRef } from 'react';

const ACCEPTED = '.mp4,.mov,.avi,.mkv,.webm,.m4v';

export default function VideoUploader({ onUpload }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [throwHand, setThrowHand] = useState('left');
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef(null);

  const pickFile = useCallback((f) => {
    if (!f) return;
    setFile(f);
    setError('');
    const url = URL.createObjectURL(f);
    setPreview(url);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) pickFile(f);
  }, [pickFile]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setUploading(true);
    setError('');
    try {
      await onUpload({ file, throwHand });
    } catch (err) {
      setError(err.message);
      setUploading(false);
    }
  };

  return (
    <div className="uploader">
      <div className="uploader-card">
        <h2>Upload Pitcher Video</h2>
        <p className="uploader-hint">
          Film from the <strong>first-base side</strong> (pitcher's right for a lefty).
          Slow-motion (60fps+) gives better arm-speed accuracy.
        </p>

        {/* Drop zone */}
        <div
          className={`drop-zone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
        >
          {preview ? (
            <video src={preview} className="preview-video" muted playsInline />
          ) : (
            <div className="drop-placeholder">
              <span className="drop-icon">🎥</span>
              <span>Drop video here or click to browse</span>
              <span className="drop-formats">MP4 · MOV · AVI · MKV · WebM</span>
            </div>
          )}
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED}
            style={{ display: 'none' }}
            onChange={e => pickFile(e.target.files[0])}
          />
        </div>

        {file && (
          <p className="filename">{file.name} ({(file.size / 1024 / 1024).toFixed(1)} MB)</p>
        )}

        {/* Throw hand selector */}
        <div className="hand-selector">
          <span className="hand-label">Pitching hand:</span>
          <div className="hand-buttons">
            {['left', 'right'].map(h => (
              <button
                key={h}
                className={`hand-btn ${throwHand === h ? 'active' : ''}`}
                onClick={() => setThrowHand(h)}
                type="button"
              >
                {h === 'left' ? '🤚 Left (Lefty)' : '✋ Right (Righty)'}
              </button>
            ))}
          </div>
        </div>

        <div className="camera-tip">
          <div className="tip-row">
            <span className="tip-icon">📷</span>
            <span>
              {throwHand === 'left'
                ? 'Camera at first base — pitcher's right side visible. Throwing (left) arm faces away from camera.'
                : 'Camera at third base — pitcher's left side visible. Throwing (right) arm faces away from camera.'}
            </span>
          </div>
        </div>

        {error && <p className="error-msg">{error}</p>}

        <button
          className="btn-primary"
          disabled={!file || uploading}
          onClick={handleSubmit}
        >
          {uploading ? 'Uploading…' : 'Analyze Delivery'}
        </button>
      </div>
    </div>
  );
}
