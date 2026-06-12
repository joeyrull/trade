import { useState, useCallback, useRef } from 'react';

const ACCEPTED = '.mp4,.mov,.avi,.mkv,.webm,.m4v';

const ANGLE_OPTIONS = [
  { value: 'side',          label: 'Side view (1B/3B line)' },
  { value: 'front',         label: 'Front view (home plate)' },
  { value: 'behind',        label: 'Behind (center field)' },
  { value: 'three_quarter', label: '3/4 angle' },
  { value: 'other',         label: 'Other' },
];

export default function VideoUploader({ onUpload }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [angle, setAngle] = useState('side');

  const [showSecond, setShowSecond] = useState(false);
  const [file2, setFile2] = useState(null);
  const [preview2, setPreview2] = useState(null);
  const [angle2, setAngle2] = useState('front');

  const [showThird, setShowThird] = useState(false);
  const [file3, setFile3] = useState(null);
  const [preview3, setPreview3] = useState(null);
  const [angle3, setAngle3] = useState('three_quarter');

  const [throwHand, setThrowHand] = useState('left');
  const [dragging, setDragging] = useState(false);
  const [dragging2, setDragging2] = useState(false);
  const [dragging3, setDragging3] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef(null);
  const inputRef2 = useRef(null);
  const inputRef3 = useRef(null);

  const pickFile = useCallback((f) => {
    if (!f) return;
    setFile(f);
    setError('');
    const url = URL.createObjectURL(f);
    setPreview(url);
  }, []);

  const pickFile2 = useCallback((f) => {
    if (!f) return;
    setFile2(f);
    setError('');
    const url = URL.createObjectURL(f);
    setPreview2(url);
  }, []);

  const pickFile3 = useCallback((f) => {
    if (!f) return;
    setFile3(f);
    setError('');
    const url = URL.createObjectURL(f);
    setPreview3(url);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) pickFile(f);
  }, [pickFile]);

  const onDrop2 = useCallback((e) => {
    e.preventDefault();
    setDragging2(false);
    const f = e.dataTransfer.files[0];
    if (f) pickFile2(f);
  }, [pickFile2]);

  const onDrop3 = useCallback((e) => {
    e.preventDefault();
    setDragging3(false);
    const f = e.dataTransfer.files[0];
    if (f) pickFile3(f);
  }, [pickFile3]);

  const removeSecond = useCallback(() => {
    setFile2(null);
    setPreview2(null);
    setShowSecond(false);
    setFile3(null);
    setPreview3(null);
    setShowThird(false);
  }, []);

  const removeThird = useCallback(() => {
    setFile3(null);
    setPreview3(null);
    setShowThird(false);
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setUploading(true);
    setError('');
    try {
      await onUpload({
        file, angle, throwHand,
        ...(file2 ? { file2, angle2 } : {}),
        ...(file2 && file3 ? { file3, angle3 } : {}),
      });
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
          <div className="file-meta-row">
            <p className="filename">{file.name} ({(file.size / 1024 / 1024).toFixed(1)} MB)</p>
            <AngleSelect value={angle} onChange={setAngle} />
          </div>
        )}

        {/* Optional second camera angle */}
        {!showSecond && (
          <button type="button" className="btn-ghost add-camera-btn" onClick={() => setShowSecond(true)}>
            + Add a second camera angle (optional)
          </button>
        )}

        {showSecond && (
          <div className="second-camera">
            <div className="second-camera-header">
              <span className="second-camera-title">Second camera angle</span>
              <button type="button" className="btn-ghost remove-camera-btn" onClick={removeSecond}>Remove</button>
            </div>
            <p className="uploader-hint">
              A second angle (e.g. front view from home plate) lets us combine both
              videos into a more accurate 3D reconstruction of the delivery.
            </p>
            <div
              className={`drop-zone ${dragging2 ? 'dragging' : ''} ${file2 ? 'has-file' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragging2(true); }}
              onDragLeave={() => setDragging2(false)}
              onDrop={onDrop2}
              onClick={() => inputRef2.current?.click()}
            >
              {preview2 ? (
                <video src={preview2} className="preview-video" muted playsInline />
              ) : (
                <div className="drop-placeholder">
                  <span>Drop second video here or click to browse</span>
                  <span className="drop-formats">MP4 · MOV · AVI · MKV · WebM</span>
                </div>
              )}
              <input
                ref={inputRef2}
                type="file"
                accept={ACCEPTED}
                style={{ display: 'none' }}
                onChange={e => pickFile2(e.target.files[0])}
              />
            </div>
            {file2 && (
              <div className="file-meta-row">
                <p className="filename">{file2.name} ({(file2.size / 1024 / 1024).toFixed(1)} MB)</p>
                <AngleSelect value={angle2} onChange={setAngle2} />
              </div>
            )}
          </div>
        )}

        {/* Optional third camera angle (only useful alongside a second) */}
        {showSecond && file2 && !showThird && (
          <button type="button" className="btn-ghost add-camera-btn" onClick={() => setShowThird(true)}>
            + Add a third camera angle (optional)
          </button>
        )}

        {showSecond && showThird && (
          <div className="second-camera">
            <div className="second-camera-header">
              <span className="second-camera-title">Third camera angle</span>
              <button type="button" className="btn-ghost remove-camera-btn" onClick={removeThird}>Remove</button>
            </div>
            <p className="uploader-hint">
              A third synced angle further constrains the 3D reconstruction —
              useful for a multi-camera rig covering the full delivery.
            </p>
            <div
              className={`drop-zone ${dragging3 ? 'dragging' : ''} ${file3 ? 'has-file' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragging3(true); }}
              onDragLeave={() => setDragging3(false)}
              onDrop={onDrop3}
              onClick={() => inputRef3.current?.click()}
            >
              {preview3 ? (
                <video src={preview3} className="preview-video" muted playsInline />
              ) : (
                <div className="drop-placeholder">
                  <span>Drop third video here or click to browse</span>
                  <span className="drop-formats">MP4 · MOV · AVI · MKV · WebM</span>
                </div>
              )}
              <input
                ref={inputRef3}
                type="file"
                accept={ACCEPTED}
                style={{ display: 'none' }}
                onChange={e => pickFile3(e.target.files[0])}
              />
            </div>
            {file3 && (
              <div className="file-meta-row">
                <p className="filename">{file3.name} ({(file3.size / 1024 / 1024).toFixed(1)} MB)</p>
                <AngleSelect value={angle3} onChange={setAngle3} />
              </div>
            )}
          </div>
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
                {h === 'left' ? 'Left (Lefty)' : 'Right (Righty)'}
              </button>
            ))}
          </div>
        </div>

        <div className="camera-tip">
          <div className="tip-row">
            {throwHand === 'left'
              ? "Camera at first base — pitcher's right side visible. Throwing (left) arm faces away from camera."
              : "Camera at third base — pitcher's left side visible. Throwing (right) arm faces away from camera."}
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

function AngleSelect({ value, onChange }) {
  return (
    <select className="angle-select" value={value} onChange={e => onChange(e.target.value)}>
      {ANGLE_OPTIONS.map(o => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  );
}
