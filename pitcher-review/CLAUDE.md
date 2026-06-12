# pitcher-review — project guide

A baseball-pitching biomechanics web app: upload pitch video(s) → pose
analysis → per-frame metrics, phase timeline, kinetic-chain sequencing, and an
annotated video. React frontend + Node/Express API + Python analysis scripts.

## Architecture

```
web/ (React/Vite)  ──HTTP──>  server/ (Express)  ──spawn──>  scripts/*.py
  AnalysisViewer               routes/analysis.js             analyze_pitcher.py  (MediaPipe engine)
  KinematicSequence            (job store, sync, fusion)      pitchcap_analyze.py (PitchCap engine, default)
  SummaryReport / LabReport                                   fuse_cameras.py     (2-cam metric fusion)
                                                              pitchcap/           (vendored PitchCap pkg)
```

`routes/analysis.js` runs each uploaded camera through an analyzer subprocess
(CLI: `<video...> --output-dir <dir> --throw-hand <l|r> --progress`; stdout =
JSON progress lines + terminal `{"status":"done", annotated_video, metrics}`),
writing `metrics.json` / `series.json` / `annotated.mp4` per camera, then
optionally cross-syncs and fuses two cameras.

## Analysis engines (selectable)

- `PITCHER_ENGINE=pitchcap` (**default**) → `pitchcap_analyze.py`. Single camera:
  delegates to `analyze_pitcher.analyze_video(..., kinematic_sequence=True)` —
  identical trusted metrics PLUS PitchCap's kinematic sequence under
  `summary.pitchcap`. 2+ cameras (CLI / `PITCHER_MULTIVIEW=1`): RTMPose 2D +
  multi-view triangulation → true metric 3D.
- `PITCHER_ENGINE=mediapipe` → original `analyze_pitcher.py` (no kinematic
  sequence).
- `PITCHER_MULTIVIEW=1` (off by default, PitchCap engine only): two uploaded
  cameras go to ONE process for true triangulation instead of per-camera +
  fusion. Needs `rtmlib`+`onnxruntime`; falls back to monocular on camera 0.

PitchCap (`server/pitchcap/`) is a standalone markerless motion-capture core;
its headline output is the kinematic sequence (pelvis→trunk→arm angular
velocity, peak timing/order). Integration detail:
`server/pitchcap/docs/INTEGRATION.md`.

## Hard invariants

- **metrics.json must be valid JSON** (Node `JSON.parse` + browser reject
  `NaN`/`Infinity`). Every write goes through `analyze_pitcher.json_sanitize()`;
  `biomech.compute_kinematic_sequence` keeps its own values finite (None lags /
  null series for occluded segments). Don't add a raw `json.dump` of metrics.
- The PitchCap engine is **purely additive** for single camera: the trusted
  per-frame metrics/peaks/sequencing must stay byte-identical to the MediaPipe
  engine; PitchCap only adds `summary.pitchcap`.
- `summary` schema consumed by the frontend: `fps, motion_fps, total_frames,
  throw_hand, tracking_quality, phases, peak, key_frames, sequencing,
  phase_confidence, auto_zoom`, optional `pitchcap`. Per-frame `metrics`:
  `hip_rotation, shoulder_rotation, hip_shoulder_sep, elbow_angle,
  elbow_height_pct, hip_rotation_speed, chest_rotation_speed, arm_speed,
  trunk_tilt, low_confidence`.

## Run / test

```bash
# server (port 3002)
cd server && node index.js
# frontend
cd web && npm run dev      # or: npm run build

# Python tests (pytest not installed by default: pip install pytest)
cd server/pitchcap && python3 -m pytest -q          # 23 tests (PitchCap math core)
cd server/scripts  && python3 -m pytest test_pitchcap_bridge.py -q   # 11 tests (bridge glue)
```

Python deps already present in the analysis env: `numpy scipy opencv-python
mediapipe matplotlib`. Multi-view path also needs `rtmlib onnxruntime`. The pose
model is `server/scripts/models/pose_landmarker_heavy.task` (auto-downloaded).

## Slow-motion caveat

`detect_frame_rates` distinguishes playback fps from motion (capture) fps; all
speed/timing metrics use motion_fps. A single ambiguous clip whose container
metadata hides its slow-mo rate may understate speeds; a confident second camera
corrects it via `--rebuild` (which also rescales the `summary.pitchcap` block).
