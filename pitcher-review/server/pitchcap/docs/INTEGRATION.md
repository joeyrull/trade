# PitchCap ↔ pitcher-review integration

How the standalone PitchCap motion-capture core is wired into the
pitcher-review web app, what it adds, and how to run it.

## What PitchCap adds to the app

PitchCap is a markerless motion-capture core (RTMPose 2D + multi-view
triangulation, or MediaPipe monocular fallback) whose headline output is the
**kinematic sequence**: pelvis → trunk → throwing-arm angular velocities over
time, each segment's peak magnitude, and the peak **timing/order** (a healthy
delivery fires proximal→distal). See `README.md` / `docs/…design.md`.

In the app it shows up as a new **"Kinematic Sequence"** tab in the analysis
viewer (rendered by `web/src/components/KinematicSequence.jsx`), populated from
`summary.pitchcap` in `metrics.json`.

## Architecture: an additive engine, not a rewrite

The app's trusted per-frame metrics, phases, peaks, sequencing and
tracking-quality (everything the UI grades against) are produced by
`scripts/analyze_pitcher.py`. PitchCap **does not replace** that schema layer —
it reuses it, so output stays byte-compatible with the existing frontend.

```
                       ┌─────────────────────────────────────────┐
 video(s) ──spawn────► │  scripts/pitchcap_analyze.py  (engine)   │
                       └───────────────┬─────────────────────────┘
              1 camera                 │                 2+ cameras
                 │                     │                     │
   ap.analyze_video(..., kinematic_sequence=True)   _analyze_multiview(...)
   (MediaPipe metrics, unchanged)                   RTMPose 2D + triangulation
                 │                     │            → true metric 3D
   + summary['pitchcap']  ◄────────────┴────────────►  web metrics + sequence
   (kinematic sequence from world landmarks)          (both from true 3D)
```

* **1 camera** — PitchCap can't out-pose MediaPipe monocularly (it *uses*
  MediaPipe), so the per-frame metrics come from `analyze_pitcher.analyze_video`
  exactly as the default engine produces them. PitchCap's contribution is the
  `summary['pitchcap']` kinematic-sequence block, computed from the same
  MediaPipe **world** landmarks via `pitchcap.biomech.compute_kinematic_sequence`.
  This is the path the current Node flow always takes (it analyzes each uploaded
  camera in its own process, then fuses).

* **2+ cameras (CLI)** — the genuine upgrade. RTMPose 2D per view + markerless
  multi-view triangulation → true metric 3D, from which both the web metrics and
  the kinematic sequence are derived. Requires `rtmlib` + `onnxruntime`; falls
  back to the 1-camera path on camera 0 with a warning if absent.

## Selecting the engine (Node)

`server/routes/analysis.js`:

```js
const ENGINE = (process.env.PITCHER_ENGINE || 'pitchcap').toLowerCase();
// 'pitchcap'  -> scripts/pitchcap_analyze.py   (default; adds the sequence)
// 'mediapipe' -> scripts/analyze_pitcher.py    (original, no sequence)
```

Both are spawned with the identical CLI/stdout contract
(`<video> --output-dir <dir> --throw-hand <hand> --progress`, JSON progress
lines, terminal `{"status":"done", annotated_video, metrics}`) and write the
same `annotated.mp4` / `metrics.json` / `series.json`. The motion-fps
`--rebuild` correction always runs `analyze_pitcher.py`, which operates purely
on the schema-compatible `series.json` either engine writes.

Set `PITCHER_ENGINE=mediapipe` to disable PitchCap entirely.

## `summary.pitchcap` schema

```jsonc
"pitchcap": {
  "engine": "pitchcap",
  "mode": "monocular",            // or "multiview"
  "n_cams": 1,
  "reprojection_error_px": null,  // multiview only; lower is better
  "kinematic_sequence": {
    "fps": 300,
    "handedness": "L",
    "sequence_order": ["pelvis", "trunk", "arm"],
    "inter_peak_lags_ms": { "pelvis_to_trunk": 0.0, "trunk_to_arm": 13.3 },
    "segments": {
      "pelvis": { "peak_degps": 956.4, "peak_time_s": 0.887, "series_degps": [ … ] },
      "trunk":  { "peak_degps": 533.2, "peak_time_s": 0.887, "series_degps": [ … ] },
      "arm":    { "peak_degps": 3422.1,"peak_time_s": 0.900, "series_degps": [ … ] }
    },
    "segment_warnings": []        // e.g. a segment whose joints were never reconstructed
  },
  "warnings": [ … ]
}
```

`peak_time_s` is `null` for a segment that was never reliably reconstructed
(its joints occluded/low-confidence all clip); the UI shows it as "—".

## Vendoring & deps

The package lives at `server/pitchcap/` (repo-root layout: `pitchcap/` package
+ `tests/` + `pyproject.toml`). `scripts/pitchcap_analyze.py` puts
`server/pitchcap/` on `sys.path` so `import pitchcap` resolves to the package.

Runtime deps already present for the app: `numpy scipy opencv-python mediapipe`.
The monocular path needs only those. The multi-view path additionally needs
`rtmlib onnxruntime` (see `requirements.txt`); when absent it falls back to
monocular. The kinematic-sequence add-on is wrapped in try/except, so a missing
package or a degenerate clip simply omits `summary.pitchcap` rather than failing
the analysis.

Run the math test suite: `cd server/pitchcap && python3 -m pytest -q` (23 tests).
Bridge glue tests: `cd server/scripts && python3 -m pytest test_pitchcap_bridge.py -q` (10 tests).

## Validation status

- Single-camera (default Node path) verified end-to-end through the HTTP API
  (`POST /api/analysis/upload` → poll `/status` → `GET /result`): the trusted
  per-frame metrics are byte-identical to the MediaPipe engine, `summary.pitchcap`
  flows through to the result, the annotated video is served, and the result is
  strict-JSON over the wire. On a real 300fps clip PitchCap recovers a clean
  pelvis→trunk→arm sequence (~913/509/4222 °/s after per-segment cutoffs).
- Multi-view orchestration (RTMPose 2D → triangulation → web schema) smoke-tested
  end-to-end on real RTMPose output (`mode: multiview`, reprojection error
  reported). Its reconstruction math has unit coverage in
  `tests/test_multiview_scale.py` / `test_triangulate.py` / `test_calibrate_extrinsic.py`.

## Enabling true two-camera triangulation

The current Node flow analyzes each uploaded camera in its own process and then
*fuses* (`fuse_cameras.py`) — it does not triangulate the two views into one 3D
reconstruction. To get true multi-view 3D, run the engine on both clips in a
single process via the CLI (needs `rtmlib` + `onnxruntime` installed and the two
clips overlapping in time with a shared audio event for sync):

```bash
python3 scripts/pitchcap_analyze.py cam0.mov cam1.mov \
    --output-dir out --throw-hand R --progress
```

Wiring this into the upload flow (send both clips to one `pitchcap_analyze`
process when 2 cameras are uploaded) is the natural next step, gated on the
multi-view dependencies being present in the deployment.
