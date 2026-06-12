# PitchCap — Handoff / Context Brief

> Paste this into a fresh conversation to bring it fully up to speed on PitchCap.
> Self-contained: assumes zero prior context.

## What PitchCap is

A standalone **Python** tool that turns 1–4 iPhone video clips of a baseball pitch into the
**kinematic sequence** — the rotational speeds of the body segments and the order they peak in:

- pelvis (hip) angular velocity (deg/s)
- trunk (torso) angular velocity (deg/s)
- throwing-arm angular velocity (deg/s)
- each segment's **peak magnitude**, **peak time**, and the **sequence order + inter-peak lags**

A healthy pitch fires proximal→distal: **pelvis → trunk → arm**. That ordering and its timing
are the core output.

It is **standalone** (not part of the FreakPitcher app). It outputs a `result.json` + `plot.png`.

## Where the code lives

- Repo root: `/Users/joeyruller/pitchcap/` (its own git repo, ~25 commits, `main` branch).
- Language: Python 3.9+. ~15 small modules under `pitchcap/`, 20 passing tests under `tests/`.
- Design rationale: `docs/superpowers/specs/2026-06-11-pitchcap-design.md`
- Build plan: `docs/superpowers/plans/2026-06-11-pitchcap.md`
- Usage + limitations: `README.md`

## How it works (pipeline)

```
videos → decode (frames + audio + fps)
       → audio cross-correlation sync (align the cameras in time)
       → RTMPose 2D pose per view (17 COCO keypoints + confidence)
       → multi-view path:  undistort+normalize → markerless extrinsic calibration
                           → global scale resolution (3-4 cams) → confidence-weighted
                           → RANSAC triangulation → 3D
         single-cam path:  MediaPipe world-landmarks → 3D
       → zero-lag Butterworth filter (NaN- and low-fps-safe)
       → segment vectors → angular velocity → peaks → kinematic sequence
       → result.json + plot.png
```

Key modules: `io_video.py`, `sync.py`, `pose2d.py` (RTMPose), `intrinsics.py` (lens
calibration), `calibrate_extrinsic.py`, `triangulate.py`, `monocular.py`, `filtering.py`,
`biomech.py`, `reconstruct.py`, `plot.py`, `main.py` (CLI).

## Why the design is the way it is (the important decisions)

1. **Angular velocity is scale-invariant.** Angles between segment vectors don't depend on
   reconstruction scale, so we can use low-friction **markerless** calibration (no per-session
   checkerboard) with no penalty on the metric we actually report.
2. **240fps matters.** It gives the filter headroom to kill jitter without clipping the fast
   arm-whip peak. Lower fps (30fps) undersamples and underestimates peaks.
3. **Two separate calibrations.** Intrinsics = one-time per-phone lens profile (reused forever,
   location-independent). Extrinsics = per-session, markerless, automatic from the body joints.
4. **2D-first model (RTMPose) + true triangulation** beats forcing one model to do 3D — crisp
   under motion blur, real 3D when ≥2 cameras.

## Current status

- ✅ Full pipeline implemented, 20 tests passing (math validated on synthetic ground truth +
  an end-to-end smoke test).
- ✅ Multi-camera (3–4 cam) reconstruction is scale-consistent; lens distortion is applied;
  robust (RANSAC) triangulation for ≥3 views.
- ✅ Runs on real footage end-to-end. First real 2-camera run produced the correct
  pelvis→trunk→arm order.
- ⚠️ Accuracy caveats (honest): absolute deg/s magnitudes are approximate; reliable parts are
  curve shape + peak timing/order. No metric-cm output without a checkerboard. Not lab-grade.

## Known limitations / backlog

- Full bundle adjustment deferred (global-scale resolution is a lighter substitute).
- Real footage needs: **240fps slo-mo**, clips **trimmed to a single throw (~2s)**, and both
  cameras genuinely **overlapping in time** (audio sync needs a shared event like ball-in-glove).

## How to run it

```bash
cd /Users/joeyruller/pitchcap
python3 -m pip install -r requirements.txt   # numpy scipy opencv rtmlib onnxruntime mediapipe ...

# two cameras (primary); optional per-phone lens profiles improve magnitudes
python3 -m pitchcap.main cam0.mov cam1.mov --handedness R
# single camera (monocular fallback, reduced accuracy)
python3 -m pitchcap.main cam0.mov --handedness L
```

Flags: `--handedness {R,L}`, `--profiles a.json b.json`, `--out-json`, `--out-png`, `--cutoff-hz`.

## Output schema (`result.json`) — what a frontend consumes

```jsonc
{
  "fps": 240,
  "n_cams": 2,
  "mode": "multiview",            // or "monocular"
  "handedness": "R",
  "reprojection_error_px": 4.2,   // calibration quality; lower is better (<5 good)
  "segments": {
    "pelvis": { "series_degps": [/* per-frame */], "peak_degps": 612.0, "peak_time_s": 0.84 },
    "trunk":  { "series_degps": [ ... ],           "peak_degps": 1015.0, "peak_time_s": 0.89 },
    "arm":    { "series_degps": [ ... ],           "peak_degps": 4200.0, "peak_time_s": 0.93 }
  },
  "sequence_order": ["pelvis", "trunk", "arm"],
  "inter_peak_lags_ms": { "pelvis_to_trunk": 50, "trunk_to_arm": 40 },
  "segment_warnings": [],         // e.g. a segment whose joints were never reconstructed
  "warnings": [ "footage is ~30 fps ...", "camera_0 using approximate intrinsics ..." ]
}
```

`series_degps` arrays are ready to plot as time series; `sequence_order`/`inter_peak_lags_ms`
are the headline summary.

## Recommended integration with a JS/TS web app

PitchCap is Python with heavy CV deps (onnxruntime, mediapipe, opencv) — **do not reimplement
it in JS**. Run it as a **Python service the JS/TS app calls over HTTP**, and consume the JSON.

**Pattern (recommended): thin FastAPI wrapper + async job.**

1. Frontend (React/Next/etc.) uploads 1–4 clips (+ handedness) to the Python service.
2. Service runs `pitchcap.main.run(...)` and returns the `result.json` payload.
3. Frontend renders `series_degps` (line charts), the `sequence_order`, peaks, and surfaces
   `warnings`.

Because processing runs a pose model per frame (seconds to minutes per clip), prefer an
**async job** shape: `POST /analyze` → `{ job_id }`, then `GET /jobs/{job_id}` → status →
result. For a quick MVP a synchronous `POST /analyze` that blocks and returns the JSON is fine.

**Minimal FastAPI sketch** (the new conversation can build this in `pitchcap/server.py`):

```python
# pip install fastapi uvicorn python-multipart
import tempfile, os
from fastapi import FastAPI, UploadFile, File, Form
from pitchcap.main import run

app = FastAPI()

@app.post("/analyze")
async def analyze(videos: list[UploadFile] = File(...), handedness: str = Form("R")):
    paths = []
    with tempfile.TemporaryDirectory() as d:
        for v in videos:
            p = os.path.join(d, v.filename)
            with open(p, "wb") as f:
                f.write(await v.read())
            paths.append(p)
        result = run(paths, handedness=handedness,
                     out_json=os.path.join(d, "r.json"),
                     out_png=os.path.join(d, "p.png"))
    return result   # the result.json dict, consumed by the JS frontend
```

Run with `uvicorn pitchcap.server:app --reload`; the JS app POSTs multipart clips to
`http://localhost:8000/analyze`. Deploy later as a container or on a GPU host for speed.

Note: `pitchcap.main.run(video_paths, profile_paths=None, handedness="R", out_json=..., out_png=...,
cutoff_hz=15.0)` is the importable entry point and returns the result dict directly — the HTTP
layer is just a wrapper around it.

## First thing to do in the new conversation

If continuing development: read `README.md` and this file, confirm `python3 -m pytest -q`
shows 20 passing, then build the FastAPI wrapper above (or whatever the website needs). The
Python engine is done and tested — integration work is mostly the HTTP/job layer and the
frontend rendering of the JSON.
