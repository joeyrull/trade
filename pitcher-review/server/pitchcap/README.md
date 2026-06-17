# PitchCap

Markerless motion-capture core for baseball pitching analysis. Turns 1–4 iPhone clips of a pitch into the **kinematic sequence**: pelvis, trunk, shoulder (upper arm), and elbow (forearm) angular velocities (deg/s), each segment's peak magnitude, and the timing/order of those peaks.

Standalone tool — outputs `result.json` + `plot.png`. Design rationale in [`docs/superpowers/specs/2026-06-11-pitchcap-design.md`](docs/superpowers/specs/2026-06-11-pitchcap-design.md).

## Why this exists

The headline output is **angular velocity, which is scale-invariant** — angles between body-segment vectors don't depend on reconstruction scale. That lets us use low-friction markerless camera calibration (no per-session checkerboard) with no accuracy penalty on the metric that matters. Recording at 240fps gives the filtering headroom to kill jitter without clipping the arm-whip peak.

## Install

```bash
pip install -r requirements.txt
```

Core math (numpy, scipy, opencv) runs the full test suite. Two extras are needed only for real clips:
- `rtmlib` + `onnxruntime` — 2D pose (RTMPose). Imported lazily.
- `soundfile` — audio sync. Falls back to zero-offset with a warning if absent.

## Usage

```bash
# Two cameras (primary path). Pass lens profiles if you've calibrated them.
python3 -m pitchcap.main throw_cam0.mov throw_cam1.mov \
  --profiles iphone_A_wide_240.json iphone_B_wide_240.json --handedness R

# Single camera (monocular fallback, reduced accuracy)
python3 -m pitchcap.main throw_cam0.mov --handedness R
```

Outputs `result.json` (per-segment time-series, peaks, sequence order, inter-peak lags, reprojection error, warnings) and `plot.png`.

### One-time lens calibration (optional but recommended)

Lens intrinsics belong to the phone, not the location — calibrate each phone **once, ever**, and reuse the profile forever. Record ~20–30s of a checkerboard at varied angles, then:

```python
from pitchcap.intrinsics import calibrate_from_video, save_profile
intr = calibrate_from_video("board_iphoneA.mov", board_size=(9, 6), square_m=0.025)
save_profile("iphone_A_wide_240.json", intr.K, intr.dist, intr.image_size)
```

## Tests

```bash
python3 -m pytest -q
```

The math core (biomech, triangulation, filtering, sync, calibration) is unit-tested against synthetic ground truth. An end-to-end smoke test drives the full geometry→3D→sequence pipeline on a synthetic two-view rig with no model downloads.

## Accuracy & known limitations (v0.1 foundation)

Honest about what at-home markerless can and cannot do:

**Reliable:** angular-velocity curve shape and peak timing/order (the kinematic sequence). Scale- and largely calibration-robust.

**Approximate:** absolute deg/s magnitudes — improve with lens profiles and more cameras.

**Resolved (multi-camera hardening pass):**
1. ✅ **3–4 camera scale consistency.** Reconstruction now runs in normalized coordinates, recovers each view's pose against camera 0, and resolves a single consistent global scale across all cameras (camera 1 defines the scale; cameras ≥3 are rescaled by aligning shared-joint depths). 3–4 views are geometrically consistent. *(See `reconstruct.py`, `tests/test_multiview_scale.py`.)*
2. ✅ **Lens distortion is applied.** Each view's 2D points are undistorted by that view's own intrinsics (`geometry.normalize_points`) before essential-matrix and triangulation math, so calibrated distortion coefficients are now used. Per-view intrinsics differences are handled too.
4. ✅ **Cross-view outlier rejection.** Triangulation uses minimal-subset (pairwise) RANSAC when ≥3 views are available, so a single bad keypoint in one view can't drag the 3D estimate.
3. ✅ **Per-segment filter cutoffs.** `compute_kinematic_sequence(..., cutoffs=DEFAULT_CUTOFFS)` filters each segment's joints at its own band (13 Hz pelvis/trunk, 18 Hz shoulder/elbow) before differentiation, so the sharp arm-whip peaks are preserved while the noisier rotation angles are filtered harder. A joint shared across segments is filtered independently for each. *(See `biomech.py`, `tests/test_biomech.py`.)*

**Remaining refinements (not blocking):**
5. **Full bundle adjustment** (joint refinement of all camera poses + 3D points) is still deferred — the current global-scale resolution is a lighter-weight substitute. Worth adding if real-clip reprojection error is high.

**Not supported by design:** metric (cm) outputs without a checkerboard; lab marker-grade (sub-mm, global-shutter) precision.

## Project layout

```
pitchcap/
  constants.py          COCO-17 indices, Intrinsics/Extrinsics dataclasses
  geometry.py           projection matrix, undistort helpers
  io_video.py           decode clips -> frames + audio + fps
  sync.py               audio cross-correlation -> frame offsets
  pose2d.py             RTMPose 2D backend (pluggable; YOLO11-pose swap noted)
  intrinsics.py         one-time lens calibration + profile I/O
  calibrate_extrinsic.py  markerless extrinsics (essential matrix + reproj error)
  triangulate.py        confidence-weighted multi-view DLT + gap fill
  monocular.py          1-camera MediaPipe world-landmark fallback
  filtering.py          zero-lag Butterworth (NaN- and short-clip-safe)
  biomech.py            segment vectors -> angular velocity -> kinematic sequence
  reconstruct.py        choose multiview vs monocular -> 3D
  plot.py               kinematic-sequence plot
  main.py               CLI
```
