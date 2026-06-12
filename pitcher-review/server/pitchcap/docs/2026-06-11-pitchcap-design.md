# PitchCap — Markerless Motion-Capture Core (Design Spec)

**Date:** 2026-06-11
**Status:** Implemented. v0.1 foundation (2-camera + monocular) plus the multi-camera hardening pass: normalized-coordinate reconstruction, global scale resolution for 3–4 cameras, lens-distortion handling, and pairwise-RANSAC robust triangulation are all built and tested (19 tests). Remaining refinements (per-segment filter cutoffs, full bundle adjustment) are noted in README; "native 3–4 cam" is now delivered.
**Tier:** Elite (per-session markerless + one-time lens calibration + bundle adjustment + confidence-weighted triangulation + native 3–4 cam)

## Goal

A standalone Python tool that ingests 1–4 iPhone video clips of a baseball pitch and outputs the **kinematic sequence**: pelvis, trunk, and throwing-arm angular velocities (deg/s) over time, plus each segment's **peak magnitude** and the **timing/order** of those peaks. Bias toward working, runnable code.

## Non-goals (for this foundation)

- No integration with the FreakPitcher HTML app yet (standalone; emits JSON others can consume).
- No player-development/coaching logic — pure measurement.
- No metric (cm) outputs — out of scope by design (see Accuracy).
- No model training in the foundation; fine-tuning is a documented later path (Step 5).

## Hardware & context

- 1–4 iPhones, **usually 2**, sometimes 1, up to 4. Capable of 240fps.
- Subject: pitcher throwing — violent rotation, fast arm, self-occlusion.
- Existing app uses MediaPipe (BlazePose); we are free to replace/hybridize.

## Core insights driving the design

1. **Primary output is angular velocity (deg/s), which is scale-invariant.** Angles between segment vectors don't depend on reconstruction scale, so an unknown global scale (inherent to markerless extrinsics) costs us nothing on the metric we care about.
2. **240fps is the enabler.** Nyquist = 120 Hz gives huge headroom to filter jitter aggressively without clipping the arm-whip peak — directly fixing the "intense motion" weakness of low-fps tracking.
3. **The kinematic sequence (peak timing & order) is the most robust output.** It barely depends on calibration quality, so it's trustworthy even in degraded conditions. Magnitudes are what better calibration buys.
4. **Use each tool for its strength.** 2D-first models (RTMPose) are crisp under motion blur; multi-view triangulation supplies true 3D; monocular 3D is the graceful fallback only.

## The two calibrations (kept strictly separate)

- **Intrinsics — one-time per (phone, lens, recording mode).** Focal length, optical center, lens distortion. A property of the hardware, independent of location/scene. Calibrated once with a checkerboard, saved to a profile, reused forever. Falls back to approximate FOV-derived intrinsics + warning if no profile exists.
- **Extrinsics — per session, markerless, automatic.** Camera positions/orientations relative to each other, recovered from the pitcher's own joints each session. Handles any change of location or phone placement with zero user friction. Scale is unknown (fine — see insight #1).

## Architecture

```
videos ─┬─> io_video ──> frames + audio + fps
        │
        ├─> sync (audio xcorr) ──> per-clip frame offsets
        │
        ├─> pose2d (RTMPose) ──> per-view 2D keypoints + confidence
        │
   [2+ cams]                              [1 cam]
        │                                    │
   calibrate_extrinsic (E-matrix,        monocular (MediaPipe
   bundle adjust) + intrinsics            world-landmarks 3D)
        │                                    │
   triangulate (conf-weighted,              │
   RANSAC, N views)                         │
        └──────────────┬─────────────────── ┘
                       v
                 filtering (zero-lag Butterworth)
                       v
                 biomech (segment vectors -> angular velocity
                          -> peaks -> sequence order/lags)
                       v
                 result.json + sequence plot.png
```

## Module breakdown

| Module | Responsibility | Interface (in → out) |
|---|---|---|
| `io_video.py` | Decode clips, extract frames + audio, normalize/record fps | path → `frames[T,H,W,3]`, `audio[]`, `fps` |
| `sync.py` | Audio cross-correlation → integer per-clip frame offsets; rolling-shutter note | `audio[]`, `fps` → `offsets[int]` |
| `pose2d.py` | RTMPose per frame → 17 COCO keypoints + confidence; pluggable backend | `frames` → `kp2d[T,17,2]`, `conf[T,17]` |
| `intrinsics.py` | **Utility**: checkerboard video → per-phone lens profile (K, dist); load/save | board video → `K`, `dist` JSON; or `load(profile)` |
| `calibrate_extrinsic.py` | Markerless extrinsics: correspondences → essential matrix (RANSAC) → recover R,t → bundle adjust over clip; report reprojection error | `kp2d` per view, `K[]` → camera poses, `reproj_err` |
| `triangulate.py` | Confidence-weighted, RANSAC multi-view DLT (2–4 views) → 3D per frame; outlier rejection; occlusion gap-fill | `kp2d[]`, `conf[]`, cams → `kp3d[T,17,3]` |
| `monocular.py` | 1-cam fallback: MediaPipe world-landmarks → 3D | `frames` → `kp3d[T,J,3]` |
| `filtering.py` | 4th-order zero-phase Butterworth on 3D positions; per-segment cutoffs; optional residual-analysis auto-cutoff | `kp3d` → smoothed `kp3d` |
| `biomech.py` | Segment vectors → angular velocity (deg/s) → peak detect → sequence order + inter-peak lags | smoothed `kp3d`, handedness → result dict |
| `plot.py` | Time-series plot of 3 curves with peak markers | result → PNG |
| `main.py` | CLI: 1–N videos (+ optional profiles) → `result.json` + plot | — |

## Component detail

### Sync
- `scipy.signal.correlate` on the clips' audio waveforms → lag in samples → frames (at 240fps, 1 frame = 4.17 ms; round to nearest frame).
- If clips differ in fps, resample to a common timeline.
- **Rolling shutter:** iPhones read the sensor row-by-row; at 240fps + fast arm this skews keypoints slightly. Optional first-order correction flag; documented as a residual vs. global-shutter lab cameras.

### 2D pose — RTMPose (primary)
- Via `rtmlib` (ONNX/`onnxruntime`), pip-installable, no full MMPose.
- Chosen over MediaPipe (crisper under motion blur) and YOLO-pose (higher keypoint *precision*, which triangulation amplifies).
- COCO-17 supplies all needed joints: L/R shoulder (5,6), L/R hip (11,12), throwing elbow (7/8), throwing wrist (9/10).
- Backend is pluggable: one-line swap to `ultralytics` YOLO11-pose.

### Intrinsics (one-time lens calibration)
- Utility script: record ~20–30 s of a printed/on-screen checkerboard at varied angles/distances per phone, per recording mode.
- `cv2.findChessboardCorners` + `cv2.calibrateCamera` → `K`, distortion coeffs → saved JSON profile keyed by `(phone_id, lens, resolution, fps)`.
- Reused indefinitely. Missing profile → approximate `K` from image width × typical iPhone FOV, with a warning surfaced in output.

### Extrinsic calibration (markerless, per session) + bundle adjustment
- Gather high-confidence joint correspondences across **many frames spanning the whole motion** (avoids degenerate single-pose geometry).
- Undistort with intrinsics → `cv2.findEssentialMat` (RANSAC) → `cv2.recoverPose` → relative `R, t` (unit scale).
- **Bundle adjustment**: refine camera extrinsics (and 3D points) by minimizing total reprojection error across the clip (`scipy.optimize.least_squares`).
- **Report reprojection error (px)** in output so a bad session's calibration is visible, not silent.
- Drop-in upgrade path: a board-based extrinsic step can replace this behind the same interface if markerless error is routinely too high.

### Triangulation (2–4 views)
- Confidence-weighted DLT per joint; RANSAC across views to reject per-joint outliers (a blurry/occluded keypoint in one view can't drag the 3D point).
- 3+ views over-determine each point and crush occlusion — automatically used when available.
- Occlusion gap-fill: short low-confidence runs interpolated; long ones flagged.

### Monocular fallback (1 cam)
- MediaPipe Pose `world_landmarks` → metric-ish 3D relative to hip center.
- Same downstream filtering/biomech. Explicitly the less-accurate path (the known depth-ambiguity weakness); used only when a single clip is provided.

### Temporal filtering
- 4th-order **zero-phase Butterworth** (`scipy.signal.filtfilt`, no lag) applied to 3D **positions** before differentiating.
- Default cutoffs: ~13 Hz pelvis/trunk, ~18 Hz arm. Optional per-signal residual-analysis to auto-select cutoff.
- Rationale: at 240fps there's ample headroom to remove jitter while preserving the peak.

### Biomechanics
- Segment unit vector `u(t)`; angular velocity `ω = |u × du/dt|` (rad/s → deg/s); `du/dt` by central finite difference on filtered 3D.
- Segments:
  - **Pelvis:** right-hip → left-hip
  - **Trunk:** hip-midpoint → shoulder-midpoint
  - **Arm (throwing):** throwing-shoulder → throwing-elbow (handedness configurable; default RHP)
- `scipy.signal.find_peaks` → peak deg/s + peak time per segment.
- **Sequence:** sort segments by peak time → report order + inter-peak lags (ms). Healthy pattern: pelvis → trunk → arm.
- Default is full-3D segment angular *speed*; optional projection onto the vertical axis for the textbook transverse-plane number.

### Output schema (`result.json`)
```json
{
  "fps": 240,
  "n_cams": 2,
  "mode": "multiview",
  "handedness": "R",
  "reprojection_error_px": 2.4,
  "segments": {
    "pelvis": {"series_degps": [], "peak_degps": 612.0, "peak_time_s": 0.84},
    "trunk":  {"series_degps": [], "peak_degps": 1015.0, "peak_time_s": 0.89},
    "arm":    {"series_degps": [], "peak_degps": 4200.0, "peak_time_s": 0.93}
  },
  "sequence_order": ["pelvis", "trunk", "arm"],
  "inter_peak_lags_ms": {"pelvis_to_trunk": 50, "trunk_to_arm": 40},
  "warnings": ["camera_2 using approximate intrinsics (no lens profile)"]
}
```
Plus `plot.png`: the three angular-velocity curves with peak markers.

## Accuracy — honest bounds

- ✅ **Reliable:** angular-velocity curve *shape* and peak *timing/order* (the kinematic sequence). Scale- and largely calibration-robust.
- ⚠️ **Approximate:** absolute deg/s magnitudes — quality scales with intrinsics + bundle adjustment + number of cameras. With lens profiles + 2–3 cams: good enough to track real change over time and roughly match ASMI ranges.
- ❌ **Not supported:** metric (cm) measurements without a board; lab marker-grade precision (global-shutter, sub-mm) is unattainable at home.
- Realistic 2–3 cam ceiling: timing within a few ms; magnitudes useful for development, not publication.

## Dependencies

```
numpy scipy opencv-python rtmlib onnxruntime mediapipe matplotlib imageio-ffmpeg soundfile
```
(Optional: `ultralytics` for the YOLO11-pose backend swap.)

## Step 5 — later accuracy upgrades (not in foundation)

- Fine-tune RTMPose on labeled pitching frames (occlusion/motion-blur cases) to improve 2D precision where it matters most.
- Optionally train a multi-view lifter on the tool's own triangulated 3D (self-supervised) for the monocular path.
- Validate against: reprojection error, a few board-scaled clips for metric ground truth, and ASMI sanity bands (peak pelvis ~500–700°/s, trunk ~900–1200°/s).

## Validation plan (foundation)

- Unit tests on `biomech.py` with synthetic 3D (known constant angular velocity → exact deg/s).
- Triangulation test on a synthetic two-view rig with known geometry.
- End-to-end smoke test on a sample clip → JSON + plot produced, reprojection error reported.

## Open questions / future

- Wrist/hand for release-point detection (deferred).
- Auto handedness detection.
- FreakPitcher integration (JSON ingest) — separate later project.
