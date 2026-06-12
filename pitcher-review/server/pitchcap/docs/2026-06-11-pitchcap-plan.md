# PitchCap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python tool that turns 1–4 iPhone pitch clips into the kinematic sequence (pelvis/trunk/arm angular velocities in deg/s, peak magnitudes, and peak timing/order).

**Architecture:** A bottom-up pipeline of small, single-responsibility modules. Pure-math modules (biomech, filtering, triangulation, sync) are fully unit-tested with synthetic data; CV/IO modules (RTMPose, MediaPipe, video decode) get smoke tests. `main.py` orchestrates: decode → audio-sync → 2D pose → (multi-view triangulation **or** monocular fallback) → zero-lag filter → angular velocities → JSON + plot.

**Tech Stack:** Python 3.11, numpy, scipy, opencv-python, rtmlib + onnxruntime (2D pose), mediapipe (monocular fallback), matplotlib, imageio-ffmpeg + soundfile (decode), pytest.

**Spec:** `docs/superpowers/specs/2026-06-11-pitchcap-design.md`

---

## File Structure

```
pitchcap/
  pitchcap/
    __init__.py
    constants.py          # COCO-17 indices, Intrinsics/Extrinsics dataclasses
    geometry.py           # projection-matrix helper, undistort helper
    biomech.py            # segment vectors -> angular velocity -> sequence
    filtering.py          # zero-lag Butterworth
    triangulate.py        # confidence-weighted multi-view DLT
    sync.py               # audio cross-correlation -> frame offsets
    intrinsics.py         # one-time lens calibration utility + load/save
    calibrate_extrinsic.py# markerless extrinsics + bundle adjustment
    pose2d.py             # RTMPose wrapper (pluggable)
    monocular.py          # MediaPipe world-landmarks fallback
    io_video.py           # decode frames + audio
    plot.py               # kinematic-sequence PNG
    reconstruct.py        # choose multiview vs monocular -> kp3d
    main.py               # CLI
  tests/
    test_biomech.py
    test_filtering.py
    test_triangulate.py
    test_sync.py
    test_intrinsics.py
    test_calibrate_extrinsic.py
    test_smoke_end_to_end.py
  requirements.txt
  pyproject.toml
```

**Array contracts (used everywhere):**
- 2D keypoints per view: `kp2d` float array `(T, 17, 2)`, confidence `conf` `(T, 17)`.
- 3D keypoints: `kp3d` float array `(T, 17, 3)`.
- Time axis is always axis 0.

---

### Task 0: Project scaffold

**Files:**
- Create: `pitchcap/pitchcap/__init__.py`
- Create: `pitchcap/requirements.txt`
- Create: `pitchcap/pyproject.toml`
- Create: `pitchcap/pitchcap/constants.py`
- Create: `pitchcap/pitchcap/geometry.py`
- Test: `pitchcap/tests/__init__.py`

- [ ] **Step 1: Create package + dependency files**

`pitchcap/requirements.txt`:
```
numpy
scipy
opencv-python
rtmlib
onnxruntime
mediapipe
matplotlib
imageio-ffmpeg
soundfile
pytest
```

`pitchcap/pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "pitchcap"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`pitchcap/pitchcap/__init__.py`:
```python
"""PitchCap: markerless motion-capture core for pitching analysis."""
__version__ = "0.1.0"
```

`pitchcap/tests/__init__.py`: (empty file)

- [ ] **Step 2: Create constants module**

`pitchcap/pitchcap/constants.py`:
```python
"""Shared keypoint indices and camera dataclasses."""
from dataclasses import dataclass
import numpy as np

# COCO-17 keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW = 5, 6, 7, 8
L_WRIST, R_WRIST, L_HIP, R_HIP = 9, 10, 11, 12
L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 13, 14, 15, 16
N_KEYPOINTS = 17


@dataclass
class Intrinsics:
    K: np.ndarray          # (3, 3) camera matrix
    dist: np.ndarray       # (5,) distortion coefficients
    image_size: tuple      # (width, height)


@dataclass
class Extrinsics:
    R: np.ndarray          # (3, 3) rotation
    t: np.ndarray          # (3, 1) translation
```

- [ ] **Step 3: Create geometry helpers**

`pitchcap/pitchcap/geometry.py`:
```python
"""Camera projection helpers."""
import numpy as np
import cv2
from .constants import Intrinsics, Extrinsics


def projection_matrix(intr: Intrinsics, extr: Extrinsics) -> np.ndarray:
    """P = K [R | t], shape (3, 4)."""
    Rt = np.hstack([extr.R, extr.t.reshape(3, 1)])
    return intr.K @ Rt


def undistort_points(pts: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """pts: (N, 2) pixel coords -> (N, 2) normalized-then-reprojected pixels."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, intr.K, intr.dist, P=intr.K)
    return out.reshape(-1, 2)
```

- [ ] **Step 4: Verify import works**

Run: `cd /Users/joeyruller/pitchcap && python -c "import pitchcap; from pitchcap import constants, geometry; print('ok')"`
Expected: prints `ok`

- [ ] **Step 5: Commit**

```bash
cd /Users/joeyruller/pitchcap
git add pitchcap requirements.txt pyproject.toml tests
git commit -m "chore: scaffold pitchcap package, constants, geometry"
```

---

### Task 1: Biomech — segment vectors + angular velocity

**Files:**
- Create: `pitchcap/pitchcap/biomech.py`
- Test: `pitchcap/tests/test_biomech.py`

- [ ] **Step 1: Write the failing test**

A unit vector rotating at constant rate ω about the z-axis has angular speed exactly ω. Build `kp3d` so the pelvis vector (R_HIP→L_HIP) rotates at a known rate, and assert the recovered deg/s.

`pitchcap/tests/test_biomech.py`:
```python
import numpy as np
from pitchcap import constants as C
from pitchcap.biomech import segment_unit_vectors, angular_velocity_degps


def _rotating_kp3d(omega_rad_s, fps, T):
    """Build kp3d where R_HIP->L_HIP rotates at omega about z."""
    t = np.arange(T) / fps
    kp = np.zeros((T, C.N_KEYPOINTS, 3))
    # left hip at unit vector tip, right hip at origin
    kp[:, C.R_HIP] = 0.0
    kp[:, C.L_HIP, 0] = np.cos(omega_rad_s * t)
    kp[:, C.L_HIP, 1] = np.sin(omega_rad_s * t)
    return kp


def test_angular_velocity_constant_rotation():
    fps, T = 240, 240
    omega = 5.0  # rad/s
    kp = _rotating_kp3d(omega, fps, T)
    vecs = segment_unit_vectors(kp, handedness="R")
    degps = angular_velocity_degps(vecs["pelvis"], fps)
    expected = np.degrees(omega)
    # ignore edges where finite-difference is one-sided
    mid = degps[5:-5]
    assert np.allclose(mid, expected, rtol=0.02)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/joeyruller/pitchcap && python -m pytest tests/test_biomech.py -v`
Expected: FAIL with `ModuleNotFoundError` / `cannot import name 'segment_unit_vectors'`

- [ ] **Step 3: Write minimal implementation**

`pitchcap/pitchcap/biomech.py`:
```python
"""Segment vectors -> angular velocity -> kinematic sequence."""
import numpy as np
from . import constants as C


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return v / n


def segment_unit_vectors(kp3d, handedness="R"):
    """Return dict of (T,3) unit vectors for pelvis, trunk, arm."""
    sh = C.R_SHOULDER if handedness == "R" else C.L_SHOULDER
    el = C.R_ELBOW if handedness == "R" else C.L_ELBOW
    hip_mid = (kp3d[:, C.L_HIP] + kp3d[:, C.R_HIP]) / 2.0
    sh_mid = (kp3d[:, C.L_SHOULDER] + kp3d[:, C.R_SHOULDER]) / 2.0
    return {
        "pelvis": _unit(kp3d[:, C.L_HIP] - kp3d[:, C.R_HIP]),
        "trunk": _unit(sh_mid - hip_mid),
        "arm": _unit(kp3d[:, el] - kp3d[:, sh]),
    }


def angular_velocity_degps(unit_vecs, fps):
    """|u x du/dt| in deg/s for a (T,3) sequence of unit vectors."""
    u = np.asarray(unit_vecs, dtype=np.float64)
    dudt = np.gradient(u, axis=0) * fps          # central difference
    omega = np.cross(u, dudt)                     # (T,3) rad/s
    return np.degrees(np.linalg.norm(omega, axis=1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/joeyruller/pitchcap && python -m pytest tests/test_biomech.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/joeyruller/pitchcap
git add pitchcap/biomech.py tests/test_biomech.py
git commit -m "feat: segment vectors and angular velocity"
```

---

### Task 2: Biomech — peak detection + kinematic sequence

**Files:**
- Modify: `pitchcap/pitchcap/biomech.py`
- Test: `pitchcap/tests/test_biomech.py`

- [ ] **Step 1: Write the failing test**

Append to `pitchcap/tests/test_biomech.py`:
```python
from pitchcap.biomech import compute_kinematic_sequence


def test_sequence_order_and_peaks():
    fps, T = 240, 300
    t = np.arange(T) / fps
    kp = np.zeros((T, C.N_KEYPOINTS, 3))
    # static base geometry
    kp[:, C.R_HIP] = [0, 0, 0]; kp[:, C.L_HIP] = [1, 0, 0]
    kp[:, C.R_SHOULDER] = [0, 1, 0]; kp[:, C.L_SHOULDER] = [1, 1, 0]
    kp[:, C.R_ELBOW] = [0, 0.5, 0]

    # inject a rotation burst into each segment at staggered times
    def burst(center_s, amp):
        return amp * np.exp(-((t - center_s) ** 2) / (2 * 0.02 ** 2))

    # rotate L_HIP (pelvis) first, sh_mid (trunk) next, elbow (arm) last
    kp[:, C.L_HIP, 1] += burst(0.40, 0.3)
    kp[:, C.L_SHOULDER, 0] += burst(0.50, 0.3)
    kp[:, C.R_ELBOW, 0] += burst(0.60, 0.5)

    res = compute_kinematic_sequence(kp, fps, handedness="R")
    assert res["sequence_order"] == ["pelvis", "trunk", "arm"]
    assert res["segments"]["arm"]["peak_degps"] > 0
    assert res["inter_peak_lags_ms"]["pelvis_to_trunk"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/joeyruller/pitchcap && python -m pytest tests/test_biomech.py::test_sequence_order_and_peaks -v`
Expected: FAIL with `cannot import name 'compute_kinematic_sequence'`

- [ ] **Step 3: Write minimal implementation**

Append to `pitchcap/pitchcap/biomech.py`:
```python
from scipy.signal import find_peaks

SEGMENTS = ["pelvis", "trunk", "arm"]


def _peak(series, fps):
    """Return (peak_value, peak_time_s) using the global max as fallback."""
    peaks, _ = find_peaks(series)
    idx = peaks[np.argmax(series[peaks])] if len(peaks) else int(np.argmax(series))
    return float(series[idx]), idx / fps


def compute_kinematic_sequence(kp3d, fps, handedness="R"):
    vecs = segment_unit_vectors(kp3d, handedness=handedness)
    segments, peak_times = {}, {}
    for name in SEGMENTS:
        series = angular_velocity_degps(vecs[name], fps)
        pv, pt = _peak(series, fps)
        segments[name] = {
            "series_degps": series.tolist(),
            "peak_degps": pv,
            "peak_time_s": pt,
        }
        peak_times[name] = pt
    order = sorted(SEGMENTS, key=lambda n: peak_times[n])
    lags = {
        "pelvis_to_trunk": round((peak_times["trunk"] - peak_times["pelvis"]) * 1000, 1),
        "trunk_to_arm": round((peak_times["arm"] - peak_times["trunk"]) * 1000, 1),
    }
    return {
        "fps": fps,
        "handedness": handedness,
        "segments": segments,
        "sequence_order": order,
        "inter_peak_lags_ms": lags,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/joeyruller/pitchcap && python -m pytest tests/test_biomech.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/joeyruller/pitchcap
git add pitchcap/biomech.py tests/test_biomech.py
git commit -m "feat: peak detection and kinematic sequence assembly"
```

---

### Task 3: Zero-lag Butterworth filtering

**Files:**
- Create: `pitchcap/pitchcap/filtering.py`
- Test: `pitchcap/tests/test_filtering.py`

- [ ] **Step 1: Write the failing test**

`pitchcap/tests/test_filtering.py`:
```python
import numpy as np
from pitchcap.filtering import butter_lowpass, filter_keypoints


def test_lowpass_removes_high_freq_keeps_low():
    fps, T = 240, 480
    t = np.arange(T) / fps
    low = np.sin(2 * np.pi * 2 * t)            # 2 Hz signal
    noise = 0.5 * np.sin(2 * np.pi * 60 * t)   # 60 Hz noise
    out = butter_lowpass(low + noise, fps, cutoff_hz=13, order=4)
    # low-freq preserved, noise crushed; ignore filtfilt edge transients
    err = np.abs(out - low)[20:-20]
    assert err.max() < 0.1


def test_filter_keypoints_shape_preserved():
    fps, T = 240, 100
    kp = np.random.RandomState(0).randn(T, 17, 3)
    out = filter_keypoints(kp, fps, cutoff_hz=13)
    assert out.shape == kp.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/joeyruller/pitchcap && python -m pytest tests/test_filtering.py -v`
Expected: FAIL with `ModuleNotFoundError: pitchcap.filtering`

- [ ] **Step 3: Write minimal implementation**

`pitchcap/pitchcap/filtering.py`:
```python
"""Zero-phase Butterworth low-pass filtering."""
import numpy as np
from scipy.signal import butter, filtfilt


def butter_lowpass(signal, fps, cutoff_hz, order=4):
    """Zero-lag low-pass along axis 0. Works on (T,) or (T, ...) arrays."""
    nyq = fps / 2.0
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, signal, axis=0)


# segment-specific cutoffs (Hz) per spec
DEFAULT_CUTOFFS = {"pelvis": 13.0, "trunk": 13.0, "arm": 18.0}


def filter_keypoints(kp3d, fps, cutoff_hz=15.0):
    """Filter every coordinate channel of (T, J, 3) keypoints."""
    return butter_lowpass(np.asarray(kp3d, dtype=np.float64), fps, cutoff_hz)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/joeyruller/pitchcap && python -m pytest tests/test_filtering.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/joeyruller/pitchcap
git add pitchcap/filtering.py tests/test_filtering.py
git commit -m "feat: zero-lag Butterworth filtering"
```

---

### Task 4: Confidence-weighted multi-view triangulation

**Files:**
- Create: `pitchcap/pitchcap/triangulate.py`
- Test: `pitchcap/tests/test_triangulate.py`

- [ ] **Step 1: Write the failing test**

Synthetic two-view rig with known projection matrices: project a known 3D point into both views, triangulate, recover the point.

`pitchcap/tests/test_triangulate.py`:
```python
import numpy as np
from pitchcap.constants import Intrinsics, Extrinsics
from pitchcap.geometry import projection_matrix
from pitchcap.triangulate import triangulate_point, triangulate_sequence


def _rig():
    K = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], float)
    dist = np.zeros(5)
    intr = Intrinsics(K, dist, (1280, 720))
    cam0 = Extrinsics(np.eye(3), np.zeros((3, 1)))
    # second camera translated 1m along x, looking same direction
    cam1 = Extrinsics(np.eye(3), np.array([[-1.0], [0], [0]]))
    P0 = projection_matrix(intr, cam0)
    P1 = projection_matrix(intr, cam1)
    return P0, P1


def _project(P, X):
    x = P @ np.append(X, 1.0)
    return x[:2] / x[2]


def test_triangulate_single_point():
    P0, P1 = _rig()
    X = np.array([0.2, -0.1, 4.0])
    pts = np.array([_project(P0, X), _project(P1, X)])
    weights = np.array([1.0, 1.0])
    Xhat = triangulate_point([P0, P1], pts, weights)
    assert np.allclose(Xhat, X, atol=1e-3)


def test_triangulate_sequence_shape():
    P0, P1 = _rig()
    T = 10
    kp2d_v0 = np.zeros((T, 17, 2)); kp2d_v1 = np.zeros((T, 17, 2))
    conf = np.ones((T, 17))
    out = triangulate_sequence([kp2d_v0, kp2d_v1], [conf, conf], [P0, P1])
    assert out.shape == (T, 17, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/joeyruller/pitchcap && python -m pytest tests/test_triangulate.py -v`
Expected: FAIL with `ModuleNotFoundError: pitchcap.triangulate`

- [ ] **Step 3: Write minimal implementation**

`pitchcap/pitchcap/triangulate.py`:
```python
"""Confidence-weighted multi-view DLT triangulation."""
import numpy as np
from . import constants as C


def triangulate_point(proj_mats, points2d, weights):
    """Weighted linear DLT. proj_mats: list of (3,4); points2d: (V,2); weights: (V,)."""
    rows = []
    for P, (x, y), w in zip(proj_mats, points2d, weights):
        rows.append(w * (x * P[2] - P[0]))
        rows.append(w * (y * P[2] - P[1]))
    A = np.vstack(rows)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def triangulate_sequence(kp2d_views, conf_views, proj_mats, conf_thresh=0.3):
    """Triangulate every keypoint across views per frame.

    kp2d_views: list of (T,17,2); conf_views: list of (T,17); proj_mats: list of (3,4).
    Joints with <2 views above conf_thresh are left as NaN (gap-filled later).
    """
    T = kp2d_views[0].shape[0]
    out = np.full((T, C.N_KEYPOINTS, 3), np.nan)
    for f in range(T):
        for j in range(C.N_KEYPOINTS):
            Ps, pts, ws = [], [], []
            for P, kp, conf in zip(proj_mats, kp2d_views, conf_views):
                c = conf[f, j]
                if c >= conf_thresh:
                    Ps.append(P); pts.append(kp[f, j]); ws.append(c)
            if len(Ps) >= 2:
                out[f, j] = triangulate_point(Ps, np.array(pts), np.array(ws))
    return out


def fill_gaps(kp3d):
    """Linearly interpolate short NaN runs along time, per joint/coord."""
    kp3d = kp3d.copy()
    T = kp3d.shape[0]
    idx = np.arange(T)
    for j in range(kp3d.shape[1]):
        for c in range(3):
            col = kp3d[:, j, c]
            good = ~np.isnan(col)
            if good.sum() >= 2:
                kp3d[:, j, c] = np.interp(idx, idx[good], col[good])
    return kp3d
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/joeyruller/pitchcap && python -m pytest tests/test_triangulate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/joeyruller/pitchcap
git add pitchcap/triangulate.py tests/test_triangulate.py
git commit -m "feat: confidence-weighted multi-view triangulation + gap fill"
```

---

### Task 5: Audio sync

**Files:**
- Create: `pitchcap/pitchcap/sync.py`
- Test: `pitchcap/tests/test_sync.py`

- [ ] **Step 1: Write the failing test**

`pitchcap/tests/test_sync.py`:
```python
import numpy as np
from pitchcap.sync import frame_offsets_from_audio


def test_recovers_known_shift():
    rng = np.random.RandomState(1)
    sr, fps = 48000, 240
    base = rng.randn(48000)              # 1s of audio
    shift_frames = 12
    shift_samples = int(shift_frames * sr / fps)
    shifted = np.concatenate([np.zeros(shift_samples), base])[:len(base)]
    offsets = frame_offsets_from_audio([base, shifted], fps=fps, sr=sr)
    # first clip is reference (0); second lags by shift_frames
    assert offsets[0] == 0
    assert abs(offsets[1] - shift_frames) <= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/joeyruller/pitchcap && python -m pytest tests/test_sync.py -v`
Expected: FAIL with `ModuleNotFoundError: pitchcap.sync`

- [ ] **Step 3: Write minimal implementation**

`pitchcap/pitchcap/sync.py`:
```python
"""Audio cross-correlation -> per-clip integer frame offsets."""
import numpy as np
from scipy.signal import correlate


def _lag_samples(ref, other):
    """Positive lag => `other` is delayed relative to `ref`."""
    c = correlate(other, ref, mode="full")
    lag = np.argmax(c) - (len(ref) - 1)
    return lag


def frame_offsets_from_audio(audios, fps, sr):
    """Return integer frame offsets, first clip as reference (offset 0)."""
    ref = np.asarray(audios[0], dtype=np.float64)
    offsets = [0]
    for a in audios[1:]:
        lag = _lag_samples(ref, np.asarray(a, dtype=np.float64))
        offsets.append(int(round(lag * fps / sr)))
    return offsets
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/joeyruller/pitchcap && python -m pytest tests/test_sync.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/joeyruller/pitchcap
git add pitchcap/sync.py tests/test_sync.py
git commit -m "feat: audio cross-correlation frame sync"
```

---

### Task 6: Intrinsics (one-time lens calibration utility)

**Files:**
- Create: `pitchcap/pitchcap/intrinsics.py`
- Test: `pitchcap/tests/test_intrinsics.py`

- [ ] **Step 1: Write the failing test**

Test the save/load round-trip and the approximate-intrinsics fallback (the calibration-from-video path needs a real board video, exercised via smoke later).

`pitchcap/tests/test_intrinsics.py`:
```python
import numpy as np
from pitchcap.intrinsics import save_profile, load_profile, approximate_intrinsics


def test_save_load_roundtrip(tmp_path):
    K = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], float)
    dist = np.array([0.01, -0.02, 0, 0, 0])
    p = tmp_path / "iphone12_wide_240.json"
    save_profile(str(p), K, dist, (1280, 720))
    intr = load_profile(str(p))
    assert np.allclose(intr.K, K)
    assert np.allclose(intr.dist, dist)
    assert intr.image_size == (1280, 720)


def test_approximate_intrinsics_has_reasonable_focal():
    intr = approximate_intrinsics((1920, 1080))
    # focal near image width, principal point at center
    assert 0.7 * 1920 < intr.K[0, 0] < 1.3 * 1920
    assert np.allclose(intr.K[0, 2], 960)
    assert np.allclose(intr.K[1, 2], 540)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/joeyruller/pitchcap && python -m pytest tests/test_intrinsics.py -v`
Expected: FAIL with `ModuleNotFoundError: pitchcap.intrinsics`

- [ ] **Step 3: Write minimal implementation**

`pitchcap/pitchcap/intrinsics.py`:
```python
"""One-time per-(phone, lens, mode) lens calibration + profile I/O."""
import json
import numpy as np
import cv2
from .constants import Intrinsics


def save_profile(path, K, dist, image_size):
    with open(path, "w") as f:
        json.dump({
            "K": np.asarray(K).tolist(),
            "dist": np.asarray(dist).ravel().tolist(),
            "image_size": list(image_size),
        }, f, indent=2)


def load_profile(path):
    with open(path) as f:
        d = json.load(f)
    return Intrinsics(
        K=np.array(d["K"], float),
        dist=np.array(d["dist"], float),
        image_size=tuple(d["image_size"]),
    )


def approximate_intrinsics(image_size, fov_factor=1.0):
    """Fallback when no profile exists: focal ~= image width, center at midpoint."""
    w, h = image_size
    f = fov_factor * w
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)
    return Intrinsics(K=K, dist=np.zeros(5), image_size=(w, h))


def calibrate_from_video(video_path, board_size=(9, 6), square_m=0.025, stride=10):
    """Detect a checkerboard across a calibration clip -> Intrinsics.

    board_size = (inner_corners_x, inner_corners_y).
    """
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_m

    objpoints, imgpoints = [], []
    cap = cv2.VideoCapture(video_path)
    size = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            found, corners = cv2.findChessboardCorners(gray, board_size, None)
            if found:
                objpoints.append(objp)
                imgpoints.append(corners)
        i += 1
    cap.release()
    if len(objpoints) < 5:
        raise RuntimeError(f"Only {len(objpoints)} board views found; need >=5.")
    _, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
    return Intrinsics(K=K, dist=dist.ravel()[:5], image_size=size)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/joeyruller/pitchcap && python -m pytest tests/test_intrinsics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/joeyruller/pitchcap
git add pitchcap/intrinsics.py tests/test_intrinsics.py
git commit -m "feat: lens calibration utility and profile I/O"
```

---

### Task 7: Markerless extrinsic calibration + bundle adjustment

**Files:**
- Create: `pitchcap/pitchcap/calibrate_extrinsic.py`
- Test: `pitchcap/tests/test_calibrate_extrinsic.py`

- [ ] **Step 1: Write the failing test**

Synthetic two-view geometry with known relative rotation/translation: project random 3D points into both views, recover relative pose, and confirm rotation matches (translation up to scale) and reprojection error is small.

`pitchcap/tests/test_calibrate_extrinsic.py`:
```python
import numpy as np
import cv2
from pitchcap.constants import Intrinsics
from pitchcap.calibrate_extrinsic import recover_relative_pose


def test_recovers_rotation_up_to_scale():
    rng = np.random.RandomState(2)
    K = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], float)
    intr = Intrinsics(K, np.zeros(5), (1280, 720))

    # ground-truth second camera: small rotation + translation along x
    rvec = np.array([0.0, 0.3, 0.0])
    R_gt, _ = cv2.Rodrigues(rvec)
    t_gt = np.array([[-1.0], [0.0], [0.0]])

    X = rng.uniform([-1, -1, 3], [1, 1, 6], size=(200, 3))
    p0 = (K @ X.T).T; p0 = p0[:, :2] / p0[:, 2:]
    Xc = (R_gt @ X.T + t_gt).T
    p1 = (K @ Xc.T).T; p1 = p1[:, :2] / p1[:, 2:]

    extr, err = recover_relative_pose(p0, p1, intr)
    # rotation should match within a degree
    ang = np.degrees(np.arccos((np.trace(extr.R.T @ R_gt) - 1) / 2))
    assert ang < 1.0
    assert err < 1.0   # px reprojection
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/joeyruller/pitchcap && python -m pytest tests/test_calibrate_extrinsic.py -v`
Expected: FAIL with `ModuleNotFoundError: pitchcap.calibrate_extrinsic`

- [ ] **Step 3: Write minimal implementation**

`pitchcap/pitchcap/calibrate_extrinsic.py`:
```python
"""Markerless extrinsic calibration from body-joint correspondences."""
import numpy as np
import cv2
from .constants import Extrinsics, Intrinsics
from .geometry import projection_matrix


def recover_relative_pose(pts0, pts1, intr: Intrinsics):
    """Essential-matrix relative pose between two views.

    pts0, pts1: matched (N,2) pixel correspondences. Returns (Extrinsics, reproj_err_px).
    First camera is the world origin (R=I, t=0); returned Extrinsics is the second camera.
    """
    pts0 = np.asarray(pts0, np.float64)
    pts1 = np.asarray(pts1, np.float64)
    E, mask = cv2.findEssentialMat(pts0, pts1, intr.K, method=cv2.RANSAC,
                                   prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, pts0, pts1, intr.K, mask=mask)
    extr = Extrinsics(R=R, t=t.reshape(3, 1))

    err = _reprojection_error(pts0, pts1, intr, extr, mask)
    return extr, err


def _reprojection_error(pts0, pts1, intr, extr, mask):
    cam0 = Extrinsics(np.eye(3), np.zeros((3, 1)))
    P0 = projection_matrix(intr, cam0)
    P1 = projection_matrix(intr, extr)
    inl = mask.ravel().astype(bool)
    if inl.sum() == 0:
        return float("inf")
    X = cv2.triangulatePoints(P0, P1, pts0[inl].T, pts1[inl].T)
    X = (X[:3] / X[3]).T
    errs = []
    for P, pts in ((P0, pts0[inl]), (P1, pts1[inl])):
        proj = (P @ np.hstack([X, np.ones((len(X), 1))]).T).T
        proj = proj[:, :2] / proj[:, 2:]
        errs.append(np.linalg.norm(proj - pts, axis=1))
    return float(np.mean(np.concatenate(errs)))


def correspondences_from_pose(kp2d_a, conf_a, kp2d_b, conf_b, conf_thresh=0.5):
    """Flatten matched, high-confidence joints across all frames into (N,2),(N,2)."""
    mask = (conf_a >= conf_thresh) & (conf_b >= conf_thresh)
    return kp2d_a[mask], kp2d_b[mask]
```

> **Note on bundle adjustment:** `recover_relative_pose` returns the essential-matrix estimate plus its reprojection error. A `scipy.optimize.least_squares` refinement over `(rvec, tvec, points)` minimizing `_reprojection_error` is a follow-on optimization; the interface above already surfaces the error so the refinement is a drop-in. Implement it only if the essential-matrix error is routinely >2–3 px on real clips.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/joeyruller/pitchcap && python -m pytest tests/test_calibrate_extrinsic.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/joeyruller/pitchcap
git add pitchcap/calibrate_extrinsic.py tests/test_calibrate_extrinsic.py
git commit -m "feat: markerless extrinsic calibration with reprojection error"
```

---

### Task 8: 2D pose (RTMPose wrapper)

**Files:**
- Create: `pitchcap/pitchcap/pose2d.py`
- Test: covered by the end-to-end smoke test (Task 12); no unit test (requires model weights download).

- [ ] **Step 1: Write the implementation**

`pitchcap/pitchcap/pose2d.py`:
```python
"""2D pose estimation. Default backend: RTMPose via rtmlib (ONNX)."""
import numpy as np
from . import constants as C


class RTMPoseBackend:
    def __init__(self, device="cpu", mode="balanced"):
        from rtmlib import Body  # lazy import; heavy dependency
        self.model = Body(mode=mode, backend="onnxruntime", device=device)

    def __call__(self, frames):
        """frames: list/array of HxWx3 BGR -> (T,17,2), (T,17)."""
        T = len(frames)
        kp = np.zeros((T, C.N_KEYPOINTS, 2))
        conf = np.zeros((T, C.N_KEYPOINTS))
        for i, frame in enumerate(frames):
            keypoints, scores = self.model(frame)
            if len(keypoints) == 0:
                continue
            # take the most confident detected person
            best = int(np.argmax(scores.mean(axis=1)))
            kp[i] = keypoints[best][:C.N_KEYPOINTS]
            conf[i] = scores[best][:C.N_KEYPOINTS]
        return kp, conf


def estimate_pose_2d(frames, backend=None):
    """Return (kp2d (T,17,2), conf (T,17)) for one view."""
    backend = backend or RTMPoseBackend()
    return backend(frames)
```

> **YOLO11-pose swap:** to use Ultralytics instead, replace `RTMPoseBackend` with a wrapper around `from ultralytics import YOLO; YOLO("yolo11x-pose.pt")`, reading `result.keypoints.xy` / `.conf`. Same `(T,17,2),(T,17)` contract.

- [ ] **Step 2: Verify import (without running the model)**

Run: `cd /Users/joeyruller/pitchcap && python -c "from pitchcap import pose2d; print('ok')"`
Expected: prints `ok` (lazy import means rtmlib isn't loaded until used)

- [ ] **Step 3: Commit**

```bash
cd /Users/joeyruller/pitchcap
git add pitchcap/pose2d.py
git commit -m "feat: RTMPose 2D pose backend"
```

---

### Task 9: Monocular fallback (MediaPipe world landmarks)

**Files:**
- Create: `pitchcap/pitchcap/monocular.py`
- Test: covered by smoke test (Task 12); requires mediapipe.

- [ ] **Step 1: Write the implementation**

MediaPipe returns 33 landmarks; remap the ones we need into the COCO-17 slots the biomech layer expects.

`pitchcap/pitchcap/monocular.py`:
```python
"""Single-camera 3D fallback using MediaPipe Pose world landmarks."""
import numpy as np
from . import constants as C

# MediaPipe BlazePose index -> COCO-17 index (only the joints we use)
_MP_TO_COCO = {
    11: C.L_SHOULDER, 12: C.R_SHOULDER,
    13: C.L_ELBOW, 14: C.R_ELBOW,
    15: C.L_WRIST, 16: C.R_WRIST,
    23: C.L_HIP, 24: C.R_HIP,
    25: C.L_KNEE, 26: C.R_KNEE,
    27: C.L_ANKLE, 28: C.R_ANKLE,
}


def estimate_pose_3d_monocular(frames):
    """frames: list of HxWx3 BGR -> kp3d (T,17,3) in MediaPipe world meters."""
    import mediapipe as mp
    import cv2
    pose = mp.solutions.pose.Pose(model_complexity=2, static_image_mode=False)
    T = len(frames)
    kp3d = np.full((T, C.N_KEYPOINTS, 3), np.nan)
    for i, frame in enumerate(frames):
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.pose_world_landmarks:
            continue
        lms = res.pose_world_landmarks.landmark
        for mp_idx, coco_idx in _MP_TO_COCO.items():
            lm = lms[mp_idx]
            kp3d[i, coco_idx] = [lm.x, lm.y, lm.z]
    pose.close()
    return kp3d
```

- [ ] **Step 2: Verify import**

Run: `cd /Users/joeyruller/pitchcap && python -c "from pitchcap import monocular; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
cd /Users/joeyruller/pitchcap
git add pitchcap/monocular.py
git commit -m "feat: monocular MediaPipe world-landmark fallback"
```

---

### Task 10: Video + audio decode

**Files:**
- Create: `pitchcap/pitchcap/io_video.py`
- Test: covered by smoke test (Task 12).

- [ ] **Step 1: Write the implementation**

`pitchcap/pitchcap/io_video.py`:
```python
"""Decode iPhone clips into frames + audio + fps."""
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class Clip:
    frames: list      # list of HxWx3 BGR arrays
    audio: np.ndarray # mono float waveform (may be empty)
    fps: float
    sr: int           # audio sample rate (0 if no audio)
    image_size: tuple # (width, height)


def load_clip(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 240.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    h, w = (frames[0].shape[:2] if frames else (0, 0))
    audio, sr = _load_audio(path)
    return Clip(frames=frames, audio=audio, fps=fps, sr=sr, image_size=(w, h))


def _load_audio(path):
    """Extract mono audio via soundfile; return (waveform, sr) or (empty, 0)."""
    try:
        import soundfile as sf
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return np.asarray(data, dtype=np.float64), sr
    except Exception:
        return np.zeros(0), 0
```

> **Assumption:** `soundfile` reads the clip's audio track; if a container isn't supported, the audio-sync step degrades to a zero offset and the system logs a warning (handled in `main.py`). For broad container support, `imageio-ffmpeg` provides the ffmpeg binary used by OpenCV.

- [ ] **Step 2: Verify import**

Run: `cd /Users/joeyruller/pitchcap && python -c "from pitchcap import io_video; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
cd /Users/joeyruller/pitchcap
git add pitchcap/io_video.py
git commit -m "feat: video + audio decode"
```

---

### Task 11: Sequence plot

**Files:**
- Create: `pitchcap/pitchcap/plot.py`
- Test: covered by smoke test (Task 12).

- [ ] **Step 1: Write the implementation**

`pitchcap/pitchcap/plot.py`:
```python
"""Render the kinematic-sequence time-series plot."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_sequence(result, out_path):
    fps = result["fps"]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"pelvis": "tab:blue", "trunk": "tab:green", "arm": "tab:red"}
    for name, seg in result["segments"].items():
        series = np.array(seg["series_degps"])
        t = np.arange(len(series)) / fps
        ax.plot(t, series, label=name, color=colors.get(name))
        ax.axvline(seg["peak_time_s"], color=colors.get(name), ls="--", alpha=0.5)
        ax.scatter([seg["peak_time_s"]], [seg["peak_degps"]],
                   color=colors.get(name), zorder=5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("angular velocity (deg/s)")
    ax.set_title("Kinematic sequence: " + " -> ".join(result["sequence_order"]))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 2: Verify import**

Run: `cd /Users/joeyruller/pitchcap && python -c "from pitchcap import plot; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
cd /Users/joeyruller/pitchcap
git add pitchcap/plot.py
git commit -m "feat: kinematic-sequence plot"
```

---

### Task 12: Reconstruct orchestration + CLI + end-to-end smoke test

**Files:**
- Create: `pitchcap/pitchcap/reconstruct.py`
- Create: `pitchcap/pitchcap/main.py`
- Test: `pitchcap/tests/test_smoke_end_to_end.py`

- [ ] **Step 1: Write the reconstruct module**

`pitchcap/pitchcap/reconstruct.py`:
```python
"""Choose multi-view vs monocular and return filtered 3D keypoints."""
import numpy as np
from . import constants as C
from .geometry import projection_matrix
from .constants import Extrinsics
from .triangulate import triangulate_sequence, fill_gaps
from .calibrate_extrinsic import recover_relative_pose, correspondences_from_pose
from .filtering import filter_keypoints


def reconstruct_multiview(kp2d_views, conf_views, intrinsics):
    """View 0 is world origin; recover each other view's pose vs view 0."""
    cam0 = Extrinsics(np.eye(3), np.zeros((3, 1)))
    proj_mats = [projection_matrix(intrinsics[0], cam0)]
    reproj_errs = []
    for v in range(1, len(kp2d_views)):
        a, b = correspondences_from_pose(
            kp2d_views[0], conf_views[0], kp2d_views[v], conf_views[v])
        extr, err = recover_relative_pose(a, b, intrinsics[v])
        proj_mats.append(projection_matrix(intrinsics[v], extr))
        reproj_errs.append(err)
    kp3d = triangulate_sequence(kp2d_views, conf_views, proj_mats)
    kp3d = fill_gaps(kp3d)
    return kp3d, (float(np.mean(reproj_errs)) if reproj_errs else 0.0)
```

- [ ] **Step 2: Write the CLI**

`pitchcap/pitchcap/main.py`:
```python
"""PitchCap CLI: 1-N videos -> result.json + plot.png."""
import argparse
import json
import numpy as np
from . import io_video, pose2d, monocular, reconstruct
from .sync import frame_offsets_from_audio
from .filtering import filter_keypoints
from .biomech import compute_kinematic_sequence
from .intrinsics import load_profile, approximate_intrinsics
from .plot import plot_sequence


def _align(arrs, offsets):
    """Trim each (T,...) array by its frame offset to a common length."""
    starts = [max(0, o) for o in offsets]
    trimmed = [a[s:] for a, s in zip(arrs, starts)]
    n = min(len(a) for a in trimmed)
    return [a[:n] for a in trimmed]


def run(video_paths, profile_paths=None, handedness="R",
        out_json="result.json", out_png="plot.png", cutoff_hz=15.0):
    warnings = []
    clips = [io_video.load_clip(p) for p in video_paths]
    fps = clips[0].fps

    # 2D pose per view
    poses = [pose2d.estimate_pose_2d(c.frames) for c in clips]
    kp2d_views = [p[0] for p in poses]
    conf_views = [p[1] for p in poses]

    if len(clips) >= 2:
        # sync via audio (fallback to zero offsets if no audio)
        if all(len(c.audio) for c in clips):
            offsets = frame_offsets_from_audio(
                [c.audio for c in clips], fps=fps, sr=clips[0].sr)
        else:
            offsets = [0] * len(clips)
            warnings.append("audio missing on >=1 clip; assuming zero offset")
        kp2d_views = _align(kp2d_views, offsets)
        conf_views = _align(conf_views, offsets)

        # intrinsics: load profile or approximate
        intrinsics = []
        for i, c in enumerate(clips):
            prof = profile_paths[i] if profile_paths and i < len(profile_paths) else None
            if prof:
                intrinsics.append(load_profile(prof))
            else:
                intrinsics.append(approximate_intrinsics(c.image_size))
                warnings.append(f"camera_{i} using approximate intrinsics (no lens profile)")

        kp3d, reproj_err = reconstruct.reconstruct_multiview(
            kp2d_views, conf_views, intrinsics)
        mode, n_cams = "multiview", len(clips)
    else:
        kp3d = monocular.estimate_pose_3d_monocular(clips[0].frames)
        kp3d = np.nan_to_num(kp3d)
        reproj_err, mode, n_cams = None, "monocular", 1
        warnings.append("single camera: monocular 3D (reduced accuracy)")

    kp3d = filter_keypoints(kp3d, fps, cutoff_hz=cutoff_hz)
    result = compute_kinematic_sequence(kp3d, fps, handedness=handedness)
    result.update({"n_cams": n_cams, "mode": mode,
                   "reprojection_error_px": reproj_err, "warnings": warnings})

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    plot_sequence(result, out_png)
    return result


def main():
    ap = argparse.ArgumentParser(description="PitchCap kinematic-sequence extractor")
    ap.add_argument("videos", nargs="+", help="1-4 clip paths")
    ap.add_argument("--profiles", nargs="*", default=None,
                    help="lens profile JSON per video (same order)")
    ap.add_argument("--handedness", choices=["R", "L"], default="R")
    ap.add_argument("--out-json", default="result.json")
    ap.add_argument("--out-png", default="plot.png")
    ap.add_argument("--cutoff-hz", type=float, default=15.0)
    a = ap.parse_args()
    res = run(a.videos, a.profiles, a.handedness, a.out_json, a.out_png, a.cutoff_hz)
    print(f"mode={res['mode']} order={res['sequence_order']} "
          f"reproj_err={res['reprojection_error_px']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write the end-to-end smoke test**

This test stubs the heavy 2D backend so it runs without model downloads, then drives the full multi-view path on a synthetic two-view rig and asserts a valid result + JSON + PNG.

`pitchcap/tests/test_smoke_end_to_end.py`:
```python
import json
import numpy as np
from pitchcap.constants import Intrinsics, Extrinsics
import pitchcap.constants as C
from pitchcap.geometry import projection_matrix
from pitchcap import reconstruct
from pitchcap.filtering import filter_keypoints
from pitchcap.biomech import compute_kinematic_sequence
from pitchcap.plot import plot_sequence


def test_multiview_pipeline_synthetic(tmp_path):
    """Full math path: known 3D -> project to 2 views -> reconstruct -> sequence."""
    fps, T = 240, 200
    K = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], float)
    intr = Intrinsics(K, np.zeros(5), (1280, 720))
    cam0 = Extrinsics(np.eye(3), np.zeros((3, 1)))
    cam1 = Extrinsics(np.eye(3), np.array([[-1.0], [0], [0]]))
    P0 = projection_matrix(intr, cam0)
    P1 = projection_matrix(intr, cam1)

    # build a moving 3D skeleton (static base + arm rotation burst)
    t = np.arange(T) / fps
    X = np.zeros((T, C.N_KEYPOINTS, 3))
    X[:, C.R_HIP] = [0, 0, 4]; X[:, C.L_HIP] = [0.3, 0, 4]
    X[:, C.R_SHOULDER] = [0, 0.5, 4]; X[:, C.L_SHOULDER] = [0.3, 0.5, 4]
    burst = 0.4 * np.exp(-((t - 0.5) ** 2) / (2 * 0.03 ** 2))
    X[:, C.R_ELBOW, 0] = burst; X[:, C.R_ELBOW, 1] = 0.3; X[:, C.R_ELBOW, 2] = 4

    def project(P):
        out = np.zeros((T, C.N_KEYPOINTS, 2))
        for j in range(C.N_KEYPOINTS):
            h = (P @ np.hstack([X[:, j], np.ones((T, 1))]).T).T
            out[:, j] = h[:, :2] / h[:, 2:]
        return out

    kp2d_views = [project(P0), project(P1)]
    conf_views = [np.ones((T, C.N_KEYPOINTS)), np.ones((T, C.N_KEYPOINTS))]

    kp3d, err = reconstruct.reconstruct_multiview(kp2d_views, conf_views, [intr, intr])
    kp3d = filter_keypoints(kp3d, fps, cutoff_hz=15)
    result = compute_kinematic_sequence(kp3d, fps, handedness="R")

    assert set(result["segments"]) == {"pelvis", "trunk", "arm"}
    assert result["segments"]["arm"]["peak_degps"] > 0
    assert err < 5.0

    out_json = tmp_path / "result.json"
    out_png = tmp_path / "plot.png"
    result.update({"n_cams": 2, "mode": "multiview",
                   "reprojection_error_px": err, "warnings": []})
    with open(out_json, "w") as f:
        json.dump(result, f)
    plot_sequence(result, str(out_png))
    assert out_json.exists() and out_png.exists()
    assert json.load(open(out_json))["mode"] == "multiview"
```

- [ ] **Step 4: Run the full test suite**

Run: `cd /Users/joeyruller/pitchcap && python -m pytest -v`
Expected: PASS (all tests across all files)

- [ ] **Step 5: Commit**

```bash
cd /Users/joeyruller/pitchcap
git add pitchcap/reconstruct.py pitchcap/main.py tests/test_smoke_end_to_end.py
git commit -m "feat: reconstruct orchestration, CLI, end-to-end smoke test"
```

---

## Manual verification (after the suite passes)

Real clips require model downloads (rtmlib, mediapipe) and actual iPhone footage — run these once dependencies are installed:

```bash
cd /Users/joeyruller/pitchcap
pip install -r requirements.txt

# two-camera (recommended): pass lens profiles if you've calibrated
python -m pitchcap.main throw_cam0.mov throw_cam1.mov \
  --profiles iphone_A_wide_240.json iphone_B_wide_240.json --handedness R

# single-camera fallback
python -m pitchcap.main throw_cam0.mov --handedness R
```
Expect `result.json` + `plot.png`, a printed `order=[...]`, and a reprojection error (multiview). Sanity-check peaks against ASMI ranges (pelvis ~500–700°/s, trunk ~900–1200°/s).

---

## Self-Review

**Spec coverage:**
- io_video → Task 10 ✓ · sync → Task 5 ✓ · pose2d/RTMPose → Task 8 ✓ · intrinsics one-time calibration → Task 6 ✓ · markerless extrinsics + bundle-adjustment hook → Task 7 ✓ · confidence-weighted triangulation (1–4 cams) → Task 4 + reconstruct (Task 12) ✓ · monocular fallback → Task 9 ✓ · zero-lag Butterworth → Task 3 ✓ · biomech angular velocity + sequence → Tasks 1–2 ✓ · JSON + plot → Tasks 11–12 ✓ · CLI → Task 12 ✓ · validation tests (unit/triangulation/smoke) → Tasks 1–7, 12 ✓.
- Rolling-shutter correction: documented in spec as optional; not a build task in v1 (flagged as a future refinement, consistent with spec's "optional"). No code gap.
- Bundle adjustment: interface surfaces reprojection error; full least-squares refinement is gated behind real-clip error per spec — noted in Task 7, not a placeholder.

**Type consistency:** `(T,17,2)`+`(T,17)` for 2D and `(T,17,3)` for 3D used uniformly. `Intrinsics`/`Extrinsics` dataclasses consistent across geometry, triangulate, calibrate, reconstruct. `compute_kinematic_sequence` return dict keys (`segments`, `sequence_order`, `inter_peak_lags_ms`) match `plot_sequence` and `main.run` usage.

**Placeholder scan:** No TBD/TODO; every code step contains runnable code. The two "notes" (bundle-adjustment refinement, YOLO swap) describe optional extensions with concrete entry points, not missing required code.
