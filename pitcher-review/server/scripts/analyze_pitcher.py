#!/usr/bin/env python3
"""
Pitcher side-view biomechanics analyzer.
Optimized for first-base-side camera angle (lefty pitcher).
Measures: torso/hip rotation, hip-shoulder separation (X-factor), arm speed.

Uses MediaPipe Tasks API (mediapipe >= 0.10).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from fractions import Fraction
from types import SimpleNamespace
import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


def json_sanitize(obj):
    """Recursively replace non-finite floats (NaN/Infinity) with None.

    Python's ``json.dump`` emits the literal ``NaN``/``Infinity`` tokens, which
    are invalid JSON: the Node server and the browser both reject them with a
    SyntaxError, turning one bad value anywhere in metrics.json into an
    HTTP-500 / unreadable-result for the whole analysis. Applied at every
    metrics.json write as the general guarantee that the output is valid JSON,
    independent of which metric/engine produced it."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    return obj


def find_ffmpeg():
    """Locate an ffmpeg binary: prefer one on PATH, otherwise fall back to
    the portable static build bundled by the imageio-ffmpeg package (so the
    H.264 re-encode works even on machines without a system ffmpeg, e.g. a
    fresh macOS install)."""
    path = shutil.which('ffmpeg')
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def find_ffprobe():
    """Locate an ffprobe binary, mirroring find_ffmpeg()."""
    path = shutil.which('ffprobe')
    if path:
        return path
    try:
        import imageio_ffmpeg
        candidate = os.path.join(os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()), 'ffprobe')
        return candidate if os.path.exists(candidate) else None
    except Exception:
        return None


def detect_frame_rates(video_path, decoded_frames, cap_fps):
    """Return (playback_fps, motion_fps, confident) for a video.

    cv2's CAP_PROP_FPS reports the container's average frame rate, which for
    variable-frame-rate slow-motion exports (e.g. an iPhone 240/300fps slow-mo
    clip muxed for ~30fps playback) can be wildly wrong — neither the rate the
    player advances frames at nor the rate the footage was captured at, which
    badly skews every speed/timing metric (and, separately, MediaPipe's
    detect_for_video timestamps).

    ffprobe exposes both pieces directly:
      - format.duration combined with the actual decoded frame count gives the
        true playback rate — what the annotated output video and the UI's
        time/frame sync should use.
      - r_frame_rate is the camera's nominal capture rate; when it's
        substantially higher than the playback rate, the footage is slow
        motion and real-world speeds/phase timings must be computed against
        it instead, or they read many times too low (and the delivery's
        phases collapse into a handful of frames).

    `confident` is False when r_frame_rate doesn't clearly indicate the true
    capture rate (see threshold below) — some slow-motion exports (e.g. iPhone
    "intent=0" HEVC clips) report an r_frame_rate that reflects neither the
    sensor rate nor the playback rate, so motion_fps falls back to
    playback_fps but may still understate the true rate by the clip's actual
    slow-mo factor. When a second camera's rate IS confidently detected,
    analysis.js cross-correlates the two clips' motion to recover the true
    rate for the unconfident one.

    Falls back to (cap_fps, cap_fps, True) — today's behaviour — if ffprobe is
    unavailable or the probe doesn't yield usable numbers.
    """
    ffprobe = find_ffprobe()
    if not ffprobe or decoded_frames <= 0:
        return cap_fps, cap_fps, True

    try:
        out = subprocess.run(
            [ffprobe, '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate',
             '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1', video_path],
            capture_output=True, text=True, timeout=10,
        ).stdout
        info = dict(line.split('=', 1) for line in out.strip().splitlines() if '=' in line)
        duration = float(info['duration'])
        r_frame_rate = float(Fraction(info['r_frame_rate']))
    except Exception:
        return cap_fps, cap_fps, True

    if duration <= 0 or r_frame_rate <= 0:
        return cap_fps, cap_fps, True

    # A genuine slow-motion capture (120/240/300fps recorded, ~30fps playback)
    # has a ratio of 4x or more. Ordinary variable-frame-rate footage (e.g. a
    # screen recording with dropped frames) can land r_frame_rate/playback_fps
    # around 1.2-1.6x without being slow motion at all — using r_frame_rate
    # there would inflate every speed by that same factor. 2x sits well below
    # the slow-mo floor and above realistic VFR drift, so it cleanly separates
    # the two cases.
    playback_fps = decoded_frames / duration
    confident = r_frame_rate >= playback_fps * 2.0
    motion_fps = r_frame_rate if confident else playback_fps
    return playback_fps, motion_fps, confident


# ── Model download ────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'pose_landmarker_heavy.task')
MODEL_URL  = ('https://storage.googleapis.com/mediapipe-models/'
              'pose_landmarker/pose_landmarker_heavy/float16/latest/'
              'pose_landmarker_heavy.task')

def ensure_model():
    if os.path.exists(MODEL_FILE):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f'[model] Downloading pose model (~28 MB)…', flush=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print(f'[model] Saved to {MODEL_FILE}', flush=True)


# ── Landmark index map ────────────────────────────────────────────────────────
# MediaPipe Pose landmarks (BlazePose 33-keypoint model)
LM = {
    'nose': 0,
    'left_shoulder': 11,  'right_shoulder': 12,
    'left_elbow':    13,  'right_elbow':    14,
    'left_wrist':    15,  'right_wrist':    16,
    'left_hip':      23,  'right_hip':      24,
    'left_knee':     25,  'right_knee':     26,
    'left_ankle':    27,  'right_ankle':    28,
}

THROW = {
    'left':  dict(shoulder=11, elbow=13, wrist=15, hip=23, knee=25, ankle=27),
    'right': dict(shoulder=12, elbow=14, wrist=16, hip=24, knee=26, ankle=28),
}
LEAD = {
    'left':  dict(shoulder=12, elbow=14, wrist=16, hip=24, knee=26, ankle=28),
    'right': dict(shoulder=11, elbow=13, wrist=15, hip=23, knee=25, ankle=27),
}

PHASE_COLORS_BGR = {
    'setup':          (180, 180, 180),
    'windup':         (0,   200, 255),
    'stride':         (255, 200, 0),
    'foot_strike':    (100, 255, 0),
    'arm_cocking':    (0,   100, 255),
    'acceleration':   (0,   0,   255),
    'release':        (255, 0,   200),
    'follow_through': (255, 100, 100),
}

SKELETON_CONNECTIONS = [
    (11, 12, (200, 200, 60),  2),   # shoulders
    (23, 24, (200, 200, 60),  2),   # hips
    (11, 23, (160, 160, 60),  2),   # left torso
    (12, 24, (160, 160, 60),  2),   # right torso
    (23, 25, (140, 140, 140), 2),   # left thigh
    (25, 27, (140, 140, 140), 2),   # left shin
    (24, 26, (140, 140, 140), 2),   # right thigh
    (26, 28, (140, 140, 140), 2),   # right shin
]

# ── Robustness thresholds ───────────────────────────────────────────────────
MIN_VISIBILITY   = 0.35  # landmark visibility below this is not trusted
MAX_POS_JUMP     = 0.15  # max plausible frame-to-frame landmark move (normalized coords)
MIN_ROT_VEC_MAG  = 0.035 # min |hip/shoulder line vector| (x-z plane) to trust its angle
MIN_MASK_CONF    = 0.4   # min person-segmentation confidence to trust a landmark there
MAX_INTERP_GAP_S = 0.1   # gaps in tracking shorter than this are linearly interpolated
                         # between the surrounding trusted samples rather than held flat;
                         # longer gaps are left held since the true value likely drifted
                         # too far to estimate from a straight line.
PEAK_MIN_VALID_FRAC = 0.34  # below this, a peak-search window was too untracked to trust

# ── Velocity filtering (Winter-style zero-lag Butterworth, see lowpass) ──────
# Cutoffs are real frequencies (Hz), so the same physical noise band is rejected
# regardless of frame rate. Arm (elbow-extension) peaks are the sharpest/fastest
# yet cleanest signal, so they keep a wider band; the depth-derived hip/shoulder
# rotation angles are noisier and their true peaks broader, so they're filtered
# harder. Tuned on synthetic pitching signals across 24-240fps.
ARM_CUTOFF_HZ = 14.0  # elbow-extension rate (arm speed)
ROT_CUTOFF_HZ = 6.0   # hip / shoulder rotation speed, trunk tilt

# ── Segmentation mask ────────────────────────────────────────────────────────
# The pitcher/background mask is used to confidence-gate the robust trackers
# (MIN_MASK_CONF); it is no longer used for any cosmetic background blur.
MASK_DOWNSCALE_W = 320  # width (px) to downsample segmentation masks to before storing

# ── Pitcher framing / zoom ───────────────────────────────────────────────────
# Wide or cluttered shots (e.g. a garage with the pitcher small in frame) hurt
# both pose-detection accuracy and the precision of every downstream angle
# measurement. A quick first pass locates the pitcher across the whole clip so
# the main pass can crop in on them, effectively raising their resolution.
SCAN_SAMPLES   = 24   # number of frames sampled in the first pass to locate the pitcher
ROI_PAD        = 0.25 # margin added around the scanned bounding box, as a fraction of its size
ROI_MIN_SPAN   = 0.92 # don't bother cropping if the ROI would already cover ~the whole frame
CROP_ALIGN     = 16   # crop width/height are snapped to a multiple of this many px


# ── Math helpers ──────────────────────────────────────────────────────────────

def angle_3pt(ax, ay, bx, by, cx, cy):
    """Angle at B between BA and BC (degrees)."""
    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = (ba[0]**2+ba[1]**2)**0.5 * (bc[0]**2+bc[1]**2)**0.5 + 1e-9
    return float(np.degrees(np.arccos(np.clip(dot / mag, -1.0, 1.0))))


def angle_3pt_3d(ax, ay, az, bx, by, bz, cx, cy, cz):
    """3D angle at B between BA and BC (degrees). Used for the elbow angle from
    MediaPipe *world* landmarks: a 3D joint angle stays well-conditioned even
    when the limb points toward/away from the camera, whereas the 2D image-plane
    angle degenerates under that foreshortening (which is exactly how a side-view
    camera sees the throwing arm through cocking and release)."""
    ba = (ax - bx, ay - by, az - bz)
    bc = (cx - bx, cy - by, cz - bz)
    dot = ba[0]*bc[0] + ba[1]*bc[1] + ba[2]*bc[2]
    mag = ((ba[0]**2+ba[1]**2+ba[2]**2)**0.5
           * (bc[0]**2+bc[1]**2+bc[2]**2)**0.5 + 1e-9)
    return float(np.degrees(np.arccos(np.clip(dot / mag, -1.0, 1.0))))


def smooth(arr, window=7, poly=3):
    if len(arr) < window + 2:
        return list(arr)
    w = window | 1           # ensure odd
    w = min(w, len(arr) - (1 - len(arr) % 2))
    if w < poly + 1:
        return list(arr)
    return savgol_filter(arr, w, poly).tolist()


def lowpass(series, fps, cutoff_hz, order=2):
    """Zero-lag (forward-backward) Butterworth low-pass filter.

    The cutoff is a real frequency in Hz, so the filter rejects the same
    physical noise band no matter the frame rate — unlike a fixed-sample-count
    smoother, whose effective cutoff scales with fps and leaves high-fps footage
    badly under-filtered (at 240fps the old gradient-then-savgol velocity read a
    hip rotation ~2x faster than the same motion shot at 30fps, purely from
    differentiation noise). filtfilt is zero-phase, so peak *timing* is
    preserved — important for the kinetic-chain sequencing metric.

    Falls back to the raw series when there are too few samples to filter
    stably; clamps the cutoff just under Nyquist so low-fps clips still filter.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 8:
        return x.tolist()
    # Cap the normalized cutoff at half-Nyquist. Frequencies above that are
    # mostly differentiation noise, and the cap matters at low frame rates: at
    # 30fps a nominal 14Hz arm cutoff would sit at 0.93*Nyquist and filter
    # almost nothing, leaving the elbow-extension velocity dominated by jitter.
    wn = min(cutoff_hz / (fps / 2.0), 0.5)
    b, a = butter(order, wn)
    padlen = 3 * (max(len(a), len(b)) - 1)
    if n <= padlen:
        return x.tolist()
    return filtfilt(b, a, x).tolist()


def best_peak(values, valid, lo, hi, n):
    """argmax of values over [lo, hi], restricted to trustworthy frames.

    A frame only qualifies if it *and both immediate neighbours* are valid —
    this rejects the one-frame velocity spikes thrown off when tracking pops
    back in mid-delivery (a foreshortened throwing arm snapping into view
    previously produced an impossible ~2500 deg/s "peak"). The result is
    flagged low-confidence when the search window was mostly untracked, so a
    delivery whose actual acceleration phase couldn't be seen reports an
    honest "uncertain" rather than whatever lone artifact survived."""
    lo, hi = max(0, lo), min(n - 1, hi)
    win = list(range(lo, hi + 1))
    dens = (sum(1 for i in win if valid[i]) / len(win)) if win else 0.0
    def neigh_ok(i):
        return valid[i] and (i == 0 or valid[i-1]) and (i == n-1 or valid[i+1])
    cand = [i for i in win if neigh_ok(i)] or [i for i in win if valid[i]]
    if cand:
        return max(cand, key=lambda i: values[i]), bool(dens < PEAK_MIN_VALID_FRAC)
    cand = [i for i in range(n) if valid[i]]
    if cand:
        return max(cand, key=lambda i: values[i]), True
    return int(np.argmax(values)), True


def derivative(series, fps):
    """Central-difference time derivative (units/sec). Apply to an already
    low-passed series (see lowpass): differentiating raw landmark-derived
    angles amplifies high-frequency jitter, and the amplification grows with
    fps (noise/Δt), which is what skewed the old peak-speed numbers."""
    return np.gradient(np.asarray(series, dtype=float), 1.0 / fps).tolist()


def mask_confidence(mask_small, x, y):
    """Sample a downsampled segmentation mask (uint8, 0-255) at normalized
    image coordinates (x, y), returning person-probability in [0, 1].
    Returns 1.0 (no opinion) if no mask is available."""
    if mask_small is None:
        return 1.0
    h, w = mask_small.shape
    mx = min(w - 1, max(0, int(x * w)))
    my = min(h - 1, max(0, int(y * h)))
    return mask_small[my, mx] / 255.0


def fill_short_gaps(valid, max_gap, fill):
    """Find runs where `valid` is False, bounded by trusted (`valid` True)
    samples on both sides, with length <= max_gap, and call `fill(i, j)` for
    each — the run spans indices [i, j) and is bounded by trusted samples
    i-1 and j. Marks filled indices valid in place. Runs touching either end
    of the series have no second bound and are left for the caller's
    hold-last/first-value behavior."""
    n = len(valid)
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:
            fill(i, j)
            for k in range(i, j):
                valid[k] = True
        i = j
    return valid


def diagnose_landmark_failures(raw, name, min_vis=MIN_VISIBILITY, max_jump=MAX_POS_JUMP, min_mask=MIN_MASK_CONF):
    """Re-walk a landmark's track with the same last-trusted-position logic
    as build_robust_series(), but tally *why* each frame fails validity
    (mutually exclusive, first-applicable cause) instead of building the
    held series. Diagnostic-only — does not affect build_robust_series or
    its callers. Returns a dict of frame counts: missing, low_vis, low_mask,
    pos_jump, ok."""
    n = len(raw)
    counts = {'missing': 0, 'low_vis': 0, 'low_mask': 0, 'pos_jump': 0, 'ok': 0}
    low_vis_values = []
    last = None
    for rec in raw:
        lm = rec['landmarks'].get(name)
        if lm is None:
            counts['missing'] += 1
            continue
        if lm['v'] < min_vis:
            counts['low_vis'] += 1
            low_vis_values.append(lm['v'])
            continue
        if mask_confidence(rec.get('_mask_small'), lm['x'], lm['y']) < min_mask:
            counts['low_mask'] += 1
            continue
        if last is not None and ((lm['x'] - last[0]) ** 2 + (lm['y'] - last[1]) ** 2) ** 0.5 > max_jump:
            counts['pos_jump'] += 1
            continue
        counts['ok'] += 1
        last = (lm['x'], lm['y'], lm['z'])
    counts['low_vis_values'] = low_vis_values
    return counts


def print_low_conf_diagnostics(raw, n, cropped, roi_norm, roi_w, roi_h,
                                lh_ok, rh_ok, ls_ok, rs_ok, te_ok, tw_ok, ts_ok,
                                hip_ang_ok, sho_ang_ok, elbow_ok, low_conf,
                                te_name, tw_name, ts_name):
    """Print an stderr-only breakdown of why frames are flagged low_confidence,
    to help diagnose clips where the annotated skeleton barely renders. Gated
    by PITCHER_DEBUG_LOWCONF so it doesn't affect normal runs. Purely
    informational — no effect on metrics.json/series.json/annotated.mp4."""
    def pct(x):
        return round(100.0 * x / n, 1) if n else 0.0

    print(f'[diag] crop: cropped={cropped} roi_norm={tuple(round(v, 3) for v in roi_norm)} '
          f'frame_pct={round(100.0 * roi_w * roi_h, 1)}', file=sys.stderr)
    print(f'[diag] low_conf: {sum(low_conf)}/{n} ({pct(sum(low_conf))}%)', file=sys.stderr)

    landmarks = [
        ('left_hip', lh_ok), ('right_hip', rh_ok),
        ('left_shoulder', ls_ok), ('right_shoulder', rs_ok),
        (te_name, te_ok), (tw_name, tw_ok), (ts_name, ts_ok),
    ]
    for name, ok in landmarks:
        invalid = sum(1 for v in ok if not v)
        causes = diagnose_landmark_failures(raw, name)
        lvv = causes['low_vis_values']
        lvv_str = (f' (vis range {min(lvv):.3f}-{max(lvv):.3f}, '
                   f'median {sorted(lvv)[len(lvv) // 2]:.3f})') if lvv else ''
        print(f'[diag]   {name}: invalid {invalid}/{n} ({pct(invalid)}%) — '
              f'missing={causes["missing"]} low_vis={causes["low_vis"]}{lvv_str} '
              f'low_mask={causes["low_mask"]} pos_jump={causes["pos_jump"]}', file=sys.stderr)

    hip_bad = sum(1 for v in hip_ang_ok if not v)
    sho_bad = sum(1 for v in sho_ang_ok if not v)
    elbow_bad = sum(1 for v in elbow_ok if not v)
    print(f'[diag]   hip rot-vec invalid: {hip_bad}/{n} ({pct(hip_bad)}%)', file=sys.stderr)
    print(f'[diag]   shoulder rot-vec invalid: {sho_bad}/{n} ({pct(sho_bad)}%)', file=sys.stderr)
    print(f'[diag]   elbow (ts/te/tw) invalid: {elbow_bad}/{n} ({pct(elbow_bad)}%)', file=sys.stderr)

    seven_bad = sum(1 for i in range(n) if not (
        lh_ok[i] and rh_ok[i] and ls_ok[i] and rs_ok[i] and te_ok[i] and tw_ok[i] and ts_ok[i]))
    n_low_conf = sum(low_conf)
    covered = sum(1 for i in range(n) if low_conf[i] and not (
        lh_ok[i] and rh_ok[i] and ls_ok[i] and rs_ok[i] and te_ok[i] and tw_ok[i] and ts_ok[i]))
    covered_pct = round(100.0 * covered / n_low_conf, 1) if n_low_conf else 0.0
    print(f'[diag]   of {n_low_conf} low_conf frames, {covered} ({covered_pct}% of low_conf) '
          f'are explained by the 7-landmark check alone (7-landmark check fails on {seven_bad}/{n} frames total)', file=sys.stderr)


def build_robust_series(raw, name, min_vis=MIN_VISIBILITY, max_jump=MAX_POS_JUMP, min_mask=MIN_MASK_CONF, max_gap=0):
    """Track a landmark's (x, y, z) across frames, holding the last trusted
    position whenever the landmark is missing, low-visibility, falls outside
    the person segmentation mask, or jumps implausibly far in a single frame
    (signs MediaPipe has locked onto the wrong object, e.g. background
    clutter). Gaps of `max_gap` frames or fewer between trusted samples are
    linearly interpolated instead of held flat, since a brief dropout is
    usually MediaPipe jitter and the true position likely moved smoothly
    across it. Returns (xs, ys, zs, valid)."""
    n = len(raw)
    xs = np.zeros(n); ys = np.zeros(n); zs = np.zeros(n)
    valid = np.zeros(n, dtype=bool)
    last = None
    for i, rec in enumerate(raw):
        lm = rec['landmarks'].get(name)
        ok = lm is not None and lm['v'] >= min_vis
        if ok and mask_confidence(rec.get('_mask_small'), lm['x'], lm['y']) < min_mask:
            ok = False
        if ok and last is not None:
            if ((lm['x'] - last[0]) ** 2 + (lm['y'] - last[1]) ** 2) ** 0.5 > max_jump:
                ok = False
        if ok:
            last = (lm['x'], lm['y'], lm['z'])
            valid[i] = True
        elif last is None and lm is not None:
            last = (lm['x'], lm['y'], lm['z'])
        if last is not None:
            xs[i], ys[i], zs[i] = last

    def fill(i, j):
        span = j - i + 1
        for k in range(i, j):
            t = (k - i + 1) / span
            xs[k] = xs[i - 1] + (xs[j] - xs[i - 1]) * t
            ys[k] = ys[i - 1] + (ys[j] - ys[i - 1]) * t
            zs[k] = zs[i - 1] + (zs[j] - zs[i - 1]) * t

    fill_short_gaps(valid, max_gap, fill)
    return xs, ys, zs, valid


def filter_angle_series(angles, valid, max_jump_deg, max_gap=0):
    """Hold the previous angle for any frame already flagged invalid, or
    whose frame-to-frame change exceeds a physically plausible bound.

    The bound scales with the number of consecutive held frames so far:
    a long run of held frames means `out[i-1]` is a stale comparison
    baseline, and the true angle may have drifted proportionally further
    during the gap. Without this scaling, one false rejection freezes
    `out` at a stale value indefinitely, since every later frame then gets
    compared against that same stale value too.

    Gaps of `max_gap` frames or fewer between trusted angles are then
    interpolated along the shortest angular path instead of held flat.

    Returns (angles, still_valid)."""
    out = list(angles)
    ok = list(valid)
    held = 0
    for i in range(1, len(out)):
        if not ok[i]:
            out[i] = out[i - 1]
            held += 1
            continue
        d = (angles[i] - out[i - 1] + 180) % 360 - 180
        if abs(d) > max_jump_deg * (held + 1):
            out[i] = out[i - 1]
            ok[i] = False
            held += 1
        else:
            held = 0

    def fill(i, j):
        span = j - i + 1
        d = (out[j] - out[i - 1] + 180) % 360 - 180
        for k in range(i, j):
            t = (k - i + 1) / span
            out[k] = out[i - 1] + d * t

    fill_short_gaps(ok, max_gap, fill)
    return out, ok


# ── Phase detection ───────────────────────────────────────────────────────────

def detect_phases(lead_ankle_y, throw_elbow_y, wrist_speed, fps):
    n = len(lead_ankle_y)
    if n < 5:
        return {'setup': (0, n - 1)}

    # Smoothing windows below are expressed as a fixed sample count for
    # ~30fps video (the historical tuning), then scaled up for higher frame
    # rates so they cover the same span of *time* — a fixed 9-sample window
    # covers 300ms at 30fps but only 37ms at 240fps, leaving per-frame
    # landmark jitter unfiltered and dominating the velocity signal.
    pos_win = max(9, round(fps * 0.15))
    lead_y  = smooth(lead_ankle_y,  window=pos_win)
    elbow_y = smooth(throw_elbow_y, window=7)
    wspeed  = smooth(wrist_speed,   window=5)

    # Ankle vertical velocity in normalized-coords/sec (not per-frame), so the
    # descending/stopping thresholds below are frame-rate independent. Without
    # this, the same physical foot-strike motion that crosses the threshold at
    # 30fps would fall short at 120fps slow-motion — exactly the frame rate
    # the UI recommends for better arm-speed accuracy.
    vel_win = max(9, round(fps * 0.08))
    ankle_vel = smooth((np.gradient(lead_y) * fps).tolist(), window=vel_win)
    # ~50ms lookback, but never less than 3 frames — a shorter window falls
    # inside the smoothing filter's edge region and reacts to its artifacts.
    lookback = max(3, round(fps * 0.05))
    start = max(lookback, 4)

    # Foot strike: ankle descending then stops
    foot_strike = n // 2
    for i in range(start, n - 4):
        if ankle_vel[i - lookback] > 0.09 and ankle_vel[i] < 0.015:
            foot_strike = i
            break

    # Release: peak wrist speed (search past ~30% of video)
    search_start = max(foot_strike + 2, int(n * 0.3))
    slice_spd = wspeed[search_start:]
    if slice_spd and max(slice_spd) > 0:
        release = search_start + int(np.argmax(slice_spd))
    else:
        release = min(foot_strike + max(3, int(fps * 0.2)), n - 2)

    if release <= foot_strike:
        release = min(foot_strike + max(2, int(fps * 0.15)), n - 2)

    # Max external rotation: min elbow y in cocking window
    ew = elbow_y[foot_strike:release] if release > foot_strike else []
    mer = foot_strike + (int(np.argmin(ew)) if ew else max(1, (release - foot_strike) // 2))
    mer = max(foot_strike, min(mer, release - 1))

    def clamp(s, e):
        return (max(0, min(int(s), n-1)), max(0, min(int(e), n-1)))

    ws  = max(0, foot_strike - int(fps * 0.8))
    ss  = max(0, foot_strike - int(fps * 0.25))
    return {
        'setup':          clamp(0, ws),
        'windup':         clamp(ws, ss),
        'stride':         clamp(ss, foot_strike),
        'foot_strike':    clamp(foot_strike, foot_strike),
        'arm_cocking':    clamp(foot_strike, mer),
        'acceleration':   clamp(mer, release),
        'release':        clamp(release, release),
        'follow_through': clamp(release, n - 1),
    }


def phase_for_frame(phases, i):
    for name in ['release', 'foot_strike', 'acceleration', 'arm_cocking',
                 'stride', 'windup', 'follow_through', 'setup']:
        if name in phases:
            s, e = phases[name]
            if s <= i <= e:
                return name
    return 'setup'


def compute_metrics(motion_fps, n, hip_angs, sho_angs, trunk_angs, elbow_angs,
                     elbow_ok, hip_ang_ok, sho_ang_ok, low_conf,
                     lay, tey, wrist_disp, elbow_height_pct):
    """Compute per-frame motion metrics, phases, and the motion_fps-dependent
    summary fields from the raw (pre-filter) angle/position series.

    Everything here is a function of motion_fps alone — the inputs are
    motion_fps-independent per-frame series extracted from pose tracking.
    Used both for the initial analysis and to rebuild a clip's metrics with a
    corrected motion_fps after cross-camera sync (see analysis.js and the
    --rebuild CLI mode below).
    """
    hip_s   = lowpass(hip_angs,   motion_fps, ROT_CUTOFF_HZ)
    sho_s   = lowpass(sho_angs,   motion_fps, ROT_CUTOFF_HZ)
    trunk_s = lowpass(trunk_angs, motion_fps, ROT_CUTOFF_HZ)
    elbow_s = lowpass(elbow_angs, motion_fps, ARM_CUTOFF_HZ)
    hip_vel   = derivative(hip_s,   motion_fps)
    chest_vel = derivative(sho_s,   motion_fps)
    arm_vel   = derivative(elbow_s, motion_fps)

    frame_metrics = []
    for i in range(n):
        # Hip-shoulder separation is the *short* angular distance between the
        # two lines, wrapped to [-180, 180] — after unwrapping, hip_s/sho_s
        # can differ by more than 180 deg even though the true separation
        # (the X-factor) never exceeds that.
        hss = ((hip_s[i] - sho_s[i] + 180) % 360) - 180
        # Wrap rotation angles back to (-180, 180] for display, now that the
        # unwrapped series has done its job for hss/velocity above.
        hip_disp = ((hip_s[i] + 180) % 360) - 180
        sho_disp = ((sho_s[i] + 180) % 360) - 180

        frame_metrics.append({
            'hip_rotation':         round(hip_disp,       2),
            'shoulder_rotation':    round(sho_disp,       2),
            'hip_shoulder_sep':     round(abs(hss),        2),
            'elbow_angle':          round(elbow_s[i],     2),
            'elbow_height_pct':     elbow_height_pct[i],
            'hip_rotation_speed':   round(hip_vel[i],     2),
            'chest_rotation_speed': round(chest_vel[i],   2),
            'arm_speed':            round(arm_vel[i],     2),
            'trunk_tilt':           round(trunk_s[i],     2),
            'low_confidence':       bool(low_conf[i]),
        })

    wrist_speed = [d * motion_fps for d in wrist_disp]
    phases = detect_phases(lay, tey, wrist_speed, motion_fps)
    frame_phases = [phase_for_frame(phases, i) for i in range(n)]

    # Summary
    arm_speeds   = [abs(m['arm_speed'])            for m in frame_metrics]
    hip_speeds   = [abs(m['hip_rotation_speed'])   for m in frame_metrics]
    chest_speeds = [abs(m['chest_rotation_speed']) for m in frame_metrics]
    hss_vals     = [m['hip_shoulder_sep']          for m in frame_metrics]
    release_f  = phases.get('release',      (0, 0))[0]
    mer_f      = phases.get('acceleration', (0, 0))[0]
    fs_f       = phases.get('foot_strike',  (0, 0))[0]
    ss_f       = phases.get('stride',       (0, 0))[0]

    # Peak-search windows. Every peak the kinetic chain produces fires between
    # the lead foot planting and shortly after release, so restrict the search
    # to that span. This is what stops a foreshortened-arm glitch back in the
    # windup from being reported as the "peak" arm speed (it previously surfaced
    # a physically impossible ~2500 deg/s and even inverted the hip->chest->arm
    # sequencing). Per-metric validity masks keep each peak on frames where
    # *that* measurement was trustworthy, rather than the global low-conf union.
    sec = lambda s: int(round(s * motion_fps))
    rot_lo  = max(0, fs_f - sec(0.10))   # hips can fire just before foot strike
    spd_hi  = min(n - 1, release_f + sec(0.15))
    arm_hi  = min(n - 1, release_f + sec(0.20))
    valid_arm = elbow_ok
    valid_rot = [bool(hip_ang_ok[i] and sho_ang_ok[i]) for i in range(n)]

    arm_peak_f,   arm_peak_lc   = best_peak(arm_speeds,   valid_arm, fs_f,   arm_hi, n)
    hip_peak_f,   hip_peak_lc   = best_peak(hip_speeds,   valid_rot, rot_lo, spd_hi, n)
    chest_peak_f, chest_peak_lc = best_peak(chest_speeds, valid_rot, rot_lo, spd_hi, n)
    hss_peak_f,   hss_peak_lc   = best_peak(hss_vals,     valid_rot, ss_f,   spd_hi, n)

    # Per-phase tracking confidence — surfaces *where* in the delivery the
    # pose tracking was/wasn't trustworthy, since an overall percentage can
    # hide a phase (e.g. acceleration/release) that's entirely low-confidence.
    phase_confidence = {}
    for name, (s, e) in phases.items():
        seg = low_conf[s:e + 1]
        phase_confidence[name] = round(100.0 * (1 - sum(seg) / len(seg)), 1) if seg else 100.0

    summary_fields = {
        'motion_fps': round(motion_fps, 2),
        'phase_confidence': phase_confidence,
        'phases': {k: {'start': int(v[0]), 'end': int(v[1])} for k, v in phases.items()},
        'peak': {
            'max_arm_speed':          round(arm_speeds[arm_peak_f], 1),
            'max_arm_speed_frame':    arm_peak_f,
            'max_arm_speed_low_confidence': arm_peak_lc,
            'max_hip_rotation_speed':   round(hip_speeds[hip_peak_f], 1),
            'max_hip_rotation_speed_frame': hip_peak_f,
            'max_hip_rotation_speed_low_confidence': hip_peak_lc,
            'max_chest_rotation_speed':   round(chest_speeds[chest_peak_f], 1),
            'max_chest_rotation_speed_frame': chest_peak_f,
            'max_chest_rotation_speed_low_confidence': chest_peak_lc,
            'max_hip_shoulder_sep':   round(hss_vals[hss_peak_f], 1),
            'max_hss_frame':          hss_peak_f,
            'max_hss_low_confidence': hss_peak_lc,
            'arm_speed_at_release':   round(arm_speeds[release_f], 1) if release_f < n else 0,
            'arm_speed_at_release_low_confidence': (not valid_arm[release_f]) if release_f < n else False,
            'hss_at_foot_strike':     round(hss_vals[fs_f], 1),
            'hss_at_foot_strike_low_confidence': bool(not valid_rot[fs_f]),
        },
        'key_frames': {
            'foot_strike':  int(fs_f),
            'max_ext_rot':  int(mer_f),
            'release':      int(release_f),
        },
        # Kinetic-chain sequencing: efficient deliveries fire hips, then chest,
        # then arm — each peak progressively later, like links in a whip. The
        # ordering is only meaningful when each peak it depends on was measured
        # on a trustworthy frame, so it carries its own low-confidence flag.
        'sequencing': {
            'hip_peak_frame':    hip_peak_f,
            'chest_peak_frame':  chest_peak_f,
            'arm_peak_frame':    arm_peak_f,
            'hip_to_chest_ms':   round((chest_peak_f - hip_peak_f) / motion_fps * 1000, 1),
            'chest_to_arm_ms':   round((arm_peak_f - chest_peak_f) / motion_fps * 1000, 1),
            'hip_to_arm_ms':     round((arm_peak_f - hip_peak_f) / motion_fps * 1000, 1),
            'proper_order':      hip_peak_f <= chest_peak_f <= arm_peak_f,
            'low_confidence':    bool(hip_peak_lc or chest_peak_lc or arm_peak_lc),
        },
    }
    return frame_metrics, frame_phases, summary_fields


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_skeleton(frame, lm_list, throw_idx, lead_idx, W, H, throw_arm_ok=True):
    """lm_list: list of 33 NormalizedLandmark objects.

    throw_arm_ok: when False, the throwing arm's robust tracking failed this
    frame (fast/blurry release motion) — skip its highlighted connections and
    joint markers rather than risk drawing a drifted line, while still
    drawing the rest of the skeleton (torso, hips, lead arm)."""
    # Scale overlay sizes to the frame resolution (constants below were
    # tuned for a ~960px-wide frame) so the skeleton stays clearly visible
    # on higher-resolution phone footage.
    scale = max(1.0, W / 960.0)

    def pt(idx):
        lm = lm_list[idx]
        return (int(lm.x * W), int(lm.y * H)), getattr(lm, 'visibility', 1.0)

    for a, b, color, thick in SKELETON_CONNECTIONS:
        pa, va = pt(a)
        pb, vb = pt(b)
        if va > 0.3 and vb > 0.3:
            cv2.line(frame, pa, pb, color, max(1, round(thick * scale)))

    # Throwing arm (highlighted)
    if throw_arm_ok:
        for (a, b), color, thick in [
            ((throw_idx['shoulder'], throw_idx['elbow']), (0, 140, 255), 4),
            ((throw_idx['elbow'],    throw_idx['wrist']),  (0, 220, 255), 4),
        ]:
            pa, va = pt(a)
            pb, vb = pt(b)
            if va > 0.3 and vb > 0.3:
                cv2.line(frame, pa, pb, color, max(1, round(thick * scale)))

    # Lead arm
    for (a, b), color, thick in [
        ((lead_idx['shoulder'], lead_idx['elbow']), (100, 100, 200), 2),
        ((lead_idx['elbow'],    lead_idx['wrist']),  (100, 100, 200), 2),
    ]:
        pa, va = pt(a)
        pb, vb = pt(b)
        if va > 0.3 and vb > 0.3:
            cv2.line(frame, pa, pb, color, max(1, round(thick * scale)))

    # Joints
    for name, idx in LM.items():
        p, vis = pt(idx)
        if vis < 0.25:
            continue
        if idx in [throw_idx['shoulder'], throw_idx['elbow'], throw_idx['wrist']]:
            if not throw_arm_ok:
                continue
            cv2.circle(frame, p, round(8 * scale), (0, 220, 255), -1)
            cv2.circle(frame, p, round(8 * scale), (255, 255, 255), max(1, round(scale)))
        elif idx in [11, 12, 23, 24]:
            cv2.circle(frame, p, round(6 * scale), (200, 200, 60), -1)
        else:
            cv2.circle(frame, p, round(4 * scale), (180, 180, 180), -1)


def draw_hud(frame, f_data, phase_name):
    H_f, W_f = frame.shape[:2]
    # Scale the HUD panel/fonts to the frame resolution (constants below
    # were tuned for a ~960px-wide frame) so the readout stays legible when
    # the video is shown small on a phone.
    scale = max(1.0, W_f / 960.0)
    m = f_data['metrics']

    pw, ph = round(292 * scale), round(312 * scale)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (pw, ph), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    pad = round(8 * scale)
    val_x = round(185 * scale)

    color = PHASE_COLORS_BGR.get(phase_name, (150, 150, 150))
    header_h = round(34 * scale)
    cv2.rectangle(frame, (0, 0), (pw, header_h), color, -1)
    cv2.putText(frame, phase_name.replace('_', ' ').upper(), (pad, round(24 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65 * scale, (0, 0, 0), max(1, round(2 * scale)))

    y = round(58 * scale)
    row_h = round(24 * scale)
    def row(lbl, val, unit='deg', good_thresh=None, lo_thresh=None):
        nonlocal y
        cv2.putText(frame, lbl, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46 * scale, (160, 210, 255), max(1, round(scale)))
        vc = (255, 255, 255)
        if good_thresh is not None and val >= good_thresh:
            vc = (80, 255, 80)
        elif lo_thresh is not None and val < lo_thresh:
            vc = (80, 80, 255)
        cv2.putText(frame, f'{val:+.1f}{unit}', (val_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52 * scale, vc, max(1, round(scale)))
        y += row_h

    row('Hip Rotation',       m.get('hip_rotation',      0.0))
    row('Shoulder Rot.',      m.get('shoulder_rotation', 0.0))
    row('Hip-Shoulder Sep',   m.get('hip_shoulder_sep',  0.0), good_thresh=25.0, lo_thresh=10.0)
    row('Elbow Height',       m.get('elbow_height_pct',  0.0), unit='%', good_thresh=5.0)
    row('Elbow Angle',        m.get('elbow_angle',        0.0))
    row('Hip Speed',  abs(m.get('hip_rotation_speed', 0.0)),   unit='deg/s', good_thresh=400.0, lo_thresh=150.0)
    row('Chest Speed', abs(m.get('chest_rotation_speed', 0.0)), unit='deg/s', good_thresh=500.0, lo_thresh=200.0)
    row('Arm Speed', abs(m.get('arm_speed', 0.0)),            unit='deg/s', good_thresh=600.0, lo_thresh=200.0)
    row('Trunk Tilt',         m.get('trunk_tilt',         0.0))

    if m.get('low_confidence'):
        cv2.putText(frame, '~ low-confidence tracking', (pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, (0, 140, 255), max(1, round(scale)))
        y += round(22 * scale)

    cv2.putText(frame, f't={f_data["time"]:.3f}s  #{f_data["frame"]}',
                (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, (100, 100, 100), max(1, round(scale)))


# ── PitchCap kinematic sequence (optional add-on) ───────────────────────────

# COCO-17 indices used by the vendored PitchCap biomech, keyed by the same
# landmark names this module already stores in each frame's `world` dict.
_KINSEQ_NAME_TO_COCO = {
    'nose': 0,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16,
}


def _segment_tracked_fractions(joint_ok, handedness, n):
    """Fraction of frames where every joint a biomech segment depends on was
    validly tracked. Returns ``None`` per segment when masks are unavailable."""
    if not joint_ok or n == 0:
        return {}
    sh = 6 if handedness == 'R' else 5      # throwing shoulder (R/L)
    el = 8 if handedness == 'R' else 7      # throwing elbow (R/L)
    wr = 10 if handedness == 'R' else 9     # throwing wrist (R/L)
    deps = {'pelvis': [11, 12], 'trunk': [11, 12, 5, 6],
            'shoulder': [sh, el], 'elbow': [sh, el, wr]}

    def frac(idxs):
        if any(ci not in joint_ok for ci in idxs):
            return None
        good = sum(1 for i in range(n) if all(joint_ok[ci][i] for ci in idxs))
        return round(good / n, 3)

    return {name: frac(idxs) for name, idxs in deps.items()}


def attach_kinematic_sequence(summary, raw, motion_fps, throw_hand, joint_ok=None):
    """Attach PitchCap's kinematic sequence (pelvis / trunk / shoulder / elbow
    angular velocity over time, each segment's peak magnitude, and the
    proximal→distal peak order + inter-peak lags) as ``summary['pitchcap']``.

    Computed from the MediaPipe *world* landmarks already extracted per frame,
    via the vendored ``pitchcap`` package. This is purely additive — the
    trusted per-frame metrics/peaks/sequencing the UI grades against are left
    exactly as the MediaPipe analyzer produced them; this surfaces PitchCap's
    distinct biomechanical reading alongside them. Best-effort: any failure
    (package not vendored, no world landmarks, degenerate clip) leaves the
    summary unchanged rather than failing the analysis.

    ``joint_ok`` (optional) maps a COCO-17 index -> per-frame validity (the same
    robust per-landmark masks the rest of the analysis uses). Untrusted joints
    are dropped to NaN so PitchCap's filter interpolates short dropouts and a
    segment that's occluded all clip is honestly reported as "not reconstructed"
    rather than producing a peak from mis-tracked world coordinates."""
    try:
        pkg_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pitchcap')
        if pkg_root not in sys.path:
            sys.path.insert(0, pkg_root)
        from pitchcap import constants as PC  # noqa: F401
        from pitchcap.filtering import DEFAULT_CUTOFFS
        from pitchcap.biomech import compute_kinematic_sequence
    except Exception:
        return

    n = len(raw)
    kp3d = np.full((n, 17, 3), np.nan)
    for i, rec in enumerate(raw):
        for name, w in rec.get('world', {}).items():
            ci = _KINSEQ_NAME_TO_COCO.get(name)
            if ci is None:
                continue
            if joint_ok is not None and ci in joint_ok and not joint_ok[ci][i]:
                continue  # untrusted this frame -> leave NaN for the filter to bridge
            kp3d[i, ci] = [w['x'], w['y'], w['z']]
    if not np.isfinite(kp3d).any():
        return

    hand = 'L' if throw_hand == 'left' else 'R'
    try:
        # Per-segment cutoffs (13Hz pelvis/trunk, 18Hz arm) filter each segment's
        # joints at its own band inside compute_kinematic_sequence.
        ks = compute_kinematic_sequence(kp3d, motion_fps, handedness=hand,
                                        cutoffs=DEFAULT_CUTOFFS)
    except Exception:
        return

    # Per-segment tracked fraction: what share of frames had every joint that
    # segment's vector depends on validly tracked. A low fraction (e.g. the
    # throwing arm on a clip where MediaPipe can't see the elbow) means that
    # segment's peak/timing was reconstructed mostly from interpolation and
    # should be read as low-confidence — surfaced so the UI can flag it rather
    # than presenting an interpolated peak as if it were measured.
    tracked = _segment_tracked_fractions(joint_ok, hand, len(raw))

    summary['pitchcap'] = {
        'engine': 'pitchcap',
        'mode': 'monocular',
        'n_cams': 1,
        'reprojection_error_px': None,
        'kinematic_sequence': {
            'fps': ks['fps'],
            'handedness': ks['handedness'],
            'sequence_order': ks['sequence_order'],
            'inter_peak_lags_ms': ks['inter_peak_lags_ms'],
            'segments': {nm: {
                'peak_degps': sg['peak_degps'],
                'peak_time_s': sg['peak_time_s'],
                'series_degps': sg['series_degps'],
                'tracked_frac': tracked.get(nm),
            } for nm, sg in ks['segments'].items()},
            'segment_warnings': ks.get('segment_warnings', []),
        },
        'warnings': ['single camera: monocular 3D from MediaPipe world landmarks '
                     '(angular-velocity magnitudes approximate; curve shape and '
                     'peak timing/order are the reliable readings)'],
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def analyze_video(video_path, output_dir, throw_hand='left', progress_cb=None,
                  kinematic_sequence=False):
    os.makedirs(output_dir, exist_ok=True)
    ensure_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f'Cannot open: {video_path}')

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    throw_idx = THROW[throw_hand]
    lead_idx  = LEAD[throw_hand]

    def emit(stage, cur, tot):
        if progress_cb:
            progress_cb({'stage': stage, 'current': cur, 'total': tot,
                         'pct': round(cur / tot * 100, 1)})

    emit('extracting', 0, max(total, 1))

    # ── Pass 0: locate the pitcher ─────────────────────────────────────────
    # Sample a handful of frames across the whole clip to find a bounding box
    # around the pitcher, then crop the main pass to that box (with margin).
    # This raises the pitcher's effective resolution and excludes background
    # clutter — the single biggest lever for wide/cluttered shots (e.g. a
    # garage) where the pitcher is small in frame.
    #
    # Each PoseLandmarker gets its own BaseOptions instance — sharing one
    # across two create_from_options() calls causes a native crash once the
    # first landmarker is closed (the underlying model resource gets torn
    # down out from under the second instance).
    scan_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_FILE),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    scanner = mp_vision.PoseLandmarker.create_from_options(scan_opts)

    sample_step = max(1, total // SCAN_SAMPLES) if total > 0 else max(1, int(fps))
    bbox = None
    i = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if i % sample_step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            scan_result = scanner.detect(mp_img)
            if scan_result.pose_landmarks:
                pts = [(lm.x, lm.y) for lm in scan_result.pose_landmarks[0]
                       if getattr(lm, 'visibility', 1.0) > 0.3]
                if pts:
                    xs, ys = zip(*pts)
                    fb = (min(xs), min(ys), max(xs), max(ys))
                    bbox = fb if bbox is None else (
                        min(bbox[0], fb[0]), min(bbox[1], fb[1]),
                        max(bbox[2], fb[2]), max(bbox[3], fb[3]),
                    )
        i += 1
    scanner.close()
    cap.release()
    cap = cv2.VideoCapture(video_path)

    roi_norm = (0.0, 0.0, 1.0, 1.0)
    if bbox is not None:
        bx0, by0, bx1, by1 = bbox
        bw, bh = max(bx1 - bx0, 1e-6), max(by1 - by0, 1e-6)
        cand = (
            max(0.0, bx0 - bw * ROI_PAD),
            max(0.0, by0 - bh * ROI_PAD),
            min(1.0, bx1 + bw * ROI_PAD),
            min(1.0, by1 + bh * ROI_PAD),
        )
        if (cand[2] - cand[0]) < ROI_MIN_SPAN or (cand[3] - cand[1]) < ROI_MIN_SPAN:
            roi_norm = cand

    crop_x0, crop_y0 = int(roi_norm[0] * W), int(roi_norm[1] * H)
    crop_x1 = max(crop_x0 + 2, int(roi_norm[2] * W))
    crop_y1 = max(crop_y0 + 2, int(roi_norm[3] * H))

    # Snap the crop to a multiple of CROP_ALIGN px on each axis. MediaPipe's
    # segmentation-mask output is a float32 ImageFrame internally, and reading
    # it back via numpy_view() hits a hard CHECK-failure crash (SIGABRT) when
    # handed an arbitrary, unaligned crop width/height.
    crop_w = max(CROP_ALIGN, ((crop_x1 - crop_x0) // CROP_ALIGN) * CROP_ALIGN)
    crop_h = max(CROP_ALIGN, ((crop_y1 - crop_y0) // CROP_ALIGN) * CROP_ALIGN)
    crop_w = min(crop_w, (W // CROP_ALIGN) * CROP_ALIGN)
    crop_h = min(crop_h, (H // CROP_ALIGN) * CROP_ALIGN)
    crop_x0 = min(crop_x0, W - crop_w)
    crop_y0 = min(crop_y0, H - crop_h)
    crop_x1, crop_y1 = crop_x0 + crop_w, crop_y0 + crop_h

    cropped = (crop_x0, crop_y0, crop_x1, crop_y1) != (0, 0, W, H)
    roi_x, roi_y = crop_x0 / W, crop_y0 / H
    roi_w, roi_h = crop_w / W, crop_h / H

    # Build PoseLandmarker in VIDEO mode for the main pass
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_FILE),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=True,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    raw = []
    frame_num = 0
    throw_side = 'left' if throw_hand == 'left' else 'right'

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        crop = bgr[crop_y0:crop_y1, crop_x0:crop_x1] if cropped else bgr
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  = int(frame_num * 1000 / fps)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        lm_list = None
        if result.pose_landmarks:
            if cropped:
                # Map landmarks from crop-relative back to full-frame
                # normalized coordinates so every downstream calculation
                # (which assumes full-frame coordinates) is unaffected.
                lm_list = [
                    SimpleNamespace(
                        x=roi_x + lm.x * roi_w,
                        y=roi_y + lm.y * roi_h,
                        z=lm.z * roi_w,
                        visibility=getattr(lm, 'visibility', 1.0),
                    )
                    for lm in result.pose_landmarks[0]
                ]
            else:
                lm_list = result.pose_landmarks[0]

        # World landmarks: metric, body-centred 3D coordinates. They're
        # independent of the crop (not in image space), so they need no ROI
        # remapping, and they give a foreshortening-robust 3D elbow angle.
        wl_list = result.pose_world_landmarks[0] if result.pose_world_landmarks else None

        mask_small = None
        if result.segmentation_masks:
            mask = result.segmentation_masks[0].numpy_view()
            if cropped:
                full_mask = np.zeros((H, W), dtype=np.float32)
                mask_resized = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                full_mask[crop_y0:crop_y1, crop_x0:crop_x1] = mask_resized
                mask = full_mask
            mh = max(1, round(MASK_DOWNSCALE_W * mask.shape[0] / mask.shape[1]))
            mask_small = cv2.resize(mask, (MASK_DOWNSCALE_W, mh), interpolation=cv2.INTER_AREA)
            mask_small = (mask_small * 255).astype(np.uint8)

        rec = {
            'frame': frame_num,
            'time':  round(frame_num / fps, 4),
            '_pose_ok':        lm_list is not None,
            '_lm_list':        lm_list,
            '_mask_small':     mask_small,
            'landmarks': {},
            'world':     {},
        }

        if lm_list:
            for name, idx in LM.items():
                lm = lm_list[idx]
                rec['landmarks'][name] = {
                    'x': round(lm.x, 5),
                    'y': round(lm.y, 5),
                    'z': round(lm.z, 5),
                    'v': round(getattr(lm, 'visibility', 1.0), 3),
                }
                if wl_list is not None:
                    wlm = wl_list[idx]
                    rec['world'][name] = {'x': wlm.x, 'y': wlm.y, 'z': wlm.z}

        raw.append(rec)
        frame_num += 1
        if frame_num % 15 == 0:
            emit('extracting', frame_num, max(total, 1))

    cap.release()
    landmarker.close()

    n = len(raw)
    if n == 0:
        raise ValueError('No frames extracted from video')

    # Recompute fps now that the true decoded frame count is known. `fps`
    # becomes the playback rate (drives output video timing, the per-frame
    # `time` field, and UI scrubbing); `motion_fps` is the rate real-world
    # motion advances at and drives every speed/timing calculation below —
    # for slow-motion footage these differ by the slow-mo factor (e.g. 10x).
    fps, motion_fps, motion_fps_confident = detect_frame_rates(video_path, n, fps)
    for rec in raw:
        rec['time'] = round(rec['frame'] / fps, 4)

    ts_name = f'{throw_side}_shoulder'
    te_name = f'{throw_side}_elbow'
    tw_name = f'{throw_side}_wrist'
    lead_side = 'right' if throw_hand == 'left' else 'left'
    la_name = f'{lead_side}_ankle'

    # Express the short-gap interpolation window in real time so it covers
    # the same span of motion regardless of frame rate.
    max_gap = max(0, round(motion_fps * MAX_INTERP_GAP_S))

    # ── Robust landmark tracks ─────────────────────────────────────────────
    # Hold the last trusted position whenever a landmark is missing,
    # low-visibility, or jumps implausibly far in one frame (a sign
    # MediaPipe has locked onto the wrong object, e.g. background clutter).
    # Short gaps (<= max_gap frames) are interpolated rather than held flat.
    lhx, lhy, lhz, lh_ok = build_robust_series(raw, 'left_hip', max_gap=max_gap)
    rhx, rhy, rhz, rh_ok = build_robust_series(raw, 'right_hip', max_gap=max_gap)
    lsx, lsy, lsz, ls_ok = build_robust_series(raw, 'left_shoulder', max_gap=max_gap)
    rsx, rsy, rsz, rs_ok = build_robust_series(raw, 'right_shoulder', max_gap=max_gap)
    tex, tey, tez, te_ok = build_robust_series(raw, te_name, max_gap=max_gap)
    twx, twy, twz, tw_ok = build_robust_series(raw, tw_name, max_gap=max_gap)
    tsx, tsy, tsz, ts_ok = build_robust_series(raw, ts_name, max_gap=max_gap)
    _, lay, _, _ = build_robust_series(raw, la_name, max_gap=max_gap)

    low_conf = [not (lh_ok[i] and rh_ok[i] and ls_ok[i] and rs_ok[i]
                      and te_ok[i] and tw_ok[i] and ts_ok[i])
                for i in range(n)]

    # Per-region tracking used to draw a partial skeleton: the torso (hips +
    # shoulders) is tracked reliably on most clips even when the throwing arm
    # (which moves fastest and foreshortens most) isn't, so a frame being
    # low_confidence overall shouldn't blank out the whole skeleton.
    torso_ok     = [lh_ok[i] and rh_ok[i] and ls_ok[i] and rs_ok[i] for i in range(n)]
    throw_arm_ok = [te_ok[i] and tw_ok[i] and ts_ok[i] for i in range(n)]

    # Wrist displacement (smoothed, robust) — used for phase detection.
    # Stored as a per-frame displacement (not yet scaled to a speed) so it
    # can be rescaled if motion_fps is later corrected (see compute_metrics).
    twx_p = smooth(twx.tolist(), window=7)
    twy_p = smooth(twy.tolist(), window=7)
    wrist_disp = [0.0] + [
        float(np.hypot(twx_p[i] - twx_p[i-1], twy_p[i] - twy_p[i-1]))
        for i in range(1, n)
    ]

    # ── Rotation angles (transverse plane, x-z) ────────────────────────────
    # atan2(dz, dx) is only meaningful when the hip/shoulder line projects to
    # a non-trivial vector; when both dx and dz are near zero, MediaPipe depth
    # noise dominates and the angle swings wildly. Hold the previous angle in
    # that case, then reject any remaining implausible frame-to-frame jumps.
    # This is a per-decoded-frame outlier-rejection threshold against gross
    # mis-tracking (MediaPipe locking onto the wrong object), not a real-time
    # motion bound — it's calibrated against MediaPipe's roughly fixed
    # per-frame coordinate jitter, so it scales with the *playback* frame
    # interval (fps) rather than motion_fps. At motion_fps (e.g. 300 for a
    # 10x slow-mo clip) the equivalent per-frame budget would be a few
    # degrees, smaller than ordinary landmark jitter, and would flag almost
    # every frame as low-confidence.
    max_rot_jump = 1200.0 / fps  # deg/frame ≈ 1200 deg/s cap

    def rotation_series(p1x, p1z, p2x, p2z, p1_ok, p2_ok):
        raw_angs, valid = [], []
        for i in range(n):
            dx, dz = p2x[i] - p1x[i], p2z[i] - p1z[i]
            mag = (dx*dx + dz*dz) ** 0.5
            ok = bool(p1_ok[i] and p2_ok[i] and mag >= MIN_ROT_VEC_MAG)
            if ok:
                ang = float(np.degrees(np.arctan2(dz, dx)))
            else:
                ang = raw_angs[-1] if raw_angs else float(np.degrees(np.arctan2(dz, dx)))
            raw_angs.append(ang)
            valid.append(ok)
        return filter_angle_series(raw_angs, valid, max_rot_jump, max_gap=max_gap)

    hip_angs, hip_ang_ok = rotation_series(lhx, lhz, rhx, rhz, lh_ok, rh_ok)
    sho_angs, sho_ang_ok = rotation_series(lsx, lsz, rsx, rsz, ls_ok, rs_ok)
    for i in range(n):
        if not hip_ang_ok[i] or not sho_ang_ok[i]:
            low_conf[i] = True

    # Unwrap before smoothing so a rotation that crosses the +/-180 deg seam
    # (common across a full delivery) doesn't read as a sudden ~360 deg jump.
    hip_angs = np.degrees(np.unwrap(np.deg2rad(hip_angs))).tolist()
    sho_angs = np.degrees(np.unwrap(np.deg2rad(sho_angs))).tolist()

    # ── Elbow angle (3D world landmarks) ───────────────────────────────────
    # The throwing arm foreshortens badly on a side-view camera through cocking
    # and release (it points toward/away from the lens), which collapses the 2D
    # image-plane elbow angle to noise — the single biggest error source for
    # arm speed on real footage. MediaPipe's metric world landmarks give a 3D
    # elbow angle that stays well-conditioned under that foreshortening; on test
    # footage it roughly halved the frame-to-frame jitter through the delivery.
    # Fall back to the 2D angle if a model build doesn't emit world landmarks.
    has_world = any(rec['world'] for rec in raw)

    def world_track(name, valid):
        """Hold the last visibility-trusted world position across untrusted
        frames, mirroring build_robust_series but for the 3D world coords."""
        xs = np.zeros(n); ys = np.zeros(n); zs = np.zeros(n); last = None
        for i, rec in enumerate(raw):
            w = rec['world'].get(name)
            if w is not None and (valid[i] or last is None):
                last = (w['x'], w['y'], w['z'])
            if last is not None:
                xs[i], ys[i], zs[i] = last
        return xs, ys, zs

    wsx, wsy, wsz = world_track(ts_name, ts_ok)
    wex, wey, wez = world_track(te_name, te_ok)
    wwx, wwy, wwz = world_track(tw_name, tw_ok)

    elbow_angs, elbow_ok, trunk_angs = [], [], []
    for i in range(n):
        ok = bool(ts_ok[i] and te_ok[i] and tw_ok[i])
        if has_world:
            ang = angle_3pt_3d(wsx[i], wsy[i], wsz[i],
                               wex[i], wey[i], wez[i],
                               wwx[i], wwy[i], wwz[i])
        else:
            ang = angle_3pt(tsx[i], tsy[i], tex[i], tey[i], twx[i], twy[i])
        if not ok and elbow_angs:
            ang = elbow_angs[-1]
        elbow_angs.append(ang)
        elbow_ok.append(ok)
        if not ok:
            low_conf[i] = True

        hip_mx, hip_my = (lhx[i] + rhx[i]) / 2, (lhy[i] + rhy[i]) / 2
        sho_mx, sho_my = (lsx[i] + rsx[i]) / 2, (lsy[i] + rsy[i]) / 2
        trunk_angs.append(float(np.degrees(np.arctan2(sho_mx - hip_mx, hip_my - sho_my))))

    if os.environ.get('PITCHER_DEBUG_LOWCONF'):
        print_low_conf_diagnostics(raw, n, cropped, roi_norm, roi_w, roi_h,
                                    lh_ok, rh_ok, ls_ok, rs_ok, te_ok, tw_ok, ts_ok,
                                    hip_ang_ok, sho_ang_ok, elbow_ok, low_conf,
                                    te_name, tw_name, ts_name)

    # Per-frame metrics, phases, and motion_fps-dependent summary fields are
    # computed by compute_metrics() from the raw series above, so they can be
    # recomputed later with a corrected motion_fps (see --rebuild below and
    # analysis.js's cross-camera sync) without re-running pose estimation.
    elbow_height_pct = [round(float(tsy[i] - tey[i]) * 100, 2) for i in range(n)]
    frame_metrics, frame_phases, mf_summary = compute_metrics(
        motion_fps, n, hip_angs, sho_angs, trunk_angs, elbow_angs,
        elbow_ok, hip_ang_ok, sho_ang_ok, low_conf,
        lay.tolist(), tey.tolist(), wrist_disp, elbow_height_pct)

    for i, rec in enumerate(raw):
        rec['metrics'] = frame_metrics[i]
        rec['phase']   = frame_phases[i]

    # Save the raw series alongside metrics.json so a corrected motion_fps
    # can be applied later (via --rebuild) without re-running pose estimation.
    series_path = os.path.join(output_dir, 'series.json')
    with open(series_path, 'w') as fp:
        json.dump({
            'n': n,
            'hip_angs': hip_angs, 'sho_angs': sho_angs,
            'trunk_angs': trunk_angs, 'elbow_angs': elbow_angs,
            'elbow_ok': elbow_ok, 'hip_ang_ok': hip_ang_ok, 'sho_ang_ok': sho_ang_ok,
            'low_conf': low_conf, 'lay': lay.tolist(), 'tey': tey.tolist(),
            'wrist_disp': wrist_disp, 'elbow_height_pct': elbow_height_pct,
        }, fp)

    summary = {
        'fps': round(fps, 2),
        'motion_fps_confident': motion_fps_confident,
        'total_frames': n,
        'duration_s': round(n / fps, 3),
        'throw_hand': throw_hand,
        'frame_width': W,
        'frame_height': H,
        'tracking_quality': round(100.0 * (1 - sum(low_conf) / n), 1),
        'auto_zoom': {
            'applied': cropped,
            'frame_pct': round(100.0 * roi_w * roi_h, 1),
        },
        **mf_summary,
    }

    # ── Pass 2: annotate video ─────────────────────────────────────────────
    emit('annotating', 0, n)
    cap2    = cv2.VideoCapture(video_path)
    raw_out = os.path.join(output_dir, '_raw.mp4')
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    writer  = cv2.VideoWriter(raw_out, fourcc, fps, (W, H))

    i = 0
    while True:
        ret, bgr = cap2.read()
        if not ret or i >= n:
            break
        rec = raw[i]

        # Draw the skeleton whenever the torso (hips + shoulders) is tracked
        # reliably — this is true for almost all frames even on clips where
        # the throwing arm's tracking is unreliable. The throwing arm's own
        # connections/joints are additionally gated on throw_arm_ok inside
        # draw_skeleton: MediaPipe can keep returning *some* pose for that arm
        # after losing it (e.g. locked onto the wrong region during a fast,
        # blurry release), and drawing that raw pose would visibly drift away
        # from the pitcher's actual arm. The HUD's low-confidence badge still
        # communicates "tracking lost" for the overall frame.
        if rec['_pose_ok'] and rec['_lm_list'] and torso_ok[i]:
            draw_skeleton(bgr, rec['_lm_list'], throw_idx, lead_idx, W, H,
                          throw_arm_ok=throw_arm_ok[i])
            if throw_arm_ok[i] and te_name in rec['landmarks']:
                hud_scale = max(1.0, W / 960.0)
                lme = rec['landmarks'][te_name]
                off  = round(12 * hud_scale)
                bound = round(55 * hud_scale)
                ex, ey = int(lme['x']*W) + off, int(lme['y']*H) - off
                ex = max(5, min(ex, W - bound)); ey = max(15, min(ey, H - 5))
                cv2.putText(bgr, f'{rec["metrics"]["elbow_angle"]:.0f}deg',
                            (ex, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * hud_scale, (255, 255, 0), max(1, round(2 * hud_scale)))
        draw_hud(bgr, rec, rec['phase'])
        writer.write(bgr)
        i += 1
        if i % 15 == 0:
            emit('annotating', i, n)

    cap2.release()
    writer.release()

    # Re-encode to H.264 for browser playback. The raw OpenCV output uses the
    # 'mp4v' (MPEG-4 Part 2) codec, which most browsers can't play — without
    # a successful re-encode here, the annotated video silently fails to load.
    ann_path = os.path.join(output_dir, 'annotated.mp4')
    ffmpeg_bin = find_ffmpeg()
    encoded = False
    if not ffmpeg_bin:
        print('[warn] No ffmpeg binary found (system PATH or imageio-ffmpeg) — '
              'annotated video will stay mp4v/MPEG-4 and may not play in the browser. '
              'Fix: pip install imageio-ffmpeg', file=sys.stderr)
    else:
        import subprocess
        cmd = [ffmpeg_bin, '-i', raw_out, '-vcodec', 'libx264', '-crf', '18',
               '-preset', 'medium', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', '-y', ann_path]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, timeout=300)
            encoded = (proc.returncode == 0 and os.path.exists(ann_path)
                       and os.path.getsize(ann_path) > 1024)
            if not encoded:
                # Surface the real reason (last few lines of ffmpeg stderr) so a
                # failed H.264 re-encode isn't silent — this is what leaves the
                # browser with an unplayable mpeg4 file.
                tail = proc.stderr.decode('utf-8', 'replace').strip().splitlines()[-8:]
                print('[warn] ffmpeg H.264 re-encode failed (rc=%d) — annotated '
                      'video will stay mp4v and may not play in the browser:\n  %s'
                      % (proc.returncode, '\n  '.join(tail)), file=sys.stderr)
        except Exception as e:
            print('[warn] ffmpeg H.264 re-encode raised %r — annotated video '
                  'will stay mp4v and may not play in the browser.' % e,
                  file=sys.stderr)
    if encoded:
        os.remove(raw_out)
    else:
        os.rename(raw_out, ann_path)

    # Serialize (strip cv2 / mediapipe objects)
    json_frames = [{
        'frame':     r['frame'],
        'time':      r['time'],
        'phase':     r['phase'],
        'metrics':   r['metrics'],
        'landmarks': {k: v for k, v in r['landmarks'].items() if isinstance(v, dict)},
    } for r in raw]

    if kinematic_sequence:
        # Feed PitchCap's biomech the same robust per-landmark validity the rest
        # of the analysis uses, keyed by COCO-17 index (hips, both shoulders,
        # throwing elbow/wrist — the joints its segment vectors need).
        te_coco = 7 if throw_hand == 'left' else 8   # L_ELBOW / R_ELBOW
        tw_coco = 9 if throw_hand == 'left' else 10  # L_WRIST / R_WRIST
        joint_ok = {
            11: lh_ok, 12: rh_ok,          # L_HIP, R_HIP
            5: ls_ok, 6: rs_ok,            # L_SHOULDER, R_SHOULDER
            te_coco: te_ok, tw_coco: tw_ok,
        }
        attach_kinematic_sequence(summary, raw, motion_fps, throw_hand, joint_ok)

    output = {'summary': summary, 'frames': json_frames}
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as fp:
        json.dump(json_sanitize(output), fp)

    emit('done', n, n)
    return {'annotated_video': ann_path, 'metrics': metrics_path}


# ── Rebuild (motion_fps correction) ─────────────────────────────────────────

def rebuild_metrics(series_path, metrics_path, motion_fps, out_path):
    """Recompute per-frame metrics, phases, and summary fields for a clip
    whose motion_fps was found to be wrong (see analysis.js's cross-camera
    correlation). Reuses the motion_fps-independent raw series saved by
    analyze_video() instead of re-running pose estimation.
    """
    with open(series_path) as fp:
        series = json.load(fp)
    with open(metrics_path) as fp:
        existing = json.load(fp)

    n = series['n']
    frame_metrics, frame_phases, mf_summary = compute_metrics(
        motion_fps, n,
        series['hip_angs'], series['sho_angs'], series['trunk_angs'], series['elbow_angs'],
        series['elbow_ok'], series['hip_ang_ok'], series['sho_ang_ok'], series['low_conf'],
        series['lay'], series['tey'], series['wrist_disp'], series['elbow_height_pct'])

    frames = existing['frames']
    for i, fr in enumerate(frames):
        fr['metrics'] = frame_metrics[i]
        fr['phase']   = frame_phases[i]

    summary = dict(existing['summary'])
    motion_fps_detected = summary['motion_fps']
    summary.update(mf_summary)
    summary['motion_fps_detected']  = motion_fps_detected
    summary['motion_fps_corrected'] = True

    # The PitchCap kinematic sequence was computed at the originally-detected
    # motion_fps and can't be recomputed here (series.json holds no 3D), but it
    # reparametrizes exactly with frame rate: angular velocity scales with fps
    # and time scales with 1/fps. Rescale so its peaks/times/axis stay consistent
    # with the corrected metrics rather than reading off by the slow-mo factor.
    pc = summary.get('pitchcap')
    if pc and motion_fps_detected and motion_fps_detected > 0:
        r = motion_fps / motion_fps_detected
        ks = pc.get('kinematic_sequence', {})
        ks['fps'] = motion_fps
        for seg in ks.get('segments', {}).values():
            if seg.get('peak_degps') is not None:
                seg['peak_degps'] = round(seg['peak_degps'] * r, 1)
            if seg.get('peak_time_s') is not None:
                seg['peak_time_s'] = round(seg['peak_time_s'] / r, 4)
            seg['series_degps'] = [None if v is None else v * r for v in seg.get('series_degps', [])]
        lags = ks.get('inter_peak_lags_ms', {})
        for k in list(lags):
            if lags[k] is not None:
                lags[k] = round(lags[k] / r, 1)

    with open(out_path, 'w') as fp:
        json.dump(json_sanitize({'summary': summary, 'frames': frames}), fp)

    return {'motion_fps': summary['motion_fps']}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Pitcher side-view biomechanics analyzer')
    parser.add_argument('video', nargs='?', help='Input video path')
    parser.add_argument('--output-dir', default='./pitcher_analysis')
    parser.add_argument('--throw-hand', choices=['left', 'right'], default='left')
    parser.add_argument('--progress',   action='store_true')
    parser.add_argument('--kinematic-sequence', action='store_true',
                         help='Also compute PitchCap\'s pelvis/trunk/arm kinematic '
                              'sequence from the world landmarks and attach it under '
                              'summary.pitchcap')
    parser.add_argument('--rebuild', metavar='SERIES_JSON',
                         help='Recompute metrics from a saved series.json using '
                              '--motion-fps, instead of running pose estimation')
    parser.add_argument('--metrics',    help='Existing metrics.json (required with --rebuild)')
    parser.add_argument('--motion-fps', type=float, help='Corrected motion_fps (required with --rebuild)')
    parser.add_argument('--output',     help='Output path for rebuilt metrics.json (required with --rebuild)')
    args = parser.parse_args()

    if args.rebuild:
        if not (args.metrics and args.motion_fps and args.output):
            print(json.dumps({'status': 'error', 'error': '--rebuild requires --metrics, --motion-fps, and --output'}),
                  flush=True, file=sys.stderr)
            sys.exit(1)
        try:
            r = rebuild_metrics(args.rebuild, args.metrics, args.motion_fps, args.output)
            print(json.dumps({'status': 'done', **r}), flush=True)
        except Exception as e:
            import traceback
            print(json.dumps({'status': 'error', 'error': str(e)}), flush=True, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
        return

    if not args.video:
        parser.error('the following arguments are required: video')

    def cb(d):
        if args.progress:
            print(json.dumps(d), flush=True)

    try:
        r = analyze_video(args.video, args.output_dir, args.throw_hand, cb,
                          kinematic_sequence=args.kinematic_sequence)
        print(json.dumps({'status': 'done', **r}), flush=True)
    except Exception as e:
        import traceback
        print(json.dumps({'status': 'error', 'error': str(e)}), flush=True, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
