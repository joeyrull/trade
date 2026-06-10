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
import sys
import urllib.request
from types import SimpleNamespace
import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


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
MIN_VISIBILITY   = 0.5   # landmark visibility below this is not trusted
MAX_POS_JUMP     = 0.15  # max plausible frame-to-frame landmark move (normalized coords)
MIN_ROT_VEC_MAG  = 0.035 # min |hip/shoulder line vector| (x-z plane) to trust its angle
MIN_LIMB_VEC_MAG = 0.04  # min |upper-arm or forearm| (x-y plane) to trust the elbow angle
MIN_MASK_CONF    = 0.4   # min person-segmentation confidence to trust a landmark there

# ── Background blur ──────────────────────────────────────────────────────────
MASK_DOWNSCALE_W = 320  # width (px) to downsample segmentation masks to before storing
BG_BLUR_KSIZE    = 45   # Gaussian blur kernel size (odd) applied to background pixels

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


def smooth(arr, window=7, poly=3):
    if len(arr) < window + 2:
        return list(arr)
    w = window | 1           # ensure odd
    w = min(w, len(arr) - (1 - len(arr) % 2))
    if w < poly + 1:
        return list(arr)
    return savgol_filter(arr, w, poly).tolist()


def angular_velocity(angles_deg, fps, window=7):
    a = np.unwrap(np.deg2rad(angles_deg))
    vel = np.gradient(np.rad2deg(a), 1.0 / fps)
    if len(vel) > window + 2:
        vel = savgol_filter(vel, window | 1, 3)
    return vel.tolist()


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


def build_robust_series(raw, name, min_vis=MIN_VISIBILITY, max_jump=MAX_POS_JUMP, min_mask=MIN_MASK_CONF):
    """Track a landmark's (x, y, z) across frames, holding the last trusted
    position whenever the landmark is missing, low-visibility, falls outside
    the person segmentation mask, or jumps implausibly far in a single frame
    (signs MediaPipe has locked onto the wrong object, e.g. background
    clutter). Returns (xs, ys, zs, valid)."""
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
    return xs, ys, zs, valid


def filter_angle_series(angles, valid, max_jump_deg):
    """Hold the previous angle for any frame already flagged invalid, or
    whose frame-to-frame change exceeds a physically plausible bound.
    Returns (angles, still_valid)."""
    out = list(angles)
    ok = list(valid)
    for i in range(1, len(out)):
        if not ok[i]:
            out[i] = out[i - 1]
            continue
        d = (out[i] - out[i - 1] + 180) % 360 - 180
        if abs(d) > max_jump_deg:
            out[i] = out[i - 1]
            ok[i] = False
    return out, ok


# ── Phase detection ───────────────────────────────────────────────────────────

def detect_phases(frames, fps):
    n = len(frames)
    if n < 5:
        return {'setup': (0, n - 1)}

    lead_y  = smooth([f['_lead_ankle_y']  for f in frames], window=9)
    elbow_y = smooth([f['_throw_elbow_y'] for f in frames], window=7)
    wspeed  = smooth([f['_wrist_speed']   for f in frames], window=5)

    ankle_vel = np.gradient(lead_y)

    # Foot strike: ankle descending then stops
    foot_strike = n // 2
    for i in range(4, n - 4):
        if ankle_vel[i-3] > 0.003 and ankle_vel[i] < 0.0005:
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


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_skeleton(frame, lm_list, throw_idx, lead_idx, W, H):
    """lm_list: list of 33 NormalizedLandmark objects."""
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


# ── Main pipeline ─────────────────────────────────────────────────────────────

def analyze_video(video_path, output_dir, throw_hand='left', progress_cb=None, blur_background=True):
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
            '_lead_ankle_y':   0.85,
            '_throw_elbow_y':  0.4,
            '_wrist_speed':    0.0,
            'landmarks': {},
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
            rec['_lead_ankle_y']  = lm_list[lead_idx['ankle']].y
            rec['_throw_elbow_y'] = lm_list[throw_idx['elbow']].y
        elif raw:
            rec['_lead_ankle_y']  = raw[-1]['_lead_ankle_y']
            rec['_throw_elbow_y'] = raw[-1]['_throw_elbow_y']

        raw.append(rec)
        frame_num += 1
        if frame_num % 15 == 0:
            emit('extracting', frame_num, max(total, 1))

    cap.release()
    landmarker.close()

    n = len(raw)
    if n == 0:
        raise ValueError('No frames extracted from video')

    ts_name = f'{throw_side}_shoulder'
    te_name = f'{throw_side}_elbow'
    tw_name = f'{throw_side}_wrist'

    # ── Robust landmark tracks ─────────────────────────────────────────────
    # Hold the last trusted position whenever a landmark is missing,
    # low-visibility, or jumps implausibly far in one frame (a sign
    # MediaPipe has locked onto the wrong object, e.g. background clutter).
    lhx, lhy, lhz, lh_ok = build_robust_series(raw, 'left_hip')
    rhx, rhy, rhz, rh_ok = build_robust_series(raw, 'right_hip')
    lsx, lsy, lsz, ls_ok = build_robust_series(raw, 'left_shoulder')
    rsx, rsy, rsz, rs_ok = build_robust_series(raw, 'right_shoulder')
    tex, tey, tez, te_ok = build_robust_series(raw, te_name)
    twx, twy, twz, tw_ok = build_robust_series(raw, tw_name)
    tsx, tsy, tsz, ts_ok = build_robust_series(raw, ts_name)

    low_conf = [not (lh_ok[i] and rh_ok[i] and ls_ok[i] and rs_ok[i]
                      and te_ok[i] and tw_ok[i] and ts_ok[i])
                for i in range(n)]

    # Wrist speed (smoothed, robust) — used for phase detection
    twx_p = smooth(twx.tolist(), window=7)
    twy_p = smooth(twy.tolist(), window=7)
    for i, rec in enumerate(raw):
        if i == 0:
            rec['_wrist_speed'] = 0.0
        else:
            rec['_wrist_speed'] = float(np.hypot(twx_p[i] - twx_p[i-1], twy_p[i] - twy_p[i-1]) * fps)

    # ── Rotation angles (transverse plane, x-z) ────────────────────────────
    # atan2(dz, dx) is only meaningful when the hip/shoulder line projects to
    # a non-trivial vector; when both dx and dz are near zero, MediaPipe depth
    # noise dominates and the angle swings wildly. Hold the previous angle in
    # that case, then reject any remaining implausible frame-to-frame jumps.
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
        return filter_angle_series(raw_angs, valid, max_rot_jump)

    hip_angs, hip_ang_ok = rotation_series(lhx, lhz, rhx, rhz, lh_ok, rh_ok)
    sho_angs, sho_ang_ok = rotation_series(lsx, lsz, rsx, rsz, ls_ok, rs_ok)
    for i in range(n):
        if not hip_ang_ok[i] or not sho_ang_ok[i]:
            low_conf[i] = True

    # Unwrap before smoothing so a rotation that crosses the +/-180 deg seam
    # (common across a full delivery) doesn't read as a sudden ~360 deg jump.
    hip_angs = np.degrees(np.unwrap(np.deg2rad(hip_angs))).tolist()
    sho_angs = np.degrees(np.unwrap(np.deg2rad(sho_angs))).tolist()

    # ── Elbow angle, trunk tilt (image-plane geometry, x-y) ────────────────
    # The 3-point elbow angle itself is well-conditioned, but its *direction*
    # becomes noise if either the upper-arm or forearm projects to a near-zero
    # 2D vector (forearm pointing straight at/away from the camera). Hold the
    # previous angle in that case.
    elbow_angs, elbow_ok, trunk_angs = [], [], []
    for i in range(n):
        ux, uy = tex[i] - tsx[i], tey[i] - tsy[i]
        fx, fy = twx[i] - tex[i], twy[i] - tey[i]
        umag, fmag = (ux*ux+uy*uy) ** 0.5, (fx*fx+fy*fy) ** 0.5
        ok = bool(ts_ok[i] and te_ok[i] and tw_ok[i]
                  and umag >= MIN_LIMB_VEC_MAG and fmag >= MIN_LIMB_VEC_MAG)
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

    hip_s   = smooth(hip_angs,   window=9)
    sho_s   = smooth(sho_angs,   window=9)
    elbow_s = smooth(elbow_angs, window=7)
    trunk_s = smooth(trunk_angs, window=9)
    hip_vel   = angular_velocity(hip_s, fps, window=9)
    chest_vel = angular_velocity(sho_s, fps, window=9)
    # Arm speed = elbow extension/flexion rate. Unlike the angle of the
    # shoulder->wrist vector, this doesn't blow up under foreshortening: the
    # 3-point angle is bounded to [0, 180] deg, so its derivative can't sweep
    # through a near-360 deg discontinuity the way a 2D vector angle can.
    arm_vel = angular_velocity(elbow_s, fps, window=7)

    # Per-frame metrics
    for i, rec in enumerate(raw):
        # Hip-shoulder separation is the *short* angular distance between the
        # two lines, wrapped to [-180, 180] — after unwrapping, hip_s/sho_s
        # can differ by more than 180 deg even though the true separation
        # (the X-factor) never exceeds that.
        hss = ((hip_s[i] - sho_s[i] + 180) % 360) - 180
        elbow_h = round(float(tsy[i] - tey[i]) * 100, 2)
        # Wrap rotation angles back to (-180, 180] for display, now that the
        # unwrapped series has done its job for hss/velocity above.
        hip_disp = ((hip_s[i] + 180) % 360) - 180
        sho_disp = ((sho_s[i] + 180) % 360) - 180

        rec['metrics'] = {
            'hip_rotation':         round(hip_disp,       2),
            'shoulder_rotation':    round(sho_disp,       2),
            'hip_shoulder_sep':     round(abs(hss),        2),
            'elbow_angle':          round(elbow_s[i],     2),
            'elbow_height_pct':     elbow_h,
            'hip_rotation_speed':   round(hip_vel[i],     2),
            'chest_rotation_speed': round(chest_vel[i],   2),
            'arm_speed':            round(arm_vel[i],     2),
            'trunk_tilt':           round(trunk_s[i],     2),
            'low_confidence':       bool(low_conf[i]),
        }

    # Phases
    phases = detect_phases(raw, fps)
    for i, rec in enumerate(raw):
        rec['phase'] = phase_for_frame(phases, i)

    # Summary
    arm_speeds   = [abs(r['metrics']['arm_speed'])           for r in raw]
    hip_speeds   = [abs(r['metrics']['hip_rotation_speed'])  for r in raw]
    chest_speeds = [abs(r['metrics']['chest_rotation_speed']) for r in raw]
    hss_vals     = [r['metrics']['hip_shoulder_sep']         for r in raw]
    release_f  = phases.get('release',      (0, 0))[0]
    mer_f      = phases.get('acceleration', (0, 0))[0]
    fs_f       = phases.get('foot_strike',  (0, 0))[0]

    def best_peak(values):
        """argmax restricted to high-confidence frames when any exist;
        otherwise fall back to all frames and flag the result."""
        hi = [i for i in range(n) if not low_conf[i]]
        if hi:
            idx = max(hi, key=lambda i: values[i])
            return idx, False
        return int(np.argmax(values)), True

    arm_peak_f,   arm_peak_lc   = best_peak(arm_speeds)
    hip_peak_f,   hip_peak_lc   = best_peak(hip_speeds)
    chest_peak_f, chest_peak_lc = best_peak(chest_speeds)
    hss_peak_f,   hss_peak_lc   = best_peak(hss_vals)

    # Per-phase tracking confidence — surfaces *where* in the delivery the
    # pose tracking was/wasn't trustworthy, since an overall percentage can
    # hide a phase (e.g. acceleration/release) that's entirely low-confidence.
    phase_confidence = {}
    for name, (s, e) in phases.items():
        seg = low_conf[s:e + 1]
        phase_confidence[name] = round(100.0 * (1 - sum(seg) / len(seg)), 1) if seg else 100.0

    summary = {
        'fps': round(fps, 2),
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
            'arm_speed_at_release_low_confidence': bool(low_conf[release_f]) if release_f < n else False,
            'hss_at_foot_strike':     round(hss_vals[fs_f], 1),
            'hss_at_foot_strike_low_confidence': bool(low_conf[fs_f]),
        },
        'key_frames': {
            'foot_strike':  int(fs_f),
            'max_ext_rot':  int(mer_f),
            'release':      int(release_f),
        },
        # Kinetic-chain sequencing: efficient deliveries fire hips, then chest,
        # then arm — each peak progressively later, like links in a whip.
        'sequencing': {
            'hip_peak_frame':    hip_peak_f,
            'chest_peak_frame':  chest_peak_f,
            'arm_peak_frame':    arm_peak_f,
            'hip_to_chest_ms':   round((chest_peak_f - hip_peak_f) / fps * 1000, 1),
            'chest_to_arm_ms':   round((arm_peak_f - chest_peak_f) / fps * 1000, 1),
            'hip_to_arm_ms':     round((arm_peak_f - hip_peak_f) / fps * 1000, 1),
            'proper_order':      hip_peak_f <= chest_peak_f <= arm_peak_f,
        },
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

        if blur_background and rec['_mask_small'] is not None:
            mask_full = cv2.resize(rec['_mask_small'], (W, H), interpolation=cv2.INTER_LINEAR)
            alpha = (mask_full.astype(np.float32) / 255.0)[..., None]
            blurred = cv2.GaussianBlur(bgr, (BG_BLUR_KSIZE, BG_BLUR_KSIZE), 0)
            bgr = (bgr.astype(np.float32) * alpha + blurred.astype(np.float32) * (1 - alpha)).astype(np.uint8)

        # Only draw the skeleton/elbow-angle overlay when the robust tracker
        # trusts this frame's landmarks. MediaPipe can keep returning *some*
        # pose for a frame even after it has lost the pitcher (e.g. locked
        # onto the wrong region during a fast, blurry release), and drawing
        # that raw pose produces a skeleton that visibly drifts away from the
        # pitcher's actual position. The HUD's low-confidence badge already
        # communicates "tracking lost" for these frames.
        if rec['_pose_ok'] and rec['_lm_list'] and not rec['metrics']['low_confidence']:
            draw_skeleton(bgr, rec['_lm_list'], throw_idx, lead_idx, W, H)
            if te_name in rec['landmarks']:
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
    if ffmpeg_bin:
        ret_code = os.system(
            f'"{ffmpeg_bin}" -i "{raw_out}" -vcodec libx264 -crf 18 -preset medium '
            f'-pix_fmt yuv420p -movflags +faststart -y "{ann_path}" 2>/dev/null'
        )
        encoded = ret_code == 0 and os.path.exists(ann_path) and os.path.getsize(ann_path) > 1024
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

    output = {'summary': summary, 'frames': json_frames}
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as fp:
        json.dump(output, fp)

    emit('done', n, n)
    return {'annotated_video': ann_path, 'metrics': metrics_path}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Pitcher side-view biomechanics analyzer')
    parser.add_argument('video',        help='Input video path')
    parser.add_argument('--output-dir', default='./pitcher_analysis')
    parser.add_argument('--throw-hand', choices=['left', 'right'], default='left')
    parser.add_argument('--progress',   action='store_true')
    parser.add_argument('--no-blur-background', action='store_false', dest='blur_background',
                         default=True, help='Disable automatic background blur in the annotated video')
    args = parser.parse_args()

    def cb(d):
        if args.progress:
            print(json.dumps(d), flush=True)

    try:
        r = analyze_video(args.video, args.output_dir, args.throw_hand, cb, args.blur_background)
        print(json.dumps({'status': 'done', **r}), flush=True)
    except Exception as e:
        import traceback
        print(json.dumps({'status': 'error', 'error': str(e)}), flush=True, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
