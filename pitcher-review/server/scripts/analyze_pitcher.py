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
import sys
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

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
    def pt(idx):
        lm = lm_list[idx]
        return (int(lm.x * W), int(lm.y * H)), getattr(lm, 'visibility', 1.0)

    for a, b, color, thick in SKELETON_CONNECTIONS:
        pa, va = pt(a)
        pb, vb = pt(b)
        if va > 0.3 and vb > 0.3:
            cv2.line(frame, pa, pb, color, thick)

    # Throwing arm (highlighted)
    for (a, b), color, thick in [
        ((throw_idx['shoulder'], throw_idx['elbow']), (0, 140, 255), 4),
        ((throw_idx['elbow'],    throw_idx['wrist']),  (0, 220, 255), 4),
    ]:
        pa, va = pt(a)
        pb, vb = pt(b)
        if va > 0.3 and vb > 0.3:
            cv2.line(frame, pa, pb, color, thick)

    # Lead arm
    for (a, b), color, thick in [
        ((lead_idx['shoulder'], lead_idx['elbow']), (100, 100, 200), 2),
        ((lead_idx['elbow'],    lead_idx['wrist']),  (100, 100, 200), 2),
    ]:
        pa, va = pt(a)
        pb, vb = pt(b)
        if va > 0.3 and vb > 0.3:
            cv2.line(frame, pa, pb, color, thick)

    # Joints
    for name, idx in LM.items():
        p, vis = pt(idx)
        if vis < 0.25:
            continue
        if idx in [throw_idx['shoulder'], throw_idx['elbow'], throw_idx['wrist']]:
            cv2.circle(frame, p, 8, (0, 220, 255), -1)
            cv2.circle(frame, p, 8, (255, 255, 255), 1)
        elif idx in [11, 12, 23, 24]:
            cv2.circle(frame, p, 6, (200, 200, 60), -1)
        else:
            cv2.circle(frame, p, 4, (180, 180, 180), -1)


def draw_hud(frame, f_data, phase_name):
    H_f, W_f = frame.shape[:2]
    m = f_data['metrics']
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (292, 245), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    color = PHASE_COLORS_BGR.get(phase_name, (150, 150, 150))
    cv2.rectangle(frame, (0, 0), (292, 34), color, -1)
    cv2.putText(frame, phase_name.replace('_', ' ').upper(), (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    y = 58
    def row(lbl, val, unit='°', good_thresh=None, lo_thresh=None):
        nonlocal y
        cv2.putText(frame, lbl, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 210, 255), 1)
        vc = (255, 255, 255)
        if good_thresh is not None and val >= good_thresh:
            vc = (80, 255, 80)
        elif lo_thresh is not None and val < lo_thresh:
            vc = (80, 80, 255)
        cv2.putText(frame, f'{val:+.1f}{unit}', (185, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, vc, 1)
        y += 24

    row('Hip Rotation',       m.get('hip_rotation',      0.0))
    row('Shoulder Rot.',      m.get('shoulder_rotation', 0.0))
    row('Hip-Shoulder Sep',   m.get('hip_shoulder_sep',  0.0), good_thresh=25.0, lo_thresh=10.0)
    row('Elbow Height',       m.get('elbow_height_pct',  0.0), unit='%', good_thresh=5.0)
    row('Elbow Angle',        m.get('elbow_angle',        0.0))
    row('Arm Speed', abs(m.get('arm_speed', 0.0)),            unit='°/s', good_thresh=600.0, lo_thresh=200.0)
    row('Trunk Tilt',         m.get('trunk_tilt',         0.0))

    cv2.putText(frame, f't={f_data["time"]:.3f}s  #{f_data["frame"]}',
                (8, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def analyze_video(video_path, output_dir, throw_hand='left', progress_cb=None):
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

    # Build PoseLandmarker in VIDEO mode
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_FILE)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    raw = []
    frame_num = 0
    throw_side = 'left' if throw_hand == 'left' else 'right'

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  = int(frame_num * 1000 / fps)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        lm_list = result.pose_landmarks[0] if result.pose_landmarks else None

        rec = {
            'frame': frame_num,
            'time':  round(frame_num / fps, 4),
            '_pose_ok':        lm_list is not None,
            '_lm_list':        lm_list,
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

    # Wrist speed (for phase detection)
    tw_name = f'{throw_side}_wrist'
    prev_wx = prev_wy = None
    for rec in raw:
        if tw_name in rec['landmarks']:
            wx = rec['landmarks'][tw_name]['x']
            wy = rec['landmarks'][tw_name]['y']
            if prev_wx is not None:
                rec['_wrist_speed'] = float(np.sqrt((wx-prev_wx)**2 + (wy-prev_wy)**2) * fps)
            prev_wx, prev_wy = wx, wy
        else:
            prev_wx = prev_wy = None

    # Angle time series
    hip_angs, sho_angs, arm_angs, elbow_angs, trunk_angs = [], [], [], [], []

    ts_name = f'{throw_side}_shoulder'
    te_name = f'{throw_side}_elbow'
    tw_name = f'{throw_side}_wrist'

    for rec in raw:
        lms = rec['landmarks']
        if not rec['_pose_ok'] or 'left_hip' not in lms:
            hip_angs.append(hip_angs[-1] if hip_angs else 0.0)
            sho_angs.append(sho_angs[-1] if sho_angs else 0.0)
            arm_angs.append(arm_angs[-1] if arm_angs else 0.0)
            elbow_angs.append(elbow_angs[-1] if elbow_angs else 90.0)
            trunk_angs.append(trunk_angs[-1] if trunk_angs else 0.0)
            continue

        lh, rh = lms['left_hip'],      lms['right_hip']
        ls, rs = lms['left_shoulder'],  lms['right_shoulder']

        # Transverse-plane rotation (using MediaPipe depth z)
        hip_ang = float(np.degrees(np.arctan2(rh['z'] - lh['z'], rh['x'] - lh['x'])))
        sho_ang = float(np.degrees(np.arctan2(rs['z'] - ls['z'], rs['x'] - ls['x'])))
        hip_angs.append(hip_ang)
        sho_angs.append(sho_ang)

        # Throwing arm angle (shoulder → wrist, in 2D frame)
        if ts_name in lms and tw_name in lms:
            ts, tw = lms[ts_name], lms[tw_name]
            arm_ang = float(np.degrees(np.arctan2(tw['y'] - ts['y'], tw['x'] - ts['x'])))
            arm_angs.append(arm_ang)
        else:
            arm_angs.append(arm_angs[-1] if arm_angs else 0.0)

        # Elbow angle
        if ts_name in lms and te_name in lms and tw_name in lms:
            ts, te, tw = lms[ts_name], lms[te_name], lms[tw_name]
            ea = angle_3pt(ts['x'], ts['y'], te['x'], te['y'], tw['x'], tw['y'])
            elbow_angs.append(ea)
        else:
            elbow_angs.append(elbow_angs[-1] if elbow_angs else 90.0)

        # Trunk tilt
        hip_mx = (lh['x'] + rh['x']) / 2
        hip_my = (lh['y'] + rh['y']) / 2
        sho_mx = (ls['x'] + rs['x']) / 2
        sho_my = (ls['y'] + rs['y']) / 2
        trunk_ang = float(np.degrees(np.arctan2(
            sho_mx - hip_mx, hip_my - sho_my)))
        trunk_angs.append(trunk_ang)

    hip_s   = smooth(hip_angs,   window=9)
    sho_s   = smooth(sho_angs,   window=9)
    arm_s   = smooth(arm_angs,   window=5)
    trunk_s = smooth(trunk_angs, window=9)
    arm_vel = angular_velocity(arm_s, fps, window=7)

    # Per-frame metrics
    for i, rec in enumerate(raw):
        lms = rec['landmarks']
        hss = hip_s[i] - sho_s[i]
        elbow_h = 0.0
        if ts_name in lms and te_name in lms:
            elbow_h = round((lms[ts_name]['y'] - lms[te_name]['y']) * 100, 2)

        rec['metrics'] = {
            'hip_rotation':      round(hip_s[i],       2),
            'shoulder_rotation': round(sho_s[i],       2),
            'hip_shoulder_sep':  round(abs(hss),        2),
            'elbow_angle':       round(elbow_angs[i],  2),
            'elbow_height_pct':  elbow_h,
            'arm_speed':         round(arm_vel[i],     2),
            'trunk_tilt':        round(trunk_s[i],     2),
        }

    # Phases
    phases = detect_phases(raw, fps)
    for i, rec in enumerate(raw):
        rec['phase'] = phase_for_frame(phases, i)

    # Summary
    arm_speeds = [abs(r['metrics']['arm_speed'])    for r in raw]
    hss_vals   = [r['metrics']['hip_shoulder_sep']  for r in raw]
    release_f  = phases.get('release',      (0, 0))[0]
    mer_f      = phases.get('acceleration', (0, 0))[0]
    fs_f       = phases.get('foot_strike',  (0, 0))[0]

    summary = {
        'fps': round(fps, 2),
        'total_frames': n,
        'duration_s': round(n / fps, 3),
        'throw_hand': throw_hand,
        'frame_width': W,
        'frame_height': H,
        'phases': {k: {'start': int(v[0]), 'end': int(v[1])} for k, v in phases.items()},
        'peak': {
            'max_arm_speed':        round(max(arm_speeds), 1),
            'max_arm_speed_frame':  int(np.argmax(arm_speeds)),
            'max_hip_shoulder_sep': round(max(hss_vals), 1),
            'max_hss_frame':        int(np.argmax(hss_vals)),
            'arm_speed_at_release': round(arm_speeds[release_f], 1) if release_f < n else 0,
            'hss_at_foot_strike':   round(hss_vals[fs_f], 1),
        },
        'key_frames': {
            'foot_strike':  int(fs_f),
            'max_ext_rot':  int(mer_f),
            'release':      int(release_f),
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
        if rec['_pose_ok'] and rec['_lm_list']:
            draw_skeleton(bgr, rec['_lm_list'], throw_idx, lead_idx, W, H)
            if te_name in rec['landmarks']:
                lme = rec['landmarks'][te_name]
                ex, ey = int(lme['x']*W)+12, int(lme['y']*H)-10
                ex = max(5, min(ex, W-55)); ey = max(15, min(ey, H-5))
                cv2.putText(bgr, f'{rec["metrics"]["elbow_angle"]:.0f}°',
                            (ex, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
        draw_hud(bgr, rec, rec['phase'])
        writer.write(bgr)
        i += 1
        if i % 15 == 0:
            emit('annotating', i, n)

    cap2.release()
    writer.release()

    # Re-encode to H.264 for browser
    ann_path = os.path.join(output_dir, 'annotated.mp4')
    ret_code = os.system(
        f'ffmpeg -i "{raw_out}" -vcodec libx264 -crf 22 -preset fast '
        f'-pix_fmt yuv420p -movflags +faststart -y "{ann_path}" 2>/dev/null'
    )
    if ret_code == 0 and os.path.exists(ann_path) and os.path.getsize(ann_path) > 1024:
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
    args = parser.parse_args()

    def cb(d):
        if args.progress:
            print(json.dumps(d), flush=True)

    try:
        r = analyze_video(args.video, args.output_dir, args.throw_hand, cb)
        print(json.dumps({'status': 'done', **r}), flush=True)
    except Exception as e:
        import traceback
        print(json.dumps({'status': 'error', 'error': str(e)}), flush=True, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
