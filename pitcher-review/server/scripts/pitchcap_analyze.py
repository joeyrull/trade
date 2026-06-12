#!/usr/bin/env python3
"""PitchCap engine — pitcher-review's stronger-motion-capture analysis backend.

CLI- and contract-compatible with ``analyze_pitcher.py`` (the Node server spawns
it as a drop-in engine):

    python3 pitchcap_analyze.py <video> [<video2> ...] --output-dir <dir> \
        --throw-hand <hand> --progress

It writes the same ``annotated.mp4`` / ``metrics.json`` / ``series.json`` the
frontend consumes, emits the same stdout progress / ``{"status":"done"}``
contract, and attaches a ``summary['pitchcap']`` block carrying PitchCap's
kinematic sequence (pelvis/trunk/arm angular-velocity curves, peak timing/order,
reconstruction mode + reprojection error).

Two paths, by camera count:

  * **1 camera** — PitchCap can't beat MediaPipe at *monocular* pose (it uses
    MediaPipe for that), so the trusted per-frame metrics/peaks/sequencing are
    produced by ``analyze_pitcher.analyze_video`` exactly as the default engine
    does; PitchCap's contribution is the kinematic-sequence layer computed from
    those world landmarks. Output is the MediaPipe engine's, plus
    ``summary['pitchcap']``.

  * **2+ cameras** — the genuine upgrade: RTMPose 2D per view + markerless
    multi-view triangulation -> true metric 3D, from which both the web metrics
    *and* the kinematic sequence are derived. Requires rtmlib + onnxruntime;
    falls back to the single-camera path on camera 0 (with a warning) if the
    multi-view stack is unavailable.

The current Node flow analyzes each camera in its own process (then fuses), so
it always takes the 1-camera path; the multi-view path is exercised via the CLI
(pass 2+ clips) and is where true 3D motion capture lives.
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import cv2

# Make the vendored pitchcap package and analyze_pitcher importable. PKG_ROOT is
# the vendored PitchCap repo root (.../server/pitchcap), which contains the inner
# `pitchcap` package — putting it (not .../server) on the path makes
# `import pitchcap` resolve to the package, not the namespace dir.
_HERE = os.path.dirname(os.path.abspath(__file__))           # .../server/scripts
_SERVER = os.path.dirname(_HERE)                             # .../server
_PKG_ROOT = os.path.join(_SERVER, 'pitchcap')               # .../server/pitchcap
for p in (_PKG_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import analyze_pitcher as ap                          # noqa: E402

MODEL_FILE = ap.MODEL_FILE  # reuse the app's already-downloaded pose model


# ── 1-camera path: trusted MediaPipe metrics + PitchCap kinematic sequence ──

def _analyze_monocular(video_path, output_dir, throw_hand, progress_cb):
    return ap.analyze_video(video_path, output_dir, throw_hand, progress_cb,
                            kinematic_sequence=True)


# ── Multi-view helpers (COCO-17 maps, world tracks, rotation) ───────────────

def _maps():
    from pitchcap import constants as C
    coco_name = {
        C.NOSE: 'nose',
        C.L_SHOULDER: 'left_shoulder', C.R_SHOULDER: 'right_shoulder',
        C.L_ELBOW: 'left_elbow', C.R_ELBOW: 'right_elbow',
        C.L_WRIST: 'left_wrist', C.R_WRIST: 'right_wrist',
        C.L_HIP: 'left_hip', C.R_HIP: 'right_hip',
        C.L_KNEE: 'left_knee', C.R_KNEE: 'right_knee',
        C.L_ANKLE: 'left_ankle', C.R_ANKLE: 'right_ankle',
    }
    coco_to_mp = {
        C.NOSE: 0,
        C.L_SHOULDER: 11, C.R_SHOULDER: 12, C.L_ELBOW: 13, C.R_ELBOW: 14,
        C.L_WRIST: 15, C.R_WRIST: 16, C.L_HIP: 23, C.R_HIP: 24,
        C.L_KNEE: 25, C.R_KNEE: 26, C.L_ANKLE: 27, C.R_ANKLE: 28,
    }
    side_coco = {
        'left':  dict(shoulder=C.L_SHOULDER, elbow=C.L_ELBOW, wrist=C.L_WRIST),
        'right': dict(shoulder=C.R_SHOULDER, elbow=C.R_ELBOW, wrist=C.R_WRIST),
    }
    return C, coco_name, coco_to_mp, side_coco


def _frame_gen(path):
    cap = cv2.VideoCapture(path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def _world_track(kp3d, coco_idx, valid, n):
    """Hold the last validity-trusted world position across untrusted frames."""
    xs = np.zeros(n); ys = np.zeros(n); zs = np.zeros(n)
    last = None
    for i in range(n):
        p = kp3d[i, coco_idx]
        if np.all(np.isfinite(p)) and (valid[i] or last is None):
            last = p
        if last is not None:
            xs[i], ys[i], zs[i] = last
    return xs, ys, zs


def _world_rotation_series(pL, pR, lok, rok, n, max_jump_deg, max_gap):
    """Transverse-plane (x-z) rotation of the line R->L from metric 3D, with a
    clip-scaled magnitude floor + the same jump rejection / gap fill the
    MediaPipe analyzer applies to its image-space rotation."""
    dxs = pL[0] - pR[0]
    dzs = pL[2] - pR[2]
    mags = np.hypot(dxs, dzs)
    base = np.array([mags[i] for i in range(n) if lok[i] and rok[i] and mags[i] > 0])
    min_mag = 0.25 * float(np.median(base)) if base.size else 0.0
    raw_angs, valid = [], []
    for i in range(n):
        ok = bool(lok[i] and rok[i] and mags[i] >= min_mag)
        ang = (float(np.degrees(np.arctan2(dzs[i], dxs[i]))) if ok
               else (raw_angs[-1] if raw_angs else float(np.degrees(np.arctan2(dzs[i], dxs[i])))))
        raw_angs.append(ang)
        valid.append(ok)
    return ap.filter_angle_series(raw_angs, valid, max_jump_deg, max_gap=max_gap)


def _extract_multiview(video_paths, emit):
    """RTMPose 2D per view + markerless triangulation -> (kp3d_world (T,17,3),
    kp2d_img (T,17,2 normalized), vis (T,17), n_cams, reproj_err, W, H, cap_fps,
    warnings). Reference view (camera 0) supplies the 2D used for image-space
    metrics + drawing. Requires rtmlib; raises if unavailable."""
    from pitchcap import io_video, pose2d, reconstruct
    from pitchcap.sync import frame_offsets_from_audio
    from pitchcap.intrinsics import approximate_intrinsics

    emit('extracting', 0, 1)
    clips = [io_video.load_clip(p) for p in video_paths]
    cap_fps = clips[0].fps
    poses = [pose2d.estimate_pose_2d(c.frames) for c in clips]
    kp2d_views = [p[0] for p in poses]
    conf_views = [p[1] for p in poses]

    if all(len(c.audio) for c in clips):
        offsets = frame_offsets_from_audio([c.audio for c in clips], fps=cap_fps, sr=clips[0].sr)
        starts = [max(0, o) for o in offsets]
        m = min(len(a) - s for a, s in zip(kp2d_views, starts))
        kp2d_views = [a[s:s + m] for a, s in zip(kp2d_views, starts)]
        conf_views = [a[s:s + m] for a, s in zip(conf_views, starts)]

    intrinsics = [approximate_intrinsics(c.image_size) for c in clips]
    kp3d_world, reproj_err = reconstruct.reconstruct_multiview(kp2d_views, conf_views, intrinsics)

    W0, H0 = clips[0].image_size
    T = kp3d_world.shape[0]
    kp2d_img = kp2d_views[0][:T].astype(float).copy()
    kp2d_img[..., 0] /= max(W0, 1)
    kp2d_img[..., 1] /= max(H0, 1)
    vis = conf_views[0][:T]
    warnings = [f'camera_{i} using approximate intrinsics (no lens profile)'
                for i in range(len(clips))]
    return (kp3d_world[:T], kp2d_img, vis, len(clips), reproj_err,
            W0, H0, cap_fps, warnings)


def _analyze_multiview(video_paths, output_dir, throw_hand, progress_cb):
    os.makedirs(output_dir, exist_ok=True)

    def emit(stage, cur, tot):
        if progress_cb:
            progress_cb({'stage': stage, 'current': cur, 'total': tot,
                         'pct': round(cur / max(tot, 1) * 100, 1)})

    C, coco_name, coco_to_mp, side_coco = _maps()
    from pitchcap.filtering import DEFAULT_CUTOFFS
    from pitchcap.biomech import compute_kinematic_sequence

    (kp3d_world, kp2d_img, vis, n_cams, reproj_err,
     W, H, cap_fps, warnings) = _extract_multiview(video_paths, emit)
    n = int(kp3d_world.shape[0])
    if n == 0:
        raise ValueError('No frames reconstructed from the multi-view clips')

    ref_path = video_paths[0]
    fps, motion_fps, motion_fps_confident = ap.detect_frame_rates(ref_path, n, cap_fps)
    max_gap = max(0, round(motion_fps * ap.MAX_INTERP_GAP_S))

    throw_side = 'left' if throw_hand == 'left' else 'right'
    lead_side = 'right' if throw_hand == 'left' else 'left'
    ts_name, te_name, tw_name = f'{throw_side}_shoulder', f'{throw_side}_elbow', f'{throw_side}_wrist'
    la_name = f'{lead_side}_ankle'
    tc = side_coco[throw_side]

    # Robust image-space landmark tracks (reused verbatim from the MP analyzer).
    raw = []
    for i in range(n):
        lm = {}
        for ci, name in coco_name.items():
            x, y = kp2d_img[i, ci]
            if np.isfinite(x) and np.isfinite(y):
                lm[name] = {'x': float(x), 'y': float(y), 'z': 0.0, 'v': float(vis[i, ci])}
        raw.append({'frame': i, 'time': round(i / fps, 4), 'landmarks': lm})

    lhx, lhy, lhz, lh_ok = ap.build_robust_series(raw, 'left_hip', max_gap=max_gap)
    rhx, rhy, rhz, rh_ok = ap.build_robust_series(raw, 'right_hip', max_gap=max_gap)
    lsx, lsy, lsz, ls_ok = ap.build_robust_series(raw, 'left_shoulder', max_gap=max_gap)
    rsx, rsy, rsz, rs_ok = ap.build_robust_series(raw, 'right_shoulder', max_gap=max_gap)
    tex, tey, tez, te_ok = ap.build_robust_series(raw, te_name, max_gap=max_gap)
    twx, twy, twz, tw_ok = ap.build_robust_series(raw, tw_name, max_gap=max_gap)
    tsx, tsy, tsz, ts_ok = ap.build_robust_series(raw, ts_name, max_gap=max_gap)
    _, lay, _, _ = ap.build_robust_series(raw, la_name, max_gap=max_gap)

    low_conf = [not (lh_ok[i] and rh_ok[i] and ls_ok[i] and rs_ok[i]
                     and te_ok[i] and tw_ok[i] and ts_ok[i]) for i in range(n)]
    torso_ok = [lh_ok[i] and rh_ok[i] and ls_ok[i] and rs_ok[i] for i in range(n)]
    throw_arm_ok = [te_ok[i] and tw_ok[i] and ts_ok[i] for i in range(n)]

    twx_p = ap.smooth(twx.tolist(), window=7)
    twy_p = ap.smooth(twy.tolist(), window=7)
    wrist_disp = [0.0] + [float(np.hypot(twx_p[i] - twx_p[i - 1], twy_p[i] - twy_p[i - 1]))
                          for i in range(1, n)]

    # Rotation + elbow angles from TRUE triangulated 3D (the multi-view upgrade).
    max_rot_jump = 1200.0 / fps
    wlh = _world_track(kp3d_world, C.L_HIP, lh_ok, n)
    wrh = _world_track(kp3d_world, C.R_HIP, rh_ok, n)
    wls = _world_track(kp3d_world, C.L_SHOULDER, ls_ok, n)
    wrs = _world_track(kp3d_world, C.R_SHOULDER, rs_ok, n)
    hip_angs, hip_ang_ok = _world_rotation_series(wlh, wrh, lh_ok, rh_ok, n, max_rot_jump, max_gap)
    sho_angs, sho_ang_ok = _world_rotation_series(wls, wrs, ls_ok, rs_ok, n, max_rot_jump, max_gap)
    for i in range(n):
        if not hip_ang_ok[i] or not sho_ang_ok[i]:
            low_conf[i] = True
    hip_angs = np.degrees(np.unwrap(np.deg2rad(hip_angs))).tolist()
    sho_angs = np.degrees(np.unwrap(np.deg2rad(sho_angs))).tolist()

    wts = _world_track(kp3d_world, tc['shoulder'], ts_ok, n)
    wte = _world_track(kp3d_world, tc['elbow'], te_ok, n)
    wtw = _world_track(kp3d_world, tc['wrist'], tw_ok, n)
    elbow_angs, elbow_ok, trunk_angs = [], [], []
    for i in range(n):
        ok = bool(ts_ok[i] and te_ok[i] and tw_ok[i])
        ang = ap.angle_3pt_3d(wts[0][i], wts[1][i], wts[2][i],
                              wte[0][i], wte[1][i], wte[2][i],
                              wtw[0][i], wtw[1][i], wtw[2][i])
        if not ok and elbow_angs:
            ang = elbow_angs[-1]
        elbow_angs.append(ang)
        elbow_ok.append(ok)
        if not ok:
            low_conf[i] = True
        hip_mx, hip_my = (lhx[i] + rhx[i]) / 2, (lhy[i] + rhy[i]) / 2
        sho_mx, sho_my = (lsx[i] + rsx[i]) / 2, (lsy[i] + rsy[i]) / 2
        trunk_angs.append(float(np.degrees(np.arctan2(sho_mx - hip_mx, hip_my - sho_my))))

    elbow_height_pct = [round(float(tsy[i] - tey[i]) * 100, 2) for i in range(n)]

    frame_metrics, frame_phases, mf_summary = ap.compute_metrics(
        motion_fps, n, hip_angs, sho_angs, trunk_angs, elbow_angs,
        elbow_ok, hip_ang_ok, sho_ang_ok, low_conf,
        lay.tolist(), tey.tolist(), wrist_disp, elbow_height_pct)

    with open(os.path.join(output_dir, 'series.json'), 'w') as fp:
        json.dump({
            'n': n, 'hip_angs': hip_angs, 'sho_angs': sho_angs,
            'trunk_angs': trunk_angs, 'elbow_angs': elbow_angs,
            'elbow_ok': elbow_ok, 'hip_ang_ok': hip_ang_ok, 'sho_ang_ok': sho_ang_ok,
            'low_conf': low_conf, 'lay': lay.tolist(), 'tey': tey.tolist(),
            'wrist_disp': wrist_disp, 'elbow_height_pct': elbow_height_pct,
        }, fp)

    hand = 'L' if throw_hand == 'left' else 'R'
    ks = compute_kinematic_sequence(kp3d_world, motion_fps, handedness=hand,
                                    cutoffs=DEFAULT_CUTOFFS)
    joint_ok = {C.L_HIP: lh_ok, C.R_HIP: rh_ok,
                C.L_SHOULDER: ls_ok, C.R_SHOULDER: rs_ok,
                tc['elbow']: te_ok, tc['wrist']: tw_ok}
    tracked = ap._segment_tracked_fractions(joint_ok, hand, n)

    summary = {
        'fps': round(fps, 2),
        'motion_fps_confident': motion_fps_confident,
        'total_frames': n,
        'duration_s': round(n / fps, 3),
        'throw_hand': throw_hand,
        'frame_width': W,
        'frame_height': H,
        'tracking_quality': round(100.0 * (1 - sum(low_conf) / n), 1),
        'auto_zoom': {'applied': False, 'frame_pct': 100.0},
        'engine': 'pitchcap',
        **mf_summary,
        'pitchcap': {
            'engine': 'pitchcap', 'mode': 'multiview', 'n_cams': n_cams,
            'reprojection_error_px': reproj_err,
            'kinematic_sequence': {
                'fps': ks['fps'], 'handedness': ks['handedness'],
                'sequence_order': ks['sequence_order'],
                'inter_peak_lags_ms': ks['inter_peak_lags_ms'],
                'segments': {nm: {'peak_degps': sg['peak_degps'],
                                  'peak_time_s': sg['peak_time_s'],
                                  'series_degps': sg['series_degps'],
                                  'tracked_frac': tracked.get(nm)}
                             for nm, sg in ks['segments'].items()},
                'segment_warnings': ks.get('segment_warnings', []),
            },
            'warnings': warnings,
        },
    }

    # Annotate (reference view).
    emit('annotating', 0, n)
    throw_idx, lead_idx = ap.THROW[throw_hand], ap.LEAD[throw_hand]
    raw_out = os.path.join(output_dir, '_raw.mp4')
    writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    def lm33(i):
        lst = [SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in range(33)]
        for ci, mp_idx in coco_to_mp.items():
            x, y = kp2d_img[i, ci]
            if np.isfinite(x) and np.isfinite(y):
                lst[mp_idx] = SimpleNamespace(x=float(x), y=float(y), z=0.0,
                                              visibility=float(vis[i, ci]))
        return lst

    i = 0
    for bgr in _frame_gen(ref_path):
        if i >= n:
            break
        if torso_ok[i]:
            ap.draw_skeleton(bgr, lm33(i), throw_idx, lead_idx, W, H, throw_arm_ok=throw_arm_ok[i])
        ap.draw_hud(bgr, {'frame': i, 'time': round(i / fps, 4),
                          'metrics': frame_metrics[i]}, frame_phases[i])
        writer.write(bgr)
        i += 1
        if i % 15 == 0:
            emit('annotating', i, n)
    writer.release()

    ann_path = os.path.join(output_dir, 'annotated.mp4')
    _reencode_h264(raw_out, ann_path)

    json_frames = []
    for i in range(n):
        lm = {}
        for ci, name in coco_name.items():
            x, y = kp2d_img[i, ci]
            if np.isfinite(x) and np.isfinite(y):
                lm[name] = {'x': round(float(x), 5), 'y': round(float(y), 5),
                            'z': 0.0, 'v': round(float(vis[i, ci]), 3)}
        json_frames.append({'frame': i, 'time': round(i / fps, 4),
                            'phase': frame_phases[i], 'metrics': frame_metrics[i],
                            'landmarks': lm})

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as fp:
        json.dump({'summary': summary, 'frames': json_frames}, fp)

    emit('done', n, n)
    return {'annotated_video': ann_path, 'metrics': os.path.join(output_dir, 'metrics.json')}


def _reencode_h264(raw_out, ann_path):
    """Re-encode the mp4v writer output to browser-playable H.264 (mirrors
    analyze_pitcher); falls back to the raw file if ffmpeg is unavailable."""
    ffmpeg_bin = ap.find_ffmpeg()
    encoded = False
    if ffmpeg_bin:
        import subprocess
        cmd = [ffmpeg_bin, '-i', raw_out, '-vcodec', 'libx264', '-crf', '18',
               '-preset', 'medium', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', '-y', ann_path]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
            encoded = (proc.returncode == 0 and os.path.exists(ann_path)
                       and os.path.getsize(ann_path) > 1024)
            if not encoded:
                tail = proc.stderr.decode('utf-8', 'replace').strip().splitlines()[-8:]
                print('[warn] ffmpeg H.264 re-encode failed (rc=%d):\n  %s'
                      % (proc.returncode, '\n  '.join(tail)), file=sys.stderr)
        except Exception as e:
            print('[warn] ffmpeg H.264 re-encode raised %r' % e, file=sys.stderr)
    else:
        print('[warn] No ffmpeg binary found — annotated video stays mp4v.', file=sys.stderr)
    if encoded:
        os.remove(raw_out)
    else:
        os.rename(raw_out, ann_path)


def analyze(video_paths, output_dir, throw_hand='left', progress_cb=None):
    if len(video_paths) == 1:
        return _analyze_monocular(video_paths[0], output_dir, throw_hand, progress_cb)
    try:
        return _analyze_multiview(video_paths, output_dir, throw_hand, progress_cb)
    except Exception as e:
        print('[warn] multi-view path unavailable (%r); falling back to camera 0 '
              'monocular analysis.' % e, file=sys.stderr)
        return _analyze_monocular(video_paths[0], output_dir, throw_hand, progress_cb)


def main():
    parser = argparse.ArgumentParser(description='PitchCap engine (pitcher-review schema)')
    parser.add_argument('video', nargs='+', help='Input video path(s); 2+ enables multi-view')
    parser.add_argument('--output-dir', default='./pitcher_analysis')
    parser.add_argument('--throw-hand', choices=['left', 'right'], default='left')
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()

    def cb(d):
        if args.progress:
            print(json.dumps(d), flush=True)

    try:
        r = analyze(args.video, args.output_dir, args.throw_hand, cb)
        print(json.dumps({'status': 'done', **r}), flush=True)
    except Exception as e:
        import traceback
        print(json.dumps({'status': 'error', 'error': str(e)}), flush=True, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
