#!/usr/bin/env python3
"""
Cross-camera metric fusion (post-processing step).

Given two analyze_pitcher.py metrics.json outputs for the same pitch shot
from different angles, plus the time offset between their clocks (from the
existing release-frame sync in analysis.js), fill each frame's
low-confidence metrics using the other camera's data at the corresponding
instant, where the other camera was confident:

- elbow_angle is a 3D joint angle (computed from MediaPipe world
  landmarks), so it's camera-orientation-invariant and can be substituted
  directly.
- hip_rotation / shoulder_rotation / trunk_tilt are transverse-plane angles
  measured in each camera's own image coordinate frame, so two stationary
  cameras differ by a roughly constant offset. That offset is estimated
  from frames where both cameras are confident at the same instant, then
  applied when borrowing the other camera's value.
- hip_shoulder_sep and the *_speed metrics are recomputed from the fused
  angle series so they stay internally consistent.

Usage:
  python3 fuse_cameras.py <self_metrics.json> <other_metrics.json> <offset_seconds> <output.json>

offset_seconds is self_time - other_time for the same real-world instant
(t_other = t_self - offset_seconds).
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_pitcher import lowpass, derivative, best_peak, ROT_CUTOFF_HZ, ARM_CUTOFF_HZ

# Minimum number of frames where *both* cameras are simultaneously confident
# needed to trust a fitted cross-camera angle offset. Below this, the
# relationship between the two cameras' coordinate frames can't be reliably
# estimated, so no fusion is applied for this pair.
MIN_OVERLAP_FRAMES = 10


def circular_mean_deg(diffs_deg):
    rad = np.deg2rad(diffs_deg)
    return float(np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))))


def nearest_index(times, t, max_dt):
    idx = int(np.searchsorted(times, t))
    cand = [j for j in (idx - 1, idx) if 0 <= j < len(times)]
    if not cand:
        return None
    best = min(cand, key=lambda j: abs(times[j] - t))
    return best if abs(times[best] - t) <= max_dt else None


def fuse(self_path, other_path, offset, out_path):
    self_d  = json.load(open(self_path))
    other_d = json.load(open(other_path))
    self_frames  = self_d['frames']
    other_frames = other_d['frames']
    n = len(self_frames)
    summary = self_d['summary']
    motion_fps = summary['motion_fps']

    self_times  = np.array([f['time'] for f in self_frames])
    other_times = np.array([f['time'] for f in other_frames])
    max_dt = (np.median(np.diff(self_times)) if n > 1 else 1.0 / summary['fps']) * 1.5

    self_hip   = np.degrees(np.unwrap(np.deg2rad([f['metrics']['hip_rotation']      for f in self_frames])))
    self_sho   = np.degrees(np.unwrap(np.deg2rad([f['metrics']['shoulder_rotation'] for f in self_frames])))
    self_trunk = np.degrees(np.unwrap(np.deg2rad([f['metrics']['trunk_tilt']        for f in self_frames])))
    self_elbow = np.array([f['metrics']['elbow_angle'] for f in self_frames])
    self_lc    = np.array([f['metrics']['low_confidence'] for f in self_frames])

    other_hip_full   = np.degrees(np.unwrap(np.deg2rad([f['metrics']['hip_rotation']      for f in other_frames])))
    other_sho_full   = np.degrees(np.unwrap(np.deg2rad([f['metrics']['shoulder_rotation'] for f in other_frames])))
    other_trunk_full = np.degrees(np.unwrap(np.deg2rad([f['metrics']['trunk_tilt']        for f in other_frames])))
    other_elbow_full = np.array([f['metrics']['elbow_angle'] for f in other_frames])
    other_lc_full    = np.array([f['metrics']['low_confidence'] for f in other_frames])

    matched       = np.full(n, -1, dtype=int)
    matched_hip   = np.full(n, np.nan)
    matched_sho   = np.full(n, np.nan)
    matched_trunk = np.full(n, np.nan)
    matched_elbow = np.full(n, np.nan)
    matched_lc    = np.ones(n, dtype=bool)

    for i, t in enumerate(self_times):
        idx = nearest_index(other_times, t - offset, max_dt)
        if idx is None:
            continue
        matched[i]       = idx
        matched_hip[i]   = other_hip_full[idx]
        matched_sho[i]   = other_sho_full[idx]
        matched_trunk[i] = other_trunk_full[idx]
        matched_elbow[i] = other_elbow_full[idx]
        matched_lc[i]    = other_lc_full[idx]

    have_match = matched >= 0
    both_ok = have_match & ~self_lc & ~matched_lc
    overlap_n = int(both_ok.sum())
    fusion_applied = overlap_n >= MIN_OVERLAP_FRAMES

    fused_hip_raw   = self_hip.copy()
    fused_sho_raw   = self_sho.copy()
    fused_trunk_raw = self_trunk.copy()
    fused_elbow     = self_elbow.copy()
    fused_lc        = self_lc.copy()
    fused_from      = ['self'] * n

    hip_offset = sho_offset = trunk_offset = None
    if fusion_applied:
        hip_offset   = circular_mean_deg(self_hip[both_ok]   - matched_hip[both_ok])
        sho_offset   = circular_mean_deg(self_sho[both_ok]   - matched_sho[both_ok])
        trunk_offset = circular_mean_deg(self_trunk[both_ok] - matched_trunk[both_ok])

        fill_mask = self_lc & have_match & ~matched_lc
        for i in np.where(fill_mask)[0]:
            fused_hip_raw[i]   = matched_hip[i]   + hip_offset
            fused_sho_raw[i]   = matched_sho[i]   + sho_offset
            fused_trunk_raw[i] = matched_trunk[i] + trunk_offset
            fused_elbow[i]     = matched_elbow[i]
            fused_lc[i]        = False
            fused_from[i]      = 'other'

    # Substituted values are only correct modulo 360 (each camera's series
    # was unwrapped from its own arbitrary base point), so re-unwrap the
    # mixed series to remove any spurious large jumps at fill boundaries.
    fused_hip   = np.degrees(np.unwrap(np.deg2rad(fused_hip_raw)))
    fused_sho   = np.degrees(np.unwrap(np.deg2rad(fused_sho_raw)))
    fused_trunk = np.degrees(np.unwrap(np.deg2rad(fused_trunk_raw)))

    # Recompute angle-derived speeds from the fused series so they stay
    # internally consistent with the (possibly substituted) angles.
    hip_s   = lowpass(fused_hip.tolist(),   motion_fps, ROT_CUTOFF_HZ)
    sho_s   = lowpass(fused_sho.tolist(),   motion_fps, ROT_CUTOFF_HZ)
    trunk_s = lowpass(fused_trunk.tolist(), motion_fps, ROT_CUTOFF_HZ)
    elbow_s = lowpass(fused_elbow.tolist(), motion_fps, ARM_CUTOFF_HZ)
    hip_vel   = derivative(hip_s,   motion_fps)
    chest_vel = derivative(sho_s,   motion_fps)
    arm_vel   = derivative(elbow_s, motion_fps)

    out_frames = []
    arm_speeds, hip_speeds, chest_speeds, hss_vals = [], [], [], []
    for i, f in enumerate(self_frames):
        hip_disp   = ((hip_s[i] + 180) % 360) - 180
        sho_disp   = ((sho_s[i] + 180) % 360) - 180
        trunk_disp = ((trunk_s[i] + 180) % 360) - 180
        hss = abs(((hip_s[i] - sho_s[i] + 180) % 360) - 180)
        m = dict(f['metrics'])
        m['hip_rotation']         = round(hip_disp, 2)
        m['shoulder_rotation']    = round(sho_disp, 2)
        m['hip_shoulder_sep']     = round(hss, 2)
        m['trunk_tilt']           = round(trunk_disp, 2)
        m['elbow_angle']          = round(elbow_s[i], 2)
        m['hip_rotation_speed']   = round(hip_vel[i], 2)
        m['chest_rotation_speed'] = round(chest_vel[i], 2)
        m['arm_speed']            = round(arm_vel[i], 2)
        m['low_confidence']       = bool(fused_lc[i])
        m['fused_from']           = fused_from[i]
        out_frames.append({**f, 'metrics': m})
        arm_speeds.append(abs(m['arm_speed']))
        hip_speeds.append(abs(m['hip_rotation_speed']))
        chest_speeds.append(abs(m['chest_rotation_speed']))
        hss_vals.append(m['hip_shoulder_sep'])

    valid = (~fused_lc).tolist()

    phases     = summary['phases']
    key_frames = summary['key_frames']
    fs_f      = key_frames['foot_strike']
    release_f = key_frames['release']
    ss_f      = phases.get('stride', {}).get('start', fs_f)
    sec = lambda s: int(round(s * motion_fps))
    rot_lo = max(0, fs_f - sec(0.10))
    spd_hi = min(n - 1, release_f + sec(0.15))
    arm_hi = min(n - 1, release_f + sec(0.20))

    arm_peak_f,   arm_peak_lc   = best_peak(arm_speeds,   valid, fs_f,   arm_hi, n)
    hip_peak_f,   hip_peak_lc   = best_peak(hip_speeds,   valid, rot_lo, spd_hi, n)
    chest_peak_f, chest_peak_lc = best_peak(chest_speeds, valid, rot_lo, spd_hi, n)
    hss_peak_f,   hss_peak_lc   = best_peak(hss_vals,     valid, ss_f,   spd_hi, n)

    phase_confidence = {}
    for name, span in phases.items():
        s, e = span['start'], span['end']
        seg = fused_lc[s:e + 1]
        phase_confidence[name] = round(100.0 * (1 - seg.sum() / len(seg)), 1) if len(seg) else 100.0

    out_summary = dict(summary)
    out_summary['tracking_quality'] = round(100.0 * (1 - fused_lc.sum() / n), 1)
    out_summary['phase_confidence'] = phase_confidence
    out_summary['fusion'] = {
        'applied':             fusion_applied,
        'overlap_frames':      overlap_n,
        'filled_frames':       int(sum(1 for x in fused_from if x == 'other')),
        'hip_offset_deg':      round(hip_offset, 2)   if fusion_applied else None,
        'shoulder_offset_deg': round(sho_offset, 2)   if fusion_applied else None,
        'trunk_offset_deg':    round(trunk_offset, 2) if fusion_applied else None,
    }
    out_summary['peak'] = dict(summary['peak'])
    out_summary['peak'].update({
        'max_arm_speed':                         round(arm_speeds[arm_peak_f], 1),
        'max_arm_speed_frame':                   arm_peak_f,
        'max_arm_speed_low_confidence':          arm_peak_lc,
        'max_hip_rotation_speed':                round(hip_speeds[hip_peak_f], 1),
        'max_hip_rotation_speed_frame':          hip_peak_f,
        'max_hip_rotation_speed_low_confidence': hip_peak_lc,
        'max_chest_rotation_speed':              round(chest_speeds[chest_peak_f], 1),
        'max_chest_rotation_speed_frame':        chest_peak_f,
        'max_chest_rotation_speed_low_confidence': chest_peak_lc,
        'max_hip_shoulder_sep':                  round(hss_vals[hss_peak_f], 1),
        'max_hss_frame':                         hss_peak_f,
        'max_hss_low_confidence':                hss_peak_lc,
        'arm_speed_at_release':                  round(arm_speeds[release_f], 1) if release_f < n else 0,
        'arm_speed_at_release_low_confidence':   (not valid[release_f]) if release_f < n else False,
        'hss_at_foot_strike':                    round(hss_vals[fs_f], 1),
        'hss_at_foot_strike_low_confidence':     bool(not valid[fs_f]),
    })
    out_summary['sequencing'] = {
        'hip_peak_frame':   hip_peak_f,
        'chest_peak_frame': chest_peak_f,
        'arm_peak_frame':   arm_peak_f,
        'hip_to_chest_ms':  round((chest_peak_f - hip_peak_f) / motion_fps * 1000, 1),
        'chest_to_arm_ms':  round((arm_peak_f - chest_peak_f) / motion_fps * 1000, 1),
        'hip_to_arm_ms':    round((arm_peak_f - hip_peak_f) / motion_fps * 1000, 1),
        'proper_order':     hip_peak_f <= chest_peak_f <= arm_peak_f,
        'low_confidence':   bool(hip_peak_lc or chest_peak_lc or arm_peak_lc),
    }

    with open(out_path, 'w') as fp:
        json.dump({'summary': out_summary, 'frames': out_frames}, fp)

    return out_summary


def main():
    if len(sys.argv) != 5:
        print('usage: fuse_cameras.py <self_metrics.json> <other_metrics.json> '
              '<offset_seconds> <output.json>', file=sys.stderr)
        sys.exit(1)
    self_path, other_path, offset_s, out_path = sys.argv[1:]
    summary = fuse(self_path, other_path, float(offset_s), out_path)
    print(json.dumps({
        'status': 'done',
        'fusion': summary['fusion'],
        'tracking_quality': summary['tracking_quality'],
    }))


if __name__ == '__main__':
    main()
