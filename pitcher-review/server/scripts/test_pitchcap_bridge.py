"""Tests for the PitchCap bridge glue in analyze_pitcher.py.

Covers the pure-Python integration logic (per-segment tracked fractions and the
kinematic-sequence attachment) on synthetic world-landmark data — no video,
model download, or MediaPipe inference required. ``import analyze_pitcher`` does
load mediapipe at import time, which is available in the analysis environment.

Run:  cd server/scripts && python3 -m pytest test_pitchcap_bridge.py -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_pitcher as ap  # noqa: E402
import pitchcap_analyze as pa  # noqa: E402


# ── _aligned_window (multi-view time alignment) ─────────────────────────────

def test_aligned_window_lagging_clip():
    starts, m = pa._aligned_window([0, 12], [100, 100])
    assert starts == [0, 12] and m == 88   # clip 1 lags 12 -> drop its first 12


def test_aligned_window_leading_clip():
    starts, m = pa._aligned_window([0, -12], [100, 100])
    assert starts == [12, 0] and m == 88   # clip 1 leads -> trim the reference


def test_aligned_window_implausible_offset_falls_back():
    starts, m = pa._aligned_window([0, 500], [100, 100])  # start would exceed clip
    assert starts == [0, 0] and m == 100   # abandon alignment, no empty slice


def test_aligned_window_no_audio_uses_common_length():
    starts, m = pa._aligned_window(None, [100, 90])
    assert starts == [0, 0] and m == 90    # still trim to a common length


# ── _segment_tracked_fractions ──────────────────────────────────────────────

def test_tracked_fractions_all_valid():
    n = 50
    T = [True] * n
    # right-handed: arm uses shoulder 6 + elbow 8
    joint_ok = {11: T, 12: T, 5: T, 6: T, 8: T, 10: T}
    fr = ap._segment_tracked_fractions(joint_ok, 'R', n)
    assert fr['pelvis'] == 1.0 and fr['trunk'] == 1.0 and fr['arm'] == 1.0


def test_tracked_fractions_partial_arm():
    n = 100
    T = [True] * n
    half = [i % 2 == 0 for i in range(n)]  # 50% valid
    joint_ok = {11: T, 12: T, 5: T, 6: T, 8: half, 10: T}
    fr = ap._segment_tracked_fractions(joint_ok, 'R', n)
    assert fr['pelvis'] == 1.0
    assert fr['trunk'] == 1.0
    assert abs(fr['arm'] - 0.5) < 1e-6  # gated by the half-valid elbow


def test_tracked_fractions_missing_mask_is_none():
    n = 10
    joint_ok = {11: [True] * n, 12: [True] * n}  # no shoulders/elbow
    fr = ap._segment_tracked_fractions(joint_ok, 'R', n)
    assert fr['pelvis'] == 1.0
    assert fr['trunk'] is None  # needs shoulders
    assert fr['arm'] is None    # needs shoulder + elbow


# ── attach_kinematic_sequence ───────────────────────────────────────────────

def _staggered_raw(n=300, fps=240):
    """World-landmark `raw` with pelvis, then trunk, then arm rotation bursts —
    the proximal→distal pattern, mirroring pitchcap's own biomech sequence test
    but in the per-frame `world` dict shape analyze_pitcher stores."""
    t = np.arange(n) / fps

    def burst(center, amp):
        return amp * np.exp(-((t - center) ** 2) / (2 * 0.02 ** 2))

    lh_y = burst(0.40, 0.3)   # pelvis fires first
    ls_x = burst(0.50, 0.3)   # trunk next
    re_x = burst(0.60, 0.5)   # throwing arm last (handedness R)

    raw = []
    for i in range(n):
        world = {
            'right_hip':      {'x': 0.0,            'y': 0.0,        'z': 0.0},
            'left_hip':       {'x': 1.0,            'y': float(lh_y[i]), 'z': 0.0},
            'right_shoulder': {'x': 0.0,            'y': 1.0,        'z': 0.0},
            'left_shoulder':  {'x': 1.0 + float(ls_x[i]), 'y': 1.0,  'z': 0.0},
            'right_elbow':    {'x': float(re_x[i]), 'y': 0.5,        'z': 0.0},
        }
        raw.append({'frame': i, 'time': i / fps, 'world': world})
    return raw, fps


def test_attach_produces_proper_sequence():
    raw, fps = _staggered_raw()
    summary = {}
    ap.attach_kinematic_sequence(summary, raw, fps, 'right')
    assert 'pitchcap' in summary
    ks = summary['pitchcap']['kinematic_sequence']
    assert ks['sequence_order'] == ['pelvis', 'trunk', 'arm']
    assert ks['segments']['arm']['peak_degps'] > 0
    assert ks['inter_peak_lags_ms']['pelvis_to_trunk'] > 0
    assert summary['pitchcap']['mode'] == 'monocular'


def test_attach_gating_drops_untracked_arm():
    raw, fps = _staggered_raw()
    n = len(raw)
    allT = [True] * n
    # right-handed arm = shoulder 6 + elbow 8; mark the elbow untracked all clip
    joint_ok = {11: allT, 12: allT, 5: allT, 6: allT, 8: [False] * n, 10: allT}
    summary = {}
    ap.attach_kinematic_sequence(summary, raw, fps, 'right', joint_ok)
    ks = summary['pitchcap']['kinematic_sequence']
    arm = ks['segments']['arm']
    # arm joints NaN'd out -> segment never reconstructed -> honest invalid
    assert arm['peak_time_s'] is None
    assert arm['tracked_frac'] == 0.0
    assert any('arm' in w for w in ks['segment_warnings'])
    # pelvis/trunk were fully tracked and still produce valid peaks
    assert ks['segments']['pelvis']['peak_time_s'] is not None
    assert ks['segments']['pelvis']['tracked_frac'] == 1.0


def test_attach_noop_without_world_landmarks():
    n = 20
    raw = [{'frame': i, 'time': i / 30.0, 'world': {}} for i in range(n)]
    summary = {}
    ap.attach_kinematic_sequence(summary, raw, 30.0, 'right')
    assert 'pitchcap' not in summary  # nothing to compute, summary untouched
