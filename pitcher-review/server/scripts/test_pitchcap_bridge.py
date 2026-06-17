"""Tests for the PitchCap bridge glue in analyze_pitcher.py.

Covers the pure-Python integration logic (per-segment tracked fractions and the
kinematic-sequence attachment) on synthetic world-landmark data — no video,
model download, or MediaPipe inference required. ``import analyze_pitcher`` does
load mediapipe at import time, which is available in the analysis environment.

Run:  cd server/scripts && python3 -m pytest test_pitchcap_bridge.py -q
"""
import os
import sys
import json

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


# ── json_sanitize (serialization safety net) ────────────────────────────────

def test_json_sanitize_replaces_non_finite():
    obj = {"a": float('inf'), "b": [1.0, float('nan'), 3.0],
           "c": {"d": -float('inf')}, "e": "x", "f": 2, "g": None}
    out = ap.json_sanitize(obj)
    assert out["a"] is None
    assert out["b"] == [1.0, None, 3.0]
    assert out["c"]["d"] is None
    assert out["e"] == "x" and out["f"] == 2 and out["g"] is None
    json.dumps(out, allow_nan=False)   # strict JSON: must not raise


# ── _segment_tracked_fractions ──────────────────────────────────────────────

def test_tracked_fractions_all_valid():
    n = 50
    T = [True] * n
    # right-handed: shoulder uses 6+8, elbow uses 6+8+10
    joint_ok = {11: T, 12: T, 5: T, 6: T, 8: T, 10: T}
    fr = ap._segment_tracked_fractions(joint_ok, 'R', n)
    assert fr['pelvis'] == 1.0 and fr['trunk'] == 1.0
    assert fr['shoulder'] == 1.0 and fr['elbow'] == 1.0


def test_tracked_fractions_partial_arm():
    n = 100
    T = [True] * n
    half = [i % 2 == 0 for i in range(n)]  # 50% valid
    joint_ok = {11: T, 12: T, 5: T, 6: T, 8: half, 10: T}
    fr = ap._segment_tracked_fractions(joint_ok, 'R', n)
    assert fr['pelvis'] == 1.0
    assert fr['trunk'] == 1.0
    assert abs(fr['shoulder'] - 0.5) < 1e-6  # gated by the half-valid elbow
    assert abs(fr['elbow'] - 0.5) < 1e-6     # also gated by the half-valid elbow


def test_tracked_fractions_missing_mask_is_none():
    n = 10
    joint_ok = {11: [True] * n, 12: [True] * n}  # no shoulders/elbow/wrist
    fr = ap._segment_tracked_fractions(joint_ok, 'R', n)
    assert fr['pelvis'] == 1.0
    assert fr['trunk'] is None     # needs shoulders
    assert fr['shoulder'] is None  # needs shoulder + elbow
    assert fr['elbow'] is None     # needs shoulder + elbow + wrist


# ── attach_kinematic_sequence ───────────────────────────────────────────────

def _staggered_raw(n=300, fps=240):
    """World-landmark `raw` with pelvis, then trunk, then shoulder, then elbow
    rotation bursts — the proximal→distal pattern, mirroring pitchcap's own
    biomech sequence test but in the per-frame `world` dict shape
    analyze_pitcher stores."""
    t = np.arange(n) / fps

    def burst(center, amp):
        return amp * np.exp(-((t - center) ** 2) / (2 * 0.02 ** 2))

    lh_y = burst(0.40, 0.3)   # pelvis fires first
    ls_x = burst(0.50, 0.3)   # trunk next
    re_x = burst(0.60, 0.3)   # shoulder (upper arm) next
    rw_x = burst(0.70, 0.5)   # elbow (forearm) last (handedness R)

    raw = []
    for i in range(n):
        world = {
            'right_hip':      {'x': 0.0,            'y': 0.0,        'z': 0.0},
            'left_hip':       {'x': 1.0,            'y': float(lh_y[i]), 'z': 0.0},
            'right_shoulder': {'x': 0.0,            'y': 1.0,        'z': 0.0},
            'left_shoulder':  {'x': 1.0 + float(ls_x[i]), 'y': 1.0,  'z': 0.0},
            'right_elbow':    {'x': float(re_x[i]), 'y': 0.5,        'z': 0.0},
            'right_wrist':    {'x': float(rw_x[i]), 'y': 0.0,        'z': 0.0},
        }
        raw.append({'frame': i, 'time': i / fps, 'world': world})
    return raw, fps


def test_attach_produces_proper_sequence():
    raw, fps = _staggered_raw()
    summary = {}
    ap.attach_kinematic_sequence(summary, raw, fps, 'right')
    assert 'pitchcap' in summary
    ks = summary['pitchcap']['kinematic_sequence']
    assert ks['sequence_order'] == ['pelvis', 'trunk', 'shoulder', 'elbow']
    assert ks['segments']['shoulder']['peak_degps'] > 0
    assert ks['segments']['elbow']['peak_degps'] > 0
    assert ks['inter_peak_lags_ms']['pelvis_to_trunk'] > 0
    assert ks['inter_peak_lags_ms']['trunk_to_shoulder'] > 0
    assert ks['inter_peak_lags_ms']['shoulder_to_elbow'] > 0
    assert summary['pitchcap']['mode'] == 'monocular'


def test_attach_gating_drops_untracked_arm():
    raw, fps = _staggered_raw()
    n = len(raw)
    allT = [True] * n
    # right-handed shoulder = 6+8, elbow = 6+8+10; mark the elbow joint untracked all clip
    joint_ok = {11: allT, 12: allT, 5: allT, 6: allT, 8: [False] * n, 10: allT}
    summary = {}
    ap.attach_kinematic_sequence(summary, raw, fps, 'right', joint_ok)
    ks = summary['pitchcap']['kinematic_sequence']
    # elbow joint NaN'd out -> both shoulder (sh->el) and elbow (el->wr) segments
    # depend on it -> neither is ever reconstructed -> honest invalid
    for name in ('shoulder', 'elbow'):
        seg = ks['segments'][name]
        assert seg['peak_time_s'] is None
        assert seg['tracked_frac'] == 0.0
        assert any(name in w for w in ks['segment_warnings'])
    # pelvis/trunk were fully tracked and still produce valid peaks
    assert ks['segments']['pelvis']['peak_time_s'] is not None
    assert ks['segments']['pelvis']['tracked_frac'] == 1.0


def test_attach_noop_without_world_landmarks():
    n = 20
    raw = [{'frame': i, 'time': i / 30.0, 'world': {}} for i in range(n)]
    summary = {}
    ap.attach_kinematic_sequence(summary, raw, 30.0, 'right')
    assert 'pitchcap' not in summary  # nothing to compute, summary untouched


# ── _load_intrinsics (per-camera calibration profiles) ──────────────────────

class _FakeClip:
    def __init__(self, image_size):
        self.image_size = image_size


def test_load_intrinsics_falls_back_without_profiles(monkeypatch, tmp_path):
    monkeypatch.setattr(pa, 'CALIBRATION_DIR', str(tmp_path))
    clips = [_FakeClip((1280, 720)), _FakeClip((1280, 720))]
    intrinsics, warnings = pa._load_intrinsics(clips)
    assert len(intrinsics) == 2
    assert all('no calibration profile' in w for w in warnings)


def test_load_intrinsics_uses_matching_profile(monkeypatch, tmp_path):
    from pitchcap.intrinsics import save_profile
    monkeypatch.setattr(pa, 'CALIBRATION_DIR', str(tmp_path))
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    save_profile(str(tmp_path / 'cam0.json'), K, np.zeros(5), (1280, 720))

    clips = [_FakeClip((1280, 720)), _FakeClip((1280, 720))]
    intrinsics, warnings = pa._load_intrinsics(clips)
    assert np.allclose(intrinsics[0].K, K)            # camera 0: calibrated profile used
    assert not any('camera_0' in w for w in warnings)
    assert any('camera_1' in w and 'no calibration profile' in w for w in warnings)  # camera 1: fallback


def test_load_intrinsics_ignores_mismatched_resolution(monkeypatch, tmp_path):
    from pitchcap.intrinsics import save_profile
    monkeypatch.setattr(pa, 'CALIBRATION_DIR', str(tmp_path))
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    save_profile(str(tmp_path / 'cam0.json'), K, np.zeros(5), (1920, 1080))  # different resolution

    clips = [_FakeClip((1280, 720))]
    intrinsics, warnings = pa._load_intrinsics(clips)
    assert not np.allclose(intrinsics[0].K, K)         # fell back to approximate, not the mismatched profile
    assert any('camera_0' in w and 'but clip is' in w for w in warnings)
