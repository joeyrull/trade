import json

import numpy as np
from pitchcap import constants as C
from pitchcap.filtering import filter_keypoints, interpolate_nans
from pitchcap.biomech import compute_kinematic_sequence


def test_interpolate_internal_nans():
    arr = np.array([0.0, np.nan, 2.0, np.nan, 4.0]).reshape(5, 1, 1)
    out = interpolate_nans(arr)
    assert np.allclose(out.ravel(), [0, 1, 2, 3, 4])


def test_filter_handles_internal_nans():
    fps, T = 240, 200
    kp = np.random.RandomState(0).randn(T, 17, 3)
    kp[50:55, C.R_ELBOW, 0] = np.nan          # short internal gap
    out = filter_keypoints(kp, fps, cutoff_hz=15)
    assert out.shape == kp.shape
    assert np.isfinite(out[:, C.R_ELBOW, 0]).all()   # gap interpolated then filtered


def test_filter_fully_nan_joint_no_crash():
    fps, T = 240, 200
    kp = np.random.RandomState(1).randn(T, 17, 3)
    kp[:, C.R_WRIST, :] = np.nan              # joint never reconstructed
    out = filter_keypoints(kp, fps, cutoff_hz=15)
    assert out.shape == kp.shape
    assert np.isnan(out[:, C.R_WRIST, :]).all()      # stays NaN, no crash
    assert np.isfinite(out[:, C.R_HIP, :]).all()     # others fine


def test_filter_short_clip_no_crash():
    fps, T = 240, 8                            # shorter than filtfilt padlen
    kp = np.random.RandomState(2).randn(T, 17, 3)
    out = filter_keypoints(kp, fps, cutoff_hz=15)
    assert out.shape == kp.shape
    assert np.isfinite(out).all()


def test_filter_low_fps_no_crash():
    # 30fps footage with a 15Hz cutoff sits at Nyquist -> must clamp, not crash
    fps, T = 30, 120
    kp = np.random.RandomState(7).randn(T, 17, 3)
    out = filter_keypoints(kp, fps, cutoff_hz=15.0)
    assert out.shape == kp.shape
    assert np.isfinite(out).all()


def test_dead_arm_joint_sorts_last_and_warns():
    fps, T = 240, 300
    t = np.arange(T) / fps
    kp = np.zeros((T, C.N_KEYPOINTS, 3))
    kp[:, C.R_HIP] = [0, 0, 0]; kp[:, C.L_HIP] = [1, 0, 0]
    kp[:, C.R_SHOULDER] = [0, 1, 0]; kp[:, C.L_SHOULDER] = [1, 1, 0]
    kp[:, C.R_WRIST] = [0, 0, 0]
    # rotate pelvis then trunk
    kp[:, C.L_HIP, 1] += 0.3 * np.exp(-((t - 0.4) ** 2) / (2 * 0.02 ** 2))
    kp[:, C.L_SHOULDER, 0] += 0.3 * np.exp(-((t - 0.5) ** 2) / (2 * 0.02 ** 2))
    # elbow never reconstructed -> both shoulder (sh->el) and elbow (el->wr) are dead
    kp[:, C.R_ELBOW, :] = np.nan
    res = compute_kinematic_sequence(kp, fps, handedness="R")
    # invalid segments sort last, in SEGMENTS order (stable sort) when tied at inf
    assert res["sequence_order"] == ["pelvis", "trunk", "shoulder", "elbow"]
    assert any("shoulder" in w for w in res["segment_warnings"])
    assert any("elbow" in w for w in res["segment_warnings"])
    assert np.isfinite(res["segments"]["pelvis"]["peak_degps"])


def test_dead_segment_output_is_json_safe():
    """A segment occluded the whole clip must not put Infinity/NaN into the
    result — json.dump would emit invalid tokens that break the Node server and
    browser JSON.parse, making an otherwise-fine analysis unreadable."""
    fps, T = 240, 300
    t = np.arange(T) / fps
    kp = np.zeros((T, C.N_KEYPOINTS, 3))
    kp[:, C.R_HIP] = [0, 0, 0]; kp[:, C.L_HIP] = [1, 0, 0]
    kp[:, C.R_SHOULDER] = [0, 1, 0]; kp[:, C.L_SHOULDER] = [1, 1, 0]
    kp[:, C.R_WRIST] = [0, 0, 0]
    kp[:, C.L_HIP, 1] += 0.3 * np.exp(-((t - 0.4) ** 2) / (2 * 0.02 ** 2))
    kp[:, C.R_ELBOW, :] = np.nan                        # throwing arm never seen
    res = compute_kinematic_sequence(kp, fps, handedness="R")

    # invalid shoulder/elbow -> lags are None (not inf), series uses null (not NaN)
    assert res["inter_peak_lags_ms"]["trunk_to_shoulder"] is None
    assert res["inter_peak_lags_ms"]["shoulder_to_elbow"] is None
    assert res["segments"]["shoulder"]["series_degps"][0] is None
    assert res["segments"]["elbow"]["series_degps"][0] is None
    # the whole structure round-trips as STRICT JSON (allow_nan=False) — exactly
    # the constraint JSON.parse enforces on the Node/browser side
    dumped = json.dumps(res, allow_nan=False)
    assert "Infinity" not in dumped and "NaN" not in dumped
