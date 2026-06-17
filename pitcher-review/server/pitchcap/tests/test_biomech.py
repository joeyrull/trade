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


from pitchcap.biomech import compute_kinematic_sequence


def test_sequence_order_and_peaks():
    fps, T = 240, 300
    t = np.arange(T) / fps
    kp = np.zeros((T, C.N_KEYPOINTS, 3))
    # static base geometry
    kp[:, C.R_HIP] = [0, 0, 0]; kp[:, C.L_HIP] = [1, 0, 0]
    kp[:, C.R_SHOULDER] = [0, 1, 0]; kp[:, C.L_SHOULDER] = [1, 1, 0]
    kp[:, C.R_ELBOW] = [0, 0.5, 0]; kp[:, C.R_WRIST] = [0, 0, 0]

    # inject a rotation burst into each segment at staggered times
    def burst(center_s, amp):
        return amp * np.exp(-((t - center_s) ** 2) / (2 * 0.02 ** 2))

    # rotate L_HIP (pelvis), sh_mid (trunk), elbow (shoulder), wrist (elbow)
    # in that order -- the proximal->distal pattern.
    kp[:, C.L_HIP, 1] += burst(0.40, 0.3)
    kp[:, C.L_SHOULDER, 0] += burst(0.50, 0.3)
    kp[:, C.R_ELBOW, 0] += burst(0.60, 0.3)
    kp[:, C.R_WRIST, 0] += burst(0.70, 0.5)

    res = compute_kinematic_sequence(kp, fps, handedness="R")
    assert res["sequence_order"] == ["pelvis", "trunk", "shoulder", "elbow"]
    assert res["segments"]["shoulder"]["peak_degps"] > 0
    assert res["segments"]["elbow"]["peak_degps"] > 0
    assert res["inter_peak_lags_ms"]["pelvis_to_trunk"] > 0
    assert res["inter_peak_lags_ms"]["trunk_to_shoulder"] > 0
    assert res["inter_peak_lags_ms"]["shoulder_to_elbow"] > 0


from pitchcap.filtering import DEFAULT_CUTOFFS


def test_per_segment_cutoffs_attenuate_arm_jitter():
    """Per-segment cutoffs filter each segment's joints before differentiation,
    so high-frequency landmark jitter on the arm doesn't inflate its peak."""
    fps, T = 240, 300
    t = np.arange(T) / fps
    kp = np.zeros((T, C.N_KEYPOINTS, 3))
    kp[:, C.R_HIP] = [0, 0, 0]; kp[:, C.L_HIP] = [1, 0, 0]
    kp[:, C.R_SHOULDER] = [0, 1, 0]; kp[:, C.L_SHOULDER] = [1, 1, 0]
    kp[:, C.R_ELBOW] = [0, 0.5, 0]
    # 2 Hz real arm motion + 60 Hz jitter on the throwing elbow
    kp[:, C.R_ELBOW, 0] += 0.3 * np.sin(2 * np.pi * 2 * t)
    kp[:, C.R_ELBOW, 0] += 0.02 * np.sin(2 * np.pi * 60 * t)

    raw = compute_kinematic_sequence(kp, fps, handedness="R")                       # no filtering
    filt = compute_kinematic_sequence(kp, fps, handedness="R", cutoffs=DEFAULT_CUTOFFS)
    # the 18Hz shoulder cutoff removes the 60Hz-driven velocity spikes
    assert filt["segments"]["shoulder"]["peak_degps"] < raw["segments"]["shoulder"]["peak_degps"]
    # structure stays valid and the real 2Hz motion survives
    assert filt["segments"]["shoulder"]["peak_degps"] > 0
    assert filt["sequence_order"]


def test_cutoffs_none_is_unfiltered_passthrough():
    """Default (cutoffs=None) must be identical to not passing cutoffs."""
    fps, T = 240, 120
    kp = np.random.RandomState(3).randn(T, C.N_KEYPOINTS, 3)
    a = compute_kinematic_sequence(kp, fps, handedness="R")
    b = compute_kinematic_sequence(kp, fps, handedness="R", cutoffs=None)
    assert a["segments"]["shoulder"]["series_degps"] == b["segments"]["shoulder"]["series_degps"]
