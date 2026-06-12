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
