"""Multi-camera (>=3) scale-consistency, lens-distortion, and robust-triangulation tests."""
import numpy as np
import cv2
import pitchcap.constants as C
from pitchcap.constants import Intrinsics
from pitchcap import reconstruct

K = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], float)

# Camera extrinsics (world->cam) as (rvec, tvec):
#   cam0 at world origin; cam1 center (1,0,0) -> baseline 1.0; cam2 center (0,2,0) -> baseline 2.0
_EXTR = [
    (np.zeros(3), np.zeros(3)),
    (np.zeros(3), np.array([-1.0, 0.0, 0.0])),
    (np.zeros(3), np.array([0.0, -2.0, 0.0])),
]


def _scene(T=10, seed=0):
    """Static cloud of 17 well-spread 3D points in front of all cameras, repeated over T frames."""
    rng = np.random.RandomState(seed)
    base = rng.uniform([-1, -1, 4], [1, 1, 6], size=(C.N_KEYPOINTS, 3))
    return np.repeat(base[None], T, axis=0)  # (T,17,3)


def _project(X3, rvec, tvec, dist):
    """X3: (N,3) world -> (N,2) pixels via cv2 (applies distortion)."""
    img, _ = cv2.projectPoints(X3.reshape(-1, 1, 3), rvec, tvec, K, dist)
    return img.reshape(-1, 2)


def _views(X, dist, n_cams):
    T = X.shape[0]
    views = []
    for rvec, tvec in _EXTR[:n_cams]:
        kp = np.zeros((T, C.N_KEYPOINTS, 2))
        for f in range(T):
            kp[f] = _project(X[f], rvec, tvec, dist)
        views.append(kp)
    return views


def test_three_camera_scale_consistency():
    """3 cams with DIFFERENT true baselines (1.0, 2.0) must reconstruct consistent metric 3D.

    The bug being fixed: cv2.recoverPose returns unit-norm translation per pair, so cam2
    (true baseline 2.0) would be reconstructed at half-scale and corrupt the joint triangulation.
    """
    X = _scene(seed=0)
    dist = np.zeros(5)
    intr = Intrinsics(K, dist, (1280, 720))
    views = _views(X, dist, 3)
    conf = [np.ones((X.shape[0], C.N_KEYPOINTS)) for _ in range(3)]

    kp3d, err = reconstruct.reconstruct_multiview(views, conf, [intr, intr, intr])

    # cam1 baseline 1.0 sets the metric scale, so reconstruction should equal true world coords
    assert np.allclose(kp3d, X, atol=1e-2)
    assert err < 1.0


def test_distortion_is_applied():
    """With real radial distortion, reconstruction is only correct if dist is undistorted first."""
    X = _scene(seed=1)
    dist = np.array([0.2, -0.05, 0.0, 0.0, 0.0])
    intr = Intrinsics(K, dist, (1280, 720))
    views = _views(X, dist, 2)
    conf = [np.ones((X.shape[0], C.N_KEYPOINTS)) for _ in range(2)]

    kp3d, err = reconstruct.reconstruct_multiview(views, conf, [intr, intr])

    assert np.allclose(kp3d, X, atol=5e-2)
    assert err < 1.0


def test_robust_triangulation_rejects_outlier():
    """A gross 2D outlier in one of three views must not corrupt that joint's 3D."""
    X = _scene(seed=2)
    dist = np.zeros(5)
    intr = Intrinsics(K, dist, (1280, 720))
    views = _views(X, dist, 3)
    j_bad = C.R_WRIST
    views[2][:, j_bad] += np.array([120.0, -90.0])  # gross pixel outlier in view 2
    conf = [np.ones((X.shape[0], C.N_KEYPOINTS)) for _ in range(3)]

    kp3d, err = reconstruct.reconstruct_multiview(views, conf, [intr, intr, intr])

    # robust triangulation should drop the bad view for that joint and still recover truth
    assert np.allclose(kp3d[:, j_bad], X[:, j_bad], atol=5e-2)
