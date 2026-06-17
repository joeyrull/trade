import numpy as np
import cv2
from pitchcap.constants import Intrinsics
from pitchcap.calibrate_extrinsic import recover_relative_pose


def test_recovers_rotation_up_to_scale():
    rng = np.random.RandomState(2)
    K = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], float)
    intr = Intrinsics(K, np.zeros(5), (1280, 720))

    # ground-truth second camera: small rotation + translation along x
    rvec = np.array([0.0, 0.3, 0.0])
    R_gt, _ = cv2.Rodrigues(rvec)
    t_gt = np.array([[-1.0], [0.0], [0.0]])

    X = rng.uniform([-1, -1, 3], [1, 1, 6], size=(200, 3))
    p0 = (K @ X.T).T; p0 = p0[:, :2] / p0[:, 2:]
    Xc = (R_gt @ X.T + t_gt).T
    p1 = (K @ Xc.T).T; p1 = p1[:, :2] / p1[:, 2:]

    extr, err = recover_relative_pose(p0, p1, intr)
    # rotation should match within a degree
    ang = np.degrees(np.arccos((np.trace(extr.R.T @ R_gt) - 1) / 2))
    assert ang < 1.0
    assert err < 1.0   # px reprojection
