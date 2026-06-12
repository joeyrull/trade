"""Markerless extrinsic calibration from body-joint correspondences."""
import numpy as np
import cv2
from .constants import Extrinsics, Intrinsics
from .geometry import projection_matrix


def recover_relative_pose(pts0, pts1, intr: Intrinsics):
    """Essential-matrix relative pose between two views.

    pts0, pts1: matched (N,2) pixel correspondences. Returns (Extrinsics, reproj_err_px).
    First camera is the world origin (R=I, t=0); returned Extrinsics is the second camera.
    """
    pts0 = np.asarray(pts0, np.float64)
    pts1 = np.asarray(pts1, np.float64)
    E, mask = cv2.findEssentialMat(pts0, pts1, intr.K, method=cv2.RANSAC,
                                   prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, pts0, pts1, intr.K, mask=mask)
    extr = Extrinsics(R=R, t=t.reshape(3, 1))

    err = _reprojection_error(pts0, pts1, intr, extr, mask)
    return extr, err


def _reprojection_error(pts0, pts1, intr, extr, mask):
    cam0 = Extrinsics(np.eye(3), np.zeros((3, 1)))
    P0 = projection_matrix(intr, cam0)
    P1 = projection_matrix(intr, extr)
    inl = mask.ravel().astype(bool)
    if inl.sum() == 0:
        return float("inf")
    X = cv2.triangulatePoints(P0, P1, pts0[inl].T, pts1[inl].T)
    X = (X[:3] / X[3]).T
    errs = []
    for P, pts in ((P0, pts0[inl]), (P1, pts1[inl])):
        proj = (P @ np.hstack([X, np.ones((len(X), 1))]).T).T
        proj = proj[:, :2] / proj[:, 2:]
        errs.append(np.linalg.norm(proj - pts, axis=1))
    return float(np.mean(np.concatenate(errs)))


def correspondences_from_pose(kp2d_a, conf_a, kp2d_b, conf_b, conf_thresh=0.5):
    """Flatten matched, high-confidence joints across all frames into (N,2),(N,2)."""
    mask = (conf_a >= conf_thresh) & (conf_b >= conf_thresh)
    return kp2d_a[mask], kp2d_b[mask]


def recover_pose_normalized(norm0, normv):
    """Relative pose from NORMALIZED correspondences (camera matrix = identity).

    norm0, normv: matched (N,2) normalized image coords. Returns (R, t_unit, inlier_mask)
    where t_unit is unit-norm (per-pair scale is arbitrary) and the pose maps camera-0
    coords into camera-v coords (x_v = R x_0 + t).
    """
    norm0 = np.asarray(norm0, np.float64)
    normv = np.asarray(normv, np.float64)
    I = np.eye(3)
    E, mask = cv2.findEssentialMat(norm0, normv, cameraMatrix=I, method=cv2.RANSAC,
                                   prob=0.999, threshold=1e-3)
    _, R, t, mask = cv2.recoverPose(E, norm0, normv, cameraMatrix=I, mask=mask)
    return R, t.reshape(3, 1), mask
