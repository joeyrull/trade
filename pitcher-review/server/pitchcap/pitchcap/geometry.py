"""Camera projection helpers."""
import numpy as np
import cv2
from .constants import Intrinsics, Extrinsics


def projection_matrix(intr: Intrinsics, extr: Extrinsics) -> np.ndarray:
    """P = K [R | t], shape (3, 4)."""
    Rt = np.hstack([extr.R, extr.t.reshape(3, 1)])
    return intr.K @ Rt


def undistort_points(pts: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """pts: (N, 2) pixel coords -> (N, 2) undistorted pixel coords (re-projected through K)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, intr.K, intr.dist, P=intr.K)
    return out.reshape(-1, 2)


def normalize_points(pts: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """pts: (N, 2) pixel coords -> (N, 2) normalized image coords (K=I), distortion removed.

    Applies this view's own intrinsics + distortion, so different-lens cameras and lens
    distortion are both handled before essential-matrix / triangulation math.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, intr.K, intr.dist, P=None)
    return out.reshape(-1, 2)
