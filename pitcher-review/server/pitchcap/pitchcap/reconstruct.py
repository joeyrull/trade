"""Choose multi-view vs monocular and return 3D keypoints.

Multi-view path works in NORMALIZED image coordinates (each view undistorted by its own
intrinsics), recovers per-pair relative pose against view 0, resolves a single consistent
global scale across cameras (so 3-4 cameras don't each carry an arbitrary per-pair scale),
then triangulates. Reprojection error is reported in pixels.
"""
import numpy as np
from . import constants as C
from .geometry import normalize_points
from .calibrate_extrinsic import recover_pose_normalized
from .triangulate import triangulate_sequence, triangulate_point, fill_gaps


def _normalize_view(kp2d, intr):
    """(T,J,2) pixel -> (T,J,2) normalized coords via this view's own K and distortion."""
    T, J = kp2d.shape[:2]
    norm = normalize_points(kp2d.reshape(-1, 2), intr)
    return norm.reshape(T, J, 2)


def _shared(maskA, maskB, *more):
    m = maskA & maskB
    for x in more:
        m = m & x
    return m


def _resolve_scale(norm_views, conf_views, R, t, v, thresh):
    """Global scale for camera v so its (0,v) reconstruction matches the (0,1) scale.

    Camera 1 defines the global scale (s=1). For camera v>=2, triangulate joints visible in
    views 0,1,v two ways — via pair (0,1) [reference] and via pair (0,v) [unit baseline] —
    and take the median ratio of camera-0 depths. Both share view-0's rays, so the depth
    ratio is exactly the scale factor needed on t_v.
    """
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = np.hstack([R[1], t[1]])
    Pv = np.hstack([R[v], t[v]])  # unit-baseline pose for camera v
    m = _shared(conf_views[0] >= thresh, conf_views[1] >= thresh, conf_views[v] >= thresh)
    n0, n1, nv = norm_views[0][m], norm_views[1][m], norm_views[v][m]
    ratios = []
    for i in range(len(n0)):
        Xref = triangulate_point([P0, P1], [n0[i], n1[i]], [1.0, 1.0])
        Xv = triangulate_point([P0, Pv], [n0[i], nv[i]], [1.0, 1.0])
        if Xref[2] > 1e-6 and Xv[2] > 1e-6:
            ratios.append(Xref[2] / Xv[2])
    return float(np.median(ratios)) if ratios else 1.0


def _reprojection_error_px(norm_views, conf_views, proj, intrinsics, kp3d, thresh):
    """Mean reprojection error in PIXELS over confident, reconstructed joints."""
    T = kp3d.shape[0]
    errs = []
    for v, (P, intr) in enumerate(zip(proj, intrinsics)):
        K = intr.K
        for f in range(T):
            for j in range(C.N_KEYPOINTS):
                if conf_views[v][f, j] < thresh:
                    continue
                X = kp3d[f, j]
                if not np.all(np.isfinite(X)):
                    continue
                xh = P @ np.append(X, 1.0)
                if abs(xh[2]) < 1e-9:
                    continue
                pred = K[:2, :2] @ (xh[:2] / xh[2]) + K[:2, 2]
                obs = K[:2, :2] @ norm_views[v][f, j] + K[:2, 2]
                errs.append(np.linalg.norm(pred - obs))
    return float(np.mean(errs)) if errs else 0.0


def reconstruct_multiview(kp2d_views, conf_views, intrinsics, conf_thresh=0.5):
    """Triangulated 3D keypoints (T,17,3) + mean reprojection error (px).

    View 0 is the world origin. Each other view's pose is recovered in normalized coords;
    cameras >=3 are rescaled to a single consistent global scale before triangulation.
    """
    V = len(kp2d_views)
    norm_views = [_normalize_view(kp2d_views[v], intrinsics[v]) for v in range(V)]

    R = [np.eye(3)]
    t = [np.zeros((3, 1))]
    for v in range(1, V):
        m = _shared(conf_views[0] >= conf_thresh, conf_views[v] >= conf_thresh)
        Rv, tv, _ = recover_pose_normalized(norm_views[0][m], norm_views[v][m])
        R.append(Rv)
        t.append(tv)

    for v in range(2, V):  # reconcile per-camera scale (no-op for the 2-camera case)
        t[v] = t[v] * _resolve_scale(norm_views, conf_views, R, t, v, conf_thresh)

    proj = [np.hstack([R[v], t[v]]) for v in range(V)]
    kp3d = triangulate_sequence(norm_views, conf_views, proj,
                                conf_thresh=conf_thresh, robust=(V >= 3))
    kp3d = fill_gaps(kp3d)
    err = _reprojection_error_px(norm_views, conf_views, proj, intrinsics, kp3d, conf_thresh)
    return kp3d, err
