"""Confidence-weighted multi-view DLT triangulation."""
import numpy as np
from . import constants as C


def triangulate_point(proj_mats, points2d, weights):
    """Weighted linear DLT. proj_mats: list of (3,4); points2d: (V,2); weights: (V,)."""
    rows = []
    for P, (x, y), w in zip(proj_mats, points2d, weights):
        rows.append(w * (x * P[2] - P[0]))
        rows.append(w * (y * P[2] - P[1]))
    A = np.vstack(rows)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def _reproj_residuals(proj_mats, pts, X):
    res = np.empty(len(proj_mats))
    Xh = np.append(X, 1.0)
    for i, (P, p) in enumerate(zip(proj_mats, pts)):
        xh = P @ Xh
        res[i] = np.inf if abs(xh[2]) < 1e-9 else np.linalg.norm(xh[:2] / xh[2] - p)
    return res


def triangulate_point_robust(proj_mats, points2d, weights, inlier_tol=0.01):
    """Minimal-subset RANSAC triangulation (needs >=3 views to reject anything).

    Triangulates from every view PAIR, scores each hypothesis by how many views it reprojects
    within ``inlier_tol`` (normalized-coord units), and re-triangulates from the largest inlier
    set. A single outlier view can't drag the result because at least one clean pair exists.
    Falls back to plain DLT for <3 views.
    """
    pts = np.asarray(points2d, float)
    ws = np.asarray(weights, float)
    n = len(proj_mats)
    if n < 3:
        return triangulate_point(proj_mats, pts, ws)
    best_inliers, best_score, best_res = None, -1, np.inf
    for i in range(n):
        for k in range(i + 1, n):
            Xh = triangulate_point([proj_mats[i], proj_mats[k]], pts[[i, k]], ws[[i, k]])
            res = _reproj_residuals(proj_mats, pts, Xh)
            inl = res <= inlier_tol
            score, total = int(inl.sum()), float(res[inl].sum())
            if score > best_score or (score == best_score and total < best_res):
                best_inliers, best_score, best_res = inl, score, total
    if best_inliers is not None and best_inliers.sum() >= 2:
        Ps = [P for P, keep in zip(proj_mats, best_inliers) if keep]
        return triangulate_point(Ps, pts[best_inliers], ws[best_inliers])
    return triangulate_point(proj_mats, pts, ws)


def triangulate_sequence(kp2d_views, conf_views, proj_mats, conf_thresh=0.3, robust=False):
    """Triangulate every keypoint across views per frame.

    kp2d_views: list of (T,17,2); conf_views: list of (T,17); proj_mats: list of (3,4).
    Joints with <2 views above conf_thresh are left as NaN (gap-filled later).
    robust=True enables per-joint cross-view outlier rejection (only acts with >=3 views).
    """
    T = kp2d_views[0].shape[0]
    out = np.full((T, C.N_KEYPOINTS, 3), np.nan)
    for f in range(T):
        for j in range(C.N_KEYPOINTS):
            Ps, pts, ws = [], [], []
            for P, kp, conf in zip(proj_mats, kp2d_views, conf_views):
                c = conf[f, j]
                if c >= conf_thresh:
                    Ps.append(P); pts.append(kp[f, j]); ws.append(c)
            if len(Ps) >= 2:
                if robust:
                    out[f, j] = triangulate_point_robust(Ps, np.array(pts), np.array(ws))
                else:
                    out[f, j] = triangulate_point(Ps, np.array(pts), np.array(ws))
    return out


def fill_gaps(kp3d):
    """Linearly interpolate short NaN runs along time, per joint/coord."""
    kp3d = kp3d.copy()
    T = kp3d.shape[0]
    idx = np.arange(T)
    for j in range(kp3d.shape[1]):
        for c in range(3):
            col = kp3d[:, j, c]
            good = ~np.isnan(col)
            if good.sum() >= 2:
                kp3d[:, j, c] = np.interp(idx, idx[good], col[good])
    return kp3d
