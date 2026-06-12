import numpy as np
from pitchcap.constants import Intrinsics, Extrinsics
from pitchcap.geometry import projection_matrix
from pitchcap.triangulate import triangulate_point, triangulate_sequence


def _rig():
    K = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], float)
    dist = np.zeros(5)
    intr = Intrinsics(K, dist, (1280, 720))
    cam0 = Extrinsics(np.eye(3), np.zeros((3, 1)))
    # second camera translated 1m along x, looking same direction
    cam1 = Extrinsics(np.eye(3), np.array([[-1.0], [0], [0]]))
    P0 = projection_matrix(intr, cam0)
    P1 = projection_matrix(intr, cam1)
    return P0, P1


def _project(P, X):
    x = P @ np.append(X, 1.0)
    return x[:2] / x[2]


def test_triangulate_single_point():
    P0, P1 = _rig()
    X = np.array([0.2, -0.1, 4.0])
    pts = np.array([_project(P0, X), _project(P1, X)])
    weights = np.array([1.0, 1.0])
    Xhat = triangulate_point([P0, P1], pts, weights)
    assert np.allclose(Xhat, X, atol=1e-3)


def test_triangulate_sequence_shape():
    P0, P1 = _rig()
    T = 10
    kp2d_v0 = np.zeros((T, 17, 2)); kp2d_v1 = np.zeros((T, 17, 2))
    conf = np.ones((T, 17))
    out = triangulate_sequence([kp2d_v0, kp2d_v1], [conf, conf], [P0, P1])
    assert out.shape == (T, 17, 3)
