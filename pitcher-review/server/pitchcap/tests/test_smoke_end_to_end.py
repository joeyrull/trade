import json
import numpy as np
from pitchcap.constants import Intrinsics, Extrinsics
import pitchcap.constants as C
from pitchcap.geometry import projection_matrix
from pitchcap import reconstruct
from pitchcap.filtering import filter_keypoints
from pitchcap.biomech import compute_kinematic_sequence
from pitchcap.plot import plot_sequence


def test_multiview_pipeline_synthetic(tmp_path):
    """Full math path: known 3D -> project to 2 views -> reconstruct -> sequence."""
    fps, T = 240, 200
    K = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], float)
    intr = Intrinsics(K, np.zeros(5), (1280, 720))
    cam0 = Extrinsics(np.eye(3), np.zeros((3, 1)))
    cam1 = Extrinsics(np.eye(3), np.array([[-1.0], [0], [0]]))
    P0 = projection_matrix(intr, cam0)
    P1 = projection_matrix(intr, cam1)

    # build a moving 3D skeleton (static base + arm rotation burst)
    t = np.arange(T) / fps
    X = np.zeros((T, C.N_KEYPOINTS, 3))
    X[:, :, 2] = 4.0  # default depth for all joints (avoids degenerate Z=0 projections)
    X[:, C.R_HIP] = [0, 0, 4]; X[:, C.L_HIP] = [0.3, 0, 4]
    X[:, C.R_SHOULDER] = [0, 0.5, 4]; X[:, C.L_SHOULDER] = [0.3, 0.5, 4]
    burst = 0.4 * np.exp(-((t - 0.5) ** 2) / (2 * 0.03 ** 2))
    X[:, C.R_ELBOW, 0] = burst; X[:, C.R_ELBOW, 1] = 0.3; X[:, C.R_ELBOW, 2] = 4

    def project(P):
        out = np.zeros((T, C.N_KEYPOINTS, 2))
        for j in range(C.N_KEYPOINTS):
            h = (P @ np.hstack([X[:, j], np.ones((T, 1))]).T).T
            out[:, j] = h[:, :2] / h[:, 2:]
        return out

    kp2d_views = [project(P0), project(P1)]
    conf_views = [np.ones((T, C.N_KEYPOINTS)), np.ones((T, C.N_KEYPOINTS))]

    kp3d, err = reconstruct.reconstruct_multiview(kp2d_views, conf_views, [intr, intr])
    kp3d = filter_keypoints(kp3d, fps, cutoff_hz=15)
    result = compute_kinematic_sequence(kp3d, fps, handedness="R")

    assert set(result["segments"]) == {"pelvis", "trunk", "arm"}
    assert result["segments"]["arm"]["peak_degps"] > 0
    assert err < 5.0

    out_json = tmp_path / "result.json"
    out_png = tmp_path / "plot.png"
    result.update({"n_cams": 2, "mode": "multiview",
                   "reprojection_error_px": err, "warnings": []})
    with open(out_json, "w") as f:
        json.dump(result, f)
    plot_sequence(result, str(out_png))
    assert out_json.exists() and out_png.exists()
    assert json.load(open(out_json))["mode"] == "multiview"
