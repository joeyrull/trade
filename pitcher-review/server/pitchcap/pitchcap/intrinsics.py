"""One-time per-(phone, lens, mode) lens calibration + profile I/O."""
import json
import numpy as np
import cv2
from .constants import Intrinsics


def save_profile(path, K, dist, image_size):
    with open(path, "w") as f:
        json.dump({
            "K": np.asarray(K).tolist(),
            "dist": np.asarray(dist).ravel().tolist(),
            "image_size": list(image_size),
        }, f, indent=2)


def load_profile(path):
    with open(path) as f:
        d = json.load(f)
    return Intrinsics(
        K=np.array(d["K"], float),
        dist=np.array(d["dist"], float),
        image_size=tuple(d["image_size"]),
    )


def approximate_intrinsics(image_size, fov_factor=1.0):
    """Fallback when no profile exists: focal ~= image width, center at midpoint."""
    w, h = image_size
    f = fov_factor * w
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)
    return Intrinsics(K=K, dist=np.zeros(5), image_size=(w, h))


def calibrate_from_video(video_path, board_size=(9, 6), square_m=0.025, stride=10):
    """Detect a checkerboard across a calibration clip -> Intrinsics.

    board_size = (inner_corners_x, inner_corners_y).
    """
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_m

    objpoints, imgpoints = [], []
    cap = cv2.VideoCapture(video_path)
    size = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            found, corners = cv2.findChessboardCorners(gray, board_size, None)
            if found:
                objpoints.append(objp)
                imgpoints.append(corners)
        i += 1
    cap.release()
    if len(objpoints) < 5:
        raise RuntimeError(f"Only {len(objpoints)} board views found; need >=5.")
    _, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
    return Intrinsics(K=K, dist=dist.ravel()[:5], image_size=size)
