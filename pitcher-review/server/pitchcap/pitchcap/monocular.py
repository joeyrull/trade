"""Single-camera 3D fallback using MediaPipe Pose world landmarks.

Uses the MediaPipe **Tasks** API (``mediapipe.tasks`` / ``PoseLandmarker``),
which is what current mediapipe (>= 0.10) ships — the legacy
``mp.solutions.pose`` API this module originally used was removed, so the old
path raised ``AttributeError`` on any modern install.

``estimate_pose_monocular`` returns the world-landmark 3D *and* the
image-normalized 2D + per-landmark visibility in a single pass, so a caller
that needs image-space positions (for drawing or 2D metrics) doesn't have to
re-run the model. ``estimate_pose_3d_monocular`` is kept as a thin
back-compat wrapper returning just the 3D array.
"""
import os
import urllib.request
import numpy as np
from . import constants as C

# MediaPipe BlazePose (33-landmark) index -> COCO-17 index (only joints we use).
# Face keypoints beyond the nose (COCO 1-4) are left NaN: biomech needs none.
_MP_TO_COCO = {
    0:  C.NOSE,
    11: C.L_SHOULDER, 12: C.R_SHOULDER,
    13: C.L_ELBOW, 14: C.R_ELBOW,
    15: C.L_WRIST, 16: C.R_WRIST,
    23: C.L_HIP, 24: C.R_HIP,
    25: C.L_KNEE, 26: C.R_KNEE,
    27: C.L_ANKLE, 28: C.R_ANKLE,
}

_MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
              'pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task')


def _default_model_path():
    return os.path.join(os.path.dirname(__file__), 'models',
                        'pose_landmarker_heavy.task')


def ensure_model(model_path=None):
    """Return a path to the pose model, downloading it on first use if absent."""
    path = model_path or _default_model_path()
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, path)
    return path


def _make_landmarker(model_path):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def estimate_pose_monocular(frames, fps=30.0, model_path=None):
    """Run MediaPipe Pose over ``frames`` (HxWx3 BGR, list or generator).

    Returns a dict:
      - ``kp3d`` (T,17,3) float: world landmarks in meters, body-centered,
        remapped to COCO-17 (unused joints NaN).
      - ``kp2d`` (T,17,2) float: image-normalized [0,1] (x, y) landmarks.
      - ``vis``  (T,17)   float: per-landmark visibility [0,1].

    Iterates ``frames`` exactly once and accumulates per-frame rows, so a
    streaming generator can be passed without holding the whole clip in memory.
    """
    import mediapipe as mp
    import cv2
    model_path = ensure_model(model_path)
    lmk = _make_landmarker(model_path)

    kp3d_rows, kp2d_rows, vis_rows = [], [], []
    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(i * 1000 / max(fps, 1e-6))
        res = lmk.detect_for_video(mp_img, ts_ms)

        kp3d = np.full((C.N_KEYPOINTS, 3), np.nan)
        kp2d = np.full((C.N_KEYPOINTS, 2), np.nan)
        vis = np.zeros(C.N_KEYPOINTS)
        wl = res.pose_world_landmarks[0] if res.pose_world_landmarks else None
        il = res.pose_landmarks[0] if res.pose_landmarks else None
        for mp_idx, coco_idx in _MP_TO_COCO.items():
            if wl is not None:
                w = wl[mp_idx]
                kp3d[coco_idx] = [w.x, w.y, w.z]
            if il is not None:
                p = il[mp_idx]
                kp2d[coco_idx] = [p.x, p.y]
                vis[coco_idx] = getattr(p, 'visibility', 1.0)
        kp3d_rows.append(kp3d)
        kp2d_rows.append(kp2d)
        vis_rows.append(vis)
    lmk.close()

    if not kp3d_rows:
        return {'kp3d': np.zeros((0, C.N_KEYPOINTS, 3)),
                'kp2d': np.zeros((0, C.N_KEYPOINTS, 2)),
                'vis': np.zeros((0, C.N_KEYPOINTS))}
    return {'kp3d': np.stack(kp3d_rows),
            'kp2d': np.stack(kp2d_rows),
            'vis': np.stack(vis_rows)}


def estimate_pose_3d_monocular(frames, fps=30.0, model_path=None):
    """Back-compat: world-landmark 3D only, (T,17,3) in MediaPipe world meters."""
    return estimate_pose_monocular(frames, fps=fps, model_path=model_path)['kp3d']
