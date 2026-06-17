"""2D pose estimation. Default backend: RTMPose via rtmlib (ONNX)."""
import numpy as np
from . import constants as C


class RTMPoseBackend:
    def __init__(self, device="cpu", mode="balanced"):
        from rtmlib import Body  # lazy import; heavy dependency
        self.model = Body(mode=mode, backend="onnxruntime", device=device)

    def __call__(self, frames):
        """frames: list/array of HxWx3 BGR -> (T,17,2), (T,17)."""
        T = len(frames)
        kp = np.zeros((T, C.N_KEYPOINTS, 2))
        conf = np.zeros((T, C.N_KEYPOINTS))
        for i, frame in enumerate(frames):
            keypoints, scores = self.model(frame)
            if len(keypoints) == 0:
                continue
            # take the most confident detected person
            best = int(np.argmax(scores.mean(axis=1)))
            kp[i] = keypoints[best][:C.N_KEYPOINTS]
            conf[i] = scores[best][:C.N_KEYPOINTS]
        return kp, conf


def estimate_pose_2d(frames, backend=None):
    """Return (kp2d (T,17,2), conf (T,17)) for one view."""
    backend = backend or RTMPoseBackend()
    return backend(frames)
