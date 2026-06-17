"""Shared keypoint indices and camera dataclasses."""
from dataclasses import dataclass
import numpy as np

# COCO-17 keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW = 5, 6, 7, 8
L_WRIST, R_WRIST, L_HIP, R_HIP = 9, 10, 11, 12
L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 13, 14, 15, 16
N_KEYPOINTS = 17


@dataclass
class Intrinsics:
    K: np.ndarray          # (3, 3) camera matrix
    dist: np.ndarray       # (5,) distortion coefficients
    image_size: tuple      # (width, height)


@dataclass
class Extrinsics:
    R: np.ndarray          # (3, 3) rotation
    t: np.ndarray          # (3, 1) translation
