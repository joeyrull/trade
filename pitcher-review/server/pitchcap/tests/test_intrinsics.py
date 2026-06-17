import numpy as np
from pitchcap.intrinsics import save_profile, load_profile, approximate_intrinsics


def test_save_load_roundtrip(tmp_path):
    K = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], float)
    dist = np.array([0.01, -0.02, 0, 0, 0])
    p = tmp_path / "iphone12_wide_240.json"
    save_profile(str(p), K, dist, (1280, 720))
    intr = load_profile(str(p))
    assert np.allclose(intr.K, K)
    assert np.allclose(intr.dist, dist)
    assert intr.image_size == (1280, 720)


def test_approximate_intrinsics_has_reasonable_focal():
    intr = approximate_intrinsics((1920, 1080))
    # focal near image width, principal point at center
    assert 0.7 * 1920 < intr.K[0, 0] < 1.3 * 1920
    assert np.allclose(intr.K[0, 2], 960)
    assert np.allclose(intr.K[1, 2], 540)
