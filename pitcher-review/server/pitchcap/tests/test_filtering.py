import numpy as np
from pitchcap.filtering import butter_lowpass, filter_keypoints


def test_lowpass_removes_high_freq_keeps_low():
    fps, T = 240, 480
    t = np.arange(T) / fps
    low = np.sin(2 * np.pi * 2 * t)            # 2 Hz signal
    noise = 0.5 * np.sin(2 * np.pi * 60 * t)   # 60 Hz noise
    out = butter_lowpass(low + noise, fps, cutoff_hz=13, order=4)
    # low-freq preserved, noise crushed; ignore filtfilt edge transients
    err = np.abs(out - low)[20:-20]
    assert err.max() < 0.1


def test_filter_keypoints_shape_preserved():
    fps, T = 240, 100
    kp = np.random.RandomState(0).randn(T, 17, 3)
    out = filter_keypoints(kp, fps, cutoff_hz=13)
    assert out.shape == kp.shape
