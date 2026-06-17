import numpy as np
from pitchcap.sync import frame_offsets_from_audio


def test_recovers_known_shift():
    rng = np.random.RandomState(1)
    sr, fps = 48000, 240
    base = rng.randn(48000)              # 1s of audio
    shift_frames = 12
    shift_samples = int(shift_frames * sr / fps)
    shifted = np.concatenate([np.zeros(shift_samples), base])[:len(base)]
    offsets = frame_offsets_from_audio([base, shifted], fps=fps, sr=sr)
    # first clip is reference (0); second lags by shift_frames
    assert offsets[0] == 0
    assert abs(offsets[1] - shift_frames) <= 1
