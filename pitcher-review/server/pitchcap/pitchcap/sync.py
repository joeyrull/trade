"""Audio cross-correlation -> per-clip integer frame offsets."""
import numpy as np
from scipy.signal import correlate


def _lag_samples(ref, other):
    """Positive lag => `other` is delayed relative to `ref`."""
    c = correlate(other, ref, mode="full")
    lag = np.argmax(c) - (len(ref) - 1)
    return lag


def frame_offsets_from_audio(audios, fps, sr):
    """Return integer frame offsets, first clip as reference (offset 0)."""
    ref = np.asarray(audios[0], dtype=np.float64)
    offsets = [0]
    for a in audios[1:]:
        lag = _lag_samples(ref, np.asarray(a, dtype=np.float64))
        offsets.append(int(round(lag * fps / sr)))
    return offsets
