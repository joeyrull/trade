"""Zero-phase Butterworth low-pass filtering."""
import numpy as np
from scipy.signal import butter, filtfilt


def _safe_wn(cutoff_hz, fps):
    """Normalized cutoff clamped to (0, 1) so butter() never rejects it.

    Low-fps footage (e.g. 30fps with a 15Hz cutoff) would otherwise put the cutoff at or
    above Nyquist. We clamp to 0.99 (near-passthrough) rather than crash.
    """
    nyq = fps / 2.0
    return min(max(cutoff_hz / nyq, 1e-4), 0.99)


def butter_lowpass(signal, fps, cutoff_hz, order=4):
    """Zero-lag low-pass along axis 0. Works on (T,) or (T, ...) arrays."""
    b, a = butter(order, _safe_wn(cutoff_hz, fps), btype="low")
    return filtfilt(b, a, signal, axis=0)


# segment-specific cutoffs (Hz) per spec
DEFAULT_CUTOFFS = {"pelvis": 13.0, "trunk": 13.0, "shoulder": 18.0, "elbow": 18.0}


def interpolate_nans(arr):
    """Linearly interpolate INTERNAL NaNs along axis 0, column-wise.

    Operates on a (T, ...) array. For each flattened column, finite samples
    are kept and internal gaps are filled via ``np.interp`` over finite
    indices (leading/trailing NaNs become the nearest finite value). Columns
    that are entirely NaN are left untouched (no data is fabricated).
    """
    a = np.asarray(arr, dtype=np.float64).copy()
    T = a.shape[0]
    flat = a.reshape(T, -1)
    idx = np.arange(T)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        finite = np.isfinite(col)
        if finite.all() or not finite.any():
            # nothing to do (fully finite) or all-NaN (leave as NaN)
            continue
        col[~finite] = np.interp(idx[~finite], idx[finite], col[finite])
    return flat.reshape(a.shape)


def filter_keypoints(kp3d, fps, cutoff_hz=15.0, order=4):
    """Filter every coordinate channel of (T, J, 3) keypoints, NaN-safe.

    - Internal NaNs are linearly interpolated first.
    - Fully-NaN columns (joints never seen by >=2 views) stay NaN.
    - Short clips (T <= filtfilt padlen) skip filtering and return the
      interpolated array unchanged instead of crashing.
    """
    arr = np.asarray(kp3d, dtype=np.float64)
    interp = interpolate_nans(arr)

    shape = interp.shape
    T = shape[0]
    flat = interp.reshape(T, -1)

    b, a = butter(order, _safe_wn(cutoff_hz, fps), btype="low")
    padlen = 3 * max(len(b), len(a))
    if T <= padlen:
        # too few frames for zero-phase padding; return interpolated as-is
        return flat.reshape(shape)

    out = flat.copy()
    finite_cols = np.isfinite(flat).all(axis=0)
    if finite_cols.any():
        out[:, finite_cols] = filtfilt(b, a, flat[:, finite_cols], axis=0)
    # fully-NaN columns are left as NaN in `out`
    return out.reshape(shape)
