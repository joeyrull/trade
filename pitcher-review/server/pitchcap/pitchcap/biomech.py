"""Segment vectors -> angular velocity -> kinematic sequence."""
import numpy as np
from . import constants as C


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return v / n


def segment_unit_vectors(kp3d, handedness="R"):
    """Return dict of (T,3) unit vectors for pelvis, trunk, shoulder, elbow.

    ``shoulder`` is the upper-arm vector (throwing shoulder->elbow); its
    orientation in the lab frame changes only via motion at the shoulder
    joint, so its angular velocity is a reasonable proxy for shoulder-segment
    rotation speed. ``elbow`` is the forearm vector (throwing elbow->wrist);
    it inherits the upper arm's own motion too (it's not an isolated local
    joint angle), but during the acceleration/release window elbow extension
    dominates that signal, consistent with how ``trunk`` already isn't a pure
    trunk-only signal either (pelvis translation moves it too) — same
    lab-frame-vector approximation throughout."""
    sh = C.R_SHOULDER if handedness == "R" else C.L_SHOULDER
    el = C.R_ELBOW if handedness == "R" else C.L_ELBOW
    wr = C.R_WRIST if handedness == "R" else C.L_WRIST
    hip_mid = (kp3d[:, C.L_HIP] + kp3d[:, C.R_HIP]) / 2.0
    sh_mid = (kp3d[:, C.L_SHOULDER] + kp3d[:, C.R_SHOULDER]) / 2.0
    return {
        "pelvis": _unit(kp3d[:, C.L_HIP] - kp3d[:, C.R_HIP]),
        "trunk": _unit(sh_mid - hip_mid),
        "shoulder": _unit(kp3d[:, el] - kp3d[:, sh]),
        "elbow": _unit(kp3d[:, wr] - kp3d[:, el]),
    }


def angular_velocity_degps(unit_vecs, fps):
    """|u x du/dt| in deg/s for a (T,3) sequence of unit vectors."""
    u = np.asarray(unit_vecs, dtype=np.float64)
    dudt = np.gradient(u, axis=0) * fps          # central difference
    omega = np.cross(u, dudt)                     # (T,3) rad/s
    return np.degrees(np.linalg.norm(omega, axis=1))


from scipy.signal import find_peaks

SEGMENTS = ["pelvis", "trunk", "shoulder", "elbow"]


def _peak(series, fps):
    """Return (peak_value, peak_time_s, valid).

    NaN-aware: a series with no finite values is a dead segment (joint
    occluded/low-confidence on every frame) and returns the invalid
    sentinel ``(0.0, inf, False)`` so callers can sort it last and warn.
    Otherwise the peak is computed over the finite samples (NaNs replaced
    with -inf for ``find_peaks``) and ``valid`` is True.
    """
    series = np.asarray(series, dtype=np.float64)
    finite = np.isfinite(series)
    if not finite.any():
        return 0.0, float("inf"), False
    safe = np.where(finite, series, -np.inf)
    peaks, _ = find_peaks(safe)
    idx = peaks[np.argmax(safe[peaks])] if len(peaks) else int(np.nanargmax(safe))
    return float(series[idx]), idx / fps, True


def compute_kinematic_sequence(kp3d, fps, handedness="R", cutoffs=None):
    """Pelvis/trunk/shoulder/elbow angular velocity -> peaks -> sequence order + lags.

    cutoffs: optional ``{segment: cutoff_hz}`` (e.g. ``filtering.DEFAULT_CUTOFFS``
    = 13Hz pelvis/trunk, 18Hz shoulder/elbow per the design spec). When given,
    each segment's source joints are zero-lag Butterworth filtered at that
    segment's own cutoff before differentiation — the arm-whip segments are
    the sharpest signal so they keep a wider band, while the noisier rotation
    angles are filtered harder. A joint shared across segments (e.g. the
    throwing shoulder, used by both trunk and the shoulder segment) is
    filtered independently for each, so no segment is constrained by
    another's cutoff. When None, ``kp3d`` is used as-is and the caller owns
    any filtering (backward-compatible default)."""
    if cutoffs:
        from .filtering import filter_keypoints
        cache = {c: filter_keypoints(kp3d, fps, cutoff_hz=c) for c in set(cutoffs.values())}
        vecs = {name: segment_unit_vectors(cache[cutoffs[name]], handedness=handedness)[name]
                for name in SEGMENTS}
    else:
        vecs = segment_unit_vectors(kp3d, handedness=handedness)
    segments, sort_times = {}, {}
    warnings = []
    for name in SEGMENTS:
        series = angular_velocity_degps(vecs[name], fps)
        pv, pt, valid = _peak(series, fps)
        segments[name] = {
            # JSON-safe: a dead (occluded-all-clip) segment's series is all NaN,
            # which json.dump would emit as the invalid ``NaN`` token and break
            # every JSON.parse consumer (the Node server, the browser). Emit null
            # for any non-finite sample instead.
            "series_degps": [None if not np.isfinite(v) else float(v) for v in series],
            "peak_degps": pv,
            # invalid segments report None (inf is only used as a sort key)
            "peak_time_s": pt if valid else None,
        }
        # invalid -> inf, so dead segments sort LAST in sequence_order
        sort_times[name] = pt
        if not valid:
            warnings.append(
                f"{name}: no valid 3D (occluded/low-confidence all frames)")
    order = sorted(SEGMENTS, key=lambda n: sort_times[n])

    def _lag(a, b):
        # None (not the inf sort sentinel) when either segment never peaked —
        # round(inf*1000) is float('inf'), which json.dump emits as the invalid
        # ``Infinity`` token and breaks JSON.parse downstream.
        if not (np.isfinite(sort_times[a]) and np.isfinite(sort_times[b])):
            return None
        return round((sort_times[a] - sort_times[b]) * 1000, 1)
    lags = {
        "pelvis_to_trunk": _lag("trunk", "pelvis"),
        "trunk_to_shoulder": _lag("shoulder", "trunk"),
        "shoulder_to_elbow": _lag("elbow", "shoulder"),
    }
    return {
        "fps": fps,
        "handedness": handedness,
        "segments": segments,
        "sequence_order": order,
        "inter_peak_lags_ms": lags,
        "segment_warnings": warnings,
    }
