"""PitchCap CLI: 1-N videos -> result.json + plot.png."""
import argparse
import json
import os
import numpy as np
from . import io_video, pose2d, monocular, reconstruct
from .sync import frame_offsets_from_audio
from .filtering import filter_keypoints
from .biomech import compute_kinematic_sequence
from .intrinsics import load_profile, approximate_intrinsics
from .plot import plot_sequence


def _align(arrs, offsets):
    """Trim each (T,...) array by its frame offset to a common length."""
    starts = [max(0, o) for o in offsets]
    trimmed = [a[s:] for a, s in zip(arrs, starts)]
    n = min(len(a) for a in trimmed)
    return [a[:n] for a in trimmed]


def run(video_paths, profile_paths=None, handedness="R",
        out_json="result.json", out_png="plot.png", cutoff_hz=15.0):
    missing = [p for p in video_paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit(
            "PitchCap: video file(s) not found: " + ", ".join(missing) +
            "\n  Run from the folder containing the clips, or pass full paths. "
            "Check the names with `ls`.")

    warnings = []
    clips = [io_video.load_clip(p) for p in video_paths]
    empty = [p for p, c in zip(video_paths, clips) if len(c.frames) == 0]
    if empty:
        raise SystemExit(
            "PitchCap: could not decode any frames from: " + ", ".join(empty) +
            "\n  Is it a valid video file? .mov/.mp4 are expected.")
    fps = clips[0].fps
    if fps < 120:
        warnings.append(
            f"footage is ~{fps:.0f} fps, not high-speed (240fps recommended). "
            "Fast-motion peaks (arm) are undersampled; record in 240fps slo-mo for reliable "
            "kinematic-sequence numbers.")

    # 2D pose per view
    poses = [pose2d.estimate_pose_2d(c.frames) for c in clips]
    kp2d_views = [p[0] for p in poses]
    conf_views = [p[1] for p in poses]

    if len(clips) >= 2:
        # sync via audio (fallback to zero offsets if no audio)
        if all(len(c.audio) for c in clips):
            offsets = frame_offsets_from_audio(
                [c.audio for c in clips], fps=fps, sr=clips[0].sr)
        else:
            offsets = [0] * len(clips)
            warnings.append("audio missing on >=1 clip; assuming zero offset")
        kp2d_views = _align(kp2d_views, offsets)
        conf_views = _align(conf_views, offsets)

        # intrinsics: load profile or approximate
        intrinsics = []
        for i, c in enumerate(clips):
            prof = profile_paths[i] if profile_paths and i < len(profile_paths) else None
            if prof:
                intrinsics.append(load_profile(prof))
            else:
                intrinsics.append(approximate_intrinsics(c.image_size))
                warnings.append(f"camera_{i} using approximate intrinsics (no lens profile)")

        kp3d, reproj_err = reconstruct.reconstruct_multiview(
            kp2d_views, conf_views, intrinsics)
        mode, n_cams = "multiview", len(clips)
    else:
        kp3d = monocular.estimate_pose_3d_monocular(clips[0].frames, fps=fps)
        kp3d = np.nan_to_num(kp3d)
        reproj_err, mode, n_cams = None, "monocular", 1
        warnings.append("single camera: monocular 3D (reduced accuracy)")

    kp3d = filter_keypoints(kp3d, fps, cutoff_hz=cutoff_hz)
    result = compute_kinematic_sequence(kp3d, fps, handedness=handedness)
    # surface dead joints (occluded/low-confidence segments) to the CLI/JSON
    warnings.extend(result.get("segment_warnings", []))
    result.update({"n_cams": n_cams, "mode": mode,
                   "reprojection_error_px": reproj_err, "warnings": warnings})
    result["fps"] = int(round(fps))

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    plot_sequence(result, out_png)
    return result


def main():
    ap = argparse.ArgumentParser(description="PitchCap kinematic-sequence extractor")
    ap.add_argument("videos", nargs="+", help="1-4 clip paths")
    ap.add_argument("--profiles", nargs="*", default=None,
                    help="lens profile JSON per video (same order)")
    ap.add_argument("--handedness", choices=["R", "L"], default="R")
    ap.add_argument("--out-json", default="result.json")
    ap.add_argument("--out-png", default="plot.png")
    ap.add_argument("--cutoff-hz", type=float, default=15.0)
    a = ap.parse_args()
    res = run(a.videos, a.profiles, a.handedness, a.out_json, a.out_png, a.cutoff_hz)
    print(f"mode={res['mode']} order={res['sequence_order']} "
          f"reproj_err={res['reprojection_error_px']}")


if __name__ == "__main__":
    main()
