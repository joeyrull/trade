# Per-camera intrinsics calibration profiles

This directory holds one-time intrinsics calibration profiles for the garage
rig's fixed cameras: `cam0.json`, `cam1.json`, `cam2.json` (matching
`video`/`video2`/`video3` upload order — same order as `CAMERAS` in
`capture/record_sync.py`).

Each file is the JSON written by `pitchcap.intrinsics.save_profile`:
`{"K": [[...]], "dist": [...], "image_size": [w, h]}`.

If a `camN.json` is present **and its `image_size` matches that camera's
clip**, `pitchcap_analyze.py`'s multi-view path uses it instead of the
`approximate_intrinsics` fallback (focal ≈ image width), which sharpens the
recovered 3D scale. If absent (or the resolution doesn't match), the analyzer
falls back automatically and notes it in `summary.pitchcap.warnings` — nothing
breaks either way.

Generate these with `capture/calibrate.py` once each camera is mounted (see
`capture/README.md`). Re-run for a camera if it's ever repositioned or
refocused.
