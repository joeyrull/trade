#!/usr/bin/env python3
"""One-time intrinsics calibration for a fixed garage camera.

Print a standard OpenCV checkerboard (default assumes 9x6 *inner* corners —
i.e. a 10x7-square board) and record a short clip with that camera, moving
the board around to cover different parts of the frame and a few tilt angles.
Then run:

  python3 calibrate.py checkerboard_cam0.avi --cam-index 0 --square-mm 25

This writes ../server/scripts/calibration/cam0.json. pitchcap_analyze.py's
multi-view path picks it up automatically for every future analysis from that
camera (matched by image size) — falls back to approximate intrinsics if the
file is missing or the resolution doesn't match. Re-run if a camera is ever
repositioned or refocused.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))             # .../pitcher-review/capture
_ROOT = os.path.dirname(_HERE)                                  # .../pitcher-review
_PKG_ROOT = os.path.join(_ROOT, 'server', 'pitchcap')           # .../pitcher-review/server/pitchcap
_CALIB_DIR = os.path.join(_ROOT, 'server', 'scripts', 'calibration')
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from pitchcap.intrinsics import calibrate_from_video, save_profile  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video', help='checkerboard calibration clip from this camera')
    ap.add_argument('--cam-index', type=int, required=True,
                    help='0/1/2 -- matches video/video2/video3 upload order')
    ap.add_argument('--board-size', default='9x6',
                    help='inner corners WxH, default 9x6 (a 10x7-square printed board)')
    ap.add_argument('--square-mm', type=float, default=25.0, help='checkerboard square size in mm')
    ap.add_argument('--stride', type=int, default=10, help='use every Nth frame of the clip')
    args = ap.parse_args()

    w, h = (int(x) for x in args.board_size.lower().split('x'))
    intr = calibrate_from_video(args.video, board_size=(w, h),
                                 square_m=args.square_mm / 1000.0, stride=args.stride)

    os.makedirs(_CALIB_DIR, exist_ok=True)
    out_path = os.path.join(_CALIB_DIR, f'cam{args.cam_index}.json')
    save_profile(out_path, intr.K, intr.dist, intr.image_size)
    print(f'Wrote {out_path}')
    print(f'  image_size: {intr.image_size}')
    print(f'  K:\n{intr.K}')
    print(f'  dist: {intr.dist}')


if __name__ == '__main__':
    main()
