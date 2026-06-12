#!/usr/bin/env python3
"""Synchronized 3-camera capture for the garage OV9281 rig.

Hardware: 3x OV9281 720p/120fps global-shutter USB cameras on one Raspberry
Pi. All three cameras' TRIGGER pins (+ a shared GND) are wired together on a
breadboard to one Pi GPIO pin, which trigger.py drives at the camera frame
rate -- this puts the sensors in external-trigger / genlock mode so every
camera exposes at the same instant.

Edit CAMERAS below once you know which /dev/videoN node is physically
mounted where (red rack / above fridge / above smith machine + plyo wall) --
see ../pitcher-review/capture/README.md for how to map device nodes to camera
angles, and which CAMERA_ANGLES values the web upload API accepts.

Usage:
  sudo pigpiod                              # once, if not already running
  python3 record_sync.py --duration 6 --out ~/pitches/take_001

Writes <out>/cam0.avi, cam1.avi, cam2.avi (MJPG, no recompression -- the
upload form already accepts .avi) + manifest.json describing each camera's
angle, frame count, and capture-start offset.
"""
import argparse
import json
import os
import threading
import time

import cv2

from trigger import start_trigger, stop_trigger

# Edit to match your garage layout. `angle` must be one of the web app's
# CAMERA_ANGLES ('side', 'front', 'behind', 'three_quarter', 'other') -- the
# first camera's angle is informational only when PITCHER_MULTIVIEW=1 (all
# clips go to one triangulation job), but still useful for transfer.py.
CAMERAS = [
    {'device': '/dev/video0', 'angle': 'side',          'label': 'red_rack'},
    {'device': '/dev/video1', 'angle': 'front',         'label': 'above_fridge'},
    {'device': '/dev/video2', 'angle': 'three_quarter', 'label': 'smith_plyo_wall'},
]

WIDTH, HEIGHT, FPS = 1280, 720, 120


def open_camera(device, width, height, fps):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    # MJPG: the OV9281's compressed output. Re-muxing into an MJPG .avi
    # (below) needs no CPU-heavy transcode, which matters when reading three
    # 720p120 streams on a Pi at once.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open {device}')
    return cap


def capture_worker(cam_idx, cam_cfg, out_dir, duration, barrier, results):
    cap = open_camera(cam_cfg['device'], WIDTH, HEIGHT, FPS)
    out_path = os.path.join(out_dir, f'cam{cam_idx}.avi')
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), FPS, (WIDTH, HEIGHT))

    barrier.wait()  # release all camera threads at the same instant
    t0 = time.monotonic()
    deadline = t0 + duration
    n = 0
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        n += 1

    cap.release()
    writer.release()
    results[cam_idx] = {'frames': n, 't0': t0, 'file': os.path.basename(out_path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--duration', type=float, required=True, help='seconds to record')
    ap.add_argument('--out', required=True, help='output directory (created if missing)')
    ap.add_argument('--gpio', type=int, default=18, help='trigger GPIO (BCM), see trigger.py')
    ap.add_argument('--no-trigger', action='store_true',
                    help="don't drive the GPIO trigger here (e.g. trigger.py already running)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    pi = None
    if not args.no_trigger:
        import pigpio
        pi = pigpio.pi()
        if not pi.connected:
            raise SystemExit('Could not connect to pigpiod -- run `sudo pigpiod` first')
        start_trigger(pi, args.gpio, FPS)

    try:
        barrier = threading.Barrier(len(CAMERAS))
        results = {}
        threads = [
            threading.Thread(target=capture_worker, args=(i, cam, args.out, args.duration, barrier, results))
            for i, cam in enumerate(CAMERAS)
        ]
        print(f'Recording {args.duration}s from {len(CAMERAS)} cameras...')
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        if pi is not None:
            stop_trigger(pi, args.gpio)
            pi.stop()

    t0_min = min(r['t0'] for r in results.values())
    manifest = {
        'fps': FPS, 'width': WIDTH, 'height': HEIGHT, 'duration_s': args.duration,
        'cameras': [
            {
                'device': CAMERAS[i]['device'],
                'angle': CAMERAS[i]['angle'],
                'label': CAMERAS[i]['label'],
                'file': results[i]['file'],
                'frames': results[i]['frames'],
                'start_offset_s': round(results[i]['t0'] - t0_min, 4),
            }
            for i in range(len(CAMERAS))
        ],
    }
    manifest_path = os.path.join(args.out, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    for i, cam in enumerate(manifest['cameras']):
        print(f"cam{i} ({cam['label']}): {cam['frames']} frames, "
              f"start offset {cam['start_offset_s']}s -> {cam['file']}")
    print(f'Wrote {manifest_path}')


if __name__ == '__main__':
    main()
