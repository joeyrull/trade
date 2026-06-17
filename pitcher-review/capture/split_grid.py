#!/usr/bin/env python3
"""Split a single OBS composite recording (3 cameras tiled in one scene) into
per-camera clips the pitcher-review pipeline expects.

If you're instead using OBS's "Source Record" plugin to capture each camera
to its own file directly, you don't need this script — just upload the 3
files through the web UI's "add a second/third camera angle" flow, or write
manifest.json by hand and use transfer.py.

This script is for the simpler OBS setup: one scene with all three camera
sources tiled in a grid, recorded as a single canvas. It crops that one file
back into camN.mp4 per source plus a manifest.json that transfer.py can read.
Because every camera comes from the same recording, inter-camera sync is
frame-exact (start_offset_s is always 0), unlike independently started
recordings.

Usage:
  # equal horizontal thirds (3 cameras side by side, the common layout)
  python3 split_grid.py obs_recording.mp4 --out ~/pitches/take_001 \\
      --angles front,side,three_quarter

  # explicit crop regions if your scene isn't equal thirds — OBS's own
  # "Transform" panel (right-click a source -> Transform -> Edit Transform)
  # shows each source's position/size on the canvas
  python3 split_grid.py obs_recording.mp4 --out ~/pitches/take_001 \\
      --crops 640x720+0+0,640x720+640+0,1280x360+0+720 \\
      --angles front,side,three_quarter
"""
import argparse
import json
import os
import subprocess
import sys

ANGLE_DEFAULTS = ['front', 'side', 'three_quarter']


def ffprobe_json(path):
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', path],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def video_stream(probe):
    for s in probe['streams']:
        if s['codec_type'] == 'video':
            return s
    raise SystemExit('No video stream found in ' + probe.get('format', {}).get('filename', '?'))


def parse_crop(spec):
    # "WxH+X+Y" -> (w, h, x, y)
    wh, x, y = spec.split('+')
    w, h = wh.split('x')
    return int(w), int(h), int(x), int(y)


def equal_thirds(width, height, n):
    # n cameras side by side in one row, equal-width slices, full height
    slice_w = width // n
    return [(slice_w, height, i * slice_w, 0) for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('recording', help='single composite recording from OBS')
    ap.add_argument('--out', required=True, help='output directory (created if missing)')
    ap.add_argument('--crops',
                     help='comma-separated WxH+X+Y per camera (default: equal horizontal thirds)')
    ap.add_argument('--angles', default=','.join(ANGLE_DEFAULTS),
                     help=f'comma-separated camera angles, default {",".join(ANGLE_DEFAULTS)}')
    ap.add_argument('--labels', help='comma-separated labels, default cam0/cam1/...')
    args = ap.parse_args()

    probe = ffprobe_json(args.recording)
    vstream = video_stream(probe)
    width, height = int(vstream['width']), int(vstream['height'])
    fps_num, fps_den = (int(x) for x in vstream['r_frame_rate'].split('/'))
    fps = fps_num / fps_den
    duration_s = float(probe['format'].get('duration', vstream.get('duration', 0)))

    angles = args.angles.split(',')
    n = len(angles)
    if n > 3:
        sys.exit('Only up to 3 cameras are supported (web upload API: video/video2/video3)')

    if args.crops:
        crops = [parse_crop(c) for c in args.crops.split(',')]
        if len(crops) != n:
            sys.exit(f'--crops has {len(crops)} regions but --angles has {n}')
    else:
        crops = equal_thirds(width, height, n)

    labels = args.labels.split(',') if args.labels else [f'cam{i}' for i in range(n)]

    os.makedirs(args.out, exist_ok=True)
    cameras = []
    for i, ((w, h, x, y), angle, label) in enumerate(zip(crops, angles, labels)):
        out_file = f'cam{i}.mp4'
        out_path = os.path.join(args.out, out_file)
        print(f'cam{i} ({label}, {angle}): cropping {w}x{h}+{x}+{y} -> {out_file}')
        subprocess.run(
            ['ffmpeg', '-y', '-i', args.recording, '-filter:v', f'crop={w}:{h}:{x}:{y}',
             '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-an', out_path],
            check=True,
        )
        cameras.append({
            'angle': angle, 'label': label, 'file': out_file,
            'start_offset_s': 0.0,  # one source recording -> frame-exact sync
        })

    manifest = {
        'fps': fps, 'width': width, 'height': height, 'duration_s': duration_s,
        'source': 'obs_grid_split', 'cameras': cameras,
    }
    manifest_path = os.path.join(args.out, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    for i, cam in enumerate(cameras):
        print(f"cam{i} ({cam['label']}): {cam['angle']} -> {cam['file']}")
    print(f'Wrote {manifest_path}')
    print('Note: re-run calibrate.py against these cropped files (not the original native '
          'camera feed) if you want calibrated true-scale 3D — the calibration is tied to '
          'the exact resolution each camera ends up at.')


if __name__ == '__main__':
    main()
