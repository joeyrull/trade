#!/usr/bin/env python3
"""Send a synced take from the garage rig to the pitcher-review web app.

Reads manifest.json from a take directory (written by record_sync.py) and
POSTs the camera files to /api/analysis/upload as video/video2/video3 with
each camera's configured angle, kicking off analysis. For true 3-camera
triangulation, run the server with PITCHER_ENGINE=pitchcap (default) and
PITCHER_MULTIVIEW=1.

Usage:
  python3 transfer.py ~/pitches/take_001 --server http://192.168.1.50:3002 --throw-hand left
"""
import argparse
import json
import os
import sys

import requests

FIELD_NAMES = ['video', 'video2', 'video3']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('take_dir', help='directory containing manifest.json + camN.avi files')
    ap.add_argument('--server', required=True, help='e.g. http://192.168.1.50:3002')
    ap.add_argument('--throw-hand', choices=['left', 'right'], default='left')
    args = ap.parse_args()

    manifest_path = os.path.join(args.take_dir, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)

    cams = manifest['cameras']
    if not cams:
        sys.exit('manifest.json has no cameras')
    if len(cams) > 3:
        print(f'Note: only the first 3 of {len(cams)} cameras are sent '
              '(the upload API accepts video/video2/video3).', file=sys.stderr)

    opened = []
    try:
        files, data = {}, {'throwHand': args.throw_hand}
        for i, cam in enumerate(cams[:3]):
            path = os.path.join(args.take_dir, cam['file'])
            fh = open(path, 'rb')
            opened.append(fh)
            files[FIELD_NAMES[i]] = (cam['file'], fh, 'video/avi')
            angle_field = 'angle' if i == 0 else f'angle{i + 1}'
            data[angle_field] = cam.get('angle', 'side')

        url = args.server.rstrip('/') + '/api/analysis/upload'
        print(f'Uploading {len(files)} camera(s) to {url} ...')
        resp = requests.post(url, files=files, data=data, timeout=300)
        resp.raise_for_status()
        result = resp.json()
        print(json.dumps(result, indent=2))
        job_id = result.get('jobId')
        print(f'Job started: {job_id}')
        print(f"Check status: {args.server.rstrip('/')}/api/analysis/status/{job_id}")
    finally:
        for fh in opened:
            fh.close()


if __name__ == '__main__':
    main()
