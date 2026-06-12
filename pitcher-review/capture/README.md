# Garage 3-camera capture rig

Scripts for a Raspberry Pi driving three OV9281 720p/120fps global-shutter USB
cameras, hardware-triggered in sync, feeding the `pitcher-review` web app's
multi-view (true 3D) analysis path.

## Hardware

- 3x Innomaker OV9281 720p/120fps global-shutter USB cameras
- 1x Raspberry Pi (any model with 3 free USB ports + GPIO)
- Breadboard + DuPont jumper wires

Planned garage placement (edit `CAMERAS` in `record_sync.py` once you know
which `/dev/videoN` maps to which physical camera):

1. Red rack
2. Above the fridge
3. Above the Smith machine / plyo wall rack

## Wiring: hardware trigger sync

These OV9281 modules have a TRIGGER pin and a GND pin for external-trigger
("genlock") mode. On the breadboard:

- Tie all three cameras' TRIGGER pins together, and to one Pi GPIO pin
  (default **BCM18**, a hardware-PWM-capable pin — see `trigger.py`).
- Tie all three cameras' GND pins together, and to a Pi GND pin.

`trigger.py` drives that GPIO with a steady square wave at the camera frame
rate (120 Hz default) using pigpio's hardware PWM (DMA-backed — accurate
enough to keep three cameras' exposures locked together; software-timed GPIO
toggling drifts too much at 120 Hz). Each trigger pulse causes all three
sensors to expose a frame at the same instant, so the resulting clips are
frame-synced without needing audio cross-correlation (these cameras have no
mic).

**Before relying on this**: check your specific OV9281 board's datasheet for
how it enters external-trigger mode — some need a V4L2/UVC control set (e.g.
via `v4l2-ctl -d /dev/videoX --list-ctrls-menu`, look for something like
`trigger_mode`) in addition to the physical trigger signal. If your boards
don't support external trigger at all, the rig still works with software-only
sync — start all three captures together (the barrier-synced threads in
`record_sync.py` already do this) and check `manifest.json`'s
`start_offset_s` per camera; offsets under one frame period (~8ms at 120fps)
are fine for the existing analysis pipeline.

## Software setup (on the Pi)

```bash
sudo apt install -y pigpio v4l-utils python3-opencv
pip3 install -r requirements.txt   # pigpio python bindings, requests
sudo systemctl enable --now pigpiod   # or `sudo pigpiod` each boot
```

Find the device nodes:

```bash
v4l2-ctl --list-devices
```

Plug them in one at a time if the ordering is ambiguous, and update the
`device` paths (and `angle`/`label`) in `CAMERAS` at the top of
`record_sync.py` to match your physical layout.

## Recording a take

```bash
python3 record_sync.py --duration 6 --out ~/pitches/take_001
```

This starts the GPIO trigger, records 6 seconds from all three cameras in
parallel (barrier-synced thread start), and writes:

- `cam0.avi`, `cam1.avi`, `cam2.avi` — MJPG (no recompression — light enough
  for a Pi running three 720p120 streams; the web upload form already accepts
  `.avi`)
- `manifest.json` — per-camera device, angle, frame count, start offset

If `trigger.py` is already running in another terminal/process, pass
`--no-trigger`.

## Sending a take to the web app

On the computer running the `pitcher-review` server (port 3002 by default,
ideally started with `PITCHER_MULTIVIEW=1` for true 3-camera triangulation):

```bash
python3 transfer.py ~/pitches/take_001 --server http://<computer-ip>:3002 --throw-hand left
```

This POSTs `cam0/1/2.avi` as `video`/`video2`/`video3` with each camera's
configured angle to `/api/analysis/upload` and prints the job ID + status URL.
You can also just copy the take directory over (e.g. `scp`/`rsync`/a shared
network drive) and upload the three files manually through the web UI's
"add a second/third camera angle" flow.

## One-time camera calibration

Once each camera is mounted in its final spot, record a ~10-15s clip of a
printed checkerboard (default 9x6 inner corners) moved around the frame and
tilted a bit, then for each camera:

```bash
python3 calibrate.py checkerboard_cam0.avi --cam-index 0 --square-mm 25
```

This writes `../server/scripts/calibration/cam0.json` (etc. for cam1/cam2).
`pitchcap_analyze.py`'s multi-view path automatically uses any `camN.json`
whose `image_size` matches that camera's clip, which sharpens the recovered
3D scale over the `approximate_intrinsics` fallback. Nothing breaks if a
profile is missing or doesn't match — it just falls back and notes it in
`summary.pitchcap.warnings`. Re-run if a camera is ever repositioned or
refocused.

## Notes / future work

- **Processing time**: RTMPose 2D estimation is the bottleneck — roughly
  ~2-2.5 fps per camera on CPU-only `onnxruntime`. At 120fps that's ~50x
  slower than real time, *per camera*, and the 3 cameras run sequentially
  (one pose pass per clip). A 4-second take (480 frames x 3 cams = 1440
  frames) would take ~10 minutes on CPU. **Keep takes short** — just the
  windup-through-release window, not the whole bullpen — and/or run the
  server with a GPU-enabled `onnxruntime` (`onnxruntime-gpu`) if available on
  the analysis machine.
- **USB bandwidth**: three simultaneous 720p/120fps MJPG streams is a lot of
  USB traffic. If frames drop, try a powered USB3 hub, or lower resolution/fps
  slightly (edit `WIDTH`/`HEIGHT`/`FPS` in `record_sync.py` and `trigger.py`'s
  `--freq` to match).
