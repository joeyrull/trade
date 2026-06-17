# Pitcher Review

Upload a pitching video, get pose tracking, joint-angle charts, a kinetic-
chain breakdown, and an annotated video with the skeleton drawn on top.

## Running it on your computer

Everything runs locally on your own machine — nothing is uploaded to the
internet, no GitHub account needed once you have the files.

1. **First time only**, install two free programs if you don't already have
   them:
   - [Node.js](https://nodejs.org) (choose the LTS version)
   - [Python 3](https://www.python.org/downloads/)
2. Double-click the launcher for your computer:
   - **Mac**: `start.command`
   - **Windows**: `start.bat`
3. The first launch installs everything it needs (takes a few minutes) and
   then opens the app in your browser automatically. Every launch after
   that is fast.
4. To stop the app, close the window the launcher opened (or press
   Ctrl+C in it).

If a launcher reports a missing-program error, follow the link it prints,
install that program, then double-click the launcher again.

## What it does

- Upload one camera angle for standard 2D pose analysis, joint angles,
  speeds, and a lab-report scorecard with MLB reference bands.
- Upload two or three synced camera angles for true 3D multi-view
  triangulation (see `capture/README.md` for camera-rig and OBS capture
  setups, including a hardware-triggered Raspberry Pi rig).

For technical/architecture details, see `CLAUDE.md`.
