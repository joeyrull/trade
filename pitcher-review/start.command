#!/bin/bash
# Double-click this file in Finder (or run it from a terminal) to set up
# (first run only) and launch Pitcher Review. A browser tab opens
# automatically once the app is ready. Close this window (or press Ctrl+C)
# to stop the app.
set -e
cd "$(dirname "$0")"

command_exists() { command -v "$1" >/dev/null 2>&1; }

pause_and_exit() {
  read -r -p "Press Enter to close this window..." _ || true
  exit 1
}

if ! command_exists node || ! command_exists npm; then
  echo "Node.js is required but wasn't found on this computer."
  echo "Install it from https://nodejs.org (choose the LTS version), then double-click this file again."
  pause_and_exit
fi

PYTHON_BIN=python3
if ! command_exists "$PYTHON_BIN"; then
  if command_exists python; then
    PYTHON_BIN=python
  else
    echo "Python 3 is required but wasn't found on this computer."
    echo "Install it from https://www.python.org/downloads/, then double-click this file again."
    pause_and_exit
  fi
fi

if [ ! -d web/node_modules ] || [ ! -d server/node_modules ]; then
  echo "Installing Node dependencies (first run only, this can take a minute)..."
  npm install
fi

if ! "$PYTHON_BIN" -c "import mediapipe" >/dev/null 2>&1; then
  echo "Installing Python dependencies (first run only, this can take a few minutes)..."
  if ! "$PYTHON_BIN" -m pip install -r requirements.txt; then
    echo "Retrying with --break-system-packages (needed on some newer Python installs)..."
    "$PYTHON_BIN" -m pip install --break-system-packages -r requirements.txt
  fi
fi

echo ""
echo "Starting Pitcher Review..."
echo "A browser tab will open automatically once it's ready."
echo "Keep this window open while you use the app -- closing it (or Ctrl+C) stops the app."
echo ""

npm run dev &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null' EXIT INT TERM

URL="http://localhost:5174"
"$PYTHON_BIN" - "$URL" <<'PYEOF' &
import sys, time, urllib.request
url = sys.argv[1]
for _ in range(60):
    try:
        urllib.request.urlopen(url, timeout=1)
        sys.exit(0)
    except Exception:
        time.sleep(1)
sys.exit(1)
PYEOF
WAIT_PID=$!
if wait "$WAIT_PID"; then
  if command_exists open; then
    open "$URL"          # macOS
  elif command_exists xdg-open; then
    xdg-open "$URL"       # Linux
  else
    echo "Open $URL in your browser."
  fi
else
  echo "The app didn't respond in time -- check the log output above for errors, or open $URL manually."
fi

wait "$SERVER_PID"
