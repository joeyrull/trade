#!/bin/bash
# Double-click this file in Finder to set up (first run only) and launch
# Pitcher Review. A Terminal window will open with server logs.
set -e

cd "$(dirname "$0")"

if [ ! -d node_modules ]; then
  echo "Installing Node dependencies (first run only)..."
  npm install
fi

if ! python3 -c "import mediapipe" >/dev/null 2>&1; then
  echo "Installing Python dependencies (first run only)..."
  pip3 install -r requirements.txt
fi

echo ""
echo "Starting Pitcher Review..."
echo "Once you see 'VITE ... ready', open http://localhost:5174 in your browser."
echo "Press Ctrl+C to stop."
echo ""

npm run dev
