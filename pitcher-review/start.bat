@echo off
setlocal
rem Double-click this file in File Explorer to set up (first run only) and
rem launch Pitcher Review. A browser tab opens automatically once the app
rem is ready. Close the "Pitcher Review" window that pops up to stop the app.

cd /d "%~dp0"

where node >nul 2>nul
if errorlevel 1 (
  echo Node.js is required but wasn't found on this computer.
  echo Install it from https://nodejs.org ^(choose the LTS version^), then double-click this file again.
  pause
  exit /b 1
)

set PYTHON_BIN=python
where python >nul 2>nul
if errorlevel 1 (
  where py >nul 2>nul
  if errorlevel 1 (
    echo Python 3 is required but wasn't found on this computer.
    echo Install it from https://www.python.org/downloads/, then double-click this file again.
    pause
    exit /b 1
  )
  set PYTHON_BIN=py -3
)

if not exist web\node_modules (
  echo Installing Node dependencies ^(first run only, this can take a minute^)...
  call npm install
)
if not exist server\node_modules (
  echo Installing Node dependencies ^(first run only, this can take a minute^)...
  call npm install
)

%PYTHON_BIN% -c "import mediapipe" >nul 2>nul
if errorlevel 1 (
  echo Installing Python dependencies ^(first run only, this can take a few minutes^)...
  %PYTHON_BIN% -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Retrying with --break-system-packages ^(needed on some newer Python installs^)...
    %PYTHON_BIN% -m pip install --break-system-packages -r requirements.txt
  )
)

echo.
echo Starting Pitcher Review...
echo A browser tab will open automatically once it's ready.
echo Close the "Pitcher Review" window to stop the app.
echo.

start "Pitcher Review" cmd /k npm run dev

echo Waiting for the app to start...
powershell -NoProfile -Command "$ok=$false; for($i=0;$i -lt 60;$i++){try{Invoke-WebRequest -Uri http://localhost:5174 -UseBasicParsing -TimeoutSec 1 | Out-Null; $ok=$true; break}catch{Start-Sleep -Seconds 1}}; if(-not $ok){exit 1}"
if errorlevel 1 (
  echo The app didn't respond in time -- check the "Pitcher Review" window for errors, or open http://localhost:5174 manually.
) else (
  start http://localhost:5174
)

pause
