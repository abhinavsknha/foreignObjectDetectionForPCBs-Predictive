@echo off
title PCB Inspection - local demo (no tunnel)
echo ===================================================
echo   PCB Defect Detection: LOCAL demo
echo   Everything runs on this laptop: http://localhost:3000
echo   No public tunnel, so it does not depend on loca.lt
echo ===================================================
echo.

netstat -ano | findstr ":8000 " | findstr LISTENING >nul
if %errorlevel%==0 (
  echo [1/2] Backend already running on port 8000, reusing it.
) else (
  echo [1/2] Starting the API and the Celery worker...
  start "FastAPI Backend" cmd /k "cd /d %~dp0backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
  start "Celery Worker" cmd /k "cd /d %~dp0backend && python -m celery -A app.celery_worker.celery_app worker --loglevel=info --pool=solo"
)

echo [2/2] Starting the website...
if not exist "%~dp0frontend\.next\BUILD_ID" (
  echo Building the website first, this takes about a minute...
  pushd "%~dp0frontend"
  call npx next build
  popd
)
start "Website (localhost:3000)" cmd /k "cd /d %~dp0frontend && npx next start -p 3000"

echo.
echo If the backend was just started, wait until the Celery Worker window prints
echo "ready" (about 30 seconds), then open http://localhost:3000
echo The header shows "Pipeline Online" when everything is up.
