@echo off
title PCB Inspection - demo launcher
echo ===================================================
echo   PCB Defect Detection: demo stack
echo   API + Celery worker + public tunnel
echo   Site: https://fodpcb-predictive.vercel.app
echo ===================================================
echo.

echo [1/3] FastAPI backend on port 8000...
start "FastAPI Backend" cmd /k "cd /d %~dp0backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"

echo [2/3] Celery worker (loads both detectors, about 20 seconds)...
start "Celery Worker" cmd /k "cd /d %~dp0backend && python -m celery -A app.celery_worker.celery_app worker --loglevel=info --pool=solo"

echo [3/3] Public tunnel...
start "Tunnel" cmd /k "npx localtunnel --port 8000 --subdomain fodpcb-predictive-api"

echo.
echo Wait until the Celery Worker window prints "ready" and the Tunnel window
echo prints "your url is: https://fodpcb-predictive-api.loca.lt".
echo Then open the site: the header shows "Pipeline Online" when everything is up.
echo.
echo If the tunnel prints a different address, the name was taken: tell Claude
echo the new address, or edit frontend/public/config.json and push.
echo.
echo Keep this laptop awake and online during the demo.
