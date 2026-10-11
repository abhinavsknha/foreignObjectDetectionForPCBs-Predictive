@echo off
echo ===================================================
echo   Starting PCB Predictive Detection Architecture
echo ===================================================

echo [1/3] Starting Next.js Frontend (Port 3000)...
start "Next.js Frontend" cmd /k "cd frontend && npm run dev"

echo [2/3] Starting FastAPI Backend (Port 8000)...
start "FastAPI Backend" cmd /k "cd backend && python -m uvicorn app.main:app --reload"

echo [3/3] Starting Celery AI Sleeper Cell Worker...
start "Celery Worker" cmd /k "cd backend && python -m celery -A app.celery_worker.celery_app worker --loglevel=info --pool=solo"

echo.
echo All microservices have been launched in separate terminal windows!
echo You can safely close this window.
