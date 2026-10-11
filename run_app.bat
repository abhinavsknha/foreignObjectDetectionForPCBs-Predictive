@echo off
echo ========================================================
echo   Launching PCB Defect Isolation Platform (Local Demo)
echo ========================================================
echo Installing/Verifying dependencies...
python -m pip install -r requirements.txt
echo.
echo Launching Streamlit Dashboard...
python -m streamlit run dashboard_app.py
pause
