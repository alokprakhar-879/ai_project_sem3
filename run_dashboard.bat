@echo off
cd /d "%~dp0"
echo Starting Adaptive Traffic Signal Control Dashboard...
python -m streamlit run dashboard/app.py
pause