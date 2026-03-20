@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python scripts/build_targeted_data.py %*
pause
