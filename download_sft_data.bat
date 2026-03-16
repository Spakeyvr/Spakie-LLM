@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python scripts/download_sft_data.py %*
pause
