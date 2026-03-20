@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python scripts/train.py --preset 180m %*
pause
