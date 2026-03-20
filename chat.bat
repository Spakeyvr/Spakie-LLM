@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python scripts/chat.py --preset 180m %*
