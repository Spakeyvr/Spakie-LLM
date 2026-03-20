@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python scripts/eval_basic_qa.py --preset 180m %*
pause
