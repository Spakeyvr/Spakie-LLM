@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python tokenizer/train_tokenizer.py %*
pause
