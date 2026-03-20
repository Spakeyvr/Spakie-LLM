@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python scripts/download_pretrain_corpus.py %*
pause
