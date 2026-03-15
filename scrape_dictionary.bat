@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python scripts/scrape_dictionary.py %*
pause
