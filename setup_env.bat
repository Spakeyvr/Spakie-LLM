@echo off
cd /d "%~dp0"
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo Installing PyTorch with CUDA 12.6...
pip install torch --index-url https://download.pytorch.org/whl/cu126

echo Installing remaining dependencies...
pip install -r requirements.txt

echo Done! Activate with: venv\Scripts\activate.bat
pause
