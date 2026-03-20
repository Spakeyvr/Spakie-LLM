@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python scripts/finetune.py --preset 180m --train-jsonl data/chat/train_targeted.jsonl --source-checkpoint sft_mixed_best.pt --output-name sft_targeted_best.pt %*
pause
