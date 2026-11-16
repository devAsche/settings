@echo off
cd /d "%~dp0ComfyUI"
echo Open http://127.0.0.1:8188 in your browser.
REM 2026-11-16 RTX 5060 Ti 16GB: removed --lowvram, added SageAttention. Old version: start_comfyui_lowvram_backup.bat
"%~dp0venv\Scripts\python.exe" main.py --listen 127.0.0.1 --port 8188 --use-sage-attention
pause
