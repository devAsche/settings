@echo off
cd /d "%~dp0ComfyUI"
echo Open http://127.0.0.1:8188 in your browser.
"%~dp0venv\Scripts\python.exe" main.py --listen 127.0.0.1 --port 8188 --lowvram
pause
