@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please follow README.md setup instructions first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" tools\serve.py
pause
