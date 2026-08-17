@echo off
setlocal
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

set "TASK_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%TASK_PYTHON%" goto run_model

set "TASK_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%TASK_PYTHON%" goto run_model

where python >nul 2>nul
if not errorlevel 1 (
  set "TASK_PYTHON=python"
  goto run_model
)

where py >nul 2>nul
if not errorlevel 1 (
  set "TASK_PYTHON=py"
  goto run_model
)

echo [ERROR] No usable Python interpreter was found.
echo Install Python 3.11 or select an interpreter in VS Code.
exit /b 1

:run_model
echo Using Python: %TASK_PYTHON%
"%TASK_PYTHON%" -m src.main
exit /b %ERRORLEVEL%
