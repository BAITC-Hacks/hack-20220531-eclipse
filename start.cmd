@echo off
setlocal
cd /d "%~dp0"
set "PROJECT_PYTHON=%~dp0work\.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" (
    py -3 -m venv "%~dp0work\.venv"
    if errorlevel 1 goto failed
)
"%PROJECT_PYTHON%" -m pip install --disable-pip-version-check -r "%~dp0requirements.txt"
if errorlevel 1 goto failed
"%PROJECT_PYTHON%" -m streamlit run "%~dp0app.py"
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo Startup failed. Read the error above. Python 3.11 or newer is required.
pause
exit /b 1
