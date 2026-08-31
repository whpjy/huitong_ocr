@echo off
setlocal
cd /d "%~dp0"

set "HUITONG_PYTHON="
if /I "%CONDA_DEFAULT_ENV%"=="huitong" if exist "%CONDA_PREFIX%\python.exe" set "HUITONG_PYTHON=%CONDA_PREFIX%\python.exe"
if not defined HUITONG_PYTHON if exist "%USERPROFILE%\.conda\envs\huitong\python.exe" set "HUITONG_PYTHON=%USERPROFILE%\.conda\envs\huitong\python.exe"

if not defined HUITONG_PYTHON (
    echo Could not find the huitong Conda environment.
    exit /b 1
)

start "Huitong OCR API" /D "%~dp0backend" "%HUITONG_PYTHON%" main.py
start "Huitong OCR Web Demo" /D "%~dp0web_demo" "%HUITONG_PYTHON%" main.py

echo API: http://127.0.0.1:8000/docs
echo Web demo: http://127.0.0.1:5173
endlocal
