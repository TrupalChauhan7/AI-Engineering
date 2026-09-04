@echo off
title FM-05 Speech-to-Clinical-Note - Windows Setup
cd /d "%~dp0"

echo ==================================================
echo   FM-05 Speech-to-Clinical-Note - Windows setup
echo   Run this once each time you sit at the lab PC.
echo ==================================================
echo.

echo [1/6] ffmpeg...
if not exist "ffmpeg.exe" (
  curl -L -o "%TEMP%\ff.zip" https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
  tar -xf "%TEMP%\ff.zip" -C "%TEMP%"
  for /d %%D in ("%TEMP%\ffmpeg-*") do (
    copy "%%D\bin\ffmpeg.exe" . >nul
    copy "%%D\bin\ffprobe.exe" . >nul
  )
)
echo     ffmpeg ready.
echo.

echo [2/6] Ollama...
if not exist "%USERPROFILE%\ollama\ollama.exe" (
  curl -L -o "%TEMP%\ollama.zip" https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip
  mkdir "%USERPROFILE%\ollama" 2>nul
  tar -xf "%TEMP%\ollama.zip" -C "%USERPROFILE%\ollama"
)
echo     Ollama ready.
echo.

echo [3/6] Starting the Ollama engine in its own window...
start "Ollama Engine - leave me open" "%USERPROFILE%\ollama\ollama.exe" serve
timeout /t 5 >nul
echo.

echo [4/6] Downloading the 3 AI models (large; only fetches what is missing)...
"%USERPROFILE%\ollama\ollama.exe" pull medgemma:4b
"%USERPROFILE%\ollama\ollama.exe" pull llama3.1:8b
"%USERPROFILE%\ollama\ollama.exe" pull atla/selene-mini
echo.

echo [5/6] Python environment + project install...
if not exist ".venv" (
  python -m venv .venv
)
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\pip install -e .
echo.

echo [6/6] Launching the demo at http://localhost:3000 ...
set PATH=%CD%;%PATH%
.venv\Scripts\python scripts\dev.py
