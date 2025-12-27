@echo off
title Spectra-Shield - Real-Time Signal Defense System
color 0A

echo.
echo ============================================================
echo    🛡️  SPECTRA-SHIELD
echo    Real-Time Electromagnetic Signal Defense System
echo ============================================================
echo.

:: Check if .env exists
if not exist ".env" (
    echo [!] .env file not found. Creating from template...
    copy .env.example .env
    echo [!] Please edit .env and add your GEMINI_API_KEY
    echo.
)

:: Menu
echo Choose an option:
echo.
echo   [1] Run Demo (Defense Scenario - with Gemini LLM)
echo   [2] Run Demo (Health Scenario)
echo   [3] Run Demo (Agriculture Scenario)
echo   [4] Run Demo (Mock Mode - no API key needed)
echo   [5] Launch Dashboard (Streamlit)
echo   [6] Run Signal Simulator Only
echo   [7] Install Dependencies
echo   [0] Exit
echo.

set /p choice="Enter choice [0-7]: "

if "%choice%"=="1" (
    echo.
    echo Starting Defense Demo with Gemini...
    python demo.py --scenario defense --duration 30
    pause
    goto :eof
)

if "%choice%"=="2" (
    echo.
    echo Starting Health Demo...
    python demo.py --scenario health --duration 30
    pause
    goto :eof
)

if "%choice%"=="3" (
    echo.
    echo Starting Agriculture Demo...
    python demo.py --scenario agriculture --duration 30
    pause
    goto :eof
)

if "%choice%"=="4" (
    echo.
    echo Starting Demo in Mock Mode (no LLM)...
    python demo.py --scenario defense --duration 20 --no-llm
    pause
    goto :eof
)

if "%choice%"=="5" (
    echo.
    echo Launching Streamlit Dashboard...
    echo Open http://localhost:8501 in your browser
    streamlit run ui/dashboard.py
    pause
    goto :eof
)

if "%choice%"=="6" (
    echo.
    echo Running Signal Simulator...
    python signal_simulator.py --scenario defense --duration 15
    pause
    goto :eof
)

if "%choice%"=="7" (
    echo.
    echo Installing dependencies...
    pip install -r requirements.txt
    pip install litellm
    echo.
    echo Done! Dependencies installed.
    pause
    goto :eof
)

if "%choice%"=="0" (
    exit
)

echo Invalid choice. Please try again.
pause
