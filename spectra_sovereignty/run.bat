@echo off
title Spectra-Shield - Real-Time Signal Defense System
color 0A

echo.
echo ============================================================
echo    SPECTRA-SHIELD
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
echo   [1] Run Demo - Defense Scenario
echo   [2] Run Demo - Health Scenario
echo   [3] Run Demo - Agriculture Scenario
echo   [4] Run Demo - Mock Mode (no API key)
echo   [5] Launch Dashboard
echo   [6] Run Signal Simulator Only
echo   [7] Install Dependencies
echo   [0] Exit
echo.

set /p choice="Enter choice [0-7]: "

if "%choice%"=="1" goto defense
if "%choice%"=="2" goto health
if "%choice%"=="3" goto agriculture
if "%choice%"=="4" goto mock
if "%choice%"=="5" goto dashboard
if "%choice%"=="6" goto simulator
if "%choice%"=="7" goto install
if "%choice%"=="0" goto end

echo Invalid choice.
pause
goto end

:defense
echo.
echo Starting Defense Demo with Gemini...
python demo.py --scenario defense --duration 30
pause
goto end

:health
echo.
echo Starting Health Demo...
python demo.py --scenario health --duration 30
pause
goto end

:agriculture
echo.
echo Starting Agriculture Demo...
python demo.py --scenario agriculture --duration 30
pause
goto end

:mock
echo.
echo Starting Demo in Mock Mode...
python demo.py --scenario defense --duration 20 --no-llm
pause
goto end

:dashboard
echo.
echo Launching Streamlit Dashboard...
echo Open http://localhost:8501 in your browser
echo Press Ctrl+C to stop
streamlit run ui/dashboard.py
pause
goto end

:simulator
echo.
echo Running Signal Simulator...
python signal_simulator.py --scenario defense --duration 15
pause
goto end

:install
echo.
echo Installing dependencies...
pip install -r requirements.txt
pip install litellm
echo.
echo Done!
pause
goto end

:end
