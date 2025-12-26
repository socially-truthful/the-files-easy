@echo off
setlocal
title DOJ Epstein Files Explorer
color 0F
:: mode con: cols=70 lines=30

:: ===================================================================
::                         HEADER
:: ===================================================================
cls
echo.
echo   ============================================================
echo        DOJ EPSTEIN FILES EXPLORER
echo   ============================================================
echo.

:: ===================================================================
::                   CHECK IF SETUP NEEDED
:: ===================================================================

if exist "venv\Scripts\python.exe" goto check_data

echo   [STEP 1/3] First time setup detected
echo.
echo   Installing Python environment and dependencies...
echo   This takes 5-10 minutes. You will see progress below.
echo.
echo   ------------------------------------------------------------
echo.
call src\setup.bat
if errorlevel 1 (
    echo.
    echo   Setup failed. Please check errors above.
    pause
    exit /b 1
)

:check_data
:: ===================================================================
::                   CHECK IF DATA EXISTS
:: ===================================================================

:: Check if search index has actual content (not just empty dir)
if not exist "data\search_index\MAIN_WRITELOCK" goto run_pipeline
goto check_ai

:run_pipeline

cls
echo.
echo   ============================================================
echo        DOJ EPSTEIN FILES EXPLORER
echo   ============================================================
echo.
echo   [STEP 2/3] No data found - starting full pipeline
echo.
echo   This will:
echo      1. Download files from DOJ (~18GB)
echo      2. Extract and process text
echo      3. Build search index
echo.
echo   You will see detailed progress for each step.
echo   The window will scroll - this is normal!
echo.
echo   ------------------------------------------------------------
echo.
echo   Starting in 5 seconds... Press Ctrl+C to cancel.
timeout /t 5
echo.
call venv\Scripts\activate.bat
pip install beautifulsoup4 --quiet 2>nul
python src\main.py all
if errorlevel 1 (
    echo.
    echo   Pipeline encountered an error.
    pause
    exit /b 1
)

:: Verify data was actually created
if not exist "data\search_index\MAIN_WRITELOCK" (
    echo.
    echo   [ERROR] Search index was not created.
    echo   The pipeline may not have completed successfully.
    pause
    exit /b 1
)

:check_ai
:: ===================================================================
::                   CHECK FOR AI ASSISTANT
:: ===================================================================

:: Check if AI is already set up
if exist "data\rag_index.json" goto launch

:launch

:: ===================================================================
::                   LAUNCH WEB INTERFACE
:: ===================================================================
cls
echo.
echo   ============================================================
echo        DOJ EPSTEIN FILES EXPLORER
echo   ============================================================
echo.
echo   ------------------------------------------------------------
echo.
echo   Starting web interface...
echo.
echo   Opening in your browser:  http://localhost:5000
echo.

:: Show AI status
if exist "data\rag_index.json" (
    echo   AI Assistant:  http://localhost:5000/chat
    echo.
)

echo   ------------------------------------------------------------
echo.
echo   Close this window or press Ctrl+C to stop.
echo.

call venv\Scripts\activate.bat

:: Start Ollama if AI is available (models stored in app directory)
set "OLLAMA_MODELS=%~dp0data\ollama_models"
if exist "data\rag_index.json" (
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if errorlevel 1 (
        start "" /min cmd /c "set OLLAMA_MODELS=%OLLAMA_MODELS% && ollama serve"
        timeout /t 2 /nobreak >nul
    )
)

:: Start server (browser opens automatically via Python)
python src\main.py serve

pause
exit /b 0
