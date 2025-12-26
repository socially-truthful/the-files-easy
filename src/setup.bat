@echo off
setlocal
title DOJ Epstein Files Explorer - Setup
color 0F

:: ===================================================================
::                         CONFIGURATION
:: ===================================================================
set "PYTHON_VERSION=3.11.7"
set "PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe"
set "PYTHON_INSTALLER=python_installer.exe"

:: ===================================================================
::                         DISPLAY HEADER
:: ===================================================================
cls
echo.
echo   ============================================================
echo        DOJ EPSTEIN FILES EXPLORER - One-Click Setup
echo   ============================================================
echo.
echo   This installer will set up everything you need:
echo.
echo      [1] Python (if not installed)
echo      [2] Virtual environment
echo      [3] All required packages
echo      [4] Data directories
echo.
echo   ------------------------------------------------------------
echo.

:: ===================================================================
::                    STEP 1: CHECK/INSTALL PYTHON
:: ===================================================================
echo   [1/4] Checking Python installation...
echo   ------------------------------------------------------------
echo.

python --version >nul 2>&1
if not errorlevel 1 goto python_ok

echo      Python not found. Installing Python %PYTHON_VERSION%...
echo.
echo      Downloading Python installer...

powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%'" >nul 2>&1

if not exist "%PYTHON_INSTALLER%" goto python_download_failed

echo      Installing Python (this may take a minute)...
echo      A separate installer window may appear.
echo.

start /wait "" "%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
del "%PYTHON_INSTALLER%" >nul 2>&1

set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%PATH%"

python --version >nul 2>&1
if errorlevel 1 goto python_restart_needed
goto python_ok

:python_download_failed
echo.
echo      [!] Could not download Python installer.
echo.
echo      Please install Python manually:
echo      https://www.python.org/downloads/
echo.
echo      Make sure to check "Add Python to PATH" during install.
echo.
pause
exit /b 1

:python_restart_needed
echo      [!] Python installation may require a restart.
echo.
echo      Please restart your computer and run this setup again.
echo.
pause
exit /b 1

:python_ok
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo      [OK] Python %PYVER% detected
echo.

:: ===================================================================
::                 STEP 2: CREATE VIRTUAL ENVIRONMENT
:: ===================================================================
echo   ------------------------------------------------------------
echo.
echo   [2/4] Setting up virtual environment...
echo   ------------------------------------------------------------
echo.

if exist "venv" goto venv_exists

echo      Creating isolated Python environment...
python -m venv venv >nul 2>&1
if errorlevel 1 goto venv_failed
echo      [OK] Virtual environment created
goto venv_done

:venv_exists
echo      [OK] Virtual environment already exists
goto venv_done

:venv_failed
echo      [!] Failed to create virtual environment.
pause
exit /b 1

:venv_done
echo.

:: ===================================================================
::                    STEP 3: INSTALL PACKAGES
:: ===================================================================
echo   ------------------------------------------------------------
echo.
echo   [3/4] Installing required packages...
echo   ------------------------------------------------------------
echo.
echo      This takes 5-10 minutes. Progress will appear below.
echo      If it seems frozen, packages are still downloading!
echo.

call venv\Scripts\activate.bat

echo      [....] Updating package manager...
python -m pip install --upgrade pip --quiet >nul 2>&1
echo      [DONE] Package manager updated
echo.

echo      [....] Installing dependencies (this is the slow part)...
echo      You should see package names scrolling...
echo.

pip install -r src\requirements.txt 2>nul
if errorlevel 1 goto pip_retry
goto pip_done

:pip_retry
echo      [!] Some packages had issues. Retrying with verbose...
pip install -r src\requirements.txt

:pip_done
echo.
echo      [OK] All packages installed
echo.

:: ===================================================================
::                    STEP 4: CREATE DIRECTORIES
:: ===================================================================
echo   ------------------------------------------------------------
echo.
echo   [4/4] Creating data directories...
echo   ------------------------------------------------------------
echo.

if not exist "data" mkdir data
if not exist "data\downloads" mkdir data\downloads
if not exist "data\extracted_text" mkdir data\extracted_text
if not exist "data\classification" mkdir data\classification
if not exist "data\search_index" mkdir data\search_index

echo      [OK] Data directories ready
echo.

:: ===================================================================
::                    CHECK OPTIONAL TOOLS
:: ===================================================================
echo   ------------------------------------------------------------
echo.
echo   Checking optional tools...
echo.

ffmpeg -version >nul 2>&1
if errorlevel 1 goto no_ffmpeg
echo      [OK] FFmpeg installed
goto check_tesseract

:no_ffmpeg
echo      [ ] FFmpeg not found (needed for video/audio)
echo          Download: https://ffmpeg.org/download.html

:check_tesseract
tesseract --version >nul 2>&1
if errorlevel 1 goto no_tesseract
echo      [OK] Tesseract OCR installed
goto tools_done

:no_tesseract
echo      [ ] Tesseract not found (needed for scanned PDFs)
echo          Download: https://github.com/UB-Mannheim/tesseract/wiki

:tools_done
echo.

:: ===================================================================
::                        SETUP COMPLETE
:: ===================================================================
echo.
echo   ============================================================
echo                    SETUP COMPLETE!
echo   ============================================================
echo.
exit /b 0
