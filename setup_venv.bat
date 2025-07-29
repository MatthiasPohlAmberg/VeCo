@echo off
REM ====================================
REM Setup-Skript für virtuelle Umgebung
REM ====================================

set VENV_DIR=.venv

echo ================================
echo Check python 3.10 version
echo ================================

REM Prüft, ob Python 3.10 verfügbar ist
where py >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python Launcher 'py' not found. Please install Python 3.10 on your local system.
    exit /b 1
)

REM Gibt die Version zur Kontrolle aus
py -3.10 --version

echo ================================
echo Creating virtual environment
echo ================================

REM Erstelle virtuelle Umgebung mit Python 3.10
py -3.10 -m venv %VENV_DIR%
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Creating virtual environment failed.
    exit /b 1
)

echo ================================
echo Activating virtual environment
echo ================================

%VENV_DIR%\Scripts\activate.bat

echo ================================
echo Update pip and install packages
echo ================================

py -m pip install --upgrade pip

cd /d "%~dp0"
IF EXIST requirements.txt (
    pip install -r requirements.txt
) ELSE (
    echo [WARN] No requirements.txt found - skipping pakage installation.
)

echo ================================
echo Installation complete.
echo ================================
pause
