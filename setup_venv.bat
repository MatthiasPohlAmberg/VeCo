@echo off
REM ====================================
REM Setup-Skript für virtuelle Umgebung
REM ====================================

set VENV_DIR=.venv
echo ================================
echo Pruefe Python 3.11-Verfügbarkeit
echo ================================

REM Prüft ob python3.11 im PATH ist
where python3.11 >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo [FEHLER] Python 3.11 wurde nicht gefunden. Bitte installiere es zuerst.
    exit /b 1
)

REM Gibt die Version zur Kontrolle aus
echo [INFO] Gefundene Python-Version:
python3.11 --version

echo ================================
echo Erstelle virtuelle Umgebung
echo ================================

REM Erstellen der virtuellen Umgebung
python3.11 -m venv %VENV_DIR%
IF %ERRORLEVEL% NEQ 0 (
    echo [FEHLER] Erstellung der Umgebung ist fehlgeschlagen.
    exit /b 1
)

echo ================================
echo Aktiviere virtuelle Umgebung
echo ================================

IF EXIST %VENV_DIR%\Scripts\activate.bat (
    call %VENV_DIR%\Scripts\activate.bat
) ELSE (
    echo [FEHLER] Aktivierung fehlgeschlagen – Datei nicht gefunden.
    exit /b 1
)

echo ================================
echo Aktualisiere pip & installiere Pakete
echo ================================

pip install --upgrade pip
IF EXIST requirements.txt (
    pip install -r requirements.txt
) ELSE (
    echo [WARNUNG] Keine requirements.txt gefunden – überspringe Paketinstallation.
)

echo ================================
echo Installation abgeschlossen.
echo ================================
pause
