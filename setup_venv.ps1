# -----------------------------------------
# PowerShell setup script for Python 3.12 + venv
# -----------------------------------------

$ErrorActionPreference = "Stop"

Write-Host "Checking for Python 3.12 availability..."

# 1) py-Launcher prüfen
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Host "Python Launcher 'py' not found. Please install Python 3.12 and add it to PATH."
    exit 1
}

# 2) Python 3.12 prüfen
$ver = & py -3.12 -c "import sys; print(sys.version)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python 3.12 not found. Please install it and ensure it is in PATH."
    exit 1
}
Write-Host "Found Python 3.12: $ver"

# 3) venv erstellen
Write-Host ""
Write-Host "Creating virtual environment (.venv) with Python 3.12..."
& py -3.12 -m venv .venv

# 4) venv aktivieren
Write-Host ""
Write-Host "Activating virtual environment..."
$activate = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $activate) {
    . $activate
} else {
    Write-Host "Activate.ps1 not found. Exiting."
    exit 1
}

# 5) Immer das venv-Python verwenden (nicht 'py')
$venvPy = Resolve-Path ".\.venv\Scripts\python.exe"
Write-Host "Using venv Python at: $venvPy"

# 6) pip upgraden + requirements installieren (in die venv)
Write-Host ""
Write-Host "Upgrading pip..."
& $venvPy -m pip install --upgrade pip

# Dateiname anpassen, falls du einen anderen verwendest
$req = "requirements.txt"
if (Test-Path $req) {
    Write-Host "Installing packages from $req ..."
    & $venvPy -m pip install -r $req
    Write-Host "Package installation complete."
} else {
    Write-Host "File '$req' not found. Skipping package installation."
}

# 7) Nachweis: welches Python ist aktiv?
Write-Host ""
& $venvPy -c "import sys; print('VENV PYTHON:', sys.executable)"
Write-Host "Virtual environment with Python 3.12 is ready."
