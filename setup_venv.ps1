# -----------------------------------------
# PowerShell setup script for Python 3.10
# -----------------------------------------

Write-Host "Checking for Python 3.10 availability..."

# Check if 'py' launcher exists
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Host "Python Launcher 'py' not found. Please install Python 3.10 and ensure it is added to PATH."
    exit 1
}

# Check if Python 3.10 is available
$pyVersion = py -3.10 --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python 3.10 not found. Please install it and ensure it is added to PATH."
    exit 1
}

Write-Host "Found Python version: $pyVersion"

# Optional: show path to Python launcher
$pythonPath = (Get-Command py).Source
Write-Host "Python launcher path: $pythonPath"

# -----------------------------------------
# Create virtual environment
# -----------------------------------------

Write-Host ""
Write-Host "Creating virtual environment using Python 3.10..."
py -3.10 -m venv .venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to create virtual environment. Exiting."
    exit 1
}

# -----------------------------------------
# Activate virtual environment
# -----------------------------------------

Write-Host ""
Write-Host "Activating virtual environment..."
# Check if the Activate.ps1 script exists
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "Failed to find Activate.ps1 script in the virtual environment. Exiting."
    exit 1
}

# -----------------------------------------
# Install dependencies
# -----------------------------------------

Write-Host ""
Write-Host "Installing packages from requirements.txt..."
if (Test-Path "requirements.txt") {
    pip install --upgrade pip
    pip install -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Package installation complete."
    } else {
        Write-Host "Failed to install packages. Please check the error messages above."
        exit 1
    }
} else {
    Write-Host "File 'requirements.txt' not found. Skipping package installation."
}

Write-Host ""
Write-Host "Virtual environment with Python 3.10 is ready."
