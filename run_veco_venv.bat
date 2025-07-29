echo ================================
echo Activating virtual environment
echo ================================
set VENV_DIR=.venv

%VENV_DIR%\Scripts\activate.bat

echo ================================
echo Running VeCo
echo ================================

py veco.py