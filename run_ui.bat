@echo off
REM Dent Container Detection UI - Launch Script for Windows

echo Starting Dent Container Detection UI...
echo.

REM Activate conda environment
echo Activating conda environment: rgbd-model
call conda activate rgbd-model

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate conda environment 'rgbd-model'
    echo Please make sure the environment exists: conda create -n rgbd-model
    pause
    exit /b 1
)

echo Conda environment activated
echo.

REM Check if streamlit is installed
where streamlit >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Streamlit not found. Installing requirements...
    pip install -r requirements.txt
)

REM Run the app
echo Launching Streamlit app...
streamlit run app.py

pause

