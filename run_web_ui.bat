@echo off
REM Activate conda environment
call conda activate rgbd-model

if errorlevel 1 (
    echo Error: Failed to activate conda environment 'rgbd-model'
    echo Please create the environment first: conda create -n rgbd-model python=3.9
    pause
    exit /b 1
)

REM Check if model file exists
if not exist "best_attention_unet_4.pth" (
    echo Warning: Model file 'best_attention_unet_4.pth' not found in current directory
    echo The application will still run, but you'll need to provide the model path
)

echo Starting Flask API server...
echo API will be available at: http://localhost:5000
echo.
echo In another terminal, start the frontend:
echo   cd web_ui ^&^& python -m http.server 8000
echo.
echo Then open: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start Flask server
python api_server.py

pause
