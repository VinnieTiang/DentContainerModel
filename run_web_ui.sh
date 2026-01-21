#!/bin/bash

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rgbd-model

# Check if conda environment is activated
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'rgbd-model'"
    echo "Please create the environment first: conda create -n rgbd-model python=3.9"
    exit 1
fi

# Check if model file exists
if [ ! -f "best_attention_unet_4.pth" ]; then
    echo "Warning: Model file 'best_attention_unet_4.pth' not found in current directory"
    echo "The application will still run, but you'll need to provide the model path"
fi

echo "Starting Flask API server..."
echo "API will be available at: http://localhost:5000"
echo ""
echo "In another terminal, start the frontend:"
echo "  cd web_ui && python -m http.server 8000"
echo ""
echo "Then open: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Flask server
python api_server.py
