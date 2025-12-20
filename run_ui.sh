#!/bin/bash

# Dent Container Detection UI - Launch Script

echo "🚀 Starting Dent Container Detection UI..."
echo ""

# Activate conda environment
echo "📦 Activating conda environment: rgbd-model"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rgbd-model

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment 'rgbd-model'"
    echo "Please make sure the environment exists: conda create -n rgbd-model"
    exit 1
fi

echo "✅ Conda environment activated"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing requirements..."
    pip install -r requirements.txt
fi

# Run the app
echo "🌐 Launching Streamlit app..."
streamlit run app.py

