# Dent Container Detection - Streamlit UI

A simple web-based interface for running inference with the trained Attention-UNet model for dent container detection.

## Features

- 📦 Load trained model (.pth file)
- 📊 Display model information and architecture details
- 📥 Upload depth maps (.npy files)
- 🚀 Run inference to generate binary segmentation masks
- 📤 Visualize results (binary mask and probability map)
- 💾 Download results as .npy files

## Installation

1. Activate your conda environment:

```bash
conda activate rgbd-model
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Activate the conda environment:

```bash
conda activate rgbd-model
```

2. Make sure your trained model file (`best_attention_unet.pth`) is in the project directory, or you can upload it through the UI.

3. Run the Streamlit app:

```bash
streamlit run app.py
```

Or use the launch scripts (they will activate conda automatically):

- Mac/Linux: `./run_ui.sh`
- Windows: `run_ui.bat`

3. The app will open in your browser (usually at `http://localhost:8501`)

## How to Use

1. **Load Model**:

   - Option 1: Check "Use default model path" if `best_attention_unet.pth` is in the current directory
   - Option 2: Upload your model file using the file uploader
   - Click "🔄 Load Model" button

2. **Upload Depth Map**:

   - Click "Browse files" and select a `.npy` depth map file
   - The input depth map will be displayed

3. **Run Inference**:

   - Adjust the binary threshold slider if needed (default: 0.5)
   - Click "🚀 Run Inference" button
   - View the binary segmentation mask and probability map

4. **Download Results**:
   - Download the binary mask as `.npy` file
   - Download the probability map as `.npy` file

## Model Information

The UI displays:

- Model architecture details
- Total and trainable parameters
- Model file size
- Current device (CPU/GPU)

## Input Format

- **Depth Map**: `.npy` file containing a 2D numpy array (H, W) with depth values
- The model expects depth maps with finite positive values

## Output Format

- **Binary Mask**: `.npy` file with values 0 (background) and 255 (dent regions)
- **Probability Map**: `.npy` file with probability values between 0 and 1

## Troubleshooting

- **Model not loading**: Make sure the model file is compatible with the model architecture
- **CUDA errors**: The app will automatically use CPU if CUDA is not available
- **File upload issues**: Ensure the `.npy` file is a valid 2D numpy array

## Notes

- The model uses GPU if available, otherwise falls back to CPU
- Inference is optimized for batch size 1
- The preprocessing pipeline matches the training pipeline exactly
