# Dent Container Detection - JavaScript Web UI

A modern JavaScript-based web interface for the Shipping Container Dent Detection System with container management and multi-panel processing workflow.

## Features

- 📦 **Container Management**: Create and name containers for inspection
- 🎯 **5-Panel Processing**: Process Back, Left, Right, Roof, and Door panels
- 📁 **Drag & Drop Upload**: Easy file upload with drag-and-drop support
- 🔧 **Model Configuration**: Load and configure the PyTorch model
- ⚙️ **Advanced Settings**: RANSAC panel extraction, thresholds, and more
- 📊 **Results Visualization**: View metrics, preview images, and download results
- 💾 **Download Results**: Download binary masks, probability maps, and overlay images

## Architecture

The application consists of two parts:

1. **Backend API Server** (`api_server.py`): Flask REST API that handles:

   - Model loading and inference
   - Container management
   - File uploads and processing
   - Results generation and download

2. **Frontend Web UI** (`web_ui/`): Pure JavaScript application with:
   - HTML/CSS/JavaScript (no framework dependencies)
   - Drag-and-drop file upload
   - Real-time processing status
   - Results visualization

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

### Starting the Backend API Server

1. Make sure your trained model file (`best_attention_unet_4.pth`) is in the project directory.

2. Start the Flask API server:

```bash
python api_server.py
```

The server will start on `http://localhost:5000`

### Starting the Frontend

1. Open `web_ui/index.html` in a modern web browser, or

2. Use a simple HTTP server (recommended):

```bash
# Python 3
cd web_ui
python -m http.server 8000

# Or using Node.js (if installed)
npx http-server web_ui -p 8000
```

3. Open your browser and navigate to `http://localhost:8000`

### Using the Application

1. **Create a Container**:

   - Enter a container name (or leave blank for auto-generated name)
   - Click "Create Container"

2. **Load Model**:

   - Enter the model path (default: `best_attention_unet_4.pth`)
   - Click "Load Model"
   - Wait for confirmation

3. **Upload Files for Each Panel**:

   - Drag and drop or click to browse for depth map (.npy file)
   - Optionally upload RGB image for overlay visualization
   - Repeat for all 5 panels (Back, Left, Right, Door, Roof)

4. **Configure Settings** (optional):

   - Adjust confidence threshold
   - Set max area and depth thresholds
   - Configure RANSAC settings if needed

5. **Process Panels**:
   - Click "Process" button for each panel
   - View results including metrics, preview images, and download options

## API Endpoints

### Health Check

- `GET /api/health` - Check API server status

### Model

- `POST /api/model/load` - Load the PyTorch model
  ```json
  {
    "model_path": "best_attention_unet_4.pth"
  }
  ```

### Containers

- `POST /api/containers/create` - Create a new container
  ```json
  {
    "name": "CONTAINER-0001"
  }
  ```
- `GET /api/containers` - List all containers
- `GET /api/containers/<container_id>` - Get container details

### Panel Processing

- `POST /api/containers/<container_id>/panels/<panel_name>/upload` - Upload depth map and RGB image

  - Form data: `depth_file` (required), `rgb_file` (optional)
  - Panel names: `back`, `left`, `right`, `roof`, `door`

- `POST /api/containers/<container_id>/panels/<panel_name>/process` - Process a panel
  ```json
  {
    "threshold": 0.5,
    "use_ransac": true,
    "camera_fov": 87,
    "adaptive_threshold": true,
    "downsample_factor": 4,
    "max_area_threshold": 100.0,
    "max_depth_threshold": 35.0
  }
  ```

### Results Download

- `GET /api/containers/<container_id>/panels/<panel_name>/results/mask` - Download binary mask (.npy)
- `GET /api/containers/<container_id>/panels/<panel_name>/results/prob` - Download probability map (.npy)
- `GET /api/containers/<container_id>/panels/<panel_name>/results/overlay` - Download overlay image (.png)
- `GET /api/containers/<container_id>/panels/<panel_name>/results/preview` - Get preview images (base64)

## File Formats

- **Depth Map**: `.npy` file containing a 2D numpy array (H, W) with depth values
- **RGB Image**: Standard image formats (PNG, JPG, JPEG) for overlay visualization
- **Model File**: `.pth` PyTorch model file

## Configuration

### Model Path

Default model path is `best_attention_unet_4.pth`. You can change it in the UI or modify `default_model_path` in `api_server.py`.

### Camera Intrinsics

The system uses `camera_intrinsics_default.json` by default. You can upload a custom intrinsics file through the API (future enhancement).

## Troubleshooting

### API Server Not Running

- Make sure Flask is installed: `pip install flask flask-cors`
- Check if port 5000 is available
- Verify the model file exists

### CORS Errors

- The Flask server has CORS enabled by default
- If you encounter CORS issues, check that `flask-cors` is installed

### File Upload Issues

- Ensure files are valid `.npy` format for depth maps
- Check file size limits (default Flask limit is 16MB)
- Verify the API server is running and accessible

### Model Loading Errors

- Verify the model file path is correct
- Check that the model architecture matches `AttentionUNet`
- Ensure PyTorch and CUDA (if using GPU) are properly installed

## Differences from Streamlit UI

1. **Workflow**: Container-based workflow with named containers
2. **Multi-Panel**: Process 5 panels per container
3. **File Management**: Better file organization per container/panel
4. **API-Based**: RESTful API allows for integration with other systems
5. **Modern UI**: Pure JavaScript, no Python dependencies for frontend

## Development

### Backend Development

- Modify `api_server.py` to add new endpoints or features
- Test API endpoints using tools like Postman or curl

### Frontend Development

- Modify files in `web_ui/` directory
- No build process required - pure HTML/CSS/JavaScript
- Use browser developer tools for debugging

## Notes

- The backend stores container data in memory (not persistent)
- For production use, consider adding a database for container storage
- File uploads are processed in memory - large files may require optimization
- The API server runs in debug mode by default (change for production)
