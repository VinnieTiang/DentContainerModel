"""
Flask API Server for Dent Container Detection
Provides REST API endpoints for the JavaScript UI
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import numpy as np
import cv2
import os
import json
import tempfile
import uuid
from pathlib import Path
from datetime import datetime
import io
from PIL import Image
from model_architecture import AttentionUNet, predict_mask, preprocess_depth, create_dent_overlay, calculate_dent_metrics
from panel_extractor import RANSACPanelExtractor, DEFAULT_CLOSING_KERNEL_SIZE, DEFAULT_CAMERA_FOV, DEFAULT_DOWNSAMPLE_FACTOR, clean_depth_map_uint16
# Thread-safe matplotlib imports
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Rectangle

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj

# Global state
model = None
model_loaded = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
default_model_path = "best_attention_unet_4.pth"

# Container storage (in production, use a database)
containers = {}
container_counter = 1

# Create temp directory for file storage
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'device': str(device),
        'model_loaded': model_loaded
    })


@app.route('/api/model/upload', methods=['POST'])
def upload_model():
    """Upload a model file"""
    try:
        if 'model_file' not in request.files:
            return jsonify({'error': 'No model file uploaded'}), 400
        
        model_file = request.files['model_file']
        
        if model_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not model_file.filename.endswith('.pth'):
            return jsonify({'error': 'Model file must be a .pth file'}), 400
        
        # Save uploaded model to temp directory
        model_filename = f"uploaded_{model_file.filename}"
        model_path = TEMP_DIR / model_filename
        model_file.save(str(model_path))
        
        return jsonify({
            'success': True,
            'message': 'Model file uploaded successfully',
            'model_path': str(model_path)
        })
    except Exception as e:
        return jsonify({'error': f'Error uploading model: {str(e)}'}), 500


@app.route('/api/model/load', methods=['POST'])
def load_model():
    """Load the PyTorch model"""
    global model, model_loaded
    
    try:
        data = request.json
        model_path = data.get('model_path', default_model_path)
        
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model file not found: {model_path}'}), 404
        
        # Initialize model
        model = AttentionUNet(in_channels=3, out_channels=1, filters=[32, 64, 128, 256])
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        
        model_loaded = True
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        
        return jsonify({
            'success': True,
            'message': 'Model loaded successfully',
            'model_path': model_path,  # Return the actual model path being used
            'model_info': {
                'architecture': 'Attention-UNet',
                'input_channels': 3,
                'output_channels': 1,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': file_size,
                'device': str(device)
            }
        })
    except Exception as e:
        return jsonify({'error': f'Error loading model: {str(e)}'}), 500


@app.route('/api/containers/create', methods=['POST'])
def create_container():
    """Create a new container"""
    global container_counter
    
    try:
        data = request.json
        container_name = data.get('name', f'CONTAINER-{container_counter:04d}')
        
        container_id = f'CONTAINER-{container_counter:04d}'
        container_counter += 1
        
        containers[container_id] = {
            'id': container_id,
            'name': container_name,
            'created_at': datetime.now().isoformat(),
            'panels': {
                'back': {'uploads': []},
                'left': {'uploads': []},
                'right': {'uploads': []},
                'roof': {'uploads': []},
                'door': {'uploads': []}
            },
            'results': {}
        }
        
        return jsonify({
            'success': True,
            'container_id': container_id,
            'container': containers[container_id]
        })
    except Exception as e:
        return jsonify({'error': f'Error creating container: {str(e)}'}), 500


@app.route('/api/containers/<container_id>', methods=['GET'])
def get_container(container_id):
    """Get container information"""
    if container_id not in containers:
        return jsonify({'error': 'Container not found'}), 404
    
    # Create a deep copy to modify for display without changing the original data
    import copy
    container_data = copy.deepcopy(containers[container_id])
    
    # Flatten 'uploads' for frontend compatibility
    # This ensures the UI can access panelData.depth_filename directly
    for panel_name, panel_data in container_data['panels'].items():
        if panel_data and 'uploads' in panel_data and panel_data['uploads']:
            # Get the most recent upload
            latest_upload = panel_data['uploads'][-1]
            
            # Merge latest upload keys into the panel_data object
            # This makes panelData.depth_filename valid again for backward compatibility
            panel_data.update(latest_upload)
            
            # Keep the uploads list so the UI can access it if needed
            # panel_data['uploads'] remains accessible
    
    return jsonify({
        'success': True,
        'container': container_data
    })


@app.route('/api/containers', methods=['GET'])
def list_containers():
    """List all containers"""
    import copy
    
    # Create deep copies and flatten uploads for frontend compatibility
    containers_list = []
    for container in containers.values():
        container_copy = copy.deepcopy(container)
        
        # Flatten 'uploads' for each panel
        for panel_name, panel_data in container_copy['panels'].items():
            if panel_data and 'uploads' in panel_data and panel_data['uploads']:
                # Get the most recent upload
                latest_upload = panel_data['uploads'][-1]
                # Merge latest upload keys into the panel_data object
                panel_data.update(latest_upload)
        
        containers_list.append(container_copy)
    
    return jsonify({
        'success': True,
        'containers': containers_list
    })


@app.route('/api/containers/<container_id>/panels/<panel_name>/upload', methods=['POST'])
def upload_panel_files(container_id, panel_name):
    """Upload depth map and RGB image for a panel"""
    if container_id not in containers:
        return jsonify({'error': 'Container not found'}), 404
    
    if panel_name not in ['back', 'left', 'right', 'roof', 'door']:
        return jsonify({'error': 'Invalid panel name'}), 400
    
    try:
        # Check if files are uploaded
        if 'depth_file' not in request.files:
            return jsonify({'error': 'No depth file uploaded'}), 400
        
        depth_file = request.files['depth_file']
        rgb_file = request.files.get('rgb_file')
        
        # Generate unique upload ID
        upload_id = str(uuid.uuid4())
        
        # Save files to disk with unique ID
        depth_path = TEMP_DIR / f"{container_id}_{panel_name}_{upload_id}_depth.npy"
        rgb_path = TEMP_DIR / f"{container_id}_{panel_name}_{upload_id}_rgb.png" if rgb_file else None
        
        # Save depth file
        depth_file.save(str(depth_path))
        depth_data = np.load(str(depth_path))
        
        # Process depth map
        original_dtype = depth_data.dtype
        is_uint16 = depth_data.dtype == np.uint16 or depth_data.dtype == np.uint32
        
        if is_uint16:
            depth_map = clean_depth_map_uint16(depth_data, apply_scale_factor=True)
        else:
            if depth_data.dtype == np.float16:
                depth_map = depth_data.astype(np.float32)
            else:
                depth_map = depth_data.astype(np.float32)
        
        # Save processed depth map with unique ID
        processed_depth_path = TEMP_DIR / f"{container_id}_{panel_name}_{upload_id}_depth_processed.npy"
        np.save(str(processed_depth_path), depth_map)
        
        # Process RGB image if provided
        rgb_shape = None
        rgb_array = None
        rgb_cropped_path = None
        rgb_cropped_array = None
        if rgb_file:
            # Save RGB file (original)
            rgb_file.save(str(rgb_path))
            rgb_image = Image.open(str(rgb_path))
            rgb_array = np.array(rgb_image)
            if rgb_array.shape[2] == 4:
                rgb_array = rgb_array[:, :, :3]
            rgb_shape = rgb_array.shape

            # If uint16 depth map, also crop RGB using same crop coordinates
            if is_uint16:
                H_rgb, W_rgb = rgb_array.shape[:2]
                left_crop = int(0.20 * W_rgb)
                right_crop = int(0.80 * W_rgb)
                top_crop = int(0.1 * H_rgb)
                bottom_crop = int(0.9 * H_rgb)

                # Crop RGB image
                rgb_cropped_array = rgb_array[top_crop:bottom_crop, left_crop:right_crop].copy()

                # Save cropped RGB with unique ID
                rgb_cropped_path = TEMP_DIR / f"{container_id}_{panel_name}_{upload_id}_rgb_cropped.png"
                rgb_cropped_image = Image.fromarray(rgb_cropped_array)
                rgb_cropped_image.save(str(rgb_cropped_path))
        
        # Calculate depth map statistics
        valid_pixels = np.sum((depth_map > 0) & np.isfinite(depth_map))
        total_pixels = depth_map.size
        valid_percentage = (valid_pixels / total_pixels * 100) if total_pixels > 0 else 0
        min_depth = float(np.nanmin(depth_map[depth_map > 0])) if valid_pixels > 0 else 0.0
        max_depth = float(np.nanmax(depth_map[depth_map > 0])) if valid_pixels > 0 else 0.0
        median_depth = float(np.nanmedian(depth_map[depth_map > 0])) if valid_pixels > 0 else 0.0
        
        # Get RANSAC preference from request (if provided)
        use_ransac_preference = request.form.get('use_ransac', 'true').lower() == 'true'
        force_rectangular_mask_preference = request.form.get('force_rectangular_mask', 'true').lower() == 'true'
        
        # Initialize uploads array if it doesn't exist
        if 'uploads' not in containers[container_id]['panels'][panel_name]:
            containers[container_id]['panels'][panel_name]['uploads'] = []
        
        # Create upload entry
        upload_entry = {
            'upload_id': upload_id,
            'depth_path': str(processed_depth_path),
            'depth_filename': depth_file.filename if hasattr(depth_file, 'filename') else 'depth_map.npy',
            'depth_shape': list(depth_map.shape),
            'depth_dtype': str(depth_map.dtype),
            'original_dtype': str(original_dtype),
            'is_uint16': is_uint16,
            'rgb_path': str(rgb_path) if rgb_path else None,
            'rgb_cropped_path': str(rgb_cropped_path) if rgb_cropped_path else None,
            'rgb_filename': rgb_file.filename if rgb_file and hasattr(rgb_file, 'filename') else None,
            'rgb_shape': list(rgb_shape) if rgb_shape else None,
            'use_ransac': use_ransac_preference,
            'force_rectangular_mask': force_rectangular_mask_preference,
            'uploaded_at': datetime.now().isoformat(),
            'depth_stats': {
                'valid_pixels': int(valid_pixels),
                'total_pixels': int(total_pixels),
                'valid_percentage': float(valid_percentage),
                'min_depth': min_depth,
                'max_depth': max_depth,
                'median_depth': median_depth
            }
        }
        
        # Append to uploads array
        containers[container_id]['panels'][panel_name]['uploads'].append(upload_entry)
        
        # Also update the top-level panel object for backward compatibility with UI
        # This ensures the UI sees the "current" active file at panelData.depth_filename
        containers[container_id]['panels'][panel_name].update(upload_entry)
        
        return jsonify({
            'success': True,
            'message': f'Files uploaded for {panel_name} panel',
            'upload_id': upload_id,
            'depth_filename': depth_file.filename if hasattr(depth_file, 'filename') else 'depth_map.npy',
            'rgb_filename': rgb_file.filename if rgb_file and hasattr(rgb_file, 'filename') else None,
            'depth_shape': list(depth_map.shape),
            'depth_dtype': str(depth_map.dtype),
            'original_dtype': str(original_dtype),
            'is_uint16': is_uint16,
            'rgb_shape': list(rgb_shape) if rgb_shape else None,
            'depth_stats': {
                'valid_pixels': int(valid_pixels),
                'total_pixels': int(total_pixels),
                'valid_percentage': float(valid_percentage),
                'min_depth': min_depth,
                'max_depth': max_depth,
                'median_depth': median_depth
            }
        })
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Error uploading files: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/containers/<container_id>/panels/<panel_name>/update-ransac', methods=['POST'])
def update_ransac_preference(container_id, panel_name):
    """Update RANSAC preference for a specific upload"""
    if container_id not in containers:
        return jsonify({'error': 'Container not found'}), 404
    
    if panel_name not in ['back', 'left', 'right', 'roof', 'door']:
        return jsonify({'error': 'Invalid panel name'}), 400
    
    panel_data = containers[container_id]['panels'].get(panel_name)
    if not panel_data or 'uploads' not in panel_data or not panel_data['uploads']:
        return jsonify({'error': f'No data uploaded for {panel_name} panel'}), 400
    
    try:
        data = request.json or {}
        upload_id = data.get('upload_id')  # Optional, defaults to most recent
        use_ransac = data.get('use_ransac', True)
        force_rectangular_mask = data.get('force_rectangular_mask', True)
        
        # Get upload data
        upload_data = get_upload_data(container_id, panel_name, upload_id)
        if upload_data is None:
            return jsonify({'error': f'Upload not found'}), 404
        
        # Update RANSAC preference for this upload
        upload_data['use_ransac'] = use_ransac
        upload_data['force_rectangular_mask'] = force_rectangular_mask
        
        return jsonify({
            'success': True,
            'message': f'RANSAC preference updated for {panel_name} panel',
            'upload_id': upload_data['upload_id'],
            'use_ransac': use_ransac,
            'force_rectangular_mask': force_rectangular_mask
        })
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Error updating RANSAC preference: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


def get_upload_data(container_id, panel_name, upload_id=None):
    """Helper function to get upload data. If upload_id is None, returns the most recent upload."""
    if container_id not in containers:
        return None
    
    panel_data = containers[container_id]['panels'].get(panel_name)
    if not panel_data or 'uploads' not in panel_data:
        return None
    
    uploads = panel_data['uploads']
    if not uploads:
        return None
    
    if upload_id:
        # Find specific upload
        for upload in uploads:
            if upload['upload_id'] == upload_id:
                return upload
        return None
    else:
        # Return most recent upload (last in array)
        return uploads[-1]


@app.route('/api/containers/<container_id>/panels/<panel_name>/process', methods=['POST'])
def process_panel(container_id, panel_name):
    """Process a panel (run inference). Accepts optional upload_id to specify which upload to process."""
    if container_id not in containers:
        return jsonify({'error': 'Container not found'}), 404
    
    if panel_name not in ['back', 'left', 'right', 'roof', 'door']:
        return jsonify({'error': 'Invalid panel name'}), 400
    
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 400
    
    try:
        # Get processing parameters
        data = request.json or {}
        upload_id = data.get('upload_id')  # Optional upload_id parameter
        threshold = data.get('threshold', 0.5)
        use_ransac = data.get('use_ransac', True)
        camera_fov = data.get('camera_fov', DEFAULT_CAMERA_FOV)
        adaptive_threshold = data.get('adaptive_threshold', True)
        residual_threshold = data.get('residual_threshold', None)
        downsample_factor = data.get('downsample_factor', DEFAULT_DOWNSAMPLE_FACTOR)
        apply_morphological_closing = data.get('apply_morphological_closing', True)
        closing_kernel_size = data.get('closing_kernel_size', DEFAULT_CLOSING_KERNEL_SIZE)
        force_rectangular_mask = data.get('force_rectangular_mask', True)
        max_area_threshold = data.get('max_area_threshold', 100.0)
        max_depth_threshold = data.get('max_depth_threshold', 35.0)
        overlay_alpha = data.get('overlay_alpha', 0.2)
        outline_thickness = data.get('outline_thickness', 2)
        intrinsics_json_path = data.get('intrinsics_json_path', None)
        
        # Get upload data (use upload_id if provided, otherwise use most recent)
        upload_data = get_upload_data(container_id, panel_name, upload_id)
        if upload_data is None:
            return jsonify({'error': f'No upload data found for {panel_name} panel' + (f' with upload_id {upload_id}' if upload_id else '')}), 400
        
        # Use RANSAC preferences from upload if not overridden
        if use_ransac is True:  # Only use upload preference if use_ransac wasn't explicitly set
            use_ransac = upload_data.get('use_ransac', True)
        if force_rectangular_mask is True:  # Only use upload preference if not explicitly set
            force_rectangular_mask = upload_data.get('force_rectangular_mask', True)
        
        # Load depth map from file
        depth_map = np.load(upload_data['depth_path'])
        
        # Load RGB image if available
        # Prefer cropped RGB if available (for uint16 depth maps) to match cropped depth dimensions
        rgb_array = None
        rgb_cropped_path = upload_data.get('rgb_cropped_path')
        if rgb_cropped_path and os.path.exists(rgb_cropped_path):
            rgb_image = Image.open(rgb_cropped_path)
            rgb_array = np.array(rgb_image)
            if rgb_array.shape[2] == 4:
                rgb_array = rgb_array[:, :, :3]
        elif upload_data.get('rgb_path'):
            rgb_image = Image.open(upload_data['rgb_path'])
            rgb_array = np.array(rgb_image)
            if rgb_array.shape[2] == 4:
                rgb_array = rgb_array[:, :, :3]
        
        # Apply RANSAC if enabled
        depth_cleaned_precomputed = None
        panel_mask_precomputed = None
        ransac_stats = None
        plane_coefficients = None
        
        if use_ransac:
            extractor = RANSACPanelExtractor(
                camera_fov=camera_fov,
                residual_threshold=residual_threshold if residual_threshold else 0.02,
                adaptive_threshold=adaptive_threshold,
                downsample_factor=downsample_factor,
                apply_morphological_closing=apply_morphological_closing,
                closing_kernel_size=closing_kernel_size,
                force_rectangular_mask=force_rectangular_mask
            )
            cleaned_depth, panel_mask, ransac_stats, plane_coefficients = extractor.extract_panel(
                depth_map,
                fill_background=False  # No median filling - use 0-background for AI model
            )
            depth_cleaned_precomputed = cleaned_depth
            panel_mask_precomputed = panel_mask
        
        # Run inference
        binary_mask, prob_mask = predict_mask(
            model,
            depth_map,
            device=str(device),
            threshold=threshold,
            depth_cleaned=depth_cleaned_precomputed,
            panel_mask=panel_mask_precomputed
        )
        
        # Calculate metrics
        depth_for_metrics = depth_cleaned_precomputed if depth_cleaned_precomputed is not None else depth_map
        depth_units_for_metrics = 'meters'
        
        metrics_kwargs = {
            'depth_map': depth_for_metrics,
            'dent_mask': binary_mask,
            'pixel_to_cm': None,
            'intrinsics_json_path': intrinsics_json_path,
            'depth_units': depth_units_for_metrics
        }
        
        if use_ransac:
            metrics_kwargs['camera_fov'] = camera_fov
            if panel_mask_precomputed is not None:
                metrics_kwargs['panel_mask'] = panel_mask_precomputed
            if plane_coefficients is not None:
                metrics_kwargs['plane_coefficients'] = plane_coefficients
        
        metrics = calculate_dent_metrics(**metrics_kwargs)
        
        # Determine status
        has_dents = metrics['num_defects'] > 0
        if metrics['area_valid'] and metrics['area_cm2'] is not None:
            area_pass = metrics['area_cm2'] <= max_area_threshold
        else:
            area_pass = True
        
        depth_pass = metrics['max_depth_mm'] <= max_depth_threshold
        
        # Check if Max Depth is less than 35mm - set to PASS with note
        note = None
        if metrics['max_depth_mm'] is not None and metrics['max_depth_mm'] < 35.0:
            status = "PASS"
            note = "Pass with minor dent"
        else:
            status = "PASS" if (not has_dents or (area_pass and depth_pass)) else "FAIL"
        
        # Create overlay if RGB image available
        overlay_image = None
        if rgb_array is not None:
            overlay_image = create_dent_overlay(
                rgb_array,
                binary_mask,
                overlay_alpha=overlay_alpha,
                outline_thickness=outline_thickness
            )
        
        # Save results to disk with upload_id
        upload_id_used = upload_data['upload_id']
        mask_path = TEMP_DIR / f"{container_id}_{panel_name}_{upload_id_used}_mask.npy"
        prob_path = TEMP_DIR / f"{container_id}_{panel_name}_{upload_id_used}_prob.npy"
        overlay_path = TEMP_DIR / f"{container_id}_{panel_name}_{upload_id_used}_overlay.png" if overlay_image is not None else None
        
        np.save(str(mask_path), binary_mask)
        np.save(str(prob_path), prob_mask)
        
        if overlay_image is not None:
            overlay_pil = Image.fromarray(overlay_image)
            overlay_pil.save(str(overlay_path))
        
        # Store file paths in results
        # Convert metrics and ransac_stats to JSON-serializable format
        metrics_stored = convert_numpy_types(metrics)
        ransac_stats_stored = convert_numpy_types(ransac_stats) if ransac_stats is not None else None
        
        result = {
            'upload_id': upload_id_used,
            'timestamp': datetime.now().isoformat(),
            'threshold': float(threshold),
            'mask_path': str(mask_path),
            'prob_path': str(prob_path),
            'overlay_path': str(overlay_path) if overlay_path else None,
            'binary_mask_shape': list(binary_mask.shape),
            'prob_mask_shape': list(prob_mask.shape),
            'metrics': metrics_stored,
            'status': status,
            'note': note,
            'ransac_stats': ransac_stats_stored,
            'overlay_shape': list(overlay_image.shape) if overlay_image is not None else None
        }
        
        # Store results keyed by panel_name (can have multiple results per panel if multiple uploads)
        if panel_name not in containers[container_id]['results']:
            containers[container_id]['results'][panel_name] = {}
        containers[container_id]['results'][panel_name][upload_id_used] = result
        
        # Prepare response (avoid sending full arrays)
        # Convert numpy types to native Python types
        metrics_clean = {
            'num_defects': int(metrics['num_defects']) if metrics['num_defects'] is not None else 0,
            'area_cm2': float(metrics['area_cm2']) if metrics['area_valid'] and metrics['area_cm2'] is not None else None,
            'area_valid': bool(metrics['area_valid']),
            'max_depth_mm': float(metrics['max_depth_mm']) if metrics['max_depth_mm'] is not None else 0.0,
            'avg_depth_mm': float(metrics['avg_depth_mm']) if metrics['avg_depth_mm'] is not None else 0.0,
            'pixel_count': int(metrics['pixel_count']) if metrics['pixel_count'] is not None else 0
        }
        
        # Convert ransac_stats if it exists
        ransac_stats_clean = None
        if ransac_stats is not None:
            ransac_stats_clean = convert_numpy_types(ransac_stats)
        
        response = {
            'success': True,
            'message': f'Processing completed for {panel_name} panel',
            'status': status,
            'note': note,
            'metrics': metrics_clean,
            'ransac_stats': ransac_stats_clean,
            'has_overlay': overlay_image is not None
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Error processing panel: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/containers/<container_id>/panels/<panel_name>/results/mask', methods=['GET'])
def get_mask(container_id, panel_name):
    """Download binary mask as .npy file. Accepts optional upload_id query parameter."""
    if container_id not in containers:
        return jsonify({'error': 'Container not found'}), 404
    
    upload_id = request.args.get('upload_id')  # Optional query parameter
    
    if panel_name not in containers[container_id]['results']:
        return jsonify({'error': 'No results available for this panel'}), 404
    
    results = containers[container_id]['results'][panel_name]
    
    # If upload_id provided, get specific result; otherwise get most recent
    if upload_id:
        result = results.get(upload_id) if isinstance(results, dict) else None
    else:
        # Get most recent result (last upload_id in dict)
        if isinstance(results, dict) and results:
            result = list(results.values())[-1]
        elif isinstance(results, dict):
            result = None
        else:
            # Legacy format (single result)
            result = results
    
    if not result:
        return jsonify({'error': 'No results available for this upload'}), 404
    
    mask_path = result['mask_path']
    
    if not os.path.exists(mask_path):
        return jsonify({'error': 'Mask file not found'}), 404
    
    # Get filename from upload data
    upload_data = get_upload_data(container_id, panel_name, upload_id or result.get('upload_id'))
    depth_filename = upload_data.get('depth_filename', 'depth_map.npy') if upload_data else 'depth_map.npy'
    base_filename = depth_filename.replace('.npy', '')
    
    return send_file(
        mask_path,
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name=f'{base_filename}_mask.npy'
    )


@app.route('/api/containers/<container_id>/panels/<panel_name>/results/prob', methods=['GET'])
def get_prob_mask(container_id, panel_name):
    """Download probability mask as .npy file. Accepts optional upload_id query parameter."""
    if container_id not in containers:
        return jsonify({'error': 'Container not found'}), 404
    
    upload_id = request.args.get('upload_id')  # Optional query parameter
    
    if panel_name not in containers[container_id]['results']:
        return jsonify({'error': 'No results available for this panel'}), 404
    
    results = containers[container_id]['results'][panel_name]
    
    # If upload_id provided, get specific result; otherwise get most recent
    if upload_id:
        result = results.get(upload_id) if isinstance(results, dict) else None
    else:
        # Get most recent result (last upload_id in dict)
        if isinstance(results, dict) and results:
            result = list(results.values())[-1]
        elif isinstance(results, dict):
            result = None
        else:
            # Legacy format (single result)
            result = results
    
    if not result:
        return jsonify({'error': 'No results available for this upload'}), 404
    
    prob_path = result['prob_path']
    
    if not os.path.exists(prob_path):
        return jsonify({'error': 'Probability mask file not found'}), 404
    
    # Get filename from upload data
    upload_data = get_upload_data(container_id, panel_name, upload_id or result.get('upload_id'))
    depth_filename = upload_data.get('depth_filename', 'depth_map.npy') if upload_data else 'depth_map.npy'
    base_filename = depth_filename.replace('.npy', '')
    
    return send_file(
        prob_path,
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name=f'{base_filename}_prob.npy'
    )


@app.route('/api/intrinsics/upload', methods=['POST'])
def upload_intrinsics():
    """Upload camera intrinsics JSON file"""
    if 'intrinsics_file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['intrinsics_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.json'):
        return jsonify({'error': 'File must be a JSON file'}), 400
    
    try:
        # Save file temporarily
        import tempfile
        
        # Validate JSON
        file_content = file.read()
        file.seek(0)  # Reset file pointer
        
        try:
            json.loads(file_content)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid JSON file'}), 400
        
        # Save to temp directory
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json', dir=str(TEMP_DIR))
        temp_file.write(file_content)
        temp_file.close()
        
        intrinsics_path = temp_file.name
        
        return jsonify({
            'success': True,
            'intrinsics_path': intrinsics_path,
            'filename': file.filename
        })
    except Exception as e:
        return jsonify({'error': f'Error uploading file: {str(e)}'}), 500


@app.route('/api/containers/<container_id>/panels/<panel_name>/rgb', methods=['GET'])
def get_rgb_image(container_id, panel_name):
    """Get RGB image as PNG. Accepts optional upload_id query parameter."""
    if container_id not in containers:
        return jsonify({'error': 'Container not found'}), 404
    
    upload_id = request.args.get('upload_id')  # Optional query parameter
    
    upload_data = get_upload_data(container_id, panel_name, upload_id)
    if upload_data is None:
        return jsonify({'error': 'Upload not found'}), 404
    
    # Prefer cropped RGB if available
    rgb_path = upload_data.get('rgb_cropped_path') or upload_data.get('rgb_path')
    if not rgb_path or not os.path.exists(rgb_path):
        return jsonify({'error': 'RGB image not available'}), 404
    
    return send_file(
        rgb_path,
        mimetype='image/png',
        as_attachment=False
    )


@app.route('/api/containers/<container_id>/panels/<panel_name>/results/overlay', methods=['GET'])
def get_overlay(container_id, panel_name):
    """Download overlay image as PNG. Accepts optional upload_id query parameter."""
    if container_id not in containers:
        return jsonify({'error': 'Container not found'}), 404
    
    upload_id = request.args.get('upload_id')  # Optional query parameter
    
    if panel_name not in containers[container_id]['results']:
        return jsonify({'error': 'No results available for this panel'}), 404
    
    results = containers[container_id]['results'][panel_name]
    
    # If upload_id provided, get specific result; otherwise get most recent
    if upload_id:
        result = results.get(upload_id) if isinstance(results, dict) else None
    else:
        # Get most recent result (last upload_id in dict)
        if isinstance(results, dict) and results:
            result = list(results.values())[-1]
        elif isinstance(results, dict):
            result = None
        else:
            # Legacy format (single result)
            result = results
    
    if not result:
        return jsonify({'error': 'No results available for this upload'}), 404
    
    overlay_path = result.get('overlay_path')
    
    if not overlay_path or not os.path.exists(overlay_path):
        return jsonify({'error': 'No overlay image available'}), 404
    
    # Get filename from upload data
    upload_data = get_upload_data(container_id, panel_name, upload_id or result.get('upload_id'))
    depth_filename = upload_data.get('depth_filename', 'depth_map.npy') if upload_data else 'depth_map.npy'
    base_filename = depth_filename.replace('.npy', '')
    
    return send_file(
        overlay_path,
        mimetype='image/png',
        as_attachment=True,
        download_name=f'{base_filename}_overlay.png'
    )


@app.route('/api/containers/<container_id>/panels/<panel_name>/ransac/preview', methods=['POST'])
def get_ransac_preview(container_id, panel_name):
    """Thread-safe RANSAC preview generation. Accepts optional upload_id in request body."""
    if container_id not in containers:
        return jsonify({'error': 'Container not found'}), 404
    
    try:
        import base64
        
        # Get parameters from request
        data = request.json or {}
        upload_id = data.get('upload_id')  # Optional upload_id parameter
        camera_fov = data.get('camera_fov', DEFAULT_CAMERA_FOV)
        adaptive_threshold = data.get('adaptive_threshold', True)
        residual_threshold = data.get('residual_threshold', None)
        downsample_factor = data.get('downsample_factor', DEFAULT_DOWNSAMPLE_FACTOR)
        apply_morphological_closing = data.get('apply_morphological_closing', True)
        closing_kernel_size = data.get('closing_kernel_size', DEFAULT_CLOSING_KERNEL_SIZE)
        force_rectangular_mask = data.get('force_rectangular_mask', True)
        
        # Get upload data
        upload_data = get_upload_data(container_id, panel_name, upload_id)
        if upload_data is None:
            return jsonify({'error': 'Upload not found'}), 404
        
        # Load depth map
        depth_map = np.load(upload_data['depth_path'])
        
        # Apply RANSAC
        extractor = RANSACPanelExtractor(
            camera_fov=camera_fov,
            residual_threshold=residual_threshold if residual_threshold else 0.02,
            adaptive_threshold=adaptive_threshold,
            downsample_factor=downsample_factor,
            apply_morphological_closing=apply_morphological_closing,
            closing_kernel_size=closing_kernel_size,
            force_rectangular_mask=force_rectangular_mask
        )
        cleaned_depth, panel_mask, ransac_stats, plane_coefficients = extractor.extract_panel(
            depth_map,
            fill_background=False  # No median filling - use 0-background
        )
        
        # --- THREAD-SAFE PLOTTING START ---
        # Use Figure directly instead of plt.subplots()
        fig = Figure(figsize=(16, 8))
        canvas = FigureCanvas(fig)
        
        # Add subplots
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        
        # Plot Panel Mask
        im1 = ax1.imshow(panel_mask, cmap='gray', vmin=0, vmax=1)
        ax1.set_title("Panel Mask (White = Panel)")
        ax1.axis('off')
        fig.colorbar(im1, ax=ax1, fraction=0.046)
        
        # Plot Cleaned Depth
        im2 = ax2.imshow(cleaned_depth, cmap='viridis')
        ax2.set_title("Cleaned Depth Map (Background Filled)")
        ax2.axis('off')
        fig.colorbar(im2, ax=ax2, fraction=0.046)
        
        # Save to buffer
        buffer = io.BytesIO()
        fig.savefig(buffer, format='PNG', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        preview_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        # --- THREAD-SAFE PLOTTING END ---
        
        # Convert stats to JSON-serializable
        stats_clean = convert_numpy_types(ransac_stats)
        
        return jsonify({
            'success': True,
            'preview_image': preview_image,
            'stats': stats_clean
        })
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Error generating RANSAC preview: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/containers/<container_id>/panels/<panel_name>/depth/preview', methods=['GET'])
def get_depth_preview(container_id, panel_name):
    """Thread-safe depth preview generation. Accepts optional upload_id query parameter."""
    if container_id not in containers:
        return jsonify({'error': 'Container not found'}), 404
    
    upload_id = request.args.get('upload_id')  # Optional query parameter
    
    upload_data = get_upload_data(container_id, panel_name, upload_id)
    if upload_data is None:
        return jsonify({'error': 'Upload not found'}), 404
    
    try:
        import base64
        
        # Load depth map
        depth_map = np.load(upload_data['depth_path'])
        original_dtype = upload_data.get('original_dtype', str(depth_map.dtype))
        is_uint16 = upload_data.get('is_uint16', False)
        
        previews = {}
        
        # --- THREAD-SAFE PLOT 1: Processed Depth ---
        fig = Figure(figsize=(8, 8))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        im = ax.imshow(depth_map, cmap='viridis')
        ax.set_title("Input Depth Map")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046)
        
        buffer = io.BytesIO()
        fig.savefig(buffer, format='PNG', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        previews['processed_depth'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # --- THREAD-SAFE PLOT 2: Raw Depth (if uint16) ---
        if is_uint16:
            upload_id_used = upload_data['upload_id']
            raw_depth_path = TEMP_DIR / f"{container_id}_{panel_name}_{upload_id_used}_depth.npy"
            if raw_depth_path.exists():
                raw_depth = np.load(str(raw_depth_path))
                H, W = raw_depth.shape
                left_crop = int(0.20 * W)
                right_crop = int(0.80 * W)
                top_crop = int(0.1 * H)
                bottom_crop = int(0.9 * H)
                
                fig_raw = Figure(figsize=(8, 8))
                canvas_raw = FigureCanvas(fig_raw)
                ax_raw = fig_raw.add_subplot(111)
                
                im_raw = ax_raw.imshow(raw_depth, cmap='viridis')
                
                # Draw bounding box
                crop_width = right_crop - left_crop
                crop_height = bottom_crop - top_crop
                rect = Rectangle(
                    (left_crop, top_crop),
                    crop_width,
                    crop_height,
                    linewidth=3,
                    edgecolor='red',
                    facecolor='none',
                    linestyle='--',
                    label='Crop Area'
                )
                ax_raw.add_patch(rect)
                ax_raw.set_title("Raw Depth Map (Original) - Red box indicates crop area")
                ax_raw.axis('off')
                ax_raw.legend(loc='upper right', framealpha=0.8)
                fig_raw.colorbar(im_raw, ax=ax_raw, fraction=0.046)
                
                buffer_raw = io.BytesIO()
                fig_raw.savefig(buffer_raw, format='PNG', bbox_inches='tight', dpi=100)
                buffer_raw.seek(0)
                previews['raw_depth'] = base64.b64encode(buffer_raw.getvalue()).decode('utf-8')
                
                # If RGB exists, show RGB images (RGB uses PIL which is thread-safe)
                rgb_path = upload_data.get('rgb_path')
                rgb_cropped_path = upload_data.get('rgb_cropped_path')
                if rgb_path and os.path.exists(rgb_path):
                    # Load original RGB
                    rgb_image = Image.open(rgb_path)
                    rgb_array = np.array(rgb_image)
                    if rgb_array.shape[2] == 4:
                        rgb_array = rgb_array[:, :, :3]

                    H_rgb, W_rgb = rgb_array.shape[:2]
                    left_crop_rgb = int(0.20 * W_rgb)
                    right_crop_rgb = int(0.80 * W_rgb)
                    top_crop_rgb = int(0.1 * H_rgb)
                    bottom_crop_rgb = int(0.9 * H_rgb)

                    # Show original RGB with crop area if cropped version exists
                    fig_rgb_orig = Figure(figsize=(8, 8))
                    canvas_rgb_orig = FigureCanvas(fig_rgb_orig)
                    ax_rgb_orig = fig_rgb_orig.add_subplot(111)
                    ax_rgb_orig.imshow(rgb_array)

                    if rgb_cropped_path and os.path.exists(rgb_cropped_path):
                        # Draw bounding box to indicate crop area
                        crop_width_rgb = right_crop_rgb - left_crop_rgb
                        crop_height_rgb = bottom_crop_rgb - top_crop_rgb
                        rect_rgb = Rectangle(
                            (left_crop_rgb, top_crop_rgb),
                            crop_width_rgb,
                            crop_height_rgb,
                            linewidth=3,
                            edgecolor='red',
                            facecolor='none',
                            linestyle='--',
                            label='Crop Area'
                        )
                        ax_rgb_orig.add_patch(rect_rgb)
                        ax_rgb_orig.set_title("RGB Image (Original) - Red box indicates crop area")
                        ax_rgb_orig.legend(loc='upper right', framealpha=0.8)
                    else:
                        ax_rgb_orig.set_title("RGB Image (Original)")

                    ax_rgb_orig.axis('off')

                    buffer_rgb_orig = io.BytesIO()
                    fig_rgb_orig.savefig(buffer_rgb_orig, format='PNG', bbox_inches='tight', dpi=100)
                    buffer_rgb_orig.seek(0)
                    previews['rgb_original'] = base64.b64encode(buffer_rgb_orig.getvalue()).decode('utf-8')

                    # Show cropped RGB if available
                    if rgb_cropped_path and os.path.exists(rgb_cropped_path):
                        rgb_cropped_image = Image.open(rgb_cropped_path)
                        rgb_cropped_array = np.array(rgb_cropped_image)
                        if rgb_cropped_array.shape[2] == 4:
                            rgb_cropped_array = rgb_cropped_array[:, :, :3]

                        fig_rgb_cropped = Figure(figsize=(8, 8))
                        canvas_rgb_cropped = FigureCanvas(fig_rgb_cropped)
                        ax_rgb_cropped = fig_rgb_cropped.add_subplot(111)
                        ax_rgb_cropped.imshow(rgb_cropped_array)
                        ax_rgb_cropped.set_title("RGB Image (Cropped)")
                        ax_rgb_cropped.axis('off')

                        buffer_rgb_cropped = io.BytesIO()
                        fig_rgb_cropped.savefig(buffer_rgb_cropped, format='PNG', bbox_inches='tight', dpi=100)
                        buffer_rgb_cropped.seek(0)
                        previews['rgb_cropped'] = base64.b64encode(buffer_rgb_cropped.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'previews': previews
        })
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Error generating preview: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get all processed results from all containers with thumbnails"""
    history = []
    import base64
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    
    for container_id, container_data in containers.items():
        container_name = container_data.get('name', container_id)
        
        # Get all results for this container
        results = container_data.get('results', {})
        
        for panel_name, panel_results in results.items():
            if not panel_results:
                continue

            # Determine if we have a single result (legacy) or a dict of results (stacked)
            results_list = []
            
            # Check if it's the new nested structure (does NOT have 'status' at top level)
            if isinstance(panel_results, dict) and 'status' not in panel_results:
                results_list = list(panel_results.values())
            else:
                results_list = [panel_results]
            
            # Iterate through ALL results for this panel
            for result_data in results_list:
                thumbnail = None
                
                # 1. PRIORITY: RGB Overlay (if available)
                if result_data.get('overlay_path') and os.path.exists(result_data['overlay_path']):
                    try:
                        overlay_img = Image.open(result_data['overlay_path'])
                        overlay_img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                        buffer = io.BytesIO()
                        overlay_img.save(buffer, format='PNG')
                        buffer.seek(0)
                        thumbnail = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    except Exception as e:
                        print(f"Error generating RGB overlay thumbnail: {e}")
                
                # 2. PRIORITY: Depth Overlay (Generate dynamically if RGB missing)
                if not thumbnail and result_data.get('mask_path') and os.path.exists(result_data['mask_path']):
                    try:
                        # We need to find the depth file associated with this result
                        upload_id = result_data.get('upload_id')
                        depth_path = None
                        
                        # Look up depth path in panel uploads
                        panel_data = container_data['panels'].get(panel_name, {})
                        if 'uploads' in panel_data:
                            for upload in panel_data['uploads']:
                                if upload['upload_id'] == upload_id:
                                    depth_path = upload['depth_path']
                                    break
                        # Fallback to top-level if not found
                        if not depth_path:
                            depth_path = panel_data.get('depth_path')

                        if depth_path and os.path.exists(depth_path):
                            # Load data
                            depth_map = np.load(depth_path)
                            binary_mask = np.load(result_data['mask_path'])
                            
                            # Create Visualization (Thread Safe)
                            fig = Figure(figsize=(4, 4), dpi=72) # Small size for thumbnail
                            canvas = FigureCanvas(fig)
                            ax = fig.add_subplot(111)
                            
                            # Plot Depth
                            ax.imshow(depth_map, cmap='viridis', interpolation='nearest')
                            
                            # Overlay Mask (Red with alpha)
                            if binary_mask.max() > 0:
                                # Normalize mask to 0-1
                                mask_norm = (binary_mask > 0).astype(np.float32)
                                overlay = np.zeros((*mask_norm.shape, 4))
                                overlay[:, :, 0] = 1.0  # Red
                                overlay[:, :, 3] = mask_norm * 0.5  # Alpha
                                ax.imshow(overlay, interpolation='nearest')
                            
                            ax.axis('off')
                            fig.tight_layout(pad=0)
                            
                            # Save to buffer
                            buffer = io.BytesIO()
                            fig.savefig(buffer, format='PNG', bbox_inches='tight', pad_inches=0)
                            buffer.seek(0)
                            
                            # Resize to thumbnail standard
                            img_pil = Image.open(buffer)
                            img_pil.thumbnail((200, 200), Image.Resampling.LANCZOS)
                            
                            final_buffer = io.BytesIO()
                            img_pil.save(final_buffer, format='PNG')
                            final_buffer.seek(0)
                            
                            thumbnail = base64.b64encode(final_buffer.getvalue()).decode('utf-8')
                    except Exception as e:
                        print(f"Error generating depth overlay thumbnail: {e}")

                # 3. PRIORITY: Binary Mask (Fallback)
                if not thumbnail and result_data.get('mask_path') and os.path.exists(result_data['mask_path']):
                    try:
                        binary_mask = np.load(result_data['mask_path'])
                        binary_mask_img = Image.fromarray((binary_mask > 0).astype(np.uint8) * 255, mode='L')
                        binary_mask_img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                        buffer = io.BytesIO()
                        binary_mask_img.save(buffer, format='PNG')
                        buffer.seek(0)
                        thumbnail = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    except Exception as e:
                        print(f"Error generating mask thumbnail: {e}")
                
                history.append({
                    'container_id': container_id,
                    'container_name': container_name,
                    'panel_name': panel_name,
                    'upload_id': result_data.get('upload_id'),
                    'status': result_data.get('status', 'UNKNOWN'),
                    'note': result_data.get('note'),
                    'timestamp': result_data.get('timestamp', ''),
                    'metrics': result_data.get('metrics', {}),
                    'num_defects': result_data.get('metrics', {}).get('num_defects', 0),
                    'area_cm2': result_data.get('metrics', {}).get('area_cm2', 0),
                    'max_depth_mm': result_data.get('metrics', {}).get('max_depth_mm', 0),
                    'thumbnail': thumbnail
                })
    
    # Sort by timestamp (newest first)
    history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return jsonify({
        'success': True,
        'history': history,
        'total': len(history)
    })

@app.route('/api/containers/<container_id>/panels/<panel_name>/results/preview', methods=['GET'])
def get_preview(container_id, panel_name):
    """Get preview images as base64 encoded strings"""
    if container_id not in containers:
        return jsonify({'error': 'Container not found'}), 404
    
    if panel_name not in containers[container_id]['results']:
        return jsonify({'error': 'No results available for this panel'}), 404
    
    # --- BUG FIX START ---
    # We must handle the nested dictionary structure (keyed by upload_id)
    raw_results = containers[container_id]['results'][panel_name]
    upload_id = request.args.get('upload_id') # Optional query param
    
    result = None
    
    # Case 1: Legacy format (direct dictionary) - unlikely but good for safety
    if 'mask_path' in raw_results:
        result = raw_results
    # Case 2: New format (Dictionary of upload_ids)
    elif isinstance(raw_results, dict) and raw_results:
        if upload_id and upload_id in raw_results:
            result = raw_results[upload_id]
        else:
            # Get the most recent one (last item in values)
            result = list(raw_results.values())[-1]
            
    if not result:
         return jsonify({'error': 'Result data structure is empty or invalid'}), 404
         
    # Safety check: Ensure mask_path actually exists in the object
    if 'mask_path' not in result:
        return jsonify({'error': 'Result found but mask_path is missing'}), 500
    # --- BUG FIX END ---

    panel_data = containers[container_id]['panels'].get(panel_name)
    
    previews = {}
    import base64
    
    # Depth map overlay with segmentation (replaces binary mask)
    # We need to find the specific depth file used for this result
    # (Try to match depth file from the upload_id if possible, or fallback to latest)
    depth_path = None
    
    # Try to find the specific upload that generated this result
    result_upload_id = result.get('upload_id')
    
    if panel_data and 'uploads' in panel_data:
        # Find upload by ID
        for upload in panel_data['uploads']:
            if upload['upload_id'] == result_upload_id:
                depth_path = upload['depth_path']
                break
        # Fallback: use top-level depth path if specific one not found
        if not depth_path and 'depth_path' in panel_data:
             depth_path = panel_data['depth_path']
    elif panel_data and 'depth_path' in panel_data:
        # Legacy fallback
        depth_path = panel_data['depth_path']

    if os.path.exists(result['mask_path']) and depth_path and os.path.exists(depth_path):
        try:
            import matplotlib
            matplotlib.use('Agg')
            # Use Object-Oriented Interface for Thread Safety!
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            
            # Load depth map and binary mask
            depth_map = np.load(depth_path)
            binary_mask = np.load(result['mask_path'])
            
            # Normalize binary mask to 0-1 range if needed
            if binary_mask.max() > 1:
                binary_mask_normalized = (binary_mask > 0).astype(np.float32)
            else:
                binary_mask_normalized = binary_mask.astype(np.float32)
            
            # Create visualization (Thread Safe)
            fig = Figure(figsize=(10, 10))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            
            # Display depth map with viridis colormap
            depth_display = ax.imshow(depth_map, cmap='viridis', interpolation='nearest')
            
            # Overlay segmentation mask with transparency
            overlay = np.zeros((*binary_mask_normalized.shape, 4))
            overlay[:, :, 0] = 1.0  # Red channel
            overlay[:, :, 3] = binary_mask_normalized * 0.5  # Alpha channel
            
            ax.imshow(overlay, interpolation='nearest')
            
            ax.set_title("Segmentation Overlay on Depth Map", fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # Add colorbar
            fig.colorbar(depth_display, ax=ax, fraction=0.046, label='Depth (meters)')
            
            # Save to buffer
            buffer = io.BytesIO()
            fig.savefig(buffer, format='PNG', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            previews['depth_overlay'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            import traceback
            print(f"Error creating depth overlay: {e}")
            print(traceback.format_exc())
            # Fallback to binary mask if overlay creation fails
            if os.path.exists(result['mask_path']):
                binary_mask = np.load(result['mask_path'])
                binary_mask_img = Image.fromarray(binary_mask, mode='L')
                buffer = io.BytesIO()
                binary_mask_img.save(buffer, format='PNG')
                buffer.seek(0)
                previews['binary_mask'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Probability mask preview
    if os.path.exists(result['prob_path']):
        prob_mask = np.load(result['prob_path'])
        # Normalize to 0-255 for display
        prob_mask_display = (prob_mask * 255).astype(np.uint8)
        prob_mask_img = Image.fromarray(prob_mask_display, mode='L')
        buffer = io.BytesIO()
        prob_mask_img.save(buffer, format='PNG')
        buffer.seek(0)
        previews['prob_mask'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Overlay preview (RGB overlay if available)
    if result.get('overlay_path') and os.path.exists(result['overlay_path']):
        overlay_img = Image.open(result['overlay_path'])
        buffer = io.BytesIO()
        overlay_img.save(buffer, format='PNG')
        buffer.seek(0)
        previews['overlay'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return jsonify({
        'success': True,
        'previews': previews
    })

@app.route('/api/containers/<container_id>/panels/<panel_name>/uploads/<upload_id>', methods=['DELETE'])
def delete_upload(container_id, panel_name, upload_id):
    """Delete a specific upload and its physical files"""
    if container_id not in containers:
        return jsonify({'error': 'Container not found'}), 404
    
    panel_data = containers[container_id]['panels'].get(panel_name)
    if not panel_data or 'uploads' not in panel_data:
        return jsonify({'error': 'Panel data not found'}), 404

    # 1. Find the upload entry
    uploads = panel_data['uploads']
    target_upload = next((u for u in uploads if u['upload_id'] == upload_id), None)
    
    if not target_upload:
        return jsonify({'error': 'Upload ID not found'}), 404

    try:
        # 2. Delete physical files (Optional - helps save space)
        if target_upload.get('depth_path') and os.path.exists(target_upload['depth_path']):
            try: os.remove(target_upload['depth_path'])
            except: pass
            
        if target_upload.get('rgb_path') and os.path.exists(target_upload['rgb_path']):
            try: os.remove(target_upload['rgb_path'])
            except: pass

        if target_upload.get('rgb_cropped_path') and os.path.exists(target_upload['rgb_cropped_path']):
            try: os.remove(target_upload['rgb_cropped_path'])
            except: pass

        # 3. Remove from the list
        panel_data['uploads'] = [u for u in uploads if u['upload_id'] != upload_id]
        
        # 4. Update the "Latest" fallback pointer (Critical for UI consistency)
        # If we deleted the latest file, we must set the pointer to the NEW latest file
        remaining_uploads = panel_data['uploads']
        if remaining_uploads:
            # Set to the last item in the new list
            new_latest = remaining_uploads[-1]
            panel_data.update(new_latest)
        else:
            # List is empty, clear top-level keys
            keys_to_clear = ['depth_path', 'depth_filename', 'rgb_path', 'rgb_filename', 'upload_id']
            for k in keys_to_clear:
                if k in panel_data:
                    del panel_data[k]

        return jsonify({'success': True, 'message': 'Upload deleted successfully'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
if __name__ == '__main__':
    print("Starting Flask API server...")
    print(f"Device: {device}")
    print(f"Default model path: {default_model_path}")
    app.run(host='0.0.0.0', port=5009, debug=True)