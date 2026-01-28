"""
Dent Container Detection - Streamlit UI
Simple interface to load model and run inference on depth maps
"""
import streamlit as st
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import os
import io
from PIL import Image
from datetime import datetime
import json
import tempfile
from model_architecture import AttentionUNet, predict_mask, preprocess_depth, create_dent_overlay, calculate_dent_metrics
from panel_extractor import RANSACPanelExtractor, DEFAULT_CLOSING_KERNEL_SIZE, DEFAULT_CAMERA_FOV, DEFAULT_DOWNSAMPLE_FACTOR, clean_depth_map_uint16

# Page configuration
st.set_page_config(
    page_title="Dent Container Detection",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with dark mode support
st.markdown("""
<style>
    /* Main header styling - adapts to theme */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .main-header {
            color: #4A9EFF;
        }
    }
    
    /* Streamlit dark mode detection */
    [data-theme="dark"] .main-header {
        color: #4A9EFF;
    }
    
    /* Model info box - adapts to theme */
    .model-info {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: #f0f2f6;
        color: #262730;
    }
    
    [data-theme="dark"] .model-info {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #333;
    }
    
    /* Metric box styling */
    .metric-box {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        background-color: #e8f4f8;
    }
    
    [data-theme="dark"] .metric-box {
        background-color: #1E1E1E;
        border: 1px solid #333;
    }
    
    /* Improve text readability in dark mode */
    [data-theme="dark"] {
        color: #FAFAFA;
    }
    
    /* Style info boxes for dark mode */
    [data-theme="dark"] .stInfo {
        background-color: #1E1E1E;
        border: 1px solid #333;
    }
    
    [data-theme="dark"] .stSuccess {
        background-color: #1E3A1E;
        border: 1px solid #2A5A2A;
    }
    
    [data-theme="dark"] .stWarning {
        background-color: #3A2E1E;
        border: 1px solid #5A4A2A;
    }
    
    [data-theme="dark"] .stError {
        background-color: #3A1E1E;
        border: 1px solid #5A2A2A;
    }
    
    /* Improve sidebar in dark mode */
    [data-theme="dark"] .css-1d391kg {
        background-color: #0E1117;
    }
    
    /* Better contrast for code blocks */
    [data-theme="dark"] code {
        background-color: #1E1E1E;
        color: #D4D4D4;
    }
    
    /* Improve button visibility in dark mode */
    [data-theme="dark"] .stButton > button {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #333;
    }
    
    [data-theme="dark"] .stButton > button:hover {
        background-color: #333;
        border-color: #4A9EFF;
    }
    
    /* Improve slider visibility */
    [data-theme="dark"] .stSlider {
        color: #FAFAFA;
    }
    
    /* Smaller font size for dent metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
    }
    
    /* Make metric containers more compact */
    [data-testid="stMetricContainer"] {
        padding: 0.5rem 0.75rem !important;
    }
    
    /* Footer styling */
    .footer-text {
        text-align: center;
        padding: 1rem;
    }
    
    [data-theme="dark"] .footer-text {
        color: #AAAAAA;
    }
    
    [data-theme="light"] .footer-text {
        color: #666666;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if 'history' not in st.session_state:
    st.session_state.history = []
if 'container_id_counter' not in st.session_state:
    st.session_state.container_id_counter = 1
if 'ransac_cleaned_depth' not in st.session_state:
    st.session_state.ransac_cleaned_depth = None
if 'ransac_panel_mask' not in st.session_state:
    st.session_state.ransac_panel_mask = None
if 'ransac_stats' not in st.session_state:
    st.session_state.ransac_stats = None

# Header
st.markdown('<div class="main-header">📦 Shipping Container Dent Detection System</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Model Information and Loading
with st.sidebar:
    st.header("🔧 Model Configuration")
    
    # Model file uploader
    st.subheader("Load Model")
    model_file = st.file_uploader(
        "Upload trained model (.pth file)",
        type=['pth'],
        help="Select your trained AttentionUNet model file"
    )
    
    # Or use default path
    default_model_path = "best_attention_unet_4.pth"
    use_default = st.checkbox("Use default model path", value=True)
    
    if use_default and os.path.exists(default_model_path):
        model_path = default_model_path
        st.success(f"✅ Using default model: {default_model_path}")
    elif model_file is not None:
        # Save uploaded file temporarily
        model_path = f"temp_{model_file.name}"
        with open(model_path, "wb") as f:
            f.write(model_file.getbuffer())
        st.success(f"✅ Model uploaded: {model_file.name}")
    else:
        model_path = None
        st.warning("⚠️ Please upload a model file or use default path")
    
    # Load model button
    if st.button("🔄 Load Model", type="primary", use_container_width=True):
        if model_path and os.path.exists(model_path):
            try:
                with st.spinner("Loading model..."):
                    # Initialize model
                    model = AttentionUNet(in_channels=3, out_channels=1, filters=[32,64,128,256])
                    
                    # Load weights
                    checkpoint = torch.load(model_path, map_location=st.session_state.device)
                    model.load_state_dict(checkpoint)
                    model.to(st.session_state.device)
                    model.eval()
                    
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.session_state.model_path = model_path
                    
                    st.success("✅ Model loaded successfully!")
                    
                    # Get model file size
                    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                    st.session_state.model_size_mb = file_size
                    
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                st.session_state.model_loaded = False
        else:
            st.error("❌ Model file not found!")
    
    st.markdown("---")
    
    # Model Information Display
    if st.session_state.model_loaded:
        st.subheader("📊 Model Information")
        
        # Count parameters
        total_params = sum(p.numel() for p in st.session_state.model.parameters())
        trainable_params = sum(p.numel() for p in st.session_state.model.parameters() if p.requires_grad)
        
        st.markdown(f"""
        <div class="model-info">
            <strong>Architecture:</strong> Attention-UNet<br>
            <strong>Input Channels:</strong> 3 (Depth + Gradients)<br>
            <strong>Output Channels:</strong> 1 (Binary Mask)<br>
            <strong>Total Parameters:</strong> {total_params:,}<br>
            <strong>Trainable Parameters:</strong> {trainable_params:,}<br>
            <strong>Model Size:</strong> {st.session_state.model_size_mb:.2f} MB<br>
            <strong>Device:</strong> {st.session_state.device}<br>
        </div>
        """, unsafe_allow_html=True)
        
        # Model architecture details
        with st.expander("🏗️ Architecture Details"):
            st.markdown("""
            **Encoder-Decoder with Attention Gates:**
            - Encoder: 4 levels with filters [32, 64, 128, 256]
            - Bottleneck: 512 filters
            - Decoder: 4 levels with skip connections
            - Attention gates at each decoder level
            - Final 1x1 convolution for binary segmentation
            
            **Input Processing:**
            - Channel 1: Normalized depth map (0-1)
            - Channel 2: Normalized X-gradient (Sobel)
            - Channel 3: Normalized Y-gradient (Sobel)
            """)
    else:
        st.info("👈 Please load a model first")
    
    st.markdown("---")
    
    # Inference Settings
    st.subheader("⚙️ Inference Settings")
    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Threshold for converting probability mask to binary mask"
    )
    
    # Area calculation now uses camera intrinsics automatically - no manual calibration needed
    pixel_to_cm = None  # Not needed when using intrinsics
    
    # Pass/Fail threshold settings
    st.markdown("---")
    st.subheader("📏 Quality Control Thresholds")
    max_area_threshold = st.number_input(
        "Max Area Threshold (cm²)",
        min_value=0.0,
        value=100.0,
        step=1.0,
        help="Maximum allowed dent area in cm² for PASS status"
    )
    max_depth_threshold = st.number_input(
        "Max Depth Threshold (mm)",
        min_value=0.0,
        value=35.0,
        step=0.5,
        help="Maximum allowed dent depth in mm for PASS status"
    )
    
    st.markdown("---")
    
    # Camera Intrinsics Settings
    st.subheader("📷 Camera Intrinsics (Optional)")
    intrinsics_file = st.file_uploader(
        "Upload Camera Intrinsics JSON File",
        type=['json'],
        help="Upload a camera intrinsics JSON file with fx, fy, cx, cy values. "
             "If not provided, will use camera_intrinsics_default.json automatically."
    )
    
    intrinsics_json_path = None
    if intrinsics_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='wb') as tmp_file:
            tmp_file.write(intrinsics_file.getvalue())
            intrinsics_json_path = tmp_file.name
            st.success(f"✅ Loaded intrinsics file: {intrinsics_file.name}")
        
        # Store in session state for cleanup later
        if 'temp_intrinsics_files' not in st.session_state:
            st.session_state.temp_intrinsics_files = []
        st.session_state.temp_intrinsics_files.append(intrinsics_json_path)
    else:
        st.info("ℹ️ Using default camera intrinsics from camera_intrinsics_default.json")
    
    st.markdown("---")
    
    # RANSAC Panel Extraction Settings
    st.subheader("🔄 RANSAC Panel Extraction (Optional)")
    use_ransac = st.checkbox(
        "Enable RANSAC panel extraction",
        value=True,
        help="Extract container panel using RANSAC plane fitting before inference. "
             "Recommended for raw depth maps with background noise."
    )
    
    if use_ransac:
        camera_fov = st.slider(
            "Camera FOV (degrees)",
            min_value=30.0,
            max_value=120.0,
            value=DEFAULT_CAMERA_FOV,
            step=5.0,
            help=f"Camera field of view for point cloud conversion and intrinsics-based area calculation. "
                 f"Default: {DEFAULT_CAMERA_FOV}° (typical for Intel RealSense, Kinect). "
                 f"Adjust to match your camera specifications for accurate measurements."
        )
        
        with st.expander("ℹ️ How to find your camera FOV"):
            st.markdown("""
            **Common Camera FOV Values:**
            - Intel RealSense D435/D455: ~87° (horizontal) or ~58° (vertical)
            - Intel RealSense L515: ~70° (diagonal)
            - Microsoft Kinect v2: ~70° (horizontal)
            - Generic depth cameras: 60-90°
            
            **How to find your camera FOV:**
            1. Check camera specifications/datasheet
            2. Look for "Field of View" or "FOV" in degrees
            3. Use horizontal FOV if available (more accurate for area calculation)
            4. If only diagonal FOV is given, use that as approximation
            
            **Why it matters:**
            - Used for converting depth maps to 3D point clouds (RANSAC)
            - Used for calculating accurate pixel sizes at different depths
            - Incorrect FOV will affect area measurements
            """)
        adaptive_threshold = st.checkbox(
            "Adaptive residual threshold",
            value=True,
            help="Automatically tune threshold based on corrugation depth"
        )
        residual_threshold = None
        if not adaptive_threshold:
            residual_threshold = st.slider(
                "Residual Threshold (meters)",
                min_value=0.005,
                max_value=0.1,
                value=0.02,
                step=0.005,
                help="Maximum distance from plane to be considered an inlier"
            )
        downsample_factor = st.slider(
            "Downsample Factor",
            min_value=1,
            max_value=8,
            value=DEFAULT_DOWNSAMPLE_FACTOR,
            step=1,
            help="Downsample factor for faster RANSAC (higher = faster but less accurate)"
        )
        
        # Morphological Closing Settings
        st.markdown("**Morphological Closing (Fill Holes):**")
        apply_morphological_closing = st.checkbox(
            "Apply morphological closing",
            value=True,
            help="Fill small holes (dents) in the panel mask. "
                 "RANSAC may reject deep dents as outliers; closing fills them back in."
        )
        closing_kernel_size = st.slider(
            "Closing Kernel Size (pixels)",
            min_value=5,
            max_value=50,
            value=DEFAULT_CLOSING_KERNEL_SIZE,
            step=2,
            help=f"Size of the closing kernel. Larger values fill larger holes, "
                 f"but may merge separate dents. Default: {DEFAULT_CLOSING_KERNEL_SIZE} pixels"
        ) if apply_morphological_closing else DEFAULT_CLOSING_KERNEL_SIZE
        
        # Rectangular Mask Enforcement
        force_rectangular_mask = st.checkbox(
            "Enforce rectangular panel mask",
            value=True,
            help="Find the largest contour and draw its rotated bounding box. "
                 "This ensures deep dents inside the panel boundary are included. "
                 "Recommended for panels with large dents."
        )
    else:
        camera_fov = DEFAULT_CAMERA_FOV
        adaptive_threshold = True
        residual_threshold = None
        downsample_factor = DEFAULT_DOWNSAMPLE_FACTOR
        apply_morphological_closing = True
        closing_kernel_size = DEFAULT_CLOSING_KERNEL_SIZE
        force_rectangular_mask = True
    
    st.markdown("---")
    
    # Overlay Settings
    st.subheader("🎨 Overlay Settings")
    overlay_alpha = st.slider(
        "Overlay Transparency",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Transparency of the red overlay on RGB image (0.0 = transparent, 1.0 = opaque)"
    )
    outline_thickness = st.slider(
        "Outline Thickness",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        help="Thickness of the red outline around dent regions (pixels)"
    )

# Main Content Area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📥 Input")
    
    # Container ID input
    container_id = st.text_input(
        "Container ID",
        value=f"CONTAINER-{st.session_state.container_id_counter:04d}",
        help="Enter a unique container identifier"
    )
    
    # File uploader for depth map
    uploaded_depth = st.file_uploader(
        "Upload Depth Map (.npy file)",
        type=['npy'],
        help="Upload a depth map in .npy format",
        key="depth_uploader"
    )
    
    # File uploader for RGB image
    uploaded_rgb = st.file_uploader(
        "Upload RGB Image (optional)",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an RGB image to overlay the segmentation mask",
        key="rgb_uploader"
    )
    
    if uploaded_depth is not None:
        try:
            # Load depth map
            depth_map_raw = np.load(uploaded_depth)
            
            # Store original dtype for display
            original_dtype = depth_map_raw.dtype
            
            # Check if uint16 format - apply special cleaning pipeline
            is_uint16 = depth_map_raw.dtype == np.uint16 or depth_map_raw.dtype == np.uint32
            st.session_state.is_uint16_format = is_uint16  # Store for later use in metrics
            
            # Display raw depth map first if uint16 format
            if is_uint16:
                st.info("🔧 Uint16/Uint32 format detected. Applying raw data cleaning pipeline (crop, unit conversion, inpainting, wall isolation, Gaussian blur)...")
                
                # Calculate crop coordinates (matching clean_depth_map_uint16)
                H, W = depth_map_raw.shape
                left_crop = int(0.20 * W)
                right_crop = int(0.80 * W)
                top_crop = int(0.1 * H)
                bottom_crop = int(0.9 * H)
                
                # Display raw/original depth map first with bounding box showing crop area
                st.subheader("Input Depth Map")
                fig_raw, ax_raw = plt.subplots(figsize=(8, 8))
                im_raw = ax_raw.imshow(depth_map_raw, cmap='viridis')
                
                # Draw bounding box to indicate crop area
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
                plt.colorbar(im_raw, ax=ax_raw, fraction=0.046)
                st.pyplot(fig_raw)
                
                # Apply cleaning pipeline
                with st.spinner("Cleaning raw depth data..."):
                    depth_map = clean_depth_map_uint16(depth_map_raw, apply_scale_factor=True)
                st.success("✅ Raw data cleaning completed.")
                
                # Display processed/cropped depth map
                st.subheader("Preprocessed Depth Map")
                fig_processed, ax_processed = plt.subplots(figsize=(8, 8))
                im_processed = ax_processed.imshow(depth_map, cmap='viridis')
                ax_processed.set_title("Preprocessed Depth Map (Cropped & Cleaned)")
                ax_processed.axis('off')
                plt.colorbar(im_processed, ax=ax_processed, fraction=0.046)
                st.pyplot(fig_processed)
            else:
                # Convert to float32 for better precision (especially important for float16 inputs)
                if depth_map_raw.dtype == np.float16:
                    st.warning("⚠️ Float16 detected. Converting to float32 for better precision. Small depth variations may have been lost.")
                depth_map = depth_map_raw.astype(np.float32)
                
                # Display depth map (non-uint16 case - show single image)
                st.subheader("Input Depth Map")
                fig, ax = plt.subplots(figsize=(8, 8))
                im = ax.imshow(depth_map, cmap='viridis')
                ax.set_title("Input Depth Map")
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)
                st.pyplot(fig)
            
            # Check resolution and provide guidance
            h, w = depth_map.shape[:2]
            total_pixels = h * w
            if total_pixels > 1_000_000:  # > 1MP (e.g., 1000x1000 or 720x1280)
                st.info(f"ℹ️ Large image detected ({h}x{w} = {total_pixels:,} pixels). "
                       f"Processing may take longer and use more memory. "
                       f"The model supports any resolution, but performance may vary if training resolution was different.")
            
            # Check minimum size requirement (model has 4 pooling levels = divide by 16)
            min_dim = min(h, w)
            if min_dim < 16:
                st.error(f"❌ Image too small! Minimum dimension is 16 pixels (after 4 pooling levels). "
                        f"Current: {h}x{w}")
            
            st.success(f"✅ Depth map loaded: {uploaded_depth.name}")
            st.info(f"Shape: {depth_map.shape}, Original Dtype: {original_dtype}, Converted to: {depth_map.dtype}")
            
            # Store in session state (as float32)
            st.session_state.depth_map = depth_map
            st.session_state.depth_filename = uploaded_depth.name
            
            # Show RANSAC-extracted panel if RANSAC is enabled
            if use_ransac:
                try:
                    with st.spinner("Extracting panel using RANSAC..."):
                        extractor = RANSACPanelExtractor(
                            camera_fov=camera_fov,
                            residual_threshold=residual_threshold if residual_threshold else 0.02,
                            adaptive_threshold=adaptive_threshold,
                            downsample_factor=downsample_factor,
                            apply_morphological_closing=apply_morphological_closing,
                            closing_kernel_size=closing_kernel_size,
                            force_rectangular_mask=force_rectangular_mask
                        )
                        # Extract panel and fill background
                        cleaned_depth, panel_mask, ransac_stats, plane_coefficients = extractor.extract_panel(
                            depth_map, 
                            fill_background=False
                        )
                        
                        # Store in session state for later use
                        st.session_state.ransac_cleaned_depth = cleaned_depth
                        st.session_state.ransac_panel_mask = panel_mask
                        st.session_state.ransac_stats = ransac_stats
                        st.session_state.ransac_camera_fov = camera_fov  # Store for metrics calculation
                        st.session_state.ransac_plane_coefficients = plane_coefficients  # Store plane coefficients
                    
                    # Display RANSAC results
                    st.subheader("🔄 RANSAC-Extracted Panel")
                    
                    # Statistics
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Panel Coverage", f"{ransac_stats['plane_percentage']:.1f}%")
                    with col_stat2:
                        st.metric("Panel Pixels", f"{ransac_stats['plane_pixel_count']:,}")
                    with col_stat3:
                        st.metric("Threshold", f"{ransac_stats['residual_threshold_used']*1000:.1f} mm")
                    
                    # Display panel mask and cleaned depth side by side
                    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                    
                    # Panel mask
                    im1 = axes[0].imshow(panel_mask, cmap='gray', vmin=0, vmax=1)
                    axes[0].set_title("Panel Mask (White = Panel)")
                    axes[0].axis('off')
                    plt.colorbar(im1, ax=axes[0], fraction=0.046)
                    
                    # Cleaned depth map
                    im2 = axes[1].imshow(cleaned_depth, cmap='viridis')
                    axes[1].set_title("Cleaned Depth Map (Background Filled)")
                    axes[1].axis('off')
                    plt.colorbar(im2, ax=axes[1], fraction=0.046)
                    
                    st.pyplot(fig)
                    
                    # Additional info
                    if 'plane_median_depth' in ransac_stats:
                        st.info(f"ℹ️ Panel median depth: {ransac_stats['plane_median_depth']:.4f} m | "
                               f"Fill method: {ransac_stats.get('fill_method', 'median')}")
                    
                except Exception as e:
                    import traceback
                    error_msg = str(e)
                    st.error(f"❌ RANSAC extraction failed: {error_msg}")
                    
                    # Provide diagnostic information
                    valid_pixels = np.sum((depth_map > 0) & np.isfinite(depth_map))
                    total_pixels = depth_map.size
                    st.warning(f"📊 Diagnostic: {valid_pixels}/{total_pixels} valid pixels ({100*valid_pixels/total_pixels:.1f}%)")
                    
                    if valid_pixels == 0:
                        st.error("⚠️ No valid depth pixels found! Check your depth map file.")
                    elif valid_pixels < 100:
                        st.warning("⚠️ Very few valid pixels. RANSAC may fail with insufficient data.")
                    
                    # Show full traceback in expander for debugging
                    with st.expander("🔍 Show full error details"):
                        st.code(traceback.format_exc())
                    
                    st.session_state.ransac_cleaned_depth = None
                    st.session_state.ransac_panel_mask = None
                    st.session_state.ransac_stats = None
            else:
                # Clear RANSAC results if disabled
                st.session_state.ransac_cleaned_depth = None
                st.session_state.ransac_panel_mask = None
                st.session_state.ransac_stats = None
            
        except Exception as e:
            st.error(f"❌ Error loading depth map: {str(e)}")
            st.session_state.depth_map = None
            st.session_state.ransac_cleaned_depth = None
            st.session_state.ransac_panel_mask = None
            st.session_state.ransac_stats = None
    else:
        st.info("👆 Please upload a depth map (.npy file)")
        st.session_state.depth_map = None
        st.session_state.ransac_cleaned_depth = None
        st.session_state.ransac_panel_mask = None
        st.session_state.ransac_stats = None
    
    # Display RGB image if uploaded
    if uploaded_rgb is not None:
        try:
            # Load RGB image
            rgb_image = Image.open(uploaded_rgb)
            rgb_array = np.array(rgb_image)

            # Convert RGBA to RGB if needed
            if rgb_array.shape[2] == 4:
                rgb_array = rgb_array[:, :, :3]

            st.success(f"✅ RGB image loaded: {uploaded_rgb.name}")
            st.info(f"Shape: {rgb_array.shape}")

            st.success(f"✅ RGB image loaded: {uploaded_rgb.name}")
            st.info(f"Shape: {rgb_array.shape}")

            # Display RGB image
            st.subheader("Input RGB Image")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(rgb_array)
            ax.set_title("Input RGB Image")
            ax.axis('off')
            st.pyplot(fig)

            # Store in session state
            st.session_state.rgb_image = rgb_array
            st.session_state.rgb_filename = uploaded_rgb.name
            
        except Exception as e:
            st.error(f"❌ Error loading RGB image: {str(e)}")
            st.session_state.rgb_image = None
    else:
        st.info("💡 Optional: Upload an RGB image to visualize overlay")
        st.session_state.rgb_image = None

with col2:
    st.header("📤 Output")
    
    # Run inference button
    if st.button("🚀 Run Inference", type="primary", use_container_width=True):
        if not st.session_state.model_loaded:
            st.error("❌ Please load a model first!")
        elif 'depth_map' not in st.session_state or st.session_state.depth_map is None:
            st.error("❌ Please upload a depth map first!")
        else:
            try:
                with st.spinner("Running inference..."):
                    # Use pre-computed RANSAC results from session state if RANSAC was enabled
                    # RANSAC is already computed when depth map is uploaded (see above)
                    depth_cleaned_precomputed = None
                    panel_mask_precomputed = None
                    if use_ransac:
                        if st.session_state.ransac_cleaned_depth is not None:
                            # Use pre-computed RANSAC results from session state
                            depth_cleaned_precomputed = st.session_state.ransac_cleaned_depth
                            panel_mask_precomputed = st.session_state.ransac_panel_mask
                        else:
                            st.warning("⚠️ RANSAC enabled but no pre-computed results found. Please upload depth map again or disable RANSAC.")
                    
                    # Run inference (uses pre-computed RANSAC results if available)
                    binary_mask, prob_mask, preprocessed_input = predict_mask(
                        st.session_state.model,
                        st.session_state.depth_map,
                        device=str(st.session_state.device),
                        threshold=threshold,
                        depth_cleaned=depth_cleaned_precomputed,
                        panel_mask=panel_mask_precomputed
                    )
                    
                    # Store results
                    st.session_state.binary_mask = binary_mask
                    st.session_state.prob_mask = prob_mask
                    
                    # Calculate dent metrics
                    # Use RANSAC-cleaned depth if available (already converted to meters)
                    # Otherwise use original depth map (may need conversion)
                    depth_for_metrics = st.session_state.ransac_cleaned_depth if st.session_state.ransac_cleaned_depth is not None else st.session_state.depth_map
                    
                    # Determine depth units: if uint16 was processed, it's already in meters
                    # Otherwise, check if it's likely in meters or mm
                    depth_units_for_metrics = 'meters'  # Default assumption
                    is_uint16_format = st.session_state.get('is_uint16_format', False)
                    if is_uint16_format:
                        # clean_depth_map_uint16 already converted mm to meters (and applied scale_factor=0.5)
                        depth_units_for_metrics = 'meters'
                    else:
                        # For non-uint16, check depth values to auto-detect
                        if depth_for_metrics is not None:
                            valid_depths = depth_for_metrics[np.isfinite(depth_for_metrics) & (depth_for_metrics > 0)]
                            if len(valid_depths) > 0:
                                median_depth = np.median(valid_depths)
                                if median_depth > 10:
                                    depth_units_for_metrics = 'mm'
                    
                    # Pass camera intrinsics (from uploaded file or default)
                    metrics_kwargs = {
                        'depth_map': depth_for_metrics,
                        'dent_mask': binary_mask,
                        'pixel_to_cm': pixel_to_cm,  # None - using intrinsics instead
                        'intrinsics_json_path': intrinsics_json_path,  # Use uploaded file or None (defaults to default file)
                        'depth_units': depth_units_for_metrics  # Explicitly set units based on preprocessing
                    }
                    
                    # Add camera intrinsics if RANSAC was used and camera_fov is available
                    if use_ransac:
                        # Get camera_fov from session state or use default
                        ransac_fov = st.session_state.get('ransac_camera_fov', DEFAULT_CAMERA_FOV)
                        metrics_kwargs['camera_fov'] = ransac_fov
                        
                        # Add panel mask for accurate depth measurement (median of normal panel surface)
                        if panel_mask_precomputed is not None:
                            metrics_kwargs['panel_mask'] = panel_mask_precomputed
                        
                        # Add plane coefficients for plane depth-based median calculation
                        if 'ransac_plane_coefficients' in st.session_state and st.session_state.ransac_plane_coefficients is not None:
                            metrics_kwargs['plane_coefficients'] = st.session_state.ransac_plane_coefficients
                    else:
                        # If RANSAC not used, try to get panel mask from session state if available
                        if 'ransac_panel_mask' in st.session_state and st.session_state.ransac_panel_mask is not None:
                            metrics_kwargs['panel_mask'] = st.session_state.ransac_panel_mask
                    
                    metrics = calculate_dent_metrics(**metrics_kwargs)
                    
                    # Determine status (PASS/FAIL)
                    has_dents = metrics['num_defects'] > 0
                    
                    # Status determination - only use area if it's valid
                    if metrics['area_valid'] and metrics['area_cm2'] is not None:
                        area_pass = metrics['area_cm2'] <= max_area_threshold
                    else:
                        area_pass = True  # Don't fail based on invalid area
                    
                    depth_pass = metrics['max_depth_mm'] <= max_depth_threshold
                    status = "PASS" if (not has_dents or (area_pass and depth_pass)) else "FAIL"
                    
                    # Store metrics
                    st.session_state.metrics = metrics
                    st.session_state.status = status
                    
                    st.success("✅ Inference completed!")
                    
                    # Show warnings about missing calibration information
                    if metrics['missing_info']:
                        for info in metrics['missing_info']:
                            st.warning(f"⚠️ {info}")
                    
                    # Notification alert when dent is detected
                    if has_dents:
                        st.error(f"🚨 ALERT: {metrics['num_defects']} dent(s) detected!")
                        if status == "FAIL":
                            area_info = f"Area={metrics['area_cm2']:.2f} cm²" if metrics['area_valid'] and metrics['area_cm2'] is not None else "Area=N/A (not calibrated)"
                            st.error(f"❌ FAIL: Dent exceeds quality thresholds! {area_info}, Max Depth={metrics['max_depth_mm']:.2f} mm")
                        else:
                            area_info = f"Area={metrics['area_cm2']:.2f} cm²" if metrics['area_valid'] and metrics['area_cm2'] is not None else "Area=N/A (not calibrated)"
                            st.warning(f"⚠️ PASS with defects: {area_info}, Depth={metrics['max_depth_mm']:.2f} mm")
                    else:
                        st.success("✅ No dents detected - Container is in good condition!")
                    
                    # Display results
                    st.subheader("Binary Segmentation Mask")
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(binary_mask, cmap='gray')
                    ax.set_title("Binary Mask (Threshold = {:.2f})".format(threshold))
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    # Display probability map
                    st.subheader("Probability Map")
                    fig, ax = plt.subplots(figsize=(8, 8))
                    im = ax.imshow(prob_mask, cmap='hot', vmin=0, vmax=1)
                    ax.set_title("Probability Map")
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046)
                    st.pyplot(fig)
                    
                    # Create overlay if RGB image is available
                    if 'rgb_image' in st.session_state and st.session_state.rgb_image is not None:
                        try:
                            # Create overlay
                            overlay_image = create_dent_overlay(
                                st.session_state.rgb_image,
                                binary_mask,
                                overlay_alpha=overlay_alpha,
                                outline_thickness=outline_thickness
                            )
                            
                            # Store overlay
                            st.session_state.overlay_image = overlay_image
                            
                            # Display overlay
                            st.subheader("🎨 Dent Overlay on RGB Image")
                            fig, ax = plt.subplots(figsize=(8, 8))
                            ax.imshow(overlay_image)
                            ax.set_title("Dent Segmentation Overlay")
                            ax.axis('off')
                            st.pyplot(fig)
                            
                        except Exception as e:
                            st.warning(f"⚠️ Could not create overlay: {str(e)}")
                            st.session_state.overlay_image = None
                    else:
                        st.info("💡 Upload an RGB image to see the overlay visualization")
                        st.session_state.overlay_image = None
                    
                    # Dent Metrics Display
                    st.subheader("📊 Dent Metrics")
                    
                    # Show intrinsics status
                    area_method = metrics.get('area_method', None)
                    
                    if area_method == 'intrinsics':
                        st.success("✅ Using camera intrinsics for area calculation")
                        if intrinsics_file is not None:
                            st.info(f"ℹ️ Using uploaded intrinsics file: {intrinsics_file.name}")
                        else:
                            st.info("ℹ️ Using default camera intrinsics from camera_intrinsics_default.json")
                    elif area_method == 'pixel_to_cm':
                        st.warning("⚠️ Falling back to pixel_to_cm method (intrinsics not available)")
                    else:
                        st.warning("⚠️ Area calculation not available - check camera intrinsics configuration")
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        if metrics['area_valid'] and metrics['area_cm2'] is not None:
                            method_label = " (Intrinsics)" if area_method == 'intrinsics' else " (pixel_to_cm)"
                            st.metric("Total Area", f"{metrics['area_cm2']:.2f} cm²", help=f"Calculation method: {area_method or 'pixel_to_cm'}")
                            if area_method:
                                st.caption(f"Method: {area_method}")
                        else:
                            st.metric("Total Area", "N/A", help="Area computation requires calibration")
                            st.caption("⚠️ Not calibrated")
                    with col_b:
                        st.metric("Max Depth", f"{metrics['max_depth_mm']:.2f} mm")
                    with col_c:
                        st.metric("Total Defects", f"{metrics['num_defects']}")
                    with col_d:
                        status_color = "🟢" if status == "PASS" else "🔴"
                        st.metric("Status", f"{status_color} {status}")
                    
                    # Additional Statistics
                    with st.expander("📈 Detailed Statistics"):
                        col_e, col_f, col_g = st.columns(3)
                        with col_e:
                            st.metric("Mask Pixels", f"{(binary_mask > 0).sum():,}")
                        with col_f:
                            total_pixels = binary_mask.size
                            percentage = (binary_mask > 0).sum() / total_pixels * 100
                            st.metric("Coverage", f"{percentage:.2f}%")
                        with col_g:
                            st.metric("Avg Depth", f"{metrics['avg_depth_mm']:.2f} mm")
                        st.metric("Max Probability", f"{prob_mask.max():.3f}")
                        st.metric("Pixel Count", f"{metrics['pixel_count']:,}")
                        
                        # Depth calculation diagnostics
                        if 'depth_stats' in metrics:
                            st.markdown("---")
                            st.markdown("**🔍 Depth Calculation Diagnostics:**")
                            depth_stats = metrics['depth_stats']
                            units_detected = depth_stats.get('depth_units_detected', 'unknown')
                            method_used = depth_stats.get('method_used', 'unknown')
                            st.markdown(f"- **Units Auto-Detected:** {units_detected}")
                            st.markdown(f"- **Method Used:** {method_used.replace('_', ' ').title()}")
                            if depth_stats.get('max_depth_value') is not None:
                                st.markdown(f"- **Max Depth Value in Map:** {depth_stats['max_depth_value']:.4f} {units_detected}")
                            st.markdown(f"- **Wall Reference Depth (median):** {depth_stats.get('reference_depth', 0):.4f} {units_detected} = {depth_stats.get('reference_depth_mm', 0):.2f} mm")
                            
                            st.markdown("**📊 Depth Metrics (Relative to Wall):**")
                            st.markdown(f"- **Median Depth:** {depth_stats.get('depth_median_mm', 0):.2f} mm (good for volume estimation)")
                            st.markdown(f"- **Raw Max Depth:** {depth_stats.get('depth_raw_max_mm', 0):.2f} mm (sensitive to noise)")
                            st.markdown(f"- **Robust Max (Blur→Max):** **{depth_stats.get('depth_robust_max_mm', 0):.2f} mm** ⭐ (median filter removes noise, then max captures true depth)")
                            
                            st.markdown("**🔍 Dent Pixel Depths (Absolute):**")
                            st.markdown(f"- **Dent Median Depth:** {depth_stats.get('dent_depth_median', 0):.4f} {units_detected} = {depth_stats.get('dent_depth_median_mm', 0):.2f} mm")
                            st.markdown(f"- **Dent Max Depth:** {depth_stats.get('dent_depth_max', 0):.4f} {units_detected} = {depth_stats.get('dent_depth_max_mm', 0):.2f} mm")
                            st.markdown(f"- **Dent Min Depth:** {depth_stats.get('dent_depth_min', 0):.4f} {units_detected} = {depth_stats.get('dent_depth_min_mm', 0):.2f} mm")
                            
                            st.markdown(f"**✅ Final Max Depth (for Pass/Fail):** **{metrics['max_depth_mm']:.2f} mm**")
                            
                            # Warning if depth seems unusually large
                            if metrics['max_depth_mm'] > 100:
                                st.warning(f"⚠️ **Large depth detected ({metrics['max_depth_mm']:.2f} mm).** This may indicate:")
                                st.markdown("""
                                - Depth map units mismatch (e.g., map is in mm but code assumed meters, or vice versa)
                                - Incorrect reference depth calculation
                                - Background noise included in dent regions
                                - Please verify your depth map units and check the diagnostics above
                                """)
                        
                        # Show calculation method details
                        st.markdown("---")
                        st.markdown("**📐 Area Calculation Method:**")
                        if area_method == 'intrinsics':
                            st.success(f"✅ **Using Camera Intrinsics**")
                            if intrinsics_file is not None:
                                st.markdown(f"- Intrinsics file: {intrinsics_file.name}")
                            else:
                                st.markdown(f"- Intrinsics file: camera_intrinsics_default.json (default)")
                            st.markdown(f"- Method: Depth-dependent pixel size calculation using fx, fy")
                            st.markdown(f"- Accuracy: High (accounts for depth variation)")
                            st.markdown(f"- Formula: `pixel_size = depth / focal_length`, then `area = Σ(pixel_size²)`")
                        elif metrics['area_valid'] and metrics['area_cm2'] is not None:
                            st.warning(f"⚠️ **Falling back to pixel_to_cm Method**")
                            st.markdown(f"- Reason: Camera intrinsics not available")
                            st.markdown(f"- Method: Constant pixel size assumption")
                            st.markdown(f"- Accuracy: Medium (doesn't account for depth variation)")
                            st.markdown(f"- 💡 **Tip**: Upload a camera intrinsics JSON file for more accurate area calculation")
                        else:
                            st.warning("⚠️ **Area calculation not available**")
                            st.markdown("- Reason: Missing camera intrinsics")
                            st.markdown("- Upload a camera intrinsics JSON file or ensure camera_intrinsics_default.json exists")
                    
                    # Save to history
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Create thumbnail (resize overlay or mask)
                    if 'overlay_image' in st.session_state and st.session_state.overlay_image is not None:
                        thumbnail = Image.fromarray(st.session_state.overlay_image)
                    else:
                        # Create a simple visualization from mask
                        mask_rgb = np.stack([binary_mask, binary_mask, binary_mask], axis=-1)
                        thumbnail = Image.fromarray(mask_rgb)
                    
                    thumbnail.thumbnail((150, 150))
                    thumb_buffer = io.BytesIO()
                    thumbnail.save(thumb_buffer, format='PNG')
                    thumb_buffer.seek(0)
                    
                    history_entry = {
                        'timestamp': timestamp,
                        'container_id': container_id,
                        'total_defects': metrics['num_defects'],
                        'status': status,
                        'area_cm2': metrics['area_cm2'] if metrics['area_valid'] and metrics['area_cm2'] is not None else None,
                        'area_valid': metrics['area_valid'],
                        'max_depth_mm': metrics['max_depth_mm'],
                        'thumbnail': thumb_buffer.getvalue(),
                        'threshold': threshold
                    }
                    
                    st.session_state.history.append(history_entry)
                    st.session_state.container_id_counter += 1
                    
                    # Download buttons
                    st.subheader("💾 Download Results")
                    output_filename = st.session_state.depth_filename.replace('.npy', '_mask.npy')
                    
                    # Prepare binary mask for download
                    buffer = io.BytesIO()
                    np.save(buffer, binary_mask)
                    buffer.seek(0)
                    
                    # Download binary mask as numpy
                    st.download_button(
                        label="Download Binary Mask (.npy)",
                        data=buffer,
                        file_name=output_filename,
                        mime="application/octet-stream"
                    )
                    
                    # Also save probability mask
                    prob_filename = st.session_state.depth_filename.replace('.npy', '_prob.npy')
                    prob_buffer = io.BytesIO()
                    np.save(prob_buffer, prob_mask)
                    prob_buffer.seek(0)
                    
                    st.download_button(
                        label="Download Probability Map (.npy)",
                        data=prob_buffer,
                        file_name=prob_filename,
                        mime="application/octet-stream"
                    )
                    
                    # Download overlay image if available
                    if 'overlay_image' in st.session_state and st.session_state.overlay_image is not None:
                        overlay_filename = st.session_state.depth_filename.replace('.npy', '_overlay.png')
                        
                        # Convert to PIL Image and save to buffer
                        overlay_pil = Image.fromarray(st.session_state.overlay_image)
                        overlay_buffer = io.BytesIO()
                        overlay_pil.save(overlay_buffer, format='PNG')
                        overlay_buffer.seek(0)
                        
                        st.download_button(
                            label="Download Overlay Image (.png)",
                            data=overlay_buffer,
                            file_name=overlay_filename,
                            mime="image/png"
                        )
                    
            except Exception as e:
                st.error(f"❌ Error during inference: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# History Panel
st.markdown("---")
st.header("📜 Processing History")

if len(st.session_state.history) == 0:
    st.info("No processing history yet. Run inference on an image to see history here.")
else:
    # Display history in reverse order (most recent first)
    history_reversed = list(reversed(st.session_state.history))
    
    # Filter options
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        filter_status = st.selectbox(
            "Filter by Status",
            ["All", "PASS", "FAIL"],
            key="history_filter_status"
        )
    with col_filter2:
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    
    # Filter history
    filtered_history = history_reversed
    if filter_status != "All":
        filtered_history = [h for h in history_reversed if h['status'] == filter_status]
    
    if len(filtered_history) == 0:
        st.info(f"No entries found with status: {filter_status}")
    else:
        # Display history entries
        for idx, entry in enumerate(filtered_history):
            with st.container():
                # Create columns for each entry
                col_thumb, col_info, col_metrics = st.columns([1, 3, 2])
                
                with col_thumb:
                    # Display thumbnail
                    thumb_img = Image.open(io.BytesIO(entry['thumbnail']))
                    st.image(thumb_img, use_container_width=True)
                
                with col_info:
                    st.markdown(f"**Container ID:** {entry['container_id']}")
                    st.markdown(f"**Timestamp:** {entry['timestamp']}")
                    status_display = f"{'🟢' if entry['status'] == 'PASS' else '🔴'} **Status:** {entry['status']}"
                    st.markdown(status_display)
                    st.markdown(f"**Total Defects:** {entry['total_defects']}")
                
                with col_metrics:
                    st.markdown("**Metrics:**")
                    if entry.get('area_valid', True) and entry.get('area_cm2') is not None:
                        st.markdown(f"- Area: {entry['area_cm2']:.2f} cm²")
                    else:
                        st.markdown("- Area: N/A (not calibrated)")
                    st.markdown(f"- Max Depth: {entry['max_depth_mm']:.2f} mm")
                    st.markdown(f"- Threshold: {entry['threshold']:.2f}")
                
                if idx < len(filtered_history) - 1:
                    st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer-text">
    <p>Shipping Container Dent Detection System | Attention-UNet Model</p>
    <p>Upload a depth map (.npy) and optionally an RGB image to generate segmentation masks and overlays</p>
</div>
""", unsafe_allow_html=True)

