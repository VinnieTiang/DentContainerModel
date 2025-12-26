"""
Dent Container Detection - Streamlit UI
Simple interface to load model and run inference on depth maps
"""
import streamlit as st
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os
import io
from PIL import Image
from datetime import datetime
import json
from model_architecture import AttentionUNet, predict_mask, preprocess_depth, create_dent_overlay, calculate_dent_metrics

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

# Header
st.markdown('<div class="main-header">📦 Dent Container Detection System</div>', unsafe_allow_html=True)
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
    default_model_path = "best_attention_unet.pth"
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
    
    # Pixel to cm conversion (for area calculation)
    st.markdown("---")
    st.subheader("📐 Area Calibration (REQUIRED for area computation)")
    
    use_calibration = st.checkbox(
        "Enable area computation",
        value=True,
        help="Uncheck to disable area computation if calibration is not available"
    )
    
    if use_calibration:
        pixel_to_cm = st.slider(
            "Pixel to CM Conversion Factor",
            min_value=0.001,
            max_value=10.0,
            value=0.20,
            step=0.001,
            format="%.4f",
            help="REQUIRED: Physical size per pixel in cm. "
                 "Default (0.20 cm/pixel) assumes: ~2.4m container width, ~2m distance, 1280px image width. "
                 "To calibrate: (1) Use camera parameters: focal_length/sensor_width * distance/image_width, "
                 "(2) Measure a known object in the image, or (3) Place a ruler and measure pixels per cm."
        )
        
        with st.expander("ℹ️ How to calibrate pixel-to-cm conversion"):
            st.markdown("""
            **Default Value (0.20 cm/pixel):**
            - Assumes: Standard 2.4m container width, ~2m inspection distance, 1280px image width
            - Typical for: Intel RealSense, Kinect, or similar depth cameras
            - **You should calibrate this for your specific setup!**
            
            **Method 1: Camera Calibration**
            - Focal length (mm)
            - Sensor width (mm) 
            - Distance to object (mm)
            - Image width (pixels)
            - Formula: `pixel_to_cm = (focal_length / sensor_width) * (distance / image_width) * 10`
            
            **Method 2: Reference Object (Recommended)**
            - Place a known-size object (e.g., 10cm ruler) in the image
            - Measure its width in pixels
            - Formula: `pixel_to_cm = object_size_cm / object_width_pixels`
            - Example: If a 10cm ruler spans 50 pixels → 10cm / 50px = 0.20 cm/pixel
            
            **Method 3: Container-Based Calibration**
            - If you know the container width (typically 2.4m = 240cm)
            - Measure container width in pixels in your image
            - Formula: `pixel_to_cm = 240cm / container_width_pixels`
            
            **⚠️ WARNING:** Without proper calibration, area measurements are NOT physically meaningful!
            """)
    else:
        pixel_to_cm = None
        st.warning("⚠️ Area computation disabled. Enable calibration above to compute real-world area.")
    
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
            depth_map = np.load(uploaded_depth)
            
            st.success(f"✅ Depth map loaded: {uploaded_depth.name}")
            st.info(f"Shape: {depth_map.shape}, Dtype: {depth_map.dtype}")
            
            # Display depth map
            st.subheader("Input Depth Map")
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(depth_map, cmap='viridis')
            ax.set_title("Input Depth Map")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
            st.pyplot(fig)
            
            # Store in session state
            st.session_state.depth_map = depth_map
            st.session_state.depth_filename = uploaded_depth.name
            
        except Exception as e:
            st.error(f"❌ Error loading depth map: {str(e)}")
            st.session_state.depth_map = None
    else:
        st.info("👆 Please upload a depth map (.npy file)")
        st.session_state.depth_map = None
    
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
                    # Run inference
                    binary_mask, prob_mask = predict_mask(
                        st.session_state.model,
                        st.session_state.depth_map,
                        device=str(st.session_state.device),
                        threshold=threshold
                    )
                    
                    # Store results
                    st.session_state.binary_mask = binary_mask
                    st.session_state.prob_mask = prob_mask
                    
                    # Calculate dent metrics
                    metrics = calculate_dent_metrics(
                        st.session_state.depth_map,
                        binary_mask,
                        pixel_to_cm=pixel_to_cm
                    )
                    
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
                            st.error(f"❌ FAIL: Dent exceeds quality thresholds! {area_info}, Depth={metrics['max_depth_mm']:.2f} mm")
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
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        if metrics['area_valid'] and metrics['area_cm2'] is not None:
                            st.metric("Total Area", f"{metrics['area_cm2']:.2f} cm²")
                        else:
                            st.metric("Total Area", "N/A", help="Area computation requires pixel-to-cm calibration")
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
                        if not metrics['area_valid']:
                            st.info("ℹ️ Area computation requires pixel-to-cm calibration. See settings panel.")
                    
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
    <p>Dent Container Detection System | Attention-UNet Model</p>
    <p>Upload a depth map (.npy) and optionally an RGB image to generate segmentation masks and overlays</p>
</div>
""", unsafe_allow_html=True)

