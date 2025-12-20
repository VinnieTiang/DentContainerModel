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
from model_architecture import AttentionUNet, predict_mask, preprocess_depth, create_dent_overlay

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
        "Binary Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Threshold for converting probability mask to binary mask"
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
                    
                    st.success("✅ Inference completed!")
                    
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
                    
                    # Statistics
                    st.subheader("📊 Statistics")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Mask Pixels", f"{(binary_mask > 0).sum():,}")
                    with col_b:
                        total_pixels = binary_mask.size
                        percentage = (binary_mask > 0).sum() / total_pixels * 100
                        st.metric("Coverage", f"{percentage:.2f}%")
                    with col_c:
                        st.metric("Max Probability", f"{prob_mask.max():.3f}")
                    
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

# Footer
st.markdown("---")
st.markdown("""
<div class="footer-text">
    <p>Dent Container Detection System | Attention-UNet Model</p>
    <p>Upload a depth map (.npy) and optionally an RGB image to generate segmentation masks and overlays</p>
</div>
""", unsafe_allow_html=True)

