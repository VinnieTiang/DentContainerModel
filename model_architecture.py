"""
Model Architecture for Dent Container Detection
Attention-UNet model definition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Optional, Tuple
import json
from pathlib import Path


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.ReLU(inplace=True),
                                 nn.Conv2d(F_int, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Upsample gating signal to match x1 size if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        psi = self.psi(g1 + x1)
        return x * psi


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, filters=[32,64,128,256]):
        super().__init__()
        f1, f2, f3, f4 = filters
        self.enc1 = ConvBlock(in_channels, f1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(f1, f2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(f2, f3)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(f3, f4)
        self.pool4 = nn.MaxPool2d(2)
        self.center = ConvBlock(f4, f4*2)

        self.att4 = AttentionGate(F_g=f4*2, F_l=f4, F_int=f4)
        self.dec4 = UpBlock(f4*2, f4)
        self.att3 = AttentionGate(F_g=f4, F_l=f3, F_int=f3)
        self.dec3 = UpBlock(f4, f3)
        self.att2 = AttentionGate(F_g=f3, F_l=f2, F_int=f2)
        self.dec2 = UpBlock(f3, f2)
        self.att1 = AttentionGate(F_g=f2, F_l=f1, F_int=f1)
        self.dec1 = UpBlock(f2, f1)

        self.final = nn.Conv2d(f1, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        c = self.center(self.pool4(e4))

        g4 = c
        x4 = self.att4(g4, e4)
        d4 = self.dec4(c, x4)

        g3 = d4
        x3 = self.att3(g3, e3)
        d3 = self.dec3(d4, x3)

        g2 = d3
        x2 = self.att2(g2, e2)
        d2 = self.dec2(d3, x2)

        g1 = d2
        x1 = self.att1(g1, e1)
        d1 = self.dec1(d2, x1)

        out = self.final(d1)
        return out


def preprocess_depth(depth: np.ndarray, 
                     target_size: Optional[Tuple[int, int]] = None,
                     depth_cleaned: Optional[np.ndarray] = None,
                     panel_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess depth map for model inference with Hard Masking for background suppression.
    Converts depth map to 3-channel input: [normalized_depth, normalized_gx, normalized_gy]
    
    This function matches the ver4 notebook training pipeline:
    - Uses pre-computed cleaned depth map (RANSAC should be applied BEFORE calling this function)
    - Normalizes depth to [0.1, 1.0] range (makes wall distinct from 0.0 background)
    - Computes gradients on cleaned depth (no cliff edges)
    - Masks gradients to remove background noise
    - Returns panel mask for hard masking predictions
    
    NOTE: RANSAC panel extraction should be done BEFORE calling this function.
    Use panel_extractor.py to extract panel, then pass the results here.
    
    Args:
        depth: Input depth map (H, W) as numpy array
               NOTE: Should be converted to float32 BEFORE calling this function (e.g., in app.py before RANSAC).
                     This function will convert to float32 if not already done, but conversion before RANSAC
                     is recommended for best precision.
        target_size: Optional target size (H, W) to resize depth and mask. If None, uses original size.
        depth_cleaned: Pre-computed cleaned depth map (H, W) with background filled.
                      If None, uses simple median fill fallback.
        panel_mask: Pre-computed panel mask (H, W) where 1.0 = panel, 0.0 = background.
                    If None, creates mask from valid pixels.
        
    Returns:
        Tuple of (input_tensor, panel_mask)
        - input_tensor: Preprocessed input tensor ready for model (3, H, W)
        - panel_mask: Binary panel mask (H, W) where 1.0 = panel, 0.0 = background
                      Resized to match input_tensor spatial dimensions
    """
    # Convert to float32 for precision (critical for gradient calculations)
    # NOTE: Conversion should ideally happen BEFORE calling this function (e.g., in app.py before RANSAC)
    # This is a defensive check - only convert if not already float32
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)
    original_shape = depth.shape[:2]  # (H, W)
    
    # --- STEP 1: Use pre-computed cleaned depth and panel mask ---
    # RANSAC should be applied BEFORE calling this function (e.g., in app.py or by caller)
    if depth_cleaned is not None and panel_mask is not None:
        # Use pre-computed RANSAC results
        depth_cleaned = depth_cleaned.astype(np.float32)
        panel_mask = panel_mask.astype(np.float32)
        
        # Verify shapes match
        if depth_cleaned.shape != depth.shape:
            raise ValueError(f"Pre-computed depth_cleaned shape {depth_cleaned.shape} doesn't match depth shape {depth.shape}")
        if panel_mask.shape != depth.shape:
            raise ValueError(f"Pre-computed panel_mask shape {panel_mask.shape} doesn't match depth shape {depth.shape}")
    else:
        # Fallback: Simple median fill if RANSAC results not provided
        # This is a fallback for cases where RANSAC wasn't applied
        valid_original = np.isfinite(depth) & (depth > 0)
        if not np.any(valid_original):
            # Fallback for empty images
            empty_mask = np.zeros((depth.shape[0], depth.shape[1]), dtype=np.float32)
            return np.zeros((3, depth.shape[0], depth.shape[1]), dtype=np.float32), empty_mask
        
        d_med = np.median(depth[valid_original])
        depth_cleaned = depth.copy()
        depth_cleaned[~valid_original] = d_med
        
        # Create a simple panel mask (all valid pixels = panel)
        panel_mask = valid_original.astype(np.float32)
    
    # --- STEP 2: Resize depth and mask to target size if specified ---
    if target_size is not None:
        target_h, target_w = target_size
        # Resize depth_cleaned for gradient computation
        depth_cleaned = cv2.resize(depth_cleaned, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        # Resize panel_mask using NEAREST to keep it binary (0 or 1)
        panel_mask = cv2.resize(panel_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        # Ensure mask is binary (0.0 or 1.0)
        panel_mask = (panel_mask > 0.5).astype(np.float32)
        # Resize original depth for normalization (use LINEAR for depth values)
        depth_for_norm = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        current_shape = (target_h, target_w)
    else:
        depth_for_norm = depth
        current_shape = original_shape
    
    # --- STEP 3: Compute normalization stats from ORIGINAL depth (before filling) ---
    # CRITICAL: We need to compute normalization stats from ORIGINAL depth (before filling)
    # to match the notebook training pipeline exactly.
    # This ensures the normalization range is computed from original valid pixels only,
    # not including the median-filled background pixels.
    # Use depth_for_norm (which is original depth, possibly resized)
    valid_original = np.isfinite(depth_for_norm) & (depth_for_norm > 0)
    if not np.any(valid_original):
        # Fallback for empty images
        return np.zeros((3, current_shape[0], current_shape[1]), dtype=np.float32), panel_mask
    
    # Get min/max from ORIGINAL valid pixels (for normalization)
    d_valid_original = depth_for_norm[valid_original]
    d_min = np.min(d_valid_original)
    d_max = np.max(d_valid_original)
    range_val = d_max - d_min
    if range_val < 1e-6:
        range_val = 1.0
    
    # --- STEP 4: Normalize Depth Channel to [0.1, 1.0] ---
    # Map valid range to [0.1, 1.0] (makes Wall distinct from 0.0 background)
    # CRITICAL: Normalize using ORIGINAL depth values (resized if target_size specified)
    depth_n = np.zeros_like(depth_for_norm)
    # Formula: 0.1 + 0.9 * (val - min) / (max - min)
    depth_n[valid_original] = 0.1 + 0.9 * ((depth_for_norm[valid_original] - d_min) / range_val)
    
    # --- STEP 5: Normalize CLEANED Depth for Gradient Computation ---
    # CRITICAL: Match training pipeline - gradients must be computed on NORMALIZED depth
    # Training computes gradients on normalized filled depth, so we do the same here
    # Normalize depth_cleaned using the same stats as original depth
    depth_cleaned_norm = np.zeros_like(depth_cleaned)
    valid_cleaned = (depth_cleaned > 0) & np.isfinite(depth_cleaned)
    if np.any(valid_cleaned):
        # Use same normalization stats (d_min, d_max, range_val) for consistency
        depth_cleaned_norm[valid_cleaned] = 0.1 + 0.9 * ((depth_cleaned[valid_cleaned] - d_min) / range_val)
    
    # --- STEP 6: Compute Gradients on NORMALIZED CLEANED Depth ---
    # CRITICAL: Match training exactly - gradients computed on normalized depth (not raw cleaned depth)
    # This matches the training pipeline where gradients are computed on depth_norm
    gx = cv2.Sobel(depth_cleaned_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_cleaned_norm, cv2.CV_32F, 0, 1, ksize=3)
    
    # --- STEP 7: Robust Normalization for Gradients ---
    def robust_norm(x):
        v = x.flatten()
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.zeros_like(x, dtype=np.float32)
        med = np.median(v)
        mad = np.median(np.abs(v - med)) + 1e-6
        out = (x - med) / (3.0 * mad)
        out = np.clip(out, -3.0, 3.0)
        out = (out - out.min()) / (out.max() - out.min() + 1e-9)
        return out.astype(np.float32)
    
    gx_n = robust_norm(gx)
    gy_n = robust_norm(gy)
    
    # --- STEP 8: MASK the Gradients ---
    # We computed gradients on cleaned data, but we force background to 0.0
    # for the final network input (removes background noise).
    valid_mask = (depth_n > 0).astype(np.float32)
    gx_n = gx_n * valid_mask
    gy_n = gy_n * valid_mask
    
    # --- STEP 9: Stack channels ---
    inp = np.stack([depth_n, gx_n, gy_n], axis=0).astype(np.float32)
    
    return inp, panel_mask


def _aggressive_internal_fill(binary_mask: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Aggressively fill black pixels inside white regions regardless of depth.
    This catches any remaining black lines that weren't filled by morphological operations.
    
    Args:
        binary_mask: Binary mask (H, W) where 255 = dent, 0 = background
        valid_mask: Valid region mask (H, W) where True = valid panel area
        
    Returns:
        Binary mask with internal black pixels filled
    """
    # Convert to binary (0 or 1) for processing
    mask_binary = (binary_mask > 127).astype(np.uint8)
    
    # Find contours of white (dent) regions
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a new mask
    filled_mask = np.zeros_like(mask_binary)
    
    # Fill each contour (this fills holes inside white regions)
    for contour in contours:
        if cv2.contourArea(contour) > 0:
            cv2.fillPoly(filled_mask, [contour], 1)
    
    # Only keep filled regions that are within valid_mask
    filled_mask = filled_mask * valid_mask.astype(np.uint8)
    
    # Convert back to 0/255 format
    return filled_mask.astype(np.uint8) * 255


def _fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    """
    Fill black holes inside white dented regions using morphological closing.
    
    Args:
        binary_mask: Binary mask (H, W) where 255 = dent, 0 = background
        
    Returns:
        Binary mask with holes filled
    """
    # Convert to binary (0 or 1) for morphological operations
    mask_binary = (binary_mask > 127).astype(np.uint8)
    
    # Use morphological closing to fill small holes
    # Kernel size of 5x5 should catch most small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filled_mask = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to 0/255 format
    return filled_mask.astype(np.uint8) * 255


def _filter_thin_components(binary_mask: np.ndarray, min_area: int = 10) -> np.ndarray:
    """
    Remove small and thin false-positive regions using connected-component analysis.
    
    Args:
        binary_mask: Binary mask (H, W) where 255 = dent, 0 = background
        min_area: Minimum area in pixels for a component to be kept (default: 10)
        
    Returns:
        Binary mask with small/thin components removed
    """
    # Convert to binary (0 or 1) for connected components
    mask_binary = (binary_mask > 127).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    
    # Create output mask
    filtered_mask = np.zeros_like(mask_binary)
    
    # Keep only components with area >= min_area
    # stats format: [x, y, width, height, area]
    # Index 4 is the area
    for label_id in range(1, num_labels):  # Skip background (label 0)
        area = stats[label_id, 4]  # Area is at index 4
        if area >= min_area:
            # Keep this component
            filtered_mask[labels == label_id] = 1
    
    # Convert back to 0/255 format
    return filtered_mask.astype(np.uint8) * 255


def predict_mask(model: nn.Module, depth: np.ndarray, device: str = 'cpu', threshold: float = 0.5,
                 target_size: Optional[Tuple[int, int]] = None,
                 min_dent_area: int = 200,
                 depth_cleaned: Optional[np.ndarray] = None,
                 panel_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on a depth map to generate binary segmentation mask with Hard Masking.
    
    This function implements "Hard Masking" for background suppression:
    - Uses pre-computed panel mask (RANSAC should be applied BEFORE calling this function)
    - Runs model inference to get raw probability map
    - Multiplies probability map by panel_mask to force background predictions to 0.0
    - Applies post-processing: aggressive fill, hole filling, and component filtering
    
    NOTE: RANSAC panel extraction should be done BEFORE calling this function.
    Use panel_extractor.py to extract panel, then pass the results here.
    
    Args:
        model: Trained AttentionUNet model
        depth: Input depth map (H, W) as numpy array
        device: Device to run inference on ('cpu' or 'cuda')
        threshold: Threshold for binary mask (default 0.5)
        target_size: Optional target size (H, W) to resize depth and mask. If None, uses original size.
        min_dent_area: Minimum area in pixels for a dent component to be kept (default: 200)
        depth_cleaned: Pre-computed cleaned depth map (H, W) with background filled.
                      If None, uses simple median fill fallback.
        panel_mask: Pre-computed panel mask (H, W) where 1.0 = panel, 0.0 = background.
                    If None, creates mask from valid pixels.
        
    Returns:
        Tuple of (binary_mask, prob_mask)
        - binary_mask: Binary segmentation mask (H, W) as numpy array (0 or 255)
                      Background areas are forced to 0 via hard masking
                      Post-processed to fill holes and remove small components
        - prob_mask: Probability mask (H, W) as numpy array (0.0 to 1.0)
                     Background areas are forced to 0.0 via hard masking
    """
    model.eval()
    original_shape = depth.shape[:2]  # (H, W)
    
    # Preprocess depth and get panel mask
    # RANSAC should be applied BEFORE calling this function (e.g., in app.py)
    inp, panel_mask = preprocess_depth(
        depth, 
        target_size=target_size,
        depth_cleaned=depth_cleaned,
        panel_mask=panel_mask
    )
    
    # Convert to tensor and add batch dimension: (1, 3, H, W)
    inp_tensor = torch.from_numpy(inp).unsqueeze(0).to(device)
    
    # Inference - get raw probability map
    with torch.no_grad():
        output = model(inp_tensor)
        raw_prob_mask = torch.sigmoid(output).cpu().numpy()[0, 0]  # (H, W)
    
    # --- HARD MASKING: Multiply probability map by panel_mask ---
    # This forces any prediction in background areas to be exactly 0.0
    prob_mask = raw_prob_mask * panel_mask
    
    # Resize prob_mask back to original size if we resized during preprocessing
    if target_size is not None and prob_mask.shape != original_shape:
        prob_mask = cv2.resize(prob_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        # Also resize panel_mask for consistency (though we don't return it)
        panel_mask_resized = cv2.resize(panel_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
        # Re-apply hard masking after resize to ensure background is still 0
        prob_mask = prob_mask * panel_mask_resized
    
    # Convert to binary mask
    binary_mask = (prob_mask > threshold).astype(np.uint8) * 255
    
    # --- POST-PROCESSING: Apply morphological operations to clean up the mask ---
    # Get valid mask (panel area) for aggressive fill
    # binary_mask is now at original_shape (after resize if needed)
    if target_size is not None:
        # We resized during preprocessing, so use the resized panel_mask
        # panel_mask_resized was already computed above
        valid_mask = (panel_mask_resized > 0.5)
    else:
        # No resize, panel_mask should match binary_mask shape
        if panel_mask.shape != binary_mask.shape:
            # Resize panel_mask to match binary_mask if there's a mismatch
            panel_mask_for_valid = cv2.resize(panel_mask, (binary_mask.shape[1], binary_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            valid_mask = (panel_mask_for_valid > 0.5)
        else:
            valid_mask = (panel_mask > 0.5)
    
    # Post-morphology aggressive fill: Fill black pixels inside white regions regardless of depth
    # This catches any remaining black lines that weren't filled by morphological operations
    binary_mask = _aggressive_internal_fill(binary_mask, valid_mask)
    
    # Fill black holes inside white dented regions
    binary_mask = _fill_holes(binary_mask)
    
    # Remove small and thin false-positive regions using connected-component analysis
    binary_mask = _filter_thin_components(binary_mask, min_area=min_dent_area)
    
    return binary_mask, prob_mask


def create_dent_overlay(rgb_image: np.ndarray, dent_mask: np.ndarray,
                       overlay_alpha: float = 0.2, outline_thickness: int = 2) -> np.ndarray:
    """
    Create a visual overlay showing dent regions on the RGB image.
    Similar to compare_dents_depth_visual_output.py
    
    Args:
        rgb_image: RGB image (H, W, 3) as numpy array (0-255 uint8)
        dent_mask: Binary mask (H, W) where WHITE (255) = dented areas, BLACK (0) = normal areas
        overlay_alpha: Transparency of the overlay fill (0.0-1.0, default: 0.2)
        outline_thickness: Thickness of the red outline in pixels (default: 2)
        
    Returns:
        RGB image with dent overlay visualization (H, W, 3) as numpy array (0-255 uint8)
    """
    # Ensure mask and RGB image have matching dimensions
    if rgb_image.shape[:2] != dent_mask.shape[:2]:
        h, w = rgb_image.shape[:2]
        dent_mask = cv2.resize(dent_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Convert RGB to float for blending
    background = rgb_image.astype(np.float32) / 255.0
    
    # Create red overlay color (RGB: red = [1, 0, 0])
    overlay_color = np.array([1.0, 0.0, 0.0])  # Red color
    
    # Create binary mask (0 or 1) from dent_mask
    mask_binary = (dent_mask > 127).astype(np.float32)
    
    # Create overlay image with red color
    overlay = np.zeros_like(background)
    overlay[mask_binary > 0] = overlay_color
    
    # Blend overlay with background using alpha transparency
    # result = background * (1 - alpha * mask) + overlay * (alpha * mask)
    alpha_mask = mask_binary * overlay_alpha
    result = background * (1.0 - alpha_mask[..., np.newaxis]) + overlay * alpha_mask[..., np.newaxis]
    
    # Add red outline for better visibility
    # Find contours of the dent mask
    mask_uint8 = (mask_binary * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw red outline on the result
    result_uint8 = (result * 255).astype(np.uint8)
    cv2.drawContours(result_uint8, contours, -1, (255, 0, 0), outline_thickness)
    
    return result_uint8


def load_camera_intrinsics(json_path: Optional[str] = None) -> dict:
    """
    Load camera intrinsics from JSON file.
    
    Args:
        json_path: Path to camera intrinsics JSON file. If None, uses default path.
        
    Returns:
        Dictionary with camera intrinsics: fx, fy, cx, cy, fov_degrees, resolution
    """
    if json_path is None:
        # Default path relative to this file
        default_path = Path(__file__).parent / "camera_intrinsics_default.json"
        json_path = str(default_path)
    
    try:
        with open(json_path, 'r') as f:
            intrinsics = json.load(f)
        
        # Extract intrinsics
        result = {
            'fx': intrinsics.get('fx'),
            'fy': intrinsics.get('fy'),
            'cx': intrinsics.get('cx'),
            'cy': intrinsics.get('cy'),
            'fov_degrees': intrinsics.get('fov_degrees', {}),
            'resolution': intrinsics.get('resolution', {})
        }
        
        return result
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"Warning: Could not load camera intrinsics from {json_path}: {e}")
        return {}


def calculate_dent_metrics(depth_map: np.ndarray, dent_mask: np.ndarray, 
                           pixel_to_cm: float = None,
                           depth_units: str = 'meters',
                           camera_fov: float = None,
                           focal_length: float = None,
                           sensor_width: float = None,
                           intrinsics_json_path: Optional[str] = None,
                           panel_mask: Optional[np.ndarray] = None) -> dict:
    """
    Calculate dent metrics including area and maximum depth using camera intrinsics.
    
    IMPORTANT: Real-world area computation requires camera calibration information.
    Uses camera intrinsics to calculate depth-dependent pixel sizes for accurate area.
    
    By default, loads intrinsics from camera_intrinsics_default.json if available.
    
    Depth measurement compares dent depth to the median of the normal panel surface
    (panel regions excluding dent regions).
    
    Args:
        depth_map: Original depth map (H, W) as numpy array in meters
        dent_mask: Binary mask (H, W) where WHITE (255) = dented areas
        pixel_to_cm: Conversion factor from pixels to cm (fallback if intrinsics not provided).
                     If camera intrinsics are provided, this is ignored in favor of intrinsics-based calculation.
        depth_units: Units of depth_map values. Options: 'meters', 'mm', 'cm', 'inches'
                     Used to convert depth differences to mm. Default: 'meters'
        camera_fov: Camera field of view in degrees (for intrinsics-based calculation).
                    If provided along with depth_map dimensions, calculates focal length.
                    If None, will try to load from intrinsics JSON file.
        focal_length: Focal length in pixels (alternative to camera_fov).
                      If provided, uses this directly for intrinsics calculation.
                      If None, will try to load from intrinsics JSON file (fx or fy).
        sensor_width: Sensor width in mm (for intrinsics-based calculation).
                      If None, uses FOV-based approximation.
        intrinsics_json_path: Path to camera intrinsics JSON file. If None, uses default path.
        panel_mask: Optional panel mask (H, W) where 1.0 = panel pixels, 0.0 = background.
                    If provided, uses median of panel regions (excluding dent regions) as reference depth.
                    If None, falls back to median of non-dent regions.
        
    Returns:
        Dictionary with metrics:
        - area_cm2: Total dent area in cm² (calculated using intrinsics if available)
        - area_valid: Boolean indicating if area computation is physically valid
        - max_depth_mm: Maximum depth difference in mm (compared to panel median)
        - num_defects: Number of separate dent regions
        - avg_depth_mm: Average depth difference in mm (compared to panel median)
        - pixel_count: Number of pixels in dent regions (always available)
        - missing_info: List of missing information needed for valid area computation
        - area_method: Method used for area calculation ('intrinsics' or 'pixel_to_cm')
    """
    missing_info = []
    
    # Load camera intrinsics from JSON file by default (prioritize over FOV-based calculation)
    intrinsics = load_camera_intrinsics(intrinsics_json_path)
    
    # Use intrinsics from JSON if available (more accurate than FOV-based calculation)
    if intrinsics.get('fx') is not None:
        if focal_length is None:
            # Use fx as focal length (or average of fx and fy if both available)
            fx = intrinsics['fx']
            fy = intrinsics.get('fy', fx)
            focal_length = (fx + fy) / 2.0  # Average focal length
    
    if camera_fov is None and intrinsics.get('fov_degrees'):
        # Use vertical FOV if available, otherwise horizontal
        fov_dict = intrinsics['fov_degrees']
        camera_fov = fov_dict.get('vertical') or fov_dict.get('horizontal')
    
    # CRITICAL: Detect and convert millimeter values to meters if needed
    # If max depth > 100, likely still in millimeters (should be < 10m for containers)
    depth_map_converted = depth_map.copy()
    if np.nanmax(np.abs(depth_map)) > 100:
        # Likely still in millimeters - convert to meters
        depth_map_converted = depth_map_converted / 1000.0
        # Filter error codes and far background
        depth_map_converted[depth_map_converted > 3.0] = 0.0
        depth_map_converted[depth_map_converted < 0] = 0.0
        missing_info.append("Depth map appeared to be in millimeters - converted to meters for calculation")
    elif np.nanmax(np.abs(depth_map)) > 10.0:
        # Values > 10m are likely noise/background for container inspection
        depth_map_converted[depth_map_converted > 10.0] = 0.0
        missing_info.append("Filtered depth values > 10m (likely background noise)")
    
    # Handle negative values (convert to positive)
    if np.any(depth_map_converted < 0):
        depth_map_converted = np.abs(depth_map_converted)
    
    # Use converted depth map for all calculations
    depth_map = depth_map_converted
    
    # Validate pixel_to_cm for area computation
    area_valid = False
    if pixel_to_cm is None:
        missing_info.append("pixel_to_cm conversion factor (required for area computation)")
    elif pixel_to_cm <= 0:
        missing_info.append("pixel_to_cm must be positive (invalid value provided)")
    elif pixel_to_cm > 10:  # Sanity check: >10 cm/pixel seems unreasonable for typical setups
        missing_info.append(f"pixel_to_cm value ({pixel_to_cm}) seems unusually large - please verify calibration")
        area_valid = False  # Flag as suspicious but don't refuse
    else:
        area_valid = True
    
    # Ensure masks match dimensions
    if depth_map.shape != dent_mask.shape:
        h, w = depth_map.shape
        dent_mask = cv2.resize(dent_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create binary mask (True for dent regions)
    dent_binary = (dent_mask > 127).astype(bool)
    
    # Count number of separate dent regions (connected components)
    mask_uint8 = (dent_binary * 255).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_uint8)
    num_defects = num_labels - 1  # Subtract 1 because background is label 0
    
    # Calculate pixel count (always available)
    num_dent_pixels = np.sum(dent_binary)
    
    # Calculate area using camera intrinsics if available, otherwise use pixel_to_cm
    area_cm2 = None
    area_method = None
    
    if num_dent_pixels > 0:
        # Try intrinsics-based calculation first
        h, w = depth_map.shape
        
        # Prioritize focal_length from intrinsics JSON over FOV-based calculation
        # (intrinsics are more accurate)
        if focal_length is not None:
            focal_length_px = focal_length
        elif camera_fov is not None:
            # Fallback: Calculate focal length from FOV
            fov_y_rad = np.deg2rad(camera_fov)
            focal_length_px = (h / 2.0) / np.tan(fov_y_rad / 2.0)
        else:
            focal_length_px = None
        
        # Calculate area using intrinsics (depth-dependent pixel size)
        if focal_length_px is not None:
            # Get depths for dent pixels
            dent_depths_for_area = depth_map[dent_binary]
            valid_depths_for_area = dent_depths_for_area[np.isfinite(dent_depths_for_area) & (dent_depths_for_area > 0)]
            
            if len(valid_depths_for_area) > 0:
                # Calculate pixel size at each depth using camera intrinsics
                # Using pinhole camera model: pixel_size_m = depth / focal_length_px
                # This gives the physical size of one pixel at that depth in meters
                pixel_size_at_depth_m = valid_depths_for_area / focal_length_px
                
                # If sensor_width is provided, use more accurate calculation
                if sensor_width is not None:
                    # Convert sensor_width from mm to meters
                    sensor_width_m = sensor_width / 1000.0
                    # Focal length in meters (from sensor geometry)
                    # focal_length_m = (sensor_width_m * focal_length_px) / w
                    # More accurate: pixel_size = depth * (sensor_width_m / focal_length_m) / w
                    # But we can simplify using the relationship: focal_length_px / w ≈ focal_length_m / sensor_width_m
                    # So: pixel_size_m = depth * sensor_width_m / (focal_length_px * w / focal_length_px * sensor_width_m / w)
                    # Simplified: pixel_size_m = depth * sensor_width_m / (focal_length_px * w) * w
                    # Actually: pixel_size_m = depth * sensor_width_m / (focal_length_m * w)
                    # Where focal_length_m = sensor_width_m * focal_length_px / w
                    focal_length_m = (sensor_width_m * focal_length_px) / w
                    pixel_size_at_depth_m = valid_depths_for_area * (sensor_width_m / (focal_length_m * w))
                
                # Convert pixel size from meters to cm
                pixel_size_cm = pixel_size_at_depth_m * 100.0
                
                # Calculate area for each pixel and sum
                pixel_area_cm2 = pixel_size_cm ** 2
                area_cm2 = np.sum(pixel_area_cm2)
                area_method = 'intrinsics'
                area_valid = True
        
        # Fallback to pixel_to_cm if intrinsics not available
        if area_cm2 is None and area_valid and pixel_to_cm is not None:
            # Calculate area in cm² using constant pixel size
            # Area = number of pixels * (pixel_to_cm)^2
            area_cm2 = num_dent_pixels * (pixel_to_cm ** 2)
            area_method = 'pixel_to_cm'
    
    if area_cm2 is None:
        area_method = None
    
    if not np.any(dent_binary):
        return {
            'area_cm2': area_cm2,
            'area_valid': area_valid,
            'max_depth_mm': 0.0,
            'num_defects': 0,
            'avg_depth_mm': 0.0,
            'pixel_count': num_dent_pixels,
            'missing_info': missing_info
        }
    
    # Calculate depth metrics
    # Get depth values in dent regions
    dent_depths = depth_map[dent_binary]
    valid_depths = dent_depths[np.isfinite(dent_depths) & (dent_depths > 0)]
    
    if len(valid_depths) == 0:
        return {
            'area_cm2': area_cm2,
            'area_valid': area_valid,
            'max_depth_mm': 0.0,
            'num_defects': num_defects,
            'avg_depth_mm': 0.0,
            'pixel_count': num_dent_pixels,
            'missing_info': missing_info
        }
    
    # Get reference depth: median of normal panel surface (panel regions excluding dent regions)
    # This represents the expected depth of the panel surface
    if panel_mask is not None:
        # Ensure panel_mask matches depth_map dimensions
        if panel_mask.shape != depth_map.shape:
            h, w = depth_map.shape
            panel_mask_resized = cv2.resize(panel_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            panel_mask_resized = panel_mask
        
        # Panel regions: where panel_mask > 0.5
        panel_binary = (panel_mask_resized > 0.5)
        
        # Normal panel surface: panel regions excluding dent regions
        normal_panel_binary = panel_binary & ~dent_binary
        
        if np.any(normal_panel_binary):
            normal_panel_depths = depth_map[normal_panel_binary]
            valid_normal_panel = normal_panel_depths[np.isfinite(normal_panel_depths) & (normal_panel_depths > 0)]
            if len(valid_normal_panel) > 0:
                reference_depth = np.median(valid_normal_panel)
            else:
                # Fallback: use all panel regions (including dents) if no valid normal panel
                panel_depths = depth_map[panel_binary]
                valid_panel = panel_depths[np.isfinite(panel_depths) & (panel_depths > 0)]
                if len(valid_panel) > 0:
                    reference_depth = np.median(valid_panel)
                else:
                    reference_depth = np.median(valid_depths)
        else:
            # Fallback: use all panel regions if no normal panel surface found
            panel_depths = depth_map[panel_binary]
            valid_panel = panel_depths[np.isfinite(panel_depths) & (panel_depths > 0)]
            if len(valid_panel) > 0:
                reference_depth = np.median(valid_panel)
            else:
                reference_depth = np.median(valid_depths)
    else:
        # Fallback: use median of non-dent regions if panel_mask not provided
        non_dent_binary = ~dent_binary
        if np.any(non_dent_binary):
            non_dent_depths = depth_map[non_dent_binary]
            valid_non_dent = non_dent_depths[np.isfinite(non_dent_depths) & (non_dent_depths > 0)]
            if len(valid_non_dent) > 0:
                reference_depth = np.median(valid_non_dent)
            else:
                reference_depth = np.median(valid_depths)
        else:
            reference_depth = np.median(valid_depths)
    
    # Calculate depth differences (dents are typically depressions, so depth < reference)
    depth_differences = reference_depth - valid_depths
    depth_differences = np.maximum(depth_differences, 0)  # Only positive differences
    
    # Convert depth differences to mm based on depth_units
    depth_conversion_factors = {
        'meters': 1000.0,  # meters to mm
        'mm': 1.0,         # already in mm
        'cm': 10.0,        # cm to mm
        'inches': 25.4     # inches to mm
    }
    
    if depth_units not in depth_conversion_factors:
        missing_info.append(f"Unknown depth_units '{depth_units}'. Assuming meters.")
        conversion_factor = 1000.0
    else:
        conversion_factor = depth_conversion_factors[depth_units]
    
    max_depth_mm = np.max(depth_differences) * conversion_factor if depth_differences.size > 0 else 0.0
    avg_depth_mm = np.mean(depth_differences) * conversion_factor if depth_differences.size > 0 else 0.0
    
    return {
        'area_cm2': area_cm2,
        'area_valid': area_valid,
        'max_depth_mm': max_depth_mm,
        'num_defects': num_defects,
        'avg_depth_mm': avg_depth_mm,
        'pixel_count': num_dent_pixels,
        'missing_info': missing_info,
        'area_method': area_method
    }

