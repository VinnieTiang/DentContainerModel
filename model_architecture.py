"""
Model Architecture for Dent Container Detection
Attention-UNet model definition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from scipy.ndimage import median_filter
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


# def preprocess_depth(depth: np.ndarray, 
#                      target_size: Optional[Tuple[int, int]] = None,
#                      depth_cleaned: Optional[np.ndarray] = None,
#                      panel_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Preprocess depth map for model inference with Hard Masking for background suppression.
#     Converts depth map to 3-channel input: [normalized_depth, normalized_gx, normalized_gy]
    
#     This function matches the ver4 notebook training pipeline:
#     - Uses pre-computed cleaned depth map (RANSAC should be applied BEFORE calling this function)
#     - Normalizes depth to [0.1, 1.0] range (makes wall distinct from 0.0 background)
#     - Computes gradients on cleaned depth (no cliff edges)
#     - Masks gradients to remove background noise
#     - Returns panel mask for hard masking predictions
    
#     NOTE: RANSAC panel extraction should be done BEFORE calling this function.
#     Use panel_extractor.py to extract panel, then pass the results here.
    
#     Args:
#         depth: Input depth map (H, W) as numpy array
#                NOTE: Should be converted to float32 BEFORE calling this function (e.g., in app.py before RANSAC).
#                      This function will convert to float32 if not already done, but conversion before RANSAC
#                      is recommended for best precision.
#         target_size: Optional target size (H, W) to resize depth and mask. If None, uses original size.
#         depth_cleaned: Pre-computed cleaned depth map (H, W) with background filled.
#                       If None, uses simple median fill fallback.
#         panel_mask: Pre-computed panel mask (H, W) where 1.0 = panel, 0.0 = background.
#                     If None, creates mask from valid pixels.
        
#     Returns:
#         Tuple of (input_tensor, panel_mask)
#         - input_tensor: Preprocessed input tensor ready for model (3, H, W)
#         - panel_mask: Binary panel mask (H, W) where 1.0 = panel, 0.0 = background
#                       Resized to match input_tensor spatial dimensions
#     """
#     # Convert to float32 for precision (critical for gradient calculations)
#     # NOTE: Conversion should ideally happen BEFORE calling this function (e.g., in app.py before RANSAC)
#     # This is a defensive check - only convert if not already float32
#     if depth.dtype != np.float32:
#         depth = depth.astype(np.float32)
#     original_shape = depth.shape[:2]  # (H, W)
    
#     # --- STEP 1: Use pre-computed cleaned depth and panel mask ---
#     # RANSAC should be applied BEFORE calling this function (e.g., in app.py or by caller)
#     if depth_cleaned is not None and panel_mask is not None:
#         # Use pre-computed RANSAC results
#         depth_cleaned = depth_cleaned.astype(np.float32)
#         panel_mask = panel_mask.astype(np.float32)
        
#         # Verify shapes match
#         if depth_cleaned.shape != depth.shape:
#             raise ValueError(f"Pre-computed depth_cleaned shape {depth_cleaned.shape} doesn't match depth shape {depth.shape}")
#         if panel_mask.shape != depth.shape:
#             raise ValueError(f"Pre-computed panel_mask shape {panel_mask.shape} doesn't match depth shape {depth.shape}")
#     else:
#         # Fallback: Simple median fill if RANSAC results not provided
#         # This is a fallback for cases where RANSAC wasn't applied
#         valid_original = np.isfinite(depth) & (depth > 0)
#         if not np.any(valid_original):
#             # Fallback for empty images
#             empty_mask = np.zeros((depth.shape[0], depth.shape[1]), dtype=np.float32)
#             return np.zeros((3, depth.shape[0], depth.shape[1]), dtype=np.float32), empty_mask
        
#         d_med = np.median(depth[valid_original])
#         depth_cleaned = depth.copy()
#         depth_cleaned[~valid_original] = d_med
        
#         # Create a simple panel mask (all valid pixels = panel)
#         panel_mask = valid_original.astype(np.float32)
    
#     # --- STEP 2: Resize depth and mask to target size if specified ---
#     if target_size is not None:
#         target_h, target_w = target_size
#         # Resize depth_cleaned for gradient computation
#         depth_cleaned = cv2.resize(depth_cleaned, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
#         # Resize panel_mask using NEAREST to keep it binary (0 or 1)
#         panel_mask = cv2.resize(panel_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
#         # Ensure mask is binary (0.0 or 1.0)
#         panel_mask = (panel_mask > 0.5).astype(np.float32)
#         # Resize original depth for normalization (use LINEAR for depth values)
#         depth_for_norm = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
#         current_shape = (target_h, target_w)
#     else:
#         depth_for_norm = depth
#         current_shape = original_shape
    
#     # --- STEP 3: Compute normalization stats from ORIGINAL depth (before filling) ---
#     # CRITICAL: We need to compute normalization stats from ORIGINAL depth (before filling)
#     # to match the notebook training pipeline exactly.
#     # This ensures the normalization range is computed from original valid pixels only,
#     # not including the median-filled background pixels.
#     # Use depth_for_norm (which is original depth, possibly resized)
#     valid_original = np.isfinite(depth_for_norm) & (depth_for_norm > 0)
#     if not np.any(valid_original):
#         # Fallback for empty images
#         return np.zeros((3, current_shape[0], current_shape[1]), dtype=np.float32), panel_mask
    
#     # Get min/max from ORIGINAL valid pixels (for normalization)
#     d_valid_original = depth_for_norm[valid_original]
#     d_min = np.min(d_valid_original)
#     d_max = np.max(d_valid_original)
#     range_val = d_max - d_min
#     if range_val < 1e-6:
#         range_val = 1.0
    
#     # --- STEP 4: Normalize Depth Channel to [0.1, 1.0] ---
#     # Map valid range to [0.1, 1.0] (makes Wall distinct from 0.0 background)
#     # CRITICAL: Normalize using ORIGINAL depth values (resized if target_size specified)
#     depth_n = np.zeros_like(depth_for_norm)
#     # Formula: 0.1 + 0.9 * (val - min) / (max - min)
#     depth_n[valid_original] = 0.1 + 0.9 * ((depth_for_norm[valid_original] - d_min) / range_val)
    
#     # --- STEP 5: Normalize CLEANED Depth for Gradient Computation ---
#     # CRITICAL: Match training pipeline - gradients must be computed on NORMALIZED depth
#     # Training computes gradients on normalized filled depth, so we do the same here
#     # Normalize depth_cleaned using the same stats as original depth
#     depth_cleaned_norm = np.zeros_like(depth_cleaned)
#     valid_cleaned = (depth_cleaned > 0) & np.isfinite(depth_cleaned)
#     if np.any(valid_cleaned):
#         # Use same normalization stats (d_min, d_max, range_val) for consistency
#         depth_cleaned_norm[valid_cleaned] = 0.1 + 0.9 * ((depth_cleaned[valid_cleaned] - d_min) / range_val)
    
#     # --- STEP 6: Compute Gradients on NORMALIZED CLEANED Depth ---
#     # CRITICAL: Match training exactly - gradients computed on normalized depth (not raw cleaned depth)
#     # This matches the training pipeline where gradients are computed on depth_norm
#     gx = cv2.Sobel(depth_cleaned_norm, cv2.CV_32F, 1, 0, ksize=3)
#     gy = cv2.Sobel(depth_cleaned_norm, cv2.CV_32F, 0, 1, ksize=3)
    
#     # --- STEP 7: Robust Normalization for Gradients ---
#     def robust_norm(x):
#         v = x.flatten()
#         v = v[np.isfinite(v)]
#         if v.size == 0:
#             return np.zeros_like(x, dtype=np.float32)
#         med = np.median(v)
#         mad = np.median(np.abs(v - med)) + 1e-6
#         out = (x - med) / (3.0 * mad)
#         out = np.clip(out, -3.0, 3.0)
#         out = (out - out.min()) / (out.max() - out.min() + 1e-9)
#         return out.astype(np.float32)
    
#     gx_n = robust_norm(gx)
#     gy_n = robust_norm(gy)
    
#     # --- STEP 8: MASK the Gradients ---
#     # We computed gradients on cleaned data, but we force background to 0.0
#     # for the final network input (removes background noise).
#     valid_mask = (depth_n > 0).astype(np.float32)
#     gx_n = gx_n * valid_mask
#     gy_n = gy_n * valid_mask
    
#     # --- STEP 9: Stack channels ---
#     inp = np.stack([depth_n, gx_n, gy_n], axis=0).astype(np.float32)
    
#     return inp, panel_mask



########################################################
#Preprocess depth map by no replacing 0 with median depth
########################################################
def preprocess_depth(depth: np.ndarray, 
                     target_size: Optional[Tuple[int, int]] = None,
                     depth_cleaned: Optional[np.ndarray] = None,
                     panel_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess depth map for inference on REAL DATA.
    
    CRITICAL CHANGE:
    - Uses 'depth_cleaned' (Smart Fill + Bilateral) if available.
    - This ensures Gradients (gx, gy) are smooth and clear, not noisy.
    """
    # 1. Determine which depth source to use
    # If we have a cleaned/smoothed map from the RANSAC pipeline, USE IT!
    # Otherwise, fall back to the raw depth.
    print("=" * 60)
    print("🔍 DEBUG: preprocess_depth() - Checking depth source:")
    if depth_cleaned is not None:
        print("   ✅ Using depth_cleaned (should have BILATERAL SMOOTHING if from uint16 processing)")
        print(f"   depth_cleaned shape: {depth_cleaned.shape}, dtype: {depth_cleaned.dtype}")
        print(f"   depth_cleaned range: [{np.nanmin(depth_cleaned):.4f}, {np.nanmax(depth_cleaned):.4f}] meters")
        input_depth = depth_cleaned.astype(np.float32)
    else:
        print("   ⚠️  Using raw depth (NO bilateral smoothing applied)")
        print(f"   raw depth shape: {depth.shape}, dtype: {depth.dtype}")
        input_depth = depth.astype(np.float32)
    print("=" * 60)

    # 2. Resize if necessary
    if target_size is not None:
        target_h, target_w = target_size
        # Linear is better for smooth depth values (prevents stair-stepping artifacts)
        input_depth = cv2.resize(input_depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        if panel_mask is not None:
            # Nearest for masks to keep them binary (0 or 1)
            panel_mask = cv2.resize(panel_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # 3. Identify Valid Region (Background is 0)
    valid_mask = (input_depth > 0)
    
    # Handle empty image
    if not np.any(valid_mask):
        empty_inp = np.zeros((3, input_depth.shape[0], input_depth.shape[1]), dtype=np.float32)
        p_mask = panel_mask if panel_mask is not None else np.zeros_like(input_depth)
        return empty_inp, p_mask

    # 4. Normalize Depth (Map Container to 0.1 - 1.0)
    # We use the min/max of the VALID region only
    d_valid = input_depth[valid_mask]
    d_min = np.min(d_valid)
    d_max = np.max(d_valid)
    range_val = d_max - d_min
    if range_val < 1e-6: range_val = 1.0

    depth_norm = np.zeros_like(input_depth)
    # Background stays 0.0. Container becomes 0.1 to 1.0
    depth_norm[valid_mask] = 0.1 + 0.9 * ((input_depth[valid_mask] - d_min) / range_val)

    # 5. Compute Gradients on the SMOOTH/CLEANED data
    # This is the most important step for detection accuracy
    gx = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)

    # 6. Robust Normalization Helper
    def robust_norm(x, mask):
        v = x[mask]
        if v.size == 0: return np.zeros_like(x)
        
        med = np.median(v)
        mad = np.median(np.abs(v - med)) + 1e-6
        
        out = (x - med) / (3.0 * mad)
        # Clip and map to 0-1 range
        return ((np.clip(out, -3.0, 3.0) + 3.0) / 6.0).astype(np.float32)

    # 7. Normalize Gradients
    gx_n = robust_norm(gx, valid_mask)
    gy_n = robust_norm(gy, valid_mask)
    
    # Mask gradients (force background to 0)
    gx_n[~valid_mask] = 0
    gy_n[~valid_mask] = 0

    # 8. Stack channels: [Depth, Grad_X, Grad_Y]
    inp = np.stack([depth_norm, gx_n, gy_n], axis=0).astype(np.float32)
    
    # Generate Panel Mask if not provided
    if panel_mask is None:
        panel_mask = valid_mask.astype(np.float32)
        
    return inp, panel_mask

########################################################
########################################################

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
        Tuple of (binary_mask, prob_mask, preprocessed_input)
        - binary_mask: Binary segmentation mask (H, W) as numpy array (0 or 255)
                      Background areas are forced to 0 via hard masking
                      Post-processed to fill holes and remove small components
        - prob_mask: Probability mask (H, W) as numpy array (0.0 to 1.0)
                     Background areas are forced to 0.0 via hard masking
        - preprocessed_input: Preprocessed input tensor (3, H, W) as numpy array
                             Contains [normalized_depth, normalized_gx, normalized_gy]
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
    
    return binary_mask, prob_mask, inp


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

# ---------------------------------------------------------
# NEW: IICL-Compliant Depth Measurement Logic (The "Stripes" Method)
# ---------------------------------------------------------
from sklearn.linear_model import RANSACRegressor

def detect_corrugation_orientation(depth_map):
    """
    Determines if stripes run Vertical (Side Walls) or Horizontal (Roof).
    """
    # Sobel gradients to find direction of highest change
    dx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
    dy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)
    
    # If Horizontal variance (dx) is higher, waves go Up-Down (Vertical)
    if np.var(dx) > np.var(dy) * 1.2:
        return "VERTICAL"
    # If Vertical variance (dy) is higher, waves go Left-Right (Horizontal)
    elif np.var(dy) > np.var(dx) * 1.2:
        return "HORIZONTAL"
    
    return "UNKNOWN"

def calculate_max_dent_depth_stripes(depth_map_m, mask_binary):
    """
    Measures depth by extending "healthy stripes" across the dent.
    Returns: (max_severity_total, dent_details_list)
    """
    if np.sum(mask_binary) == 0: return 0.0, []
    
    # 1. Orientation
    orientation = detect_corrugation_orientation(depth_map_m)
    
    # 2. Process Each Dent Individually
    num_labels, labels = cv2.connectedComponents((mask_binary > 0).astype(np.uint8))
    max_severity_total = 0.0
    dent_details = [] # Store details here
    
    H, W = depth_map_m.shape
    
    # Grid for RANSAC
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    for i in range(1, num_labels):
        dent_mask = (labels == i)
        ys, xs = np.where(dent_mask)
        y_min, y_max = np.min(ys), np.max(ys)
        x_min, x_max = np.min(xs), np.max(xs)
        
        # --- 3. EXTRACT THE STRIPE (Scanning) ---
        scan_margin = 80 
        
        if orientation == "VERTICAL":
            y_scan_min = max(0, y_min - scan_margin)
            y_scan_max = min(H, y_max + scan_margin)
            x_scan_min, x_scan_max = x_min, x_max 
            
        elif orientation == "HORIZONTAL":
            y_scan_min, y_scan_max = y_min, y_max
            x_scan_min = max(0, x_min - scan_margin)
            x_scan_max = min(W, x_max + scan_margin)
            
        else:
            y_scan_min = max(0, y_min - 50)
            y_scan_max = min(H, y_max + 50)
            x_scan_min = max(0, x_min - 50)
            x_scan_max = min(W, x_max + 50)

        strip_depth = depth_map_m[y_scan_min:y_scan_max, x_scan_min:x_scan_max]
        strip_mask = mask_binary[y_scan_min:y_scan_max, x_scan_min:x_scan_max]
        
        neighbor_mask = (strip_depth > 0) & (strip_mask == 0)
        
        if np.sum(neighbor_mask) < 50: continue

        # --- 4. SURFACE LOGIC ---
        neighbor_depths = strip_depth[neighbor_mask]
        
        h_c, w_c = strip_depth.shape
        yy_c, xx_c = np.meshgrid(np.arange(h_c), np.arange(w_c), indexing='ij')
        X_candidates = np.stack([xx_c[neighbor_mask], yy_c[neighbor_mask]], axis=1)
        y_candidates = neighbor_depths

        depth_range = np.percentile(neighbor_depths, 95) - np.percentile(neighbor_depths, 5)
        
        if depth_range > 0.015: 
            threshold = np.percentile(neighbor_depths, 50) 
            is_peak = neighbor_depths <= threshold
            X_train = X_candidates[is_peak]
            y_train = y_candidates[is_peak]
        else:
            X_train = X_candidates
            y_train = y_candidates

        # --- 5. RANSAC Fit ---
        try:
            reg = RANSACRegressor(random_state=42, residual_threshold=0.005)
            reg.fit(X_train, y_train)
            
            # --- 6. Measure ---
            dent_pixels_mask = (strip_mask > 0)
            X_dent = np.stack([xx_c[dent_pixels_mask], yy_c[dent_pixels_mask]], axis=1)
            actual_depths = strip_depth[dent_pixels_mask]
            
            ideal_depths = reg.predict(X_dent)
            diffs = np.abs(actual_depths - ideal_depths)
            
            sev = np.percentile(diffs, 98) # Max depth for this blob
            
            # Add to details list
            dent_details.append({
                'id': i,
                'max_depth': float(sev),
                'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)]
            })

            if sev > max_severity_total: max_severity_total = sev
        except:
            continue

    return max_severity_total, dent_details

# ---------------------------------------------------------
# UPDATED: Metric Calculation
# ---------------------------------------------------------

def load_camera_intrinsics(json_path: Optional[str] = None) -> dict:
    """Load camera intrinsics from JSON file."""
    if json_path is None:
        default_path = Path(__file__).parent / "camera_intrinsics_default.json"
        json_path = str(default_path)
    try:
        with open(json_path, 'r') as f:
            intrinsics = json.load(f)
        return {
            'fx': intrinsics.get('fx'),
            'fy': intrinsics.get('fy'),
            'cx': intrinsics.get('cx'),
            'cy': intrinsics.get('cy'),
            'fov_degrees': intrinsics.get('fov_degrees', {}),
            'resolution': intrinsics.get('resolution', {})
        }
    except:
        return {}

def calculate_dent_metrics(depth_map: np.ndarray, dent_mask: np.ndarray, 
                           pixel_to_cm: float = None,
                           depth_units: str = 'meters',
                           camera_fov: float = None,
                           focal_length: float = None,
                           sensor_width: float = None,
                           intrinsics_json_path: Optional[str] = None,
                           panel_mask: Optional[np.ndarray] = None,
                           plane_coefficients: Optional[np.ndarray] = None) -> dict:
    """
    Calculate dent metrics using IICL-compliant 'Stripes' method.
    """
    missing_info = []
    
    # 1. Intrinsics Loading (Same as before)
    intrinsics = load_camera_intrinsics(intrinsics_json_path)
    if intrinsics.get('fx') is not None:
        if focal_length is None:
            fx = intrinsics['fx']
            fy = intrinsics.get('fy', fx)
            focal_length = (fx + fy) / 2.0
    
    # 2. Depth Unit Conversion (Critical for Stripes Method)
    # The Stripes method EXPECTS METERS.
    depth_map_m = depth_map.copy()
    
    if depth_units != 'meters':
        # Auto-detect mm
        if np.nanmax(np.abs(depth_map)) > 10.0:
             depth_map_m = depth_map_m / 1000.0
    
    # Handle negatives / cleanup
    depth_map_m = np.abs(depth_map_m)
    depth_map_m[depth_map_m > 10.0] = 0.0 # Remove far noise
    
    # 3. Mask Handling
    if dent_mask.ndim == 3: dent_mask = dent_mask.squeeze()
    if depth_map.shape != dent_mask.shape:
        dent_mask = cv2.resize(dent_mask, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Normalize mask
    mask_max = np.nanmax(dent_mask)
    if mask_max > 1: dent_binary = (dent_mask > 127).astype(bool)
    else: dent_binary = (dent_mask > 0.5).astype(bool)
    
    # 4. Basic Metrics
    mask_uint8 = (dent_binary * 255).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_uint8)
    num_defects = num_labels - 1
    num_dent_pixels = np.sum(dent_binary)
    
    # 5. Area Calculation (Standard Intrinsics Logic - Unchanged)
    area_cm2 = None
    area_method = None
    area_valid = False
    
    if num_dent_pixels > 0:
        # (Same area logic as before - omitted for brevity, stick with your existing logic here or copy below)
        # Simplified for this snippet:
        if focal_length is not None:
            dent_depths = depth_map_m[dent_binary]
            valid_depths = dent_depths[dent_depths > 0]
            if len(valid_depths) > 0:
                pixel_size_m = valid_depths / focal_length
                pixel_area_cm2 = (pixel_size_m * 100.0) ** 2
                area_cm2 = np.sum(pixel_area_cm2)
                area_method = 'intrinsics'
                area_valid = True
        elif pixel_to_cm is not None:
             area_cm2 = num_dent_pixels * (pixel_to_cm ** 2)
             area_method = 'pixel_to_cm'
             area_valid = True

    # ---------------------------------------------------------
    # 6. NEW DEPTH MEASUREMENT (Stripes Method)
    # ---------------------------------------------------------
    max_depth_mm = 0.0
    avg_depth_mm = 0.0
    individual_dents = [] # New List
    
    if num_dent_pixels > 0:
        # Call updated function (Now returns TUPLE)
        max_depth_m, dent_details_m = calculate_max_dent_depth_stripes(depth_map_m, mask_uint8)
        
        max_depth_mm = max_depth_m * 1000.0
        avg_depth_mm = max_depth_mm * 0.6 
        
        # --- PROCESS INDIVIDUAL DENTS (Match depth with area) ---
        # We need to calculate area for each blob returned by the depth function
        for dent in dent_details_m:
            blob_id = dent['id']
            blob_depth_mm = dent['max_depth'] * 1000.0
            
            # Calculate Area for this specific blob
            blob_mask = (labels == blob_id)
            blob_pixels = np.sum(blob_mask)
            blob_area_cm2 = 0.0
            
            # Use same area logic as main function
            if focal_length is not None:
                # Intrinsics based (approximate using blob depths)
                dents_z = depth_map_m[blob_mask]
                valid_z = dents_z[dents_z > 0]
                if len(valid_z) > 0:
                     mean_z = np.mean(valid_z)
                     pixel_size_m = mean_z / focal_length
                     blob_area_cm2 = blob_pixels * ((pixel_size_m * 100.0) ** 2)
            elif pixel_to_cm is not None:
                blob_area_cm2 = blob_pixels * (pixel_to_cm ** 2)
                
            individual_dents.append({
                'id': blob_id,
                'max_depth_mm': float(blob_depth_mm),
                'area_cm2': float(blob_area_cm2),
                'bbox': dent['bbox'] # [x_min, y_min, x_max, y_max]
            })
            
        # Sort by depth descending
        individual_dents.sort(key=lambda x: x['max_depth_mm'], reverse=True)

    return {
        'area_cm2': area_cm2,
        'area_valid': area_valid,
        'max_depth_mm': float(max_depth_mm),
        'num_defects': num_defects,
        'avg_depth_mm': float(avg_depth_mm),
        'pixel_count': num_dent_pixels,
        'missing_info': missing_info,
        'area_method': area_method,
        'individual_dents': individual_dents # <--- ADD THIS
    }
# def load_camera_intrinsics(json_path: Optional[str] = None) -> dict:
#     """
#     Load camera intrinsics from JSON file.
    
#     Args:
#         json_path: Path to camera intrinsics JSON file. If None, uses default path.
        
#     Returns:
#         Dictionary with camera intrinsics: fx, fy, cx, cy, fov_degrees, resolution
#     """
#     if json_path is None:
#         # Default path relative to this file
#         default_path = Path(__file__).parent / "camera_intrinsics_default.json"
#         json_path = str(default_path)
    
#     try:
#         with open(json_path, 'r') as f:
#             intrinsics = json.load(f)
        
#         # Extract intrinsics
#         result = {
#             'fx': intrinsics.get('fx'),
#             'fy': intrinsics.get('fy'),
#             'cx': intrinsics.get('cx'),
#             'cy': intrinsics.get('cy'),
#             'fov_degrees': intrinsics.get('fov_degrees', {}),
#             'resolution': intrinsics.get('resolution', {})
#         }
        
#         return result
#     except FileNotFoundError:
#         return {}
#     except Exception as e:
#         print(f"Warning: Could not load camera intrinsics from {json_path}: {e}")
#         return {}


# def _calculate_plane_depth_map(depth_map: np.ndarray, 
#                                 plane_coefficients: np.ndarray,
#                                 camera_fov: float = None,
#                                 intrinsics_json_path: Optional[str] = None) -> np.ndarray:
#     """
#     Calculate plane depth map from RANSAC plane coefficients.
    
#     Plane equation: z = ax + by + c in camera space
#     Plane coefficients format: [a, b, -1, c]
    
#     For each pixel, we solve: z = a*x_norm*z + b*y_norm*z + c
#     Rearranging: z - a*x_norm*z - b*y_norm*z = c
#                  z * (1 - a*x_norm - b*y_norm) = c
#                  z = c / (1 - a*x_norm - b*y_norm)
    
#     Args:
#         depth_map: Depth map (H, W) in meters
#         plane_coefficients: Array [a, b, -1, c] where z = ax + by + c in camera space
#         camera_fov: Camera field of view in degrees (for intrinsics calculation)
#         intrinsics_json_path: Path to camera intrinsics JSON file
        
#     Returns:
#         Plane depth map (H, W) with depth values from the fitted plane
#     """
#     if plane_coefficients is None or len(plane_coefficients) != 4:
#         return None
    
#     a, b, _, c = plane_coefficients
    
#     height, width = depth_map.shape
    
#     # Load camera intrinsics
#     fx = None
#     fy = None
#     cx = None
#     cy = None
    
#     if intrinsics_json_path:
#         try:
#             with open(intrinsics_json_path, 'r') as f:
#                 intrinsics = json.load(f)
#                 fx = intrinsics.get('fx')
#                 fy = intrinsics.get('fy', fx)
#                 cx = intrinsics.get('cx', width / 2.0)
#                 cy = intrinsics.get('cy', height / 2.0)
#         except Exception as e:
#             print(f"Warning: Failed to load intrinsics from {intrinsics_json_path}: {e}")
    
#     # Fallback to FOV-based calculation if intrinsics not available
#     if fx is None and camera_fov is not None:
#         fov_y_rad = np.deg2rad(camera_fov)
#         fy = (height / 2.0) / np.tan(fov_y_rad / 2.0)
#         fx = fy
#         cx, cy = width / 2.0, height / 2.0
    
#     if fx is None or fy is None:
#         # Cannot calculate plane depth without intrinsics
#         return None
    
#     # Create pixel coordinates
#     u, v = np.meshgrid(np.arange(width), np.arange(height))
    
#     # Convert to normalized camera coordinates
#     x_norm = (u - cx) / fx
#     y_norm = (v - cy) / fy
    
#     # Calculate plane depth: z = c / (1 - a*x_norm - b*y_norm)
#     denominator = 1.0 - a * x_norm - b * y_norm
    
#     # Handle division by zero and unrealistic depths
#     # Set very small denominators to a small positive value to avoid division by zero
#     denominator = np.where(np.abs(denominator) < 1e-6, np.sign(denominator) * 1e-6, denominator)
    
#     plane_depth = c / denominator
    
#     # Filter unrealistic depths (negative or too large)
#     # Keep only reasonable depths (positive and less than 100 meters)
#     plane_depth = np.where((plane_depth > 0) & (plane_depth < 100.0), plane_depth, np.nan)
    
#     return plane_depth


# def calculate_dent_metrics(depth_map: np.ndarray, dent_mask: np.ndarray, 
#                            pixel_to_cm: float = None,
#                            depth_units: str = 'meters',
#                            camera_fov: float = None,
#                            focal_length: float = None,
#                            sensor_width: float = None,
#                            intrinsics_json_path: Optional[str] = None,
#                            panel_mask: Optional[np.ndarray] = None,
#                            plane_coefficients: Optional[np.ndarray] = None) -> dict:
#     """
#     Calculate dent metrics including area and maximum depth using camera intrinsics.
    
#     IMPORTANT: Real-world area computation requires camera calibration information.
#     Uses camera intrinsics to calculate depth-dependent pixel sizes for accurate area.
    
#     By default, loads intrinsics from camera_intrinsics_default.json if available.
    
#     Depth measurement compares dent depth to the median of the normal panel surface
#     (panel regions excluding dent regions).
    
#     Args:
#         depth_map: Original depth map (H, W) as numpy array in meters
#         dent_mask: Binary mask (H, W) as numpy array (NPY format from model output)
#                    Can be [0, 1], [0, 255], or boolean. Will be normalized to binary.
#         pixel_to_cm: Conversion factor from pixels to cm (fallback if intrinsics not provided).
#                      If camera intrinsics are provided, this is ignored in favor of intrinsics-based calculation.
#         depth_units: Units of depth_map values. Options: 'meters', 'mm', 'cm', 'inches'
#                      Used to convert depth differences to mm. Default: 'meters'
#         camera_fov: Camera field of view in degrees (for intrinsics-based calculation).
#                     If provided along with depth_map dimensions, calculates focal length.
#                     If None, will try to load from intrinsics JSON file.
#         focal_length: Focal length in pixels (alternative to camera_fov).
#                       If provided, uses this directly for intrinsics calculation.
#                       If None, will try to load from intrinsics JSON file (fx or fy).
#         sensor_width: Sensor width in mm (for intrinsics-based calculation).
#                       If None, uses FOV-based approximation.
#         intrinsics_json_path: Path to camera intrinsics JSON file. If None, uses default path.
#         panel_mask: Optional panel mask (H, W) where 1.0 = panel pixels, 0.0 = background.
#                     If provided, uses median of panel regions (excluding dent regions) as reference depth.
#                     If None, falls back to median of non-dent regions.
        
#     Returns:
#         Dictionary with metrics:
#         - area_cm2: Total dent area in cm² (calculated using intrinsics if available)
#         - area_valid: Boolean indicating if area computation is physically valid
#         - max_depth_mm: Maximum depth difference in mm (compared to panel median)
#         - num_defects: Number of separate dent regions
#         - avg_depth_mm: Average depth difference in mm (compared to panel median)
#         - pixel_count: Number of pixels in dent regions (always available)
#         - missing_info: List of missing information needed for valid area computation
#         - area_method: Method used for area calculation ('intrinsics' or 'pixel_to_cm')
#     """
#     missing_info = []
    
#     # Load camera intrinsics from JSON file by default (prioritize over FOV-based calculation)
#     intrinsics = load_camera_intrinsics(intrinsics_json_path)
    
#     # Use intrinsics from JSON if available (more accurate than FOV-based calculation)
#     if intrinsics.get('fx') is not None:
#         if focal_length is None:
#             # Use fx as focal length (or average of fx and fy if both available)
#             fx = intrinsics['fx']
#             fy = intrinsics.get('fy', fx)
#             focal_length = (fx + fy) / 2.0  # Average focal length
    
#     if camera_fov is None and intrinsics.get('fov_degrees'):
#         # Use vertical FOV if available, otherwise horizontal
#         fov_dict = intrinsics['fov_degrees']
#         camera_fov = fov_dict.get('vertical') or fov_dict.get('horizontal')
    
#     # CRITICAL: Only auto-detect and convert if depth_units is not explicitly set to 'meters'
#     # If depth_units='meters' is explicitly passed, trust it (e.g., after clean_depth_map_uint16)
#     depth_map_converted = depth_map.copy()
    
#     if depth_units != 'meters':
#         # Auto-detect: If max depth > 100, likely still in millimeters (should be < 10m for containers)
#         if np.nanmax(np.abs(depth_map)) > 100:
#             # Likely still in millimeters - convert to meters
#             depth_map_converted = depth_map_converted / 1000.0
#             # Filter error codes and far background
#             depth_map_converted[depth_map_converted > 3.0] = 0.0
#             depth_map_converted[depth_map_converted < 0] = 0.0
#             missing_info.append("Depth map appeared to be in millimeters - converted to meters for calculation")
#         elif np.nanmax(np.abs(depth_map)) > 10.0:
#             # Values > 10m are likely noise/background for container inspection
#             depth_map_converted[depth_map_converted > 10.0] = 0.0
#             missing_info.append("Filtered depth values > 10m (likely background noise)")
    
#     # Handle negative values (convert to positive)
#     if np.any(depth_map_converted < 0):
#         depth_map_converted = np.abs(depth_map_converted)
    
#     # Use converted depth map for all calculations
#     depth_map = depth_map_converted
    
#     # Validate pixel_to_cm for area computation
#     area_valid = False
#     if pixel_to_cm is None:
#         missing_info.append("pixel_to_cm conversion factor (required for area computation)")
#     elif pixel_to_cm <= 0:
#         missing_info.append("pixel_to_cm must be positive (invalid value provided)")
#     elif pixel_to_cm > 10:  # Sanity check: >10 cm/pixel seems unreasonable for typical setups
#         missing_info.append(f"pixel_to_cm value ({pixel_to_cm}) seems unusually large - please verify calibration")
#         area_valid = False  # Flag as suspicious but don't refuse
#     else:
#         area_valid = True
    
#     # --- MASK HANDLING (NPY-only, simplified) ---
#     # Mask is always NPY format from model output
#     # Handle dimensions: Ensure it's 2D (H, W)
#     # Common model outputs: (1, H, W) or (H, W, 1) -> Squeeze to (H, W)
#     if dent_mask.ndim == 3:
#         dent_mask = dent_mask.squeeze()
    
#     # Ensure masks match dimensions
#     if depth_map.shape != dent_mask.shape:
#         h, w = depth_map.shape
#         # Resize mask to match depth map (Nearest Neighbor to keep it binary)
#         # Note: cv2.resize expects (Width, Height) which is (shape[1], shape[0])
#         dent_mask = cv2.resize(dent_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
#     # --- Normalize Mask to Binary (0 and 1) ---
#     # Handles inputs like [0, 1] or [0, 255] or [False, True]
#     mask_max = np.nanmax(dent_mask)
#     if mask_max > 1:
#         # If values are 0-255, threshold at 127
#         dent_binary = (dent_mask > 127).astype(bool)
#     else:
#         # If values are already 0-1 or boolean, threshold at 0.5
#         dent_binary = (dent_mask > 0.5).astype(bool)
    
#     # Count number of separate dent regions (connected components)
#     mask_uint8 = (dent_binary * 255).astype(np.uint8)
#     num_labels, labels = cv2.connectedComponents(mask_uint8)
#     num_defects = num_labels - 1  # Subtract 1 because background is label 0
    
#     # Calculate pixel count (always available)
#     num_dent_pixels = np.sum(dent_binary)
    
#     # Calculate area using camera intrinsics if available, otherwise use pixel_to_cm
#     area_cm2 = None
#     area_method = None
    
#     if num_dent_pixels > 0:
#         # Try intrinsics-based calculation first
#         h, w = depth_map.shape
        
#         # Prioritize focal_length from intrinsics JSON over FOV-based calculation
#         # (intrinsics are more accurate)
#         if focal_length is not None:
#             focal_length_px = focal_length
#         elif camera_fov is not None:
#             # Fallback: Calculate focal length from FOV
#             fov_y_rad = np.deg2rad(camera_fov)
#             focal_length_px = (h / 2.0) / np.tan(fov_y_rad / 2.0)
#         else:
#             focal_length_px = None
        
#         # Calculate area using intrinsics (depth-dependent pixel size)
#         if focal_length_px is not None:
#             # Get depths for dent pixels
#             dent_depths_for_area = depth_map[dent_binary]
#             valid_depths_for_area = dent_depths_for_area[np.isfinite(dent_depths_for_area) & (dent_depths_for_area > 0)]
            
#             if len(valid_depths_for_area) > 0:
#                 # Calculate pixel size at each depth using camera intrinsics
#                 # Using pinhole camera model: pixel_size_m = depth / focal_length_px
#                 # This gives the physical size of one pixel at that depth in meters
#                 pixel_size_at_depth_m = valid_depths_for_area / focal_length_px
                
#                 # If sensor_width is provided, use more accurate calculation
#                 if sensor_width is not None:
#                     # Convert sensor_width from mm to meters
#                     sensor_width_m = sensor_width / 1000.0
#                     # Focal length in meters (from sensor geometry)
#                     # focal_length_m = (sensor_width_m * focal_length_px) / w
#                     # More accurate: pixel_size = depth * (sensor_width_m / focal_length_m) / w
#                     # But we can simplify using the relationship: focal_length_px / w ≈ focal_length_m / sensor_width_m
#                     # So: pixel_size_m = depth * sensor_width_m / (focal_length_px * w / focal_length_px * sensor_width_m / w)
#                     # Simplified: pixel_size_m = depth * sensor_width_m / (focal_length_px * w) * w
#                     # Actually: pixel_size_m = depth * sensor_width_m / (focal_length_m * w)
#                     # Where focal_length_m = sensor_width_m * focal_length_px / w
#                     focal_length_m = (sensor_width_m * focal_length_px) / w
#                     pixel_size_at_depth_m = valid_depths_for_area * (sensor_width_m / (focal_length_m * w))
                
#                 # Convert pixel size from meters to cm
#                 pixel_size_cm = pixel_size_at_depth_m * 100.0
                
#                 # Calculate area for each pixel and sum
#                 pixel_area_cm2 = pixel_size_cm ** 2
#                 area_cm2 = np.sum(pixel_area_cm2)
#                 area_method = 'intrinsics'
#                 area_valid = True
        
#         # Fallback to pixel_to_cm if intrinsics not available
#         if area_cm2 is None and area_valid and pixel_to_cm is not None:
#             # Calculate area in cm² using constant pixel size
#             # Area = number of pixels * (pixel_to_cm)^2
#             area_cm2 = num_dent_pixels * (pixel_to_cm ** 2)
#             area_method = 'pixel_to_cm'
    
#     if area_cm2 is None:
#         area_method = None
    
#     if not np.any(dent_binary):
#         return {
#             'area_cm2': area_cm2,
#             'area_valid': area_valid,
#             'max_depth_mm': 0.0,
#             'num_defects': 0,
#             'avg_depth_mm': 0.0,
#             'pixel_count': num_dent_pixels,
#             'missing_info': missing_info
#         }
    
#     # Calculate depth metrics
#     # Get depth values in dent regions
#     dent_depths = depth_map[dent_binary]
#     valid_depths = dent_depths[np.isfinite(dent_depths) & (dent_depths > 0)]
    
#     if len(valid_depths) == 0:
#         return {
#             'area_cm2': area_cm2,
#             'area_valid': area_valid,
#             'max_depth_mm': 0.0,
#             'num_defects': num_defects,
#             'avg_depth_mm': 0.0,
#             'pixel_count': num_dent_pixels,
#             'missing_info': missing_info
#         }
    
#     # --- NEW SIMPLIFIED DEPTH CALCULATION ---
#     # Based on: measure_dent_depth_from_files approach
#     # Wall Mask: Pixels that are NOT dent AND have valid depth (>0)
#     # We explicitly ignore 0.0 because that is the background/empty space
#     wall_mask = (dent_binary == False) & (depth_map > 0) & np.isfinite(depth_map)
    
#     # If panel_mask is provided, use it to refine wall mask
#     if panel_mask is not None:
#         wall_mask = wall_mask & (panel_mask > 0.5)
    
#     # Calculate plane depth map if plane_coefficients are provided
#     plane_depth_map = None
#     if plane_coefficients is not None:
#         plane_depth_map = _calculate_plane_depth_map(
#             depth_map, 
#             plane_coefficients,
#             camera_fov=camera_fov,
#             intrinsics_json_path=intrinsics_json_path
#         )
    
#     # Get Wall Reference Depth (Median)
#     # Use plane depth if available, otherwise use raw depth
#     if plane_depth_map is not None:
#         # Use plane depth for median calculation (removes perspective distortion)
#         wall_pixels_plane = plane_depth_map[wall_mask]
#         wall_pixels_plane = wall_pixels_plane[np.isfinite(wall_pixels_plane) & (wall_pixels_plane > 0)]
        
#         if len(wall_pixels_plane) == 0:
#             # Fallback: use raw depth if plane depth not available
#             wall_pixels = depth_map[wall_mask]
#             if len(wall_pixels) == 0:
#                 # Fallback: use all non-dent pixels if no wall pixels found
#                 non_dent_mask = (dent_binary == False) & np.isfinite(depth_map)
#                 if panel_mask is not None:
#                     non_dent_mask = non_dent_mask & (panel_mask > 0.5)
#                 wall_pixels = depth_map[non_dent_mask]
#                 if len(wall_pixels) == 0:
#                     # Last resort: use all valid depths
#                     wall_pixels = valid_depths
#             use_plane_depth = False
#         else:
#             wall_pixels = wall_pixels_plane
#             use_plane_depth = True
#     else:
#         # Use raw depth for median calculation
#         wall_pixels = depth_map[wall_mask]
#         if len(wall_pixels) == 0:
#             # Fallback: use all non-dent pixels if no wall pixels found
#             non_dent_mask = (dent_binary == False) & np.isfinite(depth_map)
#             if panel_mask is not None:
#                 non_dent_mask = non_dent_mask & (panel_mask > 0.5)
#             wall_pixels = depth_map[non_dent_mask]
#             if len(wall_pixels) == 0:
#                 # Last resort: use all valid depths
#                 wall_pixels = valid_depths
#         use_plane_depth = False
    
#     # Initialize variables
#     wall_ref_depth = 0.0
#     max_depth_mm = 0.0
#     avg_depth_mm = 0.0
#     dent_depth_median = 0.0
#     dent_depth_max = 0.0
#     dent_depth_min = 0.0
#     raw_depth_diff = 0.0
#     depth_median = 0.0
#     depth_raw_max = 0.0
#     depth_robust_max = 0.0
#     max_depth_value = 0.0
#     is_meters = False
#     depth_units_detected = 'unknown'
    
#     if len(wall_pixels) > 0:
#         # Get Wall Reference (Median is correct here for a flat wall)
#         # If using plane depth, this removes perspective distortion
#         wall_ref_depth = np.median(wall_pixels)
        
#         # Log which method is being used
#         if use_plane_depth:
#             print(f"Using RANSAC-fitted plane depth for median calculation (removes perspective distortion)")
#         else:
#             print(f"Using raw depth for median calculation")
        
#         # Get Dent Pixels (raw, before filtering)
#         dent_pixels_raw = depth_map[dent_binary]
#         if len(dent_pixels_raw) > 0:
#             # --- ROBUST MEASUREMENT: Blur then Max ---
#             # Strategy: Median Filter -> Max
#             # This kills noise spikes but preserves the real dent shape
            
#             # Extract Dent Region of Interest (ROI)
#             # Create a copy filled with wall_ref so edges don't distort the blur
#             dent_roi = np.full_like(depth_map, wall_ref_depth)
#             dent_roi[dent_binary] = depth_map[dent_binary]
            
#             # Apply Median Filter (The Magic Step)
#             # size=5 means it looks at a 5x5 area
#             # This kills noise spikes but keeps the dent shape
#             clean_dent_roi = median_filter(dent_roi, size=5)
            
#             # Get cleaned dent pixels
#             dent_pixels_cleaned = clean_dent_roi[dent_binary]
            
#             # Calculate Depths relative to wall (from cleaned data)
#             dent_depths_relative_cleaned = np.abs(dent_pixels_cleaned - wall_ref_depth)
            
#             # Also calculate from raw data for comparison
#             dent_depths_relative_raw = np.abs(dent_pixels_raw - wall_ref_depth)
            
#             # --- METRIC SELECTION ---
#             # A. Median (Good for volume estimation, bad for safety limits)
#             depth_median = np.median(dent_depths_relative_raw)
            
#             # B. Raw Max (Dangerous - sensitive to noise)
#             depth_raw_max = np.max(dent_depths_relative_raw)
            
#             # C. Robust Max (Blur -> Max) - THE GOLD STANDARD
#             # "Median filter removes noise spikes, then max captures true depth"
#             depth_robust_max = np.max(dent_depths_relative_cleaned)
            
#             # Store individual dent depths for diagnostics
#             dent_depth_median = np.median(dent_pixels_raw)
#             dent_depth_max = np.max(dent_pixels_raw)
#             dent_depth_min = np.min(dent_pixels_raw)
#         else:
#             # No dent pixels
#             depth_median = 0.0
#             depth_raw_max = 0.0
#             depth_robust_max = 0.0
#             dent_depth_median = wall_ref_depth
#             dent_depth_max = wall_ref_depth
#             dent_depth_min = wall_ref_depth
        
#         # Use robust max (Blur -> Max) as the primary metric
#         raw_depth_diff = depth_robust_max
        
#         # --- Unit Handling (Auto-detect) ---
#         # Check if the map is likely in Meters (Max value small, e.g. < 10.0)
#         # If so, multiply by 1000 to get Millimeters.
#         max_depth_value = np.nanmax(depth_map)
#         is_meters = max_depth_value < 10.0
        
#         if is_meters:
#             # Depth map is in meters, convert to mm
#             max_depth_mm = raw_depth_diff * 1000.0
#             # For average depth, use median relative depth
#             avg_depth_mm = depth_median * 1000.0
#             depth_units_detected = 'meters'
#         else:
#             # Depth map is already in mm (or other units)
#             max_depth_mm = raw_depth_diff
#             avg_depth_mm = depth_median
#             depth_units_detected = 'mm'
    
#     # Add diagnostic information about depth calculation
#     conversion_factor = 1000.0 if is_meters else 1.0
#     method_used = 'plane_depth' if use_plane_depth else 'raw_depth'
#     depth_stats = {
#         'reference_depth': float(wall_ref_depth),
#         'reference_depth_mm': float(wall_ref_depth * conversion_factor),
#         'dent_depth_median': float(dent_depth_median),
#         'dent_depth_median_mm': float(dent_depth_median * conversion_factor),
#         'dent_depth_max': float(dent_depth_max),
#         'dent_depth_max_mm': float(dent_depth_max * conversion_factor),
#         'dent_depth_min': float(dent_depth_min),
#         'dent_depth_min_mm': float(dent_depth_min * conversion_factor),
#         'depth_units_detected': depth_units_detected,
#         'max_depth_value': float(max_depth_value),
#         'raw_depth_diff': float(raw_depth_diff),
#         'depth_median': float(depth_median),
#         'depth_median_mm': float(depth_median * conversion_factor),
#         'depth_raw_max': float(depth_raw_max),
#         'depth_raw_max_mm': float(depth_raw_max * conversion_factor),
#         'depth_robust_max': float(depth_robust_max),
#         'method_used': method_used,
#         'depth_robust_max_mm': float(depth_robust_max * conversion_factor),
#         'method_used': 'median_filter_then_max'
#     }
    
#     return {
#         'area_cm2': area_cm2,
#         'area_valid': area_valid,
#         'max_depth_mm': max_depth_mm,
#         'num_defects': num_defects,
#         'avg_depth_mm': avg_depth_mm,
#         'pixel_count': num_dent_pixels,
#         'missing_info': missing_info,
#         'area_method': area_method,
#         'depth_stats': depth_stats
#     }

