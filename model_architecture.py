"""
Model Architecture for Dent Container Detection
Attention-UNet model definition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


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


def preprocess_depth(depth: np.ndarray) -> np.ndarray:
    """
    Preprocess depth map for model inference.
    Converts depth map to 3-channel input: [normalized_depth, normalized_gx, normalized_gy]
    
    Args:
        depth: Input depth map (H, W) as numpy array
        
    Returns:
        Preprocessed input tensor ready for model (3, H, W)
    """
    depth = depth.astype(np.float32)
    
    # 1. Normalize depth (0-1)
    valid = np.isfinite(depth) & (depth > 0)
    if np.any(valid):
        dmin, dmax = float(depth[valid].min()), float(depth[valid].max())
        denom = dmax - dmin
        if denom < 1e-8:
            denom = 1e-8
        depth_n = (depth - dmin) / denom
    else:
        depth_n = np.zeros_like(depth, dtype=np.float32)
    
    # 2. Compute gradients
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    
    # 3. Robust normalization for gradients
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
    
    # 4. Stack channels: (3, H, W)
    inp = np.stack([depth_n, gx_n, gy_n], axis=0).astype(np.float32)
    
    return inp


def predict_mask(model: nn.Module, depth: np.ndarray, device: str = 'cpu', threshold: float = 0.5) -> np.ndarray:
    """
    Run inference on a depth map to generate binary segmentation mask.
    
    Args:
        model: Trained AttentionUNet model
        depth: Input depth map (H, W) as numpy array
        device: Device to run inference on ('cpu' or 'cuda')
        threshold: Threshold for binary mask (default 0.5)
        
    Returns:
        Binary segmentation mask (H, W) as numpy array (0 or 255)
    """
    model.eval()
    
    # Preprocess
    inp = preprocess_depth(depth)
    
    # Convert to tensor and add batch dimension: (1, 3, H, W)
    inp_tensor = torch.from_numpy(inp).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(inp_tensor)
        prob_mask = torch.sigmoid(output).cpu().numpy()[0, 0]  # (H, W)
    
    # Convert to binary mask
    binary_mask = (prob_mask > threshold).astype(np.uint8) * 255
    
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


def calculate_dent_metrics(depth_map: np.ndarray, dent_mask: np.ndarray, 
                           pixel_to_cm: float = None,
                           depth_units: str = 'meters') -> dict:
    """
    Calculate dent metrics including area and maximum depth.
    
    IMPORTANT: Real-world area computation requires camera calibration information.
    Without proper calibration, area values are NOT physically meaningful.
    
    Args:
        depth_map: Original depth map (H, W) as numpy array
        dent_mask: Binary mask (H, W) where WHITE (255) = dented areas
        pixel_to_cm: Conversion factor from pixels to cm (REQUIRED for area computation).
                     If None, area computation will be refused.
                     
                     To obtain this value, you need ONE of the following:
                     1. Camera calibration: focal_length (mm) / sensor_width (mm) * distance_to_object (mm) / image_width (pixels)
                     2. Reference object: Measure a known object in the image and calculate pixels_per_cm
                     3. Manual calibration: Place a ruler or known-size object and measure pixels per cm
                     
                     WARNING: Using an uncalibrated value will produce incorrect area measurements.
        depth_units: Units of depth_map values. Options: 'meters', 'mm', 'cm', 'inches'
                     Used to convert depth differences to mm. Default: 'meters'
        
    Returns:
        Dictionary with metrics:
        - area_cm2: Total dent area in cm² (None if pixel_to_cm not provided)
        - area_valid: Boolean indicating if area computation is physically valid
        - max_depth_mm: Maximum depth difference in mm
        - num_defects: Number of separate dent regions
        - avg_depth_mm: Average depth difference in mm
        - pixel_count: Number of pixels in dent regions (always available)
        - missing_info: List of missing information needed for valid area computation
    """
    missing_info = []
    
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
    
    # REFUSE area computation if not physically valid
    if not area_valid:
        area_cm2 = None
    else:
        # Calculate area in cm²
        # Area = number of pixels * (pixel_to_cm)^2
        area_cm2 = num_dent_pixels * (pixel_to_cm ** 2)
    
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
    
    # Get reference depth (median of non-dent regions)
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
        'missing_info': missing_info
    }

