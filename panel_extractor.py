"""
RANSAC Panel Extractor for Dent Detection Preprocessing
UPDATED VERSION: Uses Smart Cleaning (Bilateral + Despiking) instead of Gaussian Blur.
"""
import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor
from typing import Tuple, Optional
import json
from pathlib import Path

# Default configuration constants
DEFAULT_CLOSING_KERNEL_SIZE = 30
DEFAULT_CAMERA_FOV = 75.0
DEFAULT_RESIDUAL_THRESHOLD = 0.02 
DEFAULT_DOWNSAMPLE_FACTOR = 4

def clean_depth_map_uint16(depth: np.ndarray, 
                           apply_scale_factor: bool = True,
                           scale_factor: float = 0.5,
                           **kwargs) -> np.ndarray:
    """
    IMPROVED CLEANING PIPELINE:
    1. Crop & Convert
    2. Smart Hole Fill (Small noise only)
    3. Local Despike (Glare removal)
    4. Bilateral Smoothing (Edge preserving)
    """
    # 1. Convert to float32
    depth = depth.astype(np.float32)
    H, W = depth.shape
    
    # 2. Crop Center
    y1, y2 = int(0.1 * H), int(0.9 * H)
    x1, x2 = int(0.20 * W), int(0.80 * W)
    depth = depth[y1:y2, x1:x2]

    # 3. Unit Conversion & Scaling
    if np.nanmax(depth) > 100:
        depth /= 1000.0  # mm to m
    
    if apply_scale_factor:
        depth *= scale_factor

    # 4. Filter Obvious Noise
    depth[depth == 0] = np.nan
    depth[depth > 3.0] = np.nan # Background > 3m

    # 5. SMART HOLE FILLING (<50px only)
    # This replaces the old "blind inpainting"
    holes_mask = np.isnan(depth).astype(np.uint8) * 255
    # Find connected components of holes
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes_mask, connectivity=8)
    small_holes_mask = np.zeros_like(holes_mask)
    
    for i in range(1, num_labels):
        # If hole is small (e.g., < 50 pixels), mark it for filling
        if stats[i, cv2.CC_STAT_AREA] < 50:
            small_holes_mask[labels == i] = 255

    # Inpaint ONLY small holes
    if np.sum(small_holes_mask) > 0:
        valid_mask = ~np.isnan(depth)
        if np.any(valid_mask):
            mn, mx = np.nanmin(depth), np.nanmax(depth)
            # Normalize to 0-255 for OpenCV inpaint
            norm_depth = ((depth - mn) / (mx - mn + 1e-6) * 255)
            norm_depth[np.isnan(norm_depth)] = 0
            norm_depth = norm_depth.astype(np.uint8)
            
            # Navier-Stokes Inpainting
            inpainted = cv2.inpaint(norm_depth, small_holes_mask, 3, cv2.INPAINT_NS)
            
            # Restore float values
            filled_vals = (inpainted.astype(np.float32) / 255.0) * (mx - mn) + mn
            
            # Apply fill
            should_fill = (small_holes_mask == 255)
            depth[should_fill] = filled_vals[should_fill]

    # Leave large holes as 0 (Background)
    depth[np.isnan(depth)] = 0.0

    # 6. LOCAL SPIKE REMOVAL (Glare Removal)
    # Removes single-pixel spikes without blurring the whole image
    if np.nanmax(depth) > 0:
        # Convert to mm (uint16) for medianBlur
        depth_mm = (depth * 1000).astype(np.uint16)
        blurred_mm = cv2.medianBlur(depth_mm, 5)
        median_blurred = blurred_mm.astype(np.float32) / 1000.0
        
        diff = np.abs(depth - median_blurred)
        # If pixel deviates > 5% from neighbors, replace with median
        mask_spikes = (diff > 0.05 * median_blurred) & (depth > 0)
        depth[mask_spikes] = median_blurred[mask_spikes]

    # 7. BILATERAL SMOOTHING
    # This is the "Magic Step": Smooths flat wall, keeps dent edges sharp
    # d=5, sigmaColor=0.05 (5cm), sigmaSpace=5.0
    print("=" * 60)
    print("🔍 DEBUG: Applying BILATERAL SMOOTHING to depth map")
    print(f"   Parameters: d=5, sigmaColor=0.05 (5cm), sigmaSpace=5.0")
    print(f"   Input depth shape: {depth.shape}, dtype: {depth.dtype}")
    print(f"   Input depth range: [{np.nanmin(depth):.4f}, {np.nanmax(depth):.4f}] meters")
    
    depth_final = cv2.bilateralFilter(depth, d=5, sigmaColor=0.05, sigmaSpace=5.0)
    
    print(f"   Output depth range: [{np.nanmin(depth_final):.4f}, {np.nanmax(depth_final):.4f}] meters")
    print("✅ BILATERAL SMOOTHING applied successfully!")
    print("=" * 60)

    return depth_final

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

class RANSACPanelExtractor:
    def __init__(self, 
                 camera_fov: float = DEFAULT_CAMERA_FOV,
                 residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
                 max_trials: int = 1000,
                 adaptive_threshold: bool = False, # Changed default to False (Fixed is safer for measurement)
                 downsample_factor: int = DEFAULT_DOWNSAMPLE_FACTOR,
                 apply_morphological_closing: bool = True,
                 closing_kernel_size: int = DEFAULT_CLOSING_KERNEL_SIZE,
                 force_rectangular_mask: bool = True):
        
        self.camera_fov = camera_fov
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.adaptive_threshold = adaptive_threshold
        self.downsample_factor = downsample_factor
        self.apply_morphological_closing = apply_morphological_closing
        self.closing_kernel_size = closing_kernel_size
        self.force_rectangular_mask = force_rectangular_mask
        self._intrinsics = load_camera_intrinsics()
    
    def _depth_to_points_camera_space(self, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        height, width = depth.shape
        if self._intrinsics.get('fx') is not None:
            fx = self._intrinsics['fx']
            fy = self._intrinsics.get('fy', fx)
            cx = self._intrinsics.get('cx', width / 2.0)
            cy = self._intrinsics.get('cy', height / 2.0)
        else:
            fov_y_rad = np.deg2rad(self.camera_fov)
            fy = (height / 2.0) / np.tan(fov_y_rad / 2.0)
            fx = fy
            cx, cy = width / 2.0, height / 2.0
        
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy
        
        # Valid mask: Depth must be > 0 and finite
        valid_mask = (depth > 0) & np.isfinite(depth)
        
        # Project
        depth_valid = depth[valid_mask]
        x_cam = x_norm[valid_mask] * depth_valid
        y_cam = y_norm[valid_mask] * depth_valid
        z_cam = depth_valid
        
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        return points_cam, valid_mask
    
    def _fit_plane_ransac(self, points: np.ndarray, threshold: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if len(points) < 3:
            return np.zeros(len(points), dtype=bool), None
        
        X = points[:, :2]  # x, y
        y = points[:, 2]   # z
        
        ransac = RANSACRegressor(
            residual_threshold=threshold,
            max_trials=self.max_trials,
            min_samples=3,
            random_state=42
        )
        try:
            ransac.fit(X, y)
            inlier_mask = ransac.inlier_mask_
            
            if hasattr(ransac.estimator_, 'coef_'):
                coef = ransac.estimator_.coef_
                intercept = ransac.estimator_.intercept_
                # z = ax + by + c  ->  a, b, -1, c
                plane_coefficients = np.array([coef[0], coef[1], -1.0, intercept])
            else:
                plane_coefficients = None
            
            return inlier_mask, plane_coefficients
        except:
            return np.ones(len(points), dtype=bool), None

    def extract_panel_mask(self, depth: np.ndarray) -> Tuple[np.ndarray, dict, Optional[np.ndarray]]:
        """
        Extract panel mask using clean, bilateral-smoothed data.
        """
        print("=" * 60)
        print("🔍 DEBUG: extract_panel_mask() - Starting RANSAC extraction")
        print(f"   Input depth shape: {depth.shape}, dtype: {depth.dtype}")
        valid_before = np.sum((depth > 0) & np.isfinite(depth))
        total_pixels = depth.size
        print(f"   Valid pixels before processing: {valid_before}/{total_pixels} ({100*valid_before/total_pixels:.1f}%)")
        
        # 1. Standardize (Just in case, though clean_depth_map_uint16 usually handles this)
        if np.nanmax(depth) > 100: depth /= 1000.0
        depth[depth > 3.0] = 0
        depth[depth < 0] = 0
        
        # --- ✅ ADDED: TEMPORARY BLUR FOR SYNTHETIC DATA ---
        # This restores the "Old Code" behavior: We smooth the data used for 
        # RANSAC plane fitting, but we do NOT overwrite the original 'depth'.
        # This makes RANSAC robust on noisy synthetic data without blurring the final output.
        print(f"   Applying temporary Gaussian Blur for RANSAC plane fitting...")
        # (15, 15) kernel with sigma=2.0 is a robust smoother similar to the old behavior
        depth_for_fitting = cv2.GaussianBlur(depth, (15, 15), 2.0)
        # ----------------------------------------------------

        # 2. Downsample for speed
        # ⚠️ IMPORTANT: We now use 'depth_for_fitting' (smoothed) instead of 'depth' (raw)
        if self.downsample_factor > 1:
            h, w = depth_for_fitting.shape
            h_ds, w_ds = h // self.downsample_factor, w // self.downsample_factor
            depth_ds = cv2.resize(depth_for_fitting, (w_ds, h_ds), interpolation=cv2.INTER_AREA)
            print(f"   Downsampled from {depth.shape} to {depth_ds.shape} (factor={self.downsample_factor})")
        else:
            depth_ds = depth_for_fitting
            
        # 3. Convert to 3D Points
        points_3d, valid_mask_ds = self._depth_to_points_camera_space(depth_ds)
        
        print(f"   Converted to 3D points: {len(points_3d)} points")
        print(f"   Residual threshold: {self.residual_threshold}m ({self.residual_threshold*1000:.1f}mm)")
        
        if len(points_3d) < 100:
            print(f"   ❌ ERROR: Too few valid points ({len(points_3d)}) for RANSAC. Need at least 100.")
            print(f"   This may be due to:")
            print(f"      - Too narrow crop region (current: 0.25-0.75 width)")
            print(f"      - Too many invalid pixels in depth map")
            print(f"      - Depth values all zeros or out of valid range")
            print("=" * 60)
            return np.zeros_like(depth), {'error': f'Too few valid points: {len(points_3d)}'}, None

        # 4. RANSAC Fit
        # Note: We trust the Bilateral Filter from preprocessing, so we don't blur again here.
        print(f"   Running RANSAC with max_trials={self.max_trials}...")
        inlier_mask_points, plane_coefficients = self._fit_plane_ransac(points_3d, self.residual_threshold)
        
        num_inliers = np.sum(inlier_mask_points)
        num_outliers = len(inlier_mask_points) - num_inliers
        inlier_percentage = 100 * num_inliers / len(points_3d) if len(points_3d) > 0 else 0
        print(f"   ✅ RANSAC completed: {num_inliers} inliers, {num_outliers} outliers ({inlier_percentage:.1f}% inliers)")
        
        # 5. Reconstruct Mask
        panel_mask_ds = np.zeros_like(depth_ds, dtype=np.float32)
        valid_indices = np.where(valid_mask_ds)
        inlier_indices = np.where(inlier_mask_points)[0]
        
        if len(inlier_indices) > 0:
            panel_mask_ds[(valid_indices[0][inlier_indices], valid_indices[1][inlier_indices])] = 1.0
        else:
            print(f"   ⚠️  WARNING: No inliers found! RANSAC may have failed.")
            
        # 6. Upsample
        if self.downsample_factor > 1:
            h, w = depth.shape
            panel_mask = cv2.resize(panel_mask_ds, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            panel_mask = panel_mask_ds
            
        # 7. Post-Processing (Closing holes & Rectangular enforce)
        if self.apply_morphological_closing:
            mask_uint8 = (panel_mask * 255).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.closing_kernel_size, self.closing_kernel_size))
            mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
            panel_mask = (mask_closed / 255.0).astype(np.float32)

        # --- ✅ NEW: FILL HOLES WITH CONVEX HULL ---
        # This bridges the gap over deep dents, ensuring they are included in the mask
        panel_mask = self._fill_holes_convex(panel_mask)
        # -------------------------------------------

        if self.force_rectangular_mask:
            panel_mask = self._enforce_rectangular_mask(panel_mask)

        final_mask_pixels = np.sum(panel_mask > 0.5)
        final_mask_percentage = 100 * final_mask_pixels / panel_mask.size if panel_mask.size > 0 else 0
        print(f"   Final panel mask: {final_mask_pixels} pixels ({final_mask_percentage:.1f}% of image)")
        print("=" * 60)
        
        stats = {
            'residual_threshold_used': self.residual_threshold,
            'num_points': len(points_3d),
            'num_inliers': int(num_inliers),
            'plane_percentage': float(final_mask_percentage),
            'plane_pixel_count': int(final_mask_pixels),
        }
        return panel_mask, stats, plane_coefficients

    def _enforce_rectangular_mask(self, mask: np.ndarray) -> np.ndarray:
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: 
            return mask
        
        largest = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest) < 100: 
            return mask
        
        rect_mask = np.zeros_like(mask_uint8)
        
        # --- FIX IS HERE: Use np.int32 instead of np.int0 ---
        box = np.int32(cv2.boxPoints(cv2.minAreaRect(largest)))
        
        cv2.fillPoly(rect_mask, [box], 255)
        return (rect_mask / 255.0).astype(np.float32)

    def _fill_holes_convex(self, mask: np.ndarray) -> np.ndarray:
        """
        Fills holes and edge indentations using Convex Hull.
        This ensures deep dents on the edge of the container are included in the mask.
        """
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask
            
        # Find the largest contour (the panel)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate Convex Hull (Rubber band effect)
        hull = cv2.convexHull(largest_contour)
        
        # Draw the filled hull
        filled_mask = np.zeros_like(mask_uint8)
        cv2.drawContours(filled_mask, [hull], -1, 255, thickness=cv2.FILLED)
        
        return (filled_mask / 255.0).astype(np.float32)

    def extract_panel(self, depth: np.ndarray, fill_background: bool = False) -> Tuple[np.ndarray, np.ndarray, dict, Optional[np.ndarray]]:
        print("=" * 60)
        print("🔍 DEBUG: extract_panel() - Processing depth map")
        print(f"   Input depth shape: {depth.shape}, dtype: {depth.dtype}")
        print(f"   Input depth range: [{np.nanmin(depth):.4f}, {np.nanmax(depth):.4f}] meters")
        print(f"   fill_background: {fill_background}")
        print("   Note: If this depth came from clean_depth_map_uint16(), it already has BILATERAL SMOOTHING")
        print("=" * 60)
        
        panel_mask, stats, plane_coef = self.extract_panel_mask(depth)
        processed_depth = depth.copy()
        
        if fill_background:
            fill_val = 0.0
            valid_panel = depth[(panel_mask > 0.5) & (depth > 0)]
            if len(valid_panel) > 0: fill_val = np.median(valid_panel)
            processed_depth[panel_mask <= 0.5] = fill_val
            print(f"✅ Background filled with median panel depth: {fill_val:.4f} meters")
        else:
            processed_depth[panel_mask <= 0.5] = 0.0
            
        return processed_depth, panel_mask, stats, plane_coef

# Convenience function
def extract_panel(depth, **kwargs):
    # FORCE fill_background to False for the AI model pipeline
    if 'fill_background' in kwargs:
        print(f"⚠️ Overriding fill_background=True -> False (AI requires 0-background)")
        kwargs['fill_background'] = False
        
    extractor = RANSACPanelExtractor(**kwargs)
    return extractor.extract_panel(depth, fill_background=False) # Force False