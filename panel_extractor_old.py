"""
RANSAC Panel Extractor for Dent Detection Preprocessing

This module extracts the container panel from raw depth maps using RANSAC plane fitting.
It isolates the panel by identifying inlier points that belong to the plane and masking out
background noise (ground, trees, etc.).

The extracted panel can then be fed to the dent detection model, which expects clean
panel-only input similar to the training data.
"""
import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional
import json
from pathlib import Path

# Default configuration constants (single source of truth)
DEFAULT_CLOSING_KERNEL_SIZE = 30
DEFAULT_CAMERA_FOV = 75.0
DEFAULT_RESIDUAL_THRESHOLD = 0.02  # Changed from 0.03 to 0.02 to match DentContainer2
DEFAULT_DOWNSAMPLE_FACTOR = 4


def clean_depth_map_uint16(depth: np.ndarray, 
                           apply_scale_factor: bool,
                           scale_factor: float = 0.5,
                           lower_percentile: float = 2.0,
                           upper_percentile: float = 98.0) -> np.ndarray:
    """
    Clean raw uint16 depth map data before RANSAC processing.
    
    This function performs raw data cleaning steps including:
    1. Unit Conversion (mm to m)
    2. Filter Obvious Noise (0 and >1.5m)
    3. Percentile-based Noise Filtering (removes outliers)
    4. Inpainting (Hole Filling)
    5. Smart Wall Isolation (Background Removal)
    6. Strong Gaussian Blur
    
    This should be called BEFORE RANSAC when processing uint16 format depth maps.
    
    Args:
        depth: Input depth map (H, W) as numpy array (can be uint16 or float32)
        apply_scale_factor: If True, apply additional scale factor
        scale_factor: Scale factor to apply if apply_scale_factor=True (default: 0.5)
        lower_percentile: Lower percentile threshold for noise removal (default: 1.0)
                         Values below this percentile will be treated as noise
        upper_percentile: Upper percentile threshold for noise removal (default: 99.0)
                         Values above this percentile will be treated as noise
        
    Returns:
        Cleaned depth map (H, W) as float32 numpy array
    """
    # Convert to float32 if needed
    depth = depth.astype(np.float32)
    H, W = depth.shape
    
    # 1. Crop to center region (remove edge artifacts)
    left_crop = int(0.25 * W)
    right_crop = int(0.85 * W)
    top_crop = int(0.1 * H)
    bottom_crop = int(0.9 * H)
    depth = depth[top_crop:bottom_crop, left_crop:right_crop].copy()
    
    # 2. Unit Conversion: Detect and convert millimeters to meters
    if np.nanmax(depth) > 100:
        # Likely in millimeters, convert to meters
        depth = depth / 1000.0
        # Apply optional scale factor if requested
        if apply_scale_factor:
            depth = depth * scale_factor 
    
    # 3. Filter Obvious Noise (Pre-Inpainting)
    # Treat 0 (sensor error) and >1.5m (background) as Invalid (NaN)
    depth[depth == 0] = np.nan
    depth[depth > 1.5] = np.nan
    
    # 4. Percentile-based Noise Filtering (remove outliers)
    # Filter out extreme values that are likely noise/outliers
    # This is done AFTER filtering obvious noise so percentiles are calculated on cleaner data
    valid_pixels = depth[(depth > 0) & np.isfinite(depth)]
    if len(valid_pixels) > 0:
        # Calculate percentile thresholds
        lower_threshold = np.nanpercentile(valid_pixels, lower_percentile)
        upper_threshold = np.nanpercentile(valid_pixels, upper_percentile)
        
        # Remove values outside percentile range (treat as noise)
        noise_mask = (depth < lower_threshold) | (depth > upper_threshold)
        depth[noise_mask] = np.nan
    
    # 5. Inpaint Small Holes (Sensor Noise)
    mask_invalid = np.isnan(depth).astype(np.uint8)
    if np.any(mask_invalid) and np.any(~np.isnan(depth)):
        # Only inpaint if there are both invalid and valid pixels
        depth_valid = depth.copy()
        depth_valid[mask_invalid == 1] = 0
        
        valid_pixels = depth[~np.isnan(depth)]
        if len(valid_pixels) > 0:
            d_min, d_max = np.nanmin(depth), np.nanmax(depth)
            if d_max - d_min > 1e-6:  # Avoid division by zero
                # Normalize to [0, 255] for inpainting
                norm_depth = ((depth_valid - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
                # Inpaint using Navier-Stokes method
                inpainted_norm = cv2.inpaint(norm_depth, mask_invalid, 3, cv2.INPAINT_NS)
                # Convert back to original scale
                depth_filled = (inpainted_norm.astype(np.float32) / 255.0) * (d_max - d_min) + d_min
            else:
                depth_filled = depth_valid
        else:
            depth_filled = depth_valid
    else:
        depth_filled = depth.copy()
        # Replace NaN with 0 for consistency
        depth_filled[np.isnan(depth_filled)] = 0.0
    
    # 6. Smart Wall Isolation (Background Removal)
    # Use median depth as wall center, with wide tolerance for tilted corners
    valid_for_median = depth_filled[(depth_filled > 0) & np.isfinite(depth_filled)]
    if len(valid_for_median) > 0:
        wall_depth = np.median(valid_for_median)
        
        # Widen window to 0.25m to catch tilted corners
        valid_range_mask = (depth_filled > (wall_depth - 0.25)) & (depth_filled < (wall_depth + 0.25))
        depth_filled[~valid_range_mask] = 0.0
    else:
        # No valid pixels found, keep as is
        depth_filled[depth_filled <= 0] = 0.0
    
    # 7. Final Denoise with Gaussian Blur
    # Only apply blur if there are valid pixels
    if np.any(depth_filled > 0):
        depth_final = cv2.GaussianBlur(depth_filled, (7, 7), 0)
    else:
        depth_final = depth_filled.copy()
    
    # 8. Remove "Near Zero" Blur Artifacts
    # Any pixel created by the blur that is essentially 0 should be 0
    depth_final[depth_final < 0.1] = 0.0
    
    return depth_final.astype(np.float32)


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


class RANSACPanelExtractor:
    """
    Extract container panel from raw depth maps using RANSAC plane fitting.
    
    This class handles:
    - Converting depth maps to 3D point clouds
    - Fitting a plane using RANSAC
    - Identifying panel pixels (inliers) vs background (outliers)
    - Returning masks or processed depth maps
    """
    
    def __init__(self, 
                 camera_fov: float = DEFAULT_CAMERA_FOV,
                 residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
                 max_trials: int = 1000,
                 adaptive_threshold: bool = True,
                 downsample_factor: int = DEFAULT_DOWNSAMPLE_FACTOR,
                 apply_morphological_closing: bool = True,
                 closing_kernel_size: int = DEFAULT_CLOSING_KERNEL_SIZE,
                 force_rectangular_mask: bool = True):
        """
        Initialize RANSAC Panel Extractor.
        
        Args:
            camera_fov: Camera field of view in degrees (default: 75.0)
            residual_threshold: Maximum distance from plane to be considered inlier (meters)
                                Only used if adaptive_threshold=False
            max_trials: Maximum RANSAC iterations (default: 1000)
            adaptive_threshold: If True, automatically tune threshold based on corrugation depth
            downsample_factor: Downsample factor for RANSAC fitting (default: 4)
                              RANSAC runs on (H/downsample_factor) x (W/downsample_factor) points
                              Then mask is upsampled back to full resolution
            apply_morphological_closing: If True, apply morphological closing to fill small holes (dents)
            closing_kernel_size: Size of the morphological closing kernel in pixels (default: 30)
                                Larger values fill larger holes, but may merge separate dents
            force_rectangular_mask: If True, enforce a rectangular mask by finding the largest contour
                                    and drawing its rotated bounding box. This ensures deep dents
                                    inside the panel boundary are included.
        """
        self.camera_fov = camera_fov
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.adaptive_threshold = adaptive_threshold
        self.downsample_factor = downsample_factor
        self.apply_morphological_closing = apply_morphological_closing
        self.closing_kernel_size = closing_kernel_size
        self.force_rectangular_mask = force_rectangular_mask
        
        # Load camera intrinsics from JSON file by default
        self._intrinsics = load_camera_intrinsics()
    
    def _depth_to_points_camera_space(self, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert depth map to 3D points in camera space.
        
        Uses camera intrinsics from JSON file if available, otherwise falls back to FOV-based calculation.
        
        Args:
            depth: Depth map (H, W) in meters
            
        Returns:
            Tuple of (points_3d, valid_mask)
            - points_3d: (N, 3) array of 3D points [x, y, z]
            - valid_mask: (H, W) boolean mask of valid depth pixels
        """
        height, width = depth.shape
        
        # Use intrinsics from JSON file if available
        if self._intrinsics.get('fx') is not None:
            fx = self._intrinsics['fx']
            fy = self._intrinsics.get('fy', fx)
            cx = self._intrinsics.get('cx', width / 2.0)
            cy = self._intrinsics.get('cy', height / 2.0)
        else:
            # Fallback: Calculate camera intrinsics from FOV
            fov_y_rad = np.deg2rad(self.camera_fov)
            # Calculate focal length from vertical FOV (more accurate for depth maps)
            fy = (height / 2.0) / np.tan(fov_y_rad / 2.0)
            fx = fy  # Assume square pixels for FOV-based fallback
            cx, cy = width / 2.0, height / 2.0
        
        # Create pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to normalized camera coordinates using fx and fy separately
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy
        
        # Get valid depth pixels (allow negative values, but filter out zero and NaN/Inf)
        # Negative depths are valid - they just indicate depth in opposite direction
        valid_mask = (depth != 0) & np.isfinite(depth)
        
        # Take absolute value for processing (mathematically equivalent for plane fitting)
        # A plane at Z=-0.4 is identical to Z=+0.4 for shape analysis
        depth_abs = np.abs(depth[valid_mask])
        
        # Back-project to 3D points in camera space (using absolute values)
        x_cam = x_norm[valid_mask] * depth_abs
        y_cam = y_norm[valid_mask] * depth_abs
        z_cam = depth_abs
        
        # Stack into Nx3 array
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        
        return points_cam, valid_mask
    
    def _calculate_adaptive_sigma(self, depth: np.ndarray,
                                 min_sigma: float = 4.0,
                                 max_sigma: float = 18.0,
                                 base_sigma: float = 8.0) -> float:
        """
        Calculate adaptive sigma for Gaussian smoothing based on depth variance.
        Matches DentContainer2 implementation exactly.
        
        Higher variance indicates more corrugation/variation, requiring more smoothing.
        Lower variance indicates flatter surfaces, requiring less smoothing.
        
        Args:
            depth: Depth map (H, W) in meters
            min_sigma: Minimum sigma value (default: 4.0)
            max_sigma: Maximum sigma value (default: 18.0)
            base_sigma: Base sigma value for normalization (default: 8.0)
            
        Returns:
            Adaptive sigma value for Gaussian smoothing
        """
        # Get valid depth pixels (allow negative values, but filter out zero and NaN/Inf)
        valid_mask = (depth != 0) & np.isfinite(depth)
        
        if not np.any(valid_mask):
            return base_sigma
        
        valid_depths = depth[valid_mask]
        
        # Calculate depth variance
        depth_variance = np.var(valid_depths)
        depth_std = np.std(valid_depths)
        
        # Calculate mean depth for normalization
        depth_mean = np.mean(valid_depths)
        
        # Normalize variance by mean depth to get relative variation
        # This accounts for different camera distances
        relative_variance = depth_variance / (depth_mean ** 2 + 1e-6)
        relative_std = depth_std / (depth_mean + 1e-6)
        
        # Scale sigma based on relative standard deviation
        # Higher relative std = more corrugation = higher sigma needed
        # Use a scaling factor: sigma = base_sigma * (1 + relative_std * scale_factor)
        scale_factor = 2.0  # Adjust this to control sensitivity
        adaptive_sigma = base_sigma * (1.0 + relative_std * scale_factor)
        
        # Clamp to min/max bounds
        adaptive_sigma = np.clip(adaptive_sigma, min_sigma, max_sigma)
        
        return float(adaptive_sigma)
    
    def _apply_gaussian_smoothing(self, depth: np.ndarray,
                                 sigma: Optional[float] = None,
                                 adaptive: bool = True) -> Tuple[np.ndarray, float]:
        """
        Apply Gaussian smoothing to depth map to flatten corrugation patterns.
        Matches DentContainer2 implementation exactly.
        
        This step is CRITICAL - it must be applied BEFORE RANSAC to match training data preprocessing.
        
        IMPORTANT: Invalid pixels (zero, NaN, Inf) are masked out before smoothing to prevent
        corruption of valid data.
        
        Args:
            depth: Depth map (H, W) in meters
            sigma: Standard deviation for Gaussian kernel (if None and adaptive=True, will be calculated)
            adaptive: If True, automatically tune sigma based on depth variance
            
        Returns:
            Tuple of (smoothed_depth_map, sigma_used)
        """
        # Filter invalid pixels FIRST (allow negative values, but filter out zero and NaN/Inf)
        valid_mask = (depth != 0) & np.isfinite(depth)
        
        if not np.any(valid_mask):
            return depth.copy(), sigma if sigma is not None else 8.0
        
        # Calculate adaptive sigma if requested (using only valid pixels)
        if adaptive and sigma is None:
            # Create a temporary depth with invalid pixels masked
            depth_for_sigma = depth.copy()
            depth_for_sigma[~valid_mask] = np.nan  # Mask invalid pixels
            sigma = self._calculate_adaptive_sigma(depth_for_sigma)
        elif sigma is None:
            sigma = 8.0  # Default fallback
        
        # Create a masked depth map for smoothing (invalid pixels set to NaN)
        # This prevents gaussian_filter from spreading invalid values
        depth_masked = depth.copy()
        depth_masked[~valid_mask] = np.nan
        
        # Apply Gaussian filter (NaN values are ignored by gaussian_filter)
        smoothed = gaussian_filter(depth_masked, sigma=sigma, mode='constant', cval=np.nan)
        
        # Create output: smoothed values where valid, original invalid pixels preserved
        smoothed_depth = depth.copy()
        smoothed_depth[valid_mask] = smoothed[valid_mask]
        # Invalid pixels remain as original (0, NaN, or Inf)
        
        return smoothed_depth, sigma
    
    def _standardize_depth_units(self, depth: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Helper: Standardizes depth map to Positive Meters.
        Handles: Type A (Negative → convert to positive), Type B (Positive → no change).
        
        Note: Type C (Millimeters/Uint16) conversion is handled by clean_depth_map_uint16()
        and should be called BEFORE this method to avoid double conversion.
        
        This method is idempotent - safe to call multiple times on the same data.
        
        Args:
            depth: Input depth map (H, W) as numpy array (should already be in meters)
            
        Returns:
            Tuple of (standardized_depth, stats_dict)
            - standardized_depth: (H, W) depth map in positive meters
            - stats_dict: Dictionary with conversion statistics
        """
        depth = depth.astype(np.float32)
        stats = {'converted_mm': False, 'converted_neg': False}

        # Handle Negative Z (Type A) - convert to positive
        if np.any(depth < 0):
            stats['converted_neg'] = True
            depth = np.abs(depth)
        
        # Final clipping: Remove anything > 3m (background/outliers)
        # This ensures consistent behavior regardless of input format
        depth[depth > 3.0] = 0.0

        return depth, stats
    
    def _calculate_adaptive_threshold(self, depth: np.ndarray) -> float:
        """
        Calculate adaptive residual threshold based on corrugation depth.
        Matches DentContainer2 implementation exactly.
        
        Args:
            depth: Depth map (H, W) - should already have negative values converted to positive
            
        Returns:
            Adaptive threshold value
        """
        # Allow negative values (though they should be converted to positive by now)
        valid_mask = (depth != 0) & np.isfinite(depth)
        
        if not np.any(valid_mask):
            return self.residual_threshold
        
        valid_depths = depth[valid_mask]
        depth_min = np.min(valid_depths)
        depth_max = np.max(valid_depths)
        depth_range = depth_max - depth_min
        depth_mean = np.mean(valid_depths)
        
        # Normalize range by mean depth
        relative_range = depth_range / (depth_mean + 1e-6)
        
        # Scale threshold based on relative corrugation depth
        scale_factor = 1.5
        adaptive_threshold = self.residual_threshold * (1.0 + relative_range * scale_factor)
        
        # Clamp to reasonable bounds
        adaptive_threshold = np.clip(adaptive_threshold, 0.015, 0.05)
        
        return float(adaptive_threshold)
    
    def _fit_plane_ransac(self, points: np.ndarray, threshold: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit a plane to 3D points using RANSAC.
        
        Args:
            points: (N, 3) array of 3D points
            threshold: Residual threshold for inlier detection
            
        Returns:
            Tuple of (inlier_mask, plane_coefficients)
            - inlier_mask: Boolean mask of inlier points (True = panel, False = background)
            - plane_coefficients: Array [a, b, -1, c] where z = ax + by + c in camera space, or None if fitting failed
        """
        if len(points) < 3:
            return np.zeros(len(points), dtype=bool), None
        
        # Fit z as a function of x and y: z = ax + by + c
        X = points[:, :2]  # x, y coordinates
        y = points[:, 2]    # z coordinates (depth)
        
        ransac = RANSACRegressor(
            residual_threshold=threshold,
            max_trials=self.max_trials,
            min_samples=3,
            random_state=42
        )
        
        try:
            ransac.fit(X, y)
            inlier_mask = ransac.inlier_mask_
            
            # Extract plane coefficients: z = ax + by + c
            # ransac.estimator_.coef_ gives [a, b] and ransac.estimator_.intercept_ gives c
            if hasattr(ransac.estimator_, 'coef_') and hasattr(ransac.estimator_, 'intercept_'):
                coef = ransac.estimator_.coef_  # [a, b]
                intercept = ransac.estimator_.intercept_  # c
                # Return as [a, b, -1, c] format for consistency with plane equation z = ax + by + c
                plane_coefficients = np.array([coef[0], coef[1], -1.0, intercept])
            else:
                plane_coefficients = None
            
            return inlier_mask, plane_coefficients
        except Exception as e:
            # If RANSAC fails, treat all points as inliers
            print(f"Warning: RANSAC fitting failed: {e}. Using all points as inliers.")
            return np.ones(len(points), dtype=bool), None
    
    def extract_panel_mask(self, depth: np.ndarray) -> Tuple[np.ndarray, dict, Optional[np.ndarray]]:
        """
        Extract panel mask from depth map using RANSAC plane fitting.
        
        CRITICAL: This function now matches DentContainer2 preprocessing exactly by:
        1. Applying Gaussian smoothing BEFORE RANSAC (to flatten corrugation patterns)
        2. Using smoothed depth for RANSAC plane fitting
        
        Args:
            depth: Input depth map (H, W) as numpy array (float32)
            
        Returns:
            Tuple of (panel_mask, stats_dict, plane_coefficients)
            - panel_mask: (H, W) binary mask where 1.0 = panel pixels, 0.0 = background
            - stats_dict: Dictionary with extraction statistics
            - plane_coefficients: Array [a, b, -1, c] where z = ax + by + c in camera space, or None if not available
        """
        # Step 1: Standardize depth units (mm -> m, negative -> positive)
        depth, unit_stats = self._standardize_depth_units(depth)
        
        # Step 2: Filter invalid pixels FIRST to prevent corruption
        # Allow negative values (now converted to positive), but filter out zero and NaN/Inf
        valid_mask_original = (depth != 0) & np.isfinite(depth)
        
        if not np.any(valid_mask_original):
            # No valid pixels at all
            panel_mask = np.zeros_like(depth, dtype=np.float32)
            stats = {
                'plane_percentage': 0.0,
                'plane_pixel_count': 0,
                'total_pixels': int(panel_mask.size),
                'residual_threshold_used': self.residual_threshold,
                'num_points': 0,
                'gaussian_smoothing_applied': True,
                'gaussian_sigma_used': 8.0,
                'morphological_closing_applied': self.apply_morphological_closing,
                'closing_kernel_size': self.closing_kernel_size if self.apply_morphological_closing else None,
                'rectangular_mask_enforced': self.force_rectangular_mask,
                'valid_pixels_before_processing': 0,
                'negative_values_converted': unit_stats['converted_neg'],
                'millimeters_converted': unit_stats['converted_mm']
            }
            return panel_mask, stats, None
        
        # Step 3: CRITICAL STEP - Apply Gaussian smoothing BEFORE RANSAC
        # This matches DentContainer2 preprocessing exactly
        # Invalid pixels are already filtered, so smoothing won't corrupt valid data
        depth_smoothed, sigma_used = self._apply_gaussian_smoothing(depth, adaptive=True)
        
        # Step 4: Calculate adaptive threshold if needed (use smoothed depth for threshold calculation)
        threshold = self.residual_threshold
        if self.adaptive_threshold:
            threshold = self._calculate_adaptive_threshold(depth_smoothed)
        
        # Step 5: Downsample for faster RANSAC fitting (use smoothed depth)
        # Filter invalid pixels before downsampling to prevent interpolation artifacts
        if self.downsample_factor > 1:
            h, w = depth_smoothed.shape
            h_ds = h // self.downsample_factor
            w_ds = w // self.downsample_factor
            
            # Create masked depth for downsampling (invalid pixels set to NaN)
            depth_for_resize = depth_smoothed.copy()
            valid_mask_smoothed = (depth_smoothed > 0) & np.isfinite(depth_smoothed)
            depth_for_resize[~valid_mask_smoothed] = np.nan
            
            # Resize (NaN values are handled, but may still cause issues)
            # Better approach: resize only valid regions
            depth_ds = cv2.resize(depth_smoothed, (w_ds, h_ds), interpolation=cv2.INTER_AREA)
            
            # Filter invalid pixels after downsampling (zero and NaN/Inf)
            depth_ds[(depth_ds == 0) | ~np.isfinite(depth_ds)] = 0.0
        else:
            depth_ds = depth_smoothed
        
        # Step 6: Convert to 3D points (this filters invalid pixels again)
        points_3d, valid_mask_ds = self._depth_to_points_camera_space(depth_ds)
        
        if len(points_3d) == 0:
            # No valid points - return empty mask with complete stats dictionary
            panel_mask = np.zeros_like(depth, dtype=np.float32)
            stats = {
                'plane_percentage': 0.0,
                'plane_pixel_count': 0,
                'total_pixels': int(panel_mask.size),
                'residual_threshold_used': threshold,
                'num_points': 0,
                'gaussian_smoothing_applied': True,
                'gaussian_sigma_used': sigma_used,
                'morphological_closing_applied': self.apply_morphological_closing,
                'closing_kernel_size': self.closing_kernel_size if self.apply_morphological_closing else None,
                'rectangular_mask_enforced': self.force_rectangular_mask
            }
            return panel_mask, stats, None
        
        # Fit plane using RANSAC
        inlier_mask_points, plane_coefficients = self._fit_plane_ransac(points_3d, threshold)
        
        # Create mask at downsampled resolution
        panel_mask_ds = np.zeros_like(depth_ds, dtype=np.float32)
        valid_indices_ds = np.where(valid_mask_ds)
        inlier_indices = np.where(inlier_mask_points)[0]
        
        if len(inlier_indices) > 0:
            inlier_pixels_ds = (valid_indices_ds[0][inlier_indices], 
                               valid_indices_ds[1][inlier_indices])
            panel_mask_ds[inlier_pixels_ds] = 1.0
        
        # Upsample mask back to full resolution
        if self.downsample_factor > 1:
            h, w = depth.shape
            panel_mask = cv2.resize(panel_mask_ds, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            panel_mask = panel_mask_ds
        
        # Apply morphological closing to fill small holes (dents that RANSAC rejected)
        if self.apply_morphological_closing:
            # Convert to uint8 for morphological operations
            mask_uint8 = (panel_mask * 255).astype(np.uint8)
            
            # Create circular kernel for morphological closing
            # This fills small black holes (dents) surrounded by white (panel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (self.closing_kernel_size, self.closing_kernel_size))
            
            # Apply morphological closing: dilation followed by erosion
            # This fills small holes while preserving the overall shape
            mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to float32
            panel_mask = (mask_closed / 255.0).astype(np.float32)
        
        # Enforce rectangular mask (fill holes inside panel boundary)
        if self.force_rectangular_mask:
            panel_mask = self._enforce_rectangular_mask(panel_mask)
        
        # Calculate statistics
        plane_pixels = (panel_mask > 0.5)
        stats = {
            'plane_percentage': float(np.sum(plane_pixels) / panel_mask.size * 100),
            'plane_pixel_count': int(np.sum(plane_pixels)),
            'total_pixels': int(panel_mask.size),
            'residual_threshold_used': threshold,
            'num_points': len(points_3d),
            'gaussian_smoothing_applied': True,  # Always applied now
            'gaussian_sigma_used': sigma_used,
            'morphological_closing_applied': self.apply_morphological_closing,
            'closing_kernel_size': self.closing_kernel_size if self.apply_morphological_closing else None,
            'rectangular_mask_enforced': self.force_rectangular_mask,
            'valid_pixels_before_processing': int(np.sum(valid_mask_original)),
            'negative_values_converted': unit_stats['converted_neg'],
            'millimeters_converted': unit_stats['converted_mm']
        }
        
        return panel_mask, stats, plane_coefficients
    
    def _enforce_rectangular_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Enforce a rectangular panel mask by finding the largest contour and drawing
        its rotated bounding box. This ensures deep dents inside the panel boundary
        are included as valid panel area.
        
        Args:
            mask: Input panel mask (H, W) as float32 array (0.0 to 1.0)
            
        Returns:
            Rectangular mask (H, W) as float32 array with solid rectangle filled
        """
        # Convert to uint8 for contour detection
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find contours of white regions (panel areas)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            # No contours found, return original mask
            return mask
        
        # Find the largest contour (assuming this is the main container wall)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if largest contour has minimum area (at least 100 pixels)
        if cv2.contourArea(largest_contour) < 100:
            # Contour too small, return original mask
            return mask
        
        # Compute rotated bounding box for the largest contour
        # This handles cases where the camera is slightly diagonal
        rotated_rect = cv2.minAreaRect(largest_contour)
        box_points = cv2.boxPoints(rotated_rect)
        box_points = box_points.astype(np.int32)  # Convert to integer coordinates
        
        # Create a new mask with the rectangle filled
        rectangular_mask = np.zeros_like(mask_uint8)
        
        # Draw filled rotated rectangle
        cv2.fillPoly(rectangular_mask, [box_points], 255)
        
        # Convert back to float32
        rectangular_mask_float = (rectangular_mask / 255.0).astype(np.float32)
        
        return rectangular_mask_float
    
    def extract_panel(self, depth: np.ndarray, 
                     fill_background: bool = False,
                     fill_value: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, dict, Optional[np.ndarray]]:
        """
        Extract panel from depth map and optionally fill background.
        
        Args:
            depth: Input depth map (H, W) as numpy array (float32)
            fill_background: If True, fill non-panel areas with fill_value
            fill_value: Value to fill background with. If None, uses median panel depth.
            
        Returns:
            Tuple of (processed_depth, panel_mask, stats_dict, plane_coefficients)
            - processed_depth: (H, W) depth map with background optionally filled
            - panel_mask: (H, W) binary mask where 1.0 = panel pixels
            - stats_dict: Dictionary with extraction statistics
            - plane_coefficients: Array [a, b, -1, c] where z = ax + by + c in camera space, or None if not available
        """
        # Standardize depth units (mm -> m, negative -> positive)
        depth, unit_stats = self._standardize_depth_units(depth)
        
        # Extract panel mask (standardizes negative values to positive if needed)
        panel_mask, stats, plane_coefficients = self.extract_panel_mask(depth)
        
        # Add unit conversion stats to the returned stats
        stats['negative_values_converted'] = unit_stats['converted_neg']
        stats['millimeters_converted'] = unit_stats['converted_mm']
        
        # Process depth map (now using converted depth)
        processed_depth = depth.copy()
        
        if fill_background:
            # Determine fill value
            fill_method = 'custom'
            if fill_value is None:
                # Use median panel depth (from converted depth)
                fill_method = 'median'
                plane_pixels = (panel_mask > 0.5)
                if np.any(plane_pixels):
                    plane_depths = depth[plane_pixels]  # Now using converted depth
                    valid_plane_depths = plane_depths[(plane_depths != 0) & np.isfinite(plane_depths)]
                    if len(valid_plane_depths) > 0:
                        fill_value = np.median(valid_plane_depths)
                        stats['plane_median_depth'] = float(fill_value)
                        stats['plane_mean_depth'] = float(np.mean(valid_plane_depths))
                    else:
                        fill_value = 0.0
                else:
                    fill_value = 0.0
            
            # Fill non-panel areas
            non_panel_mask = (panel_mask <= 0.5)
            processed_depth[non_panel_mask] = fill_value
            
            stats['fill_method'] = fill_method
            stats['fill_value'] = float(fill_value)
        else:
            # Set non-panel areas to 0
            non_panel_mask = (panel_mask <= 0.5)
            processed_depth[non_panel_mask] = 0.0
            stats['fill_method'] = 'zero'
        
        return processed_depth, panel_mask, stats, plane_coefficients


# Convenience function for quick usage
def extract_panel(depth: np.ndarray,
                 camera_fov: float = DEFAULT_CAMERA_FOV,
                 residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
                 adaptive_threshold: bool = True,
                 fill_background: bool = False,
                 apply_morphological_closing: bool = True,
                 closing_kernel_size: int = DEFAULT_CLOSING_KERNEL_SIZE,
                 force_rectangular_mask: bool = True) -> Tuple[np.ndarray, np.ndarray, dict, Optional[np.ndarray]]:
    """
    Convenience function to extract panel from depth map.
    
    Args:
        depth: Input depth map (H, W) as numpy array
        camera_fov: Camera field of view in degrees
        residual_threshold: RANSAC residual threshold (only if adaptive_threshold=False)
        adaptive_threshold: Use adaptive threshold calculation
        fill_background: Fill non-panel areas with median panel depth
        apply_morphological_closing: Apply morphological closing to fill small holes (dents)
        closing_kernel_size: Size of the morphological closing kernel in pixels
        force_rectangular_mask: If True, enforce rectangular mask by finding largest contour
                               and drawing its rotated bounding box
        
    Returns:
        Tuple of (processed_depth, panel_mask, stats_dict, plane_coefficients)
        - processed_depth: (H, W) depth map with background optionally filled
        - panel_mask: (H, W) binary mask where 1.0 = panel pixels
        - stats_dict: Dictionary with extraction statistics
        - plane_coefficients: Array [a, b, -1, c] where z = ax + by + c in camera space, or None if not available
    """
    extractor = RANSACPanelExtractor(
        camera_fov=camera_fov,
        residual_threshold=residual_threshold,
        adaptive_threshold=adaptive_threshold,
        apply_morphological_closing=apply_morphological_closing,
        closing_kernel_size=closing_kernel_size,
        force_rectangular_mask=force_rectangular_mask
    )
    return extractor.extract_panel(depth, fill_background=fill_background)

