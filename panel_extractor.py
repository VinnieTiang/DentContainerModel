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

# Default configuration constants (single source of truth)
DEFAULT_CLOSING_KERNEL_SIZE = 30
DEFAULT_CAMERA_FOV = 75.0
DEFAULT_RESIDUAL_THRESHOLD = 0.02  # Changed from 0.03 to 0.02 to match DentContainer2
DEFAULT_DOWNSAMPLE_FACTOR = 4


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
    
    def _depth_to_points_camera_space(self, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert depth map to 3D points in camera space.
        
        Args:
            depth: Depth map (H, W) in meters
            
        Returns:
            Tuple of (points_3d, valid_mask)
            - points_3d: (N, 3) array of 3D points [x, y, z]
            - valid_mask: (H, W) boolean mask of valid depth pixels
        """
        height, width = depth.shape
        
        # Calculate camera intrinsics from FOV
        fov_y_rad = np.deg2rad(self.camera_fov)
        focal_length = (height / 2.0) / np.tan(fov_y_rad / 2.0)
        cx, cy = width / 2.0, height / 2.0
        
        # Create pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to normalized camera coordinates
        x_norm = (u - cx) / focal_length
        y_norm = (v - cy) / focal_length
        
        # Get valid depth pixels
        valid_mask = (depth > 0) & np.isfinite(depth)
        
        # Back-project to 3D points in camera space
        x_cam = x_norm[valid_mask] * depth[valid_mask]
        y_cam = y_norm[valid_mask] * depth[valid_mask]
        z_cam = depth[valid_mask]
        
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
        # Get valid depth pixels
        valid_mask = (depth > 0) & np.isfinite(depth)
        
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
        
        Args:
            depth: Depth map (H, W) in meters
            sigma: Standard deviation for Gaussian kernel (if None and adaptive=True, will be calculated)
            adaptive: If True, automatically tune sigma based on depth variance
            
        Returns:
            Tuple of (smoothed_depth_map, sigma_used)
        """
        # Calculate adaptive sigma if requested
        if adaptive and sigma is None:
            sigma = self._calculate_adaptive_sigma(depth)
        elif sigma is None:
            sigma = 8.0  # Default fallback
        
        # Only smooth valid depth pixels (non-zero and finite)
        valid_mask = (depth > 0) & np.isfinite(depth)
        
        if not np.any(valid_mask):
            return depth.copy(), sigma
        
        # Create a copy and apply Gaussian filter
        smoothed_depth = depth.copy()
        
        # Apply Gaussian filter to valid regions
        # Use gaussian_filter which handles NaN/inf gracefully by only filtering valid pixels
        smoothed = gaussian_filter(depth, sigma=sigma, mode='constant', cval=0.0)
        
        # Preserve invalid pixels (set smoothed invalid pixels back to original)
        smoothed_depth[valid_mask] = smoothed[valid_mask]
        smoothed_depth[~valid_mask] = depth[~valid_mask]
        
        return smoothed_depth, sigma
    
    def _calculate_adaptive_threshold(self, depth: np.ndarray) -> float:
        """
        Calculate adaptive residual threshold based on corrugation depth.
        Matches DentContainer2 implementation exactly.
        
        Args:
            depth: Depth map (H, W)
            
        Returns:
            Adaptive threshold value
        """
        valid_mask = (depth > 0) & np.isfinite(depth)
        
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
    
    def _fit_plane_ransac(self, points: np.ndarray, threshold: float) -> np.ndarray:
        """
        Fit a plane to 3D points using RANSAC.
        
        Args:
            points: (N, 3) array of 3D points
            threshold: Residual threshold for inlier detection
            
        Returns:
            Boolean mask of inlier points (True = panel, False = background)
        """
        if len(points) < 3:
            return np.zeros(len(points), dtype=bool)
        
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
            return inlier_mask
        except Exception as e:
            # If RANSAC fails, treat all points as inliers
            print(f"Warning: RANSAC fitting failed: {e}. Using all points as inliers.")
            return np.ones(len(points), dtype=bool)
    
    def extract_panel_mask(self, depth: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Extract panel mask from depth map using RANSAC plane fitting.
        
        CRITICAL: This function now matches DentContainer2 preprocessing exactly by:
        1. Applying Gaussian smoothing BEFORE RANSAC (to flatten corrugation patterns)
        2. Using smoothed depth for RANSAC plane fitting
        
        Args:
            depth: Input depth map (H, W) as numpy array (float32)
            
        Returns:
            Tuple of (panel_mask, stats_dict)
            - panel_mask: (H, W) binary mask where 1.0 = panel pixels, 0.0 = background
            - stats_dict: Dictionary with extraction statistics
        """
        depth = depth.astype(np.float32)
        
        # CRITICAL STEP: Apply Gaussian smoothing BEFORE RANSAC
        # This matches DentContainer2 preprocessing exactly
        depth_smoothed, sigma_used = self._apply_gaussian_smoothing(depth, adaptive=True)
        
        # Calculate adaptive threshold if needed (use smoothed depth for threshold calculation)
        threshold = self.residual_threshold
        if self.adaptive_threshold:
            threshold = self._calculate_adaptive_threshold(depth_smoothed)
        
        # Downsample for faster RANSAC fitting (use smoothed depth)
        if self.downsample_factor > 1:
            h, w = depth_smoothed.shape
            h_ds = h // self.downsample_factor
            w_ds = w // self.downsample_factor
            depth_ds = cv2.resize(depth_smoothed, (w_ds, h_ds), interpolation=cv2.INTER_AREA)
        else:
            depth_ds = depth_smoothed
        
        # Convert to 3D points
        points_3d, valid_mask_ds = self._depth_to_points_camera_space(depth_ds)
        
        if len(points_3d) == 0:
            # No valid points
            panel_mask = np.zeros_like(depth, dtype=np.float32)
            return panel_mask, {
                'plane_percentage': 0.0,
                'residual_threshold_used': threshold,
                'num_points': 0
            }
        
        # Fit plane using RANSAC
        inlier_mask_points = self._fit_plane_ransac(points_3d, threshold)
        
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
            'rectangular_mask_enforced': self.force_rectangular_mask
        }
        
        return panel_mask, stats
    
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
                     fill_value: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Extract panel from depth map and optionally fill background.
        
        Args:
            depth: Input depth map (H, W) as numpy array (float32)
            fill_background: If True, fill non-panel areas with fill_value
            fill_value: Value to fill background with. If None, uses median panel depth.
            
        Returns:
            Tuple of (processed_depth, panel_mask, stats_dict)
            - processed_depth: (H, W) depth map with background optionally filled
            - panel_mask: (H, W) binary mask where 1.0 = panel pixels
            - stats_dict: Dictionary with extraction statistics
        """
        # Extract panel mask
        panel_mask, stats = self.extract_panel_mask(depth)
        
        # Process depth map
        processed_depth = depth.copy()
        
        if fill_background:
            # Determine fill value
            if fill_value is None:
                # Use median panel depth
                plane_pixels = (panel_mask > 0.5)
                if np.any(plane_pixels):
                    plane_depths = depth[plane_pixels]
                    valid_plane_depths = plane_depths[(plane_depths > 0) & np.isfinite(plane_depths)]
                    if len(valid_plane_depths) > 0:
                        fill_value = np.median(valid_plane_depths)
                    else:
                        fill_value = 0.0
                else:
                    fill_value = 0.0
            
            # Fill non-panel areas
            non_panel_mask = (panel_mask <= 0.5)
            processed_depth[non_panel_mask] = fill_value
            
            stats['fill_method'] = 'median' if fill_value is None else 'custom'
            stats['fill_value'] = float(fill_value)
        else:
            # Set non-panel areas to 0
            non_panel_mask = (panel_mask <= 0.5)
            processed_depth[non_panel_mask] = 0.0
            stats['fill_method'] = 'zero'
        
        return processed_depth, panel_mask, stats


# Convenience function for quick usage
def extract_panel(depth: np.ndarray,
                 camera_fov: float = DEFAULT_CAMERA_FOV,
                 residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
                 adaptive_threshold: bool = True,
                 fill_background: bool = False,
                 apply_morphological_closing: bool = True,
                 closing_kernel_size: int = DEFAULT_CLOSING_KERNEL_SIZE,
                 force_rectangular_mask: bool = True) -> Tuple[np.ndarray, np.ndarray, dict]:
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
        Tuple of (processed_depth, panel_mask, stats_dict)
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

