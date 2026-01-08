# RANSAC Plane Detection - Fill Value Recommendations

## Current Situation

When RANSAC detects the target plane and crops out non-plane areas, those areas need to be filled with some value in the `.npy` files. The current normalization code handles invalid values as follows:

```python
valid = np.isfinite(depth) & (depth > 0)
norm = (depth - d_min) / (d_max - d_min)
norm[~valid] = 0.0  # Invalid pixels become 0.0
```

## Problem with Current Approach

If cropped areas are filled with **zeros** or **NaN**, they become `0.0` after normalization, which is ambiguous because:

- `0.0` could mean "minimum valid depth"
- `0.0` could mean "invalid/cropped area"
- This ambiguity can confuse the model during training

## Recommended Solutions

### ✅ **Option 1: Fill with NaN (Best for Current Code)**

**Implementation:**

```python
# After RANSAC plane detection
plane_mask = ransac_inlier_mask  # Boolean mask of plane pixels
depth_cropped = depth.copy()
depth_cropped[~plane_mask] = np.nan  # Fill non-plane areas with NaN
np.save("output.npy", depth_cropped)
```

**Why this works:**

- NaN values are automatically filtered by `np.isfinite(depth)`
- They become `0.0` after normalization (consistent behavior)
- Clear semantic meaning: "no data here"

**Pros:**

- ✅ Works with existing code without changes
- ✅ Clear semantic meaning
- ✅ No ambiguity with valid depth values

**Cons:**

- ⚠️ Still becomes `0.0` after normalization (same as minimum depth)

---

### ✅ **Option 2: Fill with Plane Depth Value (RECOMMENDED)**

**Implementation:**

```python
# After RANSAC plane detection
plane_mask = ransac_inlier_mask
plane_depths = depth[plane_mask]
plane_median = np.median(plane_depths)  # or np.mean()

depth_filled = depth.copy()
depth_filled[~plane_mask] = plane_median  # Fill with plane depth
np.save("output.npy", depth_filled)
```

**Why this is better:**

- Non-plane areas have a consistent, meaningful depth value
- After normalization, they'll normalize to a value representing the plane depth
- Model learns that these areas are "at plane level" rather than "invalid"

**Pros:**

- ✅ Most semantically meaningful
- ✅ Preserves depth information
- ✅ Model learns correct associations
- ✅ Better for gradient computation

**Cons:**

- ⚠️ Requires computing plane statistics

---

### ✅ **Option 3: Fill with Large Negative Value**

**Implementation:**

```python
# After RANSAC plane detection
plane_mask = ransac_inlier_mask
INVALID_DEPTH = -9999.0  # Clearly invalid value

depth_filled = depth.copy()
depth_filled[~plane_mask] = INVALID_DEPTH
np.save("output.npy", depth_filled)
```

**Why this works:**

- Filtered by `depth > 0` check
- Becomes `0.0` after normalization (consistent)
- Clear that it's invalid

**Pros:**

- ✅ Simple to implement
- ✅ Clear semantic meaning
- ✅ Works with existing code

**Cons:**

- ⚠️ Still becomes `0.0` after normalization

---

## Impact on Model Training

### Current Approach (Zeros/NaN → 0.0):

- ❌ Model may confuse minimum depth with invalid areas
- ❌ Gradient computation at boundaries may be noisy
- ❌ Potential for false positives at cropped boundaries

### Recommended Approach (Plane Depth Value):

- ✅ Model learns consistent depth representation
- ✅ Better gradient information
- ✅ Reduced ambiguity in training
- ✅ Better generalization

## Recommended Code Change for Preprocessing

If you control the RANSAC preprocessing step, use this approach:

```python
import numpy as np
from sklearn.linear_model import RANSACRegressor

def ransac_plane_fill(depth_map, ransac_threshold=0.01):
    """
    Detect plane using RANSAC and fill non-plane areas with plane depth.

    Args:
        depth_map: Input depth map (H, W)
        ransac_threshold: Distance threshold for RANSAC inliers

    Returns:
        Filled depth map where non-plane areas have plane median depth
    """
    # Create point cloud from depth map
    h, w = depth_map.shape
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    points = np.column_stack([
        x_coords.flatten(),
        y_coords.flatten(),
        depth_map.flatten()
    ])

    # Filter valid points
    valid_mask = np.isfinite(depth_map) & (depth_map > 0)
    valid_points = points[valid_mask.flatten()]

    if len(valid_points) < 3:
        return depth_map  # Not enough points for plane fitting

    # Fit plane using RANSAC
    # Using simple linear regression: z = ax + by + c
    X = valid_points[:, :2]  # x, y coordinates
    y = valid_points[:, 2]    # depth values

    ransac = RANSACRegressor(
        residual_threshold=ransac_threshold,
        random_state=42,
        max_trials=100
    )
    ransac.fit(X, y)

    # Get inlier mask
    inlier_mask = ransac.inlier_mask_
    plane_depths = y[inlier_mask]
    plane_median = np.median(plane_depths)

    # Fill non-plane areas with plane median depth
    depth_filled = depth_map.copy()
    valid_flat = valid_mask.flatten()
    non_plane_mask = valid_flat & (~inlier_mask)

    # Set non-plane valid pixels to plane median
    depth_filled_flat = depth_filled.flatten()
    depth_filled_flat[non_plane_mask] = plane_median
    depth_filled = depth_filled_flat.reshape(h, w)

    return depth_filled
```

## Summary

**Best Practice:** Fill non-plane areas with the **median (or mean) depth of the detected plane**. This provides:

1. Consistent depth values that normalize meaningfully
2. Better training signal for the model
3. Reduced ambiguity between "minimum depth" and "invalid area"
4. Improved gradient computation

**Current Code Compatibility:** All three options work with your existing normalization code, but Option 2 (plane depth) provides the best training signal.
