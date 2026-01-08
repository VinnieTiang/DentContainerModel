# RANSAC Implementation Comparison: DentContainer2 vs DentContainerModel

## ⚠️ CRITICAL FINDING: Missing Gaussian Smoothing Step!

**Your DentContainerModel RANSAC is MISSING the Gaussian smoothing step that DentContainer2 uses before RANSAC!**

---

## Detailed Comparison

### ✅ **MATCHING Parameters:**

| Parameter | DentContainer2 | DentContainerModel | Status |
|----------|---------------|-------------------|---------|
| **Camera FOV** | 75.0° | 75.0° (DEFAULT_CAMERA_FOV) | ✅ MATCH |
| **Max Trials** | 1000 | 1000 | ✅ MATCH |
| **Random State** | 42 | 42 | ✅ MATCH |
| **Min Samples** | 3 | 3 | ✅ MATCH |
| **Adaptive Threshold** | Enabled (True) | Enabled (True) | ✅ MATCH |
| **Scale Factor** | 1.5 | 1.5 | ✅ MATCH |
| **Threshold Bounds** | [0.015, 0.05] | [0.015, 0.05] | ✅ MATCH |

### ⚠️ **DIFFERENT Parameters:**

| Parameter | DentContainer2 | DentContainerModel | Impact |
|----------|---------------|-------------------|---------|
| **Base Threshold** | 0.02m (2cm) | 0.03m (3cm) | ⚠️ Minor difference |
| **Gaussian Smoothing** | ✅ **APPLIED BEFORE RANSAC** | ❌ **NOT APPLIED** | 🔴 **CRITICAL** |
| **Downsampling** | ❌ Not used | ✅ Used (factor=4) | ⚠️ Performance optimization |
| **Morphological Closing** | ❌ Not applied | ✅ Applied (kernel=30) | ⚠️ Post-processing |
| **Rectangular Mask** | ❌ Not enforced | ✅ Enforced | ⚠️ Post-processing |

---

## 🔴 **CRITICAL DIFFERENCE #1: Gaussian Smoothing**

### DentContainer2 (Dataset Generation):
```python
# Step 1: Apply Gaussian smoothing BEFORE RANSAC
original_depth_smoothed, sigma_used = self._apply_gaussian_smoothing(
    original_depth, 
    adaptive=True  # Automatically tunes sigma based on depth variance
)

# Step 2: Apply RANSAC on SMOOTHED depth map
panel_mask = self._generate_panel_mask_ransac(
    original_depth_smoothed,  # ← Uses SMOOTHED depth!
    adaptive_threshold=True,
    max_trials=1000
)
```

**Purpose:** Flattens corrugation patterns so RANSAC can fit a plane to the main panel surface without being confused by corrugation variations.

**Adaptive Sigma Calculation:**
- Base sigma: 8.0
- Min sigma: 4.0
- Max sigma: 18.0
- Scales based on depth variance/relative std

### DentContainerModel (Inference):
```python
# ❌ NO Gaussian smoothing step!
# Directly applies RANSAC to raw depth map
panel_mask = extractor.extract_panel_mask(depth)  # ← Uses RAW depth!
```

**Problem:** Without smoothing, RANSAC may:
- Be confused by corrugation patterns
- Treat corrugation peaks/valleys as outliers
- Produce different panel masks than training data

---

## ⚠️ **DIFFERENCE #2: Base Threshold**

### DentContainer2:
```python
base_threshold = 0.02  # 2cm default
```

### DentContainerModel:
```python
DEFAULT_RESIDUAL_THRESHOLD = 0.03  # 3cm default
```

**Impact:** Minor - adaptive threshold will override this anyway, but initial threshold differs.

---

## ⚠️ **DIFFERENCE #3: Downsampling**

### DentContainer2:
- No downsampling - processes full resolution

### DentContainerModel:
- Downsamples by factor of 4 for faster RANSAC
- Upsamples mask back to full resolution using INTER_NEAREST

**Impact:** Performance optimization, but may cause slight differences in mask boundaries.

---

## ⚠️ **DIFFERENCE #4: Post-Processing**

### DentContainer2:
- No morphological closing
- No rectangular mask enforcement

### DentContainerModel:
- Morphological closing (kernel size 30) - fills small holes
- Rectangular mask enforcement - ensures rectangular panel boundary

**Impact:** These are post-processing steps that may help but weren't used during training data generation.

---

## 🔧 **Recommended Fix: Add Gaussian Smoothing**

To match DentContainer2 exactly, you need to add Gaussian smoothing BEFORE RANSAC:

```python
# Add to panel_extractor.py
from scipy.ndimage import gaussian_filter

def _apply_gaussian_smoothing(self, depth: np.ndarray, 
                             adaptive: bool = True) -> Tuple[np.ndarray, float]:
    """
    Apply Gaussian smoothing to depth map to flatten corrugation patterns.
    Matches DentContainer2 implementation exactly.
    """
    if adaptive:
        sigma = self._calculate_adaptive_sigma(depth)
    else:
        sigma = 8.0  # Default fallback
    
    valid_mask = (depth > 0) & np.isfinite(depth)
    if not np.any(valid_mask):
        return depth.copy(), sigma
    
    smoothed_depth = depth.copy()
    smoothed = gaussian_filter(depth, sigma=sigma, mode='constant', cval=0.0)
    smoothed_depth[valid_mask] = smoothed[valid_mask]
    smoothed_depth[~valid_mask] = depth[~valid_mask]
    
    return smoothed_depth, sigma

def _calculate_adaptive_sigma(self, depth: np.ndarray,
                             min_sigma: float = 4.0,
                             max_sigma: float = 18.0,
                             base_sigma: float = 8.0) -> float:
    """
    Calculate adaptive sigma for Gaussian smoothing.
    Matches DentContainer2 implementation exactly.
    """
    valid_mask = (depth > 0) & np.isfinite(depth)
    if not np.any(valid_mask):
        return base_sigma
    
    valid_depths = depth[valid_mask]
    depth_variance = np.var(valid_depths)
    depth_std = np.std(valid_depths)
    depth_mean = np.mean(valid_depths)
    
    relative_std = depth_std / (depth_mean + 1e-6)
    scale_factor = 2.0
    adaptive_sigma = base_sigma * (1.0 + relative_std * scale_factor)
    adaptive_sigma = np.clip(adaptive_sigma, min_sigma, max_sigma)
    
    return float(adaptive_sigma)

# Modify extract_panel_mask to use smoothing:
def extract_panel_mask(self, depth: np.ndarray) -> Tuple[np.ndarray, dict]:
    depth = depth.astype(np.float32)
    
    # ✅ ADD THIS: Apply Gaussian smoothing BEFORE RANSAC
    depth_smoothed, sigma_used = self._apply_gaussian_smoothing(depth, adaptive=True)
    
    # Calculate adaptive threshold if needed
    threshold = self.residual_threshold
    if self.adaptive_threshold:
        threshold = self._calculate_adaptive_threshold(depth_smoothed)  # Use smoothed depth
    
    # ... rest of RANSAC code using depth_smoothed instead of depth ...
```

---

## 📊 **Current Settings Comparison**

### What You're Currently Doing (DentContainerModel):

```python
# In app.py, when RANSAC is enabled:
extractor = RANSACPanelExtractor(
    camera_fov=75.0,                    # ✅ Matches
    residual_threshold=0.02,            # ⚠️ Different default (0.03)
    adaptive_threshold=True,            # ✅ Matches
    downsample_factor=4,                # ⚠️ Not in DentContainer2
    apply_morphological_closing=True,   # ⚠️ Not in DentContainer2
    closing_kernel_size=30,             # ⚠️ Not in DentContainer2
    force_rectangular_mask=True         # ⚠️ Not in DentContainer2
)
```

### What DentContainer2 Does:

```python
# In compare_dents_depth.py:
# 1. Apply Gaussian smoothing
original_depth_smoothed, sigma = self._apply_gaussian_smoothing(
    original_depth, adaptive=True
)

# 2. Apply RANSAC on smoothed depth
panel_mask = self._generate_panel_mask_ransac(
    original_depth_smoothed,           # ← SMOOTHED depth!
    adaptive_threshold=True,            # ✅ Matches
    max_trials=1000                     # ✅ Matches
)
# No downsampling, no morphological closing, no rectangular mask
```

---

## ✅ **Recommended Settings for Exact Match**

To match DentContainer2 exactly:

```python
extractor = RANSACPanelExtractor(
    camera_fov=75.0,                    # ✅ Must match
    residual_threshold=0.02,            # ✅ Change from 0.03 to 0.02
    adaptive_threshold=True,            # ✅ Must match
    max_trials=1000,                    # ✅ Must match
    downsample_factor=1,                 # ✅ Change from 4 to 1 (no downsampling)
    apply_morphological_closing=False,   # ✅ Change from True to False
    force_rectangular_mask=False        # ✅ Change from True to False
)

# ✅ CRITICAL: Add Gaussian smoothing BEFORE calling extract_panel
depth_smoothed, sigma = extractor._apply_gaussian_smoothing(depth, adaptive=True)
cleaned_depth, panel_mask, stats = extractor.extract_panel(
    depth_smoothed,  # ← Use SMOOTHED depth!
    fill_background=True
)
```

---

## 🎯 **Action Items**

1. **🔴 CRITICAL:** Add Gaussian smoothing step before RANSAC
2. **⚠️ IMPORTANT:** Change default residual_threshold from 0.03 to 0.02
3. **⚠️ OPTIONAL:** Disable downsampling (set downsample_factor=1) for exact match
4. **⚠️ OPTIONAL:** Disable morphological closing for exact match
5. **⚠️ OPTIONAL:** Disable rectangular mask enforcement for exact match

---

## 📝 **Summary**

**Current Status:** ❌ **NOT EXACTLY MATCHING** - Missing Gaussian smoothing step

**Impact:** Without Gaussian smoothing, RANSAC may produce different panel masks than training data, potentially reducing detection accuracy.

**Priority:** 🔴 **HIGH** - Add Gaussian smoothing to match training data preprocessing exactly.

