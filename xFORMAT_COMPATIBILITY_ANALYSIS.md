# Format Compatibility Analysis: Training vs Inference

## Summary

**✅ GOOD NEWS:** The preprocessing pipelines are **mostly compatible**, but there are **critical differences** that could affect detection accuracy.

---

## Training Data Format (DentContainer2)

### Files Generated:
1. **`*_dented_depth_raw.npy`** - Raw depth maps (before RANSAC, with background)
2. **`*_dented_depth.npy`** - **MASKED depth maps** (after RANSAC panel extraction, background = 0)
3. **`*_dent_mask.png`** - Ground truth segmentation masks

### Training Dataset Preprocessing (`_safe_normalize_and_fill`):
```python
# Training notebook does:
1. Loads *_dented_depth.npy (MASKED depth maps)
2. Fills background (0 values) with median depth
3. Normalizes to [0.1, 1.0] range
4. Computes gradients on filled depth
5. Masks gradients to remove background noise
6. Creates 3-channel input: [normalized_depth, normalized_gx, normalized_gy]
```

**Key Point:** Training uses **MASKED depth maps** (background already set to 0 by RANSAC).

---

## Inference Format (DentContainerModel)

### Input:
- Raw depth maps (`.npy` files) - can be either:
  - Raw depth maps (with background)
  - Pre-masked depth maps (background = 0)

### Inference Preprocessing (`preprocess_depth`):
```python
# Inference does:
1. Optionally applies RANSAC panel extraction (if use_ransac=True)
   - If enabled: extracts panel, fills background with median
   - If disabled: simple median fill of invalid pixels
2. Normalizes to [0.1, 1.0] range
3. Computes gradients on cleaned depth
4. Masks gradients to remove background noise
5. Creates 3-channel input: [normalized_depth, normalized_gx, normalized_gy]
```

**Key Point:** Inference can handle both raw and masked depth maps, but **RANSAC should be enabled** for best compatibility.

---

## Format Compatibility Analysis

### ✅ **COMPATIBLE Scenarios:**

#### Scenario 1: Using Pre-Masked Depth Maps (Recommended)
```
Training: *_dented_depth.npy (masked, background=0)
Inference: Pre-masked depth maps (background=0)
RANSAC: Disabled (not needed, already masked)
Result: ✅ FULLY COMPATIBLE
```

#### Scenario 2: Using Raw Depth Maps with RANSAC
```
Training: *_dented_depth.npy (masked, background=0)
Inference: Raw depth maps + RANSAC enabled
RANSAC: Enabled (extracts panel, fills background)
Result: ✅ COMPATIBLE (RANSAC mimics training preprocessing)
```

### ⚠️ **POTENTIALLY INCOMPATIBLE Scenarios:**

#### Scenario 3: Using Raw Depth Maps WITHOUT RANSAC
```
Training: *_dented_depth.npy (masked, background=0)
Inference: Raw depth maps, RANSAC disabled
RANSAC: Disabled (only median fill)
Result: ⚠️ PARTIALLY COMPATIBLE
- Background handling differs
- May include non-panel regions
- Detection accuracy may be reduced
```

---

## Critical Differences

### 1. **Background Handling**

**Training:**
- Uses masked depth maps (background = 0)
- Fills 0 values with median depth
- Only panel regions have valid depth values

**Inference (without RANSAC):**
- Uses raw depth maps (background has actual depth values)
- Fills invalid pixels with median
- May include non-panel regions (ground, structures, etc.)

**Impact:** If RANSAC is disabled, inference may process non-panel regions that weren't in training data.

### 2. **RANSAC Application**

**Training Data Generation:**
- RANSAC applied during data generation
- Masked depth maps saved to dataset
- Training never sees raw depth maps

**Inference:**
- RANSAC is **optional** (can be enabled/disabled)
- If disabled, preprocessing differs from training

**Impact:** Disabling RANSAC creates a distribution shift between training and inference.

### 3. **Normalization Range**

**Both use:** `[0.1, 1.0]` range ✅ **COMPATIBLE**

**Formula:** `0.1 + 0.9 * (val - min) / (max - min)`

### 4. **Gradient Computation**

**Both use:** 
- Sobel gradients (ksize=3) ✅ **COMPATIBLE**
- Computed on filled/cleaned depth ✅ **COMPATIBLE**
- Masked to remove background ✅ **COMPATIBLE**

### 5. **Input Format**

**Both create:** 3-channel input `[normalized_depth, normalized_gx, normalized_gy]` ✅ **COMPATIBLE**

---

## Recommendations

### ✅ **For Best Detection Accuracy:**

1. **Use Pre-Masked Depth Maps** (if available):
   ```python
   # Load masked depth maps (background = 0)
   depth = np.load("path/to/dented_depth.npy")  # Already masked
   # Disable RANSAC (not needed)
   mask = predict_mask(model, depth, use_ransac=False)
   ```

2. **Use Raw Depth Maps WITH RANSAC** (if pre-masked not available):
   ```python
   # Load raw depth maps
   depth = np.load("path/to/raw_depth.npy")  # With background
   # Enable RANSAC to match training preprocessing
   mask = predict_mask(model, depth, use_ransac=True, ransac_extractor=extractor)
   ```

3. **Avoid Raw Depth Maps WITHOUT RANSAC**:
   ```python
   # ⚠️ NOT RECOMMENDED
   depth = np.load("path/to/raw_depth.npy")
   mask = predict_mask(model, depth, use_ransac=False)  # May reduce accuracy
   ```

---

## Detection Accuracy Considerations

### ✅ **Correct Detection Expected When:**
- Using masked depth maps (training format)
- Using raw depth maps with RANSAC enabled
- Same camera setup (distance, FOV, resolution)

### ⚠️ **Reduced Detection Accuracy Possible When:**
- Using raw depth maps without RANSAC
- Different camera parameters (distance, FOV)
- Different image resolution (though model handles this)

### 🔍 **Format Verification:**

To verify your test data format matches training:

```python
import numpy as np

# Load your test depth map
depth = np.load("your_test_depth.npy")

# Check if it's masked (background = 0)
zero_pixels = np.sum(depth == 0)
total_pixels = depth.size
zero_percentage = (zero_pixels / total_pixels) * 100

if zero_percentage > 10:  # Significant background
    print("⚠️  Appears to be RAW depth map - Enable RANSAC for best results")
    print(f"   Zero pixels: {zero_percentage:.1f}%")
else:
    print("✅ Appears to be MASKED depth map - RANSAC optional")
    print(f"   Zero pixels: {zero_percentage:.1f}%")
```

---

## Conclusion

**Format Compatibility:** ✅ **YES** (when RANSAC is used correctly)

**Detection Accuracy:** ✅ **Should work correctly** if:
1. Using masked depth maps, OR
2. Using raw depth maps with RANSAC enabled

**Potential Issues:** ⚠️ **May occur** if:
1. Using raw depth maps without RANSAC
2. Camera parameters differ significantly from training

**Recommendation:** Always enable RANSAC when using raw depth maps to match training data preprocessing.

