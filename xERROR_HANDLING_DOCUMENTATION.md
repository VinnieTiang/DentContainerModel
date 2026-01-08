# Error Handling Documentation

## Dent Container Detection Pipeline

This document provides a comprehensive overview of error handling implemented throughout the entire pipeline, organized by component and scenario.

---

## 📋 Table of Contents

1. [UI Application (app.py)](#1-ui-application-apppy)
2. [Model Architecture (model_architecture.py)](#2-model-architecture-model_architecturepy)
3. [Training Pipeline (Notebook)](#3-training-pipeline-notebook)
4. [Summary Table](#summary-table)

---

## 1. UI Application (app.py)

### 1.1 Model Loading Errors

**Scenario:** User attempts to load an invalid or corrupted model file

**Location:** Lines 213-240

**Code Snippet:**

```python
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

        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.session_state.model_loaded = False
    else:
        st.error("❌ Model file not found!")
```

**How it handles:**

- ✅ Checks if model file exists before attempting to load
- ✅ Wraps model loading in try-except block
- ✅ Catches all exceptions (corrupted files, wrong architecture, missing keys, etc.)
- ✅ Displays user-friendly error message
- ✅ Resets model_loaded flag to prevent invalid state
- ✅ Shows error in UI with ❌ emoji for visibility

**Error Types Handled:**

- FileNotFoundError (handled by existence check)
- RuntimeError (wrong model architecture)
- KeyError (missing state dict keys)
- Corrupted file errors
- CUDA/device errors

---

### 1.2 Depth Map Loading Errors

**Scenario:** User uploads invalid or corrupted depth map file

**Location:** Lines 418-444

**Code Snippet:**

```python
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
```

**How it handles:**

- ✅ Validates file upload exists
- ✅ Catches numpy loading errors (corrupted .npy files)
- ✅ Catches visualization errors (invalid array shapes)
- ✅ Displays informative error message
- ✅ Clears session state to prevent invalid data propagation
- ✅ Provides user guidance ("Please upload a depth map")

**Error Types Handled:**

- ValueError (invalid array format)
- OSError (file read errors)
- MemoryError (file too large)
- Visualization errors

---

### 1.3 RGB Image Loading Errors

**Scenario:** User uploads invalid or corrupted RGB image

**Location:** Lines 447-477

**Code Snippet:**

```python
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
```

**How it handles:**

- ✅ Handles PIL image loading errors
- ✅ Handles RGBA to RGB conversion
- ✅ Catches array conversion errors
- ✅ Gracefully handles optional file (doesn't break pipeline)
- ✅ Clears state on error

**Error Types Handled:**

- PIL.UnidentifiedImageError (unsupported format)
- ValueError (invalid image data)
- IndexError (wrong array dimensions)

---

### 1.4 Inference Pipeline Errors

**Scenario:** Errors during model inference or metric calculation

**Location:** Lines 483-705

**Code Snippet:**

```python
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

                # Calculate dent metrics
                metrics = calculate_dent_metrics(
                    st.session_state.depth_map,
                    binary_mask,
                    pixel_to_cm=pixel_to_cm
                )

                # ... rest of processing ...

        except Exception as e:
            st.error(f"❌ Error during inference: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
```

**How it handles:**

- ✅ Pre-validates prerequisites (model loaded, depth map exists)
- ✅ Comprehensive try-except around entire inference pipeline
- ✅ Displays full traceback for debugging
- ✅ Prevents partial state corruption
- ✅ User-friendly error messages

**Error Types Handled:**

- CUDA out of memory errors
- Model forward pass errors
- Metric calculation errors
- Dimension mismatch errors

---

### 1.5 Overlay Creation Errors

**Scenario:** Errors when creating RGB overlay visualization

**Location:** Lines 563-586

**Code Snippet:**

```python
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
```

**How it handles:**

- ✅ Isolated error handling (doesn't break main inference)
- ✅ Uses warning instead of error (overlay is optional)
- ✅ Graceful degradation (continues without overlay)
- ✅ Clears overlay state on error

**Error Types Handled:**

- Dimension mismatch (RGB vs mask)
- OpenCV contour detection errors
- Memory errors during blending

---

## 2. Model Architecture (model_architecture.py)

### 2.1 Depth Preprocessing Errors

**Scenario:** Invalid depth maps (NaN, Inf, zero values, constant values)

**Location:** Lines 113-160 (`preprocess_depth` function)

**Code Snippet:**

```python
def preprocess_depth(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)

    # 1. Normalize depth (0-1)
    valid = np.isfinite(depth) & (depth > 0)
    if np.any(valid):
        dmin, dmax = float(depth[valid].min()), float(depth[valid].max())
        denom = dmax - dmin
        if denom < 1e-8:
            denom = 1e-8  # Prevent division by zero
        depth_n = (depth - dmin) / denom
    else:
        depth_n = np.zeros_like(depth, dtype=np.float32)  # Fallback for invalid data

    # 2. Compute gradients
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)

    # 3. Robust normalization for gradients
    def robust_norm(x):
        v = x.flatten()
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.zeros_like(x, dtype=np.float32)  # Fallback
        med = np.median(v)
        mad = np.median(np.abs(v - med)) + 1e-6  # Prevent division by zero
        out = (x - med) / (3.0 * mad)
        out = np.clip(out, -3.0, 3.0)
        out = (out - out.min()) / (out.max() - out.min() + 1e-9)  # Prevent division by zero
        return out.astype(np.float32)

    gx_n = robust_norm(gx)
    gy_n = robust_norm(gy)

    # 4. Stack channels: (3, H, W)
    inp = np.stack([depth_n, gx_n, gy_n], axis=0).astype(np.float32)

    return inp
```

**How it handles:**

- ✅ Checks for finite values (`np.isfinite`)
- ✅ Handles zero/negative depth values
- ✅ Prevents division by zero (multiple checks)
- ✅ Fallback to zeros for completely invalid data
- ✅ Robust normalization using median/MAD (resistant to outliers)
- ✅ Clipping to prevent extreme values

**Error Types Handled:**

- NaN/Inf values in depth maps
- Constant depth maps (no variation)
- Zero or negative depth values
- Division by zero errors
- Outlier values

---

### 2.2 Metric Calculation Validation

**Scenario:** Invalid calibration parameters or missing data

**Location:** Lines 245-385 (`calculate_dent_metrics` function)

**Code Snippet:**

```python
def calculate_dent_metrics(depth_map: np.ndarray, dent_mask: np.ndarray,
                           pixel_to_cm: float = None,
                           depth_units: str = 'meters') -> dict:
    missing_info = []

    # Validate pixel_to_cm for area computation
    area_valid = False
    if pixel_to_cm is None:
        missing_info.append("pixel_to_cm conversion factor (required for area computation)")
    elif pixel_to_cm <= 0:
        missing_info.append("pixel_to_cm must be positive (invalid value provided)")
    elif pixel_to_cm > 10:  # Sanity check
        missing_info.append(f"pixel_to_cm value ({pixel_to_cm}) seems unusually large - please verify calibration")
        area_valid = False
    else:
        area_valid = True

    # Ensure masks match dimensions
    if depth_map.shape != dent_mask.shape:
        h, w = depth_map.shape
        dent_mask = cv2.resize(dent_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Handle empty dent regions
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

    # Handle unknown depth units
    depth_conversion_factors = {
        'meters': 1000.0,
        'mm': 1.0,
        'cm': 10.0,
        'inches': 25.4
    }

    if depth_units not in depth_conversion_factors:
        missing_info.append(f"Unknown depth_units '{depth_units}'. Assuming meters.")
        conversion_factor = 1000.0
    else:
        conversion_factor = depth_conversion_factors[depth_units]

    # ... rest of calculation ...
```

**How it handles:**

- ✅ Validates calibration parameters (pixel_to_cm)
- ✅ Sanity checks for unreasonable values
- ✅ Handles dimension mismatches (auto-resize)
- ✅ Handles empty dent regions gracefully
- ✅ Validates depth values (finite, positive)
- ✅ Handles unknown depth units with fallback
- ✅ Returns informative `missing_info` list
- ✅ Refuses invalid area computation (sets to None)

**Error Types Handled:**

- Invalid calibration parameters
- Dimension mismatches
- Empty dent regions
- Invalid depth values
- Unknown unit specifications

---

### 2.3 Overlay Creation Errors

**Scenario:** Dimension mismatches between RGB and mask

**Location:** Lines 195-242 (`create_dent_overlay` function)

**Code Snippet:**

```python
def create_dent_overlay(rgb_image: np.ndarray, dent_mask: np.ndarray,
                       overlay_alpha: float = 0.2, outline_thickness: int = 2) -> np.ndarray:
    # Ensure mask and RGB image have matching dimensions
    if rgb_image.shape[:2] != dent_mask.shape[:2]:
        h, w = rgb_image.shape[:2]
        dent_mask = cv2.resize(dent_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Convert RGB to float for blending
    background = rgb_image.astype(np.float32) / 255.0

    # Create binary mask (0 or 1) from dent_mask
    mask_binary = (dent_mask > 127).astype(np.float32)

    # ... overlay creation ...

    # Find contours of the dent mask
    mask_uint8 = (mask_binary * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw red outline on the result
    result_uint8 = (result * 255).astype(np.uint8)
    cv2.drawContours(result_uint8, contours, -1, (255, 0, 0), outline_thickness)

    return result_uint8
```

**How it handles:**

- ✅ Auto-resizes mask to match RGB dimensions
- ✅ Handles different data types (uint8, float32)
- ✅ Safe contour detection (handles empty masks)
- ✅ Type conversions with proper casting

**Error Types Handled:**

- Dimension mismatches
- Type mismatches
- Empty masks (no contours)

---

## 3. Training Pipeline (Notebook)

### 3.1 Data Loading Errors

**Scenario:** Corrupted or missing mask files during training

**Location:** Cell 5, Lines 176-177

**Code Snippet:**

```python
def _load_mask(self, path):
    p = Path(path)
    if p.suffix.lower() == ".npy":
        m = np.load(path)
        m = (m > 0).astype(np.uint8)
    else:
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Cannot read mask: {path}")
        m = (m > 127).astype(np.uint8)
    return m
```

**How it handles:**

- ✅ Checks if file read was successful
- ✅ Raises descriptive RuntimeError with file path
- ✅ Handles both .npy and .png formats
- ✅ Provides clear error message for debugging

**Error Types Handled:**

- FileNotFoundError (implicitly via cv2.imread returning None)
- Corrupted image files
- Unsupported file formats

---

### 3.2 Normalization Errors

**Scenario:** Invalid depth maps (constant values, all zeros, all NaN)

**Location:** Cell 5, `_normalize` method

**Code Snippet:**

```python
def _normalize(self, depth):
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros_like(depth, dtype=np.float32)  # Fallback
    d = depth.copy()
    d_min = float(np.min(d[valid]))
    d_max = float(np.max(d[valid]))
    if d_max - d_min < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)  # Handle constant values
    norm = (d - d_min) / (d_max - d_min)
    norm[~valid] = 0.0
    return norm.astype(np.float32)
```

**How it handles:**

- ✅ Checks for valid (finite, positive) values
- ✅ Handles constant depth maps (no variation)
- ✅ Prevents division by zero
- ✅ Fallback to zeros for invalid data
- ✅ Masks invalid pixels to zero

**Error Types Handled:**

- All NaN/Inf values
- Constant depth maps
- Zero or negative values
- Division by zero

---

### 3.3 Gradient Computation Errors

**Scenario:** Invalid gradients (all zeros, NaN values)

**Location:** Cell 5, `_gradients` method

**Code Snippet:**

```python
def _gradients(self, depth):
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    # robust normalization
    def robust_norm(x):
        v = x.flatten()
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.zeros_like(x, dtype=np.float32)  # Fallback
        med = np.median(v)
        mad = np.median(np.abs(v - med)) + 1e-6  # Prevent division by zero
        out = (x - med) / (3.0 * mad)
        out = np.clip(out, -3.0, 3.0)
        out = (out - out.min()) / (out.max() - out.min() + 1e-9)  # Prevent division by zero
        return out.astype(np.float32)
    return robust_norm(gx), robust_norm(gy)
```

**How it handles:**

- ✅ Filters out NaN/Inf values before normalization
- ✅ Uses robust statistics (median, MAD) resistant to outliers
- ✅ Multiple division-by-zero checks
- ✅ Clipping to prevent extreme values
- ✅ Fallback to zeros for empty arrays

**Error Types Handled:**

- NaN/Inf in gradients
- Empty gradient arrays
- Division by zero
- Extreme outlier values

---

## Summary Table

| Component             | Error Scenario        | Handling Method               | User Impact                      |
| --------------------- | --------------------- | ----------------------------- | -------------------------------- |
| **Model Loading**     | Corrupted model file  | Try-except with error message | ❌ Clear error, prevents crash   |
| **Model Loading**     | Missing model file    | Existence check before load   | ❌ Clear error message           |
| **Depth Map Loading** | Invalid .npy file     | Try-except, clear state       | ❌ Error message, no crash       |
| **RGB Image Loading** | Corrupted image       | Try-except, optional handling | ⚠️ Warning, continues            |
| **Inference**         | Missing prerequisites | Pre-validation checks         | ❌ Clear error before processing |
| **Inference**         | Runtime errors        | Try-except with traceback     | ❌ Full error details            |
| **Overlay Creation**  | Dimension mismatch    | Auto-resize                   | ✅ Automatic fix                 |
| **Overlay Creation**  | Creation failure      | Warning, graceful degradation | ⚠️ Continues without overlay     |
| **Preprocessing**     | Invalid depth values  | NaN/Inf filtering             | ✅ Robust handling               |
| **Preprocessing**     | Constant depth maps   | Division-by-zero prevention   | ✅ Fallback to zeros             |
| **Metrics**           | Invalid calibration   | Validation + warnings         | ⚠️ Refuses invalid area          |
| **Metrics**           | Dimension mismatch    | Auto-resize                   | ✅ Automatic fix                 |
| **Metrics**           | Empty dent regions    | Early return with zeros       | ✅ Safe handling                 |
| **Training**          | Missing mask files    | RuntimeError with path        | ❌ Stops training, clear error   |
| **Training**          | Invalid depth maps    | Fallback normalization        | ✅ Continues training            |

---

## Key Error Handling Strategies

### 1. **Defensive Programming**

- ✅ Pre-validation of inputs
- ✅ Existence checks before file operations
- ✅ Dimension validation and auto-correction

### 2. **Graceful Degradation**

- ✅ Optional features fail gracefully (overlay)
- ✅ Fallback values for invalid data
- ✅ Continues operation when possible

### 3. **User-Friendly Messages**

- ✅ Clear error messages with emojis (❌, ⚠️, ✅)
- ✅ Actionable guidance ("Please upload...")
- ✅ Detailed tracebacks for debugging

### 4. **State Management**

- ✅ Clears invalid state on errors
- ✅ Prevents partial/corrupted state
- ✅ Validates prerequisites before operations

### 5. **Robust Data Processing**

- ✅ Handles edge cases (NaN, Inf, zeros)
- ✅ Prevents division by zero
- ✅ Uses robust statistics (median, MAD)

---

## Recommendations for Slides

### Slide 1: Overview

- Show error handling coverage across 3 main components
- Highlight defensive programming approach

### Slide 2: UI Error Handling

- Model loading errors (with code snippet)
- File upload errors (depth map, RGB)
- Inference pipeline errors

### Slide 3: Model Architecture Error Handling

- Preprocessing robustness (NaN, Inf handling)
- Metric validation (calibration checks)
- Dimension mismatch handling

### Slide 4: Training Pipeline Error Handling

- Data loading errors
- Normalization robustness
- Gradient computation safety

### Slide 5: Error Handling Strategies

- Defensive programming
- Graceful degradation
- User-friendly messages
- State management
- Robust data processing

### Slide 6: Summary Table

- Visual table showing all error scenarios
- Color-coded by severity (❌ Error, ⚠️ Warning, ✅ Auto-fix)





