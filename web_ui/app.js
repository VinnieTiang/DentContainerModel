// API Base URL
const API_BASE_URL = 'http://localhost:5009/api';
const deleteButtonStyle = document.createElement('style');
deleteButtonStyle.textContent = `
    .file-pair-container {
        position: relative; /* Needed for absolute positioning of the X button */
        border: 1px solid #eee;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 8px;
        background: #f9f9f9;
        transition: background 0.2s;
    }

    .file-pair-container:hover {
        background: #f0f0f0;
    }

    .delete-file-btn {
        position: absolute;
        top: 5px;
        right: 5px;
        background: #ff4444;
        color: white;
        border: none;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        font-size: 12px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        opacity: 0.7;
        transition: all 0.2s;
        z-index: 10;
    }

    .delete-file-btn:hover {
        opacity: 1;
        transform: scale(1.1);
        background: #cc0000;
    }
`;
document.head.appendChild(deleteButtonStyle);

// Global state
let currentContainerId = null;
let modelLoaded = false;
let containers = [];
let currentModalPanel = null;
let modalFiles = {
    depth: null,
    rgb: null
};
let modalPreviewData = null;
let intrinsicsFile = null;
let modelFile = null;
// Store multiple file pairs per panel (client-side)
let panelFilePairs = {
    back: [],
    left: [],
    right: [],
    roof: [],
    door: []
};

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    // Setup event listeners
    setupCreateContainerPage();
    setupProcessingPage();
    setupModal();
    setupHistoryModal();
    setupIntrinsicsUpload();
    setupModelUpload();
    setupUnifiedUpload();
    loadContainers();

    // Check API health
    checkAPIHealth();
}

// API Health Check
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        console.log('API Health:', data);
    } catch (error) {
        console.error('API Health Check Failed:', error);
        showNotification('API server is not running. Please start the Flask server.', 'error');
    }
}

// Create Container Page Setup
function setupCreateContainerPage() {
    const createBtn = document.getElementById('create-container-btn');
    const containerNameInput = document.getElementById('container-name');

    createBtn.addEventListener('click', async () => {
        const containerName = containerNameInput.value.trim() || `CONTAINER-${Date.now()}`;
        await createContainer(containerName);
    });

    containerNameInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            createBtn.click();
        }
    });
}

// Processing Page Setup
function setupProcessingPage() {
    // Back button
    document.getElementById('back-to-create-btn').addEventListener('click', () => {
        showPage('create-container-page');
        loadContainers();
    });

    // Model loading
    document.getElementById('load-model-btn').addEventListener('click', loadModel);

    // Threshold slider
    const thresholdSlider = document.getElementById('threshold');
    const thresholdValue = document.getElementById('threshold-value');
    thresholdSlider.addEventListener('input', (e) => {
        thresholdValue.textContent = e.target.value;
    });

    // Setup upload buttons
    document.querySelectorAll('.btn-upload').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const panelName = e.target.dataset.panel;
            openUploadModal(panelName);
        });
    });

    // Setup refresh buttons
    document.querySelectorAll('.btn-refresh').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const panelName = e.target.dataset.panel;
            await refreshPanelFiles(panelName);
        });
    });

    // Setup process buttons
    document.querySelectorAll('.btn-process').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const panelName = e.target.dataset.panel;
            await processPanel(panelName);
        });
    });

    // Setup Process All button
    const processAllBtn = document.getElementById('process-all-btn');
    if (processAllBtn) {
        console.log('Process All button found, attaching event listener');
        processAllBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('Process All button clicked');
            await processAllPanels();
        });
    } else {
        console.warn('Process All button not found during setup');
    }

    // Close viewer button
    document.getElementById('close-viewer').addEventListener('click', () => {
        closeResultsViewer();
    });
}

// Modal Setup
function setupModal() {
    const modal = document.getElementById('upload-modal');
    const closeModalBtn = document.getElementById('close-modal');
    const cancelBtn = document.getElementById('cancel-upload');
    const confirmBtn = document.getElementById('confirm-upload');
    const ransacCheckbox = document.getElementById('modal-ransac-checkbox');

    // Close modal
    closeModalBtn.addEventListener('click', closeModal);
    cancelBtn.addEventListener('click', closeModal);

    // Click outside to close
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeModal();
        }
    });

    // Confirm upload
    confirmBtn.addEventListener('click', async () => {
        await confirmUpload();
    });

    // RANSAC checkbox
    ransacCheckbox.addEventListener('change', async (e) => {
        const rectangularMaskLabel = document.getElementById('rectangular-mask-checkbox-label');
        if (e.target.checked) {
            // Show rectangular mask checkbox when RANSAC is enabled
            if (rectangularMaskLabel) {
                rectangularMaskLabel.style.display = 'block';
            }

            // --- BUG FIX HERE ---
            // Old code: if (modalFiles.depth && currentContainerId)
            // New code: Check for EITHER new file object OR existing filename
            const hasFile = modalFiles.depth || modalFiles.depthFilename;

            if (hasFile && currentContainerId) {
                await showRANSACPreview();
            }
            // --------------------
        } else {
            // Hide rectangular mask checkbox when RANSAC is disabled
            if (rectangularMaskLabel) {
                rectangularMaskLabel.style.display = 'none';
            }
            document.getElementById('ransac-preview').style.display = 'none';
        }
    });

    // Rectangular mask checkbox - update preview immediately when changed
    const rectangularMaskCheckbox = document.getElementById('modal-rectangular-mask-checkbox');
    if (rectangularMaskCheckbox) {
        rectangularMaskCheckbox.addEventListener('change', async (e) => {
            // Only update preview if RANSAC is enabled and we have depth data
            const ransacCheckbox = document.getElementById('modal-ransac-checkbox');

            // --- BUG FIX HERE AS WELL ---
            const hasFile = modalFiles.depth || modalFiles.depthFilename;

            if (ransacCheckbox && ransacCheckbox.checked && hasFile && currentContainerId) {
                await showRANSACPreview();
            }
            // ----------------------------
        });
    }

    // Preview toggle buttons
    const toggleDepthBtn = document.getElementById('toggle-depth');
    const toggleRgbBtn = document.getElementById('toggle-rgb');

    toggleDepthBtn.addEventListener('click', async () => {
        toggleDepthBtn.classList.add('active');
        toggleRgbBtn.classList.remove('active');
        // Pass the current ID if we are looking at an existing file
        await showDepthPreview(currentModalUploadId);
    });

    toggleRgbBtn.addEventListener('click', async () => {
        toggleRgbBtn.classList.add('active');
        toggleDepthBtn.classList.remove('active');
        await showRGBPreview();
    });

    // Setup modal file inputs
    setupModalFileInputs();
}
// Setup Modal File Inputs
function setupModalFileInputs() {
    const uploadAreas = document.querySelectorAll('.upload-area-modal');

    uploadAreas.forEach(area => {
        const fileInput = area.querySelector('.file-input-modal');
        const fileType = area.dataset.type;

        // Click to browse
        area.addEventListener('click', () => {
            fileInput.click();
        });

        // File selected
        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (file) {
                await handleModalFileSelect(file, fileType);
            }
        });

        // Drag and drop
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            area.addEventListener(eventName, preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            area.addEventListener(eventName, () => {
                area.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            area.addEventListener(eventName, () => {
                area.classList.remove('dragover');
            }, false);
        });

        area.addEventListener('drop', async (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                await handleModalFileSelect(files[0], fileType);
            }
        });
    });
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Open Upload Modal
function openUploadModal(panelName) {
    currentModalPanel = panelName;
    modalFiles = { depth: null, rgb: null, depthFilename: null, rgbFilename: null };
    modalPreviewData = null;

    const modal = document.getElementById('upload-modal');
    const modalTitle = document.getElementById('modal-title');
    modalTitle.textContent = `Upload Files - ${panelName.charAt(0).toUpperCase() + panelName.slice(1)} Panel`;

    // Reset modal
    document.getElementById('preview-section').style.display = 'none';
    document.getElementById('preview-placeholder').style.display = 'flex';
    document.getElementById('preview-toggle-buttons').style.display = 'none';
    document.getElementById('ransac-preview').style.display = 'none';
    document.getElementById('ransac-checkbox-section').style.display = 'none';
    const confirmBtn = document.getElementById('confirm-upload');
    confirmBtn.disabled = true;
    confirmBtn.textContent = 'Upload';
    document.getElementById('modal-ransac-checkbox').checked = true;
    const rectangularMaskLabel = document.getElementById('rectangular-mask-checkbox-label');
    if (rectangularMaskLabel) {
        rectangularMaskLabel.style.display = 'none';
    }
    const rectangularMaskCheckbox = document.getElementById('modal-rectangular-mask-checkbox');
    if (rectangularMaskCheckbox) {
        rectangularMaskCheckbox.checked = false; // Reset to default (unchecked)
    }

    // Reset toggle buttons
    document.getElementById('toggle-depth').classList.add('active');
    document.getElementById('toggle-rgb').classList.remove('active');

    // Reset status message
    const statusMessage = document.getElementById('status-message');
    statusMessage.textContent = 'Please upload a depth map (.npy file) to proceed';
    statusMessage.className = 'waiting';

    // Reset file inputs
    document.querySelectorAll('.file-input-modal').forEach(input => {
        input.value = '';
    });

    // Reset upload areas
    document.querySelectorAll('.upload-area-modal').forEach(area => {
        area.classList.remove('has-file');
        area.querySelector('.upload-placeholder').innerHTML = `
            <span class="upload-icon">${area.dataset.type === 'depth' ? '📁' : '🖼️'}</span>
            <p>Drag & drop ${area.dataset.type === 'depth' ? 'depth map (.npy)' : 'RGB image (optional)'}</p>
            <p class="upload-hint">or click to browse</p>
        `;
    });

    modal.classList.add('active');
}

// Close Modal
function closeModal() {
    const modal = document.getElementById('upload-modal');
    modal.classList.remove('active');
    currentModalPanel = null;
    modalFiles = { depth: null, rgb: null, depthFilename: null, rgbFilename: null };
    modalPreviewData = null;
}

// Handle Modal File Select
async function handleModalFileSelect(file, fileType) {
    const uploadArea = document.querySelector(`.upload-area-modal[data-type="${fileType}"]`);

    // Validate file type
    if (fileType === 'depth' && !file.name.endsWith('.npy')) {
        showNotification('Please upload a .npy file for depth map', 'error');
        return;
    }

    if (fileType === 'rgb' && !file.type.startsWith('image/')) {
        showNotification('Please upload an image file for RGB', 'error');
        return;
    }

    modalFiles[fileType] = file;
    modalFiles[`${fileType}Filename`] = file.name; // Store filename

    uploadArea.classList.add('has-file');
    uploadArea.querySelector('.upload-placeholder').innerHTML = `
        <span class="upload-icon">✅</span>
        <p class="file-name">${file.name}</p>
        <p class="upload-hint">File selected - Click to change</p>
    `;

    // If depth file selected, show preview and enable RANSAC checkbox
    if (fileType === 'depth') {
        await showDepthPreview();
        // Show RANSAC checkbox section
        document.getElementById('ransac-checkbox-section').style.display = 'block';
        const ransacCheckbox = document.getElementById('modal-ransac-checkbox');
        if (ransacCheckbox && ransacCheckbox.checked) {
            // Show the spinner and start extraction immediately
            await showRANSACPreview();
        }

        // Show toggle buttons if RGB is also available
        const hasRGB = modalFiles.rgb instanceof File || modalFiles.rgbFilename;
        if (hasRGB) {
            document.getElementById('preview-toggle-buttons').style.display = 'flex';
            document.getElementById('toggle-depth').classList.add('active');
            document.getElementById('toggle-rgb').classList.remove('active');
        }
    }

    // If RGB file selected and depth already exists, show toggle buttons
    if (fileType === 'rgb') {
        const hasDepth = modalFiles.depth instanceof File || modalFiles.depthFilename;
        if (hasDepth) {
            document.getElementById('preview-toggle-buttons').style.display = 'flex';
            document.getElementById('toggle-depth').classList.add('active');
            document.getElementById('toggle-rgb').classList.remove('active');
        }
    }

    // Check if ready
    checkReadyStatus();
}

// Show Depth Preview
async function showDepthPreview(specificUploadId = null) {
    if (!currentContainerId || !currentModalPanel) return;

    // Check if we have a File object (new upload) or just filename (existing file)
    const hasNewFile = modalFiles.depth instanceof File;
    const hasExistingFile = modalFiles.depthFilename && !hasNewFile;

    if (!hasNewFile && !hasExistingFile) return;

    const previewSection = document.getElementById('preview-section');
    const previewDiv = document.getElementById('modal-preview');
    const previewPlaceholder = document.getElementById('preview-placeholder');

    // Show loading state
    previewPlaceholder.style.display = 'none';
    previewSection.style.display = 'flex';
    previewDiv.innerHTML = '<div class="spinner"></div><p>Loading preview...</p>';

    try {
        let data = null;

        // ---------------------------------------------------------
        // SCENARIO 1: NEW FILE (User just dropped a file in modal)
        // ---------------------------------------------------------
        if (hasNewFile) {
            // 1. Validate Container
            if (!currentContainerId) {
                showNotification('Please create or select a container first', 'error');
                previewDiv.innerHTML = '<div class="preview-error">No container selected</div>';
                return;
            }

            // 2. Upload Temporary File for Preview
            const formData = new FormData();
            formData.append('depth_file', modalFiles.depth);
            if (modalFiles.rgb) {
                formData.append('rgb_file', modalFiles.rgb);
            }

            const response = await fetch(`${API_BASE_URL}/containers/${currentContainerId}/panels/${currentModalPanel}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.status}`);
            }

            data = await response.json();

            if (data.success) {
                modalPreviewData = data;
                // Update UI filenames if needed
                if (data.depth_filename) modalFiles.depthFilename = data.depth_filename;
                if (data.rgb_filename) modalFiles.rgbFilename = data.rgb_filename;

                // CRITICAL: Use the ID of the file we JUST uploaded
                if (data.upload_id) {
                    currentModalUploadId = data.upload_id;
                    specificUploadId = data.upload_id;
                }
            }
        }
        // ---------------------------------------------------------
        // SCENARIO 2: EXISTING FILE (User clicked a list item)
        // ---------------------------------------------------------
        else {
            // Fetch container data to get metadata (like is_uint16) for the specific file
            const containerResponse = await fetch(`${API_BASE_URL}/containers/${currentContainerId}`);
            const containerData = await containerResponse.json();

            if (containerData.success) {
                const panelData = containerData.container.panels[currentModalPanel];

                if (panelData) {
                    let isUint16 = false;

                    // A. If we have a specific ID, look up its specific metadata
                    if (specificUploadId && panelData.uploads) {
                        const specificUpload = panelData.uploads.find(u => u.upload_id === specificUploadId);
                        if (specificUpload) {
                            isUint16 = specificUpload.is_uint16;
                        }
                    }
                    // B. Fallback to latest
                    else {
                        isUint16 = panelData.is_uint16 || false;
                    }

                    data = {
                        success: true,
                        depth_filename: modalFiles.depthFilename,
                        rgb_filename: modalFiles.rgbFilename,
                        is_uint16: isUint16
                    };
                    modalPreviewData = data;
                }
            }
        }

        // ---------------------------------------------------------
        // FETCH AND DISPLAY PREVIEW IMAGES
        // ---------------------------------------------------------
        if (data && data.success) {
            // Construct URL
            let url = `${API_BASE_URL}/containers/${currentContainerId}/panels/${currentModalPanel}/depth/preview`;

            // Append specific ID if we have one (from click OR new upload)
            if (specificUploadId) {
                url += `?upload_id=${specificUploadId}`;
            }

            const previewResponse = await fetch(url);
            const previewData = await previewResponse.json();

            if (previewData.success && previewData.previews) {
                let html = '';

                // Show Raw Depth (for uint16)
                if (previewData.previews.raw_depth) {
                    html += `
                        <div class="preview-image-container">
                            <img src="data:image/png;base64,${previewData.previews.raw_depth}" alt="Raw Depth">
                            <p class="preview-caption">Raw Depth Map (Red box = crop area)</p>
                        </div>
                    `;
                }

                // Show Processed Depth
                if (previewData.previews.processed_depth) {
                    const caption = (modalPreviewData && modalPreviewData.is_uint16)
                        ? 'Preprocessed Depth Map'
                        : 'Input Depth Map';

                    html += `
                        <div class="preview-image-container">
                            <img src="data:image/png;base64,${previewData.previews.processed_depth}" alt="Processed Depth">
                            <p class="preview-caption">${caption}</p>
                        </div>
                    `;
                }

                previewDiv.innerHTML = html;

                // Toggle Button Logic
                const hasRGB = modalFiles.rgb instanceof File || modalFiles.rgbFilename;
                if (hasRGB) {
                    document.getElementById('preview-toggle-buttons').style.display = 'flex';
                    document.getElementById('toggle-depth').classList.add('active');
                    document.getElementById('toggle-rgb').classList.remove('active');
                }
            }
        }
    } catch (error) {
        console.error('Error loading preview:', error);
        previewDiv.innerHTML = '<p class="error-message">Error loading preview</p>';
    }
}

// // Show RGB Preview
// async function showRGBPreview() {
//     if (!currentContainerId || !currentModalPanel) return;

//     // Check if we have RGB file
//     const hasNewRGB = modalFiles.rgb instanceof File;
//     const hasExistingRGB = modalFiles.rgbFilename && !hasNewRGB;

//     if (!hasNewRGB && !hasExistingRGB) return;

//     const previewSection = document.getElementById('preview-section');
//     const previewDiv = document.getElementById('modal-preview');
//     const previewPlaceholder = document.getElementById('preview-placeholder');

//     previewPlaceholder.style.display = 'none';
//     previewSection.style.display = 'flex';
//     previewDiv.innerHTML = '<div class="spinner"></div><p>Loading RGB preview...</p>';

//     try {
//         // ============================================================
//         // STEP 1: DETERMINE IF WE ARE IN A UINT16 (CROP) SCENARIO
//         // ============================================================
//         let isUint16 = false;

//         // A. Check recent upload data (fastest)
//         if (modalPreviewData && modalPreviewData.is_uint16 !== undefined) {
//             isUint16 = modalPreviewData.is_uint16;
//         }
//         // B. Check server data (if adding RGB to existing depth)
//         else {
//             try {
//                 const containerResponse = await fetch(`${API_BASE_URL}/containers/${currentContainerId}`);
//                 const containerData = await containerResponse.json();
//                 if (containerData.success) {
//                     const panelData = containerData.container.panels[currentModalPanel];
//                     if (panelData) {
//                         if (currentModalUploadId && panelData.uploads) {
//                             const specificUpload = panelData.uploads.find(u => u.upload_id === currentModalUploadId);
//                             if (specificUpload) isUint16 = specificUpload.is_uint16;
//                             else isUint16 = panelData.is_uint16 || false;
//                         } else {
//                             isUint16 = panelData.is_uint16 || false;
//                         }
//                     }
//                 }
//             } catch (error) {
//                 console.error('Error fetching container data:', error);
//             }
//         }

//         // ============================================================
//         // STEP 2: HANDLE NEW FILE UPLOAD (STAGING PHASE)
//         // ============================================================
//         if (hasNewRGB) {

//             // SCENARIO A: Standard Image (Show Local Preview Instant)
//             if (!isUint16) {
//                 const reader = new FileReader();
//                 const imageData = await new Promise((resolve, reject) => {
//                     reader.onload = (e) => resolve(e.target.result);
//                     reader.onerror = reject;
//                     reader.readAsDataURL(modalFiles.rgb);
//                 });

//                 previewDiv.innerHTML = `
//                     <div class="preview-image-container">
//                         <img src="${imageData}" alt="RGB Image" style="max-width: 100%; height: auto;">
//                         <p class="preview-caption">RGB Image (Local Preview)</p>
//                     </div>
//                 `;
//                 return; // Done
//             }

//             // SCENARIO B: Uint16 Image (MUST Upload to get Crops)
//             if (isUint16) {
//                 const formData = new FormData();
//                 formData.append('rgb_file', modalFiles.rgb);

//                 // --- ✅ FIX START: Send Depth File + Flags to prevent 400 Error ---

//                 // 1. If we have the depth file locally (Staging phase), send it!
//                 // This satisfies servers that demand 'depth_file' in the request.
//                 if (modalFiles.depth instanceof File) {
//                     formData.append('depth_file', modalFiles.depth);
//                 }

//                 // 2. Always link to the existing ID
//                 if (currentModalUploadId) {
//                     formData.append('upload_id', currentModalUploadId);
//                 }

//                 // 3. Send all flags (mimic the main upload function)
//                 const ransacCheckbox = document.getElementById('modal-ransac-checkbox');
//                 const rectCheckbox = document.getElementById('modal-rectangular-mask-checkbox');

//                 formData.append('use_ransac', ransacCheckbox ? ransacCheckbox.checked : true);
//                 formData.append('force_rectangular_mask', rectCheckbox ? rectCheckbox.checked : true);
//                 // -----------------------------------------------------------------

//                 let url = `${API_BASE_URL}/containers/${currentContainerId}/panels/${currentModalPanel}/upload`;

//                 const response = await fetch(url, {
//                     method: 'POST',
//                     body: formData
//                 });
//                 const data = await response.json();

//                 // If upload successful, we can now ask for the generated previews
//                 if (data.success) {
//                     // Update global ID if we just created/updated one
//                     if (data.upload_id) currentModalUploadId = data.upload_id;

//                     // Now fetch the previews (which will include the crops we just made)
//                     const previewResponse = await fetch(`${API_BASE_URL}/containers/${currentContainerId}/panels/${currentModalPanel}/depth/preview?upload_id=${currentModalUploadId}`);
//                     const previewData = await previewResponse.json();

//                     if (previewData.success && previewData.previews) {
//                         let html = '';
//                         if (previewData.previews.rgb_original) {
//                             html += `
//                                 <div class="preview-image-container">
//                                     <img src="data:image/png;base64,${previewData.previews.rgb_original}" alt="RGB Original">
//                                     <p class="preview-caption">RGB Image (Original) - Red box indicates crop area</p>
//                                 </div>`;
//                         }
//                         if (previewData.previews.rgb_cropped) {
//                             html += `
//                                 <div class="preview-image-container">
//                                     <img src="data:image/png;base64,${previewData.previews.rgb_cropped}" alt="RGB Cropped">
//                                     <p class="preview-caption">RGB Image (Cropped)</p>
//                                 </div>`;
//                         }
//                         previewDiv.innerHTML = html;
//                         return; // Done
//                     }
//                 }
//             }
//         }

//         // ============================================================
//         // STEP 3: HANDLE EXISTING FILE (SERVER VIEW PHASE)
//         // ============================================================

//         // If isUint16, fetch Dual Previews
//         if (isUint16) {
//             try {
//                 let url = `${API_BASE_URL}/containers/${currentContainerId}/panels/${currentModalPanel}/depth/preview`;
//                 if (currentModalUploadId) url += `?upload_id=${currentModalUploadId}`;

//                 const previewResponse = await fetch(url);
//                 const previewData = await previewResponse.json();

//                 if (previewData.success && previewData.previews) {
//                     let html = '';
//                     if (previewData.previews.rgb_original) {
//                         html += `
//                             <div class="preview-image-container">
//                                 <img src="data:image/png;base64,${previewData.previews.rgb_original}" alt="RGB Original">
//                                 <p class="preview-caption">RGB Image (Original) - Red box indicates crop area</p>
//                             </div>`;
//                     }
//                     if (previewData.previews.rgb_cropped) {
//                         html += `
//                             <div class="preview-image-container">
//                                 <img src="data:image/png;base64,${previewData.previews.rgb_cropped}" alt="RGB Cropped">
//                                 <p class="preview-caption">RGB Image (Cropped)</p>
//                             </div>`;
//                     }
//                     if (html) {
//                         previewDiv.innerHTML = html;
//                         return;
//                     }
//                 }
//             } catch (error) {
//                 console.error('Error fetching uint16 previews:', error);
//             }
//         }

//         // Standard RGB preview (Fallback)
//         let rgbImageData = null;
//         try {
//             let url = `${API_BASE_URL}/containers/${currentContainerId}/panels/${currentModalPanel}/rgb`;
//             if (currentModalUploadId) url += `?upload_id=${currentModalUploadId}`;

//             const rgbResponse = await fetch(url);
//             if (rgbResponse.ok) {
//                 const blob = await rgbResponse.blob();
//                 rgbImageData = await new Promise((resolve, reject) => {
//                     const reader = new FileReader();
//                     reader.onload = (e) => resolve(e.target.result);
//                     reader.onerror = reject;
//                     reader.readAsDataURL(blob);
//                 });
//             }
//         } catch (error) {
//             console.error('Error fetching RGB from server:', error);
//         }

//         if (rgbImageData) {
//             previewDiv.innerHTML = `
//                 <div class="preview-image-container">
//                     <img src="${rgbImageData}" alt="RGB Image" style="max-width: 100%; height: auto;">
//                     <p class="preview-caption">RGB Image</p>
//                 </div>`;
//         } else {
//             previewDiv.innerHTML = '<p class="error-message">RGB image not available</p>';
//         }

//     } catch (error) {
//         console.error('Error loading RGB preview:', error);
//         previewDiv.innerHTML = '<p class="error-message">Error loading RGB preview</p>';
//     }
// }

// Show RGB Preview (Simplified - No Cropping)
async function showRGBPreview() {
    if (!currentContainerId || !currentModalPanel) return;

    // Check if we have RGB file
    const hasNewRGB = modalFiles.rgb instanceof File;
    const hasExistingRGB = modalFiles.rgbFilename && !hasNewRGB;

    if (!hasNewRGB && !hasExistingRGB) return;

    const previewSection = document.getElementById('preview-section');
    const previewDiv = document.getElementById('modal-preview');
    const previewPlaceholder = document.getElementById('preview-placeholder');

    previewPlaceholder.style.display = 'none';
    previewSection.style.display = 'flex';
    previewDiv.innerHTML = '<div class="spinner"></div><p>Loading RGB preview...</p>';

    try {
        // SCENARIO 1: NEW FILE (Show Local Preview Instantly)
        if (hasNewRGB) {
            const reader = new FileReader();
            const imageData = await new Promise((resolve, reject) => {
                reader.onload = (e) => resolve(e.target.result);
                reader.onerror = reject;
                reader.readAsDataURL(modalFiles.rgb);
            });

            previewDiv.innerHTML = `
                <div class="preview-image-container">
                    <img src="${imageData}" alt="RGB Image" style="max-width: 100%; height: auto;">
                    <p class="preview-caption">RGB Image (Local Preview)</p>
                </div>
            `;
            return;
        }

        // SCENARIO 2: EXISTING FILE (Fetch from Server)
        let rgbImageData = null;
        try {
            // We use the 'rgb_original' preview endpoint if available, or raw download
            // Let's try to get the preview generated by api_server.py first (base64)
            let url = `${API_BASE_URL}/containers/${currentContainerId}/panels/${currentModalPanel}/depth/preview`;
            if (currentModalUploadId) url += `?upload_id=${currentModalUploadId}`;

            const previewResponse = await fetch(url);
            const previewData = await previewResponse.json();

            if (previewData.success && previewData.previews && previewData.previews.rgb_original) {
                previewDiv.innerHTML = `
                    <div class="preview-image-container">
                        <img src="data:image/png;base64,${previewData.previews.rgb_original}" alt="RGB Image">
                        <p class="preview-caption">RGB Image</p>
                    </div>`;
                return;
            }
        } catch (error) {
            console.warn('Failed to fetch RGB preview base64, falling back to raw image', error);
        }

        // Fallback: Fetch raw image URL
        let rawUrl = `${API_BASE_URL}/containers/${currentContainerId}/panels/${currentModalPanel}/rgb`;
        if (currentModalUploadId) rawUrl += `?upload_id=${currentModalUploadId}`;

        previewDiv.innerHTML = `
            <div class="preview-image-container">
                <img src="${rawUrl}" alt="RGB Image" style="max-width: 100%; height: auto;">
                <p class="preview-caption">RGB Image</p>
            </div>`;

    } catch (error) {
        console.error('Error loading RGB preview:', error);
        previewDiv.innerHTML = '<p class="error-message">Error loading RGB preview</p>';
    }
}

// Show RANSAC Preview
async function showRANSACPreview() {
    // FIX 1: Allow execution if EITHER a new file (depth) OR existing file (depthFilename) exists
    const hasFile = modalFiles.depth || modalFiles.depthFilename;

    if (!hasFile || !currentContainerId || !currentModalPanel) return;

    const ransacPreview = document.getElementById('ransac-preview');
    const ransacStats = document.getElementById('ransac-stats');
    const ransacImage = document.getElementById('ransac-preview-image');

    ransacPreview.style.display = 'block';
    ransacStats.innerHTML = '<div class="spinner"></div><p>Extracting panel...</p>';
    ransacImage.style.display = 'none';

    try {
        // Get RANSAC parameters
        const rectangularMaskCheckbox = document.getElementById('modal-rectangular-mask-checkbox');
        const params = {
            // FIX 2: Pass the specific upload ID so backend knows which file to process
            upload_id: currentModalUploadId,
            adaptive_threshold: false,
            apply_morphological_closing: true,
            closing_kernel_size: 30,
            force_rectangular_mask: rectangularMaskCheckbox ? rectangularMaskCheckbox.checked : true
        };

        const response = await fetch(`${API_BASE_URL}/containers/${currentContainerId}/panels/${currentModalPanel}/ransac/preview`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });

        const data = await response.json();

        if (data.success) {
            const stats = data.stats || {};
            ransacStats.innerHTML = `
                <div class="ransac-stat-item">
                    <div class="ransac-stat-label">Panel Coverage</div>
                    <div class="ransac-stat-value">${stats.plane_percentage ? stats.plane_percentage.toFixed(1) : 'N/A'}%</div>
                </div>
            `;

            if (data.preview_image) {
                ransacImage.src = `data:image/png;base64,${data.preview_image}`;
                ransacImage.style.display = 'block';
            }
        } else {
            ransacStats.innerHTML = `<p class="error-message">${data.error || 'Error loading RANSAC preview'}</p>`;
        }
    } catch (error) {
        console.error('Error showing RANSAC preview:', error);
        ransacStats.innerHTML = '<p class="error-message">Error loading RANSAC preview</p>';
    }
}

// Check Ready Status
function checkReadyStatus() {
    const statusMessage = document.getElementById('status-message');
    const confirmBtn = document.getElementById('confirm-upload');

    // Check if we have a File object (new upload) or existing file (via depthFilename)
    const hasFile = modalFiles.depth instanceof File || modalFiles.depthFilename;

    if (hasFile) {
        statusMessage.textContent = '✅ Ready to Process';
        statusMessage.className = 'ready';
        confirmBtn.disabled = false;
        // Show "Update" if files already exist, "Upload" if new files
        confirmBtn.textContent = modalFiles.depth instanceof File ? 'Upload' : 'Update';
    } else {
        statusMessage.textContent = 'Please upload a depth map (.npy file) to proceed';
        statusMessage.className = 'waiting';
        confirmBtn.disabled = true;
        confirmBtn.textContent = 'Upload';
    }
}

// Confirm Upload
async function confirmUpload() {
    if (!currentContainerId || !currentModalPanel) return;

    // Store panel name
    const panelName = currentModalPanel;
    const confirmBtn = document.getElementById('confirm-upload');

    // Close check
    if (confirmBtn.textContent === 'Close') {
        closeModal();
        await updatePanelDisplay(panelName);
        return;
    }

    // --- FIX START ---
    // Check for ANY new file (Depth OR RGB)
    const hasNewDepth = modalFiles.depth instanceof File;
    const hasNewRGB = modalFiles.rgb instanceof File;
    const isUpdate = currentModalUploadId !== null;

    // We proceed if:
    // 1. It's a completely new upload (New Depth required)
    // 2. It's an update AND we have at least one new file (Depth OR RGB)
    const shouldUpload = (hasNewDepth) || (isUpdate && hasNewRGB);

    if (!shouldUpload && !isUpdate) return; // Nothing to do
    // -----------------

    confirmBtn.disabled = true;
    confirmBtn.textContent = isUpdate ? 'Updating...' : 'Uploading...';

    try {
        // Check if container exists
        if (!currentContainerId) {
            showNotification('Please create or select a container first', 'error');
            confirmBtn.textContent = hasNewFile ? 'Upload Files' : 'Update';
            return;
        }

        // Verify container exists on server
        try {
            const containerCheck = await fetch(`${API_BASE_URL}/containers/${currentContainerId}`);
            if (!containerCheck.ok) {
                if (containerCheck.status === 404) {
                    showNotification(`Container ${currentContainerId} not found. Please create a new container.`, 'error');
                    confirmBtn.textContent = hasNewFile ? 'Upload Files' : 'Update';
                    return;
                }
                throw new Error(`HTTP error! status: ${containerCheck.status}`);
            }
        } catch (error) {
            console.error('Error checking container:', error);
            showNotification('Failed to verify container. Please try again.', 'error');
            confirmBtn.textContent = hasNewFile ? 'Upload Files' : 'Update';
            return;
        }

        if (shouldUpload) {
            // CASE 1: Uploading Files (New or Update)
            const formData = new FormData();

            // Append Depth if new
            if (hasNewDepth) {
                formData.append('depth_file', modalFiles.depth);
            }

            // Append RGB if new
            if (hasNewRGB) {
                formData.append('rgb_file', modalFiles.rgb);
            }

            // Always append RANSAC settings
            const ransacCheckbox = document.getElementById('modal-ransac-checkbox');
            const rectangularMaskCheckbox = document.getElementById('modal-rectangular-mask-checkbox');
            formData.append('use_ransac', ransacCheckbox ? ransacCheckbox.checked : true);
            formData.append('force_rectangular_mask', rectangularMaskCheckbox ? rectangularMaskCheckbox.checked : true);

            // --- FIX: Pass upload_id if updating ---
            if (isUpdate) {
                formData.append('upload_id', currentModalUploadId);
            }
            // --------------------------------------

            const response = await fetch(`${API_BASE_URL}/containers/${currentContainerId}/panels/${panelName}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.success) {
                showNotification(`Files ${isUpdate ? 'updated' : 'uploaded'} for ${panelName}`, 'success');
                modalPreviewData = data;

                // If it was a NEW upload (not update), add to list
                // If it was an UPDATE, we assume the backend updated the existing record, 
                // but we should refresh our local list to match.

                if (!isUpdate) {
                    // Create new entry logic (same as before)
                    const newFilePair = {
                        upload_id: data.upload_id,
                        depth_filename: data.depth_filename || null,
                        depth_shape: data.depth_shape || [],
                        rgb_filename: data.rgb_filename || null,
                        rgb_shape: data.rgb_shape || null,
                        uploaded_at: new Date().toISOString()
                    };
                    if (!panelFilePairs[panelName]) panelFilePairs[panelName] = [];
                    panelFilePairs[panelName].push(newFilePair);
                } else {
                    // It was an update: Update local array entry
                    if (panelFilePairs[panelName]) {
                        const idx = panelFilePairs[panelName].findIndex(p => p.upload_id === currentModalUploadId);
                        if (idx !== -1) {
                            panelFilePairs[panelName][idx].depth_filename = data.depth_filename;
                            panelFilePairs[panelName][idx].rgb_filename = data.rgb_filename;
                        }
                    }
                }

                // UI Updates (keep existing logic)
                if (data.depth_filename) modalFiles.depthFilename = data.depth_filename;
                if (data.rgb_filename) modalFiles.rgbFilename = data.rgb_filename;

                // Clear file objects
                modalFiles.depth = null;
                modalFiles.rgb = null;

                // Show toggle buttons if RGB available
                if (data.rgb_filename) {
                    document.getElementById('preview-toggle-buttons').style.display = 'flex';
                }

                // Refresh preview
                await showDepthPreview(currentModalUploadId);

                await updatePanelDisplay(panelName);
                confirmBtn.disabled = false;
                confirmBtn.textContent = 'Close';
            } else {
                throw new Error(data.error || 'Failed');
            }
        } else {
            // CASE 2: No new files, just updating RANSAC settings
            // (Keep existing update-ransac logic here)
            const ransacCheckbox = document.getElementById('modal-ransac-checkbox');
            const rectangularMaskCheckbox = document.getElementById('modal-rectangular-mask-checkbox');

            if (ransacCheckbox) {
                await fetch(`${API_BASE_URL}/containers/${currentContainerId}/panels/${panelName}/update-ransac`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        upload_id: currentModalUploadId,
                        use_ransac: ransacCheckbox.checked,
                        force_rectangular_mask: rectangularMaskCheckbox ? rectangularMaskCheckbox.checked : true
                    })
                });
                showNotification(`RANSAC settings updated`, 'success');
            }
            closeModal();
            await updatePanelDisplay(panelName);
        }
    } catch (error) {
        console.error('Error uploading:', error);
        showNotification(error.message || 'Failed to upload', 'error');
        confirmBtn.disabled = false;
        confirmBtn.textContent = isUpdate ? 'Update' : 'Upload';
    }
}

// Delete a specific file pair
async function deleteFilePair(panelName, uploadId) {
    if (!uploadId) return;

    if (!confirm('Are you sure you want to delete this file? This cannot be undone.')) {
        return;
    }

    try {
        // 1. Call Backend to delete
        const response = await fetch(`${API_BASE_URL}/containers/${currentContainerId}/panels/${panelName}/uploads/${uploadId}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (data.success) {
            // 2. Remove from local state
            if (panelFilePairs[panelName]) {
                panelFilePairs[panelName] = panelFilePairs[panelName].filter(p => p.upload_id !== uploadId);
            }

            // 3. Update UI
            showNotification('File deleted successfully', 'success');
            await updatePanelDisplay(panelName);
        } else {
            showNotification(data.error || 'Failed to delete file', 'error');
        }
    } catch (error) {
        console.error('Error deleting file:', error);
        showNotification('Error deleting file', 'error');
    }
}

// Refresh Panel Files
async function refreshPanelFiles(panelName) {
    const refreshBtn = document.querySelector(`.btn-refresh[data-panel="${panelName}"]`);

    // Add spinning animation
    const originalText = refreshBtn.textContent;
    refreshBtn.textContent = '⏳';
    refreshBtn.disabled = true;

    try {
        await updatePanelDisplay(panelName);
        showNotification(`Files refreshed for ${panelName} panel`, 'success');
    } catch (error) {
        console.error('Error refreshing panel files:', error);
        showNotification('Failed to refresh files', 'error');
    } finally {
        refreshBtn.textContent = originalText;
        refreshBtn.disabled = false;
    }
}

// Update Panel Display
async function updatePanelDisplay(panelName) {
    const filesDiv = document.getElementById(`${panelName}-files`);
    const processBtn = document.querySelector(`.btn-process[data-panel="${panelName}"]`);

    // Check if elements exist (processing page might not be active)
    if (!filesDiv || !processBtn) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/containers/${currentContainerId}`);
        const data = await response.json();

        if (data.success) {
            const panelData = data.container.panels[panelName];

            // --- 1. RENDER FILE LIST ---
            // Ensure file pairs array exists
            if (panelData && panelData.uploads) {
                panelFilePairs[panelName] = panelData.uploads.map(upload => ({
                    upload_id: upload.upload_id,
                    depth_filename: upload.depth_filename || 'depth_map.npy',
                    depth_shape: upload.depth_shape || [],
                    rgb_filename: upload.rgb_filename || null,
                    rgb_shape: upload.rgb_shape || null,
                    uploaded_at: upload.uploaded_at || new Date().toISOString()
                }));
            } else {
                panelFilePairs[panelName] = [];
            }
            const filePairs = panelFilePairs[panelName];
            let html = '';

            if (filePairs.length === 0) {
                html = '<div class="uploaded-files-empty">Upload a file for dent inspection!</div>';
            } else {
                filePairs.forEach((pair, index) => {
                    // (Your existing HTML generation for file list works fine, keeping it brief here)
                    html += `<div class="file-pair-container" data-pair-index="${index}" data-upload-id="${pair.upload_id || ''}">`;
                    html += `
                                <button class="delete-file-btn" onclick="event.stopPropagation(); deleteFilePair('${panelName}', '${pair.upload_id}')" title="Delete this file">
                                    ✕
                                </button>
                            `
                    if (pair.depth_filename) {
                        html += `
                            <div class="uploaded-file-item clickable-file" onclick="reopenUploadModal('${panelName}', '${pair.upload_id}')">
                                <span class="file-icon">📁</span>
                                <div class="file-info">
                                    <div class="file-name">${pair.depth_filename}</div>
                                    <div class="file-details">Shape: ${pair.depth_shape ? pair.depth_shape.join(' × ') : 'N/A'} | Click to update</div>
                                </div>
                            </div>
                        `;
                    }

                    if (pair.rgb_filename) {
                        html += `
                            <div class="uploaded-file-item clickable-file" onclick="reopenUploadModal('${panelName}', '${pair.upload_id}')">
                                <span class="file-icon">🖼️</span>
                                <div class="file-info">
                                    <div class="file-name">${pair.rgb_filename}</div>
                                    <div class="file-details">Shape: ${pair.rgb_shape ? pair.rgb_shape.join(' × ') : 'N/A'} | Click to update</div>
                                </div>
                            </div>
                        `;
                    }
                    html += `</div>`;
                });
            }

            filesDiv.innerHTML = html;

            // Enable process button if any depth file exists
            const hasDepthFile = filePairs.length > 0 && filePairs.some(pair => pair.depth_filename);
            processBtn.disabled = !hasDepthFile;

            // --- 2. RENDER RESULTS ---
            // This is where the fix is: Just pass the results object directly.
            // Do NOT try to filter for "resultToDisplay" here.
            if (data.container.results && data.container.results[panelName]) {
                await displayExistingResults(panelName, data.container.results[panelName]);
            } else {
                // Clear results if none exist
                const resultsDiv = document.getElementById(`${panelName}-results`);
                if (resultsDiv) {
                    resultsDiv.innerHTML = '';
                }
            }
        }
    } catch (error) {
        console.error('Error updating panel display:', error);
    }
}

// Load Containers List
async function loadContainers() {
    try {
        const response = await fetch(`${API_BASE_URL}/containers`);
        const data = await response.json();

        if (data.success) {
            containers = data.containers;
            displayContainersList();
        }
    } catch (error) {
        console.error('Error loading containers:', error);
    }
}

// Display Containers List (Updated Layout)
function displayContainersList() {
    const containerListEl = document.getElementById('containers-list-container');

    // Sort containers by 'created_at' date in descending order (Newest first)
    if (containers && containers.length > 0) {
        containers.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    }

    if (containers.length === 0) {
        containerListEl.innerHTML = '<p style="text-align: center; color: #6c757d;">No containers created yet.</p>';
        return;
    }

    containerListEl.innerHTML = containers.map(cont => {
        // 1. Calculate Failures
        let failCount = 0;
        let totalProcessed = 0;

        if (cont.results) {
            // Loop through each panel (back, left, right, etc.)
            Object.values(cont.results).forEach(panelResults => {
                if (!panelResults) return;

                // Handle potential data structures (legacy single object vs new dictionary)
                let results = [];
                if (panelResults.status) {
                    results = [panelResults]; // Legacy
                } else {
                    results = Object.values(panelResults); // New Dictionary format
                }

                // Count failures
                results.forEach(res => {
                    totalProcessed++;
                    if (res && res.status === 'FAIL') {
                        failCount++;
                    }
                });
            });
        }

        // 2. Format Date
        const dateStr = new Date(cont.created_at).toLocaleString();

        // 3. Render with Flexbox Layout
        // Left Side: Name & Date
        // Right Side: Failure Count
        return `
        <div class="container-item" onclick="window.openContainer('${cont.id}')" 
             style="display: flex; justify-content: space-between; align-items: center; padding: 15px;">
            
            <div class="container-info">
                <h3 style="margin: 0 0 5px 0; font-size: 1.1rem;">${cont.name}</h3>
                <p style="margin: 0; color: #666; font-size: 0.85em;">Created: ${dateStr}</p>
            </div>

            <div class="container-stats" style="text-align: right;">
                <span style="
                    display: inline-block;
                    padding: 6px 12px;
                    border-radius: 6px;
                    background-color: ${failCount > 0 ? '#ffebee' : '#f1f8e9'};
                    color: ${failCount > 0 ? '#c62828' : '#33691e'};
                    font-weight: bold;
                    font-size: 0.9em;
                    border: 1px solid ${failCount > 0 ? '#ef9a9a' : '#c5e1a5'};
                ">
                    ${failCount > 0 ? `⚠️ Failed Detected: ${failCount}` : '✅ No Failures'}
                </span>
            </div>
            
        </div>
        `;
    }).join('');

    // Make openContainer available globally
    window.openContainer = openContainer;
}

// Reopen Upload Modal (global function)
window.reopenUploadModal = function (panelName, uploadId = null) {
    openUploadModal(panelName);
    // Pass the specific ID to the loader
    loadExistingFilesIntoModal(panelName, uploadId);
};

// Load Existing Files Into Modal
async function loadExistingFilesIntoModal(panelName, uploadId = null) {
    if (!currentContainerId) return;

    try {
        const response = await fetch(`${API_BASE_URL}/containers/${currentContainerId}`);
        const data = await response.json();

        if (data.success) {
            const panelData = data.container.panels[panelName];
            if (panelData) {
                // DEFAULT: Use the top-level data (latest)
                let targetData = panelData;

                // IF ID PROVIDED: Find the specific file in the list
                if (uploadId && panelData.uploads) {
                    const found = panelData.uploads.find(u => u.upload_id === uploadId);
                    if (found) {
                        targetData = found;
                        currentModalUploadId = uploadId; // Set global ID
                    }
                } else {
                    currentModalUploadId = targetData.upload_id || null;
                }

                // Reset modal files
                modalFiles.depth = null;
                modalFiles.rgb = null;

                // Update modal with TARGET data (Depth)
                if (targetData.depth_path || targetData.depth_filename) {
                    const depthArea = document.querySelector('.upload-area-modal[data-type="depth"]');
                    if (depthArea) {
                        depthArea.classList.add('has-file');
                        const fname = targetData.depth_filename || 'depth_map.npy';
                        modalFiles.depthFilename = fname;

                        depthArea.querySelector('.upload-placeholder').innerHTML = `
                            <span class="upload-icon">✅</span>
                            <p class="file-name">${fname}</p>
                            <p class="upload-hint">File uploaded - Click to change</p>
                        `;

                        document.getElementById('ransac-checkbox-section').style.display = 'block';
                    }
                }

                // Update modal with TARGET data (RGB)
                if (targetData.rgb_path || targetData.rgb_filename) {
                    const rgbArea = document.querySelector('.upload-area-modal[data-type="rgb"]');
                    if (rgbArea) {
                        rgbArea.classList.add('has-file');
                        const fname = targetData.rgb_filename || 'rgb_image.png';
                        modalFiles.rgbFilename = fname;

                        rgbArea.querySelector('.upload-placeholder').innerHTML = `
                            <span class="upload-icon">✅</span>
                            <p class="file-name">${fname}</p>
                            <p class="upload-hint">File uploaded - Click to change</p>
                        `;
                    }
                }

                // --- RESTORE RANSAC STATE ---
                const ransacCheckbox = document.getElementById('modal-ransac-checkbox');
                const rectangularMaskLabel = document.getElementById('rectangular-mask-checkbox-label');
                const rectangularMaskCheckbox = document.getElementById('modal-rectangular-mask-checkbox');

                if (ransacCheckbox && targetData.use_ransac !== undefined) {
                    ransacCheckbox.checked = targetData.use_ransac;

                    // Restore sub-option visibility
                    if (rectangularMaskLabel) {
                        rectangularMaskLabel.style.display = targetData.use_ransac ? 'block' : 'none';
                    }
                }

                if (rectangularMaskCheckbox && targetData.force_rectangular_mask !== undefined) {
                    rectangularMaskCheckbox.checked = targetData.force_rectangular_mask;
                }

                // Check status
                checkReadyStatus();

                // --- CRITICAL FIX: TRIGGER PREVIEWS ---
                setTimeout(async () => {
                    // 1. Show Depth Preview
                    await showDepthPreview(currentModalUploadId);

                    // 2. Show RANSAC Preview (IF CHECKED)
                    if (ransacCheckbox && ransacCheckbox.checked) {
                        // This line was missing or logic prevented it from running
                        await showRANSACPreview();
                    }
                }, 300);
            }
        }
    } catch (error) {
        console.error('Error loading existing files:', error);
    }
}

// Create Container
async function createContainer(name) {
    try {
        const response = await fetch(`${API_BASE_URL}/containers/create`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ name })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Create container response:', data); // Debug log

        if (data.success) {
            currentContainerId = data.container_id;
            const containerName = data.container?.name || data.container_id;
            showNotification(`Container "${containerName}" created successfully!`, 'success');

            // Navigate to processing page
            await openContainer(data.container_id);
        } else {
            showNotification(data.error || 'Failed to create container', 'error');
        }
    } catch (error) {
        console.error('Error creating container:', error);
        showNotification(`Failed to create container: ${error.message}. Make sure the API server is running.`, 'error');
    }
}

// Open Container
async function openContainer(containerId) {
    try {
        const response = await fetch(`${API_BASE_URL}/containers/${containerId}`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Open container response:', data);

        if (data.success) {
            currentContainerId = containerId;
            const containerTitle = document.getElementById('container-title');
            if (containerTitle) {
                containerTitle.textContent = `Processing: ${data.container.name || data.container.id}`;
            }

            // Navigate to processing page
            showPage('processing-page');

            // --- ✅ NEW ADDITION: FORCE RESET VIEWER ---
            const viewer = document.getElementById('results-viewer');
            const viewerContent = document.getElementById('viewer-content');

            // 1. Hide the viewer sidebar
            if (viewer) {
                viewer.classList.remove('active');
            }

            // 2. Clear stale content and restore placeholder
            if (viewerContent) {
                viewerContent.innerHTML = '<p class="viewer-placeholder">Select a result to view details</p>';
            }

            // 3. Remove "active" highlight from any result items (if they exist)
            document.querySelectorAll('.result-item').forEach(item => {
                item.classList.remove('active');
            });
            // -------------------------------------------

            // Load existing panel data
            await loadPanelData(data.container);
        } else {
            showNotification(data.error || 'Failed to load container', 'error');
        }
    } catch (error) {
        console.error('Error opening container:', error);
        showNotification(`Failed to load container: ${error.message}. Make sure the API server is running.`, 'error');
    }
}

// Load Panel Data (Fixed: Clears previous container state)
async function loadPanelData(container) {
    const panels = ['back', 'left', 'right', 'roof', 'door'];

    // 1. STRICT RESET: Wipe all previous data immediately
    // This ensures no "ghost files" carry over from the previous container
    panelFilePairs = {
        back: [],
        left: [],
        right: [],
        roof: [],
        door: []
    };

    try {
        // 2. Populate with new data (if any)
        for (const panelName of panels) {
            const panelData = container.panels[panelName];

            // Only map if uploads exist
            if (panelData && panelData.uploads && panelData.uploads.length > 0) {
                panelFilePairs[panelName] = panelData.uploads.map(upload => ({
                    upload_id: upload.upload_id,
                    depth_filename: upload.depth_filename || 'depth_map.npy',
                    depth_shape: upload.depth_shape || [],
                    rgb_filename: upload.rgb_filename || null,
                    rgb_shape: upload.rgb_shape || null,
                    uploaded_at: upload.uploaded_at || new Date().toISOString()
                }));
            }
            // Note: No 'else' needed because we already reset everything to [] at the top
        }

        // 3. Update display for all panels
        for (const panelName of panels) {
            await updatePanelDisplay(panelName);
        }
    } catch (error) {
        console.error('Error loading panel data:', error);
    }
}

// Setup Model Upload
function setupModelUpload() {
    const modelFileInput = document.getElementById('model-file');
    const modelUploadArea = document.getElementById('model-upload-area');
    const modelPlaceholder = document.getElementById('model-placeholder');
    const modelSelected = document.getElementById('model-selected');
    const modelFilename = document.getElementById('model-filename');
    const removeModelBtn = document.getElementById('remove-model');

    // Click to browse
    modelUploadArea.addEventListener('click', () => {
        modelFileInput.click();
    });

    // File selected
    modelFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            if (!file.name.endsWith('.pth')) {
                showNotification('Please upload a .pth model file', 'error');
                return;
            }
            modelFile = file;
            modelFilename.textContent = file.name;
            modelPlaceholder.style.display = 'none';
            modelSelected.style.display = 'flex';
            // Clear the path input when file is uploaded
            document.getElementById('model-path').value = '';
        }
    });

    // Remove file
    removeModelBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        modelFile = null;
        modelFileInput.value = '';
        modelPlaceholder.style.display = 'flex';
        modelSelected.style.display = 'none';
        // Restore default path
        document.getElementById('model-path').value = 'best_attention_unet_4.pth';
    });

    // Drag and drop handlers
    modelUploadArea.addEventListener('dragenter', (e) => {
        e.preventDefault();
        e.stopPropagation();
        modelUploadArea.classList.add('dragover');
    });

    modelUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        modelUploadArea.classList.add('dragover');
    });

    modelUploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        modelUploadArea.classList.remove('dragover');
    });

    modelUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        modelUploadArea.classList.remove('dragover');

        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            const file = files[0];
            if (!file.name.endsWith('.pth')) {
                showNotification('Please upload a .pth model file', 'error');
                return;
            }
            modelFile = file;
            modelFilename.textContent = file.name;
            modelPlaceholder.style.display = 'none';
            modelSelected.style.display = 'flex';
            // Clear the path input when file is uploaded
            document.getElementById('model-path').value = '';
        }
    });
}

// Load Model
async function loadModel() {
    const statusDiv = document.getElementById('model-status');
    const loadBtn = document.getElementById('load-model-btn');
    let modelPath = document.getElementById('model-path').value.trim();

    loadBtn.disabled = true;
    loadBtn.textContent = 'Loading...';
    statusDiv.className = 'status-message info';
    statusDiv.textContent = 'Loading model...';
    statusDiv.style.display = 'block';

    try {
        // If model file is uploaded, upload it first
        if (modelFile) {
            statusDiv.textContent = 'Uploading model file...';
            const formData = new FormData();
            formData.append('model_file', modelFile);

            const uploadResponse = await fetch(`${API_BASE_URL}/model/upload`, {
                method: 'POST',
                body: formData
            });

            const uploadData = await uploadResponse.json();
            if (!uploadData.success) {
                throw new Error(uploadData.error || 'Failed to upload model file');
            }
            modelPath = uploadData.model_path;
            statusDiv.textContent = 'Loading model...';
        } else if (!modelPath) {
            // Use default if nothing specified
            modelPath = 'best_attention_unet_4.pth';
        }

        const response = await fetch(`${API_BASE_URL}/model/load`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_path: modelPath })
        });

        const data = await response.json();

        if (data.success) {
            modelLoaded = true;

            // Show which model is being used
            const modelPathDisplay = modelFile ? modelFile.name : (modelPath || 'best_attention_unet_4.pth');
            statusDiv.className = 'status-message success';
            statusDiv.innerHTML = `
                ✅ Model loaded successfully!<br>
                <strong>Using model:</strong> ${modelPathDisplay}
            `;

            // Show detailed model information
            const modelInfoSection = document.getElementById('model-info-section');
            const modelInfoContent = document.getElementById('model-info-content');
            if (modelInfoSection && modelInfoContent) {
                modelInfoSection.style.display = 'block';
                modelInfoContent.innerHTML = `
                    <div class="model-info-item">
                        <strong>Model Path:</strong> ${data.model_path || modelPath}
                    </div>
                    <div class="model-info-item">
                        <strong>Architecture:</strong> ${data.model_info.architecture}
                    </div>
                    <div class="model-info-item">
                        <strong>Input Channels:</strong> ${data.model_info.input_channels} (Depth + Gradients)
                    </div>
                    <div class="model-info-item">
                        <strong>Output Channels:</strong> ${data.model_info.output_channels} (Binary Mask)
                    </div>
                    <div class="model-info-item">
                        <strong>Total Parameters:</strong> ${data.model_info.total_parameters.toLocaleString()}
                    </div>
                    <div class="model-info-item">
                        <strong>Trainable Parameters:</strong> ${data.model_info.trainable_parameters.toLocaleString()}
                    </div>
                    <div class="model-info-item">
                        <strong>Model Size:</strong> ${data.model_info.model_size_mb.toFixed(2)} MB
                    </div>
                    <div class="model-info-item">
                        <strong>Device:</strong> ${data.model_info.device}
                    </div>
                `;
            }

            showNotification('Model loaded successfully!', 'success');
        } else {
            modelLoaded = false;
            statusDiv.className = 'status-message error';
            statusDiv.textContent = `❌ ${data.error}`;
            showNotification(data.error || 'Failed to load model', 'error');
        }
    } catch (error) {
        modelLoaded = false;
        statusDiv.className = 'status-message error';
        statusDiv.textContent = `❌ Error: ${error.message}`;
        showNotification('Failed to load model. Make sure the API server is running.', 'error');
    } finally {
        loadBtn.disabled = false;
        loadBtn.textContent = 'Load Model';
    }
}

// Setup Intrinsics Upload
function setupIntrinsicsUpload() {
    const intrinsicsFileInput = document.getElementById('intrinsics-file');
    const intrinsicsUploadArea = document.getElementById('intrinsics-upload-area');
    const intrinsicsPlaceholder = document.getElementById('intrinsics-placeholder');
    const intrinsicsSelected = document.getElementById('intrinsics-selected');
    const intrinsicsFilename = document.getElementById('intrinsics-filename');
    const removeIntrinsicsBtn = document.getElementById('remove-intrinsics');

    // Click to browse
    intrinsicsUploadArea.addEventListener('click', () => {
        intrinsicsFileInput.click();
    });

    // File selected
    intrinsicsFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            if (!file.name.endsWith('.json')) {
                showNotification('Please upload a JSON file for camera intrinsics', 'error');
                return;
            }
            intrinsicsFile = file;
            intrinsicsFilename.textContent = file.name;
            intrinsicsPlaceholder.style.display = 'none';
            intrinsicsSelected.style.display = 'flex';
        }
    });

    // Remove file
    removeIntrinsicsBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        intrinsicsFile = null;
        intrinsicsFileInput.value = '';
        intrinsicsPlaceholder.style.display = 'flex';
        intrinsicsSelected.style.display = 'none';
    });

    // Drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        intrinsicsUploadArea.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        intrinsicsUploadArea.addEventListener(eventName, () => {
            intrinsicsUploadArea.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        intrinsicsUploadArea.addEventListener(eventName, () => {
            intrinsicsUploadArea.classList.remove('dragover');
        }, false);
    });

    intrinsicsUploadArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            const file = files[0];
            if (!file.name.endsWith('.json')) {
                showNotification('Please upload a JSON file for camera intrinsics', 'error');
                return;
            }
            intrinsicsFile = file;
            intrinsicsFilename.textContent = file.name;
            intrinsicsPlaceholder.style.display = 'none';
            intrinsicsSelected.style.display = 'flex';
        }
    });
}

// Process Panel
// Replace your existing processPanel function with this one
async function processPanel(panelName) {
    if (!currentContainerId) {
        showNotification('Please create a container first', 'error');
        return;
    }

    if (!modelLoaded) {
        showNotification('Please load the model first', 'error');
        return;
    }

    const processBtn = document.querySelector(`.btn-process[data-panel="${panelName}"]`);

    // Get ALL files for this panel
    const filePairs = panelFilePairs[panelName] || [];

    if (filePairs.length === 0) {
        showNotification('No files to process', 'error');
        return;
    }

    processBtn.disabled = true;
    processBtn.textContent = `Processing 0/${filePairs.length}...`;

    // Upload intrinsics if needed (do this once)
    let intrinsicsPath = null;
    if (intrinsicsFile) {
        const intrinsicsFormData = new FormData();
        intrinsicsFormData.append('intrinsics_file', intrinsicsFile);
        try {
            const iResp = await fetch(`${API_BASE_URL}/intrinsics/upload`, { method: 'POST', body: intrinsicsFormData });
            if (iResp.ok) {
                const iData = await iResp.json();
                intrinsicsPath = iData.intrinsics_path;
            }
        } catch (e) { console.warn("Intrinsics upload failed"); }
    }

    const panelThresholds = {
        'back': 0.03,   // Usually doors (Flat) -> Tight Threshold
        'door': 0.03,   // Doors (Flat) -> Tight Threshold
        'left': 0.03,   // Wall (Corrugated) -> Loose Threshold
        'right': 0.03,  // Wall (Corrugated) -> Loose Threshold
        'roof': 0.03    // Roof (Corrugated) -> Loose Threshold
    };

    const activeResidualThreshold = panelThresholds[panelName] || 0.05;

    console.log(`Processing ${panelName}: Using Residual Threshold = ${activeResidualThreshold}m`);

    try {
        // --- LOOP THROUGH ALL FILES ---
        for (let i = 0; i < filePairs.length; i++) {
            const pair = filePairs[i];
            processBtn.textContent = `Processing ${i + 1}/${filePairs.length}...`;

            // Skip if no upload_id (local error)
            if (!pair.upload_id) continue;

            // Prepare params for THIS SPECIFIC file
            const params = {
                upload_id: pair.upload_id, // <--- CRITICAL: Process this specific ID
                threshold: parseFloat(document.getElementById('threshold').value),
                residual_threshold: activeResidualThreshold,
                use_ransac: true,
                adaptive_threshold: false,
                apply_morphological_closing: true,
                closing_kernel_size: 30,
                force_rectangular_mask: true,
                max_area_threshold: parseFloat(document.getElementById('max-area-threshold').value),
                max_depth_threshold: parseFloat(document.getElementById('max-depth-threshold').value),
                overlay_alpha: 0.2,
                outline_thickness: 2,
                intrinsics_json_path: intrinsicsPath
            };

            // Call API
            const response = await fetch(`${API_BASE_URL}/containers/${currentContainerId}/panels/${panelName}/process`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });

            if (!response.ok) {
                console.error(`Failed to process file ${i + 1}`);
                // Optional: continue to next file even if one fails
            }
        }

        showNotification(`Finished processing ${filePairs.length} files for ${panelName}`, 'success');

        // --- UPDATE DISPLAY ---
        // Fetch the fresh data (which now contains all 5 results) and display them
        await updatePanelDisplay(panelName);

    } catch (error) {
        console.error('Error processing panel:', error);
        showNotification('Error during processing loop', 'error');
    } finally {
        processBtn.disabled = false;
        processBtn.textContent = 'Process';
    }
}


// Process All Panels (With Strict Checks)
async function processAllPanels() {
    console.log('processAllPanels called');

    // --- 1. STRICT CHECKS (Guard Clauses) ---
    // These must happen BEFORE we disable any buttons!

    if (!currentContainerId) {
        showNotification('Please create a container first', 'error');
        return; // Stops execution immediately
    }

    if (!modelLoaded) {
        showNotification('Please load the model first', 'error');
        // This ensures we DO NOT proceed to the loop below
        return; // Stops execution immediately
    }

    const processAllBtn = document.getElementById('process-all-btn');
    if (!processAllBtn) return;

    // --- 2. LOCK UI ---
    // Only lock buttons if we passed the checks above
    processAllBtn.disabled = true;
    processAllBtn.textContent = 'Processing Batch...';

    const allPanels = ['back', 'left', 'right', 'roof', 'door'];

    try {
        // --- 3. PROCESS LOOP ---
        for (const panelName of allPanels) {

            // Check if this panel has files uploaded
            // We use the local panelFilePairs to check quickly without asking the server
            const hasFiles = panelFilePairs[panelName] && panelFilePairs[panelName].length > 0;

            if (hasFiles) {
                console.log(`Triggering process for: ${panelName}`);

                // Update button text to show progress
                processAllBtn.textContent = `Processing: ${panelName.toUpperCase()}...`;

                // Run the individual panel process and WAIT for it
                await processPanel(panelName);

                // Small safety pause to allow UI to breathe
                await new Promise(resolve => setTimeout(resolve, 500));
            }
        }

        showNotification('Batch processing complete!', 'success');

    } catch (error) {
        console.error('Error in batch processing:', error);
        showNotification('Something went wrong during batch processing.', 'error');
    } finally {
        // --- 4. UNLOCK UI ---
        // Always re-enable the button, even if errors occurred
        processAllBtn.disabled = false;
        processAllBtn.textContent = 'Process All';
    }
}
// Display Existing Results (Updated with "viewer-note" style)
async function displayExistingResults(panelName, resultData) {
    const resultsDiv = document.getElementById(`${panelName}-results`);
    if (!resultsDiv) return;

    // Clear current display
    resultsDiv.innerHTML = '';

    // Normalize data
    let resultsToRender = [];
    if (resultData && !resultData.metrics && !resultData.status) {
        resultsToRender = Object.values(resultData);
        resultsToRender.sort((a, b) => new Date(b.timestamp || 0) - new Date(a.timestamp || 0));
    } else if (resultData) {
        resultsToRender = [resultData];
    }

    if (resultsToRender.length === 0) return;

    for (const result of resultsToRender) {
        let previews = {};
        const metrics = result.metrics || {};
        const status = result.status || 'UNKNOWN';
        const timestamp = result.timestamp || new Date().toISOString();
        const note = result.note;

        // 1. Create Title
        let displayTitle = `Result ${new Date(timestamp).toLocaleTimeString()}`;
        if (panelFilePairs[panelName]) {
            const match = panelFilePairs[panelName].find(f => f.upload_id === result.upload_id);
            if (match) displayTitle = match.depth_filename;
        }

        // 2. NOTE LOGIC (Updated to match Viewer style)
        // We use "viewer-note under-title" to inherit the yellow styling
        const maxDepthIsNA = !metrics.max_depth_mm || metrics.max_depth_mm === 0;
        const noteDisplay = (note && !maxDepthIsNA)
            ? `<div class="viewer-note under-title" style="font-size: 0.85em; margin-bottom: 8px;">${note}</div>`
            : '';

        const resultItem = document.createElement('div');
        resultItem.className = 'result-item';
        resultItem.style.marginBottom = "10px";
        resultItem.dataset.uploadId = result.upload_id;

        resultItem.innerHTML = `
            <div class="result-header">
                <span class="result-title" style="font-size:0.95em; font-weight:bold;">${displayTitle}</span>
                <span class="result-status ${status.toLowerCase()}">${status}</span>
            </div>
            ${noteDisplay}
            <div class="result-metrics">
                <div class="result-metric">
                    <span>Max Depth:</span>
                    <span>${metrics.max_depth_mm ? metrics.max_depth_mm.toFixed(2) + ' mm' : 'N/A'}</span>
                </div>
            </div>
        `;

        resultItem.addEventListener('click', async () => {
            try {
                const previewUrl = `${API_BASE_URL}/containers/${currentContainerId}/panels/${panelName}/results/preview?upload_id=${result.upload_id}`;
                const previewResponse = await fetch(previewUrl);
                const previewData = await previewResponse.json();
                if (previewData.success) previews = previewData.previews;
            } catch (e) { }
            showResultInViewer(panelName, result, previews);
            document.querySelectorAll('.result-item').forEach(i => i.classList.remove('active'));
            resultItem.classList.add('active');
        });

        resultsDiv.appendChild(resultItem);
    }
}

// Display Results (Immediate Update with "viewer-note" style)
async function displayResults(panelName, resultData) {
    const resultsDiv = document.getElementById(`${panelName}-results`);
    if (!resultsDiv) return;

    let filename = '';
    try {
        const containerResponse = await fetch(`${API_BASE_URL}/containers/${currentContainerId}`);
        const containerData = await containerResponse.json();
        if (containerData.success) {
            const panelData = containerData.container.panels[panelName];
            if (panelData.uploads) {
                const upload = panelData.uploads.find(u => u.upload_id === resultData.upload_id);
                if (upload) filename = upload.depth_filename;
            }
        }
    } catch (e) { }

    const metrics = resultData.metrics || {};
    const status = resultData.status || 'UNKNOWN';
    const note = resultData.note;
    const timestamp = resultData.timestamp || new Date().toISOString();
    const resultTitle = filename || `Result - ${new Date(timestamp).toLocaleTimeString()}`;

    // NOTE LOGIC (Updated to match Viewer style)
    const maxDepthIsNA = !metrics.max_depth_mm || metrics.max_depth_mm === 0;
    const noteDisplay = (note && !maxDepthIsNA)
        ? `<div class="viewer-note under-title" style="font-size: 0.85em; margin-bottom: 8px;">${note}</div>`
        : '';

    const resultItem = document.createElement('div');
    resultItem.className = 'result-item';
    resultItem.dataset.panelName = panelName;

    resultItem.innerHTML = `
        <div class="result-header">
            <span class="result-title" style="font-weight:bold;">${resultTitle}</span>
            <span class="result-status ${status.toLowerCase()}">${status}</span>
        </div>
        ${noteDisplay}
        <div class="result-metrics">
            <div class="result-metric">
                <span>Max Depth:</span>
                <span>${metrics.max_depth_mm ? metrics.max_depth_mm.toFixed(2) + ' mm' : 'N/A'}</span>
            </div>
        </div>
    `;

    resultItem.addEventListener('click', async () => {
        let previews = {};
        try {
            const previewResponse = await fetch(`${API_BASE_URL}/containers/${currentContainerId}/panels/${panelName}/results/preview?upload_id=${resultData.upload_id}`);
            const previewData = await previewResponse.json();
            if (previewData.success) previews = previewData.previews;
        } catch (e) { }

        await showResultInViewer(panelName, resultData, previews);
        document.querySelectorAll('.result-item').forEach(item => item.classList.remove('active'));
        resultItem.classList.add('active');
    });

    resultsDiv.prepend(resultItem);
}

// Show Result in Viewer
async function showResultInViewer(panelName, resultData, previews) {
    const viewer = document.getElementById('results-viewer');
    const viewerContent = document.getElementById('viewer-content');
    const metrics = resultData.metrics || {};
    const status = resultData.status || 'UNKNOWN';
    const note = resultData.note || null;

    // Only show note if Max Depth is not N/A (meaning dent was detected)
    const maxDepthIsNA = !metrics.max_depth_mm || metrics.max_depth_mm === null || metrics.max_depth_mm === undefined || metrics.max_depth_mm === 0;
    const noteDisplay = (note && !maxDepthIsNA) ? `<div class="viewer-note under-title">${note}</div>` : '';

    let html = `
        <div class="viewer-header-section">
            <h4>${panelName.charAt(0).toUpperCase() + panelName.slice(1)} Panel Results</h4>
            <span class="result-status ${status.toLowerCase()} top-right">${status}</span>
        </div>
        ${noteDisplay}
        <div class="viewer-metrics">
            <div class="viewer-metric">
                <div class="viewer-metric-label">Total Defects</div>
                <div class="viewer-metric-value">${metrics.num_defects || 0}</div>
            </div>
            <div class="viewer-metric">
                <div class="viewer-metric-label">Area (cm²)</div>
                <div class="viewer-metric-value">${metrics.area_cm2 !== null && metrics.area_cm2 !== undefined ? metrics.area_cm2.toFixed(2) : 'N/A'}</div>
            </div>
            <div class="viewer-metric">
                <div class="viewer-metric-label">Max Depth (mm)</div>
                <div class="viewer-metric-value">${metrics.max_depth_mm ? metrics.max_depth_mm.toFixed(2) : 'N/A'}</div>
            </div>
            <div class="viewer-metric">
                <div class="viewer-metric-label">Avg Depth (mm)</div>
                <div class="viewer-metric-value">${metrics.avg_depth_mm ? metrics.avg_depth_mm.toFixed(2) : 'N/A'}</div>
            </div>
        </div>
    `;

    // --- NEW: Per-Dent Breakdown ---
    if (metrics.individual_dents && metrics.individual_dents.length > 0) {
        html += `
            <div class="dent-breakdown-section" style="margin-top: 15px; background: #fff; padding: 10px; border-radius: 8px; border: 1px solid #eee;">
                <h5 style="margin: 0 0 10px 0; color: #333; font-size: 0.95em;">🔍 Dent Details (${metrics.individual_dents.length} found)</h5>
                <div style="max-height: 150px; overflow-y: auto;">
                    <table style="width: 100%; font-size: 0.85em; border-collapse: collapse;">
                        <thead style="background: #f9f9f9; color: #666;">
                            <tr>
                                <th style="text-align: left; padding: 5px;">#</th>
                                <th style="text-align: right; padding: 5px;">Depth</th>
                                <th style="text-align: right; padding: 5px;">Area</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${metrics.individual_dents.map((dent, idx) => `
                                <tr style="border-bottom: 1px solid #eee;">
                                    <td style="padding: 5px;">Dent ${idx + 1}</td>
                                    <td style="text-align: right; padding: 5px; font-weight: bold; color: ${dent.max_depth_mm > 50 ? '#d32f2f' : '#333'}">
                                        ${dent.max_depth_mm.toFixed(1)} mm
                                    </td>
                                    <td style="text-align: right; padding: 5px; color: #666;">
                                        ${dent.area_cm2.toFixed(1)} cm²
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }

    // Add preview images - show overlay first if RGB is available
    // Add preview images - show overlay first if RGB is available
    if (previews.overlay) {
        // Calculate interactive boxes if we have bbox data
        let interactiveBoxesHtml = '';

        // We need image dimensions to calculate percentages
        // binary_mask_shape is usually [Height, Width]
        const imgDims = resultData.binary_mask_shape || resultData.preprocessed_shape;

        if (metrics.individual_dents && imgDims && imgDims.length === 2) {
            const H = imgDims[0];
            const W = imgDims[1];

            interactiveBoxesHtml = metrics.individual_dents.map((dent, idx) => {
                if (!dent.bbox) return '';

                const [x_min, y_min, x_max, y_max] = dent.bbox;

                // Calculate percentages for responsive positioning
                const left = (x_min / W) * 100;
                const top = (y_min / H) * 100;
                const width = ((x_max - x_min) / W) * 100;
                const height = ((y_max - y_min) / H) * 100;

                // Tooltip content
                const tooltipText = `Dent ${idx + 1}\nDepth: ${dent.max_depth_mm.toFixed(1)}mm\nArea: ${dent.area_cm2.toFixed(1)}cm²`;

                return `
                    <div class="dent-interactive-box" 
                         style="left: ${left}%; top: ${top}%; width: ${width}%; height: ${height}%;"
                         title="${tooltipText}">
                         <span class="dent-label">${idx + 1}</span>
                    </div>
                `;
            }).join('');
        }

        html += `
            <div class="preview-image-container">
                <h5>🎨 Dent Segment Overlay (Hover for Details)</h5>
                <div class="interactive-wrapper" style="position: relative; display: inline-block; width: 100%;">
                    <img src="data:image/png;base64,${previews.overlay}" alt="Overlay" class="viewer-image" style="display: block; width: 100%;">
                    ${interactiveBoxesHtml}
                </div>
            </div>
        `;
    }

    // --- 2. NEW: INSERT REFERENCE RGB HERE ---
    // This places the specific RGB photo right under the overlay
    if (resultData.upload_id) {
        const rgbUrl = `${API_BASE_URL}/containers/${currentContainerId}/panels/${panelName}/rgb?upload_id=${resultData.upload_id}`;

        html += `
            <div class="preview-image-container">
                <h5>📸 Reference RGB (Original)</h5>
                <img src="${rgbUrl}" 
                     alt="Reference RGB" 
                     class="viewer-image"
                     style="max-height: 300px; object-fit: contain;" 
                     onerror="this.parentElement.style.display='none'">
            </div>
        `;
    }

    // Show depth overlay with segmentation (replaces binary mask)
    if (previews.depth_overlay) {
        html += `
            <div class="preview-image-container">
                <h5>📍 Segmentation Overlay on Depth Map</h5>
                <img src="data:image/png;base64,${previews.depth_overlay}" alt="Depth Overlay" class="viewer-image">
            </div>
        `;
    } else if (previews.binary_mask) {
        // Fallback to binary mask if depth overlay not available
        html += `
            <div class="preview-image-container">
                <h5>Binary Mask</h5>
                <img src="data:image/png;base64,${previews.binary_mask}" alt="Binary Mask" class="viewer-image">
            </div>
        `;
    }

    if (previews.prob_mask) {
        html += `
            <div class="preview-image-container">
                <h5>Probability Map</h5>
                <img src="data:image/png;base64,${previews.prob_mask}" alt="Probability Map" class="viewer-image">
            </div>
        `;
    }

    // Get filename for download buttons
    let filename = '';
    try {
        const containerResponse = await fetch(`${API_BASE_URL}/containers/${currentContainerId}`);
        const containerData = await containerResponse.json();
        if (containerData.success) {
            const panelData = containerData.container.panels[panelName];
            if (panelData && panelData.depth_filename) {
                filename = panelData.depth_filename.replace('.npy', '');
            }
        }
    } catch (error) {
        console.error('Error loading filename:', error);
    }

    const preprocessedFilename = filename ? `${filename}_preprocessed.npy` : 'preprocessed.npy';
    const maskFilename = filename ? `${filename}_mask.npy` : 'mask.npy';
    const probFilename = filename ? `${filename}_prob.npy` : 'prob.npy';
    const overlayFilename = filename ? `${filename}_overlay.png` : 'overlay.png';

    // Get upload_id from resultData if available
    const uploadId = resultData.upload_id ? `?upload_id=${resultData.upload_id}` : '';

    // Add download buttons
    html += `
        <div class="download-buttons">
            <a href="${API_BASE_URL}/containers/${currentContainerId}/panels/${panelName}/results/preprocessed${uploadId}" 
               class="download-btn" download="${preprocessedFilename}">Download Preprocessed NPY</a>
            <a href="${API_BASE_URL}/containers/${currentContainerId}/panels/${panelName}/results/mask${uploadId}" 
               class="download-btn" download="${maskFilename}">Download Binary Mask</a>
            <a href="${API_BASE_URL}/containers/${currentContainerId}/panels/${panelName}/results/prob${uploadId}" 
               class="download-btn" download="${probFilename}">Download Probability Map</a>
    `;

    if (resultData.has_overlay) {
        html += `
            <a href="${API_BASE_URL}/containers/${currentContainerId}/panels/${panelName}/results/overlay${uploadId}" 
               class="download-btn" download="${overlayFilename}">Download Overlay</a>
        `;
    }

    html += '</div>';

    viewerContent.innerHTML = html;
    viewer.classList.add('active');
}

// Close Results Viewer
function closeResultsViewer() {
    const viewer = document.getElementById('results-viewer');
    viewer.classList.remove('active');

    // Remove active class from result items
    document.querySelectorAll('.result-item').forEach(item => {
        item.classList.remove('active');
    });
}

// Page Navigation
function showPage(pageId) {
    console.log('Switching to page:', pageId); // Debug log
    const pages = document.querySelectorAll('.page');
    console.log('Found pages:', pages.length); // Debug log

    pages.forEach(page => {
        page.classList.remove('active');
    });

    const targetPage = document.getElementById(pageId);
    if (targetPage) {
        targetPage.classList.add('active');
        console.log('Page switched successfully'); // Debug log
    } else {
        console.error('Page not found:', pageId); // Debug log
    }
}

// Notification System
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        z-index: 10001;
        max-width: 400px;
        animation: slideIn 0.3s ease;
    `;

    if (type === 'success') {
        notification.style.backgroundColor = '#d4edda';
        notification.style.color = '#155724';
        notification.style.border = '1px solid #c3e6cb';
    } else if (type === 'error') {
        notification.style.backgroundColor = '#f8d7da';
        notification.style.color = '#721c24';
        notification.style.border = '1px solid #f5c6cb';
    } else {
        notification.style.backgroundColor = '#d1ecf1';
        notification.style.color = '#0c5460';
        notification.style.border = '1px solid #bee5eb';
    }

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 5000);
}

// Setup History Modal
function setupHistoryModal() {
    const historyBtn = document.getElementById('history-btn');
    const historyModal = document.getElementById('history-modal');
    const closeHistoryBtn = document.getElementById('close-history-modal');

    // Open history modal
    historyBtn.addEventListener('click', async () => {
        historyModal.classList.add('active');
        await loadHistory();
    });

    // Close history modal
    closeHistoryBtn.addEventListener('click', () => {
        historyModal.classList.remove('active');
    });

    // Click outside to close
    historyModal.addEventListener('click', (e) => {
        if (e.target === historyModal) {
            historyModal.classList.remove('active');
        }
    });
}

// Load History
async function loadHistory() {
    const historyContent = document.getElementById('history-content');
    historyContent.innerHTML = '<div class="spinner"></div><p>Loading history...</p>';

    try {
        const response = await fetch(`${API_BASE_URL}/history`);
        const data = await response.json();

        if (data.success) {
            if (data.history.length === 0) {
                historyContent.innerHTML = '<div class="history-empty">No processing history available</div>';
                return;
            }

            let html = '';
            data.history.forEach(item => {
                const status = item.status.toUpperCase();
                const statusClass = status === 'FAIL' ? 'fail' : (status === 'PASS' ? 'pass' : 'unknown');
                const itemClass = status === 'FAIL' ? 'failed' : (status === 'PASS' ? 'passed' : '');
                const timestamp = item.timestamp ? new Date(item.timestamp).toLocaleString() : 'Unknown';
                const thumbnailSrc = item.thumbnail ? `data:image/png;base64,${item.thumbnail}` : '';
                const note = item.note || null;
                // Only show note if dents were detected (num_defects > 0)
                const numDefects = item.num_defects || 0;
                const noteDisplay = (note && numDefects > 0) ? `<div class="history-item-note">${note}</div>` : '';

                html += `
                    <div class="history-item ${itemClass}" onclick="openContainerFromHistory('${item.container_id}', '${item.panel_name}')">
                        <div class="history-item-content">
                            ${thumbnailSrc ? `
                                <div class="history-item-thumbnail">
                                    <img src="${thumbnailSrc}" alt="Result Preview">
                                </div>
                            ` : `
                                <div class="history-item-thumbnail history-item-thumbnail-placeholder">
                                    <span>📷</span>
                                    <p>No Preview</p>
                                </div>
                            `}
                            <div class="history-item-info">
                                <div class="history-item-header">
                                    <div class="history-item-title">
                                        <div class="history-item-container-id">Container ID: ${item.container_name}</div>
                                        <div class="history-item-panel-name">${item.panel_name.charAt(0).toUpperCase() + item.panel_name.slice(1)} Panel</div>
                                    </div>
                                    <span class="history-item-status ${statusClass}">${status}</span>
                                </div>
                                ${noteDisplay}
                                <div class="history-item-details">
                                    <div class="history-item-detail">
                                        <span class="history-item-detail-label">📅 Timestamp:</span>
                                        <span class="history-item-detail-value">${timestamp}</span>
                                    </div>
                                    <div class="history-item-detail">
                                        <span class="history-item-detail-label">🔍 Total Detected Dents:</span>
                                        <span class="history-item-detail-value">${item.num_defects || 0}</span>
                                    </div>
                                    <div class="history-item-detail">
                                        <span class="history-item-detail-label">📏 Area:</span>
                                        <span class="history-item-detail-value">${item.area_cm2 !== null && item.area_cm2 !== undefined ? item.area_cm2.toFixed(2) + ' cm²' : 'N/A'}</span>
                                    </div>
                                    <div class="history-item-detail">
                                        <span class="history-item-detail-label">📊 Max Depth:</span>
                                        <span class="history-item-detail-value">${item.max_depth_mm ? item.max_depth_mm.toFixed(2) + ' mm' : 'N/A'}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });

            historyContent.innerHTML = html;
        } else {
            historyContent.innerHTML = '<div class="history-empty">Error loading history</div>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
        historyContent.innerHTML = '<div class="history-empty">Failed to load history. Make sure the API server is running.</div>';
    }
}

// Open Container from History
window.openContainerFromHistory = async function (containerId, panelName) {
    await openContainer(containerId);

    // Close history modal
    document.getElementById('history-modal').classList.remove('active');

    // Scroll to the specific panel after a short delay
    setTimeout(() => {
        const panelCard = document.querySelector(`.panel-card[data-panel="${panelName}"]`);
        if (panelCard) {
            panelCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
            // Highlight the panel briefly
            panelCard.style.transition = 'box-shadow 0.3s ease';
            panelCard.style.boxShadow = '0 0 20px rgba(31, 119, 180, 0.5)';
            setTimeout(() => {
                panelCard.style.boxShadow = '';
            }, 2000);
        }
    }, 500);
};

// Unified Folder Upload Setup
let folderFiles = {}; // Store parsed folder files

function setupUnifiedUpload() {
    const unifiedUploadBtn = document.getElementById('unified-upload-btn');
    const unifiedModal = document.getElementById('unified-upload-modal');
    const closeUnifiedModalBtn = document.getElementById('close-unified-modal');
    const cancelUnifiedBtn = document.getElementById('cancel-unified-upload');
    const confirmUnifiedBtn = document.getElementById('confirm-unified-upload');
    const folderInput = document.getElementById('folder-input');
    const folderUploadArea = document.getElementById('folder-upload-area');

    // Open unified upload modal
    unifiedUploadBtn.addEventListener('click', () => {
        if (!currentContainerId) {
            showNotification('Please create or select a container first', 'error');
            return;
        }
        openUnifiedUploadModal();
    });

    // Close unified upload modal
    closeUnifiedModalBtn.addEventListener('click', closeUnifiedUploadModal);
    cancelUnifiedBtn.addEventListener('click', closeUnifiedUploadModal);

    // Click outside to close
    unifiedModal.addEventListener('click', (e) => {
        if (e.target === unifiedModal) {
            closeUnifiedUploadModal();
        }
    });

    // Click to browse folder
    folderUploadArea.addEventListener('click', () => {
        folderInput.click();
    });

    // Handle folder selection
    folderInput.addEventListener('change', async (e) => {
        const files = Array.from(e.target.files);
        await handleFolderSelection(files);
    });

    // Drag and drop handlers for folder
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        folderUploadArea.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        folderUploadArea.addEventListener(eventName, () => {
            folderUploadArea.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        folderUploadArea.addEventListener(eventName, () => {
            folderUploadArea.classList.remove('dragover');
        }, false);
    });

    folderUploadArea.addEventListener('drop', async (e) => {
        const dt = e.dataTransfer;

        console.log('Drop event triggered. Checking DataTransfer items...');

        // Try to use FileSystemEntry API to preserve folder structure (modern browsers)
        if (dt.items && dt.items.length > 0) {
            const items = Array.from(dt.items);
            console.log(`Found ${items.length} DataTransfer items`);

            // Check all items to find a directory entry
            let directoryEntry = null;
            for (const item of items) {
                if (item.webkitGetAsEntry) {
                    const entry = item.webkitGetAsEntry();
                    console.log(`Checking entry: ${entry.name}, kind: ${entry.kind}`);

                    // Check if it's a directory by checking available methods or kind
                    const isDirectory = entry && (
                        entry.kind === 'directory' ||
                        entry.createReader ||
                        (!entry.file && entry.createReader)
                    );

                    if (isDirectory) {
                        directoryEntry = entry;
                        console.log(`✓ Found directory entry: ${entry.name}`);
                        break;
                    }
                } else {
                    console.log('webkitGetAsEntry not available on item');
                }
            }

            if (directoryEntry) {
                console.log('Using FileSystemEntry API to traverse directory:', directoryEntry.name);
                // Use FileSystemEntry API to traverse folder structure
                const files = await traverseDirectory(directoryEntry);
                console.log(`Traversed directory, found ${files.length} files with paths`);
                if (files.length > 0) {
                    // Sample some paths for debugging
                    const filesWithPath = files.filter(f => f.webkitRelativePath);
                    const filesWithoutPath = files.filter(f => !f.webkitRelativePath);
                    console.log(`  Files with webkitRelativePath: ${filesWithPath.length}`);
                    console.log(`  Files without webkitRelativePath: ${filesWithoutPath.length}`);
                    files.slice(0, 3).forEach(f => {
                        console.log(`  Sample: ${f.name} -> webkitRelativePath: ${f.webkitRelativePath || 'NOT SET'}`);
                    });
                }
                await handleFolderSelection(files);
                return;
            } else {
                console.warn('⚠️ No directory entry found in DataTransfer items. This means:');
                console.warn('  1. You might have dragged files instead of a folder, OR');
                console.warn('  2. Your browser doesn\'t support FileSystemEntry API for drag-and-drop');
                console.warn('  Falling back to regular file list (won\'t preserve folder structure)');
            }
        }

        // Fallback: Use regular file list (won't preserve folder structure)
        const files = Array.from(dt.files);
        if (files.length > 0) {
            console.warn('⚠️ Fallback: Using regular file list without folder structure');
            console.warn('  All files will appear to be in root folder');

            // Show a clear error message in the UI
            const detailsDiv = document.getElementById('folder-upload-details');
            const statusMessage = document.getElementById('unified-status-message');
            const confirmBtn = document.getElementById('confirm-unified-upload');

            if (detailsDiv && statusMessage) {
                statusMessage.textContent = '⚠️ Drag & drop does not preserve folder structure in your browser';
                statusMessage.className = 'error';
                detailsDiv.innerHTML = `
                    <div style="color: #d32f2f; padding: 1rem; background: #ffebee; border-radius: 0.5rem; margin-top: 1rem;">
                        <strong>⚠️ Folder Structure Not Detected</strong>
                        <p style="margin: 0.5rem 0 0 0;">Drag & drop folders may not preserve folder structure in all browsers.</p>
                        <p style="margin: 0.5rem 0 0 0;"><strong>Solution:</strong> Please use the "Click to browse" button instead, which properly detects folder structure.</p>
                    </div>
                `;
                confirmBtn.disabled = true;
            }

            showNotification('⚠️ Drag & drop cannot preserve folder structure. Please use "Click to browse" button instead.', 'error');
            await handleFolderSelection(files);
        }
    });

    // Confirm unified upload
    let confirmHandler = async () => {
        await confirmUnifiedUpload();
    };
    confirmUnifiedBtn.addEventListener('click', async (e) => {
        if (confirmUnifiedBtn.textContent === 'Close') {
            closeUnifiedUploadModal();
        } else {
            await confirmHandler();
        }
    });
}

// Open Unified Upload Modal
function openUnifiedUploadModal() {
    const modal = document.getElementById('unified-upload-modal');
    const folderUploadArea = document.getElementById('folder-upload-area');
    const folderPlaceholder = document.getElementById('folder-placeholder');
    const progressDiv = document.getElementById('folder-upload-progress');
    const confirmBtn = document.getElementById('confirm-unified-upload');
    const statusMessage = document.getElementById('unified-status-message');
    const detailsDiv = document.getElementById('folder-upload-details');
    const folderInput = document.getElementById('folder-input');

    // Reset state
    folderFiles = {};
    if (folderInput) folderInput.value = '';
    folderPlaceholder.innerHTML = `
        <span class="upload-icon" style="font-size: 3rem;">📁</span>
        <p style="font-size: 1.1rem; margin-top: 1rem;">Drag & drop a folder here</p>
        <p class="upload-hint">or click to browse</p>
    `;
    folderPlaceholder.style.display = 'flex';
    progressDiv.style.display = 'none';
    confirmBtn.disabled = true;
    confirmBtn.textContent = 'Upload All';
    statusMessage.textContent = 'Please select a folder to upload';
    statusMessage.className = 'waiting';
    detailsDiv.innerHTML = '';

    modal.classList.add('active');
}

// Close Unified Upload Modal
function closeUnifiedUploadModal() {
    const modal = document.getElementById('unified-upload-modal');
    modal.classList.remove('active');
    folderFiles = {};
}

// Traverse directory structure using FileSystemEntry API
async function traverseDirectory(entry, path = '') {
    const files = [];

    if (!entry) {
        console.warn('traverseDirectory: entry is null/undefined');
        return files;
    }

    // Log entry info for debugging
    const entryInfo = {
        name: entry.name,
        kind: entry.kind,
        hasFile: !!entry.file,
        hasCreateReader: !!entry.createReader
    };
    console.log(`Traversing entry: ${entry.name}, path: "${path}", kind: ${entry.kind || 'unknown'}, hasFile: ${entryInfo.hasFile}, hasCreateReader: ${entryInfo.hasCreateReader}`);

    // Determine if entry is a directory or file
    // Priority: check kind property first (most reliable)
    const isDirectory = entry.kind === 'directory' || (entry.createReader && !entry.file);
    const isFile = entry.kind === 'file' || (entry.file && !entry.createReader);

    // If kind is not set, try to determine by available methods
    if (!entry.kind) {
        if (entry.createReader) {
            // Has createReader, must be a directory
            // Continue to directory handling below
        } else if (entry.file) {
            // Has file() method, try it
            try {
                const file = await new Promise((resolve, reject) => {
                    entry.file(resolve, reject);
                });
                const relativePath = path ? `${path}/${file.name}` : file.name;
                file.webkitRelativePath = relativePath;
                files.push(file);
                console.log(`  ✓ File: ${relativePath}`);
                return files;
            } catch (error) {
                console.warn(`  ✗ Failed to read as file: ${entry.name}`, error);
                // If file() fails, might be a directory, continue to directory check
            }
        } else {
            console.warn(`  ⚠ Unknown entry type: ${entry.name}`);
            return files;
        }
    }

    // Handle as directory
    if (isDirectory || entry.createReader) {
        try {
            const reader = entry.createReader();
            const entries = await new Promise((resolve, reject) => {
                const dirEntries = [];
                const readEntries = () => {
                    reader.readEntries((entries) => {
                        if (entries.length === 0) {
                            resolve(dirEntries);
                        } else {
                            dirEntries.push(...entries);
                            readEntries(); // Continue reading until no more entries
                        }
                    }, reject);
                };
                readEntries();
            });

            console.log(`  → Directory "${entry.name}" contains ${entries.length} entries`);

            // Process all entries recursively
            for (const subEntry of entries) {
                // Skip hidden files like .DS_Store
                if (subEntry.name.startsWith('.')) {
                    console.log(`  ⊘ Skipping hidden: ${subEntry.name}`);
                    continue;
                }

                const subPath = path ? `${path}/${subEntry.name}` : subEntry.name;
                const subFiles = await traverseDirectory(subEntry, subPath);
                files.push(...subFiles);
            }
        } catch (error) {
            console.error(`Error reading directory entry "${entry.name}":`, error);
        }
    }
    // Handle as file
    else if (isFile && entry.file) {
        try {
            const file = await new Promise((resolve, reject) => {
                entry.file(resolve, reject);
            });
            const relativePath = path ? `${path}/${file.name}` : file.name;
            file.webkitRelativePath = relativePath;
            files.push(file);
            console.log(`  ✓ File: ${relativePath}`);
        } catch (error) {
            console.warn(`Error reading file entry "${entry.name}":`, error);
        }
    }

    return files;
}

// Handle Folder Selection
async function handleFolderSelection(files) {
    if (!files || files.length === 0) return;

    // Map folder names to panel names
    const folderToPanelMap = {
        'back': 'back',
        'left_wall': 'left',
        'right_wall': 'right',
        'roof': 'roof',
        'door': 'door'
    };

    // Valid folder names (normalized to lowercase)
    const validFolderNames = Object.keys(folderToPanelMap).map(f => f.toLowerCase());

    // Parse folder structure
    const parsedFiles = {};

    // Check if files have webkitRelativePath (from folder input) or not (from drag and drop)
    const hasWebkitPath = files.some(file => file.webkitRelativePath);

    // Warn if no webkitRelativePath (means FileSystemEntry API wasn't used for drag & drop)
    if (!hasWebkitPath && files.length > 0) {
        console.warn('⚠️ WARNING: Files do not have webkitRelativePath. This means:');
        console.warn('  1. Either you dragged files (not a folder), OR');
        console.warn('  2. The FileSystemEntry API is not working, and folder structure is lost');
        console.warn('  For best results, please use "Click to browse" to select the folder.');
    }

    // Debug: Log ALL file paths to understand structure
    console.log('Parsing folder structure. Total files:', files.length);
    console.log('Files with webkitRelativePath:', files.filter(f => f.webkitRelativePath).length);
    console.log('=== ALL FILE PATHS ===');
    const allPaths = [];
    const folderStructure = new Set();

    files.forEach((file, idx) => {
        const path = file.webkitRelativePath || file.name || 'N/A';
        const pathParts = path.split('/').filter(p => p.length > 0);
        allPaths.push({
            index: idx + 1,
            name: file.name,
            path: path,
            parts: pathParts,
            hasSubfolder: pathParts.length >= 2
        });

        // Track folder structure
        if (pathParts.length >= 1) {
            folderStructure.add(pathParts[0].toLowerCase());
        }
    });

    console.table(allPaths);
    console.log('=== FOLDER STRUCTURE FOUND ===');
    console.log('First-level folders/directories:', Array.from(folderStructure).sort());
    console.log('Expected folders:', validFolderNames);

    // Check if files are in subfolders or directly in root
    const filesInSubfolders = files.filter(file => {
        const path = file.webkitRelativePath || '';
        return path.includes('/') && path.split('/').length >= 2;
    });
    const filesInRoot = files.filter(file => {
        const path = file.webkitRelativePath || '';
        return !path.includes('/') || path.split('/').length === 1;
    });

    console.log(`Files in subfolders: ${filesInSubfolders.length}`);
    console.log(`Files in root: ${filesInRoot.length}`);

    if (filesInSubfolders.length === 0 && hasWebkitPath) {
        // Files are directly in root, not in subfolders
        console.warn('⚠️ WARNING: No files found in subfolders! All files appear to be in the root folder.');
        console.warn('Please make sure you select the PARENT folder that contains subfolders (back/, door/, etc.)');
    }

    files.forEach(file => {
        let folderName = null;

        if (hasWebkitPath && file.webkitRelativePath) {
            // Get relative path from webkitRelativePath
            // Path could be: "folder/file" or "parent_folder/subfolder/file"
            const path = file.webkitRelativePath;
            const pathParts = path.split('/').filter(part => part.length > 0); // Remove empty parts

            console.log(`Processing file: ${file.name}, path: ${path}, pathParts: [${pathParts.join(', ')}]`);

            // Path must have at least 2 parts (subfolder/file)
            if (pathParts.length < 2) {
                console.log(`Skipping file ${file.name}: path has only ${pathParts.length} part(s), need at least 2 (subfolder/file)`);
                return;
            }

            // Find which part matches a valid folder name
            // Usually it's the first or second part depending on whether the selected folder name is included
            for (let i = 0; i < pathParts.length - 1; i++) {
                const part = pathParts[i].toLowerCase();
                console.log(`  Checking path part ${i}: "${part}"`);
                if (validFolderNames.includes(part)) {
                    folderName = part;
                    console.log(`  ✓ Found matching folder: "${folderName}"`);
                    break;
                }
            }
        } else {
            // For drag and drop, try to infer from file name or path
            // This is less reliable, but we'll try
            const fileName = file.name.toLowerCase();
            // Check if filename starts with a known folder name
            for (const folder of validFolderNames) {
                if (fileName.startsWith(folder + '_') || fileName.startsWith(folder + '/')) {
                    folderName = folder;
                    break;
                }
            }
        }

        if (!folderName) {
            console.log(`Skipping file ${file.name}: no matching folder name found in path`);
            return;
        }

        const fileName = file.name || (file.webkitRelativePath ? file.webkitRelativePath.split('/').pop() : '');
        const extension = fileName.split('.').pop().toLowerCase();

        // Map folder name to panel name
        const panelName = folderToPanelMap[folderName];
        if (!panelName) {
            console.log(`Skipping file ${file.name}: folder name "${folderName}" not mapped to a panel`);
            return; // Skip unknown folders
        }

        if (!parsedFiles[panelName]) {
            parsedFiles[panelName] = { npy: null, png: null };
        }

        if (extension === 'npy') {
            parsedFiles[panelName].npy = file;
            console.log(`Found .npy file for ${panelName} panel: ${fileName}`);
        } else if (extension === 'png' || extension === 'jpg' || extension === 'jpeg') {
            parsedFiles[panelName].png = file;
            console.log(`Found .png file for ${panelName} panel: ${fileName}`);
        }
    });

    folderFiles = parsedFiles;

    // Update UI
    const folderPlaceholder = document.getElementById('folder-placeholder');
    const statusMessage = document.getElementById('unified-status-message');
    const confirmBtn = document.getElementById('confirm-unified-upload');
    const detailsDiv = document.getElementById('folder-upload-details');

    // Check which panels have files
    const panelsWithFiles = Object.keys(parsedFiles).filter(panel => parsedFiles[panel].npy);
    const missingPanels = Object.keys(folderToPanelMap).filter(folder => {
        const panelName = folderToPanelMap[folder];
        return !parsedFiles[panelName] || !parsedFiles[panelName].npy;
    }).map(folder => folder);

    // Debug: Log parsed results
    console.log('Parsed files summary:', parsedFiles);
    console.log('Panels with files:', panelsWithFiles);
    console.log('Missing panels:', missingPanels);

    if (panelsWithFiles.length === 0) {
        // Show more detailed error information
        let debugInfo = '';
        if (files.length > 0) {
            // Show all unique folder structures found
            const uniquePaths = new Set();
            const rootFiles = [];
            const subfolderFiles = [];

            files.forEach(f => {
                const path = f.webkitRelativePath || f.name;
                const parts = path.split('/').filter(p => p.length > 0);
                const pathStr = path;

                if (parts.length <= 1) {
                    rootFiles.push(pathStr);
                } else {
                    subfolderFiles.push(pathStr);
                    uniquePaths.add(parts[0]); // Track first-level folders
                }
            });

            let pathsDisplay = '';
            if (rootFiles.length > 0) {
                pathsDisplay += `<strong>Files in root folder (${rootFiles.length}):</strong><br>${rootFiles.slice(0, 10).map(p => `  • ${p}`).join('<br>')}`;
                if (rootFiles.length > 10) {
                    pathsDisplay += `<br>  ... and ${rootFiles.length - 10} more`;
                }
            }

            if (subfolderFiles.length > 0) {
                if (pathsDisplay) pathsDisplay += '<br><br>';
                pathsDisplay += `<strong>Files in subfolders (${subfolderFiles.length}):</strong><br>${subfolderFiles.slice(0, 10).map(p => `  • ${p}`).join('<br>')}`;
                if (subfolderFiles.length > 10) {
                    pathsDisplay += `<br>  ... and ${subfolderFiles.length - 10} more`;
                }
            }

            if (uniquePaths.size > 0) {
                pathsDisplay += `<br><br><strong>First-level folders found:</strong> ${Array.from(uniquePaths).sort().join(', ')}`;
            }

            debugInfo = `<br><br><small style="color: #666; text-align: left; display: block; max-width: 700px; margin: 0.5rem auto; font-family: monospace; font-size: 0.85rem; max-height: 300px; overflow-y: auto; padding: 0.5rem; background: #f5f5f5; border-radius: 0.25rem;">${pathsDisplay}</small>`;

            // Check if files are in root (not in subfolders)
            const allInRoot = subfolderFiles.length === 0 && rootFiles.length > 0;

            if (allInRoot) {
                debugInfo += `<br><small style="color: #ff9800; display: block; margin-top: 0.5rem;"><strong>⚠️ Issue detected:</strong> All files are in the root folder, none in subfolders. Please select the <strong>parent folder</strong> that contains subfolders like "back/", "door/", etc.</small>`;
            } else if (subfolderFiles.length > 0 && uniquePaths.size > 0) {
                const expectedFolders = validFolderNames;
                const foundFolders = Array.from(uniquePaths).map(f => f.toLowerCase());
                const missingExpected = expectedFolders.filter(f => !foundFolders.includes(f));
                if (missingExpected.length > 0) {
                    debugInfo += `<br><small style="color: #ff9800; display: block; margin-top: 0.5rem;"><strong>⚠️ Missing expected folders:</strong> ${missingExpected.join(', ')}</small>`;
                }
            }
        }

        statusMessage.textContent = 'No valid files found. Please check folder structure.';
        statusMessage.className = 'error';
        confirmBtn.disabled = true;
        folderPlaceholder.innerHTML = `
            <span class="upload-icon" style="font-size: 3rem;">❌</span>
            <p style="font-size: 1.1rem; margin-top: 1rem; color: #d32f2f;">Invalid folder structure</p>
            <p class="upload-hint">Expected subfolders: back/, door/, roof/, left_wall/, right_wall/<br>Each subfolder should contain a .npy file${debugInfo}</p>
        `;
        detailsDiv.innerHTML = `<p style="color: #d32f2f;">No valid .npy files found in expected subfolders. Please check your folder structure.</p>`;
        return;
    }

    // Show parsed files
    let detailsHtml = '<strong>Files found:</strong><ul style="text-align: left; margin-top: 0.5rem;">';
    panelsWithFiles.forEach(panel => {
        const npyName = parsedFiles[panel].npy.name;
        const pngName = parsedFiles[panel].png ? parsedFiles[panel].png.name : 'None';
        detailsHtml += `<li><strong>${panel}:</strong> ${npyName}${parsedFiles[panel].png ? ` + ${pngName}` : ''}</li>`;
    });
    detailsHtml += '</ul>';

    if (missingPanels.length > 0) {
        detailsHtml += `<p style="color: #ff9800; margin-top: 0.5rem;">Missing: ${missingPanels.join(', ')}</p>`;
    }

    detailsDiv.innerHTML = detailsHtml;
    folderPlaceholder.innerHTML = `
        <span class="upload-icon" style="font-size: 3rem;">✅</span>
        <p style="font-size: 1.1rem; margin-top: 1rem; color: #2e7d32;">Folder parsed successfully</p>
        <p class="upload-hint">${panelsWithFiles.length} panel(s) ready to upload</p>
    `;

    statusMessage.textContent = `Ready to upload ${panelsWithFiles.length} panel(s)`;
    statusMessage.className = 'ready';
    confirmBtn.disabled = false;
}

// Confirm Unified Upload
async function confirmUnifiedUpload() {
    if (!currentContainerId || Object.keys(folderFiles).length === 0) return;

    const confirmBtn = document.getElementById('confirm-unified-upload');
    const progressDiv = document.getElementById('folder-upload-progress');
    const statusMessage = document.getElementById('unified-status-message');
    const detailsDiv = document.getElementById('folder-upload-details');

    confirmBtn.disabled = true;
    progressDiv.style.display = 'block';
    statusMessage.textContent = 'Uploading files...';

    const panelNames = Object.keys(folderFiles);
    let successCount = 0;
    let failCount = 0;
    const errors = [];

    try {
        // Upload files for each panel
        for (const panelName of panelNames) {
            const files = folderFiles[panelName];
            if (!files.npy) continue;

            detailsDiv.innerHTML = `<p>Uploading ${panelName} panel...</p>`;

            try {
                const formData = new FormData();
                formData.append('depth_file', files.npy);
                if (files.png) {
                    formData.append('rgb_file', files.png);
                }
                // Use default RANSAC settings
                formData.append('use_ransac', 'true');
                formData.append('force_rectangular_mask', 'true');

                const response = await fetch(`${API_BASE_URL}/containers/${currentContainerId}/panels/${panelName}/upload`, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
                    throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                if (data.success) {
                    successCount++;

                    // Add new file pair to the panel's file pairs array
                    const newFilePair = {
                        upload_id: data.upload_id,
                        depth_filename: data.depth_filename || null,
                        depth_shape: data.depth_shape || [],
                        rgb_filename: data.rgb_filename || null,
                        rgb_shape: data.rgb_shape || null,
                        uploaded_at: new Date().toISOString()
                    };

                    // Initialize array if it doesn't exist
                    if (!panelFilePairs[panelName]) {
                        panelFilePairs[panelName] = [];
                    }

                    // Append the new file pair
                    panelFilePairs[panelName].push(newFilePair);

                    // Update panel display
                    await updatePanelDisplay(panelName);
                } else {
                    failCount++;
                    errors.push(`${panelName}: ${data.error || 'Upload failed'}`);
                }
            } catch (error) {
                console.error(`Error uploading ${panelName}:`, error);
                failCount++;
                errors.push(`${panelName}: ${error.message}`);
            }
        }

        // Show results
        if (successCount > 0) {
            showNotification(`Successfully uploaded ${successCount} panel(s)`, 'success');
            statusMessage.textContent = `Upload complete! ${successCount} panel(s) uploaded.`;
            statusMessage.className = 'success';
        }

        if (failCount > 0) {
            showNotification(`${failCount} panel(s) failed to upload`, 'error');
            if (errors.length > 0) {
                statusMessage.textContent = `Upload completed with errors: ${errors.join('; ')}`;
                statusMessage.className = 'error';
            }
        }

        detailsDiv.innerHTML = `
            <p><strong>Results:</strong></p>
            <ul style="text-align: left; margin-top: 0.5rem;">
                <li style="color: #2e7d32;">✓ Success: ${successCount}</li>
                <li style="color: #d32f2f;">✗ Failed: ${failCount}</li>
            </ul>
        `;

        // Enable close button after a delay
        setTimeout(() => {
            confirmBtn.textContent = 'Close';
            confirmBtn.disabled = false;
        }, 2000);

    } catch (error) {
        console.error('Error in unified upload:', error);
        showNotification('Failed to upload files', 'error');
        statusMessage.textContent = `Error: ${error.message}`;
        statusMessage.className = 'error';
        confirmBtn.disabled = false;
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }

    /* Interactive Dent Map */
    .dent-interactive-box {
        position: absolute;
        border: 2px solid transparent; /* Invisible by default */
        cursor: help;
        transition: all 0.2s ease;
        z-index: 10;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* Show box on hover */
    .dent-interactive-box:hover {
        border-color: #FFEB3B; /* Bright Yellow border */
        background-color: rgba(255, 235, 59, 0.2); /* Semi-transparent yellow fill */
        box-shadow: 0 0 8px rgba(0,0,0,0.5);
    }

    /* The little number label inside */
    .dent-label {
        font-size: 10px;
        color: white;
        background: rgba(0,0,0,0.7);
        padding: 1px 4px;
        border-radius: 4px;
        opacity: 0; /* Hidden by default */
        transition: opacity 0.2s;
        pointer-events: none;
    }

    .dent-interactive-box:hover .dent-label {
        opacity: 1; /* Show number on hover */
    }
`;
document.head.appendChild(style);

// Add to your style block at the bottom of app.js
const ransacStyle = `
    #ransac-preview {
        transition: all 0.3s ease;
        border-left: 3px solid #4CAF50; /* Green accent line to show it belongs to enabled option */
    }
    .ransac-stat-item {
        background: white;
        padding: 5px 10px;
        border-radius: 4px;
        border: 1px solid #eee;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .ransac-stat-label { font-weight: bold; color: #555; font-size: 0.8em; }
    .ransac-stat-value { color: #2196F3; font-weight: bold; }
`;
document.querySelector('style').textContent += ransacStyle;