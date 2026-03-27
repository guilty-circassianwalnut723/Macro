// Global state
const state = {
    currentTask: '',
    currentSubType: '',
    currentSplitType: '',
    currentImageCountCategory: '',
    currentFilter: 'all',  // New: filter type
    currentPage: 1,
    totalPages: 1,
    totalSamples: 0,
    isLoading: false,
    currentSamples: [],  // Current page samples
    spatialHasSubTypes: false  // Whether backend detected sub_type layer for spatial
};

// DOM elements
const elements = {
    taskSelect: document.getElementById('task-select'),
    subTypeSelect: document.getElementById('sub-type-select'),
    subTypeGroup: document.getElementById('spatial-sub-type-group'),
    splitTypeSelect: document.getElementById('split-type-select'),
    imageCountSelect: document.getElementById('image-count-select'),
    filterSelect: document.getElementById('filter-select'),
    currentTask: document.getElementById('current-task'),
    currentSubType: document.getElementById('current-sub-type'),
    currentSplitType: document.getElementById('current-split-type'),
    currentImageCount: document.getElementById('current-image-count'),
    currentFilter: document.getElementById('current-filter'),
    totalSamples: document.getElementById('total-samples'),
    loading: document.getElementById('loading'),
    errorMessage: document.getElementById('error-message'),
    samplesContainer: document.getElementById('samples-container'),
    prevPageBtn: document.getElementById('prev-page'),
    nextPageBtn: document.getElementById('next-page'),
    pageInfo: document.getElementById('page-info'),
    backToTopBtn: document.getElementById('back-to-top')
};

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    // Initialize state from URL parameters
    initializeFromURL();
    loadTasks().then(() => {
        // Restore selection state after task list is loaded
        restoreStateFromURL();
    });
    setupEventListeners();
    setupImageModal();
    setupImageClickDelegate();
    setupBackToTop();
});

// Set up image click event delegation (only needs to be bound once)
function setupImageClickDelegate() {
    elements.samplesContainer.addEventListener('click', (e) => {
        if (e.target.tagName === 'IMG' && e.target.closest('.image-item')) {
            const src = e.target.src;
            const modalImg = window.imageModal.querySelector('img');
            modalImg.src = src;
            window.imageModal.classList.add('active');
        }
    });
}

// Set up event listeners
function setupEventListeners() {
    elements.taskSelect.addEventListener('change', handleTaskChange);
    elements.subTypeSelect.addEventListener('change', handleSubTypeChange);
    elements.splitTypeSelect.addEventListener('change', handleSplitTypeChange);
    elements.imageCountSelect.addEventListener('change', handleImageCountChange);
    elements.filterSelect.addEventListener('change', handleFilterChange);
    elements.prevPageBtn.addEventListener('click', () => changePage(-1));
    elements.nextPageBtn.addEventListener('click', () => changePage(1));
}

// Set up back-to-top functionality
function setupBackToTop() {
    // Listen for scroll event, show/hide back-to-top button
    window.addEventListener('scroll', () => {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        if (scrollTop > 300) {
            elements.backToTopBtn.classList.add('visible');
        } else {
            elements.backToTopBtn.classList.remove('visible');
        }
    });
    
    // Click back-to-top button
    elements.backToTopBtn.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
}

// URL parameter management functions
function updateURL() {
    const params = new URLSearchParams();
    if (state.currentTask) params.set('task', state.currentTask);
    if (state.currentSubType) params.set('sub_type', state.currentSubType);
    if (state.currentSplitType) params.set('split_type', state.currentSplitType);
    if (state.currentImageCountCategory) params.set('image_count_category', state.currentImageCountCategory);
    if (state.currentFilter !== 'all') params.set('filter', state.currentFilter);
    if (state.currentPage > 1) params.set('page', state.currentPage);
    
    const newURL = params.toString() ? `${window.location.pathname}?${params.toString()}` : window.location.pathname;
    window.history.pushState({}, '', newURL);
}

// Initialize state from URL parameters
function initializeFromURL() {
    const params = new URLSearchParams(window.location.search);
    state.currentTask = params.get('task') || '';
    state.currentSubType = params.get('sub_type') || '';
    state.currentSplitType = params.get('split_type') || '';
    state.currentImageCountCategory = params.get('image_count_category') || '';
    state.currentFilter = params.get('filter') || 'all';
    state.currentPage = parseInt(params.get('page') || '1', 10);
}

// Restore state from URL (called after task list is loaded)
async function restoreStateFromURL() {
    // Restore task selection
    if (state.currentTask) {
        elements.taskSelect.value = state.currentTask;
        
        // If spatial, first query backend whether sub_type layer exists
        if (state.currentTask === 'spatial') {
            await loadSpatialSubTypes();
            
            // If backend has sub_type layer, restore subtype selection
            if (state.spatialHasSubTypes && state.currentSubType) {
                elements.subTypeSelect.value = state.currentSubType;
            }
        }
        
        // Restore split type selection
        if (state.currentSplitType) {
            elements.splitTypeSelect.value = state.currentSplitType;
            
            // Load image count categories
            await loadImageCountCategories();
            
            // Restore image count category selection
            if (state.currentImageCountCategory) {
                elements.imageCountSelect.value = state.currentImageCountCategory;
                // Restore filter selection
                if (state.currentFilter) {
                    elements.filterSelect.value = state.currentFilter;
                }
                // Load sample data
                await loadSamples(true);
            }
        }
    }
    
    updateCurrentSelection();
}

// Set up image modal
function setupImageModal() {
    const modal = document.createElement('div');
    modal.className = 'image-modal';
    modal.innerHTML = '<span class="close-btn">&times;</span><img src="" alt="">';
    document.body.appendChild(modal);
    
    modal.addEventListener('click', (e) => {
        if (e.target === modal || e.target.classList.contains('close-btn')) {
            modal.classList.remove('active');
        }
    });
    
    // Save modal reference
    window.imageModal = modal;
}

// Load task list
async function loadTasks() {
    try {
        const response = await fetch('/api/tasks');
        const tasks = await response.json();
        
        elements.taskSelect.innerHTML = '<option value="">Please select...</option>';
        tasks.forEach(task => {
            const option = document.createElement('option');
            option.value = task;
            option.textContent = task.charAt(0).toUpperCase() + task.slice(1);
            elements.taskSelect.appendChild(option);
        });
    } catch (error) {
        showError('Failed to load task list: ' + error.message);
    }
}

// Handle task change
async function handleTaskChange() {
    const task = elements.taskSelect.value;
    
    // Reset all downstream state
    state.currentTask = task;
    state.currentSubType = '';
    state.currentSplitType = '';
    state.currentImageCountCategory = '';
    state.currentPage = 1;
    state.spatialHasSubTypes = false;
    
    // Reset downstream UI
    elements.splitTypeSelect.value = '';
    elements.imageCountSelect.innerHTML = '<option value="">Please select...</option>';
    elements.subTypeGroup.style.display = 'none';
    elements.subTypeSelect.innerHTML = '<option value="">Please select...</option>';
    
    updateCurrentSelection();
    clearSamples();
    
    if (!task) return;
    
    // If spatial, first query backend whether sub_type layer exists
    if (task === 'spatial') {
        await loadSpatialSubTypes();
        // loadSpatialSubTypes will decide whether to show sub_type selector based on result
        // If no sub_type, split_type has been reset, no need to load categories
    }
    // Non-spatial needs no extra action, wait for user to select split_type
}

// Load Spatial subtypes
async function loadSpatialSubTypes() {
    try {
        const response = await fetch('/api/spatial_sub_types');
        const subTypes = await response.json();
        
        // Record whether backend actually has sub_type
        state.spatialHasSubTypes = subTypes.length > 0;
        
        if (subTypes.length === 0) {
            // filter layout: no sub_type layer, hide sub_type selector
            elements.subTypeGroup.style.display = 'none';
            elements.subTypeSelect.innerHTML = '<option value="">Please select...</option>';
            state.currentSubType = '';
            return;
        }
        
        // final layout: show sub_type selector and fill options
        elements.subTypeGroup.style.display = 'flex';
        elements.subTypeSelect.innerHTML = '<option value="">Please select...</option>';
        subTypes.forEach(subType => {
            const option = document.createElement('option');
            option.value = subType;
            option.textContent = subType.charAt(0).toUpperCase() + subType.slice(1);
            elements.subTypeSelect.appendChild(option);
        });
    } catch (error) {
        showError('Failed to load subtype list: ' + error.message);
    }
}

// Handle subtype change
async function handleSubTypeChange() {
    state.currentSubType = elements.subTypeSelect.value;
    // Clear downstream
    state.currentImageCountCategory = '';
    state.currentPage = 1;
    
    elements.imageCountSelect.innerHTML = '<option value="">Please select...</option>';
    
    updateCurrentSelection();
    clearSamples();
    
    // If split type is set, reload image count categories
    if (state.currentSplitType && state.currentSubType) {
        await loadImageCountCategories();
    }
}

// Handle split type change
async function handleSplitTypeChange() {
    state.currentSplitType = elements.splitTypeSelect.value;
    // Clear downstream
    state.currentImageCountCategory = '';
    state.currentPage = 1;
    
    elements.imageCountSelect.innerHTML = '<option value="">Please select...</option>';
    
    updateCurrentSelection();
    clearSamples();
    
    if (!state.currentTask || !state.currentSplitType) return;
    
    // spatial: if sub_type layer exists, sub_type must be selected first; otherwise load directly
    if (state.currentTask === 'spatial' && state.spatialHasSubTypes) {
        if (state.currentSubType) {
            await loadImageCountCategories();
        }
        // If sub_type not yet selected, wait for user
    } else {
        await loadImageCountCategories();
    }
}

// Load image count categories
async function loadImageCountCategories() {
    if (!state.currentTask || !state.currentSplitType) {
        return;
    }
    
    try {
        let url = `/api/image_count_categories?task=${state.currentTask}&split_type=${state.currentSplitType}`;
        // Only pass sub_type when backend confirmed sub_type layer exists and user has selected it
        if (state.currentTask === 'spatial' && state.spatialHasSubTypes && state.currentSubType) {
            url += `&sub_type=${state.currentSubType}`;
        }
        
        const response = await fetch(url);
        const categories = await response.json();
        
        elements.imageCountSelect.innerHTML = '<option value="">Please select...</option>';
        categories.forEach(category => {
            const option = document.createElement('option');
            option.value = category;
            option.textContent = category;
            elements.imageCountSelect.appendChild(option);
        });
    } catch (error) {
        showError('Failed to load image count categories: ' + error.message);
    }
}

// Handle image count category change
async function handleImageCountChange() {
    state.currentImageCountCategory = elements.imageCountSelect.value;
    state.currentPage = 1;
    updateCurrentSelection();
    updateURL();
    
    if (state.currentImageCountCategory) {
        await loadSamples(true);
    } else {
        clearSamples();
    }
}

// Handle filter type change
async function handleFilterChange() {
    state.currentFilter = elements.filterSelect.value;
    state.currentPage = 1;
    updateCurrentSelection();
    updateURL();
    
    if (state.currentImageCountCategory) {
        await loadSamples(true);
    } else {
        clearSamples();
    }
}

// Load sample data
async function loadSamples(reset = false) {
    if (!state.currentTask || !state.currentSplitType || !state.currentImageCountCategory) {
        return;
    }
    
    // For spatial with backend sub_type layer, subtype must be selected
    if (state.currentTask === 'spatial' && state.spatialHasSubTypes && !state.currentSubType) {
        return;
    }
    
    if (state.isLoading) {
        return;
    }
    
    state.isLoading = true;
    showLoading(true);
    hideError();
    
    try {
        let url = `/api/samples?task=${state.currentTask}&split_type=${state.currentSplitType}&image_count_category=${state.currentImageCountCategory}&page=${state.currentPage}&filter=${state.currentFilter}`;
        // Only pass sub_type when backend confirmed sub_type layer and it is selected
        if (state.currentTask === 'spatial' && state.spatialHasSubTypes && state.currentSubType) {
            url += `&sub_type=${state.currentSubType}`;
        }
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
            state.isLoading = false;
            showLoading(false);
            updatePagination();
            return;
        }
        
        state.currentPage = data.current_page;
        state.totalPages = data.total_pages;
        state.totalSamples = data.total_samples;
        state.currentSamples = data.samples;
        
        updatePagination();
        renderSamples(state.currentSamples, true);  // Always reset display
        updateCurrentSelection();
        updateURL();
        
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (error) {
        showError('Failed to load sample data: ' + error.message);
    } finally {
        state.isLoading = false;
        showLoading(false);
        updatePagination();  // Ensure pagination is updated in finally block
    }
}


// Render samples
function renderSamples(samples, reset = false) {
    if (samples.length === 0) {
        elements.samplesContainer.innerHTML = '<div style="text-align: center; padding: 40px; color: #999;">No data available</div>';
        return;
    }
    
    // Calculate start index of current page
    const startIndex = (state.currentPage - 1) * 10 + 1;
    
    // Render samples
    const samplesHtml = samples.map((sample, index) => {
        const sampleIndex = startIndex + index;
        return renderSample(sample, sampleIndex);
    }).join('');
    
    elements.samplesContainer.innerHTML = samplesHtml;
    
    // Image click event was bound via event delegation at initialization, no need to bind again
}

// Render a single sample
function renderSample(sample, index) {
    const uniqueId = sample.unique_id || sample.source_file || `sample-${index}`;
    const imageCount = sample.image_count || (sample.input_images ? sample.input_images.length : 0);
    
    // Render input images
    const inputImagesHtml = sample.input_images && sample.input_images.length > 0
        ? sample.input_images.map((img, idx) => {
            const imageUrl = `/api/image?path=${encodeURIComponent(img)}`;
            return `
                <div class="image-item">
                    <img src="${imageUrl}" alt="Input ${idx + 1}" loading="lazy" onerror="this.parentElement.classList.add('error'); this.parentElement.innerHTML='<span>Image load failed</span>';">
                    <div class="image-label">Input ${idx + 1}</div>
                </div>
            `;
        }).join('')
        : '<div style="color: #999; text-align: center; padding: 20px;">No input images</div>';
    
    // Render output image
    const outputImageHtml = sample.output_image
        ? `
            <div class="image-item output-image-item">
                <img src="/api/image?path=${encodeURIComponent(sample.output_image)}" alt="Output" loading="lazy" onerror="this.parentElement.classList.add('error'); this.parentElement.innerHTML='<span>Image load failed</span>';">
                <div class="image-label">Output</div>
            </div>
        `
        : '<div style="color: #999; text-align: center; padding: 20px;">No output image</div>';
    
    // Render other info
    const metaInfo = [];
    if (sample.guidance_score !== undefined) metaInfo.push(`<span class="score-item">Guidance: ${sample.guidance_score}</span>`);
    if (sample.training_score !== undefined) metaInfo.push(`<span class="score-item">Training: ${sample.training_score}</span>`);
    if (sample.temporal_score !== undefined) metaInfo.push(`<span class="score-item">Temporal: ${sample.temporal_score}</span>`);
    
    // Get prompt text (supports multiple field names)
    const promptText = sample.text || sample.instruction || sample.prompt || '';
    
    const infoHtml = Object.entries(sample)
        .filter(([key]) => !['input_images', 'output_image', 'text', 'instruction', 'prompt', 'unique_id', 'source_file', 'source_line', 'true_index'].includes(key))
        .filter(([key]) => !key.startsWith('_'))
        .map(([key, value]) => {
            if (Array.isArray(value)) {
                value = value.map(v => typeof v === 'boolean' ? (v ? '✓' : '✗') : v).join(', ');
            } else if (typeof value === 'boolean') {
                value = value ? '✓' : '✗';
            } else if (typeof value === 'object' && value !== null) {
                value = JSON.stringify(value);
            }
            // HTML-escape the value to prevent text like <image 1> from being treated as HTML tags
            const escapedKey = escapeHtml(String(key));
            const escapedValue = escapeHtml(String(value));
            return `<div class="info-row"><span class="info-label">${escapedKey}:</span><span class="info-value">${escapedValue}</span></div>`;
        }).join('');
    
    return `
        <div class="sample-card">
            <div class="sample-header">
                <div class="sample-id">#${index} - ${uniqueId}</div>
                <div class="sample-meta">
                    <div class="meta-item"><strong>Image Count:</strong> ${imageCount}</div>
                    ${metaInfo.join('')}
                </div>
            </div>
            ${promptText ? `<div class="sample-text">${escapeHtml(promptText)}</div>` : ''}
            <div class="images-section">
                <div class="section-title">Input Images:</div>
                <div class="images-grid">${inputImagesHtml}</div>
            </div>
            <div class="images-section">
                <div class="section-title">Output Image:</div>
                <div class="output-image-container">${outputImageHtml}</div>
            </div>
            ${infoHtml ? `<div class="sample-info">${infoHtml}</div>` : ''}
        </div>
    `;
}

// HTML escaping
function escapeHtml(text) {
    if (text == null) {
        return '';
    }
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML.replace(/\n/g, '<br>');
}

// Update current selection display
function updateCurrentSelection() {
    elements.currentTask.textContent = state.currentTask || '-';
    elements.currentSubType.textContent = (state.currentTask === 'spatial' && state.spatialHasSubTypes && state.currentSubType) ? state.currentSubType : '-';
    elements.currentSplitType.textContent = state.currentSplitType || '-';
    elements.currentImageCount.textContent = state.currentImageCountCategory || '-';
    elements.totalSamples.textContent = state.totalSamples;
    
    // Update filter type display
    const filterLabels = {
        'all': 'All',
        'pass': 'Pass',
        'fail': 'Fail'
    };
    elements.currentFilter.textContent = filterLabels[state.currentFilter] || 'All';
}

// Update pagination info
function updatePagination() {
    elements.pageInfo.textContent = `Page ${state.currentPage} / ${state.totalPages} (total ${state.totalSamples} samples)`;
    elements.prevPageBtn.disabled = state.currentPage <= 1 || state.isLoading;
    elements.nextPageBtn.disabled = state.currentPage >= state.totalPages || state.isLoading;
}

// Switch page
function changePage(delta) {
    const newPage = state.currentPage + delta;
    if (newPage >= 1 && newPage <= state.totalPages && !state.isLoading) {
        state.currentPage = newPage;
        updateURL();
        loadSamples(true);
    }
}

// Clear samples
function clearSamples() {
    elements.samplesContainer.innerHTML = '';
    state.currentPage = 1;
    state.totalPages = 1;
    state.totalSamples = 0;
    state.currentSamples = [];
    state.isLoading = false;
    updatePagination();
    updateCurrentSelection();
    updateURL();
}

// Show/hide loading indicator
function showLoading(show) {
    elements.loading.style.display = show ? 'block' : 'none';
}

// Show error
function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorMessage.style.display = 'block';
}

// Hide error
function hideError() {
    elements.errorMessage.style.display = 'none';
}

