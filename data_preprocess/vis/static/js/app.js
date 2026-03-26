// 全局状态
const state = {
    currentTask: '',
    currentSubType: '',
    currentSplitType: '',
    currentImageCountCategory: '',
    currentFilter: 'all',  // 新增：筛选类型
    currentPage: 1,
    totalPages: 1,
    totalSamples: 0,
    isLoading: false,
    currentSamples: [],  // 当前页的样本
    spatialHasSubTypes: false  // 后端检测到的 spatial 是否有 sub_type 层
};

// DOM 元素
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

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    // 从URL参数初始化状态
    initializeFromURL();
    loadTasks().then(() => {
        // 在加载完任务列表后，恢复选择状态
        restoreStateFromURL();
    });
    setupEventListeners();
    setupImageModal();
    setupImageClickDelegate();
    setupBackToTop();
});

// 设置图片点击事件委托（只需要绑定一次）
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

// 设置事件监听器
function setupEventListeners() {
    elements.taskSelect.addEventListener('change', handleTaskChange);
    elements.subTypeSelect.addEventListener('change', handleSubTypeChange);
    elements.splitTypeSelect.addEventListener('change', handleSplitTypeChange);
    elements.imageCountSelect.addEventListener('change', handleImageCountChange);
    elements.filterSelect.addEventListener('change', handleFilterChange);
    elements.prevPageBtn.addEventListener('click', () => changePage(-1));
    elements.nextPageBtn.addEventListener('click', () => changePage(1));
}

// 设置返回顶部功能
function setupBackToTop() {
    // 监听滚动事件，显示/隐藏返回顶部按钮
    window.addEventListener('scroll', () => {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        if (scrollTop > 300) {
            elements.backToTopBtn.classList.add('visible');
        } else {
            elements.backToTopBtn.classList.remove('visible');
        }
    });
    
    // 点击返回顶部按钮
    elements.backToTopBtn.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
}

// URL参数管理函数
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

// 从URL参数初始化状态
function initializeFromURL() {
    const params = new URLSearchParams(window.location.search);
    state.currentTask = params.get('task') || '';
    state.currentSubType = params.get('sub_type') || '';
    state.currentSplitType = params.get('split_type') || '';
    state.currentImageCountCategory = params.get('image_count_category') || '';
    state.currentFilter = params.get('filter') || 'all';
    state.currentPage = parseInt(params.get('page') || '1', 10);
}

// 从URL恢复状态（在任务列表加载完成后调用）
async function restoreStateFromURL() {
    // 恢复任务选择
    if (state.currentTask) {
        elements.taskSelect.value = state.currentTask;
        
        // 如果是 spatial，先查询后端是否有 sub_type 层
        if (state.currentTask === 'spatial') {
            await loadSpatialSubTypes();
            
            // 若后端有 sub_type 层，恢复子类型选择
            if (state.spatialHasSubTypes && state.currentSubType) {
                elements.subTypeSelect.value = state.currentSubType;
            }
        }
        
        // 恢复 split type 选择
        if (state.currentSplitType) {
            elements.splitTypeSelect.value = state.currentSplitType;
            
            // 加载 image count categories
            await loadImageCountCategories();
            
            // 恢复 image count category 选择
            if (state.currentImageCountCategory) {
                elements.imageCountSelect.value = state.currentImageCountCategory;
                // 恢复 filter 选择
                if (state.currentFilter) {
                    elements.filterSelect.value = state.currentFilter;
                }
                // 加载样本数据
                await loadSamples(true);
            }
        }
    }
    
    updateCurrentSelection();
}

// 设置图片模态框
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
    
    // 保存模态框引用
    window.imageModal = modal;
}

// 加载任务列表
async function loadTasks() {
    try {
        const response = await fetch('/api/tasks');
        const tasks = await response.json();
        
        elements.taskSelect.innerHTML = '<option value="">请选择...</option>';
        tasks.forEach(task => {
            const option = document.createElement('option');
            option.value = task;
            option.textContent = task.charAt(0).toUpperCase() + task.slice(1);
            elements.taskSelect.appendChild(option);
        });
    } catch (error) {
        showError('加载任务列表失败: ' + error.message);
    }
}

// 处理任务变化
async function handleTaskChange() {
    const task = elements.taskSelect.value;
    
    // 重置所有下级状态
    state.currentTask = task;
    state.currentSubType = '';
    state.currentSplitType = '';
    state.currentImageCountCategory = '';
    state.currentPage = 1;
    state.spatialHasSubTypes = false;
    
    // 重置下级 UI
    elements.splitTypeSelect.value = '';
    elements.imageCountSelect.innerHTML = '<option value="">请选择...</option>';
    elements.subTypeGroup.style.display = 'none';
    elements.subTypeSelect.innerHTML = '<option value="">请选择...</option>';
    
    updateCurrentSelection();
    clearSamples();
    
    if (!task) return;
    
    // 如果是 spatial，先查询后端是否有 sub_type 层
    if (task === 'spatial') {
        await loadSpatialSubTypes();
        // loadSpatialSubTypes 会根据结果决定是否显示 sub_type 选择器
        // 若无 sub_type，此时 split_type 已重置，无需再加载 categories
    }
    // 非 spatial 无需额外操作，等用户选 split_type
}

// 加载 Spatial 子类型
async function loadSpatialSubTypes() {
    try {
        const response = await fetch('/api/spatial_sub_types');
        const subTypes = await response.json();
        
        // 记录后端是否真的有 sub_type
        state.spatialHasSubTypes = subTypes.length > 0;
        
        if (subTypes.length === 0) {
            // filter 布局：无 sub_type 层，隐藏 sub_type 选择器
            elements.subTypeGroup.style.display = 'none';
            elements.subTypeSelect.innerHTML = '<option value="">请选择...</option>';
            state.currentSubType = '';
            return;
        }
        
        // final 布局：显示 sub_type 选择器并填充选项
        elements.subTypeGroup.style.display = 'flex';
        elements.subTypeSelect.innerHTML = '<option value="">请选择...</option>';
        subTypes.forEach(subType => {
            const option = document.createElement('option');
            option.value = subType;
            option.textContent = subType.charAt(0).toUpperCase() + subType.slice(1);
            elements.subTypeSelect.appendChild(option);
        });
    } catch (error) {
        showError('加载子类型列表失败: ' + error.message);
    }
}

// 处理子类型变化
async function handleSubTypeChange() {
    state.currentSubType = elements.subTypeSelect.value;
    // 清空下级
    state.currentImageCountCategory = '';
    state.currentPage = 1;
    
    elements.imageCountSelect.innerHTML = '<option value="">请选择...</option>';
    
    updateCurrentSelection();
    clearSamples();
    
    // 如果有 split type，重新加载 image count categories
    if (state.currentSplitType && state.currentSubType) {
        await loadImageCountCategories();
    }
}

// 处理分割类型变化
async function handleSplitTypeChange() {
    state.currentSplitType = elements.splitTypeSelect.value;
    // 清空下级
    state.currentImageCountCategory = '';
    state.currentPage = 1;
    
    elements.imageCountSelect.innerHTML = '<option value="">请选择...</option>';
    
    updateCurrentSelection();
    clearSamples();
    
    if (!state.currentTask || !state.currentSplitType) return;
    
    // spatial：有 sub_type 层时需要先选 sub_type；无 sub_type 层直接加载
    if (state.currentTask === 'spatial' && state.spatialHasSubTypes) {
        if (state.currentSubType) {
            await loadImageCountCategories();
        }
        // 若还未选 sub_type，等用户选
    } else {
        await loadImageCountCategories();
    }
}

// 加载图像数量类别
async function loadImageCountCategories() {
    if (!state.currentTask || !state.currentSplitType) {
        return;
    }
    
    try {
        let url = `/api/image_count_categories?task=${state.currentTask}&split_type=${state.currentSplitType}`;
        // 只有在后端确认有 sub_type 层，且用户已选 sub_type 时才传入
        if (state.currentTask === 'spatial' && state.spatialHasSubTypes && state.currentSubType) {
            url += `&sub_type=${state.currentSubType}`;
        }
        
        const response = await fetch(url);
        const categories = await response.json();
        
        elements.imageCountSelect.innerHTML = '<option value="">请选择...</option>';
        categories.forEach(category => {
            const option = document.createElement('option');
            option.value = category;
            option.textContent = category;
            elements.imageCountSelect.appendChild(option);
        });
    } catch (error) {
        showError('加载图像数量类别失败: ' + error.message);
    }
}

// 处理图像数量类别变化
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

// 处理筛选类型变化
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

// 加载样本数据
async function loadSamples(reset = false) {
    if (!state.currentTask || !state.currentSplitType || !state.currentImageCountCategory) {
        return;
    }
    
    // 对于 spatial 且后端有 sub_type 层时，必须选择子类型
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
        // 只有后端确认有 sub_type 层且已选时才传入
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
        renderSamples(state.currentSamples, true);  // 每次都是重置显示
        updateCurrentSelection();
        updateURL();
        
        // 滚动到顶部
        window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (error) {
        showError('加载样本数据失败: ' + error.message);
    } finally {
        state.isLoading = false;
        showLoading(false);
        updatePagination();  // 确保在finally中也更新分页状态
    }
}


// 渲染样本
function renderSamples(samples, reset = false) {
    if (samples.length === 0) {
        elements.samplesContainer.innerHTML = '<div style="text-align: center; padding: 40px; color: #999;">暂无数据</div>';
        return;
    }
    
    // 计算当前页的起始索引
    const startIndex = (state.currentPage - 1) * 10 + 1;
    
    // 渲染样本
    const samplesHtml = samples.map((sample, index) => {
        const sampleIndex = startIndex + index;
        return renderSample(sample, sampleIndex);
    }).join('');
    
    elements.samplesContainer.innerHTML = samplesHtml;
    
    // 图片点击事件已经通过事件委托在初始化时绑定，无需再次绑定
}

// 渲染单个样本
function renderSample(sample, index) {
    const uniqueId = sample.unique_id || sample.source_file || `sample-${index}`;
    const imageCount = sample.image_count || (sample.input_images ? sample.input_images.length : 0);
    
    // 渲染输入图片
    const inputImagesHtml = sample.input_images && sample.input_images.length > 0
        ? sample.input_images.map((img, idx) => {
            const imageUrl = `/api/image?path=${encodeURIComponent(img)}`;
            return `
                <div class="image-item">
                    <img src="${imageUrl}" alt="Input ${idx + 1}" loading="lazy" onerror="this.parentElement.classList.add('error'); this.parentElement.innerHTML='<span>图片加载失败</span>';">
                    <div class="image-label">Input ${idx + 1}</div>
                </div>
            `;
        }).join('')
        : '<div style="color: #999; text-align: center; padding: 20px;">无输入图片</div>';
    
    // 渲染输出图片
    const outputImageHtml = sample.output_image
        ? `
            <div class="image-item output-image-item">
                <img src="/api/image?path=${encodeURIComponent(sample.output_image)}" alt="Output" loading="lazy" onerror="this.parentElement.classList.add('error'); this.parentElement.innerHTML='<span>图片加载失败</span>';">
                <div class="image-label">Output</div>
            </div>
        `
        : '<div style="color: #999; text-align: center; padding: 20px;">无输出图片</div>';
    
    // 渲染其他信息
    const metaInfo = [];
    if (sample.guidance_score !== undefined) metaInfo.push(`<span class="score-item">Guidance: ${sample.guidance_score}</span>`);
    if (sample.training_score !== undefined) metaInfo.push(`<span class="score-item">Training: ${sample.training_score}</span>`);
    if (sample.temporal_score !== undefined) metaInfo.push(`<span class="score-item">Temporal: ${sample.temporal_score}</span>`);
    
    // 获取prompt文本（支持多种字段名）
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
            // 对value进行HTML转义，防止<image 1>这样的文本被误认为是HTML标签
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

// HTML 转义
function escapeHtml(text) {
    if (text == null) {
        return '';
    }
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML.replace(/\n/g, '<br>');
}

// 更新当前选择显示
function updateCurrentSelection() {
    elements.currentTask.textContent = state.currentTask || '-';
    elements.currentSubType.textContent = (state.currentTask === 'spatial' && state.spatialHasSubTypes && state.currentSubType) ? state.currentSubType : '-';
    elements.currentSplitType.textContent = state.currentSplitType || '-';
    elements.currentImageCount.textContent = state.currentImageCountCategory || '-';
    elements.totalSamples.textContent = state.totalSamples;
    
    // 更新筛选类型显示
    const filterLabels = {
        'all': '全部',
        'pass': '通过',
        'fail': '未通过'
    };
    elements.currentFilter.textContent = filterLabels[state.currentFilter] || '全部';
}

// 更新分页信息
function updatePagination() {
    elements.pageInfo.textContent = `第 ${state.currentPage} / ${state.totalPages} 页 (共 ${state.totalSamples} 个样本)`;
    elements.prevPageBtn.disabled = state.currentPage <= 1 || state.isLoading;
    elements.nextPageBtn.disabled = state.currentPage >= state.totalPages || state.isLoading;
}

// 切换页面
function changePage(delta) {
    const newPage = state.currentPage + delta;
    if (newPage >= 1 && newPage <= state.totalPages && !state.isLoading) {
        state.currentPage = newPage;
        updateURL();
        loadSamples(true);
    }
}

// 清空样本
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

// 显示/隐藏加载提示
function showLoading(show) {
    elements.loading.style.display = show ? 'block' : 'none';
}

// 显示错误
function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorMessage.style.display = 'block';
}

// 隐藏错误
function hideError() {
    elements.errorMessage.style.display = 'none';
}

