// 全局状态
let currentBaseline = null;  // 当前选择的baseline
let currentTask = null;
let selectedExps = [];  // 支持多选（完整的实验名称列表）
let selectedExpBases = [];  // 选中的实验基础名称列表（多选）
let selectedSteps = [];  // 选中的step列表（多选）
let expStructure = null;  // 实验结构数据 {base_names, steps, exp_map}
let currentCategory = null;  // 当前选择的类别
let sampleIds = [];
let currentPage = 1;
let totalPages = 1;  // 总页数
let currentEvaluator = 'gpt';  // 当前选择的评估器：'gpt' 或 'gemini'
const itemsPerPage = 10;

const API_BASE = '';

// 自然排序函数（处理数字排序，如 exp2 在 exp10 之前）
function naturalSort(arr) {
    return arr.slice().sort((a, b) => {
        // 将字符串分割成文本和数字部分
        const regex = /(\d+)/g;
        const aParts = a.split(regex);
        const bParts = b.split(regex);
        
        const minLength = Math.min(aParts.length, bParts.length);
        
        for (let i = 0; i < minLength; i++) {
            const aPart = aParts[i];
            const bPart = bParts[i];
            
            // 如果都是数字，按数字大小比较
            if (/^\d+$/.test(aPart) && /^\d+$/.test(bPart)) {
                const numA = parseInt(aPart, 10);
                const numB = parseInt(bPart, 10);
                if (numA !== numB) {
                    return numA - numB;
                }
            } else {
                // 否则按字符串比较
                if (aPart !== bPart) {
                    return aPart < bPart ? -1 : 1;
                }
            }
        }
        
        // 如果前面的部分都相同，长度短的排在前面
        return aParts.length - bParts.length;
    });
}

// 获取排序后的实验列表
function getSortedExps(exps) {
    return naturalSort(exps);
}

// 根据选择的实验基础名称和step，组合成完整的实验名称列表
// 注意：Python 的 None key 被 JSON 序列化后变为字符串 "null"，需用 'null' 查找
function combineExpNames() {
    if (!expStructure || selectedExpBases.length === 0) {
        selectedExps = [];
        return;
    }
    
    const combined = [];
    for (const baseName of selectedExpBases) {
        if (expStructure.exp_map[baseName]) {
            // 对于没有step的实验（Python None key 序列化为字符串 'null'），总是加入
            if (expStructure.exp_map[baseName]['null']) {
                combined.push(expStructure.exp_map[baseName]['null']);
            }
            
            // 对于有step的实验，根据选中的steps（多选）组合
            if (selectedSteps.length > 0) {
                for (const step of selectedSteps) {
                    if (expStructure.exp_map[baseName][step]) {
                        combined.push(expStructure.exp_map[baseName][step]);
                    }
                }
            }
        }
    }
    selectedExps = combined;
}

// 从完整的实验名称列表中解析出实验基础名称和step
function parseExpNamesFromFullNames(fullNames) {
    if (!fullNames || fullNames.length === 0) {
        selectedExpBases = [];
        selectedSteps = [];
        return;
    }
    
    // 解析每个完整名称，提取基础名称和step
    const basesSet = new Set();
    const stepsSet = new Set();
    
    for (const fullName of fullNames) {
        // 尝试从expStructure中查找
        if (expStructure && expStructure.exp_map) {
            for (const [baseName, steps] of Object.entries(expStructure.exp_map)) {
                for (const [step, expName] of Object.entries(steps)) {
                    if (expName === fullName) {
                        basesSet.add(baseName);
                        // 收集所有的step（排除null，因为null对应的实验会直接显示）
                        if (step !== null && step !== 'null') {
                            stepsSet.add(step);
                        }
                        break;
                    }
                }
            }
        }
    }
    
    selectedExpBases = Array.from(basesSet);
    selectedSteps = Array.from(stepsSet);
}

// 路由管理
function updateURL(baseline, task, exps, category, page = 1, evaluator = 'gpt') {
    const params = new URLSearchParams();
    if (baseline) params.set('baseline', baseline);
    if (task) params.set('task', task);
    if (exps && exps.length > 0) params.set('exps', exps.join(','));
    if (category) params.set('category', category);
    if (page && page > 1) params.set('page', page);
    // 始终写入 evaluator，确保刷新后能正确恢复（默认 gpt 也写入）
    if (evaluator) params.set('evaluator', evaluator);
    
    const newURL = window.location.pathname + (params.toString() ? '?' + params.toString() : '');
    window.history.pushState({ baseline, task, exps, category, page, evaluator }, '', newURL);
}

function parseURL() {
    const params = new URLSearchParams(window.location.search);
    const expsParam = params.get('exps');
    return {
        baseline: params.get('baseline') || null,
        task: params.get('task') || null,
        exps: expsParam ? expsParam.split(',').filter(e => e) : [],
        category: params.get('category') || null,
        page: parseInt(params.get('page')) || 1,
        evaluator: params.get('evaluator') || 'gpt'
    };
}

// 从URL加载页面
async function loadFromURL() {
    const { baseline, task, exps, category, page, evaluator } = parseURL();
    currentEvaluator = evaluator;
    updateEvaluatorButtons();
    
    // 首先加载baselines
    const baselines = await loadBaselines();
    
    if (baseline) {
        currentBaseline = baseline;
        // 传入 baselines 参数，确保按钮正确渲染
        renderBaselineButtons(baselines || window.allBaselines || []);
        
        if (task) {
            try {
                const response = await fetch(`${API_BASE}/api/tasks/${baseline}`);
                const tasks = await response.json();
                
                if (tasks.includes(task)) {
                    currentTask = task;
                    renderButtons('task-buttons', tasks || [],  'task', null);
                    
                    // 加载实验结构
                    await loadExps(baseline, task);
                    
                    if (exps && exps.length > 0) {
                        try {
                            // 解析URL中的实验名称，恢复选择的实验基础名称和step
                            parseExpNamesFromFullNames(exps);
                            
                            // 验证并更新UI
                            if (expStructure) {
                                // 验证选中的实验基础名称是否有效
                                const validBases = selectedExpBases.filter(b => expStructure.base_names.includes(b));
                                selectedExpBases = validBases;
                                
                                // 验证选中的steps是否有效
                                const validSteps = selectedSteps.filter(step => 
                                    expStructure.steps && expStructure.steps.includes(step)
                                );
                                selectedSteps = validSteps;
                                
                                // 重新组合实验名称
                                combineExpNames();
                                
                                // 更新UI
                                renderExpBaseButtons(expStructure.base_names);
                                
                                // 检查是否有需要step的实验
                                let hasStepBasedExps = false;
                                if (selectedExpBases.length > 0 && expStructure.steps && expStructure.steps.length > 0) {
                                    for (const base of selectedExpBases) {
                                        if (expStructure.exp_map[base]) {
                                            const steps = Object.keys(expStructure.exp_map[base]).filter(s => s !== 'null' && s !== null);
                                            if (steps.length > 0) {
                                                hasStepBasedExps = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                                
                                if (hasStepBasedExps) {
                                    renderExpStepButtons(expStructure.steps);
                                } else {
                                    if (selectedExpBases.length > 0) {
                                        document.getElementById('exp-step-buttons').innerHTML = '<span class="empty-state">这些实验无需选择step</span>';
                                    } else {
                                        document.getElementById('exp-step-buttons').innerHTML = '<span class="empty-state">请先选择实验名称</span>';
                                    }
                                }
                                
                                // 加载categories
                                if (selectedExps.length > 0) {
                                    const firstExp = selectedExps[0];
                                    const catResponse = await fetch(`${API_BASE}/api/categories/${baseline}/${task}/${firstExp}`);
                                    const categories = await catResponse.json();
                                    renderCategoryButtons(categories);
                                    
                                    if (category && categories.includes(category)) {
                                        currentCategory = category;
                                        currentPage = page;
                                        // 更新 category 按钮 active 状态
                                        document.querySelectorAll('#category-buttons .btn').forEach(btn => {
                                            btn.classList.toggle('active', btn.textContent.trim() === category);
                                        });
                                        await loadData();
                                    } else {
                                        currentCategory = null;
                                    }
                                } else {
                                    hideDataAreas();
                                    document.getElementById('category-buttons').innerHTML = '<span class="empty-state">请先选择实验</span>';
                                }
                            } else {
                                // expStructure 为 null，可能是加载失败
                                console.warn('expStructure is null, cannot load data from URL');
                                hideDataAreas();
                                document.getElementById('category-buttons').innerHTML = '<span class="empty-state">请先选择实验</span>';
                            }
                        } catch (error) {
                            console.error('Error loading exps from URL:', error);
                            hideDataAreas();
                            document.getElementById('category-buttons').innerHTML = '<span class="empty-state">请先选择实验</span>';
                        }
                    } else {
                        // 没有 exps 参数，清空状态
                        hideDataAreas();
                        document.getElementById('category-buttons').innerHTML = '<span class="empty-state">请先选择实验</span>';
                    }
                } else {
                    await loadTasks(baseline);
                }
            } catch (error) {
                console.error('Error loading tasks from URL:', error);
                await loadTasks(baseline);
            }
        } else {
            await loadTasks(baseline);
        }
    }
}

// 隐藏数据区域
function hideDataAreas() {
    const comparisonArea = document.getElementById('comparison-area');
    if (comparisonArea) comparisonArea.style.display = 'none';
    const samplesArea = document.getElementById('samples-area');
    if (samplesArea) samplesArea.style.display = 'none';
    const oursArea = document.getElementById('ours-area');
    if (oursArea) oursArea.style.display = 'none';
    const benchmarkArea = document.getElementById('benchmark-area');
    if (benchmarkArea) benchmarkArea.style.display = 'none';
    const floatingPagination = document.getElementById('floating-pagination');
    if (floatingPagination) floatingPagination.style.display = 'none';
}

// 转义 HTML 并高亮显示 <image X> 标签
function formatPrompt(text) {
    if (!text) return 'N/A';
    
    const escapeHtml = (str) => {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return str.replace(/[&<>"']/g, m => map[m]);
    };
    
    const placeholders = [];
    let placeholderIndex = 0;
    let processed = text.replace(/<image\s+(\d+)>/gi, (match) => {
        const placeholder = `__IMAGE_TAG_${placeholderIndex}__`;
        placeholders.push({ placeholder, tag: match });
        placeholderIndex++;
        return placeholder;
    });
    
    let escaped = escapeHtml(processed);
    
    placeholders.forEach(({ placeholder, tag }) => {
        const escapedTag = escapeHtml(tag);
        escaped = escaped.replace(placeholder, `<span class="image-tag">${escapedTag}</span>`);
    });
    
    return escaped;
}

// 初始化
async function init() {
    window.addEventListener('popstate', (event) => {
        if (event.state) {
            const { baseline, task, exps, category, page, evaluator } = event.state;
            currentBaseline = baseline;
            currentTask = task;
            selectedExps = exps || [];
            currentCategory = category || null;
            currentPage = page || 1;
            currentEvaluator = evaluator || 'gpt';
            loadFromURL();
        } else {
            loadFromURL();
        }
    });
    
    await loadFromURL();
}

// 加载baselines
async function loadBaselines() {
    try {
        const response = await fetch(`${API_BASE}/api/baselines`);
        const baselines = await response.json();
        // 保存到全局变量，以便后续使用
        window.allBaselines = baselines;
        renderBaselineButtons(baselines);
        return baselines;
    } catch (error) {
        console.error('Error loading baselines:', error);
        document.getElementById('baseline-buttons').innerHTML = '<span class="empty-state">加载失败</span>';
        return [];
    }
}

// 渲染baseline按钮
function renderBaselineButtons(baselines) {
    const container = document.getElementById('baseline-buttons');
    if (!baselines || baselines.length === 0) {
        container.innerHTML = '<span class="empty-state">暂无数据</span>';
        return;
    }
    
    container.innerHTML = baselines.map(baseline => {
        const isActive = baseline === currentBaseline;
        return `<button class="btn baseline-btn ${isActive ? 'active' : ''}" 
                       data-baseline="${baseline}"
                       onclick="selectBaseline('${baseline}')">
                ${baseline}
               </button>`;
    }).join('');
}

// 加载tasks
async function loadTasks(baseline) {
    try {
        const response = await fetch(`${API_BASE}/api/tasks/${baseline}`);
        const tasks = await response.json();
        // 将 ours 加入任务列表，与 customization 等使用相同样式
        renderButtons('task-buttons', tasks || [],  'task', null);
    } catch (error) {
        console.error('Error loading tasks:', error);
        document.getElementById('task-buttons').innerHTML = '<span class="empty-state">加载失败</span>';
    }
}

// 加载exps（改为加载实验结构）
async function loadExps(baseline, task) {
    try {
        const apiUrl = task === 'ours' 
            ? `${API_BASE}/api/ours-exp-structure/${baseline}`
            : `${API_BASE}/api/exp-structure/${baseline}/${task}`;
        const response = await fetch(apiUrl);
        expStructure = await response.json();
        
        if (!expStructure || !expStructure.base_names || expStructure.base_names.length === 0) {
            document.getElementById('exp-base-buttons').innerHTML = '<span class="empty-state">暂无数据</span>';
            document.getElementById('exp-step-buttons').innerHTML = '<span class="empty-state">请先选择实验名称</span>';
            return;
        }
        
        // 渲染实验基础名称按钮
        renderExpBaseButtons(expStructure.base_names);
        
        // 检查是否有选中的实验基础名称需要step
        let hasStepBasedExps = false;
        if (selectedExpBases.length > 0 && expStructure.steps && expStructure.steps.length > 0) {
            for (const base of selectedExpBases) {
                if (expStructure.exp_map[base]) {
                    // 检查是否有step（排除None）
                    const steps = Object.keys(expStructure.exp_map[base]).filter(s => s !== 'null' && s !== null);
                    if (steps.length > 0) {
                        hasStepBasedExps = true;
                        break;
                    }
                }
            }
        }
        
        // 如果有需要step的实验，显示step按钮
        if (hasStepBasedExps) {
            // 过滤出仍然有效的steps
            const validSteps = selectedSteps.filter(step => 
                expStructure.steps && expStructure.steps.includes(step)
            );
            selectedSteps = validSteps;
            renderExpStepButtons(expStructure.steps);
        } else {
            // 清空step选择
            selectedSteps = [];
            if (selectedExpBases.length > 0) {
                document.getElementById('exp-step-buttons').innerHTML = '<span class="empty-state">这些实验无需选择step</span>';
            } else {
                document.getElementById('exp-step-buttons').innerHTML = '<span class="empty-state">请先选择实验名称</span>';
            }
        }
        
        // 清空category按钮
        document.getElementById('category-buttons').innerHTML = '<span class="empty-state">请先选择实验</span>';
    } catch (error) {
        console.error('Error loading exp structure:', error);
        document.getElementById('exp-base-buttons').innerHTML = '<span class="empty-state">加载失败</span>';
        document.getElementById('exp-step-buttons').innerHTML = '<span class="empty-state">加载失败</span>';
    }
}

// 渲染实验基础名称按钮
function renderExpBaseButtons(baseNames) {
    const container = document.getElementById('exp-base-buttons');
    if (!baseNames || baseNames.length === 0) {
        container.innerHTML = '<span class="empty-state">暂无数据</span>';
        return;
    }
    
    const sortedBases = naturalSort(baseNames);
    container.innerHTML = sortedBases.map(baseName => {
        const isSelected = selectedExpBases.includes(baseName);
        return `<button class="btn ${isSelected ? 'selected' : ''}" 
                       onclick="toggleExpBase('${baseName}')">
                ${baseName}
               </button>`;
    }).join('');
}

// 渲染step按钮
function renderExpStepButtons(steps) {
    const container = document.getElementById('exp-step-buttons');
    if (!steps || steps.length === 0) {
        container.innerHTML = '<span class="empty-state">暂无step数据</span>';
        return;
    }
    
    const sortedSteps = naturalSort(steps);
    container.innerHTML = sortedSteps.map(step => {
        const isSelected = selectedSteps.includes(step);
        return `<button class="btn ${isSelected ? 'selected' : ''}" 
                       onclick="toggleStep('${step}')">
                ${step}
               </button>`;
    }).join('');
}

// 加载categories
async function loadCategories(baseline, task, exp) {
    try {
        const response = await fetch(`${API_BASE}/api/categories/${baseline}/${task}/${exp}`);
        const categories = await response.json();
        renderCategoryButtons(categories);
    } catch (error) {
        console.error('Error loading categories:', error);
        document.getElementById('category-buttons').innerHTML = '<span class="empty-state">加载失败</span>';
    }
}

// 渲染category按钮
function renderCategoryButtons(categories) {
    const container = document.getElementById('category-buttons');
    if (!categories || categories.length === 0) {
        container.innerHTML = '<span class="empty-state">暂无数据</span>';
        return;
    }
    
    const allCategories = ['1-3', '4-5', '6-7', '>=8'];
    container.innerHTML = allCategories.map(cat => {
        const isAvailable = categories.includes(cat);
        const isActive = cat === currentCategory;
        
        if (isAvailable) {
            return `<button class="btn ${isActive ? 'active' : ''}" 
                           onclick="selectCategory('${cat}')">
                    ${cat}
                   </button>`;
        } else {
            return `<button class="btn disabled" disabled>
                    ${cat}
                   </button>`;
        }
    }).join('');
}

// 渲染按钮
function renderButtons(containerId, items, type, context) {
    const container = document.getElementById(containerId);
    if (!items || items.length === 0) {
        container.innerHTML = '<span class="empty-state">暂无数据</span>';
        return;
    }

    if (type === 'exp') {
        // 多选模式
        container.innerHTML = items.map(item => {
            const isSelected = selectedExps.includes(item);
            return `<button class="btn ${isSelected ? 'selected' : ''}" 
                           onclick="toggleExp('${item}')">
                    ${item}
                   </button>`;
        }).join('');
    } else {
        container.innerHTML = items.map(item => {
            const isActive = (type === 'task' && item === currentTask);
            const dataAttr = type === 'task' ? `data-task="${item}"` : '';
            return `<button class="btn ${isActive ? 'active' : ''}" 
                           ${dataAttr}
                           onclick="select${type.charAt(0).toUpperCase() + type.slice(1)}('${item}')">
                    ${item}
                   </button>`;
        }).join('');
    }
}

// 选择baseline
async function selectBaseline(baseline, updateUrl = true) {
    if (currentBaseline === baseline) return;
    
    currentBaseline = baseline;
    currentTask = null;
    selectedExpBases = [];
    selectedSteps = [];
    selectedExps = [];
    currentCategory = null;
    expStructure = null;
    currentPage = 1;
    
    // 更新baseline按钮的active状态
    const baselineButtons = document.querySelectorAll('#baseline-buttons .btn');
    baselineButtons.forEach(btn => {
        const btnBaseline = btn.getAttribute('data-baseline') || btn.textContent.trim();
        if (btnBaseline === baseline) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    // 清空其他选择
    document.getElementById('exp-base-buttons').innerHTML = '<span class="empty-state">请先选择任务</span>';
    document.getElementById('exp-step-buttons').innerHTML = '<span class="empty-state">请先选择实验名称</span>';
    document.getElementById('category-buttons').innerHTML = '<span class="empty-state">请先选择实验</span>';
    hideDataAreas();
    
    // 加载任务
    await loadTasks(baseline);
    
    if (updateUrl) {
        updateURL(baseline, null, [], null, 1, currentEvaluator);
    }
}

// 选择task
async function selectTask(task, updateUrl = true) {
    // 保存之前的选择状态
    const previousExpBases = [...selectedExpBases];
    const previousSteps = [...selectedSteps];
    const previousCategory = currentCategory;
    
    currentTask = task;
    expStructure = null;
    currentPage = 1;
    
    // 更新任务按钮的 active 状态（ours 与其他任务使用相同样式）
    const taskButtons = document.querySelectorAll('#task-buttons .btn');
    taskButtons.forEach(btn => {
        const btnTask = btn.getAttribute('data-task') || btn.textContent.trim();
        if (btnTask === task) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    // Ours 视图：特殊处理（无需 category）
    if (task === 'ours') {
        await loadExps(currentBaseline, 'ours');
        if (expStructure && previousExpBases.length > 0) {
            const validBases = previousExpBases.filter(base => expStructure.base_names.includes(base));
            if (validBases.length > 0) {
                selectedExpBases = validBases;
                renderExpBaseButtons(expStructure.base_names);
                let hasStepBasedExps = false;
                if (expStructure.steps && expStructure.steps.length > 0) {
                    for (const base of selectedExpBases) {
                        if (expStructure.exp_map[base]) {
                            const steps = Object.keys(expStructure.exp_map[base]).filter(s => s !== 'null' && s !== null);
                            if (steps.length > 0) { hasStepBasedExps = true; break; }
                        }
                    }
                }
                if (hasStepBasedExps) {
                    const validSteps = previousSteps.filter(step => expStructure.steps && expStructure.steps.includes(step));
                    selectedSteps = validSteps;
                    renderExpStepButtons(expStructure.steps);
                } else {
                    selectedSteps = [];
                    document.getElementById('exp-step-buttons').innerHTML = selectedExpBases.length > 0 
                        ? '<span class="empty-state">这些实验无需选择step</span>' : '<span class="empty-state">请先选择实验名称</span>';
                }
                combineExpNames();
                if (selectedExps.length > 0) {
                    await loadData();
                } else hideDataAreas();
            } else {
                selectedExpBases = []; selectedSteps = []; selectedExps = [];
                hideDataAreas();
            }
        } else {
            selectedExpBases = []; selectedSteps = []; selectedExps = [];
            hideDataAreas();
        }
        if (updateUrl) updateURL(currentBaseline, 'ours', selectedExps, currentCategory, currentPage, currentEvaluator);
        return;
    }
    
    // 加载新任务的实验结构
    await loadExps(currentBaseline, task);
    
    // 尝试恢复之前选择的实验基础名称（如果它们在新任务下也存在）
    if (expStructure && previousExpBases.length > 0) {
        const validBases = previousExpBases.filter(base => expStructure.base_names.includes(base));
        if (validBases.length > 0) {
            selectedExpBases = validBases;
            renderExpBaseButtons(expStructure.base_names);
            
            // 检查是否需要step
            let hasStepBasedExps = false;
            if (expStructure.steps && expStructure.steps.length > 0) {
                for (const base of selectedExpBases) {
                    if (expStructure.exp_map[base]) {
                        const steps = Object.keys(expStructure.exp_map[base]).filter(s => s !== 'null' && s !== null);
                        if (steps.length > 0) {
                            hasStepBasedExps = true;
                            break;
                        }
                    }
                }
            }
            
            // 如果之前选择的steps仍然有效，尝试恢复
            if (hasStepBasedExps) {
                // 过滤出在新任务下仍然有效的steps
                const validSteps = previousSteps.filter(step => 
                    expStructure.steps && expStructure.steps.includes(step)
                );
                selectedSteps = validSteps;
                renderExpStepButtons(expStructure.steps);
            } else {
                selectedSteps = [];
                if (selectedExpBases.length > 0) {
                    document.getElementById('exp-step-buttons').innerHTML = '<span class="empty-state">这些实验无需选择step</span>';
                } else {
                    document.getElementById('exp-step-buttons').innerHTML = '<span class="empty-state">请先选择实验名称</span>';
                }
            }
            
            // 组合实验名称
            combineExpNames();
            
            // 如果有选中的实验，尝试恢复类别选择
            if (selectedExps.length > 0) {
                const firstExp = selectedExps[0];
                const catResponse = await fetch(`${API_BASE}/api/categories/${currentBaseline}/${task}/${firstExp}`);
                const categories = await catResponse.json();
                renderCategoryButtons(categories);
                
                if (previousCategory && categories.includes(previousCategory)) {
                    currentCategory = previousCategory;
                    document.querySelectorAll('#category-buttons .btn').forEach(btn => {
                        btn.classList.toggle('active', btn.textContent.trim() === previousCategory);
                    });
                    await loadData();
                } else {
                    currentCategory = null;
                    hideDataAreas();
                }
            } else {
                currentCategory = null;
                document.getElementById('category-buttons').innerHTML = '<span class="empty-state">请先选择实验</span>';
                hideDataAreas();
            }
        } else {
            // 如果之前的实验基础名称在新任务下都不存在，清空选择
            selectedExpBases = [];
            selectedSteps = [];
            selectedExps = [];
            currentCategory = null;
            document.getElementById('category-buttons').innerHTML = '<span class="empty-state">请先选择实验</span>';
            hideDataAreas();
        }
    } else {
        // 如果没有之前的选择，清空
        selectedExpBases = [];
        selectedSteps = [];
        selectedExps = [];
        currentCategory = null;
        document.getElementById('category-buttons').innerHTML = '<span class="empty-state">请先选择实验</span>';
        hideDataAreas();
    }
    
    if (updateUrl) {
        updateURL(currentBaseline, task, selectedExps, currentCategory, currentPage, currentEvaluator);
    }
}

// 选择evaluator
async function selectEvaluator(evaluator) {
    if (evaluator !== 'gpt' && evaluator !== 'gemini') {
        return;
    }
    currentEvaluator = evaluator;
    updateEvaluatorButtons();
    
    // 如果有选中的实验和category，重新加载数据
    if (currentBaseline && currentTask && selectedExps.length > 0 && currentCategory) {
        await loadData();
    }
    
    updateURL(currentBaseline, currentTask, selectedExps, currentCategory, currentPage, currentEvaluator);
}

// 更新evaluator按钮状态
function updateEvaluatorButtons() {
    const buttons = document.querySelectorAll('#evaluator-buttons .btn');
    buttons.forEach(btn => {
        const evaluator = btn.textContent.toLowerCase().includes('gpt') ? 'gpt' : 'gemini';
        if (evaluator === currentEvaluator) {
            btn.classList.add('active');
            btn.classList.remove('selected');
        } else {
            btn.classList.remove('active');
            btn.classList.remove('selected');
        }
    });
}

// 切换实验基础名称（多选）
async function toggleExpBase(baseName) {
    // 保存之前的steps和category选择
    const previousSteps = [...selectedSteps];
    const previousCategory = currentCategory;
    
    const index = selectedExpBases.indexOf(baseName);
    if (index > -1) {
        selectedExpBases.splice(index, 1);
    } else {
        selectedExpBases.push(baseName);
    }
    
    // 重新渲染按钮
    if (expStructure) {
        renderExpBaseButtons(expStructure.base_names);
        
        // 检查是否有选中的实验基础名称需要step
        let hasStepBasedExps = false;
        if (selectedExpBases.length > 0 && expStructure.steps && expStructure.steps.length > 0) {
            for (const base of selectedExpBases) {
                if (expStructure.exp_map[base]) {
                    // 检查是否有step（排除None）
                    const steps = Object.keys(expStructure.exp_map[base]).filter(s => s !== 'null' && s !== null);
                    if (steps.length > 0) {
                        hasStepBasedExps = true;
                        break;
                    }
                }
            }
        }
        
        // 如果有需要step的实验，显示step按钮
        if (hasStepBasedExps) {
            // 过滤出仍然有效的steps（即所有选中的实验基础名称都有这个step）
            const validSteps = previousSteps.filter(step => {
                if (!expStructure.steps || !expStructure.steps.includes(step)) {
                    return false;
                }
                for (const base of selectedExpBases) {
                    if (expStructure.exp_map[base] && !expStructure.exp_map[base][step]) {
                        return false;
                    }
                }
                return true;
            });
            selectedSteps = validSteps;
            renderExpStepButtons(expStructure.steps);
        } else {
            // 如果所有实验都不需要step，清空step选择
            selectedSteps = [];
            if (selectedExpBases.length > 0) {
                document.getElementById('exp-step-buttons').innerHTML = '<span class="empty-state">这些实验无需选择step</span>';
            } else {
                document.getElementById('exp-step-buttons').innerHTML = '<span class="empty-state">请先选择实验名称</span>';
            }
        }
    }
    
    // 更新完整的实验名称列表
    combineExpNames();
    
    // 如果有选中的实验，尝试保持类别选择
    if (selectedExps.length > 0) {
        const firstExp = selectedExps[0];
        // 尝试保持之前选择的类别
        const catResponse = await fetch(`${API_BASE}/api/categories/${currentBaseline}/${currentTask}/${firstExp}`);
        const categories = await catResponse.json();
        renderCategoryButtons(categories);
        
        // 优先使用previousCategory，如果没有则使用currentCategory
        const categoryToCheck = previousCategory || currentCategory;
        if (categoryToCheck && categories.includes(categoryToCheck)) {
            currentCategory = categoryToCheck;
            document.querySelectorAll('#category-buttons .btn').forEach(btn => {
                btn.classList.toggle('active', btn.textContent.trim() === currentCategory);
            });
            await loadData();
        } else {
            currentCategory = null;
            hideDataAreas();
        }
    } else {
        // 如果没有选中的实验，但之前有类别选择，保持类别选择（等待用户选择实验）
        if (currentCategory) {
            // 保持类别选择，但不显示数据
            hideDataAreas();
        } else {
            document.getElementById('category-buttons').innerHTML = '<span class="empty-state">请先选择实验</span>';
            hideDataAreas();
        }
    }
    
    updateURL(currentBaseline, currentTask, selectedExps, currentCategory, currentPage, currentEvaluator);
}

// 切换step（多选）
async function toggleStep(step) {
    // 保存之前的类别选择和页码
    const previousCategory = currentCategory;
    const previousPage = currentPage;
    
    // 切换step选择
    const index = selectedSteps.indexOf(step);
    if (index > -1) {
        selectedSteps.splice(index, 1);
    } else {
        selectedSteps.push(step);
    }
    
    // 重新渲染step按钮
    if (expStructure && expStructure.steps) {
        renderExpStepButtons(expStructure.steps);
    }
    
    // 更新完整的实验名称列表
    combineExpNames();
    
    // 如果有选中的实验，尝试保持类别选择
    if (selectedExps.length > 0) {
        const firstExp = selectedExps[0];
        const catResponse = await fetch(`${API_BASE}/api/categories/${currentBaseline}/${currentTask}/${firstExp}`);
        const categories = await catResponse.json();
        renderCategoryButtons(categories);
        
        // 优先使用previousCategory，如果没有则使用currentCategory
        const categoryToCheck = previousCategory || currentCategory;
        if (categoryToCheck && categories.includes(categoryToCheck)) {
            currentCategory = categoryToCheck;
            document.querySelectorAll('#category-buttons .btn').forEach(btn => {
                btn.classList.toggle('active', btn.textContent.trim() === currentCategory);
            });
            currentPage = previousPage;
            await loadData();
        } else {
            currentCategory = null;
            hideDataAreas();
        }
    } else {
        // 如果没有选中的实验
        if (currentCategory) {
            hideDataAreas();
        } else {
            document.getElementById('category-buttons').innerHTML = '<span class="empty-state">请先选择实验</span>';
            hideDataAreas();
        }
    }
    
    updateURL(currentBaseline, currentTask, selectedExps, currentCategory, currentPage, currentEvaluator);
}

// 选择category
async function selectCategory(category) {
    currentCategory = category;
    currentPage = 1;
    
    // 更新category按钮的active状态
    const categoryButtons = document.querySelectorAll('#category-buttons .btn');
    categoryButtons.forEach(btn => {
        const btnCategory = btn.textContent.trim();
        if (btnCategory === category) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    // 加载数据
    if (currentBaseline && currentTask && selectedExps.length > 0) {
        await loadData();
    }
    
    updateURL(currentBaseline, currentTask, selectedExps, category, 1, currentEvaluator);
}

// 加载数据
async function loadData() {
    if (!currentBaseline || !currentTask || selectedExps.length === 0) return;
    
    // Ours 视图：仅加载聚合得分，无样例
    if (currentTask === 'ours') {
        await loadOursData();
        return;
    }
    
    // 常规任务需要 category
    if (!currentCategory) return;

    // 加载对比统计
    await loadComparison();
    
    // 隐藏 benchmark 区域（已不支持）
    const benchmarkEl = document.getElementById('benchmark-area');
    if (benchmarkEl) benchmarkEl.style.display = 'none';
    
    // 加载样本列表
    await loadSamples();
}


// 渲染benchmark分数表格
function renderBenchmarkScores(data) {
    const container = document.getElementById('benchmark-area');
    if (!container) return;
    
    // 对选中的实验进行自然排序
    const sortedExps = getSortedExps(selectedExps);
    
    let html = '<div class="benchmark-title">Benchmark 评测结果</div>';
    
    for (const benchmarkData of data.comparison_data) {
        const benchmark = benchmarkData.benchmark;
        const metrics = benchmarkData.metrics;
        
        html += `<div class="benchmark-section">`;
        html += `<div class="benchmark-name">${benchmark.toUpperCase()}</div>`;
        html += '<div class="benchmark-table-container"><table class="benchmark-table">';
        html += '<thead><tr><th>指标</th>';
        sortedExps.forEach(exp => {
            html += `<th class="exp-header">${exp}</th>`;
        });
        html += '</tr></thead><tbody>';
        
        for (const [metricName, expValues] of Object.entries(metrics)) {
            html += '<tr>';
            // 格式化指标名称
            const formattedName = metricName.split('_').map(word => 
                word.charAt(0).toUpperCase() + word.slice(1)
            ).join(' ');
            html += `<td class="metric-name">${formattedName}</td>`;
            
            sortedExps.forEach(exp => {
                const value = expValues[exp];
                let displayValue = 'N/A';
                if (value !== null && value !== undefined) {
                    if (typeof value === 'number') {
                        displayValue = value.toFixed(2);
                    } else if (typeof value === 'object') {
                        // 处理对象类型的值（如 {mean: 0.95, count: 100}）
                        if (value.mean !== undefined) {
                            displayValue = value.mean.toFixed(2);
                            if (value.count !== undefined) {
                                displayValue += ` (${value.count} samples)`;
                            }
                        } else {
                            displayValue = JSON.stringify(value);
                        }
                    } else {
                        displayValue = value;
                    }
                }
                html += `<td class="metric-value">${displayValue}</td>`;
            });
            
            html += '</tr>';
        }
        
        html += '</tbody></table></div></div>';
    }
    
    container.innerHTML = html;
}

// 加载 Ours 视图数据
async function loadOursData() {
    try {
        const expsParam = selectedExps.join(',');
        const response = await fetch(`${API_BASE}/api/ours-scores?baseline=${currentBaseline}&exps=${expsParam}&evaluator=${currentEvaluator}`);
        const data = await response.json();
        
        if (data && data.task_rows) {
            renderOursScores(data);
            document.getElementById('ours-area').style.display = 'block';
            document.getElementById('comparison-area').style.display = 'none';
            document.getElementById('samples-area').style.display = 'none';
            const benchmarkArea3 = document.getElementById('benchmark-area');
            if (benchmarkArea3) benchmarkArea3.style.display = 'none';
            const floatingPagination = document.getElementById('floating-pagination');
            if (floatingPagination) floatingPagination.style.display = 'none';
            
            const oursBenchmarkSection = document.getElementById('ours-benchmark-section');
            if (oursBenchmarkSection) oursBenchmarkSection.style.display = 'none';
        } else {
            document.getElementById('ours-area').style.display = 'none';
        }
    } catch (error) {
        console.error('Error loading ours scores:', error);
        document.getElementById('ours-area').style.display = 'none';
    }
}

// 加载 Ours 页面下的第三方评测数据（支持多选）

// 统一格式化数字为 2 位小数
function formatScore(val) {
    if (val === undefined || val === null) return 'N/A';
    if (typeof val === 'number' && !isNaN(val)) return val.toFixed(2);
    if (typeof val === 'string' && val !== 'N/A') {
        const n = parseFloat(val);
        if (!isNaN(n)) return n.toFixed(2);
    }
    return String(val);
}

// 渲染多个 Ours 第三方评测表格

// 渲染单个 Ours 第三方评测表格（可折叠）
function renderSingleOursBenchmarkBlock(data) {
    const benchmark = data.benchmark || '';
    const table = data.comparison_table || [];
    const sortedExps = getSortedExps(selectedExps);
    
    if (table.length === 0) return '';
    
    const avgRow = table.find(r => (r.metric_name || '').toLowerCase() === 'average');
    const sampleCountRow = table.find(r => (r.metric_name || '').includes('样本数'));
    const detailRows = table.filter(r => {
        const name = (r.metric_name || '').toLowerCase();
        return name !== 'average' && !(r.metric_name || '').includes('样本数');
    });
    const hasDetails = detailRows.length > 0;
    const safeId = benchmark.replace(/[^a-z0-9]/gi, '_');
    
    const formatCell = (val) => {
        if (val === undefined || val === null) return 'N/A';
        if (typeof val === 'object' && val !== null) {
            const os = val.overall_score;
            if (os !== undefined && os !== 'N/A' && os !== null) {
                return typeof os === 'number' ? formatScore(os) : os;
            }
            return 'N/A';
        }
        if (typeof val === 'number') return formatScore(val);
        if (typeof val === 'string' && val.endsWith('%')) {
            const n = parseFloat(val);
            if (!isNaN(n)) return n.toFixed(2) + '%';
        }
        return String(val);
    };
    
    let blockHtml = `<div class="ours-benchmark-block">`;
    blockHtml += `<div class="ours-benchmark-header" onclick="toggleOursBenchmarkExpand('${safeId}')">`;
    blockHtml += `<span class="expand-icon" id="ours-benchmark-expand-icon-${safeId}">${hasDetails ? '▶' : ''}</span> `;
    blockHtml += `${benchmark.toUpperCase()} 评测结果</div>`;
    blockHtml += `<div class="ours-benchmark-table-wrap"><table class="ours-table ours-benchmark-table"><thead><tr><th>指标</th>`;
    sortedExps.forEach(exp => { blockHtml += `<th class="exp-header">${exp}</th>`; });
    blockHtml += '</tr></thead><tbody>';
    
    const formatCellWithCount = (v) => {
        const score = formatCell(typeof v === 'object' ? (v && v.overall_score) : v);
        const cnt = (v && typeof v === 'object' && v.count !== undefined && v.count !== null) ? v.count : null;
        return cnt !== null ? `${score} <span class="sample-count">n=${cnt}</span>` : score;
    };
    
    if (avgRow) {
        blockHtml += '<tr class="benchmark-total-row">';
        blockHtml += `<td class="metric-name">Average</td>`;
        sortedExps.forEach(exp => {
            const v = avgRow[exp];
            blockHtml += `<td class="metric-value">${formatCellWithCount(v)}</td>`;
        });
        blockHtml += '</tr>';
    }
    if (sampleCountRow) {
        blockHtml += '<tr class="benchmark-sample-row">';
        blockHtml += `<td class="metric-name">${sampleCountRow.metric_name || '样本数'}</td>`;
        sortedExps.forEach(exp => {
            const v = sampleCountRow[exp];
            blockHtml += `<td class="metric-value">${v !== undefined && v !== null && v !== 'N/A' ? v : (typeof v === 'object' ? 'N/A' : (v || 'N/A'))}</td>`;
        });
        blockHtml += '</tr>';
    }
    
    if (hasDetails) {
        detailRows.forEach(row => {
            blockHtml += `<tr class="benchmark-detail-row ours-benchmark-detail-${safeId}" style="display: none;">`;
            blockHtml += `<td class="metric-name">${row.metric_name || ''}</td>`;
            sortedExps.forEach(exp => {
                const v = row[exp];
                blockHtml += `<td class="metric-value">${formatCellWithCount(v)}</td>`;
            });
            blockHtml += '</tr>';
        });
    }
    
    blockHtml += '</tbody></table></div></div>';
    return blockHtml;
}

// 切换 Ours 第三方评测详情的展开/收起

// 渲染 Ours 得分表格（支持点击任务展开 category 详情）
function renderOursScores(data) {
    const container = document.getElementById('ours-table-container');
    if (!container) return;
    
    const sortedExps = getSortedExps(data.exps || selectedExps);
    
    let html = '<table class="ours-table"><thead><tr><th>任务 / 指标</th>';
    sortedExps.forEach(exp => {
        html += `<th class="exp-header">${exp}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    const taskDisplayNames = {
        'customization': 'Customization',
        'illustration': 'Illustration',
        'spatial': 'Spatial',
        'temporal': 'Temporal'
    };
    
    data.task_rows.forEach((taskRow, idx) => {
        const taskName = taskRow.task;
        const displayName = taskDisplayNames[taskName] || taskName;
        const hasCategories = sortedExps.some(exp => taskRow.categories[exp] && Object.keys(taskRow.categories[exp]).length > 0);
        const taskCount = taskRow.task_count || {};
        const catCount = taskRow.categories_count || {};
        
        html += `<tr class="task-row" data-task-idx="${idx}" onclick="toggleOursTaskExpand(${idx})">`;
        html += `<td class="metric-name"><span class="expand-icon">${hasCategories ? '▶' : ''}</span> ${displayName} (average)</td>`;
        sortedExps.forEach(exp => {
            const val = taskRow.task_mean[exp];
            const cnt = taskCount[exp];
            const cntStr = (cnt !== undefined && cnt !== null && cnt > 0) ? ` <span class="sample-count">n=${cnt}</span>` : '';
            html += `<td class="metric-value">${val !== null && val !== undefined ? formatScore(val) + cntStr : 'N/A'}</td>`;
        });
        html += '</tr>';
        
        if (hasCategories) {
            const catOrder = ['1-3', '4-5', '6-7', '>=8'];
            catOrder.forEach(cat => {
                const anyHasCat = sortedExps.some(exp => taskRow.categories[exp] && taskRow.categories[exp][cat] !== undefined);
                if (!anyHasCat) return;
                
                html += `<tr class="category-detail-row ours-expand-row ours-expand-${idx}" style="display: none;">`;
                html += `<td class="metric-name">└ ${cat}</td>`;
                sortedExps.forEach(exp => {
                    const val = taskRow.categories[exp] && taskRow.categories[exp][cat];
                    const cnt = catCount[exp] && catCount[exp][cat];
                    const cntStr = (cnt !== undefined && cnt !== null && cnt > 0) ? ` <span class="sample-count">n=${cnt}</span>` : '';
                    html += `<td class="metric-value">${val !== null && val !== undefined ? formatScore(val) + cntStr : 'N/A'}</td>`;
                });
                html += '</tr>';
            });
        }
    });
    
    html += '<tr class="total-row"><td class="metric-name">Total (average)</td>';
    const totalCount = data.total_count || {};
    sortedExps.forEach(exp => {
        const val = data.total_row && data.total_row[exp];
        const cnt = totalCount[exp];
        const cntStr = (cnt !== undefined && cnt !== null && cnt > 0) ? ` <span class="sample-count">n=${cnt}</span>` : '';
        html += `<td class="metric-value">${val !== null && val !== undefined ? formatScore(val) + cntStr : 'N/A'}</td>`;
    });
    html += '</tr>';
    
    html += '</tbody></table>';
    container.innerHTML = html;
}

// 切换 Ours 任务行的展开/收起
function toggleOursTaskExpand(idx) {
    const detailRows = document.querySelectorAll('.ours-expand-' + idx);
    const taskRow = document.querySelector('.task-row[data-task-idx="' + idx + '"]');
    if (!taskRow) return;
    
    const isExpanded = taskRow.classList.contains('expanded');
    if (isExpanded) {
        taskRow.classList.remove('expanded');
        detailRows.forEach(r => { r.style.display = 'none'; });
        const icon = taskRow.querySelector('.expand-icon');
        if (icon) icon.textContent = '▶';
    } else {
        taskRow.classList.add('expanded');
        detailRows.forEach(r => { r.style.display = 'table-row'; });
        const icon = taskRow.querySelector('.expand-icon');
        if (icon) icon.textContent = '▼';
    }
}

// 加载对比统计
async function loadComparison() {
    try {
        const expsParam = selectedExps.join(',');
        const response = await fetch(`${API_BASE}/api/comparison?baseline=${currentBaseline}&task=${currentTask}&category=${currentCategory}&exps=${expsParam}&evaluator=${currentEvaluator}`);
        const comparison = await response.json();
        
        if (comparison) {
            renderComparison(comparison);
            document.getElementById('comparison-area').style.display = 'block';
        }
    } catch (error) {
        console.error('Error loading comparison:', error);
    }
}

// 渲染对比统计
function renderComparison(comparison) {
    const table = document.getElementById('comparison-table');
    const thead = table.querySelector('thead tr');
    const tbody = table.querySelector('tbody');
    
    // 清空表头和数据
    thead.innerHTML = '<th>指标</th>';
    tbody.innerHTML = '';
    
    // 对选中的实验进行自然排序
    const sortedExps = getSortedExps(selectedExps);
    
    // 添加实验列
    sortedExps.forEach(exp => {
        const th = document.createElement('th');
        th.className = 'exp-header';
        th.textContent = exp;
        thead.appendChild(th);
    });
    
    // 格式化指标名称（首字母大写，下划线替换为空格）
    const formatMetricName = (name) => {
        return name.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    };
    
    // 添加数据行
    if (comparison.comparison_table && comparison.comparison_table.length > 0) {
        comparison.comparison_table.forEach(row => {
            const tr = document.createElement('tr');
            
            // 指标名称（格式化显示）
            const metricTd = document.createElement('td');
            metricTd.className = 'metric-name';
            const metricName = row.metric_name || row.metric || '';
            metricTd.textContent = formatMetricName(metricName);
            tr.appendChild(metricTd);
            
            // 各实验的值（使用排序后的顺序）
            sortedExps.forEach(exp => {
                const valueTd = document.createElement('td');
                valueTd.className = 'metric-value';
                const value = row[exp];
                if (value !== undefined && value !== null) {
                    if (typeof value === 'object' && value.mean !== undefined) {
                        valueTd.textContent = `${formatScore(value.mean)} (中位数: ${formatScore(value.median)})`;
                    } else {
                        valueTd.textContent = typeof value === 'number' ? formatScore(value) : value;
                    }
                } else {
                    valueTd.textContent = 'N/A';
                }
                tr.appendChild(valueTd);
            });
            
            tbody.appendChild(tr);
        });
    }
    
    // 添加评估器标识到表格标题
    const comparisonTitle = document.querySelector('.comparison-title');
    if (comparisonTitle) {
        const evaluatorLabel = currentEvaluator === 'gpt' ? 'GPT-4o' : 'Gemini';
        comparisonTitle.textContent = `实验对比统计 (${evaluatorLabel} 评分) - ${currentBaseline}`;
    }
}

// 加载样本列表
async function loadSamples() {
    try {
        const expsParam = selectedExps.join(',');
        const response = await fetch(`${API_BASE}/api/page?baseline=${currentBaseline}&task=${currentTask}&category=${currentCategory}&exps=${expsParam}&page=${currentPage}&evaluator=${currentEvaluator}`);
        const data = await response.json();
        
        if (data && data.samples) {
            sampleIds = data.sample_ids || [];
            // 保存指标名称到全局变量
            window.metric1Name = data.metric1_name || 'metric1';
            window.metric2Name = data.metric2_name || 'metric2';
            renderSamples(data.samples);
            renderPagination(data.total_pages || 1, data.current_page || 1);
            document.getElementById('samples-area').style.display = 'block';
        }
    } catch (error) {
        console.error('Error loading samples:', error);
    }
}

// 渲染样本
function renderSamples(samples) {
    const list = document.getElementById('samples-list');
    
    if (!samples || samples.length === 0) {
        list.innerHTML = '<div class="empty-state">暂无样本数据</div>';
        return;
    }
    
    list.innerHTML = samples.map(sample => renderSample(sample)).join('');
}

// 渲染单个样本
function renderSample(sample) {
    // 对选中的实验进行自然排序
    const sortedExps = getSortedExps(selectedExps);
    
    // 获取instruction（从第一个实验获取，应该都一样）
    const firstExp = sortedExps[0];
    const instruction = sample.instruction || 'N/A';
    const inputImages = sample.input_images || [];
    const targetImage = sample.target_image;
    
    // 渲染输入图像
    const inputImagesHtml = inputImages.map((img, idx) => `
        <div class="image-wrapper input">
            <img src="${API_BASE}/api/image?path=${encodeURIComponent(img)}" 
                 alt="输入图像 ${idx + 1}" 
                 onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'200\' height=\'200\'%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\'%3E图片加载失败%3C/text%3E%3C/svg%3E'">
            <div class="image-label">输入 ${idx + 1}</div>
        </div>
    `).join('');
    
    // 渲染目标图像（如果有）
    const targetImageHtml = targetImage ? `
        <div class="image-wrapper target">
            <img src="${API_BASE}/api/image?path=${encodeURIComponent(targetImage)}" 
                 alt="目标图像" 
                 onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'200\' height=\'200\'%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\'%3E图片加载失败%3C/text%3E%3C/svg%3E'">
            <div class="image-label">目标图像</div>
        </div>
    ` : '';
    
    // 渲染各实验的输出（使用排序后的顺序）
    const experimentsHtml = sortedExps.map(exp => {
        const expData = sample.experiments && sample.experiments[exp];
        if (!expData) {
            return `
                <div class="experiment-block">
                    <div class="experiment-header">${exp}</div>
                    <div class="empty-state">无数据</div>
                </div>
            `;
        }
        
        // 获取指标名称（从全局变量或使用默认值）
        const metric1Name = window.metric1Name || 'metric1';
        const metric2Name = window.metric2Name || 'metric2';
        
        // 格式化指标名称（首字母大写，下划线替换为空格）
        const formatMetricName = (name) => {
            return name.split('_').map(word => 
                word.charAt(0).toUpperCase() + word.slice(1)
            ).join(' ');
        };
        
        const evaluatorLabel = currentEvaluator === 'gpt' ? 'GPT-4o' : 'Gemini';
        
        // 单图像任务
        const outputImage = expData.output_image || '';
        
        // 区分0分和None（无评分）：0分显示为"0"，None显示为"N/A"
        const metric1Score = (expData.metric1_score !== undefined && expData.metric1_score !== null) ? expData.metric1_score : 'N/A';
        const metric2Score = (expData.metric2_score !== undefined && expData.metric2_score !== null) ? expData.metric2_score : 'N/A';
        const reasoning = expData.reasoning || '';
        
        // 对于customization任务，显示consistency_scores_list（如果有）
        let consistencyScoresListHtml = '';
        if (expData.consistency_scores_list && Array.isArray(expData.consistency_scores_list) && expData.consistency_scores_list.length > 0) {
            consistencyScoresListHtml = `
                <div class="score-item-detail" style="font-size: 0.85em; color: #666; margin-top: 5px;">
                    原始得分: [${expData.consistency_scores_list.join(', ')}]
                </div>
            `;
        }
        
        return `
            <div class="experiment-block">
                <div class="experiment-header">${exp} <span class="evaluator-badge">(${evaluatorLabel})</span></div>
                <div class="experiment-images">
                    ${outputImage ? `
                        <div class="image-wrapper output">
                            <img src="${API_BASE}/api/image?path=${encodeURIComponent(outputImage)}" 
                                 alt="生成图像" 
                                 onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'200\' height=\'200\'%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\'%3E图片加载失败%3C/text%3E%3C/svg%3E'">
                            <div class="image-label">生成图像</div>
                        </div>
                    ` : '<div class="empty-state">无输出图像</div>'}
                </div>
                <div class="experiment-scores">
                    <div class="score-item">
                        <div class="score-item-label">${formatMetricName(metric1Name)}</div>
                        <div class="score-item-value ${metric1Score === 'FAILED' ? 'failed' : ''}">${metric1Score === 'FAILED' ? '评分失败' : metric1Score}</div>
                        ${consistencyScoresListHtml}
                    </div>
                    <div class="score-item">
                        <div class="score-item-label">${formatMetricName(metric2Name)}</div>
                        <div class="score-item-value ${metric2Score === 'FAILED' ? 'failed' : ''}">${metric2Score === 'FAILED' ? '评分失败' : metric2Score}</div>
                    </div>
                </div>
                ${reasoning ? `
                    <div class="reasoning-section">
                        <button class="reasoning-toggle" onclick="toggleReasoning(this)">
                            显示/隐藏 ${evaluatorLabel} 理由
                        </button>
                        <div class="reasoning-content">
                            ${reasoning}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');
    
    // 构建图片展示区域
    let imagesSection = '';
    if (inputImagesHtml) {
        imagesSection += `<div class="input-images-container">${inputImagesHtml}</div>`;
    }
    if (targetImageHtml) {
        if (inputImagesHtml) {
            imagesSection += '<div class="separator">→</div>';
        }
        imagesSection += targetImageHtml;
    }
    
    return `
        <div class="sample-block">
            <div class="sample-prompt">
                <strong>Prompt:</strong> ${formatPrompt(instruction)}
            </div>
            <div class="sample-images">
                ${imagesSection}
            </div>
            <div class="experiments-comparison">
                ${experimentsHtml}
            </div>
        </div>
    `;
}

// 切换理由显示
function toggleReasoning(button) {
    const content = button.nextElementSibling;
    content.classList.toggle('show');
}

// 渲染分页
function renderPagination(totalPagesParam, currentPageNum) {
    const pagination = document.getElementById('pagination');
    const floatingPagination = document.getElementById('floating-pagination');
    const floatingPrevBtn = document.getElementById('floating-prev-btn');
    const floatingNextBtn = document.getElementById('floating-next-btn');
    const floatingPageInfo = document.getElementById('floating-page-info');
    const floatingPageText = document.getElementById('floating-page-text');
    
    currentPage = currentPageNum;
    totalPages = totalPagesParam;
    
    if (totalPages <= 1) {
        pagination.innerHTML = '';
        if (floatingPagination) {
            floatingPagination.style.display = 'none';
        }
        return;
    }

    // 更新顶部分页
    pagination.innerHTML = `
        <button class="pagination-btn ${currentPage === 1 ? 'disabled' : ''}" 
                onclick="changePage(${currentPage - 1})" 
                ${currentPage === 1 ? 'disabled' : ''}>
            上一页
        </button>
        <span class="page-info">第 ${currentPage} / ${totalPages} 页</span>
        <button class="pagination-btn ${currentPage === totalPages ? 'disabled' : ''}" 
                onclick="changePage(${currentPage + 1})" 
                ${currentPage === totalPages ? 'disabled' : ''}>
            下一页
        </button>
    `;
    
    // 更新浮动分页按钮
    if (floatingPagination) {
        floatingPagination.style.display = 'flex';
        floatingPageText.textContent = `${currentPage} / ${totalPages}`;
        
        // 更新上一页按钮
        if (floatingPrevBtn) {
            if (currentPage === 1) {
                floatingPrevBtn.disabled = true;
            } else {
                floatingPrevBtn.disabled = false;
            }
        }
        
        // 更新下一页按钮
        if (floatingNextBtn) {
            if (currentPage === totalPages) {
                floatingNextBtn.disabled = true;
            } else {
                floatingNextBtn.disabled = false;
            }
        }
    }
}

// 切换页面
function changePage(page) {
    if (page < 1 || page > totalPages) return;
    currentPage = page;
    updateURL(currentBaseline, currentTask, selectedExps, currentCategory, page, currentEvaluator);
    loadSamples();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// 页面加载时初始化
init();
