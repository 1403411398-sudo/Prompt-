/**
 * PromptForge - AI Prompt 自动优化系统
 * Main Application Logic v2
 */

// ──────────────────────── State ────────────────────────
const state = {
    taskType: 'classification',
    algorithm: 'random_search',
    maxIterations: 8,
    useLlmJudge: false,
    model: 'qwen3.5-35b-a3b',
    multiTask: false,
    multiTaskTypes: ['classification', 'summarization', 'translation'],
    isRunning: false,
    results: null,
    multiResults: null,
    charts: {},
    tasks: {},
    models: [],
    taskData: [],        // Full dataset for current task
    selectedIndices: [],  // Selected data indices for evaluation
};

const API_BASE = 'http://127.0.0.1:8000'; //

const MODEL_DISPLAY = {
    'qwen3.5-35b-a3b': { short: 'Qwen3.5', color: '#6366f1' },
    'deepseek-v3': { short: 'DeepSeek-V3', color: '#06b6d4' },
    'kimi-k2': { short: 'Kimi-K2', color: '#f59e0b' },
};

// ──────────────────────── DOM References ────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ──────────────────────── Initialization ────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initModelCards();
    initTaskButtons();
    initAlgoButtons();
    initSlider();
    initToggle();
    initMultiTask();
    initStartButton();
    initActionButtons();
    initDataSelector();
    loadTaskData();
    loadModels();
    loadFullData();
});

// ──────────────────────── Model Cards ────────────────────────
function initModelCards() {
    $$('.model-card').forEach(card => {
        card.addEventListener('click', () => {
            $$('.model-card').forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            state.model = card.dataset.model;
            updateHeaderModelBadge();
        });
    });
}

function updateHeaderModelBadge() {
    const display = MODEL_DISPLAY[state.model] || { short: state.model, color: '#6366f1' };
    const badge = $('#headerModelBadge');
    const dot = $('#headerModelDot');
    const name = $('#headerModelName');

    name.textContent = display.short;
    badge.style.borderColor = display.color + '33';
    badge.style.background = display.color + '14';
    badge.style.color = display.color;
    dot.style.background = display.color;
    dot.style.boxShadow = `0 0 6px ${display.color}66`;
}

// ──────────────────────── Task Selection ────────────────────────
function initTaskButtons() {
    $$('.task-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('.task-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.taskType = btn.dataset.task;
            loadFullData();
        });
    });
}

// ──────────────────────── Algorithm Selection ────────────────────────
function initAlgoButtons() {
    $$('.algo-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('.algo-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.algorithm = btn.dataset.algo;
        });
    });
}

// ──────────────────────── Slider ────────────────────────
function initSlider() {
    const slider = $('#iterSlider');
    const value = $('#iterValue');
    slider.addEventListener('input', () => {
        state.maxIterations = parseInt(slider.value);
        value.textContent = slider.value;
    });
}

// ──────────────────────── Toggles ────────────────────────
function initToggle() {
    $('#llmJudgeToggle').addEventListener('change', (e) => {
        state.useLlmJudge = e.target.checked;
    });
}

// ──────────────────────── Multi-Task ────────────────────────
function initMultiTask() {
    $('#multiTaskToggle').addEventListener('change', (e) => {
        state.multiTask = e.target.checked;
        $('#multiTaskTasks').style.display = e.target.checked ? 'flex' : 'none';
    });

    $$('#multiTaskTasks input[type="checkbox"]').forEach(cb => {
        cb.addEventListener('change', () => {
            state.multiTaskTypes = Array.from(
                $$('#multiTaskTasks input[type="checkbox"]:checked')
            ).map(c => c.value);
        });
    });
}

// ──────────────────────── Data Loading ────────────────────────
async function loadTaskData() {
    try {
        const response = await fetch(`${API_BASE}/api/tasks`);
        if (response.ok) {
            state.tasks = await response.json();
            let totalSamples = 0;
            Object.values(state.tasks).forEach(t => { totalSamples += t.count || 0; });
            $('#heroSamples').textContent = totalSamples;
            updateDataPreview();
        }
    } catch (e) {
        console.warn('Could not load task data:', e);
    }
}

async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/api/models`);
        if (response.ok) {
            const data = await response.json();
            state.models = data.models || [];
            $('#heroModels').textContent = state.models.length;
        }
    } catch (e) {
        console.warn('Could not load models:', e);
    }
}

async function loadFullData() {
    const container = $('#dataPreviewSamples');
    const countBadge = $('#dataPreviewCount');

    container.innerHTML = '<div class="data-sample-placeholder">加载数据中...</div>';

    const taskTypes = ['classification', 'summarization', 'translation'];
    const taskLabels = { classification: '分类', summarization: '摘要', translation: '翻译' };
    const allData = [];

    try {
        const results = await Promise.all(
            taskTypes.map(t => fetch(`${API_BASE}/api/task_data/${t}`).then(r => r.json()))
        );

        results.forEach((result, i) => {
            const tt = taskTypes[i];
            (result.data || []).forEach(item => {
                allData.push({ ...item, task_type: tt });
            });
        });

        state.taskData = allData;
        // Select all by default
        state.selectedIndices = allData.map((_, i) => i);
        renderDataItems();
    } catch (e) {
        container.innerHTML = '<div class="data-sample-placeholder">加载数据失败</div>';
        countBadge.textContent = '—';
    }
}

function renderDataItems() {
    const container = $('#dataPreviewSamples');
    const labelClassMap = { classification: 'cls', summarization: 'sum', translation: 'trans' };
    const taskLabelMap = { classification: '分类', summarization: '摘要', translation: '翻译' };

    if (!state.taskData.length) {
        container.innerHTML = '<div class="data-sample-placeholder">无数据</div>';
        updateSelectedCount();
        return;
    }

    let html = '';
    state.taskData.forEach((item, globalIdx) => {
        const checked = state.selectedIndices.includes(globalIdx) ? 'checked' : '';
        const labelClass = labelClassMap[item.task_type] || 'cls';
        let labelText = '';
        let contentText = '';

        if (item.task_type === 'classification') {
            labelText = item.label || '';
            contentText = item.text || '';
        } else if (item.task_type === 'summarization') {
            labelText = '摘要';
            contentText = item.text || '';
        } else if (item.task_type === 'translation') {
            labelText = '翻译';
            contentText = item.source || '';
        }

        const taskTag = taskLabelMap[item.task_type] || '';
        html += `<label class="data-select-item" data-task="${item.task_type}">
            <input type="checkbox" class="data-checkbox" data-global-index="${globalIdx}" ${checked}>
            <span class="data-task-tag ${labelClass}">${taskTag}</span>
            <span class="data-sample-label ${labelClass}">${escapeHtml(labelText)}</span>
            <span class="data-sample-text">${escapeHtml(contentText)}</span>
        </label>`;
    });

    container.innerHTML = html;

    // Listen for checkbox changes
    container.querySelectorAll('.data-checkbox').forEach(cb => {
        cb.addEventListener('change', () => {
            const idx = parseInt(cb.dataset.globalIndex);
            if (cb.checked) {
                if (!state.selectedIndices.includes(idx)) state.selectedIndices.push(idx);
            } else {
                state.selectedIndices = state.selectedIndices.filter(i => i !== idx);
            }
            updateSelectedCount();
        });
    });

    updateSelectedCount();
}

function initDataSelector() {
    $('#btnSelectAll').addEventListener('click', () => {
        state.selectedIndices = state.taskData.map((_, i) => i);
        $$('.data-checkbox').forEach(cb => cb.checked = true);
        updateSelectedCount();
    });
    $('#btnSelectNone').addEventListener('click', () => {
        state.selectedIndices = [];
        $$('.data-checkbox').forEach(cb => cb.checked = false);
        updateSelectedCount();
    });
}

function updateSelectedCount() {
    const total = state.taskData.length;
    const selected = state.selectedIndices.length;
    $('#dataPreviewCount').textContent = `已选 ${selected} / ${total}`;
}

function getSelectedIndicesForTask(taskType) {
    // Return the original data indices for items selected that match the given task type
    const indices = [];
    state.selectedIndices.forEach(globalIdx => {
        const item = state.taskData[globalIdx];
        if (item && item.task_type === taskType) {
            indices.push(item.index);
        }
    });
    return indices.length > 0 ? indices : null;
}

// ──────────────────────── Start ────────────────────────
function initStartButton() {
    $('#startBtn').addEventListener('click', startOptimization);
}

function initActionButtons() {
    $('#btnRestart').addEventListener('click', () => {
        $('#resultsSection').classList.add('hidden');
        $('#progressSection').classList.add('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    $('#btnExport').addEventListener('click', exportReport);
}

// ──────────────────────── Optimization ────────────────────────
async function startOptimization() {
    if (state.isRunning) return;

    state.isRunning = true;
    setStatus('running', '优化中...');
    const startBtn = $('#startBtn');
    startBtn.disabled = true;
    startBtn.innerHTML = '<div class="spinner"></div> <span>优化中...</span>';

    // Show progress
    $('#progressSection').classList.remove('hidden');
    $('#resultsSection').classList.add('hidden');
    clearLog();
    resetProgress();

    // Show model in progress
    const modelDisplay = MODEL_DISPLAY[state.model] || { short: state.model };
    $('#statModel').textContent = modelDisplay.short;

    // Scroll to progress
    setTimeout(() => {
        $('#progressSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 200);

    try {
        if (state.multiTask) {
            await runMultiTaskOptimization();
        } else {
            await runSingleOptimization();
        }
    } catch (err) {
        addLog(`❌ 错误: ${err.message}`, 'error');
        setStatus('ready', '就绪');
    } finally {
        state.isRunning = false;
        startBtn.disabled = false;
        startBtn.innerHTML = `
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5,3 19,12 5,21"/></svg>
            <span>开始优化</span>`;
    }
}

async function runSingleOptimization() {
    const taskName = { classification: '文本分类', summarization: '文本摘要', translation: '中英翻译' }[state.taskType];
    const algoName = { random_search: '随机搜索', genetic: '遗传算法', bayesian: '贝叶斯优化' }[state.algorithm];
    const modelDisplay = MODEL_DISPLAY[state.model] || { short: state.model };

    addLog(`🚀 开始 ${taskName} 任务 - ${algoName}`, 'info');
    addLog(`📊 迭代: ${state.maxIterations}  模型: ${modelDisplay.short}`, 'info');

    const selectedForTask = getSelectedIndicesForTask(state.taskType);
    const selectedCount = selectedForTask ? selectedForTask.length : (state.tasks[state.taskType]?.count || '全部');
    addLog(`📋 评估数据: ${selectedCount} 条`, 'info');

    // Get custom prompt from textarea
    const customPromptText = ($('#customPromptInput')?.value || '').trim();
    const customPrompts = customPromptText ? [customPromptText] : null;
    if (customPrompts) {
        addLog(`✏️ 使用自定义初始 Prompt`, 'info');
    }
    addLog(`⏳ 实时流式输出已开启，每次迭代完成后立即显示结果`, 'info');

    $('#progressDesc').textContent = `正在使用 ${modelDisplay.short} + ${algoName} 搜索最优 Prompt ...`;
    $('#statIter').textContent = `0 / ${state.maxIterations}`;

    const startTime = Date.now();
    const timerInterval = setInterval(() => {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
        $('#statTime').textContent = `${elapsed}s`;
    }, 1000);

    let bestSoFar = 0;
    let iterCount = 0;

    try {
        // Use SSE streaming endpoint
        const response = await fetch(`${API_BASE}/api/optimize_stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                task_type: state.taskType,
                algorithm: state.algorithm,
                max_iterations: state.maxIterations,
                use_llm_judge: state.useLlmJudge,
                model: state.model,
                data_indices: getSelectedIndicesForTask(state.taskType),
                custom_prompts: customPrompts,
            }),
        });

        if (!response.ok) {
            clearInterval(timerInterval);
            const err = await response.json();
            throw new Error(err.detail || 'Optimization failed');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const jsonStr = line.substring(6).trim();
                if (!jsonStr) continue;

                try {
                    const msg = JSON.parse(jsonStr);

                    if (msg.type === 'iteration') {
                        iterCount++;
                        bestSoFar = Math.max(bestSoFar, msg.score);
                        const progress = (iterCount / state.maxIterations) * 100;

                        // Update UI in real time
                        $('#progressBar').style.width = `${progress}%`;
                        $('#statIter').textContent = `${iterCount} / ${state.maxIterations}`;
                        $('#statCurrent').textContent = msg.score.toFixed(4);
                        $('#statBest').textContent = bestSoFar.toFixed(4);

                        const promptPreview = msg.prompt.length > 60
                            ? msg.prompt.substring(0, 60) + '...' : msg.prompt;
                        const isBest = msg.score >= bestSoFar;
                        addLog(
                            `<span class="log-iter">[Iter ${msg.iteration}]</span> ` +
                            `得分: <span class="log-score">${msg.score.toFixed(4)}</span> ` +
                            `最佳: <span class="log-best">${bestSoFar.toFixed(4)}</span> ` +
                            (isBest ? '⭐ ' : '') +
                            `<span class="log-prompt-preview">${escapeHtml(promptPreview)}</span>`
                        );
                    } else if (msg.type === 'complete') {
                        clearInterval(timerInterval);
                        state.results = msg.result;

                        $('#progressBar').style.width = '100%';
                        $('#progressDesc').textContent = '优化完成!';

                        setStatus('ready', '就绪');
                        showResults(msg.result);
                        addLog(`✅ 优化完成! 最佳得分: ${msg.result.best_score.toFixed(4)}`, 'success');
                    } else if (msg.type === 'error') {
                        clearInterval(timerInterval);
                        throw new Error(msg.message);
                    }
                } catch (parseErr) {
                    if (parseErr.message !== msg?.message) {
                        console.warn('SSE parse error:', parseErr, jsonStr);
                    } else {
                        throw parseErr;
                    }
                }
            }
        }

        clearInterval(timerInterval);

    } catch (err) {
        clearInterval(timerInterval);
        throw err;
    }
}

async function runMultiTaskOptimization() {
    const modelDisplay = MODEL_DISPLAY[state.model] || { short: state.model };
    addLog(`🚀 开始多任务优化模式 (${modelDisplay.short})`, 'info');
    addLog(`📋 任务列表: ${state.multiTaskTypes.join(', ')}`, 'info');

    $('#progressDesc').textContent = `正在使用 ${modelDisplay.short} 对 ${state.multiTaskTypes.length} 个任务进行优化...`;

    const startTime = Date.now();
    const timerInterval = setInterval(() => {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
        $('#statTime').textContent = `${elapsed}s`;
    }, 1000);

    try {
        const response = await fetch(`${API_BASE}/api/optimize_multi`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                task_types: state.multiTaskTypes,
                algorithm: state.algorithm,
                max_iterations: state.maxIterations,
                use_llm_judge: state.useLlmJudge,
                model: state.model,
            }),
        });

        clearInterval(timerInterval);

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Multi-task optimization failed');
        }

        const data = await response.json();
        state.multiResults = data.results;

        // Show best single result for display
        const firstTask = Object.keys(data.results)[0];
        if (data.results[firstTask]?.result) {
            state.results = data.results[firstTask].result;
            await animateResults(data.results[firstTask].result);
            showResults(data.results[firstTask].result);
            showMultiTaskResults(data.results);
        }

        setStatus('ready', '就绪');
        addLog(`✅ 多任务优化完成!`, 'success');
    } catch (err) {
        clearInterval(timerInterval);
        throw err;
    }
}

// ──────────────────────── Animate Progress ────────────────────────
async function animateResults(result) {
    const total = result.all_results.length;
    let bestSoFar = 0;

    for (let i = 0; i < total; i++) {
        const r = result.all_results[i];
        const progress = ((i + 1) / total) * 100;

        bestSoFar = Math.max(bestSoFar, r.score);

        $('#progressBar').style.width = `${progress}%`;
        $('#statIter').textContent = `${i + 1} / ${total}`;
        $('#statCurrent').textContent = r.score.toFixed(4);
        $('#statBest').textContent = bestSoFar.toFixed(4);

        const promptPreview = r.prompt.length > 60 ? r.prompt.substring(0, 60) + '...' : r.prompt;
        const isBest = r.score >= bestSoFar;
        addLog(
            `<span class="log-iter">[Iter ${r.iteration}]</span> ` +
            `得分: <span class="log-score">${r.score.toFixed(4)}</span> ` +
            `最佳: <span class="log-best">${bestSoFar.toFixed(4)}</span> ` +
            (isBest ? '⭐ ' : '') +
            `<span class="log-prompt-preview">${escapeHtml(promptPreview)}</span>`
        );

        // Small delay for visual effect
        await sleep(80);
    }

    // Final progress
    $('#progressBar').style.width = '100%';
    $('#progressDesc').textContent = '优化完成!';
}

// ──────────────────────── Display Results ────────────────────────
function showResults(result) {
    const section = $('#resultsSection');
    section.classList.remove('hidden');

    // Best prompt
    $('#bestScore').textContent = result.best_score.toFixed(4);
    $('#bestPromptText').textContent = result.best_prompt;

    const modelDisplay = MODEL_DISPLAY[state.model] || { short: state.model };

    const metricStr = Object.entries(result.best_metrics || {})
        .filter(([k]) => !['task_type', 'primary_score', 'primary_metric'].includes(k))
        .map(([k, v]) => `<span>📐 ${k}: ${typeof v === 'number' ? v.toFixed(4) : v}</span>`)
        .join('');
    $('#bestPromptMeta').innerHTML =
        `<span>🔄 总迭代: ${result.total_iterations}</span>` +
        `<span>⏱️ 耗时: ${result.duration.toFixed(1)}s</span>` +
        `<span>🧠 模型: ${modelDisplay.short}</span>` +
        `<span>⚙️ 算法: ${result.algorithm}</span>` +
        metricStr;

    // Charts
    renderScoreCurve(result);
    renderScoreDistribution(result);
    renderKeywordContributions(result);
    renderResultsTable(result);

    // Scroll to results
    setTimeout(() => {
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 300);
}

// ──────────────────────── Charts ────────────────────────
function renderScoreCurve(result) {
    destroyChart('scoreCurveChart');

    const ctx = $('#scoreCurveChart').getContext('2d');
    const labels = result.score_curve.map((_, i) => i + 1);

    state.charts.scoreCurveChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: '当前得分',
                    data: result.score_curve,
                    borderColor: 'rgba(99,102,241,0.8)',
                    backgroundColor: 'rgba(99,102,241,0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 4,
                    pointBackgroundColor: 'rgba(99,102,241,1)',
                    borderWidth: 2,
                },
                {
                    label: '最佳得分',
                    data: result.best_score_curve,
                    borderColor: 'rgba(52,211,153,0.9)',
                    backgroundColor: 'rgba(52,211,153,0.05)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                    pointBackgroundColor: 'rgba(52,211,153,1)',
                    borderWidth: 2,
                    borderDash: [5, 3],
                },
            ],
        },
        options: chartOptions('迭代次数', '得分'),
    });
}

function renderScoreDistribution(result) {
    destroyChart('scoreDistChart');

    const ctx = $('#scoreDistChart').getContext('2d');
    const scores = result.score_curve;

    // Create histogram bins
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const range = max - min || 0.1;
    const binCount = 8;
    const binWidth = range / binCount;
    const bins = Array(binCount).fill(0);
    const binLabels = [];

    for (let i = 0; i < binCount; i++) {
        const low = min + i * binWidth;
        binLabels.push(low.toFixed(3));
    }

    scores.forEach(s => {
        let bin = Math.floor((s - min) / binWidth);
        if (bin >= binCount) bin = binCount - 1;
        bins[bin]++;
    });

    const colors = bins.map((_, i) => {
        const t = i / (binCount - 1);
        const r = Math.round(99 + (52 - 99) * t);
        const g = Math.round(102 + (211 - 102) * t);
        const b = Math.round(241 + (153 - 241) * t);
        return `rgba(${r},${g},${b},0.7)`;
    });

    state.charts.scoreDistChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: binLabels,
            datasets: [{
                label: '频次',
                data: bins,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace('0.7', '1')),
                borderWidth: 1,
                borderRadius: 6,
            }],
        },
        options: chartOptions('得分区间', '频次'),
    });
}

function renderKeywordContributions(result) {
    destroyChart('keywordChart');

    const kw = result.keyword_contributions || {};
    const entries = Object.entries(kw).filter(([_, v]) => v !== 0).slice(0, 20);

    if (entries.length === 0) {
        $('#keywordSection').style.display = 'none';
        return;
    }

    $('#keywordSection').style.display = '';
    const ctx = $('#keywordChart').getContext('2d');
    const labels = entries.map(([k]) => k.length > 10 ? k.substring(0, 10) + '…' : k);
    const values = entries.map(([_, v]) => v);

    const colors = values.map(v =>
        v > 0 ? 'rgba(52,211,153,0.7)' : 'rgba(251,113,133,0.7)'
    );
    const borderColors = values.map(v =>
        v > 0 ? 'rgba(52,211,153,1)' : 'rgba(251,113,133,1)'
    );

    state.charts.keywordChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: '贡献度',
                data: values,
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 1,
                borderRadius: 4,
            }],
        },
        options: {
            ...chartOptions('关键词', '边际贡献'),
            indexAxis: 'y',
        },
    });
}

function renderResultsTable(result) {
    const tbody = $('#resultsTableBody');
    tbody.innerHTML = '';

    const bestScore = result.best_score;

    result.all_results.forEach(r => {
        const isBest = r.score === bestScore;
        const metricsStr = Object.entries(r.metrics || {})
            .filter(([k]) => !['task_type', 'primary_score', 'primary_metric'].includes(k))
            .map(([k, v]) => `${k}: ${typeof v === 'number' ? v.toFixed(4) : v}`)
            .join(', ');

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${r.iteration}</td>
            <td class="prompt-cell" title="${escapeHtml(r.prompt)}">${escapeHtml(r.prompt)}</td>
            <td class="score-cell ${isBest ? 'best-row' : ''}">${r.score.toFixed(4)} ${isBest ? '⭐' : ''}</td>
            <td class="metrics-cell">${metricsStr}</td>
        `;
        tbody.appendChild(tr);
    });
}

function showMultiTaskResults(results) {
    const section = $('#multiResultsSection');
    section.classList.remove('hidden');

    destroyChart('multiTaskChart');
    const ctx = $('#multiTaskChart').getContext('2d');

    const taskNames = { classification: '文本分类', summarization: '文本摘要', translation: '中英翻译' };
    const labels = [];
    const scores = [];
    const colors = ['rgba(99,102,241,0.7)', 'rgba(168,85,247,0.7)', 'rgba(34,211,238,0.7)'];

    Object.entries(results).forEach(([task, data], i) => {
        if (data.result) {
            labels.push(taskNames[task] || task);
            scores.push(data.result.best_score);
        }
    });

    state.charts.multiTaskChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: '最佳得分',
                data: scores,
                backgroundColor: colors.slice(0, labels.length),
                borderColor: colors.slice(0, labels.length).map(c => c.replace('0.7', '1')),
                borderWidth: 1,
                borderRadius: 8,
                barThickness: 60,
            }],
        },
        options: chartOptions('任务', '最佳得分'),
    });
}

// ──────────────────────── Chart Helpers ────────────────────────
function chartOptions(xLabel, yLabel) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: '#94a3b8',
                    font: { family: "'Inter', sans-serif", size: 12 },
                    padding: 16,
                    usePointStyle: true,
                    pointStyleWidth: 10,
                },
            },
            tooltip: {
                backgroundColor: 'rgba(15, 17, 23, 0.95)',
                titleColor: '#f1f5f9',
                bodyColor: '#94a3b8',
                borderColor: 'rgba(99,102,241,0.3)',
                borderWidth: 1,
                padding: 12,
                cornerRadius: 8,
                titleFont: { family: "'Inter', sans-serif", weight: '600' },
                bodyFont: { family: "'JetBrains Mono', monospace", size: 12 },
            },
        },
        scales: {
            x: {
                ticks: { color: '#64748b', font: { size: 11 } },
                grid: { color: 'rgba(148,163,184,0.06)' },
                title: { display: true, text: xLabel, color: '#64748b', font: { size: 12 } },
            },
            y: {
                ticks: { color: '#64748b', font: { size: 11 } },
                grid: { color: 'rgba(148,163,184,0.06)' },
                title: { display: true, text: yLabel, color: '#64748b', font: { size: 12 } },
            },
        },
    };
}

function destroyChart(id) {
    if (state.charts[id]) {
        state.charts[id].destroy();
        delete state.charts[id];
    }
}

// ──────────────────────── Log ────────────────────────
function addLog(message, type = 'default') {
    const log = $('#liveLog');
    const line = document.createElement('div');
    line.className = 'log-line';
    line.innerHTML = message;
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
}

function clearLog() {
    $('#liveLog').innerHTML = '';
}

// ──────────────────────── Progress ────────────────────────
function resetProgress() {
    $('#progressBar').style.width = '0%';
    $('#statIter').textContent = '0 / 0';
    $('#statCurrent').textContent = '—';
    $('#statBest').textContent = '—';
    $('#statTime').textContent = '0s';
    $('#statModel').textContent = '—';
}

// ──────────────────────── Status ────────────────────────
function setStatus(type, text) {
    const pill = $('#statusPill');
    pill.className = 'status-pill' + (type === 'running' ? ' running' : '');
    pill.querySelector('.status-text').textContent = text;
}

// ──────────────────────── Export ────────────────────────
function exportReport() {
    if (!state.results) return;

    const r = state.results;
    const modelDisplay = MODEL_DISPLAY[state.model] || { short: state.model };

    let report = `# Prompt Optimization Report\n\n`;
    report += `## Configuration\n`;
    report += `- Task: ${r.task_type}\n`;
    report += `- Algorithm: ${r.algorithm}\n`;
    report += `- Model: ${modelDisplay.short}\n`;
    report += `- Iterations: ${r.total_iterations}\n`;
    report += `- Duration: ${r.duration.toFixed(1)}s\n\n`;
    report += `## Best Result\n`;
    report += `- **Score**: ${r.best_score.toFixed(4)}\n`;
    report += `- **Prompt**: ${r.best_prompt}\n\n`;
    report += `## Metrics\n`;
    Object.entries(r.best_metrics || {}).forEach(([k, v]) => {
        report += `- ${k}: ${typeof v === 'number' ? v.toFixed(4) : v}\n`;
    });
    report += `\n## Score Curve\n`;
    r.score_curve.forEach((s, i) => {
        report += `Iteration ${i + 1}: ${s.toFixed(4)}\n`;
    });
    report += `\n## Keyword Contributions\n`;
    Object.entries(r.keyword_contributions || {}).forEach(([k, v]) => {
        report += `- "${k}": ${v.toFixed(4)}\n`;
    });
    report += `\n## All Results\n`;
    r.all_results.forEach(res => {
        report += `\n### Iteration ${res.iteration} (Score: ${res.score.toFixed(4)})\n`;
        report += `Prompt: ${res.prompt}\n`;
    });

    const blob = new Blob([report], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `prompt_optimization_report_${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
}

// ──────────────────────── Utilities ────────────────────────
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
