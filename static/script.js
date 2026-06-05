/**
 * script.js
 * OralScan AI – Frontend Logic
 * Handles: file upload, drag-drop, API call, result rendering
 */

'use strict';

/* ── State ── */
let selectedFile = null;

/* ── DOM References ── */
const dropZone       = document.getElementById('drop-zone');
const dropInner      = document.getElementById('drop-inner');
const fileInput      = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const previewImg     = document.getElementById('preview-img');
const previewFname   = document.getElementById('preview-filename');
const previewSize    = document.getElementById('preview-size');
const analyzeBtn     = document.getElementById('analyze-btn');

const emptyState     = document.getElementById('empty-state');
const loadingOverlay = document.getElementById('loading-overlay');
const resultsContent = document.getElementById('results-content');

const rbPct          = document.getElementById('rb-pct');
const rbSeverity     = document.getElementById('rb-severity');
const severityFill   = document.getElementById('severity-fill');
const resultBanner   = document.getElementById('result-banner');

const resOriginal    = document.getElementById('res-original');
const resMask        = document.getElementById('res-mask');
const resOverlay     = document.getElementById('res-overlay');

const mpDice         = document.getElementById('mp-dice');
const mpIou          = document.getElementById('mp-iou');
const mpAcc          = document.getElementById('mp-acc');

/* ── Hero metric refs ── */
const hmDice = document.getElementById('hm-dice');
const hmIou  = document.getElementById('hm-iou');
const hmAcc  = document.getElementById('hm-acc');

/* ══════════════════════════════════════════
   Model Status Check
══════════════════════════════════════════ */
async function fetchModelInfo() {
  try {
    const res  = await fetch('/model_info');
    const data = await res.json();

    const chip = document.getElementById('model-status-chip');
    const txt  = document.getElementById('model-status-text');

    if (data.model_loaded) {
      chip.classList.add('online');
      txt.textContent = 'Model Online';
    } else {
      chip.classList.add('offline');
      txt.textContent = 'Demo Mode';
    }

    // Populate hero metrics
    const m = data.metrics;
    if (m) {
      hmDice.textContent = (m.dice  * 100).toFixed(1) + '%';
      hmIou.textContent  = (m.iou   * 100).toFixed(1) + '%';
      hmAcc.textContent  = (m.accuracy * 100).toFixed(1) + '%';
    }
  } catch {
    document.getElementById('model-status-text').textContent = 'Status Unknown';
  }
}

/* ══════════════════════════════════════════
   File Handling
══════════════════════════════════════════ */
function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewFname.textContent = file.name.length > 24
      ? file.name.substring(0, 22) + '…'
      : file.name;
    previewSize.textContent = formatBytes(file.size);

    dropInner.classList.add('hidden');
    previewContainer.classList.remove('hidden');
    analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

function resetUpload() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src  = '';
  previewContainer.classList.add('hidden');
  dropInner.classList.remove('hidden');
  analyzeBtn.disabled = true;

  // Hide results
  resultsContent.classList.add('hidden');
  emptyState.classList.remove('hidden');
}

/* File input change */
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) { selectedFile = file; showPreview(file); }
});

/* Drag & Drop */
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    selectedFile = file;
    showPreview(file);
  }
});

/* Click on drop zone (but not on buttons) */
dropZone.addEventListener('click', (e) => {
  if (
    e.target.closest('button') ||
    e.target === fileInput ||
    selectedFile
  ) return;
  fileInput.click();
});

/* ══════════════════════════════════════════
   Loading Steps Animation
══════════════════════════════════════════ */
const STEPS = ['ls-1', 'ls-2', 'ls-3', 'ls-4'];
let stepTimer = null;
let currentStep = 0;

function startLoadingSteps() {
  currentStep = 0;
  STEPS.forEach(id => document.getElementById(id).classList.remove('active'));
  document.getElementById(STEPS[0]).classList.add('active');

  stepTimer = setInterval(() => {
    currentStep++;
    if (currentStep < STEPS.length) {
      STEPS.forEach(id => document.getElementById(id).classList.remove('active'));
      document.getElementById(STEPS[currentStep]).classList.add('active');
    }
  }, 900);
}

function stopLoadingSteps() {
  clearInterval(stepTimer);
}

/* ══════════════════════════════════════════
   Render Results
══════════════════════════════════════════ */
function renderResults(data) {
  // Images
  resOriginal.src = `data:image/png;base64,${data.original_b64}`;
  resMask.src     = `data:image/png;base64,${data.mask_b64}`;
  resOverlay.src  = `data:image/png;base64,${data.overlay_b64}`;

  // Cancer %
  const pct = data.cancer_pct;
  rbPct.textContent      = pct.toFixed(2) + '%';
  rbSeverity.textContent = data.severity;

  // Severity bar fill (capped at 100%)
  const fillPct = Math.min(pct * 1.5, 100);  // scale for visual emphasis
  severityFill.style.width = fillPct + '%';

  // Banner color class
  resultBanner.className = 'result-banner banner-' + data.severity_level;

  // Model metrics
  const m = data.model_metrics;
  if (m) {
    mpDice.textContent = (m.dice     * 100).toFixed(1) + '%';
    mpIou.textContent  = (m.iou      * 100).toFixed(1) + '%';
    mpAcc.textContent  = (m.accuracy * 100).toFixed(1) + '%';
  }
}

/* ══════════════════════════════════════════
   Main Analysis Call
══════════════════════════════════════════ */
async function runAnalysis() {
  if (!selectedFile) return;

  // Show loading
  emptyState.classList.add('hidden');
  resultsContent.classList.add('hidden');
  loadingOverlay.classList.remove('hidden');
  startLoadingSteps();
  analyzeBtn.disabled = true;

  try {
    const form = new FormData();
    form.append('file', selectedFile);

    const res = await fetch('/predict', {
      method: 'POST',
      body: form,
    });

    const data = await res.json();

    if (!res.ok || data.error) {
      throw new Error(data.error || 'Server error');
    }

    stopLoadingSteps();
    loadingOverlay.classList.add('hidden');

    renderResults(data);
    resultsContent.classList.remove('hidden');

    // Smooth scroll to results on mobile
    if (window.innerWidth < 1024) {
      document.getElementById('results-panel')
        .scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

  } catch (err) {
    stopLoadingSteps();
    loadingOverlay.classList.add('hidden');
    emptyState.classList.remove('hidden');
    alert('Analysis failed: ' + err.message);
    console.error(err);
  } finally {
    analyzeBtn.disabled = false;
  }
}

/* ══════════════════════════════════════════
   Init
══════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  fetchModelInfo();
});
