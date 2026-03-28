/**
 * PostureCoach — Frontend Application
 * Real-time posture analysis with camera + image upload
 */

// ── Configuration ──────────────────────────────────────────────────
const API_BASE = window.location.origin;
const ENDPOINTS = {
    health: `${API_BASE}/api/health`,
    analyze: `${API_BASE}/api/analyze`,
    analyzeFrame: `${API_BASE}/api/analyze-frame`,
    modelInfo: `${API_BASE}/api/model-info`,
};

// ── State ──────────────────────────────────────────────────────────
const state = {
    mode: 'upload',           // 'upload' | 'camera'
    cameraStream: null,
    isLiveMode: false,
    liveInterval: null,
    isAnalyzing: false,
    stats: {
        total: 0,
        good: 0,
        bad: 0,
        confidences: [],
    },
};

// ── DOM References ─────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
    // Nav
    statusDot: $('#status-dot'),
    statusText: $('#status-text'),

    // Mode
    btnUploadMode: $('#btn-upload-mode'),
    btnCameraMode: $('#btn-camera-mode'),

    // Upload
    uploadZone: $('#upload-zone'),
    fileInput: $('#file-input'),
    btnBrowse: $('#btn-browse'),

    // Camera
    cameraZone: $('#camera-zone'),
    cameraVideo: $('#camera-video'),
    cameraCanvas: $('#camera-canvas'),
    btnCapture: $('#btn-capture'),
    btnToggleLive: $('#btn-toggle-live'),

    // Live overlay
    liveOverlay: $('#live-overlay'),
    liveStatusBanner: $('#live-status-banner'),
    livePostureText: $('#live-posture-text'),
    liveConfidenceText: $('#live-confidence-text'),

    // Preview
    previewZone: $('#preview-zone'),
    previewImage: $('#preview-image'),
    btnAnalyze: $('#btn-analyze'),
    btnClear: $('#btn-clear'),

    // Results
    resultsEmpty: $('#results-empty'),
    resultsLoading: $('#results-loading'),
    resultsDisplay: $('#results-display'),
    postureResult: $('#posture-result'),
    confCircle: $('#conf-circle'),
    confValue: $('#conf-value'),
    postureLabel: $('#posture-label'),
    postureClass: $('#posture-class'),
    inferenceTime: $('#inference-time'),
    annotatedContainer: $('#annotated-container'),
    annotatedImage: $('#annotated-image'),
    detectionsList: $('#detections-list'),
    btnNewAnalysis: $('#btn-new-analysis'),

    // Stats
    statTotal: $('#stat-total'),
    statGood: $('#stat-good'),
    statBad: $('#stat-bad'),
    statAvgConf: $('#stat-avg-conf'),
};


// ══════════════════════════════════════════════════════════════════
//  INITIALIZATION
// ══════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    checkServerHealth();
    bindEvents();
    initScrollAnimations();
    initCounterAnimations();
    initNavScrollBehavior();
});


// ── Server Health ──────────────────────────────────────────────────
async function checkServerHealth() {
    try {
        const res = await fetch(ENDPOINTS.health);
        const data = await res.json();
        if (data.status === 'ok') {
            dom.statusDot.classList.add('online');
            dom.statusDot.classList.remove('offline');
            dom.statusText.textContent = 'Model Ready';
        } else {
            throw new Error('Bad status');
        }
    } catch {
        dom.statusDot.classList.add('offline');
        dom.statusDot.classList.remove('online');
        dom.statusText.textContent = 'Server Offline';
    }
}

// Periodic health check
setInterval(checkServerHealth, 15000);


// ══════════════════════════════════════════════════════════════════
//  EVENT BINDING
// ══════════════════════════════════════════════════════════════════

function bindEvents() {
    // Mode switching
    dom.btnUploadMode.addEventListener('click', () => switchMode('upload'));
    dom.btnCameraMode.addEventListener('click', () => switchMode('camera'));

    // Upload
    dom.uploadZone.addEventListener('click', () => dom.fileInput.click());
    dom.btnBrowse.addEventListener('click', (e) => { e.stopPropagation(); dom.fileInput.click(); });
    dom.fileInput.addEventListener('change', handleFileSelect);

    // Drag & drop
    dom.uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); dom.uploadZone.classList.add('drag-over'); });
    dom.uploadZone.addEventListener('dragleave', () => dom.uploadZone.classList.remove('drag-over'));
    dom.uploadZone.addEventListener('drop', handleDrop);

    // Camera
    dom.btnCapture.addEventListener('click', captureFrame);
    dom.btnToggleLive.addEventListener('click', toggleLiveMode);

    // Preview / Analysis
    dom.btnAnalyze.addEventListener('click', analyzeCurrentImage);
    dom.btnClear.addEventListener('click', clearPreview);
    dom.btnNewAnalysis.addEventListener('click', resetAnalysis);

    // Smooth scroll for nav links
    $$('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = document.querySelector(link.getAttribute('href'));
            if (target) target.scrollIntoView({ behavior: 'smooth' });
        });
    });

    // Hero buttons
    const btnGetStarted = $('#btn-get-started');
    const btnViewMetrics = $('#btn-view-metrics');
    if (btnGetStarted) {
        btnGetStarted.addEventListener('click', (e) => {
            e.preventDefault();
            document.querySelector('#analyzer').scrollIntoView({ behavior: 'smooth' });
        });
    }
    if (btnViewMetrics) {
        btnViewMetrics.addEventListener('click', (e) => {
            e.preventDefault();
            document.querySelector('#metrics').scrollIntoView({ behavior: 'smooth' });
        });
    }
}


// ══════════════════════════════════════════════════════════════════
//  MODE SWITCHING
// ══════════════════════════════════════════════════════════════════

function switchMode(mode) {
    state.mode = mode;

    // Update buttons
    dom.btnUploadMode.classList.toggle('active', mode === 'upload');
    dom.btnCameraMode.classList.toggle('active', mode === 'camera');

    // Stop live mode if switching away from camera
    if (mode !== 'camera') {
        stopLiveMode();
        stopCamera();
    }

    // Show/hide zones
    if (mode === 'upload') {
        dom.cameraZone.style.display = 'none';
        dom.uploadZone.style.display = 'flex';
        dom.previewZone.style.display = 'none';
    } else {
        dom.uploadZone.style.display = 'none';
        dom.previewZone.style.display = 'none';
        dom.cameraZone.style.display = 'flex';
        startCamera();
    }
}


// ══════════════════════════════════════════════════════════════════
//  FILE UPLOAD
// ══════════════════════════════════════════════════════════════════

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) loadImageFile(file);
}

function handleDrop(e) {
    e.preventDefault();
    dom.uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        loadImageFile(file);
    }
}

function loadImageFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        dom.previewImage.src = e.target.result;
        dom.uploadZone.style.display = 'none';
        dom.previewZone.style.display = 'flex';
    };
    reader.readAsDataURL(file);
}

function clearPreview() {
    dom.previewImage.src = '';
    dom.previewZone.style.display = 'none';
    dom.uploadZone.style.display = 'flex';
    dom.fileInput.value = '';
}


// ══════════════════════════════════════════════════════════════════
//  CAMERA
// ══════════════════════════════════════════════════════════════════

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
        });
        state.cameraStream = stream;
        dom.cameraVideo.srcObject = stream;
    } catch (err) {
        console.error('Camera access failed:', err);
        alert('Could not access camera. Please ensure camera permissions are granted.');
        switchMode('upload');
    }
}

function stopCamera() {
    if (state.cameraStream) {
        state.cameraStream.getTracks().forEach(t => t.stop());
        state.cameraStream = null;
        dom.cameraVideo.srcObject = null;
    }
}

function captureFrame() {
    // Stop live mode if running
    stopLiveMode();

    const video = dom.cameraVideo;
    const canvas = dom.cameraCanvas;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.85);

    // Show in preview and auto-analyze
    dom.previewImage.src = dataUrl;
    dom.cameraZone.style.display = 'none';
    dom.previewZone.style.display = 'flex';
    stopCamera();

    // Auto-trigger analysis
    analyzeCurrentImage();
}

function toggleLiveMode() {
    if (state.isLiveMode) {
        stopLiveMode();
    } else {
        startLiveMode();
    }
}

function startLiveMode() {
    state.isLiveMode = true;
    dom.btnToggleLive.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
        Stop Live
    `;
    dom.btnToggleLive.classList.add('live-active');

    // Show the live overlay on top of the video
    if (dom.liveOverlay) {
        dom.liveOverlay.style.display = 'flex';
        dom.livePostureText.textContent = 'Analyzing...';
        dom.liveConfidenceText.textContent = '--';
        dom.liveStatusBanner.className = 'live-status-banner';
    }

    // Analyze frames every 1 second
    analyzeLiveFrame(); // immediate first analysis
    state.liveInterval = setInterval(analyzeLiveFrame, 1000);
}

function stopLiveMode() {
    state.isLiveMode = false;
    if (state.liveInterval) {
        clearInterval(state.liveInterval);
        state.liveInterval = null;
    }
    dom.btnToggleLive.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        Start Live Analysis
    `;
    dom.btnToggleLive.classList.remove('live-active');

    // Hide the live overlay
    if (dom.liveOverlay) {
        dom.liveOverlay.style.display = 'none';
    }
}

async function analyzeLiveFrame() {
    if (state.isAnalyzing || !state.cameraStream) return;
    state.isAnalyzing = true;

    try {
        const video = dom.cameraVideo;
        const canvas = dom.cameraCanvas;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.7);

        const res = await fetch(ENDPOINTS.analyzeFrame, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataUrl }),
        });

        const data = await res.json();
        if (data.posture && data.posture !== 'unknown') {
            const isGood = data.posture === 'Good';
            const confPercent = Math.round(data.confidence * 100);

            // Update LIVE overlay on the video
            if (dom.liveOverlay) {
                dom.livePostureText.textContent = isGood ? '✓ Good Posture' : '✗ Bad Posture';
                dom.liveConfidenceText.textContent = `${confPercent}%`;
                dom.liveStatusBanner.className = `live-status-banner ${isGood ? 'good' : 'bad'}`;
            }

            // Also update the results panel
            showResults({
                posture: data.posture,
                confidence: data.confidence,
                detections: [{
                    class: data.posture,
                    confidence: data.confidence,
                }],
                inference_time_ms: 0,
            }, true);
            updateStats(data.posture, data.confidence);
        }
    } catch (err) {
        console.error('Live analysis error:', err);
    } finally {
        state.isAnalyzing = false;
    }
}


// ══════════════════════════════════════════════════════════════════
//  ANALYSIS
// ══════════════════════════════════════════════════════════════════

async function analyzeCurrentImage() {
    if (state.isAnalyzing) return;
    state.isAnalyzing = true;

    const imgSrc = dom.previewImage.src;
    if (!imgSrc) return;

    // Show loading
    showState('loading');

    try {
        const res = await fetch(ENDPOINTS.analyze, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imgSrc }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || 'Analysis failed');
        }

        const data = await res.json();
        showResults(data, false);
        updateStats(data.posture, data.confidence);
    } catch (err) {
        console.error('Analysis error:', err);
        alert(`Analysis failed: ${err.message}`);
        showState('empty');
    } finally {
        state.isAnalyzing = false;
    }
}


// ══════════════════════════════════════════════════════════════════
//  RESULT DISPLAY
// ══════════════════════════════════════════════════════════════════

function showState(s) {
    dom.resultsEmpty.style.display = s === 'empty' ? 'flex' : 'none';
    dom.resultsLoading.style.display = s === 'loading' ? 'flex' : 'none';
    dom.resultsDisplay.style.display = s === 'results' ? 'flex' : 'none';
}

function showResults(data, isLive) {
    showState('results');

    const isGood = data.posture === 'Good';
    const confPercent = Math.round(data.confidence * 100);

    // Posture result card
    dom.postureResult.className = `posture-result ${isGood ? 'good' : 'bad'}`;

    // Confidence ring
    const circumference = 2 * Math.PI * 52; // r=52
    const offset = circumference - (data.confidence * circumference);
    dom.confCircle.style.strokeDasharray = circumference;
    dom.confCircle.style.strokeDashoffset = offset;

    // Text
    dom.confValue.textContent = confPercent;
    dom.postureLabel.textContent = isGood ? '✓ Good Posture' : '✗ Bad Posture';
    dom.postureClass.textContent = isGood
        ? 'Your posture looks great! Keep it up.'
        : 'Your posture needs improvement. Try sitting straighter.';

    if (data.inference_time_ms) {
        dom.inferenceTime.textContent = `Inference: ${data.inference_time_ms}ms`;
    } else {
        dom.inferenceTime.textContent = '';
    }

    // Annotated image
    if (data.annotated_image && !isLive) {
        dom.annotatedContainer.style.display = 'block';
        dom.annotatedImage.src = data.annotated_image;
    } else {
        dom.annotatedContainer.style.display = 'none';
    }

    // Detection list
    dom.detectionsList.innerHTML = '';
    if (data.detections && data.detections.length > 0) {
        data.detections.forEach(det => {
            const item = document.createElement('div');
            item.className = 'detection-item';
            const clsLower = det.class.toLowerCase();
            item.innerHTML = `
                <span class="det-class ${clsLower}">${det.class}</span>
                <span class="det-conf">${(det.confidence * 100).toFixed(1)}%</span>
            `;
            dom.detectionsList.appendChild(item);
        });
    }
}

function resetAnalysis() {
    showState('empty');
    if (state.mode === 'upload') {
        clearPreview();
    } else {
        dom.previewZone.style.display = 'none';
        dom.cameraZone.style.display = 'flex';
        startCamera();
    }
}


// ══════════════════════════════════════════════════════════════════
//  SESSION STATS
// ══════════════════════════════════════════════════════════════════

function updateStats(posture, confidence) {
    state.stats.total++;
    if (posture === 'Good') state.stats.good++;
    else state.stats.bad++;
    state.stats.confidences.push(confidence);

    dom.statTotal.textContent = state.stats.total;
    dom.statGood.textContent = state.stats.good;
    dom.statBad.textContent = state.stats.bad;

    const avg = state.stats.confidences.reduce((a, b) => a + b, 0) / state.stats.confidences.length;
    dom.statAvgConf.textContent = `${(avg * 100).toFixed(1)}%`;
}


// ══════════════════════════════════════════════════════════════════
//  SCROLL ANIMATIONS
// ══════════════════════════════════════════════════════════════════

function initScrollAnimations() {
    // Add animate-on-scroll class to elements
    const animTargets = $$('.metric-card, .step-card, .session-stat, .config-item, .training-config');
    animTargets.forEach(el => el.classList.add('animate-on-scroll'));

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, i) => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.classList.add('visible');
                }, i * 80);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.15, rootMargin: '0px 0px -40px 0px' });

    animTargets.forEach(el => observer.observe(el));

    // Metric bar fills
    const metricObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const fill = entry.target.querySelector('.metric-fill');
                if (fill) {
                    const width = fill.style.getPropertyValue('--fill');
                    fill.style.width = width;
                }
                metricObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.3 });

    $$('.metric-card').forEach(card => metricObserver.observe(card));
}


// ══════════════════════════════════════════════════════════════════
//  COUNTER ANIMATIONS
// ══════════════════════════════════════════════════════════════════

function initCounterAnimations() {
    const counterObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const counters = entry.target.querySelectorAll('[data-count]');
                counters.forEach(el => animateCounter(el));
                counterObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.3 });

    $$('.hero-stats, .metrics-grid').forEach(section => counterObserver.observe(section));
}

function animateCounter(el) {
    const target = parseFloat(el.dataset.count);
    const isFloat = target % 1 !== 0;
    const duration = 1500;
    const startTime = performance.now();

    function update(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = target * eased;

        el.textContent = isFloat ? current.toFixed(2) : Math.round(current);

        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            el.textContent = isFloat ? target.toFixed(2) : target;
        }
    }

    requestAnimationFrame(update);
}


// ══════════════════════════════════════════════════════════════════
//  NAV SCROLL BEHAVIOR
// ══════════════════════════════════════════════════════════════════

function initNavScrollBehavior() {
    const navbar = $('#navbar');
    const sections = $$('section[id]');

    window.addEventListener('scroll', () => {
        // Navbar background
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }

        // Active section tracking
        let current = '';
        sections.forEach(section => {
            const top = section.offsetTop - 100;
            if (window.scrollY >= top) {
                current = section.getAttribute('id');
            }
        });

        $$('.nav-link').forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('data-section') === current) {
                link.classList.add('active');
            }
        });
    });
}
