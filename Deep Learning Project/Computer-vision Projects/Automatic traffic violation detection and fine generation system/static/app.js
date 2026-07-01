const statusNode = document.getElementById('status');
const previewNode = document.getElementById('preview');
const detectionsBody = document.getElementById('detectionsBody');
const finesBox = document.getElementById('finesBox');
const rawResponseNode = document.getElementById('rawResponse');
const modelPathNode = document.getElementById('modelPath');
const classCountNode = document.getElementById('classCount');
const fineCountNode = document.getElementById('fineCount');

function setStatus(message) {
    statusNode.textContent = message;
}

function formatConfidence(value) {
    return typeof value === 'number' ? value.toFixed(3) : '-';
}

function renderDetections(items) {
    detectionsBody.innerHTML = '';
    if (!items || !items.length) {
        detectionsBody.innerHTML = '<tr><td colspan="4" class="muted">No detections returned.</td></tr>';
        return;
    }

    for (const item of items) {
        const row = document.createElement('tr');
        const labelCell = document.createElement('td');
        labelCell.textContent = item.label ?? '-';
        const confidenceCell = document.createElement('td');
        confidenceCell.textContent = formatConfidence(item.confidence);
        const boxCell = document.createElement('td');
        boxCell.textContent = Array.isArray(item.bbox) ? item.bbox.join(', ') : '-';
        const trackCell = document.createElement('td');
        trackCell.textContent = item.track_id ?? '-';
        row.append(labelCell, confidenceCell, boxCell, trackCell);
        detectionsBody.appendChild(row);
    }
}

function renderFines(items) {
    if (!items || !items.length) {
        finesBox.innerHTML = '<span class="muted">No fine records generated for this scan.</span>';
        fineCountNode.textContent = '0';
        return;
    }

    fineCountNode.textContent = String(items.length);
    finesBox.innerHTML = '';

    for (const item of items) {
        const card = document.createElement('div');
        card.className = 'fine-card';
        card.innerHTML = `<strong>${item.violation_type ?? 'violation'}</strong><div class="muted">Fine: ${item.fine_amount ?? '-'} | Plate: ${item.plate_number ?? 'UNKNOWN'}</div><div class="muted">Evidence: ${item.evidence_path ?? '-'}</div>`;
        finesBox.appendChild(card);
    }
}

function renderPreview(result) {
    if (result.annotated_image_url) {
        previewNode.innerHTML = `<img src="${result.annotated_image_url}" alt="Annotated result" />`;
        return;
    }

    if (result.output_video_url) {
        previewNode.innerHTML = `<video controls src="${result.output_video_url}"></video>`;
        return;
    }

    previewNode.innerHTML = '<span class="muted">No preview available.</span>';
}

async function loadMetadata() {
    const response = await fetch('/metadata');
    const data = await response.json();
    modelPathNode.textContent = data.model ?? 'unknown';
    classCountNode.textContent = data.classes ? String(Object.keys(data.classes).length) : '0';
}

async function loadFines() {
    const response = await fetch('/fines');
    const data = await response.json();
    renderFines(data.slice(0, 5));
}

async function analyze() {
    const fileInput = document.getElementById('mediaFile');
    const confidence = document.getElementById('confidence').value || '0.4';
    const frameStride = document.getElementById('frameStride').value || '4';
    const mode = document.querySelector('input[name="mode"]:checked').value;
    const file = fileInput.files[0];

    if (!file) {
        setStatus('Choose a file first.');
        return;
    }

    let endpoint = '/detect/image';
    if (mode === 'video' || (mode === 'auto' && file.type.startsWith('video'))) {
        endpoint = '/detect/video';
    }

    const formData = new FormData();
    formData.append('file', file);

    setStatus('Running model inference...');

    try {
        const response = await fetch(`${endpoint}?conf=${encodeURIComponent(confidence)}&frame_stride=${encodeURIComponent(frameStride)}`, {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.detail || 'Request failed');
        }

        renderPreview(result);
        renderDetections(result.detections || []);
        renderFines(result.fines || []);
        rawResponseNode.textContent = JSON.stringify(result, null, 2);
        setStatus(`Done. ${result.summary?.detections ?? 0} detections, ${result.summary?.fines ?? 0} fines.`);
        await loadFines();
    } catch (error) {
        setStatus(error.message || 'Inference failed.');
    }
}

document.getElementById('analyzeBtn').addEventListener('click', analyze);
document.getElementById('refreshBtn').addEventListener('click', loadFines);

loadMetadata();
loadFines();
