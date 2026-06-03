document.addEventListener('DOMContentLoaded', () => {
  const runBtn = document.getElementById('runBtn');
  const imageInput = document.getElementById('imageInput');
  const uploadedImg = document.getElementById('uploadedImg');
  const originalXaiImg = document.getElementById('originalXaiImg');
  const predText = document.getElementById('predText');
  const confText = document.getElementById('confText');
  const probsList = document.getElementById('probsList');
  const overlayImg = document.getElementById('overlayImg');
  
  // Custom selectors
  const dropZone = document.getElementById('dropZone');
  const previewPanel = document.getElementById('previewPanel');
  const fileName = document.getElementById('fileName');
  const fileSize = document.getElementById('fileSize');
  const modelSelect = document.getElementById('modelSelect');
  const ttaContainer = document.getElementById('ttaContainer');
  const ttaCheckbox = document.getElementById('ttaCheckbox');
  const scannerStatus = document.getElementById('scannerStatus');
  
  // Results panels
  const emptyResultsState = document.getElementById('emptyResultsState');
  const resultsPanel = document.getElementById('resultsPanel');
  const diagnosisCard = document.getElementById('diagnosisCard');
  const urgencyText = document.getElementById('urgencyText');
  const gradCamSection = document.getElementById('gradCamSection');
  
  // Download report button
  const downloadReportBtn = document.getElementById('downloadReportBtn');

  // Format file size
  function formatBytes(bytes, decimals = 2) {
    if (!+bytes) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
  }

  // Handle file preview loading
  function handleFile(file) {
    if (!file) return;
    
    fileName.textContent = file.name;
    fileSize.textContent = `Size: ${formatBytes(file.size)}`;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      uploadedImg.src = e.target.result;
      originalXaiImg.src = e.target.result;
      previewPanel.style.display = 'block';
      runBtn.disabled = false;
      
      resultsPanel.style.display = 'none';
      emptyResultsState.style.display = 'flex';
    };
    reader.readAsDataURL(file);
  }

  // File Input listener
  imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    handleFile(file);
  });

  // Drag and Drop Events
  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    }, false);
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const file = dt.files[0];
    if (file && file.type.startsWith('image/')) {
      imageInput.files = dt.files;
      handleFile(file);
    }
  });

  // Removed model selection TTA toggling, TTA works for both models now

  // Run Inference Diagnostic
  runBtn.addEventListener('click', async () => {
    const file = imageInput.files[0];
    if (!file) return;

    const model = modelSelect.value;
    const tta = ttaCheckbox.checked;

    // Show spinner overlay
    scannerStatus.style.display = 'flex';
    runBtn.disabled = true;
    modelSelect.disabled = true;
    ttaCheckbox.disabled = true;

    const form = new FormData();
    form.append('image', file);
    form.append('model', model);
    form.append('tta', tta);

    try {
      const res = await fetch('/predict', { method: 'POST', body: form });
      const data = await res.json();

      scannerStatus.style.display = 'none';
      runBtn.disabled = false;
      modelSelect.disabled = false;
      ttaCheckbox.disabled = false;

      if (data.error) {
        alert('Error running model prediction: ' + data.error);
        return;
      }

      // 1. Populating assessment text
      const predVal = data.predicted.toLowerCase();
      const isHealthy = predVal === 'no tumor' || predVal === 'notumor';
      predText.textContent = data.predicted.toUpperCase();
      
      diagnosisCard.className = 'assessment-card';
      if (isHealthy) {
        diagnosisCard.classList.add('healthy');
        urgencyText.textContent = 'Classified as No Tumor.';
      } else {
        diagnosisCard.classList.add('tumor-detected');
        if (predVal === 'glioma') {
          urgencyText.textContent = 'Classified as Glioma.';
        } else if (predVal === 'meningioma') {
          urgencyText.textContent = 'Classified as Meningioma.';
        } else {
          urgencyText.textContent = 'Classified as Pituitary.';
        }
      }

      // 2. Confidence percentage
      const confPercent = (data.confidence * 100);
      confText.textContent = confPercent.toFixed(2) + '%';

      // 3. Render Probability Bars
      probsList.innerHTML = '';
      const entries = Object.entries(data.probs);
      
      let maxClass = '';
      let maxVal = -1;
      for (const [k, v] of entries) {
        if (v > maxVal) {
          maxVal = v;
          maxClass = k;
        }
      }

      entries.forEach(([className, val]) => {
        const percentage = (val * 100).toFixed(2);
        const isMax = className === maxClass;
        
        const barDiv = document.createElement('div');
        barDiv.className = `bar-item ${isMax ? 'highest' : ''}`;
        
        barDiv.innerHTML = `
          <div class="bar-labels">
            <span class="bar-name">${className}</span>
            <span class="bar-percentage">${percentage}%</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill" style="width: 0%"></div>
          </div>
        `;
        
        probsList.appendChild(barDiv);

        setTimeout(() => {
          barDiv.querySelector('.progress-fill').style.width = `${percentage}%`;
        }, 50);
      });

      // 4. Render Grad-CAM
      if (data.overlay) {
        overlayImg.src = 'data:image/png;base64,' + data.overlay;
        gradCamSection.style.display = 'block';
      } else {
        gradCamSection.style.display = 'none';
      }

      // Transition views
      emptyResultsState.style.display = 'none';
      resultsPanel.style.display = 'block';

    } catch (err) {
      scannerStatus.style.display = 'none';
      runBtn.disabled = false;
      modelSelect.disabled = false;
      ttaCheckbox.disabled = false;
      alert('Request failed. Check server console logs.');
    }
  });

  // Exposing PDF Exporter
  downloadReportBtn.addEventListener('click', async () => {
    const file = imageInput.files[0];
    if (!file) return;

    const originalText = downloadReportBtn.innerHTML;
    downloadReportBtn.disabled = true;
    downloadReportBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating PDF...';

    const form = new FormData();
    form.append('image', file);

    try {
      const res = await fetch('/download_report', {
        method: 'POST',
        body: form
      });

      if (!res.ok) {
        throw new Error('Server generated error.');
      }

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Report_${Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

    } catch (e) {
      alert('Failed to generate report: ' + e.message);
    } finally {
      downloadReportBtn.disabled = false;
      downloadReportBtn.innerHTML = originalText;
    }
  });
});
