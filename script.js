const API_URL = 'http://localhost:5000/predict';

const fileInput     = document.getElementById('fileInput');
const dropZone      = document.getElementById('dropZone');
const dropIdle      = document.getElementById('dropIdle');
const dropPreview   = document.getElementById('dropPreview');
const previewImg    = document.getElementById('previewImg');
const removeBtn     = document.getElementById('removeBtn');
const chooseBtn     = document.getElementById('chooseBtn');
const predictBtn    = document.getElementById('predictBtn');
const resultSection = document.getElementById('resultSection');
const ageValue      = document.getElementById('ageValue');
const ageBar        = document.getElementById('ageBar');
const genderValue   = document.getElementById('genderValue');
const genderIcon    = document.getElementById('genderIcon');
const genderConf    = document.getElementById('genderConf');
const errorBox      = document.getElementById('errorBox');

let selectedFile = null;

// Nút chọn ảnh từ máy
chooseBtn.addEventListener('click', () => fileInput.click());

// Click vào drop zone cũng mở file dialog
dropZone.addEventListener('click', (e) => {
  if (!e.target.closest('.btn-remove') && !e.target.closest('#chooseBtn')) {
    fileInput.click();
  }
});

fileInput.addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

// Drag & drop
dropZone.addEventListener('dragover', e => {
  e.preventDefault(); dropZone.classList.add('over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('over');
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

removeBtn.addEventListener('click', e => { e.stopPropagation(); resetAll(); });

function handleFile(file) {
  if (!file.type.startsWith('image/')) { alert('Chọn file ảnh (JPG, PNG, WEBP)'); return; }
  if (file.size > 10 * 1024 * 1024) { alert('File tối đa 10MB'); return; }

  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    dropIdle.style.display = 'none';
    dropPreview.style.display = 'flex';
    predictBtn.disabled = false;
    hideResult();
  };
  reader.readAsDataURL(file);
}

function resetAll() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  dropIdle.style.display = 'flex';
  dropPreview.style.display = 'none';
  predictBtn.disabled = true;
  hideResult();
}

// Phân tích
predictBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  setLoading(true);
  hideResult();

  const form = new FormData();
  form.append('file', selectedFile);

  try {
    const res  = await fetch(API_URL, { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok || data.error) { showError(data.error || 'Lỗi không xác định'); return; }
    showResult(data);
  } catch {
    showError('Không kết nối được server. Hãy chạy main.py trước!');
  } finally {
    setLoading(false);
  }
});

function setLoading(on) {
  predictBtn.querySelector('.btn-inner').style.display   = on ? 'none' : 'flex';
  predictBtn.querySelector('.btn-loading').style.display = on ? 'flex' : 'none';
  predictBtn.disabled = on;
}

function showResult(data) {
  resultSection.style.display = 'flex';
  errorBox.style.display = 'none';

  const age = data.age;
  ageValue.textContent = age;
  setTimeout(() => { ageBar.style.width = Math.min(age, 100) + '%'; }, 50);

  const isMale = data.gender === 'Nam';
  genderIcon.textContent = isMale ? '♂' : '♀';
  genderIcon.style.color = isMale ? '#60a5fa' : '#f9a8d4';
  genderValue.textContent = data.gender;
  genderValue.style.color = isMale ? '#60a5fa' : '#f9a8d4';
  genderConf.textContent  = `Độ chắc: ${data.gender_confidence}%`;
}

function showError(msg) {
  resultSection.style.display = 'flex';
  errorBox.style.display = 'block';
  errorBox.textContent = '⚠️ ' + msg;
}

function hideResult() {
  resultSection.style.display = 'none';
  ageBar.style.width = '0%';
}