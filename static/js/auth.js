// ─── AUTH LOGIC ────────────────────────────────────────────────────────────────

// ─── HELPERS ──────────────────────────────────────────────────────────────────
function showStep(id) {
  document.querySelectorAll('.auth-step').forEach(s => s.classList.remove('active'));
  const el = document.getElementById(id);
  if (el) el.classList.add('active');
}

function setLoading(btnTextId, spinnerId, loading) {
  document.getElementById(btnTextId).style.opacity = loading ? '0' : '1';
  document.getElementById(spinnerId).classList.toggle('hidden', !loading);
}

function showError(id, msg) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.classList.remove('hidden');
}

function hideError(id) {
  document.getElementById(id).classList.add('hidden');
}

function togglePw(inputId, btn) {
  const inp = document.getElementById(inputId);
  if (inp.type === 'password') { inp.type = 'text'; btn.textContent = '🙈'; }
  else { inp.type = 'password'; btn.textContent = '👁'; }
}

function getOTP(containerSelector) {
  return Array.from(document.querySelectorAll(containerSelector + ' .otp-box'))
    .map(i => i.value).join('');
}

// ─── OTP BOX NAVIGATION ───────────────────────────────────────────────────────
function setupOtpBoxes(containerSelector) {
  const boxes = document.querySelectorAll(containerSelector + ' .otp-box');
  boxes.forEach((box, i) => {
    box.addEventListener('input', e => {
      box.value = box.value.replace(/[^0-9]/g,'').slice(-1);
      box.classList.toggle('filled', box.value !== '');
      if (box.value && i < boxes.length - 1) boxes[i+1].focus();
    });
    box.addEventListener('keydown', e => {
      if (e.key === 'Backspace' && !box.value && i > 0) {
        boxes[i-1].focus();
        boxes[i-1].value = '';
        boxes[i-1].classList.remove('filled');
      }
    });
    box.addEventListener('paste', e => {
      e.preventDefault();
      const pasted = (e.clipboardData || window.clipboardData).getData('text').replace(/\D/g,'');
      boxes.forEach((b, j) => {
        b.value = pasted[j] || '';
        b.classList.toggle('filled', !!b.value);
      });
      const next = Math.min(pasted.length, boxes.length - 1);
      boxes[next].focus();
    });
  });
}

// ─── RESEND TIMER ─────────────────────────────────────────────────────────────
function startResendTimer(btnId, timerId, seconds = 60) {
  const btn   = document.getElementById(btnId);
  const timer = document.getElementById(timerId);
  btn.disabled = true;
  timer.classList.remove('hidden');
  let left = seconds;
  const iv = setInterval(() => {
    timer.textContent = `(${left}s)`;
    left--;
    if (left < 0) {
      clearInterval(iv);
      btn.disabled = false;
      timer.classList.add('hidden');
      timer.textContent = '';
    }
  }, 1000);
}

// ─── REGISTER ─────────────────────────────────────────────────────────────────
async function handleRegister() {
  hideError('reg-error');
  const name  = document.getElementById('reg-name').value.trim();
  const email = document.getElementById('reg-email').value.trim();
  const pw    = document.getElementById('reg-password').value;

  if (!name || !email || !pw) { showError('reg-error','All fields are required'); return; }
  if (pw.length < 6) { showError('reg-error','Password must be at least 6 characters'); return; }

  setLoading('reg-btn-text','reg-spinner', true);
  try {
    const d = await api('/api/register', { name, email, password: pw });
    if (d.error) { showError('reg-error', d.error); return; }
    document.getElementById('otp-reg-subtitle').textContent =
      `6-digit code sent to ${email}`;
    showStep('step-otp-register');
    setupOtpBoxes('#step-otp-register');
    startResendTimer('resend-reg-btn','resend-reg-timer');
    document.querySelector('#step-otp-register .otp-box').focus();
  } catch(e) {
    showError('reg-error','Network error. Please try again.');
  } finally {
    setLoading('reg-btn-text','reg-spinner', false);
  }
}

async function handleVerifyRegisterOTP() {
  hideError('otp-reg-error');
  const otp = getOTP('#step-otp-register');
  if (otp.length !== 6) { showError('otp-reg-error','Enter all 6 digits'); return; }

  setLoading('otp-reg-btn-text','otp-reg-spinner', true);
  try {
    const d = await api('/api/verify-register-otp', { otp });
    if (d.error) { showError('otp-reg-error', d.error); return; }
    showToast('Account created! Redirecting...', 'success');
    setTimeout(() => window.location.href = d.redirect || '/dashboard', 800);
  } catch(e) {
    showError('otp-reg-error','Network error');
  } finally {
    setLoading('otp-reg-btn-text','otp-reg-spinner', false);
  }
}

// ─── LOGIN ────────────────────────────────────────────────────────────────────
async function handleLogin() {
  hideError('login-error');
  const email = document.getElementById('login-email').value.trim();
  const pw    = document.getElementById('login-password').value;

  if (!email || !pw) { showError('login-error','All fields are required'); return; }

  setLoading('login-btn-text','login-spinner', true);
  try {
    const d = await api('/api/login', { email, password: pw });
    if (d.error) { showError('login-error', d.error); return; }
    document.getElementById('otp-login-subtitle').textContent =
      `6-digit code sent to ${email}`;
    showStep('step-otp-login');
    setupOtpBoxes('#step-otp-login');
    startResendTimer('resend-login-btn','resend-login-timer');
    document.querySelector('#step-otp-login .otp-box').focus();
  } catch(e) {
    showError('login-error','Network error. Please try again.');
  } finally {
    setLoading('login-btn-text','login-spinner', false);
  }
}

async function handleVerifyLoginOTP() {
  hideError('otp-login-error');
  const otp = getOTP('#step-otp-login');
  if (otp.length !== 6) { showError('otp-login-error','Enter all 6 digits'); return; }

  setLoading('otp-login-btn-text','otp-login-spinner', true);
  try {
    const d = await api('/api/verify-login-otp', { otp });
    if (d.error) { showError('otp-login-error', d.error); return; }
    showToast('Login successful!', 'success');
    setTimeout(() => window.location.href = d.redirect || '/dashboard', 800);
  } catch(e) {
    showError('otp-login-error','Network error');
  } finally {
    setLoading('otp-login-btn-text','otp-login-spinner', false);
  }
}

// ─── RESEND ───────────────────────────────────────────────────────────────────
async function resendOTP(purpose) {
  try {
    const d = await api('/api/resend-otp', { purpose });
    if (d.error) { showToast(d.error, 'error'); return; }
    showToast('OTP resent to your email', 'success');
    startResendTimer(`resend-${purpose}-btn`, `resend-${purpose}-timer`);
  } catch(e) {
    showToast('Failed to resend OTP', 'error');
  }
}

// ─── INIT ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Allow Enter key to submit
  document.getElementById('reg-password')?.addEventListener('keypress', e => {
    if (e.key === 'Enter') handleRegister();
  });
  document.getElementById('login-password')?.addEventListener('keypress', e => {
    if (e.key === 'Enter') handleLogin();
  });
});
