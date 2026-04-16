// ─── ZenPose Practice Engine — Stable Hold-Based Detection ───────────────────

// ─── CONSTANTS ────────────────────────────────────────────────────────────────
const HOLD_REQUIRED_MS  = 15000;  // 15 seconds to complete one rep
const DETECT_INTERVAL   = 700;    // ms between frames sent to server
const MIN_ACCURACY      = 65;     // % — minimum to start/continue a hold

// ─── STATE ────────────────────────────────────────────────────────────────────
const state = {
  cameraOn:          false,
  detecting:         false,
  voiceOn:           true,
  targetPose:        null,
  lastVoice:         0,
  voiceCooldown:     5000,
  detectionInterval: null,
  stream:            null,
  allPoseData:       {},

  // Streak / session
  currentStreak:     0,
  maxStreak:         0,
  sessionStreak:     0,
  repCount:          0,   // completed reps (each = 15s hold)

  // Session timer
  sessionStartTime:  null,
  sessionPose:       null,
  sessionSeconds:    0,
  sessionActive:     false,
  sessionDetections: [],

  // Hold timer — tracks current 15s window
  holdStartTime:     null,   // when user first entered correct pose
  holdActive:        false,  // user is currently in a valid hold
  holdElapsedMs:     0,      // how many ms they have held so far
  holdTick:          null,   // rAF handle for hold progress bar
};

// ─── DOM ──────────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ─── CAMERA ───────────────────────────────────────────────────────────────────
async function toggleCamera() {
  state.cameraOn ? stopCamera() : await startCamera();
}

async function startCamera() {
  const video = $('user-video');

  // Camera requires a secure context (localhost or https)
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    console.error('Camera error: mediaDevices API unavailable. Open via http://localhost:5000');
    $('video-placeholder').innerHTML = `
      <div class="vp-icon">🔒</div>
      <div class="vp-text">Camera Blocked</div>
      <div class="vp-sub">Open the app at <b>http://localhost:5000</b> — not 127.0.0.1 or an IP address.</div>`;
    state.cameraOn = true;
    $('live-badge').classList.remove('hidden');
    $('camera-btn').textContent = '⏹ Stop';
    startDetectionLoop();
    startSessionTimer();
    showToast('❌ Open the app at http://localhost:5000 to enable camera.', 'error');
    return;
  }

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width:{ideal:640}, height:{ideal:480}, facingMode:'user' }, audio: false,
    });
    video.srcObject = state.stream;
    await video.play();
    state.cameraOn = true;
    $('video-placeholder').classList.add('hidden');
    $('live-badge').classList.remove('hidden');
    $('camera-btn').textContent = '⏹ Stop Camera';
    $('camera-btn').style.cssText = 'background:rgba(239,68,68,0.15);border-color:rgba(239,68,68,0.4);color:#f87171;';
    startDetectionLoop();
    startSessionTimer();
    speak(`Camera started. Get into ${getPoseName(state.targetPose)} position.`);
  } catch(err) {
    console.error('Camera error:', err.name, err.message);

    let errMsg = 'Camera unavailable';
    let errSub = 'Using AI simulation instead';
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      errMsg = 'Camera Permission Denied';
      errSub = 'Click the camera icon in your browser address bar and allow access, then reload.';
      showToast('❌ Camera permission denied — allow camera access in your browser and reload.', 'error');
    } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
      errMsg = 'No Camera Found';
      errSub = 'No camera detected on this device. Running AI simulation.';
      showToast('No camera detected — running simulation mode.', 'info');
    } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
      errMsg = 'Camera In Use';
      errSub = 'Camera is being used by another app. Close it and reload.';
      showToast('❌ Camera is in use by another app. Close it and reload.', 'error');
    } else {
      showToast('Camera unavailable — running simulation mode', 'info');
    }

    state.cameraOn = true;
    $('video-placeholder').innerHTML = `
      <div class="vp-icon">🤖</div>
      <div class="vp-text">${errMsg}</div>
      <div class="vp-sub">${errSub}</div>`;
    $('live-badge').classList.remove('hidden');
    $('camera-btn').textContent = '⏹ Stop';
    startDetectionLoop();
    startSessionTimer();
  }
}

function stopCamera() {
  const video = $('user-video');
  if (state.stream) { state.stream.getTracks().forEach(t => t.stop()); state.stream = null; }
  video.srcObject = null;
  state.cameraOn = false;
  clearInterval(state.detectionInterval);
  state.detectionInterval = null;
  cancelAnimationFrame(state.holdTick);
  stopSessionTimer();
  resetHoldTimer();
  $('live-badge').classList.add('hidden');
  $('video-placeholder').classList.remove('hidden');
  $('video-placeholder').innerHTML = `
    <div class="vp-icon">📹</div>
    <div class="vp-text">Camera Off</div>
    <div class="vp-sub">Press Space or click Start Camera</div>`;
  $('camera-btn').textContent = '📹 Start Camera';
  $('camera-btn').style.cssText = '';
  resetUI();
}

// ─── SESSION TIMER ────────────────────────────────────────────────────────────
function startSessionTimer() {
  state.sessionStartTime = Date.now();
  state.sessionActive    = true;
  state.sessionPose      = state.targetPose;
  state.sessionDetections = [];
  $('end-session-btn').disabled = false;
  tickSessionTimer();
}

function tickSessionTimer() {
  if (!state.sessionActive) return;
  state.sessionSeconds = Math.floor((Date.now() - state.sessionStartTime) / 1000);
  const m = Math.floor(state.sessionSeconds / 60);
  const s = state.sessionSeconds % 60;
  $('session-timer').textContent = `${m}:${s.toString().padStart(2,'0')}`;
  requestAnimationFrame(tickSessionTimer);
}

function stopSessionTimer() {
  state.sessionActive = false;
  $('end-session-btn').disabled = true;
}

async function endSession() {
  if (!state.sessionActive && !state.cameraOn) return;
  stopCamera();
  const dets   = state.sessionDetections;
  const avgAcc = dets.length > 0
    ? dets.reduce((s, d) => s + d.accuracy, 0) / dets.length : 0;
  try {
    await api('/api/save-session', {
      pose_name:    state.sessionPose || state.targetPose || 'unknown',
      duration_sec: state.sessionSeconds,
      avg_accuracy: Math.round(avgAcc * 10) / 10,
      max_streak:   state.sessionStreak,
    });
    showToast(`Session saved! ${state.repCount} reps completed.`, 'success');
  } catch(e) {
    showToast('Session could not be saved', 'error');
  }
}

// ─── HOLD TIMER ───────────────────────────────────────────────────────────────
function startHold() {
  if (state.holdActive) return;
  state.holdActive    = true;
  state.holdStartTime = Date.now();
  tickHoldBar();
}

function breakHold() {
  if (!state.holdActive) return;
  state.holdActive    = false;
  state.holdStartTime = null;
  state.holdElapsedMs = 0;
  cancelAnimationFrame(state.holdTick);
  renderHoldBar(0);
}

function resetHoldTimer() {
  state.holdActive    = false;
  state.holdStartTime = null;
  state.holdElapsedMs = 0;
  cancelAnimationFrame(state.holdTick);
  renderHoldBar(0);
}

function tickHoldBar() {
  if (!state.holdActive) return;
  state.holdElapsedMs = Date.now() - state.holdStartTime;
  const pct = Math.min(state.holdElapsedMs / HOLD_REQUIRED_MS, 1);
  renderHoldBar(pct);
  if (pct < 1) {
    state.holdTick = requestAnimationFrame(tickHoldBar);
  }
}

function renderHoldBar(pct) {
  const bar  = $('hold-bar-fill');
  const lbl  = $('hold-bar-label');
  const secs = Math.ceil((HOLD_REQUIRED_MS - (pct * HOLD_REQUIRED_MS)) / 1000);
  if (bar) bar.style.width = (pct * 100).toFixed(1) + '%';
  if (lbl) {
    if (pct <= 0)       lbl.textContent = 'Hold the pose for 15s to count a rep';
    else if (pct >= 1)  lbl.textContent = '✅ Rep complete!';
    else                lbl.textContent = `Hold… ${secs}s remaining`;
  }
}

// ─── FRAME CAPTURE ────────────────────────────────────────────────────────────
function captureFrame() {
  const video = $('user-video');
  if (!video.videoWidth) return null;
  const cap = document.createElement('canvas');
  cap.width = video.videoWidth; cap.height = video.videoHeight;
  const c = cap.getContext('2d');
  c.translate(cap.width, 0); c.scale(-1, 1);
  c.drawImage(video, 0, 0);
  return cap.toDataURL('image/jpeg', 0.75);
}

// ─── DETECTION LOOP ───────────────────────────────────────────────────────────
function startDetectionLoop() {
  if (state.detectionInterval) clearInterval(state.detectionInterval);
  state.detectionInterval = setInterval(runDetection, DETECT_INTERVAL);
}

async function runDetection() {
  if (state.detecting || !state.targetPose) return;
  state.detecting = true;

  const heldMs   = state.holdActive ? (Date.now() - state.holdStartTime) : 0;
  const countRep = state.holdActive && heldMs >= HOLD_REQUIRED_MS;

  try {
    const frame = captureFrame();
    const body  = {
      target_pose:  state.targetPose,
      hold_seconds: Math.floor(heldMs / 1000),
      count_rep:    countRep,
    };
    if (frame) body.frame = frame;

    const res = await fetch('/api/detect', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    const d = await res.json();
    if (!d.error) handleDetectionResult(d, countRep);
  } catch(e) {
    console.error('Detection error:', e);
  } finally {
    state.detecting = false;
  }
}

// ─── DETECTION RESULT HANDLER ─────────────────────────────────────────────────
function handleDetectionResult(d, wasCountRep) {
  const targetMatched = d.target_matched;
  const accuracy      = d.accuracy || 0;
  const isCorrect     = d.is_correct;

  // Manage hold timer based on whether user is in correct pose
  if (targetMatched && accuracy >= MIN_ACCURACY) {
    startHold();
  } else {
    breakHold();
  }

  // Rep completion
  if (wasCountRep && isCorrect) {
    state.repCount++;
    $('correct-count').textContent = state.repCount;
    resetHoldTimer();
    speak(`Rep ${state.repCount} complete!`);
    showToast(`🎉 Rep ${state.repCount} complete! Hold again for next rep.`, 'success');
  } else if (wasCountRep && !isCorrect) {
    resetHoldTimer();
    showToast('⚠️ Hold completed but accuracy was too low. Fix your form and try again.', 'warning');
    speak('Accuracy too low. Check your form and hold again.');
  }

  state.sessionDetections.push({ accuracy, is_correct: isCorrect });
  updateUI(d);
}

// ─── UI UPDATE ────────────────────────────────────────────────────────────────
function updateUI(d) {
  const acc        = d.accuracy  || 0;
  const poseName   = d.pose      || '';
  const dispName   = d.display_name || poseName;
  const emoji      = d.emoji     || '🧘';
  const fb         = d.feedback  || {};
  const alertLevel = d.alert_level   || 'warning';
  const alertMsg   = d.alert_message || '';
  const streakData = d.streak    || {};

  $('ab-fill').style.width = acc + '%';
  $('ab-pct').textContent  = acc.toFixed(1) + '%';
  $('acc-val').textContent = acc.toFixed(1) + '%';

  const circumference = 326.7;
  $('score-ring-fill').style.strokeDashoffset = circumference - (acc / 100) * circumference;
  $('score-num').textContent = Math.round(acc);

  const ringColors = { perfect:'#22c55e', good:'#4ade80', warning:'#f59e0b', error:'#ef4444' };
  $('score-ring-fill').style.stroke = ringColors[alertLevel] || '#f59e0b';

  $('score-status').textContent = acc >= 85 ? fb.good_msg || 'Excellent!' :
                                  acc >= 65 ? 'Getting there!' : 'Needs correction';
  $('score-status').className   = `score-status${acc >= 85 ? ' good' : acc < 55 ? ' bad' : ''}`;

  const alertEl = $('score-alert');
  alertEl.textContent = alertMsg;
  alertEl.className   = `score-alert ${alertLevel}`;

  const badge = $('alert-badge');
  badge.textContent = alertMsg;
  badge.className   = `alert-badge ${alertLevel}`;
  badge.classList.remove('hidden');
  clearTimeout(badge._t);
  badge._t = setTimeout(() => badge.classList.add('hidden'), 3500);

  $('detected-pose-name').textContent = `${emoji} ${dispName}`;
  $('detected-pose-desc').textContent = fb.description || '';
  $('pdc-dot').classList.add('active');
  if (fb.benefits) {
    $('pdc-benefits').textContent = '✦ ' + fb.benefits;
    $('pdc-benefits').classList.remove('hidden');
  }
  if (fb.difficulty) {
    const colors = { Beginner:'#22c55e', Intermediate:'#f59e0b', Advanced:'#ef4444' };
    const db = $('difficulty-badge');
    db.textContent = fb.difficulty;
    db.style.background = colors[fb.difficulty] || '#888';
    db.classList.remove('hidden');
  }

  if (d.annotated_frame) {
    const ov = $('annotated-overlay');
    ov.src = d.annotated_frame;
    ov.classList.remove('hidden');
  }

  if (fb.cues?.length) {
    $('cues-list').innerHTML = fb.cues.map(c => `<div class="cue-item">${c}</div>`).join('');
  }

  const corrections = fb.corrections || [];
  if (corrections.length && !d.target_matched) {
    $('corrections-card').classList.remove('hidden');
    $('corrections-list').innerHTML = corrections.map(c => `<div class="correction-item">${c}</div>`).join('');
  } else {
    $('corrections-card').classList.add('hidden');
  }

  if (streakData.current !== undefined) {
    state.currentStreak = streakData.current;
    state.maxStreak     = streakData.max;
    state.sessionStreak = Math.max(state.sessionStreak, streakData.current);
    $('streak-count').textContent = streakData.current;
    $('max-streak').textContent   = streakData.max;
  }

  if (streakData.milestone) {
    showToast(`🔥 ${streakData.current} streak! Amazing!`, 'success');
  }

  const now = Date.now();
  if (state.voiceOn && (now - state.lastVoice) > state.voiceCooldown) {
    let msg = '';
    if (d.target_matched && acc >= 85) {
      msg = fb.good_msg || `${dispName}. Excellent form!`;
      state.voiceCooldown = 8000;
    } else if (!d.target_matched && corrections.length) {
      msg = corrections[0];
      state.voiceCooldown = 5000;
    } else if (fb.cues?.length) {
      msg = fb.cues[Math.floor(Math.random() * fb.cues.length)];
      state.voiceCooldown = 6000;
    }
    if (msg) {
      speak(msg);
      state.lastVoice = now;
      $('vc-message').textContent = msg;
      $('vc-message').classList.add('speaking');
      setTimeout(() => $('vc-message').classList.remove('speaking'), 3000);
    }
  }
}

// ─── VOICE ────────────────────────────────────────────────────────────────────
function speak(text) {
  if (!state.voiceOn || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.9; u.pitch = 1.0; u.volume = 0.95;
  const voices = window.speechSynthesis.getVoices();
  const v = voices.find(v => v.lang.startsWith('en') && /female|samantha|karen|moira/i.test(v.name))
         || voices.find(v => v.lang.startsWith('en'));
  if (v) u.voice = v;
  window.speechSynthesis.speak(u);
}

function toggleVoice() {
  state.voiceOn = !state.voiceOn;
  const on = state.voiceOn;
  $('voice-toggle').textContent = on ? '🔊 Voice On' : '🔇 Voice Off';
  const btn = $('vc-toggle-btn');
  btn.textContent = on ? 'ON' : 'OFF';
  btn.classList.toggle('off', !on);
  $('vc-icon').textContent = on ? '🔊' : '🔇';
  $('vc-message').textContent = on ? 'Voice guidance enabled.' : 'Voice guidance disabled.';
  if (on) speak("Voice guidance enabled.");
  else window.speechSynthesis?.cancel();
}

// ─── POSE REFERENCE IMAGE ─────────────────────────────────────────────────────
// Shows an instant local guide card from already-loaded pose data (no spinner,
// no network wait). Silently fetches a Wikipedia photo in the background; if it
// loads, it replaces the guide card. If it fails, the guide card stays.
function loadPoseReferenceImage(pose) {
  const loading     = $('ref-img-loading');
  const img         = $('pose-ref-img');
  const placeholder = $('ref-img-placeholder');
  const badge       = $('prc-badge');

  if (!img) return;

  // Hide loading spinner and the raw img element
  img.classList.add('hidden');
  if (loading) loading.classList.add('hidden');

  // ── Show instant reference card ───────────────────────────────────────────
  if (placeholder) {
    const pd   = state.allPoseData[pose] || {};
    const emoji = pd.emoji  || '🧘';
    const name  = pd.display_name || pose.replace(/_/g, ' ');
    const cue   = (pd.cues && pd.cues[0]) || pd.description || '';
    const diff  = pd.difficulty || '';
    const diffColor = { Beginner:'#22c55e', Intermediate:'#f59e0b', Advanced:'#ef4444' }[diff] || '#9ba8a0';
    placeholder.innerHTML =
      '<div style="display:flex;flex-direction:column;align-items:center;gap:8px;'
      + 'padding:16px;width:100%;text-align:center;">'
      + `<span style="font-size:3rem;line-height:1;">${emoji}</span>`
      + `<span style="font-size:0.85rem;font-weight:700;color:#c8d8d0;">${name}</span>`
      + (diff
          ? `<span style="font-size:0.65rem;padding:2px 10px;border-radius:20px;`
            + `background:${diffColor}22;color:${diffColor};border:1px solid ${diffColor}44;">${diff}</span>`
          : '')
      + (cue
          ? `<span style="font-size:0.72rem;color:#9ba8a0;line-height:1.5;max-width:170px;">${cue}</span>`
          : '')
      + '</div>';
    placeholder.classList.remove('hidden');
    if (badge) badge.textContent = 'Guide';
  }

  // ── Silently try Wikipedia photo in background ────────────────────────────
  img.onload = () => {
    if (placeholder) placeholder.classList.add('hidden');
    img.classList.remove('hidden');
    if (badge) badge.textContent = 'Photo';
  };
  img.onerror = () => { /* keep guide card — it's already visible */ };
  img.src = `/api/pose-image/${encodeURIComponent(pose)}`;
}

function showPlaceholder(pose) {
  const loading     = $('ref-img-loading');
  const placeholder = $('ref-img-placeholder');
  const phEmoji     = $('ref-ph-emoji');
  if (loading) loading.classList.add('hidden');
  if (placeholder) {
    placeholder.classList.remove('hidden');
    if (phEmoji) phEmoji.textContent = state.allPoseData[pose]?.emoji || '🧘';
  }
}

// ─── POSE SELECTION ───────────────────────────────────────────────────────────
function selectPose(pose) {
  // Prevent switching while camera is live
  if (state.cameraOn) {
    showToast('Stop the camera before selecting a different pose.', 'warning');
    return;
  }

  state.targetPose = pose;
  state.repCount   = 0;
  $('correct-count').textContent = '0';
  resetHoldTimer();
  loadPoseReferenceImage(pose);

  const poseData = state.allPoseData[pose] || {};
  const dispName = poseData.display_name || pose.replace(/_/g,' ');
  const emoji    = poseData.emoji || '🧘';

  $('target-pose-label').textContent = `Target: ${emoji} ${dispName}`;
  $('target-pose-label').style.color = '#22c55e';

  document.querySelectorAll('.sp-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.pose === pose));
  document.querySelectorAll('.ref-card').forEach(c =>
    c.classList.toggle('active', c.id === 'ref-' + pose));

  if (poseData.cues?.length)
    $('cues-list').innerHTML = poseData.cues.map(c => `<div class="cue-item">${c}</div>`).join('');
  if (poseData.description) $('detected-pose-desc').textContent = poseData.description;
  if (poseData.difficulty) {
    const colors = { Beginner:'#22c55e', Intermediate:'#f59e0b', Advanced:'#ef4444' };
    const db = $('difficulty-badge');
    db.textContent = poseData.difficulty;
    db.style.background = colors[poseData.difficulty] || '#888';
    db.classList.remove('hidden');
  }

  if (state.voiceOn) {
    const cue = poseData.cues?.[0] || '';
    speak(`${dispName}. ${cue}`);
    state.lastVoice = Date.now();
  }

  showToast(`Selected: ${emoji} ${dispName}. Press Space to start camera!`, 'info');
}

function getPoseName(pose) {
  if (!pose) return 'your pose';
  const pd = state.allPoseData[pose];
  return pd ? pd.display_name : pose.replace(/_/g,' ');
}

// ─── RESET ────────────────────────────────────────────────────────────────────
function resetUI() {
  $('score-num').textContent = '0';
  $('score-ring-fill').style.strokeDashoffset = '326.7';
  $('ab-fill').style.width   = '0%';
  $('ab-pct').textContent    = '0%';
  $('acc-val').textContent   = '–';
  $('detected-pose-name').textContent = 'Waiting…';
  $('detected-pose-desc').textContent = 'Start camera and get into position';
  $('pdc-dot').classList.remove('active');
  $('score-status').textContent = 'No pose detected';
  $('score-status').className   = 'score-status';
  $('score-alert').textContent  = '';
  $('score-alert').className    = 'score-alert';
  $('corrections-card').classList.add('hidden');
  $('difficulty-badge').classList.add('hidden');
  $('pdc-benefits').classList.add('hidden');
  $('vc-message').textContent = 'Voice feedback will guide your pose corrections.';
  $('session-timer').textContent = '0:00';
  state.sessionSeconds = 0;
  state.repCount       = 0;
  state.sessionStreak  = 0;
  $('correct-count').textContent = '0';
}

// ─── UTILITY ──────────────────────────────────────────────────────────────────
async function api(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return r.json();
}

// ─── INIT ─────────────────────────────────────────────────────────────────────
async function loadPoseData() {
  try {
    const d = await fetch('/api/poses').then(r => r.json());
    state.allPoseData = {};
    (d.poses || []).forEach(p => {
      const fb = d.feedback?.[p] || {};
      state.allPoseData[p] = {
        display_name: d.display_names?.[p] || p,
        emoji:        d.emojis?.[p]        || '🧘',
        cues:         fb.cues              || [],
        corrections:  fb.corrections       || [],
        description:  fb.description       || '',
        difficulty:   fb.difficulty        || 'Beginner',
        benefits:     fb.benefits          || '',
      };
    });
  } catch(e) { console.warn('Could not load pose data', e); }
}

document.addEventListener('DOMContentLoaded', async () => {
  if (window.speechSynthesis) {
    window.speechSynthesis.getVoices();
    window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
  }

  await loadPoseData();

  // Inject hold-progress bar into page dynamically if not in practice.html
  if (!$('hold-bar-fill')) {
    const anchor = $('score-alert')?.parentElement;
    if (anchor) {
      const holdDiv = document.createElement('div');
      holdDiv.style.cssText = 'margin:14px 0 8px;padding:0 4px;';
      holdDiv.innerHTML = `
        <div id="hold-bar-label"
             style="font-size:12px;color:#9ba8a0;margin-bottom:6px;text-align:center;">
          Hold the pose for 15s to count a rep
        </div>
        <div style="background:rgba(255,255,255,0.08);border-radius:8px;
                    height:10px;overflow:hidden;">
          <div id="hold-bar-fill"
               style="height:100%;width:0%;
                      background:linear-gradient(90deg,#22c55e,#4ade80);
                      border-radius:8px;transition:width 0.25s linear;">
          </div>
        </div>`;
      anchor.appendChild(holdDiv);
    }
  }

  if (typeof loadSidebarUser === 'function') loadSidebarUser();

  const pose = new URLSearchParams(window.location.search).get('pose');
  selectPose(pose || 'tadasana');

  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    if (e.code === 'Space') { e.preventDefault(); toggleCamera(); }
    if (e.code === 'KeyV')  toggleVoice();
    if (e.code === 'KeyE' && state.cameraOn) endSession();
  });
});
