// ─── ZenPose Practice Engine — Enhanced ──────────────────────────────────────

// ─── STATE ────────────────────────────────────────────────────────────────────
const state = {
  cameraOn:        false,
  detecting:       false,
  voiceOn:         true,
  targetPose:      null,
  lastVoice:       0,
  voiceCooldown:   5000,
  detectionInterval: null,
  stream:          null,
  allPoseData:     {},

  // Streak
  currentStreak:   0,
  maxStreak:       0,
  sessionStreak:   0,   // max streak this session
  correctCount:    0,

  // Session timer
  sessionStartTime: null,
  sessionPose:      null,
  holdSeconds:      0,
  sessionActive:    false,
  sessionDetections: [],  // [{accuracy, is_correct}]
};

// ─── DOM ──────────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ─── CAMERA ───────────────────────────────────────────────────────────────────
async function toggleCamera() {
  state.cameraOn ? stopCamera() : await startCamera();
}

async function startCamera() {
  const video = $('user-video');
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
    speak("Camera started. Get into position to begin.");
  } catch(err) {
    // Simulation fallback
    state.cameraOn = true;
    $('video-placeholder').innerHTML = `
      <div class="vp-icon">🤖</div>
      <div class="vp-text">Simulation Mode</div>
      <div class="vp-sub">No camera — using AI simulation</div>`;
    $('live-badge').classList.remove('hidden');
    $('camera-btn').textContent = '⏹ Stop';
    startDetectionLoop();
    startSessionTimer();
    showToast('Camera unavailable — running simulation mode', 'info');
  }
}

function stopCamera() {
  const video = $('user-video');
  if (state.stream) { state.stream.getTracks().forEach(t => t.stop()); state.stream = null; }
  video.srcObject = null;
  state.cameraOn = false;
  clearInterval(state.detectionInterval);
  state.detectionInterval = null;
  stopSessionTimer();
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
  updateTimerDisplay();
}

function updateTimerDisplay() {
  if (!state.sessionActive) return;
  const elapsed = Math.floor((Date.now() - state.sessionStartTime) / 1000);
  state.holdSeconds = elapsed;
  const m = Math.floor(elapsed / 60);
  const s = elapsed % 60;
  $('session-timer').textContent = `${m}:${s.toString().padStart(2,'0')}`;
  requestAnimationFrame(updateTimerDisplay);
}

function stopSessionTimer() {
  state.sessionActive = false;
  $('end-session-btn').disabled = true;
}

async function endSession() {
  if (!state.sessionActive && !state.cameraOn) return;
  stopCamera();

  // Calculate session average
  const dets = state.sessionDetections;
  const avgAcc = dets.length > 0
    ? dets.reduce((s, d) => s + d.accuracy, 0) / dets.length
    : 0;

  try {
    await api('/api/save-session', {
      pose_name:    state.sessionPose || state.targetPose || 'unknown',
      duration_sec: state.holdSeconds,
      avg_accuracy: Math.round(avgAcc * 10) / 10,
      max_streak:   state.sessionStreak,
    });
    showToast(`Session saved! ${state.holdSeconds}s held, avg ${avgAcc.toFixed(1)}%`, 'success');
  } catch(e) {
    showToast('Session could not be saved', 'error');
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
  state.detectionInterval = setInterval(runDetection, 600);
}

async function runDetection() {
  if (state.detecting) return;
  state.detecting = true;
  try {
    const frame = captureFrame();
    const body  = {
      target_pose:  state.targetPose,
      hold_seconds: state.holdSeconds,
    };
    if (frame) body.frame = frame;
    const res = await fetch('/api/detect', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    const d = await res.json();
    if (!d.error) updateUI(d);
  } catch(e) {
    console.error('Detection error:', e);
  } finally {
    state.detecting = false;
  }
}

// ─── UI UPDATE ────────────────────────────────────────────────────────────────
function updateUI(d) {
  const acc        = d.accuracy  || 0;
  const poseName   = d.pose      || '';
  const dispName   = d.display_name || poseName;
  const emoji      = d.emoji     || '🧘';
  const isCorrect  = d.is_correct;
  const fb         = d.feedback  || {};
  const alertLevel = d.alert_level   || 'warning';
  const alertMsg   = d.alert_message || '';
  const streakData = d.streak    || {};

  // ── Accuracy bar ──
  $('ab-fill').style.width  = acc + '%';
  $('ab-pct').textContent   = acc.toFixed(1) + '%';
  $('acc-val').textContent  = acc.toFixed(1) + '%';

  // ── Score ring ──
  const circumference = 326.7;
  $('score-ring-fill').style.strokeDashoffset = circumference - (acc / 100) * circumference;
  $('score-num').textContent = Math.round(acc);

  // Ring colour
  const ringColors = { perfect:'#22c55e', good:'#4ade80', warning:'#f59e0b', error:'#ef4444' };
  $('score-ring-fill').style.stroke = ringColors[alertLevel] || '#f59e0b';

  // Score status + alert
  $('score-status').textContent = acc >= 85 ? fb.good_msg || 'Excellent!' :
                                  acc >= 65 ? 'Getting there!' : 'Needs correction';
  $('score-status').className   = `score-status${acc >= 85 ? ' good' : acc < 55 ? ' bad' : ''}`;

  const alertEl = $('score-alert');
  alertEl.textContent  = alertMsg;
  alertEl.className    = `score-alert ${alertLevel}`;

  // ── Video alert badge ──
  const badge = $('alert-badge');
  badge.textContent = alertMsg;
  badge.className   = `alert-badge ${alertLevel}`;
  badge.classList.remove('hidden');
  clearTimeout(badge._t);
  badge._t = setTimeout(() => badge.classList.add('hidden'), 3500);

  // ── Detected pose ──
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

  // ── Annotated frame ──
  if (d.annotated_frame) {
    const ov = $('annotated-overlay');
    ov.src = d.annotated_frame;
    ov.classList.remove('hidden');
  }

  // ── Cues ──
  if (fb.cues?.length) {
    $('cues-list').innerHTML = fb.cues.map(c => `<div class="cue-item">${c}</div>`).join('');
  }

  // ── Corrections ──
  const corrections = fb.corrections || [];
  if (corrections.length && !isCorrect) {
    $('corrections-card').classList.remove('hidden');
    $('corrections-list').innerHTML = corrections.map(c => `<div class="correction-item">${c}</div>`).join('');
  } else {
    $('corrections-card').classList.add('hidden');
  }

  // ── Streak ──
  if (streakData.current !== undefined) {
    state.currentStreak = streakData.current;
    state.maxStreak     = streakData.max;
    state.sessionStreak = Math.max(state.sessionStreak, streakData.current);
    $('streak-count').textContent = streakData.current;
    $('max-streak').textContent   = streakData.max;
  }

  // ── Correct count ──
  if (isCorrect) {
    state.correctCount++;
    $('correct-count').textContent = state.correctCount;
  }

  // ── Streak milestone toast ──
  if (streakData.milestone) {
    showToast(`🔥 ${streakData.current} streak! Amazing form!`, 'success');
  }

  // ── Session recording ──
  state.sessionDetections.push({ accuracy: acc, is_correct: isCorrect });

  // ── Voice ──
  const now = Date.now();
  if (state.voiceOn && (now - state.lastVoice) > state.voiceCooldown) {
    let msg = '';
    if (isCorrect && acc >= 85) {
      msg = fb.good_msg || `${dispName}. Excellent form!`;
      state.voiceCooldown = 7000;
    } else if (corrections.length && !isCorrect) {
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

// ─── POSE SELECTION ───────────────────────────────────────────────────────────
function selectPose(pose) {
  state.targetPose = pose;
  const poseData   = state.allPoseData[pose] || {};
  const dispName   = poseData.display_name || pose.replace(/_/g,' ');
  const emoji      = poseData.emoji || '🧘';

  // Reset session per-pose stats
  state.correctCount = 0;
  $('correct-count').textContent = '0';

  $('target-pose-label').textContent = `Target: ${emoji} ${dispName}`;
  $('target-pose-label').style.color = '#22c55e';

  document.querySelectorAll('.sp-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.pose === pose));
  document.querySelectorAll('.ref-card').forEach(c =>
    c.classList.toggle('active', c.id === 'ref-' + pose));

  if (poseData.cues?.length) {
    $('cues-list').innerHTML = poseData.cues.map(c => `<div class="cue-item">${c}</div>`).join('');
  }
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

  if (!state.cameraOn) showToast(`Selected: ${emoji} ${dispName}. Press Space to start camera!`, 'info');
}

// ─── RESET ────────────────────────────────────────────────────────────────────
function resetUI() {
  $('score-num').textContent = '0';
  $('score-ring-fill').style.strokeDashoffset = '326.7';
  $('ab-fill').style.width   = '0%';
  $('ab-pct').textContent    = '0%';
  $('acc-val').textContent   = '–';
  $('detected-pose-name').textContent = 'Waiting…';
  $('detected-pose-desc').textContent = 'Start camera and select a target pose';
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
  state.holdSeconds   = 0;
  state.correctCount  = 0;
  state.sessionStreak = 0;
  $('correct-count').textContent = '0';
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
  loadSidebarUser();

  const pose = new URLSearchParams(window.location.search).get('pose');
  selectPose(pose || 'tadasana');

  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    if (e.code === 'Space') { e.preventDefault(); toggleCamera(); }
    if (e.code === 'KeyV')  toggleVoice();
    if (e.code === 'KeyE' && state.cameraOn) endSession();
  });
});
