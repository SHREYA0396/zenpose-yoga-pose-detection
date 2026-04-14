// ─── ZENPOSE PRACTICE ENGINE ──────────────────────────────────────────────────

// ─── STATE ────────────────────────────────────────────────────────────────────
const state = {
  cameraOn:     false,
  detecting:    false,
  voiceOn:      true,
  targetPose:   null,
  lastVoice:    0,
  voiceCooldown: 5000,   // ms between voice announcements
  detectionInterval: null,
  stream:       null,
  lastPose:     null,
  goodPoseStart: null,
  consecutiveGood: 0,
};

const DISPLAY_NAMES = {
  tadasana: "Tadasana (Mountain Pose)",
  vrikshasana: "Vrikshasana (Tree Pose)",
  warrior_i: "Warrior I",
  warrior_ii: "Warrior II",
  goddess: "Goddess Pose",
  downward_dog: "Downward Dog",
  cobra: "Cobra Pose",
  plank: "Plank",
  triangle: "Triangle Pose",
  child_pose: "Child's Pose",
};

const POSE_EMOJIS = {
  tadasana:'🧍', vrikshasana:'🌲', warrior_i:'⚔️', warrior_ii:'🗡️',
  goddess:'🌟', downward_dog:'🐕', cobra:'🐍', plank:'📐',
  triangle:'🔺', child_pose:'🙏'
};

// ─── DOM REFS ─────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const video       = $('user-video');
const canvas      = $('overlay-canvas');
const ctx         = canvas ? canvas.getContext('2d') : null;
const placeholder = $('video-placeholder');
const liveBadge   = $('live-badge');
const accVal      = $('acc-val');
const abFill      = $('ab-fill');
const abPct       = $('ab-pct');
const scoreNum    = $('score-num');
const scoreRing   = $('score-ring-fill');
const scoreStatus = $('score-status');
const detectedName= $('detected-pose-name');
const detectedDesc= $('detected-pose-desc');
const pdcDot      = $('pdc-dot');
const vcMessage   = $('vc-message');
const cuesList    = $('cues-list');
const corrCard    = $('corrections-card');
const corrList    = $('corrections-list');
const cameraBtnEl = $('camera-btn');
const targetLabel = $('target-pose-label');

// ─── CAMERA ───────────────────────────────────────────────────────────────────
async function toggleCamera() {
  if (state.cameraOn) {
    stopCamera();
  } else {
    await startCamera();
  }
}

async function startCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = state.stream;
    await video.play();
    state.cameraOn = true;
    placeholder.classList.add('hidden');
    liveBadge.classList.remove('hidden');
    cameraBtnEl.textContent = '⏹ Stop Camera';
    cameraBtnEl.style.background = 'rgba(239,68,68,0.15)';
    cameraBtnEl.style.borderColor = 'rgba(239,68,68,0.4)';
    cameraBtnEl.style.color = '#f87171';
    startDetectionLoop();
    speak("Camera started. Select a pose to begin practice.");
  } catch(err) {
    console.error('Camera error:', err);
    showToast('Camera access denied. Check browser permissions.', 'error');
    // Simulate mode
    state.cameraOn = true;
    placeholder.innerHTML = `
      <div class="vp-icon">🤖</div>
      <div class="vp-text">Simulation Mode</div>
      <div class="vp-sub">No camera — using AI simulation</div>`;
    liveBadge.classList.remove('hidden');
    cameraBtnEl.textContent = '⏹ Stop';
    startDetectionLoop();
  }
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
  }
  video.srcObject = null;
  state.cameraOn = false;
  clearInterval(state.detectionInterval);
  state.detectionInterval = null;
  liveBadge.classList.add('hidden');
  placeholder.classList.remove('hidden');
  placeholder.innerHTML = `
    <div class="vp-icon">📹</div>
    <div class="vp-text">Camera Off</div>
    <div class="vp-sub">Click "Start Camera" to begin</div>`;
  cameraBtnEl.textContent = '📹 Start Camera';
  cameraBtnEl.style.cssText = '';
  resetUI();
}

// ─── CAPTURE FRAME ────────────────────────────────────────────────────────────
function captureFrame() {
  if (!video.videoWidth) return null;
  const cap = document.createElement('canvas');
  cap.width  = video.videoWidth;
  cap.height = video.videoHeight;
  const c = cap.getContext('2d');
  // Mirror for natural feel
  c.translate(cap.width, 0);
  c.scale(-1, 1);
  c.drawImage(video, 0, 0, cap.width, cap.height);
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
    const body  = { target_pose: state.targetPose };
    if (frame) body.frame = frame;

    const res = await fetch('/api/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const d = await res.json();
    if (d.error) { console.warn('Detect error:', d.error); return; }
    updateUI(d);
  } catch(e) {
    console.error('Detection loop error:', e);
  } finally {
    state.detecting = false;
  }
}

// ─── UPDATE UI ────────────────────────────────────────────────────────────────
function updateUI(d) {
  const acc       = d.accuracy || 0;
  const poseName  = d.pose || '';
  const dispName  = d.display_name || DISPLAY_NAMES[poseName] || poseName;
  const isCorrect = d.is_correct;
  const fb        = d.feedback || {};

  // — Accuracy bar —
  abFill.style.width = acc + '%';
  abPct.textContent  = acc.toFixed(1) + '%';
  accVal.textContent = acc.toFixed(1) + '%';

  // — Score ring —
  const circumference = 326.7;
  const offset = circumference - (acc / 100) * circumference;
  scoreRing.style.strokeDashoffset = offset;
  scoreNum.textContent = Math.round(acc);

  // — Ring color by accuracy —
  if (acc >= 85) {
    scoreRing.style.stroke = '#22c55e';
    scoreStatus.textContent = fb.good_msg || 'Excellent pose!';
    scoreStatus.className = 'score-status good';
  } else if (acc >= 65) {
    scoreRing.style.stroke = '#f59e0b';
    scoreStatus.textContent = 'Getting there — check the cues';
    scoreStatus.className = 'score-status';
  } else {
    scoreRing.style.stroke = '#ef4444';
    scoreStatus.textContent = 'Needs adjustment — follow corrections';
    scoreStatus.className = 'score-status bad';
  }

  // — Detected pose —
  const emoji = POSE_EMOJIS[poseName] || '🧘';
  detectedName.textContent = `${emoji} ${dispName}`;
  detectedDesc.textContent = fb.description || '';
  pdcDot.classList.add('active');

  // — Annotated frame —
  if (d.annotated_frame) {
    const overlay = $('annotated-overlay');
    overlay.src = d.annotated_frame;
    overlay.classList.remove('hidden');
  }

  // — Cues —
  if (fb.cues && fb.cues.length) {
    cuesList.innerHTML = fb.cues.map(c => `<div class="cue-item">${c}</div>`).join('');
  }

  // — Corrections —
  const corrections = fb.corrections || [];
  if (corrections.length > 0 && !isCorrect) {
    corrCard.classList.remove('hidden');
    corrList.innerHTML = corrections.map(c => `<div class="correction-item">${c}</div>`).join('');
  } else {
    corrCard.classList.add('hidden');
    corrList.innerHTML = '';
  }

  // — Good pose streak —
  if (isCorrect) {
    state.consecutiveGood++;
    if (state.consecutiveGood === 5) {
      showToast('🎉 Holding it well! Great form!', 'success');
    }
  } else {
    state.consecutiveGood = 0;
  }

  // — Voice feedback —
  const now = Date.now();
  if (state.voiceOn && (now - state.lastVoice) > state.voiceCooldown) {
    let voiceMsg = '';
    if (isCorrect && acc >= 85) {
      voiceMsg = fb.good_msg || `${dispName}. Excellent form!`;
      state.voiceCooldown = 7000;
    } else if (corrections.length > 0) {
      voiceMsg = corrections[0];
      state.voiceCooldown = 5000;
    } else if (fb.cues && fb.cues.length > 0) {
      const cueIdx = Math.floor(Math.random() * fb.cues.length);
      voiceMsg = fb.cues[cueIdx];
      state.voiceCooldown = 6000;
    }
    if (voiceMsg) {
      speak(voiceMsg);
      state.lastVoice = now;
      vcMessage.textContent = voiceMsg;
      vcMessage.classList.add('speaking');
      setTimeout(() => vcMessage.classList.remove('speaking'), 3000);
    }
  }

  state.lastPose = poseName;
}

// ─── VOICE (Web Speech API) ───────────────────────────────────────────────────
function speak(text) {
  if (!state.voiceOn) return;
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate   = 0.92;
  utter.pitch  = 1.0;
  utter.volume = 0.95;
  // Prefer a calm female English voice if available
  const voices = window.speechSynthesis.getVoices();
  const preferred = voices.find(v =>
    v.lang.startsWith('en') && v.name.toLowerCase().includes('female')
  ) || voices.find(v =>
    v.lang.startsWith('en') && (v.name.includes('Samantha') || v.name.includes('Karen') || v.name.includes('Moira'))
  ) || voices.find(v => v.lang.startsWith('en'));
  if (preferred) utter.voice = preferred;
  window.speechSynthesis.speak(utter);
}

function toggleVoice() {
  state.voiceOn = !state.voiceOn;
  const btn1 = $('voice-toggle');
  const btn2 = $('vc-toggle-btn');
  const icon = $('vc-icon');
  if (state.voiceOn) {
    if (btn1) btn1.textContent = '🔊 Voice On';
    if (btn2) { btn2.textContent = 'ON'; btn2.classList.remove('off'); }
    if (icon) icon.textContent = '🔊';
    vcMessage.textContent = 'Voice guidance enabled.';
    speak("Voice guidance enabled.");
  } else {
    if (btn1) btn1.textContent = '🔇 Voice Off';
    if (btn2) { btn2.textContent = 'OFF'; btn2.classList.add('off'); }
    if (icon) icon.textContent = '🔇';
    vcMessage.textContent = 'Voice guidance disabled.';
    window.speechSynthesis && window.speechSynthesis.cancel();
  }
}

// ─── POSE SELECTION ───────────────────────────────────────────────────────────
function selectPose(pose) {
  state.targetPose = pose;
  const dispName   = DISPLAY_NAMES[pose] || pose;
  const emoji      = POSE_EMOJIS[pose] || '🧘';

  // Update target label
  if (targetLabel) {
    targetLabel.textContent = `Target: ${emoji} ${dispName}`;
    targetLabel.style.color = '#22c55e';
  }

  // Highlight sidebar btn
  document.querySelectorAll('.sp-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.pose === pose);
  });

  // Highlight ref card
  document.querySelectorAll('.ref-card').forEach(c => {
    c.classList.toggle('active', c.id === 'ref-' + pose);
  });

  // Update cues from pose metadata
  updatePoseCues(pose);

  // Announce via voice
  if (state.voiceOn) {
    speak(`${dispName}. ${getPoseCue(pose)}`);
    state.lastVoice = Date.now();
  }

  // Reset consecutive counter
  state.consecutiveGood = 0;

  if (!state.cameraOn) {
    showToast(`Pose selected: ${dispName}. Start camera to detect!`, 'info');
  }
}

function getPoseCue(pose) {
  const cues = {
    tadasana:     "Stand tall with feet together and arms at your sides.",
    vrikshasana:  "Balance on one leg, raise the other foot to your inner thigh.",
    warrior_i:    "Bend your front knee to ninety degrees and raise both arms.",
    warrior_ii:   "Extend both arms parallel to the floor and gaze over your front hand.",
    goddess:      "Step wide, turn feet out, and bend knees deep.",
    downward_dog: "Form an inverted V shape with hips high.",
    cobra:        "Lift your chest and arch your back gently.",
    plank:        "Keep your body in a straight line from head to heels.",
    triangle:     "Extend your arms in a vertical line over a wide stance.",
    child_pose:   "Sit back on your heels and extend arms forward to the mat.",
  };
  return cues[pose] || "Begin the pose now.";
}

function updatePoseCues(pose) {
  // Fetch from API for live cues
  fetch('/api/poses')
    .then(r => r.json())
    .then(d => {
      const poseData = d.feedback && d.feedback[pose];
      if (poseData && poseData.cues) {
        cuesList.innerHTML = poseData.cues
          .map(c => `<div class="cue-item">${c}</div>`).join('');
      }
      if (detectedDesc && poseData) {
        detectedDesc.textContent = poseData.description || '';
      }
    })
    .catch(() => {});
}

// ─── RESET UI ─────────────────────────────────────────────────────────────────
function resetUI() {
  scoreNum.textContent = '0';
  scoreRing.style.strokeDashoffset = '326.7';
  abFill.style.width = '0%';
  abPct.textContent  = '0%';
  accVal.textContent = '–';
  detectedName.textContent = 'Waiting...';
  detectedDesc.textContent = 'Start camera and select a target pose';
  pdcDot.classList.remove('active');
  scoreStatus.textContent = 'No pose detected';
  scoreStatus.className   = 'score-status';
  corrCard.classList.add('hidden');
  vcMessage.textContent = 'Voice feedback will guide your pose corrections.';
}

// ─── INIT ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Pre-load voices
  if (window.speechSynthesis) {
    window.speechSynthesis.getVoices();
    window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
  }

  // Load sidebar user
  loadSidebarUser();

  // Check URL params for auto-select
  const params = new URLSearchParams(window.location.search);
  const poseParam = params.get('pose');
  if (poseParam) {
    selectPose(poseParam);
  } else {
    // Default: select tadasana
    selectPose('tadasana');
  }

  // Keyboard shortcut: Space = toggle camera
  document.addEventListener('keydown', e => {
    if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
      e.preventDefault();
      toggleCamera();
    }
    if (e.code === 'KeyV') toggleVoice();
  });
});
