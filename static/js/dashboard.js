// ─── DASHBOARD ────────────────────────────────────────────────────────────────

const DISPLAY_NAMES = {
  tadasana: "Tadasana", vrikshasana: "Vrikshasana", warrior_i: "Warrior I",
  warrior_ii: "Warrior II", goddess: "Goddess", downward_dog: "Downward Dog",
  cobra: "Cobra", plank: "Plank", triangle: "Triangle", child_pose: "Child's Pose"
};

async function loadDashboard() {
  try {
    const d = await api('/api/me');
    if (!d.user) { window.location.href = '/login'; return; }

    // Sidebar
    document.getElementById('sidebar-name').textContent = d.user.name;
    document.getElementById('sidebar-email').textContent = d.user.email;
    document.getElementById('user-avatar').textContent = d.user.name[0].toUpperCase();

    // Stats
    document.getElementById('stat-total').textContent = d.total_sessions || 0;

    let bestPose = '–', bestAcc = 0, totalAcc = 0, posesTried = 0;
    if (d.stats && d.stats.length > 0) {
      posesTried = d.stats.length;
      d.stats.forEach(s => {
        totalAcc += s.avg_acc;
        if (s.avg_acc > bestAcc) { bestAcc = s.avg_acc; bestPose = DISPLAY_NAMES[s.pose_name] || s.pose_name; }
      });
      document.getElementById('stat-best-pose').textContent = bestPose;
      document.getElementById('stat-avg').textContent = (totalAcc / posesTried).toFixed(1) + '%';
      document.getElementById('stat-poses').textContent = posesTried;

      // Pose bars
      const bars = document.getElementById('pose-bars');
      bars.innerHTML = '';
      d.stats.forEach(s => {
        const name = DISPLAY_NAMES[s.pose_name] || s.pose_name;
        const acc  = s.avg_acc.toFixed(1);
        bars.innerHTML += `
          <div class="pose-bar-item">
            <div class="pbi-label">
              <span class="pbi-name">${name}</span>
              <span class="pbi-val">${acc}%</span>
            </div>
            <div class="pbi-bar">
              <div class="pbi-fill" style="width:0%" data-target="${acc}"></div>
            </div>
          </div>`;
      });
      // Animate bars
      setTimeout(() => {
        document.querySelectorAll('.pbi-fill').forEach(el => {
          el.style.width = el.dataset.target + '%';
        });
      }, 100);

      // Pose library accuracy
      d.stats.forEach(s => {
        const el = document.getElementById('acc-' + s.pose_name);
        if (el) el.textContent = s.avg_acc.toFixed(1) + '%';
      });
    } else {
      document.getElementById('stat-best-pose').textContent = '–';
      document.getElementById('stat-avg').textContent = '–';
      document.getElementById('stat-poses').textContent = '0';
    }

    // History
    const histEl = document.getElementById('history-list');
    if (d.history && d.history.length > 0) {
      histEl.innerHTML = d.history.map(h => {
        const name = DISPLAY_NAMES[h.pose_name] || h.pose_name;
        const time = new Date(h.detected_at).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
        return `<div class="history-item">
          <span class="hi-pose">${name}</span>
          <span class="hi-acc">${h.accuracy.toFixed(1)}%</span>
          <span class="hi-time">${time}</span>
        </div>`;
      }).join('');
    }

  } catch(e) {
    console.error('Dashboard error:', e);
    showToast('Failed to load dashboard', 'error');
  }
}

document.addEventListener('DOMContentLoaded', loadDashboard);
