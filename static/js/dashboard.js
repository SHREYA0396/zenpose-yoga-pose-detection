// ─── ZenPose Dashboard — Enhanced with Chart.js Analytics ────────────────────

let weeklyChartInst = null;
let poseChartInst   = null;
let allPoseMeta     = {};    // {pose_name: {display_name, emoji, category}}
let allStats        = {};    // pose_name → stats object

// ─── LOAD POSE META ───────────────────────────────────────────────────────────
async function loadPoseMeta() {
  try {
    const d = await api('/api/poses');
    allPoseMeta = {};
    (d.poses || []).forEach(p => {
      allPoseMeta[p] = {
        display_name: d.display_names[p] || p,
        emoji:        d.emojis[p]        || '🧘',
        category:     d.categories[p]    || '',
      };
    });
  } catch(e) { console.warn('Could not load pose meta', e); }
}

// ─── MAIN LOAD ────────────────────────────────────────────────────────────────
async function loadDashboard() {
  try {
    await loadPoseMeta();
    const d = await api('/api/me');
    if (!d.user) { window.location.href = '/login'; return; }

    // ── Sidebar ──
    document.getElementById('sidebar-name').textContent  = d.user.name;
    document.getElementById('sidebar-email').textContent = d.user.email;
    document.getElementById('user-avatar').textContent   = d.user.name[0].toUpperCase();

    // ── Top stats ──
    document.getElementById('stat-total').textContent = d.total_sessions  ?? '0';
    document.getElementById('stat-avg').textContent   = d.avg_accuracy ? d.avg_accuracy + '%' : '–';
    document.getElementById('stat-poses').textContent = d.stats?.length ?? '0';

    const cur = d.streak?.current ?? 0;
    const mx  = d.streak?.max     ?? 0;
    document.getElementById('stat-streak').textContent     = cur;
    document.getElementById('streak-max-label').textContent = mx > 0 ? `Best: ${mx}` : '';

    // ── Build stats map ──
    allStats = {};
    (d.stats || []).forEach(s => { allStats[s.pose_name] = s; });

    // ── Charts ──
    buildWeeklyChart(d.weekly_chart);
    buildPoseChart(d.pose_chart);

    // ── Recommendations ──
    buildRecommendations(d.recommendations);

    // ── History ──
    buildHistory(d.history);

    // ── Pose library ──
    buildPoseLibrary();

  } catch(e) {
    console.error('Dashboard load error:', e);
    showToast('Failed to load dashboard data', 'error');
  }
}

// ─── WEEKLY CHART ─────────────────────────────────────────────────────────────
function buildWeeklyChart(data) {
  const el = document.getElementById('weeklyChart');
  if (!el || !window.Chart) return;

  const hasData = data && data.accuracy && data.accuracy.some(v => v !== null);
  if (!hasData) {
    document.getElementById('weekly-nodata')?.classList.remove('hidden');
    return;
  }

  if (weeklyChartInst) weeklyChartInst.destroy();
  weeklyChartInst = new Chart(el, {
    type: 'line',
    data: {
      labels: data.labels,
      datasets: [{
        label: 'Avg Accuracy %',
        data:  data.accuracy,
        borderColor: '#22c55e',
        backgroundColor: 'rgba(34,197,94,0.08)',
        borderWidth: 2.5,
        pointRadius: 5,
        pointBackgroundColor: '#22c55e',
        pointBorderColor: '#080a0b',
        pointBorderWidth: 2,
        tension: 0.4,
        fill: true,
        spanGaps: true,
      }, {
        label: 'Sessions',
        data:  data.session_counts,
        borderColor: '#d4a853',
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        borderDash: [4,4],
        pointRadius: 3,
        pointBackgroundColor: '#d4a853',
        tension: 0.4,
        yAxisID: 'y2',
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#9ba8a0', font: { size: 11 } } },
        tooltip: {
          backgroundColor: '#0e1114',
          borderColor: 'rgba(34,197,94,0.2)',
          borderWidth: 1,
          titleColor: '#f0f4f0',
          bodyColor: '#9ba8a0',
          callbacks: {
            label: ctx => ctx.datasetIndex === 0
              ? ` Accuracy: ${ctx.raw ?? '–'}%`
              : ` Sessions: ${ctx.raw}`,
          },
        },
      },
      scales: {
        x: { ticks: { color: '#5a6662', font:{size:10} }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: {
          ticks: { color: '#9ba8a0', font:{size:10}, callback: v => v + '%' },
          grid:  { color: 'rgba(255,255,255,0.04)' },
          min: 0, max: 100,
        },
        y2: {
          position: 'right',
          ticks: { color: '#d4a853', font:{size:10} },
          grid:  { drawOnChartArea: false },
          min: 0,
        },
      },
    },
  });
}

// ─── POSE CHART ───────────────────────────────────────────────────────────────
function buildPoseChart(data) {
  const el = document.getElementById('poseChart');
  if (!el || !window.Chart) return;

  if (!data || !data.labels?.length) {
    document.getElementById('pose-nodata')?.classList.remove('hidden');
    return;
  }

  const colors = data.accuracy.map(a =>
    a >= 85 ? 'rgba(34,197,94,0.7)' :
    a >= 65 ? 'rgba(245,158,11,0.7)' :
              'rgba(239,68,68,0.7)');

  if (poseChartInst) poseChartInst.destroy();
  poseChartInst = new Chart(el, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [{
        label: 'Avg Accuracy %',
        data:  data.accuracy,
        backgroundColor: colors,
        borderColor: colors.map(c => c.replace('0.7','1')),
        borderWidth: 1,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0e1114',
          borderColor: 'rgba(34,197,94,0.2)',
          borderWidth: 1,
          titleColor: '#f0f4f0',
          bodyColor: '#9ba8a0',
          callbacks: {
            label: ctx => ` ${ctx.raw}% (${data.counts[ctx.dataIndex]} sessions)`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: '#5a6662', font:{size:9}, maxRotation: 45 },
          grid:  { color: 'rgba(255,255,255,0.03)' },
        },
        y: {
          ticks: { color: '#9ba8a0', font:{size:10}, callback: v => v + '%' },
          grid:  { color: 'rgba(255,255,255,0.04)' },
          min: 0, max: 100,
        },
      },
    },
  });
}

// ─── RECOMMENDATIONS ──────────────────────────────────────────────────────────
function buildRecommendations(recs) {
  const el = document.getElementById('recommendations-list');
  if (!el || !recs?.length) return;

  el.innerHTML = recs.map(r => {
    const hasAcc = r.avg_accuracy !== null;
    const accClass = !hasAcc ? 'new' : r.avg_accuracy >= 80 ? 'high' : r.avg_accuracy >= 60 ? 'medium' : 'low';
    const accText  = !hasAcc ? 'Not tried' : r.avg_accuracy + '%';
    return `
      <div class="rec-item">
        <div class="rec-emoji">${r.emoji}</div>
        <div class="rec-info">
          <div class="rec-name">${r.display_name}</div>
          <div class="rec-tip">${r.tip}</div>
          <a href="/practice?pose=${r.pose}" class="rec-btn">Practice →</a>
        </div>
        <div class="rec-acc">
          <div class="rec-acc-val ${accClass}">${accText}</div>
          <div style="font-size:0.65rem;color:var(--text-3)">${r.sessions} sessions</div>
        </div>
      </div>`;
  }).join('');
}

// ─── HISTORY ──────────────────────────────────────────────────────────────────
function buildHistory(history) {
  const el = document.getElementById('history-list');
  if (!el || !history?.length) return;

  el.innerHTML = history.map(h => {
    const meta  = allPoseMeta[h.pose_name] || {};
    const name  = meta.display_name || h.pose_name;
    const emoji = meta.emoji || '🧘';
    const time  = new Date(h.detected_at).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
    const accClass = h.accuracy >= 85 ? 'acc-high' : h.accuracy >= 65 ? 'acc-mid' : 'acc-low';
    const tick  = h.is_correct ? '✅' : '❌';
    return `
      <div class="history-item">
        <span class="hi-emoji">${emoji}</span>
        <span class="hi-pose">${name}</span>
        <span class="hi-tick">${tick}</span>
        <span class="hi-acc ${accClass}">${h.accuracy.toFixed(1)}%</span>
        <span class="hi-time">${time}</span>
      </div>`;
  }).join('');
}

// ─── POSE LIBRARY ─────────────────────────────────────────────────────────────
function buildPoseLibrary(filterCat = 'all') {
  const el = document.getElementById('pose-library');
  if (!el) return;

  const poses = Object.entries(allPoseMeta).filter(([k, v]) =>
    filterCat === 'all' || v.category === filterCat);

  el.innerHTML = poses.map(([key, meta]) => {
    const stat = allStats[key];
    const acc  = stat ? stat.avg_acc.toFixed(1) + '%' : '–';
    const accClass = !stat ? '' : stat.avg_acc >= 80 ? 'plc-acc-high' : stat.avg_acc >= 60 ? 'plc-acc-mid' : 'plc-acc-low';
    return `
      <div class="pose-lib-card" data-cat="${meta.category}">
        <div class="plc-emoji">${meta.emoji}</div>
        <div class="plc-name">${meta.display_name}</div>
        <div class="plc-cat">${meta.category}</div>
        <div class="plc-acc ${accClass}" id="acc-${key}">${acc}</div>
        <a href="/practice?pose=${key}" class="plc-practice">Practice →</a>
      </div>`;
  }).join('');
}

function filterPoses(cat, btn) {
  document.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  buildPoseLibrary(cat);
}

// ─── INIT ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', loadDashboard);
