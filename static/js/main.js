// ─── SHARED UTILITIES ─────────────────────────────────────────────────────────

// Toast notification
function showToast(msg, type = 'info', duration = 3000) {
  let el = document.getElementById('toast');
  if (!el) {
    el = document.createElement('div');
    el.id = 'toast';
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.className = `show ${type}`;
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.className = el.className.replace('show','').trim(); }, duration);
}

// Generic API call
async function api(endpoint, body = null, method = null) {
  const opts = {
    method:  method || (body ? 'POST' : 'GET'),
    headers: { 'Content-Type': 'application/json' },
  };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(endpoint, opts);
  return res.json();
}

// Logout
async function logout() {
  await api('/api/logout');
  window.location.href = '/login';
}

// Load user info into sidebar
async function loadSidebarUser() {
  try {
    const d = await api('/api/me');
    if (d.user) {
      const name = d.user.name || 'User';
      if (document.getElementById('sidebar-name'))
        document.getElementById('sidebar-name').textContent = name;
      if (document.getElementById('sidebar-email'))
        document.getElementById('sidebar-email').textContent = d.user.email || '';
      if (document.getElementById('user-avatar'))
        document.getElementById('user-avatar').textContent = name[0].toUpperCase();
    }
  } catch(e) {}
}

// Greeting
function setGreeting(elId) {
  const el = document.getElementById(elId);
  if (!el) return;
  const h = new Date().getHours();
  let g = 'Good evening';
  if (h < 12) g = 'Good morning';
  else if (h < 17) g = 'Good afternoon';
  el.textContent = g + '! Ready to practice?';
}

document.addEventListener('DOMContentLoaded', () => {
  loadSidebarUser();
  setGreeting('dash-greeting');
});
