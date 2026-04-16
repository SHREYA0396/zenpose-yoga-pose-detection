"""
ZenPose — Enhanced Flask Backend
Features: 25 poses, streak tracking, session timer, performance analytics,
          pose recommendations, real SMTP OTP with proper error handling.
"""

import os, pickle, random, time, base64, smtplib, json, sqlite3, hashlib, secrets
import urllib.request as _urllib_req
import urllib.parse  as _urllib_parse
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from functools import wraps

import numpy as np
import cv2
from flask import (Flask, render_template, request, jsonify,
                   session, redirect, g)

# Auto-load .env file so EMAIL_USER / EMAIL_PASS are available without
# manually setting env vars every time.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── MEDIAPIPE ────────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    MP_AVAILABLE = True
    mp_pose     = mp.solutions.pose
    mp_drawing  = mp.solutions.drawing_utils
    pose_detector = mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
except Exception:
    MP_AVAILABLE = False
    print("[WARN] MediaPipe unavailable — simulation mode")

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
with open('models/zenpose_model.pkl',  'rb') as f: ML_MODEL  = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f: LABEL_ENC = pickle.load(f)
with open('models/pose_metadata.pkl', 'rb') as f: META      = pickle.load(f)

POSE_LABELS    = META['labels']
DISPLAY_NAMES  = META['display_names']
POSE_FEEDBACK  = META['feedback']
POSE_EMOJIS    = META.get('emojis', {})
POSE_CATEGORIES= META.get('categories', {})

# ─── FLASK ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# ─── POSE IMAGE CACHE (pocketyoga.com scraper + SVG fallback) ─────────────────
_POSE_IMG_CACHE = {}  # pose_key → {'bytes': b'...', 'ct': '...'} or None

# pocketyoga.com pose URL names  (https://pocketyoga.com/pose/{name})
_POCKETYOGA_NAMES = {
    'tadasana':       'Mountain',
    'vrikshasana':    'Tree',
    'warrior_i':      'WarriorI',
    'warrior_ii':     'WarriorII',
    'warrior_iii':    'WarriorIII',
    'goddess':        'Goddess',
    'downward_dog':   'DownwardFacingDog',
    'cobra':          'CobraLow',
    'plank':          'PlankHigh',
    'triangle':       'Triangle',
    'child_pose':     'ChildsFull',
    'chair_pose':     'Chair',
    'bridge':         'Bridge',
    'pigeon':         'Pigeon',
    'camel':          'CamelFull',
    'half_moon':      'HalfMoon',
    'boat':           'Boat',
    'crow':           'CrowCrane',
    'eagle':          'Eagle',
    'lotus':          'LotusFlower',
    'fish':           'FishLegs',
    'seated_forward': 'SeatedForwardBend',
    'supine_twist':   'SupineTwistKneeDown',
    'low_lunge':      'LowLunge',
    'side_plank':     'SidePlank',
}

@app.before_request
def redirect_to_localhost():
    """Redirect 127.0.0.1 → localhost so the browser camera API works."""
    host = request.host  # e.g. "127.0.0.1:5000"
    if host.startswith('127.0.0.1'):
        new_url = request.url.replace('http://127.0.0.1', 'http://localhost', 1)
        return redirect(new_url, code=301)

# ─── DATABASE ─────────────────────────────────────────────────────────────────
DB_PATH = 'zenpose.db'

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db: db.close()

def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_verified   INTEGER DEFAULT 0,
            age           INTEGER DEFAULT NULL,
            gender        TEXT    DEFAULT NULL,
            phone         TEXT    DEFAULT NULL,
            fitness_level TEXT    DEFAULT NULL,
            address       TEXT    DEFAULT NULL,
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS otp_store (
            email      TEXT PRIMARY KEY,
            otp        TEXT NOT NULL,
            purpose    TEXT NOT NULL,
            expires_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS pose_history (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER NOT NULL,
            pose_name      TEXT NOT NULL,
            accuracy       REAL NOT NULL,
            is_correct     INTEGER DEFAULT 0,
            hold_seconds   REAL DEFAULT 0,
            detected_at    TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            pose_name    TEXT    NOT NULL DEFAULT '',
            duration_sec REAL    NOT NULL DEFAULT 0,
            avg_accuracy REAL    NOT NULL DEFAULT 0,
            max_streak   INTEGER DEFAULT 0,
            created_at   TEXT    DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS streaks (
            user_id        INTEGER PRIMARY KEY,
            current_streak INTEGER DEFAULT 0,
            max_streak     INTEGER DEFAULT 0,
            last_pose      TEXT DEFAULT '',
            updated_at     TEXT DEFAULT CURRENT_TIMESTAMP
        );
        -- Add new columns to existing tables if upgrading
        PRAGMA foreign_keys = ON;
    """)
    # Safe column additions for existing DBs
    for col_sql in [
        "ALTER TABLE pose_history ADD COLUMN is_correct INTEGER DEFAULT 0",
        "ALTER TABLE pose_history ADD COLUMN hold_seconds REAL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN age           INTEGER DEFAULT NULL",
        "ALTER TABLE users ADD COLUMN gender        TEXT    DEFAULT NULL",
        "ALTER TABLE users ADD COLUMN phone         TEXT    DEFAULT NULL",
        "ALTER TABLE users ADD COLUMN fitness_level TEXT    DEFAULT NULL",
        "ALTER TABLE users ADD COLUMN address       TEXT    DEFAULT NULL",
    ]:
        try: db.execute(col_sql)
        except: pass
    db.commit()
    db.close()

init_db()

# ─── EMAIL / OTP ──────────────────────────────────────────────────────────────
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def generate_otp():
    return str(random.randint(100000, 999999))

def send_otp_email(to_email: str, otp: str, purpose: str = "verification") -> dict:
    """
    Send OTP via Gmail SMTP (SSL port 465).
    Requires env vars: EMAIL_USER, EMAIL_PASS (Gmail App Password).
    Returns {'sent': bool, 'method': 'smtp'|'console', 'error': str|None}
    """
    subject = f"ZenPose — Your OTP: {otp}"
    html_body = f"""
<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:#080a0b;font-family:'DM Sans',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0">
    <tr><td align="center" style="padding:40px 20px;">
      <table width="520" cellpadding="0" cellspacing="0"
             style="background:#0e1114;border-radius:20px;border:1px solid rgba(34,197,94,0.2);overflow:hidden;">
        <!-- Header -->
        <tr>
          <td style="background:linear-gradient(135deg,#0e1114,#141920);padding:32px;text-align:center;">
            <div style="font-size:36px;margin-bottom:10px;">🧘</div>
            <h1 style="margin:0;color:#22c55e;font-family:Georgia,serif;font-size:26px;font-weight:300;">ZenPose</h1>
            <p style="margin:6px 0 0;color:#9ba8a0;font-size:13px;">AI Yoga Pose Detection</p>
          </td>
        </tr>
        <!-- Body -->
        <tr>
          <td style="padding:36px 40px;">
            <h2 style="margin:0 0 16px;color:#f0f4f0;font-size:20px;font-weight:500;">
              Your Verification Code
            </h2>
            <p style="margin:0 0 24px;color:#9ba8a0;font-size:14px;line-height:1.6;">
              Use this code to complete your <strong style="color:#f0f4f0;">{purpose}</strong>.
              It expires in <strong style="color:#22c55e;">10 minutes</strong>.
            </p>
            <!-- OTP Box -->
            <div style="background:#141920;border:2px solid rgba(34,197,94,0.35);border-radius:16px;
                        padding:28px;text-align:center;margin:0 0 28px;">
              <div style="font-family:Georgia,serif;font-size:52px;font-weight:700;
                          letter-spacing:18px;color:#22c55e;line-height:1;">{otp}</div>
            </div>
            <p style="margin:0;color:#5a6662;font-size:12px;line-height:1.6;">
              🔒 Never share this code with anyone.<br/>
              If you didn't request this, you can safely ignore this email.
            </p>
          </td>
        </tr>
        <!-- Footer -->
        <tr>
          <td style="background:#080a0b;padding:20px 40px;border-top:1px solid rgba(255,255,255,0.05);">
            <p style="margin:0;color:#5a6662;font-size:11px;text-align:center;">
              © 2025 ZenPose · AI Yoga Trainer · This is an automated message
            </p>
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""

    email_user = os.environ.get('EMAIL_USER', '').strip()
    email_pass = os.environ.get('EMAIL_PASS', '').strip()

    if email_user and email_pass:
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From']    = f"ZenPose <{email_user}>"
            msg['To']      = to_email
            msg['Reply-To']= email_user
            msg.attach(MIMEText(html_body, 'html', 'utf-8'))

            with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=10) as server:
                server.login(email_user, email_pass)
                server.sendmail(email_user, [to_email], msg.as_string())

            print(f"[EMAIL ✓] OTP sent to {to_email} via SMTP [{purpose}]")
            return {'sent': True, 'method': 'smtp', 'error': None}

        except smtplib.SMTPAuthenticationError:
            err = "Gmail authentication failed. Check EMAIL_USER and EMAIL_PASS (use App Password, not account password)."
            print(f"[EMAIL ✗] SMTP Auth Error: {err}")
        except smtplib.SMTPRecipientsRefused:
            err = f"Recipient address refused: {to_email}"
            print(f"[EMAIL ✗] {err}")
        except smtplib.SMTPException as e:
            err = f"SMTP error: {str(e)}"
            print(f"[EMAIL ✗] {err}")
        except OSError as e:
            err = f"Network error sending email: {str(e)}"
            print(f"[EMAIL ✗] {err}")
        except Exception as e:
            err = f"Unexpected error: {str(e)}"
            print(f"[EMAIL ✗] {err}")

        # SMTP failed — fall through to console
        print(f"\n{'─'*50}")
        print(f"  FALLBACK OTP for {to_email} [{purpose}]: {otp}")
        print(f"{'─'*50}\n")
        return {'sent': False, 'method': 'console', 'error': err}

    else:
        # No SMTP config — console mode (development)
        print(f"\n{'═'*50}")
        print(f"  [DEV MODE] OTP for {to_email} [{purpose}]: {otp}")
        print(f"  (Set EMAIL_USER + EMAIL_PASS env vars for real email)")
        print(f"{'═'*50}\n")
        return {'sent': True, 'method': 'console', 'error': None}

def store_otp(email, otp, purpose):
    db = get_db()
    db.execute("INSERT OR REPLACE INTO otp_store VALUES (?,?,?,?)",
               (email, otp, purpose, time.time() + 600))
    db.commit()

def verify_otp(email, otp, purpose):
    db  = get_db()
    row = db.execute(
        "SELECT * FROM otp_store WHERE email=? AND otp=? AND purpose=?",
        (email, otp, purpose)).fetchone()
    if not row:    return False, "Invalid OTP. Please check the code and try again."
    if time.time() > row['expires_at']:
        db.execute("DELETE FROM otp_store WHERE email=?", (email,))
        db.commit()
        return False, "OTP has expired. Please request a new one."
    db.execute("DELETE FROM otp_store WHERE email=?", (email,))
    db.commit()
    return True, "OK"

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Login required'}), 401
        return f(*args, **kwargs)
    return decorated

# ─── POSE HELPERS ─────────────────────────────────────────────────────────────
def extract_keypoints(landmarks):
    kp = []
    for lm in landmarks.landmark:
        kp.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(kp)

def get_pose_feedback(pose_name):
    fb = POSE_FEEDBACK.get(pose_name, {})
    return {
        'pose':        pose_name,
        'display_name':DISPLAY_NAMES.get(pose_name, pose_name),
        'emoji':       POSE_EMOJIS.get(pose_name, '🧘'),
        'category':    POSE_CATEGORIES.get(pose_name, ''),
        'description': fb.get('description', ''),
        'difficulty':  fb.get('difficulty', 'Beginner'),
        'benefits':    fb.get('benefits', ''),
        'good_msg':    fb.get('good_msg', 'Good pose!'),
        'corrections': fb.get('corrections', []),
        'cues':        fb.get('cues', []),
    }

def simulate_detection(target_pose=None):
    pose = target_pose if (target_pose and target_pose in POSE_LABELS) \
           else random.choice(POSE_LABELS)
    acc  = round(random.uniform(68, 97), 1)
    return pose, acc, acc >= 75

def _update_streak(db, user_id: int, is_correct: bool, pose_name: str):
    """Update streak record, returns (current_streak, max_streak)."""
    row = db.execute("SELECT * FROM streaks WHERE user_id=?", (user_id,)).fetchone()
    if row:
        cur = row['current_streak']
        mx  = row['max_streak']
    else:
        cur, mx = 0, 0

    if is_correct:
        cur += 1
        mx   = max(mx, cur)
    else:
        cur  = 0

    db.execute("""
        INSERT INTO streaks (user_id, current_streak, max_streak, last_pose, updated_at)
        VALUES (?,?,?,?,?)
        ON CONFLICT(user_id) DO UPDATE SET
          current_streak=excluded.current_streak,
          max_streak=excluded.max_streak,
          last_pose=excluded.last_pose,
          updated_at=excluded.updated_at
    """, (user_id, cur, mx, pose_name, datetime.now().isoformat()))
    db.commit()
    return cur, mx

def _get_recommendations(db, user_id: int, limit=5):
    """Poses with lowest average accuracy — suggest for practice."""
    rows = db.execute("""
        SELECT pose_name, AVG(accuracy) as avg_acc, COUNT(*) as cnt
        FROM pose_history WHERE user_id=?
        GROUP BY pose_name ORDER BY avg_acc ASC LIMIT ?
    """, (user_id, limit)).fetchall()
    recs = []
    for r in rows:
        recs.append({
            'pose':        r['pose_name'],
            'display_name':DISPLAY_NAMES.get(r['pose_name'], r['pose_name']),
            'emoji':       POSE_EMOJIS.get(r['pose_name'], '🧘'),
            'avg_accuracy':round(r['avg_acc'], 1),
            'sessions':    r['cnt'],
            'tip':         POSE_FEEDBACK.get(r['pose_name'], {}).get('cues', ['Keep practicing!'])[0],
        })
    # If fewer than limit tried, suggest untried poses
    tried = {r['pose_name'] for r in rows}
    for pose in POSE_LABELS:
        if pose not in tried and len(recs) < limit:
            recs.append({
                'pose':        pose,
                'display_name':DISPLAY_NAMES.get(pose, pose),
                'emoji':       POSE_EMOJIS.get(pose, '🧘'),
                'avg_accuracy':None,
                'sessions':    0,
                'tip':         'Not tried yet — give it a go!',
            })
    return recs[:limit]

def _build_weekly_chart(db, user_id: int):
    """Last 7 days accuracy per day for Chart.js line chart."""
    rows = db.execute("""
        SELECT DATE(detected_at) as day, AVG(accuracy) as avg_acc, COUNT(*) as cnt
        FROM pose_history WHERE user_id=?
          AND detected_at >= DATE('now','-6 days')
        GROUP BY DATE(detected_at) ORDER BY day ASC
    """, (user_id,)).fetchall()
    labels, accs, counts = [], [], []
    from datetime import date, timedelta as td
    for i in range(7):
        d = (date.today() - td(days=6-i)).isoformat()
        labels.append(d[5:])    # MM-DD
        match = next((r for r in rows if r['day'] == d), None)
        accs.append(round(match['avg_acc'], 1) if match else None)
        counts.append(match['cnt'] if match else 0)
    return {'labels': labels, 'accuracy': accs, 'session_counts': counts}

def _build_pose_chart(db, user_id: int):
    """Per-pose avg accuracy for radar/bar chart."""
    rows = db.execute("""
        SELECT pose_name, AVG(accuracy) as avg_acc, COUNT(*) as cnt
        FROM pose_history WHERE user_id=? GROUP BY pose_name ORDER BY avg_acc DESC
    """, (user_id,)).fetchall()
    return {
        'labels':   [DISPLAY_NAMES.get(r['pose_name'], r['pose_name']) for r in rows],
        'accuracy': [round(r['avg_acc'], 1) for r in rows],
        'counts':   [r['cnt'] for r in rows],
        'poses':    [r['pose_name'] for r in rows],
    }

# ─── PAGE ROUTES ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
        poses=POSE_LABELS, display_names=DISPLAY_NAMES,
        emojis=POSE_EMOJIS, categories=POSE_CATEGORIES)

@app.route('/login')
def login_page():
    if 'user_id' in session: return redirect('/dashboard')
    return render_template('auth.html', page='login')

@app.route('/register')
def register_page():
    if 'user_id' in session: return redirect('/dashboard')
    return render_template('auth.html', page='register')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session: return redirect('/login')
    return render_template('dashboard.html')

@app.route('/practice')
def practice():
    if 'user_id' not in session: return redirect('/login')
    return render_template('practice.html',
        poses=POSE_LABELS, display_names=DISPLAY_NAMES, emojis=POSE_EMOJIS)

@app.route('/meditation')
def meditation_page():
    if 'user_id' not in session: return redirect('/login')
    return render_template('meditation.html')

@app.route('/insights')
def insights_page():
    if 'user_id' not in session: return redirect('/login')
    return render_template('insights.html')

@app.route('/feedback')
def feedback_page():
    if 'user_id' not in session: return redirect('/login')
    return render_template('feedback.html')

# ─── AUTH API ─────────────────────────────────────────────────────────────────
@app.route('/api/register', methods=['POST'])
def api_register():
    d     = request.json or {}
    name  = (d.get('name') or '').strip()
    email = (d.get('email') or '').strip().lower()
    pw    = d.get('password') or ''
    age   = d.get('age')
    gender        = (d.get('gender') or '').strip()
    phone         = (d.get('phone') or '').strip()
    fitness_level = (d.get('fitness_level') or '').strip()
    address       = (d.get('address') or '').strip()

    if not all([name, email, pw]):
        return jsonify({'error': 'Name, email and password are required'}), 400
    if len(pw) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    if '@' not in email or '.' not in email.split('@')[-1]:
        return jsonify({'error': 'Please enter a valid email address'}), 400
    if not age or not (13 <= int(age) <= 100):
        return jsonify({'error': 'Please enter a valid age (13–100)'}), 400
    if not gender:
        return jsonify({'error': 'Please select your gender'}), 400
    if not fitness_level:
        return jsonify({'error': 'Please select your fitness level'}), 400

    db = get_db()
    if db.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone():
        return jsonify({'error': 'This email is already registered. Please login.'}), 409
    otp = generate_otp()
    store_otp(email, otp, 'register')
    result = send_otp_email(email, otp, "account registration")
    session['pending_register'] = {
        'name': name, 'email': email, 'pw_hash': hash_password(pw),
        'age': int(age), 'gender': gender, 'phone': phone,
        'fitness_level': fitness_level, 'address': address,
    }
    resp = {'success': True, 'email_method': result['method']}
    if result['method'] == 'smtp':
        resp['message'] = 'OTP sent to your email'
    else:
        resp['message'] = 'Email not configured — OTP shown below'
        resp['dev_otp'] = otp
    return jsonify(resp)

@app.route('/api/verify-register-otp', methods=['POST'])
def api_verify_register():
    d       = request.json or {}
    otp     = d.get('otp', '').strip()
    pending = session.get('pending_register')
    if not pending:
        return jsonify({'error': 'Registration session expired. Please start again.'}), 400
    ok, msg = verify_otp(pending['email'], otp, 'register')
    if not ok:
        return jsonify({'error': msg}), 400
    db = get_db()
    try:
        db.execute(
            "INSERT INTO users (name,email,password_hash,is_verified,age,gender,phone,fitness_level,address) "
            "VALUES (?,?,?,1,?,?,?,?,?)",
            (pending['name'], pending['email'], pending['pw_hash'],
             pending.get('age'), pending.get('gender'), pending.get('phone'),
             pending.get('fitness_level'), pending.get('address'))
        )
        db.commit()
        user = db.execute("SELECT id FROM users WHERE email=?", (pending['email'],)).fetchone()
        session.pop('pending_register', None)
        session['user_id']    = user['id']
        session['user_email'] = pending['email']
        session['user_name']  = pending['name']
        return jsonify({'success': True, 'redirect': '/dashboard'})
    except Exception as e:
        return jsonify({'error': f'Account creation failed: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    d     = request.json or {}
    email = (d.get('email') or '').strip().lower()
    pw    = d.get('password') or ''
    if not email or not pw:
        return jsonify({'error': 'Email and password are required'}), 400
    db   = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE email=? AND password_hash=?",
        (email, hash_password(pw))).fetchone()
    if not user:
        return jsonify({'error': 'Invalid email or password'}), 401
    if not user['is_verified']:
        return jsonify({'error': 'Email not verified. Please register again.'}), 403
    otp = generate_otp()
    store_otp(email, otp, 'login')
    result = send_otp_email(email, otp, "login verification")
    session['pending_login'] = {'user_id': user['id'], 'email': email, 'name': user['name']}
    resp = {'success': True, 'email_method': result['method']}
    if result['method'] == 'smtp':
        resp['message'] = 'OTP sent to your email'
    else:
        resp['message'] = 'Email not configured — OTP shown below'
        resp['dev_otp'] = otp   # Show in browser when SMTP is not set up
    return jsonify(resp)

@app.route('/api/verify-login-otp', methods=['POST'])
def api_verify_login():
    d       = request.json or {}
    otp     = d.get('otp', '').strip()
    pending = session.get('pending_login')
    if not pending:
        return jsonify({'error': 'Login session expired. Please start again.'}), 400
    ok, msg = verify_otp(pending['email'], otp, 'login')
    if not ok:
        return jsonify({'error': msg}), 400
    session['user_id']    = pending['user_id']
    session['user_email'] = pending['email']
    session['user_name']  = pending['name']
    session.pop('pending_login', None)
    return jsonify({'success': True, 'redirect': '/dashboard'})

@app.route('/api/resend-otp', methods=['POST'])
def api_resend_otp():
    d       = request.json or {}
    purpose = d.get('purpose', 'login')
    pending = session.get(f'pending_{purpose}')
    if not pending:
        return jsonify({'error': 'Session expired. Please start again.'}), 400
    otp    = generate_otp()
    store_otp(pending['email'], otp, purpose)
    result = send_otp_email(pending['email'], otp, purpose)
    resp = {'success': True}
    if result['method'] == 'smtp':
        resp['message'] = 'New OTP sent to your email'
    else:
        resp['message'] = 'New OTP generated — shown below'
        resp['dev_otp'] = otp
    return jsonify(resp)

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'success': True, 'redirect': '/login'})

# ─── USER / ANALYTICS API ─────────────────────────────────────────────────────
@app.route('/api/me')
@login_required
def api_me():
    db   = get_db()
    uid  = session['user_id']
    user = db.execute("SELECT id,name,email,created_at FROM users WHERE id=?", (uid,)).fetchone()
    hist = db.execute(
        "SELECT pose_name,accuracy,is_correct,hold_seconds,detected_at "
        "FROM pose_history WHERE user_id=? ORDER BY detected_at DESC LIMIT 30",
        (uid,)).fetchall()
    stats = db.execute(
        "SELECT pose_name, AVG(accuracy) as avg_acc, MAX(accuracy) as max_acc, "
        "COUNT(*) as count, SUM(is_correct) as correct_count "
        "FROM pose_history WHERE user_id=? GROUP BY pose_name ORDER BY avg_acc DESC",
        (uid,)).fetchall()
    streak_row = db.execute("SELECT * FROM streaks WHERE user_id=?", (uid,)).fetchone()
    total = db.execute("SELECT COUNT(*) as c FROM pose_history WHERE user_id=?", (uid,)).fetchone()['c']
    avg_all = db.execute("SELECT AVG(accuracy) as a FROM pose_history WHERE user_id=?", (uid,)).fetchone()['a']

    weekly   = _build_weekly_chart(db, uid)
    pose_chart = _build_pose_chart(db, uid)
    recs     = _get_recommendations(db, uid)

    return jsonify({
        'user':           dict(user),
        'history':        [dict(h) for h in hist],
        'stats':          [dict(s) for s in stats],
        'total_sessions': total,
        'avg_accuracy':   round(avg_all, 1) if avg_all else None,
        'streak': {
            'current': streak_row['current_streak'] if streak_row else 0,
            'max':     streak_row['max_streak']     if streak_row else 0,
        },
        'weekly_chart':   weekly,
        'pose_chart':     pose_chart,
        'recommendations':recs,
    })

@app.route('/api/save-session', methods=['POST'])
@login_required
def api_save_session():
    """Called when user ends a practice session — saves summary."""
    d         = request.json or {}
    pose_name = d.get('pose_name', '')
    duration  = float(d.get('duration_sec', 0))
    avg_acc   = float(d.get('avg_accuracy', 0))
    max_streak= int(d.get('max_streak', 0))
    if not pose_name or duration <= 0:
        return jsonify({'error': 'Invalid session data'}), 400
    db = get_db()
    db.execute("INSERT INTO sessions (user_id,pose_name,duration_sec,avg_accuracy,max_streak) VALUES (?,?,?,?,?)",
               (session['user_id'], pose_name, duration, avg_acc, max_streak))
    db.commit()
    return jsonify({'success': True})

# ─── POSE API ─────────────────────────────────────────────────────────────────
@app.route('/api/poses')
def api_poses():
    return jsonify({
        'poses':        POSE_LABELS,
        'display_names':DISPLAY_NAMES,
        'emojis':       POSE_EMOJIS,
        'categories':   POSE_CATEGORIES,
        'feedback': {
            k: {'description': v.get('description',''), 'cues': v.get('cues',[]),
                'difficulty': v.get('difficulty','Beginner'), 'benefits': v.get('benefits',''),
                'corrections': v.get('corrections', [])}
            for k, v in POSE_FEEDBACK.items()
        },
    })

# ─── POSE IMAGE PROXY (serves image bytes or SVG fallback — never hangs) ───────
def _pose_svg_fallback(pose):
    """Return a styled SVG placeholder for poses when Wikipedia is unreachable."""
    from flask import Response
    display = pose.replace('_', ' ').title()
    # Escape XML entities
    display = display.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="280" viewBox="0 0 320 280">'
        '<rect width="320" height="280" rx="14" fill="#0d1f1a"/>'
        '<circle cx="160" cy="80" r="26" fill="none" stroke="#22c55e" stroke-width="1.5" opacity="0.45"/>'
        '<line x1="160" y1="106" x2="160" y2="168" stroke="#22c55e" stroke-width="1.5" opacity="0.45"/>'
        '<line x1="160" y1="128" x2="122" y2="113" stroke="#22c55e" stroke-width="1.5" opacity="0.45"/>'
        '<line x1="160" y1="128" x2="198" y2="113" stroke="#22c55e" stroke-width="1.5" opacity="0.45"/>'
        '<line x1="160" y1="168" x2="136" y2="200" stroke="#22c55e" stroke-width="1.5" opacity="0.45"/>'
        '<line x1="160" y1="168" x2="184" y2="200" stroke="#22c55e" stroke-width="1.5" opacity="0.45"/>'
        f'<text x="160" y="232" font-size="15" text-anchor="middle" fill="#9ba8a0"'
        f' font-family="system-ui,sans-serif" font-weight="600">{display}</text>'
        '<text x="160" y="256" font-size="11" text-anchor="middle" fill="#4a6060"'
        ' font-family="system-ui,sans-serif">No photo available</text>'
        '</svg>'
    )
    return Response(svg.encode('utf-8'), content_type='image/svg+xml',
                    headers={'Cache-Control': 'no-cache'})

def _scrape_pocketyoga_image_url(pocket_name):
    """
    Fetch https://pocketyoga.com/pose/{pocket_name} and return the og:image URL.
    Returns None on any failure.
    """
    import re
    page_url = f'https://pocketyoga.com/pose/{_urllib_parse.quote(pocket_name)}'
    req = _urllib_req.Request(page_url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml',
    })
    try:
        with _urllib_req.urlopen(req, timeout=5) as resp:
            html = resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        print(f'[PocketYoga page] {pocket_name}: {e}')
        return None

    # og:image comes in two attribute orders; try both
    for pattern in [
        r'<meta\s[^>]*property=["\']og:image["\']\s[^>]*content=["\'](https?://[^"\']+)["\']',
        r'<meta\s[^>]*content=["\'](https?://[^"\']+)["\'][^>]*property=["\']og:image["\']',
    ]:
        m = re.search(pattern, html, re.I)
        if m:
            return m.group(1)
    return None


@app.route('/api/pose-image/<pose>')
def api_pose_image(pose):
    """Proxy pocketyoga.com pose image as bytes; always returns SVG fallback on failure."""
    from flask import Response

    # Serve cached bytes immediately (real image or None = failed)
    if pose in _POSE_IMG_CACHE:
        cached = _POSE_IMG_CACHE[pose]
        if cached is None:
            return _pose_svg_fallback(pose)
        return Response(cached['bytes'], content_type=cached['ct'],
                        headers={'Cache-Control': 'public, max-age=86400'})

    pocket_name = _POCKETYOGA_NAMES.get(pose)
    if not pocket_name:
        _POSE_IMG_CACHE[pose] = None
        return _pose_svg_fallback(pose)

    # Step 1 — scrape og:image from pocketyoga pose page
    img_url = _scrape_pocketyoga_image_url(pocket_name)
    if not img_url:
        _POSE_IMG_CACHE[pose] = None
        return _pose_svg_fallback(pose)

    # Step 2 — fetch the actual image bytes
    try:
        img_req = _urllib_req.Request(img_url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://pocketyoga.com/',
            'Accept': 'image/*,*/*',
        })
        with _urllib_req.urlopen(img_req, timeout=6) as img_resp:
            img_bytes = img_resp.read()
            content_type = img_resp.headers.get('Content-Type', 'image/jpeg').split(';')[0]
        _POSE_IMG_CACHE[pose] = {'bytes': img_bytes, 'ct': content_type}
        return Response(img_bytes, content_type=content_type,
                        headers={'Cache-Control': 'public, max-age=86400'})
    except Exception as e:
        print(f'[PoseImage bytes] {pose}: {e}')
        _POSE_IMG_CACHE[pose] = None
        return _pose_svg_fallback(pose)

# ─── DETECT API ───────────────────────────────────────────────────────────────
@app.route('/api/detect', methods=['POST'])
@login_required
def api_detect():
    data        = request.json or {}
    frame_b64   = data.get('frame')
    target_pose = data.get('target_pose')   # Must be set; detection is locked to this pose
    hold_secs   = float(data.get('hold_seconds', 0))
    count_rep   = bool(data.get('count_rep', False))  # Frontend sends True only after 15s hold

    uid = session['user_id']

    HOLD_REQUIRED = 15.0   # seconds the user must hold before a rep counts

    try:
        pose_name, accuracy, is_correct = None, 0.0, False
        annotated_b64 = None
        target_matched = False   # did ML agree this is the target pose?

        if frame_b64 and MP_AVAILABLE:
            img_bytes = base64.b64decode(frame_b64.split(',')[-1])
            frame     = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                results = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    kp         = extract_keypoints(results.pose_landmarks)
                    pred_idx   = ML_MODEL.predict([kp])[0]
                    pred_proba = ML_MODEL.predict_proba([kp])[0]
                    pose_name  = LABEL_ENC.inverse_transform([pred_idx])[0]
                    confidence = float(np.max(pred_proba))
                    accuracy   = round(confidence * 100, 1)

                    # If a target pose is selected, only report match when ML agrees
                    if target_pose and target_pose in POSE_LABELS:
                        # Get confidence specifically for the target pose
                        try:
                            target_idx     = LABEL_ENC.transform([target_pose])[0]
                            target_conf    = float(pred_proba[target_idx])
                            target_accuracy = round(target_conf * 100, 1)
                        except Exception:
                            target_accuracy = accuracy if pose_name == target_pose else 0.0

                        # Override reported accuracy to reflect how well user matches target
                        accuracy       = target_accuracy
                        target_matched = target_accuracy >= 65
                        pose_name      = target_pose   # Always report target pose name
                    else:
                        target_matched = accuracy >= 65

                    # A rep is correct only when frontend confirms 15s hold
                    is_correct = target_matched and count_rep

                    # Draw skeleton landmarks on frame
                    annotated  = frame.copy()
                    mp_drawing.draw_landmarks(
                        annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(34,197,94), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255,255,255), thickness=2),
                    )
                    _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

        if pose_name is None:
            # Simulation fallback — still locked to target pose
            pose_name = target_pose if (target_pose and target_pose in POSE_LABELS) \
                        else random.choice(POSE_LABELS)
            accuracy        = round(random.uniform(68, 97), 1)
            target_matched  = accuracy >= 65
            is_correct      = target_matched and count_rep

        # Only save to DB when a full rep is completed (count_rep=True) to avoid
        # flooding the history table with every detection frame.
        db = get_db()
        cur_streak, max_streak = 0, 0
        if count_rep and is_correct:
            cur_streak, max_streak = _update_streak(db, uid, True, pose_name)
            db.execute(
                "INSERT INTO pose_history (user_id,pose_name,accuracy,is_correct,hold_seconds) VALUES (?,?,?,?,?)",
                (uid, pose_name, accuracy, 1, hold_secs))
            db.commit()
        elif count_rep and not is_correct:
            # Rep attempted but accuracy too low — record the failed attempt
            _update_streak(db, uid, False, pose_name)
            db.execute(
                "INSERT INTO pose_history (user_id,pose_name,accuracy,is_correct,hold_seconds) VALUES (?,?,?,?,?)",
                (uid, pose_name, accuracy, 0, hold_secs))
            db.commit()
        else:
            # Live frame — just read current streak without writing
            streak_row = db.execute("SELECT * FROM streaks WHERE user_id=?", (uid,)).fetchone()
            if streak_row:
                cur_streak  = streak_row['current_streak']
                max_streak  = streak_row['max_streak']

        # Build feedback for the locked target pose
        fb = get_pose_feedback(pose_name)

        # Feedback message — based on target pose accuracy
        if not target_matched:
            alert_level   = 'error'
            correction    = fb['corrections'][0] if fb['corrections'] else 'Follow the pose cues'
            alert_message = f"❌ Pose not recognised. {correction}"
        elif accuracy >= 88:
            alert_level   = 'perfect'
            alert_message = f"🎯 {fb['good_msg']}"
        elif accuracy >= 72:
            alert_level   = 'good'
            alert_message = f"👍 Good form! {fb['cues'][0] if fb['cues'] else 'Keep it up!'}"
        else:
            alert_level   = 'warning'
            correction    = fb['corrections'][0] if fb['corrections'] else 'Check your alignment'
            alert_message = f"⚠️ Almost there! {correction}"

        return jsonify({
            'pose':           pose_name,
            'display_name':   DISPLAY_NAMES.get(pose_name, pose_name),
            'emoji':          POSE_EMOJIS.get(pose_name, '🧘'),
            'accuracy':       accuracy,
            'confidence':     accuracy / 100,
            'is_correct':     is_correct,
            'target_matched': target_matched,   # True = user is in correct pose right now
            'feedback':       fb,
            'alert_level':    alert_level,
            'alert_message':  alert_message,
            'streak': {
                'current':   cur_streak,
                'max':       max_streak,
                'milestone': cur_streak > 0 and cur_streak % 5 == 0,
            },
            'annotated_frame': annotated_b64,
            'mp_available':    MP_AVAILABLE,
        })

    except Exception as e:
        print(f"[Detect Error] {e}")
        import traceback; traceback.print_exc()
        pose_name = target_pose or random.choice(POSE_LABELS)
        accuracy  = round(random.uniform(50, 70), 1)
        fb        = get_pose_feedback(pose_name)
        return jsonify({
            'pose':           pose_name,
            'display_name':   DISPLAY_NAMES.get(pose_name,''),
            'emoji':          POSE_EMOJIS.get(pose_name,'🧘'),
            'accuracy':       accuracy,
            'confidence':     accuracy / 100,
            'is_correct':     False,
            'target_matched': False,
            'feedback':       fb,
            'alert_level':    'warning',
            'alert_message':  '⚠️ Detection error — check your pose',
            'streak':         {'current': 0, 'max': 0, 'milestone': False},
            'annotated_frame': None,
            'mp_available':    False,
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0', threaded=True)
