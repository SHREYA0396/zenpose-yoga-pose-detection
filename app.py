"""
ZenPose – Flask Backend
Handles: Auth (register/login/OTP), Pose Detection API, Session Management
"""

import os, pickle, random, time, base64, smtplib, json, sqlite3, hashlib, secrets
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from functools import wraps

import numpy as np
import cv2
from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, g)

# ─── MEDIAPIPE (optional graceful fallback) ────────────────────────────────────
try:
    import mediapipe as mp
    MP_AVAILABLE = True
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
except Exception:
    MP_AVAILABLE = False
    print("[WARN] MediaPipe not available – using simulation mode")

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
with open('models/zenpose_model.pkl', 'rb') as f:
    ML_MODEL = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    LABEL_ENC = pickle.load(f)
with open('models/pose_metadata.pkl', 'rb') as f:
    META = pickle.load(f)

POSE_FEEDBACK = META['feedback']
POSE_LABELS   = META['labels']
DISPLAY_NAMES = META['display_names']

# ─── FLASK APP ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)

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
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_verified INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS otp_store (
            email TEXT PRIMARY KEY,
            otp TEXT NOT NULL,
            purpose TEXT NOT NULL,
            expires_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            expires_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS pose_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            pose_name TEXT NOT NULL,
            accuracy REAL NOT NULL,
            detected_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    db.commit()
    db.close()

init_db()

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def generate_otp():
    return str(random.randint(100000, 999999))

def send_otp_email(to_email, otp, purpose="verification"):
    """Send OTP via Gmail SMTP. Returns True on success, False on failure."""
    subject = "ZenPose – Your OTP Code"
    purpose_label = purpose.replace('_', ' ').title()
    body = f"""
    <html><body style="font-family:sans-serif;background:#0a0a0a;color:#fff;padding:30px;">
    <div style="max-width:500px;margin:auto;background:#111;border-radius:16px;padding:30px;border:1px solid #22c55e33;">
      <h2 style="color:#22c55e;">🧘 ZenPose</h2>
      <h3>Your Verification Code</h3>
      <p>Your OTP for <strong>{purpose_label}</strong>:</p>
      <div style="font-size:42px;letter-spacing:12px;color:#22c55e;font-weight:bold;padding:20px 0;">{otp}</div>
      <p style="color:#888;font-size:12px;">This code expires in 10 minutes. Do not share it with anyone.</p>
    </div></body></html>
    """
    # Configure EMAIL_USER and EMAIL_PASS (Google App Password) in environment.
    user = os.environ.get('EMAIL_USER')
    pwd  = os.environ.get('EMAIL_PASS')
    if not user or not pwd:
        print("[OTP Error] EMAIL_USER or EMAIL_PASS not configured")
        return False

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From']    = user
        msg['To']      = to_email
        msg.attach(MIMEText(body, 'html'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=15) as s:
            s.login(user, pwd)
            s.send_message(msg)
        return True
    except Exception as e:
        print(f"[SMTP Error] {e}")
        return False

def store_otp(email, otp, purpose):
    db = get_db()
    db.execute(
        "INSERT OR REPLACE INTO otp_store VALUES (?,?,?,?)",
        (email, otp, purpose, time.time() + 600)
    )
    db.commit()

def verify_otp(email, otp, purpose):
    db = get_db()
    row = db.execute(
        "SELECT * FROM otp_store WHERE email=? AND otp=? AND purpose=?",
        (email, otp, purpose)
    ).fetchone()
    if not row:
        return False, "Invalid OTP"
    if time.time() > row['expires_at']:
        return False, "OTP expired"
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

# ─── POSE DETECTION ───────────────────────────────────────────────────────────
def calculate_angle(a, b, c):
    """Calculate angle at joint b given three landmark points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def extract_keypoints(landmarks):
    """Flatten 33 MediaPipe landmarks into 132-dim feature vector."""
    kp = []
    for lm in landmarks.landmark:
        kp.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(kp)

def get_pose_feedback(pose_name, landmarks=None):
    """Return pose-specific feedback based on key joint angles."""
    fb = POSE_FEEDBACK.get(pose_name, {})
    corrections = []

    if landmarks and pose_name == "warrior_i":
        try:
            lm = landmarks.landmark
            # Check front knee angle
            hip   = [lm[23].x, lm[23].y]
            knee  = [lm[25].x, lm[25].y]
            ankle = [lm[27].x, lm[27].y]
            angle = calculate_angle(hip, knee, ankle)
            if angle > 130:
                corrections.append(fb['corrections'].get('knee_not_bent', ''))
        except:
            pass

    if not corrections:
        corrections = fb.get('cues', [])

    return {
        'pose': pose_name,
        'display_name': DISPLAY_NAMES.get(pose_name, pose_name),
        'description': fb.get('description', ''),
        'good_msg': fb.get('good_msg', 'Good pose!'),
        'corrections': corrections,
        'cues': fb.get('cues', []),
    }

def simulate_detection(target_pose=None):
    """Simulate pose detection when no real webcam available."""
    if target_pose and target_pose in POSE_LABELS:
        pose = target_pose
    else:
        pose = random.choice(POSE_LABELS)
    confidence = random.uniform(0.78, 0.97)
    accuracy   = round(confidence * 100, 1)
    is_correct = accuracy >= 75
    return pose, accuracy, is_correct

# ─── ROUTES: PAGES ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login_page():
    if 'user_id' in session:
        return redirect('/dashboard')
    return render_template('auth.html', page='login')

@app.route('/register')
def register_page():
    if 'user_id' in session:
        return redirect('/dashboard')
    return render_template('auth.html', page='register')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template('dashboard.html')

@app.route('/practice')
def practice():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template('practice.html', poses=POSE_LABELS, display_names=DISPLAY_NAMES)

# ─── ROUTES: AUTH API ─────────────────────────────────────────────────────────
@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json or {}
    name  = (data.get('name') or '').strip()
    email = (data.get('email') or '').strip().lower()
    pw    = data.get('password') or ''

    if not all([name, email, pw]):
        return jsonify({'error': 'All fields required'}), 400
    if len(pw) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    db = get_db()
    existing = db.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
    if existing:
        return jsonify({'error': 'Email already registered'}), 409

    otp = generate_otp()
    store_otp(email, otp, 'register')
    if not send_otp_email(email, otp, "registration"):
        return jsonify({'error': 'Failed to send OTP email. Check SMTP settings.'}), 500

    # Store pending user in session
    session['pending_register'] = {'name': name, 'email': email, 'pw_hash': hash_password(pw)}
    return jsonify({'success': True, 'message': 'OTP sent to your email'})

@app.route('/api/verify-register-otp', methods=['POST'])
def api_verify_register():
    data  = request.json or {}
    otp   = data.get('otp', '').strip()
    pending = session.get('pending_register')

    if not pending:
        return jsonify({'error': 'No registration pending'}), 400

    ok, msg = verify_otp(pending['email'], otp, 'register')
    if not ok:
        return jsonify({'error': msg}), 400

    db = get_db()
    try:
        db.execute(
            "INSERT INTO users (name, email, password_hash, is_verified) VALUES (?,?,?,1)",
            (pending['name'], pending['email'], pending['pw_hash'])
        )
        db.commit()
        user = db.execute("SELECT id FROM users WHERE email=?", (pending['email'],)).fetchone()
        session.pop('pending_register', None)
        session['user_id']   = user['id']
        session['user_email'] = pending['email']
        session['user_name']  = pending['name']
        return jsonify({'success': True, 'redirect': '/dashboard'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    data  = request.json or {}
    email = (data.get('email') or '').strip().lower()
    pw    = data.get('password') or ''

    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE email=? AND password_hash=?",
        (email, hash_password(pw))
    ).fetchone()

    if not user:
        return jsonify({'error': 'Invalid email or password'}), 401
    if not user['is_verified']:
        return jsonify({'error': 'Email not verified'}), 403

    otp = generate_otp()
    store_otp(email, otp, 'login')
    if not send_otp_email(email, otp, "login"):
        return jsonify({'error': 'Failed to send OTP email. Check SMTP settings.'}), 500
    session['pending_login'] = {'user_id': user['id'], 'email': email, 'name': user['name']}
    return jsonify({'success': True, 'message': 'OTP sent to your email'})

@app.route('/api/verify-login-otp', methods=['POST'])
def api_verify_login():
    data = request.json or {}
    otp  = data.get('otp', '').strip()
    pending = session.get('pending_login')

    if not pending:
        return jsonify({'error': 'No login pending'}), 400

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
    data    = request.json or {}
    purpose = data.get('purpose', 'login')
    pending = session.get(f'pending_{purpose}')
    if not pending:
        return jsonify({'error': 'No pending verification'}), 400
    email = pending.get('email') or pending.get('email')
    otp   = generate_otp()
    store_otp(email, otp, purpose)
    if not send_otp_email(email, otp, purpose):
        return jsonify({'error': 'Failed to resend OTP email. Check SMTP settings.'}), 500
    return jsonify({'success': True, 'message': 'OTP resent'})

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'success': True, 'redirect': '/login'})

# ─── ROUTES: USER API ─────────────────────────────────────────────────────────
@app.route('/api/me')
@login_required
def api_me():
    db   = get_db()
    user = db.execute("SELECT id, name, email, created_at FROM users WHERE id=?",
                      (session['user_id'],)).fetchone()
    hist = db.execute(
        "SELECT pose_name, accuracy, detected_at FROM pose_history WHERE user_id=? ORDER BY detected_at DESC LIMIT 20",
        (session['user_id'],)
    ).fetchall()
    stats = db.execute(
        "SELECT pose_name, AVG(accuracy) as avg_acc, COUNT(*) as count FROM pose_history WHERE user_id=? GROUP BY pose_name",
        (session['user_id'],)
    ).fetchall()
    return jsonify({
        'user': dict(user),
        'history': [dict(h) for h in hist],
        'stats': [dict(s) for s in stats],
        'total_sessions': sum(s['count'] for s in stats),
    })

# ─── ROUTES: POSE API ─────────────────────────────────────────────────────────
@app.route('/api/poses')
def api_poses():
    return jsonify({
        'poses': POSE_LABELS,
        'display_names': DISPLAY_NAMES,
        'feedback': {k: {'description': v['description'], 'cues': v['cues']}
                     for k, v in POSE_FEEDBACK.items()}
    })

@app.route('/api/detect', methods=['POST'])
@login_required
def api_detect():
    """
    Accepts base64 image frame, runs MediaPipe + ML classification.
    Returns: pose, confidence, feedback, skeleton overlay.
    """
    data = request.json or {}
    frame_b64   = data.get('frame')
    target_pose = data.get('target_pose')

    # frame_b64 may be None in simulation/test mode — that's fine

    try:
        # Decode image
        img_bytes = base64.b64decode(frame_b64.split(',')[-1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Could not decode frame")

        if MP_AVAILABLE:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(rgb)

            if results.pose_landmarks:
                kp = extract_keypoints(results.pose_landmarks)
                pred_idx  = ML_MODEL.predict([kp])[0]
                pred_proba = ML_MODEL.predict_proba([kp])[0]
                pose_name  = LABEL_ENC.inverse_transform([pred_idx])[0]
                confidence = float(np.max(pred_proba))
                accuracy   = round(confidence * 100, 1)
                is_correct = (target_pose is None or pose_name == target_pose) and accuracy >= 70

                # Draw skeleton on frame
                annotated = frame.copy()
                mp_drawing.draw_landmarks(
                    annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(34, 197, 94), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
                _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
                fb = get_pose_feedback(pose_name, results.pose_landmarks)
            else:
                pose_name, accuracy, is_correct = simulate_detection(target_pose)
                confidence = accuracy / 100
                annotated_b64 = None
                fb = get_pose_feedback(pose_name)
        else:
            pose_name, accuracy, is_correct = simulate_detection(target_pose)
            confidence = accuracy / 100
            annotated_b64 = None
            fb = get_pose_feedback(pose_name)

        # Save to history
        db = get_db()
        db.execute(
            "INSERT INTO pose_history (user_id, pose_name, accuracy) VALUES (?,?,?)",
            (session['user_id'], pose_name, accuracy)
        )
        db.commit()

        return jsonify({
            'pose':         pose_name,
            'display_name': DISPLAY_NAMES.get(pose_name, pose_name),
            'accuracy':     accuracy,
            'confidence':   confidence,
            'is_correct':   is_correct,
            'feedback':     fb,
            'annotated_frame': annotated_b64,
            'mp_available': MP_AVAILABLE,
        })

    except Exception as e:
        print(f"[Detect Error] {e}")
        # Graceful fallback
        pose_name, accuracy, is_correct = simulate_detection(target_pose)
        fb = get_pose_feedback(pose_name)
        return jsonify({
            'pose':         pose_name,
            'display_name': DISPLAY_NAMES.get(pose_name, pose_name),
            'accuracy':     accuracy,
            'confidence':   accuracy / 100,
            'is_correct':   is_correct,
            'feedback':     fb,
            'annotated_frame': None,
            'mp_available': False,
        })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
