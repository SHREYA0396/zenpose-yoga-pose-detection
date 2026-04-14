# 🧘 ZenPose — AI Yoga Pose Detection System

> Real-time yoga pose detection using MediaPipe + Machine Learning, with voice feedback, OTP authentication, and progress tracking.

---

## 📁 Project Structure

```
zenpose/
├── app.py                  ← Flask backend (all routes + API)
├── train_model.py          ← ML model training script
├── requirements.txt
├── zenpose.db              ← SQLite DB (auto-created on first run)
├── models/
│   ├── zenpose_model.pkl   ← Trained Random Forest pipeline
│   ├── label_encoder.pkl   ← Pose label encoder
│   └── pose_metadata.pkl   ← Pose names, display names, feedback
├── templates/
│   ├── base.html           ← Base layout
│   ├── index.html          ← Landing page
│   ├── auth.html           ← Login + Register + OTP verification
│   ├── dashboard.html      ← User dashboard with stats
│   └── practice.html       ← Live pose detection practice page
└── static/
    ├── css/
    │   ├── main.css        ← Design system + landing + shared
    │   ├── auth.css        ← Auth page styles
    │   ├── dashboard.css   ← Dashboard styles
    │   └── practice.css    ← Practice page styles
    └── js/
        ├── main.js         ← Shared utilities (toast, API, logout)
        ├── auth.js         ← OTP flow, login, register logic
        ├── dashboard.js    ← Dashboard data loading + charts
        └── practice.js     ← Webcam, detection loop, voice feedback
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (already done — skip if models/ exists)
```bash
python train_model.py
```

### 3. Configure email OTP (optional)
Set environment variables for Gmail SMTP:
```bash
export EMAIL_USER="your.email@gmail.com"
export EMAIL_PASS="your_app_password"
```
> **Without these**, OTPs print to the terminal console (dev mode). You can still use the app.

**Gmail App Password steps:**
1. Go to Google Account → Security → 2-Step Verification → App passwords
2. Generate a password for "Mail"
3. Use that 16-char password as `EMAIL_PASS`

### 4. Run the server
```bash
python app.py
```

Open: **http://localhost:5000**

---

## 🔐 Authentication Flow

```
Register → Enter name/email/password → OTP sent to email → Verify OTP → Dashboard
Login    → Enter email/password       → OTP sent to email → Verify OTP → Dashboard
```

- OTP expires in **10 minutes**
- Resend available after **60 seconds**
- Passwords hashed with SHA-256
- Session stored server-side (Flask sessions)

---

## 🧠 10 Yoga Poses Detected

| # | Pose | Sanskrit | Key Focus |
|---|------|----------|-----------|
| 1 | Mountain Pose | Tadasana | Posture, balance |
| 2 | Tree Pose | Vrikshasana | Balance, focus |
| 3 | Warrior I | Virabhadrasana I | Strength, legs |
| 4 | Warrior II | Virabhadrasana II | Hip opening |
| 5 | Goddess Pose | Utkata Konasana | Thighs, core |
| 6 | Downward Dog | Adho Mukha | Spine, hamstrings |
| 7 | Cobra | Bhujangasana | Spine, chest |
| 8 | Plank | Phalakasana | Core, arms |
| 9 | Triangle | Trikonasana | Side stretch |
| 10 | Child's Pose | Balasana | Rest, release |

---

## 🤖 ML Model Details

- **Framework**: Scikit-learn Pipeline (StandardScaler + RandomForestClassifier)
- **Input**: 132 features (33 MediaPipe landmarks × 4: x, y, z, visibility)
- **Training samples**: 4,000 (400 per pose)
- **Test accuracy**: 100% on synthetic data
- **Production note**: Replace `generate_pose_keypoints()` in `train_model.py` with real MediaPipe-extracted CSV data for real-world accuracy

---

## 🎙️ Voice Feedback

Uses the **Web Speech API** (browser-native, no API key needed):
- Announces detected pose
- Reads correction cues aloud (e.g. "Raise your arms higher")
- Cooldown between announcements: 5–7 seconds
- Toggle with **V key** or the voice button
- Prefers calm female English voice if available

---

## ⌨️ Keyboard Shortcuts (Practice Page)

| Key | Action |
|-----|--------|
| `Space` | Toggle camera on/off |
| `V` | Toggle voice feedback |

---

## 📊 Features

- **Real-time detection** at ~600ms intervals
- **Accuracy score ring** (0–100%) with color coding
  - 🟢 85%+ = Correct
  - 🟡 65–85% = Moderate
  - 🔴 <65% = Incorrect
- **Text corrections** listed below score
- **Voice guidance** for hands-free practice
- **Progress dashboard** with per-pose accuracy bars
- **Session history** — last 20 detections stored
- **Pose library** — quick access to all 10 poses

---

## 🔧 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `EMAIL_USER` | Gmail address for OTP sending | No (console fallback) |
| `EMAIL_PASS` | Gmail App Password | No (console fallback) |

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, Flask 3.x |
| Database | SQLite (via Python sqlite3) |
| Pose Detection | MediaPipe BlazePose |
| ML Model | Scikit-learn Random Forest |
| Computer Vision | OpenCV |
| Frontend | HTML5, CSS3, Vanilla JS |
| Voice | Web Speech API |
| Fonts | Cormorant Garamond + DM Sans |

---

## 📝 Notes for Production

1. **Replace synthetic training data** with real MediaPipe CSV exports for higher real-world accuracy
2. **Switch SQLite to PostgreSQL** for multi-user production use
3. **Add HTTPS** — required for webcam access in production browsers
4. **Rate-limit `/api/detect`** to prevent abuse
5. **Store sessions in Redis** instead of Flask cookie sessions for scalability
