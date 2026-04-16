# 🧘 ZenPose v2.0 — AI Yoga Pose Detection

> 25 poses · Real-time detection · Meditation music · Ayurvedic insights · Streak tracking · Email OTP · Voice feedback

## Quick Start

```bash
pip install -r requirements.txt
python train_model.py     # retrain model (scikit-learn 1.7.2 required)
python app.py             # open http://localhost:5000
```

> **Important:** Always open the app at `http://localhost:5000` — not `127.0.0.1:5000`.  
> The camera API requires `localhost` (or HTTPS). The app auto-redirects if needed.

---

## Email OTP Setup (Gmail)

1. Google Account → Security → 2-Step Verification → App Passwords → Generate
2. Set environment variables before running:

```bash
# Windows CMD
set EMAIL_USER=you@gmail.com
set EMAIL_PASS=abcd efgh ijkl mnop
python app.py

# Windows PowerShell
$env:EMAIL_USER="you@gmail.com"
$env:EMAIL_PASS="abcd efgh ijkl mnop"
python app.py

# Linux / Mac
export EMAIL_USER="you@gmail.com"
export EMAIL_PASS="abcd efgh ijkl mnop"
python app.py
```

Without these vars, OTPs print to the terminal (dev mode — app works fully).

---

## Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page |
| `/register` | Register with email OTP verification |
| `/login` | Login with email OTP verification |
| `/dashboard` | Analytics — weekly chart, pose accuracy, recommendations |
| `/practice` | Live pose detection with reference photos & voice feedback |
| `/meditation` | Ambient sound therapy + guided breathing exercises |
| `/insights` | Yoga facts, Ayurvedic herbs, all 25 pose impacts |
| `/feedback` | Coming soon |

---

## Features

### Practice
- **Live AI detection** — MediaPipe keypoints + Random Forest classifier
- **Pose reference photos** — Wikipedia photo loads automatically on pose selection
- **15-second hold system** — rep counts only after a sustained hold
- **Voice guidance** — Web Speech API reads corrections and cues aloud
- **Streak tracking** — consecutive correct holds, milestone alerts at every 5
- **Session timer** — tracks hold time, saved on End Session

### Meditation (`/meditation`)
- **6 ambient sounds** generated in-browser via Web Audio API (no downloads):
  - Alpha Waves (10 Hz) · Theta Waves (6 Hz) · Delta Waves (2 Hz)
  - Singing Bowl 432 Hz · Gentle Rain · Om Drone (136.1 Hz)
- **Guided breathing** with animated circle — 3 techniques:
  - 4-7-8 Breathing · Box Breathing · Calming Breath
- Tips on posture, timing, and building a lasting meditation practice

### Do You Know (`/insights`) — 3 tabs
- **Yoga Insights** — 8 science-backed facts (cortisol, GABA, brain rewiring, sleep)
- **Healing Herbs** — 8 Ayurvedic herbs with benefits + daily usage tips:
  - Ashwagandha · Tulsi · Brahmi · Turmeric · Shatavari · Triphala · Moringa · Giloy
- **Pose Impact Library** — All 25 poses, each expandable with:
  - Why do it · Physical impact · Mental impact · Direct practice link

### Dashboard
- Weekly accuracy line chart (Chart.js)
- Per-pose accuracy bar chart
- Personalised pose recommendations (based on lowest accuracy)
- Recent session history (last 30 detections)
- Full pose library with category filters

---

## 25 Poses

| Category | Poses |
|----------|-------|
| Standing | Tadasana · Warrior I · Warrior II · Goddess · Triangle · Chair · Low Lunge |
| Balance | Vrikshasana · Warrior III · Half Moon · Eagle |
| Core | Plank · Boat · Side Plank |
| Backbend | Cobra · Bridge · Camel · Fish |
| Inversion | Downward Dog |
| Hip Opener | Pigeon |
| Seated | Lotus · Seated Forward Bend |
| Twist | Supine Twist |
| Restorative | Child's Pose |
| Arm Balance | Crow |

---

## Keyboard Shortcuts (Practice page)

| Key | Action |
|-----|--------|
| `Space` | Toggle camera |
| `V` | Toggle voice |
| `E` | End session |

---

## Tech Stack

- **Backend:** Flask · SQLite · Python 3.10
- **ML:** MediaPipe · Scikit-learn 1.7.2 · OpenCV · NumPy
- **Frontend:** Vanilla JS · Chart.js · Web Audio API · Web Speech API
- **Auth:** Gmail SMTP · OTP verification
- **Model:** Random Forest pipeline · 99.95% test accuracy · 25 pose classes
