# 🧘 ZenPose  — AI Yoga Pose Detection

> 25 poses · Streak tracking · Session timer · Chart.js analytics · Email OTP · Voice feedback

## Quick Start

```bash
pip install -r requirements.txt
python train_model.py   # optional — models already included
python app.py           # http://localhost:5000
```

## Email OTP Setup (Gmail)

1. Google Account → Security → 2-Step Verification → App passwords → Generate
2. Set env vars before running:

```bash
# Linux/Mac
export EMAIL_USER="you@gmail.com"
export EMAIL_PASS="abcd efgh ijkl mnop"

# Windows CMD
set EMAIL_USER=you@gmail.com
set EMAIL_PASS=abcd efgh ijkl mnop
```

Without these vars, OTPs print to terminal (dev mode — app works fully).

## 25 Poses

Tadasana · Vrikshasana · Warrior I/II/III · Goddess · Downward Dog · Cobra · Plank ·
Triangle · Child's Pose · Chair · Bridge · Pigeon · Camel · Half Moon · Boat · Crow ·
Eagle · Lotus · Fish · Seated Forward Bend · Supine Twist · Low Lunge · Side Plank

## Features

- **Streak tracking** — consecutive correct detections, resets on error, milestone alerts
- **Session timer** — real-time hold time, saved on End Session
- **Conditional alerts** — Perfect / Good / Warning / Error per accuracy tier
- **Chart.js analytics** — weekly accuracy line chart + per-pose bar chart
- **Pose recommendations** — surfaces your weakest poses
- **Voice guidance** — Web Speech API, correction cues, toggle with V key
- **25 ML pose classes** — Random Forest, 99.95% test accuracy

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Toggle camera |
| V | Toggle voice |
| E | End session |

## Tech Stack

Flask · SQLite · MediaPipe · Scikit-learn · OpenCV · Chart.js · Web Speech API · Gmail SMTP
