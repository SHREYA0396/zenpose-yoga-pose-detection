"""
ZenPose - Yoga Pose Model Training
Trains on 10 yoga poses using synthetic keypoint data + MediaPipe landmark structure
In production: replace synthetic data with real labeled keypoint CSVs
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# ─── 10 YOGA POSES ────────────────────────────────────────────────────────────
POSE_LABELS = [
    "tadasana",        # Mountain Pose
    "vrikshasana",     # Tree Pose
    "warrior_i",       # Warrior I
    "warrior_ii",      # Warrior II
    "goddess",         # Goddess Pose
    "downward_dog",    # Downward Dog
    "cobra",           # Cobra Pose
    "plank",           # Plank
    "triangle",        # Triangle Pose
    "child_pose",      # Child's Pose
]

POSE_DISPLAY_NAMES = {
    "tadasana": "Tadasana (Mountain Pose)",
    "vrikshasana": "Vrikshasana (Tree Pose)",
    "warrior_i": "Warrior I",
    "warrior_ii": "Warrior II",
    "goddess": "Goddess Pose",
    "downward_dog": "Downward Dog",
    "cobra": "Cobra Pose",
    "plank": "Plank",
    "triangle": "Triangle Pose",
    "child_pose": "Child's Pose",
}

POSE_FEEDBACK = {
    "tadasana": {
        "cues": ["Stand tall with feet together", "Arms relaxed at sides", "Weight evenly distributed"],
        "corrections": {
            "arms_too_wide": "Bring your arms closer to your body",
            "knees_bent": "Straighten your knees fully",
            "head_down": "Lift your chin parallel to the floor",
        },
        "good_msg": "Perfect Tadasana! Body is aligned.",
        "description": "Mountain Pose: Stand tall, grounded and stable."
    },
    "vrikshasana": {
        "cues": ["Balance on one leg", "Raise other foot to inner thigh", "Arms above head, palms together"],
        "corrections": {
            "foot_low": "Raise your foot higher on the inner thigh",
            "arms_not_raised": "Lift both arms overhead and join palms",
            "wobbling": "Fix your gaze on a point to maintain balance",
        },
        "good_msg": "Great Vrikshasana! Balanced beautifully.",
        "description": "Tree Pose: Root down, reach up like a tree."
    },
    "warrior_i": {
        "cues": ["Front knee bent at 90°", "Back leg straight", "Arms raised overhead"],
        "corrections": {
            "knee_not_bent": "Bend your front knee to 90 degrees",
            "arms_low": "Raise both arms fully overhead",
            "hips_not_square": "Square your hips to the front",
        },
        "good_msg": "Excellent Warrior I! Strong and powerful.",
        "description": "Warrior I: Strength and determination."
    },
    "warrior_ii": {
        "cues": ["Arms extended parallel to floor", "Front knee over ankle", "Gaze over front hand"],
        "corrections": {
            "arms_dropping": "Keep arms parallel to the floor",
            "knee_forward": "Align knee over ankle, not past it",
            "torso_leaning": "Keep torso upright between arms",
        },
        "good_msg": "Wonderful Warrior II! Arms strong and open.",
        "description": "Warrior II: Open chest, steady gaze."
    },
    "goddess": {
        "cues": ["Wide stance, feet turned out", "Knees bent deep", "Arms at shoulder height"],
        "corrections": {
            "knees_not_bent": "Bend your knees deeper over toes",
            "arms_low": "Raise arms to shoulder height",
            "feet_parallel": "Turn your feet outward at 45 degrees",
        },
        "good_msg": "Superb Goddess Pose! Deep and powerful.",
        "description": "Goddess Pose: Strength and grace combined."
    },
    "downward_dog": {
        "cues": ["Hips high, form inverted V", "Arms and legs straight", "Heels pressing toward floor"],
        "corrections": {
            "hips_low": "Lift your hips higher toward ceiling",
            "arms_bent": "Straighten your arms fully",
            "back_rounded": "Lengthen your spine, avoid rounding",
        },
        "good_msg": "Perfect Downward Dog! Great inverted V shape.",
        "description": "Downward Dog: Energize and lengthen the spine."
    },
    "cobra": {
        "cues": ["Chest lifted, back arched", "Elbows slightly bent", "Legs flat on ground"],
        "corrections": {
            "elbows_straight": "Keep a slight bend in your elbows",
            "shoulders_up": "Roll shoulders back and down away from ears",
            "hips_lifting": "Keep hips and thighs pressed to floor",
        },
        "good_msg": "Beautiful Cobra! Chest open and lifted.",
        "description": "Cobra Pose: Open heart, strengthen spine."
    },
    "plank": {
        "cues": ["Body in straight line", "Core engaged", "Wrists under shoulders"],
        "corrections": {
            "hips_sagging": "Engage your core, lift hips to body level",
            "hips_too_high": "Lower hips to form a straight line",
            "head_dropping": "Keep head in line with spine",
        },
        "good_msg": "Strong Plank! Body perfectly aligned.",
        "description": "Plank: Full body strength and stability."
    },
    "triangle": {
        "cues": ["Wide stance, reach forward then down", "Both arms in vertical line", "Look up at top hand"],
        "corrections": {
            "arms_not_aligned": "Stack both arms in a vertical line",
            "torso_forward": "Keep torso open, not leaning forward",
            "gaze_down": "Turn your head to look at the top hand",
        },
        "good_msg": "Excellent Triangle! Open and extended.",
        "description": "Triangle Pose: Open sides, full extension."
    },
    "child_pose": {
        "cues": ["Sit back on heels", "Arms extended forward", "Forehead touching mat"],
        "corrections": {
            "hips_raised": "Sit hips back toward heels",
            "arms_bent": "Extend arms fully forward on mat",
            "head_up": "Rest forehead gently on the mat",
        },
        "good_msg": "Perfect Child's Pose! Fully relaxed and grounded.",
        "description": "Child's Pose: Rest, restore, and breathe."
    },
}

# ─── SYNTHETIC DATA GENERATION ────────────────────────────────────────────────
# MediaPipe Pose has 33 landmarks × 4 values (x, y, z, visibility) = 132 features
# We generate pose-specific synthetic data with realistic angle distributions

def generate_pose_keypoints(pose_name, n_samples=300):
    """
    Generate synthetic but anatomically plausible keypoint data for each pose.
    Each pose has characteristic angle ranges for key joints.
    In production: replace with real CSV data from MediaPipe extractions.
    """
    np.random.seed(POSE_LABELS.index(pose_name) * 42)
    samples = []

    for _ in range(n_samples):
        # Base 132 features (33 landmarks × 4)
        kp = np.random.randn(132) * 0.05

        # Pose-specific joint angle signatures (normalized 0-1 range)
        if pose_name == "tadasana":
            # Standing upright: legs straight, arms down
            kp[0:4] = [0.5, 0.1, 0, 1]      # nose top-center
            kp[24:28] = [0.5, 0.9, 0, 1]    # hips centered low
            kp[48:52] = [0.48, 0.97, 0, 1]  # left ankle
            kp[52:56] = [0.52, 0.97, 0, 1]  # right ankle
            kp[28:32] = [0.3, 0.5, 0, 1]    # left arm down
            kp[32:36] = [0.7, 0.5, 0, 1]    # right arm down
        elif pose_name == "vrikshasana":
            # One leg raised
            kp[0:4] = [0.5, 0.1, 0, 1]
            kp[48:52] = [0.5, 0.95, 0, 1]   # standing foot
            kp[52:56] = [0.55, 0.6, 0, 1]   # raised foot mid thigh
            kp[28:32] = [0.4, 0.05, 0, 1]   # left arm up
            kp[32:36] = [0.6, 0.05, 0, 1]   # right arm up
        elif pose_name == "warrior_i":
            kp[0:4] = [0.5, 0.15, 0, 1]
            kp[24:28] = [0.48, 0.55, 0, 1]  # hip
            kp[36:40] = [0.35, 0.75, 0, 1]  # front knee bent
            kp[40:44] = [0.65, 0.85, 0, 1]  # back knee straight
            kp[28:32] = [0.35, 0.05, 0, 1]
            kp[32:36] = [0.65, 0.05, 0, 1]  # arms up
        elif pose_name == "warrior_ii":
            kp[0:4] = [0.5, 0.3, 0, 1]
            kp[28:32] = [0.05, 0.35, 0, 1]  # left arm extended
            kp[32:36] = [0.95, 0.35, 0, 1]  # right arm extended
            kp[36:40] = [0.35, 0.65, 0, 1]  # front knee
            kp[40:44] = [0.75, 0.8, 0, 1]   # back knee
        elif pose_name == "goddess":
            kp[0:4] = [0.5, 0.2, 0, 1]
            kp[24:28] = [0.5, 0.6, 0, 1]    # hips low and wide
            kp[48:52] = [0.15, 0.9, 0, 1]
            kp[52:56] = [0.85, 0.9, 0, 1]   # feet wide
            kp[28:32] = [0.2, 0.4, 0, 1]
            kp[32:36] = [0.8, 0.4, 0, 1]    # arms out at sides
        elif pose_name == "downward_dog":
            kp[0:4] = [0.5, 0.5, -0.2, 1]   # head down
            kp[24:28] = [0.5, 0.1, 0, 1]    # hips high
            kp[28:32] = [0.2, 0.4, 0, 1]
            kp[32:36] = [0.8, 0.4, 0, 1]    # hands on ground
            kp[48:52] = [0.3, 0.85, 0, 1]
            kp[52:56] = [0.7, 0.85, 0, 1]   # feet on ground
        elif pose_name == "cobra":
            kp[0:4] = [0.5, 0.2, 0.3, 1]    # head up and back
            kp[24:28] = [0.5, 0.75, -0.1, 1] # hips low
            kp[28:32] = [0.3, 0.55, 0, 1]
            kp[32:36] = [0.7, 0.55, 0, 1]   # arms pushing up
        elif pose_name == "plank":
            kp[0:4] = [0.5, 0.35, 0, 1]
            kp[24:28] = [0.5, 0.45, 0, 1]   # hips level
            kp[28:32] = [0.25, 0.5, 0, 1]
            kp[32:36] = [0.75, 0.5, 0, 1]   # arms straight down
            kp[48:52] = [0.35, 0.5, 0, 1]
            kp[52:56] = [0.65, 0.5, 0, 1]   # feet back
        elif pose_name == "triangle":
            kp[0:4] = [0.5, 0.4, 0, 1]
            kp[24:28] = [0.5, 0.55, 0, 1]
            kp[28:32] = [0.05, 0.35, 0, 1]  # top arm up
            kp[32:36] = [0.4, 0.85, 0, 1]   # bottom arm to ankle
            kp[48:52] = [0.2, 0.95, 0, 1]
            kp[52:56] = [0.8, 0.95, 0, 1]   # wide feet
        elif pose_name == "child_pose":
            kp[0:4] = [0.5, 0.8, -0.3, 1]   # head low and forward
            kp[24:28] = [0.5, 0.6, -0.1, 1] # hips back on heels
            kp[28:32] = [0.3, 0.9, 0, 1]
            kp[32:36] = [0.7, 0.9, 0, 1]    # arms extended forward

        # Add realistic noise
        noise = np.random.randn(132) * 0.03
        samples.append(kp + noise)

    return np.array(samples)


def train_and_save():
    print("=" * 60)
    print("ZenPose Model Training")
    print("=" * 60)

    X_list, y_list = [], []

    for pose in POSE_LABELS:
        print(f"  Generating data for: {pose}...")
        X = generate_pose_keypoints(pose, n_samples=400)
        y = [pose] * len(X)
        X_list.append(X)
        y_list.extend(y)

    X = np.vstack(X_list)
    y = np.array(y_list)

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Poses: {len(POSE_LABELS)}")

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Pipeline: scaler + Random Forest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("\nTraining Random Forest pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save
    os.makedirs('models', exist_ok=True)
    with open('models/zenpose_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    with open('models/pose_metadata.pkl', 'wb') as f:
        pickle.dump({
            'labels': POSE_LABELS,
            'display_names': POSE_DISPLAY_NAMES,
            'feedback': POSE_FEEDBACK,
        }, f)

    print("\n✓ Models saved to models/")
    print("=" * 60)
    return acc


if __name__ == "__main__":
    train_and_save()
