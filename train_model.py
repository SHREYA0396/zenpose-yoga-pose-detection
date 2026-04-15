"""
ZenPose — 25 Pose Training Script
Generates anatomically-distinct synthetic MediaPipe keypoints per pose.
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

POSE_LABELS = [
    "tadasana","vrikshasana","warrior_i","warrior_ii","warrior_iii",
    "goddess","downward_dog","cobra","plank","triangle","child_pose",
    "chair_pose","bridge","pigeon","camel","half_moon","boat","crow",
    "eagle","lotus","fish","seated_forward","supine_twist","low_lunge","side_plank",
]

POSE_DISPLAY_NAMES = {
    "tadasana":"Tadasana (Mountain Pose)","vrikshasana":"Vrikshasana (Tree Pose)",
    "warrior_i":"Warrior I","warrior_ii":"Warrior II","warrior_iii":"Warrior III",
    "goddess":"Goddess Pose","downward_dog":"Downward Dog","cobra":"Cobra Pose",
    "plank":"Plank","triangle":"Triangle Pose","child_pose":"Child's Pose",
    "chair_pose":"Chair Pose","bridge":"Bridge Pose","pigeon":"Pigeon Pose",
    "camel":"Camel Pose","half_moon":"Half Moon","boat":"Boat Pose",
    "crow":"Crow Pose","eagle":"Eagle Pose","lotus":"Lotus Pose",
    "fish":"Fish Pose","seated_forward":"Seated Forward Bend",
    "supine_twist":"Supine Twist","low_lunge":"Low Lunge","side_plank":"Side Plank",
}

POSE_EMOJIS = {
    "tadasana":"🧍","vrikshasana":"🌲","warrior_i":"⚔️","warrior_ii":"🗡️",
    "warrior_iii":"🦅","goddess":"🌟","downward_dog":"🐕","cobra":"🐍",
    "plank":"📐","triangle":"🔺","child_pose":"🙏","chair_pose":"🪑",
    "bridge":"🌉","pigeon":"🕊️","camel":"🐫","half_moon":"🌙",
    "boat":"⛵","crow":"🐦","eagle":"🦅","lotus":"🪷",
    "fish":"🐟","seated_forward":"🧘","supine_twist":"🌀","low_lunge":"🏹",
    "side_plank":"💎",
}

POSE_CATEGORIES = {
    "tadasana":"Standing","vrikshasana":"Balance","warrior_i":"Standing",
    "warrior_ii":"Standing","warrior_iii":"Balance","goddess":"Standing",
    "downward_dog":"Inversion","cobra":"Backbend","plank":"Core",
    "triangle":"Standing","child_pose":"Restorative","chair_pose":"Standing",
    "bridge":"Backbend","pigeon":"Hip Opener","camel":"Backbend",
    "half_moon":"Balance","boat":"Core","crow":"Arm Balance",
    "eagle":"Balance","lotus":"Seated","fish":"Backbend",
    "seated_forward":"Forward Bend","supine_twist":"Twist",
    "low_lunge":"Standing","side_plank":"Core",
}

POSE_FEEDBACK = {
    "tadasana":{"cues":["Stand with feet together or hip-width apart","Arms relaxed, palms forward","Crown of head lifting toward ceiling"],"corrections":["Bring your arms closer to your body","Straighten your knees fully","Lift chin parallel to the floor"],"good_msg":"Perfect Tadasana! Body fully aligned.","description":"Mountain Pose: Foundation of all standing poses.","difficulty":"Beginner","benefits":"Improves posture, strengthens thighs and ankles."},
    "vrikshasana":{"cues":["Root standing foot into the earth","Raise foot to inner thigh or calf","Press palms together overhead"],"corrections":["Raise your foot higher on the inner thigh","Lift both arms overhead and join palms","Fix gaze on a still point for balance"],"good_msg":"Beautiful Vrikshasana! Balanced and tall.","description":"Tree Pose: Cultivate balance, focus, and stillness.","difficulty":"Beginner","benefits":"Strengthens legs and core, improves balance."},
    "warrior_i":{"cues":["Front knee bent at 90° over ankle","Back foot at 45°, heel grounded","Arms raised overhead, palms facing"],"corrections":["Bend front knee to 90 degrees","Raise both arms fully overhead","Square hips toward front foot"],"good_msg":"Excellent Warrior I! Powerful and grounded.","description":"Warrior I: Embody strength and fierce grace.","difficulty":"Beginner","benefits":"Strengthens legs, opens hips and chest."},
    "warrior_ii":{"cues":["Arms extended parallel to floor","Front knee over second toe","Gaze over front middle finger"],"corrections":["Keep arms parallel to floor — don't let them droop","Align knee over ankle, not collapsing inward","Keep torso upright between arms"],"good_msg":"Wonderful Warrior II! Arms strong, stance wide.","description":"Warrior II: Open chest, steady gaze, hold your ground.","difficulty":"Beginner","benefits":"Builds leg endurance, opens hips, strengthens arms."},
    "warrior_iii":{"cues":["Standing leg straight and engaged","Hips level and squared to floor","Arms extended forward, body parallel to floor"],"corrections":["Keep standing leg straight and engaged","Level your hips — don't let one side drop","Extend arms forward or place hands on hips"],"good_msg":"Incredible Warrior III! Full body alignment!","description":"Warrior III: Balance, focus, and full body strength.","difficulty":"Intermediate","benefits":"Tones entire body, improves balance and focus."},
    "goddess":{"cues":["Wide stance, toes turned out 45°","Bend knees deeply over toes","Arms at shoulder height, elbows 90°"],"corrections":["Bend knees deeper over toes","Raise arms to shoulder height","Turn feet outward at 45 degrees"],"good_msg":"Superb Goddess! Deep, strong and radiant.","description":"Goddess Pose: Connect with inner strength and vitality.","difficulty":"Beginner","benefits":"Strengthens inner thighs, glutes, and core."},
    "downward_dog":{"cues":["Press palms flat, fingers spread wide","Lift hips forming inverted V","Press heels toward the floor"],"corrections":["Lift hips higher toward ceiling","Straighten arms fully","Lengthen spine — avoid rounding back"],"good_msg":"Perfect Downward Dog! Spine long and strong.","description":"Downward Dog: Energize and decompress the spine.","difficulty":"Beginner","benefits":"Stretches hamstrings, spine. Strengthens arms, shoulders."},
    "cobra":{"cues":["Legs flat, tops of feet pressing down","Hands under shoulders, elbows hugging body","Inhale, lift chest gently"],"corrections":["Keep slight bend in elbows","Roll shoulders back from ears","Keep hips firmly into the mat"],"good_msg":"Beautiful Cobra! Heart open and lifted.","description":"Cobra Pose: Awaken spine, open heart, build back strength.","difficulty":"Beginner","benefits":"Strengthens spine, opens chest and lungs."},
    "plank":{"cues":["Wrists directly under shoulders","Body one straight line head to heels","Engage core, glutes, and quads"],"corrections":["Engage core — lift hips to body level","Lower hips — don't let them pike up","Keep head in line with spine"],"good_msg":"Rock-solid Plank! Perfect straight line.","description":"Plank: Build full-body strength and stability.","difficulty":"Beginner","benefits":"Strengthens core, arms, wrists, and spine."},
    "triangle":{"cues":["Wide stance, front toes forward","Extend arm down to shin or floor","Stack both arms in vertical line"],"corrections":["Stack arms in vertical line","Keep torso open toward ceiling","Look up at top hand"],"good_msg":"Excellent Triangle! Open and extended.","description":"Triangle Pose: Open side body, lengthen, find space.","difficulty":"Beginner","benefits":"Stretches hamstrings, groins, spine."},
    "child_pose":{"cues":["Sit hips back toward heels","Extend arms long on mat","Rest forehead gently on mat"],"corrections":["Sit hips back toward heels","Extend arms fully forward","Rest forehead on the mat"],"good_msg":"Perfect Child's Pose! Fully restored.","description":"Child's Pose: Rest, release, and restore.","difficulty":"Beginner","benefits":"Releases back, hips, thighs. Calms nervous system."},
    "chair_pose":{"cues":["Feet together or hip-width","Bend knees as if sitting in chair","Arms raised overhead alongside ears"],"corrections":["Bend knees more deeply toward 90 degrees","Raise arms overhead alongside ears","Shift weight into heels"],"good_msg":"Strong Chair Pose! Thighs burning well.","description":"Chair Pose: Build heat, strength, and determination.","difficulty":"Beginner","benefits":"Strengthens thighs, glutes, and ankles."},
    "bridge":{"cues":["Feet hip-width, knees bent","Press feet into floor, lift hips","Clasp hands under body"],"corrections":["Press feet firmly and lift hips higher","Roll shoulders under and clasp hands","Keep knees hip-width apart"],"good_msg":"Beautiful Bridge! Hips lifted and open.","description":"Bridge Pose: Open chest, strengthen back.","difficulty":"Beginner","benefits":"Strengthens back, glutes, hamstrings."},
    "pigeon":{"cues":["Front shin parallel to mat","Back leg extended straight behind","Fold forward over front leg"],"corrections":["Square your hips — don't tilt to one side","Keep back leg straight and extended","Fold forward to deepen hip stretch"],"good_msg":"Deep Pigeon! Hips beautifully open.","description":"Pigeon Pose: Ultimate hip opener.","difficulty":"Intermediate","benefits":"Deeply opens hip flexors and rotators."},
    "camel":{"cues":["Kneel with hips over knees","Lift chest and arch back","Reach hands toward heels"],"corrections":["Keep hips over knees as you arch back","Lift chest before dropping head back","Engage core to protect lower back"],"good_msg":"Amazing Camel! Full backbend achieved!","description":"Camel Pose: Deep heart-opener, builds courage.","difficulty":"Intermediate","benefits":"Stretches front body, opens chest and hip flexors."},
    "half_moon":{"cues":["Standing leg straight and strong","Bottom hand on floor under shoulder","Top arm reaching straight up"],"corrections":["Straighten and engage standing leg","Stack top hip over bottom hip","Extend top arm toward ceiling"],"good_msg":"Radiant Half Moon! Perfectly balanced.","description":"Half Moon: Balance, grace, and luminous energy.","difficulty":"Intermediate","benefits":"Strengthens ankles, legs, core."},
    "boat":{"cues":["Balance on sit bones","Legs raised, shins parallel","Arms forward parallel to floor"],"corrections":["Lift feet higher — shins parallel","Straighten spine — avoid rounding","Keep arms active and parallel to floor"],"good_msg":"Strong Boat! Core fully engaged!","description":"Boat Pose: Ignite core strength.","difficulty":"Intermediate","benefits":"Strengthens core, hip flexors, spine."},
    "crow":{"cues":["Hands flat, shoulder-width","Knees on backs of upper arms","Shift weight forward, lift feet"],"corrections":["Place knees high on triceps — not low on wrists","Shift weight forward slowly","Engage core and look forward, not down"],"good_msg":"Flying in Crow! Amazing arm balance!","description":"Crow Pose: Gateway to arm balances.","difficulty":"Advanced","benefits":"Builds arm, wrist, core strength."},
    "eagle":{"cues":["Bend standing knee slightly","Cross thigh over other, wrap if possible","Cross same-side arm for double wrap"],"corrections":["Bend standing knee more deeply","Squeeze thighs together for stability","Lift elbows to shoulder height"],"good_msg":"Beautiful Eagle! Perfectly wrapped.","description":"Eagle Pose: Release tension, strengthen legs.","difficulty":"Intermediate","benefits":"Releases shoulder and hip tension."},
    "lotus":{"cues":["Sit cross-legged, feet on opposite thighs","Spine tall and erect","Hands on knees, palms up or down"],"corrections":["Ease feet onto thighs gently — never force","Lengthen your spine upward","Relax shoulders away from ears"],"good_msg":"Serene Lotus! Grounded and open.","description":"Lotus Pose: Classic meditation seat.","difficulty":"Advanced","benefits":"Opens hips and knees. Calms the mind."},
    "fish":{"cues":["Slide hands under hips","Press elbows down, arch chest up","Top of head lightly touches floor"],"corrections":["Press elbows firmly to lift chest","Keep legs active and together","Breathe deeply into expanded chest"],"good_msg":"Beautiful Fish! Chest wide and open.","description":"Fish Pose: Open the throat and chest.","difficulty":"Beginner","benefits":"Opens chest, throat. Stimulates thyroid."},
    "seated_forward":{"cues":["Legs extended, feet flexed","Hinge from hips, not waist","Hold feet, ankles, or shins"],"corrections":["Hinge from hips — don't round from waist","Keep spine long as you reach forward","Flex feet to protect knees"],"good_msg":"Deep Seated Forward Fold! Spine long.","description":"Seated Forward Bend: Deep back body stretch.","difficulty":"Beginner","benefits":"Stretches spine, shoulders, hamstrings."},
    "supine_twist":{"cues":["Draw one knee to chest","Guide knee across body with opposite hand","Extend other arm out, look away"],"corrections":["Let both shoulders stay flat on mat","Stack knees for gentler twist","Breathe deeply — let gravity work"],"good_msg":"Perfect Supine Twist! Spine is happy!","description":"Supine Twist: Release spine, restore balance.","difficulty":"Beginner","benefits":"Releases spinal tension, aids digestion."},
    "low_lunge":{"cues":["Front knee over ankle at 90°","Back knee lowered to mat","Hips sinking toward floor"],"corrections":["Align front knee over ankle","Lower back knee to mat gently","Sink hips lower toward ground"],"good_msg":"Perfect Low Lunge! Deep hip opening.","description":"Low Lunge: Open hip flexors, build strength.","difficulty":"Beginner","benefits":"Stretches hip flexors, strengthens legs."},
    "side_plank":{"cues":["Feet stacked or staggered","Bottom arm straight under shoulder","Top arm reaching toward ceiling"],"corrections":["Lift hips — keep body in straight line","Stack wrist under shoulder","Reach top arm straight up"],"good_msg":"Powerful Side Plank! Perfect alignment.","description":"Side Plank: Build oblique strength and stability.","difficulty":"Intermediate","benefits":"Strengthens obliques, arms, wrists."},
}

# ─── Distinct keypoint signatures per pose ────────────────────────────────────
SIGS = {
    "tadasana":       {0:[.50,.10,0,1],11:[.42,.28,0,1],12:[.58,.28,0,1],13:[.37,.45,0,1],14:[.63,.45,0,1],15:[.34,.60,0,1],16:[.66,.60,0,1],23:[.45,.65,0,1],24:[.55,.65,0,1],25:[.45,.82,0,1],26:[.55,.82,0,1],27:[.45,.97,0,1],28:[.55,.97,0,1]},
    "vrikshasana":    {0:[.50,.08,0,1],11:[.42,.25,0,1],12:[.58,.25,0,1],13:[.38,.10,0,1],14:[.62,.10,0,1],15:[.48,.03,0,1],16:[.52,.03,0,1],23:[.48,.60,0,1],24:[.52,.60,0,1],25:[.48,.78,0,1],26:[.57,.55,0,1],27:[.48,.96,0,1],28:[.54,.42,0,1]},
    "warrior_i":      {0:[.50,.12,0,1],11:[.43,.28,0,1],12:[.57,.28,0,1],13:[.38,.12,0,1],14:[.62,.12,0,1],15:[.44,.04,0,1],16:[.56,.04,0,1],23:[.44,.55,0,1],24:[.56,.55,0,1],25:[.35,.72,0,1],26:[.62,.80,0,1],27:[.30,.92,0,1],28:[.70,.92,0,1]},
    "warrior_ii":     {0:[.50,.28,0,1],11:[.42,.38,0,1],12:[.58,.38,0,1],13:[.15,.40,0,1],14:[.85,.40,0,1],15:[.04,.42,0,1],16:[.96,.42,0,1],23:[.44,.58,0,1],24:[.60,.58,0,1],25:[.32,.75,0,1],26:[.72,.80,0,1],27:[.25,.95,0,1],28:[.80,.95,0,1]},
    "warrior_iii":    {0:[.50,.40,.15,1],11:[.42,.42,.15,1],12:[.58,.42,.15,1],13:[.38,.42,.15,1],14:[.62,.42,.15,1],15:[.30,.42,.20,1],16:[.70,.42,.20,1],23:[.48,.52,.05,1],24:[.52,.52,.05,1],25:[.50,.70,0,1],26:[.50,.38,.20,1],27:[.50,.90,0,1],28:[.50,.28,.25,1]},
    "goddess":        {0:[.50,.18,0,1],11:[.36,.32,0,1],12:[.64,.32,0,1],13:[.22,.38,0,1],14:[.78,.38,0,1],15:[.18,.28,0,1],16:[.82,.28,0,1],23:[.44,.58,0,1],24:[.56,.58,0,1],25:[.25,.75,0,1],26:[.75,.75,0,1],27:[.18,.93,0,1],28:[.82,.93,0,1]},
    "downward_dog":   {0:[.50,.52,-.20,1],11:[.38,.40,.05,1],12:[.62,.40,.05,1],13:[.25,.55,.10,1],14:[.75,.55,.10,1],15:[.18,.68,0,1],16:[.82,.68,0,1],23:[.46,.12,.20,1],24:[.54,.12,.20,1],25:[.42,.52,.05,1],26:[.58,.52,.05,1],27:[.40,.85,0,1],28:[.60,.85,0,1]},
    "cobra":          {0:[.50,.18,.25,1],11:[.38,.35,.10,1],12:[.62,.35,.10,1],13:[.30,.50,.05,1],14:[.70,.50,.05,1],15:[.28,.60,0,1],16:[.72,.60,0,1],23:[.46,.75,-.08,1],24:[.54,.75,-.08,1],25:[.46,.88,-.05,1],26:[.54,.88,-.05,1],27:[.46,.97,-.03,1],28:[.54,.97,-.03,1]},
    "plank":          {0:[.50,.32,0,1],11:[.38,.40,0,1],12:[.62,.40,0,1],13:[.30,.52,0,1],14:[.70,.52,0,1],15:[.25,.62,0,1],16:[.75,.62,0,1],23:[.46,.45,0,1],24:[.54,.45,0,1],25:[.44,.55,0,1],26:[.56,.55,0,1],27:[.42,.65,0,1],28:[.58,.65,0,1]},
    "triangle":       {0:[.52,.35,0,1],11:[.42,.42,0,1],12:[.58,.42,0,1],13:[.08,.30,0,1],14:[.60,.72,0,1],15:[.04,.22,0,1],16:[.58,.88,0,1],23:[.46,.55,0,1],24:[.58,.55,0,1],25:[.30,.75,0,1],26:[.72,.75,0,1],27:[.22,.95,0,1],28:[.78,.95,0,1]},
    "child_pose":     {0:[.50,.82,-.30,1],11:[.38,.72,-.15,1],12:[.62,.72,-.15,1],13:[.28,.85,-.10,1],14:[.72,.85,-.10,1],15:[.20,.92,-.05,1],16:[.80,.92,-.05,1],23:[.46,.58,-.10,1],24:[.54,.58,-.10,1],25:[.44,.48,-.08,1],26:[.56,.48,-.08,1],27:[.42,.38,-.05,1],28:[.58,.38,-.05,1]},
    "chair_pose":     {0:[.50,.12,0,1],11:[.43,.28,0,1],12:[.57,.28,0,1],13:[.42,.10,0,1],14:[.58,.10,0,1],15:[.44,.03,0,1],16:[.56,.03,0,1],23:[.45,.55,0,1],24:[.55,.55,0,1],25:[.42,.72,0,1],26:[.58,.72,0,1],27:[.40,.94,0,1],28:[.60,.94,0,1]},
    "bridge":         {0:[.50,.90,-.20,1],11:[.40,.80,-.10,1],12:[.60,.80,-.10,1],13:[.32,.88,-.05,1],14:[.68,.88,-.05,1],15:[.28,.95,0,1],16:[.72,.95,0,1],23:[.45,.35,.30,1],24:[.55,.35,.30,1],25:[.42,.62,.05,1],26:[.58,.62,.05,1],27:[.40,.88,0,1],28:[.60,.88,0,1]},
    "pigeon":         {0:[.50,.22,0,1],11:[.38,.38,0,1],12:[.62,.38,0,1],13:[.28,.55,0,1],14:[.72,.55,0,1],15:[.22,.70,0,1],16:[.78,.70,0,1],23:[.35,.62,-.05,1],24:[.62,.60,-.05,1],25:[.22,.68,-.08,1],26:[.70,.80,-.05,1],27:[.18,.82,-.05,1],28:[.72,.96,-.02,1]},
    "camel":          {0:[.50,.55,.20,1],11:[.38,.30,.15,1],12:[.62,.30,.15,1],13:[.30,.48,.05,1],14:[.70,.48,.05,1],15:[.28,.65,-.05,1],16:[.72,.65,-.05,1],23:[.44,.52,.10,1],24:[.56,.52,.10,1],25:[.42,.68,-.02,1],26:[.58,.68,-.02,1],27:[.40,.85,-.05,1],28:[.60,.85,-.05,1]},
    "half_moon":      {0:[.50,.28,0,1],11:[.42,.35,0,1],12:[.58,.35,0,1],13:[.08,.40,0,1],14:[.55,.82,0,1],15:[.04,.32,0,1],16:[.52,.95,0,1],23:[.44,.55,0,1],24:[.55,.55,0,1],25:[.50,.72,0,1],26:[.50,.38,.15,1],27:[.50,.92,0,1],28:[.50,.28,.18,1]},
    "boat":           {0:[.50,.20,0,1],11:[.38,.35,0,1],12:[.62,.35,0,1],13:[.25,.50,0,1],14:[.75,.50,0,1],15:[.18,.58,0,1],16:[.82,.58,0,1],23:[.44,.62,-.10,1],24:[.56,.62,-.10,1],25:[.36,.42,.15,1],26:[.64,.42,.15,1],27:[.30,.28,.20,1],28:[.70,.28,.20,1]},
    "crow":           {0:[.50,.35,.15,1],11:[.38,.45,.10,1],12:[.62,.45,.10,1],13:[.30,.60,.05,1],14:[.70,.60,.05,1],15:[.25,.72,0,1],16:[.75,.72,0,1],23:[.44,.38,.20,1],24:[.56,.38,.20,1],25:[.38,.42,.22,1],26:[.62,.42,.22,1],27:[.35,.38,.28,1],28:[.65,.38,.28,1]},
    "eagle":          {0:[.50,.12,0,1],11:[.43,.28,0,1],12:[.57,.28,0,1],13:[.40,.22,0,1],14:[.60,.22,0,1],15:[.45,.18,0,1],16:[.55,.18,0,1],23:[.46,.55,0,1],24:[.54,.55,0,1],25:[.44,.72,0,1],26:[.56,.65,0,1],27:[.46,.90,0,1],28:[.50,.78,.05,1]},
    "lotus":          {0:[.50,.18,0,1],11:[.38,.32,0,1],12:[.62,.32,0,1],13:[.35,.50,0,1],14:[.65,.50,0,1],15:[.25,.68,0,1],16:[.75,.68,0,1],23:[.40,.68,-.10,1],24:[.60,.68,-.10,1],25:[.22,.80,-.12,1],26:[.78,.80,-.12,1],27:[.58,.75,-.10,1],28:[.42,.75,-.10,1]},
    "fish":           {0:[.50,.25,.22,1],11:[.40,.45,.10,1],12:[.60,.45,.10,1],13:[.36,.60,.05,1],14:[.64,.60,.05,1],15:[.32,.75,0,1],16:[.68,.75,0,1],23:[.46,.72,-.05,1],24:[.54,.72,-.05,1],25:[.44,.86,-.03,1],26:[.56,.86,-.03,1],27:[.43,.97,-.01,1],28:[.57,.97,-.01,1]},
    "seated_forward": {0:[.50,.62,-.15,1],11:[.40,.52,-.10,1],12:[.60,.52,-.10,1],13:[.35,.62,-.08,1],14:[.65,.62,-.08,1],15:[.28,.75,-.05,1],16:[.72,.75,-.05,1],23:[.44,.68,-.12,1],24:[.56,.68,-.12,1],25:[.44,.80,-.08,1],26:[.56,.80,-.08,1],27:[.43,.95,-.03,1],28:[.57,.95,-.03,1]},
    "supine_twist":   {0:[.30,.88,-.05,1],11:[.38,.80,-.08,1],12:[.62,.80,-.08,1],13:[.10,.80,-.05,1],14:[.65,.80,-.05,1],15:[.05,.80,-.02,1],16:[.68,.80,-.02,1],23:[.44,.65,-.08,1],24:[.60,.50,-.12,1],25:[.60,.42,-.15,1],26:[.56,.75,-.05,1],27:[.62,.35,-.18,1],28:[.54,.92,-.02,1]},
    "low_lunge":      {0:[.50,.14,0,1],11:[.43,.28,0,1],12:[.57,.28,0,1],13:[.40,.12,0,1],14:[.60,.12,0,1],15:[.44,.04,0,1],16:[.56,.04,0,1],23:[.44,.52,0,1],24:[.56,.52,0,1],25:[.32,.70,0,1],26:[.65,.78,-.05,1],27:[.26,.88,0,1],28:[.68,.95,-.03,1]},
    "side_plank":     {0:[.50,.28,.10,1],11:[.38,.38,.05,1],12:[.62,.38,.05,1],13:[.30,.30,.08,1],14:[.58,.55,0,1],15:[.25,.22,.12,1],16:[.55,.68,0,1],23:[.44,.45,.05,1],24:[.58,.45,.05,1],25:[.46,.58,.02,1],26:[.58,.62,0,1],27:[.46,.72,0,1],28:[.55,.76,0,1]},
}


def generate_pose_keypoints(pose_name, n_samples=400):
    np.random.seed(POSE_LABELS.index(pose_name) * 17 + 7)
    sig = SIGS.get(pose_name, {})
    samples = []
    for _ in range(n_samples):
        kp = np.random.randn(132) * 0.035
        for lm_idx, vals in sig.items():
            start = lm_idx * 4
            kp[start:start+4] = np.array(vals) + np.random.randn(4) * 0.022
        samples.append(kp)
    return np.array(samples)


def train_and_save():
    print("=" * 60)
    print("  ZenPose - Training 25 Pose Model")
    print("=" * 60)
    X_list, y_list = [], []
    for pose in POSE_LABELS:
        print(f"  [{POSE_LABELS.index(pose)+1:02d}/25] {pose}")
        X = generate_pose_keypoints(pose, 400)
        X_list.append(X)
        y_list.extend([pose] * 400)

    X = np.vstack(X_list)
    y = np.array(y_list)
    print(f"\n  Dataset: {X.shape[0]} samples x {X.shape[1]} features")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    pipe = Pipeline([
        ('sc', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=300, max_depth=18, min_samples_split=4, random_state=42, n_jobs=-1)),
    ])
    print("  Training...")
    pipe.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, pipe.predict(X_te))
    print(f"  [OK] Accuracy: {acc*100:.2f}%\n")
    print(classification_report(y_te, pipe.predict(X_te), target_names=le.classes_))

    os.makedirs('models', exist_ok=True)
    pickle.dump(pipe, open('models/zenpose_model.pkl','wb'))
    pickle.dump(le,   open('models/label_encoder.pkl','wb'))
    pickle.dump({'labels':POSE_LABELS,'display_names':POSE_DISPLAY_NAMES,'emojis':POSE_EMOJIS,
                 'categories':POSE_CATEGORIES,'feedback':POSE_FEEDBACK},
                open('models/pose_metadata.pkl','wb'))
    print("  [OK] Models saved to models/")
    print("=" * 60)

if __name__ == "__main__":
    train_and_save()
