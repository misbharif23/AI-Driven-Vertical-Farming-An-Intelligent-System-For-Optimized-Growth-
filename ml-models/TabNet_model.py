# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import firebase_admin
from firebase_admin import credentials, db
import time

# =====================================================
# 1. DEVICE SETUP
# =====================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Running on {device.upper()}")

# =====================================================
# 2. FIREBASE INITIALIZATION
# =====================================================
cred = credentials.Certificate(
    r"C:\Users\Misbha\Desktop\FYP\tabNet\esp-sensor-data-fyp-firebase-adminsdk-fbsvc-408ce45cc0.json"
)

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://esp-sensor-data-fyp-default-rtdb.firebaseio.com/'
})

farm_ref = db.reference("vertical_farm")

# =====================================================
# 3. PREDEFINED CROPS
# =====================================================
AVAILABLE_CROPS = [
    "rice","maize","chickpea","kidneybeans","pigeonpeas",
    "mothbeans","mungbean","blackgram","lentil",
    "pomegranate","banana","mango","grapes","watermelon",
    "muskmelon","apple","orange","papaya","coconut",
    "cotton","jute","coffee"
]

default_structure = {
    "available_crops": AVAILABLE_CROPS,
    "selected_crop": "rice",

    "sensors": {
        "temperature": 0,
        "humidity": 0,
        "ph": 0,
        "N": 0,
        "P": 0,
        "K": 0
    },

    "ai_output": {
        "predicted_crop": "",
        "suggested_conditions": {
            "temperature": 0,
            "humidity": 0,
            "ph": 0,
            "N": 0,
            "P": 0,
            "K": 0
        }
    }
}

if not farm_ref.get():
    farm_ref.set(default_structure)
    print("✅ Firebase structure created")

# =====================================================
# 4. LOAD DATA
# =====================================================
df = pd.read_csv(r"C:\Users\Misbha\Desktop\FYP\tabNet\data.csv")

target = 'label'
cat_cols = ['soil_type', 'water_source_type', 'growth_stage']
num_cols = [c for c in df.columns if c not in cat_cols and c != target]

df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode labels
target_encoder = LabelEncoder()
df[target] = target_encoder.fit_transform(df[target])

for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

scaler = QuantileTransformer(output_distribution='normal')
df[num_cols] = scaler.fit_transform(df[num_cols])

features = cat_cols + num_cols
X = df[features].values
y = df[target].values

cat_idxs = [i for i, f in enumerate(features) if f in cat_cols]
cat_dims = [df[col].nunique() for col in cat_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================
# 5. TRAIN TABNET
# =====================================================
print("🌱 Training TabNet...")

clf = TabNetClassifier(
    n_d=64, n_a=64,
    n_steps=5,
    gamma=1.5,
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
    device_name=device
)

clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric=['accuracy'],
    max_epochs=50,
    patience=10,
    batch_size=256
)

print("✅ Training Complete")

# =====================================================
# 6. SENSOR LIMITS (REALISTIC VALUES)
# =====================================================
SENSOR_LIMITS = {
    "temperature": (18, 35),
    "humidity": (40, 90),
    "ph": (5.5, 7.5),
    "N": (20, 180),
    "P": (10, 120),
    "K": (20, 180),
}

# =====================================================
# 7. INVERSE MECHANISM
# =====================================================
def suggest_optimal_conditions(target_crop):

    class_id = target_encoder.transform([target_crop])[0]

    input_tensor = torch.tensor(
        X_test[0:1],
        dtype=torch.float32,
        requires_grad=True
    ).to(device)

    optimizer = torch.optim.Adam([input_tensor], lr=0.1)

    clf.network.eval()

    for _ in range(100):
        optimizer.zero_grad()

        output, _ = clf.network(input_tensor)
        probs = torch.softmax(output, dim=1)

        loss = -probs[0, class_id]
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            input_tensor.clamp_(-3, 3)

    optimized = input_tensor.detach().cpu().numpy()
    opt_num = optimized[:, len(cat_cols):]
    real_values = scaler.inverse_transform(opt_num)[0]

    results = {}

    for i, name in enumerate(num_cols):
        if name in SENSOR_LIMITS:
            mn, mx = SENSOR_LIMITS[name]
            results[name] = float(np.clip(real_values[i], mn, mx))

    return results

# =====================================================
# 8. FIREBASE HELPERS
# =====================================================
def get_selected_crop():
    return db.reference("vertical_farm/selected_crop").get()

def send_ai_output(predicted_crop, conditions):

    db.reference("vertical_farm/ai_output").set({
        "predicted_crop": predicted_crop,
        "suggested_conditions": conditions
    })

# =====================================================
# 9. REALTIME LOOP
# =====================================================
print("🔥 Waiting for crop selection...")

last_crop = None

while True:

    crop = get_selected_crop()

    if crop and crop != last_crop:

        print(f"\n🌾 New crop selected: {crop}")

        conditions = suggest_optimal_conditions(crop)

        send_ai_output(crop, conditions)

        print("✅ Firebase updated")

        last_crop = crop

    time.sleep(5)
    # =====================================================
# SAVE MODEL
# =====================================================
clf.save_model("growbot_tabnet")

print("✅ TabNet model saved")