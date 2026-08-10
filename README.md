#  GrowBotOS — AI-Driven Vertical Farming System

> ** 1st Place — COMSATS Career Expo 2025**

An end-to-end intelligent vertical farming prototype that combines distributed embedded sensing, a TabNet environmental prediction model, and a fully self-supervised visual plant-health diagnosis pipeline — all integrated through Firebase and a live React dashboard.

**84.95% plant-health classification accuracy · Zero manual image labels · Closed-loop actuator control · Sub-30s visual monitoring cycle**

---

## 📸 System Overview


The system operates across four tightly integrated layers:

| Layer | Technology | Role |
|---|---|---|
| **Embedded** | 2× ESP32 (Arduino C++) | Sensor acquisition, relay actuation via UART |
| **Edge AI** | Raspberry Pi 5, Python | TabNet + MobileNetV3 inference, sensor fusion |
| **Cloud** | Firebase Realtime Database | Global shared memory, WebSocket push |
| **Dashboard** | React / Vercel | Live monitoring, crop selection, AI outputs |

---

##  Key Features

- **No manual image labeling** — 10,686 unlabeled leaf images → fully labeled 7-class dataset via SimCLR + K-Means
- **TabNet crop intelligence** — predicts optimal temperature, humidity, NPK, and pH set-points per crop using sequential attention
- **Self-supervised visual diagnosis** — detects 7 plant-health states from live camera feed
- **Purple-light preprocessing** — grey-world white balance + ExG/HSV dual-mask segmentation for grow-light environments
- **Sensor fusion** — CNN visual output (70%) + rule-based NPK/pH gate (30%) for ambiguous deficiency cases
- **Closed-loop control** — PI controller drives fan, pump, grow light, and heater actuators from TabNet set-points
- **Real-time dashboard** — Firebase WebSocket delivers sensor readings and AI results to the React UI
- **Graceful degradation** — if camera fails → sensor-only diagnosis; if Firebase drops → Pi services run from cache; if a service crashes → systemd restarts it in under 2 seconds

---

##  Repository Structure

```
│
├── hardware/
│   ├── esp32_tray1/           # ESP32 firmware — Tray 1 (DHT22, NPK, BH1750, MQ-135, relay)
│   ├── esp32_tray2/           # ESP32 firmware — Tray 2 (same stack, independent node)
│   └── schematics/            # Wiring diagrams, pinout reference, RS-485 setup
│
├── edge-ai/                   # All code that runs on Raspberry Pi 5
│   ├── pi_tabnet.py           # TabNet inference → Firebase → actuator command loop
│   ├── pi_cnn.py              # Camera capture → preprocessing → MobileNetV3 inference
│   ├── sensor_fusion.py       # 70/30 visual + sensor-gate fusion logic
│   ├── actuator_control.py    # PI controller for fan, pump, LED, heater
│   └── requirements.txt
│
├── ml-models/
│   ├── simclr_pretraining/    # SimCLR contrastive pretraining on PlantVillage
│   ├── kmeans_clustering/     # PCA → MiniBatch K-Means → pseudo-label generation
│   ├── mobilenetv3_finetune/  # Multi-head MobileNetV3-Small fine-tuning
│   └── tabnet_training/       # TabNet on Crop Recommendation Dataset
│
├── dashboard/                 # React web app (deployed on Vercel)
│   └── src/
│
├── docs/
│   ├── IEEE_paper.pdf         # Conference paper (4-page, two-column IEEE format)
│   ├── thesis.pdf             # Full undergraduate thesis
│   └── poster.pdf
│
└── assets/
    ├── hardware_diagram.png   # Hardware block diagram
    └── architecture_diagram.png  # 4-layer software architecture
```

---

##  How It Works

### 1. Sensing (ESP32 × 2)
Each grow tray has its own independent ESP32 that reads:
- **DHT22** — temperature & humidity
- **NPK 7-in-1 soil sensor** — nitrogen, phosphorus, potassium, pH, moisture, soil temp, conductivity (RS-485 / Modbus RTU)
- **BH1750** — light intensity (I2C, 1–65,535 lux)
- **MQ-135** — air quality
- **Water level sensor** — analog ADC

Readings are validated, packaged as JSON, and sent to the Raspberry Pi 5 over dedicated UART lines. Invalid readings are marked `null` rather than propagated.

---

### 2. TabNet Environmental Prediction (`pi_tabnet.py`)
A **TabNet** model trained on the [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) predicts:
- Which crop the current sensor environment best matches
- Optimal set-points (temperature, humidity, NPK, pH) for the operator's selected crop

Set-points are published to Firebase and consumed by a **PI controller** that drives actuator relays every second.

**Why TabNet?** Its sequential attention mechanism selects the most relevant sensor features at each decision step, giving interpretable predictions without manual feature engineering.

---

### 3. Self-Supervised Visual Diagnosis (`pi_cnn.py`)

A three-stage pipeline that requires **zero manual image-level annotation**:

```
10,686 unlabeled leaf images
        ↓
  SimCLR pretraining          (contrastive self-supervised, NT-Xent loss, τ=0.5)
        ↓
  576-dim feature embeddings
        ↓
  PCA (50 components, ~85% variance) → MiniBatch K-Means (k=7)
        ↓
  Cluster-level manual labeling     (7 plant-health categories)
        ↓
  Balance to 737 images/class → 589/73/75 train/val/test split
        ↓
  MobileNetV3-Small fine-tuning     (2-stage: frozen backbone → full unfreeze)
        ↓
  84.95% test accuracy · macro F1 = 0.8482
```

**7 health classes:** Healthy · Water Stress · Nitrogen Deficiency · Phosphorus Deficiency · Potassium Deficiency · Temperature Stress · pH Imbalance

**Purple-light correction** (applied before every inference):
1. Grey-world white balance (R×1.45, G×0.35, B×1.30)
2. ExG + HSV dual-mask leaf segmentation
3. Resize to 224×224, ImageNet normalization

---

### 4. Sensor Fusion
```
final_probs = 0.70 × neural_probs + 0.30 × sensor_gate
```
The sensor gate fires based on live NPK/pH/temperature threshold breaches, resolving visually ambiguous early-stage nutrient deficiencies that the CNN alone struggles to distinguish.

---

### 5. Cloud & Dashboard
- All sensor data, AI outputs, and operator settings live in **Firebase Realtime Database**
- React dashboard on **Vercel** subscribes via Firebase JS SDK → updates instantly
- Firebase paths: `Sensors/Tray1`, `Sensors/Tray2`, `verticalfarm/aioutput`, `verticalfarm/cnnoutput`, `verticalfarm/selectedcrop`

---

##  Results

### CNN Classification — Final Test Set (525 images, 75/class)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Healthy | 0.7791 | 0.8933 | 0.8323 |
| Water Stress | 0.8889 | 0.8533 | 0.8707 |
| Nitrogen Deficiency | 0.8281 | 0.7067 | 0.7626 |
| Phosphorus Deficiency | 0.8427 | **1.0000** | **0.9146** |
| Potassium Deficiency | 0.8714 | 0.8133 | 0.8414 |
| Temperature Stress | **0.9265** | 0.8400 | 0.8811 |
| pH Imbalance | 0.8289 | 0.8400 | 0.8344 |
| **Macro Avg.** | **0.8522** | **0.8495** | **0.8482** |

**Best validation accuracy: 87.28% · Final test accuracy: 84.95%**

### Clustering Quality (SimCLR + K-Means)

| Metric | Score | Interpretation |
|---|---|---|
| Calinski-Harabasz Index | 1065.54 | Strong inter-cluster separation |
| Silhouette Score | 0.1836 | Acceptable for continuous biological variation |
| Davies-Bouldin Index | 1.8673 | Moderate cluster overlap (expected for stress symptoms) |

### System Performance

| Metric | Value |
|---|---|
| CNN inference cycle | Within 30-second visual monitoring window |
| UART packet loss (soil sensor) | < 0.5% across extended testing |
| Firebase → dashboard latency | Near-instantaneous WebSocket push |
| Systemd service restart time | < 2 seconds on crash |
| Validation → test accuracy gap | 2.33% (within normal generalization bounds) |

---

## 🔧 Hardware Components

| Component | Purpose | Protocol |
|---|---|---|
| ESP32 (×2) | Microcontroller per tray | UART to Pi |
| DHT22 | Temperature & humidity | Digital GPIO |
| NPK 7-in-1 Soil Sensor | N, P, K, pH, moisture, soil temp, conductivity | RS-485 / Modbus RTU |
| BH1750 | Light intensity (lux) | I2C |
| MQ-135 | Air quality | Analog ADC |
| Water Level Sensor | Tray water level | Analog ADC |
| Relay Module | Switch fan, pump, LED, heater | GPIO |
| Raspberry Pi 5 (8GB) | Edge AI, coordination | Wi-Fi, UART |
| USB Camera | Live leaf imaging | USB |
| 12V PSU | Power supply for actuators | — |

---

##  Setup & Installation

### ESP32 Firmware
1. Open `hardware/esp32_tray1/` in Arduino IDE
2. Install libraries: `DHT sensor library`, `BH1750`, `ModbusMaster`, `ArduinoJson`
3. Set your UART baud rate (`115200`) and pin assignments in `config.h`
4. Flash to ESP32 via USB

### Raspberry Pi 5 (Edge AI)
```bash
git clone https://github.com/your-username/GrowBotOS.git
cd GrowBotOS/edge-ai
pip install -r requirements.txt

# Add your Firebase service account key
cp firebase_key.example.json firebase_key.json
# (fill in your credentials)

# Register and start systemd services
sudo cp services/pi_tabnet.service /etc/systemd/system/
sudo cp services/pi_cnn.service /etc/systemd/system/
sudo systemctl enable pi_tabnet pi_cnn
sudo systemctl start pi_tabnet pi_cnn
```

### React Dashboard
```bash
cd dashboard
npm install
# Set your Firebase config in .env
npm run dev        # local dev
# or deploy to Vercel
vercel deploy
```

### Train the Models (optional — pretrained weights included)
```bash
# SimCLR pretraining
cd ml-models/simclr_pretraining
python train_simclr.py

# K-Means clustering + pseudo-label generation
cd ../kmeans_clustering
python cluster_and_label.py

# MobileNetV3 fine-tuning
cd ../mobilenetv3_finetune
python finetune.py

# TabNet training
cd ../tabnet_training
python train_tabnet.py
```

---

##  Tech Stack

**Embedded:** C++ · Arduino Framework · ESP32 · Modbus RTU · RS-485

**Edge AI:** Python · PyTorch · PyTorch-TabNet · OpenCV · scikit-learn · NumPy

**ML Models:** SimCLR · MobileNetV3-Small · TabNet · K-Means · PCA

**Cloud:** Firebase Realtime Database · Firebase JS SDK

**Frontend:** React · Vercel

**DevOps:** systemd · Git
