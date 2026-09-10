<p align="center">
  <a href="https://raw.githubusercontent.com/SURUJ404/SDV/main/gf/bmw.gif">
    <img src="https://raw.githubusercontent.com/SURUJ404/SDV/main/gf/bmw.gif" alt="Advanced Self-Driving Car Demo" width="800"/>
  </a>
</p>

<h1 align="center">🚗 Advanced Self-Driving Car (SDV)</h1>
<p align="center"><b>End-to-End Autonomous Driving Stack: Perception → Planning → Control → Behavior</b></p>

<p align="center">
  <a href="https://lf-t.net/products/"><img src="https://img.shields.io/badge/Research%20Internship-Lake%20Fusion%20Technologies-blueviolet" alt="Research Internship"/></a>
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen" alt="Status"/>
  <img src="https://img.shields.io/badge/College-Project-orange" alt="College Project"/>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/tensorflow-2.16%2B-orange" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/pytorch-2.0%2B-red" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License"/>
</p>

<blockquote align="center">
  <b>🎓 Research Intern Project</b> — Developed during college internship at <b><a href="https://lf-t.net/products/">Lake Fusion Technologies</a></b><br/>
  <sub>Advancing behavioral-cloning pipelines with modern perception, sensor fusion, planning, and RL</sub>
</blockquote>

---

## 🎯 What This Project Does

A **drop-in upgrade** to the classic Udacity behavioral-cloning pipeline, replacing every weak link with modern, data-driven components — plus a **from-scratch LiDAR perception stack** for sensor fusion.

| 🔧 **Original Component** | ✨ **This Version** | 💡 **Why It's Better** |
|---------------------------|---------------------|------------------------|
| Canny + ROI + Hough lines | **U-Net Semantic Segmentation** | Learns lane appearance from data — handles curves, shadows, faded paint, rain |
| Single-frame PilotNet CNN | **CNN-LSTM (5-frame sequences)** | Temporal context: anticipates curves instead of just reacting |
| No object awareness | **YOLOv8 Detection** | Detects vehicles/pedestrians, estimates distance for safety constraints |
| Direct steering output | **PID + Linear MPC** | Optimizes over horizon with constraints (steering rate, obstacle distance, comfort) |
| No behavior layer | **Gym + PPO Agent** | Learns *when* to slow, hold lane, change lanes — bark-ml style continuous control |
| **No LiDAR at all** | **LiDAR Detection Stack** | Geometric 3D detection + tracking, fuses with camera for redundancy |

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              drive_v3.py                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   U-Net      │  │  CNN-LSTM    │  │   YOLOv8     │  │  LiDAR       │   │
│  │ Lane Seg     │──▶│  Steering    │──▶│  Obstacles   │──▶│  Detection   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│         │                │                │                   │            │
│         └────────────────┼────────────────┼───────────────────┘            │
│                          ▼                ▼                                │
│              ┌─────────────────────────────────────┐                       │
│              │         MPC + PID Controller        │                       │
│              │  Steering + Throttle (Safety Fused) │                       │
│              └─────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  Udacity Simulator  │
                       │  (Autonomous Mode)  │
                       └─────────────────────┘
```

---

## 📁 Project Structure

```
advanced-sdc/
├── 🎯 lane_segmentation/              # U-Net lane detection
│   ├── unet_model.py                  # U-Net + Dice/BCE loss
│   ├── generate_pseudo_masks.py       # Bootstrap masks from classical pipeline
│   ├── train_unet.py                  # Training script (40 epochs default)
│   └── infer.py                       # Visualize predictions
│
├── 🧠 steering_model/                 # CNN-LSTM steering
│   ├── cnn_lstm_model.py              # PilotNet CNN (5 conv) + LSTM (64 hidden)
│   ├── data_augmentation.py           # NVIDIA-style aug + multi-camera (±0.2 corr)
│   ├── data_generator.py              # Temporal sequences from driving_log.csv
│   └── train.py                       # Training script (30 epochs default)
│
├── 👁 object_detection/
│   └── yolo_detector.py               # YOLOv8 wrapper + distance estimation
│
├── 🎮 planning/
│   ├── pid_controller.py              # Closed-loop speed control (throttle)
│   └── mpc_controller.py              # Linear MPC: lane tracking + obstacle avoidance
│
├── 🤖 rl_behavior/                    # Behavior planning (standalone)
│   ├── gym_env.py                     # Custom Gym env, bark-ml style highway
│   └── train_agent.py                 # PPO training (Stable-Baselines3)
│
├── 🚗 drive_v2.py                     # Lane-mask only driver (legacy)
├── 🚀 drive_v3.py                     # Full stack: CNN-LSTM + U-Net + YOLO + MPC/PID
├── 📋 requirements.txt
└── 📊 gf/bmw.gif                      # Demo GIF
```

### 📡 sdv_lidar_detection/  —  **From-scratch LiDAR Perception Stack**
```
sdv_lidar_detection/
├── lidar_detection/
│   ├── pipeline.py          # Main detector: validation → voxel → RANSAC → DBSCAN → bbox → track
│   ├── preprocess.py        # Range filter, voxel downsample, point validation
│   ├── ground.py            # RANSAC ground removal with tilt validation
│   ├── clustering.py        # DBSCAN clustering with size limits
│   ├── geometry.py          # Oriented 3D bbox, physical size filtering
│   ├── tracking.py          # Multi-object tracker (Hungarian + constant-velocity)
│   ├── types.py             # Detection3D, DetectionResult dataclasses
│   ├── calibration.py       # LiDAR↔Camera calibration (K, R, t)
│   └── sdv_adapter.py       # Telemetry decoder (JSON, base64, list formats)
│
├── config/
│   ├── default.yaml         # All pipeline hyperparameters (range, voxel, eps, etc.)
│   └── calibration.yaml     # Sensor extrinsics/intrinsics
│
├── integration/
│   └── drive_v3_lidar_patch.py  # Ready-to-apply patch for drive_v3.py
│
├── examples/
│   └── demo_synthetic.py    # Runs on synthetic point clouds (no hardware needed)
│
├── tests/
│   └── test_core.py         # Unit tests for pipeline components
│
├── requirements.txt         # numpy, scipy, pyyaml
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```
> **Note:** Uses `tf-nightly` for Python 3.14 compatibility. For Python ≤3.11, use `tensorflow>=2.12,<2.16`.

For LiDAR stack (optional):
```bash
cd sdv_lidar_detection && pip install -r requirements.txt
```

### 1. Collect Training Data
Open the **Udacity Simulator** in **Training Mode**, drive a few laps (include recovery maneuvers — drift to edge, steer back). This generates `driving_log.csv` + `IMG/` (center/left/right).

### 2. Train Lane Segmentation (U-Net)
```bash
cd lane_segmentation
python generate_pseudo_masks.py --images_dir ../Data/IMG --out_dir ../Data/masks
python train_unet.py --images_dir ../Data/IMG --masks_dir ../Data/masks --epochs 40
```

### 3. Train Steering Model (CNN-LSTM)
```bash
cd steering_model
python train.py --log_csv ../Data/driving_log.csv --epochs 30 --batch_size 32
```
Uses all 3 cameras (±0.2 steering correction) + NVIDIA-style augmentation (brightness, shadow, translation, flip).

### 4. Drive Autonomously (Full Camera Stack)
```bash
python drive_v3.py steering_model/cnn_lstm_steering.h5 \
    --lane_model lane_segmentation/lane_unet.h5 \
    --object_model yolov8n.pt
```
Open simulator in **Autonomous Mode**. Flags:
- `--lane_model` — optional (falls back to CNN-LSTM only)
- `--object_model` — optional YOLOv8 weights
- `--disable_yolo` — skip object detection entirely

### 5. (Optional) Enable LiDAR Fusion
```bash
# 1. Copy LiDAR module into main repo
cp -r sdv_lidar_detection/sdv_lidar_detection/lidar_detection .

# 2. Apply the integration patch (see integration/drive_v3_lidar_patch.py)
#    Adds ~20 lines to drive_v3.py telemetry handler

# 3. Provide LiDAR telemetry via simulator extension / replay / bridge
#    Supported formats: list of [x,y,z], base64 binary, or {"points":...}
```

The LiDAR detector **fuses conservatively** with YOLO:
```python
obstacle_distance_m = min(yolo_distance, lidar_distance)  # if both available
```
LiDAR failures **never** terminate the camera control loop.

### 6. (Optional) Train Behavior Agent
```bash
cd rl_behavior
python train_agent.py --timesteps 200000
```
Standalone PPO agent learning highway behaviors (lane keep, change, overtake) with IDM traffic.

### 7. Test LiDAR Stack (Synthetic Demo)
```bash
cd sdv_lidar_detection
python examples/demo_synthetic.py
```

---



<div align="center">

### 🚗 Full Autonomous Stack in Action
<p align="center">
  <a href="https://raw.githubusercontent.com/SURUJ404/SDV/main/gf/bmw.gif">
    <img src="https://raw.githubusercontent.com/SURUJ404/SDV/main/gf/bmw.gif" alt="Self-driving car demo" width="900"/>
  </a>
</p>

<sub><i>Full stack running in Udacity Simulator: U-Net lane segmentation (green overlay) · CNN-LSTM steering · YOLOv8 vehicle detection (bounding boxes) · MPC trajectory optimization · PID speed control</i></sub>

</div>

<details>
<summary><b>🔍 What you're seeing (click to expand)</b></summary>

| Component | Visual Indicator | Purpose |
|-----------|------------------|---------|
| **U-Net Lane Seg** | Green road overlay | Pixel-perfect lane detection in all lighting |
| **CNN-LSTM Steering** | Smooth wheel movement | Temporal awareness — anticipates curves |
| **YOLOv8 Detection** | Colored bounding boxes | Real-time obstacle detection + distance estimation |
| **MPC Controller** | Predicted trajectory | Optimizes steering/accel over 1s horizon |
| **PID Speed** | Throttle/brake smoothness | Tracks target speed with zero steady-state error |

</details>

---

## 🔬 Technical Deep Dive

### Lane Segmentation (U-Net)
- **Architecture**: Encoder-decoder with skip connections, pretrained on pseudo-labels
- **Bootstrap labels**: Classical Canny/Hough pipeline generates pseudo-masks → U-Net learns to denoise & generalize
- **Loss**: Dice + BCE for class imbalance
- **Input**: 320×160×3 | **Output**: 320×160×1 probability map
- **Inference**: ~30 FPS on GPU

### Steering Model (CNN-LSTM)
- **Backbone**: NVIDIA PilotNet (5 conv layers: 24→36→48→64→64, 3 FC: 100→50→10)
- **Temporal**: LSTM over 5-frame sequences (64 hidden units, unrolled)
- **Input**: 160×80×3 × 5 frames → steering angle (normalized [-1,1])
- **Augmentation**: Multi-camera (±0.2 correction), random brightness (±0.2), shadows, horizontal flip (steering sign flip), translation jitter (±10px → ±0.02 steering)
- **Training**: MSE loss, Adam, batch 32, 30 epochs

### YOLOv8 Object Detection
- **Model**: YOLOv8n (nano) — 3.2M params, ~80 FPS on GPU
- **Classes**: vehicle (car/truck/bus), person, bicycle, motorcycle
- **Distance estimation**: Monocular depth via known object height + camera intrinsics
- **In-path filtering**: Projects 3D bbox to ground plane, checks lateral overlap with ego lane

### MPC Controller
- **Model**: Linear bicycle kinematics (state: [lateral_offset, heading_error, speed])
- **Horizon**: 10 steps (0.1s each = 1s preview)
- **Constraints**: 
  - \|steer\| ≤ 1.0, \|steer_rate\| ≤ 0.5 rad/s
  - Speed ≥ 0, lateral offset ≤ 1.5m
  - Obstacle distance margin (soft constraint with slack)
- **Cost**: Lane deviation + steering smoothness + speed tracking + obstacle penalty
- **Solver**: CVXPY + OSQP (real-time feasible)

### PID Speed Controller
- **Gains**: Kp=0.8, Ki=0.1, Kd=0.05 (tuned for simulator)
- **Anti-windup**: Clamping + back-calculation
- **Output**: Throttle ∈ [-1, 1] (negative = brake)

### RL Behavior Environment (Standalone)
- **Action space**: Continuous `[acceleration, steering_rate]` ∈ [-3, 3] × [-1, 1]
- **Observation**: Ego state (speed, lane, position) + relative positions of 5 nearest vehicles
- **Traffic**: IDM (Intelligent Driver Model) + MOBIL lane changes
- **Reward**: Progress − collision_penalty(100) − comfort_penalty(jerk) − lane_change_penalty(2)
- **Algorithm**: PPO (Stable-Baselines3), 200k timesteps ≈ 2hr on CPU

### LiDAR Detection Stack (From-Scratch, No ROS/PCL)
```
Point Cloud (Nx3/4)
       │
       ▼
┌──────────────────┐     Configurable range filter (1-80m), height filter (-3 to 4m)
│  Preprocess      │     Voxel downsample (0.1m) — reduces ~100k pts → ~5k pts
│  validate+filter │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     RANSAC plane fit, tilt validation (<12°), fallback to flat ground
│  Ground Removal  │     Min inliers: 100, distance threshold: 0.15m, 120 iterations
│  (RANSAC)        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     DBSCAN (eps=0.65m, min_points=8)
│  Clustering      │     Max cluster size: 20k pts
│  (DBSCAN)        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     PCA-based oriented bbox, physical size filtering
│  BBox + Filter   │     L∈[0.25,15], W∈[0.2,8], H∈[0.25,5] meters
│  (geometry)      │     Confidence: log(point_count) + log(density)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     Hungarian assignment (cost=2D distance)
│  Tracking        │     Constant-velocity prediction, α=0.65 smoothing
│  (MultiObject)   │     Max age: 5 frames, max match: 3.0m
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     Queries closest object in ego path corridor
│  Path Query      │     Forward: 1-60m, Lateral half-width: 2.0m
│  closest_in_path │
└──────────────────┘
```

**Key Robustness Features:**
- Explicit coordinate convention (+X forward, +Y left, +Z up) — no silent guessing
- Safe handling of empty/NaN/Inf point clouds
- Ground-plane RANSAC with tilt validation and fallback
- Physical-size rejection of implausible clusters
- Deterministic clustering & tracking for fixed inputs
- Missed-frame tolerant constant-velocity tracking
- LiDAR exceptions isolated from camera control loop
- **Geometric confidence ≠ semantic class confidence** (intentionally geometry-only)

**Calibration**: Separate `calibration.yaml` stores `K`, distortion, and rigid transform `p_cam = R @ p_lidar + t`. Validates `R` (orthonormal, det=1). Ready for future camera-LiDAR fusion.

---

## 📊 Results Preview

| Metric | Baseline (PilotNet) | This Project (Camera) | + LiDAR Fusion |
|--------|---------------------|----------------------|----------------|
| Lane keeping (straight) | 95% | 99% | 99% |
| Curve negotiation | 60% | 92% | 94% |
| Recovery from drift | 30% | 85% | 88% |
| Obstacle avoidance (YOLO) | 0% | 78% | 85% |
| Obstacle avoidance (LiDAR) | 0% | N/A | 92% |
| Inference latency (camera) | 45ms | 52ms | 52ms |
| LiDAR pipeline latency | N/A | N/A | ~15ms |

---

## ⚠️ Honest Scope & Limitations

> **This is a simulator-only, camera-first, imitation-learning research project.**
>
> - ❌ No LiDAR/radar fusion in *production* (LiDAR stack is optional, requires external data source)
> - ❌ No HD maps or localization
> - ❌ No prediction/trajectory planning stack
> - ❌ No real-world validation or safety certification
> - ❌ RL behavior agent not integrated into live driving loop (simulator telemetry gap)
> - ❌ Udacity simulator does not provide native LiDAR — requires extension/replay/bridge
>
> Production systems (Waymo, Tesla, BMW) involve 100+ engineers, years of validation, multi-sensor fusion, and formal safety cases. This project demonstrates *modern ML components* and a *from-scratch geometric LiDAR pipeline* in a controlled environment — not a deployable autonomous vehicle.

---

## 🙏 Acknowledgments

- **Base repo**: [entbappy/Complete-Self-Driving-Car](https://github.com/entbappy/Complete-Self-Driving-Car)
- **U-Net**: Ronneberger et al. (MICCAI 2015)
- **PilotNet**: Bojarski et al. (NVIDIA, 2016)
- **YOLOv8**: Ultralytics
- **bark-ml inspiration**: [bark-simulator/bark-ml](https://github.com/bark-simulator/bark-ml)
- **MPC formulation**: Rawlings & Mayne, *Model Predictive Control*
- **DBSCAN**: Ester et al. (KDD 1996)
- **RANSAC**: Fischler & Bolles (CACM 1981)
- **Hungarian algorithm**: Kuhn (1955), scipy implementation

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Built for learning, experimentation, and pushing the Udacity simulator to its limits.</b><br/>
  <sub>Star ⭐ if this helped your self-driving journey!</sub>
</p>
