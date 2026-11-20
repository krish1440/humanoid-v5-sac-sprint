# Humanoid-v5 SAC Sprint — 8,900+ Score Olympic Locomotion

[![Environment](https://img.shields.io/badge/Environment-Gymnasium%20Humanoid--v5-blue.svg)](https://gymnasium.farama.org/environments/mujoco/humanoid/)
[![Algorithm](https://img.shields.io/badge/Algorithm-Soft%20Actor--Critic%20(SAC)-orange.svg)](https://arxiv.org/abs/1801.01290)
[![Physics Engine](https://img.shields.io/badge/Physics-MuJoCo%203.x-green.svg)](https://mujoco.org/)
[![Peak Score](https://img.shields.io/badge/Peak%20Score-8%2C900.0-gold.svg)](#benchmark-results)
[![Success Rate](https://img.shields.io/badge/Success%20Rate-100%25%20(Full%201%2C000%20Steps)-brightgreen.svg)](#evaluation-results)

An end-to-end, high-performance Deep Reinforcement Learning project that trains **Gymnasium's MuJoCo Humanoid-v5** robot to transition from chaotic stumbling into an elite **+3.20 m/s (~11.5 km/h) bipedal Olympic sprinter** using vectorized Soft Actor-Critic (SAC).

---

## 🏆 Key Benchmark Results

In standard RL literature, `Humanoid-v5` is considered "solved" at **6,000 points**. This trained policy achieves **8,900.0 peak points** with **100% completion rate**:

| Metric | Official Solved Threshold | Our Trained SAC Policy | Status |
| :--- | :---: | :---: | :---: |
| **All-Time Peak Score** | 6,000.0 | **8,900.0** | 🔥 **+48.3% over threshold** |
| **Deterministic Test Score (Avg)** | ~5,500 – 6,000 | **8,512.0** | 🚀 **Elite Consistency** |
| **Full 1,000-Step Survival Rate** | ~70 – 80% | **100.0% (5 / 5)** | 🛡️ **Zero Falls** |
| **Forward Sprint Velocity (Avg)** | ~1.5 – 2.0 m/s | **+3.20 m/s (~11.5 km/h)** | ⚡ **Olympic Sprint** |
| **Peak Instant Velocity** | ~2.5 m/s | **+3.44 m/s (~12.4 km/h)** | 🏃 **Maximum Thrust** |
| **Total Environment Steps** | — | **38,046,000+** | 🧠 **74,059 Episodes** |

---

## 📊 Evaluation Summary (`test.py`)

Deterministic evaluation over 5 consecutive seeded episodes with the Champion Model (`best_model.pt`):

```text
=================================================================
                     EVALUATION SUMMARY REPORT
=================================================================
 Model Evaluated:       BEST RECORD MODEL (Step 38,015,799)
 Checkpoint Path:       checkpoints/best_model.pt
 Total Episodes Tested: 5
 Average Score:         8512.0
 Max Score:             8809.9
 Min Score:             8313.3
 Full 1,000-Step Runs:  5 / 5 (100.0% Success Rate)
 Avg Sprint Score:      8512.0 (on completed runs)
 Overall Avg Velocity:  +3.20 m/s (~11.52 km/h)
=================================================================
```

Full per-episode telemetry is saved in [`checkpoints/backup_38m_milestone_record8900/evaluation_results.json`](checkpoints/backup_38m_milestone_record8900/evaluation_results.json).

---

## 🧠 Algorithmic & Architecture Highlights

1. **Soft Actor-Critic (SAC)**:
   - **Squashed Gaussian Actor**: Deep MLP with LayerNorm layers outputting action distributions squashed into $[-1.0, 1.0]$ via numerical Tanh mapping.
   - **Twin Q-Critics ($Q_1, Q_2$)**: Two independent action-value networks minimizing overestimation bias via clipped double-Q estimation.
   - **Automatic Entropy Temperature ($\alpha$)**: Dynamic entropy tuning with safety bounding ($\alpha \in [0.028, 0.35]$) ensuring stability without losing motor exploration elasticity.
2. **Observation Normalizer**:
   - Online running mean and standard deviation computation with $[-10, 10]$ outlier clipping, critical for scaling diverse joint position, velocity, and actuator force ranges.
3. **8-Core Parallel Vectorized Training (`train_parallel.py`)**:
   - Vectorized environment stepped concurrently across 8 MuJoCo instances.
   - Live visual Pygame HUD with dynamic real-time telemetry, hotkeys, and automatic checkpointing.
4. **Camera Chase Tracking (`test.py`)**:
   - Smooth 3D drone camera tracking (`camera_name="track"`) following the humanoid's torso across infinite ground plane.

---

## 📁 Repository Structure

```text
├── agent.py                                           # SAC Agent, Actor, Twin-Critics & Replay Buffer
├── test.py                                            # Interactive 3D evaluation runner with HUD & camera tracking
├── train_parallel.py                                  # Vectorized 8-worker parallel training engine
├── train.py                                           # Single-worker training engine
├── best_model.pt                                      # 8,900.0 Champion Model weights
├── latest_checkpoint.pt                               # 38M Milestone latest weights
├── training_log.csv                                   # Full training log (74,059 episodes)
├── checkpoints/
│   ├── best_model.pt                                  # Active Champion Model
│   ├── latest_checkpoint.pt                           # Active latest checkpoint
│   ├── backup_38m_milestone_record8900/               # 38.0M Milestone (Score 8,900.0 + JSON eval)
│   ├── backup_latest_27_35m_record8043/               # 27.35M Milestone (Score 8,043.5 + JSON eval)
│   ├── backup_27m_milestone_record7993/               # 27.0M Milestone (Score 7,993.4)
│   ├── backup_latest_25_8m_record7952/                # 25.8M Milestone (Score 7,952.8)
│   ├── backup_latest_24m_milestone/                   # 24.0M Milestone
│   ├── backup_current_23m_milestone/                  # 23.4M Milestone
│   └── backup_1m_milestone/                           # Initial 1.0M baseline
└── logs/
    └── training_log.csv                               # Complete episode telemetry history
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/krish1440/humanoid-v5-sac-sprint.git
cd humanoid-v5-sac-sprint
pip install gymnasium[mujoco] mujoco pygame torch numpy
```

### 2. Run Evaluation (Watch the Robot Sprint)
To run the champion model with visual 3D drone camera tracking:
```bash
# Evaluate champion model for 5 episodes
python test.py best 5

# Evaluate champion model for 20 episodes
python test.py best 20

# Evaluate latest checkpoint continuously
python test.py latest
```

#### Visual Evaluation Controls:
* `[SPACE]` : Pause / Resume evaluation
* `[ESC]` : Exit cleanly and print Summary Report

---

### 3. Training
To train from scratch or resume from existing checkpoint:
```bash
# Vectorized 8-core parallel training (Recommended)
python train_parallel.py

# Single-worker training
python train.py
```

#### Training Hotkeys:
* `[Spacebar]` : Pause / Resume training
* `[V]` : Toggle between **Turbo Mode** (maximum training FPS) and **Live 3D Mode**
* `[S]` : Instant on-demand checkpoint save
* `[ESC]` : Safe save & clean shutdown

---

## 📜 Milestone Evolution History

| Milestone | Steps | Episodes | Best Score | Avg Fwd Velocity | Completion Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Initial Baseline** | 1,000,000 | 5,876 | 5,319.6 | +1.42 m/s | 0% (Stumbles) |
| **Stable Walk** | 15,008,329 | 30,452 | 7,675.4 | +2.28 m/s | 95% (19/20) |
| **Sprint Acceleration** | 22,224,045 | 44,293 | 7,852.3 | +2.54 m/s | ~65% |
| **8,000 Barrier Broken** | 27,324,405 | 54,129 | 8,043.5 | +2.46 m/s | 100% (20/20) |
| **Champion Olympic Sprint** | **38,015,799** | **74,000** | **8,900.0** | **+3.20 m/s** | **100% (5/5)** |

---

## 📄 License
This project is licensed under the MIT License. Built with [Gymnasium](https://gymnasium.farama.org/) and [MuJoCo](https://mujoco.org/).
