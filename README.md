# ROPE: A Three-Axis Evaluation Framework for Source Reliability, Target Observability, and Driving Performance in Cooperative and Standalone Automated Driving

**ROPE (Reliability–Observability–Performance Evaluation)** is a diagnostic framework for decomposing the performance difference between Cooperative Automated Driving (CAD) and Standalone Automated Driving (SAD). Rather than treating CAD and SAD as separate systems, ROPE defines them as two deployment configurations of the same ADS pipeline — differing only in their target-state source (V2X vs. LiDAR) and cooperative decision logic. It then isolates *where* the difference originates: in the information source, in target observation quality, or in downstream driving behavior.

This repository ships both the **analysis framework** (`rope/`) and the **driving stack** (`driving/`) used to generate the input data.

---

## Repository Structure

```
├── rope/                        # ROPE analysis framework
│   ├── pipeline.py              # Main pipeline: CSV → parquet → metrics
│   ├── config.py                # ROPEConfig dataclass
│   ├── utils.py                 # Shared utilities
│   ├── axes/
│   │   ├── r_axis.py            # R-Axis: source reliability (V2X delay, PRR, LiDAR quality)
│   │   ├── o_axis.py            # O-Axis: target observability (OST — availability, accuracy, consistency)
│   │   ├── p_axis.py            # P-Axis: driving performance (safety, stability, efficiency)
│   │   ├── cross_axis.py        # Cross-axis: R→O and ΔO→ΔP propagation analysis
│   │   └── timewise_axis.py     # T-Axis: time-series residuals aligned to maneuver events
│   ├── modules/
│   │   ├── io.py                # Parquet I/O and statistics helpers
│   │   ├── events.py            # Event detection (merge trigger, ETrA detection)
│   │   └── uncertainty.py       # OST (Object Stable Tracking) computation
│   ├── viz/                     # Matplotlib figure generators per axis
│   └── ui/                      # PyQt5 GUI (main_window, panels, canvas)
│
├── driving/                     # Cooperative AV driving stack (ROS-based)
│   ├── common/                  # Shared modules (real vehicle + CARLA)
│   │   ├── selfdriving/         # Autonomous driving node (Pure Pursuit + APID control)
│   │   │   ├── main.py          # ROS node entry point
│   │   │   ├── control/         # Controller: purepursuit.py, apid.py
│   │   │   └── transmitter/     # Vehicle CAN transmitters: ioniq5, avante, carla
│   │   ├── sharing_info/        # Cooperative perception & planning node
│   │   │   ├── main.py          # ROS node entry point
│   │   │   ├── perception/      # LiDAR obstacle handler, object simulator
│   │   │   ├── planning/        # Local path planner, velocity planner
│   │   │   ├── hd_map/          # Lanelet2 HD map parser (KIAPI, Midan, Solbat, Yeongjong)
│   │   │   └── dr_compensator.py  # Dead-reckoning latency compensator
│   │   ├── ui/                  # PyQt5 + RViz monitoring UI
│   │   ├── msg/                 # Custom ROS messages (ShareInfo, Path, Obstacle)
│   │   └── utils/               # TTC calculator, coordinate transforms, map config
│   ├── real_vehicle/            # Real-vehicle-specific code
│   │   ├── v2x/                 # OBU hardware V2X interface (UDP socket)
│   │   ├── utils/make_data.py   # ROS bag → ROPE CSV logger
│   │   └── manager/             # Launch scripts per vehicle (ioniq5, avante)
│   ├── carla/                   # CARLA simulation-specific code
│   │   ├── bridge/              # CARLA↔ROS bridge, V2V channel emulator
│   │   ├── scenarios/           # CLM and ETrA scenario classes
│   │   ├── map/carla_map.py     # CARLA Town06 Road40 map wrapper
│   │   ├── config/channel_model.yaml  # V2X channel profiles (field-calibrated)
│   │   ├── utils/make_data_carla.py   # CARLA → ROPE CSV logger
│   │   └── manager/             # Experiment automation scripts
│   └── tests/                   # Unit tests (DR compensator)
│
├── main.py                      # ROPE entry point (CLI + GUI)
└── requirements.txt             # Python dependencies for ROPE
```

---

## How the Two Parts Relate

```
driving/                              rope/
────────                              ──────
[Real vehicle / CARLA simulation]
  CAD run (WC) ──┐
  SAD run (WOC) ─┤  make_data.py / make_data_carla.py
                 └→ paired CSV logs per trial
                                         ↓
                                   pipeline.py
                                   (windowed preprocessing)
                                         ↓
                          ┌──────────────┼──────────────┐
                        R-Axis        O-Axis          P-Axis
                     (source       (target         (driving
                    reliability)  observability)  performance)
                          └──────────────┼──────────────┘
                                   Cross-axis
                                 (R→O, ΔO→ΔP)
                                         ↓
                              figures + tables → rope_output/
```

Each CSV row is one 10 Hz timestep. ROPE applies a fixed analysis window around the key maneuver event, computes per-scenario metrics, and runs all axes. CAD runs are labeled **WC** (With Cooperation); SAD runs are labeled **WOC** (Without Cooperation).

---

## ROPE: Setup & Usage

### Requirements

- Python ≥ 3.8
- `pip install -r requirements.txt`

### CLI

```bash
# Run full analysis on a CSV directory
python main.py --csv data/SIMULATION --out rope_output --axes R,O,P

# Paired Wilcoxon for field data (matched WC/WOC trials)
python main.py --csv data/FIELD --out rope_output --paired
```

### GUI

```bash
python main.py
```

### Programmatic

```python
from rope.pipeline import ROPEPipeline
from rope.config import ROPEConfig

pipeline = ROPEPipeline(config=ROPEConfig(axes=['R', 'O', 'P']))
results = pipeline.run('data/SIMULATION', 'rope_output')
```

---

## ROPE Axes

| Axis | Question | Key metrics |
|------|----------|-------------|
| **R** | Do CAD and SAD differ in source reliability? | V2X: delay, PRR, valid\_rate; LiDAR: detection distance, delay |
| **O** | Does that source difference translate into target observability difference? | OST (Object Stable Tracking): availability, accuracy, temporal consistency |
| **P** | Does observability difference propagate to driving outcomes? | Safety: TTC, DRAC, headway; Stability: jerk RMS, yaw RMS; Efficiency: merge time, avoidance distance |
| **Cross** | Where does the R→O→P propagation chain hold or break? | Spearman ρ (ΔR→ΔOST, ΔOST→ΔP, ΔR→ΔP) |
| **T** | How do metrics evolve over the maneuver window? | Time-series residuals, event-aligned traces |

Paired comparison uses Wilcoxon signed-rank test with Cohen's d effect size. Multiple comparisons are corrected with Holm–Bonferroni.

---

## Datasets

| Dataset | Pairs | Environment | Notes |
|---------|-------|-------------|-------|
| **SIM** | 287 | CARLA (no NPC) | Controlled 2×2 factorial: V2X × LiDAR degradation |
| **SIM-NPC** | 294 | CARLA (with NPC traffic) | Same factorial + ambient traffic |
| **FIELD** | 34 | Real road (Yeongjong / KIAPI) | Ioniq5 (ego) + Avante (target) |

---

## Scenarios

| ID | Full name | Description |
|----|-----------|-------------|
| CLM1–4 | Cooperative Lane Merging | Ego vehicle merges into target lane; cooperative gap negotiation via V2X (CAD) vs. LiDAR-only detection (SAD) |
| ETrA1–4 | Emergency Trajectory Alignment | Ego yields to approaching target; alignment strategy selected based on V2X-shared intent (CAD) vs. LiDAR detection (SAD) |

Conditions C1–C4 apply a 2×2 factorial of V2X channel (baseline / degraded) × LiDAR (baseline / degraded).

---

## Driving Stack: Setup

Requires **ROS Noetic** (Ubuntu 20.04) and, for simulation, **CARLA 0.9.13**.

```bash
# Build custom ROS messages
cp -r driving/common/msg ~/catkin_ws/src/
cd ~/catkin_ws && catkin_make && source devel/setup.bash
```

### CARLA Simulation

```bash
cd driving/carla/manager

./run_all.sh          # Full experiment grid (WC + WOC × all conditions)
./run_treatment.sh    # DR compensation enabled (treatment arm)
./run_rerun.sh        # Rerun trials listed in trials_rerun.txt
```

`auto_run.py` key options:

| Option | Description |
|--------|-------------|
| `--wc` / `--woc` | WC (CAD) or WOC (SAD) only |
| `--nonpc` / `--npc` | Without / with NPC traffic |
| `--speeds 50 90 130` | Target speeds (km/h) |
| `--only clm` / `--only etra` | Scenario filter |
| `--dr` | Enable dead-reckoning latency compensation |
| `--resume` | Skip trials with existing CSV output |
| `--dry-run` | Print trial list without executing |

### Real Vehicle

```bash
cd driving/real_vehicle/manager
./ioniq5.sh     # Ego (Ioniq5)
./avante.sh     # Target (Avante CN7)
./bag.sh        # Start rosbag recording
./kill_all.sh   # Stop all nodes
```

---

## CSV Log Format

Both `make_data.py` (field) and `make_data_carla.py` (CARLA) produce CSVs with a common schema:

| Column group | Description |
|---|---|
| `ts_ns` | Timestamp (nanoseconds) |
| `ego_enu_x/y`, `ego_h`, `ego_velocity` | Ego state (ENU frame) |
| `v2v_delay`, `v2v_packet_rate`, `v2v_distance` | V2X channel metrics (R-axis inputs) |
| `lidar_dist_*`, `lidar_delay` | LiDAR obstacle metrics (R-axis inputs) |
| `ttc`, `drac`, `headway` | Pre-computed safety metrics (P-axis) |
| `dr_*` | Dead-reckoning compensation log (if `--dr` enabled) |

---

## Citation

> 김가나, "ROPE: 협력 자율주행과 단독 자율주행의 정보원 신뢰도, Target 관측 품질, 주행 성능 3축 평가 체계," 공학박사학위논문, 인하대학교 대학원, 2026.

> G. Kim, "ROPE: A Three-Axis Evaluation Framework for Source Reliability, Target Observability, and Driving Performance in Cooperative and Standalone Automated Driving," Ph.D. Dissertation, Inha University, 2026.
