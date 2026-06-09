# CCAV-T — Cooperative & Connected AV Testing

V2V 기반 협력 자율주행 시험 시스템. 실차(Ioniq5 / Avante)와 CARLA 시뮬레이터 양쪽에서 동작하며, CLM(차선 변경 합류) · ETrA(긴급 차량 회피) 시나리오를 지원한다.

---

## 디렉토리 구조

```
CCAV-T-clean/
├── common/                  # 실차 · CARLA 공통 모듈
│   ├── msg/                 # ROS 커스텀 메시지 + 빌드 파일
│   ├── sharing_info/        # 인지 · 계획 (hd_map, perception, planning, DR 보정)
│   ├── selfdriving/         # 차량 제어 (Pure Pursuit + APID, transmitter)
│   ├── ui/                  # PyQt 모니터링 UI + RViz 시각화
│   └── utils/               # 공유 유틸 (TTC 계산, 지도 설정 등)
│
├── real_vehicle/            # 실차 전용
│   ├── v2x/                 # OBU 하드웨어 V2X 통신 (socket)
│   ├── utils/               # make_data.py — 실차 CSV 로거
│   └── manager/             # 실행 스크립트 (ioniq5, avante, bag 등)
│
├── carla/                   # CARLA 시뮬레이션 전용
│   ├── bridge/              # carla_ros_bridge.py · v2v_bridge.py
│   ├── scenarios/           # CLM / ETrA 시나리오 클래스
│   ├── map/                 # CarlaMAP (Town06 Road40)
│   ├── config/              # channel_model.yaml (V2X 채널 프로파일)
│   ├── utils/               # make_data_carla.py — CARLA CSV 로거
│   └── manager/             # 자동화 스크립트 (run_all, run_treatment 등)
│
└── tests/                   # 단위 테스트
```

---

## 아키텍처

### 모듈 통신 흐름

```
[CARLA bridge / 실차 센서]
        ↓
[sharing_info]  인지(LiDAR/V2X) → DR 보정 → 경로 계획 → EgoShareInfo 발행
        ↓
[v2x / v2v_bridge]  V2V 채널 에뮬레이션 (delay · PRR)
        ↓
[selfdriving]  Pure Pursuit + APID → 액추에이터 명령
        ↓
[차량 / CARLA actor]
```

### 시나리오 조건 (2×2 Factorial)

| 조건 | V2X 채널 | LiDAR |
|------|----------|-------|
| C1 | baseline (yeongjong) | baseline |
| C2 | degraded | baseline |
| C3 | baseline | degraded |
| C4 | degraded | degraded |

---

## ROS 환경 설정

```bash
# Primary (Ego)
export ROS_IP={ego_IP}
export ROS_MASTER_URI=http://$ROS_IP:11311
export ROS_HOSTNAME=$ROS_IP

# Secondary (Target)
export ROS_MASTER_URI=http://{ego_IP}:11311
export ROS_HOSTNAME={target_IP}
```

빌드:
```bash
cd ~/catkin_ws && catkin_make && source devel/setup.bash
```

---

## 실차 실행

```bash
cd real_vehicle/manager

./ioniq5.sh       # Ioniq5 ego
./avante.sh       # Avante target
./sim_ego.sh      # 시뮬레이터 모드 ego
./sim_target.sh   # 시뮬레이터 모드 target
./bag.sh          # rosbag 녹화 (ego)
./bag_target.sh   # rosbag 녹화 (target)
./kill_all.sh     # 전체 종료
```

---

## CARLA 시뮬레이션 실행

### 단일 실행 (수동)

```bash
cd carla/manager

# WC (V2X 협력)
./carla_wc.sh [SCENARIO] [TEST_MODE] [CHANNEL] [SPEED_KMH] [CONDITION]
# 예) ./carla_wc.sh CLM2 faster yeongjong 90 c1

# WOC (LiDAR 단독)
./carla_woc.sh [SCENARIO] [TEST_MODE] [CHANNEL] [SPEED_KMH] [CONDITION]
```

### 전체 자동 실행 (Baseline)

```bash
cd carla/manager
./run_all.sh
# [1] WOC + nonpc: 50, 90, 130 km/h  → 288쌍
# [2] WOC + npc:   30, 70, 110 km/h
```

### Treatment 실행 (DR 보정 활성화)

```bash
cd carla/manager
./run_treatment.sh
# WOC + nonpc + DR enabled → data/treatment/ 출력, 288쌍
```

### 특정 시나리오 재실행

```bash
cd carla/manager
# trials_rerun.txt 편집 후:
./run_rerun.sh
```

### auto_run.py 주요 옵션

```
--wc / --woc          WC 또는 WOC 모드만 실행
--nonpc / --npc       NPC 없음 / 있음
--speeds 50 90 130    속도 지정
--only clm / etra     시나리오 필터
--dr                  DR 보정 활성화 (treatment 모드)
--resume              기존 CSV 있으면 건너뜀
--dry-run             실제 실행 없이 런 목록만 출력
--trials-file FILE    특정 시나리오 키 파일로 실행
```

---

## Dead-Reckoning (DR) 보정 모듈

`common/sharing_info/dr_compensator.py`

수신된 target state에 latency만큼 등속 외삽하여 stale 데이터 보정.

```
px_c = px + vx · (τ/1000)
py_c = py + vy · (τ/1000)
```

**Safety guards**:
- τ > 300 ms → raw 그대로 출력
- |v| < 0.5 m/s → 보정 안 함
- NaN 입력 → 보정 안 함

**활성화** (`sharing_info/main.py` 인수):
```bash
python3 main.py ego CarlaMap 1 --dr-enabled --dr-target both
```

`--dr-target` 옵션: `both` | `v2x_only` | `lidar_only`

**로그**: `/ego/dr_log` 토픽 (Float32MultiArray 12개 필드) → CSV `dr_*` 컬럼으로 기록.

---

## 커스텀 ROS 메시지 (`common/msg/`)

| 메시지 | 주요 필드 |
|--------|----------|
| `ShareInfo` | state, signal, pose, velocity, target_velocity, paths, obstacles |
| `Path` | pose (x, y) |
| `Obstacle` | pose, velocity, distance, dangerous, lidar_delay |

---

## 단위 테스트

```bash
cd tests
python3 test_dr_compensator.py
# All 6 tests passed.
```

---

## 주요 설정 파일

| 파일 | 설명 |
|------|------|
| `carla/config/channel_model.yaml` | V2X 채널 프로파일 (yeongjong/degraded/ideal) |
| `common/selfdriving/control/configs/*.ini` | 차량별 제어 파라미터 |
| `common/sharing_info/hd_map/maps/*.json` | HD 맵 (Midan, KIAPI, Solbat 등) |
| `common/ui/yamls/end_point.yaml` | 시나리오 종료 지점 좌표 |
