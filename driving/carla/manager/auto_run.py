#!/usr/bin/env python3
"""CARLA 자동 데이터 취득.
사용법: python3 auto_run.py [--dry-run] [--resume]
"""
import subprocess, time, os, signal, argparse, itertools, glob
from datetime import datetime, timedelta

_procs = []

DIR      = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.abspath(os.path.join(DIR, '../..'))
DATA_DIR = os.path.join(ROOT, 'data', 'Carla')
LOG_FILE = os.path.join(DIR, 'auto_run.log')

# ── 실험 매트릭스 ─────────────────────────────────────────────────────────
SCENARIOS   = [f'CLM{i}'  for i in range(1, 5)] + [f'ETrA{i}' for i in range(1, 5)]
SPEED_MODES = ['same', 'slower', 'faster']
SPEEDS      = [50, 90, 130]
MODES       = ['wc', 'woc']
CONDITIONS  = ['c1', 'c2', 'c3', 'c4']   # 2×2 factorial: V2X × LiDAR
NPC_MODES   = ['npc', 'nonpc']

CONDITION_V2X   = {'c1': 'yeongjong', 'c2': 'degraded', 'c3': 'yeongjong', 'c4': 'degraded'}
CONDITION_LIDAR = {'c1': 'baseline',   'c2': 'baseline',  'c3': 'degraded',  'c4': 'degraded'}

TIMEOUT_BY_SPEED = {30: 220, 50: 90, 70: 120, 90: 90, 110: 90, 130: 70}
TIMEOUT_SEC  = 150  # fallback
COOLDOWN_SEC = 8
APP_COOLDOWN = 3

KILL_PATTERNS = [
    'roscore', 'rosmaster', 'carla_ros_bridge', 'v2v_bridge',
    'sharing_info', 'self_driving', 'make_data_carla', 'make_data',
    'carla_wc.sh', 'carla_woc.sh', 'carla_bridge.sh', 'carla_ego.sh', 'carla_target.sh',
    'rostopic',
]
KILL_APP_PATTERNS = [
    'v2v_bridge', 'sharing_info', 'self_driving', 'make_data_carla', 'make_data',
    'carla_ego.sh', 'carla_target.sh', 'rostopic',
]
TEST_MODE_NUM = {'same': 0, 'slower': 1, 'faster': 2}

SCENARIO_NUM = {
    'CLM1':1,'CLM2':2,'CLM3':3,'CLM4':4,
    'ETrA1':7,'ETrA2':8,'ETrA3':9,'ETrA4':10,
}
# ego lane-change signal per scenario: 1=left, 2=right, 0=none(ETrA/emergency)
SCENARIO_SIGNAL = {
    'CLM1':2,'CLM2':1,'CLM3':2,'CLM4':1,
    'ETrA1':0,'ETrA2':0,'ETrA3':0,'ETrA4':0,
}
SPEED_RATIO  = {'slower': 0.83, 'same': 1.0, 'faster': 1.17}

# ─────────────────────────────────────────────────────────────────────────────
def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def fmt_td(seconds):
    return str(timedelta(seconds=int(seconds)))

def print_progress(i, total, ok, skip, fail, run_times, label):
    pct  = i / total * 100
    bar  = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
    avg  = sum(run_times) / len(run_times) if run_times else TIMEOUT_SEC
    remaining = (total - i) * avg
    elapsed   = sum(run_times)
    print(f"\r  [{bar}] {pct:5.1f}%  {i}/{total}  "
          f"OK={ok} SKIP={skip} FAIL={fail}  "
          f"avg={avg:.0f}s  ETA={fmt_td(remaining)}  elapsed={fmt_td(elapsed)}",
          end='', flush=True)

def csv_exists(scenario, mode, speed_mode, speed_kmh, condition, npc, dr=False):
    subdir = 'treatment' if dr else 'Carla'
    suffix = '_dr_both' if dr else ''
    pat = os.path.join(ROOT, 'data', subdir,
                       f'carla_{scenario}_{mode.upper()}_{speed_mode}_{condition}_{npc}_{speed_kmh}kmh{suffix}_*.csv')
    return bool(glob.glob(pat))

def load_trials(path):
    trials = []
    for line in open(path):
        key = line.strip().split()[0] if line.strip() else ''
        if not key.startswith('carla_'):
            continue
        p = key.split('_')
        speed = int(p[6].replace('kmh', ''))
        # tuple order: (scenario, speed_mode, speed_kmh, mode, condition, npc)
        trials.append((p[1], p[3], speed, p[2].lower(), p[4], p[5]))
    return trials

def _reap_procs():
    global _procs
    for p in _procs:
        try: p.wait(timeout=0)
        except subprocess.TimeoutExpired: pass
    _procs[:] = [p for p in _procs if p.returncode is None]

def kill_all():
    for pat in KILL_PATTERNS:
        subprocess.run(['pkill', '-9', '-f', pat], capture_output=True)
    time.sleep(COOLDOWN_SEC)
    _reap_procs()

def kill_apps():
    for pat in KILL_APP_PATTERNS:
        subprocess.run(['pkill', '-9', '-f', pat], capture_output=True)
    time.sleep(APP_COOLDOWN)
    _reap_procs()

def _sh(cmd):
    p = subprocess.Popen(
        ['bash', '-c', f'source ~/catkin_ws/devel/setup.bash && {cmd}'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={**os.environ, 'ROS_MASTER_URI': 'http://localhost:11311'},
    )
    _procs.append(p)
    return p

def start_bridge(scenario, speed_mode, speed_kmh, npc):
    _sh('roscore')
    time.sleep(1)
    npc_flag = '--npc' if npc == 'npc' else ''
    _sh(f'python3 {ROOT}/carla/bridge/carla_ros_bridge.py '
        f'--scenario {scenario} --test_mode {speed_mode} --speed {speed_kmh} {npc_flag}')
    time.sleep(5)
    log('[bridge] bridge ready')

def reconfigure_bridge(scenario, speed_mode, speed_kmh, npc):
    snum = SCENARIO_NUM[scenario]
    tnum = TEST_MODE_NUM[speed_mode]
    nint = 1 if npc == 'npc' else 0
    ready_proc = subprocess.Popen(
        ['rostopic', 'echo', '-n', '1', '/carla/ready'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    time.sleep(0.3)
    pub('/carla/reset', f'{snum}, {speed_kmh}, {nint}, {tnum}')
    try:
        ready_proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        ready_proc.kill(); ready_proc.communicate()
        log('[bridge] WARNING: /carla/ready timeout — proceeding anyway')

def pub(topic, data, timeout=10):
    p = subprocess.Popen(
        ['rostopic', 'pub', '-1', topic, 'std_msgs/Float32MultiArray', f'data: [{data}]'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill(); p.wait()

def pub_parallel(pairs):
    """여러 topic을 동시에 발행하고 모두 완료될 때까지 대기."""
    procs = [
        subprocess.Popen(
            ['rostopic', 'pub', '-1', topic, 'std_msgs/Float32MultiArray', f'data: [{data}]'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        for topic, data in pairs
    ]
    for p in procs:
        try: p.wait(timeout=10)
        except subprocess.TimeoutExpired: p.kill(); p.wait()

def auto_set_start(scenario, speed_mode, speed_kmh):
    snum   = SCENARIO_NUM.get(scenario, 1)
    v_ref  = speed_kmh / 3.6
    tv_vel = v_ref * SPEED_RATIO.get(speed_mode, 1.0)
    # state=0 초기화 (동시 발행)
    pub_parallel([
        ('/ego/user_input',    f'0, 0, {v_ref:.4f}, {snum}, 0'),
        ('/target/user_input', f'0, 0, {tv_vel:.4f}, {snum}, 0'),
    ])
    time.sleep(1)
    pub('/carla/set', '1')
    time.sleep(2)
    # state=1 시작 (동시 발행 — ego/target gap 없애기)
    esig = SCENARIO_SIGNAL.get(scenario, 0)
    pub_parallel([
        ('/ego/user_input',    f'1, {esig}, {v_ref:.4f}, {snum}, 0'),
        ('/target/user_input', f'1, 0, {tv_vel:.4f}, {snum}, 0'),
    ])

def wait_scenario_done(timeout):
    try:
        r = subprocess.run(
            ['rostopic', 'echo', '-n', '1', '/scenario/done'],
            timeout=timeout, capture_output=True, text=True
        )
        return 'True' in r.stdout or 'true' in r.stdout
    except subprocess.TimeoutExpired:
        return False

def run_trial(scenario, speed_mode, mode, speed_kmh, condition, npc, dr=False):
    channel = CONDITION_V2X[condition]
    t0 = time.monotonic()

    kill_apps()
    reconfigure_bridge(scenario, speed_mode, speed_kmh, npc)

    woc    = '--woc' if mode == 'woc' else ''
    dr_arg = 'true' if dr else ''
    _sh(f'python3 {ROOT}/carla/bridge/v2v_bridge.py --channel {channel} {woc}')
    _sh(f'bash {DIR}/carla_ego.sh {speed_mode} {mode} {condition} {npc} {dr_arg}')
    _sh(f'bash {DIR}/carla_target.sh {speed_mode} {mode} {condition} {dr_arg}')

    time.sleep(6)
    auto_set_start(scenario, speed_mode, speed_kmh)
    done = wait_scenario_done(TIMEOUT_BY_SPEED.get(speed_kmh, TIMEOUT_SEC))
    time.sleep(2)
    return done, time.monotonic() - t0

def _setup_signals():
    def _handler(sig, frame):
        log('\n[SIGNAL] 정리 중...')
        kill_all()
        exit(0)
    signal.signal(signal.SIGINT,  _handler)
    signal.signal(signal.SIGTERM, _handler)

def main():
    _setup_signals()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run',  action='store_true')
    parser.add_argument('--resume',   action='store_true')
    parser.add_argument('--only',     choices=['clm', 'etra'], default=None)
    parser.add_argument('--nonpc',    action='store_true', help='nonpc 조건만 실행')
    parser.add_argument('--npc',      action='store_true', help='npc 조건만 실행')
    parser.add_argument('--wc',       action='store_true', help='wc 모드만 실행')
    parser.add_argument('--woc',      action='store_true', help='woc 모드만 실행')
    parser.add_argument('--speeds',      type=int, nargs='+', default=None, help='속도 목록 (예: --speeds 30 70 110)')
    parser.add_argument('--trials-file', default=None, help='실행할 시나리오 키 파일 (txt)')
    parser.add_argument('--dr',          action='store_true', help='treatment 모드: DR 보정 활성화, data/treatment/ 출력')
    args = parser.parse_args()

    scenarios  = [s for s in SCENARIOS if args.only is None or s.lower().startswith(args.only)]
    npc_modes  = ['nonpc'] if args.nonpc else (['npc'] if args.npc else NPC_MODES)
    speeds     = args.speeds if args.speeds else SPEEDS
    modes      = ['wc'] if args.wc else (['woc'] if args.woc else MODES)
    runs = list(itertools.product(scenarios, SPEED_MODES, speeds, modes, CONDITIONS, npc_modes))
    if args.trials_file:
        runs = load_trials(args.trials_file)
    total = len(runs)
    log(f"총 {total}개 런 시작  (resume={args.resume}  dry-run={args.dry_run}  dr={args.dr})")

    ok_cnt = skip_cnt = fail_cnt = 0
    run_times = []
    start_wall = time.monotonic()

    if not args.dry_run:
        first = next((r for r in runs
                      if not (args.resume and csv_exists(r[0], r[3], r[1], r[2], r[4], r[5], args.dr))), None)
        if first:
            start_bridge(first[0], first[1], first[2], first[5])

    for i, (scenario, speed_mode, speed_kmh, mode, condition, npc) in enumerate(runs, 1):
        label = f"{scenario} {mode.upper()} {condition} {npc} {speed_mode} {speed_kmh}km/h"

        if args.resume and csv_exists(scenario, mode, speed_mode, speed_kmh, condition, npc, args.dr):
            skip_cnt += 1
            print_progress(i, total, ok_cnt, skip_cnt, fail_cnt, run_times, label)
            continue

        print()
        log(f"[{i:3}/{total}] ▶ {label}")

        if args.dry_run:
            ok_cnt += 1
            print_progress(i, total, ok_cnt, skip_cnt, fail_cnt, run_times, label)
            continue

        done, elapsed = run_trial(scenario, speed_mode, mode, speed_kmh, condition, npc, args.dr)
        run_times.append(elapsed)

        if done:
            ok_cnt += 1
            log(f"[{i:3}/{total}] ✓ {label}  ({elapsed:.0f}s)")
        else:
            fail_cnt += 1
            log(f"[{i:3}/{total}] ✗ TIMEOUT {label}  ({elapsed:.0f}s)")

        print_progress(i, total, ok_cnt, skip_cnt, fail_cnt, run_times, label)

    kill_all()
    total_elapsed = time.monotonic() - start_wall
    print()
    log(f"=== 완료  OK={ok_cnt}  SKIP={skip_cnt}  TIMEOUT={fail_cnt}  총소요={fmt_td(total_elapsed)} ===")

if __name__ == '__main__':
    main()
