#!/usr/bin/env python3
"""CARLA simulation data logger — real-vehicle CSV format compatible."""
import rospy, csv, os, sys, time, math
from collections import deque
from datetime import datetime

import setproctitle
setproctitle.setproctitle("make_data_carla")

from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Pose
from ccavt.msg import ShareInfo

PACKET_SIZE = 373   # bytes (fixed OBU size from real data)
WINDOW_SEC  = 10.0  # rolling window for packet metrics
N_LIDAR        = 10   # max ego lidar columns
N_TARGET_LIDAR = 3    # max target lidar columns

SCENARIO_MAP = {**{i: f'CLM{i}' for i in range(1, 7)},
                **{i: f'ETrA{i-6}' for i in range(7, 13)}}

def _lidar_header(prefix, n):
    cols = []
    for i in range(n):
        cols += [f'{prefix}_{i}_enu_x', f'{prefix}_{i}_enu_y', f'{prefix}_{i}_h',
                 f'{prefix}_{i}_velocity', f'{prefix}_{i}_danger', f'{prefix}_{i}_lidar_delay']
    return cols

HEADER = (
    ['ts_ns', 'ego_state', 'ego_signal', 'ego_enu_x', 'ego_enu_y', 'ego_h', 'ego_velocity']
    + _lidar_header('lidar', N_LIDAR)
    + ['v2v_cnt', 'v2v_state', 'v2v_signal', 'v2v_enu_x', 'v2v_enu_y', 'v2v_h', 'v2v_velocity',
       'v2v_perf_state', 'v2v_v2x', 'v2v_rtt', 'v2v_mbps', 'v2v_packet_size',
       'v2v_packet_rate', 'v2v_distance', 'v2v_delay']
    + _lidar_header('target_lidar', N_TARGET_LIDAR)
    + ['target_cnt', 'target_state', 'target_signal', 'target_enu_x', 'target_enu_y',
       'target_h', 'target_velocity', 'target_ts',
       'target_v2v_ts', 'target_v2v_state', 'target_v2v_signal',
       'target_v2v_enu_x', 'target_v2v_enu_y', 'target_v2v_h', 'target_v2v_velocity',
       'target_v2v_rtt', 'target_v2v_mbps', 'target_v2v_packet_size',
       'target_v2v_packet_rate', 'target_v2v_distance', 'target_v2v_delay',
       'dt_rx_tx_ms']
    + ['ego_lateral_acc', 'ego_longitudinal_acc', 'ego_pitch_rate', 'ego_roll_rate', 'ego_yaw_rate',
       'target_lateral_acc', 'target_longitudinal_acc', 'target_pitch_rate',
       'target_roll_rate', 'target_yaw_rate']
    + ['ego_lc_response_ms', 'target_lc_response_ms',
       'ego_emv_dist_at_emg', 'target_emv_dist_at_emg']
    + ['dr_applied_v2x', 'dr_tau_v2x_ms', 'dr_raw_x_v2x', 'dr_raw_y_v2x', 'dr_cor_x_v2x', 'dr_cor_y_v2x',
       'dr_applied_lidar', 'dr_tau_lidar_ms', 'dr_raw_x_lidar', 'dr_raw_y_lidar', 'dr_cor_x_lidar', 'dr_cor_y_lidar']
)


class MakeDataCarla:
    def __init__(self):
        rospy.init_node('make_data_carla')
        self.set_values()
        self.set_protocols()

    def set_values(self):
        self.ego    = dict(state=0, signal=0, x=0, y=0, h=0, v=0)
        self.target = dict(state=0, signal=0, x=0, y=0, h=0, v=0, ts=0)
        self.ego_obstacles    = []
        self.target_obstacles = []
        self.ego_accel    = [0.0, 0.0]  # [longitudinal_acc, lateral_acc] from /carla/ego/actuator
        self.target_accel = [0.0, 0.0]

        # Perceived target (noisy) published by sharing_info
        self.v2v_target        = dict(state=0, signal=0, v=0, x=0, y=0, ts=0)
        self.target_v2v_target = dict(state=0, signal=0, v=0, x=0, y=0, ts=0)
        self.v2v_cnt    = 0
        self.target_cnt = 0

        # Noise stats window: (monotonic_time, received 0/1, delay_ms)
        self._ego_win    = deque()   # ego's perception of target
        self._target_win = deque()   # target's perception of ego (target_v2v_*)
        self.noise_recv     = 0.0
        self.noise_delay_ms = 0.0
        self.target_noise_recv     = 0.0
        self.target_noise_delay_ms = 0.0

        self._prev_ego_h    = None
        self._prev_target_h = None
        self._prev_ts_sec   = None

        # v2v_bridge channel stats: [dropped(0/1), prr_pct, distance, delay_ms]
        _nan = float('nan')
        self.ch_ego    = {'dropped': True,  'prr': 0.0, 'dist': 0.0, 'delay': _nan}
        self.ch_target = {'dropped': True,  'prr': 0.0, 'dist': 0.0, 'delay': _nan}

        self.scenario   = 0
        self.speed_kmh  = 0
        self.test_mode       = 'same'
        self.sensor_condition = 'c1'
        self.npc_mode = 'nonpc'
        self.with_coop  = True
        self.csv_file   = None
        self.csv_init   = False
        self.dr_mode    = False
        self.dr_log     = [float('nan')] * 12

        _nan = float('nan')
        # CLM: lane-change response time
        self._ego_lc_signal_t   = None   # first time ego signal ∈ [1,2]
        self._target_lc_signal_t = None
        self.ego_lc_response_ms   = _nan
        self.target_lc_response_ms = _nan
        # ETrA: EmV distance at emergency detection
        self.emv_pos = None              # (x, y) from /simulator/object
        self.ego_emv_dist_at_emg    = _nan
        self.target_emv_dist_at_emg = _nan
        self._ego_emg_detected    = False
        self._target_emg_detected = False

    def set_protocols(self):
        rospy.Subscriber('/ego/EgoShareInfo',    ShareInfo,         self.ego_cb)
        rospy.Subscriber('/target/EgoShareInfo', ShareInfo,         self.target_cb)
        rospy.Subscriber('/ego/v2v_target',      Float32MultiArray, self.v2v_target_cb)
        rospy.Subscriber('/target/v2v_target',   Float32MultiArray, self.target_v2v_target_cb)
        rospy.Subscriber('/ego/noise_stats',     Float32MultiArray, self.ego_noise_cb)
        rospy.Subscriber('/target/noise_stats',  Float32MultiArray, self.target_noise_cb)
        rospy.Subscriber('/carla/ego/actuator',    Float32MultiArray, self.ego_actuator_cb)
        rospy.Subscriber('/carla/target/actuator', Float32MultiArray, self.target_actuator_cb)
        rospy.Subscriber('/ego/CommunicationPerformance',    Float32MultiArray, self.ego_comm_cb)
        rospy.Subscriber('/target/CommunicationPerformance', Float32MultiArray, self.target_comm_cb)
        rospy.Subscriber('/ego/user_input', Float32MultiArray, self.user_input_cb)
        rospy.Subscriber('/ego/test_mode',        String, self.test_mode_cb)
        rospy.Subscriber('/ego/with_coop',        String, self.with_coop_cb)
        rospy.Subscriber('/ego/sensor_condition', String, lambda m: setattr(self, 'sensor_condition', m.data))
        rospy.Subscriber('/carla/npc_mode',       String, lambda m: setattr(self, 'npc_mode', m.data))
        rospy.Subscriber('/ego/planner_state',    String, self._ego_planner_state_cb)
        rospy.Subscriber('/target/planner_state', String, self._target_planner_state_cb)
        rospy.Subscriber('/simulator/object',     Pose,   self._emv_cb)
        rospy.Subscriber('/ego/dr_log', Float32MultiArray, self._dr_log_cb)

    # ── callbacks ────────────────────────────────────────────────────────────
    def ego_cb(self, msg: ShareInfo):
        prev_sig = self.ego.get('signal', 0)
        self.ego.update(state=int(msg.state.data), signal=int(msg.signal.data),
                        x=msg.pose.x, y=msg.pose.y, h=msg.pose.theta,
                        v=float(msg.velocity.data))
        self.ego_obstacles = list(msg.obstacles)
        sig = self.ego['signal']
        if prev_sig == 0 and sig in [1, 2] and self._ego_lc_signal_t is None:
            self._ego_lc_signal_t = rospy.get_time()

    def target_cb(self, msg: ShareInfo):
        prev_sig = self.target.get('signal', 0)
        self.target.update(state=int(msg.state.data), signal=int(msg.signal.data),
                           x=msg.pose.x, y=msg.pose.y, h=msg.pose.theta,
                           v=float(msg.velocity.data),
                           ts=rospy.Time.now().to_nsec())
        self.target_obstacles = list(msg.obstacles)
        self.target_cnt += 1
        sig = self.target['signal']
        if prev_sig == 0 and sig in [1, 2] and self._target_lc_signal_t is None:
            self._target_lc_signal_t = rospy.get_time()

    def _emv_cb(self, msg: Pose):
        self.emv_pos = (msg.position.x, msg.position.y)

    def _dr_log_cb(self, msg: Float32MultiArray):
        if len(msg.data) >= 12:
            self.dr_log = list(msg.data[:12])

    def _ego_planner_state_cb(self, msg: String):
        state = msg.data
        if state in ('CHANGE', 'CHANGING') and self._ego_lc_signal_t is not None and math.isnan(self.ego_lc_response_ms):
            self.ego_lc_response_ms = (rospy.get_time() - self._ego_lc_signal_t) * 1000.0
        if state == 'EMERGENCY_CHANGE' and not self._ego_emg_detected and self.emv_pos:
            ex, ey = self.ego['x'], self.ego['y']
            self.ego_emv_dist_at_emg = math.sqrt((ex - self.emv_pos[0])**2 + (ey - self.emv_pos[1])**2)
            self._ego_emg_detected = True

    def _target_planner_state_cb(self, msg: String):
        state = msg.data
        if state in ('CHANGE', 'CHANGING') and self._target_lc_signal_t is not None and math.isnan(self.target_lc_response_ms):
            self.target_lc_response_ms = (rospy.get_time() - self._target_lc_signal_t) * 1000.0
        if state == 'EMERGENCY_CHANGE' and not self._target_emg_detected and self.emv_pos:
            tx, ty = self.target['x'], self.target['y']
            self.target_emv_dist_at_emg = math.sqrt((tx - self.emv_pos[0])**2 + (ty - self.emv_pos[1])**2)
            self._target_emg_detected = True

    def v2v_target_cb(self, msg: Float32MultiArray):
        d = msg.data
        if len(d) >= 5:
            self.v2v_target.update(state=d[0], signal=d[1], v=d[2], x=d[3], y=d[4],
                                   ts=rospy.Time.now().to_nsec())
            self.v2v_cnt += 1

    def target_v2v_target_cb(self, msg: Float32MultiArray):
        d = msg.data
        if len(d) >= 5:
            self.target_v2v_target.update(state=d[0], signal=d[1], v=d[2], x=d[3], y=d[4],
                                          ts=rospy.Time.now().to_nsec())

    def _push_window(self, win, recv, delay_ms):
        now = time.monotonic()
        win.append((now, recv, delay_ms))
        while win and win[0][0] < now - WINDOW_SEC:
            win.popleft()

    def ego_noise_cb(self, msg: Float32MultiArray):
        d = msg.data
        if len(d) < 5:
            return
        self.noise_recv, self.noise_delay_ms = d[0], d[4]
        self._push_window(self._ego_win, d[0], d[4])

    def target_noise_cb(self, msg: Float32MultiArray):
        d = msg.data
        if len(d) < 5:
            return
        self.target_noise_recv, self.target_noise_delay_ms = d[0], d[4]
        self._push_window(self._target_win, d[0], d[4])

    def ego_actuator_cb(self, msg: Float32MultiArray):
        if len(msg.data) >= 2:
            self.ego_accel = [float(msg.data[0]), float(msg.data[1])]

    def target_actuator_cb(self, msg: Float32MultiArray):
        if len(msg.data) >= 2:
            self.target_accel = [float(msg.data[0]), float(msg.data[1])]

    def user_input_cb(self, msg: Float32MultiArray):
        sc = int(msg.data[3]) if len(msg.data) > 3 else 0
        if sc != 0 and not self.csv_init and '_' in self.test_mode:
            self.scenario  = sc
            self.speed_kmh = round(float(msg.data[2]) * 3.6 / 10) * 10 if len(msg.data) > 2 else 0
            self._init_csv()
            self.csv_init  = True

    def test_mode_cb(self, msg: String):
        self.test_mode = msg.data

    def with_coop_cb(self, msg: String):
        self.with_coop = msg.data.lower() == 'true'

    def _parse_comm(self, msg):
        d = msg.data
        if len(d) < 4:
            return
        dropped = d[0] > 0.5
        return {'dropped': dropped, 'prr': d[1], 'dist': d[2],
                'delay': float('nan') if dropped else d[3]}

    def ego_comm_cb(self, msg: Float32MultiArray):
        r = self._parse_comm(msg)
        if r: self.ch_ego = r

    def target_comm_cb(self, msg: Float32MultiArray):
        r = self._parse_comm(msg)
        if r: self.ch_target = r

    # ── metrics ──────────────────────────────────────────────────────────────
    def _perf_metrics(self, win):
        if not win:
            return 0.0, 0.0, 0.0, 0.0, 0.0  # v2x, rtt, mbps, pkt_rate, delay
        total  = len(win)
        recvd  = sum(1 for _, r, _ in win if r > 0.5)
        delays = [d for _, r, d in win if r > 0.5]
        avg_delay = sum(delays) / len(delays) if delays else 0.0
        pkt_rate  = recvd / total * 100.0
        mbps      = recvd * PACKET_SIZE * 8 / WINDOW_SEC / 1e6
        v2x       = 1.0 if win[-1][1] > 0.5 else 0.0
        rtt       = avg_delay * 2.0
        return v2x, rtt, mbps, pkt_rate, avg_delay

    def _distance(self):
        return math.sqrt((self.ego['x'] - self.target['x'])**2 +
                         (self.ego['y'] - self.target['y'])**2)

    # ── csv ──────────────────────────────────────────────────────────────────
    def _init_csv(self):
        sc_name = SCENARIO_MAP.get(self.scenario, f'S{self.scenario}')
        mode    = 'WC' if self.test_mode.startswith('WC') else 'WOC'
        speed   = self.test_mode.split('_')[-1]  # same/slower/faster
        ts      = datetime.now().strftime('%m%d%H%M%S')
        subdir  = 'treatment' if self.dr_mode else 'Carla'
        suffix  = '_dr_both' if self.dr_mode else ''
        out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', subdir)
        os.makedirs(out_dir, exist_ok=True)
        self.csv_file = os.path.join(
            out_dir,
            f'carla_{sc_name}_{mode}_{speed}_{self.sensor_condition}_{self.npc_mode}_{self.speed_kmh}kmh{suffix}_{ts}.csv')
        with open(self.csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(HEADER)
        rospy.loginfo(f'[make_data_carla] logging → {self.csv_file}')

    def _write(self):
        if not self.csv_init:
            return
        ts_ns = rospy.Time.now().to_nsec()
        dist  = self._distance()

        def _obs_row(obs, n):
            row = []
            for i in range(n):
                if i < len(obs):
                    o = obs[i]
                    row += [o.pose.x, o.pose.y, o.pose.theta,
                            float(o.velocity.data), int(o.dangerous.data),
                            float(o.lidar_delay.data)]
                else:
                    row += ['', '', '', '', '', '']
            return row

        lidar_row        = _obs_row(self.ego_obstacles,    N_LIDAR)
        target_lidar_row = _obs_row(self.target_obstacles, N_TARGET_LIDAR)

        # ego V2V performance — channel stats from v2v_bridge
        v2x, _, mbps, _, _ = self._perf_metrics(self._ego_win)
        pkt_rate = self.ch_ego['prr']
        delay    = self.ch_ego['delay']   # NaN if dropped
        rtt      = delay * 2.0

        _nan = float('nan')
        if self.ch_ego.get('dropped', False):
            v2v_state = v2v_signal = v2v_x = v2v_y = v2v_v = _nan
        else:
            v2v_state  = self.v2v_target['state']
            v2v_signal = self.v2v_target['signal']
            v2v_x      = self.v2v_target['x']
            v2v_y      = self.v2v_target['y']
            v2v_v      = self.v2v_target['v']

        v2v_row = [self.v2v_cnt,
                   v2v_state, v2v_signal,
                   v2v_x, v2v_y,
                   self.target['h'],
                   v2v_v,
                   0,
                   v2x, rtt, mbps, PACKET_SIZE, pkt_rate, dist, delay]

        # target ground truth
        t = self.target
        target_row = [self.target_cnt, t['state'], t['signal'],
                      t['x'], t['y'], t['h'], t['v'], t['ts']]

        # target_v2v (target's noisy perception of ego via V2X)
        tv2x, trtt, tmbps, _, _ = self._perf_metrics(self._target_win)
        tpkt_rate = self.ch_target['prr']
        tdelay    = self.ch_target['delay']
        tv = self.target_v2v_target
        if self.ch_target.get('dropped', False):
            tv_state = tv_signal = tv_x = tv_y = tv_v = _nan
        else:
            tv_state, tv_signal = tv['state'], tv['signal']
            tv_x, tv_y, tv_v    = tv['x'], tv['y'], tv['v']
        target_v2v_row = [tv['ts'],
                          tv_state, tv_signal,
                          tv_x, tv_y, self.ego['h'], tv_v,
                          trtt, tmbps, PACKET_SIZE, tpkt_rate, dist, tdelay]

        dt_rx_tx = delay

        # yaw_rate (deg/s): heading 변화율, 나머지 pitch/roll은 평지라 0
        ts_sec = ts_ns / 1e9
        def _yaw_rate(prev_h, cur_h, dt):
            if prev_h is None or dt <= 0: return 0.0
            dh = (cur_h - prev_h + 180) % 360 - 180  # wraparound 처리
            return dh / dt

        dt = (ts_sec - self._prev_ts_sec) if self._prev_ts_sec else 0.0
        ego_yr    = _yaw_rate(self._prev_ego_h,    self.ego['h'],    dt)
        target_yr = _yaw_rate(self._prev_target_h, self.target['h'], dt)

        self._prev_ego_h    = self.ego['h']
        self._prev_target_h = self.target['h']
        self._prev_ts_sec   = ts_sec

        ego_imu    = [self.ego_accel[0],    self.ego_accel[1],    0, 0, ego_yr]
        target_imu = [self.target_accel[0], self.target_accel[1], 0, 0, target_yr]

        row = ([ts_ns,
                self.ego['state'], self.ego['signal'],
                self.ego['x'], self.ego['y'], self.ego['h'], self.ego['v']]
               + lidar_row + v2v_row + target_lidar_row + target_row + target_v2v_row
               + [dt_rx_tx] + ego_imu + target_imu
               + [self.ego_lc_response_ms, self.target_lc_response_ms,
                  self.ego_emv_dist_at_emg, self.target_emv_dist_at_emg]
               + self.dr_log)

        if self.csv_file is None:
            return
        with open(self.csv_file, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    def execute(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self._write()
            rate.sleep()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--npc', default=None, choices=['npc', 'nonpc'])
    ap.add_argument('--dr',  action='store_true', help='treatment mode: log to data/treatment/')
    args, _ = ap.parse_known_args()
    m = MakeDataCarla()
    if args.npc:
        m.npc_mode = args.npc
    if args.dr:
        m.dr_mode = True
    m.execute()
