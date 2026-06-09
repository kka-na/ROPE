"""
carla_ros_bridge.py — CARLA Actor ↔ ROS 인터페이스

역할:
  1. CARLA 서버에 연결, 시나리오별 actor(EV/TV/EmV) spawn
  2. 20Hz 루프:
     - CARLA actor 위치/속도 → /{type}/simulator/inform (Quaternion) 발행
       (sharing_info/ros_manager.py의 simulator_inform_cb가 소비)
     - EmV 위치 → /simulator/object (Pose) 발행
       (sharing_info/ros_manager.py의 simulator_object_cb가 소비)
     - /carla/{type}/actuator 구독 → actor.apply_control()

좌표 변환: CARLA 좌손계(Y반전) ↔ 기존 ROS 우손계
  ros_x = carla_x,  ros_y = -carla_y,  ros_yaw_deg = -carla_yaw_deg
"""
import os
import sys
import math
import argparse
import rospy
import carla
from geometry_msgs.msg import Quaternion, Pose
from std_msgs.msg import Float32MultiArray, Bool
try:
    from ccavt.msg import ShareInfo
    _HAS_SHAREINFO = True
except ImportError:
    _HAS_SHAREINFO = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenarios.clm_scenario  import CLMScenario
from scenarios.etra_scenario import ETrAScenario


C2R_Y   = -1.0   # CARLA Y → ROS Y 부호
C2R_YAW = -1.0   # CARLA yaw → ROS yaw 부호

# 차량별 throttle 범위: accel_norm [0,1] → [thr_min, thr_max]
_VEHICLE = {
    'ego':    {'thr_min': 0.72, 'thr_max': 1.0},  # audi.etron
    'target': {'thr_min': 0.5,  'thr_max': 1.0},  # seat.leon
}



class CarlaRosBridge:
    def __init__(self, scenario_id: str, test_mode: str, seed: int = 42, speed_kmh: int = 30, npc: bool = False):
        # CARLA 연결
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(60.0)
        print('[bridge] connecting to CARLA...')
        if 'Town06' not in self.client.get_world().get_map().name:
            print('[bridge] loading Town06...')
            self.world = self.client.load_world('Town06')
        else:
            self.world = self.client.get_world()
        print(f'[bridge] map: {self.world.get_map().name}')

        # 동기 모드 설정
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05   # 20Hz physics
        self.world.apply_settings(settings)

        # 이전 실행에서 남은 actor 정리
        print('[bridge] cleaning up previous actors...')
        for actor in self.world.get_actors().filter('vehicle.*'):
            actor.destroy()
        for actor in self.world.get_actors().filter('sensor.*'):
            actor.destroy()
        self.world.tick()

        # 시나리오 선택 및 actor spawn
        is_etra = scenario_id.startswith('ETrA')
        ScenarioCls = ETrAScenario if is_etra else CLMScenario

        print(f'[bridge] spawning scenario {scenario_id} at {speed_kmh} km/h...')
        self._seed = seed
        self.scenario = ScenarioCls(scenario_id, test_mode,
                                    world=self.world, seed=seed, speed_kmh=speed_kmh)
        actors = self.scenario.setup(npc=npc)
        print(f'[bridge] spawned {len(actors)} actors')

        self.ego_actor = actors[0]
        self.tv_actor  = actors[1]
        self.emv_actor = actors[2] if len(actors) > 2 else None

        # 스폰 직후 물리 비활성화 → 미끄러짐 방지 (spin에서 활성화)
        self.ego_actor.set_simulate_physics(False)
        self.tv_actor.set_simulate_physics(False)

        # actuator 수신 (selfdriving/transmitter/carla_transmitter.py에서 발행)
        self._ego_ctrl  = carla.VehicleControl()
        self._tv_ctrl   = carla.VehicleControl()
        self._ego_accel = 0.0
        self._tv_accel  = 0.0
        self._ego_tgt_v = 0.0
        self._tv_tgt_v  = 0.0
        self._prev_state    = 0
        self._init_v        = None
        self._init_ticks    = 0
        self._pending_reset = None
        self._pending_set   = False
        rospy.Subscriber('/carla/ego/actuator',    Float32MultiArray, self._ego_ctrl_cb)
        rospy.Subscriber('/carla/target/actuator', Float32MultiArray, self._tv_ctrl_cb)
        if _HAS_SHAREINFO:
            from ccavt.msg import ShareInfo
            rospy.Subscriber('/ego/EgoShareInfo',    ShareInfo, lambda m: setattr(self, '_ego_tgt_v', m.target_velocity.data * 3.6))
            rospy.Subscriber('/target/EgoShareInfo', ShareInfo, lambda m: setattr(self, '_tv_tgt_v',  m.target_velocity.data * 3.6))
        rospy.Subscriber('/carla/reset',           Float32MultiArray, self._reset_cb)
        rospy.Subscriber('/carla/set',             Float32MultiArray, self._set_cb)
        rospy.Subscriber('/ego/user_input',        Float32MultiArray, self._user_input_cb)

        # 위치 발행
        self._pub_ego  = rospy.Publisher('/ego/simulator/inform',    Quaternion,       queue_size=1)
        self._pub_tv   = rospy.Publisher('/target/simulator/inform', Quaternion,       queue_size=1)
        self._pub_emv  = rospy.Publisher('/simulator/object',        Pose,             queue_size=1)
        self._pub_npc  = rospy.Publisher('/carla/npc_info',          Float32MultiArray, queue_size=1)
        self._pub_ready = rospy.Publisher('/carla/ready', Bool, queue_size=1)

        self._npc_on    = npc
        self.npc_actors = getattr(self.scenario, 'npc_actors', []) if npc else []
        self._npc_t0    = None

        # 경로 시각화용 (EgoShareInfo 구독 → CARLA debug draw)
        self._ego_path, self._tv_path = [], []
        if _HAS_SHAREINFO:
            rospy.Subscriber('/ego/EgoShareInfo',    ShareInfo, lambda m: self._path_cb(m, 'ego'))
            rospy.Subscriber('/target/EgoShareInfo', ShareInfo, lambda m: self._path_cb(m, 'tv'))

    def _path_cb(self, msg, who):
        pts = [(p.pose.x, p.pose.y) for p in msg.paths]
        if who == 'ego':
            self._ego_path = pts
        else:
            self._tv_path = pts

    def _draw_paths(self):
        try:
            dbg = self.world.debug
            for rx, ry in self._ego_path[::3]:
                dbg.draw_point(carla.Location(rx, -ry, 0.3), 0.08,
                               carla.Color(255, 50, 50), 0.12)
            for rx, ry in self._tv_path[::3]:
                dbg.draw_point(carla.Location(rx, -ry, 0.3), 0.08,
                               carla.Color(50, 50, 255), 0.12)
        except Exception:
            pass

    # ── reset 콜백 ────────────────────────────────────────────────────
    _SCENARIO_ID = {1:'CLM1',2:'CLM2',3:'CLM3',4:'CLM4',7:'ETrA1',8:'ETrA2',9:'ETrA3',10:'ETrA4'}
    _TEST_MODE   = {0:'same', 1:'slower', 2:'faster'}

    def _reset_cb(self, msg):
        self._pending_reset = msg  # CARLA API는 spin() 메인 스레드에서만 호출

    def _do_reset(self, msg):
        d = list(msg.data)
        if len(d) >= 4:
            sid       = self._SCENARIO_ID.get(int(d[0]), 'CLM2')
            speed_kmh = int(d[1])
            self._npc_on = bool(d[2])
            test_mode = self._TEST_MODE.get(int(d[3]), 'same')
            is_etra   = sid.startswith('ETrA')
            self.scenario.cleanup()
            self.scenario = (ETrAScenario if is_etra else CLMScenario)(
                sid, test_mode, world=self.world, seed=self._seed, speed_kmh=speed_kmh)
            print(f'[bridge] reset: {sid} {test_mode} {speed_kmh}km/h npc={self._npc_on}')
        else:
            print('[bridge] reset: respawning same scenario...')
            self.scenario.cleanup()
        self._ego_ctrl = carla.VehicleControl()
        self._tv_ctrl  = carla.VehicleControl()
        actors = self.scenario.setup(npc=self._npc_on)
        self.ego_actor = actors[0]
        self.tv_actor  = actors[1]
        self.emv_actor = actors[2] if len(actors) > 2 else None
        self.ego_actor.set_simulate_physics(False)
        self.tv_actor.set_simulate_physics(False)
        self.npc_actors = getattr(self.scenario, 'npc_actors', []) if self._npc_on else []
        self._npc_t0    = None
        self._update_spectator()
        print('[bridge] reset done — press Set to enable physics')
        self._pub_ready.publish(Bool(data=True))

    def _set_cb(self, msg):
        self._pending_set = True  # CARLA API는 spin() 메인 스레드에서만 호출

    def _user_input_cb(self, msg):
        state = int(msg.data[0])
        if self._prev_state == 0 and state == 1:
            self._init_v     = float(msg.data[2])
            self._init_ticks = 20   # 1s @ 20Hz
            self._npc_t0     = rospy.get_time()
            print(f'[bridge] start: will set initial velocity {self._init_v*3.6:.1f} km/h for {self._init_ticks} ticks')
        self._prev_state = state

    # ── actuator 콜백 ──────────────────────────────────────────────────
    def _ego_ctrl_cb(self, msg):
        self._ego_accel = float(msg.data[0])
        self._ego_ctrl  = self._to_carla_ctrl(msg.data[0], msg.data[1], _VEHICLE['ego'])

    def _tv_ctrl_cb(self, msg):
        self._tv_accel = float(msg.data[0])
        self._tv_ctrl  = self._to_carla_ctrl(msg.data[0], msg.data[1], _VEHICLE['target'])

    def _to_carla_ctrl(self, accel_norm: float, steer_rad: float, vp: dict) -> carla.VehicleControl:
        ctrl = carla.VehicleControl()
        if accel_norm >= 0:
            ctrl.throttle = vp['thr_min'] + float(accel_norm) * (vp['thr_max'] - vp['thr_min'])
            ctrl.brake    = 0.0
        else:
            ctrl.throttle = 0.0
            ctrl.brake    = min(-float(accel_norm), 1.0)
        ctrl.steer = max(-1.0, min(1.0, -steer_rad / math.radians(35)))
        return ctrl

    # ── 위치 발행 ─────────────────────────────────────────────────────
    def _actor_to_quaternion(self, actor: carla.Actor) -> Quaternion:
        """CARLA actor → Quaternion(x=ros_x, y=ros_y, z=speed, w=yaw_deg)"""
        tf  = actor.get_transform()
        vel = actor.get_velocity()
        msg = Quaternion()
        msg.x = tf.location.x
        msg.y = tf.location.y * C2R_Y
        msg.z = math.sqrt(vel.x**2 + vel.y**2)
        msg.w = tf.rotation.yaw * C2R_YAW
        return msg

    def _emv_to_pose(self, actor: carla.Actor) -> Pose:
        tf  = actor.get_transform()
        vel = actor.get_velocity()
        msg = Pose()
        msg.position.x    = tf.location.x
        msg.position.y    = tf.location.y * C2R_Y
        msg.orientation.y = math.atan2(vel.y * C2R_Y, vel.x)  # heading (rad)
        return msg

    # ── 메인 루프 ─────────────────────────────────────────────────────
    def _update_spectator(self):
        # 뒤에 있는 차량 추적 (동쪽 진행 → x가 작을수록 후방)
        ego_loc = self.ego_actor.get_transform().location
        tv_loc  = self.tv_actor.get_transform().location
        target  = ego_loc if ego_loc.x < tv_loc.x else tv_loc

        desired = carla.Location(x=target.x - 20, y=target.y, z=target.z + 15)
        spec = self.world.get_spectator()
        cur  = spec.get_transform().location

        alpha = 0.1
        smooth = carla.Location(
            x=cur.x + alpha * (desired.x - cur.x),
            y=cur.y + alpha * (desired.y - cur.y),
            z=cur.z + alpha * (desired.z - cur.z),
        )
        spec.set_transform(carla.Transform(smooth, carla.Rotation(pitch=-30, yaw=0)))

    def _enable_physics(self):
        self.ego_actor.set_simulate_physics(True)
        self.tv_actor.set_simulate_physics(True)
        print('[bridge] physics enabled')

    def _tick_npcs(self):
        if self._npc_t0 is None or not self.npc_actors:
            return
        dt   = rospy.get_time() - self._npc_t0
        data = []
        for npc in self.npc_actors:
            actor = npc['actor']
            if not actor.is_alive:
                continue
            new_x = npc['x0'] + npc['speed'] * dt
            y     = npc['y']
            loc   = carla.Location(x=new_x, y=y, z=0)
            try:
                wp  = self.world.get_map().get_waypoint(loc)
                loc = carla.Location(x=new_x, y=y, z=wp.transform.location.z + 0.5)
            except Exception:
                loc.z = 0.5
            actor.set_transform(carla.Transform(loc, carla.Rotation(yaw=0.0)))
            data += [new_x, y * C2R_Y, 0.0, npc['speed']]   # x, ros_y, heading_ros, speed
        if data:
            self._pub_npc.publish(Float32MultiArray(data=data))

    def spin(self):
        print('[bridge] spinning at 20Hz — press Set to enable physics...')
        self._update_spectator()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self._pending_reset is not None:
                msg, self._pending_reset = self._pending_reset, None
                self._do_reset(msg)
                rate.sleep()
                continue
            if self._pending_set:
                self._pending_set = False
                self._enable_physics()
                self._prev_state = 0
                self._init_v     = None
                self._init_ticks = 0
                print('[bridge] set: physics enabled')
                rate.sleep()
                continue
            if self._init_v is not None and self._init_ticks > 0:
                tv_v = self._init_v * getattr(self.scenario, 'speed_ratio', 1.0)
                self.scenario.set_initial_velocity(self.ego_actor, self._init_v)
                self.scenario.set_initial_velocity(self.tv_actor,  tv_v)
                self._init_ticks -= 1
                ego_neutral = carla.VehicleControl(throttle=self._ego_ctrl.throttle, steer=self._ego_ctrl.steer, brake=0)
                tv_neutral  = carla.VehicleControl(throttle=self._tv_ctrl.throttle,  steer=self._tv_ctrl.steer,  brake=0)
                self.ego_actor.apply_control(ego_neutral)
                self.tv_actor.apply_control(tv_neutral)
                if self._init_ticks == 0:
                    print(f'[bridge] initial velocity applied: ego={self._init_v*3.6:.1f} tv={tv_v*3.6:.1f} km/h')
                    self._init_v = None
            else:
                self.ego_actor.apply_control(self._ego_ctrl)
                self.tv_actor.apply_control(self._tv_ctrl)
            self.world.tick()
            eq = self._actor_to_quaternion(self.ego_actor)
            tq = self._actor_to_quaternion(self.tv_actor)
            self._pub_ego.publish(eq)
            self._pub_tv.publish(tq)
            if self.emv_actor:
                self._pub_emv.publish(self._emv_to_pose(self.emv_actor))
            self._tick_npcs()
            self._draw_paths()
            self._update_spectator()
            rate.sleep()

    def cleanup(self):
        self.scenario.cleanup()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario',  default='CLM2')
    parser.add_argument('--test_mode', default='same',
                        choices=['slower', 'same', 'faster'])
    parser.add_argument('--seed',  type=int, default=42)
    parser.add_argument('--speed', type=int, default=30)
    parser.add_argument('--npc',   action='store_true', help='spawn ghost NPC vehicles')
    args, _ = parser.parse_known_args()

    print('[bridge] init_node...')
    rospy.init_node('carla_ros_bridge')
    print('[bridge] init_node done')
    bridge = CarlaRosBridge(args.scenario, args.test_mode, args.seed, args.speed, npc=args.npc)
    rospy.on_shutdown(bridge.cleanup)

    try:
        bridge.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
