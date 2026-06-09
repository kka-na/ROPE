#!/usr/bin/env python
import rospy
import math
import tf
import random
import os
import numpy as np
from collections import deque

from pyproj import Proj, Transformer
from threading import Timer

from ccavt.msg import *
from novatel_oem7_msgs.msg import INSPVA
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Pose, Point
from jsk_recognition_msgs.msg import BoundingBoxArray
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Float32MultiArray,Bool, String
from dr_compensator import DRCompensator


class ROSManager:
    def __init__(self, type,test,  map, oh, lpp):
        rospy.init_node(f'{type}_share_info')
        self.type = type
        self.test = test
        self.map = map
        self.oh = oh
        self.lpp = lpp
        self.set_values()
        self.set_protocol()
    
    def set_values(self):
        self.car = {'fix': 'No','x': 0, 'y': 0, 't': 0, 'v': 0}
        self.user_input = {'state': 0, 'signal': 0, 'target_velocity': 0, 'scenario':0, 'mode':0, 'with_coop': True}
        self.lidar_obstacles = []
        self.dangerous_obstacle = []
        self.obstacle_caution = False
        self.target_info = [0,0,0, 0, 0]
        self.target_info_v2v = [0,0,0, 0, 0]  # [state, signal, velocity, x, y, heading]
        self.target_gt_v = 0.0  # CARLA ground truth target velocity (noise 적용 전)
        self._v2x_delay_ms = 0.0
        self._dr_log = [0.0] * 12
        self.dr = DRCompensator()  # disabled by default; main.py --dr-enabled activates
        self.target_path = []
        self.target_velocity = 0
        self.test_mode = 'same'  # test_mode 저장

        # Auto signal 관련 변수
        self.auto_signal_sent = False
        self.speed_reached_time = None
        self.scenario_done = False

        # Endpoint 관련 변수
        self.endpoints = self.load_endpoints()
        self.endpoint_threshold = 3.0  # 3m 이내면 도착으로 간주

        # Emergency 관련 변수 추가
        self.emergency_active = False
        self.emergency_cooldown = False
        self.emergency_timer = None
        self.cooldown_timer = None
        self.emergency_duration = 5.0  # 5초 동안 발행
        self.cooldown_duration = 20.0  # 50초 동안 발행 금지

        # LiDAR 조건 (sensor_condition 토픽으로 설정)
        self.lidar_preset = 'baseline'
        self.sensor_condition = 'c1'
        self.npc_obstacles = []
        self._obu_delay_buf = deque()    # (release_time, noisy_data)
        self._lidar_delay_buf = deque()  # (release_time, obstacle_list)
        self._ge_state = {'baseline': 'GOOD', 'degraded': 'GOOD'}

        proj_wgs84 = Proj(proj='latlong', datum='WGS84') 
        proj_enu = Proj(proj='aeqd', datum='WGS84', lat_0=self.map.base_lla[0], lon_0=self.map.base_lla[1], h_0=self.map.base_lla[2])
        self.geo2enu_transformer = Transformer.from_proj(proj_wgs84, proj_enu)
        self.enu2geo_transformter = Transformer.from_proj(proj_enu, proj_wgs84)



        rospy.loginfo("Sharing Info set")

    def set_protocol(self):
        # ── publishers 먼저 정의 (latch 토픽 콜백보다 앞서야 함) ──────────
        self.pub_emergency_user_input = rospy.Publisher(f'/{self.type}/user_input', Float32MultiArray, queue_size=1)
        self.pub_ego_share_info       = rospy.Publisher(f'/{self.type}/EgoShareInfo', ShareInfo, queue_size=1)
        self.pub_dangerous_obstacle   = rospy.Publisher(f'/{self.type}/dangerous_obstacle', Float32MultiArray, queue_size=1)
        self.pub_obs_caution          = rospy.Publisher(f'{self.type}/obs_caution', Bool, queue_size=1)
        self.pub_planner_state        = rospy.Publisher(f'/{self.type}/planner_state', String, queue_size=1)
        self.pub_lmap_viz             = rospy.Publisher('/lmap_viz', MarkerArray, queue_size=10, latch=True)
        self.pub_inter_pt             = rospy.Publisher(f'/{self.type}/inter_pt', Marker, queue_size=10)
        self.pub_bsd_zone             = rospy.Publisher(f'/{self.type}/bsd_zone', Marker, queue_size=10)
        self.pub_endpoints            = rospy.Publisher('/endpoints', Marker, queue_size=10, latch=True)
        self.pub_endpoint_reached     = rospy.Publisher(f'/{self.type}/endpoint_reached', Bool, queue_size=1)
        self.pub_scenario_done        = rospy.Publisher('/scenario/done', Bool, queue_size=1, latch=True)
        self.pub_noise_stats          = rospy.Publisher(f'/{self.type}/noise_stats', Float32MultiArray, queue_size=1)
        self.pub_v2v_target           = rospy.Publisher(f'/{self.type}/v2v_target',  Float32MultiArray, queue_size=1)
        self.pub_dr_log               = rospy.Publisher(f'/{self.type}/dr_log',       Float32MultiArray, queue_size=1)

        # ── subscribers ───────────────────────────────────────────────────
        if self.test == 1:
            if self.type == 'ego':
                rospy.Subscriber('/target/EgoShareInfo', ShareInfo, self.target_share_info_cb)
            else:
                rospy.Subscriber('/ego/EgoShareInfo', ShareInfo, self.target_share_info_cb)
        else:
            rospy.Subscriber(f'/{self.type}/TargetShareInfo', ShareInfo, self.target_share_info_cb)

        rospy.Subscriber('/novatel/oem7/inspva', INSPVA, self.novatel_inspva_cb)
        rospy.Subscriber('/novatel/oem7/odom', Odometry, self.novatel_odom_cb)
        rospy.Subscriber('/mobinha/perception/lidar/track_box', BoundingBoxArray, self.lidar_cluster_cb)
        rospy.Subscriber('/simulator/object', Pose, self.simulator_object_cb)
        rospy.Subscriber(f'/{self.type}/user_input', Float32MultiArray, self.user_input_cb)
        rospy.Subscriber(f'/{self.type}/simulator/inform', Quaternion, self.simulator_inform_cb)
        rospy.Subscriber(f'/{self.type}/with_coop',        String, self.with_coop_cb)
        rospy.Subscriber(f'/{self.type}/test_mode',        String, self.test_mode_cb)
        rospy.Subscriber(f'/{self.type}/sensor_condition', String, self._sensor_condition_cb)
        rospy.Subscriber('/carla/reset', Float32MultiArray, self.carla_reset_cb)
        rospy.Subscriber('/scenario/done', Bool, lambda m: setattr(self, 'scenario_done', m.data))
        rospy.Subscriber(f'/{self.type}/CommunicationPerformance', Float32MultiArray, self._comm_perf_cb)
        if self.test == 1:
            rospy.Subscriber('/carla/npc_info', Float32MultiArray, self._npc_info_cb)

        self.pub_lmap_viz.publish(self.map.lmap_viz)
        self.publish_endpoints()

    def novatel_inspva_cb(self, msg):
        self.car['fix'] = 'Ok'
        e,n,u =  self.geo2enu_transformer.transform(msg.longitude, msg.latitude, 5)
        self.car['x'] = e
        self.car['y'] = n
        self.car['t'] = 88.5-msg.azimuth
    
    def novatel_odom_cb(self, msg): 
        self.car['v'] = msg.twist.twist.linear.x

    def simulator_inform_cb(self, msg):
        self.car['fix'] = 'Ok'
        self.car['x'] = msg.x
        self.car['y'] = msg.y
        self.car['t'] = msg.w
        self.car['v'] = msg.z

    def target_share_info_cb(self, msg: ShareInfo):
        if self.test == 1:
            self.target_gt_v = float(msg.velocity.data)  # ground truth — 노이즈 전

        # WOC 시뮬: Gilbert-Elliott LiDAR 노이즈
        if self.test == 1 and not self.user_input.get('with_coop', True):
            self.process_simulated_lidar(msg)
            return

        # WC 시뮬: Gaussian 측정 노이즈만 적용 (드롭·딜레이는 v2v_bridge 담당)
        if self.test == 1:
            paths = [[pt.pose.x, pt.pose.y] for pt in msg.paths]
            nx, ny, nh, nv = self._obu_noise(msg.pose.x, msg.pose.y,
                                              msg.pose.theta, float(msg.velocity.data))
            self.pub_noise_stats.publish(Float32MultiArray(data=[
                1.0, nx - msg.pose.x, ny - msg.pose.y, nv - float(msg.velocity.data), 0.0
            ]))
            self.target_info_v2v = [int(msg.state.data), int(msg.signal.data), nv, nx, ny, nh]
            self.target_path = paths
            self.update_target_info()
            return

        # 실차: 노이즈 없이 직접 저장
        self.target_info_v2v = [int(msg.state.data), int(msg.signal.data),
                                 float(msg.velocity.data), msg.pose.x, msg.pose.y, msg.pose.theta]
        self.target_path = [[pt.pose.x, pt.pose.y] for pt in msg.paths]
        self.update_target_info()

    def user_input_cb(self, msg):
        mode = int(msg.data[4]) if len(msg.data) > 4 else 0
        self.user_input['mode'] = mode
        if not self.emergency_active:
            self.user_input['state']  = int(msg.data[0])
            self.user_input['signal'] = int(msg.data[1])
            self.user_input['target_velocity'] = float(msg.data[2])
            self.user_input['scenario'] = int(msg.data[3])
        else:
            self.user_input['state']  = int(msg.data[0])
            self.user_input['target_velocity'] = float(msg.data[2])
            self.user_input['scenario'] = int(msg.data[3])
    
    def lidar_cluster_cb(self, msg):
        obstacles = []
        dangerous_id = 99999

        for i, obj in enumerate(msg.boxes):
            if int(obj.header.seq) < 3:
                continue
            enu = self.oh.object2enu([obj.pose.position.x, obj.pose.position.y])
            if enu is None:
                continue
            else:
                nx, ny = enu
            lidar_delay = obj.pose.orientation.x
            quaternion = (
                0, 
                obj.pose.orientation.y, 
                obj.pose.orientation.z, 
                obj.pose.orientation.w
            )
            _, _, heading = tf.transformations.euler_from_quaternion(quaternion)
            
            distance = self.oh.distance(self.car['x'], self.car['y'], nx, ny)
            v_rel = ( obj.value if obj.value != 0 else 0 ) + self.car['v']

            obstacles.append([i, nx, ny, heading, v_rel, distance,lidar_delay])
            
        
    
        filtered_obstacles = self.lpp.phelper.filtering_by_lanelet_from_list(obstacles)
        filtered_obstacles, dangerous_id = self.oh.get_frenets_from_list(filtered_obstacles)
        
        lidar_obstacles = []
        for i, obs in enumerate(filtered_obstacles):
            if obs[0] == dangerous_id:
                dangerous = 1
                self.dangerous_obstacle = obs
            else:
                dangerous = 0
            obstacle = Obstacle()
            obstacle.cls.data = 0
            obstacle.id.data = i
            obstacle.pose.x = obs[1]
            obstacle.pose.y = obs[2]
            obstacle.pose.theta = obs[3]
            obstacle.velocity.data = obs[4]
            obstacle.distance.data = obs[5]
            obstacle.dangerous.data = dangerous
            obstacle.lidar_delay.data = obs[6]
            lidar_obstacles.append(obstacle)

        self.lidar_obstacles = lidar_obstacles
        # Update target_info if in without cooperation mode
        self.update_target_info()

    def simulator_object_cb(self, msg):
        obstacles = []
        dangerous_id = 99999
        min_s = 100
        nx = msg.position.x
        ny = msg.position.y
        heading = self.oh.refine_heading_by_lane([nx, ny])
        if heading is None:
            heading = math.degrees(msg.orientation.y)
        distance = self.oh.distance(self.car['x'], self.car['y'], nx, ny)
        v_rel = self.car['v']
        frenet = self.oh.object2frenet((nx, ny))
        if frenet is None:
            return
        else:
            s, d = frenet

        obstacles.append([0, nx, ny, heading, v_rel, distance])
        if 1 < s < 100:
            if s < min_s and -2 < d < 2 :
                dangerous_id = 0
                min_s = s
        
        lidar_obstacles = []
        dangerous_obstacle = []
        for i, obs in enumerate(obstacles):
            if obs[0] == dangerous_id:
                dangerous = 1
                dangerous_obstacle = obs
            else:
                dangerous = 0
            obstacle = Obstacle()
            obstacle.cls.data = 0
            obstacle.id.data = 0
            obstacle.pose.x = obs[1]
            obstacle.pose.y = obs[2]
            obstacle.pose.theta = obs[3]
            obstacle.velocity.data = obs[4]
            obstacle.distance.data = obs[5]
            obstacle.dangerous.data = dangerous
            obstacle.lidar_delay.data = 0.0
            lidar_obstacles.append(obstacle)

        self.lidar_obstacles = lidar_obstacles
        self.dangerous_obstacle = dangerous_obstacle
        # Update target_info if in without cooperation mode
        self.update_target_info()

    def with_coop_cb(self, msg):
        """Callback for with_coop topic"""
        self.user_input['with_coop'] = (msg.data == 'true')
        # Update target_info when with_coop flag changes
        self.update_target_info()

    def test_mode_cb(self, msg):
        self.test_mode = msg.data
        self.user_input['test_mode'] = msg.data
        if hasattr(self, 'lpp') and self.lpp:
            self.lpp.test_mode = self.test_mode

    def get_closest_lidar_obstacle(self):
        all_obs = (self.lidar_obstacles or []) + self.npc_obstacles
        if not all_obs:
            return None
        return min(all_obs, key=lambda o: o.distance.data)

    def _npc_info_cb(self, msg):
        d = list(msg.data)
        n = len(d) // 4
        p = self._LIDAR_PRESETS[self.lidar_preset]
        obstacles = []
        for i in range(n):
            x, y, h, v = d[i*4], d[i*4+1], d[i*4+2], d[i*4+3]
            dist = math.sqrt((x - self.car['x'])**2 + (y - self.car['y'])**2)
            if dist > p['range'] or not self._detect():
                continue
            obs = Obstacle()
            obs.cls.data      = 0
            obs.id.data       = i + 10
            obs.pose.x        = x + p['bias_x'] + np.random.normal(0, p['std_x'])
            obs.pose.y        = y + p['bias_y'] + np.random.normal(0, p['std_y'])
            obs.pose.theta    = h + p['bias_h'] + np.random.normal(0, p['std_h'])
            obs.velocity.data = max(0.0, v + p['bias_v'] + np.random.normal(0, p['std_v']))
            obs.distance.data = dist
            obs.dangerous.data = 0
            delay_ms = max(0.0, random.gauss(p['mu_delay'], p['sigma_delay']))
            obs.lidar_delay.data = delay_ms / 1000.0
            obstacles.append(obs)
        self.npc_obstacles = obstacles

    def build_target_info_from_lidar(self):
        """Build target_info structure from lidar obstacle data"""
        closest_obs = self.get_closest_lidar_obstacle()

        if closest_obs is None:
            # No lidar obstacles, return default
            return [0, 0, 0, 0, 0]

        # Build target_info: [state, signal, velocity, x, y]
        # For lidar-based: state=0, signal=0 (no signal info from lidar)
        target_info = [
            0,  # state (no state info from lidar)
            0,  # signal (no signal info from lidar)
            float(closest_obs.velocity.data),  # velocity from lidar
            float(closest_obs.pose.x),  # x position
            float(closest_obs.pose.y)   # y position
        ]

        return target_info

    def process_simulated_lidar(self, msg):
        """시뮬 LiDAR: field 실측 기반 검출률·노이즈 모델."""
        if not self._detect():
            self.lidar_obstacles = []
            self.pub_noise_stats.publish(Float32MultiArray(data=[0.0, 0.0, 0.0, 0.0, 0.0]))
            return

        tx, ty = msg.pose.x, msg.pose.y
        distance = math.sqrt((tx - self.car['x'])**2 + (ty - self.car['y'])**2)
        p = self._LIDAR_PRESETS[self.lidar_preset]
        if distance > p['range']:
            self.lidar_obstacles = []
            self.pub_noise_stats.publish(Float32MultiArray(data=[0.0, 0.0, 0.0, 0.0, 0.0]))
            return
        nx = tx + p['bias_x'] + np.random.normal(0, p['std_x'])
        ny = ty + p['bias_y'] + np.random.normal(0, p['std_y'])
        nh = msg.pose.theta + p['bias_h'] + np.random.normal(0, p['std_h'])
        nv = float(msg.velocity.data) + p['bias_v'] + np.random.normal(0, p['std_v'])

        self.pub_noise_stats.publish(Float32MultiArray(data=[
            1.0, nx - tx, ny - ty, nv - float(msg.velocity.data), 0.0
        ]))

        obstacle = Obstacle()
        obstacle.cls.data = 0
        obstacle.id.data = 0
        obstacle.pose.x = nx
        obstacle.pose.y = ny
        obstacle.pose.theta = nh
        obstacle.velocity.data = nv
        obstacle.distance.data = distance
        obstacle.dangerous.data = 0
        delay_ms = max(0.0, random.gauss(p['mu_delay'], p['sigma_delay']))
        obstacle.lidar_delay.data = delay_ms / 1000.0
        self._lidar_delay_buf.append((rospy.get_time() + delay_ms / 1000.0, [obstacle]))

    def _comm_perf_cb(self, msg):
        if len(msg.data) >= 4 and msg.data[0] < 0.5:  # not dropped
            self._v2x_delay_ms = float(msg.data[3])

    def update_target_info(self):
        _nan = float('nan')
        if self.user_input.get('with_coop', True):
            raw = self.target_info_v2v
            if len(raw) >= 5:
                h = float(raw[5]) if len(raw) > 5 else 0.0
                x_c, y_c, applied, _ = self.dr.compensate(
                    float(raw[3]), float(raw[4]), float(raw[2]), h, self._v2x_delay_ms, 'v2x')
                self.target_info = [raw[0], raw[1], raw[2], x_c, y_c]
                self._dr_log = [float(applied), self._v2x_delay_ms,
                                float(raw[3]), float(raw[4]), x_c, y_c,
                                0.0, _nan, _nan, _nan, _nan, _nan]
            else:
                self.target_info = raw if raw else [0, 0, 0, 0, 0]
                self._dr_log = [0.0] * 12
        else:
            obs = self.get_closest_lidar_obstacle()
            if obs is not None:
                tau_ms = float(obs.lidar_delay.data) * 1000.0
                rx, ry = float(obs.pose.x), float(obs.pose.y)
                x_c, y_c, applied, _ = self.dr.compensate(
                    rx, ry, float(obs.velocity.data),
                    math.degrees(float(obs.pose.theta)), tau_ms, 'lidar')
                self.target_info = [0, 0, float(obs.velocity.data), x_c, y_c]
                self._dr_log = [0.0, _nan, _nan, _nan, _nan, _nan,
                                float(applied), tau_ms, rx, ry, x_c, y_c]
            else:
                self.target_info = [0, 0, 0, 0, 0]
                self._dr_log = [0.0] * 12
        if len(self.target_info) >= 5:
            self.pub_v2v_target.publish(Float32MultiArray(data=[float(v) for v in self.target_info[:5]]))
        self.pub_dr_log.publish(Float32MultiArray(data=self._dr_log))

    def stop_emergency_publishing(self):
        """5초 emergency 발행 종료 후 50초 쿨다운 시작"""
        self.emergency_active = False
        self.emergency_cooldown = True
        rospy.loginfo("Emergency publishing stopped - starting 20s cooldown")
        
        # 50초 쿨다운 타이머 시작
        self.cooldown_timer = Timer(self.cooldown_duration, self.reset_emergency_cooldown)
        self.cooldown_timer.start()

    def reset_emergency_cooldown(self):
        """50초 쿨다운 해제"""
        self.emergency_cooldown = False
        rospy.loginfo("Emergency cooldown finished - ready to publish emergency again")


    def calc_world_pose(self, x, y):
        la, ln, al = self.enu2geo_transformter.transform(x, y, 5)
        return [la, ln]

    def organize_share_info(self, _path, _target_velocity, merge_safety):
        share_info = ShareInfo()
        if self.car['fix'] == 'No':
            return share_info
        share_info.state.data = int(self.user_input['state'])
        if self.lpp.pstate == 'EMERGENCY_CHANGE':
            share_info.signal.data = 7
        elif merge_safety != 0:
            share_info.signal.data = 4 if merge_safety == 1 else 5
        elif self.lpp.change_state and self.lpp.temp_signal in [1, 2]:
            share_info.signal.data = self.lpp.temp_signal
        else:
            share_info.signal.data = int(self.user_input['signal'])

        target_vel = _target_velocity
        share_info.target_velocity.data = target_vel
        share_info.pose.x = self.car['x']
        share_info.pose.y = self.car['y']
        share_info.pose.theta = self.car['t']
        share_info.velocity.data = self.car['v']
        if _path != None:
            for xy in _path:
                path = Path()
                path.pose.x = xy[0]
                path.pose.y = xy[1]
                share_info.paths.append(path)
        for obstacle in (self.lidar_obstacles or []) + self.npc_obstacles:
            share_info.obstacles.append(obstacle)
        
        return share_info
    
    def publish(self, lpp_res, vp_res):
        share_info = self.organize_share_info(lpp_res[1],vp_res,lpp_res[5])
        self.pub_ego_share_info.publish(share_info)
        self.pub_planner_state.publish(String(data=self.lpp.pstate))
        if self.type == 'ego':
            self.pub_obs_caution.publish(Bool(lpp_res[4]))
        self.pub_dangerous_obstacle.publish(Float32MultiArray(data=list(self.dangerous_obstacle)))
    
    def publish_inter_pt(self, inter_pt):
        # WC 모드(With Communication)일 때만 merge point 표시
        if not self.user_input.get('with_coop', True):
            # WOC 모드면 마커 삭제
            marker = Marker()
            marker.action = Marker.DELETE
            marker.id = 1
            marker.ns = 'intersection'
            self.pub_inter_pt.publish(marker)
            return

        if inter_pt is not None:
            marker = Marker()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.header.frame_id = 'world'
            marker.ns = 'intersection'
            marker.id = 1
            marker.lifetime = rospy.Duration(0)
            marker.scale.x = 1
            marker.scale.y = 1
            marker.scale.z = 1
            marker.color.r = 33/255
            marker.color.g = 255/255
            marker.color.b = 144/255
            marker.color.a = 0.7
            marker.pose.position.x = inter_pt[0]
            marker.pose.position.y = inter_pt[1]
            marker.pose.position.z = 1.0
            self.pub_inter_pt.publish(marker)
        else:
            # inter_pt가 None이면 마커 삭제
            marker = Marker()
            marker.action = Marker.DELETE
            marker.id = 1
            marker.ns = 'intersection'
            self.pub_inter_pt.publish(marker)

    def publish_bsd_zone(self, bsd_info):
        """
        BSD zone 시각화
        bsd_info: {'active': bool, 'signal': int, 'bsd_range': list, 'local_path': list}
        """
        if not bsd_info or not bsd_info.get('active', False):
            # WOC 모드가 아니거나 차선 변경 신호가 없으면 삭제
            marker = Marker()
            marker.action = Marker.DELETE
            marker.id = 1
            marker.ns = 'bsd_zone'
            self.pub_bsd_zone.publish(marker)
            return

        signal = bsd_info.get('signal', 0)
        if signal not in [1, 2]:  # 차선 변경 신호가 없으면 표시 안함
            return

        bsd_range = bsd_info.get('bsd_range', [-12, 10, 1.5, 6.0])
        ts_rear, ts_front, td_min, td_max = bsd_range

        # Ego 차량 위치와 heading
        ego_x = self.car['x']
        ego_y = self.car['y']
        ego_heading = math.radians(self.car['t'])

        # BSD zone을 LINE_STRIP으로 그리기
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = rospy.Time.now()
        marker.ns = 'bsd_zone'
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(0.1)  # 0.1초 후 자동 삭제

        # 색상 설정 (signal에 따라)
        if signal == 1:  # 좌회전 - 파란색
            marker.color.r = 0.2
            marker.color.g = 0.5
            marker.color.b = 1.0
            marker.color.a = 0.5
            lateral_offset = 1  # 왼쪽
        else:  # 우회전 - 노란색
            marker.color.r = 1.0
            marker.color.g = 0.8
            marker.color.b = 0.0
            marker.color.a = 0.5
            lateral_offset = -1  # 오른쪽

        marker.scale.x = 0.3  # 선 두께

        # BSD zone 박스의 네 모서리 계산
        # 차량 좌표계: x=앞, y=왼쪽
        from geometry_msgs.msg import Point

        # 박스 모서리 (차량 좌표계)
        corners = [
            (ts_rear, lateral_offset * td_min),  # 뒤-안쪽
            (ts_front, lateral_offset * td_min), # 앞-안쪽
            (ts_front, lateral_offset * td_max), # 앞-바깥쪽
            (ts_rear, lateral_offset * td_max),  # 뒤-바깥쪽
            (ts_rear, lateral_offset * td_min),  # 닫기
        ]

        # 차량 좌표계 -> 월드 좌표계 변환
        for local_x, local_y in corners:
            # 회전 + 이동
            world_x = ego_x + local_x * math.cos(ego_heading) - local_y * math.sin(ego_heading)
            world_y = ego_y + local_x * math.sin(ego_heading) + local_y * math.cos(ego_heading)

            point = Point()
            point.x = world_x
            point.y = world_y
            point.z = 0.5  # 지면에서 약간 위
            marker.points.append(point)

        self.pub_bsd_zone.publish(marker)


    def publish_emergency(self, emergency, caution=False):
        """
        Emergency 발행 - check_emergency 또는 caution 감지 시 signal=7 전송
        """
        if self.user_input['scenario'] >= 7:
            # caution: 차선 변경 후 dangerous_obstacle이 off-path가 돼도 signal=7 보장
            if emergency == 'emergency':
                # 쿨다운 중이면 무시
                if self.emergency_cooldown:
                    return

                # 처음 emergency면 타이머 시작
                if not self.emergency_active:
                    self.emergency_active = True
                    rospy.loginfo("Emergency started")

                    if self.emergency_timer:
                        self.emergency_timer.cancel()
                    self.emergency_timer = Timer(7.0, self.stop_emergency)
                    self.emergency_timer.start()

                # signal=7로 발행
                self.publish_signal(7)
                
            # else:  # 'normal'
            #     # 모든 상태 리셋
            #     self.reset_emergency()
            #     # signal=0으로 발행
            #     self.publish_signal(0)

    def publish_stop(self):
        msg = Float32MultiArray()
        msg.data = [0.0, 0.0, 0.0, float(self.user_input.get('scenario', 0))]
        self.pub_emergency_user_input.publish(msg)
        rospy.loginfo(f"[ENDPOINT] {self.type} stop signal sent (state=0)")

    def publish_signal(self, signal_value):
        """signal 값으로 메시지 발행"""
        msg = Float32MultiArray()
        msg.data = [
            float(self.user_input['state']),
            float(signal_value),
            float(self.user_input['target_velocity']),
            float(self.user_input['scenario'])
        ]
        self.user_input['signal'] = signal_value
        self.pub_emergency_user_input.publish(msg)

    def stop_emergency(self):
        """5초 후 emergency 중단"""
        self.emergency_active = False
        self.emergency_cooldown = True
        rospy.loginfo("Emergency stopped - cooldown started")
        
        # signal=0으로 변경
        self.publish_signal(0)
        
        # 50초 후 쿨다운 해제
        if self.cooldown_timer:
            self.cooldown_timer.cancel()
        self.cooldown_timer = Timer(20.0, self.end_cooldown)
        self.cooldown_timer.start()

    def end_cooldown(self):
        """쿨다운 해제"""
        self.emergency_cooldown = False
        rospy.loginfo("Cooldown ended")

    def reset_emergency(self):
        """Emergency 상태 완전 리셋"""
        self.emergency_active = False
        self.emergency_cooldown = False

        if self.emergency_timer:
            self.emergency_timer.cancel()
        if self.cooldown_timer:
            self.cooldown_timer.cancel()

    def carla_reset_cb(self, msg):
        self.auto_signal_sent = False
        self.speed_reached_time = None
        self.user_input['signal'] = 0
        self.car['v'] = 0
        self.target_info_v2v = []
        self.lpp.setting_values()
        rospy.loginfo(f"[{self.type}] state reset on /carla/reset")

    # ── LiDAR 노이즈 프리셋 (2×2 Factorial) ───────────────────────────
    _LIDAR_PRESETS = {
        'baseline': {                    # C1/C2 — field 실측 (noise_model.md)
            'bias_x': -1.50, 'std_x': 0.42,
            'bias_y':  0.13, 'std_y': 0.41,
            'bias_h': -0.36, 'std_h': 5.18,
            'bias_v':  0.72, 'std_v': 0.81,
            'alpha': 0.0063, 'beta': 0.041,
            'mu_delay': 50.3, 'sigma_delay': 30.1,
            'range': 130,
        },
        'degraded': {                    # C3/C4 — 악천후/원거리
            'bias_x': -1.50, 'std_x': 0.42,
            'bias_y':  0.13, 'std_y': 0.41,
            'bias_h': -0.36, 'std_h': 5.18,
            'bias_v':  0.72, 'std_v': 0.81,
            'alpha': 0.020, 'beta': 0.025,
            'mu_delay': 160.0, 'sigma_delay': 30.1,
            'range': 80,
        },
    }
    _OBU_NOISE = {
        'bias_x': +0.136, 'std_x': 0.267,
        'bias_y': -0.160, 'std_y': 0.327,
        'bias_h':  0.00,  'std_h': 0.08,   # deg
        'bias_v': -0.011, 'std_v': 0.038,
        # 드롭·딜레이는 v2v_bridge(channel 레벨)에서 처리
    }

    def _sensor_condition_cb(self, msg):
        self.sensor_condition = msg.data
        self.lidar_preset = 'degraded' if msg.data in ['c3', 'c4'] else 'baseline'
        rospy.loginfo(f'[CONDITION] {self.type} sensor_condition={msg.data} → lidar={self.lidar_preset}')

    def _detect(self):
        p = self._LIDAR_PRESETS[self.lidar_preset]
        s = self._ge_state[self.lidar_preset]
        if s == 'GOOD':
            if random.random() < p['alpha']:
                self._ge_state[self.lidar_preset] = 'BAD'
        else:
            if random.random() < p['beta']:
                self._ge_state[self.lidar_preset] = 'GOOD'
        return self._ge_state[self.lidar_preset] == 'GOOD'

    def _obu_noise(self, x, y, h, v):
        """OBU state에 Gaussian noise 적용"""
        p = self._OBU_NOISE
        return (x + np.random.normal(p['bias_x'], p['std_x']),
                y + np.random.normal(p['bias_y'], p['std_y']),
                h + np.random.normal(p['bias_h'], p['std_h']),
                v + np.random.normal(p['bias_v'], p['std_v']))

    def flush_obu_buffer(self):
        """메인 루프에서 호출 — delay 경과한 OBU 메시지 처리"""
        now = rospy.get_time()
        while self._obu_delay_buf and self._obu_delay_buf[0][0] <= now:
            _, d = self._obu_delay_buf.popleft()
            self.target_info_v2v = [d['state'], d['signal'], d['v'], d['x'], d['y']]
            self.target_path = d['paths']
            self.update_target_info()

    def flush_lidar_buffer(self):
        """메인 루프에서 호출 — delay 경과한 LiDAR 감지 결과 처리"""
        now = rospy.get_time()
        while self._lidar_delay_buf and self._lidar_delay_buf[0][0] <= now:
            _, obstacles = self._lidar_delay_buf.popleft()
            self.lidar_obstacles = obstacles
            self.update_target_info()

    _SPEED_RATIO = {'slower': 0.83, 'same': 1.0, 'faster': 1.17}
    _SIGNAL_MAP  = {1: {'ego': 2}, 2: {'ego': 1},
                    3: {'ego': 2, 'target': 1}, 4: {'ego': 1, 'target': 2},
                    5: {'ego': 2}, 6: {'ego': 2}}

    def check_auto_signal(self):
        if self.auto_signal_sent:
            return
        if self.user_input.get('state', 0) != 1:
            return
        scenario = self.user_input.get('scenario', 0)
        if scenario < 1 or scenario > 6:
            return
        my_tv = self.user_input['target_velocity'] * 3.6
        if my_tv < 1.0:
            return
        if self.type == 'ego':
            self._auto_signal_ego(scenario, my_tv)
        else:
            self._auto_signal_target(scenario, my_tv)

    def _faster_reached(self, my_tv, thr):
        """목표속도가 더 높은 차량(나중에 달성)이 도달했는지 확인.
        ratio >= 1.0: TV가 더 빠름 → TV 속도 기준
        ratio <  1.0: ego가 더 빠름 → ego 속도 기준"""
        mode  = self.test_mode.split('_')[-1] if '_' in self.test_mode else 'same'
        ratio = self._SPEED_RATIO.get(mode, 1.0)

        # CARLA: ground truth 속도로 판정 (WC/WOC 동일 기준)
        if self.test == 1:
            target_v_kmh = self.target_gt_v * 3.6
        else:
            tv_info      = self.target_info_v2v if self.user_input.get('with_coop', True) else self.target_info
            target_v_kmh = tv_info[2] * 3.6 if len(tv_info) >= 3 else None

        if ratio >= 1.0:
            if self.type == 'ego':
                if target_v_kmh is None: return False
                return abs(target_v_kmh - my_tv * ratio) <= thr
            else:
                return abs(self.car['v'] * 3.6 - my_tv) <= thr
        else:
            if self.type == 'ego':
                return abs(self.car['v'] * 3.6 - my_tv) <= thr
            else:
                if target_v_kmh is None: return False
                return abs(target_v_kmh - my_tv / ratio) <= thr

    def _auto_signal_ego(self, scenario, my_tv):
        thr = max(3.0, my_tv * 0.05)
        if self.speed_reached_time is None and not self._faster_reached(my_tv, thr):
            return
        self._countdown_and_fire(scenario)

    def _auto_signal_target(self, scenario, my_tv):
        thr = max(3.0, my_tv * 0.05)
        if self.speed_reached_time is None and not self._faster_reached(my_tv, thr):
            return
        self._countdown_and_fire(scenario)

    def _countdown_and_fire(self, scenario):
        now = rospy.get_time()
        if self.speed_reached_time is None:
            self.speed_reached_time = now
            rospy.loginfo(f"[AUTO SIGNAL] {self.type} reached target speed — waiting 3s")
            return
        if now - self.speed_reached_time < 3.0:
            return
        signal = self._SIGNAL_MAP.get(scenario, {}).get(self.type)
        if signal is None:
            return
        self.user_input['signal'] = signal
        self.auto_signal_sent = True
        rospy.loginfo(f"[AUTO SIGNAL] {self.type} sent signal {signal} for CLM{scenario}")
        Timer(3.0, lambda: self.user_input.update({'signal': 0})).start()

    def load_endpoints(self):
        """ui/yaml/end_point.yaml에서 endpoint 좌표 로드"""
        import yaml
        import os
        yaml_path = os.path.join(os.path.dirname(__file__), '../ui/yamls/end_point.yaml')
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            endpoints = []
            for key in ['point1', 'point2', 'point3', 'point4', 'point5', 'point6']:
                if key in data:
                    endpoints.append(data[key])
            rospy.loginfo(f"[ENDPOINT] Loaded {len(endpoints)} endpoints")
            return endpoints
        except Exception as e:
            rospy.logwarn(f"[ENDPOINT] Failed to load endpoints: {e}")
            return []

    _CARLA_ENDPOINT_X = 570  # Town06 Road40 커브 시작점 (임시값, 실측 후 수정)

    def check_endpoint(self):
        """차량이 endpoint에 도달했는지 확인 — 어느 한 차라도 먼저 도달하면 즉시 종료"""
        if self.scenario_done:
            return True

        car_x = self.car['x']
        car_y = self.car['y']

        if self.test == 1:
            if self.user_input.get('state', 0) == 1 and car_x > self._CARLA_ENDPOINT_X:
                rospy.loginfo(f"[ENDPOINT] {self.type} reached curve start (x={car_x:.1f}) — broadcasting done")
                self.pub_scenario_done.publish(Bool(data=True))
                self.pub_endpoint_reached.publish(Bool(data=True))
                self.scenario_done = True
                return True
            return False

        if not self.endpoints:
            return False
        for i, ep in enumerate(self.endpoints):
            dist = math.sqrt((car_x - ep[0])**2 + (car_y - ep[1])**2)
            if dist < self.endpoint_threshold:
                rospy.loginfo(f"[ENDPOINT] {self.type} reached endpoint {i+1}")
                self.pub_scenario_done.publish(Bool(data=True))
                self.pub_endpoint_reached.publish(Bool(data=True))
                return True
        return False

    def publish_endpoints(self):
        """Endpoint 위치를 마커로 시각화"""
        if not self.endpoints:
            return

        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "endpoints"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 3.0
        marker.scale.y = 3.0
        marker.scale.z = 3.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8

        for ep in self.endpoints:
            p = Point()
            p.x = ep[0]
            p.y = ep[1]
            p.z = 0.0
            marker.points.append(p)

        self.pub_endpoints.publish(marker)
