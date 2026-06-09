#!/usr/bin/env python3
import rospy
import sys
import signal
import argparse

import setproctitle
setproctitle.setproctitle("sharing_info")
from ros_manager import ROSManager
from planning.local_path_planner import LocalPathPlanner
from planning.velocity_planner import VelocityPlanner
from perception.obstacle_handler import ObstacleHandler

from hd_map.map import MAP

def signal_handler(sig, frame):
    sys.exit(0)

def load_map(map_name):
    if map_name == 'CarlaMap':
        import carla, sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', 'carla'))
        from map.carla_map import CarlaMAP
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        return CarlaMAP(client.get_world())
    return MAP(map_name)

class SharingInfo():
    def __init__(self, type, map_name, test):
        self.map = load_map(map_name)
        self.lpp = LocalPathPlanner(self.map, type, is_carla=(map_name == 'CarlaMap'))
        self.vp = VelocityPlanner(type, is_carla=(map_name == 'CarlaMap'))
        self.oh = ObstacleHandler(self.lpp.phelper)
        self.RM = ROSManager(type, test, self.map, self.oh, self.lpp)
        rospy.on_shutdown(self.lpp._flush_log)
        self.set_values()
    
    def set_values(self):
        self.local_path = None
        self.limit_local_path = None
        self.local_waypoints = None
        self.local_lane_number = None
        self.vp_result = 0
        self.target_signal = 0

    def update_value(self):
        self.lpp.update_value(self.RM.car, self.RM.user_input, self.RM.target_info, self.RM.target_path, self.RM.dangerous_obstacle)
        self.vp.update_value(self.RM.user_input, self.RM.car, self.RM.target_info)
        self.oh.update_value(self.RM.car, self.lpp.local_path,
                             (self.RM.lidar_obstacles or []) + self.RM.npc_obstacles)

    def path_planning(self):
        lpp_result = self.lpp.execute()
        if lpp_result is not None:
            return lpp_result

    def velocity_planning(self, lpp_result):
        vp_result = self.vp.execute(lpp_result)
        return vp_result

    def perception_handling(self):
        emergency = self.oh.check_emergency(self.RM.dangerous_obstacle)
        return emergency
    
    def execute(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            self.update_value()

            # simulator_inform을 아직 못 받은 경우 (position=0,0) → 경로 계산 스킵
            # 이 상태에서 경로를 계산하면 NaN이 발생하고 target_velocity가 오염됨
            if self.RM.car['fix'] == 'No':
                rate.sleep()
                continue

            self.RM.flush_obu_buffer()
            self.RM.flush_lidar_buffer()
            self.RM.check_auto_signal()

            if self.RM.check_endpoint():
                rospy.loginfo(f"[ENDPOINT] Stopping node - endpoint reached")
                self.RM.publish_stop()
                rospy.signal_shutdown("Endpoint reached")
                break

            try:
                lpp_result = self.path_planning()
                if lpp_result is not None:
                    vp_result = self.velocity_planning(lpp_result)
                    emergency = self.perception_handling()
                    self.RM.publish(lpp_result, vp_result)
                    self.RM.publish_inter_pt(self.lpp.get_interpt())
                    self.RM.publish_bsd_zone(self.lpp.get_bsd_info())
                    self.RM.publish_emergency(emergency, lpp_result[4])
            except Exception as e:
                import traceback
                rospy.logwarn(f'[{self.RM.type}] execute error: {e}\n{traceback.format_exc()}')
            rate.sleep()


def main():
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dr-enabled', action='store_true')
    parser.add_argument('--dr-target', default='both', choices=['both', 'v2x_only', 'lidar_only'])
    args, remaining = parser.parse_known_args(sys.argv[1:])

    type     = remaining[0] if len(remaining) > 0 else 'ego'
    map_name = remaining[1] if len(remaining) > 1 else 'Solbat'
    test     = int(remaining[2]) if len(remaining) > 2 else 0

    si = SharingInfo(type, map_name, test)
    if args.dr_enabled:
        from dr_compensator import DRCompensator
        si.RM.dr = DRCompensator(enabled=True, target=args.dr_target)
        rospy.loginfo(f'[DR] enabled, target={args.dr_target}')
    si.execute()

if __name__=="__main__":
    main()