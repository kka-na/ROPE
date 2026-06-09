"""
CarlaTransmitter: simulator.py의 CARLA 대체판.
selfdriving/control에서 오는 actuator를 /carla/{type}/actuator 토픽으로 발행.
carla/bridge/carla_ros_bridge.py가 이를 구독해 CARLA actor에 적용.
"""
import rospy
from std_msgs.msg import Float32MultiArray


class CarlaTransmitter:
    def __init__(self, vehicle_type: str):
        self.type = vehicle_type
        self.pub = rospy.Publisher(
            f'/carla/{vehicle_type}/actuator', Float32MultiArray, queue_size=1
        )
        self.car = {'state': 0, 'x': 0, 'y': 0, 't': 0, 'v': 0}
        self.obstacles = []

    def set_actuator(self, actuator):
        """actuator: (accel_norm, steer_deg) — control.py 출력 형식."""
        import math
        accel_norm = float(actuator[0])
        steer_rad  = math.radians(actuator[1])
        self.pub.publish(Float32MultiArray(data=[accel_norm, steer_rad]))

    def set_user_input(self, msg):
        pass  # CARLA에서 초기 위치 변경 불필요

    async def execute(self):
        pass  # bridge가 world.tick() 담당

    def cleanup(self):
        pass
