"""
BaseScenario: CARLA world 공통 유틸 (actor spawn, sensor attach, cleanup).
seed 고정으로 WC/WOC 대응쌍의 초기 조건을 동일하게 보장.
"""
import math
import carla
import random


# 차량 blueprint: 기존 실험 차량(IONIQ5, Avante)에 가장 근접
BLUEPRINT = {
    'ego':    'vehicle.audi.etron',
    'target': 'vehicle.seat.leon',
    'emv':    'vehicle.dodge.charger_police',
}

_NPC_BPS = [
    'vehicle.tesla.model3', 'vehicle.toyota.prius',
    'vehicle.nissan.patrol_2021', 'vehicle.lincoln.mkz_2020',
    'vehicle.mercedes.coupe_2020', 'vehicle.bmw.grandtourer', 'vehicle.audi.a2',
]


class BaseScenario:
    def __init__(self, world: carla.World, seed: int = 42):
        self.world  = world
        self.bp_lib = world.get_blueprint_library()
        self.actors = []
        random.seed(seed)

    def spawn_vehicle(self, role: str, transform: carla.Transform) -> carla.Actor:
        # road surface z에 맞춰 높이 보정 (0.5m 위에 spawn)
        wp = self.world.get_map().get_waypoint(transform.location)
        loc = transform.location
        loc.z = wp.transform.location.z + 0.5
        transform = carla.Transform(loc, transform.rotation)
        bp = self.bp_lib.find(BLUEPRINT[role])
        bp.set_attribute('role_name', role)
        actor = self.world.try_spawn_actor(bp, transform)
        if actor is None:
            raise RuntimeError(f"Failed to spawn {role} at {transform.location}")
        self.actors.append(actor)
        return actor

    def attach_lidar(self, vehicle: carla.Actor, range_m: float = 130.0) -> carla.Sensor:
        """Semantic LiDAR — 실험 Hesai Pandar 64 근사."""
        bp = self.bp_lib.find('sensor.lidar.ray_cast_semantic')
        bp.set_attribute('range',             str(range_m))
        bp.set_attribute('channels',          '64')
        bp.set_attribute('rotation_frequency','20')
        bp.set_attribute('points_per_second', '640000')
        sensor = self.world.spawn_actor(
            bp, carla.Transform(carla.Location(z=2.0)), attach_to=vehicle
        )
        self.actors.append(sensor)
        return sensor

    def set_initial_velocity(self, actor: carla.Actor, speed_mps: float):
        """actor 초기 속도 설정 (진행 방향 기준)."""
        yaw_rad = math.radians(actor.get_transform().rotation.yaw)
        v = carla.Vector3D(
            x=speed_mps * math.cos(yaw_rad),
            y=speed_mps * math.sin(yaw_rad),
            z=0
        )
        actor.set_target_velocity(v)

    def spawn_ghost(self, x, y, yaw_deg, bp_idx=0):
        """Physics-off ghost NPC. Returns actor or None on spawn failure."""
        bp = self.bp_lib.find(_NPC_BPS[bp_idx % len(_NPC_BPS)])
        bp.set_attribute('role_name', 'npc')
        tf = carla.Transform(carla.Location(x=x, y=y, z=0), carla.Rotation(yaw=yaw_deg))
        wp = self.world.get_map().get_waypoint(tf.location)
        loc = carla.Location(x=x, y=y, z=wp.transform.location.z + 0.5)
        actor = self.world.try_spawn_actor(bp, carla.Transform(loc, carla.Rotation(yaw=yaw_deg)))
        if actor:
            actor.set_simulate_physics(False)
            self.actors.append(actor)
        return actor

    def cleanup(self):
        for actor in reversed(self.actors):
            if actor.is_alive:
                actor.destroy()
        self.actors.clear()
