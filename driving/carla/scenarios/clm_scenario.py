"""
CLMScenario: CLM1~4 시나리오 초기 조건.

맵: Town06  Road 40  (471m 직선, yaw=0, 동쪽 진행)
  lane -3: y=237.65 / lane -4: y=241.15 / lane -5: y=244.65

차간 거리 규칙: gap(m) = speed_kmh  (한국 차간거리 규칙)
"""
import random
import carla
from .base_scenario import BaseScenario

SPEED_RATIO = {'slower': 0.83, 'same': 1.0, 'faster': 1.17}

_YAW = 0.0
_Y1  = 237.65
_Y2  = 241.15
_Y3  = 244.65
_X0  = 135.0  # Town06 Road40 서쪽 끝 (spawn 시작점)

def _tf(x, y):
    return carla.Transform(carla.Location(x=x, y=y, z=0), carla.Rotation(yaw=_YAW))


_NPC_CONFIGS = {
    # 같은 레인(ego/tv): x>=280 (gap 최대 130m 감안해 앞에 배치)
    # 다른 레인: x<135 (뒤에서 따라옴)
    'CLM1': [  # ego Y1@135+g, tv Y2@135
        {'x': 280, 'y': _Y1}, {'x': 340, 'y': _Y1},  # Y1 ahead of ego
        {'x': 170, 'y': _Y2}, {'x': 220, 'y': _Y2},  # Y2 ahead of tv
        {'x':  80, 'y': _Y3},                          # Y3 behind
    ],
    'CLM2': [  # ego Y2@135, tv Y1@135+g
        {'x': 170, 'y': _Y2}, {'x': 230, 'y': _Y2},  # Y2 ahead of ego
        {'x': 280, 'y': _Y1}, {'x': 340, 'y': _Y1},  # Y1 ahead of tv
        {'x':  80, 'y': _Y3},                          # Y3 behind
    ],
    'CLM3': [  # ego Y1@135, tv Y2@135
        {'x': 170, 'y': _Y1}, {'x': 240, 'y': _Y1},  # Y1 ahead of ego
        {'x': 185, 'y': _Y2}, {'x': 255, 'y': _Y2},  # Y2 ahead of tv
        {'x':  75, 'y': _Y3}, {'x': 110, 'y': _Y3},  # Y3 behind
    ],
    'CLM4': [  # ego Y3@135, tv Y1@135
        {'x': 170, 'y': _Y3}, {'x': 230, 'y': _Y3},  # Y3 ahead of ego
        {'x': 185, 'y': _Y1}, {'x': 250, 'y': _Y1},  # Y1 ahead of tv
        {'x':  80, 'y': _Y2},                          # Y2 behind
    ],
}


class CLMScenario(BaseScenario):
    def __init__(self, scenario_id: str, test_mode: str,
                 world: carla.World, seed: int = 42, speed_kmh: int = 30):
        super().__init__(world, seed)
        self.scenario_id = scenario_id
        self.speed_ratio = SPEED_RATIO[test_mode]
        self.speed_kmh = speed_kmh
        self.gap = speed_kmh
        self.npc_actors = []

    def setup(self, npc=False):
        g = self.gap
        print(f'[CLM] scenario={self.scenario_id} speed_km/h→gap={g}m')
        configs = {
            'CLM1': {'ego': _tf(_X0 + g, _Y1), 'tv': _tf(_X0,     _Y2)},
            'CLM2': {'ego': _tf(_X0,     _Y2), 'tv': _tf(_X0 + g, _Y1)},
            'CLM3': {'ego': _tf(_X0,     _Y1), 'tv': _tf(_X0,     _Y2)},
            'CLM4': {'ego': _tf(_X0,     _Y3), 'tv': _tf(_X0,     _Y1)},
        }
        cfg = configs[self.scenario_id]
        ego = self.spawn_vehicle('ego',    cfg['ego'])
        tv  = self.spawn_vehicle('target', cfg['tv'])
        self.attach_lidar(ego)
        self.attach_lidar(tv)
        self.npc_actors = []
        if npc:
            for i, nc in enumerate(_NPC_CONFIGS[self.scenario_id]):
                speed = random.uniform(0.8, 1.2) * self.speed_kmh / 3.6
                actor = self.spawn_ghost(nc['x'], nc['y'], _YAW, i)
                if actor:
                    self.npc_actors.append({'actor': actor, 'x0': nc['x'], 'y': nc['y'], 'speed': speed})
            print(f'[CLM] spawned {len(self.npc_actors)} NPC ghosts')
        return ego, tv
