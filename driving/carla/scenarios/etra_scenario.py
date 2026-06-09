"""
ETrAScenario: ETrA1~4 시나리오 초기 조건.

맵: Town06  Road 40  (471m 직선, yaw=0, 동쪽 진행)
  lane -3: y=237.65 / lane -4: y=241.15

d_safe = v * t_reaction (2s) → 속도에 따라 스케일
  safe  gap: 1.2 × d_safe  (ETrA1/4)
  unsafe gap: 0.6 × d_safe (ETrA2/3)
"""
import random
import carla
from .base_scenario import BaseScenario

SPEED_RATIO = {'slower': 0.83, 'same': 1.0, 'faster': 1.17}

_YAW   = 0.0
_Y1    = 237.65
_Y2    = 241.15
_Y3    = 244.65
_Y4    = 248.15   # lane -6 (3.5m 간격 추정)
_X_EGO = 135.0
_X_EMV = 305.0   # 170m 전방 (동쪽)

def _tf(x, y, yaw=_YAW):
    return carla.Transform(carla.Location(x=x, y=y, z=0), carla.Rotation(yaw=yaw))


_NPC_CONFIGS = {
    # 같은 레인(emv 포함): 레인 내 가장 앞 차량(emv) 보다 더 앞에 배치
    # 다른 레인: 뒤에서 따라옴
    'ETrA1': [  # ego Y1@135, tv Y1@~115, emv Y1@305
        {'x': 330, 'y': _Y1}, {'x': 370, 'y': _Y1},  # Y1 ahead of emv
        {'x':  80, 'y': _Y2}, {'x':  95, 'y': _Y3},  # other lanes behind
    ],
    'ETrA2': [  # ego Y2@135, tv Y2@~125, emv Y2@305
        {'x': 330, 'y': _Y2}, {'x': 370, 'y': _Y2},  # Y2 ahead of emv
        {'x':  80, 'y': _Y1}, {'x':  95, 'y': _Y3},  # other lanes behind
    ],
    'ETrA3': [  # ego Y1@135, tv Y2@~125, emv Y1@305
        {'x': 330, 'y': _Y1}, {'x': 370, 'y': _Y1},  # Y1 ahead of emv
        {'x': 180, 'y': _Y2}, {'x': 230, 'y': _Y2},  # Y2 ahead of tv
        {'x':  80, 'y': _Y3},                          # Y3 behind
    ],
    'ETrA4': [  # ego Y3@135, tv Y2@~145, emv (Y3+Y4)/2@305
        {'x': 330, 'y': _Y3}, {'x': 370, 'y': _Y3},  # Y3 ahead of emv
        {'x': 185, 'y': _Y2}, {'x': 250, 'y': _Y2},  # Y2 ahead of tv
        {'x':  80, 'y': _Y1},                          # Y1 behind
    ],
}


class ETrAScenario(BaseScenario):
    def __init__(self, scenario_id: str, test_mode: str = 'same',
                 world: carla.World = None, seed: int = 42, speed_kmh: int = 30):
        super().__init__(world, seed)
        self.scenario_id = scenario_id
        self.speed_ratio = SPEED_RATIO[test_mode]
        self.speed_kmh = speed_kmh
        self.npc_actors = []
        d_safe = (speed_kmh / 3.6) * 2.0
        d_safe_gap   = round(d_safe * 1.2)
        d_unsafe_gap = round(d_safe * 0.6)
        self._cfg = {
            'ETrA1': {'ego': _tf(_X_EGO,                _Y1),
                      'tv':  _tf(_X_EGO - d_safe_gap,   _Y1),
                      'emv': _tf(_X_EMV,                _Y1)},
            'ETrA2': {'ego': _tf(_X_EGO,                _Y2),
                      'tv':  _tf(_X_EGO - d_unsafe_gap, _Y2),
                      'emv': _tf(_X_EMV,                _Y2)},
            'ETrA3': {'ego': _tf(_X_EGO,                _Y1),
                      'tv':  _tf(_X_EGO - d_unsafe_gap, _Y2),
                      'emv': _tf(_X_EMV,                _Y1)},
            'ETrA4': {'ego': _tf(_X_EGO,                _Y3),
                      'tv':  _tf(_X_EGO + d_unsafe_gap, _Y2),
                      'emv': _tf(_X_EMV, (_Y3 + _Y4) / 2, yaw=25.0)},
        }

    def setup(self, npc=False):
        cfg = self._cfg[self.scenario_id]
        ego = self.spawn_vehicle('ego',    cfg['ego'])
        tv  = self.spawn_vehicle('target', cfg['tv'])
        emv = self.spawn_vehicle('emv',    cfg['emv'])
        emv.set_target_velocity(carla.Vector3D(0, 0, 0))
        self.attach_lidar(ego)
        self.attach_lidar(tv)
        self.npc_actors = []
        if npc:
            for i, nc in enumerate(_NPC_CONFIGS[self.scenario_id]):
                speed = random.uniform(0.8, 1.2) * self.speed_kmh / 3.6
                actor = self.spawn_ghost(nc['x'], nc['y'], _YAW, i)
                if actor:
                    self.npc_actors.append({'actor': actor, 'x0': nc['x'], 'y': nc['y'], 'speed': speed})
            print(f'[ETrA] spawned {len(self.npc_actors)} NPC ghosts')
        return ego, tv, emv
