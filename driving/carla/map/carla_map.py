"""
CarlaMAP: CARLA Map API → 기존 MAP 클래스와 동일한 lanelets/tiles 인터페이스.
LocalPathPlanner(carla_map, type) 에 그대로 주입 가능 — 코드 수정 없음.

좌표계: CARLA 좌손계(Y반전) → ROS 우손계
  ros_x = carla_x
  ros_y = -carla_y
"""
import carla
from collections import defaultdict
from visualization_msgs.msg import MarkerArray


class CarlaMAP:
    def __init__(self, world, tile_size=5, radius=200.0, center=None):
        """
        Args:
            world: carla.World
            tile_size: 공간 해시 타일 크기 (기존 MAP과 동일하게 5)
            radius: 시나리오 중심점 기준 waypoint 수집 반경 (m)
            center: carla.Location — None이면 전체 맵 사용
        """
        self.tile_size = tile_size
        self.base_lla = [0.0, 0.0, 0.0]  # CARLA는 GPS 불필요 (dummy)
        self.lmap_viz = MarkerArray()     # RViz 시각화 (빈 값, 기존 코드 호환용)
        self.mlmap_viz = MarkerArray()

        cmap = world.get_map()
        all_wps = cmap.generate_waypoints(1.0)

        # 반경 필터링
        if center is not None:
            all_wps = [w for w in all_wps
                       if w.transform.location.distance(center) < radius]

        self.lanelets, self.tiles = self._build(all_wps, cmap)

    # ── public (기존 MAP과 동일한 속성) ────────────────────────────────
    # self.lanelets, self.tiles, self.tile_size, self.base_lla 제공

    # ── 내부 빌드 ──────────────────────────────────────────────────────
    def _build(self, all_wps, cmap):
        # 1. 주행 차선만, (road_id, lane_id) 기준 그룹핑
        groups = defaultdict(list)
        for wp in all_wps:
            if wp.lane_type != carla.LaneType.Driving:
                continue
            if wp.is_junction:
                continue
            groups[(wp.road_id, wp.lane_id)].append(wp)

        # 2. s(도로 진행 방향 arc length) 기준 오름차순 정렬
        # s=0이 도로 기준 시작점, 증가 방향이 주행 방향
        for key in groups:
            groups[key].sort(key=lambda w: w.s)

        # 3. lanelets dict 생성
        lanelets = {}
        for key, wps in groups.items():
            lid = f"{key[0]}_{key[1]}"
            waypoints = [(w.transform.location.x, -w.transform.location.y)
                         for w in wps]
            lanelets[lid] = {
                'waypoints':     waypoints,
                'laneNo':        None,      # step 5에서 채움
                'adjacentLeft':  None,
                'adjacentRight': None,
                'successor':     [],
            }

        # 4. adjacentLeft/Right
        # CARLA get_left/right_lane()은 해당 차선의 주행 방향 기준 좌우를 반환.
        for key, wps in groups.items():
            lid = f"{key[0]}_{key[1]}"
            ref = wps[len(wps) // 2]
            l = ref.get_left_lane()
            r = ref.get_right_lane()
            lk = (ref.road_id, l.lane_id) if (l and l.lane_type == carla.LaneType.Driving) else None
            rk = (ref.road_id, r.lane_id) if (r and r.lane_type == carla.LaneType.Driving) else None
            lanelets[lid]['adjacentLeft']  = f"{lk[0]}_{lk[1]}" if (lk and lk in groups) else None
            lanelets[lid]['adjacentRight'] = f"{rk[0]}_{rk[1]}" if (rk and rk in groups) else None

        # 5. laneNo: 같은 road에서 오른쪽부터 1, 2, 3 ...
        #    CARLA: 음수 lane_id = 도로 진행 방향
        road_lanes = defaultdict(list)
        for key in groups:
            road_lanes[key[0]].append(key[1])

        for road_id, lane_ids in road_lanes.items():
            driving = sorted([l for l in lane_ids if l < 0])  # -1 → 가장 오른쪽
            for i, lid_num in enumerate(reversed(driving)):    # -1=1번, -2=2번
                lid = f"{road_id}_{lid_num}"
                if lid in lanelets:
                    lanelets[lid]['laneNo'] = i + 1

        # 6. successor
        for key, wps in groups.items():
            lid = f"{key[0]}_{key[1]}"
            for nwp in wps[-1].next(1.5):
                nkey = (nwp.road_id, nwp.lane_id)
                nid  = f"{nkey[0]}_{nkey[1]}"
                if nid in lanelets and nid != lid:
                    lanelets[lid]['successor'].append(nid)

        # 7. tiles (공간 해시)
        tiles = {}
        ts = self.tile_size
        for lid, data in lanelets.items():
            for i, (x, y) in enumerate(data['waypoints']):
                row, col = int(x // ts), int(y // ts)
                tiles.setdefault((row, col), {}).setdefault(
                    lid, {'waypoints': [], 'idx': []}
                )
                tiles[(row, col)][lid]['waypoints'].append((x, y))
                tiles[(row, col)][lid]['idx'].append(i)

        # 진단: 스폰 위치(x≈135, y≈-241) 근처 lane 확인 (Town06 Road40 lane-4)
        for lid, data in lanelets.items():
            wps = data['waypoints']
            if any(abs(x - 135) < 10 and abs(y - (-241)) < 5 for x, y in wps[:5]):
                print(f'[CarlaMAP] spawn-lane {lid}: first={wps[0]}, last={wps[-1]}, len={len(wps)}')
                print(f'[CarlaMAP]   laneNo={data["laneNo"]} adjL={data["adjacentLeft"]} adjR={data["adjacentRight"]} succ={data["successor"]}')
                for nb in [data['adjacentLeft'], data['adjacentRight']]:
                    if nb and nb in lanelets:
                        nbw = lanelets[nb]['waypoints']
                        print(f'[CarlaMAP]   neighbor {nb}: len={len(nbw)} first={nbw[0] if nbw else None}')
                break

        return lanelets, tiles
