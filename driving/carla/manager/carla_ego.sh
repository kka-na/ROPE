#!/bin/bash
# carla_ego.sh — ego 스택 (sharing_info + selfdriving + make_data)
# 사용법: ./carla_ego.sh [TEST_MODE] [wc|woc] [CONDITION]

TEST_MODE=${1:-same}
MODE=${2:-wc}
CONDITION=${3:-c1}
NPC=${4:-nonpc}
DR_MODE=${5:-}       # 'true' → DR 보정 활성화
DR_TARGET=${6:-both}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

DR_ARGS=""
DR_MAKE=""
if [ "$DR_MODE" = "true" ]; then
    DR_ARGS="--dr-enabled --dr-target $DR_TARGET"
    DR_MAKE="--dr"
fi

source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311

cleanup() { kill $(jobs -p) 2>/dev/null; }
trap cleanup EXIT

cd "$ROOT/sharing_info"
python3 main.py ego CarlaMap 1 $DR_ARGS &
sleep 1

cd "$ROOT/selfdriving"
python3 main.py ego carla CarlaMap &

cd "$ROOT/utils"
python3 make_data_carla.py --npc "$NPC" $DR_MAKE &
sleep 1

if [ "$MODE" = "woc" ]; then
    rostopic pub --latch /ego/with_coop  std_msgs/String "data: 'false'" &
    rostopic pub --latch /ego/test_mode  std_msgs/String "data: 'WOC_${TEST_MODE}'" &
else
    rostopic pub --latch /ego/with_coop  std_msgs/String "data: 'true'" &
    rostopic pub --latch /ego/test_mode  std_msgs/String "data: 'WC_${TEST_MODE}'" &
fi

rostopic pub --latch /ego/sensor_condition std_msgs/String "data: '${CONDITION}'" &

wait
