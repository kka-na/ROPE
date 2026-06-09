#!/bin/bash
# carla_target.sh — target 스택 (sharing_info + selfdriving)
# 사용법: ./carla_target.sh [TEST_MODE] [wc|woc] [CONDITION]

TEST_MODE=${1:-same}
MODE=${2:-wc}
CONDITION=${3:-c1}
DR_MODE=${4:-}       # 'true' → DR 보정 활성화
DR_TARGET=${5:-both}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

DR_ARGS=""
if [ "$DR_MODE" = "true" ]; then
    DR_ARGS="--dr-enabled --dr-target $DR_TARGET"
fi

source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311

cleanup() { kill $(jobs -p) 2>/dev/null; }
trap cleanup EXIT

cd "$ROOT/sharing_info"
python3 main.py target CarlaMap 1 $DR_ARGS &
sleep 1

cd "$ROOT/selfdriving"
python3 main.py target carla CarlaMap &

if [ "$MODE" = "woc" ]; then
    rostopic pub --latch /target/with_coop std_msgs/String "data: 'false'" &
    rostopic pub --latch /target/test_mode std_msgs/String "data: 'WOC_${TEST_MODE}'" &
else
    rostopic pub --latch /target/with_coop std_msgs/String "data: 'true'" &
    rostopic pub --latch /target/test_mode std_msgs/String "data: 'WC_${TEST_MODE}'" &
fi

rostopic pub --latch /target/sensor_condition std_msgs/String "data: '${CONDITION}'" &

wait
