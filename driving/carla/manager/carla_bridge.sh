#!/bin/bash
# carla_bridge.sh — CARLA bridge + ROS core
# 사용법: ./carla_bridge.sh [SCENARIO] [TEST_MODE] [CHANNEL] [SPEED_KMH] [wc|woc]

SCENARIO=${1:-CLM2}
TEST_MODE=${2:-same}
CHANNEL=${3:-yeongjong}
SPEED_KMH=${4:-30}
MODE=${5:-wc}
NPC=${6:-nonpc}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

source ~/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311

cleanup() { kill $(jobs -p) 2>/dev/null; }
trap cleanup EXIT

roscore &
sleep 1
rostopic pub --latch /carla/npc_mode std_msgs/String "data: '$NPC'" &

cd "$ROOT/carla/bridge"
NPC_FLAG=""
[ "$NPC" = "npc" ] && NPC_FLAG="--npc"
python3 carla_ros_bridge.py \
    --scenario "$SCENARIO" \
    --test_mode "$TEST_MODE" \
    --speed "$SPEED_KMH" $NPC_FLAG &
sleep 3

if [ "$MODE" = "woc" ]; then
    python3 "$ROOT/carla/bridge/v2v_bridge.py" --channel "$CHANNEL" --woc &
else
    python3 "$ROOT/carla/bridge/v2v_bridge.py" --channel "$CHANNEL" &
fi

wait
