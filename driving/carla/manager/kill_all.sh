#!/bin/bash
pkill -9 -f 'sharing_info'
pkill -9 -f 'self_driving'
pkill -9 -f 'carla_ros_bridge'
pkill -9 -f 'v2v_bridge'
pkill -9 -f 'make_data_carla'
pkill -9 -f 'rostopic'
pkill -9 roscore
pkill -9 rosmaster
echo "done"
