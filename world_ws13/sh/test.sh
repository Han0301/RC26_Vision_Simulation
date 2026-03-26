#!/bin/bash
source /home/awwsome/RC/world_ws5/devel/setup.bash

roslaunch zwei map1.launch 1> /dev/null &
arena1_launch_PID=$!
echo "map1.launch 已启动，PID: $arena1_launch_PID"

sleep 8

roslaunch 3dto2d zbuffer_test1.launch &
zbuffer_test1_PID=$!
echo "zbuffer_func_debug_node 已启动，PID：$zbuffer_test1_PID"

#rosrun wpr_simulation keyboard_vel_ctrl_node

# 捕获Ctrl+C信号，停止所有节点和roscore（关键，避免进程残留）
trap kill_node SIGINT

kill_node()
{
    for pid in $arena1_launch_PID $zbuffer_test1_PID; do
    kill $pid && wait $pid
    done
    exit 0
}

wait   # 脚本不会直接退出