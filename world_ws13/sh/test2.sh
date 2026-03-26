#!/bin/bash

source /home/awwsome/RC/world_ws5/devel/setup.bash

kill_node()
{
    local pid=$1
    # echo $pid
    if judge_running $pid;then
        kill $pid && wait $pid
        echo "进程 $pid 已退出"
    else
        echo "进程 $pid 不存在，无需杀死"
    fi
    return 0
}

judge_running()
{
    local pid=$1
    # echo $pid
    if ps -p "$pid"; then
        return 0
    else 
        return 1
    fi
}

# 跑gazebo
roslaunch zwei arena1.launch 1> /dev/null &
arena1_launch_pid=$!

# 跑 zbuffer
rosrun 3dto2d zbuffer_func_debug_node 1> /dev/null &
zbuffer_func_debug_node_pid=$!

# trap ctrl-c 结束gazebo
trap "kill_node $zbuffer_func_debug_node_pid" SIGINT

# 判断gazebo的pid还在不在跑
wait $zbuffer_func_debug_node_pid

# 不在跑等gazebo完全退出 就结束gazebo
if ! judge_running $zbuffer_func_debug_node_pid;then
    kill_node $arena1_launch_pid
fi

wait