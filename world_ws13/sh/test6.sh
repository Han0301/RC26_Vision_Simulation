#!/bin/bash

trap "exit 0" SIGTSTP

source /home/awwsome/RC/world_ws5/devel/setup.bash

kill_node()
{
    local pid=$1
    # echo $pid
    if judge_running $pid;then
        kill $pid && wait $pid
        echo "进程 $pid 已杀死"
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
    
roslaunch zwei arena1.launch 1> /dev/null &
launch_pid=$!

sleep 5

rosrun wpr_simulation keyboard_vel_ctrl

kill_node $launch_pid && wait $launch_pid

wait