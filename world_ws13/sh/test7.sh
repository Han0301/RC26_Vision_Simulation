#!/bin/bash

trap "exit 0" SIGTSTP

source /home/awwsome/RC/world_ws5/devel/setup.bash

kill_node()
{
    local pid=$1
    # echo $pid
    if judge_running $pid;then
        kill $pid >/dev/null 2>&1 && wait $pid
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
    if ps -p "$pid" >/dev/null 2>&1; then
        return 0
    else 
        return 1
    fi
}

launch_name="map"
launch=".launch"
launch_num=166

for i in $(seq 1 $launch_num);do 
    name=$launch_name$i$launch
    echo "运行 $name"
    roslaunch zwei $name 1> /dev/null &
    launch_pid=$!

    sleep 5

    echo "正在杀死 $name"
    kill_node $launch_pid
    sleep 2

done

wait