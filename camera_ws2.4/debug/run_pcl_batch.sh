#!/bin/bash
# 批量运行所有距离的 bag 文件，收集滤波后点数统计

source /opt/ros/noetic/setup.bash
source /home/h/RC2026/camera_ws2.4/devel/setup.bash

EXE=/home/h/RC2026/camera_ws2.4/devel/lib/camera/test_node
BAG_DIR=/home/h
OUTPUT_CSV=/tmp/kfs_pcl_summary.csv

# 汇总CSV头
echo "组别,帧数,点数均值,点数σ,点数Δ,点数P95-P5,点数max,点数min" > "$OUTPUT_CSV"

# 距离文件夹列表 (按距离排序)
BAGS=(
    "1.05m距离面静态1"
    "1.05m距离面静态2"
    "1.2m距离面静态1"
    "1.2m距离面静态2"
    "1.2m距离面静态3"
    "1.2m距离面静态4"
    "1.35m距离面静态1"
    "1.35m距离面静态2"
    "1.35m距离面静态3"
    "1.35m距离面静态4"
    "1.5m距离面静态1"
    "1.5m距离面静态2"
    "1.5m距离面静态3"
    "1.5m距离面静态4"
    "1.8m距离面静态1"
    "1.8m距离面静态2"
    "1.8m距离面静态3"
    "1.8m距离面静态4"
    "2.1m距离面静态1"
    "2.1m距离面静态2"
    "2.1m距离面静态3"
    "2.1m距离面静态4"
    "2.4m距离面静态1"
    "2.4m距离面静态2"
    "2.4m距离面静态3"
    "2.4m距离面静态4"
    "2.65m距离面静态1"
    "2.65m距离面静态2"
    "2.65m距离面静态3"
    "2.65m距离面静态4"
)

for name in "${BAGS[@]}"; do
    bag_path="${BAG_DIR}/${name}.bag"
    tmp_dir="/tmp/kfs_${name}/"
    pcl_file="${tmp_dir}pcl_count.txt"

    if [ ! -f "$bag_path" ]; then
        echo "⚠ 跳过(文件不存在): $bag_path"
        continue
    fi

    echo "▶ 处理: $name ..."
    # 静默运行 (stdout/stderr 丢弃)
    timeout 120 "$EXE" "$bag_path" > /dev/null 2>&1

    if [ -f "$pcl_file" ]; then
        # 用python计算统计值
        stats=$(python3 -c "
import sys
with open('$pcl_file') as f:
    vals = [float(l.strip()) for l in f if l.strip()]
n = len(vals)
if n == 0:
    print('0,0,0,0,0,0,0')
else:
    vals.sort()
    avg = sum(vals)/n
    var = sum((v-avg)**2 for v in vals)/n
    std = var**0.5
    delta = vals[-1] - vals[0]
    p5 = vals[int(n*0.05)]
    p95 = vals[int(n*0.95)]
    print(f'{n},{avg:.1f},{std:.1f},{delta:.1f},{p95-p5:.1f},{vals[-1]:.1f},{vals[0]:.1f}')
")
        echo "$name,$stats" >> "$OUTPUT_CSV"
        echo "  ✓ 完成, 帧数: $(echo $stats | cut -d, -f1)"
    else
        echo "  ✗ 失败(无输出文件)"
        echo "$name,0,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL" >> "$OUTPUT_CSV"
    fi
done

echo ""
echo "========================================="
echo "汇总结果: $OUTPUT_CSV"
echo "========================================="
cat "$OUTPUT_CSV"
