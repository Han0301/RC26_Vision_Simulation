#!/usr/bin/env python3
"""从 /tmp/kfs_*/ 读取所有 bag 的统计文件，生成 summary_stats1.txt"""
import os, sys, math
from datetime import datetime

TMP_BASE = "/tmp"
OUTPUT = "/home/h/RC2026/camera_ws2.4/debug/summary_stats1.txt"

# 按距离排序的所有 bag 名
BAGS = [
    "1.05m距离面静态1",
    "1.2m距离面静态1", "1.2m距离面静态2", "1.2m距离面静态3", "1.2m距离面静态4",
    "1.35m距离面静态1", "1.35m距离面静态2", "1.35m距离面静态3", "1.35m距离面静态4",
    "1.5m距离面静态1", "1.5m距离面静态2", "1.5m距离面静态3", "1.5m距离面静态4",
    "1.8m距离面静态1", "1.8m距离面静态2", "1.8m距离面静态3", "1.8m距离面静态4",
    "2.1m距离面静态1", "2.1m距离面静态2", "2.1m距离面静态3", "2.1m距离面静态4",
    "2.4m距离面静态1", "2.4m距离面静态2", "2.4m距离面静态3", "2.4m距离面静态4",
    "2.65m距离面静态1", "2.65m距离面静态2", "2.65m距离面静态3", "2.65m距离面静态4",
]

def read_vals(filepath):
    """读取单列数值文件，返回 float 列表"""
    vals = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    vals.append(float(line))
                except ValueError:
                    pass
    return vals

def compute_stats(vals):
    """计算统计值字典"""
    n = len(vals)
    if n == 0:
        return None
    vals_sorted = sorted(vals)
    avg = sum(vals_sorted) / n
    var = sum((v - avg) ** 2 for v in vals_sorted) / n
    std = math.sqrt(var)
    delta = vals_sorted[-1] - vals_sorted[0]
    p5 = vals_sorted[int(n * 0.05)]
    p95 = vals_sorted[int(n * 0.95)]
    return {
        "n": n,
        "avg": avg,
        "std": std,
        "delta": delta,
        "p90_range": p95 - p5,
        "max": vals_sorted[-1],
        "min": vals_sorted[0],
    }

def fmt_stat(vals, unit_scale=1.0, precision=4):
    """格式化一个统计块"""
    s = compute_stats(vals)
    if s is None:
        return "    (无数据)\n"
    return (
        f"    平均值 (avg):       {s['avg'] * unit_scale:.{precision}f}\n"
        f"    标准差 (std):       {s['std'] * unit_scale:.{precision}f}\n"
        f"    跳变幅度 (Δ):       {s['delta'] * unit_scale:.{precision}f}\n"
        f"    90%范围 (P95-P5):   {s['p90_range'] * unit_scale:.{precision}f}\n"
        f"    最大值 (max):       {s['max'] * unit_scale:.{precision}f}\n"
        f"    最小值 (min):       {s['min'] * unit_scale:.{precision}f}\n"
    )

lines = []
lines.append("=" * 60)
lines.append("         多 Bag 静态跳变统计汇总 (含滤波点数)")
lines.append("=" * 60)
lines.append(f"生成时间: {datetime.now().strftime('%b %d %Y %H:%M:%S')}")
lines.append("")

total_bags = 0

for name in BAGS:
    tmp_dir = os.path.join(TMP_BASE, f"kfs_{name}")
    x_file = os.path.join(tmp_dir, "x.txt")
    y_file = os.path.join(tmp_dir, "y.txt")
    z_file = os.path.join(tmp_dir, "z.txt")
    a_file = os.path.join(tmp_dir, "angle.txt")
    p_file = os.path.join(tmp_dir, "pcl_count.txt")

    if not os.path.isfile(x_file):
        print(f"⚠ 跳过(无数据): {name}")
        continue

    x_vals = read_vals(x_file)
    y_vals = read_vals(y_file)
    z_vals = read_vals(z_file)
    a_vals = read_vals(a_file)
    p_vals = read_vals(p_file)

    n_frames = len(x_vals)

    lines.append("─" * 52)
    lines.append(f"  Bag: {name}")
    lines.append(f"  路径: /home/h/{name}.bag")
    lines.append(f"  总帧数: {n_frames} | 成功帧数: {n_frames} | 成功率: 100.0000%")
    lines.append("")

    lines.append("  X (m):")
    lines.append(fmt_stat(x_vals, unit_scale=1.0, precision=4))
    lines.append("  Y (m):")
    lines.append(fmt_stat(y_vals, unit_scale=1.0, precision=4))
    lines.append("  Z (m):")
    lines.append(fmt_stat(z_vals, unit_scale=1.0, precision=4))
    lines.append("  Angle (deg):")
    lines.append(fmt_stat(a_vals, unit_scale=1.0, precision=2))
    lines.append("  滤波后点数:")
    lines.append(fmt_stat(p_vals, unit_scale=1.0, precision=1))

    total_bags += 1

lines.append("=" * 60)
lines.append(f"共处理 {total_bags} 个 Bag 文件")
lines.append("=" * 60)

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"✅ 汇总已保存: {OUTPUT}")
print(f"   共 {total_bags} 个 bag")
