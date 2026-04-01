import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 你的数据
x_list = [0.11246416, 0.09716127, 0.10113878, 0.10883719, 0.10343845, 0.10765433]
y_list = [0.18696044, 0.18482668, 0.18139391, 0.19041498, 0.18342799, 0.18940471]

# 画布
fig, ax = plt.subplots(figsize=(8, 8))

# ===================== 标准直角坐标系（原点 0,0）=====================
ax.spines['left'].set_position('zero')    # Y轴固定在 x=0
ax.spines['bottom'].set_position('zero')  # X轴固定在 y=0
ax.spines['right'].set_visible(False)     # 隐藏右边框
ax.spines['top'].set_visible(False)       # 隐藏上边框

# ===================== 绘制 原点(0,0) 同心圆 =====================
# 半径适配你的数据大小（从小到大）
radii = [0.05, 0.10, 0.15, 0.20, 0.25]
for r in radii:
    circle = Circle((0, 0), radius=r, fill=False, edgecolor='blue', linewidth=1.5)
    ax.add_patch(circle)

# ===================== 绘制你的散点 =====================
ax.scatter(x_list, y_list, s=6, c='red', zorder=3)

# ===================== 关键：坐标轴范围（能看到原点+数据）=====================
ax.axis('equal')       # 强制正圆
ax.set_xlim(-0.3, 0.3)  # 缩小范围！适配你的数据
ax.set_ylim(-0.3, 0.3)
ax.grid(True, alpha=0.3)
ax.set_title('1022vans-bias', fontsize=12)

plt.show()
