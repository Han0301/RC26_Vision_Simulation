# RC26 Vision Simulation — World Workspace

> 机器人竞赛视觉仿真工作区，基于 ROS Noetic，用于 3D 感知、2D 映射与机械臂视觉抓取。

## 项目概述

本项目是 RC26 机器人竞赛的视觉仿真系统，包含以下核心功能包：

| 包名 | 功能 |
|------|------|
| `3dto2d` | 3D 空间到 2D 图像的关键点映射、Z-buffer 遮挡处理、HSV 目标检测 |
| `rc_msgs` | 自定义 ROS 消息定义 |
| `wpr_simulation` | WPR 机器人仿真（传感器、控制、导航） |
| `zwei` | Zwei 机器人模型与仿真配置 |

## 使用方法

### 环境要求

- Ubuntu 20.04 + ROS Noetic
- OpenCV + PCL (点云库)

```bash
# 安装依赖
sudo apt install ros-noetic-desktop-full
sudo apt install ros-noetic-vision-msgs ros-noetic-pcl-ros
```

### 编译

```bash
cd world_ws
catkin_make
# 或使用 catkin build
```

### 运行示例

```bash
source devel/setup.bash

# 启动 WPR 仿真环境
roslaunch wpr_simulation wpb_sim.launch

# 启动 Z-buffer 感知节点
rosrun 3dto2d zbuffer_func_node

# HSV 颜色检测
rosrun 3dto2d hsv_detection_node
```

## 版本迭代日志

### world_ws5 — 远古初始版
- 初始仿真工作区搭建，基础 3D→2D 坐标映射
- Z-buffer 遮挡处理的早期试验
- HSV 空框过滤原型

### world_ws6 — 正式重构版
- 重构 `3dto2d` 包目录，引入 `package/` 模块化组织
- 新增 `world_to_camera` 外参标定工具
- 新增 `camera_calibration.h` 相机内参标定
- 删除大量旧版试验文件（`t2dto3d*`、`zbuffer_func_final*` 等），清理代码库
- 优化 Z-buffer 简化算法性能

### world_ws7 — HSV 功能测试版
- **新增** `zbuffer_func_node` 主节点，集成完整 Z-buffer 管线
- 引入 `3dto2d_ten_utils` 通用工具库
- 新增大量调试图像输出（`debug/`），可视化 HSV 检测中间结果
- 重构 `set_box_lists` 区域选择逻辑，增加有效点阈值判断
- 优化 `update_debug_image` 多 ROI 拼接显示

### world_ws8 — set_box_lists 重构版
- **移除** `debug_hsv` 调试模块（不再单独调试）
- **移除** `zbuffer_simplify` 旧版简化算法
- **新增** `occlusion_handing` 遮挡处理模块
- **修改** `zbuffer_func.cpp`，集成新的遮挡处理

### world_ws9 — 旗舰版
- **新增** `camera_calibration.cpp` 相机标定实现（原仅头文件）
- **优化** `occlusion_handing` 遮挡处理算法
  - 重构宏定义，参数可配置化
  - 新增 `roi_valid_flag` 有效区域标志位
  - 引入 Eigen 线性代数库支持
  - 增加现代 C++ 随机数生成与异常处理
- **增强** `method_math` 工具库
  - 新增 PCD 点云文件读写支持
  - 新增 `tvectovector3d` 平移向量转换
  - 新增 `getpath` 路径搜索功能
  - 新增 `readPoseFromTxt` 位姿文件解析
- **编译优化** 启用 `-O2 -Wall -Wextra` 编译选项
- 优化 `world_to_camera` 外参转换精度

### world_ws10 — 数据集录制版
*待更新*

### world_ws11
*待更新*

### world_ws12
*待更新*

### world_ws13
*待更新*

### world_ws13.5
*待更新*

## 目录结构

```
world_ws/
├── src/
│   ├── 3dto2d/          # 3D→2D 感知包（核心算法）
│   │   ├── src/         # 源文件
│   │   ├── include/     # 头文件
│   │   ├── launch/      # 启动文件
│   │   └── debug/       # 调试图像
│   ├── rc_msgs/         # 自定义 ROS 消息
│   ├── wpr_simulation/  # WPR 机器人仿真
│   └── zwei/            # Zwei 机器人模型
├── build/               # 编译产物
├── devel/               # 开发空间
└── README.md
```
