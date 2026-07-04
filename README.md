# RC26 Vision Simulation — World Workspace

> 机器人竞赛视觉仿真工作区 · ROS Noetic · Z-buffer 感知管线 · 数据集自动录制

## 项目概述

本项目是 RC26 机器人竞赛的**视觉感知与数据集录制**工作区，核心是基于 **Z-buffer 的 3D→2D 映射遮挡处理管线**，用于仿真环境中机械臂视觉抓取。

### 核心功能包

| 包名 | 功能 |
|------|------|
| `3dto2d` | **核心感知算法**：Z-buffer 遮挡处理、HSV 目标检测、相机标定、PID 运动控制 |
| `rc_msgs` | 自定义 ROS 消息（感知结果、控制指令） |
| `wpr_simulation` | WPR 机器人仿真环境（传感器、控制、导航、机械臂） |
| `zwei` | Zwei 机器人模型 + **地图数据集**（`map1_add/` 含数百张地图配置） |

### 配套工具脚本

| 脚本 | 用途 |
|------|------|
| `random_create_map.py` | 随机生成地图配置（12 点位规则），用于训练数据增强 |
| `change_data.py` | URDF 日志 → C++ 二维 vector 格式转换 |
| `change_debug_to_data.py` | 调试图像自动裁剪、重命名并整理为数据集 |

## 使用方法

### 环境要求

- **Ubuntu 20.04** + ROS Noetic (Neotic Ninjemys)
- OpenCV (`cv_bridge`)、PCL (`pcl_ros`)、Eigen3
- 依赖安装：
```bash
sudo apt install ros-noetic-desktop-full
sudo apt install ros-noetic-vision-msgs ros-noetic-pcl-ros
```

### 编译

```bash
cd /path/to/world_ws
catkin_make
# 或使用 catkin build
source devel/setup.bash
```

### 运行感知管线

**主入口 — Z-buffer 感知节点**（推荐使用 launch 文件启动）：

```bash
roslaunch 3dto2d zbuffer_func.launch
```

launch 文件可配置参数：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `save_root_path` | 数据集录制保存根目录 | `/home/awwsome/datasets/juanZhou_gazebo20/` |
| `map_file_path` | 地图文件路径 | `$(find zwei)/map1_add/` |
| `txt_root_name` | 地图文件名前缀 | `map_` |
| `image_root_name` | 图像目录名 | `juanZhou_gazebo20` |

也可直接运行节点：

```bash
rosrun 3dto2d zbuffer_func_node
```

### 仿真环境

WPR 机器人仿真提供多种场景启动方式：

```bash
# RoboCup 标准赛场
roslaunch wpr_simulation wpb_stage_robocup.launch

# 走廊环境
roslaunch wpr_simulation wpb_stage_corridor.launch

# 简单桌面测试
roslaunch wpr_simulation wpb_table.launch

# 查看完整场景列表：
ls src/wpr_simulation/launch/
```

### 数据工具

```bash
# 随机生成地图（用于训练数据增强）
python3 src/random_create_map.py

# URDF 日志转 C++ 格式
python3 src/change_data.py <input_log> <output_txt>

# 调试图像转数据集
python3 src/change_debug_to_data.py
```

## 稳定版本

| 版本 | 标签 | 描述 |
|------|------|------|
| **world_ws9** | `world_ws/v9.0` | 旗舰版 — 完整的 Z-buffer 感知管线 + 相机标定 |
| **world_ws13** | `world_ws/v13.0` | 数据集录制版 — BaseMoveController + PID 控制 |
| **world_ws13.5** | `world_ws/v13.5` | 最终版 — 大规模地图数据更新与最终优化 |

> 从 [GitHub Releases](https://github.com/Han0301/RC26_Vision_Simulation/releases) 下载对应版本的源码压缩包。

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

### world_ws10 — 数据集自动录制版
- **新增** `change_debug_to_data.py`: 调试图像转数据集脚本，支持自动标注
- **新增** `random_create_map.py`: 随机地图生成器，用于训练数据增强
- **新增** `move_controller.cpp/h`: 运动控制模块，支持自动遍历采集
- **新增** `zbuffer_func.launch`: Z-buffer 功能的便捷启动文件
- **新增** `label_100.txt`/`label_150.txt`: 标注数据文件
- **新增** `map1_add`: 扩展地图配置文件
- **移除** `camera_calibration.cpp`（整合至其他模块）
- **移除** `debug/` 调试图像目录（不再需要手动调试）
- **优化** `occlusion_handing` 遮挡参数的标定精度
- **优化** `method_math` 数学工具函数性能

### world_ws11 — 偏差与边缘处理引入版
- **新增** `deviation_handing.cpp/h`: 偏差处理模块，补偿检测偏差
- **新增** `edge_handing.cpp`: 边缘检测处理，提升边缘定位精度
- **新增** `hsv_handing.cpp/h`: HSV 颜色检测独立模块
- **新增** `zbuffer_test1.launch`: 测试用启动文件
- **新增** `debug/` 调试图像目录
- **新增** 顶层 `CMakeLists.txt` 工作区构建配置
- **移除** `move_controller` 运动控制模块（迁移至其他工作区）
- **移除** 数据集相关脚本（`change_debug_to_data.py`, `random_create_map.py`）
- **优化** `occlusion_handing` 遮挡处理算法
- **优化** `zbuffer_func` 主节点性能

### world_ws12 — PID 控制与运动恢复版
- **新增** `PID.cpp`: PID 控制器模块，提升运动控制精度
- **新增** `basemove2.cpp`: 基础运动控制 v2
- **恢复** `camera_calibration.cpp`: 相机标定功能重新整合
- **恢复** `move_controller.cpp/h`: 运动控制模块回迁
- **恢复** 数据集相关脚本（`change_debug_to_data.py`, `random_create_map.py`）
- **恢复** `zbuffer_func.launch` 启动文件
- **移除** `deviation_handing` 偏差处理（整合至其他模块）
- **移除** `edge_handing` / `hsv_handing` 边缘与 HSV 模块
- **优化** `occlusion_handing` 遮挡处理
- **优化** `zbuffer_func` 主节点融合运动控制

### world_ws13 — BaseMoveController 重构版
- **新增** `BaseMoveController.cpp/h`: 重构基础运动控制器，替代旧版 basemove2
- **新增** `change_data.py`: 数据格式转换与处理脚本
- **新增** `01_400.txt`/`label_400.txt`: 更新标注数据集（400 级）
- **移除** `basemove2.cpp`: 被 BaseMoveController 替代
- **移除** `label_100.txt`/`label_150.txt`: 旧版标注数据
- **优化** `occlusion_handing` 遮挡处理算法精度
- **优化** `PID.cpp` 控制器参数
- **优化** `zbuffer_func` 主节点性能与稳定性

### world_ws13.5 — 地图数据更新与最终优化版
- **优化** `zbuffer_func.cpp`: 主节点性能微调与稳定性改进
- **更新** `map_*.txt` 地图数据集（大规模地图数据更新）
- **更新** `flag.txt` 地图标志位配置
- **更新** `wpr_simulation.zip` 仿真环境包

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
