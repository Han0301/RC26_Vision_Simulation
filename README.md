# Ten-Vision
world_ws5: 仿真中测试的远古版本

world_ws6: 仿真中重构的一版（正式版）

world_ws7: 仿真中测试hsv相关功能的测试版

world_ws8: 仿真中重构set_box_lists函数的测试版本

world_ws9: 仿真中的旗舰版

world_ws10: 仿真中自动录制数据集



yolo11_Custom_12roi: 针对12个位置的roi照片定制的yolo模型, 用于筛空(3分类)


yolo11_zb_12roi_c2:结合zb的遮挡处理, 输入全局图像和r,t生成roi进行循环推理


yolo11_Custom_pointsize: 结合pointsize进行损失加权, 让模型敢给丰富图像信息(即高pointsize)的图片高置信度

yolo11_Custom_atten: 引入注意力模块, yolo11_Custom定制筛空模型

yolo11_Custom_atten2: 在原有的基础上, 采用12roi+单张roi共同训练的方式, 防止模型过于依赖注意力模块, 不在局限单张roi的场景, 支持任意数量图片的输入

camera_ws: 提取方块面的位姿

merge_ws: 上车代码, 后缀表示版本
