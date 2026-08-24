# 视觉测试数据

本目录只保存体积较小、可随代码提交的视觉测试样本，不用于存放完整数据集、
录制视频或批量测试输出。大型数据仍按项目根目录 README 的约定通过网盘或群文件
共享。

## 当前样本

### `target_test_1.jpg`

- 用途：对比 V1 外框轮廓法与 V2 内区矩形法的静态检测结果；
- 来源：项目成员使用项目相机实拍；
- 调用入口：`python -m vision.test.test_outer_frame_target_detector`、
  `python -m vision.test.test_inner_region_target_detector`；
- SHA-256：`F0E9C002BF5503B1F63467317B7BA29538E00F790117A663BA2D1AFF185D1353`。

新增样本时应说明来源、用途和许可，不得提交包含个人敏感信息或来源不明的图片。
