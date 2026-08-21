# 视觉核心源码

本目录保存视觉子系统可复用的采集、帧传递和靶标检测代码。可视化演示与硬件
检查脚本放在 `vision/test/`，相机和检测参数放在 `vision/configs/config.yaml`。
更上层的运行方法及靶标算法说明见 [`vision/README.md`](../README.md)。

## 文件说明

| 文件或目录 | 作用 | 当前状态 |
| --- | --- | --- |
| `data_types.py` | 定义采集、靶标检测和无人机跟踪共用的数据结构 | 公共数据接口 |
| `hik_camera.py` | 封装海康 USB 工业相机的配置、同步取流和资源释放 | 已用于实机采集 |
| `calibrate_camera_intrinsics.py` | 自动筛选棋盘视角并计算、保存相机内参与畸变 | 可运行的标定工具 |
| `capture_camera_images.py` | 实时预览工业相机画面并按键保存原始分辨率图片 | 可运行的采集工具 |
| `record_drone_video.py` | 直接调用工业相机逐帧录制无人机训练视频 | 可运行的采集工具 |
| `latest_frame_buffer.py` | 在线程间传递最新一帧，主动丢弃来不及处理的旧帧 | 实时低延迟方案 |
| `camera_worker.py` | 在后台线程中管理相机生命周期并持续发布帧 | 已用于实时测试 |
| `outer_frame_target_detector.py` | V1 外框轮廓法靶标检测器 | 可运行的实验基线 |
| `inner_region_target_detector.py` | V2 内区矩形法靶标检测器 | 可运行的实验版本 |
| `drone_detector.py` | 定义后端无关的无人机检测接口及 Ultralytics CUDA 实现 | Windows 实机验证版本 |
| `drone_tracker.py` | 单目标时序关联、丢失管理和相机视轴角误差解算 | 后端无关核心逻辑 |
| `target_pose_estimator.py` | 根据靶标四角和相机内参解算 `R_CT`、`t_CT` | 位姿解算工具 |
| `laser_pixel_predictor.py` | 根据激光外参和靶标位姿预测激光点像素坐标 | 几何预测工具 |
| `MvImport/` | 海康 MVS SDK 的 Python 绑定 | 第三方随附代码，不在此处维护 |

V2 当前用于检测黑框内部的亮色矩形，并验证候选四边形外侧的暗色窄带；算法说明
和静态、实时测试命令记录在上级 README 中。

## 数据流与职责

```text
HikCamera.grab()
        │ FramePacket
        ▼
CameraWorker ──publish──> LatestFrameBuffer
                                  │ 最新 FramePacket
                    ┌─────────────┼────────────────────────┐
                    ▼             ▼                        ▼
       OuterFrameTargetDetector  InnerRegion...  DroneDetectorBackend
                    │             │                        │ candidates
                    └──────┬──────┘                        ▼
                           ▼                         DroneTracker
                    TargetDetection                       │
                                                          ▼
                                                DroneTrackingResult
```

- `CameraWorker` 所在线程独占相机对象；检测与 OpenCV 窗口绘制在消费线程执行。
- `LatestFrameBuffer` 是单槽缓冲区，不是帧队列。消费者处理较慢时会跳帧，以
  保证拿到的画面尽可能新；录像和逐帧分析应另外使用队列。
- 两个检测器均为无帧间状态的单帧算法，使用相同输入输出接口；调用程序按实验
  需要选择其中一个，不应在同一实时循环中无目的地重复处理同一帧。
- `DroneDetectorBackend` 只负责输出原图坐标候选；`DroneTracker` 保存单目标锁定
  状态并输出角误差。替换成 RKNN 后端时不应把框关联或角度公式复制到后端中。
- `DroneTrackingResult.found=False` 表示当前帧没有新测量，角度字段为 `None`；即使
  短时丢失仍保留 `track_id`，控制端也不得复用上一帧角度。
- `FramePacket.timestamp` 当前由 `time.perf_counter()` 产生，单位为秒，只用于
  计算进程内时间间隔和延迟，不可作为日期时间或跨设备同步时间。
- 像素坐标遵循 OpenCV 约定：原点在左上角，`x/u` 向右，`y/v` 向下。三维坐标、
  云台方向和角度单位以 [`interface/coordinate-system.md`](../../interface/coordinate-system.md)
  为准。

## 相机生命周期

相机应按以下顺序使用：

```text
open -> start -> grab（循环）-> stop -> close
```

实时程序通常只需要启动和停止 `CameraWorker`，由它执行上述过程。`stop()` 会等待
采集线程退出并释放相机，同时永久关闭关联的 `LatestFrameBuffer`；重新启动整条
流水线时应创建新的缓冲区和工作线程实例。即使采集线程设为守护线程，程序退出前
仍应主动调用 `stop()`，避免相机句柄未正常释放。

## 依赖与运行条件

- Python、NumPy、OpenCV；读取 YAML 配置的入口还需要 PyYAML。
- 当前无人机 PT 推理后端需要与模型匹配的 `ultralytics==8.4.121`、PyTorch 和
  CUDA；配置默认使用设备 0 与 FP16。模型文件和具体运行库不进入 Git。
- 使用工业相机前，需要安装与系统及 Python 架构匹配的海康 MVS SDK 和运行库。
- `hik_camera.py` 当前打开枚举到的第一台 USB 相机，仅支持 Bayer BG8、Bayer GB8
  两种输入像素格式。
- `camera.gamma_enable/gamma_selector/gamma` 控制相机 gamma；启用时 selector 只
  支持 `User` 或 `sRGB`。`camera.pixel_format` 只支持 `BayerBG8` 或 `BayerGB8`，
  配置成其他格式会在访问硬件前报错。
- `MvImport/` 来源于海康机器人 MVS SDK 的 Python 绑定，当前版本为 `4.8.0.1`
  （以 `MvImport/__init__.py` 为准）。这些绑定文件未随附独立许可证，其使用和
  再分发遵循官方 MVS SDK 许可协议；仓库公开发布前需要再次确认再分发条件。
- `MvImport/` 视为 SDK 随附代码。除非升级 SDK 或修复明确的兼容问题，否则不要
  对其中生成的绑定文件做格式化或批量修改。

## 当前限制

- V1 检测器依赖黑色外框与背景分离，在暗背景和光照不均的纸质靶标场景中稳定性
  有限。详细原因与 V2 说明见 [`vision/README.md`](../README.md)。
- V2 检测器不依赖黑框外侧背景，但仅靠亮色四边形和外侧暗色窄带无法区分所有
  暗背景中的普通亮色矩形，现阶段仍属于实验算法。
- 无人机角度解算第一版忽略镜头畸变，并假定相机视轴与云台机械零位对齐；输出
  是目标相对当前相机光轴的角误差，不是云台绝对角度或可直接下发的电机指令。
