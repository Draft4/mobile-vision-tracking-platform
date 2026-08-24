# 视觉模块

## 相机内参自动标定

标定工具使用项目现有的海康工业相机接口，适配 **8×11 个方格、单格边长
15 mm** 的棋盘（OpenCV 检测 7×10 个内部角点）。安装海康 MVS SDK、OpenCV、
NumPy 和 PyYAML，连接相机并确认 `vision/configs/config.yaml` 中的采集参数后运行：

```powershell
python -m vision.src.calibrate_camera_intrinsics
```

程序会实时显示检测角点和采样进度，不需要手动拍照。标定时请保持相机的分辨率、
焦距和对焦状态不变，并缓慢移动棋盘，使它出现在画面中心、四周、不同距离，且
包含正视、上下倾斜和左右倾斜的视角。模糊、过小、靠近画面边缘或与已有样本
过于相似的画面不会被采纳。按 `q`、`Esc` 或 `Ctrl+C` 可取消，取消时不写结果。

采样与视角覆盖达到要求后，程序会自动计算内参、剔除重投影误差异常的样本，
并在总体 RMS 重投影误差不超过 1 像素时保存到：

```text
vision/configs/camera_intrinsics.json
```

结果包含图像尺寸、`fx/fy/cx/cy`、3×3 相机矩阵、按
`[k1, k2, p1, p2, k3]` 排列的畸变系数、棋盘规格、有效样本数、重投影误差和
UTC 标定时间。默认不会覆盖已有结果；确认需要重新标定时使用：

```powershell
python -m vision.src.calibrate_camera_intrinsics --force
```

也可通过 `--config` 和 `--output` 指定其他配置或输出位置。当前 JSON 只作为完整
标定结果保存，不会自动修改 `config.yaml`；应将其中的 `_fx/_fy/_cx/_cy` 和五项
畸变系数手动写入配置，后续位姿和激光像素解算会直接读取这些配置值。

## 激光外参离线标定

激光外参工具使用 V2 亮色内区四角估计靶面姿态，再由用户在放大窗口中点击
激光点。每张图片点击 5 次并取中位数；全部有效命中点经过像素残差共识筛选和
三维直线拟合，最终得到相机光学坐标系中的激光直线。先采集原始 PNG：

```powershell
python -m vision.src.capture_camera_images `
  --output-dir vision/datasets/laser_calibration_images `
  --format png `
  --prefix laser
```

建议在 1～5 m 范围内采集至少 15 张，覆盖近、中、远距离，并保证相机、激光、
分辨率、焦距和对焦状态不变。`camera.target_W/target_L` 必须对应 V2 检测的亮色
内区尺寸，而不是 A4 外缘或黑框外沿。完成精确相机标定后，将同分辨率的
`_fx/_fy/_cx/_cy` 和 `[k1,k2,p1,p2,k3]` 写入 `config.yaml`，再运行：

```powershell
python -m vision.src.calibrate_laser_extrinsics
```

全图窗口左键选择激光附近区域，放大窗口左键添加点击、右键撤销最后一次点击；
`Enter` 确认，`R` 重做当前图片，`B` 返回全图重选，`S` 跳过，`Q` 或 `Esc`
取消整次标定。点击结果只在正常完成时统一保存，中途取消不会写入或覆盖结果文件。

默认结果写入 `vision/configs/laser_extrinsics.json`。文件包含每张图片的角点、
靶标姿态、人工点击、三维命中点、内点状态、拟合/留一像素误差以及最终
`origin_camera_m`、`direction_camera`。结果文件已存在时需使用 `--force` 才会
覆盖。质量不达标时仍会保存外参并返回退出码 2，但终端和 JSON 会标记警告，
不应直接写入正式配置；几何上无法拟合时只保存失败报告。

输出的 `origin_camera_m` 是拟合直线上距相机光心最近的点，并非物理出射口；它
与单位方向 `direction_camera` 共同定义激光直线，可手动复制到 `laser` 配置段。

## 无人机图片采集

安装海康 MVS SDK、连接 USB 工业相机并确认 `vision/configs/config.yaml` 中的
`camera` 参数后，可实时预览画面并按键保存用于数据集制作的图片：

```powershell
python -m vision.src.capture_camera_images
```

先单击预览窗口使其获得键盘焦点；按 `p` 保存当前帧，按 `q`、`Esc` 或
`Ctrl+C` 退出。预览画面可能为适应窗口而等比例缩小，但保存的是相机返回的原始
分辨率图像，不包含文字或检测标记。图片默认以高质量 JPG 保存到
`vision/datasets/captured_images/`，该目录已被 `.gitignore` 排除，不会进入
Git 历史。

可以指定输出目录，或使用 PNG 无损保存：

```powershell
python -m vision.src.capture_camera_images --output-dir D:/drone_images --format png
```

可通过 `--prefix` 指定输出文件名前缀；只允许字母、数字、下划线和连字符。

## 无人机视频采集

安装海康 MVS SDK、连接 USB 工业相机并确认 `vision/configs/config.yaml` 中的
`camera` 参数后，可直接录制用于 YOLO 数据集制作的无人机视频：

```powershell
python -m vision.src.record_drone_video
```

默认视频保存到 `vision/recordings/drone_日期_时间.mp4`。预览窗口只用于观察，
其中的录像状态文字不会写入视频；按 `q`、`Esc` 或 `Ctrl+C` 会正常结束录像并
释放相机。`vision/recordings/` 和常见视频格式已被 `.gitignore` 排除，录制结果
不会进入 Git 历史。

可以指定录制时长和输出位置，例如录制 60 秒：

```powershell
python -m vision.src.record_drone_video --duration 60 --output vision/recordings/drone_01.mp4
```

无显示器运行时使用 `--no-preview`；通过 `--fps` 可以覆盖视频帧率，通过
`--codec` 可以指定四字符 OpenCV 编码器。默认输出帧率为 80 FPS，默认编码器为
适用于 MP4 的 `mp4v`。输出帧率应尽量接近相机实际采集帧率；设置过高会使视频
播放速度变快，可以先通过实时检测画面中的 `Camera FPS` 确认实际值。脚本直接
逐帧调用 `HikCamera.grab()`，不经过会丢弃旧帧的实时检测缓冲区。停止后会报告
实际采集平均帧率及由相机帧号发现的跳帧数，便于检查 USB 带宽、曝光和帧率配置。

## 无人机实时检测、单目标跟踪与角误差

`drone_detector.py` 将模型推理结果转换为后端无关的检测候选，
`drone_tracker.py` 负责单目标关联，并使用相机内参计算目标相对当前相机视轴的
偏航与仰角误差。视觉层只返回测量结果，不直接发送电机命令，也不包含 PID、
滤波、运动预测或云台限位逻辑。

当前 Windows 验证后端使用 `vision/models/best.pt` 和 NVIDIA CUDA 设备 0。安装与
模型训练环境匹配的 `ultralytics==8.4.121`，连接工业相机后运行：

```powershell
python -m vision.test.realtime_drone_tracking
```

程序以最新帧方式运行推理，画面会显示所有候选框、当前锁定框、相机标定主点、
像素脱靶量、角误差、相机/检测帧率、当前延迟和滚动 P95 延迟。按 `q`、`Esc` 或
`Ctrl+C` 退出。首帧从置信度至少为 0.50 的候选中选择离标定主点最近的目标；
锁定后按前一帧目标中心进行轻量时序关联，并允许置信度降低至 0.25。

公共输出由 `DroneTrackingResult` 表示，核心字段包括：

- `found`、`track_id`、`frame_id` 和采集 `timestamp`；
- 原图坐标中的 `bbox_xyxy`、`center`、`pixel_error_u/v`；
- 弧度单位的 `yaw_error_rad` 和 `elevation_error_rad`。

目标第一次漏检时便返回 `found=False`，检测框、像素误差和角误差均为 `None`，
调用方不得继续使用上一帧角度。跟踪器会在内部保留锁定 0.5 秒；目标及时重现时
沿用原 `track_id`，超时后重新选择目标并分配新编号。角度符号遵循公共坐标约定：
目标在画面左侧时偏航为正，目标在画面上方时仰角为正。

角度解算会严格检查运行画面尺寸是否等于 `camera.calibrated_image_width/height`
（当前为 1280×1024），不一致时直接报错，避免错误缩放内参。第一版暂不应用畸变
系数，并假定相机视轴与云台机械零位方向对齐；进入闭环前仍需完成相机—云台外参、
电机方向、机械限位和低速安全联调。性能验证目标为检测处理不低于 20 Hz、P95
端到端延迟不高于 100 ms；实际结果以演示画面和目标 Windows 设备实测为准。

后续部署 RK3588 时新增实现同一 `detect(frame)` 接口的 RKNN 后端，并确保输出框
映射回原始图像坐标即可；上层锁定和角度解算无需依赖 Ultralytics 或 PyTorch。

## 靶标检测方法

### V1：外框轮廓法（实验基线）

- 实现：`vision/src/outer_frame_target_detector.py`
- 类名：`OuterFrameTargetDetector`
- 接口：`detect(FramePacket) -> TargetDetection`
- 状态：保留用于对照实验、参数验证和明亮背景场景，不作为最终纸质靶标方案。

该方法依次执行反色 Otsu 二值化、形态学处理、带子轮廓的黑色外框查找、四边形拟合和内外区域亮度评分。配置继续使用 `config.yaml` 中的 `target` 节。当前形态学处理先执行开运算以减少黑框与邻近暗色区域的细小粘连，再执行闭运算修复边框缺口；如果原始二值图已经干净，可以将对应迭代次数设为 `0`。

屏幕目标通常具有均匀的发光背景，黑框容易在反色二值图中形成独立、完整的白色环，因此识别效果较好。打印靶标依靠环境反射光；当黑框直接接触暗背景时，两者会在二值图中粘连，外框不再是独立矩形轮廓。打印边框较细、断裂、光照不均，或者内部红色圆环变暗并参与轮廓，也可能导致闭环、子轮廓或四边形检查失败。

面积、子轮廓、面积比例和四边形检查都发生在最终评分之前。因此，如果目标在这些阶段被过滤，仅降低 `min_confidence` 或亮度评分门槛不会改善识别。

静态图片测试：

```powershell
python -m vision.test.test_outer_frame_target_detector
```

安装海康 MVS SDK 并连接相机后，可运行实时测试：

```powershell
python -m vision.test.realtime_outer_frame_target_detection
```

实时程序使用独立线程持续采集最新画面，在主线程中绘制靶标四边形、四个角点、
中心、置信度和处理延迟。画面左上角还会显示相机实际采集帧率和检测处理帧率，
刷新周期由 `camera.fps_window_seconds` 配置。按 `q`、`Esc` 或 `Ctrl+C` 退出。

### V2：内区矩形法（实验版本）

实现使用 `vision/src/inner_region_target_detector.py` 和 `InnerRegionTargetDetector`，并保持 `detect(FramePacket) -> TargetDetection` 接口不变，以便与 V1 并行对比。

V2 不再依赖黑框外侧背景，而是：

1. 检测黑框内部的亮色矩形区域；
2. 允许内部圆环形成少量暗色孔洞；
3. 从内区四边形向外扩展窄带，验证该区域具有足够高的暗色像素比例；
4. 不检查暗色带以外的背景；
5. 使用屏幕、纸张、暗背景和复杂背景样本与 V1 做同条件对比。

面积、画面边缘、凸四边形、内部白色比例、外侧黑色比例和最低内外灰度差属于
硬性过滤条件；通过过滤的候选再根据面积、内部白色比例、外侧黑色比例和内外
灰度差进行加权评分。和 V1 相比，V2 不要求黑框成为独立闭环，因此更适合黑框
与暗背景相邻的场景。普通亮色矩形紧邻暗色背景时仍可能满足这些特征，后续需要
结合内部圆环或其他靶标特征进一步降低误检。

V2 使用 `config.yaml` 中独立的 `target_v2` 配置段。当前 V1、V2 静态测试均使用
`vision/data/target_test_1.jpg`，便于在同一场景下对比。静态图片测试：

```powershell
python -m vision.test.test_inner_region_target_detector
```

安装海康 MVS SDK 并连接相机后，可运行实时测试：

```powershell
python -m vision.test.realtime_inner_region_target_detection
```

实时程序绘制内区四边形、四个角点、中心、置信度和处理延迟，并在画面左上角
显示相机实际采集帧率和检测处理帧率。按 `q`、`Esc` 或 `Ctrl+C` 退出。
