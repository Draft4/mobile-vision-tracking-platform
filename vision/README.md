# 视觉模块

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
