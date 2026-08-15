# 视觉模块

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

实时程序使用独立线程持续采集最新画面，在主线程中绘制靶标四边形、四个角点、中心、置信度和处理延迟。按 `q`、`Esc` 或 `Ctrl+C` 退出。

### V2：内区矩形法（实验版本）

实现使用 `vision/src/inner_region_target_detector.py` 和 `InnerRegionTargetDetector`，并保持 `detect(FramePacket) -> TargetDetection` 接口不变，以便与 V1 并行对比。

V2 不再依赖黑框外侧背景，而是：

1. 检测黑框内部的亮色矩形区域；
2. 允许内部圆环形成少量暗色孔洞；
3. 从内区四边形向外扩展窄带，验证该区域具有足够高的暗色像素比例；
4. 不检查暗色带以外的背景；
5. 使用屏幕、纸张、暗背景和复杂背景样本与 V1 做同条件对比。

面积、画面边缘、凸四边形、内部白色比例和外侧黑色比例属于硬性过滤条件；
通过过滤的候选再根据面积、内部白色比例、外侧黑色比例和内外灰度差进行加权
评分。和 V1 相比，V2 不要求黑框成为独立闭环，因此更适合黑框与暗背景相邻的
场景。普通亮色矩形紧邻暗色背景时仍可能满足这些特征，后续需要结合内部圆环或
其他靶标特征进一步降低误检。

V2 使用 `config.yaml` 中独立的 `target_v2` 配置段。当前 V1、V2 静态测试均使用
`vision/data/target_test_1.jpg`，便于在同一场景下对比。静态图片测试：

```powershell
python -m vision.test.test_inner_region_target_detector
```

安装海康 MVS SDK 并连接相机后，可运行实时测试：

```powershell
python -m vision.test.realtime_inner_region_target_detection
```

实时程序绘制内区四边形、四个角点、中心、置信度和处理延迟。按 `q`、`Esc`
或 `Ctrl+C` 退出。
