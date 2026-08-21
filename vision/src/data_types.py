"""视觉模块内部使用的公共数据结构。

采集、缓冲、检测等模块通过本文件中的数据类传递结果，避免使用含义不明的
元组或字典。图像及角点数组均遵循 OpenCV 像素坐标系：原点位于左上角，
``x/u`` 向右、``y/v`` 向下。

数据类不会自动复制 NumPy 数组。创建对象的一方应确保数组在使用期间不会被
相机 SDK 或其他线程覆写；当前 :class:`HikCamera` 会在发布前复制帧数据。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FramePacket:
    """系统内部传递的一帧图像及其采集元数据。

    Attributes:
        frame: OpenCV 图像数组。
        frame_id: 相机提供的单调递增帧编号，用于判断帧的新旧。
        timestamp: 采集端记录的时间戳，单位为秒。当前相机实现使用单调时钟，
            适合计算延迟和时间间隔，不表示日期或 UTC 时间。
    """

    frame: np.ndarray
    frame_id: int
    timestamp: float


@dataclass
class TargetDetection:
    """单帧靶标检测结果。

    Attributes:
        found: 是否得到满足全部过滤条件的靶标。
        corners: 四个靶标角点，形状为 ``(4, 2)``。当前检测器按左上角起始、
            图像坐标系顺时针排列；未检测到目标时为 ``None``。
        center: 靶标中心的 ``(x, y)`` 像素坐标；未检测到目标时为 ``None``。
        area: 检测轮廓的像素面积。
        confidence: 检测器给出的归一化评分，范围为 ``[0, 1]``。
    """

    found: bool
    corners: Optional[np.ndarray] = None
    center: Optional[tuple[float, float]] = None
    area: float = 0.0
    confidence: float = 0.0


@dataclass(frozen=True)
class DroneDetectionCandidate:
    """推理后端输出的一个无人机检测候选。

    Attributes:
        bbox_xyxy: 原始图像像素坐标中的 ``(x1, y1, x2, y2)`` 检测框。
        confidence: 检测置信度，范围为 ``[0, 1]``。
        class_id: 推理模型给出的类别编号。
        class_name: 推理模型给出的类别名称。
    """

    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> tuple[float, float]:
        """返回检测框的几何中心 ``(u, v)``。"""

        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class DroneTrackingResult:
    """一帧无人机锁定状态及其相对当前相机光轴的角误差。

    ``found=False`` 时本帧没有可用于控制的新测量，检测框、中心、像素误差和
    角误差均为 ``None``。短暂丢失期间 ``track_id`` 可继续保留，便于调用方
    区分“等待原目标重现”和“尚未锁定目标”；不得把空角度解释为零误差。

    角度单位为弧度，符号遵循公共坐标约定：偏航向左为正，仰角向上为正。
    """

    found: bool
    frame_id: int
    timestamp: float
    track_id: Optional[int] = None
    bbox_xyxy: Optional[tuple[float, float, float, float]] = None
    center: Optional[tuple[float, float]] = None
    confidence: float = 0.0
    pixel_error_u: Optional[float] = None
    pixel_error_v: Optional[float] = None
    yaw_error_rad: Optional[float] = None
    elevation_error_rad: Optional[float] = None
