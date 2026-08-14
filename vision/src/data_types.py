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
