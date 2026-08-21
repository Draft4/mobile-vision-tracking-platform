"""后端可替换的无人机检测接口及 Ultralytics CUDA 实现。

锁定和角度解算只依赖本模块输出的普通检测候选，不依赖 PyTorch 张量。后续在
RK3588 上部署时，应新增实现相同 ``detect()`` 接口的 RKNN 后端，并把检测框
映射回原始图像像素坐标；无需修改上层跟踪和几何代码。
"""

import math
from pathlib import Path
from typing import Protocol

import numpy as np

from .data_types import DroneDetectionCandidate


VISION_DIR = Path(__file__).resolve().parents[1]


class DroneDetectorBackend(Protocol):
    """无人机推理后端需要遵守的最小接口。"""

    def detect(self, frame: np.ndarray) -> list[DroneDetectionCandidate]:
        """检测一帧并返回原始图像坐标中的无人机候选。"""


class UltralyticsDroneDetector:
    """使用 Ultralytics ``best.pt`` 模型执行单帧无人机检测。"""

    def __init__(self, config: dict):
        detector_config = config["drone_tracking"]
        self._validate_config(detector_config)

        model_path = Path(detector_config["model_path"])
        if not model_path.is_absolute():
            model_path = VISION_DIR / model_path
        self.model_path = model_path.resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"无人机模型不存在：{self.model_path}")

        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "缺少 ultralytics；请安装与 best.pt 匹配的 ultralytics==8.4.121"
            ) from error

        self.device = detector_config["device"]
        self.precision = detector_config["precision"]
        self.image_size = int(detector_config["image_size"])
        self.confidence_threshold = float(
            detector_config["maintain_confidence"]
        )
        self.iou_threshold = float(detector_config["iou_threshold"])
        self.max_detections = int(detector_config["max_detections"])
        self.class_name = str(detector_config["class_name"])

        self.model = YOLO(str(self.model_path))
        names = self._normalise_names(self.model.names)
        matching_ids = [
            class_id
            for class_id, name in names.items()
            if name == self.class_name
        ]
        if not matching_ids:
            raise ValueError(
                f"模型类别中没有 {self.class_name!r}：{names}"
            )
        self.class_id = matching_ids[0]

    @staticmethod
    def _validate_config(detector_config: dict) -> None:
        """在加载大型模型前检查推理配置。"""

        if detector_config["backend"] != "ultralytics":
            raise ValueError("当前仅实现 backend=ultralytics")
        if int(detector_config["image_size"]) <= 0:
            raise ValueError("image_size 必须大于 0")
        if detector_config["precision"] not in {"fp16", "fp32"}:
            raise ValueError("precision 必须为 fp16 或 fp32")
        if not 0.0 <= float(detector_config["maintain_confidence"]) <= 1.0:
            raise ValueError("maintain_confidence 必须位于 [0, 1]")
        if not 0.0 <= float(detector_config["iou_threshold"]) <= 1.0:
            raise ValueError("iou_threshold 必须位于 [0, 1]")
        if int(detector_config["max_detections"]) <= 0:
            raise ValueError("max_detections 必须大于 0")

    @staticmethod
    def _normalise_names(names) -> dict[int, str]:
        """把 Ultralytics 的列表或字典类别表统一为字典。"""

        if isinstance(names, dict):
            return {int(class_id): str(name) for class_id, name in names.items()}
        return {class_id: str(name) for class_id, name in enumerate(names)}

    def detect(self, frame: np.ndarray) -> list[DroneDetectionCandidate]:
        """运行模型并把张量结果转换为后端无关的数据结构。

        当前模型使用项目录像脚本采集的数据训练。为保持训练和实时推理的数值
        通道顺序一致，这里直接传入 ``FramePacket.frame``，不额外交换 RGB/BGR。
        """

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("无人机检测输入必须是 H x W x 3 图像")
        if frame.dtype != np.uint8:
            raise ValueError("无人机检测输入必须是 uint8 图像")

        results = self.model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            classes=[self.class_id],
            device=self.device,
            quantize=self.precision,
            verbose=False,
        )
        if len(results) != 1:
            raise RuntimeError(f"单帧推理返回了 {len(results)} 个结果")

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy()
        height, width = frame.shape[:2]

        candidates: list[DroneDetectionCandidate] = []
        for coordinates, confidence, class_id_value in zip(
            xyxy,
            confidences,
            class_ids,
        ):
            class_id = int(class_id_value)
            if class_id != self.class_id:
                continue

            x1, y1, x2, y2 = map(float, coordinates)
            confidence_value = float(confidence)
            values = (x1, y1, x2, y2, confidence_value)
            if not all(math.isfinite(value) for value in values):
                continue

            x1 = min(max(x1, 0.0), float(width))
            x2 = min(max(x2, 0.0), float(width))
            y1 = min(max(y1, 0.0), float(height))
            y2 = min(max(y2, 0.0), float(height))
            if x2 <= x1 or y2 <= y1:
                continue

            candidates.append(
                DroneDetectionCandidate(
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=confidence_value,
                    class_id=class_id,
                    class_name=self.class_name,
                )
            )
        return candidates
