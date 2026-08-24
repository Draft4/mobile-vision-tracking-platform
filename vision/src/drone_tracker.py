"""无人机单目标锁定和相机视轴角误差解算。

该模块不直接控制云台，也不保存陈旧角度。目标短暂漏检时会立即返回
``found=False``，但在配置的超时时间内保留内部锁定，便于原目标重现后继续使用
同一 ``track_id``。角误差单位和正方向遵循 ``interface/coordinate-system.md``。
"""

import math

from .data_types import (
    DroneDetectionCandidate,
    DroneTrackingResult,
    FramePacket,
)
from .drone_detector import DroneDetectorBackend


class DroneTracker:
    """组合推理后端、轻量单目标关联和角度解算。"""

    def __init__(self, config: dict, detector: DroneDetectorBackend):
        self.detector = detector
        self._load_config(config)

        self._next_track_id = 1
        self._locked_track_id = None
        self._last_center = None
        self._last_seen_timestamp = None

    def _load_config(self, config: dict) -> None:
        """读取并验证锁定参数与相机内参。"""

        camera_config = config["camera"]
        tracking_config = config["drone_tracking"]

        self.fx = float(camera_config["_fx"])
        self.fy = float(camera_config["_fy"])
        self.cx = float(camera_config["_cx"])
        self.cy = float(camera_config["_cy"])
        self.image_width = int(camera_config["calibrated_image_width"])
        self.image_height = int(camera_config["calibrated_image_height"])

        self.acquire_confidence = float(
            tracking_config["acquire_confidence"]
        )
        self.maintain_confidence = float(
            tracking_config["maintain_confidence"]
        )
        self.reacquire_timeout_seconds = float(
            tracking_config["reacquire_timeout_seconds"]
        )
        self.association_distance_ratio = float(
            tracking_config["association_distance_ratio"]
        )

        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("相机 fx 和 fy 必须大于 0")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("标定图像宽高必须大于 0")
        if not 0.0 <= self.cx < self.image_width:
            raise ValueError("相机主点 cx 必须位于标定图像内")
        if not 0.0 <= self.cy < self.image_height:
            raise ValueError("相机主点 cy 必须位于标定图像内")
        if not 0.0 <= self.maintain_confidence <= 1.0:
            raise ValueError("maintain_confidence 必须位于 [0, 1]")
        if not self.maintain_confidence <= self.acquire_confidence <= 1.0:
            raise ValueError("acquire_confidence 必须不小于 maintain_confidence")
        if self.reacquire_timeout_seconds <= 0.0:
            raise ValueError("reacquire_timeout_seconds 必须大于 0")
        if not 0.0 < self.association_distance_ratio <= 1.0:
            raise ValueError("association_distance_ratio 必须位于 (0, 1]")

    def process(self, frame_packet: FramePacket) -> DroneTrackingResult:
        """对一帧执行检测、目标关联和角度解算。"""

        candidates = self.detector.detect(frame_packet.frame)
        return self.update(frame_packet, candidates)

    def update(
        self,
        frame_packet: FramePacket,
        candidates: list[DroneDetectionCandidate],
    ) -> DroneTrackingResult:
        """使用已得到的候选更新锁定状态，便于后端替换和单元测试。"""

        self._validate_frame(frame_packet)

        if self._locked_track_id is None:
            selected = self._select_for_acquisition(candidates)
            if selected is None:
                return self._lost_result(frame_packet)
            self._start_new_lock(selected, frame_packet.timestamp)
            return self._build_found_result(frame_packet, selected)

        selected = self._select_for_maintenance(candidates)
        if selected is not None:
            self._last_center = selected.center
            self._last_seen_timestamp = frame_packet.timestamp
            return self._build_found_result(frame_packet, selected)

        elapsed_since_seen = frame_packet.timestamp - self._last_seen_timestamp
        if elapsed_since_seen < 0.0:
            raise ValueError("FramePacket.timestamp 不得倒退")
        if elapsed_since_seen <= self.reacquire_timeout_seconds:
            return self._lost_result(frame_packet)

        self._clear_lock()
        selected = self._select_for_acquisition(candidates)
        if selected is None:
            return self._lost_result(frame_packet)
        self._start_new_lock(selected, frame_packet.timestamp)
        return self._build_found_result(frame_packet, selected)

    def _validate_frame(self, frame_packet: FramePacket) -> None:
        """确保运行分辨率与内参标定分辨率完全一致。"""

        height, width = frame_packet.frame.shape[:2]
        actual_size = (width, height)
        expected_size = (self.image_width, self.image_height)
        if actual_size != expected_size:
            raise ValueError(
                f"运行图像尺寸 {actual_size} 与内参标定尺寸 {expected_size} 不一致"
            )

    def _select_for_acquisition(
        self,
        candidates: list[DroneDetectionCandidate],
    ):
        """从高置信度候选中选择离相机主点最近的目标。"""

        eligible = [
            candidate
            for candidate in candidates
            if candidate.confidence >= self.acquire_confidence
        ]
        if not eligible:
            return None

        return min(
            eligible,
            key=lambda candidate: (
                (candidate.center[0] - self.cx) ** 2
                + (candidate.center[1] - self.cy) ** 2,
                -candidate.confidence,
            ),
        )

    def _select_for_maintenance(
        self,
        candidates: list[DroneDetectionCandidate],
    ):
        """选择上一目标中心附近且达到维持阈值的候选。"""

        eligible = [
            candidate
            for candidate in candidates
            if candidate.confidence >= self.maintain_confidence
        ]
        if not eligible or self._last_center is None:
            return None

        selected = min(
            eligible,
            key=lambda candidate: (
                math.hypot(
                    candidate.center[0] - self._last_center[0],
                    candidate.center[1] - self._last_center[1],
                ),
                -candidate.confidence,
            ),
        )
        distance = math.hypot(
            selected.center[0] - self._last_center[0],
            selected.center[1] - self._last_center[1],
        )
        maximum_distance = self.association_distance_ratio * math.hypot(
            self.image_width,
            self.image_height,
        )
        return selected if distance <= maximum_distance else None

    def _start_new_lock(
        self,
        candidate: DroneDetectionCandidate,
        timestamp: float,
    ) -> None:
        """为新目标分配一次锁定会话编号。"""

        self._locked_track_id = self._next_track_id
        self._next_track_id += 1
        self._last_center = candidate.center
        self._last_seen_timestamp = timestamp

    def _clear_lock(self) -> None:
        """清除已超时的锁定状态。"""

        self._locked_track_id = None
        self._last_center = None
        self._last_seen_timestamp = None

    def _build_found_result(
        self,
        frame_packet: FramePacket,
        candidate: DroneDetectionCandidate,
    ) -> DroneTrackingResult:
        """根据框中心和针孔内参计算两轴角误差。"""

        target_u, target_v = candidate.center
        pixel_error_u = target_u - self.cx
        pixel_error_v = target_v - self.cy
        yaw_error_rad = math.atan2(-pixel_error_u, self.fx)
        elevation_error_rad = math.atan2(-pixel_error_v, self.fy)

        return DroneTrackingResult(
            found=True,
            frame_id=frame_packet.frame_id,
            timestamp=frame_packet.timestamp,
            track_id=self._locked_track_id,
            bbox_xyxy=candidate.bbox_xyxy,
            center=candidate.center,
            confidence=candidate.confidence,
            pixel_error_u=pixel_error_u,
            pixel_error_v=pixel_error_v,
            yaw_error_rad=yaw_error_rad,
            elevation_error_rad=elevation_error_rad,
        )

    def _lost_result(self, frame_packet: FramePacket) -> DroneTrackingResult:
        """返回不含陈旧目标测量的丢失结果。"""

        return DroneTrackingResult(
            found=False,
            frame_id=frame_packet.frame_id,
            timestamp=frame_packet.timestamp,
            track_id=self._locked_track_id,
        )
