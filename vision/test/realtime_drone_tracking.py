"""海康工业相机无人机实时检测、单目标跟踪与角误差可视化。"""

import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

from vision.src.camera_worker import CameraWorker
from vision.src.drone_detector import UltralyticsDroneDetector
from vision.src.drone_tracker import DroneTracker
from vision.src.hik_camera import HikCamera
from vision.src.latest_frame_buffer import LatestFrameBuffer


def load_config():
    """读取视觉模块统一配置。"""

    vision_dir = Path(__file__).resolve().parents[1]
    config_path = vision_dir / "configs" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resize_for_display(frame, max_width=1200, max_height=800):
    """等比例缩小显示画面，不改变检测和角度解算使用的原图。"""

    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return frame
    return cv2.resize(
        frame,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def _format_number(value, unit="", precision=2):
    """把可选数值格式化为适合画面显示的短文本。"""

    if value is None:
        return "--"
    return f"{value:.{precision}f}{unit}"


def draw_tracking(
    frame,
    candidates,
    result,
    tracker,
    detect_ms,
    latency_ms,
    latency_p95_ms,
    camera_fps,
    detection_fps,
):
    """在原始画面的副本上绘制候选、锁定状态和性能指标。"""

    display_frame = frame.copy()
    principal_point = (int(round(tracker.cx)), int(round(tracker.cy)))

    # 未被选中的候选使用细灰框，便于观察模型输出和关联结果。
    selected_bbox = result.bbox_xyxy if result.found else None
    for candidate in candidates:
        if selected_bbox is not None and candidate.bbox_xyxy == selected_bbox:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in candidate.bbox_xyxy)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
        cv2.putText(
            display_frame,
            f"{candidate.confidence:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (160, 160, 160),
            1,
        )

    cv2.drawMarker(
        display_frame,
        principal_point,
        (255, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=28,
        thickness=2,
    )

    if result.found:
        x1, y1, x2, y2 = (int(round(value)) for value in result.bbox_xyxy)
        target_center = tuple(int(round(value)) for value in result.center)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.drawMarker(
            display_frame,
            target_center,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=22,
            thickness=2,
        )
        cv2.line(display_frame, principal_point, target_center, (0, 255, 255), 2)
        status_text = (
            f"TRACKING  track={result.track_id}  confidence={result.confidence:.2f}"
        )
        status_color = (0, 255, 0)
    elif result.track_id is not None:
        status_text = f"TEMPORARILY LOST  track={result.track_id}"
        status_color = (0, 165, 255)
    else:
        status_text = "SEARCHING"
        status_color = (0, 0, 255)

    yaw_deg = (
        None if result.yaw_error_rad is None else math.degrees(result.yaw_error_rad)
    )
    elevation_deg = (
        None
        if result.elevation_error_rad is None
        else math.degrees(result.elevation_error_rad)
    )
    camera_fps_text = _format_number(camera_fps, precision=1)
    detection_fps_text = _format_number(detection_fps, precision=1)
    p95_text = _format_number(latency_p95_ms, "ms", precision=1)

    lines = [
        (status_text, status_color, 0.8),
        (
            "pixel error: "
            f"u={_format_number(result.pixel_error_u, 'px', 1)}  "
            f"v={_format_number(result.pixel_error_v, 'px', 1)}",
            (0, 255, 255),
            0.6,
        ),
        (
            "angle error: "
            f"yaw={_format_number(result.yaw_error_rad, 'rad', 4)} "
            f"({_format_number(yaw_deg, 'deg', 2)})  "
            f"elevation={_format_number(result.elevation_error_rad, 'rad', 4)} "
            f"({_format_number(elevation_deg, 'deg', 2)})",
            (0, 255, 255),
            0.55,
        ),
        (
            f"frame={result.frame_id}  inference={detect_ms:.1f}ms  "
            f"latency={latency_ms:.1f}ms  P95={p95_text}",
            (0, 255, 255),
            0.55,
        ),
        (
            f"Camera FPS: {camera_fps_text}  Detection FPS: {detection_fps_text}",
            (0, 255, 255),
            0.6,
        ),
    ]
    for index, (text, color, scale) in enumerate(lines):
        cv2.putText(
            display_frame,
            text,
            (20, 35 + index * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
        )

    return display_frame


def main():
    config = load_config()
    camera = HikCamera(config)
    frame_buffer = LatestFrameBuffer()
    camera_worker = CameraWorker(camera, frame_buffer)
    detector = UltralyticsDroneDetector(config)
    tracker = DroneTracker(config, detector)

    last_frame_id = -1
    fps_window_seconds = float(config["camera"]["fps_window_seconds"])
    fps_window_start_time = None
    fps_window_start_frame_id = None
    fps_window_start_frame_timestamp = None
    processed_frames_in_window = 0
    camera_fps = None
    detection_fps = None
    recent_latencies_ms = deque(maxlen=200)

    try:
        camera_worker.start()

        while True:
            frame_packet = frame_buffer.wait_for_new(last_frame_id, timeout=5.0)
            if frame_packet is None:
                print("等待相机画面超时，或相机采集线程已经结束")
                break
            last_frame_id = frame_packet.frame_id

            detect_start = time.perf_counter()
            candidates = detector.detect(frame_packet.frame)
            result = tracker.update(frame_packet, candidates)
            detect_ms = (time.perf_counter() - detect_start) * 1000.0
            latency_ms = (time.perf_counter() - frame_packet.timestamp) * 1000.0
            recent_latencies_ms.append(latency_ms)
            latency_p95_ms = float(np.percentile(recent_latencies_ms, 95))

            # 相机 FPS 使用帧号与采集时间，检测 FPS 使用实际完成推理的帧数。
            stats_now = time.perf_counter()
            if fps_window_start_time is None:
                fps_window_start_time = stats_now
                fps_window_start_frame_id = frame_packet.frame_id
                fps_window_start_frame_timestamp = frame_packet.timestamp
            else:
                processed_frames_in_window += 1
                processing_elapsed = stats_now - fps_window_start_time
                capture_elapsed = (
                    frame_packet.timestamp - fps_window_start_frame_timestamp
                )
                if processing_elapsed >= fps_window_seconds:
                    captured_frames = frame_packet.frame_id - fps_window_start_frame_id
                    if capture_elapsed > 0.0:
                        camera_fps = captured_frames / capture_elapsed
                    detection_fps = processed_frames_in_window / processing_elapsed

                    fps_window_start_time = stats_now
                    fps_window_start_frame_id = frame_packet.frame_id
                    fps_window_start_frame_timestamp = frame_packet.timestamp
                    processed_frames_in_window = 0

            display_frame = draw_tracking(
                frame_packet.frame,
                candidates,
                result,
                tracker,
                detect_ms,
                latency_ms,
                latency_p95_ms,
                camera_fps,
                detection_fps,
            )
            cv2.imshow(
                "Realtime Drone Tracking",
                resize_for_display(display_frame),
            )
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        print("收到 Ctrl+C，正在停止相机")
    finally:
        try:
            camera_worker.stop()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
