"""海康相机 V1 外框轮廓法实时检测与可视化测试。"""

import time
from pathlib import Path

import cv2
import yaml

from vision.src.camera_worker import CameraWorker
from vision.src.hik_camera import HikCamera
from vision.src.latest_frame_buffer import LatestFrameBuffer
from vision.src.outer_frame_target_detector import OuterFrameTargetDetector


def load_config():
    vision_dir = Path(__file__).resolve().parents[1]
    config_path = vision_dir / "configs" / "config.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def draw_detection(
    frame,
    result,
    frame_id,
    detect_ms,
    latency_ms,
    camera_fps,
    detection_fps,
):
    """在原始画面的副本上绘制检测结果。"""
    display_frame = frame.copy()
    has_target = (
        result.found and result.corners is not None and result.center is not None
    )

    if has_target:
        corners = result.corners.round().astype(int)
        labels = ["TL", "TR", "BR", "BL"]

        for index in range(4):
            p1 = tuple(map(int, corners[index]))
            p2 = tuple(map(int, corners[(index + 1) % 4]))
            cv2.line(display_frame, p1, p2, (0, 255, 0), 2)

        for point, label in zip(corners, labels):
            x, y = map(int, point)
            cv2.circle(display_frame, (x, y), 6, (255, 0, 0), -1)
            cv2.putText(
                display_frame,
                label,
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        center_x, center_y = result.center
        center = (int(round(center_x)), int(round(center_y)))
        cv2.drawMarker(
            display_frame,
            center,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=25,
            thickness=2,
        )

        status_text = f"TARGET  confidence={result.confidence:.2f}"
        status_color = (0, 255, 0)
    else:
        status_text = "TARGET NOT FOUND"
        status_color = (0, 0, 255)

    cv2.putText(
        display_frame,
        status_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
    )
    cv2.putText(
        display_frame,
        f"frame={frame_id}  detect={detect_ms:.1f}ms  latency={latency_ms:.1f}ms",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    camera_fps_text = "--" if camera_fps is None else f"{camera_fps:.1f}"
    detection_fps_text = "--" if detection_fps is None else f"{detection_fps:.1f}"
    cv2.putText(
        display_frame,
        f"Camera FPS: {camera_fps_text}  Detection FPS: {detection_fps_text}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    return display_frame


def resize_for_display(frame, max_width=1000, max_height=700):
    """等比例缩小显示画面，不改变检测使用的原始图像。"""
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


def main():
    config = load_config()
    camera = HikCamera(config)
    frame_buffer = LatestFrameBuffer()
    camera_worker = CameraWorker(camera, frame_buffer)
    detector = OuterFrameTargetDetector(config)

    last_frame_id = -1
    # 使用配置窗口平滑显示帧率；首个完整窗口结束前保持为 None。
    fps_window_seconds = float(config["camera"]["fps_window_seconds"])
    fps_window_start_time = None
    fps_window_start_frame_id = None
    fps_window_start_frame_timestamp = None
    processed_frames_in_window = 0
    camera_fps = None
    detection_fps = None

    try:
        camera_worker.start()

        while True:
            frame_packet = frame_buffer.wait_for_new(last_frame_id, timeout=5.0)

            if frame_packet is None:
                print("等待相机画面超时，或相机采集线程已经结束")
                break

            last_frame_id = frame_packet.frame_id

            detect_start = time.perf_counter()
            result = detector.detect(frame_packet)
            detect_ms = (time.perf_counter() - detect_start) * 1000.0
            latency_ms = (time.perf_counter() - frame_packet.timestamp) * 1000.0

            # 帧号差反映相机采集速度，完成检测的帧数反映处理速度。
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
                    captured_frame_count = (
                        frame_packet.frame_id - fps_window_start_frame_id
                    )
                    if capture_elapsed > 0.0:
                        camera_fps = captured_frame_count / capture_elapsed
                    detection_fps = processed_frames_in_window / processing_elapsed

                    fps_window_start_time = stats_now
                    fps_window_start_frame_id = frame_packet.frame_id
                    fps_window_start_frame_timestamp = frame_packet.timestamp
                    processed_frames_in_window = 0

            display_frame = draw_detection(
                frame_packet.frame,
                result,
                frame_packet.frame_id,
                detect_ms,
                latency_ms,
                camera_fps,
                detection_fps,
            )
            display_frame = resize_for_display(display_frame)

            cv2.imshow("Realtime Outer Frame Target Detection", display_frame)
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
