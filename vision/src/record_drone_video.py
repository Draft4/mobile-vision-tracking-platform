"""使用海康工业相机录制无人机训练视频。

本脚本直接使用 :class:`HikCamera` 的同步取流接口，逐帧写入视频，不经过会主动
丢弃旧帧的 ``LatestFrameBuffer``。默认输出到 ``vision/recordings/``；该目录和
常见视频格式已由项目根目录的 ``.gitignore`` 排除。
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from vision.src.hik_camera import HikCamera


VISION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = VISION_DIR / "configs" / "config.yaml"


def parse_args():
    """解析录像参数。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = VISION_DIR / "recordings" / f"drone_{timestamp}.mp4"

    parser = argparse.ArgumentParser(description="使用海康工业相机录制无人机视频")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"视频保存路径（默认：{default_output}）",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"相机配置文件（默认：{DEFAULT_CONFIG_PATH}）",
    )
    parser.add_argument(
        "--fps", type=float, default=80, help="输出视频帧率（默认：80）"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="录制时长，单位为秒；0 表示持续录制直到手动停止（默认：0）",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="四字符 OpenCV 视频编码器（默认：mp4v）",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="不显示实时预览；可使用 Ctrl+C 或 --duration 停止",
    )
    args = parser.parse_args()

    if args.fps is not None and args.fps <= 0.0:
        parser.error("--fps 必须大于 0")
    if args.duration < 0.0:
        parser.error("--duration 必须大于等于 0")
    if len(args.codec) != 4:
        parser.error("--codec 必须恰好包含 4 个字符")

    return args


def load_config(config_path: Path):
    """读取 YAML 相机配置。"""
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or "camera" not in config:
        raise ValueError(f"配置文件缺少 camera 节：{config_path}")

    return config


def main():
    args = parse_args()
    config = load_config(args.config)
    output_fps = args.fps
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    camera = HikCamera(config)
    writer = None
    recorded_frames = 0
    skipped_frames = 0
    last_frame_id = None
    record_start = None

    try:
        camera.open()
        camera.start()

        first_packet = camera.grab()
        height, width = first_packet.frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*args.codec)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            output_fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(
                f"无法创建视频文件：{output_path}；请检查扩展名和编码器 {args.codec}"
            )

        print(f"开始录像：{width}x{height} @ {output_fps:.2f} FPS")
        print(f"保存位置：{output_path}")
        if args.no_preview:
            print("按 Ctrl+C 停止录像")
        else:
            print("按 q、Esc 或 Ctrl+C 停止录像")

        frame_packet = first_packet
        record_start = time.perf_counter()

        while True:
            video_frame = frame_packet.frame
            writer.write(video_frame)
            recorded_frames += 1

            if last_frame_id is not None:
                skipped_frames += max(frame_packet.frame_id - last_frame_id - 1, 0)
            last_frame_id = frame_packet.frame_id

            elapsed = time.perf_counter() - record_start
            if not args.no_preview:
                preview = video_frame.copy()
                cv2.putText(
                    preview,
                    f"REC  {elapsed:.1f}s  frames={recorded_frames}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Drone Video Recording", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if args.duration > 0.0 and elapsed >= args.duration:
                break

            frame_packet = camera.grab()

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停止录像")
    finally:
        if writer is not None:
            writer.release()
        if camera.is_start:
            camera.stop()
        if camera.is_open:
            camera.close()
        if not args.no_preview:
            cv2.destroyAllWindows()

    if recorded_frames > 0 and record_start is not None:
        elapsed = time.perf_counter() - record_start
        actual_fps = recorded_frames / elapsed if elapsed > 0.0 else 0.0
        print(
            f"录像完成：{recorded_frames} 帧，{elapsed:.2f} 秒，"
            f"采集平均 {actual_fps:.2f} FPS"
        )
        if skipped_frames > 0:
            print(f"相机帧号检测到跳过 {skipped_frames} 帧，请检查带宽或曝光设置")
        print(f"视频已保存：{output_path}")


if __name__ == "__main__":
    main()
