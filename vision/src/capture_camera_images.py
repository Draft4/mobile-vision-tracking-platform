"""使用海康工业相机实时预览并按键保存图片。

脚本直接遵循 ``HikCamera`` 的同步采集生命周期。预览画面可以缩小显示，但按
``p`` 保存的是相机返回的原始分辨率帧，不包含窗口提示或其他绘制内容。默认
输出目录位于已被项目 ``.gitignore`` 排除的 ``vision/datasets/`` 下。
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from vision.src.hik_camera import HikCamera


VISION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = VISION_DIR / "configs" / "config.yaml"
DEFAULT_OUTPUT_DIR = VISION_DIR / "datasets" / "captured_images"


def parse_args():
    """解析图片采集参数。"""
    parser = argparse.ArgumentParser(
        description="实时预览海康工业相机画面并按 p 保存当前帧"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"图片保存目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"相机配置文件（默认：{DEFAULT_CONFIG_PATH}）",
    )
    parser.add_argument(
        "--format",
        dest="image_format",
        choices=("jpg", "png"),
        default="jpg",
        help="保存格式：jpg 或 png（默认：jpg）",
    )
    return parser.parse_args()


def load_config(config_path: Path):
    """读取 YAML 相机配置。"""
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or "camera" not in config:
        raise ValueError(f"配置文件缺少 camera 节：{config_path}")

    return config


def resize_for_display(frame, max_width=1280, max_height=800):
    """仅为实时预览等比例缩小画面，不改变保存图片的分辨率。"""
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


def build_image_path(output_dir: Path, image_format: str, frame_id: int):
    """使用拍摄时间和相机帧号生成不易重复的文件名。"""
    captured_at = datetime.now()
    milliseconds = captured_at.microsecond // 1000
    filename = (
        f"drone_{captured_at:%Y%m%d_%H%M%S}_{milliseconds:03d}_"
        f"frame_{frame_id:08d}.{image_format}"
    )
    return output_dir / filename


def save_frame(frame, output_path: Path, image_format: str):
    """保存未绘制标记的原始分辨率帧。"""
    parameters = []
    if image_format == "jpg":
        parameters = [cv2.IMWRITE_JPEG_QUALITY, 95]

    if not cv2.imwrite(str(output_path), frame, parameters):
        raise RuntimeError(f"图片保存失败：{output_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = HikCamera(config)
    saved_count = 0

    try:
        camera.open()
        camera.start()

        print(f"图片保存目录：{output_dir}")
        print("请先选中预览窗口；按 p 保存当前原始帧，按 q、Esc 或 Ctrl+C 退出")

        while True:
            frame_packet = camera.grab()
            display_frame = resize_for_display(frame_packet.frame)
            cv2.imshow("Industrial Camera Image Capture", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("p"), ord("P")):
                output_path = build_image_path(
                    output_dir,
                    args.image_format,
                    frame_packet.frame_id,
                )
                save_frame(frame_packet.frame, output_path, args.image_format)
                saved_count += 1
                print(f"已保存第 {saved_count} 张：{output_path}")
            elif key in (ord("q"), ord("Q"), 27):
                break

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停止相机")
    finally:
        if camera.is_start:
            camera.stop()
        if camera.is_open:
            camera.close()
        cv2.destroyAllWindows()

    print(f"图片采集结束，共保存 {saved_count} 张")


if __name__ == "__main__":
    main()
