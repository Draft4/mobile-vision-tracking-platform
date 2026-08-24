"""使用海康工业相机自动采样棋盘格并标定相机内参。

棋盘固定为 8 x 11 个方格，即 7 x 10 个内部角点，单格边长为 15 mm。
程序只在内存中保留角点，不保存采样图片；达到数量和视角覆盖要求后自动计算
OpenCV 针孔模型内参，并将结果原子写入 JSON 文件。
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from vision.src.hik_camera import HikCamera


VISION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = VISION_DIR / "configs" / "config.yaml"
DEFAULT_OUTPUT_PATH = VISION_DIR / "configs" / "camera_intrinsics.json"

# 用户打印的棋盘为 8 x 11 个方格；OpenCV 需要的是内部角点数量。
BOARD_SQUARES = (8, 11)
PATTERN_SIZE = (7, 10)
SQUARE_SIZE_MM = 15.0

MIN_REQUIRED_SAMPLES = 20
MIN_BOARD_AREA_RATIO = 0.02
MIN_BORDER_MARGIN_RATIO = 0.01
MIN_SHARPNESS = 100.0
MIN_CAPTURE_INTERVAL_SECONDS = 0.4
MAX_RMS_ERROR_PX = 1.0
OUTLIER_ERROR_FLOOR_PX = 1.0

# 两个候选在所有维度上都小于以下差异时，视为重复视角。
MIN_CENTER_DELTA = 0.08
MIN_SCALE_DELTA = 0.03
MIN_ROTATION_DELTA_RADIANS = math.radians(8.0)
MIN_PERSPECTIVE_DELTA = math.log(1.08)
PERSPECTIVE_RATIO_LIMIT = math.log(1.08)

SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    40,
    0.001,
)


@dataclass
class CandidateMetrics:
    """一帧棋盘角点的质量和视角描述。"""

    center: tuple[float, float]
    area_ratio: float
    sharpness: float
    rotation: float
    log_top_bottom_ratio: float
    log_left_right_ratio: float
    center_bin: tuple[int, int]
    scale_bin: str
    perspective_classes: frozenset[str]

    @property
    def descriptor(self) -> tuple[float, ...]:
        return (
            self.center[0],
            self.center[1],
            math.sqrt(self.area_ratio),
            self.rotation,
            self.log_top_bottom_ratio,
            self.log_left_right_ratio,
        )


@dataclass
class CalibrationSample:
    """一组已采纳的亚像素角点及其视角描述。"""

    image_points: np.ndarray
    metrics: CandidateMetrics


@dataclass
class CalibrationResult:
    """一次 OpenCV 标定的结果和逐视角重投影误差。"""

    rms_error_px: float
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rotation_vectors: list[np.ndarray]
    translation_vectors: list[np.ndarray]
    per_view_errors_px: list[float]


@dataclass
class Coverage:
    """当前样本对位置、尺度和倾斜类别的覆盖情况。"""

    center_bins: set[tuple[int, int]]
    scale_bins: set[str]
    perspective_classes: set[str]

    @property
    def non_center_bin_count(self) -> int:
        return len(self.center_bins - {(1, 1)})

    @property
    def is_complete(self) -> bool:
        return (
            (1, 1) in self.center_bins
            and self.non_center_bin_count >= 4
            and len(self.scale_bins) >= 2
            and {"front", "pitch", "yaw"}.issubset(self.perspective_classes)
        )


def parse_args() -> argparse.Namespace:
    """解析标定工具参数。"""

    parser = argparse.ArgumentParser(
        description="自动采样 8x11 方格、15 mm 单格的棋盘并标定海康相机内参"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"相机 YAML 配置（默认：{DEFAULT_CONFIG_PATH}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"标定 JSON 输出路径（默认：{DEFAULT_OUTPUT_PATH}）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许覆盖已经存在的输出 JSON",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """读取并检查相机 YAML 配置。"""

    if not config_path.is_file():
        raise FileNotFoundError(f"相机配置不存在：{config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or "camera" not in config:
        raise ValueError(f"配置文件缺少 camera 节：{config_path}")
    return config


def validate_output_path(output_path: Path, force: bool) -> None:
    """在打开相机前检查输出路径，避免完成标定后才发现无法保存。"""

    if output_path.exists() and output_path.is_dir():
        raise IsADirectoryError(f"输出路径是目录：{output_path}")
    if output_path.exists() and not force:
        raise FileExistsError(
            f"输出文件已经存在：{output_path}；如需覆盖请添加 --force"
        )


def create_object_points() -> np.ndarray:
    """生成以毫米为单位的棋盘内部角点三维坐标。"""

    columns, rows = PATTERN_SIZE
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    points[:, :2] *= SQUARE_SIZE_MM
    return points


OBJECT_POINTS = create_object_points()


def detect_checkerboard(gray: np.ndarray) -> Optional[np.ndarray]:
    """检测完整棋盘并返回原始分辨率上的亚像素角点。"""

    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        flags |= getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)
        flags |= getattr(cv2, "CALIB_CB_ACCURACY", 0)
        found, corners = cv2.findChessboardCornersSB(
            gray,
            PATTERN_SIZE,
            flags=flags,
        )
    else:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(
            gray,
            PATTERN_SIZE,
            flags=flags,
        )

    if not found or corners is None:
        return None

    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.cornerSubPix(
        gray,
        corners,
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=SUBPIX_CRITERIA,
    )


def checkerboard_quad(corners: np.ndarray) -> np.ndarray:
    """从行优先角点序列提取左上、右上、右下、左下四角。"""

    columns, rows = PATTERN_SIZE
    points = corners.reshape(-1, 2)
    return np.array(
        [
            points[0],
            points[columns - 1],
            points[rows * columns - 1],
            points[(rows - 1) * columns],
        ],
        dtype=np.float32,
    )


def calculate_sharpness(gray: np.ndarray, quad: np.ndarray) -> float:
    """在棋盘包围框内计算拉普拉斯方差。"""

    x, y, width, height = cv2.boundingRect(quad)
    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + width, gray.shape[1])
    y1 = min(y + height, gray.shape[0])
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def classify_scale(area_ratio: float) -> str:
    """把棋盘投影面积划分为远、中、近三个尺度档。"""

    if area_ratio < 0.08:
        return "small"
    if area_ratio < 0.20:
        return "medium"
    return "large"


def describe_candidate(
    gray: np.ndarray,
    corners: np.ndarray,
) -> CandidateMetrics:
    """计算候选棋盘的归一化位置、面积、角度和透视特征。"""

    image_height, image_width = gray.shape[:2]
    quad = checkerboard_quad(corners)
    top_left, top_right, bottom_right, bottom_left = quad

    area_ratio = abs(float(cv2.contourArea(quad))) / (image_width * image_height)
    center_point = quad.mean(axis=0)
    center = (
        float(center_point[0] / image_width),
        float(center_point[1] / image_height),
    )
    center_bin = (
        min(int(center[0] * 3), 2),
        min(int(center[1] * 3), 2),
    )

    top_length = max(float(np.linalg.norm(top_right - top_left)), 1e-6)
    bottom_length = max(float(np.linalg.norm(bottom_right - bottom_left)), 1e-6)
    left_length = max(float(np.linalg.norm(bottom_left - top_left)), 1e-6)
    right_length = max(float(np.linalg.norm(bottom_right - top_right)), 1e-6)
    log_top_bottom_ratio = math.log(top_length / bottom_length)
    log_left_right_ratio = math.log(left_length / right_length)

    top_edge = top_right - top_left
    rotation = math.atan2(float(top_edge[1]), float(top_edge[0])) % math.pi

    perspective_classes: set[str] = set()
    if (
        abs(log_top_bottom_ratio) < PERSPECTIVE_RATIO_LIMIT
        and abs(log_left_right_ratio) < PERSPECTIVE_RATIO_LIMIT
    ):
        perspective_classes.add("front")
    if abs(log_top_bottom_ratio) >= PERSPECTIVE_RATIO_LIMIT:
        perspective_classes.add("pitch")
    if abs(log_left_right_ratio) >= PERSPECTIVE_RATIO_LIMIT:
        perspective_classes.add("yaw")

    return CandidateMetrics(
        center=center,
        area_ratio=area_ratio,
        sharpness=calculate_sharpness(gray, quad),
        rotation=rotation,
        log_top_bottom_ratio=log_top_bottom_ratio,
        log_left_right_ratio=log_left_right_ratio,
        center_bin=center_bin,
        scale_bin=classify_scale(area_ratio),
        perspective_classes=frozenset(perspective_classes),
    )


def rotation_difference(first: float, second: float) -> float:
    """计算以 pi 为周期的棋盘平面旋转差。"""

    difference = abs(first - second) % math.pi
    return min(difference, math.pi - difference)


def is_novel_view(
    metrics: CandidateMetrics,
    samples: list[CalibrationSample],
) -> bool:
    """判断候选是否与每个已有样本存在足够大的视角差异。"""

    candidate = metrics.descriptor
    for sample in samples:
        previous = sample.metrics.descriptor
        differences = (
            abs(candidate[0] - previous[0]),
            abs(candidate[1] - previous[1]),
            abs(candidate[2] - previous[2]),
            rotation_difference(candidate[3], previous[3]),
            abs(candidate[4] - previous[4]),
            abs(candidate[5] - previous[5]),
        )
        thresholds = (
            MIN_CENTER_DELTA,
            MIN_CENTER_DELTA,
            MIN_SCALE_DELTA,
            MIN_ROTATION_DELTA_RADIANS,
            MIN_PERSPECTIVE_DELTA,
            MIN_PERSPECTIVE_DELTA,
        )
        if all(
            difference < threshold
            for difference, threshold in zip(differences, thresholds)
        ):
            return False
    return True


def evaluate_candidate(
    gray: np.ndarray,
    corners: np.ndarray,
    samples: list[CalibrationSample],
    last_capture_time: float,
    current_time: float,
) -> tuple[Optional[CandidateMetrics], str]:
    """按完整性、清晰度、时间间隔和多样性筛选候选。"""

    image_height, image_width = gray.shape[:2]
    points = corners.reshape(-1, 2)
    x_margin = image_width * MIN_BORDER_MARGIN_RATIO
    y_margin = image_height * MIN_BORDER_MARGIN_RATIO

    if (
        float(points[:, 0].min()) < x_margin
        or float(points[:, 0].max()) > image_width - x_margin
        or float(points[:, 1].min()) < y_margin
        or float(points[:, 1].max()) > image_height - y_margin
    ):
        return None, "Move the checkerboard away from the image border"

    metrics = describe_candidate(gray, corners)
    if metrics.area_ratio < MIN_BOARD_AREA_RATIO:
        return None, "Move the checkerboard closer"
    if metrics.sharpness < MIN_SHARPNESS:
        return None, "Hold still or improve focus / exposure"
    if current_time - last_capture_time < MIN_CAPTURE_INTERVAL_SECONDS:
        return None, "Hold this view briefly"
    if not is_novel_view(metrics, samples):
        return None, "Change position, distance, rotation, or tilt"
    return metrics, "Accepted"


def summarize_coverage(samples: list[CalibrationSample]) -> Coverage:
    """汇总已采纳样本的视角覆盖情况。"""

    center_bins = {sample.metrics.center_bin for sample in samples}
    scale_bins = {sample.metrics.scale_bin for sample in samples}
    perspective_classes: set[str] = set()
    for sample in samples:
        perspective_classes.update(sample.metrics.perspective_classes)
    return Coverage(center_bins, scale_bins, perspective_classes)


def next_sampling_hint(samples: list[CalibrationSample], coverage: Coverage) -> str:
    """给出当前最优先补充的样本类型。"""

    if len(samples) < MIN_REQUIRED_SAMPLES:
        return "Keep moving the checkerboard slowly"
    if (1, 1) not in coverage.center_bins:
        return "Move the checkerboard to the image center"
    if coverage.non_center_bin_count < 4:
        return "Visit more image edges and corners"
    if len(coverage.scale_bins) < 2:
        return "Change the checkerboard distance"
    if "front" not in coverage.perspective_classes:
        return "Hold the checkerboard front-facing"
    if "pitch" not in coverage.perspective_classes:
        return "Tilt the checkerboard up or down"
    if "yaw" not in coverage.perspective_classes:
        return "Tilt the checkerboard left or right"
    return "Coverage complete; calibrating"


def calculate_per_view_errors(
    samples: list[CalibrationSample],
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    rotation_vectors: list[np.ndarray],
    translation_vectors: list[np.ndarray],
) -> list[float]:
    """计算每个视角所有角点的像素 RMSE。"""

    errors: list[float] = []
    for sample, rotation, translation in zip(
        samples,
        rotation_vectors,
        translation_vectors,
    ):
        projected, _ = cv2.projectPoints(
            OBJECT_POINTS,
            rotation,
            translation,
            camera_matrix,
            distortion_coefficients,
        )
        difference = sample.image_points.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(difference**2, axis=1)))))
    return errors


def run_calibration(
    samples: list[CalibrationSample],
    image_size: tuple[int, int],
) -> CalibrationResult:
    """使用标准五参数畸变模型完成一次 OpenCV 相机标定。"""

    object_points = [OBJECT_POINTS.copy() for _ in samples]
    image_points = [sample.image_points for sample in samples]
    (
        rms_error,
        camera_matrix,
        distortion_coefficients,
        rotation_vectors,
        translation_vectors,
    ) = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    per_view_errors = calculate_per_view_errors(
        samples,
        camera_matrix,
        distortion_coefficients,
        rotation_vectors,
        translation_vectors,
    )
    return CalibrationResult(
        rms_error_px=float(rms_error),
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        rotation_vectors=rotation_vectors,
        translation_vectors=translation_vectors,
        per_view_errors_px=per_view_errors,
    )


def calibrate_and_reject_outliers(
    samples: list[CalibrationSample],
    image_size: tuple[int, int],
) -> tuple[list[CalibrationSample], CalibrationResult, int]:
    """反复剔除重投影误差异常的视角并重新标定。"""

    retained = list(samples)
    removed_count = 0

    while True:
        result = run_calibration(retained, image_size)
        errors = np.asarray(result.per_view_errors_px, dtype=np.float64)
        median = float(np.median(errors))
        mad = float(np.median(np.abs(errors - median)))
        threshold = max(
            OUTLIER_ERROR_FLOOR_PX,
            median + 2.5 * mad,
        )
        keep_indices = [
            index for index, error in enumerate(errors) if error <= threshold
        ]

        if len(keep_indices) == len(retained) or len(keep_indices) < 3:
            return retained, result, removed_count

        removed_count += len(retained) - len(keep_indices)
        retained = [retained[index] for index in keep_indices]


def calibration_is_valid(
    result: CalibrationResult,
    image_size: tuple[int, int],
) -> bool:
    """验证焦距、主点和总体重投影误差。"""

    width, height = image_size
    fx = float(result.camera_matrix[0, 0])
    fy = float(result.camera_matrix[1, 1])
    cx = float(result.camera_matrix[0, 2])
    cy = float(result.camera_matrix[1, 2])
    return (
        math.isfinite(result.rms_error_px)
        and bool(np.all(np.isfinite(result.camera_matrix)))
        and bool(np.all(np.isfinite(result.distortion_coefficients)))
        and all(math.isfinite(error) for error in result.per_view_errors_px)
        and result.rms_error_px <= MAX_RMS_ERROR_PX
        and fx > 0.0
        and fy > 0.0
        and 0.0 <= cx < width
        and 0.0 <= cy < height
    )


def build_output_payload(
    result: CalibrationResult,
    image_size: tuple[int, int],
    sample_count: int,
) -> dict:
    """构造稳定、可直接供后续模块读取的 JSON 数据结构。"""

    width, height = image_size
    camera_matrix = np.asarray(result.camera_matrix, dtype=np.float64)
    distortion = np.asarray(
        result.distortion_coefficients,
        dtype=np.float64,
    ).reshape(-1)
    if distortion.size < 5:
        raise RuntimeError("OpenCV 返回的畸变系数少于五项")

    return {
        "schema_version": 1,
        "camera_model": "opencv_pinhole",
        "image_width": int(width),
        "image_height": int(height),
        "intrinsics": {
            "fx": float(camera_matrix[0, 0]),
            "fy": float(camera_matrix[1, 1]),
            "cx": float(camera_matrix[0, 2]),
            "cy": float(camera_matrix[1, 2]),
        },
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients_order": ["k1", "k2", "p1", "p2", "k3"],
        "distortion_coefficients": [float(value) for value in distortion[:5]],
        "checkerboard": {
            "squares_columns": BOARD_SQUARES[0],
            "squares_rows": BOARD_SQUARES[1],
            "inner_corners_columns": PATTERN_SIZE[0],
            "inner_corners_rows": PATTERN_SIZE[1],
            "square_size_mm": SQUARE_SIZE_MM,
        },
        "valid_sample_count": int(sample_count),
        "rms_reprojection_error_px": float(result.rms_error_px),
        "per_view_reprojection_errors_px": [
            float(error) for error in result.per_view_errors_px
        ],
        "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_json_atomically(output_path: Path, payload: dict) -> None:
    """先写同目录临时文件，再原子替换最终 JSON。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def resize_for_display(
    frame: np.ndarray,
    max_width: int = 1280,
    max_height: int = 800,
) -> np.ndarray:
    """只缩小预览图，不改变用于检测和标定的原始图像。"""

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


def draw_status(
    rgb_frame: np.ndarray,
    corners: Optional[np.ndarray],
    samples: list[CalibrationSample],
    status: str,
    sharpness: Optional[float],
) -> np.ndarray:
    """绘制角点、覆盖进度和下一步移动提示。"""

    # canvas = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    canvas = rgb_frame
    if corners is not None:
        cv2.drawChessboardCorners(canvas, PATTERN_SIZE, corners, True)

    coverage = summarize_coverage(samples)
    perspective = coverage.perspective_classes
    sharpness_text = "--" if sharpness is None else f"{sharpness:.1f}"
    lines = [
        f"Samples: {len(samples)}/{MIN_REQUIRED_SAMPLES}  Sharpness: {sharpness_text}",
        (
            f"Position: center={'Y' if (1, 1) in coverage.center_bins else 'N'} "
            f"edges={coverage.non_center_bin_count}/4  "
            f"scales={len(coverage.scale_bins)}/2"
        ),
        (
            f"Views: front={'Y' if 'front' in perspective else 'N'} "
            f"pitch={'Y' if 'pitch' in perspective else 'N'} "
            f"yaw={'Y' if 'yaw' in perspective else 'N'}"
        ),
        status,
        "Press Q or Esc to cancel",
    ]
    for index, line in enumerate(lines):
        y = 32 + index * 30
        cv2.putText(
            canvas,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (80, 255, 80),
            2,
            cv2.LINE_AA,
        )
    return resize_for_display(canvas)


def run(config: dict, output_path: Path) -> bool:
    """执行实时自动采样，成功保存返回 True，用户取消返回 False。"""

    camera = HikCamera(config)
    samples: list[CalibrationSample] = []
    image_size: Optional[tuple[int, int]] = None
    last_capture_time = -math.inf
    status = "Show the full checkerboard to the camera"

    try:
        camera.open()
        camera.start()
        print("自动标定已启动，请缓慢移动、远近移动并向不同方向倾斜棋盘")
        print("按 q、Esc 或 Ctrl+C 可取消；取消时不会生成 JSON")

        while True:
            frame_packet = camera.grab()
            rgb_frame = frame_packet.frame
            height, width = rgb_frame.shape[:2]
            current_size = (width, height)
            if image_size is None:
                image_size = current_size
            elif current_size != image_size:
                raise RuntimeError(f"采样期间分辨率由 {image_size} 变为 {current_size}")

            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            corners = detect_checkerboard(gray)
            sharpness: Optional[float] = None

            if corners is None:
                status = "Show the full checkerboard to the camera"
            else:
                current_time = time.monotonic()
                metrics, status = evaluate_candidate(
                    gray,
                    corners,
                    samples,
                    last_capture_time,
                    current_time,
                )
                if metrics is not None:
                    sharpness = metrics.sharpness
                    samples.append(
                        CalibrationSample(
                            image_points=corners.copy(),
                            metrics=metrics,
                        )
                    )
                    last_capture_time = current_time
                    coverage = summarize_coverage(samples)
                    status = next_sampling_hint(samples, coverage)
                    print(
                        f"已采纳样本 {len(samples)}："
                        f"清晰度={metrics.sharpness:.1f}，"
                        f"面积={metrics.area_ratio:.3f}，提示={status}"
                    )

                    if len(samples) >= MIN_REQUIRED_SAMPLES and coverage.is_complete:
                        print("样本覆盖要求已满足，正在计算相机内参……")
                        retained, result, removed_count = calibrate_and_reject_outliers(
                            samples, image_size
                        )
                        samples = retained
                        if removed_count:
                            print(f"已剔除 {removed_count} 个重投影误差异常样本")

                        coverage = summarize_coverage(samples)
                        if (
                            len(samples) < MIN_REQUIRED_SAMPLES
                            or not coverage.is_complete
                        ):
                            status = next_sampling_hint(samples, coverage)
                        elif calibration_is_valid(result, image_size):
                            payload = build_output_payload(
                                result,
                                image_size,
                                len(samples),
                            )
                            write_json_atomically(output_path, payload)
                            print(
                                f"标定成功：RMS={result.rms_error_px:.4f} px，"
                                f"有效样本={len(samples)}"
                            )
                            print(f"结果已保存到：{output_path}")
                            return True
                        else:
                            worst_index = int(np.argmax(result.per_view_errors_px))
                            worst_error = result.per_view_errors_px[worst_index]
                            samples.pop(worst_index)
                            status = (
                                f"RMS {result.rms_error_px:.2f}px too high; "
                                "collect a better view"
                            )
                            print(
                                f"本次结果未达到 {MAX_RMS_ERROR_PX:.1f} px 要求，"
                                f"已移除最差样本（{worst_error:.3f} px）并继续采样"
                            )
                else:
                    candidate_metrics = describe_candidate(gray, corners)
                    sharpness = candidate_metrics.sharpness

            preview = draw_status(
                rgb_frame,
                corners,
                samples,
                status,
                sharpness,
            )
            cv2.imshow("Camera Intrinsic Calibration", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("用户取消标定，未生成结果文件")
                return False
    finally:
        if camera.is_start:
            try:
                camera.stop()
            except Exception as error:
                print(f"停止相机取流失败：{error}")
        if camera.is_open:
            try:
                camera.close()
            except Exception as error:
                print(f"关闭相机失败：{error}")
        cv2.destroyAllWindows()


def main() -> int:
    """命令行入口。"""

    args = parse_args()
    config_path = args.config.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    try:
        validate_output_path(output_path, args.force)
        config = load_config(config_path)
        return 0 if run(config, output_path) else 130
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，标定已取消，未生成结果文件")
        return 130
    except Exception as error:
        print(f"标定失败：{error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
