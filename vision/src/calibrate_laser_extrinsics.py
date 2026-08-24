"""通过离线靶标图片和人工点击标定激光束相对相机的空间直线。"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from .data_types import FramePacket
from .inner_region_target_detector import InnerRegionTargetDetector
from .laser_pixel_predictor import predict_laser_pixel
from .target_pose_estimator import estimate_target_pose


VISION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = VISION_DIR / "configs" / "config.yaml"
DEFAULT_INPUT_DIR = VISION_DIR / "datasets" / "laser_calibration_images"
DEFAULT_OUTPUT_PATH = VISION_DIR / "configs" / "laser_extrinsics.json"

IMAGE_SUFFIXES = {".png"}
CLICK_COUNT = 5
CLICK_MAX_SPREAD_PX = 2.0
POSE_MAX_RMS_ERROR_PX = 1.5
RANSAC_MAX_ERROR_PX = 4.0
MIN_PAIR_DISTANCE_M = 0.10
MIN_VALID_SAMPLES = 15
MIN_INLIERS = 10
MIN_INLIER_RATIO = 0.70
MIN_HIT_SPAN_M = 2.0
MAX_LOO_RMS_ERROR_PX = 2.0
MAX_LOO_P95_ERROR_PX = 4.0
TARGET_MARGIN_M = 0.005
ZOOM_HALF_SIZE_PX = 50
ZOOM_SCALE = 8


class UserAbort(RuntimeError):
    """用户主动取消整次标定。"""


@dataclass
class ImageRecord:
    """一张标定图片的检测、点击和三维重建结果。"""

    path: Path
    status: str = "pending"
    reason: Optional[str] = None
    image_size: Optional[tuple[int, int]] = None
    raw_corners: Optional[np.ndarray] = None
    refined_corners: Optional[np.ndarray] = None
    rotation: Optional[np.ndarray] = None
    translation: Optional[np.ndarray] = None
    pose_rms_error_px: Optional[float] = None
    clicks: list[tuple[float, float]] = field(default_factory=list)
    laser_pixel: Optional[tuple[float, float]] = None
    click_max_spread_px: Optional[float] = None
    hit_camera_m: Optional[np.ndarray] = None
    is_inlier: Optional[bool] = None
    fit_error_px: Optional[float] = None
    loo_error_px: Optional[float] = None


@dataclass(frozen=True)
class LineFitResult:
    """激光直线拟合结果及全部样本的像素残差。"""

    origin_camera_m: np.ndarray
    direction_camera: np.ndarray
    inlier_mask: np.ndarray
    residuals_px: np.ndarray


def parse_args():
    """解析离线激光外参标定参数。"""

    parser = argparse.ArgumentParser(
        description="从靶标图片和人工激光点点击拟合激光外参"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"标定图片目录（默认：{DEFAULT_INPUT_DIR}）",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"视觉配置文件（默认：{DEFAULT_CONFIG_PATH}）",
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
        help="允许覆盖已经存在的输出文件",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """读取并检查标定所需配置。"""

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or not isinstance(config.get("camera"), dict):
        raise ValueError(f"配置文件缺少 camera 节：{config_path}")

    camera_matrix, distortion, image_size = camera_parameters(config)
    if not np.all(np.isfinite(camera_matrix)):
        raise ValueError("相机内参必须为有限数值")
    if camera_matrix[0, 0] <= 0.0 or camera_matrix[1, 1] <= 0.0:
        raise ValueError("相机焦距 fx、fy 必须大于 0")
    if distortion.shape != (5,):
        raise ValueError("distortion_coefficients 必须包含五个数值")
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("标定图像宽高必须大于 0")
    if not 0.0 <= camera_matrix[0, 2] < image_size[0]:
        raise ValueError("相机主点 cx 必须位于标定图像内")
    if not 0.0 <= camera_matrix[1, 2] < image_size[1]:
        raise ValueError("相机主点 cy 必须位于标定图像内")

    target_width, target_height = target_size_m(config)
    if target_width <= 0.0 or target_height <= 0.0:
        raise ValueError("靶标尺寸必须大于 0")

    return config


def camera_parameters(config) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """返回相机矩阵、五参数畸变和标定图像尺寸。"""

    camera = config["camera"]
    camera_matrix = np.array(
        [
            [float(camera["_fx"]), 0.0, float(camera["_cx"])],
            [0.0, float(camera["_fy"]), float(camera["_cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.asarray(
        camera["distortion_coefficients"], dtype=np.float64
    ).reshape(-1)
    if distortion.shape != (5,) or not np.all(np.isfinite(distortion)):
        raise ValueError("distortion_coefficients 必须包含五个有限数值")

    image_size = (
        int(camera["calibrated_image_width"]),
        int(camera["calibrated_image_height"]),
    )
    return camera_matrix, distortion, image_size


def target_size_m(config) -> tuple[float, float]:
    """返回 V2 亮色内区的宽和高，单位为米。"""

    camera = config["camera"]
    return float(camera["target_W"]) / 100.0, float(camera["target_L"]) / 100.0


def target_object_points(config) -> np.ndarray:
    """构造 TL、TR、BR、BL 顺序的靶标平面角点。"""

    width, height = target_size_m(config)
    return np.array(
        [
            [-width / 2.0, -height / 2.0, 0.0],
            [width / 2.0, -height / 2.0, 0.0],
            [width / 2.0, height / 2.0, 0.0],
            [-width / 2.0, height / 2.0, 0.0],
        ],
        dtype=np.float64,
    )


def refine_target_corners(gray, corners) -> np.ndarray:
    """以检测四角为初值进行亚像素细化。"""

    points = np.asarray(corners, dtype=np.float32).reshape(4, 1, 2).copy()
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.01,
    )
    cv2.cornerSubPix(gray, points, (7, 7), (-1, -1), criteria)
    return points.reshape(4, 2).astype(np.float64)


def pose_reprojection_rms(corners, rotation, translation, config) -> float:
    """计算四个靶标角点的 RMS 重投影误差。"""

    camera_matrix, distortion, _ = camera_parameters(config)
    rotation_vector, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))
    projected, _ = cv2.projectPoints(
        target_object_points(config),
        rotation_vector,
        np.asarray(translation, dtype=np.float64).reshape(3),
        camera_matrix,
        distortion,
    )
    residuals = projected.reshape(4, 2) - np.asarray(corners, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))


def list_images(input_dir: Path) -> list[Path]:
    """按文件名返回目录第一层中的 PNG 标定图片。"""

    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def prepare_images(image_paths, config) -> list[ImageRecord]:
    """自动检测靶标、细化角点并预先筛选姿态质量。"""

    detector = InnerRegionTargetDetector(config)
    _, _, expected_size = camera_parameters(config)
    records = []

    for index, path in enumerate(image_paths):
        record = ImageRecord(path=path)
        records.append(record)
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            record.status = "image_failed"
            record.reason = "图片读取失败"
            continue

        height, width = frame.shape[:2]
        record.image_size = (width, height)
        if record.image_size != expected_size:
            record.status = "image_size_mismatch"
            record.reason = f"图片尺寸 {record.image_size} 与内参尺寸 {expected_size} 不一致"
            continue

        detection = detector.detect(
            FramePacket(frame=frame, frame_id=index, timestamp=0.0)
        )
        if not detection.found or detection.corners is None:
            record.status = "target_not_found"
            record.reason = "V2 靶标检测失败"
            continue

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            record.raw_corners = np.asarray(detection.corners, dtype=np.float64)
            record.refined_corners = refine_target_corners(gray, record.raw_corners)
            record.rotation, record.translation = estimate_target_pose(
                record.refined_corners, config
            )
            record.pose_rms_error_px = pose_reprojection_rms(
                record.refined_corners,
                record.rotation,
                record.translation,
                config,
            )
        except (ValueError, RuntimeError, cv2.error) as error:
            record.status = "pose_failed"
            record.reason = str(error)
            continue

        if record.pose_rms_error_px > POSE_MAX_RMS_ERROR_PX:
            record.status = "pose_reprojection_failed"
            record.reason = (
                f"靶标重投影 RMS {record.pose_rms_error_px:.3f} px "
                f"超过 {POSE_MAX_RMS_ERROR_PX:.1f} px"
            )
            continue

        record.status = "ready"

    return records


def aggregate_clicks(clicks) -> tuple[tuple[float, float], float]:
    """返回多次点击的逐坐标中位数和最大离散距离。"""

    points = np.asarray(clicks, dtype=np.float64)
    if points.shape != (CLICK_COUNT, 2):
        raise ValueError(f"每张图片必须提供 {CLICK_COUNT} 次点击")

    median = np.median(points, axis=0)
    spread = float(np.max(np.linalg.norm(points - median, axis=1)))
    return (float(median[0]), float(median[1])), spread


def resize_for_display(frame, max_width=1280, max_height=800):
    """缩小全图预览并返回原图到显示图的缩放比例。"""

    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return frame.copy(), 1.0
    resized = cv2.resize(
        frame,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def display_to_image_point(x, y, scale, offset_x=0.0, offset_y=0.0):
    """把显示窗口点击坐标映射回原图坐标。"""

    if scale <= 0.0:
        raise ValueError("显示缩放比例必须大于 0")
    return offset_x + float(x) / scale, offset_y + float(y) / scale


def _window_closed(name: str) -> bool:
    """检查用户是否关闭了 OpenCV 窗口。"""

    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def select_zoom_center(frame, record: ImageRecord, config):
    """在全图中选择激光点附近区域。"""

    window_name = "Laser Calibration - Select Spot"
    preview = frame.copy()
    corners = record.refined_corners.round().astype(np.int32)
    cv2.polylines(preview, [corners], True, (0, 255, 0), 2)

    camera_matrix, distortion, _ = camera_parameters(config)
    rotation_vector, _ = cv2.Rodrigues(record.rotation)
    cv2.drawFrameAxes(
        preview,
        camera_matrix,
        distortion,
        rotation_vector,
        record.translation,
        0.05,
        2,
    )
    cv2.putText(
        preview,
        "Click near laser spot | S skip | Q/Esc abort",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    translation = np.asarray(record.translation).reshape(3)
    cv2.putText(
        preview,
        (
            f"Pose t=({translation[0]:.3f}, {translation[1]:.3f}, "
            f"{translation[2]:.3f})m | RMS={record.pose_rms_error_px:.3f}px"
        ),
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    display, scale = resize_for_display(preview)
    state = {"point": None}

    def on_mouse(event, x, y, _flags, _parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["point"] = display_to_image_point(x, y, scale)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    cv2.imshow(window_name, display)

    while state["point"] is None:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("s"), ord("S")):
            cv2.destroyWindow(window_name)
            return "skip", None
        if key in (ord("q"), ord("Q"), 27) or _window_closed(window_name):
            cv2.destroyWindow(window_name)
            raise UserAbort("用户取消激光外参标定")

    cv2.destroyWindow(window_name)
    return "selected", state["point"]


def collect_zoom_clicks(frame, center):
    """在原图局部放大窗口中收集五次激光中心点击。"""

    height, width = frame.shape[:2]
    crop_width = min(2 * ZOOM_HALF_SIZE_PX, width)
    crop_height = min(2 * ZOOM_HALF_SIZE_PX, height)
    center_x = int(round(center[0]))
    center_y = int(round(center[1]))
    x0 = int(np.clip(center_x - crop_width // 2, 0, width - crop_width))
    y0 = int(np.clip(center_y - crop_height // 2, 0, height - crop_height))
    crop = frame[y0 : y0 + crop_height, x0 : x0 + crop_width]

    window_name = "Laser Calibration - Zoom"
    clicks = []

    def on_mouse(event, x, y, _flags, _parameter):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < CLICK_COUNT:
            clicks.append(
                display_to_image_point(x, y, ZOOM_SCALE, x0, y0)
            )
        elif event == cv2.EVENT_RBUTTONDOWN and clicks:
            clicks.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        display = cv2.resize(
            crop,
            None,
            fx=ZOOM_SCALE,
            fy=ZOOM_SCALE,
            interpolation=cv2.INTER_NEAREST,
        )
        for point in clicks:
            x = int(round((point[0] - x0) * ZOOM_SCALE))
            y = int(round((point[1] - y0) * ZOOM_SCALE))
            cv2.drawMarker(
                display,
                (x, y),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                16,
                1,
            )

        spread = None
        if len(clicks) == CLICK_COUNT:
            _, spread = aggregate_clicks(clicks)
        message = f"Clicks {len(clicks)}/{CLICK_COUNT}"
        if spread is not None:
            message += f" | spread {spread:.2f}px"
        cv2.putText(
            display,
            message,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            display,
            "Enter confirm | R reset | B back | S skip | Q abort",
            (10, display.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("r"), ord("R")):
            clicks.clear()
        elif key in (ord("b"), ord("B")):
            cv2.destroyWindow(window_name)
            return "back", [], None, None
        elif key in (ord("s"), ord("S")):
            cv2.destroyWindow(window_name)
            return "skip", [], None, None
        elif key in (ord("q"), ord("Q"), 27) or _window_closed(window_name):
            cv2.destroyWindow(window_name)
            raise UserAbort("用户取消激光外参标定")
        elif key in (10, 13) and len(clicks) == CLICK_COUNT:
            laser_pixel, spread = aggregate_clicks(clicks)
            if spread <= CLICK_MAX_SPREAD_PX:
                cv2.destroyWindow(window_name)
                return "confirmed", list(clicks), laser_pixel, spread
            print(
                f"点击最大离散距离 {spread:.2f} px 超过 "
                f"{CLICK_MAX_SPREAD_PX:.1f} px，请按 R 重新点击"
            )


def collect_laser_pixel(frame, record: ImageRecord, config):
    """完成一张图片的全图定位和局部精细点击。"""

    while True:
        status, center = select_zoom_center(frame, record, config)
        if status == "skip":
            return "skip", [], None, None

        status, clicks, laser_pixel, spread = collect_zoom_clicks(frame, center)
        if status == "back":
            continue
        return status, clicks, laser_pixel, spread


def reconstruct_hit_point(laser_pixel, rotation, translation, config) -> np.ndarray:
    """将原图激光像素与靶面求交，返回相机坐标系三维点。"""

    camera_matrix, distortion, _ = camera_parameters(config)
    pixel = np.asarray(laser_pixel, dtype=np.float64).reshape(1, 1, 2)
    normalized = cv2.undistortPoints(pixel, camera_matrix, distortion).reshape(2)
    ray = np.array([normalized[0], normalized[1], 1.0], dtype=np.float64)

    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    normal = rotation[:, 2]
    denominator = float(np.dot(normal, ray))
    if abs(denominator) < 1e-9:
        raise RuntimeError("相机射线与靶面平行")

    scale = float(np.dot(normal, translation) / denominator)
    if scale <= 0.0:
        raise RuntimeError("激光像素对应的靶面交点位于相机后方")

    hit_camera = scale * ray
    if not np.all(np.isfinite(hit_camera)):
        raise RuntimeError("激光三维命中点包含非有限值")

    hit_target = rotation.T @ (hit_camera - translation)
    target_width, target_height = target_size_m(config)
    if (
        abs(hit_target[0]) > target_width / 2.0 + TARGET_MARGIN_M
        or abs(hit_target[1]) > target_height / 2.0 + TARGET_MARGIN_M
    ):
        raise RuntimeError("人工点击对应的三维点不在靶标范围内")

    return hit_camera


def canonicalize_line(point, direction) -> tuple[np.ndarray, np.ndarray]:
    """规范化直线方向，并返回线上距相机光心最近的点。"""

    point = np.asarray(point, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("激光直线方向长度必须大于 0")
    direction = direction / norm
    if direction[2] < 0.0:
        direction = -direction
    origin = point - direction * float(np.dot(direction, point))
    return origin, direction


def fit_line_svd(points) -> tuple[np.ndarray, np.ndarray]:
    """使用总最小二乘拟合三维直线。"""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("拟合激光直线至少需要两个三维点")
    center = np.mean(points, axis=0)
    _, _, right_vectors = np.linalg.svd(points - center, full_matrices=False)
    return canonicalize_line(center, right_vectors[0])


def line_pixel_residuals(origin, direction, records, config) -> np.ndarray:
    """计算候选激光直线在各靶面上的回投影像素误差。"""

    residuals = np.full(len(records), np.inf, dtype=np.float64)
    for index, record in enumerate(records):
        try:
            predicted = predict_laser_pixel(
                origin,
                direction,
                record.rotation,
                record.translation,
                config,
            )
        except (ValueError, RuntimeError, cv2.error):
            continue
        residuals[index] = float(
            np.linalg.norm(np.asarray(predicted) - np.asarray(record.laser_pixel))
        )
    return residuals


def fit_laser_line(records, config) -> LineFitResult:
    """使用二点共识搜索和 SVD 精修拟合激光直线。"""

    points = np.asarray([record.hit_camera_m for record in records], dtype=np.float64)
    if len(points) < 2:
        raise RuntimeError("至少需要两个有效三维命中点")

    best_mask = None
    best_score = None
    for first, second in combinations(range(len(points)), 2):
        delta = points[second] - points[first]
        if np.linalg.norm(delta) < MIN_PAIR_DISTANCE_M:
            continue
        origin, direction = canonicalize_line(points[first], delta)
        residuals = line_pixel_residuals(origin, direction, records, config)
        mask = residuals <= RANSAC_MAX_ERROR_PX
        count = int(np.count_nonzero(mask))
        if count < 2:
            continue
        inlier_errors = residuals[mask]
        score = (-count, float(np.median(inlier_errors)), float(np.mean(inlier_errors)))
        if best_score is None or score < best_score:
            best_score = score
            best_mask = mask

    if best_mask is None:
        raise RuntimeError("没有找到可用的激光直线候选")

    mask = best_mask
    for _ in range(3):
        origin, direction = fit_line_svd(points[mask])
        residuals = line_pixel_residuals(origin, direction, records, config)
        new_mask = residuals <= RANSAC_MAX_ERROR_PX
        if np.count_nonzero(new_mask) < 2 or np.array_equal(new_mask, mask):
            break
        mask = new_mask

    origin, direction = fit_line_svd(points[mask])
    residuals = line_pixel_residuals(origin, direction, records, config)
    final_mask = residuals <= RANSAC_MAX_ERROR_PX
    if np.count_nonzero(final_mask) >= 2 and not np.array_equal(final_mask, mask):
        origin, direction = fit_line_svd(points[final_mask])
        residuals = line_pixel_residuals(origin, direction, records, config)
    mask = residuals <= RANSAC_MAX_ERROR_PX

    return LineFitResult(origin, direction, mask, residuals)


def leave_one_out_errors(records, inlier_mask, config) -> np.ndarray:
    """逐个排除内点并预测该样本，计算留一像素误差。"""

    errors = np.full(len(records), np.nan, dtype=np.float64)
    inlier_indices = np.flatnonzero(inlier_mask)
    for held_index in inlier_indices:
        retained = [index for index in inlier_indices if index != held_index]
        if len(retained) < 2:
            continue
        points = [records[index].hit_camera_m for index in retained]
        origin, direction = fit_line_svd(points)
        residual = line_pixel_residuals(
            origin, direction, [records[held_index]], config
        )[0]
        if np.isfinite(residual):
            errors[held_index] = residual
    return errors


def error_metrics(values) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """返回有限误差的 RMS、P95 和最大值。"""

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None, None
    rms = float(np.sqrt(np.mean(values**2)))
    return rms, float(np.percentile(values, 95)), float(np.max(values))


def evaluate_quality(records, fit: LineFitResult, loo_errors) -> dict:
    """汇总质量门槛、距离覆盖和警告。"""

    valid_count = len(records)
    inlier_count = int(np.count_nonzero(fit.inlier_mask))
    inlier_ratio = inlier_count / valid_count if valid_count else 0.0
    inlier_points = np.asarray(
        [record.hit_camera_m for record, keep in zip(records, fit.inlier_mask) if keep]
    )
    if len(inlier_points) >= 2:
        positions = inlier_points @ fit.direction_camera
        hit_span = float(np.max(positions) - np.min(positions))
    else:
        hit_span = 0.0

    fit_rms, fit_p95, fit_max = error_metrics(fit.residuals_px[fit.inlier_mask])
    loo_rms, loo_p95, loo_max = error_metrics(loo_errors)

    distances = [
        float(np.linalg.norm(record.translation))
        for record, keep in zip(records, fit.inlier_mask)
        if keep
    ]
    coverage = {
        "near_1_0_to_2_3_m": sum(1.0 <= value < 2.3 for value in distances),
        "middle_2_3_to_3_7_m": sum(2.3 <= value < 3.7 for value in distances),
        "far_3_7_to_5_0_m": sum(3.7 <= value <= 5.0 for value in distances),
    }

    warnings = []
    if valid_count < MIN_VALID_SAMPLES:
        warnings.append(f"有效样本 {valid_count} 少于 {MIN_VALID_SAMPLES}")
    if inlier_count < MIN_INLIERS:
        warnings.append(f"内点 {inlier_count} 少于 {MIN_INLIERS}")
    if inlier_ratio < MIN_INLIER_RATIO:
        warnings.append(f"内点率 {inlier_ratio:.1%} 低于 {MIN_INLIER_RATIO:.0%}")
    if hit_span < MIN_HIT_SPAN_M:
        warnings.append(f"三维命中点跨度 {hit_span:.3f} m 小于 {MIN_HIT_SPAN_M:.1f} m")
    if loo_rms is None or loo_rms > MAX_LOO_RMS_ERROR_PX:
        warnings.append(
            "留一 RMS 无法计算"
            if loo_rms is None
            else f"留一 RMS {loo_rms:.3f} px 超过 {MAX_LOO_RMS_ERROR_PX:.1f} px"
        )
    if loo_p95 is None or loo_p95 > MAX_LOO_P95_ERROR_PX:
        warnings.append(
            "留一 P95 无法计算"
            if loo_p95 is None
            else f"留一 P95 {loo_p95:.3f} px 超过 {MAX_LOO_P95_ERROR_PX:.1f} px"
        )
    for name, count in coverage.items():
        if count < 5:
            warnings.append(f"距离分段 {name} 仅有 {count} 个内点，建议至少 5 个")

    hard_quality_passed = (
        valid_count >= MIN_VALID_SAMPLES
        and inlier_count >= MIN_INLIERS
        and inlier_ratio >= MIN_INLIER_RATIO
        and hit_span >= MIN_HIT_SPAN_M
        and loo_rms is not None
        and loo_rms <= MAX_LOO_RMS_ERROR_PX
        and loo_p95 is not None
        and loo_p95 <= MAX_LOO_P95_ERROR_PX
    )
    return {
        "passed": hard_quality_passed,
        "valid_sample_count": valid_count,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "hit_span_m": hit_span,
        "fit_rms_error_px": fit_rms,
        "fit_p95_error_px": fit_p95,
        "fit_max_error_px": fit_max,
        "loo_rms_error_px": loo_rms,
        "loo_p95_error_px": loo_p95,
        "loo_max_error_px": loo_max,
        "distance_coverage": coverage,
        "warnings": warnings,
    }


def _optional_array(value):
    """把可选 NumPy 数组转换为 JSON 列表。"""

    if value is None:
        return None
    return np.asarray(value).tolist()


def record_payload(record: ImageRecord, input_dir: Path) -> dict:
    """构造单张图片的可复算 JSON 数据。"""

    try:
        image_name = record.path.relative_to(input_dir).as_posix()
    except ValueError:
        image_name = record.path.name
    return {
        "image": image_name,
        "status": record.status,
        "reason": record.reason,
        "image_size": list(record.image_size) if record.image_size else None,
        "raw_corners_px": _optional_array(record.raw_corners),
        "refined_corners_px": _optional_array(record.refined_corners),
        "rotation_target_to_camera": _optional_array(record.rotation),
        "translation_target_to_camera_m": _optional_array(record.translation),
        "pose_rms_error_px": record.pose_rms_error_px,
        "laser_clicks_px": [list(point) for point in record.clicks],
        "laser_pixel_px": list(record.laser_pixel) if record.laser_pixel else None,
        "click_max_spread_px": record.click_max_spread_px,
        "hit_camera_m": _optional_array(record.hit_camera_m),
        "is_inlier": record.is_inlier,
        "fit_error_px": record.fit_error_px,
        "loo_error_px": record.loo_error_px,
    }


def build_output_payload(config, input_dir, records, fit=None, quality=None) -> dict:
    """构造完整激光外参标定结果。"""

    camera_matrix, distortion, image_size = camera_parameters(config)
    if fit is None:
        status = "failed"
        laser_line = None
    else:
        status = "success" if quality["passed"] else "warning"
        laser_line = {
            "coordinate_frame": "camera_optical",
            "origin_definition": "closest_point_to_camera_origin",
            "origin_camera_m": fit.origin_camera_m.tolist(),
            "direction_camera": fit.direction_camera.tolist(),
        }

    return {
        "schema_version": 1,
        "status": status,
        "laser_line": laser_line,
        "camera": {
            "model": "opencv_pinhole",
            "image_width": image_size[0],
            "image_height": image_size[1],
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients_order": ["k1", "k2", "p1", "p2", "k3"],
            "distortion_coefficients": distortion.tolist(),
        },
        "target": {
            "detector": "InnerRegionTargetDetector",
            "width_m": target_size_m(config)[0],
            "height_m": target_size_m(config)[1],
        },
        "thresholds": {
            "click_count": CLICK_COUNT,
            "click_max_spread_px": CLICK_MAX_SPREAD_PX,
            "pose_max_rms_error_px": POSE_MAX_RMS_ERROR_PX,
            "ransac_max_error_px": RANSAC_MAX_ERROR_PX,
            "minimum_valid_samples": MIN_VALID_SAMPLES,
            "minimum_inliers": MIN_INLIERS,
            "minimum_inlier_ratio": MIN_INLIER_RATIO,
            "minimum_hit_span_m": MIN_HIT_SPAN_M,
            "maximum_loo_rms_error_px": MAX_LOO_RMS_ERROR_PX,
            "maximum_loo_p95_error_px": MAX_LOO_P95_ERROR_PX,
        },
        "quality": quality,
        "samples": [record_payload(record, input_dir) for record in records],
        "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_json_atomically(output_path: Path, payload: dict) -> None:
    """先写同目录临时文件，再原子替换最终 JSON。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def print_laser_yaml(fit: LineFitResult) -> None:
    """打印可手动复制到 config.yaml 的激光外参。"""

    origin = ", ".join(f"{value:.12g}" for value in fit.origin_camera_m)
    direction = ", ".join(f"{value:.12g}" for value in fit.direction_camera)
    print("\nlaser:")
    print(f"  origin_camera_m: [{origin}]")
    print(f"  direction_camera: [{direction}]")


def run_calibration(input_dir: Path, output_path: Path, config) -> int:
    """执行预扫描、人工点击、鲁棒拟合和结果保存。"""

    image_paths = list_images(input_dir)
    if not image_paths:
        raise RuntimeError(f"标定目录中没有图片：{input_dir}")

    print(f"开始预扫描 {len(image_paths)} 张图片")
    records = prepare_images(image_paths, config)
    ready_records = [record for record in records if record.status == "ready"]
    print(f"靶标姿态预扫描通过 {len(ready_records)} 张")
    for record in records:
        if record.status != "ready":
            print(f"[跳过] {record.path.name}: {record.reason}")

    accepted = []
    for index, record in enumerate(ready_records, start=1):
        frame = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if frame is None:
            record.status = "image_failed"
            record.reason = "人工点击前图片读取失败"
            print(f"[跳过] {record.path.name}: {record.reason}")
            continue
        print(f"[{index}/{len(ready_records)}] 点击 {record.path.name}")
        status, clicks, laser_pixel, spread = collect_laser_pixel(frame, record, config)
        if status == "skip":
            record.status = "user_skipped"
            record.reason = "用户跳过"
            continue

        record.clicks = clicks
        record.laser_pixel = laser_pixel
        record.click_max_spread_px = spread
        try:
            record.hit_camera_m = reconstruct_hit_point(
                laser_pixel,
                record.rotation,
                record.translation,
                config,
            )
        except (ValueError, RuntimeError, cv2.error) as error:
            record.status = "geometry_failed"
            record.reason = str(error)
            print(f"[跳过] {record.path.name}: {record.reason}")
            continue

        record.status = "accepted"
        accepted.append(record)

    cv2.destroyAllWindows()

    if len(accepted) < 2:
        quality = {
            "passed": False,
            "warnings": ["少于两个有效三维命中点，无法拟合激光直线"],
        }
        payload = build_output_payload(config, input_dir, records, quality=quality)
        write_json_atomically(output_path, payload)
        print(f"标定失败报告已保存：{output_path}")
        return 1

    try:
        fit = fit_laser_line(accepted, config)
    except (ValueError, RuntimeError, cv2.error) as error:
        quality = {
            "passed": False,
            "valid_sample_count": len(accepted),
            "warnings": [f"激光直线在数学上不可解：{error}"],
        }
        payload = build_output_payload(config, input_dir, records, quality=quality)
        write_json_atomically(output_path, payload)
        print(f"标定失败报告已保存：{output_path}")
        return 1

    loo_errors = leave_one_out_errors(accepted, fit.inlier_mask, config)
    for index, record in enumerate(accepted):
        record.is_inlier = bool(fit.inlier_mask[index])
        record.fit_error_px = (
            float(fit.residuals_px[index])
            if np.isfinite(fit.residuals_px[index])
            else None
        )
        record.loo_error_px = (
            float(loo_errors[index]) if np.isfinite(loo_errors[index]) else None
        )

    quality = evaluate_quality(accepted, fit, loo_errors)
    payload = build_output_payload(config, input_dir, records, fit, quality)
    write_json_atomically(output_path, payload)

    if quality["passed"]:
        print(f"\n激光外参标定通过，结果已保存：{output_path}")
        for warning in quality["warnings"]:
            print(f"- 覆盖警告：{warning}")
    else:
        print(f"\n警告：激光外参未达到质量门槛，结果仍已保存：{output_path}")
        for warning in quality["warnings"]:
            print(f"- {warning}")
    print_laser_yaml(fit)
    return 0 if quality["passed"] else 2


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_dir.is_dir():
        print(f"标定图片目录不存在：{input_dir}")
        return 1
    if output_path.exists() and not args.force:
        print(f"输出文件已经存在：{output_path}；需要覆盖时请使用 --force")
        return 1

    try:
        config = load_config(config_path)
        return run_calibration(input_dir, output_path, config)
    except UserAbort as error:
        cv2.destroyAllWindows()
        print(f"{error}，本次未写入结果文件")
        return 130
    except (KeyError, TypeError, OSError, ValueError, RuntimeError, cv2.error) as error:
        cv2.destroyAllWindows()
        print(f"激光外参标定失败：{error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
