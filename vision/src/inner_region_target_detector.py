"""V2 内区矩形法靶标检测器。

本算法将亮色内区作为二值图前景，筛选面积合理的凸四边形，再向四边形外侧
扩展窄带以验证黑色边框。通过硬性过滤的候选根据面积、内部白色比例、外侧
黑色比例和灰度对比度加权评分，最终返回得分最高的目标。

输入：FramePacket；输出：TargetDetection。
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .data_types import FramePacket, TargetDetection


# 仅用于人工观察预处理结果；实时检测时开启会在每帧覆盖同一张调试图片。
DEBUG_BINARY_OUTPUT = False


class InnerRegionTargetDetector:
    """检测黑框包围的亮色矩形内区。

    检测参数读取自配置文件的 ``target_v2`` 节。检测器不保存帧间状态，可与 V1
    使用相同的 ``detect(FramePacket) -> TargetDetection`` 接口进行对照测试。
    """

    def __init__(self, config):
        self.config = config
        self._validate_config()

    def detect(self, frame_packet: FramePacket) -> TargetDetection:
        """检测单帧中评分最高的亮色内区候选。"""

        frame = frame_packet.frame

        # 输入检测
        if frame is None:
            return TargetDetection(found=False)
        if frame.size == 0:
            return TargetDetection(found=False)

        # 二值化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = self._preprocess(gray)

        if binary is None:
            return TargetDetection(found=False)

        # RETR_LIST 保留嵌套在黑框中的亮色内区轮廓，不依赖完整的层级关系。
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return TargetDetection(found=False)

        # 计算图像的面积
        image_height, image_width = binary.shape[:2]
        image_area = float(image_height * image_width)

        # 所有几何与黑白比例硬过滤通过后，只保留加权评分最高的候选。
        best_result: Optional[TargetDetection] = None
        best_score = -1.0

        for contour in contours:
            # 面积过滤
            area = cv2.contourArea(contour)
            area_ratio = area / image_area

            if (
                not self.config["target_v2"]["min_inner_area_ratio"]
                <= area_ratio
                <= self.config["target_v2"]["max_inner_area_ratio"]
            ):
                continue

            # 排除超出画面范围的目标
            x, y, w, h = cv2.boundingRect(contour)

            if x <= 0 or y <= 0:
                continue
            if x + w > image_width - 1 or y + h > image_height - 1:
                continue

            # 凸包消除边缘的小凹陷，使后续四边形拟合更稳定。
            hull = cv2.convexHull(contour)

            # 多边形拟合
            perimeter = cv2.arcLength(hull, True)

            if perimeter <= 0:
                continue

            epsilon = (
                float(self.config["target_v2"]["approx_epsilon_ratio"]) * perimeter
            )
            approx = cv2.approxPolyDP(hull, epsilon, True)

            # 内区边缘在透视投影下仍应近似为凸四边形。
            if len(approx) != 4:
                continue
            if not cv2.isContourConvex(approx):
                continue

            # 计算四个角点并统一为 TL、TR、BR、BL 顺序。
            corners = approx.reshape(4, 2).astype(np.float32)
            corners = self._order_corners(corners)

            # 沿中心向外等比例扩张，构造用于验证黑框的外侧窄带。
            center = np.mean(corners, axis=0)

            expanded_corners = center + self.config["target_v2"]["outer_scale"] * (
                corners - center
            )

            # outer_mask - inner_mask 得到候选四边形外侧的窄带掩膜。
            inner_mask = np.zeros_like(binary)
            outer_mask = np.zeros_like(binary)

            cv2.fillConvexPoly(inner_mask, corners.round().astype(np.int32), 255)
            cv2.fillConvexPoly(
                outer_mask, expanded_corners.round().astype(np.int32), 255
            )

            # 内区允许存在圆环等少量暗色图案，但主体应保持为白色。
            inner_pixels = inner_mask > 0
            if np.any(inner_pixels):
                white_ratio = np.mean(binary[inner_pixels] == 255)
            else:
                white_ratio = 0.0

            # 内部白色像素的比例低于最小值直接排除
            if white_ratio < self.config["target_v2"]["min_inner_white_ratio"]:
                continue

            # 外侧窄带应主要落在靶标的黑色边框上。
            band_mask = cv2.subtract(outer_mask, inner_mask)
            band_pixels = band_mask > 0

            # 计算窄带中的黑色像素比例。
            if np.any(band_pixels):
                dark_ratio = np.mean(binary[band_pixels] == 0)
            else:
                dark_ratio = 0.0

            # 黑色像素比例低于最小值时，候选外侧不具备稳定黑框特征。
            if dark_ratio < self.config["target_v2"]["min_outer_dark_ratio"]:
                continue

            # 计算内部和外部的平均亮度，以便后续计算对比得分
            inner_gray_mean = np.mean(gray[inner_pixels])
            band_gray_mean = np.mean(gray[band_pixels])
            contrast_gray_mean = abs(inner_gray_mean - band_gray_mean)

            # 过滤内区与外侧暗带平均灰度差不足的候选。
            if contrast_gray_mean < self.config["target_v2"]["min_score_contrast"]:
                continue

            # 计算得分
            score = self._score_candidate(
                area_ratio=area_ratio,
                white_ratio=white_ratio,
                dark_ratio=dark_ratio,
                contrast_gray_mean=contrast_gray_mean,
            )

            # 进行对比，找到最佳
            if score > best_score:
                center = self._compute_center(corners)
                best_result = TargetDetection(
                    found=True,
                    corners=corners,
                    center=center,
                    area=float(area),
                    confidence=float(score),
                )
                best_score = score

        if best_result is None:
            return TargetDetection(found=False)

        if best_result.confidence < self.config["target_v2"]["min_confidence"]:
            return TargetDetection(found=False)

        return best_result

    def _preprocess(self, gray) -> np.ndarray:
        """将灰度图转换为突出亮色内区的普通二值图。"""

        # 高斯滤波
        k = self.config["target_v2"]["blur_kernel_size"]
        blurred = cv2.GaussianBlur(gray, (k, k), 0)

        # 二值化
        if self.config["target_v2"]["threshold_mode"] == "otsu":
            _, binary = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            _, binary = cv2.threshold(
                blurred,
                self.config["target_v2"]["threshold_value"],
                255,
                cv2.THRESH_BINARY,
            )

        # 形态学处理
        morph_size = self.config["target_v2"]["morph_kernel_size"]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))

        # 闭运算
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self.config["target_v2"]["morph_close_iterations"],
        )

        # 开运算
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
            iterations=self.config["target_v2"]["morph_open_iterations"],
        )

        if DEBUG_BINARY_OUTPUT:
            cv2.imshow("Inner Region Binary", binary)
            binary_path = (
                Path(__file__).resolve().parents[1] / "data" / "inner_region_binary.png"
            )
            if not cv2.imwrite(str(binary_path), binary):
                raise RuntimeError(f"二值图保存失败: {binary_path}")

        return binary

    @staticmethod
    def _order_corners(points) -> np.ndarray:
        """将四个角点按左上角起始、图像坐标系顺时针排序。"""

        points = np.asarray(points, dtype=np.float32)

        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        ordered = points[np.argsort(angles)]

        # 从最接近左上的点开始，同时保留四点的顺时针环形顺序
        start_index = int(np.argmin(ordered.sum(axis=1)))
        return np.roll(ordered, -start_index, axis=0)

    def _score_candidate(self, area_ratio, white_ratio, dark_ratio, contrast_gray_mean):
        """按面积、黑白像素比例和灰度对比度计算归一化加权得分。"""
        area_score = self.config["target_v2"]["area_score_weight"] * np.clip(
            area_ratio / self.config["target_v2"]["full_size_score_area_ratio"],
            0.0,
            1.0,
        )

        white_score = self.config["target_v2"]["white_score_weight"] * white_ratio

        dark_score = self.config["target_v2"]["dark_score_weight"] * dark_ratio

        contrast_score = self.config["target_v2"]["contrast_score_weight"] * np.clip(
            contrast_gray_mean / self.config["target_v2"]["full_size_score_contrast"],
            0.0,
            1.0,
        )

        score = area_score + white_score + dark_score + contrast_score

        return score

    @staticmethod
    def _compute_center(corners) -> tuple[float, float]:
        """根据四边形两条对角线的交点计算靶心。

        投影会改变边长和角度，但保持直线的相交关系，因此该交点仍对应真实
        矩形两条对角线的交点。
        """

        tl = corners[0]
        tr = corners[1]
        br = corners[2]
        bl = corners[3]

        center = InnerRegionTargetDetector._line_intersection(tl, br, tr, bl)

        # 数值退化时使用顶点均值作为保底结果。
        if center is None:
            center_array = np.mean(corners, axis=0)
            return (float(center_array[0]), float(center_array[1]))

        return center

    @staticmethod
    def _line_intersection(p1, p2, p3, p4) -> Optional[tuple[float, float]]:
        """计算直线 ``p1-p2`` 与 ``p3-p4`` 的交点。

        如果两条线近似平行，返回 None
        """

        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        # denominator 接近 0 代表两条直线近似平行
        if abs(denominator) < 1e-6:
            return None

        determinant_1 = x1 * y2 - y1 * x2

        determinant_2 = x3 * y4 - y3 * x4

        px = (determinant_1 * (x3 - x4) - (x1 - x2) * determinant_2) / denominator

        py = (determinant_1 * (y3 - y4) - (y1 - y2) * determinant_2) / denominator

        return (float(px), float(py))

    def _validate_config(self) -> None:
        """在处理图像前检查 V2 内区检测参数。"""

        target_config = self.config.get("target_v2")
        if not isinstance(target_config, dict):
            raise ValueError("缺少 target_v2 配置段")

        required_keys = {
            "threshold_mode",
            "threshold_value",
            "blur_kernel_size",
            "morph_kernel_size",
            "morph_close_iterations",
            "morph_open_iterations",
            "min_inner_area_ratio",
            "max_inner_area_ratio",
            "approx_epsilon_ratio",
            "outer_scale",
            "min_inner_white_ratio",
            "min_outer_dark_ratio",
            "full_size_score_area_ratio",
            "full_size_score_contrast",
            "area_score_weight",
            "white_score_weight",
            "dark_score_weight",
            "contrast_score_weight",
            "min_confidence",
        }
        missing_keys = sorted(required_keys - target_config.keys())
        if missing_keys:
            raise ValueError(f"target_v2 缺少配置项: {', '.join(missing_keys)}")

        if target_config["threshold_mode"] not in {"otsu", "fixed"}:
            raise ValueError("target_v2.threshold_mode 必须为 otsu 或 fixed")

        if not 0 <= target_config["threshold_value"] <= 255:
            raise ValueError("target_v2.threshold_value 必须位于 [0, 255]")

        blur_kernel_size = target_config["blur_kernel_size"]
        if (
            not isinstance(blur_kernel_size, int)
            or isinstance(blur_kernel_size, bool)
            or blur_kernel_size <= 0
            or blur_kernel_size % 2 == 0
        ):
            raise ValueError("target_v2.blur_kernel_size 必须为正奇数")

        morph_kernel_size = target_config["morph_kernel_size"]
        if (
            not isinstance(morph_kernel_size, int)
            or isinstance(morph_kernel_size, bool)
            or morph_kernel_size <= 0
        ):
            raise ValueError("target_v2.morph_kernel_size 必须为正整数")

        for name in ("morph_close_iterations", "morph_open_iterations"):
            value = target_config[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"target_v2.{name} 必须为非负整数")

        if not (
            0.0
            < target_config["min_inner_area_ratio"]
            < target_config["max_inner_area_ratio"]
            < 1.0
        ):
            raise ValueError("target_v2 内区面积比例范围错误")

        if not 0.0 < target_config["approx_epsilon_ratio"] < 1.0:
            raise ValueError("target_v2.approx_epsilon_ratio 必须位于 (0, 1)")

        if target_config["outer_scale"] <= 1.0:
            raise ValueError("target_v2.outer_scale 必须大于 1")

        if not 0.0 <= target_config["min_inner_white_ratio"] <= 1.0:
            raise ValueError("target_v2.min_inner_white_ratio 必须位于 [0, 1]")

        if not 0.0 <= target_config["min_outer_dark_ratio"] <= 1.0:
            raise ValueError("target_v2.min_outer_dark_ratio 必须位于 [0, 1]")

        if not 0.0 < target_config["full_size_score_area_ratio"] <= 1.0:
            raise ValueError("target_v2.full_size_score_area_ratio 必须位于 (0, 1]")

        if not 0.0 < target_config["full_size_score_contrast"] <= 255.0:
            raise ValueError("target_v2.full_size_score_contrast 必须位于 (0, 255]")

        weight_names = (
            "area_score_weight",
            "white_score_weight",
            "dark_score_weight",
            "contrast_score_weight",
        )
        weights = [target_config[name] for name in weight_names]
        if any(not 0.0 <= weight <= 1.0 for weight in weights):
            raise ValueError("target_v2 评分权重必须位于 [0, 1]")
        if not np.isclose(sum(weights), 1.0, atol=1e-6):
            raise ValueError("target_v2 评分权重之和必须为 1")

        if not 0.0 <= target_config["min_confidence"] <= 1.0:
            raise ValueError("target_v2.min_confidence 必须位于 [0, 1]")
