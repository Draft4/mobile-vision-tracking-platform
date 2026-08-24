"""V1 外框轮廓法靶标检测器（实验基线）。

方法流程：反色二值化、先开后闭的形态学处理、查找带内部子轮廓的黑色外框、
四边形拟合，以及内外区域亮度评分。

适用条件：黑框闭合且具有足够像素宽度，黑框外侧背景能够在二值图中
与黑框分离。明亮、均匀的屏幕背景通常较容易满足这些条件。

已知局限：暗背景可能与黑框粘连；较细、断裂或光照不均的打印边框
可能无法形成闭环；变暗的内部圆环可能改变轮廓拓扑。面积、子轮廓和
四边形检查发生在评分之前，因此降低置信度不能修复这些失败。

本实现继续用于对照实验、参数验证和明亮背景场景，不作为最终纸质
靶标方案。下一版计划改为检测靶标内部亮色矩形，并验证其外侧暗色带。

输入：FramePacket；输出：TargetDetection。
"""

from typing import Optional

import cv2
import numpy as np

from .data_types import FramePacket, TargetDetection


class OuterFrameTargetDetector:
    """基于黑色外框轮廓检测靶标的 V1 实验基线。

    检测参数从传入配置的 ``target`` 节读取。检测器不保存帧间状态，因此同一
    实例可以依次处理不同帧。预处理函数中保留了 ``binary`` 调试窗口代码，默认
    关闭；需要观察二值图时可取消对应注释，并由调用程序处理 OpenCV 窗口事件。
    """

    def __init__(self, config):
        self.config = config
        self._validate_config()

    def detect(self, frame_packet: FramePacket) -> TargetDetection:
        """检测单帧中的最佳外框候选。

        无有效输入、没有候选或最佳候选低于置信度门槛时，返回
        ``TargetDetection(found=False)``。
        """

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

        # CCOMP 提供两级轮廓层级，用于验证黑框是否包含内部孔洞。
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            return TargetDetection(found=False)
        if hierarchy is None:
            return TargetDetection(found=False)

        # 计算图像的面积
        image_height, image_width = gray.shape[:2]
        image_area = float(image_height * image_width)

        # 所有硬性几何过滤通过后，仅保留评分最高的候选。
        best_result: Optional[TargetDetection] = None
        best_score = -1.0

        # 对轮廓进行遍历
        for index, contour in enumerate(contours):
            # 只考虑最外层轮廓
            parent_index = hierarchy[0][index][3]
            if parent_index != -1:
                continue

            # 面积过滤
            contour_area = cv2.contourArea(contour)

            if contour_area <= 0:
                continue

            # 黑框在反色二值图中应当形成一个带内部孔洞的外轮廓
            first_child = hierarchy[0][index][2]
            if first_child == -1:
                continue

            inner_area_ratio = cv2.contourArea(contours[first_child]) / contour_area
            if not (
                self.config["target"]["min_inner_area_ratio"]
                <= inner_area_ratio
                <= self.config["target"]["max_inner_area_ratio"]
            ):
                continue

            area_ratio = contour_area / image_area

            if (
                area_ratio < self.config["target"]["min_area_ratio"]
                or area_ratio > self.config["target"]["max_area_ratio"]
            ):
                continue

            # 凸包可消除边缘的小凹陷，使后续四边形拟合更稳定。
            hull = cv2.convexHull(contour)

            perimeter = cv2.arcLength(hull, True)

            if perimeter <= 0:
                continue

            # 多边形逼近
            epsilon = self.config["target"]["approx_epsilon_ratio"] * perimeter
            approx = cv2.approxPolyDP(hull, epsilon, True)

            # 检测是否为四边形
            if len(approx) != 4:
                continue

            # 计算出四个角点的坐标
            corners = approx.reshape(4, 2).astype(np.float32)
            # 统一角点顺序
            corners = self._order_corners(corners)

            # 角点贴住画面边缘时通常表示外框被裁切，不能作为完整靶标使用。
            if (
                np.any(corners[:, 0] <= 0)
                or np.any(corners[:, 0] >= image_width - 1)
                or np.any(corners[:, 1] <= 0)
                or np.any(corners[:, 1] >= image_height - 1)
            ):
                continue

            # 对候选四边形打分
            score = self._score_candidate(
                gray=gray,
                corners=corners,
                area_ratio=area_ratio,
            )

            # 进行对比，找到最佳
            if score > best_score:
                center = self._compute_center(corners)
                best_result = TargetDetection(
                    found=True,
                    corners=corners,
                    center=center,
                    area=float(contour_area),
                    confidence=float(score),
                )

                best_score = score

        if best_result is None:
            return TargetDetection(False)

        if best_result.confidence < self.config["target"]["min_confidence"]:
            return TargetDetection(False)

        return best_result

    def _preprocess(self, gray) -> np.ndarray:
        """将灰度图转换为突出暗色区域的反色二值图。"""

        # 高斯滤波
        k = self.config["target"]["blur_kernel_size"]

        blurred = cv2.GaussianBlur(gray, (k, k), 0)

        # 二值化
        if self.config["target"]["threshold_mode"] == "otsu":
            _, binary = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        elif self.config["target"]["threshold_mode"] == "fixed":
            _, binary = cv2.threshold(
                blurred,
                self.config["target"]["threshold_value"],
                255,
                cv2.THRESH_BINARY_INV,
            )
        else:
            print("不支持当前 threshold_mode")
            return None

        # 形态学处理
        morph_size = self.config["target"]["morph_kernel_size"]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))

        # 开运算
        if self.config["target"]["morph_open_iterations"] > 0:
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                kernel,
                iterations=self.config["target"]["morph_open_iterations"],
            )

        # 闭运算
        if self.config["target"]["morph_close_iterations"] > 0:
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=self.config["target"]["morph_close_iterations"],
            )

        # 需要现场观察阈值分割效果时，可临时取消下面一行的注释。
        # cv2.imshow("binary", binary)
        return binary

    def _score_candidate(self, gray, corners, area_ratio) -> float:
        """根据内外亮度差、边框黑度和目标尺寸对候选评分。

        最佳匹配目标应该满足：
            1. 面积不能太小
            2. 外围区域比较暗
            3. 内部区域比外围亮
            4. 黑框通常形成“外轮廓 + 内轮廓”

        最终的得分范围为：
            0 ~ 1
        """

        # 以下 mask 只评估候选四边形内部，不包含黑框外侧背景。
        outer_mask = np.zeros_like(gray, dtype=np.uint8)

        outer_polygon = corners.round().astype(np.int32)

        cv2.fillConvexPoly(outer_mask, outer_polygon, 255)

        # 构造内部区域
        geometric_center = np.mean(corners, axis=0)

        inner_corners = geometric_center + self.config["target"]["inner_scale"] * (
            corners - geometric_center
        )

        inner_polygon = inner_corners.round().astype(np.int32)

        inner_mask = np.zeros_like(gray, dtype=np.uint8)

        cv2.fillConvexPoly(inner_mask, inner_polygon, 255)

        # 外围区域 = outer - inner
        border_mask = cv2.subtract(outer_mask, inner_mask)

        # 防止 mask 为空
        border_pixels = cv2.countNonZero(border_mask)
        inner_pixels = cv2.countNonZero(inner_mask)

        if border_pixels == 0 or inner_pixels == 0:
            return 0.0

        # 计算边缘和内部平均灰度

        mean_border = cv2.mean(gray, mask=border_mask)[0]
        mean_inner = cv2.mean(gray, mask=inner_mask)[0]

        # 比较外围和内部的平均灰度得分，差别越大分数越高
        contrast_score = (mean_inner - mean_border) / 255.0
        contrast_score = float(np.clip(contrast_score, 0.0, 1.0))

        # 评价外围黑框的黑度
        darkness_score = (255.0 - mean_border) / 255.0
        darkness_score = float(np.clip(darkness_score, 0.0, 1.0))

        # 黑框必须明显暗于内部，避免较大的纯黑四边形仅凭面积通过
        if (
            contrast_score < self.config["target"]["min_contrast_score"]
            or darkness_score < self.config["target"]["min_darkness_score"]
        ):
            return 0.0

        # 面积评分
        size_score = float(
            np.clip(
                area_ratio / self.config["target"]["full_size_score_area_ratio"],
                0.0,
                1.0,
            )
        )

        # 综合评分
        score = 0.50 * contrast_score + 0.30 * darkness_score + 0.20 * size_score

        return float(np.clip(score, 0.0, 1.0))

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

        center = OuterFrameTargetDetector._line_intersection(tl, br, tr, bl)

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
        """在处理图像前检查 V1 外框检测参数。"""

        if (
            self.config["target"]["blur_kernel_size"] <= 0
            or self.config["target"]["blur_kernel_size"] % 2 == 0
        ):
            raise ValueError("blur_kernel_size 必须为正奇数")

        if self.config["target"]["morph_kernel_size"] <= 0:
            raise ValueError("morph_kernel_size 必须为正数")

        if not 0.0 < self.config["target"]["inner_scale"] < 1.0:
            raise ValueError("ineer_scale 必须位于 （0，1）")

        if not (
            0.0
            <= self.config["target"]["min_area_ratio"]
            < self.config["target"]["max_area_ratio"]
            <= 1.0
        ):
            raise ValueError("area_ratio 参数范围错误")

        if not (
            self.config["target"]["threshold_mode"] == "otsu"
            or self.config["target"]["threshold_mode"] == "fixed"
        ):
            raise ValueError("二值化 threshold_mode 配置错误")

        if not (0 <= self.config["target"]["threshold_value"] <= 255):
            raise ValueError("threshold_value 参数范围错误")

        if not (
            0.0
            < self.config["target"]["min_inner_area_ratio"]
            < self.config["target"]["max_inner_area_ratio"]
            < 1.0
        ):
            raise ValueError("inner_area_ratio 参数范围错误")

        if not (
            0.0 <= self.config["target"]["min_contrast_score"] <= 1.0
            and 0.0 <= self.config["target"]["min_darkness_score"] <= 1.0
        ):
            raise ValueError("亮度评分参数范围错误")

        if self.config["target"]["full_size_score_area_ratio"] <= 0.0:
            raise ValueError("full_size_score_area_ratio 必须大于 0")
