"""根据靶标四角估计靶标坐标系相对相机的位姿。"""

import cv2
import numpy as np


def estimate_target_pose(corners, config) -> tuple[np.ndarray, np.ndarray]:
    """返回靶标坐标到相机光学坐标的 ``R_CT`` 和 ``t_CT``。

    角点应按左上、右上、右下、左下排列。靶标坐标原点位于矩形中心，
    ``x`` 向右、``y`` 向下、``z`` 垂直靶面；平移向量单位为米。
    """

    image_points = np.asarray(corners, dtype=np.float64)
    if image_points.shape != (4, 2):
        raise ValueError("corners 必须是形状为 (4, 2) 的数组")

    camera_config = config["camera"]
    fx = float(camera_config["_fx"])
    fy = float(camera_config["_fy"])
    cx = float(camera_config["_cx"])
    cy = float(camera_config["_cy"])
    width_m = float(camera_config["target_W"]) / 100.0
    length_m = float(camera_config["target_L"]) / 100.0

    if fx <= 0.0 or fy <= 0.0 or width_m <= 0.0 or length_m <= 0.0:
        raise ValueError("相机焦距和靶标尺寸必须大于 0")

    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    object_points = np.array(
        [
            [-width_m / 2.0, -length_m / 2.0, 0.0],
            [width_m / 2.0, -length_m / 2.0, 0.0],
            [width_m / 2.0, length_m / 2.0, 0.0],
            [-width_m / 2.0, length_m / 2.0, 0.0],
        ],
        dtype=np.float64,
    )

    # 当前配置未记录畸变系数，输入角点按已去畸变或畸变可忽略处理。
    distortion = np.zeros((5, 1), dtype=np.float64)
    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not success:
        raise RuntimeError("靶标位姿解算失败")

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    return rotation_matrix, translation_vector.reshape(3)
