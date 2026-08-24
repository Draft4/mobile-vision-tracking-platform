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
    distortion = np.asarray(
        camera_config["distortion_coefficients"], dtype=np.float64
    ).reshape(-1)
    width_m = float(camera_config["target_W"]) / 100.0
    length_m = float(camera_config["target_L"]) / 100.0

    if fx <= 0.0 or fy <= 0.0 or width_m <= 0.0 or length_m <= 0.0:
        raise ValueError("相机焦距和靶标尺寸必须大于 0")
    if distortion.shape != (5,) or not np.all(np.isfinite(distortion)):
        raise ValueError("distortion_coefficients 必须包含五个有限数值")

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

    success, rotation_vectors, translation_vectors, _ = cv2.solvePnPGeneric(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not success:
        raise RuntimeError("靶标位姿解算失败")

    best_result = None
    best_error = float("inf")

    for rotation_vector, translation_vector in zip(
        rotation_vectors, translation_vectors
    ):
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        translation = translation_vector.reshape(3)
        camera_points = (rotation_matrix @ object_points.T).T + translation
        if np.any(camera_points[:, 2] <= 0.0):
            continue

        projected, _ = cv2.projectPoints(
            object_points,
            rotation_vector,
            translation_vector,
            camera_matrix,
            distortion,
        )
        residuals = projected.reshape(4, 2) - image_points
        rms_error = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
        if rms_error < best_error:
            best_result = (rotation_matrix, translation)
            best_error = rms_error

    if best_result is None:
        raise RuntimeError("靶标位姿解算结果位于相机后方")

    return best_result
