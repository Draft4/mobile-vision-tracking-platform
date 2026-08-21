"""根据激光外参和靶标位姿预测激光点像素坐标。"""

import numpy as np


def predict_laser_pixel(
    laser_origin_camera,
    laser_direction_camera,
    rotation_target_to_camera,
    translation_target_to_camera,
    config,
) -> tuple[float, float]:
    """返回激光射线与靶面交点的 ``(u, v)`` 像素坐标。"""

    origin_camera = np.asarray(laser_origin_camera, dtype=np.float64).reshape(3)
    direction_camera = np.asarray(
        laser_direction_camera, dtype=np.float64
    ).reshape(3)
    rotation = np.asarray(rotation_target_to_camera, dtype=np.float64)
    translation = np.asarray(
        translation_target_to_camera, dtype=np.float64
    ).reshape(3)

    if rotation.shape != (3, 3):
        raise ValueError("R_CT 必须是形状为 (3, 3) 的数组")

    # 将激光射线转换到靶标坐标系，靶面方程为 z_T = 0。
    origin_target = rotation.T @ (origin_camera - translation)
    direction_target = rotation.T @ direction_camera

    if abs(direction_target[2]) < 1e-9:
        raise RuntimeError("激光射线与靶面平行，无法计算交点")

    distance = -origin_target[2] / direction_target[2]
    if distance <= 0.0:
        raise RuntimeError("靶面交点不在激光传播方向上")

    hit_camera = origin_camera + distance * direction_camera
    if hit_camera[2] <= 0.0:
        raise RuntimeError("激光交点位于相机后方")

    camera_config = config["camera"]
    fx = float(camera_config["_fx"])
    fy = float(camera_config["_fy"])
    cx = float(camera_config["_cx"])
    cy = float(camera_config["_cy"])

    u = fx * hit_camera[0] / hit_camera[2] + cx
    v = fy * hit_camera[1] / hit_camera[2] + cy
    return float(u), float(v)
