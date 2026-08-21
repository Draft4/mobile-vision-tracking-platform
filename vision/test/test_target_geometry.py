"""靶标位姿和激光落点几何工具的单元测试。"""

import unittest

import cv2
import numpy as np

from vision.src.laser_pixel_predictor import predict_laser_pixel
from vision.src.target_pose_estimator import estimate_target_pose


def make_config():
    return {
        "camera": {
            "_fx": 1000.0,
            "_fy": 800.0,
            "_cx": 500.0,
            "_cy": 400.0,
            "target_W": 20.0,
            "target_L": 10.0,
        }
    }


class TargetGeometryTest(unittest.TestCase):
    def test_pose_estimator_recovers_synthetic_front_view(self):
        config = make_config()
        object_points = np.array(
            [
                [-0.1, -0.05, 0.0],
                [0.1, -0.05, 0.0],
                [0.1, 0.05, 0.0],
                [-0.1, 0.05, 0.0],
            ],
            dtype=np.float64,
        )
        camera_matrix = np.array(
            [
                [1000.0, 0.0, 500.0],
                [0.0, 800.0, 400.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        image_points, _ = cv2.projectPoints(
            object_points,
            np.zeros(3),
            np.array([0.0, 0.0, 2.0]),
            camera_matrix,
            np.zeros(5),
        )

        rotation, translation = estimate_target_pose(
            image_points.reshape(4, 2),
            config,
        )

        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(translation, [0.0, 0.0, 2.0], atol=1e-6)

    def test_pose_estimator_rejects_invalid_corner_shape(self):
        with self.assertRaisesRegex(ValueError, "corners"):
            estimate_target_pose(np.zeros((3, 2)), make_config())

    def test_laser_ray_is_projected_to_target_plane(self):
        pixel = predict_laser_pixel(
            laser_origin_camera=[0.0, 0.0, 0.0],
            laser_direction_camera=[0.1, 0.0, 1.0],
            rotation_target_to_camera=np.eye(3),
            translation_target_to_camera=[0.0, 0.0, 2.0],
            config=make_config(),
        )

        np.testing.assert_allclose(pixel, [600.0, 400.0], atol=1e-9)

    def test_laser_ray_parallel_to_target_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "平行"):
            predict_laser_pixel(
                laser_origin_camera=[0.0, 0.0, 0.0],
                laser_direction_camera=[1.0, 0.0, 0.0],
                rotation_target_to_camera=np.eye(3),
                translation_target_to_camera=[0.0, 0.0, 2.0],
                config=make_config(),
            )


if __name__ == "__main__":
    unittest.main()
