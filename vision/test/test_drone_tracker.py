"""无人机单目标关联、丢失语义和角误差解算单元测试。"""

import math
import unittest

import numpy as np

from vision.src.data_types import DroneDetectionCandidate, FramePacket
from vision.src.drone_tracker import DroneTracker


class _UnusedDetector:
    """测试直接调用 ``update()``，该后端不应被执行。"""

    def detect(self, frame):
        raise AssertionError("本测试不应运行推理后端")


def make_config():
    """构造便于手算的测试内参与跟踪参数。"""

    return {
        "camera": {
            "_fx": 1000.0,
            "_fy": 800.0,
            "_cx": 500.0,
            "_cy": 400.0,
            "calibrated_image_width": 1000,
            "calibrated_image_height": 800,
        },
        "drone_tracking": {
            "acquire_confidence": 0.50,
            "maintain_confidence": 0.25,
            "reacquire_timeout_seconds": 0.5,
            "association_distance_ratio": 0.10,
        },
    }


def make_packet(frame_id, timestamp, width=1000, height=800):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    return FramePacket(frame=frame, frame_id=frame_id, timestamp=timestamp)


def make_candidate(center_u, center_v, confidence=0.9, half_size=10.0):
    return DroneDetectionCandidate(
        bbox_xyxy=(
            center_u - half_size,
            center_v - half_size,
            center_u + half_size,
            center_v + half_size,
        ),
        confidence=confidence,
        class_id=0,
        class_name="drone",
    )


class DroneTrackerTest(unittest.TestCase):
    def setUp(self):
        self.tracker = DroneTracker(make_config(), _UnusedDetector())

    def test_principal_point_has_zero_error(self):
        result = self.tracker.update(
            make_packet(1, 1.0),
            [make_candidate(500.0, 400.0)],
        )

        self.assertTrue(result.found)
        self.assertEqual(result.track_id, 1)
        self.assertEqual(result.pixel_error_u, 0.0)
        self.assertEqual(result.pixel_error_v, 0.0)
        self.assertEqual(result.yaw_error_rad, 0.0)
        self.assertEqual(result.elevation_error_rad, 0.0)

    def test_right_and_down_target_produce_negative_angles(self):
        result = self.tracker.update(
            make_packet(1, 1.0),
            [make_candidate(600.0, 480.0)],
        )

        self.assertAlmostEqual(result.pixel_error_u, 100.0)
        self.assertAlmostEqual(result.pixel_error_v, 80.0)
        self.assertAlmostEqual(result.yaw_error_rad, math.atan2(-100.0, 1000.0))
        self.assertAlmostEqual(
            result.elevation_error_rad,
            math.atan2(-80.0, 800.0),
        )

    def test_left_and_up_target_produce_positive_angles(self):
        result = self.tracker.update(
            make_packet(1, 1.0),
            [make_candidate(400.0, 320.0)],
        )

        self.assertGreater(result.yaw_error_rad, 0.0)
        self.assertGreater(result.elevation_error_rad, 0.0)

    def test_acquisition_uses_threshold_and_nearest_principal_point(self):
        candidates = [
            make_candidate(501.0, 400.0, confidence=0.49),
            make_candidate(700.0, 400.0, confidence=0.99),
            make_candidate(520.0, 400.0, confidence=0.60),
        ]

        result = self.tracker.update(make_packet(1, 1.0), candidates)

        self.assertTrue(result.found)
        self.assertEqual(result.center, (520.0, 400.0))

    def test_locked_target_uses_lower_threshold_and_temporal_proximity(self):
        self.tracker.update(
            make_packet(1, 1.0),
            [make_candidate(500.0, 400.0, confidence=0.9)],
        )
        result = self.tracker.update(
            make_packet(2, 1.1),
            [
                make_candidate(510.0, 400.0, confidence=0.25),
                make_candidate(580.0, 400.0, confidence=0.99),
            ],
        )

        self.assertTrue(result.found)
        self.assertEqual(result.track_id, 1)
        self.assertEqual(result.center, (510.0, 400.0))

    def test_first_miss_returns_no_stale_measurement_and_keeps_lock(self):
        self.tracker.update(
            make_packet(1, 1.0),
            [make_candidate(500.0, 400.0)],
        )

        missed = self.tracker.update(make_packet(2, 1.1), [])
        recovered = self.tracker.update(
            make_packet(3, 1.2),
            [make_candidate(510.0, 400.0, confidence=0.3)],
        )

        self.assertFalse(missed.found)
        self.assertEqual(missed.track_id, 1)
        self.assertIsNone(missed.bbox_xyxy)
        self.assertIsNone(missed.center)
        self.assertIsNone(missed.pixel_error_u)
        self.assertIsNone(missed.pixel_error_v)
        self.assertIsNone(missed.yaw_error_rad)
        self.assertIsNone(missed.elevation_error_rad)
        self.assertTrue(recovered.found)
        self.assertEqual(recovered.track_id, 1)

    def test_far_candidate_does_not_replace_lock_before_timeout(self):
        self.tracker.update(
            make_packet(1, 1.0),
            [make_candidate(500.0, 400.0)],
        )

        result = self.tracker.update(
            make_packet(2, 1.2),
            [make_candidate(800.0, 600.0, confidence=0.99)],
        )

        self.assertFalse(result.found)
        self.assertEqual(result.track_id, 1)
        self.assertIsNone(result.yaw_error_rad)

    def test_timeout_clears_lock_and_reacquires_with_new_id(self):
        self.tracker.update(
            make_packet(1, 1.0),
            [make_candidate(500.0, 400.0)],
        )

        result = self.tracker.update(
            make_packet(2, 1.6),
            [make_candidate(800.0, 600.0, confidence=0.9)],
        )

        self.assertTrue(result.found)
        self.assertEqual(result.track_id, 2)
        self.assertEqual(result.center, (800.0, 600.0))

    def test_resolution_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "与内参标定尺寸"):
            self.tracker.update(
                make_packet(1, 1.0, width=1280, height=1024),
                [],
            )

    def test_timestamp_cannot_go_backwards_while_locked(self):
        self.tracker.update(
            make_packet(1, 2.0),
            [make_candidate(500.0, 400.0)],
        )

        with self.assertRaisesRegex(ValueError, "timestamp 不得倒退"):
            self.tracker.update(make_packet(2, 1.9), [])


if __name__ == "__main__":
    unittest.main()
