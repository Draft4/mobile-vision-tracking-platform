"""激光外参离线标定核心几何与鲁棒拟合测试。"""

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vision.src.calibrate_laser_extrinsics import (
    CLICK_COUNT,
    ImageRecord,
    aggregate_clicks,
    build_output_payload,
    camera_parameters,
    canonicalize_line,
    display_to_image_point,
    evaluate_quality,
    fit_laser_line,
    leave_one_out_errors,
    list_images,
    reconstruct_hit_point,
    write_json_atomically,
)
from vision.src.capture_camera_images import build_image_path, validate_prefix
from vision.src.laser_pixel_predictor import predict_laser_pixel


def make_config():
    return {
        "camera": {
            "_fx": 900.0,
            "_fy": 880.0,
            "_cx": 640.0,
            "_cy": 512.0,
            "calibrated_image_width": 1280,
            "calibrated_image_height": 1024,
            "distortion_coefficients": [0.05, -0.02, 0.001, -0.001, 0.005],
            "target_W": 100.0,
            "target_L": 100.0,
        }
    }


def make_synthetic_records(config):
    rng = np.random.default_rng(7)
    source_point = np.array([0.02, 0.03, 0.0], dtype=np.float64)
    true_origin, true_direction = canonicalize_line(
        source_point, np.array([0.008, -0.004, 1.0])
    )
    distances = np.concatenate(
        [
            np.linspace(1.1, 2.1, 5),
            np.linspace(2.4, 3.4, 5),
            np.linspace(3.8, 4.8, 5),
        ]
    )
    records = []

    for index, distance in enumerate(distances):
        rotation = np.eye(3)
        translation = np.array([0.0, 0.0, distance])
        pixel = np.asarray(
            predict_laser_pixel(
                true_origin,
                true_direction,
                rotation,
                translation,
                config,
            )
        )
        pixel += rng.normal(0.0, 0.15, size=2)
        if index == 7:
            pixel += np.array([24.0, -18.0])

        hit = reconstruct_hit_point(pixel, rotation, translation, config)
        records.append(
            ImageRecord(
                path=Path(f"sample_{index:02d}.png"),
                status="accepted",
                rotation=rotation,
                translation=translation,
                laser_pixel=(float(pixel[0]), float(pixel[1])),
                hit_camera_m=hit,
            )
        )

    return records, true_origin, true_direction


class LaserExtrinsicCalibrationTest(unittest.TestCase):
    def test_click_median_is_robust_to_one_bad_click(self):
        clicks = [
            (100.0, 200.0),
            (100.2, 199.8),
            (99.9, 200.1),
            (100.1, 200.0),
            (110.0, 190.0),
        ]

        median, spread = aggregate_clicks(clicks)

        np.testing.assert_allclose(median, [100.1, 200.0], atol=1e-12)
        self.assertGreater(spread, 2.0)

    def test_click_count_is_required(self):
        with self.assertRaisesRegex(ValueError, str(CLICK_COUNT)):
            aggregate_clicks([(1.0, 2.0)] * (CLICK_COUNT - 1))

    def test_display_points_map_to_original_pixels(self):
        self.assertEqual(display_to_image_point(320, 240, 0.5), (640.0, 480.0))
        self.assertEqual(
            display_to_image_point(400, 240, 8.0, 100, 50),
            (150.0, 80.0),
        )

    def test_reconstruct_hit_point_handles_distortion(self):
        config = make_config()
        rotation = np.eye(3)
        translation = np.array([0.0, 0.0, 2.0])
        expected_hit = np.array([0.1, -0.05, 2.0])
        direction = expected_hit / expected_hit[2]
        pixel = predict_laser_pixel(
            [0.0, 0.0, 0.0], direction, rotation, translation, config
        )

        actual_hit = reconstruct_hit_point(pixel, rotation, translation, config)

        np.testing.assert_allclose(actual_hit, expected_hit, atol=1e-10)

    def test_reconstruct_hit_point_rejects_outside_target(self):
        config = make_config()
        rotation = np.eye(3)
        translation = np.array([0.0, 0.0, 2.0])
        pixel = predict_laser_pixel(
            [0.0, 0.0, 0.0],
            [0.35, 0.0, 1.0],
            rotation,
            translation,
            config,
        )

        with self.assertRaisesRegex(RuntimeError, "靶标范围"):
            reconstruct_hit_point(pixel, rotation, translation, config)

    def test_robust_fit_rejects_pixel_outlier(self):
        config = make_config()
        records, expected_origin, expected_direction = make_synthetic_records(config)

        fit = fit_laser_line(records, config)
        loo_errors = leave_one_out_errors(records, fit.inlier_mask, config)
        quality = evaluate_quality(records, fit, loo_errors)

        self.assertFalse(bool(fit.inlier_mask[7]))
        self.assertGreaterEqual(np.count_nonzero(fit.inlier_mask), 14)
        np.testing.assert_allclose(
            fit.direction_camera, expected_direction, atol=5e-4
        )
        np.testing.assert_allclose(fit.origin_camera_m, expected_origin, atol=3e-3)
        self.assertTrue(quality["passed"], quality["warnings"])

    def test_short_dataset_does_not_pass_quality_gate(self):
        config = make_config()
        records, _, _ = make_synthetic_records(config)
        records = records[:5]

        fit = fit_laser_line(records, config)
        loo_errors = leave_one_out_errors(records, fit.inlier_mask, config)
        quality = evaluate_quality(records, fit, loo_errors)

        self.assertFalse(quality["passed"])
        self.assertTrue(any("有效样本" in item for item in quality["warnings"]))

    def test_invalid_distortion_length_is_rejected(self):
        config = make_config()
        config["camera"]["distortion_coefficients"] = [0.0] * 4

        with self.assertRaisesRegex(ValueError, "五个"):
            camera_parameters(config)

    def test_json_is_written_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "laser.json"
            write_json_atomically(output_path, {"schema_version": 1, "ok": True})
            self.assertIn('"ok": true', output_path.read_text(encoding="utf-8"))

    def test_image_listing_is_sorted_non_recursive_and_png_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.png").touch()
            (root / "a.PNG").touch()
            (root / "ignored.jpg").touch()
            nested = root / "nested"
            nested.mkdir()
            (nested / "c.png").touch()

            self.assertEqual(
                [path.name for path in list_images(root)],
                ["a.PNG", "b.png"],
            )

    def test_full_result_payload_is_strict_json(self):
        config = make_config()
        records, _, _ = make_synthetic_records(config)
        fit = fit_laser_line(records, config)
        loo_errors = leave_one_out_errors(records, fit.inlier_mask, config)
        quality = evaluate_quality(records, fit, loo_errors)
        for index, record in enumerate(records):
            record.is_inlier = bool(fit.inlier_mask[index])
            record.fit_error_px = float(fit.residuals_px[index])
            record.loo_error_px = (
                float(loo_errors[index]) if np.isfinite(loo_errors[index]) else None
            )

        payload = build_output_payload(
            config, Path("."), records, fit=fit, quality=quality
        )

        serialized = json.dumps(payload, allow_nan=False)
        self.assertIn('"status": "success"', serialized)

    def test_capture_prefix_is_validated_and_used(self):
        self.assertEqual(validate_prefix("laser_calibration"), "laser_calibration")
        with self.assertRaises(argparse.ArgumentTypeError):
            validate_prefix("../laser")

        path = build_image_path(Path("images"), "png", 12, "laser")
        self.assertTrue(path.name.startswith("laser_"))
        self.assertTrue(path.name.endswith("frame_00000012.png"))


if __name__ == "__main__":
    unittest.main()
