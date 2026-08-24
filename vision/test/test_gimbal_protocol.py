"""视觉到云台二进制协议的硬件无关单元测试。"""

import struct
import unittest

from vision.src.gimbal_protocol import (
    ProtocolError,
    TrackingState,
    VISION_MEASUREMENT_SIZE,
    VisionMeasurement,
    VisionMode,
    crc16_ccitt_false,
    decode_vision_measurement,
    encode_vision_measurement,
)


class GimbalProtocolTest(unittest.TestCase):
    def test_crc_standard_vector(self):
        self.assertEqual(crc16_ccitt_false(b"123456789"), 0x29B1)

    def test_tracking_measurement_round_trip(self):
        measurement = VisionMeasurement(
            mode=VisionMode.DRONE,
            tracking_state=TrackingState.TRACKING,
            packet_sequence=65535,
            frame_id=123456,
            track_id=7,
            yaw_error_rad=0.1234564,
            elevation_error_rad=-0.025,
            confidence=0.913,
            measurement_age_ms=42,
        )

        frame = encode_vision_measurement(measurement)
        decoded = decode_vision_measurement(frame)

        self.assertEqual(len(frame), VISION_MEASUREMENT_SIZE)
        self.assertEqual(frame[:2], b"\xA5\x5A")
        self.assertEqual(decoded.mode, VisionMode.DRONE)
        self.assertEqual(decoded.tracking_state, TrackingState.TRACKING)
        self.assertEqual(decoded.packet_sequence, 65535)
        self.assertEqual(decoded.frame_id, 123456)
        self.assertEqual(decoded.track_id, 7)
        self.assertAlmostEqual(decoded.yaw_error_rad, 0.123456)
        self.assertAlmostEqual(decoded.elevation_error_rad, -0.025)
        self.assertEqual(decoded.confidence, 0.913)
        self.assertEqual(decoded.measurement_age_ms, 42)

    def test_wire_layout_matches_document(self):
        measurement = VisionMeasurement(
            mode=VisionMode.TARGET,
            tracking_state=TrackingState.TRACKING,
            packet_sequence=0x1234,
            frame_id=0x01020304,
            track_id=0x05060708,
            yaw_error_rad=0.1,
            elevation_error_rad=-0.05,
            confidence=0.75,
            measurement_age_ms=25,
        )

        fields = struct.unpack("<2sBBBBHIIiiHHBBH", encode_vision_measurement(measurement))

        self.assertEqual(fields[0], b"\xA5\x5A")
        self.assertEqual(fields[1:5], (1, 0x20, 32, 0x01))
        self.assertEqual(fields[5:8], (0x1234, 0x01020304, 0x05060708))
        self.assertEqual(fields[8:10], (100000, -50000))
        self.assertEqual(fields[10:14], (25, 750, 1, 1))

    def test_lost_measurement_encodes_zero_placeholders(self):
        measurement = VisionMeasurement(
            mode=VisionMode.DRONE,
            tracking_state=TrackingState.TEMP_LOST,
            packet_sequence=9,
            frame_id=100,
            track_id=3,
            measurement_age_ms=20,
        )

        frame = encode_vision_measurement(measurement)
        fields = struct.unpack("<2sBBBBHIIiiHHBBH", frame)
        decoded = decode_vision_measurement(frame)

        self.assertEqual(fields[4], 0)
        self.assertEqual(fields[8:10], (0, 0))
        self.assertEqual(fields[11], 0)
        self.assertFalse(decoded.angles_valid)
        self.assertIsNone(decoded.yaw_error_rad)
        self.assertIsNone(decoded.elevation_error_rad)
        self.assertEqual(decoded.track_id, 3)

    def test_measurement_age_saturates_to_uint16(self):
        measurement = VisionMeasurement(
            mode=VisionMode.TARGET,
            tracking_state=TrackingState.SEARCHING,
            packet_sequence=1,
            frame_id=2,
            measurement_age_ms=100000,
        )

        decoded = decode_vision_measurement(encode_vision_measurement(measurement))

        self.assertEqual(decoded.measurement_age_ms, 65535)

    def test_crc_error_is_rejected(self):
        measurement = VisionMeasurement(
            mode=VisionMode.TARGET,
            tracking_state=TrackingState.SEARCHING,
            packet_sequence=1,
            frame_id=2,
        )
        damaged = bytearray(encode_vision_measurement(measurement))
        damaged[8] ^= 0x01

        with self.assertRaisesRegex(ProtocolError, "CRC"):
            decode_vision_measurement(damaged)

    def test_tracking_requires_angles(self):
        with self.assertRaisesRegex(ValueError, "必须提供两个角误差"):
            VisionMeasurement(
                mode=VisionMode.DRONE,
                tracking_state=TrackingState.TRACKING,
                packet_sequence=0,
                frame_id=1,
            )

    def test_non_tracking_rejects_stale_angles(self):
        with self.assertRaisesRegex(ValueError, "不得携带角误差"):
            VisionMeasurement(
                mode=VisionMode.DRONE,
                tracking_state=TrackingState.TEMP_LOST,
                packet_sequence=0,
                frame_id=1,
                yaw_error_rad=0.1,
                elevation_error_rad=0.2,
            )

    def test_idle_mode_cannot_track(self):
        with self.assertRaisesRegex(ValueError, "IDLE 模式不能"):
            VisionMeasurement(
                mode=VisionMode.IDLE,
                tracking_state=TrackingState.TRACKING,
                packet_sequence=0,
                frame_id=1,
                yaw_error_rad=0.0,
                elevation_error_rad=0.0,
            )


if __name__ == "__main__":
    unittest.main()
