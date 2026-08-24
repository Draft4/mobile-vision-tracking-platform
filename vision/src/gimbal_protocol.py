"""视觉到云台的硬件无关二进制协议编解码。

本模块实现 ``interface/communication-protocol.md`` 中的 v0.1 单向协议，负责把
视觉任务模式、检测状态和相机视轴角误差编码成固定 32 字节数据帧，并提供对称的
解码函数用于单元测试和联调工具。它不打开串口，也不依赖 RK3588 的设备节点、
GPIO 或其他板级信息；硬件发送逻辑应放在后续独立的 UART 模块中。

协议使用小端序、微弧度定点角度和 CRC-16/CCITT-FALSE。无有效目标时，角度字段
在数据帧中写零，但解码结果仍为 ``None``，避免把“没有测量”误认为“误差为零”。
"""

from dataclasses import dataclass
from enum import IntEnum
import math
import struct
from typing import Optional, Type, TypeVar


SOF = b"\xA5\x5A"
PROTOCOL_VERSION = 1
VISION_MEASUREMENT_TYPE = 0x20
VISION_MEASUREMENT_SIZE = 32

_ANGLE_SCALE = 1_000_000
_CONFIDENCE_SCALE = 1_000
_UINT16_MAX = (1 << 16) - 1
_UINT32_MAX = (1 << 32) - 1
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1

_FRAME_WITHOUT_CRC = struct.Struct("<2sBBBBHIIiiHHBB")
_CRC = struct.Struct("<H")


class ProtocolError(ValueError):
    """数据不符合视觉测量协议时抛出的异常。"""


class VisionMode(IntEnum):
    """产生测量的视觉任务。"""

    IDLE = 0
    TARGET = 1
    DRONE = 2


class TrackingState(IntEnum):
    """当前图像是否产生了可用于控制的新测量。"""

    SEARCHING = 0
    TRACKING = 1
    TEMP_LOST = 2
    FAULT = 3


_EnumType = TypeVar("_EnumType", bound=IntEnum)


def _coerce_enum(value: IntEnum, enum_type: Type[_EnumType], name: str) -> _EnumType:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 不是有效的 {enum_type.__name__}: {value!r}") from exc


def _validate_integer(value: int, minimum: int, maximum: Optional[int], name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须是整数")
    if value < minimum or (maximum is not None and value > maximum):
        upper = "无限制" if maximum is None else str(maximum)
        raise ValueError(f"{name} 必须位于 [{minimum}, {upper}]，实际为 {value}")


def _validate_finite(value: float, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} 必须是有限数值") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} 必须是有限数值")
    return converted


def _rad_to_urad(value: float, name: str) -> int:
    encoded = int(round(value * _ANGLE_SCALE))
    if not _INT32_MIN <= encoded <= _INT32_MAX:
        raise ValueError(f"{name} 超出 int32 微弧度编码范围")
    return encoded


@dataclass(frozen=True)
class VisionMeasurement:
    """一帧待发送或已解码的视觉测量。

    ``TRACKING`` 状态必须同时提供两个弧度角误差；其他状态必须将它们设为
    ``None``。``measurement_age_ms`` 大于 65535 时会在编码阶段饱和，而负值
    始终视为调用错误。
    """

    mode: VisionMode
    tracking_state: TrackingState
    packet_sequence: int
    frame_id: int
    track_id: int = 0
    yaw_error_rad: Optional[float] = None
    elevation_error_rad: Optional[float] = None
    confidence: float = 0.0
    measurement_age_ms: int = 0

    def __post_init__(self) -> None:
        mode = _coerce_enum(self.mode, VisionMode, "mode")
        state = _coerce_enum(self.tracking_state, TrackingState, "tracking_state")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "tracking_state", state)

        _validate_integer(self.packet_sequence, 0, _UINT16_MAX, "packet_sequence")
        _validate_integer(self.frame_id, 0, _UINT32_MAX, "frame_id")
        _validate_integer(self.track_id, 0, _UINT32_MAX, "track_id")
        _validate_integer(self.measurement_age_ms, 0, None, "measurement_age_ms")

        confidence = _validate_finite(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence 必须位于 [0, 1]")
        object.__setattr__(self, "confidence", confidence)

        if state == TrackingState.TRACKING:
            if mode == VisionMode.IDLE:
                raise ValueError("IDLE 模式不能产生 TRACKING 测量")
            if self.yaw_error_rad is None or self.elevation_error_rad is None:
                raise ValueError("TRACKING 状态必须提供两个角误差")

            yaw = _validate_finite(self.yaw_error_rad, "yaw_error_rad")
            elevation = _validate_finite(
                self.elevation_error_rad,
                "elevation_error_rad",
            )
            _rad_to_urad(yaw, "yaw_error_rad")
            _rad_to_urad(elevation, "elevation_error_rad")
            object.__setattr__(self, "yaw_error_rad", yaw)
            object.__setattr__(self, "elevation_error_rad", elevation)
        else:
            if self.yaw_error_rad is not None or self.elevation_error_rad is not None:
                raise ValueError("非 TRACKING 状态不得携带角误差")
            if confidence != 0.0:
                raise ValueError("非 TRACKING 状态的 confidence 必须为 0")

    @property
    def angles_valid(self) -> bool:
        """当前对象是否包含可用于控制的角误差。"""

        return self.tracking_state == TrackingState.TRACKING


def crc16_ccitt_false(data: bytes) -> int:
    """计算 CRC-16/CCITT-FALSE，初值为 ``0xFFFF``。"""

    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def encode_vision_measurement(measurement: VisionMeasurement) -> bytes:
    """将视觉测量编码为协议规定的 32 字节数据帧。"""

    if not isinstance(measurement, VisionMeasurement):
        raise TypeError("measurement 必须是 VisionMeasurement")

    if measurement.angles_valid:
        flags = 0x01
        yaw_error_urad = _rad_to_urad(
            measurement.yaw_error_rad,
            "yaw_error_rad",
        )
        elevation_error_urad = _rad_to_urad(
            measurement.elevation_error_rad,
            "elevation_error_rad",
        )
        confidence_permille = int(round(measurement.confidence * _CONFIDENCE_SCALE))
    else:
        flags = 0
        yaw_error_urad = 0
        elevation_error_urad = 0
        confidence_permille = 0

    measurement_age_ms = min(measurement.measurement_age_ms, _UINT16_MAX)
    body = _FRAME_WITHOUT_CRC.pack(
        SOF,
        PROTOCOL_VERSION,
        VISION_MEASUREMENT_TYPE,
        VISION_MEASUREMENT_SIZE,
        flags,
        measurement.packet_sequence,
        measurement.frame_id,
        measurement.track_id,
        yaw_error_urad,
        elevation_error_urad,
        measurement_age_ms,
        confidence_permille,
        int(measurement.tracking_state),
        int(measurement.mode),
    )
    return body + _CRC.pack(crc16_ccitt_false(body))


def decode_vision_measurement(frame: bytes) -> VisionMeasurement:
    """校验并解码一个完整的 32 字节视觉测量帧。"""

    try:
        raw = bytes(frame)
    except (TypeError, ValueError) as exc:
        raise TypeError("frame 必须是 bytes-like 对象") from exc

    if len(raw) != VISION_MEASUREMENT_SIZE:
        raise ProtocolError(
            f"视觉测量帧必须为 {VISION_MEASUREMENT_SIZE} 字节，实际为 {len(raw)}"
        )

    body = raw[:-_CRC.size]
    received_crc = _CRC.unpack(raw[-_CRC.size :])[0]
    expected_crc = crc16_ccitt_false(body)
    if received_crc != expected_crc:
        raise ProtocolError(
            f"CRC 校验失败：收到 0x{received_crc:04X}，期望 0x{expected_crc:04X}"
        )

    (
        sof,
        version,
        message_type,
        frame_length,
        flags,
        packet_sequence,
        frame_id,
        track_id,
        yaw_error_urad,
        elevation_error_urad,
        measurement_age_ms,
        confidence_permille,
        tracking_state_value,
        mode_value,
    ) = _FRAME_WITHOUT_CRC.unpack(body)

    if sof != SOF:
        raise ProtocolError("帧头错误")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"不支持的协议版本: {version}")
    if message_type != VISION_MEASUREMENT_TYPE:
        raise ProtocolError(f"不支持的消息类型: 0x{message_type:02X}")
    if frame_length != VISION_MEASUREMENT_SIZE:
        raise ProtocolError(f"帧内长度字段错误: {frame_length}")

    try:
        tracking_state = TrackingState(tracking_state_value)
        mode = VisionMode(mode_value)
    except ValueError as exc:
        raise ProtocolError("帧内模式或跟踪状态无效") from exc

    angles_valid = bool(flags & 0x01)
    if angles_valid != (tracking_state == TrackingState.TRACKING):
        raise ProtocolError("angles_valid 与 tracking_state 不一致")

    if angles_valid:
        yaw_error_rad = yaw_error_urad / _ANGLE_SCALE
        elevation_error_rad = elevation_error_urad / _ANGLE_SCALE
        if confidence_permille > _CONFIDENCE_SCALE:
            raise ProtocolError("置信度编码超出 0～1000")
        confidence = confidence_permille / _CONFIDENCE_SCALE
    else:
        if yaw_error_urad != 0 or elevation_error_urad != 0:
            raise ProtocolError("无效测量的角度占位字段必须为 0")
        if confidence_permille != 0:
            raise ProtocolError("无效测量的置信度占位字段必须为 0")
        yaw_error_rad = None
        elevation_error_rad = None
        confidence = 0.0

    try:
        return VisionMeasurement(
            mode=mode,
            tracking_state=tracking_state,
            packet_sequence=packet_sequence,
            frame_id=frame_id,
            track_id=track_id,
            yaw_error_rad=yaw_error_rad,
            elevation_error_rad=elevation_error_rad,
            confidence=confidence,
            measurement_age_ms=measurement_age_ms,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"视觉测量字段无效: {exc}") from exc


__all__ = [
    "PROTOCOL_VERSION",
    "ProtocolError",
    "SOF",
    "TrackingState",
    "VISION_MEASUREMENT_SIZE",
    "VISION_MEASUREMENT_TYPE",
    "VisionMeasurement",
    "VisionMode",
    "crc16_ccitt_false",
    "decode_vision_measurement",
    "encode_vision_measurement",
]
