"""海康机器人 USB 工业相机的同步采集封装。

本模块通过海康 MVS SDK 的 Python 绑定枚举并打开第一台 USB 相机，按配置设置
常用采集参数，再由 ``grab()`` 同步取得一帧并包装为 ``FramePacket``。标准生命
周期为 ``open() -> start() -> grab() -> stop() -> close()``，这些操作应由同一
采集线程串行调用；类本身不提供并发访问保护。

``MvImport`` 中的文件来自相机 SDK。本模块只负责把该目录加入模块搜索路径，
部署机器仍需正确安装与相机、操作系统架构匹配的 MVS 运行库。
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .data_types import FramePacket

# 海康 MVS SDK 的 Python 封装随项目放在当前目录下的 MvImport 中。
# 使用文件自身的位置构造路径，避免启动目录变化导致导入失败。
mv_import_dir = Path(__file__).resolve().parent / "MvImport"
if str(mv_import_dir) not in sys.path:
    sys.path.insert(0, str(mv_import_dir))

# SDK 绑定通过通配导入公开相机类、像素格式常量及 ctypes 辅助符号。
# 此处沿用厂商生成代码的导入方式，其余业务模块不应使用通配导入。
from MvCameraControl_class import *


class HikCamera:
    """管理一台海康 USB 相机及其同步取流生命周期。"""

    def __init__(self, config):
        self.config = config
        self._validate_config()
        self.is_open = False
        self.is_start = False

    def _validate_config(self) -> None:
        """在访问硬件前检查必需的相机配置值。"""

        camera_config = self.config["camera"]

        if camera_config["target_frame_rate"] <= 0.0:
            raise ValueError("target_frame_rate 必须大于 0")

        if camera_config["gain_auto"] not in {"Off", "Once", "Continuous"}:
            raise ValueError("gain_auto 配置错误")

        if camera_config["gain"] < 0.0:
            raise ValueError("gain 必须大于等于 0")

        if camera_config["exposure_auto"] not in {"Off", "Once", "Continuous"}:
            raise ValueError("exposure_auto 配置错误")

        if camera_config["exposure_time_us"] <= 0.0:
            raise ValueError("exposure_time_us 必须大于 0")

        if camera_config["balance_white_auto"] not in {
            "Off",
            "Once",
            "Continuous",
        }:
            raise ValueError("balance_white_auto 配置错误")

        if camera_config["fps_window_seconds"] <= 0.0:
            raise ValueError("fps_window_seconds 必须大于 0")

        if camera_config["trigger_mode"] not in {"Off", "On"}:
            raise ValueError("trigger_mode 配置错误")

        if not isinstance(camera_config["acquisition_frame_rate_enable"], bool):
            raise ValueError("acquisition_frame_rate_enable 必须为布尔值")

        if not isinstance(camera_config["gamma_enable"], bool):
            raise ValueError("gamma_enable 必须为布尔值")

        if camera_config["gamma_selector"] not in {"User", "sRGB"}:
            raise ValueError("gamma_selector 配置错误")

        if camera_config["gamma"] <= 0.0:
            raise ValueError("gamma 必须为正值")

        if camera_config["pixel_format"] not in {"BayerBG8", "BayerGB8"}:
            raise ValueError("pixel_format 只支持 BayerBG8 或 BayerGB8")

    def _check(self, ret, message):
        """把必要 SDK 调用的非零返回码转换为 Python 异常。"""

        if ret != 0:
            raise RuntimeError(f"{message} failed, ret = 0x{ret &  0xffffffff:08x}")

    def _print_feature_result(self, name, value, ret):
        """输出可选相机特性的设置结果，便于适配不同型号。"""

        status = "OK" if ret == 0 else "SKIP"
        suffix = "" if ret == 0 else f", ret=0x{ret & 0xffffffff:08x}"
        print(f"[{status}] {name} = {value}{suffix}")

    def _set_enum_if_supported(self, name, value):
        """尝试设置枚举特性；不支持时记录结果但不中止启动。"""

        ret = self.cam.MV_CC_SetEnumValueByString(name, value)
        self._print_feature_result(name, value, ret)
        return ret == 0

    def _set_bool_if_supported(self, name, value):
        """尝试设置布尔特性；不支持时记录结果但不中止启动。"""

        ret = self.cam.MV_CC_SetBoolValue(name, value)
        self._print_feature_result(name, value, ret)
        return ret == 0

    def _set_float_if_supported(self, name, value):
        """尝试设置浮点特性；不支持时记录结果但不中止启动。"""

        ret = self.cam.MV_CC_SetFloatValue(name, value)
        self._print_feature_result(name, f"{value:.2f}", ret)
        return ret == 0

    def _set_config_for_camera(self):
        """以尽力而为方式设置不同型号相机共有的采集特性。"""

        # 关闭自动增益
        self._set_enum_if_supported("GainAuto", self.config["camera"]["gain_auto"])
        if self.config["camera"]["gain_auto"] == "Off":
            self._set_float_if_supported("Gain", self.config["camera"]["gain"])

        # 关闭触发模式
        self._set_enum_if_supported(
            "TriggerMode", self.config["camera"]["trigger_mode"]
        )

        # 设置曝光时间
        self._set_enum_if_supported(
            "ExposureAuto", self.config["camera"]["exposure_auto"]
        )
        if self.config["camera"]["exposure_auto"] == "Off":
            self._set_float_if_supported(
                "ExposureTime", self.config["camera"]["exposure_time_us"]
            )

        # 设置自动白平衡
        self._set_enum_if_supported(
            "BalanceWhiteAuto", self.config["camera"]["balance_white_auto"]
        )

        # 设置获取帧数
        self._set_bool_if_supported(
            "AcquisitionFrameRateEnable",
            self.config["camera"]["acquisition_frame_rate_enable"],
        )

        # 设置 gamma
        self._set_bool_if_supported(
            "GammaEnable", self.config["camera"]["gamma_enable"]
        )

        if self.config["camera"]["gamma_enable"]:
            self._set_enum_if_supported(
                "GammaSelector", self.config["camera"]["gamma_selector"]
            )
            self._set_float_if_supported("Gamma", self.config["camera"]["gamma"])

        # 设置相机的像素格式
        self._set_enum_if_supported(
            "PixelFormat", self.config["camera"]["pixel_format"]
        )

    def open(self):
        """枚举并打开第一台 USB 相机，然后读取帧负载大小。"""

        # 获取设备列表
        device_list = MV_CC_DEVICE_INFO_LIST()

        ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, device_list)

        self._check(ret, "EnumDevices")

        print("camera count: ", device_list.nDeviceNum)

        if device_list.nDeviceNum == 0:
            raise RuntimeError("没有找到 USB 工业相机")

        device_info = cast(
            device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)
        ).contents

        # 创建句柄，打开相机
        self.cam = MvCamera()

        ret = self.cam.MV_CC_CreateHandle(device_info)
        self._check(ret, "CreateHandle")

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive)
        self._check(ret, "OpenDevice")

        # 设置相机配置参数
        self._set_config_for_camera()

        # 获取一帧图像需要多大的内存
        payload = MVCC_INTVALUE_EX()
        memset(byref(payload), 0, sizeof(payload))
        ret = self.cam.MV_CC_GetIntValueEx("PayloadSize", payload)
        self._check(ret, "Get PayloadSize")
        self.payload_size = int(payload.nCurValue)
        print("payload size:", self.payload_size)

        self.is_open = True

    def start(self):
        """分配复用缓冲区并开始取流；调用前必须成功执行 ``open()``。"""

        # 创建图像缓存区
        self.data_buf = (c_ubyte * self.payload_size)()
        self.frame_info = MV_FRAME_OUT_INFO_EX()

        # 开始采集数据
        ret = self.cam.MV_CC_StartGrabbing()
        self._check(ret, "StratGrabbing")

        self.is_start = True

    def grab(self):
        """同步等待并返回一帧独立持有数据的 :class:`FramePacket`。

        当前仅处理 Bayer BG8 和 Bayer GB8 像素格式。时间戳在 SDK 返回图像后
        使用单调时钟记录，可用于计算本进程内的处理延迟。
        """

        memset(byref(self.frame_info), 0, sizeof(self.frame_info))

        ret = self.cam.MV_CC_GetOneFrameTimeout(
            self.data_buf, self.payload_size, self.frame_info, 1000
        )

        self._check(ret, "get frame")

        height = self.frame_info.nHeight
        width = self.frame_info.nWidth

        # SDK 缓冲区会在下一次取流时复用，因此先复制有效负载再进行转换。
        raw = np.ctypeslib.as_array(self.data_buf)[: self.frame_info.nFrameLen].copy()

        if self.frame_info.enPixelType == PixelType_Gvsp_BayerBG8:
            bayer = raw.reshape(height, width)
            frame = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB)
        elif self.frame_info.enPixelType == PixelType_Gvsp_BayerGB8:
            bayer = raw.reshape(height, width)
            frame = cv2.cvtColor(bayer, cv2.COLOR_BayerGB2RGB)
        else:
            raise RuntimeError(f"不支持该像素格式: {self.frame_info.enPixelType}")

        return FramePacket(
            frame=frame.copy(),
            frame_id=int(self.frame_info.nFrameNum),
            timestamp=time.perf_counter(),
        )

    def stop(self):
        """停止取流；应仅在已成功启动后调用。"""

        self.cam.MV_CC_StopGrabbing()
        self.is_start = False

    def close(self):
        """关闭设备并销毁 SDK 句柄；应在停止取流后调用。"""

        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()
        self.is_open = False
