"""线程安全的单槽最新帧缓冲区。

该缓冲区用于解耦相机采集线程和图像处理线程。生产者发布新帧时直接覆盖旧帧，
消费者始终取得当前最新结果，因此处理速度暂时低于采集速度时不会累积延迟，
代价是中间帧可能被跳过。它不是保存全部帧的队列，不适合录像或逐帧离线处理。

关闭操作不可逆：``close()`` 会唤醒所有等待者，之后发布和等待均不再返回帧。
"""

import threading
from typing import Optional
from .data_types import FramePacket


class LatestFrameBuffer:
    """在线程之间传递最新 :class:`FramePacket` 的单槽缓冲区。"""

    def __init__(self):
        # 数据只在 Condition 持锁期间读写；不会在锁外单独访问。
        self._packet: Optional[FramePacket] = None

        # Condition 同时提供互斥访问和“新帧/关闭”状态通知。
        self._condition = threading.Condition()
        self._closed = False

    def publish(self, packet: FramePacket) -> None:
        """发布新帧并唤醒等待者；已有帧会被直接覆盖。"""

        with self._condition:
            if self._closed:
                return

            self._packet = packet
            self._condition.notify_all()

    def wait_for_new(
        self, last_frame_id: int = -1, timeout: Optional[float] = None
    ) -> Optional[FramePacket]:
        """等待帧编号大于 ``last_frame_id`` 的新帧。

        超时或缓冲区关闭时返回 ``None``。调用方处理成功后应保存返回帧的
        ``frame_id``，并在下一次等待时传回，以避免重复处理同一帧。
        """
        with self._condition:
            # wait_for 会正确处理虚假唤醒，并在每次唤醒后重新检查谓词。
            success = self._condition.wait_for(
                lambda: self._closed
                or (self._packet is not None and self._packet.frame_id > last_frame_id),
                timeout=timeout,
            )

            if not success:
                return None

            if self._closed:
                return None

            return self._packet

    def close(self):
        """永久关闭缓冲区，并唤醒所有正在等待的线程。"""

        with self._condition:
            self._closed = True
            self._condition.notify_all()
