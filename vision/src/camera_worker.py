"""后台相机采集任务。

``CameraWorker`` 独占相机的打开、采集和关闭生命周期，并把新帧发布到
``LatestFrameBuffer``。图像检测和界面绘制应在其他线程完成，避免阻塞相机取流。

停止采用协作式机制：主线程设置 ``_stop_event``，采集线程在一次 ``grab()``
返回后观察该状态并退出。实例停止时也会永久关闭关联的最新帧缓冲区；如需重新
启动完整流水线，应新建缓冲区和工作线程实例。
"""

import threading

from .latest_frame_buffer import LatestFrameBuffer


class CameraWorker:
    """持续采集相机帧并发布最新帧的后台线程控制器。"""

    def __init__(self, camera, frame_buffer: LatestFrameBuffer):
        self.camera = camera
        self.frame_buffer = frame_buffer

        # _thread 保存线程对象，用于防止重复启动并在停止时等待资源释放。
        self._thread = None

        # Event 是线程安全的停止信号，不直接强制终止正在执行的 grab()。
        self._stop_event = threading.Event()

    def start(self):
        """启动采集线程；同一实例不允许并发启动两次。"""

        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("CameraWorker 已经启动")

        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._run, name="CameraWorker", daemon=True
        )

        self._thread.start()

    def _run(self):
        """执行相机生命周期和采集循环，仅由后台线程调用。"""

        try:
            if not self.camera.is_open:
                self.camera.open()

            if not self.camera.is_start:
                self.camera.start()

            while not self._stop_event.is_set():
                frame_packet = self.camera.grab()

                if frame_packet is None:
                    continue

                self.frame_buffer.publish(frame_packet)

        finally:
            # 无论正常停止还是采集异常，都尝试释放设备并唤醒消费者。
            self.camera.stop()
            self.camera.close()
            self.frame_buffer.close()

    def stop(self):
        """请求线程停止，并等待相机和缓冲区完成清理。"""

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("线程没有正常结束")
            self._thread = None
