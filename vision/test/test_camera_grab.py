import cv2
from vision.src.hik_camera import HikCamera
from vision.src.latest_frame_buffer import LatestFrameBuffer
from pathlib import Path
import yaml


def main():
    vision_dir = Path(__file__).resolve().parents[1]
    config_path = vision_dir / "configs" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    camera = HikCamera(config)
    camera.open()
    camera.start()

    frame_buffer = LatestFrameBuffer()

    last_frame_id = -1
    while True:
        frame_packet = camera.grab()
        frame_buffer.publish(frame_packet)

        latest_packet = frame_buffer.wait_for_new(last_frame_id)

        if latest_packet is None:
            break

        cv2.imshow("Frame", latest_packet.frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            camera.stop()
            camera.close()
            break


if __name__ == "__main__":
    main()
