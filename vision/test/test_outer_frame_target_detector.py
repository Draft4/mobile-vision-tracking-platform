import cv2
from pathlib import Path
import yaml

from vision.src.data_types import FramePacket
from vision.src.outer_frame_target_detector import OuterFrameTargetDetector


def main():
    vision_dir = Path(__file__).resolve().parents[1]
    config_path = vision_dir / "configs" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    detector = OuterFrameTargetDetector(config)

    image_path = vision_dir / "data" / "target_test_1.jpg"
    frame = cv2.imread(str(image_path))

    if frame is None:
        raise RuntimeError("图片读取失败")

    frame_packet = FramePacket(frame=frame, frame_id=0, timestamp=0.0)
    result = detector.detect(frame_packet)

    if not result.found:
        print("没有检查到靶标")
        cv2.putText(
            frame,
            "Target not found",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
    else:
        print("检测到靶标")
        print(f"中心: {result.center}")
        print(f"角点: {result.corners}")
        print(f"面积: {result.area}")
        print(f"置信度: {result.confidence}")

        if result.corners is None or result.center is None:
            raise RuntimeError("检测成功，但角点或中心坐标为空")

        corners = result.corners.round().astype(int)
        labels = ["TL", "TR", "BR", "BL"]

        for i in range(4):
            p1 = tuple(map(int, corners[i]))
            p2 = tuple(map(int, corners[(i + 1) % 4]))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

        for point, label in zip(corners, labels):
            x, y = point
            x, y = int(x), int(y)
            cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)

            cv2.putText(
                frame,
                label,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        cx, cy = result.center
        cx = int(round(cx))
        cy = int(round(cy))

        cv2.drawMarker(
            frame,
            (cx, cy),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=25,
            thickness=2,
        )

        cv2.putText(
            frame,
            f"center=({cx},{cy})",
            (cx + 10, cy + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    height, width = frame.shape[:2]
    display_scale = min(1000 / width, 700 / height, 1.0)
    display_frame = cv2.resize(
        frame,
        None,
        fx=display_scale,
        fy=display_scale,
        interpolation=cv2.INTER_AREA,
    )

    cv2.imshow("Outer Frame Target Detection", display_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
