"""Importable perception stubs. No detection or real control yet."""
import cv2

from .contracts import Detection


def detect(frame) -> list[Detection]:
    return []


def annotate(frame, detections: list[Detection]) -> bytes:
    image = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(detection["track_id"]), (x1, max(y1, 20)), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("JPEG encoding failed")
    return encoded.tobytes()


def follow_command(target: Detection, frame_shape) -> tuple[int, int, int, int]:
    return (0, 0, 0, 0)


def photo_ready(target: Detection, frame_shape) -> bool:
    return False
