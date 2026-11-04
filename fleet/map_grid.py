"""Grid labels shared with team mapping commit 28d262a."""
import cv2

def cell_name(c, r):
    return f"{chr(65+c)}{r+1}"

def jpg(frame):
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
