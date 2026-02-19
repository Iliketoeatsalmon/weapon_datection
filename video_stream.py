import cv2
import threading
from typing import Optional
from config import VideoConfig


class VideoStream:
    """
    Asynchronous video capture.
    """
    def __init__(self, cfg: VideoConfig):
        self.cfg = cfg
        self.cap = cv2.VideoCapture(self.cfg.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)

        self.grabbed, self.frame = self.cap.read()
        if not self.grabbed:
            raise RuntimeError("Failed to read initial frame from camera.")

        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if not grabbed:
                continue
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self) -> Optional[cv2.Mat]:
        with self.lock:
            if not self.grabbed:
                return None
            return self.frame.copy()

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()