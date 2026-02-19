import time
from collections import Counter
import cv2
from config import HUDConfig
from detector import Detection


class HUDOverlay:
    def __init__(self, cfg: HUDConfig):
        self.cfg = cfg
        self.fps = 0.0
        self.frame_count = 0
        self.last_time = time.time()
        self.class_counter = Counter()

    def update_fps(self):
        self.frame_count += 1
        now = time.time()
        if now - self.last_time >= 1.0:
            self.fps = self.frame_count / (now - self.last_time)
            self.frame_count = 0
            self.last_time = now

    def update_class_stats(self, detections: list[Detection]):
        self.class_counter.clear()
        for cls_name, *_ in detections:
            self.class_counter[cls_name] += 1

    def draw(self, frame_bgr, last_alert_info: str):
        h, w, _ = frame_bgr.shape
        y = 20

        # FPS
        cv2.putText(
            frame_bgr,
            f"FPS: {self.fps:.1f}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.cfg.font_scale,
            (0, 255, 255),
            self.cfg.font_thickness,
        )
        y += 25

        # Class stats
        if self.class_counter:
            cls_text = " | ".join([f"{k}: {v}" for k, v in self.class_counter.items()])
        else:
            cls_text = "No weapon detected"
        cv2.putText(
            frame_bgr,
            cls_text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.cfg.font_scale,
            (255, 255, 0),
            self.cfg.font_thickness,
        )
        y += 25

        # Last alert
        if last_alert_info:
            cv2.putText(
                frame_bgr,
                f"Last alert: {last_alert_info}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.cfg.font_scale,
                (0, 200, 255),
                self.cfg.font_thickness,
            )
