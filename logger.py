import os
import csv
from datetime import datetime
import cv2
from config import LoggingConfig
from detector import Detection


class EventLogger:
    def __init__(self, cfg: LoggingConfig):
        self.cfg = cfg
        log_dir = os.path.dirname(self.cfg.log_csv) or "."
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.cfg.crop_dir, exist_ok=True)

        if not os.path.exists(self.cfg.log_csv):
            with open(self.cfg.log_csv, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "weapon_type",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "crop_filename",
                ])

    def log_event(self, det: Detection, frame_bgr) -> str:
        cls_name, conf, x1, y1, x2, y2 = det
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        crop_filename = ""

        if self.cfg.enable_crop_save:
            crop = frame_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size > 0:
                crop_filename = f"{ts}_{cls_name}_{conf:.2f}.jpg"
                crop_path = os.path.join(self.cfg.crop_dir, crop_filename)
                cv2.imwrite(crop_path, crop)

        with open(self.cfg.log_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ts.replace("_", " "),
                cls_name,
                f"{conf:.4f}",
                x1,
                y1,
                x2,
                y2,
                crop_filename,
            ])

        return crop_filename
