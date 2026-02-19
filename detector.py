from typing import List, Tuple
import cv2
from ultralytics import YOLO
from config import ModelConfig

# (class_name, conf, x1, y1, x2, y2)
Detection = Tuple[str, float, int, int, int, int]


class WeaponDetector:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        print("[Detector] Loading model:", self.cfg.model_path)
        self.model = YOLO(self.cfg.model_path)
        self.class_names = self.model.names
        print("[Detector] Model classes:", self.class_names)

    def detect(self, frame_bgr) -> List[Detection]:
        """
        Run YOLO inference on BGR frame and return list of detections.
        """
        results = self.model(
            frame_bgr,
            imgsz=self.cfg.img_size,
            conf=self.cfg.conf_th,
            iou=self.cfg.iou_th,
            verbose=False
        )[0]

        detections: List[Detection] = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = self.class_names.get(cls_id, str(cls_id))

            if cls_name not in self.cfg.weapon_names:
                continue

            detections.append((cls_name, conf, x1, y1, x2, y2))

        return detections

    @staticmethod
    def draw_boxes(frame_bgr, detections: List[Detection]):
        for cls_name, conf, x1, y1, x2, y2 in detections:
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(
                frame_bgr,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
