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
        raw_names = self.model.names
        # Ultralytics may return class names as dict or list depending on version/model.
        self.class_names = raw_names if isinstance(raw_names, dict) else {i: n for i, n in enumerate(raw_names)}
        self.model_names_norm = {str(name).strip().lower() for name in self.class_names.values()}
        self.name_aliases = {
            "handgun": "pistol",
            "gun": "pistol",
            "bomb": "grenade",
        }
        self.allowed_weapon_names = set()
        for name in self.cfg.weapon_names:
            norm_name = str(name).strip().lower()
            mapped_name = self.name_aliases.get(norm_name, norm_name)
            self.allowed_weapon_names.add(mapped_name)

        print("[Detector] Model classes:", self.class_names)
        unknown = sorted(self.allowed_weapon_names - self.model_names_norm)
        if unknown:
            print(
                "[Detector][WARN] weapon_names not found in model classes:",
                unknown,
            )
            print("[Detector][WARN] Use names from model classes above.")

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

            if cls_name.strip().lower() not in self.allowed_weapon_names:
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
