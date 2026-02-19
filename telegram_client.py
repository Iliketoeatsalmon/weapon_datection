import cv2
import threading
from datetime import datetime
import requests
from config import TelegramConfig
from detector import Detection


class TelegramClient:
    def __init__(self, cfg: TelegramConfig):
        self.cfg = cfg
        self.base_url = f"https://api.telegram.org/bot{self.cfg.bot_token}"

    def _send_photo_sync(self, frame_bgr, caption: str):
        ok, img_encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            print("[Telegram] Failed to encode image")
            return

        files = {"photo": ("alert.jpg", img_encoded.tobytes(), "image/jpeg")}
        data = {"chat_id": self.cfg.chat_id, "caption": caption}

        try:
            r = requests.post(f"{self.base_url}/sendPhoto", data=data, files=files, timeout=self.cfg.timeout)
            if not r.ok:
                print("[Telegram] Error:", r.text)
        except Exception as e:
            print("[Telegram] Exception:", e)

    def send_alert_async(self, frame_bgr, det: Detection, crop_filename: str = ""):
        cls_name, conf, _, _, _, _ = det
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        caption = (
            f"⚠️ Weapon detected\n"
            f"Type: {cls_name}\n"
            f"Confidence: {conf:.2f}\n"
            f"Time: {ts}\n"
        )
        if crop_filename:
            caption += f"Crop: {crop_filename}\n"

        t = threading.Thread(target=self._send_photo_sync, args=(frame_bgr, caption), daemon=True)
        t.start()
