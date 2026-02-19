import cv2
from datetime import datetime

from config import load_config
from video_stream import VideoStream
from detector import WeaponDetector
from alert import AlertManager
from telegram_client import TelegramClient
from logger import EventLogger
from hud import HUDOverlay


def main():
    cfg = load_config("config.yaml")

    stream = VideoStream(cfg.video).start()
    detector = WeaponDetector(cfg.model)
    alert_manager = AlertManager(cfg.alert)
    telegram_client = TelegramClient(cfg.telegram)
    logger = EventLogger(cfg.logging)
    hud = HUDOverlay(cfg.hud)

    print("[Main] System started. Press ESC to exit.")

    try:
        while True:
            frame = stream.read()
            if frame is None:
                continue

            # HUD: FPS
            hud.update_fps()

            # Detection
            detections = detector.detect(frame)
            detector.draw_boxes(frame, detections)
            hud.update_class_stats(detections)

            # Alert logic
            if alert_manager.update(detections) and detections:
                # เลือก detection ที่ conf สูงสุด
                best_det = max(detections, key=lambda d: d[1])
                cls_name, conf, *_ = best_det

                crop_filename = logger.log_event(best_det, frame)

                # HUD last alert info
                alert_text = f"{cls_name} {conf:.2f} @ {datetime.now().strftime('%H:%M:%S')}"
                alert_manager.set_last_alert_info(alert_text)

                print(f"[Main] ALERT: {alert_text} (crop: {crop_filename})")

                telegram_client.send_alert_async(frame, best_det, crop_filename)

            # HUD overlay
            hud.draw(frame, alert_manager.get_last_alert_info())

            if cfg.hud.show_window:
                cv2.imshow(cfg.hud.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break

    finally:
        stream.stop()
        cv2.destroyAllWindows()
        print("[Main] Stopped.")


if __name__ == "__main__":
    main()
