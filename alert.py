import time
from collections import deque
from typing import List
from config import AlertConfig
from detector import Detection


class AlertManager:
    def __init__(self, cfg: AlertConfig):
        self.cfg = cfg
        self.history = deque(maxlen=self.cfg.history_len)
        self.last_alert_time = 0.0
        self.last_alert_info = ""

    def update(self, detections: List[Detection]) -> bool:
        """
        Return True if we should trigger an alert this frame.
        """
        now = time.time()
        weapon_found = len(detections) > 0
        self.history.append(weapon_found)

        # Cooldown
        if now - self.last_alert_time < self.cfg.cooldown_sec:
            return False

        count_weapon = sum(self.history)
        if weapon_found and count_weapon >= self.cfg.min_weapon_frames:
            self.last_alert_time = now
            self.history.clear()
            return True

        return False

    def set_last_alert_info(self, text: str):
        self.last_alert_info = text

    def get_last_alert_info(self) -> str:
        return self.last_alert_info
