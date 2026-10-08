from weapon_detection.alert import AlertManager
from weapon_detection.config import AlertConfig

DET = [("pistol", 0.9, 0, 0, 10, 10)]


def make(cooldown=15, history=10, min_frames=3):
    return AlertManager(AlertConfig(cooldown_sec=cooldown, history_len=history, min_weapon_frames=min_frames))


def test_no_alert_without_detections():
    assert make().update([]) is False


def test_needs_min_frames_before_alert():
    m = make(min_frames=3)
    assert [m.update(DET) for _ in range(3)] == [False, False, True]


def test_cooldown_blocks_second_alert():
    m = make(cooldown=60, min_frames=1)
    assert m.update(DET) is True
    assert m.update(DET) is False


def test_last_alert_info_roundtrip():
    m = make()
    m.set_last_alert_info("rifle 0.90")
    assert m.get_last_alert_info() == "rifle 0.90"
