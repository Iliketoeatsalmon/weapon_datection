from pathlib import Path

from weapon_detection.config import load_config

ROOT = Path(__file__).resolve().parent.parent


def test_repo_config_loads():
    cfg = load_config(str(ROOT / "config.yaml"))
    assert cfg.model.weapon_names
    assert cfg.alert.min_weapon_frames >= 1


def test_env_overrides_telegram_secrets(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    cfg = load_config(str(ROOT / "config.yaml"))
    assert (cfg.telegram.bot_token, cfg.telegram.chat_id) == ("tok", "42")
