import os
from dataclasses import dataclass

import yaml
from dotenv import load_dotenv


@dataclass
class ModelConfig:
    model_path: str
    img_size: int
    conf_th: float
    iou_th: float
    weapon_names: list[str]


@dataclass
class VideoConfig:
    source: int | str
    width: int
    height: int


@dataclass
class AlertConfig:
    cooldown_sec: int
    history_len: int
    min_weapon_frames: int


@dataclass
class TelegramConfig:
    bot_token: str
    chat_id: str
    timeout: int


@dataclass
class LoggingConfig:
    log_csv: str
    crop_dir: str
    enable_crop_save: bool


@dataclass
class HUDConfig:
    show_window: bool
    window_name: str
    font_scale: float
    font_thickness: int


@dataclass
class AppConfig:
    model: ModelConfig
    video: VideoConfig
    alert: AlertConfig
    telegram: TelegramConfig
    logging: LoggingConfig
    hud: HUDConfig


def load_config(path: str = "config.yaml") -> AppConfig:
    load_dotenv()
    with open(path) as f:
        data = yaml.safe_load(f)

    # Secrets come from the environment, never from the committed YAML.
    tg = data["telegram"]
    tg["bot_token"] = os.getenv("TELEGRAM_BOT_TOKEN", tg.get("bot_token", ""))
    tg["chat_id"] = os.getenv("TELEGRAM_CHAT_ID", tg.get("chat_id", ""))

    return AppConfig(
        model=ModelConfig(**data["model"]),
        video=VideoConfig(**data["video"]),
        alert=AlertConfig(**data["alert"]),
        telegram=TelegramConfig(**data["telegram"]),
        logging=LoggingConfig(**data["logging"]),
        hud=HUDConfig(**data["hud"]),
    )
