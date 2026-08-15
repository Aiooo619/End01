from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class StyleConfig:
    style_id: str
    display_name: str
    discord_channel_id: int | None
    trigger_token: str
    minimum_approved_images: int
    enabled: bool = True


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_root: Path
    bot_token: str
    guild_id: int | None
    forum_channel_id: int | None
    allowed_user_ids: frozenset[int]
    max_attachment_mb: int
    styles: dict[str, StyleConfig]


def _optional_int(value: str | int | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_settings(project_root: Path | None = None) -> Settings:
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    load_dotenv(root / ".env")

    styles_path = root / "config" / "styles.local.yaml"
    if not styles_path.exists():
        styles_path = root / "config" / "styles.yaml"
    if not styles_path.exists():
        styles_path = root / "config" / "styles.example.yaml"

    runtime_path = root / "config" / "runtime.local.yaml"
    runtime = _load_yaml(runtime_path) if runtime_path.exists() else {}

    raw_styles = _load_yaml(styles_path).get("styles", {})
    styles: dict[str, StyleConfig] = {}
    for style_id, raw in raw_styles.items():
        style = StyleConfig(
            style_id=style_id,
            display_name=str(raw.get("display_name", style_id)),
            discord_channel_id=_optional_int(raw.get("discord_channel_id")),
            trigger_token=str(raw.get("trigger_token", f"{style_id}_style")),
            minimum_approved_images=int(raw.get("minimum_approved_images", 50)),
            enabled=bool(raw.get("enabled", True)),
        )
        if style.enabled:
            styles[style_id] = style

    allowed_value = str(runtime.get("allowed_user_ids", os.getenv("DISCORD_ALLOWED_USER_IDS", "")))
    allowed = frozenset(
        int(part.strip())
        for part in allowed_value.split(",")
        if part.strip()
    )
    data_root_value = os.getenv("AI_STYLE_DATA_ROOT", "").strip()
    data_root = Path(data_root_value).expanduser().resolve() if data_root_value else root

    return Settings(
        project_root=root,
        data_root=data_root,
        bot_token=os.getenv("DISCORD_BOT_TOKEN", "").strip(),
        guild_id=_optional_int(runtime.get("guild_id", os.getenv("DISCORD_GUILD_ID"))),
        forum_channel_id=_optional_int(
            runtime.get("forum_channel_id", os.getenv("DISCORD_FORUM_CHANNEL_ID"))
        ),
        allowed_user_ids=allowed,
        max_attachment_mb=int(os.getenv("MAX_ATTACHMENT_MB", "25")),
        styles=styles,
    )
