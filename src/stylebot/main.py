from __future__ import annotations

import logging
import sys
from pathlib import Path

from .bot import create_bot
from .config import load_settings


def configure_logging(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "stylebot.log", encoding="utf-8"),
        ],
    )


def main() -> int:
    settings = load_settings()
    configure_logging(settings.project_root)
    if not settings.bot_token:
        print("缺少 DISCORD_BOT_TOKEN。請先建立 .env。", file=sys.stderr)
        return 2
    if not settings.styles:
        print("沒有啟用的風格。請檢查 config/styles.yaml。", file=sys.stderr)
        return 2
    bot = create_bot(settings)
    bot.run(settings.bot_token, log_handler=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

