from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import load_settings


def main() -> int:
    settings = load_settings()
    manifest = settings.project_root / "state" / "arknights_top50_curation.json"
    records = json.loads(manifest.read_text(encoding="utf-8"))
    style = settings.styles["arknights_portrait"]
    deleted = 0
    for record in records:
        message_id = record.get("discord_message_id")
        if not message_id:
            continue
        url = f"https://discord.com/api/v10/channels/{style.discord_channel_id}/messages/{message_id}"
        request = urllib.request.Request(
            url, method="DELETE", headers={"Authorization": f"Bot {settings.bot_token}", "User-Agent": "End01-curator"}
        )
        while True:
            try:
                with urllib.request.urlopen(request, timeout=30):
                    pass
                deleted += 1
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    break
                if exc.code == 429:
                    retry = json.loads(exc.read().decode("utf-8")).get("retry_after", 1)
                    time.sleep(float(retry) + 0.1)
                    continue
                raise
        record.pop("discord_message_id", None)
        if record.get("status") != "unresolved":
            record["status"] = "downloaded"
        time.sleep(0.35)
    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"deleted={deleted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
