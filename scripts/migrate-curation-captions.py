from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "state" / "arknights_top50_curation.json").read_text(encoding="utf-8"))
    captions = root / "datasets" / "arknights_portrait" / "captions"
    approved = root / "datasets" / "arknights_portrait" / "approved"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = root / "work" / "caption_backups" / f"curation-migration-{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    migrated = 0
    for record in manifest:
        if record.get("status") != "approved" or not record.get("filename"):
            continue
        source = captions / f"{Path(record['filename']).stem}.txt"
        matches = sorted(approved.glob(f"{Path(record['filename']).stem}_*"))
        if not source.exists() or len(matches) != 1:
            continue
        target = captions / f"{matches[0].stem}.txt"
        if target.exists():
            shutil.copy2(target, backup / target.name)
        shutil.copy2(source, target)
        migrated += 1
    report = {"migrated": migrated, "backup": backup.as_posix()}
    (backup / "migration-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
