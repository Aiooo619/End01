from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


KEEP_TERMS = (
    "dress", "footwear", "shoe", "boot", "heel", "sock", "thighhigh",
    "pants", "shorts", "skirt", "shirt", "blouse", "coat", "jacket",
    "cape", "belt", "glove", "sleeve", "collar", "ribbon", "hat",
    "headwear", "jewelry", "earring", "necklace", "bracelet", "eyewear",
    "glasses", "apron", "uniform", "armor", "vest", "scarf", "necktie",
    "bowtie", "hood", "fabric", "leather", "metal", "transparent",
    "id card", "vial", "test tube", "bag", "pouch", "harness", "strap",
    "tassel", "brooch", "zipper", "overalls", "sweater", "cardigan",
    "coat on shoulders", "jacket on shoulders", "asymmetrical legwear",
)
COMPOSITION = ("solo", "full body", "standing", "simple background", "white background")
REQUIRED = ("arknights_portrait_style", "character costume design")
BLOCK_TAGS = {"ears through headwear", "adjusting eyewear"}


def clean_caption(text: str) -> tuple[str, list[str]]:
    tags = [tag.strip().lower() for tag in text.replace("\n", " ").split(",") if tag.strip()]
    kept: list[str] = list(REQUIRED)
    for tag in tags:
        if tag in REQUIRED:
            continue
        if tag not in BLOCK_TAGS and (tag in COMPOSITION or any(term in tag for term in KEEP_TERMS)):
            if tag not in kept:
                kept.append(tag)
    if "solo" not in kept:
        kept.append("solo")
    if "full body" not in kept:
        kept.append("full body")
    removed = [tag for tag in tags if tag not in kept and tag not in REQUIRED]
    return ", ".join(kept), removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean costume-design training captions")
    parser.add_argument("--root", type=Path, default=Path("datasets/arknights_portrait/captions"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    files = sorted(args.root.resolve().glob("*.txt"))
    if not files:
        raise SystemExit("No caption files found")

    changes: list[dict] = []
    removed_counts: Counter[str] = Counter()
    for path in files:
        original = path.read_text(encoding="utf-8").strip()
        cleaned, removed = clean_caption(original)
        removed_counts.update(removed)
        changes.append({
            "file": path.name,
            "before": original,
            "after": cleaned,
            "removed_count": len(removed),
        })

    report = {
        "files": len(files),
        "changed": sum(item["before"] != item["after"] for item in changes),
        "removed_tags": sum(removed_counts.values()),
        "top_removed": removed_counts.most_common(30),
        "samples": changes[:5],
    }
    if args.apply:
        project_root = Path(__file__).resolve().parents[1]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = project_root / "work" / "caption_backups" / stamp
        backup.mkdir(parents=True, exist_ok=False)
        for path, item in zip(files, changes, strict=True):
            shutil.copy2(path, backup / path.name)
            path.write_text(item["after"] + "\n", encoding="utf-8")
        report["backup"] = backup.as_posix()
        report_path = backup / "cleaning-report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
