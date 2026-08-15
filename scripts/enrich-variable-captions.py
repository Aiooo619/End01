from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from stylebot.captioner import WDTagger

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
REQUIRED = {"arknights_portrait_style", "character costume design", "solo", "full body"}

# These are deliberately limited to visible traits that should remain prompt-controllable.
# Identity, facial-expression and rendering-style tags are not added here.
VARIABLE_GROUPS = {
    "subject": {
        "1girl", "1boy", "male focus", "androgynous",
    },
    "species": {
        "animal ears", "cat ears", "dog ears", "fox ears", "wolf ears",
        "rabbit ears", "pointy ears", "tail", "cat tail", "dog tail",
        "fox tail", "wolf tail", "multiple tails", "horns", "halo", "wings",
    },
    "hair": {
        "short hair", "medium hair", "long hair", "very long hair", "ponytail",
        "twintails", "braid", "braided hair",
    },
    "pose": {
        "standing", "sitting", "walking", "hand on own hip", "hand up",
        "salute", "arms at sides", "crossed arms", "holding", "holding weapon",
        "weapon", "spread arms", "outstretched arm",
    },
}
VARIABLES = set().union(*VARIABLE_GROUPS.values())
GROUP_ORDER = tuple(VARIABLE_GROUPS)


def estimated_tokens(tags: list[str]) -> int:
    text = ", ".join(tags)
    return (
        len(re.findall(r"[\u3400-\u9fff]", text)) * 2
        + len(re.findall(r"[A-Za-z0-9_'-]+", text))
        + len(re.findall(r"[,.;:，。；：]", text))
    )


def split_tags(text: str) -> list[str]:
    return [tag.strip().lower() for tag in text.replace("\n", " ").split(",") if tag.strip()]


def detected_variables(tagger: WDTagger, image: Path) -> list[str]:
    raw = split_tags(tagger.caption(image, "_discard_trigger_"))[2:]
    found = set(raw) & VARIABLES
    # WDTagger occasionally emits both labels for an androgynous design. A caption
    # must not teach mutually exclusive subject tokens for the same single figure.
    if "1girl" in found and "1boy" in found:
        found.discard("1boy")
        found.discard("male focus")
    return [tag for group in GROUP_ORDER for tag in sorted(VARIABLE_GROUPS[group]) if tag in found]


def enrich(base: list[str], variables: list[str]) -> list[str]:
    base = [tag for tag in base if tag not in VARIABLES]
    combined = list(dict.fromkeys([*base, *variables]))
    protected = REQUIRED | set(variables)
    while estimated_tokens(combined) > 70:
        removable = next(
            (index for index in range(len(combined) - 1, -1, -1) if combined[index] not in protected),
            None,
        )
        if removable is None:
            break
        combined.pop(removable)
    return combined


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# v003 可控變量與資料比例審計",
        "",
        f"- 已核准圖片：{summary['images']}",
        f"- 已審計 caption：{summary['images']}",
        f"- 含可控變量：{summary['with_variables']}",
        f"- 最高估算 token：{summary['max_estimated_tokens']}",
        f"- 超過 70 token：{summary['over_budget']}",
        "",
        "## 變量計數",
        "",
        "| 分組 | 標籤 | 圖片數 | 比例 |",
        "|---|---|---:|---:|",
    ]
    for group in GROUP_ORDER:
        counts = summary["groups"][group]
        if not counts:
            lines.append(f"| {group} | （未偵測） | 0 | 0.0% |")
        for tag, count in counts.items():
            lines.append(f"| {group} | `{tag}` | {count} | {count / summary['images']:.1%} |")
    lines.extend([
        "",
        "## 判讀原則",
        "",
        "這些標籤描述圖片中確實存在、但不應被綁定到風格觸發詞的特徵。自動標註只作為 v004 前的初篩；低頻或明顯失衡的項目應人工抽查，必要時補充相反案例後再訓練。",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Add prompt-controllable visual variables to captions")
    parser.add_argument("--approved-root", type=Path, default=Path("datasets/arknights_portrait/approved"))
    parser.add_argument("--caption-root", type=Path, default=Path("datasets/arknights_portrait/captions"))
    parser.add_argument("--cache-root", type=Path, default=Path("cache/wd-tagger"))
    parser.add_argument("--report", type=Path, default=Path("records/evaluations/v003-variable-balance.md"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    images = sorted(path for path in args.approved_root.resolve().iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    missing = [image.name for image in images if not (args.caption_root.resolve() / f"{image.stem}.txt").exists()]
    if missing:
        raise SystemExit(f"Missing captions for {len(missing)} approved images: {missing[:5]}")

    tagger = WDTagger(args.cache_root.resolve())
    changes = []
    group_counts = {group: Counter() for group in GROUP_ORDER}
    for image in images:
        caption = args.caption_root.resolve() / f"{image.stem}.txt"
        before = caption.read_text(encoding="utf-8").strip()
        variables = detected_variables(tagger, image)
        after_tags = enrich(split_tags(before), variables)
        for group, allowed in VARIABLE_GROUPS.items():
            group_counts[group].update(tag for tag in variables if tag in allowed)
        changes.append({
            "caption": caption,
            "before": before,
            "after": ", ".join(after_tags),
            "variables": variables,
            "tokens": estimated_tokens(after_tags),
        })

    summary = {
        "images": len(images),
        "changed": sum(row["before"] != row["after"] for row in changes),
        "with_variables": sum(bool(row["variables"]) for row in changes),
        "max_estimated_tokens": max(row["tokens"] for row in changes),
        "over_budget": sum(row["tokens"] > 70 for row in changes),
        "groups": {
            group: dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
            for group, counts in group_counts.items()
        },
    }

    if args.apply:
        project_root = Path(__file__).resolve().parents[1]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = project_root / "work" / "caption_backups" / f"variables-{stamp}"
        backup.mkdir(parents=True, exist_ok=False)
        for row in changes:
            shutil.copy2(row["caption"], backup / row["caption"].name)
            row["caption"].write_text(row["after"] + "\n", encoding="utf-8")
        (backup / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown(args.report.resolve(), summary)
        summary["backup"] = backup.as_posix()
        summary["report"] = args.report.resolve().as_posix()

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
