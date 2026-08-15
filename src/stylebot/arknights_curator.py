from __future__ import annotations

import argparse
import json
import re
import unicodedata
import urllib.request
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


RAW_ROOT = (
    "https://raw.githubusercontent.com/ArknightsAssets/ArknightsAssets/cn/"
    "assets/torappu/dynamicassets/arts/characters/"
)
RAW_ROOT_2 = (
    "https://raw.githubusercontent.com/ArknightsAssets/ArknightsAssets2/cn/"
    "assets/dyn/arts/characters/"
)


def normalized(value: str) -> str:
    value = value.replace("ł", "l").replace("Ł", "L")
    value = "".join(
        char for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )
    return re.sub(r"[^\w]+", "", value.casefold(), flags=re.UNICODE)


def load_character_ids(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload.get("characters"), list):
        items = payload["characters"]
    else:
        items = [{"key": key, "value": value} for key, value in payload.items()]
    result: dict[str, str] = {}
    for item in items:
        name = item["value"].get("name")
        if name:
            name_key = normalized(name)
            current = result.get(name_key, "")
            if not current.startswith("char_") or item["key"].startswith("char_"):
                result[name_key] = item["key"]
    return result


def choose_art(character_id: str, tree_path: Path) -> str | None:
    entries = json.loads(tree_path.read_text(encoding="utf-8"))["tree"]
    prefix = f"{character_id}/"
    paths = [item["path"] for item in entries if item.get("type") == "blob" and item["path"].startswith(prefix)]
    exact_e2 = f"{character_id}/{character_id}_2.png"
    exact_e0 = f"{character_id}/{character_id}_1.png"
    if exact_e2 in paths:
        return exact_e2
    if exact_e0 in paths:
        return exact_e0
    clean = [path for path in paths if path.endswith(".png") and not path.endswith("b.png")]
    return clean[0] if clean else None


def build_manifest(root: Path) -> list[dict]:
    roster_path = root / "config" / "arknights_popularity_top50.json"
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    english = root / "work" / "character_table_en.json"
    ids = load_character_ids(english if english.exists() else root / "work" / "character_table.json")
    chinese = root / "work" / "character_table_zh.json"
    if chinese.exists():
        ids.update(load_character_ids(chinese))
    tree_paths = [root / "work" / "characters_tree.json", root / "work" / "characters_tree2.json"]
    records: list[dict] = []
    for item in roster["characters"]:
        names = [item["preferred"], *item.get("lookup", []), item["group"]]
        character_id = next((ids.get(normalized(name)) for name in names if ids.get(normalized(name))), None)
        art = None
        source_root = None
        if character_id:
            for tree_path, candidate_root in zip(tree_paths, (RAW_ROOT, RAW_ROOT_2)):
                if tree_path.exists():
                    art = choose_art(character_id, tree_path)
                    if art:
                        source_root = candidate_root
                        break
        records.append(
            {
                **item,
                "character_id": character_id,
                "asset_path": art,
                "source_url": source_root + art if art and source_root else None,
                "status": "ready" if art else "unresolved",
            }
        )
    return records


def download_one(record: dict, destination: Path) -> dict:
    if not record["source_url"]:
        return record
    filename = f"{record['rank']:02d}_{record['character_id']}.png"
    target = destination / filename
    request = urllib.request.Request(record["source_url"], headers={"User-Agent": "End01-curator"})
    with urllib.request.urlopen(request, timeout=90) as response:
        target.write_bytes(response.read())
    with Image.open(target) as image:
        image.verify()
    with Image.open(target) as image:
        record.update(filename=filename, width=image.width, height=image.height, status="downloaded")
    return record


def collect(root: Path, workers: int = 6) -> list[dict]:
    records = build_manifest(root)
    destination = root / "datasets" / "arknights_top50" / "curation"
    destination.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, item, destination): item for item in records}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                futures[future]["status"] = "failed"
                futures[future]["error"] = str(exc)
    manifest_path = root / "state" / "arknights_top50_curation.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def update_record(root: Path, rank: int, **changes: object) -> dict:
    path = root / "state" / "arknights_top50_curation.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    record = next(item for item in records if item["rank"] == rank)
    record.update(changes)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def accept_record(root: Path, rank: int, answers: dict[str, str], training_filename: str | None = None) -> tuple[Path, Path]:
    manifest = root / "state" / "arknights_top50_curation.json"
    records = json.loads(manifest.read_text(encoding="utf-8"))
    record = next(item for item in records if item["rank"] == rank)
    source = root / "datasets" / "arknights_top50" / "curation" / record["filename"]
    approved = root / "datasets" / "arknights_portrait" / "approved" / (training_filename or record["filename"])
    caption = root / "datasets" / "arknights_portrait" / "captions" / f"{source.stem}.txt"
    approved.parent.mkdir(parents=True, exist_ok=True)
    caption.parent.mkdir(parents=True, exist_ok=True)
    if not approved.exists():
        shutil.copy2(source, approved)
    fields = [
        "arknights_portrait_style",
        "character costume design",
        "full body",
        answers.get("form", record["preferred"]),
        answers.get("structure", ""),
        answers.get("materials", ""),
        answers.get("focus", "clothing structure"),
    ]
    caption.write_text(", ".join(item.strip() for item in fields if item.strip()) + "\n", encoding="utf-8")
    doc = root / "records" / "curation" / "arknights_top50" / f"{rank:02d}_{record['character_id']}.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(
        "\n".join(
            [
                f"# #{rank} {record['group']}",
                "",
                f"- Confirmed form: {answers.get('form', record['preferred'])}",
                f"- Clothing structure: {answers.get('structure', '')}",
                f"- Materials/accessories: {answers.get('materials', '')}",
                f"- Learning focus: {answers.get('focus', '')}",
                f"- Notes/exclusions: {answers.get('notes', '')}",
                f"- Source: {record['source_url']}",
                f"- Ranking source: https://www.reddit.com/r/arknights/comments/1r4yd99/",
                "- Decision: approved by human review",
                "",
            ]
        ),
        encoding="utf-8",
    )
    update_record(root, rank, status="approved", answers=answers)
    return approved, caption


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect traceable Arknights top-50 character art")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    records = collect(args.root.resolve())
    counts = {status: sum(item["status"] == status for item in records) for status in {item["status"] for item in records}}
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
