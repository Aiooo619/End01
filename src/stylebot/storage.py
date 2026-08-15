from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import sqlite3
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .config import Settings, StyleConfig


ALLOWED_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


class IngestError(ValueError):
    pass


@dataclass(frozen=True)
class IngestResult:
    status: str
    style_id: str
    sha256: str
    filename: str
    width: int
    height: int
    size_bytes: int


class DatasetStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db_path = settings.data_root / "state" / "stylebot.sqlite3"
        self.events_root = settings.data_root / "records" / "events"
        self.queue_root = settings.data_root / "queues"
        self._prepare()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _prepare(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_root.mkdir(parents=True, exist_ok=True)
        self.queue_root.mkdir(parents=True, exist_ok=True)
        for style in self.settings.styles.values():
            for bucket in ("incoming", "approved", "rejected", "captions"):
                (self.settings.data_root / "datasets" / style.style_id / bucket).mkdir(
                    parents=True, exist_ok=True
                )
        with closing(self._connect()) as db, db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS images (
                    sha256 TEXT PRIMARY KEY,
                    style_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    discord_message_id TEXT NOT NULL,
                    discord_user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _event(self, event: dict) -> None:
        now = datetime.now(UTC)
        event = {"timestamp": now.isoformat(), **event}
        path = self.events_root / f"{now.date().isoformat()}.jsonl"
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    @staticmethod
    def _safe_stem(filename: str) -> str:
        stem = Path(filename).stem
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
        return stem[:60] or "image"

    def ingest(
        self,
        style: StyleConfig,
        payload: bytes,
        original_filename: str,
        discord_message_id: str,
        discord_user_id: str,
    ) -> IngestResult:
        max_bytes = self.settings.max_attachment_mb * 1024 * 1024
        if not payload:
            raise IngestError("附件是空的。")
        if len(payload) > max_bytes:
            raise IngestError(f"附件超過 {self.settings.max_attachment_mb}MB 限制。")

        try:
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
            with Image.open(io.BytesIO(payload)) as image:
                image_format = (image.format or "").upper()
                width, height = image.size
        except (UnidentifiedImageError, OSError) as exc:
            raise IngestError("附件不是有效的 JPEG、PNG 或 WebP 圖片。") from exc

        if image_format not in ALLOWED_FORMATS:
            raise IngestError("只接受 JPEG、PNG 或 WebP 圖片。")
        if width < 512 or height < 512:
            raise IngestError(f"解析度 {width}×{height} 太低，短邊至少需要 512px。")

        digest = hashlib.sha256(payload).hexdigest()
        with closing(self._connect()) as db, db:
            existing = db.execute(
                "SELECT style_id, filename, width, height, size_bytes FROM images WHERE sha256 = ?",
                (digest,),
            ).fetchone()
        if existing:
            return IngestResult(
                status="duplicate",
                style_id=existing["style_id"],
                sha256=digest,
                filename=existing["filename"],
                width=existing["width"],
                height=existing["height"],
                size_bytes=existing["size_bytes"],
            )

        extension = ALLOWED_FORMATS[image_format]
        filename = f"{self._safe_stem(original_filename)}_{digest[:12]}{extension}"
        relative_path = Path("datasets") / style.style_id / "incoming" / filename
        destination = self.settings.data_root / relative_path
        destination.write_bytes(payload)
        now = datetime.now(UTC).isoformat()

        try:
            with closing(self._connect()) as db, db:
                db.execute(
                    """
                    INSERT INTO images (
                        sha256, style_id, filename, local_path, status, width,
                        height, size_bytes, discord_message_id, discord_user_id,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'incoming', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        digest,
                        style.style_id,
                        filename,
                        relative_path.as_posix(),
                        width,
                        height,
                        len(payload),
                        discord_message_id,
                        discord_user_id,
                        now,
                        now,
                    ),
                )
        except Exception:
            destination.unlink(missing_ok=True)
            raise

        result = IngestResult(
            status="received",
            style_id=style.style_id,
            sha256=digest,
            filename=filename,
            width=width,
            height=height,
            size_bytes=len(payload),
        )
        self._event({"event": "image_received", **asdict(result)})
        return result

    def status(self, style_id: str) -> dict[str, int]:
        with closing(self._connect()) as db, db:
            rows = db.execute(
                "SELECT status, COUNT(*) AS count FROM images WHERE style_id = ? GROUP BY status",
                (style_id,),
            ).fetchall()
        counts = {"incoming": 0, "approved": 0, "rejected": 0}
        counts.update({row["status"]: row["count"] for row in rows})
        return counts

    def pending(self, style_id: str, limit: int = 10) -> list[dict[str, str | int]]:
        with closing(self._connect()) as db, db:
            rows = db.execute(
                "SELECT sha256, filename, width, height, discord_message_id FROM images WHERE style_id = ? AND status = 'incoming' ORDER BY created_at LIMIT ?",
                (style_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def approved_items(self, style_id: str) -> list[dict[str, str | int]]:
        with closing(self._connect()) as db, db:
            rows = db.execute(
                "SELECT sha256, filename, local_path, width, height FROM images WHERE style_id = ? AND status = 'approved' ORDER BY created_at",
                (style_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _queue_caption(self, row: sqlite3.Row, destination: Path) -> None:
        caption_root = self.queue_root / "captions"
        caption_root.mkdir(parents=True, exist_ok=True)
        job = {
            "job_id": f"caption-{row['sha256'][:16]}",
            "style_id": row["style_id"],
            "sha256": row["sha256"],
            "image_path": destination.relative_to(self.settings.data_root).as_posix(),
            "status": "queued",
            "created_at": datetime.now(UTC).isoformat(),
        }
        (caption_root / f"{job['job_id']}.json").write_text(
            json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def queue_missing_captions(self) -> int:
        with closing(self._connect()) as db, db:
            rows = db.execute(
                "SELECT sha256, style_id, filename, local_path FROM images WHERE status = 'approved' ORDER BY created_at"
            ).fetchall()
        queued = 0
        for row in rows:
            image_path = self.settings.data_root / row["local_path"]
            caption_path = (
                self.settings.data_root
                / "datasets"
                / row["style_id"]
                / "captions"
                / f"{image_path.stem}.txt"
            )
            job_path = self.queue_root / "captions" / f"caption-{row['sha256'][:16]}.json"
            if not caption_path.exists() and not job_path.exists():
                self._queue_caption(row, image_path)
                queued += 1
        return queued

    def approve(self, style_id: str, limit: int = 100) -> int:
        with closing(self._connect()) as db, db:
            rows = db.execute(
                "SELECT sha256, style_id, filename, local_path FROM images WHERE style_id = ? AND status = 'incoming' ORDER BY created_at LIMIT ?",
                (style_id, limit),
            ).fetchall()
            moved: list[tuple[sqlite3.Row, Path]] = []
            try:
                for row in rows:
                    source = self.settings.data_root / row["local_path"]
                    destination = self.settings.data_root / "datasets" / style_id / "approved" / row["filename"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(source, destination)
                    moved.append((row, destination))
                    db.execute(
                        "UPDATE images SET status = 'approved', local_path = ?, updated_at = ? WHERE sha256 = ?",
                        (
                            destination.relative_to(self.settings.data_root).as_posix(),
                            datetime.now(UTC).isoformat(),
                            row["sha256"],
                        ),
                    )
                    self._queue_caption(row, destination)
            except Exception:
                for row, destination in reversed(moved):
                    if destination.exists():
                        shutil.move(destination, self.settings.data_root / row["local_path"])
                raise
        if rows:
            self._event({"event": "images_approved", "style_id": style_id, "count": len(rows)})
        return len(rows)

    def reject(self, style_id: str, limit: int = 100, message_id: str | None = None) -> int:
        query = (
            "SELECT sha256, style_id, filename, local_path FROM images "
            "WHERE style_id = ? AND status = 'incoming'"
        )
        parameters: list[str | int] = [style_id]
        if message_id is not None:
            query += " AND discord_message_id = ?"
            parameters.append(message_id)
        query += " ORDER BY created_at LIMIT ?"
        parameters.append(limit)
        with closing(self._connect()) as db, db:
            rows = db.execute(query, parameters).fetchall()
            moved: list[tuple[sqlite3.Row, Path]] = []
            try:
                for row in rows:
                    source = self.settings.data_root / row["local_path"]
                    destination = self.settings.data_root / "datasets" / style_id / "rejected" / row["filename"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(source, destination)
                    moved.append((row, destination))
                    db.execute(
                        "UPDATE images SET status = 'rejected', local_path = ?, updated_at = ? WHERE sha256 = ?",
                        (
                            destination.relative_to(self.settings.data_root).as_posix(),
                            datetime.now(UTC).isoformat(),
                            row["sha256"],
                        ),
                    )
            except Exception:
                for row, destination in reversed(moved):
                    if destination.exists():
                        shutil.move(destination, self.settings.data_root / row["local_path"])
                raise
        if rows:
            self._event({"event": "images_rejected", "style_id": style_id, "count": len(rows)})
        return len(rows)

    def queue_training(self, style: StyleConfig) -> Path:
        counts = self.status(style.style_id)
        if counts["approved"] < style.minimum_approved_images:
            raise IngestError(
                f"已批准 {counts['approved']} 張，至少需要 {style.minimum_approved_images} 張。"
            )
        job_id = f"{style.style_id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
        job = {
            "job_id": job_id,
            "style_id": style.style_id,
            "status": "queued",
            "approved_images": counts["approved"],
            "created_at": datetime.now(UTC).isoformat(),
        }
        destination = self.queue_root / f"{job_id}.json"
        destination.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        self._event({"event": "training_queued", **job})
        return destination
