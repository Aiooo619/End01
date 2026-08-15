from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path

from .config import Settings


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    style_id: str
    version: str
    checkpoint: str
    model_type: str
    status: str
    path: str
    source_job_id: str


class ModelRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db_path = settings.data_root / "state" / "stylebot.sqlite3"
        self._prepare()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _prepare(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_versions (
                    model_id TEXT PRIMARY KEY,
                    style_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    checkpoint TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    source_job_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    promoted_at TEXT
                );
                CREATE TABLE IF NOT EXISTS generations (
                    generation_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    negative_prompt TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    strength REAL NOT NULL,
                    pose_path TEXT,
                    image_path TEXT NOT NULL,
                    discord_message_id TEXT,
                    selected INTEGER NOT NULL DEFAULT 0,
                    purpose TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    generation_id TEXT,
                    model_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    comment TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS iterations (
                    iteration_id TEXT PRIMARY KEY,
                    style_id TEXT NOT NULL,
                    parent_model_id TEXT NOT NULL,
                    change_summary TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def register_model(
        self,
        style_id: str,
        version: str,
        checkpoint: str,
        path: Path,
        source_job_id: str,
        metadata: dict | None = None,
    ) -> ModelRecord:
        style = self.settings.styles[style_id]
        relative = path.resolve().relative_to(self.settings.data_root).as_posix()
        model_id = f"{style_id}:{version}:{checkpoint}"
        record = ModelRecord(
            model_id=model_id,
            style_id=style_id,
            version=version,
            checkpoint=checkpoint,
            model_type=style.model_type,
            status="draft",
            path=relative,
            source_job_id=source_job_id,
        )
        now = datetime.now(UTC).isoformat()
        with closing(self._connect()) as db, db:
            db.execute(
                """
                INSERT INTO model_versions (
                    model_id, style_id, version, checkpoint, model_type, status,
                    path, source_job_id, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET path=excluded.path,
                    metadata_json=excluded.metadata_json
                """,
                (
                    *asdict(record).values(),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                ),
            )
        return record

    def list_models(self, style_id: str | None = None, production_only: bool = False) -> list[ModelRecord]:
        clauses: list[str] = []
        params: list[str] = []
        if style_id:
            clauses.append("style_id = ?")
            params.append(style_id)
        if production_only:
            clauses.append("status = 'production'")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with closing(self._connect()) as db, db:
            rows = db.execute(
                "SELECT model_id, style_id, version, checkpoint, model_type, status, path, source_job_id FROM model_versions"
                + where
                + " ORDER BY created_at DESC",
                params,
            ).fetchall()
        return [ModelRecord(**dict(row)) for row in rows]

    def get_model(self, model_id: str) -> ModelRecord | None:
        with closing(self._connect()) as db, db:
            row = db.execute(
                "SELECT model_id, style_id, version, checkpoint, model_type, status, path, source_job_id FROM model_versions WHERE model_id = ?",
                (model_id,),
            ).fetchone()
        return ModelRecord(**dict(row)) if row else None

    def promote(self, model_id: str) -> ModelRecord:
        record = self.get_model(model_id)
        if not record:
            raise ValueError("找不到模型版本。")
        now = datetime.now(UTC).isoformat()
        with closing(self._connect()) as db, db:
            db.execute(
                "UPDATE model_versions SET status = 'archived' WHERE style_id = ? AND status = 'production'",
                (record.style_id,),
            )
            db.execute(
                "UPDATE model_versions SET status = 'production', promoted_at = ? WHERE model_id = ?",
                (now, model_id),
            )
        return ModelRecord(**{**asdict(record), "status": "production"})

    def record_generation(
        self,
        model_id: str,
        prompt: str,
        negative_prompt: str,
        seed: int,
        strength: float,
        image_path: Path,
        purpose: str,
        pose_path: str | None = None,
    ) -> str:
        generation_id = uuid.uuid4().hex[:16]
        relative = image_path.resolve().relative_to(self.settings.data_root).as_posix()
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO generations VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?, ?)",
                (
                    generation_id, model_id, prompt, negative_prompt, seed, strength,
                    pose_path, relative, purpose, datetime.now(UTC).isoformat(),
                ),
            )
        return generation_id

    def attach_message(self, generation_id: str, message_id: str) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                "UPDATE generations SET discord_message_id = ? WHERE generation_id = ?",
                (message_id, generation_id),
            )

    def select_by_message(self, message_id: str) -> str | None:
        with closing(self._connect()) as db, db:
            row = db.execute(
                "SELECT generation_id FROM generations WHERE discord_message_id = ?",
                (message_id,),
            ).fetchone()
            if row:
                db.execute(
                    "UPDATE generations SET selected = 1 WHERE generation_id = ?",
                    (row["generation_id"],),
                )
        return row["generation_id"] if row else None

    def add_feedback(
        self,
        model_id: str,
        user_id: str,
        category: str,
        comment: str,
        generation_id: str | None = None,
    ) -> str:
        feedback_id = uuid.uuid4().hex[:16]
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    feedback_id, generation_id, model_id, user_id, category,
                    comment, datetime.now(UTC).isoformat(),
                ),
            )
        return feedback_id

    def create_iteration(self, parent_model_id: str, change_summary: str) -> str:
        record = self.get_model(parent_model_id)
        if not record:
            raise ValueError("找不到父模型。")
        iteration_id = f"iter-{uuid.uuid4().hex[:12]}"
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO iterations VALUES (?, ?, ?, ?, 'planned', ?)",
                (
                    iteration_id, record.style_id, parent_model_id, change_summary,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return iteration_id
