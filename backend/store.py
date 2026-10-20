"""SQLite repository: each write is a short transaction, no network side effects."""
import json
import hashlib
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Store:
    def __init__(self, directory: Path):
        self.directory = directory
        self.path = directory / "budget.sqlite3"

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "uploads").mkdir(exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS receipts (
                    id TEXT PRIMARY KEY, source TEXT NOT NULL,
                    image_hash TEXT UNIQUE, image_path TEXT,
                    store_name TEXT, purchased_at TEXT, total_amount INTEGER,
                    items TEXT NOT NULL DEFAULT '[]', raw TEXT,
                    predictions TEXT, engine TEXT NOT NULL DEFAULT 'manual',
                    status TEXT NOT NULL DEFAULT 'review',
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                CREATE INDEX IF NOT EXISTS receipt_date ON receipts(purchased_at);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, receipt_id TEXT NOT NULL UNIQUE REFERENCES receipts(id) ON DELETE CASCADE,
                    status TEXT NOT NULL DEFAULT 'queued', error TEXT,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(receipts)")}
            for name, declaration in {
                "owner_id": "TEXT NOT NULL DEFAULT 'local'", "memo": "TEXT NOT NULL DEFAULT ''",
                "category": "TEXT NOT NULL DEFAULT '기타'", "date_inferred": "INTEGER NOT NULL DEFAULT 0",
                "allow_training": "INTEGER NOT NULL DEFAULT 0", "content_hash": "TEXT",
            }.items():
                if name not in columns:
                    db.execute(f"ALTER TABLE receipts ADD COLUMN {name} {declaration}")
            # The original global unique image_hash remains compatible while its
            # value becomes owner-scoped. Keep the actual image digest separately.
            for row in db.execute("SELECT id,owner_id,image_hash FROM receipts WHERE image_hash IS NOT NULL AND content_hash IS NULL").fetchall():
                scoped = hashlib.sha256(f"{row['owner_id']}:{row['image_hash']}".encode()).hexdigest()
                db.execute("UPDATE receipts SET content_hash=?,image_hash=? WHERE id=?", (row['image_hash'], scoped, row['id']))
            for row in db.execute("SELECT id,created_at FROM receipts WHERE purchased_at IS NULL").fetchall():
                try:
                    created=datetime.fromisoformat(row['created_at'])
                    if created.tzinfo is None:created=created.replace(tzinfo=timezone.utc)
                    temporary_date=created.astimezone(ZoneInfo('Asia/Seoul')).date().isoformat()
                except (ValueError,TypeError):
                    temporary_date=datetime.now(ZoneInfo('Asia/Seoul')).date().isoformat()
                db.execute('UPDATE receipts SET purchased_at=?,date_inferred=1 WHERE id=?',(temporary_date,row['id']))
            db.executescript("""
                CREATE INDEX IF NOT EXISTS receipt_owner_date ON receipts(owner_id,purchased_at);
                CREATE TABLE IF NOT EXISTS monthly_memos (
                    owner_id TEXT NOT NULL, month TEXT NOT NULL, memo TEXT NOT NULL DEFAULT '' CHECK(length(memo)<=200),
                    PRIMARY KEY(owner_id,month)
                );
                CREATE TABLE IF NOT EXISTS corrections (
                    id TEXT PRIMARY KEY, receipt_id TEXT NOT NULL REFERENCES receipts(id) ON DELETE CASCADE,
                    owner_id TEXT NOT NULL, before_json TEXT NOT NULL, after_json TEXT NOT NULL,
                    changed_fields TEXT NOT NULL, ocr_snapshot TEXT, predictions_snapshot TEXT,
                    content_hash TEXT, engine TEXT NOT NULL, training_allowed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                CREATE INDEX IF NOT EXISTS correction_owner ON corrections(owner_id,receipt_id);
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at INTEGER NOT NULL
                );
                PRAGMA user_version=2;
            """)

    @staticmethod
    def receipt(row):
        if row is None:
            return None
        value = dict(row)
        for key in ("items", "raw", "predictions"):
            value[key] = json.loads(value[key]) if value[key] else None
        value.pop("image_path", None)
        value.pop("image_hash", None)
        value.pop("owner_id", None)
        value["has_image"] = bool(row["image_path"])
        return value

    def get(self, receipt_id, owner_id="local"):
        with self.connect() as db:
            return self.receipt(db.execute("SELECT * FROM receipts WHERE id=? AND owner_id=?", (receipt_id,owner_id)).fetchone())

    def job(self, job_id, owner_id="local"):
        with self.connect() as db:
            row = db.execute("SELECT j.* FROM jobs j JOIN receipts r ON r.id=j.receipt_id WHERE j.id=? AND r.owner_id=?", (job_id,owner_id)).fetchone()
            return dict(row) if row else None
