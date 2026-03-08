"""
FinOS — Personal AI Financial Operating System
Database Layer — database.py

Uses SQLite (local/dev) or PostgreSQL (production via DATABASE_URL env var).
All records are isolated per user_id. Schema is auto-created on first run.
"""

import os
import json
import time
import random
import string
import sqlite3
from typing import List, Dict, Optional, Any
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "")
USE_POSTGRES = DATABASE_URL.startswith("postgresql") or DATABASE_URL.startswith("postgres")

if USE_POSTGRES:
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        print("[DB] psycopg2 not installed — falling back to SQLite")
        USE_POSTGRES = False

SQLITE_PATH = os.environ.get("SQLITE_PATH", "./finos.db")

# ─── COLLECTIONS (maps to table names) ───────────────────────────────────────
COLLECTIONS = ["expenses", "investments", "assets", "recurring", "savings"]


def _generate_id() -> str:
    ts = str(int(time.time() * 1000))
    rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{ts}_{rand}"


class Database:
    """
    Unified database interface.
    All records are stored as JSON blobs with a user_id column for isolation.
    Primary key is a string ID generated client-side-style.
    """

    def __init__(self):
        if USE_POSTGRES:
            self._conn_pg = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
            self._conn_pg.autocommit = True
            self._backend = "postgres"
        else:
            self._conn_sqlite = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
            self._conn_sqlite.row_factory = sqlite3.Row
            self._backend = "sqlite"

        self._init_schema()
        print(f"[DB] Using backend: {self._backend}")

    # ── SCHEMA ────────────────────────────────────────────────────────────────
    def _init_schema(self):
        for col in COLLECTIONS:
            if self._backend == "sqlite":
                self._conn_sqlite.execute(f"""
                    CREATE TABLE IF NOT EXISTS {col} (
                        id         TEXT PRIMARY KEY,
                        user_id    TEXT NOT NULL,
                        data       TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                """)
                self._conn_sqlite.execute(f"CREATE INDEX IF NOT EXISTS idx_{col}_user ON {col}(user_id)")
                self._conn_sqlite.commit()
            else:
                cur = self._conn_pg.cursor()
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {col} (
                        id         TEXT PRIMARY KEY,
                        user_id    TEXT NOT NULL,
                        data       JSONB NOT NULL,
                        created_at TEXT NOT NULL
                    )
                """)
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{col}_user ON {col}(user_id)")

    # ── CURSOR HELPER ─────────────────────────────────────────────────────────
    def _execute(self, sql: str, params: tuple = ()):
        if self._backend == "sqlite":
            return self._conn_sqlite.execute(sql, params)
        else:
            cur = self._conn_pg.cursor()
            cur.execute(sql, params)
            return cur

    def _commit(self):
        if self._backend == "sqlite":
            self._conn_sqlite.commit()

    # ── CRUD ──────────────────────────────────────────────────────────────────
    def add_record(self, user_id: str, collection: str, record: Dict) -> Dict:
        """Insert a new record. Assigns id and created_at automatically."""
        if collection not in COLLECTIONS:
            raise ValueError(f"Unknown collection: {collection}")

        record_id = _generate_id()
        created_at = datetime.utcnow().isoformat()
        record["id"] = record_id
        record["created_at"] = created_at

        if self._backend == "sqlite":
            self._execute(
                f"INSERT INTO {collection} (id, user_id, data, created_at) VALUES (?,?,?,?)",
                (record_id, user_id, json.dumps(record), created_at)
            )
        else:
            self._execute(
                f"INSERT INTO {collection} (id, user_id, data, created_at) VALUES (%s,%s,%s,%s)",
                (record_id, user_id, json.dumps(record), created_at)
            )
        self._commit()
        return record

    def get_records(
        self,
        user_id: str,
        collection: str,
        limit: int = 200,
        order: str = "DESC"
    ) -> List[Dict]:
        """Fetch all records for a user, ordered by creation time."""
        if collection not in COLLECTIONS:
            raise ValueError(f"Unknown collection: {collection}")

        if self._backend == "sqlite":
            cur = self._execute(
                f"SELECT data FROM {collection} WHERE user_id=? ORDER BY created_at {order} LIMIT ?",
                (user_id, limit)
            )
            rows = cur.fetchall()
            return [json.loads(row[0]) for row in rows]
        else:
            cur = self._execute(
                f"SELECT data FROM {collection} WHERE user_id=%s ORDER BY created_at {order} LIMIT %s",
                (user_id, limit)
            )
            rows = cur.fetchall()
            return [dict(row["data"]) for row in rows]

    def get_record(self, user_id: str, collection: str, record_id: str) -> Optional[Dict]:
        """Fetch a single record by id."""
        if collection not in COLLECTIONS:
            raise ValueError(f"Unknown collection: {collection}")

        if self._backend == "sqlite":
            cur = self._execute(
                f"SELECT data FROM {collection} WHERE user_id=? AND id=?",
                (user_id, record_id)
            )
            row = cur.fetchone()
            return json.loads(row[0]) if row else None
        else:
            cur = self._execute(
                f"SELECT data FROM {collection} WHERE user_id=%s AND id=%s",
                (user_id, record_id)
            )
            row = cur.fetchone()
            return dict(row["data"]) if row else None

    def update_record(
        self,
        user_id: str,
        collection: str,
        record_id: str,
        updates: Dict
    ) -> Optional[Dict]:
        """Merge updates into an existing record."""
        existing = self.get_record(user_id, collection, record_id)
        if existing is None:
            return None

        existing.update(updates)
        existing["updated_at"] = datetime.utcnow().isoformat()

        if self._backend == "sqlite":
            self._execute(
                f"UPDATE {collection} SET data=? WHERE user_id=? AND id=?",
                (json.dumps(existing), user_id, record_id)
            )
        else:
            self._execute(
                f"UPDATE {collection} SET data=%s WHERE user_id=%s AND id=%s",
                (json.dumps(existing), user_id, record_id)
            )
        self._commit()
        return existing

    def delete_record(self, user_id: str, collection: str, record_id: str) -> bool:
        """Delete a single record."""
        if collection not in COLLECTIONS:
            raise ValueError(f"Unknown collection: {collection}")

        if self._backend == "sqlite":
            self._execute(
                f"DELETE FROM {collection} WHERE user_id=? AND id=?",
                (user_id, record_id)
            )
        else:
            self._execute(
                f"DELETE FROM {collection} WHERE user_id=%s AND id=%s",
                (user_id, record_id)
            )
        self._commit()
        return True

    def delete_all_for_user(self, user_id: str, collection: str) -> int:
        """Wipe all records for a user in a collection. Returns count deleted."""
        if collection not in COLLECTIONS:
            raise ValueError(f"Unknown collection: {collection}")

        if self._backend == "sqlite":
            cur = self._execute(
                f"DELETE FROM {collection} WHERE user_id=?", (user_id,)
            )
            count = cur.rowcount
        else:
            cur = self._execute(
                f"DELETE FROM {collection} WHERE user_id=%s", (user_id,)
            )
            count = cur.rowcount
        self._commit()
        return count

    def count_records(self, user_id: str, collection: str) -> int:
        """Count records for a user in a collection."""
        if self._backend == "sqlite":
            cur = self._execute(
                f"SELECT COUNT(*) FROM {collection} WHERE user_id=?", (user_id,)
            )
        else:
            cur = self._execute(
                f"SELECT COUNT(*) FROM {collection} WHERE user_id=%s", (user_id,)
            )
        row = cur.fetchone()
        return row[0] if row else 0

    def get_kv(self, user_id: str, key: str) -> Optional[Any]:
        """
        Simple key-value store per user, backed by a 'kv' pseudo-collection.
        Stored in the 'savings' table with a special prefix to avoid collisions.
        """
        kv_id = f"__kv__{key}"
        record = self.get_record(user_id, "savings", kv_id)
        return record.get("value") if record else None

    def set_kv(self, user_id: str, key: str, value: Any) -> None:
        """Set a key-value pair per user."""
        kv_id = f"__kv__{key}"
        existing = self.get_record(user_id, "savings", kv_id)
        if existing:
            self.update_record(user_id, "savings", kv_id, {"value": value})
        else:
            # Direct insert with custom id
            created_at = datetime.utcnow().isoformat()
            record = {"id": kv_id, "created_at": created_at, "value": value, "__kv": True}
            if self._backend == "sqlite":
                self._execute(
                    "INSERT OR REPLACE INTO savings (id, user_id, data, created_at) VALUES (?,?,?,?)",
                    (kv_id, user_id, json.dumps(record), created_at)
                )
            else:
                self._execute(
                    "INSERT INTO savings (id, user_id, data, created_at) VALUES (%s,%s,%s,%s) "
                    "ON CONFLICT (id) DO UPDATE SET data=EXCLUDED.data",
                    (kv_id, user_id, json.dumps(record), created_at)
                )
            self._commit()
