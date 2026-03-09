"""AstraOS — Database Layer (SQLite / PostgreSQL)"""
import os, json, time, random, string, sqlite3
from typing import List, Dict, Optional
from datetime import datetime

DATABASE_URL = os.environ.get("DATABASE_URL", "")
USE_PG = DATABASE_URL.startswith(("postgresql","postgres"))
if USE_PG:
    try: import psycopg2, psycopg2.extras
    except: USE_PG = False

SQLITE_PATH = os.environ.get("SQLITE_PATH", "./astraos.db")
COLLECTIONS = ["expenses","investments","assets","recurring","debts","savings"]

def _gen_id():
    return f"{int(time.time()*1000)}_{(''.join(random.choices(string.ascii_lowercase+string.digits,k=6)))}"

class Database:
    def __init__(self):
        if USE_PG:
            self._pg = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
            self._pg.autocommit = True
            self._backend = "postgres"
        else:
            self._sq = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
            self._sq.row_factory = sqlite3.Row
            self._backend = "sqlite"
        self._init_schema()

    def _init_schema(self):
        for col in COLLECTIONS:
            if self._backend == "sqlite":
                self._sq.execute(f"CREATE TABLE IF NOT EXISTS {col} (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, data TEXT NOT NULL, created_at TEXT NOT NULL)")
                self._sq.execute(f"CREATE INDEX IF NOT EXISTS idx_{col}_user ON {col}(user_id)")
                self._sq.commit()
            else:
                cur = self._pg.cursor()
                cur.execute(f"CREATE TABLE IF NOT EXISTS {col} (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, data JSONB NOT NULL, created_at TEXT NOT NULL)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{col}_user ON {col}(user_id)")

    def _exec(self, sql, params=()):
        if self._backend == "sqlite": return self._sq.execute(sql, params)
        else: cur=self._pg.cursor(); cur.execute(sql,params); return cur

    def _commit(self):
        if self._backend == "sqlite": self._sq.commit()

    def add_record(self, user_id, collection, record):
        if collection not in COLLECTIONS: raise ValueError(f"Unknown: {collection}")
        rid = _gen_id(); now = datetime.utcnow().isoformat()
        record.update({"id": rid, "created_at": now})
        p = "?" if self._backend=="sqlite" else "%s"
        self._exec(f"INSERT INTO {collection} (id,user_id,data,created_at) VALUES ({p},{p},{p},{p})",
                   (rid, user_id, json.dumps(record), now))
        self._commit(); return record

    def get_records(self, user_id, collection, limit=500):
        if collection not in COLLECTIONS: raise ValueError()
        p = "?" if self._backend=="sqlite" else "%s"
        cur = self._exec(f"SELECT data FROM {collection} WHERE user_id={p} ORDER BY created_at DESC LIMIT {p}", (user_id, limit))
        rows = cur.fetchall()
        return [json.loads(r[0]) if self._backend=="sqlite" else dict(r["data"]) for r in rows]

    def get_record(self, user_id, collection, rid):
        p = "?" if self._backend=="sqlite" else "%s"
        cur = self._exec(f"SELECT data FROM {collection} WHERE user_id={p} AND id={p}", (user_id, rid))
        row = cur.fetchone()
        if not row: return None
        return json.loads(row[0]) if self._backend=="sqlite" else dict(row["data"])

    def update_record(self, user_id, collection, rid, updates):
        existing = self.get_record(user_id, collection, rid)
        if not existing: return None
        existing.update(updates); existing["updated_at"] = datetime.utcnow().isoformat()
        p = "?" if self._backend=="sqlite" else "%s"
        self._exec(f"UPDATE {collection} SET data={p} WHERE user_id={p} AND id={p}", (json.dumps(existing), user_id, rid))
        self._commit(); return existing

    def delete_record(self, user_id, collection, rid):
        p = "?" if self._backend=="sqlite" else "%s"
        self._exec(f"DELETE FROM {collection} WHERE user_id={p} AND id={p}", (user_id, rid))
        self._commit(); return True
