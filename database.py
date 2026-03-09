"""
AstraOS — Database Layer v3.0
Full financial schema: transactions, portfolio, SIPs, market data, summaries, insights
Supports SQLite (local) and PostgreSQL (production)
"""
import os, json, time, random, string, sqlite3
from typing import List, Dict, Optional, Any
from datetime import datetime, date, timedelta

DATABASE_URL = os.environ.get("DATABASE_URL", "")
USE_PG = DATABASE_URL.startswith(("postgresql", "postgres"))
if USE_PG:
    try:
        import psycopg2, psycopg2.extras
    except ImportError:
        USE_PG = False

SQLITE_PATH = os.environ.get("SQLITE_PATH", "./astraos.db")

VALID_USERS = {"naveen", "sri", "ramesh", "raja"}

def _gen_id(prefix=""):
    ts = int(time.time() * 1000)
    rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"{prefix}{ts}_{rand}"


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

    # ── Schema ────────────────────────────────────────────────────────────────
    def _init_schema(self):
        tables = [
            # Core financial records
            ("transactions", """
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                type TEXT NOT NULL,
                category TEXT,
                description TEXT,
                amount REAL NOT NULL,
                date TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            """),
            ("investments", """
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                fund_name TEXT NOT NULL,
                investment_type TEXT NOT NULL,
                invested_amount REAL NOT NULL,
                current_value REAL,
                units REAL DEFAULT 0,
                nav REAL DEFAULT 0,
                start_date TEXT,
                last_updated TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            """),
            ("sip_schedules", """
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                investment_id TEXT,
                fund_name TEXT NOT NULL,
                amount REAL NOT NULL,
                frequency TEXT DEFAULT 'monthly',
                start_date TEXT NOT NULL,
                next_due TEXT,
                is_active INTEGER DEFAULT 1,
                total_paid REAL DEFAULT 0,
                installments_paid INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            """),
            ("assets", """
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                description TEXT,
                quantity REAL DEFAULT 0,
                purchase_price REAL NOT NULL,
                current_value REAL,
                purchase_date TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            """),
            ("recurring_payments", """
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                category TEXT,
                amount REAL NOT NULL,
                frequency TEXT DEFAULT 'monthly',
                next_due TEXT,
                is_active INTEGER DEFAULT 1,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            """),
            ("savings_goals", """
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                goal_name TEXT NOT NULL,
                target_amount REAL NOT NULL,
                current_amount REAL DEFAULT 0,
                target_date TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            """),
            ("debts", """
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                debt_type TEXT,
                creditor TEXT,
                principal REAL NOT NULL,
                remaining REAL,
                interest_rate REAL DEFAULT 0,
                emi REAL DEFAULT 0,
                due_date TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            """),
            ("market_data", """
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                asset_class TEXT NOT NULL,
                price REAL NOT NULL,
                currency TEXT DEFAULT 'INR',
                fetched_at TEXT NOT NULL,
                source TEXT DEFAULT 'api'
            """),
            ("financial_summary", """
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                period TEXT NOT NULL,
                income REAL DEFAULT 0,
                expenses REAL DEFAULT 0,
                investments REAL DEFAULT 0,
                savings REAL DEFAULT 0,
                net_worth REAL DEFAULT 0,
                cash_balance REAL DEFAULT 0,
                updated_at TEXT NOT NULL
            """),
            ("ai_insights", """
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                insight_type TEXT,
                content TEXT NOT NULL,
                priority INTEGER DEFAULT 1,
                is_read INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            """),
            ("notifications", """
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                type TEXT,
                title TEXT,
                message TEXT,
                is_read INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            """),
            ("income_records", """
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                source TEXT NOT NULL,
                amount REAL NOT NULL,
                date TEXT NOT NULL,
                is_recurring INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            """),
        ]

        for table_name, cols in tables:
            if self._backend == "sqlite":
                self._sq.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({cols})")
                self._sq.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table_name}_user ON {table_name}(user_id)"
                    if "user_id" in cols else f"SELECT 1"
                )
            else:
                pg_cols = cols.replace("INTEGER", "INT").replace("REAL", "FLOAT")
                cur = self._pg.cursor()
                cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({pg_cols})")
                if "user_id" in cols:
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_user ON {table_name}(user_id)"
                    )
        if self._backend == "sqlite":
            self._sq.commit()

    # ── Internals ─────────────────────────────────────────────────────────────
    def _p(self):
        return "?" if self._backend == "sqlite" else "%s"

    def _exec(self, sql, params=()):
        if self._backend == "sqlite":
            return self._sq.execute(sql, params)
        else:
            cur = self._pg.cursor()
            cur.execute(sql, params)
            return cur

    def _commit(self):
        if self._backend == "sqlite":
            self._sq.commit()

    def _row_to_dict(self, row) -> dict:
        if row is None:
            return None
        if self._backend == "sqlite":
            return dict(row)
        return dict(row)

    def _rows_to_dicts(self, rows) -> list:
        return [self._row_to_dict(r) for r in rows]

    def _decode(self, row: dict) -> dict:
        """Decode metadata JSON fields."""
        if row and "metadata" in row and isinstance(row["metadata"], str):
            try:
                row["metadata"] = json.loads(row["metadata"])
            except Exception:
                row["metadata"] = {}
        return row

    # ── Generic CRUD ──────────────────────────────────────────────────────────
    def _insert(self, table: str, record: dict) -> dict:
        keys = list(record.keys())
        vals = [record[k] for k in keys]
        p = self._p()
        placeholders = ",".join([p] * len(keys))
        cols = ",".join(keys)
        self._exec(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", vals)
        self._commit()
        return record

    def _update_field(self, table: str, record_id: str, user_id: str, field: str, value):
        p = self._p()
        self._exec(
            f"UPDATE {table} SET {field}={p} WHERE id={p} AND user_id={p}",
            (value, record_id, user_id)
        )
        self._commit()

    # ── Transactions ──────────────────────────────────────────────────────────
    def add_transaction(self, user_id: str, tx_type: str, category: str,
                        description: str, amount: float, date_str: str,
                        metadata: dict = None) -> dict:
        record = {
            "id": _gen_id("tx_"),
            "user_id": user_id,
            "type": tx_type,
            "category": category or "other",
            "description": description or "",
            "amount": amount,
            "date": date_str or date.today().isoformat(),
            "metadata": json.dumps(metadata or {}),
            "created_at": datetime.utcnow().isoformat(),
        }
        return self._insert("transactions", record)

    def get_transactions(self, user_id: str, tx_type: str = None,
                         limit: int = 200, since_date: str = None) -> list:
        p = self._p()
        conditions = [f"user_id={p}"]
        params = [user_id]
        if tx_type:
            conditions.append(f"type={p}")
            params.append(tx_type)
        if since_date:
            conditions.append(f"date>={p}")
            params.append(since_date)
        where = " AND ".join(conditions)
        params.append(limit)
        cur = self._exec(
            f"SELECT * FROM transactions WHERE {where} ORDER BY date DESC, created_at DESC LIMIT {p}",
            params
        )
        return [self._decode(self._row_to_dict(r)) for r in cur.fetchall()]

    def delete_transaction(self, user_id: str, tx_id: str) -> bool:
        p = self._p()
        self._exec(f"DELETE FROM transactions WHERE id={p} AND user_id={p}", (tx_id, user_id))
        self._commit()
        return True

    # ── Investments ────────────────────────────────────────────────────────────
    def add_investment(self, user_id: str, fund_name: str, inv_type: str,
                       amount: float, units: float = 0, nav: float = 0,
                       start_date: str = None, metadata: dict = None) -> dict:
        now = date.today().isoformat()
        record = {
            "id": _gen_id("inv_"),
            "user_id": user_id,
            "fund_name": fund_name,
            "investment_type": inv_type,
            "invested_amount": amount,
            "current_value": amount,
            "units": units,
            "nav": nav,
            "start_date": start_date or now,
            "last_updated": now,
            "metadata": json.dumps(metadata or {}),
            "created_at": datetime.utcnow().isoformat(),
        }
        return self._insert("investments", record)

    def get_investments(self, user_id: str) -> list:
        p = self._p()
        cur = self._exec(
            f"SELECT * FROM investments WHERE user_id={p} ORDER BY created_at DESC", (user_id,)
        )
        return [self._decode(self._row_to_dict(r)) for r in cur.fetchall()]

    def update_investment_value(self, user_id: str, inv_id: str,
                                 current_value: float, nav: float = None):
        p = self._p()
        self._exec(
            f"UPDATE investments SET current_value={p}, last_updated={p} WHERE id={p} AND user_id={p}",
            (current_value, date.today().isoformat(), inv_id, user_id)
        )
        if nav:
            self._exec(f"UPDATE investments SET nav={p} WHERE id={p} AND user_id={p}",
                       (nav, inv_id, user_id))
        self._commit()

    def delete_investment(self, user_id: str, inv_id: str) -> bool:
        p = self._p()
        self._exec(f"DELETE FROM investments WHERE id={p} AND user_id={p}", (inv_id, user_id))
        self._commit()
        return True

    # ── SIP Schedules ─────────────────────────────────────────────────────────
    def add_sip(self, user_id: str, fund_name: str, amount: float,
                frequency: str = "monthly", start_date: str = None,
                investment_id: str = None, months_back: int = 0) -> dict:
        today = date.today()
        start = start_date or today.isoformat()
        # Calculate next due from start_date
        if start_date:
            sd = datetime.fromisoformat(start_date).date()
        else:
            sd = today
        # Next due is 1 month from today if starting now
        next_due = (today.replace(day=sd.day) if sd.day <= 28 else today).isoformat()
        total_paid = amount * months_back
        record = {
            "id": _gen_id("sip_"),
            "user_id": user_id,
            "investment_id": investment_id or "",
            "fund_name": fund_name,
            "amount": amount,
            "frequency": frequency,
            "start_date": start,
            "next_due": next_due,
            "is_active": 1,
            "total_paid": total_paid,
            "installments_paid": months_back,
            "metadata": json.dumps({"months_back": months_back}),
            "created_at": datetime.utcnow().isoformat(),
        }
        return self._insert("sip_schedules", record)

    def get_sips(self, user_id: str, active_only: bool = False) -> list:
        p = self._p()
        cond = f"user_id={p}"
        params = [user_id]
        if active_only:
            cond += f" AND is_active={p}"
            params.append(1)
        cur = self._exec(f"SELECT * FROM sip_schedules WHERE {cond} ORDER BY created_at DESC", params)
        return [self._decode(self._row_to_dict(r)) for r in cur.fetchall()]

    def mark_sip_paid(self, user_id: str, sip_id: str) -> dict:
        p = self._p()
        cur = self._exec(f"SELECT * FROM sip_schedules WHERE id={p} AND user_id={p}", (sip_id, user_id))
        sip = self._row_to_dict(cur.fetchone())
        if not sip:
            return None
        new_total = (sip["total_paid"] or 0) + sip["amount"]
        new_count = (sip["installments_paid"] or 0) + 1
        self._exec(
            f"UPDATE sip_schedules SET total_paid={p}, installments_paid={p} WHERE id={p} AND user_id={p}",
            (new_total, new_count, sip_id, user_id)
        )
        self._commit()
        return sip

    # ── Assets ────────────────────────────────────────────────────────────────
    def add_asset(self, user_id: str, asset_type: str, description: str,
                  quantity: float, purchase_price: float, current_value: float = None,
                  purchase_date: str = None, metadata: dict = None) -> dict:
        record = {
            "id": _gen_id("ast_"),
            "user_id": user_id,
            "asset_type": asset_type,
            "description": description or f"{asset_type} asset",
            "quantity": quantity,
            "purchase_price": purchase_price,
            "current_value": current_value or purchase_price,
            "purchase_date": purchase_date or date.today().isoformat(),
            "metadata": json.dumps(metadata or {}),
            "created_at": datetime.utcnow().isoformat(),
        }
        return self._insert("assets", record)

    def get_assets(self, user_id: str) -> list:
        p = self._p()
        cur = self._exec(f"SELECT * FROM assets WHERE user_id={p} ORDER BY created_at DESC", (user_id,))
        return [self._decode(self._row_to_dict(r)) for r in cur.fetchall()]

    def update_asset_value(self, user_id: str, asset_id: str, current_value: float):
        p = self._p()
        self._exec(
            f"UPDATE assets SET current_value={p} WHERE id={p} AND user_id={p}",
            (current_value, asset_id, user_id)
        )
        self._commit()

    def update_asset_values_by_type(self, user_id: str, asset_type: str,
                                     price_per_unit: float):
        """Update current value for all assets of given type based on price_per_unit * quantity."""
        assets = self.get_assets(user_id)
        for a in assets:
            if a["asset_type"] == asset_type and a.get("quantity", 0) > 0:
                new_val = a["quantity"] * price_per_unit
                self.update_asset_value(user_id, a["id"], new_val)

    def delete_asset(self, user_id: str, asset_id: str) -> bool:
        p = self._p()
        self._exec(f"DELETE FROM assets WHERE id={p} AND user_id={p}", (asset_id, user_id))
        self._commit()
        return True

    # ── Recurring Payments ────────────────────────────────────────────────────
    def add_recurring(self, user_id: str, name: str, category: str,
                      amount: float, frequency: str = "monthly",
                      next_due: str = None, metadata: dict = None) -> dict:
        if not next_due:
            today = date.today()
            if frequency == "monthly":
                next_mo = today.replace(day=1) + timedelta(days=32)
                next_due = next_mo.replace(day=today.day).isoformat()
            elif frequency == "weekly":
                next_due = (today + timedelta(days=7)).isoformat()
            else:
                next_due = today.isoformat()
        record = {
            "id": _gen_id("rec_"),
            "user_id": user_id,
            "name": name,
            "category": category or "other",
            "amount": amount,
            "frequency": frequency,
            "next_due": next_due,
            "is_active": 1,
            "metadata": json.dumps(metadata or {}),
            "created_at": datetime.utcnow().isoformat(),
        }
        return self._insert("recurring_payments", record)

    def get_recurring(self, user_id: str, active_only: bool = True) -> list:
        p = self._p()
        cond = f"user_id={p}"
        params = [user_id]
        if active_only:
            cond += f" AND is_active={p}"
            params.append(1)
        cur = self._exec(
            f"SELECT * FROM recurring_payments WHERE {cond} ORDER BY next_due ASC", params
        )
        return [self._decode(self._row_to_dict(r)) for r in cur.fetchall()]

    def delete_recurring(self, user_id: str, rec_id: str) -> bool:
        p = self._p()
        self._exec(f"DELETE FROM recurring_payments WHERE id={p} AND user_id={p}", (rec_id, user_id))
        self._commit()
        return True

    # ── Savings Goals ─────────────────────────────────────────────────────────
    def add_savings_goal(self, user_id: str, goal_name: str, target_amount: float,
                         current_amount: float = 0, target_date: str = None) -> dict:
        record = {
            "id": _gen_id("sav_"),
            "user_id": user_id,
            "goal_name": goal_name,
            "target_amount": target_amount,
            "current_amount": current_amount,
            "target_date": target_date or "",
            "metadata": json.dumps({}),
            "created_at": datetime.utcnow().isoformat(),
        }
        return self._insert("savings_goals", record)

    def get_savings_goals(self, user_id: str) -> list:
        p = self._p()
        cur = self._exec(f"SELECT * FROM savings_goals WHERE user_id={p}", (user_id,))
        return [self._decode(self._row_to_dict(r)) for r in cur.fetchall()]

    def update_savings_goal(self, user_id: str, goal_id: str,
                             amount_to_add: float = 0, new_total: float = None):
        p = self._p()
        cur = self._exec(f"SELECT * FROM savings_goals WHERE id={p} AND user_id={p}", (goal_id, user_id))
        goal = self._row_to_dict(cur.fetchone())
        if not goal:
            return None
        new_amount = new_total if new_total is not None else (goal["current_amount"] + amount_to_add)
        self._exec(
            f"UPDATE savings_goals SET current_amount={p} WHERE id={p} AND user_id={p}",
            (new_amount, goal_id, user_id)
        )
        self._commit()
        return goal

    def delete_savings_goal(self, user_id: str, goal_id: str) -> bool:
        p = self._p()
        self._exec(f"DELETE FROM savings_goals WHERE id={p} AND user_id={p}", (goal_id, user_id))
        self._commit()
        return True

    # ── Debts ─────────────────────────────────────────────────────────────────
    def add_debt(self, user_id: str, debt_type: str, creditor: str,
                 principal: float, interest_rate: float = 0, emi: float = 0,
                 due_date: str = None) -> dict:
        record = {
            "id": _gen_id("dbt_"),
            "user_id": user_id,
            "debt_type": debt_type or "personal",
            "creditor": creditor or "Unknown",
            "principal": principal,
            "remaining": principal,
            "interest_rate": interest_rate,
            "emi": emi,
            "due_date": due_date or "",
            "metadata": json.dumps({}),
            "created_at": datetime.utcnow().isoformat(),
        }
        return self._insert("debts", record)

    def get_debts(self, user_id: str) -> list:
        p = self._p()
        cur = self._exec(f"SELECT * FROM debts WHERE user_id={p}", (user_id,))
        return [self._decode(self._row_to_dict(r)) for r in cur.fetchall()]

    # ── Income Records ─────────────────────────────────────────────────────────
    def add_income(self, user_id: str, source: str, amount: float,
                   date_str: str = None, is_recurring: bool = False,
                   metadata: dict = None) -> dict:
        record = {
            "id": _gen_id("inc_"),
            "user_id": user_id,
            "source": source,
            "amount": amount,
            "date": date_str or date.today().isoformat(),
            "is_recurring": 1 if is_recurring else 0,
            "metadata": json.dumps(metadata or {}),
            "created_at": datetime.utcnow().isoformat(),
        }
        return self._insert("income_records", record)

    def get_income(self, user_id: str, limit: int = 100) -> list:
        p = self._p()
        cur = self._exec(
            f"SELECT * FROM income_records WHERE user_id={p} ORDER BY date DESC LIMIT {p}",
            (user_id, limit)
        )
        return [self._decode(self._row_to_dict(r)) for r in cur.fetchall()]

    # ── Market Data ────────────────────────────────────────────────────────────
    def upsert_market_price(self, symbol: str, asset_class: str, price: float,
                             currency: str = "INR", source: str = "api"):
        p = self._p()
        now = datetime.utcnow().isoformat()
        # Delete old and insert new
        self._exec(f"DELETE FROM market_data WHERE symbol={p}", (symbol,))
        self._insert("market_data", {
            "id": _gen_id("mkt_"),
            "symbol": symbol,
            "asset_class": asset_class,
            "price": price,
            "currency": currency,
            "fetched_at": now,
            "source": source,
        })

    def get_market_price(self, symbol: str) -> Optional[dict]:
        p = self._p()
        cur = self._exec(f"SELECT * FROM market_data WHERE symbol={p}", (symbol,))
        row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def get_all_market_data(self) -> list:
        cur = self._exec("SELECT * FROM market_data ORDER BY fetched_at DESC", ())
        return [self._row_to_dict(r) for r in cur.fetchall()]

    # ── AI Insights ────────────────────────────────────────────────────────────
    def add_insight(self, user_id: str, insight_type: str, content: str, priority: int = 1):
        self._insert("ai_insights", {
            "id": _gen_id("ins_"),
            "user_id": user_id,
            "insight_type": insight_type,
            "content": content,
            "priority": priority,
            "is_read": 0,
            "created_at": datetime.utcnow().isoformat(),
        })

    def get_insights(self, user_id: str, limit: int = 10) -> list:
        p = self._p()
        cur = self._exec(
            f"SELECT * FROM ai_insights WHERE user_id={p} ORDER BY priority DESC, created_at DESC LIMIT {p}",
            (user_id, limit)
        )
        return [self._row_to_dict(r) for r in cur.fetchall()]

    def mark_insight_read(self, user_id: str, insight_id: str):
        p = self._p()
        self._exec(f"UPDATE ai_insights SET is_read={p} WHERE id={p} AND user_id={p}", (1, insight_id, user_id))
        self._commit()

    # ── Notifications ─────────────────────────────────────────────────────────
    def add_notification(self, user_id: str, notif_type: str, title: str, message: str):
        self._insert("notifications", {
            "id": _gen_id("ntf_"),
            "user_id": user_id,
            "type": notif_type,
            "title": title,
            "message": message,
            "is_read": 0,
            "created_at": datetime.utcnow().isoformat(),
        })

    def get_notifications(self, user_id: str, unread_only: bool = False) -> list:
        p = self._p()
        cond = f"user_id={p}"
        params = [user_id]
        if unread_only:
            cond += f" AND is_read={p}"
            params.append(0)
        cur = self._exec(
            f"SELECT * FROM notifications WHERE {cond} ORDER BY created_at DESC LIMIT 50", params
        )
        return [self._row_to_dict(r) for r in cur.fetchall()]

    def mark_notification_read(self, user_id: str, notif_id: str):
        p = self._p()
        self._exec(f"UPDATE notifications SET is_read={p} WHERE id={p} AND user_id={p}", (1, notif_id, user_id))
        self._commit()

    # ── Financial Summary (computed) ──────────────────────────────────────────
    def compute_financial_summary(self, user_id: str) -> dict:
        today = date.today()
        current_month = f"{today.year}-{str(today.month).zfill(2)}"

        # Transactions for current month
        txs = self.get_transactions(user_id, since_date=f"{current_month}-01")
        month_expenses = sum(t["amount"] for t in txs if t["type"] == "expense")
        month_income = sum(t["amount"] for t in txs if t["type"] == "income")

        # All income
        all_income = self.get_income(user_id)
        total_income = sum(i["amount"] for i in all_income)

        # All expenses
        all_expenses = self.get_transactions(user_id, tx_type="expense")
        total_expenses = sum(t["amount"] for t in all_expenses)

        # Investments
        investments = self.get_investments(user_id)
        total_invested = sum(i["invested_amount"] for i in investments)
        total_portfolio = sum(i.get("current_value") or i["invested_amount"] for i in investments)

        # SIPs
        sips = self.get_sips(user_id)
        total_sip_paid = sum(s.get("total_paid", 0) for s in sips)

        # Assets
        assets = self.get_assets(user_id)
        total_asset_value = sum(a.get("current_value") or a["purchase_price"] for a in assets)
        total_asset_cost = sum(a["purchase_price"] for a in assets)

        # Savings
        savings = self.get_savings_goals(user_id)
        total_savings = sum(s.get("current_amount", 0) for s in savings)

        # Debts
        debts = self.get_debts(user_id)
        total_debt = sum(d.get("remaining") or d["principal"] for d in debts)

        # Net worth = portfolio + assets + savings - debts
        net_worth = total_portfolio + total_asset_value + total_savings - total_debt

        # Cash balance estimate: income - expenses - invested
        cash_balance = max(0, total_income - total_expenses - total_invested - total_sip_paid)

        return {
            "user_id": user_id,
            "period": current_month,
            "total_income": total_income,
            "total_expenses": total_expenses,
            "month_income": month_income,
            "month_expenses": month_expenses,
            "total_invested": total_invested,
            "total_portfolio_value": total_portfolio,
            "portfolio_gain": total_portfolio - total_invested,
            "portfolio_gain_pct": ((total_portfolio - total_invested) / total_invested * 100) if total_invested > 0 else 0,
            "total_asset_value": total_asset_value,
            "asset_gain": total_asset_value - total_asset_cost,
            "total_savings": total_savings,
            "total_debt": total_debt,
            "net_worth": net_worth,
            "cash_balance": cash_balance,
            "sip_count": len([s for s in sips if s.get("is_active")]),
            "active_recurring": len(self.get_recurring(user_id, active_only=True)),
        }

    # ── Full user data (for AI context) ───────────────────────────────────────
    def get_full_user_data(self, user_id: str) -> dict:
        return {
            "expenses": self.get_transactions(user_id, tx_type="expense", limit=100),
            "income": self.get_income(user_id, limit=50),
            "investments": self.get_investments(user_id),
            "sips": self.get_sips(user_id),
            "assets": self.get_assets(user_id),
            "recurring": self.get_recurring(user_id),
            "savings": self.get_savings_goals(user_id),
            "debts": self.get_debts(user_id),
            "summary": self.compute_financial_summary(user_id),
            "market_data": self.get_all_market_data(),
            "insights": self.get_insights(user_id, limit=5),
            "notifications": self.get_notifications(user_id, unread_only=True),
        }
