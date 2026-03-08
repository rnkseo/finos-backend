"""
FinOS — Personal AI Financial Operating System
FastAPI Backend — main.py
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
from datetime import datetime, date
from database import Database
from ai_engine import AIEngine

# ─── APP SETUP ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinOS API",
    description="Personal AI Financial Operating System Backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Database()
ai = AIEngine()

# ─── USERS ────────────────────────────────────────────────────────────────────
USERS = {
    "naveen": {"password": "neevaN",  "display_name": "Naveen"},
    "sri":    {"password": "irS",     "display_name": "Sri"},
    "ramesh": {"password": "hsemaR",  "display_name": "Ramesh"},
    "raja":   {"password": "ajaR",    "display_name": "Raja"},
}

# ─── PYDANTIC MODELS ──────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class ExpenseCreate(BaseModel):
    description: str
    amount: float = Field(..., gt=0)
    category: str = "other"
    date: Optional[str] = None
    source: Optional[str] = "manual"

class InvestmentCreate(BaseModel):
    fund_name: str
    amount: float = Field(..., gt=0)
    type: str = "other"
    date: Optional[str] = None
    current_value: Optional[float] = None
    source: Optional[str] = "manual"

class AssetCreate(BaseModel):
    asset_type: str
    quantity: float
    unit: str = "g"
    purchase_price: float = Field(..., gt=0)
    current_value: Optional[float] = None
    description: Optional[str] = ""
    date: Optional[str] = None
    source: Optional[str] = "manual"

class RecurringCreate(BaseModel):
    name: str
    amount: float = Field(..., gt=0)
    frequency: str = "monthly"
    category: str = "other"
    start_date: Optional[str] = None
    source: Optional[str] = "manual"

class SavingCreate(BaseModel):
    description: str
    amount: float = Field(..., gt=0)
    account: str = "General Savings"
    date: Optional[str] = None
    source: Optional[str] = "manual"

class AIMessage(BaseModel):
    message: str
    user_id: Optional[str] = None
    context: Optional[dict] = {}

class UpdateValueRequest(BaseModel):
    current_value: float

# ─── AUTH HELPERS ─────────────────────────────────────────────────────────────
def get_current_user(x_user: str = Header(default="")):
    user_id = x_user.lower().strip()
    if not user_id or user_id not in USERS:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user_id

def get_today() -> str:
    return date.today().isoformat()

# ─── HEALTH CHECK ─────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "FinOS API is running", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# ─── AUTH ENDPOINTS ───────────────────────────────────────────────────────────
@app.post("/api/auth/login")
async def login(body: LoginRequest):
    user_id = body.username.strip().lower()
    user = USERS.get(user_id)
    if not user or user["password"] != body.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {
        "success": True,
        "user_id": user_id,
        "display_name": user["display_name"],
        "token": f"bearer_{user_id}_{int(datetime.utcnow().timestamp())}"
    }

# ─── EXPENSE ENDPOINTS ────────────────────────────────────────────────────────
@app.get("/api/expenses")
async def get_expenses(
    user_id: str = Depends(get_current_user),
    category: Optional[str] = None,
    month: Optional[str] = None,
    limit: int = 200
):
    expenses = db.get_records(user_id, "expenses", limit=limit)
    if category:
        expenses = [e for e in expenses if e.get("category") == category]
    if month:
        expenses = [e for e in expenses if (e.get("date") or "").startswith(month)]
    return {"data": expenses, "count": len(expenses)}

@app.post("/api/expenses")
async def create_expense(body: ExpenseCreate, user_id: str = Depends(get_current_user)):
    record = {
        "description": body.description,
        "amount": body.amount,
        "category": body.category,
        "date": body.date or get_today(),
        "source": body.source,
    }
    saved = db.add_record(user_id, "expenses", record)
    return {"success": True, "data": saved}

@app.delete("/api/expenses/{record_id}")
async def delete_expense(record_id: str, user_id: str = Depends(get_current_user)):
    db.delete_record(user_id, "expenses", record_id)
    return {"success": True}

@app.get("/api/expenses/summary")
async def expense_summary(user_id: str = Depends(get_current_user)):
    expenses = db.get_records(user_id, "expenses")
    today = datetime.now()
    current_month = f"{today.year}-{str(today.month).zfill(2)}"
    monthly = sum(e["amount"] for e in expenses if (e.get("date") or "").startswith(current_month))
    total = sum(e["amount"] for e in expenses)
    by_category: dict = {}
    for e in expenses:
        cat = e.get("category", "other")
        by_category[cat] = by_category.get(cat, 0) + e["amount"]
    return {
        "monthly": monthly,
        "total": total,
        "by_category": by_category,
        "count": len(expenses)
    }

# ─── INVESTMENT ENDPOINTS ─────────────────────────────────────────────────────
@app.get("/api/investments")
async def get_investments(user_id: str = Depends(get_current_user)):
    records = db.get_records(user_id, "investments")
    return {"data": records, "count": len(records)}

@app.post("/api/investments")
async def create_investment(body: InvestmentCreate, user_id: str = Depends(get_current_user)):
    record = {
        "fund_name": body.fund_name,
        "name": body.fund_name,
        "amount": body.amount,
        "type": body.type,
        "date": body.date or get_today(),
        "current_value": body.current_value or body.amount,
        "source": body.source,
    }
    saved = db.add_record(user_id, "investments", record)
    return {"success": True, "data": saved}

@app.put("/api/investments/{record_id}")
async def update_investment(
    record_id: str,
    body: UpdateValueRequest,
    user_id: str = Depends(get_current_user)
):
    updated = db.update_record(user_id, "investments", record_id, {"current_value": body.current_value})
    return {"success": True, "data": updated}

@app.delete("/api/investments/{record_id}")
async def delete_investment(record_id: str, user_id: str = Depends(get_current_user)):
    db.delete_record(user_id, "investments", record_id)
    return {"success": True}

@app.get("/api/investments/summary")
async def investment_summary(user_id: str = Depends(get_current_user)):
    records = db.get_records(user_id, "investments")
    total_invested = sum(r["amount"] for r in records)
    total_current = sum(r.get("current_value", r["amount"]) for r in records)
    pnl = total_current - total_invested
    pct = (pnl / total_invested * 100) if total_invested > 0 else 0.0
    return {
        "total_invested": total_invested,
        "total_current": total_current,
        "pnl": pnl,
        "pnl_pct": round(pct, 2),
        "count": len(records)
    }

# ─── ASSET ENDPOINTS ──────────────────────────────────────────────────────────
@app.get("/api/assets")
async def get_assets(user_id: str = Depends(get_current_user)):
    records = db.get_records(user_id, "assets")
    return {"data": records, "count": len(records)}

@app.post("/api/assets")
async def create_asset(body: AssetCreate, user_id: str = Depends(get_current_user)):
    record = {
        "asset_type": body.asset_type,
        "quantity": body.quantity,
        "unit": body.unit,
        "purchase_price": body.purchase_price,
        "current_value": body.current_value or body.purchase_price,
        "description": body.description,
        "date": body.date or get_today(),
        "source": body.source,
    }
    saved = db.add_record(user_id, "assets", record)
    return {"success": True, "data": saved}

@app.put("/api/assets/{record_id}")
async def update_asset(
    record_id: str,
    body: UpdateValueRequest,
    user_id: str = Depends(get_current_user)
):
    updated = db.update_record(user_id, "assets", record_id, {"current_value": body.current_value})
    return {"success": True, "data": updated}

@app.delete("/api/assets/{record_id}")
async def delete_asset(record_id: str, user_id: str = Depends(get_current_user)):
    db.delete_record(user_id, "assets", record_id)
    return {"success": True}

@app.get("/api/assets/summary")
async def asset_summary(user_id: str = Depends(get_current_user)):
    records = db.get_records(user_id, "assets")
    total_purchase = sum(r.get("purchase_price", 0) for r in records)
    total_current = sum(r.get("current_value", r.get("purchase_price", 0)) for r in records)
    by_type: dict = {}
    for r in records:
        t = r.get("asset_type", "other")
        by_type[t] = by_type.get(t, 0) + r.get("current_value", r.get("purchase_price", 0))
    return {
        "total_purchase": total_purchase,
        "total_current": total_current,
        "pnl": total_current - total_purchase,
        "by_type": by_type,
        "count": len(records)
    }

# ─── RECURRING ENDPOINTS ──────────────────────────────────────────────────────
@app.get("/api/recurring")
async def get_recurring(user_id: str = Depends(get_current_user)):
    records = db.get_records(user_id, "recurring")
    return {"data": records, "count": len(records)}

@app.post("/api/recurring")
async def create_recurring(body: RecurringCreate, user_id: str = Depends(get_current_user)):
    record = {
        "name": body.name,
        "amount": body.amount,
        "frequency": body.frequency,
        "category": body.category,
        "start_date": body.start_date or get_today(),
        "source": body.source,
        "active": True,
    }
    saved = db.add_record(user_id, "recurring", record)
    return {"success": True, "data": saved}

@app.delete("/api/recurring/{record_id}")
async def delete_recurring(record_id: str, user_id: str = Depends(get_current_user)):
    db.delete_record(user_id, "recurring", record_id)
    return {"success": True}

@app.get("/api/recurring/summary")
async def recurring_summary(user_id: str = Depends(get_current_user)):
    records = db.get_records(user_id, "recurring")
    monthly_total = sum(
        r["amount"] for r in records if r.get("frequency") == "monthly"
    )
    weekly_total = sum(
        r["amount"] * 4 for r in records if r.get("frequency") == "weekly"
    )
    yearly_total = sum(
        r["amount"] / 12 for r in records if r.get("frequency") == "yearly"
    )
    return {
        "monthly_outflow": round(monthly_total + weekly_total + yearly_total, 2),
        "active_count": len(records),
        "by_frequency": {
            "monthly": sum(r["amount"] for r in records if r.get("frequency") == "monthly"),
            "weekly": sum(r["amount"] for r in records if r.get("frequency") == "weekly"),
            "yearly": sum(r["amount"] for r in records if r.get("frequency") == "yearly"),
            "quarterly": sum(r["amount"] for r in records if r.get("frequency") == "quarterly"),
        }
    }

# ─── SAVINGS ENDPOINTS ────────────────────────────────────────────────────────
@app.get("/api/savings")
async def get_savings(user_id: str = Depends(get_current_user)):
    records = db.get_records(user_id, "savings")
    return {"data": records, "count": len(records)}

@app.post("/api/savings")
async def create_saving(body: SavingCreate, user_id: str = Depends(get_current_user)):
    record = {
        "description": body.description,
        "amount": body.amount,
        "account": body.account,
        "date": body.date or get_today(),
        "source": body.source,
    }
    saved = db.add_record(user_id, "savings", record)
    return {"success": True, "data": saved}

@app.delete("/api/savings/{record_id}")
async def delete_saving(record_id: str, user_id: str = Depends(get_current_user)):
    db.delete_record(user_id, "savings", record_id)
    return {"success": True}

@app.get("/api/savings/summary")
async def savings_summary(user_id: str = Depends(get_current_user)):
    records = db.get_records(user_id, "savings")
    total = sum(r["amount"] for r in records)
    by_account: dict = {}
    for r in records:
        acc = r.get("account", "General")
        by_account[acc] = by_account.get(acc, 0) + r["amount"]
    return {"total": total, "by_account": by_account, "count": len(records)}

# ─── AI ENDPOINTS ─────────────────────────────────────────────────────────────
@app.post("/api/ai/parse")
async def ai_parse(body: AIMessage, user_id: str = Depends(get_current_user)):
    """
    Parse a natural language financial message using the AI engine.
    Returns structured intent + data + response text.
    """
    all_data = {
        "expenses":    db.get_records(user_id, "expenses",    limit=50),
        "investments": db.get_records(user_id, "investments", limit=50),
        "assets":      db.get_records(user_id, "assets",      limit=50),
        "recurring":   db.get_records(user_id, "recurring",   limit=50),
        "savings":     db.get_records(user_id, "savings",     limit=50),
    }

    result = await ai.parse_message(
        message=body.message,
        user_id=user_id,
        user_data=all_data,
        context=body.context or {}
    )

    # Auto-persist any records the AI wants to create
    if result.get("intent") == "add_expense" and result.get("data", {}).get("amount", 0) > 0:
        d = result["data"]
        saved = db.add_record(user_id, "expenses", {
            "description": d.get("description", body.message),
            "amount":      d["amount"],
            "category":    d.get("category", "other"),
            "date":        d.get("date", get_today()),
            "source":      "ai",
        })
        result["saved_record"] = saved

    elif result.get("intent") == "add_investment" and result.get("data", {}).get("amount", 0) > 0:
        d = result["data"]
        saved = db.add_record(user_id, "investments", {
            "fund_name":     d.get("fund_name", "Investment"),
            "name":          d.get("fund_name", "Investment"),
            "amount":        d["amount"],
            "type":          d.get("investment_type", "other"),
            "date":          d.get("date", get_today()),
            "current_value": d["amount"],
            "source":        "ai",
        })
        result["saved_record"] = saved

    elif result.get("intent") == "add_asset" and result.get("data", {}).get("amount", 0) > 0:
        d = result["data"]
        saved = db.add_record(user_id, "assets", {
            "asset_type":    d.get("asset_type", "other"),
            "quantity":      d.get("quantity", 1),
            "unit":          "g",
            "purchase_price": d["amount"],
            "current_value": d["amount"],
            "description":   d.get("description", f"{d.get('asset_type','asset')} purchase"),
            "date":          d.get("date", get_today()),
            "source":        "ai",
        })
        result["saved_record"] = saved

    elif result.get("intent") == "add_recurring" and result.get("data", {}).get("amount", 0) > 0:
        d = result["data"]
        saved = db.add_record(user_id, "recurring", {
            "name":       d.get("payment_name", body.message),
            "amount":     d["amount"],
            "frequency":  d.get("frequency", "monthly"),
            "category":   d.get("category", "other"),
            "start_date": d.get("date", get_today()),
            "active":     True,
            "source":     "ai",
        })
        result["saved_record"] = saved

    elif result.get("intent") == "add_saving" and result.get("data", {}).get("amount", 0) > 0:
        d = result["data"]
        saved = db.add_record(user_id, "savings", {
            "description": d.get("description", "Savings"),
            "amount":      d["amount"],
            "account":     d.get("account", "General Savings"),
            "date":        d.get("date", get_today()),
            "source":      "ai",
        })
        result["saved_record"] = saved

    return result

@app.get("/api/ai/insights")
async def ai_insights(user_id: str = Depends(get_current_user)):
    """Generate AI-powered financial insights for the user."""
    all_data = {
        "expenses":    db.get_records(user_id, "expenses",    limit=200),
        "investments": db.get_records(user_id, "investments", limit=50),
        "assets":      db.get_records(user_id, "assets",      limit=50),
        "savings":     db.get_records(user_id, "savings",     limit=100),
        "recurring":   db.get_records(user_id, "recurring",   limit=50),
    }
    insights = ai.generate_insights(all_data)
    return {"insights": insights}

@app.get("/api/ai/health-score")
async def health_score(user_id: str = Depends(get_current_user)):
    """Compute the AI financial health score."""
    all_data = {
        "expenses":    db.get_records(user_id, "expenses",    limit=200),
        "investments": db.get_records(user_id, "investments", limit=50),
        "assets":      db.get_records(user_id, "assets",      limit=50),
        "savings":     db.get_records(user_id, "savings",     limit=100),
        "recurring":   db.get_records(user_id, "recurring",   limit=50),
    }
    score_data = ai.compute_health_score(all_data)
    return score_data

@app.get("/api/ai/predictions")
async def ai_predictions(user_id: str = Depends(get_current_user)):
    """Generate 6-month financial predictions."""
    all_data = {
        "expenses":    db.get_records(user_id, "expenses",    limit=200),
        "investments": db.get_records(user_id, "investments", limit=50),
        "savings":     db.get_records(user_id, "savings",     limit=100),
        "recurring":   db.get_records(user_id, "recurring",   limit=50),
    }
    predictions = ai.generate_predictions(all_data)
    return predictions

# ─── ANALYTICS ENDPOINTS ──────────────────────────────────────────────────────
@app.get("/api/analytics/overview")
async def analytics_overview(user_id: str = Depends(get_current_user)):
    """Full analytics overview for the dashboard."""
    expenses    = db.get_records(user_id, "expenses",    limit=500)
    investments = db.get_records(user_id, "investments", limit=200)
    assets      = db.get_records(user_id, "assets",      limit=200)
    savings     = db.get_records(user_id, "savings",     limit=200)
    recurring   = db.get_records(user_id, "recurring",   limit=50)

    today = datetime.now()
    current_month = f"{today.year}-{str(today.month).zfill(2)}"
    monthly_expenses = sum(e["amount"] for e in expenses if (e.get("date") or "").startswith(current_month))
    total_expenses   = sum(e["amount"] for e in expenses)
    total_invested   = sum(i["amount"] for i in investments)
    total_current    = sum(i.get("current_value", i["amount"]) for i in investments)
    total_savings    = sum(s["amount"] for s in savings)
    assets_value     = sum(a.get("current_value", a.get("purchase_price", 0)) for a in assets)
    monthly_recurring = sum(
        r["amount"] for r in recurring if r.get("frequency") == "monthly"
    )

    by_category: dict = {}
    for e in expenses:
        cat = e.get("category", "other")
        by_category[cat] = by_category.get(cat, 0) + e["amount"]

    net_worth = total_current + total_savings + assets_value

    return {
        "monthly_expenses":   monthly_expenses,
        "total_expenses":     total_expenses,
        "total_invested":     total_invested,
        "portfolio_value":    total_current,
        "investment_pnl":     total_current - total_invested,
        "total_savings":      total_savings,
        "assets_value":       assets_value,
        "net_worth":          net_worth,
        "monthly_recurring":  monthly_recurring,
        "by_category":        by_category,
        "transaction_count":  len(expenses),
    }

@app.get("/api/analytics/monthly-trend")
async def monthly_trend(user_id: str = Depends(get_current_user), months: int = 6):
    """Expense trend for the last N months."""
    expenses = db.get_records(user_id, "expenses", limit=500)
    today = datetime.now()
    result = []
    for i in range(months - 1, -1, -1):
        yr = today.year if today.month - i > 0 else today.year - 1
        mo = (today.month - i - 1) % 12 + 1
        key = f"{yr}-{str(mo).zfill(2)}"
        label = datetime(yr, mo, 1).strftime("%b %y")
        total = sum(e["amount"] for e in expenses if (e.get("date") or "").startswith(key))
        result.append({"month": label, "key": key, "total": total})
    return {"data": result}

# ─── SERVER ENTRY POINT ───────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
