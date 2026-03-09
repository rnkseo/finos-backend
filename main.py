"""
AstraOS — FastAPI Backend v3.0
Financial action execution engine + AI orchestration + market data
"""
import os, json, asyncio
from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime, date, timedelta

from database import Database, VALID_USERS
from ai_engine import AIEngine
from market_data import get_market_service

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="AstraOS API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Database()
ai = AIEngine()
market = get_market_service(db)

# ── Auth ──────────────────────────────────────────────────────────────────────
VALID_USERS_PASSWORDS = {
    "naveen": "neevaN",
    "sri": "irS",
    "ramesh": "hsemaR",
    "raja": "ajaR"
}

def get_user(x_user: Optional[str] = None) -> str:
    u = (x_user or "").lower().strip()
    if u not in VALID_USERS:
        raise HTTPException(status_code=401, detail="Invalid user. Use: naveen, sri, ramesh, raja")
    return u

# ── Models ─────────────────────────────────────────────────────────────────────
class LoginBody(BaseModel):
    username: str
    password: str

class ChatBody(BaseModel):
    message: str
    context: Optional[dict] = None

class RecordBody(BaseModel):
    data: dict

class SIPPayBody(BaseModel):
    sip_id: str

class SavingsUpdateBody(BaseModel):
    goal_id: str
    amount: float

class MarketUpdateBody(BaseModel):
    symbol: str
    price: float
    asset_class: Optional[str] = "commodity"

# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": "3.0",
        "ai": "groq/llama-3.3-70b",
        "timestamp": datetime.utcnow().isoformat(),
        "features": ["financial-engine", "market-data", "sip-management", "health-score", "predictions"]
    }

# ── Auth ───────────────────────────────────────────────────────────────────────
@app.post("/api/login")
def login(body: LoginBody):
    username = body.username.lower().strip()
    password = body.password
    expected = VALID_USERS_PASSWORDS.get(username)
    if not expected or expected != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {
        "success": True,
        "user": username,
        "display_name": username.capitalize(),
        "message": f"Welcome back, {username.capitalize()}!"
    }

# ── MAIN CHAT / AI FINANCIAL PROCESSOR ────────────────────────────────────────
@app.post("/api/chat")
async def chat(body: ChatBody, background_tasks: BackgroundTasks,
               x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    message = (body.message or "").strip()
    if not message:
        raise HTTPException(400, detail="Empty message")

    # Get full user financial context
    user_data = db.get_full_user_data(user)

    # AI processes the message
    result = await ai.process_message(message, user, user_data, body.context or {})

    # Execute financial actions based on AI decision
    executed = []
    errors = []
    if not result.get("clarification_needed") and result.get("actions"):
        for action in result["actions"]:
            try:
                action_result = await _execute_action(user, action, user_data)
                executed.append(action_result)
            except Exception as e:
                errors.append({"action": action.get("action"), "error": str(e)})

    # Generate post-action insights if something was executed
    if executed:
        # Refresh user data for accurate insights
        refreshed_data = db.get_full_user_data(user)
        insights = ai.generate_insights(refreshed_data)
        updated_summary = refreshed_data.get("summary", {})
    else:
        insights = result.get("insights", [])
        updated_summary = user_data.get("summary", {})

    # Generate notifications for important events
    if executed:
        background_tasks.add_task(_generate_notifications, user, executed)

    return {
        "response": result.get("response", "Processed."),
        "intent": result.get("intent"),
        "clarification_needed": result.get("clarification_needed", False),
        "clarification_question": result.get("clarification_question"),
        "executed_actions": executed,
        "errors": errors,
        "financial_impact": result.get("financial_impact", {}),
        "insights": insights,
        "summary": updated_summary,
    }

async def _execute_action(user: str, action: dict, user_data: dict) -> dict:
    """Execute a single financial action and return result."""
    action_type = action.get("action")
    data = action.get("data", {})
    today = date.today().isoformat()

    if action_type == "create_expense":
        record = db.add_transaction(
            user_id=user,
            tx_type="expense",
            category=data.get("category", "other"),
            description=data.get("description", ""),
            amount=float(data.get("amount", 0)),
            date_str=data.get("date", today),
            metadata=data.get("metadata", {})
        )
        return {"type": "expense_created", "record": record}

    elif action_type == "create_income":
        record = db.add_income(
            user_id=user,
            source=data.get("source", "Income"),
            amount=float(data.get("amount", 0)),
            date_str=data.get("date", today),
            is_recurring=data.get("is_recurring", False),
        )
        # Also record as transaction
        db.add_transaction(user, "income", data.get("source", "income"),
                           data.get("source", "Income"),
                           float(data.get("amount", 0)), data.get("date", today))
        return {"type": "income_created", "record": record}

    elif action_type == "create_investment":
        amount = float(data.get("amount", 0))
        record = db.add_investment(
            user_id=user,
            fund_name=data.get("fund_name", "Investment"),
            inv_type=data.get("investment_type", "other"),
            amount=amount,
            start_date=data.get("date", today),
        )
        # Record as transaction (cash outflow)
        db.add_transaction(user, "investment", data.get("investment_type", "investment"),
                           f"Investment in {data.get('fund_name', 'Fund')}", amount, data.get("date", today))
        return {"type": "investment_created", "record": record}

    elif action_type == "create_sip":
        amount = float(data.get("amount", 0))
        fund_name = data.get("fund_name", "Mutual Fund")
        months_back = int(data.get("months_back", 0))

        # Create investment record first
        total_invested = amount * max(1, months_back)
        inv_record = db.add_investment(
            user_id=user,
            fund_name=fund_name,
            inv_type="mutual_fund",
            amount=total_invested,
            start_date=data.get("start_date", today),
        )

        # Create SIP schedule
        sip_record = db.add_sip(
            user_id=user,
            fund_name=fund_name,
            amount=amount,
            frequency=data.get("frequency", "monthly"),
            investment_id=inv_record["id"],
            months_back=months_back,
        )

        # Record historical transactions
        if months_back > 0:
            today_date = date.today()
            for m in range(months_back, 0, -1):
                hist_date = today_date.replace(day=1) - timedelta(days=1)
                for _ in range(m - 1):
                    hist_date = hist_date.replace(day=1) - timedelta(days=1)
                hist_date = hist_date.replace(day=min(today_date.day, 28))
                db.add_transaction(user, "investment", "mutual_fund",
                                   f"SIP: {fund_name}", amount, hist_date.isoformat())

        # Try to fetch current NAV
        try:
            nav_data = await market.fetch_mutual_fund_nav(fund_name)
            if nav_data and nav_data.get("nav") and amount > 0:
                units = total_invested / nav_data["nav"]
                current_val = units * nav_data["nav"]
                db.update_investment_value(user, inv_record["id"], current_val, nav_data["nav"])
                inv_record.update({"units": units, "nav": nav_data["nav"], "current_value": current_val})
        except Exception:
            pass

        return {"type": "sip_created", "sip": sip_record, "investment": inv_record}

    elif action_type == "create_asset":
        amount = float(data.get("purchase_price", data.get("amount", 0)))
        qty = float(data.get("quantity", 1))
        asset_type = data.get("asset_type", "other")

        # Try to get current market price for valuation
        current_value = amount
        if asset_type == "gold":
            market_data = db.get_market_price("GOLD_INR_GRAM")
            if market_data:
                current_value = qty * market_data["price"]
        elif asset_type == "silver":
            market_data = db.get_market_price("SILVER_INR_GRAM")
            if market_data:
                current_value = qty * market_data["price"]

        record = db.add_asset(
            user_id=user,
            asset_type=asset_type,
            description=data.get("description", f"{qty} {asset_type}"),
            quantity=qty,
            purchase_price=amount,
            current_value=current_value,
            purchase_date=data.get("date", today),
        )
        # Cash outflow
        db.add_transaction(user, "asset_purchase", asset_type,
                           f"Bought {asset_type}", amount, data.get("date", today))
        return {"type": "asset_created", "record": record}

    elif action_type == "create_recurring":
        record = db.add_recurring(
            user_id=user,
            name=data.get("name", "Recurring Payment"),
            category=data.get("category", "other"),
            amount=float(data.get("amount", 0)),
            frequency=data.get("frequency", "monthly"),
        )
        return {"type": "recurring_created", "record": record}

    elif action_type == "create_savings_goal":
        record = db.add_savings_goal(
            user_id=user,
            goal_name=data.get("goal_name", "Savings Goal"),
            target_amount=float(data.get("target_amount", data.get("current_amount", 0))),
            current_amount=float(data.get("current_amount", 0)),
            target_date=data.get("target_date"),
        )
        return {"type": "savings_goal_created", "record": record}

    elif action_type == "create_debt":
        record = db.add_debt(
            user_id=user,
            debt_type=data.get("debt_type", "personal"),
            creditor=data.get("creditor", "Unknown"),
            principal=float(data.get("principal", data.get("amount", 0))),
            interest_rate=float(data.get("interest_rate", 0)),
            emi=float(data.get("emi", 0)),
            due_date=data.get("due_date"),
        )
        return {"type": "debt_created", "record": record}

    elif action_type == "update_prices":
        asset_type = data.get("asset_type", "gold")
        price = float(data.get("price_per_unit", 0))
        if price > 0:
            symbol = "GOLD_INR_GRAM" if asset_type == "gold" else "SILVER_INR_GRAM"
            db.upsert_market_price(symbol, "commodity", price)
            db.update_asset_values_by_type(user, asset_type, price)
        return {"type": "prices_updated", "asset_type": asset_type, "price": price}

    elif action_type == "query_data":
        return {"type": "query", "data": db.compute_financial_summary(user)}

    return {"type": "unknown", "action": action_type}


async def _generate_notifications(user: str, executed: list):
    """Generate smart notifications after financial actions."""
    for action in executed:
        action_type = action.get("type")
        if action_type == "expense_created":
            rec = action.get("record", {})
            if rec.get("amount", 0) > 5000:
                db.add_notification(user, "alert", "Large Expense",
                    f"₹{rec.get('amount', 0):,.0f} spent on {rec.get('category', 'other')}. Review if expected.")
        elif action_type == "sip_created":
            sip = action.get("sip", {})
            db.add_notification(user, "info", "SIP Scheduled",
                f"SIP of ₹{sip.get('amount', 0):,.0f}/month set up for {sip.get('fund_name')}.")
        elif action_type == "debt_created":
            rec = action.get("record", {})
            db.add_notification(user, "warning", "Debt Recorded",
                f"Debt of ₹{rec.get('principal', 0):,.0f} from {rec.get('creditor')} recorded. Plan repayment.")


# ── FINANCIAL DATA ENDPOINTS ────────────────────────────────────────────────────

@app.get("/api/summary")
def get_summary(x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    summary = db.compute_financial_summary(user)
    health = ai.compute_health_score(db.get_full_user_data(user))
    return {**summary, "health_score": health}

@app.get("/api/dashboard")
def get_dashboard(x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    user_data = db.get_full_user_data(user)
    summary = user_data.get("summary", {})
    health = ai.compute_health_score(user_data)
    insights = ai.generate_insights(user_data)
    predictions = ai.generate_predictions(user_data)

    # Recent transactions (last 10)
    recent = db.get_transactions(user, limit=10)

    # Upcoming payments (next 30 days)
    today = date.today()
    next_month = (today + timedelta(days=30)).isoformat()
    upcoming = [r for r in db.get_recurring(user) if r.get("next_due", "") <= next_month]

    # Category breakdown (last 30 days)
    since = (today - timedelta(days=30)).isoformat()
    recent_expenses = db.get_transactions(user, tx_type="expense", since_date=since)
    cat_breakdown = {}
    for e in recent_expenses:
        cat = e.get("category", "other")
        cat_breakdown[cat] = cat_breakdown.get(cat, 0) + e.get("amount", 0)

    # Market data
    market_data = db.get_all_market_data()

    return {
        "summary": summary,
        "health": health,
        "insights": insights,
        "recent_transactions": recent,
        "upcoming_payments": upcoming[:5],
        "category_breakdown": cat_breakdown,
        "expense_forecast": predictions.get("expense_forecast", [])[:3],
        "investment_projection": predictions.get("investment_projection", [])[:3],
        "market_data": market_data,
        "notifications": db.get_notifications(user, unread_only=True)[:5],
    }

@app.get("/api/transactions")
def get_transactions(tx_type: Optional[str] = None, limit: int = 100,
                     x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    return {"items": db.get_transactions(user, tx_type=tx_type, limit=limit)}

@app.delete("/api/transactions/{tx_id}")
def delete_transaction(tx_id: str, x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    db.delete_transaction(user, tx_id)
    return {"success": True}

@app.get("/api/investments")
def get_investments(x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    investments = db.get_investments(user)
    total_invested = sum(i.get("invested_amount", 0) for i in investments)
    total_current = sum(i.get("current_value") or i.get("invested_amount", 0) for i in investments)
    return {
        "items": investments,
        "total_invested": total_invested,
        "total_current_value": total_current,
        "total_gain": total_current - total_invested,
        "gain_pct": ((total_current - total_invested) / total_invested * 100) if total_invested > 0 else 0,
    }

@app.delete("/api/investments/{inv_id}")
def delete_investment(inv_id: str, x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    db.delete_investment(user, inv_id)
    return {"success": True}

@app.get("/api/sips")
def get_sips(x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    sips = db.get_sips(user)
    return {"items": sips, "total_monthly": sum(s.get("amount", 0) for s in sips if s.get("is_active"))}

@app.post("/api/sips/{sip_id}/pay")
def pay_sip(sip_id: str, x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    sip = db.mark_sip_paid(user, sip_id)
    if not sip:
        raise HTTPException(404, "SIP not found")
    # Record transaction
    db.add_transaction(user, "investment", "mutual_fund",
                       f"SIP payment: {sip.get('fund_name')}", sip.get("amount", 0), date.today().isoformat())
    return {"success": True, "sip": sip}

@app.get("/api/assets")
def get_assets(x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    assets = db.get_assets(user)
    total_cost = sum(a.get("purchase_price", 0) for a in assets)
    total_value = sum(a.get("current_value") or a.get("purchase_price", 0) for a in assets)
    return {
        "items": assets,
        "total_cost": total_cost,
        "total_current_value": total_value,
        "total_gain": total_value - total_cost,
    }

@app.delete("/api/assets/{asset_id}")
def delete_asset(asset_id: str, x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    db.delete_asset(user, asset_id)
    return {"success": True}

@app.get("/api/recurring")
def get_recurring(x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    items = db.get_recurring(user)
    return {"items": items, "total_monthly": sum(r.get("amount", 0) for r in items if r.get("frequency") == "monthly")}

@app.delete("/api/recurring/{rec_id}")
def delete_recurring(rec_id: str, x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    db.delete_recurring(user, rec_id)
    return {"success": True}

@app.get("/api/savings")
def get_savings(x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    goals = db.get_savings_goals(user)
    return {"items": goals, "total": sum(g.get("current_amount", 0) for g in goals)}

@app.delete("/api/savings/{goal_id}")
def delete_savings(goal_id: str, x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    db.delete_savings_goal(user, goal_id)
    return {"success": True}

@app.get("/api/debts")
def get_debts(x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    debts = db.get_debts(user)
    return {"items": debts, "total_debt": sum(d.get("remaining") or d.get("principal", 0) for d in debts)}

@app.get("/api/income")
def get_income(x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    income = db.get_income(user)
    return {"items": income, "total": sum(i.get("amount", 0) for i in income)}

@app.get("/api/health-score")
def get_health_score(x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    user_data = db.get_full_user_data(user)
    return ai.compute_health_score(user_data)

@app.get("/api/insights")
def get_insights(x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    user_data = db.get_full_user_data(user)
    return {
        "insights": ai.generate_insights(user_data),
        "stored": db.get_insights(user),
    }

@app.get("/api/predictions")
def get_predictions(x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    user_data = db.get_full_user_data(user)
    return ai.generate_predictions(user_data)

@app.get("/api/analytics")
def get_analytics(x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    user_data = db.get_full_user_data(user)
    summary = user_data.get("summary", {})
    predictions = ai.generate_predictions(user_data)
    health = ai.compute_health_score(user_data)

    # Monthly expense history
    expense_history = predictions.get("expense_history", [])

    # Category breakdown (all time)
    all_expenses = db.get_transactions(user, tx_type="expense")
    cat_breakdown = {}
    for e in all_expenses:
        cat = e.get("category", "other")
        cat_breakdown[cat] = cat_breakdown.get(cat, 0) + e.get("amount", 0)

    # Monthly investment trend
    investments = db.get_investments(user)
    inv_by_type = {}
    for i in investments:
        t = i.get("investment_type", "other")
        inv_by_type[t] = inv_by_type.get(t, 0) + i.get("invested_amount", 0)

    # Asset allocation
    assets = db.get_assets(user)
    asset_alloc = {}
    for a in assets:
        t = a.get("asset_type", "other")
        asset_alloc[t] = asset_alloc.get(t, a.get("current_value", a.get("purchase_price", 0)))
        asset_alloc[t] += a.get("current_value", a.get("purchase_price", 0))

    return {
        "summary": summary,
        "health": health,
        "expense_history": expense_history,
        "expense_forecast": predictions.get("expense_forecast", []),
        "category_breakdown": cat_breakdown,
        "investment_by_type": inv_by_type,
        "asset_allocation": asset_alloc,
        "investment_projection": predictions.get("investment_projection", []),
        "net_worth_forecast": predictions.get("net_worth_forecast", []),
        "ai_recommendation": predictions.get("ai_recommendation", ""),
    }

# ── Market Data ────────────────────────────────────────────────────────────────
@app.get("/api/market")
async def get_market_data(x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    # Fetch fresh prices
    prices = await market.fetch_all()
    stored = db.get_all_market_data()
    return {"prices": prices, "stored": stored}

@app.post("/api/market/refresh")
async def refresh_market(x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    prices = await market.fetch_all()
    asset_result = await market.update_asset_values(db, user)
    portfolio_result = await market.update_portfolio_values(db, user)
    return {
        "prices": prices,
        "assets_updated": asset_result,
        "portfolio_updated": portfolio_result,
    }

@app.post("/api/market/update")
def update_market_manual(body: MarketUpdateBody, x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    db.upsert_market_price(body.symbol, body.asset_class, body.price)
    return {"success": True, "symbol": body.symbol, "price": body.price}

# ── Notifications ─────────────────────────────────────────────────────────────
@app.get("/api/notifications")
def get_notifications(x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    return {"items": db.get_notifications(user)}

@app.post("/api/notifications/{notif_id}/read")
def mark_notification_read(notif_id: str, x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    db.mark_notification_read(user, notif_id)
    return {"success": True}

# ── Generic collection endpoints (legacy compatibility) ───────────────────────
COLLECTION_MAP = {
    "expenses": lambda user, limit: db.get_transactions(user, tx_type="expense", limit=limit),
    "investments": lambda user, limit: db.get_investments(user),
    "assets": lambda user, limit: db.get_assets(user),
    "recurring": lambda user, limit: db.get_recurring(user),
    "debts": lambda user, limit: db.get_debts(user),
    "savings": lambda user, limit: db.get_savings_goals(user),
}

@app.get("/api/{collection}")
def get_collection(collection: str, x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    if collection not in COLLECTION_MAP:
        raise HTTPException(404, f"Unknown collection: {collection}")
    items = COLLECTION_MAP[collection](user, 200)
    return {"items": items, "count": len(items)}

@app.delete("/api/{collection}/{record_id}")
def delete_from_collection(collection: str, record_id: str,
                            x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    if collection == "expenses":
        db.delete_transaction(user, record_id)
    elif collection == "investments":
        db.delete_investment(user, record_id)
    elif collection == "assets":
        db.delete_asset(user, record_id)
    elif collection == "recurring":
        db.delete_recurring(user, record_id)
    elif collection == "savings":
        db.delete_savings_goal(user, record_id)
    else:
        raise HTTPException(404)
    return {"success": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
