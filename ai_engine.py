"""
FinOS — Personal AI Financial Operating System
AI Engine — ai_engine.py

Responsibilities:
  - Parse natural language financial messages → structured intent + data
  - Generate financial insights from user data
  - Compute health scores
  - Generate spending predictions
  - Produce AI recommendations

Calls Anthropic Claude API when ANTHROPIC_API_KEY is set.
Falls back to deterministic rule-based parsing otherwise.
"""

import os
import re
import json
import math
import httpx
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, date, timedelta
from collections import defaultdict

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"


class AIEngine:

    def __init__(self):
        self.has_claude = bool(ANTHROPIC_API_KEY)
        if self.has_claude:
            print("[AI] Anthropic Claude API key found — using Claude for parsing")
        else:
            print("[AI] No Anthropic key — using rule-based parser")

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC: parse_message
    # ══════════════════════════════════════════════════════════════════════════
    async def parse_message(
        self,
        message: str,
        user_id: str,
        user_data: Dict,
        context: Dict
    ) -> Dict:
        """
        Main entry point. Returns:
        {
            "intent": str,
            "data": dict,
            "response": str,
            "insights": list[str]
        }
        """
        if self.has_claude:
            result = await self._call_claude(message, user_id, user_data)
            if result:
                return result
        # Fallback
        return self._rule_based_parse(message, user_data)

    # ══════════════════════════════════════════════════════════════════════════
    # CLAUDE API CALL
    # ══════════════════════════════════════════════════════════════════════════
    async def _call_claude(
        self,
        message: str,
        user_id: str,
        user_data: Dict
    ) -> Optional[Dict]:
        system_prompt = self._build_system_prompt(user_id, user_data)
        payload = {
            "model": CLAUDE_MODEL,
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [{"role": "user", "content": message}]
        }
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload,
                    headers=headers
                )
                resp.raise_for_status()
                body = resp.json()
                text = "".join(b.get("text", "") for b in body.get("content", []))
                return self._parse_json_response(text)
        except Exception as e:
            print(f"[AI] Claude API error: {e}")
            return None

    def _build_system_prompt(self, user_id: str, user_data: Dict) -> str:
        expenses    = user_data.get("expenses", [])
        investments = user_data.get("investments", [])
        savings     = user_data.get("savings", [])
        recurring   = user_data.get("recurring", [])
        assets      = user_data.get("assets", [])

        today = date.today().isoformat()
        current_month = datetime.now().strftime("%Y-%m")
        monthly_spend = sum(
            e["amount"] for e in expenses
            if (e.get("date") or "").startswith(current_month)
        )
        total_invested = sum(i["amount"] for i in investments)
        total_savings  = sum(s["amount"] for s in savings)

        return f"""You are FinOS AI — an expert personal financial assistant for user '{user_id}'.

Current financial snapshot (today: {today}):
- Monthly expenses so far: ₹{monthly_spend:,.0f}
- Total invested: ₹{total_invested:,.0f}
- Total savings: ₹{total_savings:,.0f}
- Active recurring payments: {len(recurring)}
- Physical assets tracked: {len(assets)}

Your job: parse the user's natural language message about their finances and return ONLY valid JSON.

Return this exact JSON structure (no markdown, no preamble, no extra text):
{{
  "intent": "<one of: add_expense | add_investment | add_asset | add_recurring | add_saving | update_asset_price | query_report | query_health | query_insights | general_chat>",
  "data": {{
    "description": "<short description>",
    "amount": <rupee amount as number, 0 if none>,
    "category": "<food|transport|shopping|utilities|health|entertainment|education|other>",
    "date": "<YYYY-MM-DD, default today: {today}>",
    "asset_type": "<gold|silver|property|vehicle|other>",
    "quantity": <numeric quantity, 0 if none>,
    "fund_name": "<fund or stock name if present>",
    "investment_type": "<mutual_fund|stocks|crypto|bonds|ppf|fd|other>",
    "frequency": "<monthly|weekly|yearly|quarterly>",
    "payment_name": "<name of recurring payment>",
    "account": "<savings account or goal name>",
    "price_per_unit": <price per gram or unit if mentioned, else 0>
  }},
  "response": "<warm, concise, professional confirmation or answer — 1-2 sentences>",
  "insights": ["<optional: 0-2 short insight strings based on the new data>"]
}}

Classification rules:
- "spent/bought/paid/purchased" → add_expense
- "invested/SIP/mutual fund/stock/crypto/bond/PPF/FD" → add_investment
- "gold/silver/property/vehicle" (buying) → add_asset
- "every month/weekly/monthly/rent/hostel/EMI/subscription" → add_recurring
- "saved/saving/savings/deposit" → add_saving
- silver/gold price update → update_asset_price
- "show report/analytics/chart/how much" → query_report
- "health score/financial health" → query_health
- anything else → general_chat

Category detection for expenses:
food/lunch/dinner/breakfast/chai/tea/coffee/restaurant/snack → food
petrol/fuel/bus/auto/cab/uber/ola/train/metro/toll → transport
shirt/clothes/shopping/amazon/flipkart/dress/t-shirt/shoes → shopping
electricity/bill/internet/mobile/recharge/wifi/gas → utilities
doctor/medicine/hospital/medical/pharmacy/clinic → health
movie/game/netflix/spotify/entertainment/concert → entertainment
course/book/college/school/fee/tuition → education

ONLY return the JSON object. Nothing else."""

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        text = text.strip()
        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract first { ... } block
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
        return None

    # ══════════════════════════════════════════════════════════════════════════
    # RULE-BASED PARSER (fallback)
    # ══════════════════════════════════════════════════════════════════════════
    def _rule_based_parse(self, message: str, user_data: Dict) -> Dict:
        lower = message.lower()
        today = date.today().isoformat()
        amount = self._extract_amount(message)
        detected_date = self._extract_date(message) or today

        intent = "general_chat"
        data: Dict[str, Any] = {"amount": amount, "date": detected_date}
        response = ""
        insights: List[str] = []

        # ── Expense ──────────────────────────────────────────────────────────
        if re.search(r"\b(spent|spend|bought|buy|paid|pay|purchased|purchase|expense|cost)\b", lower):
            intent = "add_expense"
            category = self._detect_expense_category(lower)
            data["category"] = category
            data["description"] = message.strip()
            response = f"Got it! ₹{amount:,.0f} logged under {category}. 📝"
            if amount > 2000:
                insights.append(f"That's a significant {category} expense. You may want to track this category closely.")

        # ── Investment ────────────────────────────────────────────────────────
        elif re.search(r"\b(invest|invested|sip|mutual fund|stock|crypto|ppf|fd|bond|nifty|sensex)\b", lower):
            intent = "add_investment"
            inv_type = self._detect_investment_type(lower)
            fund_name = self._extract_fund_name(message) or "Investment"
            data["fund_name"] = fund_name
            data["investment_type"] = inv_type
            response = f"Investment of ₹{amount:,.0f} in {fund_name} recorded! 📈"
            if amount >= 5000:
                insights.append("Great job investing! Consistent SIPs build significant wealth over time.")

        # ── Asset: gold/silver/property ────────────────────────────────────
        elif re.search(r"\b(gold|silver|property|land|house|vehicle|car|bike)\b", lower):
            if re.search(r"\b(price|rate|now|today|current)\b", lower):
                intent = "update_asset_price"
                asset_type = "gold" if "gold" in lower else "silver"
                data["asset_type"] = asset_type
                response = f"Got it! I'll update the current {asset_type} price to ₹{amount:,.0f}. 💰"
            else:
                intent = "add_asset"
                asset_type = (
                    "gold" if "gold" in lower
                    else "silver" if "silver" in lower
                    else "property" if re.search(r"property|land|house", lower)
                    else "vehicle"
                )
                qty = self._extract_quantity(message)
                data["asset_type"] = asset_type
                data["quantity"] = qty
                data["description"] = f"{asset_type} purchase"
                response = f"Asset recorded: {qty}g of {asset_type} at ₹{amount:,.0f}. 🥇"

        # ── Recurring ─────────────────────────────────────────────────────────
        elif re.search(r"\b(every|monthly|weekly|annual|yearly|recurring|rent|hostel|emi|subscription|subscribe)\b", lower):
            intent = "add_recurring"
            freq = (
                "weekly"    if "weekly" in lower or "every week" in lower
                else "yearly"  if re.search(r"yearly|annual|every year", lower)
                else "quarterly" if "quarterly" in lower
                else "monthly"
            )
            payment_name = self._extract_payment_name(message) or message.strip()
            data["payment_name"] = payment_name
            data["frequency"] = freq
            data["category"] = self._detect_recurring_category(lower)
            response = f"Recurring payment '{payment_name}' of ₹{amount:,.0f}/{freq} set up! 🔔"

        # ── Saving ─────────────────────────────────────────────────────────
        elif re.search(r"\b(saved|saving|savings|deposit|deposited|set aside)\b", lower):
            intent = "add_saving"
            account = self._extract_account_name(message) or "General Savings"
            data["description"] = message.strip()
            data["account"] = account
            response = f"₹{amount:,.0f} added to {account}. Keep saving! 💪"

        # ── Report query ──────────────────────────────────────────────────
        elif re.search(r"\b(report|analytics|chart|graph|show|display|summarize|summary|how much)\b", lower):
            intent = "query_report"
            response = "Here's an overview of your finances. Detailed charts are in the Analytics section. 📊"

        # ── Health score ──────────────────────────────────────────────────
        elif re.search(r"\b(health score|financial health|score|how.*(doing|am i))\b", lower):
            intent = "query_health"
            score_data = self.compute_health_score(user_data)
            response = f"Your financial health score is {score_data['score']}/100. {score_data['summary']} 💡"

        # ── Fallback ──────────────────────────────────────────────────────
        else:
            intent = "general_chat"
            response = (
                "I'm your FinOS AI assistant! You can tell me about:\n"
                "• Expenses: 'I spent ₹500 on food'\n"
                "• Investments: 'I invested ₹5000 in HDFC Mutual Fund'\n"
                "• Assets: 'I bought 10g of gold for ₹6000'\n"
                "• Recurring: 'I pay ₹2000 rent monthly'\n"
                "• Savings: 'I saved ₹3000 this month'"
            )

        return {
            "intent": intent,
            "data": data,
            "response": response,
            "insights": insights
        }

    # ══════════════════════════════════════════════════════════════════════════
    # HELPER EXTRACTORS
    # ══════════════════════════════════════════════════════════════════════════
    def _extract_amount(self, text: str) -> float:
        # Match ₹500, Rs 500, 500 rupees, 500
        patterns = [
            r"₹\s*([\d,]+(?:\.\d+)?)",
            r"[Rr][Ss]\.?\s*([\d,]+(?:\.\d+)?)",
            r"([\d,]+(?:\.\d+)?)\s*(?:rupees?|rs\.?)",
            r"\b([\d,]+(?:\.\d+)?)\b",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                try:
                    val = float(m.group(1).replace(",", ""))
                    if val > 0:
                        return val
                except ValueError:
                    pass
        return 0.0

    def _extract_date(self, text: str) -> Optional[str]:
        today = date.today()
        lower = text.lower()
        if "today" in lower:
            return today.isoformat()
        if "yesterday" in lower:
            return (today - timedelta(days=1)).isoformat()
        m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
        if m:
            return m.group(1)
        m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", text)
        if m:
            try:
                d, mo, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if yr < 100:
                    yr += 2000
                return date(yr, mo, d).isoformat()
            except Exception:
                pass
        return None

    def _extract_quantity(self, text: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:gram|grams|gm|g\b|kg|kilogram)", text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if "kg" in m.group().lower():
                val *= 1000
            return val
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        if m:
            return float(m.group(1))
        return 1.0

    def _extract_fund_name(self, text: str) -> Optional[str]:
        # Try to find text after "in" or "into"
        m = re.search(r"\b(?:in|into)\s+([A-Za-z][A-Za-z0-9\s&]+?)(?:\s+(?:fund|mutual|for|of|at|₹|rs)|\s*$)", text, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if len(name) > 2:
                return name
        return None

    def _extract_payment_name(self, text: str) -> Optional[str]:
        # Try to extract the main subject
        m = re.search(r"(?:pay|paying|payment for|pay for)\s+([A-Za-z][A-Za-z0-9\s]+?)(?:\s+(?:every|for|of|₹|rs|\d)|$)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None

    def _extract_account_name(self, text: str) -> Optional[str]:
        m = re.search(r"(?:in|into|to|for)\s+([A-Za-z][A-Za-z0-9\s]+?)(?:\s+(?:account|fund|wallet|goal)|\s*$)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None

    def _detect_expense_category(self, lower: str) -> str:
        rules = [
            ("food",          r"food|lunch|dinner|breakfast|chai|tea|coffee|restaurant|hotel|snack|meal|biryani|dosa|idly|swiggy|zomato|eat|drink"),
            ("transport",     r"petrol|fuel|bus|auto|cab|uber|ola|train|metro|toll|diesel|rapido|bike|travel|ticket|fare"),
            ("shopping",      r"shirt|clothes|shopping|amazon|flipkart|dress|t.shirt|shoes|bag|jeans|fashion|mall|cloth"),
            ("utilities",     r"electricity|bill|internet|mobile|recharge|wifi|gas|water|broadband|dth|cable"),
            ("health",        r"doctor|medicine|hospital|medical|pharmacy|clinic|health|tablet|injection|dental|eye"),
            ("entertainment", r"movie|game|netflix|spotify|prime|entertainment|concert|event|theatre|cricket|match|pub|bar"),
            ("education",     r"course|book|college|school|fee|tuition|coaching|class|tutorial|udemy|study"),
        ]
        for cat, pattern in rules:
            if re.search(pattern, lower):
                return cat
        return "other"

    def _detect_investment_type(self, lower: str) -> str:
        if re.search(r"mutual fund|mf|sip|flexi|bluechip|midcap|smallcap|largecap|debt fund|hybrid", lower):
            return "mutual_fund"
        if re.search(r"stock|share|nifty|sensex|equity|ipo|bse|nse|zerodha|groww", lower):
            return "stocks"
        if re.search(r"crypto|bitcoin|btc|eth|ethereum|solana|usdt|coin", lower):
            return "crypto"
        if re.search(r"bond|debenture|ncd|government bond|g-sec", lower):
            return "bonds"
        if re.search(r"\bppf\b|public provident", lower):
            return "ppf"
        if re.search(r"\bfd\b|fixed deposit|recurring deposit|\brd\b", lower):
            return "fd"
        return "other"

    def _detect_recurring_category(self, lower: str) -> str:
        if re.search(r"rent|hostel|pg|accommodation|house|flat|room", lower):
            return "housing"
        if re.search(r"electricity|internet|mobile|wifi|gas|water|bill", lower):
            return "utilities"
        if re.search(r"emi|loan|mortgage|car loan|home loan|personal loan", lower):
            return "loan"
        if re.search(r"netflix|spotify|prime|subscription|saas|software|app", lower):
            return "subscriptions"
        if re.search(r"insurance|lic|policy|premium|health insurance", lower):
            return "insurance"
        if re.search(r"petrol|fuel|bus pass|train pass|metro", lower):
            return "transport"
        return "other"

    # ══════════════════════════════════════════════════════════════════════════
    # FINANCIAL HEALTH SCORE
    # ══════════════════════════════════════════════════════════════════════════
    def compute_health_score(self, user_data: Dict) -> Dict:
        expenses    = user_data.get("expenses", [])
        investments = user_data.get("investments", [])
        assets      = user_data.get("assets", [])
        savings     = user_data.get("savings", [])
        recurring   = user_data.get("recurring", [])

        total_expenses   = sum(e.get("amount", 0) for e in expenses)
        total_savings    = sum(s.get("amount", 0) for s in savings)
        total_invested   = sum(i.get("amount", 0) for i in investments)
        assets_value     = sum(a.get("current_value", a.get("purchase_price", 0)) for a in assets)
        monthly_recurring = sum(r.get("amount", 0) for r in recurring if r.get("frequency") == "monthly")

        scores: Dict[str, int] = {}

        # 1. Savings rate (0-25)
        total_financial = total_expenses + total_savings
        savings_rate = (total_savings / total_financial * 100) if total_financial > 0 else 0
        scores["savings_rate"] = min(25, int(savings_rate * 0.5))

        # 2. Expense control (0-25)
        # Lower recurring-to-savings ratio is better
        if total_savings > 0:
            rec_ratio = monthly_recurring / (total_savings / 12 + 1)
            scores["expense_control"] = min(25, max(0, int(25 - rec_ratio * 5)))
        else:
            scores["expense_control"] = 5 if len(expenses) > 0 else 0

        # 3. Investment diversity (0-25)
        inv_types = len(set(i.get("type", "other") for i in investments))
        inv_score = min(20, inv_types * 7)
        if total_invested > 10000:
            inv_score = min(25, inv_score + 5)
        scores["investment_diversity"] = inv_score

        # 4. Asset coverage (0-25)
        asset_types = len(set(a.get("asset_type", "other") for a in assets))
        asset_score = min(20, asset_types * 8)
        if assets_value > 5000:
            asset_score = min(25, asset_score + 5)
        scores["asset_coverage"] = asset_score

        total_score = sum(scores.values())

        if total_score >= 75:
            grade = "A"
            summary = "Excellent financial health! Your saving, investing, and asset-building habits are strong."
        elif total_score >= 60:
            grade = "B"
            summary = "Good financial health. Consider diversifying investments and building more savings."
        elif total_score >= 40:
            grade = "C"
            summary = "Moderate financial health. Focus on increasing your savings rate and reducing recurring outflows."
        elif total_score >= 20:
            grade = "D"
            summary = "Needs improvement. Start tracking all expenses and set a monthly savings goal immediately."
        else:
            grade = "F"
            summary = "Getting started. Log your first expenses and savings to begin your financial journey."

        return {
            "score": total_score,
            "grade": grade,
            "summary": summary,
            "breakdown": {
                "savings_rate":          scores["savings_rate"],
                "expense_control":       scores["expense_control"],
                "investment_diversity":  scores["investment_diversity"],
                "asset_coverage":        scores["asset_coverage"],
            },
            "max_score": 100
        }

    # ══════════════════════════════════════════════════════════════════════════
    # INSIGHTS GENERATOR
    # ══════════════════════════════════════════════════════════════════════════
    def generate_insights(self, user_data: Dict) -> List[str]:
        expenses    = user_data.get("expenses", [])
        investments = user_data.get("investments", [])
        savings     = user_data.get("savings", [])
        recurring   = user_data.get("recurring", [])

        insights: List[str] = []
        today = datetime.now()
        current_month = f"{today.year}-{str(today.month).zfill(2)}"

        # Monthly spend
        monthly_expenses = sum(
            e["amount"] for e in expenses
            if (e.get("date") or "").startswith(current_month)
        )
        if monthly_expenses > 0:
            daily_avg = monthly_expenses / today.day
            insights.append(f"Daily average spend this month: ₹{daily_avg:,.0f}")

        # Category breakdown
        cat_totals: Dict[str, float] = defaultdict(float)
        for e in expenses:
            cat_totals[e.get("category", "other")] += e.get("amount", 0)
        if cat_totals:
            top_cat, top_val = max(cat_totals.items(), key=lambda x: x[1])
            insights.append(f"Highest spend category: {top_cat} at ₹{top_val:,.0f} total")

        # Savings momentum
        total_savings = sum(s.get("amount", 0) for s in savings)
        if total_savings > 0 and monthly_expenses > 0:
            months_covered = total_savings / monthly_expenses
            insights.append(f"Your savings cover {months_covered:.1f} months of current spending")

        # Investment return
        total_invested = sum(i.get("amount", 0) for i in investments)
        total_current  = sum(i.get("current_value", i.get("amount", 0)) for i in investments)
        if total_invested > 0:
            pnl_pct = ((total_current - total_invested) / total_invested) * 100
            direction = "up" if pnl_pct >= 0 else "down"
            insights.append(f"Portfolio is {direction} {abs(pnl_pct):.1f}% overall")

        # Recurring burden
        monthly_recurring = sum(
            r.get("amount", 0) for r in recurring if r.get("frequency") == "monthly"
        )
        if monthly_recurring > 0 and monthly_expenses > 0:
            rec_pct = (monthly_recurring / monthly_expenses) * 100
            insights.append(f"Recurring payments are {rec_pct:.0f}% of your monthly expenses")

        # Alert: high spending day
        day_totals: Dict[str, float] = defaultdict(float)
        for e in expenses:
            if e.get("date"):
                day_totals[e["date"]] += e.get("amount", 0)
        if day_totals:
            peak_day, peak_val = max(day_totals.items(), key=lambda x: x[1])
            if peak_val > 1000:
                insights.append(f"Highest spending day: {peak_day} at ₹{peak_val:,.0f}")

        return insights[:6]  # Return top 6 insights

    # ══════════════════════════════════════════════════════════════════════════
    # PREDICTIONS
    # ══════════════════════════════════════════════════════════════════════════
    def generate_predictions(self, user_data: Dict) -> Dict:
        expenses    = user_data.get("expenses", [])
        investments = user_data.get("investments", [])
        savings     = user_data.get("savings", [])
        recurring   = user_data.get("recurring", [])

        today = datetime.now()

        # Monthly expense history for last 6 months
        monthly_data: Dict[str, float] = {}
        for i in range(5, -1, -1):
            yr  = today.year if today.month - i > 0 else today.year - 1
            mo  = (today.month - i - 1) % 12 + 1
            key = f"{yr}-{str(mo).zfill(2)}"
            monthly_data[key] = sum(
                e["amount"] for e in expenses
                if (e.get("date") or "").startswith(key)
            )

        values = list(monthly_data.values())
        non_zero = [v for v in values if v > 0]
        avg_monthly = sum(non_zero) / len(non_zero) if non_zero else 0

        # Linear trend
        if len(non_zero) >= 2:
            n = len(non_zero)
            x_mean = (n - 1) / 2
            y_mean = sum(non_zero) / n
            num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(non_zero))
            den = sum((i - x_mean) ** 2 for i in range(n))
            slope = num / den if den != 0 else 0
        else:
            slope = 0

        # 6-month expense forecast
        expense_forecast = []
        for i in range(1, 7):
            mo_date = date(today.year, today.month, 1)
            # advance i months
            new_mo = (mo_date.month + i - 1) % 12 + 1
            new_yr = mo_date.year + (mo_date.month + i - 1) // 12
            label = datetime(new_yr, new_mo, 1).strftime("%b %y")
            predicted = max(0, round(avg_monthly + slope * i))
            expense_forecast.append({"month": label, "predicted": predicted})

        # 12-month investment projection at 12% annual (1% per month)
        total_invested  = sum(i.get("amount", 0) for i in investments)
        total_current   = sum(i.get("current_value", i.get("amount", 0)) for i in investments)
        invest_forecast = []
        base = total_current or total_invested
        for i in range(1, 13):
            new_mo = (today.month + i - 1) % 12 + 1
            new_yr = today.year + (today.month + i - 1) // 12
            label = datetime(new_yr, new_mo, 1).strftime("%b %y")
            projected = round(base * math.pow(1.01, i))
            invest_forecast.append({"month": label, "projected": projected})

        # Savings projection
        total_savings = sum(s.get("amount", 0) for s in savings)
        monthly_recurring = sum(r.get("amount", 0) for r in recurring if r.get("frequency") == "monthly")
        savings_6mo = round(total_savings + max(0, (avg_monthly * 0.2 - monthly_recurring)) * 6)

        # AI recommendation
        recommendation = self._generate_recommendation(
            avg_monthly, total_invested, total_savings, slope, monthly_recurring
        )

        return {
            "expense_forecast":     expense_forecast,
            "investment_projection": invest_forecast,
            "predicted_monthly_expense": round(avg_monthly + slope),
            "predicted_investment_return_12mo": round(base * math.pow(1.01, 12) - base),
            "predicted_savings_6mo": savings_6mo,
            "ai_recommendation":    recommendation,
            "trend": "increasing" if slope > 50 else "decreasing" if slope < -50 else "stable"
        }

    def _generate_recommendation(
        self,
        avg_expense: float,
        total_invested: float,
        total_savings: float,
        trend: float,
        monthly_recurring: float
    ) -> str:
        if avg_expense == 0 and total_invested == 0 and total_savings == 0:
            return ("Start by recording your daily expenses for 30 days. "
                    "Once you understand your spending patterns, allocate 20% of income to savings "
                    "and start a ₹500/month SIP in a diversified equity mutual fund.")

        tips = []
        if trend > 100:
            tips.append(f"Your expenses are trending upward by ₹{trend:,.0f}/month. "
                        "Identify and cut discretionary spending.")
        if total_savings > 0 and avg_expense > 0:
            runway = total_savings / avg_expense
            if runway < 3:
                tips.append("Build an emergency fund of at least 3 months of expenses before investing aggressively.")
        if total_invested == 0 and total_savings > 10000:
            tips.append("You have savings but no investments. Consider starting a monthly SIP even at ₹1,000 to beat inflation.")
        if monthly_recurring > avg_expense * 0.5:
            tips.append(f"₹{monthly_recurring:,.0f} in recurring payments is over 50% of your average monthly spend — review subscriptions.")

        if tips:
            return " ".join(tips)
        return ("You're managing your finances well. "
                "Continue your current saving and investment habits. "
                "Consider rebalancing your portfolio every 6 months for optimal returns.")
