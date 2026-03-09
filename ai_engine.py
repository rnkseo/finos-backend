"""
AstraOS — AI Financial Engine v3.0
Multi-stage financial classification + Groq LLaMA reasoning
Acts as a CFO: classifies intent → executes financial actions → generates insights
"""
import os, re, json, math, httpx
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from collections import defaultdict

GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL_MAIN  = "llama-3.3-70b-versatile"   # complex reasoning
GROQ_MODEL_FAST  = "llama-3.1-8b-instant"       # classification / fast tasks


FINANCIAL_SYSTEM_PROMPT = """You are AstraOS CFO — an AI Chief Financial Officer, accountant, and investment manager.

Your role is NOT to chat. Your role is to PARSE financial input and return structured JSON.

You manage a personal finance system with these data types:
- expenses: daily spending (food, transport, shopping, utilities, health, entertainment, education, rent, other)
- investments: mutual funds, stocks, crypto, bonds, PPF, FD, NPS
- sip: systematic investment plan (recurring mutual fund contributions)
- assets: physical assets (gold, silver, property, vehicle, jewelry)
- recurring: recurring payments (rent, EMI, subscriptions, bills, insurance)
- savings: savings goals (emergency fund, vacation, car, house, etc.)
- debts: loans and liabilities (home loan, car loan, personal loan, credit card)
- income: salary, freelance, rental income, business income

RETURN ONLY VALID JSON. NO MARKDOWN. NO EXPLANATION. NO PREAMBLE.

JSON structure:
{
  "intent": "<expense|investment|sip|asset|recurring|savings|debt|income|update_asset_price|query|ambiguous|general>",
  "confidence": <0.0-1.0>,
  "actions": [
    {
      "action": "<create_expense|create_investment|create_sip|create_asset|create_recurring|create_savings_goal|create_debt|create_income|update_prices|query_data>",
      "data": {
        "description": "<string>",
        "amount": <number>,
        "category": "<string>",
        "date": "<YYYY-MM-DD>",
        "fund_name": "<string>",
        "investment_type": "<mutual_fund|stocks|crypto|bonds|ppf|fd|nps|other>",
        "asset_type": "<gold|silver|property|vehicle|jewelry|other>",
        "quantity": <number>,
        "frequency": "<monthly|weekly|yearly|quarterly>",
        "name": "<string>",
        "source": "<string>",
        "creditor": "<string>",
        "interest_rate": <number>,
        "emi": <number>,
        "months_back": <number>,
        "sip_funds": ["<fund1>", "<fund2>"],
        "price_per_unit": <number>,
        "target_amount": <number>,
        "current_amount": <number>
      }
    }
  ],
  "clarification_needed": false,
  "clarification_question": null,
  "response": "<1-2 sentence professional confirmation>",
  "financial_impact": {
    "cash_change": <negative for outflow, positive for inflow>,
    "asset_change": <increase in asset value>,
    "investment_change": <increase in investment value>,
    "debt_change": <increase in debt>
  },
  "insights": ["<0-2 brief financial insights>"]
}

CLASSIFICATION RULES:
- "spent/bought/paid/purchased/expense" + amount → expense
- "invested/sip/mutual fund/mf/stock/equity/crypto/ppf/fd/bonds/nps" → investment or sip
- "gold/silver/property/land/house/vehicle/car/bike" (buying) → asset
- "every month/weekly/rent/hostel/emi/subscription/recurring" → recurring
- "saved/saving/goal/emergency fund/target" → savings
- "loan/borrowed/debt/owe/credit card/mortgage" → debt
- "salary/income/earned/received/freelance/got paid" → income
- "gold price/silver rate/price now/current rate" → update_asset_price
- "show/how much/total/report/summary/balance" → query
- amount present but type unclear → ambiguous (ask clarification)
- no financial content → general

SIP DETECTION:
- "SIP of ₹2000 in HDFC, ICICI, Nippon for 6 months" → create 3 separate sip actions with months_back=6
- "started SIP 3 months ago" → months_back=3
- "paying SIP since January" → calculate months_back from January to today

MULTI-ACTION: A single message can produce multiple actions.
Example: "I have SIP in HDFC and ICICI for ₹2000 each" → 2 sip actions

IMPORTANT: For ambiguous amounts (e.g. "I paid 5000" with no context), set clarification_needed=true."""


class AIEngine:

    def __init__(self):
        self.has_groq = bool(GROQ_API_KEY)
        if self.has_groq:
            print("[AI] Groq API key found — using LLaMA 3.3 70B")
        else:
            print("[AI] No Groq key — rule-based parser active")

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC: process_message — main entry point
    # Returns structured result with actions, response, insights
    # ══════════════════════════════════════════════════════════════════════════
    async def process_message(self, message: str, user_id: str,
                               user_data: Dict, context: Dict = None) -> Dict:
        context = context or {}

        # Stage 1: Quick safety/relevance check
        if self._is_injection_attempt(message):
            return self._safe_response("I can only help with financial management tasks.")

        # Stage 2: Try Groq AI classification
        if self.has_groq:
            result = await self._call_groq(message, user_id, user_data)
            if result:
                return result

        # Stage 3: Rule-based fallback
        return self._rule_based_parse(message, user_data)

    # ══════════════════════════════════════════════════════════════════════════
    # GROQ API CALL
    # ══════════════════════════════════════════════════════════════════════════
    async def _call_groq(self, message: str, user_id: str, user_data: Dict) -> Optional[Dict]:
        summary = user_data.get("summary", {})
        today = date.today().isoformat()

        context_snippet = f"""User financial context (today: {today}):
- Net worth: ₹{summary.get('net_worth', 0):,.0f}
- Cash balance (est.): ₹{summary.get('cash_balance', 0):,.0f}
- Monthly expenses so far: ₹{summary.get('month_expenses', 0):,.0f}
- Total invested: ₹{summary.get('total_invested', 0):,.0f}
- Portfolio value: ₹{summary.get('total_portfolio_value', 0):,.0f}
- Active SIPs: {summary.get('sip_count', 0)}
- Total debt: ₹{summary.get('total_debt', 0):,.0f}"""

        messages = [
            {"role": "system", "content": FINANCIAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"{context_snippet}\n\nUser message: {message}"}
        ]

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    f"{GROQ_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": GROQ_MODEL_MAIN,
                        "messages": messages,
                        "max_tokens": 1500,
                        "temperature": 0.1,  # Low temp for consistent JSON
                    }
                )
                resp.raise_for_status()
                body = resp.json()
                text = body["choices"][0]["message"]["content"]
                return self._parse_json_response(text)
        except Exception as e:
            print(f"[AI] Groq error: {e}")
            return None

    async def _call_groq_analysis(self, prompt: str) -> Optional[str]:
        """Call Groq for free-form analysis/insights."""
        if not self.has_groq:
            return None
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{GROQ_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": GROQ_MODEL_MAIN,
                        "messages": [
                            {"role": "system", "content": "You are an expert financial analyst. Be concise, precise, and data-driven. Provide actionable insights in 3-5 bullet points."},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 800,
                        "temperature": 0.3,
                    }
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[AI] Analysis error: {e}")
            return None

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
        return None

    # ══════════════════════════════════════════════════════════════════════════
    # RULE-BASED PARSER (fallback when no Groq)
    # ══════════════════════════════════════════════════════════════════════════
    def _rule_based_parse(self, message: str, user_data: Dict) -> Dict:
        lower = message.lower()
        today = date.today().isoformat()
        amount = self._extract_amount(message)
        detected_date = self._extract_date(message) or today

        # Detect intent
        if re.search(r"\b(salary|income|earned|received|freelance|got paid|credited)\b", lower):
            return self._build_action("income", [{
                "action": "create_income",
                "data": {
                    "source": self._extract_source(message) or "Salary",
                    "amount": amount,
                    "date": detected_date,
                }
            }], f"Income of ₹{amount:,.0f} recorded. 💰",
            financial_impact={"cash_change": amount})

        if re.search(r"\b(spent|spend|bought|buy|paid|pay|purchased|expense|cost|eating|ate)\b", lower):
            cat = self._detect_expense_category(lower)
            return self._build_action("expense", [{
                "action": "create_expense",
                "data": {
                    "description": message.strip(),
                    "amount": amount,
                    "category": cat,
                    "date": detected_date,
                }
            }], f"₹{amount:,.0f} expense recorded under {cat}. 📝",
            financial_impact={"cash_change": -amount})

        if re.search(r"\b(sip|systematic investment)\b", lower):
            funds = self._extract_multiple_funds(message)
            months_back = self._extract_months_back(message)
            if not funds:
                funds = [self._extract_fund_name(message) or "Mutual Fund"]
            actions = [{
                "action": "create_sip",
                "data": {
                    "fund_name": fund,
                    "amount": amount,
                    "frequency": "monthly",
                    "months_back": months_back,
                    "investment_type": "mutual_fund",
                }
            } for fund in funds]
            total = amount * len(funds) * months_back if months_back else amount * len(funds)
            return self._build_action("sip", actions,
                f"SIP of ₹{amount:,.0f}/month set up for {', '.join(funds)}. 📈",
                financial_impact={"cash_change": -total, "investment_change": total})

        if re.search(r"\b(invest|invested|mutual fund|stock|crypto|ppf|fd|bond|nifty)\b", lower):
            inv_type = self._detect_investment_type(lower)
            fund_name = self._extract_fund_name(message) or "Investment"
            return self._build_action("investment", [{
                "action": "create_investment",
                "data": {
                    "fund_name": fund_name,
                    "amount": amount,
                    "investment_type": inv_type,
                    "date": detected_date,
                }
            }], f"₹{amount:,.0f} invested in {fund_name}. 📈",
            financial_impact={"cash_change": -amount, "investment_change": amount})

        if re.search(r"\b(gold|silver|property|land|house|vehicle|car|bike)\b", lower):
            if re.search(r"\b(price|rate|now|today|current)\b", lower):
                asset_type = "gold" if "gold" in lower else "silver"
                return self._build_action("update_asset_price", [{
                    "action": "update_prices",
                    "data": {"asset_type": asset_type, "price_per_unit": amount}
                }], f"{asset_type.title()} price updated to ₹{amount:,.0f}/gram. 💛")
            else:
                asset_type = self._detect_asset_type(lower)
                qty = self._extract_quantity(message)
                return self._build_action("asset", [{
                    "action": "create_asset",
                    "data": {
                        "asset_type": asset_type,
                        "quantity": qty,
                        "purchase_price": amount,
                        "description": f"{qty}g {asset_type}" if asset_type in ("gold","silver") else asset_type,
                        "date": detected_date,
                    }
                }], f"{asset_type.title()} asset recorded: {qty} units at ₹{amount:,.0f}. 🥇",
                financial_impact={"cash_change": -amount, "asset_change": amount})

        if re.search(r"\b(every|monthly|weekly|recurring|rent|hostel|emi|subscription)\b", lower):
            freq = self._detect_frequency(lower)
            name = self._extract_payment_name(message) or message[:40]
            cat = self._detect_recurring_category(lower)
            return self._build_action("recurring", [{
                "action": "create_recurring",
                "data": {
                    "name": name,
                    "amount": amount,
                    "frequency": freq,
                    "category": cat,
                }
            }], f"Recurring payment '{name}' of ₹{amount:,.0f}/{freq} scheduled. 🔔")

        if re.search(r"\b(saved|saving|goal|emergency|target|fund)\b", lower):
            goal_name = self._extract_account_name(message) or "Savings Goal"
            return self._build_action("savings", [{
                "action": "create_savings_goal",
                "data": {
                    "goal_name": goal_name,
                    "current_amount": amount,
                    "target_amount": amount,
                }
            }], f"₹{amount:,.0f} added to {goal_name}. 🎯",
            financial_impact={"cash_change": -amount})

        if re.search(r"\b(loan|borrowed|debt|owe|mortgage|emi due)\b", lower):
            return self._build_action("debt", [{
                "action": "create_debt",
                "data": {
                    "debt_type": "personal",
                    "creditor": "Unknown",
                    "principal": amount,
                    "description": message.strip(),
                }
            }], f"Debt of ₹{amount:,.0f} recorded. 📋",
            financial_impact={"debt_change": amount})

        if re.search(r"\b(show|how much|total|report|summary|balance|health|score|analytics)\b", lower):
            return self._build_action("query", [{
                "action": "query_data",
                "data": {"query_type": "summary"}
            }], "Here's your financial summary. Check the dashboard for detailed analytics. 📊")

        # Ambiguous amount
        if amount > 0:
            return {
                "intent": "ambiguous",
                "confidence": 0.4,
                "actions": [],
                "clarification_needed": True,
                "clarification_question": f"I see ₹{amount:,.0f}. Was this an expense, investment, income, or savings?",
                "response": f"I see ₹{amount:,.0f} mentioned. Could you clarify: was this an expense, investment, income, or savings?",
                "financial_impact": {},
                "insights": []
            }

        return self._build_action("general", [], (
            "I'm your AstraOS CFO. Tell me about:\n"
            "• Expenses: 'Spent ₹500 on food'\n"
            "• Investments: 'Invested ₹5000 in HDFC Mutual Fund'\n"
            "• SIP: 'SIP of ₹2000 in Nippon India for 6 months'\n"
            "• Assets: 'Bought 10g gold for ₹72000'\n"
            "• Income: 'Received salary ₹50000'\n"
            "• Recurring: 'Pay ₹8000 rent monthly'"
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # HELPER BUILDERS
    # ══════════════════════════════════════════════════════════════════════════
    def _build_action(self, intent: str, actions: list, response: str,
                      financial_impact: dict = None, insights: list = None) -> Dict:
        return {
            "intent": intent,
            "confidence": 0.8,
            "actions": actions,
            "clarification_needed": False,
            "clarification_question": None,
            "response": response,
            "financial_impact": financial_impact or {},
            "insights": insights or [],
        }

    def _safe_response(self, msg: str) -> Dict:
        return self._build_action("blocked", [], msg)

    def _is_injection_attempt(self, text: str) -> bool:
        patterns = [
            r"ignore previous", r"ignore all", r"disregard",
            r"you are now", r"pretend you", r"act as",
            r"system prompt", r"jailbreak", r"DAN mode",
        ]
        lower = text.lower()
        return any(re.search(p, lower) for p in patterns)

    # ══════════════════════════════════════════════════════════════════════════
    # EXTRACTORS
    # ══════════════════════════════════════════════════════════════════════════
    def _extract_amount(self, text: str) -> float:
        patterns = [
            r"₹\s*([\d,]+(?:\.\d+)?)\s*(?:k|K|lakh|L|cr)?",
            r"[Rr][Ss]\.?\s*([\d,]+(?:\.\d+)?)\s*(?:k|K|lakh|L)?",
            r"([\d,]+(?:\.\d+)?)\s*(?:rupees?|rs\.?)",
            r"\b([\d,]+(?:\.\d+)?)\b",
        ]
        multipliers = {"k": 1000, "K": 1000, "lakh": 100000, "L": 100000, "cr": 10000000}
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                try:
                    val = float(m.group(1).replace(",", ""))
                    if val > 0:
                        # Check for multiplier suffix
                        suffix_match = re.search(r"([\d,]+)\s*(k|K|lakh|L|cr)\b", text)
                        if suffix_match:
                            val = float(suffix_match.group(1).replace(",", ""))
                            val *= multipliers.get(suffix_match.group(2), 1)
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
        if "last month" in lower:
            first = today.replace(day=1) - timedelta(days=1)
            return first.replace(day=1).isoformat()
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
        m = re.search(r"([\d.]+)\s*(?:gram|grams|gm|g\b|kg)", text, re.I)
        if m:
            val = float(m.group(1))
            if "kg" in m.group().lower():
                val *= 1000
            return val
        m = re.search(r"([\d.]+)\s*(?:sqft|sq\.?ft|square feet)", text, re.I)
        if m:
            return float(m.group(1))
        m = re.search(r"([\d.]+)", text)
        return float(m.group(1)) if m else 1.0

    def _extract_fund_name(self, text: str) -> Optional[str]:
        m = re.search(
            r"\b(?:in|into|for)\s+([A-Za-z][A-Za-z0-9\s&]+?)(?:\s+(?:fund|mutual|for|at|₹|rs|sip)|\s*$)",
            text, re.I
        )
        if m:
            name = m.group(1).strip()
            if 2 < len(name) < 40:
                return name
        return None

    def _extract_multiple_funds(self, text: str) -> List[str]:
        """Extract multiple fund names from a comma/and separated list."""
        # Pattern: "in HDFC, ICICI, and Nippon" or "in HDFC and ICICI"
        m = re.search(r"\bin\s+((?:[A-Za-z][A-Za-z0-9\s&]+?)(?:,\s*(?:and\s+)?[A-Za-z][A-Za-z0-9\s&]+)*)", text, re.I)
        if m:
            raw = m.group(1)
            # Split by comma and "and"
            parts = re.split(r",\s*(?:and\s+)?|\s+and\s+", raw)
            funds = [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]
            if len(funds) > 1:
                return funds
        return []

    def _extract_months_back(self, text: str) -> int:
        """Extract how many months back a SIP has been running."""
        m = re.search(r"(\d+)\s*months?\s*(?:ago|back|since|for)", text, re.I)
        if m:
            return int(m.group(1))
        # "since January" type
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }
        today = date.today()
        for mon_name, mon_num in months.items():
            if mon_name in text.lower():
                diff = (today.month - mon_num) % 12
                return max(1, diff)
        return 0

    def _extract_payment_name(self, text: str) -> Optional[str]:
        m = re.search(r"(?:pay|paying|payment for|pay for)\s+([A-Za-z][A-Za-z0-9\s]+?)(?:\s+(?:every|for|of|₹|rs|\d)|$)", text, re.I)
        if m:
            return m.group(1).strip()
        return None

    def _extract_account_name(self, text: str) -> Optional[str]:
        m = re.search(r"(?:in|into|to|for)\s+([A-Za-z][A-Za-z0-9\s]+?)(?:\s+(?:account|fund|wallet|goal)|$)", text, re.I)
        if m:
            return m.group(1).strip()
        return None

    def _extract_source(self, text: str) -> Optional[str]:
        m = re.search(r"(?:from|via|by|as)\s+([A-Za-z][A-Za-z0-9\s]+?)(?:\s+|\.|$)", text, re.I)
        if m:
            return m.group(1).strip()
        return "Salary"

    def _detect_expense_category(self, lower: str) -> str:
        rules = [
            ("food",          r"food|lunch|dinner|breakfast|chai|tea|coffee|restaurant|snack|meal|biryani|swiggy|zomato|eat|drink|hotel"),
            ("transport",     r"petrol|fuel|bus|auto|cab|uber|ola|train|metro|toll|diesel|rapido|travel|ticket|fare"),
            ("shopping",      r"shirt|clothes|shopping|amazon|flipkart|dress|shoes|bag|mall|cloth|purchase"),
            ("utilities",     r"electricity|bill|internet|mobile|recharge|wifi|gas|water|broadband"),
            ("health",        r"doctor|medicine|hospital|medical|pharmacy|clinic|tablet|dental"),
            ("entertainment", r"movie|game|netflix|spotify|prime|concert|event|pub|bar"),
            ("education",     r"course|book|college|school|fee|tuition|coaching|udemy"),
            ("rent",          r"rent|hostel|pg|accommodation|house|flat|room"),
        ]
        for cat, pattern in rules:
            if re.search(pattern, lower):
                return cat
        return "other"

    def _detect_investment_type(self, lower: str) -> str:
        if re.search(r"mutual fund|mf|sip|flexi|bluechip|midcap|smallcap|largecap|debt fund|hybrid|nav", lower):
            return "mutual_fund"
        if re.search(r"stock|share|nifty|sensex|equity|ipo|bse|nse|zerodha|groww", lower):
            return "stocks"
        if re.search(r"crypto|bitcoin|btc|eth|ethereum|solana|usdt", lower):
            return "crypto"
        if re.search(r"bond|debenture|g-sec|government", lower):
            return "bonds"
        if re.search(r"\bppf\b|public provident", lower):
            return "ppf"
        if re.search(r"\bfd\b|fixed deposit|\brd\b|recurring deposit", lower):
            return "fd"
        if re.search(r"\bnps\b|national pension", lower):
            return "nps"
        return "other"

    def _detect_asset_type(self, lower: str) -> str:
        if "gold" in lower:
            return "gold"
        if "silver" in lower:
            return "silver"
        if re.search(r"property|land|house|flat|plot", lower):
            return "property"
        if re.search(r"vehicle|car|bike|motorcycle|scooter", lower):
            return "vehicle"
        if "jewelry" in lower or "jewel" in lower:
            return "jewelry"
        return "other"

    def _detect_frequency(self, lower: str) -> str:
        if re.search(r"weekly|every week", lower):
            return "weekly"
        if re.search(r"yearly|annual|every year", lower):
            return "yearly"
        if "quarterly" in lower:
            return "quarterly"
        return "monthly"

    def _detect_recurring_category(self, lower: str) -> str:
        if re.search(r"rent|hostel|pg|house|flat|room", lower):
            return "housing"
        if re.search(r"electricity|internet|mobile|wifi|gas|water|bill", lower):
            return "utilities"
        if re.search(r"emi|loan|mortgage", lower):
            return "loan"
        if re.search(r"netflix|spotify|prime|subscription", lower):
            return "subscriptions"
        if re.search(r"insurance|lic|policy|premium", lower):
            return "insurance"
        return "other"

    # ══════════════════════════════════════════════════════════════════════════
    # FINANCIAL HEALTH SCORE
    # ══════════════════════════════════════════════════════════════════════════
    def compute_health_score(self, user_data: Dict) -> Dict:
        summary = user_data.get("summary", {})
        expenses    = user_data.get("expenses", [])
        investments = user_data.get("investments", [])
        assets      = user_data.get("assets", [])
        savings     = user_data.get("savings", [])
        recurring   = user_data.get("recurring", [])
        debts       = user_data.get("debts", [])
        income      = user_data.get("income", [])
        sips        = user_data.get("sips", [])

        total_income     = summary.get("total_income", sum(i.get("amount", 0) for i in income))
        total_expenses   = summary.get("total_expenses", sum(e.get("amount", 0) for e in expenses))
        total_invested   = summary.get("total_invested", sum(i.get("invested_amount", 0) for i in investments))
        total_portfolio  = summary.get("total_portfolio_value", total_invested)
        total_assets     = summary.get("total_asset_value", sum(a.get("current_value", 0) for a in assets))
        total_savings    = summary.get("total_savings", sum(s.get("current_amount", 0) for s in savings))
        total_debt       = summary.get("total_debt", sum(d.get("remaining", d.get("principal", 0)) for d in debts))
        net_worth        = summary.get("net_worth", 0)
        monthly_recurring = sum(r.get("amount", 0) for r in recurring if r.get("frequency") == "monthly")

        scores = {}

        # 1. Savings rate (0-25)
        total_cash_flow = total_income + total_expenses
        if total_income > 0:
            savings_rate = (total_savings / total_income * 100)
            scores["savings_rate"] = min(25, int(savings_rate * 0.5))
        elif total_savings > 0:
            scores["savings_rate"] = 10
        else:
            scores["savings_rate"] = 0

        # 2. Expense control (0-25): recurring vs income ratio
        if total_income > 0:
            expense_ratio = total_expenses / total_income
            scores["expense_control"] = max(0, min(25, int((1 - expense_ratio) * 30)))
        elif total_expenses > 0:
            scores["expense_control"] = 5
        else:
            scores["expense_control"] = 0

        # 3. Investment diversity (0-25): variety + size
        inv_types = set(i.get("investment_type", "other") for i in investments)
        active_sips = len([s for s in sips if s.get("is_active")])
        inv_score = min(15, len(inv_types) * 5) + min(10, active_sips * 3)
        if total_portfolio > 50000:
            inv_score = min(25, inv_score + 3)
        scores["investment_diversity"] = inv_score

        # 4. Debt management (0-25): low debt relative to assets
        if net_worth > 0:
            debt_ratio = total_debt / (net_worth + total_debt + 1)
            scores["debt_management"] = max(0, min(25, int((1 - debt_ratio) * 25)))
        elif total_debt == 0:
            scores["debt_management"] = 15
        else:
            scores["debt_management"] = 0

        total_score = sum(scores.values())

        if total_score >= 80:
            grade, summary_text = "A+", "Exceptional financial health! You're building wealth strategically."
        elif total_score >= 70:
            grade, summary_text = "A", "Excellent! Strong savings, diverse investments, and managed debt."
        elif total_score >= 60:
            grade, summary_text = "B", "Good financial health. Small improvements can push you to excellent."
        elif total_score >= 45:
            grade, summary_text = "C", "Moderate. Focus on increasing investments and reducing debt."
        elif total_score >= 25:
            grade, summary_text = "D", "Needs work. Prioritize emergency fund and debt reduction."
        else:
            grade, summary_text = "F", "Getting started. Every rupee tracked is a step forward."

        return {
            "score": total_score,
            "grade": grade,
            "summary": summary_text,
            "breakdown": scores,
            "max_score": 100,
            "areas_to_improve": self._get_improvement_areas(scores, user_data),
        }

    def _get_improvement_areas(self, scores: Dict, user_data: Dict) -> List[str]:
        areas = []
        if scores.get("savings_rate", 0) < 15:
            areas.append("Increase savings rate — aim for 20% of income")
        if scores.get("expense_control", 0) < 12:
            areas.append("Reduce discretionary spending")
        if scores.get("investment_diversity", 0) < 12:
            areas.append("Diversify investments across mutual funds, stocks, and gold")
        if scores.get("debt_management", 0) < 12:
            areas.append("Pay down high-interest debt faster")
        return areas[:3]

    # ══════════════════════════════════════════════════════════════════════════
    # INSIGHTS GENERATOR
    # ══════════════════════════════════════════════════════════════════════════
    def generate_insights(self, user_data: Dict) -> List[str]:
        summary = user_data.get("summary", {})
        expenses = user_data.get("expenses", [])
        investments = user_data.get("investments", [])
        savings = user_data.get("savings", [])
        recurring = user_data.get("recurring", [])
        sips = user_data.get("sips", [])

        insights = []
        today = datetime.now()
        current_month = f"{today.year}-{str(today.month).zfill(2)}"

        # Monthly spend pace
        month_expenses = summary.get("month_expenses", 0)
        if month_expenses > 0 and today.day > 0:
            daily_avg = month_expenses / today.day
            projected = daily_avg * 30
            insights.append(f"At this pace, monthly spend will be ₹{projected:,.0f}")

        # Category breakdown
        cat_totals = defaultdict(float)
        for e in expenses[-50:]:  # Last 50 expenses
            cat_totals[e.get("category", "other")] += e.get("amount", 0)
        if cat_totals:
            top_cat, top_val = max(cat_totals.items(), key=lambda x: x[1])
            insights.append(f"Top spending: {top_cat} at ₹{top_val:,.0f} total")

        # Portfolio performance
        total_invested = summary.get("total_invested", 0)
        total_portfolio = summary.get("total_portfolio_value", 0)
        if total_invested > 0:
            gain_pct = summary.get("portfolio_gain_pct", 0)
            direction = "📈 up" if gain_pct >= 0 else "📉 down"
            insights.append(f"Portfolio is {direction} {abs(gain_pct):.1f}% (₹{abs(total_portfolio - total_invested):,.0f})")

        # SIP summary
        active_sips = [s for s in sips if s.get("is_active")]
        if active_sips:
            total_sip = sum(s.get("amount", 0) for s in active_sips)
            insights.append(f"{len(active_sips)} active SIPs investing ₹{total_sip:,.0f}/month")

        # Upcoming recurring
        today_str = date.today().isoformat()
        upcoming = [r for r in recurring if r.get("next_due", "") >= today_str]
        if upcoming:
            next_due = upcoming[0]
            insights.append(f"Next payment: {next_due.get('name')} ₹{next_due.get('amount', 0):,.0f} on {next_due.get('next_due')}")

        # Savings progress
        for goal in savings[:2]:
            target = goal.get("target_amount", 0)
            current = goal.get("current_amount", 0)
            if target > 0:
                pct = min(100, current / target * 100)
                insights.append(f"Goal '{goal.get('goal_name')}': {pct:.0f}% complete (₹{current:,.0f}/₹{target:,.0f})")

        return insights[:6]

    # ══════════════════════════════════════════════════════════════════════════
    # PREDICTIONS / FORECAST
    # ══════════════════════════════════════════════════════════════════════════
    def generate_predictions(self, user_data: Dict) -> Dict:
        expenses = user_data.get("expenses", [])
        investments = user_data.get("investments", [])
        summary = user_data.get("summary", {})
        sips = user_data.get("sips", [])
        recurring = user_data.get("recurring", [])

        today = datetime.now()

        # Monthly expense history
        monthly_data = {}
        for i in range(5, -1, -1):
            mo = (today.month - i - 1) % 12 + 1
            yr = today.year if (today.month - i) > 0 else today.year - 1
            key = f"{yr}-{str(mo).zfill(2)}"
            monthly_data[key] = sum(
                e.get("amount", 0) for e in expenses
                if (e.get("date") or "").startswith(key)
            )

        values = list(monthly_data.values())
        non_zero = [v for v in values if v > 0]
        avg_monthly = sum(non_zero) / len(non_zero) if non_zero else 0

        # Linear trend
        slope = 0
        if len(non_zero) >= 2:
            n = len(non_zero)
            x_mean = (n - 1) / 2
            y_mean = sum(non_zero) / n
            num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(non_zero))
            den = sum((i - x_mean) ** 2 for i in range(n))
            slope = num / den if den != 0 else 0

        # 6-month expense forecast
        expense_forecast = []
        for i in range(1, 7):
            new_mo = (today.month + i - 1) % 12 + 1
            new_yr = today.year + (today.month + i - 1) // 12
            label = datetime(new_yr, new_mo, 1).strftime("%b '%y")
            predicted = max(0, round(avg_monthly + slope * i))
            expense_forecast.append({"month": label, "predicted": predicted, "baseline": round(avg_monthly)})

        # Investment projection (SIP compounding)
        total_portfolio = summary.get("total_portfolio_value", 0)
        monthly_sip_total = sum(s.get("amount", 0) for s in sips if s.get("is_active"))

        invest_forecast = []
        base = total_portfolio
        for i in range(1, 13):
            new_mo = (today.month + i - 1) % 12 + 1
            new_yr = today.year + (today.month + i - 1) // 12
            label = datetime(new_yr, new_mo, 1).strftime("%b '%y")
            # Compound at 12% annual (1% per month) + monthly SIP
            base = base * 1.01 + monthly_sip_total
            invest_forecast.append({"month": label, "projected": round(base), "sip_contribution": monthly_sip_total})

        # Net worth projection
        net_worth = summary.get("net_worth", 0)
        nw_forecast = []
        nw = net_worth
        for i in range(1, 7):
            new_mo = (today.month + i - 1) % 12 + 1
            new_yr = today.year + (today.month + i - 1) // 12
            label = datetime(new_yr, new_mo, 1).strftime("%b '%y")
            # Net worth grows by portfolio appreciation minus predicted expenses
            nw += monthly_sip_total * 1.01 - max(0, avg_monthly - summary.get("month_income", 0) / today.month * 30)
            nw_forecast.append({"month": label, "net_worth": round(nw)})

        return {
            "expense_history": [{"month": k, "amount": v} for k, v in monthly_data.items()],
            "expense_forecast": expense_forecast,
            "investment_projection": invest_forecast,
            "net_worth_forecast": nw_forecast,
            "predicted_monthly_expense": round(avg_monthly + slope),
            "predicted_sip_corpus_12mo": round(invest_forecast[-1]["projected"] if invest_forecast else base),
            "trend": "increasing" if slope > 100 else "decreasing" if slope < -100 else "stable",
            "monthly_sip_total": monthly_sip_total,
            "ai_recommendation": self._generate_recommendation(user_data, avg_monthly, slope),
        }

    def _generate_recommendation(self, user_data: Dict, avg_expense: float, trend: float) -> str:
        summary = user_data.get("summary", {})
        sips = user_data.get("sips", [])
        debts = user_data.get("debts", [])

        tips = []
        net_worth = summary.get("net_worth", 0)
        total_debt = summary.get("total_debt", 0)
        total_invested = summary.get("total_invested", 0)
        active_sips = len([s for s in sips if s.get("is_active")])

        if trend > 200:
            tips.append(f"Expenses growing by ₹{trend:.0f}/month — review discretionary spending.")
        if total_debt > net_worth * 0.4 and total_debt > 0:
            tips.append("Debt is >40% of net worth. Prioritize debt reduction.")
        if active_sips == 0 and total_invested < 10000:
            tips.append("Start a monthly SIP of at least ₹500 to build long-term wealth.")
        if active_sips > 0 and active_sips < 3:
            tips.append("Consider diversifying across 3-4 SIPs in different fund categories.")
        if avg_expense > 0 and summary.get("total_savings", 0) < avg_expense * 3:
            tips.append("Build an emergency fund of 3-6 months of expenses before aggressive investing.")

        if not tips:
            return ("Your financial trajectory looks healthy. Continue regular SIP investments and maintain emergency reserves. Consider annual portfolio rebalancing.")
        return " ".join(tips)
