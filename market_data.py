"""
AstraOS — Market Data Service v3.0
Fetches real-time: Gold (INR/gram), Silver (INR/gram), Mutual Fund NAV
Uses multiple free APIs with fallbacks
"""
import httpx
import asyncio
import json
from datetime import datetime, date
from typing import Optional, Dict


# Known mutual fund codes (AMFI India)
KNOWN_FUNDS = {
    "hdfc": "100033",           # HDFC Top 100 Fund
    "hdfc top": "100033",
    "hdfc flexi": "100025",
    "icici": "120503",          # ICICI Pru Bluechip
    "icici bluechip": "120503",
    "icici pru": "120503",
    "nippon": "118825",         # Nippon India Large Cap
    "nippon india": "118825",
    "axis": "120503",
    "axis bluechip": "120503",
    "sbi": "125494",            # SBI Bluechip
    "sbi bluechip": "125494",
    "mirae": "118989",
    "mirae asset": "118989",
    "parag parikh": "122639",
    "ppfas": "122639",
}

# Gold price sources (free)
GOLD_SOURCES = [
    "https://api.metals.live/v1/spot/gold",          # metals.live
    "https://data.fixer.io/api/latest",               # fallback
]


class MarketDataService:

    def __init__(self, db=None):
        self.db = db
        self._gold_cache: Optional[float] = None
        self._silver_cache: Optional[float] = None
        self._cache_date: Optional[str] = None

    async def fetch_all(self) -> Dict:
        """Fetch gold, silver prices and return dict of symbol→price."""
        results = {}
        today = date.today().isoformat()

        # Check cache (refresh once per day)
        if self._cache_date == today and self._gold_cache:
            results["GOLD_INR_GRAM"] = self._gold_cache
            results["SILVER_INR_GRAM"] = self._silver_cache
            return results

        gold = await self._fetch_gold_inr()
        silver = await self._fetch_silver_inr()

        if gold:
            results["GOLD_INR_GRAM"] = gold
            self._gold_cache = gold
        if silver:
            results["SILVER_INR_GRAM"] = silver
            self._silver_cache = silver
        self._cache_date = today

        # Save to DB
        if self.db:
            for symbol, price in results.items():
                self.db.upsert_market_price(symbol, "commodity", price, "INR", "api")

        return results

    async def _fetch_gold_inr(self) -> Optional[float]:
        """Fetch gold price in INR per gram."""
        # Try metals.live (USD/troy oz) then convert to INR/gram
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.get("https://api.metals.live/v1/spot/gold")
                if r.status_code == 200:
                    data = r.json()
                    usd_per_oz = data.get("price") or (data[0]["price"] if isinstance(data, list) else None)
                    if usd_per_oz:
                        usd_inr = await self._get_usd_inr_rate()
                        inr_per_gram = (usd_per_oz * usd_inr) / 31.1035
                        return round(inr_per_gram, 2)
        except Exception:
            pass

        # Fallback: use goldapi.io style calculation with known approximate
        # Gold ~$2300/oz, USD/INR ~83.5, = $2300*83.5/31.1035 ≈ ₹6180/gram
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.get(
                    "https://www.goldapi.io/api/XAU/INR",
                    headers={"x-access-token": "goldapi-demo-key"}
                )
                if r.status_code == 200:
                    data = r.json()
                    price_oz = data.get("price")
                    if price_oz:
                        return round(price_oz / 31.1035, 2)
        except Exception:
            pass

        # Final fallback: approximate current market rate
        # Gold ~₹7200/gram (2025 rate)
        return 7200.0

    async def _fetch_silver_inr(self) -> Optional[float]:
        """Fetch silver price in INR per gram."""
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.get("https://api.metals.live/v1/spot/silver")
                if r.status_code == 200:
                    data = r.json()
                    usd_per_oz = data.get("price") or (data[0]["price"] if isinstance(data, list) else None)
                    if usd_per_oz:
                        usd_inr = await self._get_usd_inr_rate()
                        inr_per_gram = (usd_per_oz * usd_inr) / 31.1035
                        return round(inr_per_gram, 2)
        except Exception:
            pass
        # Fallback: ~₹85/gram (2025)
        return 85.0

    async def _get_usd_inr_rate(self) -> float:
        """Fetch USD/INR exchange rate."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    "https://api.exchangerate-api.com/v4/latest/USD"
                )
                if r.status_code == 200:
                    data = r.json()
                    return data.get("rates", {}).get("INR", 83.5)
        except Exception:
            pass
        return 83.5  # fallback

    async def fetch_mutual_fund_nav(self, fund_name: str) -> Optional[Dict]:
        """Fetch NAV for a mutual fund from AMFI India (free, no auth)."""
        fund_lower = fund_name.lower().strip()
        scheme_code = None

        # Look up known fund codes
        for key, code in KNOWN_FUNDS.items():
            if key in fund_lower:
                scheme_code = code
                break

        if not scheme_code:
            # Try searching AMFI
            scheme_code = await self._search_amfi_fund(fund_name)

        if not scheme_code:
            return None

        try:
            url = f"https://api.mfapi.in/mf/{scheme_code}"
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.get(url)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("status") == "SUCCESS" and data.get("data"):
                        latest = data["data"][0]
                        nav = float(latest.get("nav", 0))
                        scheme_name = data.get("meta", {}).get("scheme_name", fund_name)
                        return {
                            "nav": nav,
                            "scheme_name": scheme_name,
                            "scheme_code": scheme_code,
                            "date": latest.get("date"),
                            "fund_name": fund_name,
                        }
        except Exception:
            pass
        return None

    async def _search_amfi_fund(self, fund_name: str) -> Optional[str]:
        """Search AMFI for a fund code by name."""
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.get("https://api.mfapi.in/mf/search",
                                     params={"q": fund_name})
                if r.status_code == 200:
                    results = r.json()
                    if results:
                        return str(results[0].get("schemeCode"))
        except Exception:
            pass
        return None

    async def update_portfolio_values(self, db, user_id: str) -> Dict:
        """Update current values for all user investments based on latest NAV."""
        investments = db.get_investments(user_id)
        updated = []
        for inv in investments:
            if inv.get("investment_type") in ("mutual_fund", "sip"):
                nav_data = await self.fetch_mutual_fund_nav(inv["fund_name"])
                if nav_data and nav_data.get("nav") and inv.get("units", 0) > 0:
                    new_value = nav_data["nav"] * inv["units"]
                    db.update_investment_value(user_id, inv["id"], new_value, nav_data["nav"])
                    updated.append({
                        "fund": inv["fund_name"],
                        "nav": nav_data["nav"],
                        "new_value": new_value
                    })
        return {"updated": updated, "count": len(updated)}

    async def update_asset_values(self, db, user_id: str) -> Dict:
        """Update gold/silver asset values based on current market prices."""
        prices = await self.fetch_all()
        gold_price = prices.get("GOLD_INR_GRAM", 7200)
        silver_price = prices.get("SILVER_INR_GRAM", 85)

        assets = db.get_assets(user_id)
        updated = []
        for asset in assets:
            if asset["asset_type"] == "gold" and asset.get("quantity", 0) > 0:
                new_val = asset["quantity"] * gold_price
                db.update_asset_value(user_id, asset["id"], new_val)
                updated.append({"type": "gold", "quantity": asset["quantity"],
                                  "price": gold_price, "value": new_val})
            elif asset["asset_type"] == "silver" and asset.get("quantity", 0) > 0:
                new_val = asset["quantity"] * silver_price
                db.update_asset_value(user_id, asset["id"], new_val)
                updated.append({"type": "silver", "quantity": asset["quantity"],
                                  "price": silver_price, "value": new_val})

        return {"updated": updated, "gold_price": gold_price, "silver_price": silver_price}


# Singleton
_market_service: Optional[MarketDataService] = None


def get_market_service(db=None) -> MarketDataService:
    global _market_service
    if _market_service is None:
        _market_service = MarketDataService(db)
    return _market_service
