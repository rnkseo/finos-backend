from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urldefrag
import os
import json
import re
import time
from collections import Counter
from datetime import datetime, date

app = FastAPI(title="SearchRNK SERP Compare API v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.searchrnk.com", "https://searchrnk.com", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}

# ──────────────────────────────────────────────────────────────
# TOKEN USAGE TRACKER (in-memory, daily reset)
# ──────────────────────────────────────────────────────────────

_token_tracker = {
    "date": str(date.today()),
    "tokens_used": 0,
    "daily_limit": 120_000,   # conservative daily Groq limit
    "model_primary": "llama-3.3-70b-versatile",
    "model_fallback": "llama-3.1-8b-instant",
    "threshold_pct": 0.80,
}

def _reset_if_new_day():
    today = str(date.today())
    if _token_tracker["date"] != today:
        _token_tracker["date"] = today
        _token_tracker["tokens_used"] = 0

def get_active_model() -> str:
    _reset_if_new_day()
    pct = _token_tracker["tokens_used"] / _token_tracker["daily_limit"]
    return _token_tracker["model_primary"] if pct < _token_tracker["threshold_pct"] else _token_tracker["model_fallback"]

def record_tokens(n: int):
    _reset_if_new_day()
    _token_tracker["tokens_used"] += n


# ──────────────────────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    try:
        url, _ = urldefrag(url)
        p = urlparse(url)
        path = p.path.rstrip("/") or "/"
        return f"{p.scheme.lower()}://{p.netloc.lower()}{path}"
    except Exception:
        return url


def detect_intent_ai(url: str, title: str, h1: str, meta: str) -> str:
    combined = f"{url} {title} {h1} {meta}".lower()
    buy_signals = ["buy", "shop", "price", "order", "purchase", "cart", "checkout", "add to cart", "get started", "sign up", "subscribe", "pricing", "plan"]
    commercial_signals = ["best", "top", "review", "compare", "vs", "versus", "ranking", "rated", "alternatives", "worth it", "pros and cons", "which is better"]
    navigational_signals = ["login", "sign in", "account", "dashboard", "portal", "support", "contact"]
    info_signals = ["how to", "what is", "guide", "tutorial", "learn", "tips", "examples", "definition", "explained", "why", "when", "who"]

    scores = {
        "Transactional": sum(2 for s in buy_signals if s in combined),
        "Commercial": sum(2 for s in commercial_signals if s in combined),
        "Navigational": sum(2 for s in navigational_signals if s in combined),
        "Informational": sum(1 for s in info_signals if s in combined),
    }
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "Informational"


def detect_content_type(h2_count: int, word_count: int, title: str, h1: str) -> str:
    combined = f"{title} {h1}".lower()
    if any(w in combined for w in ["top ", "best ", "list of", " ways", "tips ", "reasons", "examples", "tools", "plugins", "resources"]):
        return "Listicle"
    if any(w in combined for w in ["how to", "guide", "tutorial", "step by step", "setup", "install", "configure"]):
        return "How-to Guide"
    if any(w in combined for w in ["review", "vs", "versus", "compared", "comparison", "alternatives"]):
        return "Comparison / Review"
    if any(w in combined for w in ["what is", "definition", "meaning", "explained", "overview"]):
        return "Explainer"
    if h2_count >= 8 and word_count >= 2500:
        return "Pillar / Long-form"
    if word_count < 700:
        return "Short-form"
    return "Article"


def compute_readability(text: str) -> dict:
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) > 3]
    if not sentences:
        return {"label": "Unknown", "avg_words": 0}
    avg = sum(len(s.split()) for s in sentences) / len(sentences)
    long_words = sum(1 for w in text.split() if len(w) > 7)
    long_pct = long_words / max(len(text.split()), 1) * 100
    label = "Easy" if avg < 14 and long_pct < 20 else "Moderate" if avg < 24 else "Hard"
    return {"label": label, "avg_words": round(avg, 1), "complex_word_pct": round(long_pct, 1)}


def detect_schema_full(soup) -> dict:
    schemas = []
    raw_schemas = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            raw = script.string or "{}"
            data = json.loads(raw)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    t = item.get("@type", "")
                    if t:
                        schemas.append(str(t))
                    # Capture key fields
                    preview = {k: v for k, v in item.items() if k in ["@type", "name", "description", "url", "image", "author", "datePublished", "headline"]}
                    raw_schemas.append(preview)
        except Exception:
            pass
    return {"types": list(set(schemas)), "count": len(schemas), "previews": raw_schemas[:5]}


def extract_date(soup) -> str:
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        return time_tag.get("datetime", "")[:10]
    for prop in ["article:published_time", "og:article:published_time", "article:modified_time", "article:published"]:
        m = soup.find("meta", property=prop)
        if m and m.get("content"):
            return m["content"][:10]
    for itemprop in ["datePublished", "dateModified"]:
        m = soup.find(attrs={"itemprop": itemprop})
        if m:
            val = m.get("content") or m.get_text(strip=True)
            if val:
                return str(val)[:10]
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "{}")
            if isinstance(data, dict):
                date_val = data.get("datePublished") or data.get("dateModified")
                if date_val:
                    return str(date_val)[:10]
        except Exception:
            pass
    return "Not Found"


def extract_keyword_intelligence(text: str, keyword: str) -> dict:
    words = re.findall(r"\b[a-z]+\b", text.lower())
    kw_lower = keyword.lower()
    kw_parts = set(kw_lower.split())

    # N-grams
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]
    fourgrams = [f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}" for i in range(len(words) - 3)]

    # Semantic variations (n-grams containing keyword parts)
    relevant_bi = Counter(ng for ng in bigrams if any(kp in ng for kp in kw_parts) and ng != kw_lower)
    relevant_tri = Counter(ng for ng in trigrams if any(kp in ng for kp in kw_parts) and ng != kw_lower)
    relevant_four = Counter(ng for ng in fourgrams if any(kp in ng for kp in kw_parts))

    variations = (
        [ng for ng, _ in relevant_tri.most_common(5)] +
        [ng for ng, _ in relevant_bi.most_common(5)] +
        [ng for ng, _ in relevant_four.most_common(3)]
    )
    # Deduplicate
    seen = set()
    unique_vars = []
    for v in variations:
        if v not in seen and v != kw_lower:
            seen.add(v)
            unique_vars.append(v)

    # Topic clusters (frequent multi-word phrases)
    all_trigrams = Counter(trigrams)
    stop_words = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old", "see", "two", "way", "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use", "with", "that", "this", "from", "they", "will", "been", "have", "here", "more", "also", "into", "than", "then", "them", "some", "what", "when", "your", "said"}
    topic_phrases = [ng for ng, cnt in all_trigrams.most_common(30) if cnt >= 2 and not any(w in stop_words for w in ng.split())]

    return {
        "primary": keyword,
        "variations": unique_vars[:12],
        "topic_clusters": topic_phrases[:8],
        "total_variation_count": len(unique_vars),
    }


def compute_ai_seo_score(data: dict, keyword: str, comp_data: list = None) -> dict:
    """
    Context-aware, multi-dimensional SEO scoring.
    Returns score + breakdown by dimension.
    """
    dims = {
        "content_relevance": 0,   # max 25
        "on_page_signals": 0,     # max 20
        "structure_depth": 0,     # max 15
        "technical_health": 0,    # max 20
        "content_quality": 0,     # max 10
        "ux_signals": 0,          # max 10
    }

    # ── Content Relevance (25pts) ──
    wc = data.get("word_count", 0)
    intent = data.get("intent", "Informational")

    # Intent-adjusted word count scoring
    if intent in ("Transactional", "Navigational"):
        # Short pages are fine for these
        if wc >= 500: dims["content_relevance"] += 10
        elif wc >= 200: dims["content_relevance"] += 6
    else:
        if wc >= 3000: dims["content_relevance"] += 15
        elif wc >= 2000: dims["content_relevance"] += 12
        elif wc >= 1200: dims["content_relevance"] += 8
        elif wc >= 600: dims["content_relevance"] += 4
        elif wc >= 200: dims["content_relevance"] += 2

    kd = data.get("keyword_density", 0)
    if 0.3 <= kd <= 3.0: dims["content_relevance"] += 8
    elif 0.1 <= kd <= 4.0: dims["content_relevance"] += 4

    kw_count = data.get("keyword_count", 0)
    if kw_count >= 5: dims["content_relevance"] += 2

    # Keyword in H2/H3 headings
    if data.get("keyword_in_h2"): dims["content_relevance"] += 3
    if data.get("keyword_in_h3"): dims["content_relevance"] += 2

    # ── On-Page Signals (20pts) ──
    if data.get("keyword_in_title"): dims["on_page_signals"] += 8
    if data.get("keyword_in_h1"): dims["on_page_signals"] += 7
    if data.get("keyword_in_meta"): dims["on_page_signals"] += 5

    # Title length
    tl = data.get("title_length", 0)
    if 50 <= tl <= 60: dims["on_page_signals"] += 0  # already counted in keyword presence
    elif tl > 70: dims["on_page_signals"] -= 2

    # ── Structure Depth (15pts) ──
    h1c = data.get("h1_count", 0)
    h2c = data.get("h2_count", 0)
    h3c = data.get("h3_count", 0)

    if h1c == 1: dims["structure_depth"] += 4
    elif h1c == 0: dims["structure_depth"] -= 2

    if h2c >= 8: dims["structure_depth"] += 6
    elif h2c >= 5: dims["structure_depth"] += 4
    elif h2c >= 3: dims["structure_depth"] += 2
    elif h2c >= 1: dims["structure_depth"] += 1

    if h3c >= 5: dims["structure_depth"] += 5
    elif h3c >= 3: dims["structure_depth"] += 3
    elif h3c >= 1: dims["structure_depth"] += 1

    # ── Technical Health (20pts) ──
    if data.get("canonical"): dims["technical_health"] += 6
    if data.get("canonical_match"): dims["technical_health"] += 2

    og = data.get("og_tags", {})
    og_score = sum(1 for k in ["og:title", "og:description", "og:image", "og:type", "og:url"] if og.get(k))
    dims["technical_health"] += min(og_score * 1, 5)

    if data.get("og_complete"): dims["technical_health"] += 2

    speed = data.get("load_speed")
    if isinstance(speed, dict):
        mob = speed.get("mobile", {})
        if isinstance(mob, dict) and isinstance(mob.get("score"), int):
            ms = mob["score"]
            if ms >= 90: dims["technical_health"] += 5
            elif ms >= 70: dims["technical_health"] += 3
            elif ms >= 50: dims["technical_health"] += 1

    # ── Content Quality (10pts) ──
    r = data.get("readability", {})
    rl = r if isinstance(r, str) else r.get("label", "Unknown")
    dims["content_quality"] += {"Easy": 8, "Moderate": 6, "Hard": 3}.get(rl, 0)

    schema = data.get("schema_types", [])
    if schema: dims["content_quality"] += 2

    # ── UX Signals (10pts) ──
    imgs = data.get("image_count", 0)
    if imgs >= 8: dims["ux_signals"] += 5
    elif imgs >= 4: dims["ux_signals"] += 3
    elif imgs >= 1: dims["ux_signals"] += 1

    il = data.get("internal_links", 0)
    if il >= 15: dims["ux_signals"] += 5
    elif il >= 8: dims["ux_signals"] += 3
    elif il >= 3: dims["ux_signals"] += 1

    # Clamp each dimension
    dims["content_relevance"] = max(0, min(25, dims["content_relevance"]))
    dims["on_page_signals"] = max(0, min(20, dims["on_page_signals"]))
    dims["structure_depth"] = max(0, min(15, dims["structure_depth"]))
    dims["technical_health"] = max(0, min(20, dims["technical_health"]))
    dims["content_quality"] = max(0, min(10, dims["content_quality"]))
    dims["ux_signals"] = max(0, min(10, dims["ux_signals"]))

    total = sum(dims.values())
    return {"total": min(total, 100), "breakdown": dims}


def compute_ranking_probability(user_data: dict, competitors: list, keyword: str) -> dict:
    """Estimate ranking probability based on comparative signals."""
    if not competitors:
        return {"probability": 50, "label": "Moderate", "rationale": "No competitors to compare against."}

    user_score = user_data.get("seo_score", 0)
    comp_scores = [c.get("seo_score", 0) for c in competitors if c.get("seo_score", 0) > 0]
    avg_comp = sum(comp_scores) / len(comp_scores) if comp_scores else 50

    # Base probability from score gap
    gap = user_score - avg_comp
    base = 50 + (gap * 0.8)

    # Adjust for keyword signals
    if user_data.get("keyword_in_title") and user_data.get("keyword_in_h1"):
        base += 8
    if not user_data.get("keyword_in_title"):
        base -= 10
    if not user_data.get("canonical"):
        base -= 5
    if user_data.get("schema_types"):
        base += 4
    if user_data.get("og_complete"):
        base += 3

    speed = user_data.get("load_speed", {})
    if isinstance(speed, dict):
        mob = speed.get("mobile", {})
        if isinstance(mob, dict) and isinstance(mob.get("score"), int):
            if mob["score"] < 50:
                base -= 12
            elif mob["score"] >= 90:
                base += 6

    prob = max(5, min(95, round(base)))
    label = "High" if prob >= 70 else "Moderate" if prob >= 45 else "Low"
    return {"probability": prob, "label": label}


# ──────────────────────────────────────────────────────────────
# GROQ AI ANALYSIS
# ──────────────────────────────────────────────────────────────

async def call_groq_ai(comparison: dict) -> dict:
    """Call Groq API with smart model switching and token tracking."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return _fallback_ai(comparison)

    model = get_active_model()
    gaps = comparison.get("gaps", [])
    user_score = comparison.get("user_score", 0)
    comp_avg = comparison.get("competitor_avg_score", 0)
    keyword = comparison.get("keyword", "")
    comp_avgs = comparison.get("competitor_averages", {})
    intent = comparison.get("user_intent", "Informational")
    content_type = comparison.get("user_content_type", "Article")
    ranking_prob = comparison.get("ranking_probability", {})

    high_gaps = [g for g in gaps if g["severity"] == "high"]
    med_gaps = [g for g in gaps if g["severity"] == "medium"]
    low_gaps = [g for g in gaps if g["severity"] == "low"]

    system_prompt = """You are a 10+ year veteran SEO strategist who thinks contextually and strategically, not by mechanical rules.
You understand that ranking is about satisfying search intent better than competitors, not just technical checklists.
You provide expert, specific, actionable advice. You explain WHY things matter for ranking, not just WHAT to fix.
Always respond with valid JSON only — no markdown, no preamble."""

    user_prompt = f"""Analyze this SEO comparison and generate expert recommendations.

KEYWORD: "{keyword}"
SEARCH INTENT: {intent}
CONTENT FORMAT: {content_type}
USER PAGE SCORE: {user_score}/100
COMPETITOR AVG SCORE: {comp_avg}/100
RANKING PROBABILITY: {ranking_prob.get('probability', 50)}% ({ranking_prob.get('label', 'Moderate')})

HIGH-PRIORITY GAPS:
{json.dumps(high_gaps, indent=2)}

MEDIUM-PRIORITY GAPS:
{json.dumps(med_gaps, indent=2)}

COMPETITOR AVERAGES:
{json.dumps(comp_avgs, indent=2)}

Generate expert SEO recommendations. Think like an experienced SEO consultant analyzing why pages rank and what creates competitive advantage.

Return ONLY this JSON structure:
{{
  "executive_summary": "2-3 sentences: current state, key advantage/disadvantage, most important next step. Be specific.",
  "ranking_rationale": "1-2 sentences explaining WHY the page is or isn't positioned to rank well for this specific keyword and intent.",
  "high_impact": [
    {{"action": "specific action to take", "why": "why this will move rankings", "effort": "low|medium|high", "impact": "high"}}
  ],
  "medium_impact": [
    {{"action": "...", "why": "...", "effort": "low|medium|high", "impact": "medium"}}
  ],
  "low_priority": [
    {{"action": "...", "why": "...", "effort": "low|medium|high", "impact": "low"}}
  ],
  "content_strategy": "2-3 sentences on how to restructure/expand content to match search intent and beat competitors.",
  "quick_wins": ["actionable quick win 1 (under 30 mins)", "actionable quick win 2", "actionable quick win 3"],
  "competitive_advantages": ["things the user page does better than competitors"],
  "content_gaps": ["specific topic or section missing that competitors cover"]
}}"""

    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.4,
                    "max_tokens": 1800,
                },
                timeout=aiohttp.ClientTimeout(total=45)
            ) as resp:
                if resp.status != 200:
                    return _fallback_ai(comparison)
                data = await resp.json()
                usage = data.get("usage", {})
                record_tokens(usage.get("total_tokens", 500))

                content = data["choices"][0]["message"]["content"].strip()
                # Strip markdown code fences if present
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
                result = json.loads(content)
                result["_model"] = model
                result["_tokens"] = usage.get("total_tokens", 0)
                return result
    except Exception as e:
        return _fallback_ai(comparison)


def _fallback_ai(comparison: dict) -> dict:
    gaps = comparison.get("gaps", [])
    user_score = comparison.get("user_score", 0)
    comp_avg = comparison.get("competitor_avg_score", 0)
    keyword = comparison.get("keyword", "")
    diff = user_score - comp_avg
    direction = "ahead of" if diff >= 0 else "behind"
    delta = abs(diff)

    high = [g for g in gaps if g["severity"] == "high"]
    med = [g for g in gaps if g["severity"] == "medium"]
    low = [g for g in gaps if g["severity"] == "low"]

    return {
        "executive_summary": f"Your page scores {user_score}/100, {delta} points {direction} the competitor average of {comp_avg}/100. {len(high)} critical issues require immediate attention to improve ranking position.",
        "ranking_rationale": f"With {len(gaps)} total gaps identified, addressing the high-priority issues — especially keyword placement and technical signals — will have the most direct impact on rankings for '{keyword}'.",
        "high_impact": [{"action": g["message"], "why": f"Affects: {g['metric']}", "effort": "medium", "impact": "high"} for g in high[:4]],
        "medium_impact": [{"action": g["message"], "why": f"Affects: {g['metric']}", "effort": "medium", "impact": "medium"} for g in med[:4]],
        "low_priority": [{"action": g["message"], "why": f"Affects: {g['metric']}", "effort": "low", "impact": "low"} for g in low[:3]],
        "content_strategy": "Focus on matching the dominant content format of top-ranking pages, ensure comprehensive topic coverage, and align your content depth with the search intent for this keyword.",
        "quick_wins": ["Add primary keyword to title tag if missing", "Set up canonical tag to prevent duplicate indexing", "Complete Open Graph tags for social sharing"],
        "competitive_advantages": ["Your current strengths will update after full analysis"],
        "content_gaps": ["Run a full content audit to identify missing topic sections"],
        "_model": "fallback",
        "_tokens": 0,
    }


# ──────────────────────────────────────────────────────────────
# PAGESPEED
# ──────────────────────────────────────────────────────────────

async def run_pagespeed_full(url: str) -> dict:
    key = os.environ.get("PAGESPEED_API_KEY", "").strip()
    empty = {"score": None, "fcp": "N/A", "lcp": "N/A", "tbt": "N/A", "cls": "N/A", "si": "N/A", "rating": "No Data"}

    if not key:
        return {"mobile": dict(empty), "desktop": dict(empty)}

    def parse_metric(audits, audit_key, unit="s"):
        item = audits.get(audit_key, {})
        if "displayValue" in item:
            return item["displayValue"]
        if "numericValue" in item:
            val = item["numericValue"]
            return f"{int(val)} ms" if unit == "ms" else f"{val / 1000:.2f} s"
        return "N/A"

    def rating(s):
        if not isinstance(s, int): return "N/A"
        return "Fast" if s >= 90 else "Needs Work" if s >= 50 else "Slow"

    async def fetch_strategy(strategy):
        try:
            api_url = (
                f"https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
                f"?url={url}&strategy={strategy}&category=performance&key={key}"
            )
            async with aiohttp.ClientSession() as sess:
                async with sess.get(api_url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    data = await resp.json()
                    if "error" in data or "lighthouseResult" not in data:
                        return None
                    lh = data["lighthouseResult"]
                    audits = lh["audits"]
                    perf_score = lh["categories"]["performance"]["score"]
                    score = int(perf_score * 100) if perf_score is not None else 0
                    return {
                        "score": score,
                        "fcp": parse_metric(audits, "first-contentful-paint"),
                        "lcp": parse_metric(audits, "largest-contentful-paint"),
                        "tbt": parse_metric(audits, "total-blocking-time", "ms"),
                        "cls": parse_metric(audits, "cumulative-layout-shift", ""),
                        "si": parse_metric(audits, "speed-index"),
                        "rating": rating(score),
                    }
        except Exception:
            return None

    try:
        mob, desk = await asyncio.gather(
            fetch_strategy("mobile"), fetch_strategy("desktop"), return_exceptions=True
        )
    except Exception:
        mob, desk = None, None

    return {
        "mobile": mob if (mob and not isinstance(mob, Exception)) else dict(empty),
        "desktop": desk if (desk and not isinstance(desk, Exception)) else dict(empty),
    }


# ──────────────────────────────────────────────────────────────
# PAGE EXTRACTOR
# ──────────────────────────────────────────────────────────────

async def extract_full_page(session, url: str, keyword: str, manual: dict) -> dict:
    kw = keyword.lower().strip()
    result = {
        "url": url, "page_status": 0,
        "title": "", "title_length": 0,
        "meta_description": "", "meta_desc_length": 0,
        "h1": "", "h1_count": 0, "h2_count": 0, "h3_count": 0,
        "h2_headings": [], "h3_headings": [],
        "keyword_in_title": False, "keyword_in_h1": False,
        "keyword_in_meta": False, "keyword_in_h2": False, "keyword_in_h3": False,
        "keyword_count": 0, "keyword_density": 0.0,
        "keyword_intelligence": {},
        "word_count": 0, "intent": "Unknown", "content_type": "Unknown",
        "published_date": "Not Found", "readability": {"label": "Unknown", "avg_words": 0},
        "schema": {"types": [], "count": 0, "previews": []},
        "schema_types": [],  # kept for backward compat
        "internal_links": 0, "external_links": 0,
        "image_count": 0, "canonical": None, "canonical_match": False,
        "og_tags": {}, "og_complete": False,
        "load_speed": None, "seo_score": 0, "seo_score_breakdown": {},
        "da": manual.get("da", "N/A"),
        "pa": manual.get("pa", "N/A"),
        "backlinks": manual.get("backlinks", "N/A"),
        "plagiarism": manual.get("plagiarism", "N/A"),
    }

    try:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=25),
            ssl=False, headers=HEADERS, allow_redirects=True
        ) as resp:
            result["page_status"] = resp.status
            if resp.status != 200:
                return result
            text = await resp.text(errors="ignore")

        soup = BeautifulSoup(text, "html.parser")
        page_domain = urlparse(url).netloc

        # Title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        result["title"] = title[:300]
        result["title_length"] = len(title)
        result["keyword_in_title"] = kw in title.lower()

        # Meta description
        meta_desc_tag = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
        desc = meta_desc_tag.get("content", "").strip() if meta_desc_tag else ""
        result["meta_description"] = desc[:400]
        result["meta_desc_length"] = len(desc)
        result["keyword_in_meta"] = kw in desc.lower()

        # Headings
        h1_tags = soup.find_all("h1")
        result["h1_count"] = len(h1_tags)
        result["h1"] = h1_tags[0].get_text(strip=True)[:250] if h1_tags else ""
        result["keyword_in_h1"] = kw in result["h1"].lower()

        h2_tags = soup.find_all("h2")
        result["h2_count"] = len(h2_tags)
        result["h2_headings"] = [h.get_text(strip=True)[:150] for h in h2_tags[:20]]
        h2_text = " ".join(h.lower() for h in result["h2_headings"])
        result["keyword_in_h2"] = kw in h2_text

        h3_tags = soup.find_all("h3")
        result["h3_count"] = len(h3_tags)
        result["h3_headings"] = [h.get_text(strip=True)[:120] for h in h3_tags[:20]]
        h3_text = " ".join(h.lower() for h in result["h3_headings"])
        result["keyword_in_h3"] = kw in h3_text

        # Clean body text
        for tag in soup.find_all(["script", "style", "noscript", "nav", "header", "footer", "aside", "iframe"]):
            tag.decompose()
        clean_text = soup.get_text(" ", strip=True)
        words = clean_text.lower().split()
        result["word_count"] = len(words)

        # Keyword count & density
        kw_parts = kw.split()
        kw_count = sum(
            1 for i in range(len(words) - len(kw_parts) + 1)
            if words[i:i + len(kw_parts)] == kw_parts
        )
        result["keyword_count"] = kw_count
        result["keyword_density"] = round((kw_count / max(len(words), 1)) * 100, 2)
        result["keyword_intelligence"] = extract_keyword_intelligence(clean_text, keyword)

        # Metadata
        result["intent"] = detect_intent_ai(url, title, result["h1"], desc)
        result["content_type"] = detect_content_type(result["h2_count"], result["word_count"], title, result["h1"])
        result["readability"] = compute_readability(clean_text)
        result["published_date"] = extract_date(soup)

        # Schema
        schema_info = detect_schema_full(soup)
        result["schema"] = schema_info
        result["schema_types"] = schema_info["types"]  # backward compat

        # Links
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            if not href.startswith("http"):
                continue
            if urlparse(href).netloc == page_domain:
                result["internal_links"] += 1
            else:
                result["external_links"] += 1

        # Images
        result["image_count"] = len(soup.find_all("img"))

        # Canonical
        can_tag = soup.find("link", rel="canonical")
        if can_tag and can_tag.get("href"):
            result["canonical"] = can_tag["href"].strip()
            result["canonical_match"] = normalize_url(can_tag["href"].strip()) == normalize_url(url)

        # OG tags
        og_props = ["og:title", "og:description", "og:image", "og:type", "og:url"]
        og = {}
        for prop in og_props:
            m = soup.find("meta", property=prop)
            if m and m.get("content"):
                og[prop] = m["content"][:300]
        result["og_tags"] = og
        result["og_complete"] = bool(og.get("og:title") and og.get("og:description") and og.get("og:image"))

        # PageSpeed
        result["load_speed"] = await run_pagespeed_full(url)

        # SEO Score (context-aware)
        score_result = compute_ai_seo_score(result, keyword)
        result["seo_score"] = score_result["total"]
        result["seo_score_breakdown"] = score_result["breakdown"]

    except Exception as e:
        result["_error"] = str(e)

    return result


# ──────────────────────────────────────────────────────────────
# COMPARISON ENGINE
# ──────────────────────────────────────────────────────────────

def compute_comparison(results: list, user_url: str, keyword: str) -> dict:
    user = next((r for r in results if r.get("is_user")), None)
    competitors = [r for r in results if not r.get("is_user") and r.get("page_status", 0) == 200]

    if not user or not competitors:
        return {
            "gaps": [], "competitor_averages": {},
            "user_score": user.get("seo_score", 0) if user else 0,
            "competitor_avg_score": 0,
            "ranking_probability": {"probability": 0, "label": "Unknown"},
        }

    def avg(key):
        vals = [r.get(key, 0) for r in competitors if isinstance(r.get(key), (int, float)) and r.get(key, 0) > 0]
        return round(sum(vals) / len(vals), 1) if vals else 0

    comp_avg = {
        "word_count": avg("word_count"),
        "h2_count": avg("h2_count"),
        "h3_count": avg("h3_count"),
        "keyword_count": avg("keyword_count"),
        "internal_links": avg("internal_links"),
        "image_count": avg("image_count"),
        "seo_score": avg("seo_score"),
        "keyword_density": avg("keyword_density"),
    }

    gaps = []

    # Word count gap (context-aware)
    wc_diff = user["word_count"] - comp_avg["word_count"]
    intent = user.get("intent", "Informational")
    if wc_diff < -400 and intent not in ("Transactional", "Navigational"):
        gaps.append({
            "type": "content", "severity": "high" if wc_diff < -1200 else "medium",
            "metric": "Content Depth",
            "user_val": f"{user['word_count']:,} words",
            "comp_avg": f"{int(comp_avg['word_count']):,} words",
            "message": f"Your content is {abs(int(wc_diff)):,} words shorter than the competitor average. For '{keyword}' ({intent} intent), deeper coverage signals authority to Google. Add sections covering missing sub-topics."
        })

    # Keyword in title
    kw_title_count = sum(1 for c in competitors if c.get("keyword_in_title"))
    if not user.get("keyword_in_title") and kw_title_count > 0:
        gaps.append({
            "type": "keyword", "severity": "high",
            "metric": "Keyword in Title Tag",
            "user_val": "Missing",
            "comp_avg": f"{kw_title_count}/{len(competitors)} competitors",
            "message": f"Primary keyword '{keyword}' is absent from your title tag — the single most important on-page signal. {kw_title_count} of {len(competitors)} competitors include it. This alone can explain significant ranking gaps."
        })

    # Keyword in H1
    if not user.get("keyword_in_h1"):
        kw_h1_count = sum(1 for c in competitors if c.get("keyword_in_h1"))
        if kw_h1_count >= 1:
            gaps.append({
                "type": "keyword", "severity": "high",
                "metric": "Keyword in H1",
                "user_val": "Missing",
                "comp_avg": f"{kw_h1_count}/{len(competitors)} competitors",
                "message": f"Keyword missing from H1 — your page's primary content signal. Google gives significant weight to H1 alignment with the target query. {kw_h1_count} competitors include it."
            })

    # Keyword density
    kd = user.get("keyword_density", 0)
    if kd < 0.2:
        gaps.append({
            "type": "keyword", "severity": "medium",
            "metric": "Keyword Density",
            "user_val": f"{kd}%",
            "comp_avg": f"~{comp_avg['keyword_density']}%",
            "message": f"Keyword appears too infrequently ({kd}% density). Use '{keyword}' more naturally throughout your content — in intro, body sections, and conclusion. Target 0.5–2% density."
        })

    # Keyword in meta description
    if not user.get("keyword_in_meta"):
        kw_meta_count = sum(1 for c in competitors if c.get("keyword_in_meta"))
        if kw_meta_count >= len(competitors) // 2:
            gaps.append({
                "type": "keyword", "severity": "medium",
                "metric": "Keyword in Meta Description",
                "user_val": "Missing",
                "comp_avg": f"{kw_meta_count}/{len(competitors)} competitors",
                "message": f"Keyword absent from meta description. Google bolds matching keywords in search snippets — increasing CTR. {kw_meta_count} competitors use it."
            })

    # H2 heading gap
    h2_diff = user["h2_count"] - comp_avg["h2_count"]
    if h2_diff < -3:
        gaps.append({
            "type": "structure", "severity": "medium",
            "metric": "H2 Heading Structure",
            "user_val": f"{user['h2_count']} H2s",
            "comp_avg": f"~{int(comp_avg['h2_count'])} H2s",
            "message": f"Your page has {abs(int(h2_diff))} fewer H2 headings than competitors. More structured sections mean more opportunities for keyword variations and improved crawlability."
        })

    # Canonical missing
    if not user.get("canonical"):
        gaps.append({
            "type": "technical", "severity": "medium",
            "metric": "Canonical Tag",
            "user_val": "Missing",
            "comp_avg": "Expected",
            "message": "No canonical tag found. Without a canonical, Google may index duplicate or near-duplicate versions of your page, splitting ranking signals and diluting authority."
        })

    # Schema missing
    if not user.get("schema_types"):
        comp_schema = sum(1 for c in competitors if c.get("schema_types"))
        if comp_schema >= 1:
            gaps.append({
                "type": "technical", "severity": "low",
                "metric": "Structured Data (Schema)",
                "user_val": "None detected",
                "comp_avg": f"{comp_schema}/{len(competitors)} use schema",
                "message": f"{comp_schema} competitors use structured data markup. Schema enables rich results (stars, FAQs, breadcrumbs) — directly increasing CTR and signaling content authority."
            })

    # OG tags
    if not user.get("og_complete"):
        gaps.append({
            "type": "technical", "severity": "low",
            "metric": "Open Graph Tags",
            "user_val": "Incomplete",
            "comp_avg": "Full set expected",
            "message": "OG tags (title, description, image) are incomplete. These control how your page appears when shared on social platforms — affecting referral traffic and indirect ranking signals."
        })

    # Internal links gap
    il_diff = user["internal_links"] - comp_avg["internal_links"]
    if il_diff < -10 and comp_avg["internal_links"] > 5:
        gaps.append({
            "type": "structure", "severity": "medium",
            "metric": "Internal Linking",
            "user_val": f"{user['internal_links']} links",
            "comp_avg": f"~{int(comp_avg['internal_links'])} links",
            "message": f"Your page has {abs(int(il_diff))} fewer internal links. Internal links distribute PageRank, improve crawl depth, and keep users on-site longer — all positive ranking factors."
        })

    # Image gap
    img_diff = user["image_count"] - comp_avg["image_count"]
    if img_diff < -4 and comp_avg["image_count"] > 3:
        gaps.append({
            "type": "content", "severity": "low",
            "metric": "Visual Content",
            "user_val": f"{user['image_count']} images",
            "comp_avg": f"~{int(comp_avg['image_count'])} images",
            "message": f"Competitors use significantly more visuals ({int(comp_avg['image_count'])} avg). Images improve dwell time, reduce bounce rate, and provide alt-text keyword opportunities."
        })

    # PageSpeed mobile
    speed = user.get("load_speed", {})
    if isinstance(speed, dict):
        mob = speed.get("mobile", {})
        if isinstance(mob, dict) and isinstance(mob.get("score"), int) and mob["score"] < 50:
            gaps.append({
                "type": "speed", "severity": "high",
                "metric": "Mobile Page Speed",
                "user_val": f"{mob['score']}/100",
                "comp_avg": "Check competitors",
                "message": f"Mobile speed score of {mob['score']}/100 is critically low. Core Web Vitals are a confirmed Google ranking factor since 2021. 53% of mobile users leave sites taking >3s to load."
            })

    # Sort
    order = {"high": 0, "medium": 1, "low": 2}
    gaps.sort(key=lambda g: order.get(g["severity"], 3))

    # Content type note
    user_type = user.get("content_type", "Unknown")
    comp_types = [c.get("content_type", "Unknown") for c in competitors]
    most_common_type = Counter(comp_types).most_common(1)[0][0] if comp_types else "Unknown"
    content_type_note = None
    if user_type != most_common_type and most_common_type != "Unknown":
        content_type_note = f"⚠️ Format mismatch: Your page is '{user_type}' but {comp_types.count(most_common_type)}/{len(competitors)} top competitors use '{most_common_type}' format — which may better match search intent for this query."

    # Freshness note
    freshness_note = None
    comp_dates = [c.get("published_date") for c in competitors if c.get("published_date") not in ("Not Found", None, "")]
    if comp_dates:
        recent = [d for d in comp_dates if d >= "2024"]
        user_date = user.get("published_date", "Not Found")
        if recent and user_date == "Not Found":
            freshness_note = f"📅 {len(recent)}/{len(competitors)} competitors have date-stamped content (recently updated). Add datePublished schema and a 'Last updated' line to signal freshness."

    # Ranking probability
    ranking_prob = compute_ranking_probability(user, competitors, keyword)

    return {
        "user_score": user.get("seo_score", 0),
        "competitor_avg_score": round(comp_avg["seo_score"], 0),
        "competitor_averages": comp_avg,
        "gaps": gaps,
        "keyword": keyword,
        "user_intent": user.get("intent", "Unknown"),
        "user_content_type": user.get("content_type", "Unknown"),
        "content_type_note": content_type_note,
        "freshness_note": freshness_note,
        "ranking_probability": ranking_prob,
        "user_url": user_url,
        "score_breakdown": user.get("seo_score_breakdown", {}),
    }


# ──────────────────────────────────────────────────────────────
# ENDPOINT
# ──────────────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    keyword: str
    user_url: str
    competitor_urls: list
    manual_data: dict = {}


@app.post("/serp-compare")
async def serp_compare(request: CompareRequest):
    keyword = request.keyword.strip()
    if not keyword:
        return {"error": "keyword is required"}

    def clean_url(u):
        u = u.strip()
        if not u.startswith("http"):
            u = "https://" + u
        return normalize_url(u)

    user_url = clean_url(request.user_url)
    competitor_urls = [clean_url(u) for u in request.competitor_urls if u.strip()]
    all_urls = [user_url] + competitor_urls
    manual_data = request.manual_data or {}

    async def generator():
        all_results = []
        connector = aiohttp.TCPConnector(limit=20, ssl=False)

        async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:

            async def process(idx, url):
                data = await extract_full_page(session, url, keyword, manual_data.get(url, {}))
                data["is_user"] = (idx == 0)
                data["rank"] = "Your Page" if idx == 0 else f"#{idx}"
                data["_idx"] = idx
                return data

            tasks = [asyncio.create_task(process(i, u)) for i, u in enumerate(all_urls)]

            for coro in asyncio.as_completed(tasks):
                result = await coro
                all_results.append(result)
                yield json.dumps({"type": "page_result", "data": result}) + "\n"

            # Sort back into original order
            all_results.sort(key=lambda r: r.get("_idx", 99))
            comparison = compute_comparison(all_results, user_url, keyword)

            # AI analysis
            ai_result = await call_groq_ai(comparison)
            comparison["ai"] = ai_result

            yield json.dumps({"type": "comparison", "data": comparison}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(generator(), media_type="application/x-ndjson")


@app.get("/health")
async def health():
    _reset_if_new_day()
    pct = _token_tracker["tokens_used"] / _token_tracker["daily_limit"]
    return {
        "status": "ok",
        "service": "SearchRNK SERP Compare v2",
        "active_model": get_active_model(),
        "token_usage_pct": round(pct * 100, 1),
        "tokens_today": _token_tracker["tokens_used"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
