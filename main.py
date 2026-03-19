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
from collections import Counter

app = FastAPI(title="SearchRNK SERP Compare API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.searchrnk.com", "https://searchrnk.com", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


class CompareRequest(BaseModel):
    keyword: str
    user_url: str
    competitor_urls: list
    manual_data: dict = {}   # keyed by URL: {da, pa, backlinks, plagiarism}


# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────

def normalize_url(url: str) -> str:
    try:
        url, _ = urldefrag(url)
        p = urlparse(url)
        path = p.path.rstrip("/") or "/"
        return f"{p.scheme.lower()}://{p.netloc.lower()}{path}"
    except Exception:
        return url


def detect_intent(url: str, title: str, h1: str) -> str:
    combined = f"{url} {title} {h1}".lower()
    if any(w in combined for w in ["buy", "shop", "price", "order", "purchase", "cart", "checkout"]):
        return "Transactional"
    if any(w in combined for w in ["best", "top", "review", "compare", "vs", "versus", "ranking", "rated"]):
        return "Commercial Investigation"
    return "Informational"


def detect_content_type(h2_count: int, word_count: int, title: str, h1: str) -> str:
    combined = f"{title} {h1}".lower()
    if any(w in combined for w in ["top ", "best ", "list of", " ways", "tips ", "reasons", "examples"]):
        return "Listicle"
    if any(w in combined for w in ["how to", "guide", "tutorial", "step by step"]):
        return "How-to Guide"
    if h2_count >= 8 and word_count >= 2000:
        return "Long-form Article"
    if word_count < 700:
        return "Short Article"
    return "Article"


def compute_readability(text: str) -> str:
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) > 3]
    if not sentences:
        return "Unknown"
    avg = sum(len(s.split()) for s in sentences) / len(sentences)
    if avg < 15:
        return "Easy"
    if avg < 25:
        return "Moderate"
    return "Hard"


def detect_schema(soup) -> list:
    schemas = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "{}")
            if isinstance(data, dict):
                t = data.get("@type", "")
                if t:
                    schemas.append(str(t))
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        t = item.get("@type", "")
                        if t:
                            schemas.append(str(t))
        except Exception:
            pass
    return list(set(schemas))


def extract_date(soup) -> str:
    # <time datetime="">
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        return time_tag.get("datetime", "")[:10]

    # meta property
    for prop in ["article:published_time", "og:article:published_time", "article:modified_time"]:
        m = soup.find("meta", property=prop)
        if m and m.get("content"):
            return m["content"][:10]

    # itemprop
    for itemprop in ["datePublished", "dateModified"]:
        m = soup.find(attrs={"itemprop": itemprop})
        if m:
            val = m.get("content") or m.get_text(strip=True)
            if val:
                return str(val)[:10]

    # JSON-LD
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "{}")
            if isinstance(data, dict):
                date = data.get("datePublished") or data.get("dateModified")
                if date:
                    return str(date)[:10]
        except Exception:
            pass

    return "Not Found"


def extract_keyword_variations(text: str, keyword: str, top: int = 8) -> list:
    words = re.findall(r"\b[a-z]+\b", text.lower())
    kw_parts = set(keyword.lower().split())

    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]

    relevant = [ng for ng in bigrams + trigrams
                if any(kp in ng for kp in kw_parts) and ng != keyword.lower()]

    counter = Counter(relevant)
    return [ng for ng, _ in counter.most_common(top)]


def compute_seo_score(data: dict, keyword: str) -> int:
    score = 0

    # 1. Content Depth (25pts)
    wc = data.get("word_count", 0)
    if wc >= 2500:
        score += 20
    elif wc >= 1500:
        score += 14
    elif wc >= 800:
        score += 8
    elif wc >= 300:
        score += 3

    kd = data.get("keyword_density", 0)
    if 0.3 <= kd <= 3.5:
        score += 5
    elif kd > 0:
        score += 2

    # 2. On-Page SEO (20pts)
    if data.get("keyword_in_title"):
        score += 7
    if data.get("keyword_in_h1"):
        score += 7
    if data.get("keyword_in_meta"):
        score += 6

    # 3. Structure (15pts)
    if data.get("h1_count", 0) >= 1:
        score += 5
    h2 = data.get("h2_count", 0)
    if h2 >= 8:
        score += 6
    elif h2 >= 4:
        score += 4
    elif h2 >= 2:
        score += 2
    h3 = data.get("h3_count", 0)
    if h3 >= 4:
        score += 4
    elif h3 >= 2:
        score += 2

    # 4. Technical SEO (20pts)
    if data.get("canonical"):
        score += 8
    og = data.get("og_tags", {})
    if og.get("og:title") and og.get("og:description"):
        score += 7

    speed = data.get("load_speed")
    if isinstance(speed, dict):
        mob = speed.get("mobile", {})
        if isinstance(mob, dict):
            ms = mob.get("score", 0)
            if isinstance(ms, int):
                if ms >= 90:
                    score += 5
                elif ms >= 60:
                    score += 3
                elif ms >= 30:
                    score += 1

    # 5. Content Quality (10pts)
    r = data.get("readability", "Unknown")
    score += {"Easy": 10, "Moderate": 6, "Hard": 2}.get(r, 0)

    # 6. Media & Schema (10pts)
    imgs = data.get("image_count", 0)
    if imgs >= 5:
        score += 5
    elif imgs >= 2:
        score += 3
    elif imgs >= 1:
        score += 1

    if data.get("schema_types"):
        score += 5

    return min(score, 100)


def compute_comparison(results: list, user_url: str, keyword: str) -> dict:
    user = next((r for r in results if r.get("is_user")), None)
    competitors = [r for r in results if not r.get("is_user") and r.get("page_status", 0) == 200]

    if not user or not competitors:
        return {"gaps": [], "competitor_averages": {}, "user_score": 0, "competitor_avg_score": 0}

    def avg(key):
        vals = [r.get(key, 0) for r in competitors
                if isinstance(r.get(key), (int, float)) and r.get(key, 0) > 0]
        return round(sum(vals) / len(vals), 1) if vals else 0

    comp_avg = {
        "word_count": avg("word_count"),
        "h2_count": avg("h2_count"),
        "h3_count": avg("h3_count"),
        "keyword_count": avg("keyword_count"),
        "internal_links": avg("internal_links"),
        "image_count": avg("image_count"),
        "seo_score": avg("seo_score"),
    }

    gaps = []

    # Word count
    wc_diff = user["word_count"] - comp_avg["word_count"]
    if wc_diff < -300:
        gaps.append({
            "type": "content",
            "severity": "high" if wc_diff < -1000 else "medium",
            "metric": "Word Count",
            "user_val": str(user["word_count"]),
            "comp_avg": str(int(comp_avg["word_count"])),
            "message": f"Your page has {abs(int(wc_diff)):,} fewer words than the competitor average ({int(comp_avg['word_count']):,} words). Expand your content to match their depth."
        })

    # H2 headings
    h2_diff = user["h2_count"] - comp_avg["h2_count"]
    if h2_diff < -3:
        gaps.append({
            "type": "structure",
            "severity": "medium",
            "metric": "H2 Headings",
            "user_val": str(user["h2_count"]),
            "comp_avg": str(int(comp_avg["h2_count"])),
            "message": f"Your page has {abs(int(h2_diff))} fewer H2 headings than competitors (avg: {int(comp_avg['h2_count'])}). Add more section headings to improve structure and keyword coverage."
        })

    # Keyword in title
    kw_title_count = sum(1 for c in competitors if c.get("keyword_in_title"))
    if not user.get("keyword_in_title") and kw_title_count >= len(competitors) / 2:
        gaps.append({
            "type": "keyword",
            "severity": "high",
            "metric": "Keyword in Title",
            "user_val": "Missing",
            "comp_avg": f"{kw_title_count}/{len(competitors)} competitors",
            "message": f"Primary keyword '{keyword}' is missing from your page title — {kw_title_count} of {len(competitors)} competitors include it. This is one of the highest-impact on-page SEO signals."
        })

    # Keyword in H1
    if not user.get("keyword_in_h1"):
        kw_h1_count = sum(1 for c in competitors if c.get("keyword_in_h1"))
        if kw_h1_count >= 1:
            gaps.append({
                "type": "keyword",
                "severity": "high",
                "metric": "Keyword in H1",
                "user_val": "Missing",
                "comp_avg": f"{kw_h1_count}/{len(competitors)} competitors",
                "message": f"Primary keyword missing from H1 — {kw_h1_count} competitors include it in their main heading. Your H1 is your strongest on-page signal."
            })

    # Keyword in meta
    if not user.get("keyword_in_meta"):
        kw_meta_count = sum(1 for c in competitors if c.get("keyword_in_meta"))
        if kw_meta_count >= len(competitors) / 2:
            gaps.append({
                "type": "keyword",
                "severity": "medium",
                "metric": "Keyword in Meta Description",
                "user_val": "Missing",
                "comp_avg": f"{kw_meta_count}/{len(competitors)} competitors",
                "message": f"Primary keyword missing from meta description — affects click-through rates from search results. {kw_meta_count} competitors include it."
            })

    # Keyword density
    kd = user.get("keyword_density", 0)
    if kd < 0.3:
        gaps.append({
            "type": "keyword",
            "severity": "medium",
            "metric": "Keyword Density",
            "user_val": f"{kd:.2f}%",
            "comp_avg": "0.5–2%",
            "message": f"Keyword density is very low ({kd:.2f}%). Use your primary keyword more naturally throughout the content — aim for 0.5–2% density."
        })

    # Canonical
    if not user.get("canonical"):
        gaps.append({
            "type": "technical",
            "severity": "medium",
            "metric": "Canonical Tag",
            "user_val": "Missing",
            "comp_avg": "Present",
            "message": "No canonical tag detected — add a self-referential canonical to prevent duplicate content indexing fragmentation."
        })

    # Schema
    if not user.get("schema_types"):
        comp_schema_count = sum(1 for c in competitors if c.get("schema_types"))
        if comp_schema_count >= 1:
            gaps.append({
                "type": "technical",
                "severity": "low",
                "metric": "Schema Markup",
                "user_val": "None",
                "comp_avg": f"{comp_schema_count}/{len(competitors)} competitors use schema",
                "message": "No structured data found. Adding schema (Article, FAQ, Breadcrumb) can improve rich snippets and click-through rates."
            })

    # OG tags
    if not user.get("og_complete"):
        gaps.append({
            "type": "technical",
            "severity": "low",
            "metric": "Open Graph Tags",
            "user_val": "Incomplete",
            "comp_avg": "Complete",
            "message": "Open Graph tags (og:title, og:description, og:image) are missing or incomplete — critical for social sharing appearance."
        })

    # Internal links
    il_diff = user["internal_links"] - comp_avg["internal_links"]
    if il_diff < -10 and comp_avg["internal_links"] > 5:
        gaps.append({
            "type": "structure",
            "severity": "medium",
            "metric": "Internal Links",
            "user_val": str(user["internal_links"]),
            "comp_avg": str(int(comp_avg["internal_links"])),
            "message": f"Your page has {abs(int(il_diff))} fewer internal links than competitors. Add contextual internal links to distribute link equity and improve crawlability."
        })

    # Images
    img_diff = user["image_count"] - comp_avg["image_count"]
    if img_diff < -3 and comp_avg["image_count"] > 2:
        gaps.append({
            "type": "content",
            "severity": "low",
            "metric": "Images / Media",
            "user_val": str(user["image_count"]),
            "comp_avg": str(int(comp_avg["image_count"])),
            "message": f"Your page has {abs(int(img_diff))} fewer images than competitors. Add relevant visuals to improve engagement and time-on-page."
        })

    # Speed
    speed = user.get("load_speed", {})
    if isinstance(speed, dict):
        mob = speed.get("mobile", {})
        if isinstance(mob, dict):
            mob_score = mob.get("score", "--")
            if isinstance(mob_score, int) and mob_score < 50:
                gaps.append({
                    "type": "speed",
                    "severity": "high",
                    "metric": "Page Speed (Mobile)",
                    "user_val": f"{mob_score}/100",
                    "comp_avg": "Check competitors",
                    "message": f"Mobile speed score is {mob_score}/100 — critically slow. 53% of users abandon sites that take more than 3 seconds to load. Fix immediately."
                })

    # Sort by severity
    order = {"high": 0, "medium": 1, "low": 2}
    gaps.sort(key=lambda g: order.get(g["severity"], 3))

    # Content type match
    user_type = user.get("content_type", "Unknown")
    comp_types = [c.get("content_type", "Unknown") for c in competitors]
    most_common = Counter(comp_types).most_common(1)[0][0] if comp_types else "Unknown"
    content_type_note = None
    if user_type != most_common and most_common != "Unknown":
        content_type_note = f"Most top competitors use '{most_common}' format, but your page is '{user_type}'. Consider restructuring to match the dominant SERP format."

    # Freshness note
    freshness_note = None
    comp_dates = [c.get("published_date") for c in competitors if c.get("published_date") not in ("Not Found", None, "")]
    if comp_dates:
        recent_dates = [d for d in comp_dates if d > "2024"]
        if recent_dates and user.get("published_date", "Not Found") == "Not Found":
            freshness_note = f"{len(recent_dates)} of {len(competitors)} competitors have recently updated content. Add date schema and refresh your content to signal freshness to Google."

    return {
        "user_score": user.get("seo_score", 0),
        "competitor_avg_score": round(comp_avg["seo_score"], 0),
        "competitor_averages": comp_avg,
        "gaps": gaps,
        "keyword": keyword,
        "content_type_note": content_type_note,
        "freshness_note": freshness_note,
        "user_url": user_url,
    }


# ─────────────────────────────────────────────
# PAGESPEED
# ─────────────────────────────────────────────

async def run_pagespeed_full(url: str) -> dict:
    key = os.environ.get("PAGESPEED_API_KEY", "").strip()
    empty = {"score": "--", "fcp": "--", "lcp": "--", "tbt": "--", "cls": "--", "si": "--", "rating": "N/A"}

    if not key:
        return {"mobile": dict(empty), "desktop": dict(empty)}

    def get_metric(audits, audit_key, unit="s"):
        item = audits.get(audit_key, {})
        if "displayValue" in item:
            return item["displayValue"]
        if "numericValue" in item:
            val = item["numericValue"]
            if unit == "ms":
                return f"{int(val)} ms"
            return f"{val / 1000:.2f} s"
        return "--"

    def rating(score):
        if not isinstance(score, int):
            return "N/A"
        if score >= 90:
            return "Fast"
        if score >= 50:
            return "Moderate"
        return "Slow"

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
                        "fcp": get_metric(audits, "first-contentful-paint"),
                        "lcp": get_metric(audits, "largest-contentful-paint"),
                        "tbt": get_metric(audits, "total-blocking-time", "ms"),
                        "cls": get_metric(audits, "cumulative-layout-shift", ""),
                        "si": get_metric(audits, "speed-index"),
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

    if isinstance(mob, Exception) or mob is None:
        mob = dict(empty)
    if isinstance(desk, Exception) or desk is None:
        desk = dict(empty)

    return {"mobile": mob, "desktop": desk}


# ─────────────────────────────────────────────
# PAGE EXTRACTOR
# ─────────────────────────────────────────────

async def extract_full_page(session, url: str, keyword: str, manual: dict) -> dict:
    kw = keyword.lower().strip()
    result = {
        "url": url, "page_status": 0,
        "title": "", "title_length": 0,
        "meta_description": "", "meta_desc_length": 0,
        "h1": "", "h1_count": 0, "h2_count": 0, "h3_count": 0,
        "keyword_in_title": False, "keyword_in_h1": False,
        "keyword_in_meta": False, "keyword_in_h2": False, "keyword_in_h3": False,
        "keyword_count": 0, "keyword_density": 0.0, "keyword_variations": [],
        "word_count": 0, "intent": "Unknown", "content_type": "Unknown",
        "published_date": "Not Found", "readability": "Unknown",
        "schema_types": [], "internal_links": 0, "external_links": 0,
        "image_count": 0, "canonical": None, "canonical_match": False,
        "og_tags": {}, "og_complete": False,
        "load_speed": None, "seo_score": 0,
        "da": manual.get("da", "Not Provided"),
        "pa": manual.get("pa", "Not Provided"),
        "backlinks": manual.get("backlinks", "Not Provided"),
        "plagiarism": manual.get("plagiarism", "Not Provided"),
    }

    try:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=20),
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
        result["title"] = title[:250]
        result["title_length"] = len(title)
        result["keyword_in_title"] = kw in title.lower()

        # Meta description
        meta_desc_tag = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
        desc = meta_desc_tag.get("content", "").strip() if meta_desc_tag else ""
        result["meta_description"] = desc[:350]
        result["meta_desc_length"] = len(desc)
        result["keyword_in_meta"] = kw in desc.lower()

        # Headings
        h1_tags = soup.find_all("h1")
        result["h1_count"] = len(h1_tags)
        result["h1"] = h1_tags[0].get_text(strip=True)[:200] if h1_tags else ""
        result["keyword_in_h1"] = kw in result["h1"].lower()

        h2_tags = soup.find_all("h2")
        result["h2_count"] = len(h2_tags)
        h2_text = " ".join(h.get_text(strip=True).lower() for h in h2_tags)
        result["keyword_in_h2"] = kw in h2_text

        h3_tags = soup.find_all("h3")
        result["h3_count"] = len(h3_tags)
        h3_text = " ".join(h.get_text(strip=True).lower() for h in h3_tags)
        result["keyword_in_h3"] = kw in h3_text

        # Clean content
        for tag in soup.find_all(["script", "style", "noscript", "nav", "header", "footer", "aside"]):
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
        result["keyword_variations"] = extract_keyword_variations(clean_text, keyword)

        # Metadata
        result["intent"] = detect_intent(url, title, result["h1"])
        result["content_type"] = detect_content_type(result["h2_count"], result["word_count"], title, result["h1"])
        result["readability"] = compute_readability(clean_text)
        result["published_date"] = extract_date(soup)
        result["schema_types"] = detect_schema(soup)

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

        # SEO Score (after speed is fetched)
        result["seo_score"] = compute_seo_score(result, keyword)

    except Exception:
        pass

    return result


# ─────────────────────────────────────────────
# ENDPOINT
# ─────────────────────────────────────────────

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
                data["is_user"] = idx == 0
                data["rank"] = "Your Page" if idx == 0 else f"#{idx}"
                data["_idx"] = idx
                return data

            tasks = [asyncio.create_task(process(i, u)) for i, u in enumerate(all_urls)]

            for coro in asyncio.as_completed(tasks):
                result = await coro
                all_results.append(result)
                yield json.dumps({"type": "page_result", "data": result}) + "\n"

            # Sort results back into original order for comparison
            all_results.sort(key=lambda r: r.get("_idx", 99))
            comparison = compute_comparison(all_results, user_url, keyword)
            yield json.dumps({"type": "comparison", "data": comparison}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(generator(), media_type="application/x-ndjson")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "SearchRNK SERP Compare"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
