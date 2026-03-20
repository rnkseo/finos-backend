"""
SearchRNK SERP Intelligence API v3 — God-Level SEO Engine
50+ signals | GPT OSS 20B → LLaMA 3.1 8B | Fixed PageSpeed | Full E-E-A-T
"""
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import aiohttp, asyncio, os, json, re, logging
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urldefrag
from collections import Counter
from datetime import date

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("searchrnk")

app = FastAPI(title="SearchRNK SEO Intelligence v3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Upgrade-Insecure-Requests": "1",
}

# ──────────────────────────────────────────────────────────────
# MODEL SWITCHING SYSTEM — GPT OSS 20B → LLaMA 3.1 8B
# ──────────────────────────────────────────────────────────────
_usage = {
    "date": str(date.today()),
    "models": {
        "openai/gpt-oss-20b": {"requests": 0, "tokens": 0, "req_limit": 1000, "tok_limit": 200_000},
        "llama-3.1-8b-instant": {"requests": 0, "tokens": 0, "req_limit": 14400, "tok_limit": 500_000},
    },
    "threshold": 0.80,
    "primary": "openai/gpt-oss-20b",
    "fallback": "llama-3.1-8b-instant",
}

def _reset_if_new_day():
    today = str(date.today())
    if _usage["date"] != today:
        _usage["date"] = today
        for m in _usage["models"].values():
            m["requests"] = 0
            m["tokens"] = 0

def get_active_model() -> str:
    _reset_if_new_day()
    primary = _usage["primary"]
    pd = _usage["models"][primary]
    req_pct = pd["requests"] / pd["req_limit"]
    tok_pct = pd["tokens"] / pd["tok_limit"]
    if max(req_pct, tok_pct) >= _usage["threshold"]:
        logger.info(f"Switching to fallback model — primary at {max(req_pct, tok_pct)*100:.0f}% capacity")
        return _usage["fallback"]
    return primary

def record_usage(model: str, tokens: int):
    _reset_if_new_day()
    if model in _usage["models"]:
        _usage["models"][model]["requests"] += 1
        _usage["models"][model]["tokens"] += tokens

def get_model_status() -> dict:
    _reset_if_new_day()
    status = {}
    for name, data in _usage["models"].items():
        status[name] = {
            "req_pct": round(data["requests"] / data["req_limit"] * 100, 1),
            "tok_pct": round(data["tokens"] / data["tok_limit"] * 100, 1),
            "active": name == get_active_model(),
        }
    return status

# ──────────────────────────────────────────────────────────────
# URL UTILITIES
# ──────────────────────────────────────────────────────────────
def normalize_url(url: str) -> str:
    try:
        url, _ = urldefrag(url)
        p = urlparse(url)
        path = p.path.rstrip("/") or "/"
        return f"{p.scheme.lower()}://{p.netloc.lower()}{path}"
    except:
        return url

# ──────────────────────────────────────────────────────────────
# PAGESPEED — FIXED WITH RETRY + LOGGING
# ──────────────────────────────────────────────────────────────
async def run_pagespeed(url: str, strategy: str = "mobile", retries: int = 2) -> dict:
    key = os.environ.get("PAGESPEED_API_KEY", "").strip()
    empty = {"score": None, "fcp": "N/A", "lcp": "N/A", "tbt": "N/A", "cls": "N/A", "si": "N/A",
             "fid": "N/A", "inp": "N/A", "rating": "No Data", "error": None}
    if not key:
        empty["error"] = "PAGESPEED_API_KEY not configured"
        logger.warning("PageSpeed: API key not set")
        return empty

    api_url = (f"https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
               f"?url={url}&strategy={strategy}&category=PERFORMANCE&key={key}")

    def parse_metric(audits: dict, key: str, unit: str = "s") -> str:
        item = audits.get(key, {})
        if "displayValue" in item:
            return item["displayValue"]
        if "numericValue" in item:
            val = item["numericValue"]
            return f"{int(val)} ms" if unit == "ms" else f"{val / 1000:.2f} s"
        return "N/A"

    def score_rating(s: int) -> str:
        return "Fast" if s >= 90 else "Needs Work" if s >= 50 else "Slow"

    for attempt in range(retries + 1):
        try:
            connector = aiohttp.TCPConnector(ssl=False, limit=5)
            async with aiohttp.ClientSession(connector=connector) as sess:
                async with sess.get(api_url, timeout=aiohttp.ClientTimeout(total=65)) as resp:
                    logger.info(f"PSI {strategy} [{url[:50]}] → HTTP {resp.status}")
                    if resp.status != 200:
                        text = await resp.text()
                        logger.warning(f"PSI error: {text[:200]}")
                        if attempt < retries:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        empty["error"] = f"PSI HTTP {resp.status}"
                        return empty

                    data = await resp.json()

                    if "error" in data:
                        logger.warning(f"PSI API error: {data['error']}")
                        empty["error"] = str(data["error"].get("message", "API Error"))
                        return empty

                    if "lighthouseResult" not in data:
                        logger.warning("PSI: No lighthouseResult in response")
                        empty["error"] = "No Lighthouse data"
                        return empty

                    lh = data["lighthouseResult"]
                    audits = lh.get("audits", {})
                    cats = lh.get("categories", {})
                    perf = cats.get("performance", {})
                    raw_score = perf.get("score")
                    score = int(raw_score * 100) if raw_score is not None else 0

                    # Core Web Vitals from categories if available
                    cwv = data.get("loadingExperience", {}).get("metrics", {})

                    return {
                        "score": score,
                        "fcp": parse_metric(audits, "first-contentful-paint"),
                        "lcp": parse_metric(audits, "largest-contentful-paint"),
                        "tbt": parse_metric(audits, "total-blocking-time", "ms"),
                        "cls": parse_metric(audits, "cumulative-layout-shift", ""),
                        "si": parse_metric(audits, "speed-index"),
                        "fid": cwv.get("FIRST_INPUT_DELAY_MS", {}).get("percentile", "N/A"),
                        "inp": cwv.get("INTERACTION_TO_NEXT_PAINT", {}).get("percentile", "N/A"),
                        "rating": score_rating(score),
                        "error": None,
                    }

        except asyncio.TimeoutError:
            logger.warning(f"PSI timeout attempt {attempt+1} for {url}")
            if attempt < retries:
                await asyncio.sleep(3)
        except Exception as e:
            logger.warning(f"PSI exception attempt {attempt+1}: {e}")
            if attempt < retries:
                await asyncio.sleep(2)

    empty["error"] = "Timeout after retries"
    return empty

async def run_pagespeed_full(url: str) -> dict:
    mob, desk = await asyncio.gather(
        run_pagespeed(url, "mobile"),
        run_pagespeed(url, "desktop"),
        return_exceptions=True,
    )
    empty = {"score": None, "fcp": "N/A", "lcp": "N/A", "tbt": "N/A", "cls": "N/A",
             "si": "N/A", "rating": "No Data", "error": "Exception"}
    return {
        "mobile": mob if not isinstance(mob, Exception) else dict(empty),
        "desktop": desk if not isinstance(desk, Exception) else dict(empty),
    }

# ──────────────────────────────────────────────────────────────
# SIGNAL EXTRACTORS
# ──────────────────────────────────────────────────────────────
def detect_intent(url, title, h1, meta):
    combined = f"{url} {title} {h1} {meta}".lower()
    scores = {
        "Transactional": sum(2 for w in ["buy","shop","price","order","purchase","cart","checkout","get started","sign up","subscribe","pricing","plan","free trial"] if w in combined),
        "Commercial": sum(2 for w in ["best","top","review","compare","vs","versus","ranking","rated","alternatives","worth it","pros and cons","recommended"] if w in combined),
        "Navigational": sum(2 for w in ["login","sign in","account","dashboard","portal","support","contact us","home page"] if w in combined),
        "Informational": sum(1 for w in ["how to","what is","guide","tutorial","learn","tips","examples","definition","explained","why","when","overview","introduction"] if w in combined),
    }
    top = max(scores, key=scores.get)
    return top if scores[top] > 0 else "Informational"

def detect_content_type(h2: int, wc: int, title: str, h1: str) -> str:
    c = f"{title} {h1}".lower()
    if any(w in c for w in ["top ","best ","list of"," ways","tips ","reasons","examples","tools","plugins","resources","ideas"]): return "Listicle"
    if any(w in c for w in ["how to","guide","tutorial","step by step","setup","install","configure","create"]): return "How-to Guide"
    if any(w in c for w in ["review","vs","versus","compared","comparison","alternatives","better","winner"]): return "Comparison"
    if any(w in c for w in ["what is","definition","meaning","explained","overview","introduction","understand"]): return "Explainer"
    if any(w in c for w in ["case study","results","success","outcome","experiment","test"]): return "Case Study"
    if h2 >= 8 and wc >= 2500: return "Pillar Article"
    if wc < 700: return "Short-form"
    return "Article"

def extract_readability(text: str) -> dict:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip().split()) > 3]
    if not sentences:
        return {"label": "Unknown", "avg_words_per_sentence": 0, "complex_word_pct": 0, "flesch_approx": 0}
    words = text.split()
    avg = sum(len(s.split()) for s in sentences) / len(sentences)
    long_words = sum(1 for w in words if len(re.sub(r"[^a-z]","",w.lower())) > 7)
    long_pct = long_words / max(len(words), 1) * 100
    # Approximate Flesch Reading Ease
    syllables = sum(max(1, len(re.findall(r"[aeiouy]+", w.lower()))) for w in words)
    flesch = 206.835 - 1.015 * avg - 84.6 * (syllables / max(len(words), 1))
    flesch = max(0, min(100, round(flesch)))
    label = "Easy" if avg < 14 and long_pct < 20 else "Moderate" if avg < 24 else "Complex"
    return {"label": label, "avg_words_per_sentence": round(avg, 1), "complex_word_pct": round(long_pct, 1), "flesch_approx": flesch}

def extract_schema(soup) -> dict:
    schemas, previews = [], []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "{}")
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    t = item.get("@type", "")
                    if t: schemas.append(str(t))
                    preview = {k: v for k, v in item.items() if k in ["@type","name","description","url","image","author","datePublished","headline","@context"]}
                    if preview: previews.append(preview)
        except: pass
    return {"types": list(set(schemas)), "count": len(schemas), "previews": previews[:4]}

def extract_date(soup) -> str:
    for tag in soup.find_all("time", attrs={"datetime": True}):
        v = tag.get("datetime","")[:10]; 
        if v: return v
    for prop in ["article:published_time","og:article:published_time","article:modified_time","date"]:
        m = soup.find("meta", property=prop)
        if m and m.get("content"): return m["content"][:10]
    for name in ["date","published","publish-date","article:published_time"]:
        m = soup.find("meta", attrs={"name": name})
        if m and m.get("content"): return m["content"][:10]
    for attr in ["datePublished","dateModified","dateCreated"]:
        m = soup.find(attrs={"itemprop": attr})
        if m:
            v = m.get("content") or m.get_text(strip=True)
            if v: return str(v)[:10]
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "{}")
            if isinstance(data, dict):
                for k in ["datePublished","dateModified","dateCreated","uploadDate"]:
                    if data.get(k): return str(data[k])[:10]
        except: pass
    return "Not Found"

def extract_eeat_signals(soup, text: str) -> dict:
    signals = []
    # Author
    has_author = bool(soup.find(attrs={"itemprop": re.compile("author|person", re.I)}) or
                     soup.find("a", rel="author") or
                     soup.find(class_=re.compile("author|byline|writer", re.I)) or
                     "by " in text[:500].lower())
    if has_author: signals.append("author_byline")
    # Citations
    ext_links = [a.get("href","") for a in soup.find_all("a", href=True)]
    authority_domains = [u for u in ext_links if any(d in u for d in [".edu",".gov","wikipedia.org","pubmed","ncbi","scholar","reuters","bbc","nytimes","nature.com","science.org"])]
    if len(authority_domains) >= 2: signals.append("authority_citations")
    # About/bio section
    if soup.find(id=re.compile("about|bio|author", re.I)) or soup.find(class_=re.compile("about-author|author-bio|contributor", re.I)):
        signals.append("author_bio")
    # Expert credentials
    cred_terms = ["certified","expert","years of experience","phd","mba","specialist","professional","researcher","professor"]
    if any(term in text.lower() for term in cred_terms): signals.append("credentials_mentioned")
    # Sources/references section
    if any(w in text.lower() for w in ["references","sources","bibliography","further reading","citations"]):
        signals.append("references_section")
    # Last updated
    if any(w in text.lower() for w in ["last updated","updated on","modified","reviewed by"]):
        signals.append("freshness_signals")
    return {"count": len(signals), "signals": signals}

def extract_featured_snippet_eligibility(soup, text: str) -> dict:
    reasons = []
    # Numbered lists
    if soup.find_all("ol"): reasons.append("has_numbered_list")
    # Table
    if soup.find("table"): reasons.append("has_data_table")
    # Definition patterns
    if re.search(r"(?:is defined as|refers to|is a type of|means that|is an?)\s+\w+", text[:3000].lower()):
        reasons.append("has_definition_pattern")
    # Question-answer format
    if soup.find_all(["dt", "dd"]) or soup.find(class_=re.compile("faq|question|accordion", re.I)):
        reasons.append("has_faq_format")
    # Concise paragraph after H2
    h2s = soup.find_all("h2")
    for h2 in h2s[:3]:
        next_p = h2.find_next_sibling("p")
        if next_p:
            p_words = len(next_p.get_text().split())
            if 40 <= p_words <= 120:
                reasons.append("has_snippet_paragraph")
                break
    eligible = len(reasons) >= 2
    return {"eligible": eligible, "score": len(reasons), "reasons": reasons}

def extract_keyword_intelligence(text: str, keyword: str) -> dict:
    kw_lower = keyword.lower()
    words = re.findall(r"\b[a-z]+\b", text.lower())
    kw_parts = set(kw_lower.split())
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
    fourgrams = [f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}" for i in range(len(words)-3)]
    
    relevant = Counter(ng for ng in bigrams+trigrams+fourgrams if any(kp in ng for kp in kw_parts) and ng != kw_lower)
    stop = {"the","and","for","are","but","not","you","all","can","was","one","our","out","get","has","how","new","see","way","who","did","its","let","use","with","that","this","from","they","will","have","here","more","also","into","than","them","some","what","when","your","said","been","make"}
    topic_trigrams = Counter(ng for ng, cnt in Counter(trigrams).most_common(50) if cnt >= 2 and not any(w in stop for w in ng.split()))
    variations = [ng for ng, _ in relevant.most_common(15) if ng not in stop]
    return {
        "primary": keyword,
        "variations": variations[:12],
        "topic_clusters": [ng for ng, _ in topic_trigrams.most_common(8)],
        "total_variation_count": len(variations),
        "semantic_richness": min(100, len(set(words)) * 100 // max(len(words), 1)),
    }

def compute_seo_score(d: dict, keyword: str) -> dict:
    dims = {"content_relevance": 0, "on_page_signals": 0, "structure": 0, "technical": 0, "content_quality": 0, "ux_media": 0}
    intent = d.get("intent", "Informational")
    wc = d.get("word_count", 0)
    
    # Content Relevance (25)
    if intent in ("Transactional","Navigational"):
        if wc >= 400: dims["content_relevance"] += 10
        elif wc >= 150: dims["content_relevance"] += 5
    else:
        if wc >= 3000: dims["content_relevance"] += 15
        elif wc >= 2000: dims["content_relevance"] += 12
        elif wc >= 1200: dims["content_relevance"] += 8
        elif wc >= 600: dims["content_relevance"] += 4
    kd = d.get("keyword_density", 0)
    if 0.3 <= kd <= 3.0: dims["content_relevance"] += 7
    elif 0.1 <= kd <= 4.0: dims["content_relevance"] += 3
    if d.get("keyword_in_h2"): dims["content_relevance"] += 2
    if d.get("keyword_in_h3"): dims["content_relevance"] += 1
    
    # On-Page Signals (20)
    if d.get("keyword_in_title"): dims["on_page_signals"] += 8
    if d.get("keyword_in_h1"): dims["on_page_signals"] += 7
    if d.get("keyword_in_meta"): dims["on_page_signals"] += 5
    
    # Structure (15)
    h1c = d.get("h1_count", 0)
    if h1c == 1: dims["structure"] += 5
    elif h1c == 0: dims["structure"] -= 3
    h2c = d.get("h2_count", 0)
    dims["structure"] += 6 if h2c >= 7 else 4 if h2c >= 4 else 2 if h2c >= 2 else 0
    h3c = d.get("h3_count", 0)
    dims["structure"] += 4 if h3c >= 5 else 2 if h3c >= 2 else 0
    if d.get("has_toc"): dims["structure"] += 2
    if d.get("heading_hierarchy_ok"): dims["structure"] += 1
    
    # Technical (20)
    if d.get("canonical"): dims["technical"] += 5
    if d.get("canonical_match"): dims["technical"] += 2
    og = d.get("og_tags", {})
    dims["technical"] += min(sum(1 for k in ["og:title","og:description","og:image","og:type","og:url"] if og.get(k)), 5)
    if d.get("og_complete"): dims["technical"] += 2
    if d.get("twitter_tags"): dims["technical"] += 1
    if d.get("is_https"): dims["technical"] += 2
    if d.get("has_viewport"): dims["technical"] += 1
    if d.get("robots_indexable"): dims["technical"] += 2
    sp = d.get("load_speed", {})
    if isinstance(sp, dict):
        mob = sp.get("mobile", {})
        if isinstance(mob.get("score"), int):
            ms = mob["score"]
            dims["technical"] += 5 if ms >= 90 else 3 if ms >= 70 else 1 if ms >= 50 else 0
    
    # Content Quality (10)
    r = d.get("readability", {})
    rl = r if isinstance(r, str) else r.get("label", "Unknown")
    dims["content_quality"] += {"Easy": 5, "Moderate": 4, "Complex": 2}.get(rl, 0)
    if d.get("schema_types"): dims["content_quality"] += 3
    if d.get("featured_snippet_eligible", {}).get("eligible"): dims["content_quality"] += 1
    eeat = d.get("eeat_signals", {}).get("count", 0)
    dims["content_quality"] += min(eeat, 2)
    
    # UX / Media (10)
    imgs = d.get("image_count", 0)
    dims["ux_media"] += 5 if imgs >= 8 else 3 if imgs >= 4 else 1 if imgs >= 1 else 0
    il = d.get("internal_links", 0)
    dims["ux_media"] += 5 if il >= 15 else 3 if il >= 7 else 1 if il >= 2 else 0
    if d.get("has_video"): dims["ux_media"] += 1
    if d.get("has_faq"): dims["ux_media"] += 1
    
    # Clamp
    caps = {"content_relevance": 25, "on_page_signals": 20, "structure": 15, "technical": 20, "content_quality": 10, "ux_media": 10}
    for k, cap in caps.items():
        dims[k] = max(0, min(cap, dims[k]))
    
    return {"total": min(sum(dims.values()), 100), "breakdown": dims}

def compute_ranking_probability(user: dict, competitors: list, keyword: str) -> dict:
    if not competitors:
        return {"probability": 50, "label": "Moderate", "factors": []}
    comp_scores = [c.get("seo_score", 0) for c in competitors if c.get("seo_score", 0) > 0]
    avg_comp = sum(comp_scores) / len(comp_scores) if comp_scores else 50
    gap = user.get("seo_score", 0) - avg_comp
    base = 50 + gap * 0.9
    
    factors = []
    if user.get("keyword_in_title") and user.get("keyword_in_h1"):
        base += 9; factors.append({"factor": "Keyword in title + H1", "impact": "+9%", "positive": True})
    elif not user.get("keyword_in_title"):
        base -= 12; factors.append({"factor": "Keyword missing from title", "impact": "-12%", "positive": False})
    if not user.get("canonical"):
        base -= 5; factors.append({"factor": "No canonical tag", "impact": "-5%", "positive": False})
    if user.get("schema_types"):
        base += 5; factors.append({"factor": "Structured data present", "impact": "+5%", "positive": True})
    eeat = user.get("eeat_signals", {}).get("count", 0)
    if eeat >= 3:
        base += 6; factors.append({"factor": "Strong E-E-A-T signals", "impact": "+6%", "positive": True})
    elif eeat == 0:
        base -= 4; factors.append({"factor": "No E-E-A-T signals detected", "impact": "-4%", "positive": False})
    if user.get("featured_snippet_eligible", {}).get("eligible"):
        base += 5; factors.append({"factor": "Featured snippet eligible", "impact": "+5%", "positive": True})
    sp = user.get("load_speed", {})
    if isinstance(sp, dict):
        mob = sp.get("mobile", {})
        if isinstance(mob.get("score"), int):
            if mob["score"] < 50:
                base -= 12; factors.append({"factor": "Critical mobile speed issue", "impact": "-12%", "positive": False})
            elif mob["score"] >= 90:
                base += 7; factors.append({"factor": "Excellent mobile speed", "impact": "+7%", "positive": True})
    if user.get("og_complete"):
        base += 3; factors.append({"factor": "Complete Open Graph tags", "impact": "+3%", "positive": True})
    
    prob = max(5, min(95, round(base)))
    label = "High" if prob >= 68 else "Moderate" if prob >= 42 else "Low"
    return {"probability": prob, "label": label, "factors": factors[:8]}

def compute_comparison(results: list, user_url: str, keyword: str) -> dict:
    user = next((r for r in results if r.get("is_user")), None)
    competitors = [r for r in results if not r.get("is_user") and r.get("page_status", 0) == 200]
    if not user or not competitors:
        return {"gaps": [], "competitor_averages": {}, "user_score": user.get("seo_score",0) if user else 0, "competitor_avg_score": 0}

    def avg(key):
        vals = [r.get(key,0) for r in competitors if isinstance(r.get(key),(int,float)) and r.get(key,0) > 0]
        return round(sum(vals)/len(vals), 1) if vals else 0

    comp_avg = {k: avg(k) for k in ["word_count","h2_count","h3_count","keyword_count","keyword_density","internal_links","image_count","seo_score"]}
    gaps = []

    # --- Content depth ---
    wc_diff = user["word_count"] - comp_avg["word_count"]
    intent = user.get("intent","Informational")
    if wc_diff < -400 and intent not in ("Transactional","Navigational"):
        gaps.append({"type":"content","severity":"high" if wc_diff<-1200 else "medium","metric":"Content Depth",
            "user_val":f"{user['word_count']:,} words","comp_avg":f"{int(comp_avg['word_count']):,} words",
            "message":f"Your page is {abs(int(wc_diff)):,} words shorter than the competitor average. For '{keyword}' ({intent} intent), content depth directly signals expertise to Google. Competitors rank because their comprehensive content satisfies every reader question — expand yours to match."})

    # --- Keyword in title ---
    kw_title_n = sum(1 for c in competitors if c.get("keyword_in_title"))
    if not user.get("keyword_in_title") and kw_title_n > 0:
        gaps.append({"type":"keyword","severity":"high","metric":"Keyword in Title Tag",
            "user_val":"Missing","comp_avg":f"{kw_title_n}/{len(competitors)} competitors",
            "message":f"Primary keyword '{keyword}' is absent from your title tag — the single highest-weighted on-page signal. {kw_title_n}/{len(competitors)} competitors include it. Google reads your title first; no keyword there = no relevance signal for that query."})

    # --- Keyword in H1 ---
    kw_h1_n = sum(1 for c in competitors if c.get("keyword_in_h1"))
    if not user.get("keyword_in_h1") and kw_h1_n >= 1:
        gaps.append({"type":"keyword","severity":"high","metric":"Keyword in H1",
            "user_val":"Missing","comp_avg":f"{kw_h1_n}/{len(competitors)} competitors",
            "message":f"Your H1 doesn't include the target keyword. {kw_h1_n} competitors do. H1 is the strongest content signal after the title tag — it anchors your page's topical relevance."})

    # --- Keyword density ---
    kd = user.get("keyword_density",0)
    if kd < 0.2:
        gaps.append({"type":"keyword","severity":"medium","metric":"Keyword Density",
            "user_val":f"{kd}%","comp_avg":f"~{comp_avg['keyword_density']}%",
            "message":f"Keyword appears only {kd}% of the time. This signals weak topical focus to Google's NLP. Use '{keyword}' naturally in your intro, body sections, and conclusion. Target 0.5–2%."})
    elif kd > 4.0:
        gaps.append({"type":"keyword","severity":"medium","metric":"Keyword Stuffing Risk",
            "user_val":f"{kd}%","comp_avg":"< 3% ideal",
            "message":f"Keyword density of {kd}% may trigger Google's over-optimisation penalty. Natural writing that satisfies intent outranks keyword-stuffed pages. Reduce to 0.5–2%."})

    # --- Keyword in meta ---
    kw_meta_n = sum(1 for c in competitors if c.get("keyword_in_meta"))
    if not user.get("keyword_in_meta") and kw_meta_n >= len(competitors)//2:
        gaps.append({"type":"keyword","severity":"medium","metric":"Keyword in Meta Description",
            "user_val":"Missing","comp_avg":f"{kw_meta_n}/{len(competitors)} competitors",
            "message":f"Google bolds matching keywords in search snippets, directly boosting CTR. {kw_meta_n} competitors include it in their meta description. Higher CTR improves your ranking position over time."})

    # --- H2 structure ---
    h2_diff = user["h2_count"] - comp_avg["h2_count"]
    if h2_diff < -3:
        gaps.append({"type":"structure","severity":"medium","metric":"H2 Heading Coverage",
            "user_val":f"{user['h2_count']} H2s","comp_avg":f"~{int(comp_avg['h2_count'])} H2s",
            "message":f"Your page has {abs(int(h2_diff))} fewer H2 headings. Each H2 is a topical sub-section signal — more H2s means more topic coverage, more keyword variations, and better crawlability. Competitors are satisfying more user sub-intents."})

    # --- Canonical ---
    if not user.get("canonical"):
        gaps.append({"type":"technical","severity":"medium","metric":"Canonical Tag Missing",
            "user_val":"Not set","comp_avg":"Expected",
            "message":"Without a canonical, Google may find multiple URL variants of your page (with/without trailing slash, with params) and split ranking signals between them — effectively reducing your page's authority."})

    # --- Schema ---
    comp_schema_n = sum(1 for c in competitors if c.get("schema_types"))
    if not user.get("schema_types") and comp_schema_n >= 1:
        gaps.append({"type":"technical","severity":"low","metric":"Structured Data (Schema)",
            "user_val":"None","comp_avg":f"{comp_schema_n}/{len(competitors)} use schema",
            "message":f"{comp_schema_n} competitors use structured data. Schema unlocks rich results — star ratings, FAQs, breadcrumbs in SERPs. These increase CTR by up to 30% and signal content authority to Google."})

    # --- OG tags ---
    if not user.get("og_complete"):
        gaps.append({"type":"technical","severity":"low","metric":"Open Graph Tags Incomplete",
            "user_val":"Partial","comp_avg":"Full set expected",
            "message":"Incomplete OG tags (missing og:title, og:description, or og:image) degrade how your page looks when shared on LinkedIn, Twitter, Slack. Poor social previews reduce referral traffic — an indirect ranking signal."})

    # --- E-E-A-T ---
    user_eeat = user.get("eeat_signals",{}).get("count",0)
    comp_eeat_avg = sum(c.get("eeat_signals",{}).get("count",0) for c in competitors) / len(competitors)
    if user_eeat < comp_eeat_avg - 1:
        gaps.append({"type":"authority","severity":"medium","metric":"E-E-A-T Signals",
            "user_val":f"{user_eeat} signals","comp_avg":f"~{comp_eeat_avg:.0f} signals",
            "message":f"Your page has fewer Experience, Expertise, Authoritativeness, Trustworthiness signals. Add: author byline, credentials, external citations from authority sources (.edu, .gov), and a 'last updated' date. Google's Quality Rater guidelines weight these heavily for YMYL queries."})

    # --- Featured snippet ---
    user_snip = user.get("featured_snippet_eligible",{}).get("eligible",False)
    comp_snip_n = sum(1 for c in competitors if c.get("featured_snippet_eligible",{}).get("eligible",False))
    if not user_snip and comp_snip_n >= 1:
        gaps.append({"type":"content","severity":"low","metric":"Featured Snippet Eligibility",
            "user_val":"Not eligible","comp_avg":f"{comp_snip_n}/{len(competitors)} eligible",
            "message":f"Your content isn't structured to win featured snippets (position zero). Add: numbered step lists, definition paragraphs (40-60 words answering the query directly), or FAQ sections. {comp_snip_n} competitors are eligible."})

    # --- Internal links ---
    il_diff = user["internal_links"] - comp_avg["internal_links"]
    if il_diff < -10 and comp_avg["internal_links"] > 5:
        gaps.append({"type":"structure","severity":"medium","metric":"Internal Linking",
            "user_val":f"{user['internal_links']} links","comp_avg":f"~{int(comp_avg['internal_links'])} links",
            "message":f"Your page links to {abs(int(il_diff))} fewer internal pages than competitors. Internal links distribute PageRank throughout your site, improve crawl depth, and keep users on-site longer. Each link is a 'vote' for that destination page."})

    # --- PageSpeed ---
    sp = user.get("load_speed",{})
    if isinstance(sp, dict):
        mob = sp.get("mobile",{})
        if isinstance(mob.get("score"), int):
            if mob["score"] < 50:
                gaps.append({"type":"speed","severity":"high","metric":"Mobile Page Speed Critical",
                    "user_val":f"{mob['score']}/100","comp_avg":"Check competitors",
                    "message":f"Mobile speed score of {mob['score']}/100 is critically poor. Core Web Vitals are a direct Google ranking factor. 53% of mobile users leave sites taking >3s. Fix: compress images, eliminate render-blocking resources, enable caching."})
            elif mob["score"] < 70:
                gaps.append({"type":"speed","severity":"medium","metric":"Mobile Page Speed",
                    "user_val":f"{mob['score']}/100","comp_avg":"Target: 80+",
                    "message":f"Mobile speed needs improvement ({mob['score']}/100). Poor Core Web Vitals hurt both ranking position and conversion rates. Target 80+ to stay competitive."})

    # --- Image alt text ---
    user_imgs = user.get("image_count",0)
    user_alt = user.get("alt_text_coverage",0)
    if user_imgs >= 3 and user_alt < 50:
        gaps.append({"type":"content","severity":"low","metric":"Image Alt Text Coverage",
            "user_val":f"{user_alt}% coverage","comp_avg":"Target: 80%+",
            "message":f"Only {user_alt}% of your images have alt text. Alt text provides keyword signals, enables image search rankings, and improves accessibility scores. Missing alt text is missed SEO opportunity on every image."})

    order = {"high":0,"medium":1,"low":2}
    gaps.sort(key=lambda g: order.get(g["severity"],3))

    # Notes
    user_type = user.get("content_type","Unknown")
    comp_types = [c.get("content_type","Unknown") for c in competitors]
    top_type = Counter(comp_types).most_common(1)[0][0] if comp_types else "Unknown"
    content_type_note = None
    if user_type != top_type and top_type != "Unknown":
        content_type_note = f"Format mismatch: Your page is '{user_type}' but {comp_types.count(top_type)}/{len(competitors)} top competitors use '{top_type}' format. SERP format preference signals what Google thinks best satisfies this query's intent."

    freshness_note = None
    comp_dates = [c.get("published_date") for c in competitors if c.get("published_date") not in ("Not Found",None,"")]
    if comp_dates:
        recent = [d for d in comp_dates if d >= "2024"]
        if recent and user.get("published_date","Not Found") == "Not Found":
            freshness_note = f"{len(recent)}/{len(competitors)} competitors have date-stamped content. Freshness signals (datePublished schema, visible date, 'last updated' line) help Google understand recency — important for time-sensitive queries."

    return {
        "user_score": user.get("seo_score",0),
        "competitor_avg_score": round(comp_avg["seo_score"],0),
        "competitor_averages": comp_avg,
        "gaps": gaps,
        "keyword": keyword,
        "user_intent": user.get("intent","Unknown"),
        "user_content_type": user.get("content_type","Unknown"),
        "content_type_note": content_type_note,
        "freshness_note": freshness_note,
        "ranking_probability": compute_ranking_probability(user, competitors, keyword),
        "score_breakdown": user.get("seo_score_breakdown",{}),
        "user_url": user_url,
        "comp_count": len(competitors),
    }

# ──────────────────────────────────────────────────────────────
# GROQ AI — GPT OSS 20B → LLaMA FALLBACK
# ──────────────────────────────────────────────────────────────
async def call_ai(comparison: dict) -> dict:
    api_key = os.environ.get("GROQ_API_KEY","").strip()
    if not api_key:
        return _ai_fallback(comparison)

    model = get_active_model()
    logger.info(f"AI: using model {model}")
    gaps = comparison.get("gaps",[])
    keyword = comparison.get("keyword","")
    user_score = comparison.get("user_score",0)
    comp_avg = comparison.get("competitor_avg_score",0)
    prob = comparison.get("ranking_probability",{})
    intent = comparison.get("user_intent","Informational")
    content_type = comparison.get("user_content_type","Article")
    bd = comparison.get("score_breakdown",{})

    system = """You are the world's most experienced SEO strategist with 15+ years mastering Google's algorithms.
You think contextually and strategically, never mechanically. You understand that ranking is about satisfying search intent better than competitors — through content quality, authority signals, technical excellence, and user experience.
You give specific, actionable advice that accounts for keyword intent, competitor patterns, and ranking probability.
Respond ONLY with valid JSON. No markdown, no code fences, no preamble."""

    prompt = f"""Analyse this SEO comparison for "{keyword}" and generate a god-level strategic response.

CONTEXT:
- Keyword: "{keyword}" | Intent: {intent} | Content Type: {content_type}
- Your Page Score: {user_score}/100 | Competitor Avg: {comp_avg}/100
- Ranking Probability: {prob.get('probability',50)}% ({prob.get('label','Moderate')})
- Score Breakdown: {json.dumps(bd)}

HIGH-PRIORITY GAPS:
{json.dumps([g for g in gaps if g['severity']=='high'], indent=2)}

MEDIUM GAPS:
{json.dumps([g for g in gaps if g['severity']=='medium'], indent=2)}

Return ONLY this JSON (no markdown):
{{
  "executive_summary": "3 sentences: current ranking position, core problem, single most important next step. Be brutally specific.",
  "ranking_rationale": "2 sentences: WHY this page will or won't rank for this specific keyword+intent. Reference the actual data.",
  "high_impact": [{{"action":"specific action","why":"exact ranking mechanism affected","effort":"low|medium|high","time_to_impact":"days|weeks|months"}}],
  "medium_impact": [{{"action":"...","why":"...","effort":"...","time_to_impact":"..."}}],
  "low_priority": [{{"action":"...","why":"...","effort":"...","time_to_impact":"..."}}],
  "content_strategy": "3 sentences on content restructuring to dominate this keyword. Reference the intent type and competitor format.",
  "quick_wins": ["action achievable in <1 hour that moves ranking", "..."],
  "competitive_advantages": ["specific thing your page already does better"],
  "content_gaps": ["specific topic/section missing that competitors cover and users search for"],
  "serp_opportunity": "1 sentence on the biggest untapped SERP feature opportunity (featured snippet, PAA, image pack, etc.)"
}}"""

    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role":"system","content":system},{"role":"user","content":prompt}],
                      "temperature": 0.35, "max_tokens": 2000},
                timeout=aiohttp.ClientTimeout(total=50)
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Groq API {resp.status}")
                    return _ai_fallback(comparison)
                data = await resp.json()
                usage = data.get("usage",{})
                total_tokens = usage.get("total_tokens",500)
                record_usage(model, total_tokens)
                content = data["choices"][0]["message"]["content"].strip()
                content = re.sub(r"^```(?:json)?\s*","",content).strip()
                content = re.sub(r"\s*```$","",content).strip()
                result = json.loads(content)
                result["_model"] = model
                result["_tokens"] = total_tokens
                result["_model_status"] = get_model_status()
                return result
    except Exception as e:
        logger.warning(f"AI error: {e}")
        return _ai_fallback(comparison)

def _ai_fallback(comparison: dict) -> dict:
    gaps = comparison.get("gaps",[])
    user_score = comparison.get("user_score",0)
    comp_avg = comparison.get("competitor_avg_score",0)
    high = [g for g in gaps if g["severity"]=="high"]
    med = [g for g in gaps if g["severity"]=="medium"]
    low = [g for g in gaps if g["severity"]=="low"]
    diff = user_score - comp_avg
    return {
        "executive_summary": f"Your page scores {user_score}/100 vs competitor average of {comp_avg}/100 — {'you lead by' if diff>=0 else 'you trail by'} {abs(diff)} points. {len(high)} critical issues require immediate attention. Focus on keyword placement and technical foundations first.",
        "ranking_rationale": f"With {len(gaps)} total gaps, your page's current ranking potential is limited by missing keyword signals and technical issues. Addressing the {len(high)} high-priority gaps will produce the fastest ranking gains.",
        "high_impact": [{"action":g["message"],"why":f"Affects: {g['metric']}","effort":"medium","time_to_impact":"weeks"} for g in high[:4]],
        "medium_impact": [{"action":g["message"],"why":f"Affects: {g['metric']}","effort":"medium","time_to_impact":"weeks"} for g in med[:3]],
        "low_priority": [{"action":g["message"],"why":f"Affects: {g['metric']}","effort":"low","time_to_impact":"months"} for g in low[:3]],
        "content_strategy": "Match the dominant content format of ranking competitors. Ensure comprehensive topic coverage addressing every user sub-intent. Add expert signals including author info, citations, and freshness dates.",
        "quick_wins": ["Add primary keyword to title tag","Set canonical tag to self-reference","Complete Open Graph og:title, og:description, og:image"],
        "competitive_advantages": ["Analysis will surface your advantages after full data loads"],
        "content_gaps": ["Run gap analysis to identify missing topic sections"],
        "serp_opportunity": "Add FAQ schema to target People Also Ask boxes for this query.",
        "_model": "fallback", "_tokens": 0,
    }

# ──────────────────────────────────────────────────────────────
# PAGE EXTRACTOR — 50+ SIGNALS
# ──────────────────────────────────────────────────────────────
async def extract_page(session, url: str, keyword: str, manual: dict) -> dict:
    kw = keyword.lower().strip()
    r = {
        "url": url, "page_status": 0,
        "title": "", "title_length": 0,
        "meta_description": "", "meta_desc_length": 0,
        "h1": "", "h1_count": 0, "h2_count": 0, "h3_count": 0,
        "h2_headings": [], "h3_headings": [], "heading_structure": [],
        "keyword_in_title": False, "keyword_in_h1": False, "keyword_in_meta": False,
        "keyword_in_h2": False, "keyword_in_h3": False,
        "keyword_count": 0, "keyword_density": 0.0, "keyword_intelligence": {},
        "word_count": 0, "paragraph_count": 0,
        "intent": "Unknown", "content_type": "Unknown",
        "published_date": "Not Found", "readability": {},
        "schema": {}, "schema_types": [],
        "internal_links": 0, "external_links": 0,
        "image_count": 0, "alt_text_coverage": 0,
        "canonical": None, "canonical_match": False,
        "og_tags": {}, "og_complete": False, "twitter_tags": {},
        "is_https": url.startswith("https://"),
        "has_viewport": False, "robots_indexable": True,
        "has_faq": False, "has_table": False, "has_video": False,
        "has_toc": False, "has_author": False,
        "heading_hierarchy_ok": False,
        "eeat_signals": {}, "featured_snippet_eligible": {},
        "content_html_ratio": 0,
        "load_speed": None, "seo_score": 0, "seo_score_breakdown": {},
        # Manual
        "da": manual.get("da","N/A"), "pa": manual.get("pa","N/A"),
        "backlinks": manual.get("backlinks","N/A"), "plagiarism": manual.get("plagiarism","N/A"),
    }
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=28), ssl=False, headers=HEADERS, allow_redirects=True) as resp:
            r["page_status"] = resp.status
            if resp.status != 200: return r
            html = await resp.text(errors="ignore")

        soup = BeautifulSoup(html, "html.parser")
        domain = urlparse(url).netloc

        # Title
        tt = soup.find("title")
        title = tt.get_text(strip=True) if tt else ""
        r["title"] = title[:300]; r["title_length"] = len(title)
        r["keyword_in_title"] = kw in title.lower()

        # Meta description
        md = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
        desc = md.get("content","").strip() if md else ""
        r["meta_description"] = desc[:400]; r["meta_desc_length"] = len(desc)
        r["keyword_in_meta"] = kw in desc.lower()

        # Robots meta
        robots_m = soup.find("meta", attrs={"name": re.compile(r"^robots$", re.I)})
        if robots_m:
            rc = robots_m.get("content","").lower()
            r["robots_indexable"] = "noindex" not in rc

        # Viewport
        vp = soup.find("meta", attrs={"name": re.compile(r"^viewport$", re.I)})
        r["has_viewport"] = bool(vp)

        # Headings
        h1s = soup.find_all("h1")
        r["h1_count"] = len(h1s)
        r["h1"] = h1s[0].get_text(strip=True)[:250] if h1s else ""
        r["keyword_in_h1"] = kw in r["h1"].lower()
        h2s = soup.find_all("h2")
        r["h2_count"] = len(h2s)
        r["h2_headings"] = [h.get_text(strip=True)[:150] for h in h2s[:25]]
        h3s = soup.find_all("h3")
        r["h3_count"] = len(h3s)
        r["h3_headings"] = [h.get_text(strip=True)[:120] for h in h3s[:25]]
        h2_text = " ".join(r["h2_headings"]).lower()
        r["keyword_in_h2"] = kw in h2_text
        h3_text = " ".join(r["h3_headings"]).lower()
        r["keyword_in_h3"] = kw in h3_text

        # Heading structure (full tree for heading hierarchy check)
        all_h = soup.find_all(["h1","h2","h3","h4"])
        r["heading_structure"] = [{"tag": h.name, "text": h.get_text(strip=True)[:100]} for h in all_h[:30]]
        # Hierarchy check: no H2 before H1 exists, H3s follow H2s
        h_levels = [int(h.name[1]) for h in all_h]
        r["heading_hierarchy_ok"] = (len(h_levels) >= 2 and h_levels[0] == 1 and all(h_levels[i] <= h_levels[i-1]+1 for i in range(1,len(h_levels))))

        # Clean body text
        for tag in soup.find_all(["script","style","noscript","nav","header","footer","aside","iframe"]):
            tag.decompose()
        clean = soup.get_text(" ", strip=True)
        words = clean.lower().split()
        r["word_count"] = len(words)
        r["paragraph_count"] = len(soup.find_all("p"))
        r["content_html_ratio"] = round(len(clean) / max(len(html), 1) * 100, 1)

        # Keyword metrics
        kw_parts = kw.split()
        kw_count = sum(1 for i in range(len(words)-len(kw_parts)+1) if words[i:i+len(kw_parts)] == kw_parts)
        r["keyword_count"] = kw_count
        r["keyword_density"] = round(kw_count / max(len(words),1) * 100, 2)
        r["keyword_intelligence"] = extract_keyword_intelligence(clean, keyword)

        # Intent + type
        r["intent"] = detect_intent(url, title, r["h1"], desc)
        r["content_type"] = detect_content_type(r["h2_count"], r["word_count"], title, r["h1"])
        r["readability"] = extract_readability(clean)
        r["published_date"] = extract_date(soup)

        # Schema
        schema = extract_schema(soup)
        r["schema"] = schema; r["schema_types"] = schema["types"]

        # Special content signals
        r["has_faq"] = bool(soup.find(class_=re.compile("faq|accordion|qa|question",re.I)) or soup.find_all(["dt","dd"]) or "frequently asked" in clean.lower())
        r["has_table"] = bool(soup.find("table"))
        r["has_video"] = bool(soup.find("iframe", src=re.compile("youtube|vimeo|wistia|loom",re.I)) or soup.find("video"))
        r["has_toc"] = bool(soup.find(id=re.compile("toc|table.?of.?content",re.I)) or soup.find(class_=re.compile("toc|table.?of.?content",re.I)) or "table of contents" in clean.lower())
        r["has_author"] = bool(soup.find(attrs={"itemprop":re.compile("author|person",re.I)}) or soup.find("a",rel="author") or soup.find(class_=re.compile("author|byline|writer",re.I)))

        # Links
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            if not href.startswith("http"): continue
            if urlparse(href).netloc == domain: r["internal_links"] += 1
            else: r["external_links"] += 1

        # Images + alt text
        imgs = soup.find_all("img")
        r["image_count"] = len(imgs)
        with_alt = sum(1 for img in imgs if img.get("alt","").strip())
        r["alt_text_coverage"] = round(with_alt / max(len(imgs),1) * 100) if imgs else 100

        # Canonical
        can = soup.find("link", rel="canonical")
        if can and can.get("href"):
            r["canonical"] = can["href"].strip()
            r["canonical_match"] = normalize_url(can["href"].strip()) == normalize_url(url)

        # OG + Twitter
        og_keys = ["og:title","og:description","og:image","og:type","og:url"]
        og = {}
        for k in og_keys:
            m = soup.find("meta", property=k)
            if m and m.get("content"): og[k] = m["content"][:300]
        r["og_tags"] = og
        r["og_complete"] = bool(og.get("og:title") and og.get("og:description") and og.get("og:image"))
        tw = {}
        for k in ["twitter:card","twitter:title","twitter:description","twitter:image"]:
            m = soup.find("meta", attrs={"name":k})
            if m and m.get("content"): tw[k] = m["content"][:200]
        r["twitter_tags"] = tw

        # E-E-A-T
        r["eeat_signals"] = extract_eeat_signals(soup, clean)

        # Featured snippet eligibility
        r["featured_snippet_eligible"] = extract_featured_snippet_eligibility(soup, clean)

        # PageSpeed
        r["load_speed"] = await run_pagespeed_full(url)

        # SEO Score
        score_r = compute_seo_score(r, keyword)
        r["seo_score"] = score_r["total"]
        r["seo_score_breakdown"] = score_r["breakdown"]

    except Exception as e:
        r["_error"] = str(e)
        logger.error(f"Extract error [{url}]: {e}")

    return r

# ──────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────
class CompareRequest(BaseModel):
    keyword: str
    user_url: str
    competitor_urls: list
    manual_data: dict = {}

@app.post("/serp-compare")
async def serp_compare(req: CompareRequest):
    keyword = req.keyword.strip()
    if not keyword: return {"error": "keyword required"}

    def clean(u):
        u = u.strip()
        if not u.startswith("http"): u = "https://" + u
        return normalize_url(u)

    user_url = clean(req.user_url)
    comp_urls = [clean(u) for u in req.competitor_urls if u.strip()]
    all_urls = [user_url] + comp_urls
    manual = req.manual_data or {}

    async def stream():
        results = []
        conn = aiohttp.TCPConnector(limit=20, ssl=False)
        async with aiohttp.ClientSession(headers=HEADERS, connector=conn) as sess:
            async def process(idx, url):
                d = await extract_page(sess, url, keyword, manual.get(url,{}))
                d["is_user"] = (idx == 0); d["rank"] = "Your Page" if idx == 0 else f"#{idx}"; d["_idx"] = idx
                return d
            tasks = [asyncio.create_task(process(i,u)) for i,u in enumerate(all_urls)]
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                yield json.dumps({"type":"page_result","data":result}) + "\n"
            results.sort(key=lambda r: r.get("_idx",99))
            comparison = compute_comparison(results, user_url, keyword)
            ai = await call_ai(comparison)
            comparison["ai"] = ai
            yield json.dumps({"type":"comparison","data":comparison}) + "\n"
            yield json.dumps({"type":"done","model_status":get_model_status()}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")

@app.get("/test-pagespeed")
async def test_psi(url: str = Query(...)):
    key = os.environ.get("PAGESPEED_API_KEY","")
    result = await run_pagespeed(url, "mobile")
    return {"url": url, "key_present": bool(key), "key_prefix": key[:6]+"..." if key else "NOT SET", "result": result}

@app.get("/health")
async def health():
    return {"status":"ok","version":"v3","active_model":get_active_model(),"model_status":get_model_status()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
