"""
SearchRNK Intelligence — Backend API v3
Dual AI: GPT-20 (openai/gpt-oss-20b) primary | LLaMA 3.1 8B fallback
Auto-switches based on daily quota, resets at midnight UTC.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import aiohttp, asyncio, os, json, re, time, random   # ← ADDED: random
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urldefrag
from collections import Counter
from datetime import datetime, date
# ═══════════════════════════════════════════
app = FastAPI(title="SearchRNK Intelligence API v3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.searchrnk.com", "https://searchrnk.com", "*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}

# ═══════════════════════════════════════════
# ← ADDED: fallback UA list + bot-challenge detector
# Only used when the primary request returns a challenge page or 403/429/503.
# Sites that work normally never hit this code path.
# ═══════════════════════════════════════════
_FALLBACK_HEADERS = [
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
    },
    {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
    },
]

_BLOCK_PATTERNS = [
    "checking your browser", "please wait", "just a moment",
    "enable javascript", "cloudflare", "ddos-guard",
    "captcha", "robot", "bot detection", "access denied",
    "are you human", "security check", "cf-browser-verification",
    "challenge-platform", "ray id",
    # Additional Cloudflare / anti-bot variants
    "verifying your connection", "verifying you are human",
    "attention required", "one more step", "please turn javascript on",
    "browser check", "human verification", "site protection",
    "403 forbidden", "access to this page has been denied",
    "performance & security by", "please complete the security check",
]

def _is_blocked(status: int, body: str, title: str = "") -> bool:
    if status in (403, 429, 503):
        return True
    snippet = body[:3000].lower()
    # Any single pattern match in the title is an immediate block signal
    title_lower = title.lower()
    if any(p in title_lower for p in _BLOCK_PATTERNS):
        return True
    return sum(1 for p in _BLOCK_PATTERNS if p in snippet) >= 2
def _raw_title(html: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    return m.group(1).strip() if m else ""

# ═══════════════════════════════════════════

# ═══════════════════════════════════════════
# DUAL AI MODEL MANAGER
# GPT-20 (openai/gpt-oss-20b): 1K req/day, 200K tokens/day — PRIMARY
# LLaMA 3.1 8B (llama-3.1-8b-instant): 14.4K req/day, 500K tokens/day — FALLBACK
# ═══════════════════════════════════════════
_ai_state = {
    "date": str(date.today()),
    # GPT-20 limits
    "gpt_requests_used": 0,
    "gpt_tokens_used":   0,
    "gpt_req_limit":     950,    # leave buffer below 1000/day
    "gpt_token_limit":   190_000,
    # LLaMA limits
    "llama_requests_used": 0,
    "llama_tokens_used":   0,
    "llama_req_limit":     14_000,
    "llama_token_limit":   480_000,
}
def _reset_if_new_day():
    today = str(date.today())
    if _ai_state["date"] != today:
        _ai_state.update({
            "date": today,
            "gpt_requests_used":   0, "gpt_tokens_used":   0,
            "llama_requests_used": 0, "llama_tokens_used": 0,
        })
def get_active_model() -> dict:
    """Return model config based on quota availability. GPT-20 is always primary."""
    _reset_if_new_day()
    gpt_ok = (
        _ai_state["gpt_requests_used"] < _ai_state["gpt_req_limit"] and
        _ai_state["gpt_tokens_used"]   < _ai_state["gpt_token_limit"]
    )
    if gpt_ok:
        return {
            "id":       "openai/gpt-oss-20b",
            "label":    "GPT-20 (OSS 20B)",
            "provider": "openrouter",
            "max_tok":  2000,
        }
    # Fallback to LLaMA 3.1 8B
    return {
        "id":       "llama-3.1-8b-instant",
        "label":    "LLaMA 3.1 8B",
        "provider": "groq",
        "max_tok":  1800,
    }
def record_usage(provider: str, tokens: int):
    _reset_if_new_day()
    if provider == "openrouter":
        _ai_state["gpt_requests_used"]   += 1
        _ai_state["gpt_tokens_used"]     += tokens
    else:
        _ai_state["llama_requests_used"] += 1
        _ai_state["llama_tokens_used"]   += tokens
# ═══════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════
def normalize_url(url: str) -> str:
    try:
        url, _ = urldefrag(url)
        p = urlparse(url)
        path = p.path.rstrip("/") or "/"
        return f"{p.scheme.lower()}://{p.netloc.lower()}{path}"
    except Exception:
        return url
def detect_intent(url, title, h1, meta) -> str:
    combined = f"{url} {title} {h1} {meta}".lower()
    scores = {
        "Transactional": sum(2 for s in ["buy","shop","price","order","purchase","cart","checkout","add to cart","get started","sign up","subscribe","pricing","plan"] if s in combined),
        "Commercial":    sum(2 for s in ["best","top","review","compare","vs","versus","ranking","rated","alternatives","worth it","pros and cons","which is better"] if s in combined),
        "Navigational":  sum(2 for s in ["login","sign in","account","dashboard","portal","support","contact"] if s in combined),
        "Informational": sum(1 for s in ["how to","what is","guide","tutorial","learn","tips","examples","definition","explained","why","when","who"] if s in combined),
    }
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "Informational"
def detect_content_type(h2_count, word_count, title, h1) -> str:
    combined = f"{title} {h1}".lower()
    if any(w in combined for w in ["top ","best ","list of"," ways","tips ","reasons","examples","tools","plugins","resources"]): return "Listicle"
    if any(w in combined for w in ["how to","guide","tutorial","step by step","setup","install","configure"]): return "How-to Guide"
    if any(w in combined for w in ["review","vs","versus","compared","comparison","alternatives"]): return "Comparison / Review"
    if any(w in combined for w in ["what is","definition","meaning","explained","overview"]): return "Explainer"
    if h2_count >= 8 and word_count >= 2500: return "Pillar / Long-form"
    if word_count < 700: return "Short-form"
    return "Article"
def compute_readability(text: str) -> dict:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip().split()) > 3]
    if not sentences: return {"label":"Unknown","avg_words":0,"complex_word_pct":0}
    avg  = sum(len(s.split()) for s in sentences) / len(sentences)
    long = sum(1 for w in text.split() if len(w) > 7)
    pct  = long / max(len(text.split()), 1) * 100
    label = "Easy" if avg < 14 and pct < 20 else "Moderate" if avg < 24 else "Hard"
    return {"label":label, "avg_words":round(avg,1), "complex_word_pct":round(pct,1)}
def detect_schema(soup) -> dict:
    schemas, previews = [], []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data  = json.loads(script.string or "{}")
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    t = item.get("@type","")
                    if t: schemas.append(str(t))
                    previews.append({k:v for k,v in item.items() if k in ["@type","name","description","url","image","author","datePublished","headline"]})
        except Exception:
            pass
    return {"types":list(set(schemas)), "count":len(schemas), "previews":previews[:5]}
def extract_date(soup) -> str:
    t = soup.find("time", attrs={"datetime":True})
    if t: return t.get("datetime","")[:10]
    for prop in ["article:published_time","og:article:published_time","article:modified_time"]:
        m = soup.find("meta", property=prop)
        if m and m.get("content"): return m["content"][:10]
    for itemprop in ["datePublished","dateModified"]:
        m = soup.find(attrs={"itemprop":itemprop})
        if m:
            val = m.get("content") or m.get_text(strip=True)
            if val: return str(val)[:10]
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "{}")
            if isinstance(data, dict):
                d = data.get("datePublished") or data.get("dateModified")
                if d: return str(d)[:10]
        except Exception:
            pass
    return "Not Found"
def extract_keyword_intelligence(text: str, keyword: str) -> dict:
    words    = re.findall(r"\b[a-z]+\b", text.lower())
    kw_lower = keyword.lower()
    kw_parts = set(kw_lower.split())
    bigrams  = [f"{words[i]} {words[i+1]}"          for i in range(len(words)-1)]
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
    fourgrams= [f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}" for i in range(len(words)-3)]
    rel_bi   = Counter(ng for ng in bigrams   if any(kp in ng for kp in kw_parts) and ng != kw_lower)
    rel_tri  = Counter(ng for ng in trigrams  if any(kp in ng for kp in kw_parts) and ng != kw_lower)
    rel_four = Counter(ng for ng in fourgrams if any(kp in ng for kp in kw_parts))
    variations, seen = [], set()
    for v in ([ng for ng,_ in rel_tri.most_common(5)] +
              [ng for ng,_ in rel_bi.most_common(5)]  +
              [ng for ng,_ in rel_four.most_common(3)]):
        if v not in seen and v != kw_lower:
            seen.add(v); variations.append(v)
    STOP = {"the","and","for","are","but","not","you","all","can","was","one","our","out","day","get","has","him","his","how","new","now","old","see","two","way","who","did","its","let","say","too","use","with","that","this","from","they","will","been","have","here","more","also","into","than","then","them","some","what","when","your"}
    topics = [ng for ng,cnt in Counter(trigrams).most_common(30)
              if cnt >= 2 and not any(w in STOP for w in ng.split())]
    return {
        "primary":   keyword,
        "variations":variations[:12],
        "topic_clusters":topics[:8],
        "total_variation_count":len(variations),
    }
def compute_seo_score(data: dict, keyword: str) -> dict:
    dims = {k:0 for k in ["content_relevance","on_page_signals","structure_depth","technical_health","content_quality","ux_signals"]}
    wc, intent = data.get("word_count",0), data.get("intent","Informational")
    # Content relevance (max 25)
    if intent in ("Transactional","Navigational"):
        dims["content_relevance"] += 10 if wc>=500 else 6 if wc>=200 else 0
    else:
        dims["content_relevance"] += 15 if wc>=3000 else 12 if wc>=2000 else 8 if wc>=1200 else 4 if wc>=600 else 2 if wc>=200 else 0
    kd = data.get("keyword_density",0)
    if 0.3<=kd<=3.0:  dims["content_relevance"]+=8
    elif 0.1<=kd<=4.0:dims["content_relevance"]+=4
    if data.get("keyword_count",0)>=5: dims["content_relevance"]+=2
    if data.get("keyword_in_h2"):  dims["content_relevance"]+=3
    if data.get("keyword_in_h3"):  dims["content_relevance"]+=2
    # On-page signals (max 20)
    if data.get("keyword_in_title"): dims["on_page_signals"]+=8
    if data.get("keyword_in_h1"):    dims["on_page_signals"]+=7
    if data.get("keyword_in_meta"):  dims["on_page_signals"]+=5
    tl = data.get("title_length",0)
    if tl>70: dims["on_page_signals"]-=2
    # Structure (max 15)
    h1c,h2c,h3c = data.get("h1_count",0),data.get("h2_count",0),data.get("h3_count",0)
    dims["structure_depth"] += 4 if h1c==1 else (-2 if h1c==0 else 0)
    dims["structure_depth"] += 6 if h2c>=8 else 4 if h2c>=5 else 2 if h2c>=3 else 1 if h2c>=1 else 0
    dims["structure_depth"] += 5 if h3c>=5 else 3 if h3c>=3 else 1 if h3c>=1 else 0
    # Technical (max 20)
    if data.get("canonical"):      dims["technical_health"]+=6
    if data.get("canonical_match"):dims["technical_health"]+=2
    og = data.get("og_tags",{})
    dims["technical_health"] += min(sum(1 for k in ["og:title","og:description","og:image","og:type","og:url"] if og.get(k)),5)
    if data.get("og_complete"):    dims["technical_health"]+=2
    sp = data.get("load_speed",{})
    if isinstance(sp,dict):
        mob = sp.get("mobile",{})
        if isinstance(mob,dict) and isinstance(mob.get("score"),(int,float)):
            ms=mob["score"]
            dims["technical_health"]+=5 if ms>=90 else 3 if ms>=70 else 1 if ms>=50 else 0
    # Content quality (max 10)
    r = data.get("readability",{})
    rl = r if isinstance(r,str) else r.get("label","Unknown")
    dims["content_quality"] += {"Easy":8,"Moderate":6,"Hard":3}.get(rl,0)
    if data.get("schema_types"): dims["content_quality"]+=2
    # UX (max 10)
    imgs=data.get("image_count",0)
    dims["ux_signals"]+=5 if imgs>=8 else 3 if imgs>=4 else 1 if imgs>=1 else 0
    il=data.get("internal_links",0)
    dims["ux_signals"]+=5 if il>=15 else 3 if il>=8 else 1 if il>=3 else 0
    for k,maxv in [("content_relevance",25),("on_page_signals",20),("structure_depth",15),("technical_health",20),("content_quality",10),("ux_signals",10)]:
        dims[k]=max(0,min(maxv,dims[k]))
    return {"total":min(sum(dims.values()),100), "breakdown":dims}
def compute_ranking_probability(user: dict, competitors: list, keyword: str) -> dict:
    if not competitors:
        return {"probability":50,"label":"Moderate","rationale":"No competitors to compare against."}
    user_score  = user.get("seo_score",0)
    comp_scores = [c.get("seo_score",0) for c in competitors if c.get("seo_score",0)>0]
    avg_comp    = sum(comp_scores)/len(comp_scores) if comp_scores else 50
    base = 50 + (user_score - avg_comp) * 0.8
    if user.get("keyword_in_title") and user.get("keyword_in_h1"): base+=8
    if not user.get("keyword_in_title"): base-=10
    if not user.get("canonical"):        base-=5
    if user.get("schema_types"):         base+=4
    if user.get("og_complete"):          base+=3
    sp = user.get("load_speed",{})
    if isinstance(sp,dict):
        mob=sp.get("mobile",{})
        if isinstance(mob,dict) and isinstance(mob.get("score"),(int,float)):
            if mob["score"]<50:   base-=12
            elif mob["score"]>=90:base+=6
    prob  = max(5,min(95,round(base)))
    label = "High" if prob>=70 else "Moderate" if prob>=45 else "Low"
    return {"probability":prob,"label":label}
# ═══════════════════════════════════════════
# PAGESPEED
# ═══════════════════════════════════════════
async def run_pagespeed(url: str) -> dict:
    key   = os.environ.get("PAGESPEED_API_KEY","").strip()
    empty = {"score":None,"fcp":"N/A","lcp":"N/A","tbt":"N/A","cls":"N/A","si":"N/A","rating":"No Data"}
    if not key: return {"mobile":dict(empty),"desktop":dict(empty)}
    def parse_metric(audits, audit_key, unit="s"):
        item = audits.get(audit_key,{})
        if "displayValue" in item: return item["displayValue"]
        if "numericValue"  in item:
            val=item["numericValue"]
            return f"{int(val)} ms" if unit=="ms" else f"{val/1000:.2f} s"
        return "N/A"
    def rating(s):
        if not isinstance(s,(int,float)): return "N/A"
        return "Fast" if s>=90 else "Needs Work" if s>=50 else "Slow"
    async def fetch_strategy(strategy):
        try:
            api_url=(f"https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
                     f"?url={url}&strategy={strategy}&category=performance&key={key}")
            async with aiohttp.ClientSession() as sess:
                async with sess.get(api_url,timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    data=await resp.json()
                    if "error" in data or "lighthouseResult" not in data: return None
                    lh=data["lighthouseResult"]; audits=lh["audits"]
                    perf=lh["categories"]["performance"]["score"]
                    score=int(perf*100) if perf is not None else 0
                    return {"score":score,
                            "fcp":parse_metric(audits,"first-contentful-paint"),
                            "lcp":parse_metric(audits,"largest-contentful-paint"),
                            "tbt":parse_metric(audits,"total-blocking-time","ms"),
                            "cls":parse_metric(audits,"cumulative-layout-shift",""),
                            "si": parse_metric(audits,"speed-index"),
                            "rating":rating(score)}
        except Exception:
            return None
    try:
        mob,desk = await asyncio.gather(fetch_strategy("mobile"),fetch_strategy("desktop"),return_exceptions=True)
    except Exception:
        mob=desk=None
    return {
        "mobile":  mob  if (mob  and not isinstance(mob,Exception))  else dict(empty),
        "desktop": desk if (desk and not isinstance(desk,Exception)) else dict(empty),
    }
# ═══════════════════════════════════════════
# PAGE EXTRACTOR
# ═══════════════════════════════════════════
async def extract_page(session, url: str, keyword: str, manual: dict) -> dict:
    kw = keyword.lower().strip()
    result = {
        "url":url,"page_status":0,
        "title":"","title_length":0,
        "meta_description":"","meta_desc_length":0,
        "h1":"","h1_count":0,"h2_count":0,"h3_count":0,
        "h2_headings":[],"h3_headings":[],
        "keyword_in_title":False,"keyword_in_h1":False,"keyword_in_meta":False,
        "keyword_in_h2":False,"keyword_in_h3":False,
        "keyword_count":0,"keyword_density":0.0,"keyword_intelligence":{},
        "word_count":0,"intent":"Unknown","content_type":"Unknown",
        "published_date":"Not Found","readability":{"label":"Unknown","avg_words":0},
        "schema":{"types":[],"count":0,"previews":[]},"schema_types":[],
        "internal_links":0,"external_links":0,"image_count":0,
        "canonical":None,"canonical_match":False,
        "og_tags":{},"og_complete":False,"load_speed":None,
        "seo_score":0,"seo_score_breakdown":{},
        "da":manual.get("da","N/A"),"pa":manual.get("pa","N/A"),
        "backlinks":manual.get("backlinks","N/A"),"plagiarism":manual.get("plagiarism","N/A"),
    }
    try:
        # ── Primary attempt: exactly as original ──────────────────────────────
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=25), ssl=False,
                               headers=HEADERS, allow_redirects=True) as resp:
            result["page_status"] = resp.status
            text = await resp.text(errors="ignore")

        # ── ADDED: bot-challenge handling + Google Cache fallback ───────────
        # Sites that work normally skip this entire block.
        if _is_blocked(result["page_status"], text, _raw_title(text)):

            # Step 1 — retry with 2 fallback User-Agents
            for fb_headers in _FALLBACK_HEADERS:
                await asyncio.sleep(random.uniform(1.0, 2.0))
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=25), ssl=False,
                                           headers=fb_headers, allow_redirects=True) as resp2:
                        result["page_status"] = resp2.status
                        candidate = await resp2.text(errors="ignore")
                        if not _is_blocked(resp2.status, candidate, _raw_title(candidate)):
                            text = candidate
                            break
                except Exception:
                    continue

            # Step 2 — if still blocked, try Google Cache (free, no API key needed)
            if _is_blocked(result["page_status"], text, _raw_title(text)):
                cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{url}&hl=en"
                try:
                    cache_headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Referer": "https://www.google.com/",
                    }
                    async with session.get(cache_url, timeout=aiohttp.ClientTimeout(total=20),
                                           ssl=False, headers=cache_headers,
                                           allow_redirects=True) as cr:
                        if cr.status == 200:
                            cached = await cr.text(errors="ignore")
                            if not _is_blocked(cr.status, cached, _raw_title(cached)) and len(cached) > 2000:
                                text = cached
                                result["page_status"] = 200
                                result["_crawl_note"] = "Data sourced from Google Cache (site blocked direct access)"
                except Exception:
                    pass

        # ── If still blocked after all attempts, return clean empty result ────
        if result["page_status"] != 200 or _is_blocked(result["page_status"], text, _raw_title(text)):
            result["_block_reason"] = "Bot protection active — could not retrieve real page content"
            return result
        # ─────────────────────────────────────────────────────────────────────

        soup=BeautifulSoup(text,"html.parser")
        domain=urlparse(url).netloc
        # Title
        tt=soup.find("title"); title=tt.get_text(strip=True) if tt else ""
        result.update({"title":title[:300],"title_length":len(title),"keyword_in_title":kw in title.lower()})
        # Meta description
        mt=soup.find("meta",attrs={"name":re.compile(r"^description$",re.I)})
        desc=mt.get("content","").strip() if mt else ""
        result.update({"meta_description":desc[:400],"meta_desc_length":len(desc),"keyword_in_meta":kw in desc.lower()})
        # Headings
        h1s=soup.find_all("h1"); result["h1_count"]=len(h1s)
        result["h1"]=h1s[0].get_text(strip=True)[:250] if h1s else ""
        result["keyword_in_h1"]=kw in result["h1"].lower()
        h2s=soup.find_all("h2"); result["h2_count"]=len(h2s)
        result["h2_headings"]=[h.get_text(strip=True)[:150] for h in h2s[:20]]
        result["keyword_in_h2"]=kw in " ".join(h.lower() for h in result["h2_headings"])
        h3s=soup.find_all("h3"); result["h3_count"]=len(h3s)
        result["h3_headings"]=[h.get_text(strip=True)[:120] for h in h3s[:20]]
        result["keyword_in_h3"]=kw in " ".join(h.lower() for h in result["h3_headings"])
        # Clean body text
        for tag in soup.find_all(["script","style","noscript","nav","header","footer","aside","iframe"]): tag.decompose()
        clean=soup.get_text(" ",strip=True); words=clean.lower().split()
        result["word_count"]=len(words)
        # Keyword density
        kw_parts=kw.split()
        kw_count=sum(1 for i in range(len(words)-len(kw_parts)+1) if words[i:i+len(kw_parts)]==kw_parts)
        result["keyword_count"]=kw_count
        result["keyword_density"]=round((kw_count/max(len(words),1))*100,2)
        result["keyword_intelligence"]=extract_keyword_intelligence(clean,keyword)
        # Metadata
        result["intent"]=detect_intent(url,title,result["h1"],desc)
        result["content_type"]=detect_content_type(result["h2_count"],result["word_count"],title,result["h1"])
        result["readability"]=compute_readability(clean)
        result["published_date"]=extract_date(soup)
        # Schema
        si=detect_schema(soup); result["schema"]=si; result["schema_types"]=si["types"]
        # Links
        for a in soup.find_all("a",href=True):
            href=urljoin(url,a["href"])
            if not href.startswith("http"): continue
            if urlparse(href).netloc==domain: result["internal_links"]+=1
            else: result["external_links"]+=1
        # Images
        result["image_count"]=len(soup.find_all("img"))
        # Canonical
        ct=soup.find("link",rel="canonical")
        if ct and ct.get("href"):
            result["canonical"]=ct["href"].strip()
            result["canonical_match"]=normalize_url(ct["href"].strip())==normalize_url(url)
        # OG tags
        og={}
        for prop in ["og:title","og:description","og:image","og:type","og:url"]:
            m=soup.find("meta",property=prop)
            if m and m.get("content"): og[prop]=m["content"][:300]
        result["og_tags"]=og
        result["og_complete"]=bool(og.get("og:title") and og.get("og:description") and og.get("og:image"))
        # PageSpeed
        result["load_speed"]=await run_pagespeed(url)
        # Score
        sr=compute_seo_score(result,keyword)
        result["seo_score"]=sr["total"]; result["seo_score_breakdown"]=sr["breakdown"]
    except Exception as e:
        result["_error"]=str(e)
    return result
# ═══════════════════════════════════════════
# COMPARISON ENGINE
# ═══════════════════════════════════════════
def compute_comparison(results: list, user_url: str, keyword: str) -> dict:
    user  = next((r for r in results if r.get("is_user")),None)
    comps = [r for r in results if not r.get("is_user") and r.get("page_status",0)==200]
    if not user or not comps:
        return {"gaps":[],"competitor_averages":{},"user_score":user.get("seo_score",0) if user else 0,
                "competitor_avg_score":0,"ranking_probability":{"probability":0,"label":"Unknown"}}
    def avg(key):
        vs=[r.get(key,0) for r in comps if isinstance(r.get(key),(int,float)) and r.get(key,0)>0]
        return round(sum(vs)/len(vs),1) if vs else 0
    comp_avg={k:avg(k) for k in ["word_count","h2_count","h3_count","keyword_count",
                                   "internal_links","image_count","seo_score","keyword_density"]}
    gaps=[]
    # ─── Gap: Word count
    wc_diff=user["word_count"]-comp_avg["word_count"]
    intent=user.get("intent","Informational")
    if wc_diff<-400 and intent not in ("Transactional","Navigational"):
        sev="critical" if wc_diff<-1500 else "high" if wc_diff<-1200 else "medium"
        gaps.append({"type":"content","severity":sev,"metric":"Content Depth",
            "user_val":f"{user['word_count']:,} words","comp_avg":f"{int(comp_avg['word_count']):,} words",
            "message":f"Your content is {abs(int(wc_diff)):,} words shorter than the competitor average. "
                      f"For '{keyword}' ({intent} intent), deeper coverage signals authority to Google. "
                      f"Add sections covering missing sub-topics, FAQs, and supporting data."})
    # ─── Gap: Keyword in title
    kw_title_cnt=sum(1 for c in comps if c.get("keyword_in_title"))
    if not user.get("keyword_in_title") and kw_title_cnt>0:
        gaps.append({"type":"keyword","severity":"critical","metric":"Keyword in Title Tag",
            "user_val":"Missing","comp_avg":f"{kw_title_cnt}/{len(comps)} competitors",
            "message":f"Primary keyword '{keyword}' is absent from your title tag — the single most important on-page signal. "
                      f"{kw_title_cnt}/{len(comps)} competitors include it. This alone explains significant ranking gaps."})
    # ─── Gap: Keyword in H1
    if not user.get("keyword_in_h1"):
        kw_h1_cnt=sum(1 for c in comps if c.get("keyword_in_h1"))
        if kw_h1_cnt>=1:
            gaps.append({"type":"keyword","severity":"high","metric":"Keyword in H1 Tag",
                "user_val":"Missing","comp_avg":f"{kw_h1_cnt}/{len(comps)} competitors",
                "message":f"Keyword missing from H1 — your page's primary content declaration. "
                          f"Google gives significant weight to H1 alignment with the target query. "
                          f"{kw_h1_cnt} competitors include it."})
    # ─── Gap: Keyword density
    kd=user.get("keyword_density",0)
    if kd<0.2:
        gaps.append({"type":"keyword","severity":"medium","metric":"Keyword Density",
            "user_val":f"{kd}%","comp_avg":f"~{comp_avg['keyword_density']}%",
            "message":f"Keyword appears too infrequently ({kd}% density). Use '{keyword}' more naturally throughout — "
                      f"in the intro, body sections, and conclusion. Target 0.5–2% with semantic variation."})
    # ─── Gap: Keyword in meta
    if not user.get("keyword_in_meta"):
        kw_meta_cnt=sum(1 for c in comps if c.get("keyword_in_meta"))
        if kw_meta_cnt>=len(comps)//2:
            gaps.append({"type":"keyword","severity":"medium","metric":"Keyword in Meta Description",
                "user_val":"Missing","comp_avg":f"{kw_meta_cnt}/{len(comps)} competitors",
                "message":f"Keyword absent from meta description. Google bolds matching keywords in search snippets "
                          f"— directly increasing CTR. {kw_meta_cnt} competitors use it."})
    # ─── Gap: H2 structure
    h2_diff=user["h2_count"]-comp_avg["h2_count"]
    if h2_diff<-3:
        gaps.append({"type":"structure","severity":"medium","metric":"H2 Heading Structure",
            "user_val":f"{user['h2_count']} H2s","comp_avg":f"~{int(comp_avg['h2_count'])} H2s",
            "message":f"Your page has {abs(int(h2_diff))} fewer H2 headings than competitors. More structured sections mean "
                      f"more keyword variation opportunities, better crawlability, and improved user experience."})
    # ─── Gap: Multiple H1s
    if user.get("h1_count",0)>1:
        gaps.append({"type":"structure","severity":"high","metric":"Multiple H1 Tags",
            "user_val":f"{user['h1_count']} H1 tags","comp_avg":"1 H1 (standard)",
            "message":f"You have {user['h1_count']} H1 tags — only 1 is correct. Multiple H1s dilute the primary content signal "
                      f"and confuse search engines about your page's main topic."})
    # ─── Gap: Missing H1
    if user.get("h1_count",0)==0:
        gaps.append({"type":"structure","severity":"critical","metric":"Missing H1 Tag",
            "user_val":"0 H1 tags","comp_avg":"1 H1 (standard)",
            "message":"No H1 tag detected. The H1 is the primary on-page content signal — its absence means Google "
                      "cannot determine the page's main topic from structure."})
    # ─── Gap: Canonical
    if not user.get("canonical"):
        gaps.append({"type":"technical","severity":"medium","metric":"Missing Canonical Tag",
            "user_val":"Missing","comp_avg":"Expected for all pages",
            "message":"No canonical tag found. Without it, Google may index duplicate or near-duplicate versions of your page, "
                      "splitting ranking signals and diluting authority across multiple URLs."})
    # ─── Gap: Canonical mismatch
    if user.get("canonical") and not user.get("canonical_match"):
        gaps.append({"type":"technical","severity":"high","metric":"Canonical URL Mismatch",
            "user_val":"External canonical","comp_avg":"Self-referencing canonical",
            "message":"Your canonical tag points to a different URL than the current page. This tells Google to credit "
                      "another URL for this page's content — actively harming this page's ranking ability."})
    # ─── Gap: Schema
    if not user.get("schema_types"):
        comp_schema=sum(1 for c in comps if c.get("schema_types"))
        if comp_schema>=1:
            gaps.append({"type":"technical","severity":"low","metric":"Structured Data (Schema)",
                "user_val":"None detected","comp_avg":f"{comp_schema}/{len(comps)} use schema",
                "message":f"{comp_schema} competitors use structured data markup. Schema enables rich results "
                          f"(stars, FAQs, breadcrumbs) — directly increasing CTR and signalling content authority to Google."})
    # ─── Gap: OG incomplete
    if not user.get("og_complete"):
        gaps.append({"type":"technical","severity":"low","metric":"Open Graph Tags Incomplete",
            "user_val":"Partial or missing","comp_avg":"Full set (5 tags)",
            "message":"OG tags (title, description, image) are incomplete. These control how your page appears when shared "
                      "on social platforms — affecting referral traffic quality and indirect ranking signals."})
    # ─── Gap: Internal links
    il_diff=user["internal_links"]-comp_avg["internal_links"]
    if il_diff<-10 and comp_avg["internal_links"]>5:
        gaps.append({"type":"structure","severity":"medium","metric":"Internal Link Count",
            "user_val":f"{user['internal_links']} links","comp_avg":f"~{int(comp_avg['internal_links'])} links",
            "message":f"Your page has {abs(int(il_diff))} fewer internal links. Internal links distribute PageRank, "
                      f"improve crawl depth, and keep users navigating — all positive ranking signals."})
    # ─── Gap: Images
    img_diff=user["image_count"]-comp_avg["image_count"]
    if img_diff<-4 and comp_avg["image_count"]>3:
        gaps.append({"type":"content","severity":"low","metric":"Visual Content",
            "user_val":f"{user['image_count']} images","comp_avg":f"~{int(comp_avg['image_count'])} images",
            "message":f"Competitors average {int(comp_avg['image_count'])} images vs your {user['image_count']}. "
                      f"Images improve dwell time, reduce bounce rate, and provide alt-text keyword opportunities."})
    # ─── Gap: Mobile speed
    sp=user.get("load_speed",{})
    if isinstance(sp,dict):
        mob=sp.get("mobile",{})
        if isinstance(mob,dict) and isinstance(mob.get("score"),(int,float)) and mob["score"]<50:
            gaps.append({"type":"speed","severity":"critical","metric":"Mobile Page Speed",
                "user_val":f"{mob['score']}/100","comp_avg":"Aim ≥ 90/100",
                "message":f"Mobile speed score of {mob['score']}/100 is critically low. Core Web Vitals are a confirmed "
                          f"Google ranking factor since 2021. A score below 50 actively suppresses your mobile rankings."})
        elif isinstance(mob,dict) and isinstance(mob.get("score"),(int,float)) and mob["score"]<70:
            gaps.append({"type":"speed","severity":"high","metric":"Mobile Page Speed",
                "user_val":f"{mob['score']}/100","comp_avg":"Aim ≥ 90/100",
                "message":f"Mobile speed score of {mob['score']}/100 needs improvement. 53% of mobile users abandon "
                          f"sites taking over 3 seconds to load — directly impacting bounce rate and rankings."})
    # Sort by severity
    order={"critical":0,"high":1,"medium":2,"low":3}
    gaps.sort(key=lambda g:order.get(g["severity"],4))
    # Notes
    user_type=user.get("content_type","Unknown")
    comp_types=[c.get("content_type","Unknown") for c in comps]
    most_common=Counter(comp_types).most_common(1)[0][0] if comp_types else "Unknown"
    content_type_note=None
    if user_type!=most_common and most_common!="Unknown":
        content_type_note=(f"⚠️ Format mismatch: your page is '{user_type}' but "
                           f"{comp_types.count(most_common)}/{len(comps)} competitors use '{most_common}' format — "
                           f"which may better match the search intent for this query.")
    comp_dates=[c.get("published_date") for c in comps if c.get("published_date") not in ("Not Found",None,"")]
    freshness_note=None
    if comp_dates:
        recent=[d for d in comp_dates if d>="2024"]
        if recent and user.get("published_date","Not Found")=="Not Found":
            freshness_note=(f"📅 {len(recent)}/{len(comps)} competitors have date-stamped content (recently updated). "
                            f"Adding datePublished schema and a 'Last updated' line signals freshness to Google.")
    ranking_prob=compute_ranking_probability(user,comps,keyword)
    return {
        "user_score":         user.get("seo_score",0),
        "competitor_avg_score":round(comp_avg["seo_score"],0),
        "competitor_averages": comp_avg,
        "gaps":               gaps,
        "keyword":            keyword,
        "user_intent":        user.get("intent","Unknown"),
        "user_content_type":  user.get("content_type","Unknown"),
        "content_type_note":  content_type_note,
        "freshness_note":     freshness_note,
        "ranking_probability":ranking_prob,
        "user_url":           user_url,
        "score_breakdown":    user.get("seo_score_breakdown",{}),
    }
# ═══════════════════════════════════════════
# AI ANALYSIS (DUAL MODEL)
# ═══════════════════════════════════════════
async def call_ai(comparison: dict) -> dict:
    model_cfg = get_active_model()
    gaps       = comparison.get("gaps",[])
    user_score = comparison.get("user_score",0)
    comp_avg   = comparison.get("competitor_avg_score",0)
    keyword    = comparison.get("keyword","")
    intent     = comparison.get("user_intent","Informational")
    ctype      = comparison.get("user_content_type","Article")
    prob       = comparison.get("ranking_probability",{})
    high_gaps  = [g for g in gaps if g["severity"] in ("critical","high")]
    med_gaps   = [g for g in gaps if g["severity"]=="medium"]
    low_gaps   = [g for g in gaps if g["severity"]=="low"]
    system_prompt = (
        "You are a 12+ year veteran SEO strategist. You think contextually — ranking is about satisfying "
        "search intent better than competitors, not mechanical checklists. "
        "Provide expert, specific, actionable advice. Explain WHY things matter for rankings, not just WHAT to fix. "
        "Be direct and practical. Always respond with valid JSON only — no markdown, no preamble, no backticks."
    )
    user_prompt = f"""Analyse this SEO comparison and generate expert, context-aware recommendations.
KEYWORD: "{keyword}"
SEARCH INTENT: {intent}
CONTENT FORMAT: {ctype}
USER SEO SCORE: {user_score}/100
COMPETITOR AVG: {comp_avg}/100
RANKING PROBABILITY: {prob.get('probability',50)}% ({prob.get('label','Moderate')})
CRITICAL/HIGH GAPS (fix first):
{json.dumps(high_gaps[:6], indent=2)}
MEDIUM GAPS:
{json.dumps(med_gaps[:4], indent=2)}
LOW PRIORITY:
{json.dumps(low_gaps[:3], indent=2)}
Return ONLY this JSON structure (no markdown, no backticks):
{{
  "executive_summary": "2-3 sentences: current state, key competitive advantage or disadvantage, single most important next action. Be specific and data-driven.",
  "ranking_rationale": "1-2 sentences explaining WHY this page is or isn't positioned to rank for this keyword and intent.",
  "critical_actions": [{{"action":"...", "why":"...", "effort":"low|medium|high"}}],
  "high_impact": [{{"action":"...", "why":"...", "effort":"low|medium|high"}}],
  "medium_impact": [{{"action":"...", "why":"...", "effort":"low|medium|high"}}],
  "low_priority": [{{"action":"...", "why":"...", "effort":"low|medium|high"}}],
  "content_strategy": "2-3 sentences on restructuring/expanding content to match intent and beat competitors.",
  "quick_wins": ["under-30-min action 1", "action 2", "action 3"],
  "competitive_advantages": ["specific thing user page does better than competitors"],
  "content_gaps": ["specific topic or section missing that top competitors cover"]
}}"""
    # ─── Try primary model (GPT-20 via OpenRouter)
    if model_cfg["provider"] == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY","").strip()
        if key:
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization":f"Bearer {key}","Content-Type":"application/json",
                                 "HTTP-Referer":"https://www.searchrnk.com","X-Title":"SearchRNK Intelligence"},
                        json={"model":model_cfg["id"],
                              "messages":[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
                              "temperature":0.35,"max_tokens":model_cfg["max_tok"]},
                        timeout=aiohttp.ClientTimeout(total=50)
                    ) as resp:
                        if resp.status == 200:
                            data   = await resp.json()
                            usage  = data.get("usage",{})
                            record_usage("openrouter", usage.get("total_tokens",600))
                            content = data["choices"][0]["message"]["content"].strip()
                            content = re.sub(r"^```(?:json)?\s*","",content)
                            content = re.sub(r"\s*```$","",content)
                            result  = json.loads(content)
                            result["_model"] = model_cfg["label"]
                            return result
            except Exception as e:
                pass  # fall through to LLaMA
    # ─── Fallback: LLaMA 3.1 8B via Groq
    groq_key = os.environ.get("GROQ_API_KEY","").strip()
    if groq_key:
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization":f"Bearer {groq_key}","Content-Type":"application/json"},
                    json={"model":"llama-3.1-8b-instant",
                          "messages":[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
                          "temperature":0.4,"max_tokens":1800},
                    timeout=aiohttp.ClientTimeout(total=45)
                ) as resp:
                    if resp.status == 200:
                        data   = await resp.json()
                        usage  = data.get("usage",{})
                        record_usage("groq", usage.get("total_tokens",500))
                        content = data["choices"][0]["message"]["content"].strip()
                        content = re.sub(r"^```(?:json)?\s*","",content)
                        content = re.sub(r"\s*```$","",content)
                        result  = json.loads(content)
                        result["_model"] = "LLaMA 3.1 8B"
                        return result
        except Exception:
            pass
    return _fallback_ai(comparison, model_cfg["label"])
def _fallback_ai(comparison: dict, model_label="AI Analysis") -> dict:
    gaps=comparison.get("gaps",[])
    us=comparison.get("user_score",0); ca=comparison.get("competitor_avg_score",0)
    keyword=comparison.get("keyword",""); diff=us-ca
    high=[g for g in gaps if g["severity"] in ("critical","high")]
    med= [g for g in gaps if g["severity"]=="medium"]
    low= [g for g in gaps if g["severity"]=="low"]
    return {
        "executive_summary": f"Your page scores {us}/100, {abs(diff)} points {'ahead of' if diff>=0 else 'behind'} the competitor average ({ca}/100). {len(high)} critical issues require immediate attention.",
        "ranking_rationale": f"Addressing the top {min(3,len(high))} high-priority gaps — especially keyword placement and technical signals — will have the most direct impact on rankings for '{keyword}'.",
        "critical_actions": [{"action":g["message"],"why":f"Metric: {g['metric']}","effort":"medium"} for g in high[:3]],
        "high_impact":      [{"action":g["message"],"why":f"Metric: {g['metric']}","effort":"medium"} for g in high[3:6]],
        "medium_impact":    [{"action":g["message"],"why":f"Metric: {g['metric']}","effort":"medium"} for g in med[:4]],
        "low_priority":     [{"action":g["message"],"why":f"Metric: {g['metric']}","effort":"low"}    for g in low[:3]],
        "content_strategy": "Match the dominant content format of top-ranking pages, ensure comprehensive topic coverage, and align content depth with the search intent for this keyword.",
        "quick_wins": ["Add primary keyword to title tag if missing","Set canonical tag to prevent duplicate indexing","Complete Open Graph tags for better social sharing"],
        "competitive_advantages": ["Your current strengths will be identified after full gap analysis"],
        "content_gaps": ["Run a full content audit against top-3 ranking pages to identify missing sections"],
        "_model": model_label,
        "_tokens": 0,
    }
# ═══════════════════════════════════════════
# CHAT ENDPOINT
# ═══════════════════════════════════════════
async def call_ai_chat(message: str, history: list, context: str, keyword: str) -> str:
    model_cfg = get_active_model()
    system = f"""You are an expert SEO strategist assistant embedded inside the SearchRNK Intelligence tool.
You have FULL context of the user's current SEO analysis. Use this data to give specific, accurate answers.
ANALYSIS CONTEXT:
{context}
STRICT RULES:
1. ONLY answer questions related to SEO, this analysis, rankings, or digital marketing.
2. If asked about unrelated topics (weather, cooking, etc.), politely redirect: "I'm focused on your SEO analysis — ask me anything about your rankings, gaps, or strategy!"
3. Always be specific — reference actual numbers, URLs, and data from the context above.
4. Be direct, practical, and actionable. Think like a senior SEO consultant.
5. If the context doesn't have enough data, say so clearly and suggest what to look for.
6. Format responses clearly. Use bold **text** for key points. Use bullet points with newlines for lists.
7. Never make up data that isn't in the context."""
    msgs = [{"role":"system","content":system}]
    for h in history[-10:]:
        if h.get("role") in ("user","assistant"):
            msgs.append({"role":h["role"],"content":h["content"]})
    msgs.append({"role":"user","content":message})
    # Try GPT-20 first
    if model_cfg["provider"] == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY","").strip()
        if key:
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization":f"Bearer {key}","Content-Type":"application/json",
                                 "HTTP-Referer":"https://www.searchrnk.com","X-Title":"SearchRNK Chat"},
                        json={"model":model_cfg["id"],"messages":msgs,"temperature":0.5,"max_tokens":700},
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status == 200:
                            data   = await resp.json()
                            record_usage("openrouter", data.get("usage",{}).get("total_tokens",300))
                            return data["choices"][0]["message"]["content"].strip()
            except Exception:
                pass
    # Fallback LLaMA
    groq_key = os.environ.get("GROQ_API_KEY","").strip()
    if groq_key:
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization":f"Bearer {groq_key}","Content-Type":"application/json"},
                    json={"model":"llama-3.1-8b-instant","messages":msgs,"temperature":0.5,"max_tokens":700},
                    timeout=aiohttp.ClientTimeout(total=25)
                ) as resp:
                    if resp.status == 200:
                        data   = await resp.json()
                        record_usage("groq", data.get("usage",{}).get("total_tokens",300))
                        return data["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
    return "I'm having trouble connecting to the AI service right now. Please try again in a moment."
# ═══════════════════════════════════════════
# REQUEST MODELS
# ═══════════════════════════════════════════
class CompareRequest(BaseModel):
    keyword:          str
    user_url:         str
    competitor_urls:  list
    manual_data:      dict = {}
class ChatRequest(BaseModel):
    message:  str
    history:  list = []
    context:  str  = ""
    keyword:  str  = ""
class RegenRequest(BaseModel):
    comparison: dict
    keyword:    str = ""
# ═══════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════
@app.post("/serp-compare")
async def serp_compare(request: CompareRequest):
    keyword = request.keyword.strip()
    if not keyword:
        return {"error":"keyword is required"}
    def clean(u):
        u=u.strip()
        if not u.startswith("http"): u="https://"+u
        return normalize_url(u)
    user_url   = clean(request.user_url)
    comp_urls  = [clean(u) for u in request.competitor_urls if u.strip()]
    all_urls   = [user_url] + comp_urls
    manual     = request.manual_data or {}
    async def generator():
        all_results = []
        conn = aiohttp.TCPConnector(limit=20, ssl=False)
        async with aiohttp.ClientSession(headers=HEADERS, connector=conn) as session:
            async def process(idx, url):
                data = await extract_page(session, url, keyword, manual.get(url,{}))
                data["is_user"] = (idx==0)
                data["rank"]    = "Your Page" if idx==0 else f"#{idx}"
                data["_idx"]    = idx
                return data
            tasks = [asyncio.create_task(process(i,u)) for i,u in enumerate(all_urls)]
            for coro in asyncio.as_completed(tasks):
                result = await coro
                all_results.append(result)
                yield json.dumps({"type":"page_result","data":result}) + "\n"
            all_results.sort(key=lambda r:r.get("_idx",99))
            comparison = compute_comparison(all_results, user_url, keyword)
            ai_result  = await call_ai(comparison)
            comparison["ai"] = ai_result
            yield json.dumps({"type":"comparison","data":comparison}) + "\n"
            yield json.dumps({"type":"done"}) + "\n"
    return StreamingResponse(generator(), media_type="application/x-ndjson")
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    reply = await call_ai_chat(
        message  = request.message,
        history  = request.history,
        context  = request.context,
        keyword  = request.keyword,
    )
    return {"reply": reply}
@app.post("/ai-regenerate")
async def ai_regenerate(request: RegenRequest):
    ai_result = await call_ai(request.comparison)
    return {"ai": ai_result}
@app.get("/health")
async def health():
    _reset_if_new_day()
    return {
        "status":       "ok",
        "service":      "SearchRNK Intelligence API v3",
        "active_model": get_active_model()["label"],
        "gpt_requests_used":      _ai_state["gpt_requests_used"],
        "gpt_requests_remaining": _ai_state["gpt_req_limit"] - _ai_state["gpt_requests_used"],
        "gpt_tokens_used":        _ai_state["gpt_tokens_used"],
        "llama_requests_used":    _ai_state["llama_requests_used"],
        "date": _ai_state["date"],
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
