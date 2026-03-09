"""
AstraOS — FastAPI Backend
Handles data persistence + Grok AI proxy
"""
import os, json
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
import httpx
from database import Database

app = FastAPI(title="AstraOS API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

db = Database()

GROK_API_KEY = os.environ.get("GROK_API_KEY", "")
GROK_BASE_URL = "https://api.x.ai/v1"

VALID_USERS = {"naveen","sri","ramesh","raja"}

def get_user(x_user: Optional[str] = Header(None)) -> str:
    u = (x_user or "").lower().strip()
    if u not in VALID_USERS:
        raise HTTPException(status_code=401, detail="Invalid user")
    return u

# ─── HEALTH ───────────────────────────────────────────────
@app.get("/api/health")
def health(): return {"status": "ok", "version": "2.0"}

# ─── CRUD ENDPOINTS ───────────────────────────────────────
COLLECTIONS = ["expenses","investments","assets","recurring","debts","savings"]

class RecordBody(BaseModel):
    data: dict

@app.get("/api/{collection}")
def get_records(collection: str, x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    if collection not in COLLECTIONS: raise HTTPException(404)
    items = db.get_records(user, collection)
    return {"items": items, "count": len(items)}

@app.post("/api/{collection}")
def create_record(collection: str, body: RecordBody, x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    if collection not in COLLECTIONS: raise HTTPException(404)
    record = db.add_record(user, collection, body.data)
    return {"item": record, "success": True}

@app.put("/api/{collection}/{record_id}")
def update_record(collection: str, record_id: str, body: RecordBody, x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    if collection not in COLLECTIONS: raise HTTPException(404)
    record = db.update_record(user, collection, record_id, body.data)
    if not record: raise HTTPException(404)
    return {"item": record, "success": True}

@app.delete("/api/{collection}/{record_id}")
def delete_record(collection: str, record_id: str, x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    if collection not in COLLECTIONS: raise HTTPException(404)
    db.delete_record(user, collection, record_id)
    return {"success": True}

# ─── GROK AI PROXY ────────────────────────────────────────
class ChatBody(BaseModel):
    messages: list
    model: Optional[str] = "grok-3-latest"
    system: Optional[str] = None

@app.post("/api/chat")
async def chat(body: ChatBody, x_user: Optional[str] = Header(None)):
    user = get_user(x_user)
    if not GROK_API_KEY:
        raise HTTPException(status_code=503, detail="GROK_API_KEY not configured")

    msgs = body.messages
    if body.system:
        msgs = [{"role":"system","content":body.system}] + msgs

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{GROK_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"},
            json={"model": body.model, "messages": msgs, "max_tokens": 2000, "temperature": 0.7}
        )
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return {"content": content, "model": body.model}
