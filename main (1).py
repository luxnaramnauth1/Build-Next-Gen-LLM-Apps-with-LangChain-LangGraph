"""
HITL feedback lab backend (FastAPI)
  POST /api/chat            -> streams an assistant reply in the Vercel AI SDK UI-message-stream format
  POST /api/feedback        -> stores/updates feedback for one assistant message, returns fresh stats
  GET  /api/feedback/stats  -> aggregate counts (overall + per session)
Feedback lives in a simple in-memory dict (lost on restart) - fine for the lab.
Without OPENAI_API_KEY the chat route uses a deterministic mock model.
"""
import asyncio, json, os, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="HITL Feedback Lab")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------- in-memory stores
RESPONSES: dict[str, set] = {}          # sessionId -> assistant message ids generated
FEEDBACK: dict[tuple, dict] = {}        # (sessionId, messageId) -> latest feedback record

# ---------------------------------------------------------- mock model
def mock_answer(q: str) -> str:
    q = q.lower()
    if "clarif" in q:
        return ("Sure, let me put it more simply: plan changes are applied at the start "
                "of your next billing cycle, so you will not be charged twice in the same month.")
    if "refund" in q:
        return ("I believe refunds are available within 30 days of purchase, but I'm not fully "
                "certain this applies to annual plans. Please confirm with support before relying on it.")
    if "invoice" in q or "billing" in q:
        return ("Invoices are generated on the 1st of each month and emailed to the account owner. "
                "You can also download past invoices from Settings > Billing.")
    if "cancel" in q:
        return ("You can cancel anytime from Settings > Subscription. Your access continues "
                "until the end of the current billing period.")
    if "plan" in q or "subscription" in q or "upgrade" in q:
        return ("Yes, you can change your subscription plan next month. The new plan starts at the "
                "beginning of your next billing cycle, and any price difference is shown before you confirm.")
    return ("I'm not sure I understood that. Could you tell me whether your question is about "
            "plans, billing, invoices, or cancellations?")

async def openai_stream(q: str):
    from openai import AsyncOpenAI           # optional path, used only if OPENAI_API_KEY is set
    client = AsyncOpenAI()
    stream = await client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"), stream=True,
        messages=[{"role": "system", "content": "You are a concise support assistant."},
                  {"role": "user", "content": q}])
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

async def mock_stream(q: str):
    for word in mock_answer(q).split(" "):
        yield word + " "
        await asyncio.sleep(0.035)           # simulate token streaming

def sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"

@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    session_id = body.get("sessionId", "anon")
    last_user = next((m for m in reversed(body.get("messages", [])) if m.get("role") == "user"), {})
    text = "".join(p.get("text", "") for p in last_user.get("parts", []) if p.get("type") == "text")
    msg_id, text_id = "msg_" + uuid.uuid4().hex[:10], "txt_" + uuid.uuid4().hex[:6]
    RESPONSES.setdefault(session_id, set()).add(msg_id)   # counts toward "Total responses"
    tokens = openai_stream(text) if os.getenv("OPENAI_API_KEY") else mock_stream(text)

    async def gen():
        yield sse({"type": "start", "messageId": msg_id})
        yield sse({"type": "text-start", "id": text_id})
        async for t in tokens:
            yield sse({"type": "text-delta", "id": text_id, "delta": t})
        yield sse({"type": "text-end", "id": text_id})
        yield sse({"type": "finish"})
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
        headers={"x-vercel-ai-ui-message-stream": "v1", "Cache-Control": "no-cache", "x-accel-buffering": "no"})

# ---------------------------------------------------------- feedback
class Feedback(BaseModel):
    messageId: str
    rating: Optional[Literal["up", "down"]] = None
    flagged: bool = False
    comment: Optional[str] = Field(default=None, max_length=1000)
    timestamp: str
    sessionId: str

def stats_for(session_id: Optional[str] = None) -> dict:
    recs = [r for (s, _), r in FEEDBACK.items() if session_id in (None, s)]
    total = sum(len(v) for k, v in RESPONSES.items() if session_id in (None, k))
    rated = sum(1 for r in recs if r["rating"])
    touched = sum(1 for r in recs if r["rating"] or r["flagged"] or r["comment"])
    return {"total_responses": total,
            "thumbs_up": sum(1 for r in recs if r["rating"] == "up"),
            "thumbs_down": sum(1 for r in recs if r["rating"] == "down"),
            "flagged": sum(1 for r in recs if r["flagged"]),
            "comments": sum(1 for r in recs if r["comment"]),
            "feedback_rate": round(touched / total, 2) if total else 0.0,
            "satisfaction": round(sum(1 for r in recs if r["rating"] == "up") / rated, 2) if rated else None}

@app.post("/api/feedback")
def post_feedback(fb: Feedback):
    # keyed by (session, message): changing your mind updates the record instead of double counting
    FEEDBACK[(fb.sessionId, fb.messageId)] = fb.model_dump()
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] feedback session={fb.sessionId[:8]} msg={fb.messageId} "
          f"rating={fb.rating} flagged={fb.flagged} comment={fb.comment!r}", flush=True)
    return {"ok": True, "stats": stats_for(fb.sessionId)}

@app.get("/api/feedback/stats")
def get_stats(sessionId: Optional[str] = None):
    return {"session": stats_for(sessionId) if sessionId else None, "overall": stats_for()}

# serve the built React app (production-style single server) if it exists
DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
