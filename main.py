"""
LLM Reliability Middleware (FastAPI)
------------------------------------
POST /api/chat  ->  wraps an OpenAI chat-completion call with:
  * request timeout
  * retries with exponential backoff (429 / 5xx / timeouts / connection errors only)
  * in-memory cache with TTL
  * token + retry logging
  * a stable, frontend-friendly JSON response (never leaks raw provider errors)

Run:   OPENAI_API_KEY=sk-... uvicorn main:app --reload
Test:  no key set -> MOCK mode (simulated provider with injectable failures)
"""
import hashlib
import logging
import os
import random
import threading
import time
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------- config
API_KEY = os.getenv("OPENAI_API_KEY")          # never hard-coded
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # fixed model ...
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))  # ... and temperature
TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "20"))       # per-attempt timeout
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))      # retries AFTER first try
BASE_DELAY_S = float(os.getenv("LLM_BASE_DELAY_S", "0.5"))  # 0.5s, 1s, 2s
CACHE_TTL_S = int(os.getenv("CACHE_TTL_S", "300"))        # 5 minutes
MOCK = API_KEY is None or os.getenv("MOCK_LLM") == "1"

SYSTEM_PROMPT = ("You are a helpful support assistant. Answer briefly and "
                 "accurately. If you are unsure, say so.")

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("llm-middleware")

# ---------------------------------------------------------------- errors
class ProviderError(Exception):
    """Normalized provider failure so retry logic is provider-agnostic."""
    def __init__(self, status: Optional[int], kind: str, msg: str = ""):
        super().__init__(msg or kind)
        self.status, self.kind = status, kind

def is_retryable(err: ProviderError) -> bool:
    # Transient: rate limit, server errors, timeouts, network problems.
    # NOT retryable: 400/401/403/404/422 -> retrying cannot succeed and wastes quota.
    if err.kind in ("timeout", "connection"):
        return True
    return err.status == 429 or (err.status is not None and err.status >= 500)

# ---------------------------------------------------------------- providers
_mock_attempts = {}  # message -> attempts seen (to simulate "fail N times then succeed")

def _mock_provider(system: str, user: str):
    """Simulated provider. Trigger failures with tags in the message."""
    n = _mock_attempts.get(user, 0) + 1
    _mock_attempts[user] = n
    time.sleep(0.4)  # pretend network/model latency
    if "[sim-400]" in user:
        raise ProviderError(400, "bad_request", "invalid input")
    if "[sim-429]" in user and n <= 2:
        raise ProviderError(429, "rate_limit", "Rate limit reached")
    if "[sim-timeout]" in user and n <= 1:
        raise ProviderError(None, "timeout", "request timed out")
    if "[sim-down]" in user:
        raise ProviderError(503, "server_error", "service unavailable")
    _mock_attempts.pop(user, None)
    ptok = max(1, (len(system) + len(user)) // 4)
    answer = ("Yes. You can change your subscription plan at any time; the change "
              "takes effect at the start of your next billing cycle.")
    return answer, ptok, len(answer) // 4

def _openai_provider(system: str, user: str):
    import openai
    client = openai.OpenAI(api_key=API_KEY, timeout=TIMEOUT_S, max_retries=0)  # we own retries
    try:
        r = client.chat.completions.create(
            model=MODEL, temperature=TEMPERATURE,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
    except openai.APITimeoutError as e:
        raise ProviderError(None, "timeout", str(e))
    except openai.APIConnectionError as e:
        raise ProviderError(None, "connection", str(e))
    except openai.APIStatusError as e:
        raise ProviderError(e.status_code, "http_error", str(e))
    return r.choices[0].message.content, r.usage.prompt_tokens, r.usage.completion_tokens

call_provider = _mock_provider if MOCK else _openai_provider

# ---------------------------------------------------------------- cache (TTL)
_cache, _cache_lock = {}, threading.Lock()
stats = {"requests": 0, "cache_hits": 0, "retries": 0, "failures": 0,
         "prompt_tokens": 0, "completion_tokens": 0}

def cache_key(system: str, user: str) -> str:
    # normalise whitespace/case so trivially different repeats still hit the cache
    norm = " ".join(user.lower().split())
    return hashlib.sha256(f"{MODEL}|{system}|{norm}".encode()).hexdigest()

def cache_get(key):
    with _cache_lock:
        item = _cache.get(key)
        if item and item["exp"] > time.time():
            return item["val"]
        _cache.pop(key, None)  # expired -> evict (avoids stale answers)
        return None

def cache_set(key, val):
    with _cache_lock:
        _cache[key] = {"val": val, "exp": time.time() + CACHE_TTL_S}

# ---------------------------------------------------------------- core wrapper
def call_llm_with_retry(system: str, user: str):
    """Returns (answer, meta). Raises ProviderError only after retries are exhausted
    or on a non-retryable error."""
    retries = 0
    while True:
        try:
            answer, ptok, ctok = call_provider(system, user)
            return answer, {"retried": retries > 0, "retry_count": retries,
                            "prompt_tokens": ptok, "completion_tokens": ctok}
        except ProviderError as err:
            if not is_retryable(err) or retries >= MAX_RETRIES:
                err.retries = retries
                raise
            delay = BASE_DELAY_S * (2 ** retries) + random.uniform(0, 0.1)  # 0.5, 1, 2 (+jitter)
            retries += 1
            stats["retries"] += 1
            log.warning("/api/chat - transient error (%s status=%s); retry %d/%d in %.2fs",
                        err.kind, err.status, retries, MAX_RETRIES, delay)
            time.sleep(delay)

# ---------------------------------------------------------------- API
app = FastAPI(title="LLM Reliability Middleware")

class ChatRequest(BaseModel):
    user_message: str

def envelope(answer, meta, error=None):
    """One response shape for success AND failure -> frontend never branches on format."""
    return {"answer": answer, "error": error, "meta": meta}

@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content=envelope(
        None, {"retried": False, "retry_count": 0, "cached": False},
        {"code": "INVALID_REQUEST", "message": "Please send JSON like {\"user_message\": \"...\"}."}))

@app.post("/api/chat")
def chat(req: ChatRequest):
    t0 = time.time()
    stats["requests"] += 1
    msg = req.user_message.strip()
    if not msg:
        return JSONResponse(status_code=400, content=envelope(
            None, {"retried": False, "retry_count": 0, "cached": False},
            {"code": "EMPTY_MESSAGE", "message": "Please type a message."}))

    key = cache_key(SYSTEM_PROMPT, msg)
    hit = cache_get(key)
    if hit:
        stats["cache_hits"] += 1
        ms = int((time.time() - t0) * 1000)
        log.info("/api/chat - CACHE HIT retries=0 tokens=0 (saved %d) latency=%dms",
                 hit["total"], ms)
        return envelope(hit["answer"], {"retried": False, "retry_count": 0, "cached": True,
                        "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": ms})
    try:
        answer, meta = call_llm_with_retry(SYSTEM_PROMPT, msg)
    except ProviderError as err:
        stats["failures"] += 1
        ms = int((time.time() - t0) * 1000)
        retries = getattr(err, "retries", 0)
        # Raw provider detail goes to the LOG only, never to the client.
        log.error("/api/chat - FAILED kind=%s status=%s retries=%d latency=%dms detail=%s",
                  err.kind, err.status, retries, ms, err)
        retryable = is_retryable(err)
        return JSONResponse(status_code=503 if retryable else 400, content=envelope(
            None, {"retried": retries > 0, "retry_count": retries, "cached": False, "latency_ms": ms},
            {"code": "LLM_UNAVAILABLE" if retryable else "REQUEST_REJECTED",
             "message": ("Our assistant is busy right now. Please try again in a moment."
                         if retryable else "We couldn't process that message. Please rephrase it.")}))

    total = meta["prompt_tokens"] + meta["completion_tokens"]
    stats["prompt_tokens"] += meta["prompt_tokens"]
    stats["completion_tokens"] += meta["completion_tokens"]
    cache_set(key, {"answer": answer, "total": total})
    ms = int((time.time() - t0) * 1000)
    meta.update({"cached": False, "latency_ms": ms})
    log.info("/api/chat - retries=%d tokens=%d (prompt=%d, completion=%d) latency=%dms",
             meta["retry_count"], total, meta["prompt_tokens"], meta["completion_tokens"], ms)
    return envelope(answer, meta)

@app.get("/api/metrics")
def metrics():
    """Simple counters for cost/reliability visibility."""
    s = dict(stats)
    s["cache_hit_rate"] = round(s["cache_hits"] / s["requests"], 2) if s["requests"] else 0
    s["mode"] = "mock" if MOCK else "openai"
    return s
