import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI

app = FastAPI(
    title="AI Sasuke API",
    description="OpenAI-compatible AI Sasuke API with streaming and client-key limits.",
    version="1.0.0",
)

UPSTREAM_BASE_URL = os.getenv("HMS_BASE_URL", "https://100milionseek.top/v1")
UPSTREAM_MODEL = os.getenv("HMS_MODEL", "deepseek-v4.1-flash-free")

# These are client access keys, not the upstream provider secret.
# Override them in the hosting panel if you ever need to rotate them.
PREMIUM_KEY = os.getenv("SASUKE_PREMIUM_KEY", "sasukepremium")
STANDARD_KEY = os.getenv("SASUKE_STANDARD_KEY", "sasuke")
STANDARD_DAILY_LIMIT = int(os.getenv("SASUKE_DAILY_LIMIT", "100"))

_usage: dict[tuple[str, str, str], int] = {}
_usage_lock = threading.Lock()
WATERMARK = "developer: sasuke"
SASUKE_SYSTEM_PROMPT = (
    "You are AI Sasuke. Do not mention the upstream model or provider. "
    "You were made by Sasuke. His Instagram ID is @frsaske. "
    "He studies in SSV, 11th class."
)
_stats = {"total_requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def add_system_prompt(body: dict[str, Any]) -> dict[str, Any]:
    updated = dict(body)
    messages = updated.get("messages") or []
    updated["messages"] = [
        {"role": "system", "content": SASUKE_SYSTEM_PROMPT},
        *[message for message in messages if message.get("role") != "system"],
    ]
    return updated


def record_usage(usage: Any) -> None:
    if usage is None:
        return
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump(exclude_none=True)
    elif hasattr(usage, "dict"):
        usage = usage.dict(exclude_none=True)
    with _usage_lock:
        _stats["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        _stats["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        _stats["total_tokens"] += int(usage.get("total_tokens", 0) or 0)


def record_request() -> None:
    with _usage_lock:
        _stats["total_requests"] += 1


def get_client() -> OpenAI:
    upstream_key = os.getenv("HMS_API_KEY")
    if not upstream_key:
        raise HTTPException(status_code=503, detail="Upstream API is not configured")
    return OpenAI(api_key=upstream_key, base_url=UPSTREAM_BASE_URL)


def authenticate(api_key_header: str | None) -> str:
    if not api_key_header or not api_key_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Use Authorization: Bearer <client-key>")

    client_key = api_key_header.split(" ", 1)[1].strip()
    if client_key == PREMIUM_KEY:
        return "premium"
    if client_key == STANDARD_KEY:
        return "standard"
    raise HTTPException(status_code=401, detail="Invalid client API key")


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


def consume_quota(tier: str, ip: str) -> int | None:
    if tier == "premium":
        return None

    day = datetime.now(timezone.utc).date().isoformat()
    usage_key = (day, tier, ip)
    with _usage_lock:
        used = _usage.get(usage_key, 0)
        if used >= STANDARD_DAILY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail=f"Daily limit crossed ({STANDARD_DAILY_LIMIT} requests). Try again tomorrow or use the premium key.",
            )
        _usage[usage_key] = used + 1
        return STANDARD_DAILY_LIMIT - used - 1


def sse_stream(request_body: dict[str, Any]):
    try:
        yield f": {WATERMARK}\n\n"
        upstream_body = dict(request_body)
        upstream_body["model"] = request_body.get("model") or UPSTREAM_MODEL
        upstream_body["stream"] = True
        upstream_body.setdefault("stream_options", {"include_usage": True})
        stream = get_client().chat.completions.create(
            **upstream_body,
        )
        for chunk in stream:
            record_usage(getattr(chunk, "usage", None))
            # Pydantic v1/v2 compatible serialization for OpenAI SDK chunks.
            if hasattr(chunk, "model_dump"):
                payload = chunk.model_dump(exclude_none=True)
            else:
                payload = chunk.dict(exclude_none=True)
            if getattr(chunk, "choices", None) or getattr(chunk, "usage", None):
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        error = {"error": {"message": str(exc), "type": "upstream_error"}}
        yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


@app.get("/health")
def health():
    return JSONResponse(
        {"status": "ok", "service": "ai-sasuke"},
        headers={"X-Sasuke-Watermark": WATERMARK, "X-Developer": "sasuke"},
    )


@app.get("/")
def dashboard():
    with _usage_lock:
        payload = {
            "service": "AI Sasuke",
            "developer": "sasuke",
            "watermark": WATERMARK,
            "total_requests": _stats["total_requests"],
            "prompt_tokens": _stats["prompt_tokens"],
            "completion_tokens": _stats["completion_tokens"],
            "total_tokens": _stats["total_tokens"],
            "free_key_limit": f"{STANDARD_DAILY_LIMIT} requests per UTC day per IP",
        }
    return JSONResponse(payload)


@app.post("/v1/chat/completions")
def chat_completions(
    body: dict[str, Any],
    request: Request,
    authorization: str | None = Header(default=None),
):
    tier = authenticate(authorization)
    remaining = consume_quota(tier, client_ip(request))
    record_request()
    response_headers = {
        "X-Sasuke-Watermark": WATERMARK,
        "X-Developer": "sasuke",
    }
    if remaining is not None:
        response_headers.update({
            "X-RateLimit-Limit": str(STANDARD_DAILY_LIMIT),
            "X-RateLimit-Remaining": str(remaining),
        })

    if body.get("stream") is False:
        try:
            upstream_body = add_system_prompt(body)
            upstream_body["model"] = body.get("model") or UPSTREAM_MODEL
            upstream_body["stream"] = False
            response = get_client().chat.completions.create(
                **upstream_body,
            )
            record_usage(getattr(response, "usage", None))
            if hasattr(response, "model_dump"):
                payload = response.model_dump(exclude_none=True)
                return JSONResponse(payload, headers=response_headers)
            payload = response.dict(exclude_none=True)
            return JSONResponse(payload, headers=response_headers)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return StreamingResponse(
        sse_stream(add_system_prompt(body)),
        media_type="text/event-stream",
        headers={
            **response_headers,
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
    )
