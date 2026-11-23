import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI

app = FastAPI(
    title="Sasuke API Wrapper",
    description="OpenAI-compatible streaming proxy with simple client-key limits.",
    version="1.0.0",
)

UPSTREAM_BASE_URL = os.getenv("HMS_BASE_URL", "https://100milionseek.top/v1")
UPSTREAM_MODEL = os.getenv("HMS_MODEL", "deepseek-v4.1-flash-free")

# These are client access keys, not the upstream provider secret.
# Override them in the hosting panel if you ever need to rotate them.
PREMIUM_KEY = os.getenv("SASUKE_PREMIUM_KEY", "sasukepremium")
STANDARD_KEY = os.getenv("SASUKE_STANDARD_KEY", "sasuke")
STANDARD_DAILY_LIMIT = int(os.getenv("SASUKE_DAILY_LIMIT", "100"))

_usage: dict[tuple[str, str], int] = {}
_usage_lock = threading.Lock()


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


def consume_quota(tier: str) -> None:
    if tier == "premium":
        return

    day = datetime.now(timezone.utc).date().isoformat()
    usage_key = (day, tier)
    with _usage_lock:
        used = _usage.get(usage_key, 0)
        if used >= STANDARD_DAILY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail=f"Daily limit crossed ({STANDARD_DAILY_LIMIT} requests). Try again tomorrow or use the premium key.",
            )
        _usage[usage_key] = used + 1


def sse_stream(request_body: dict[str, Any]):
    try:
        stream = get_client().chat.completions.create(
            **request_body,
            model=request_body.get("model") or UPSTREAM_MODEL,
            stream=True,
        )
        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            # Pydantic v1/v2 compatible serialization for OpenAI SDK chunks.
            if hasattr(chunk, "model_dump"):
                payload = chunk.model_dump(exclude_none=True)
            else:
                payload = chunk.dict(exclude_none=True)
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        error = {"error": {"message": str(exc), "type": "upstream_error"}}
        yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


@app.get("/health")
def health():
    return {"status": "ok", "service": "sasuke-api-wrapper"}


@app.post("/v1/chat/completions")
def chat_completions(
    body: dict[str, Any],
    authorization: str | None = Header(default=None),
):
    tier = authenticate(authorization)
    consume_quota(tier)

    if body.get("stream") is False:
        try:
            response = get_client().chat.completions.create(
                **body,
                model=body.get("model") or UPSTREAM_MODEL,
                stream=False,
            )
            if hasattr(response, "model_dump"):
                return JSONResponse(response.model_dump(exclude_none=True))
            return JSONResponse(response.dict(exclude_none=True))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return StreamingResponse(
        sse_stream(body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
