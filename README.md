# Sasuke API Wrapper

OpenAI-compatible FastAPI proxy for `https://100milionseek.top/v1` with streaming responses.

## Endpoints

- `GET /health`
- `POST /v1/chat/completions`
- Interactive docs: `/docs`

## Client keys

- `sasukepremium`: unlimited requests
- `sasuke`: 100 requests per UTC day per client IP, then `429 Daily limit crossed`

Every response includes the watermark `developer: sasuke` in `X-Sasuke-Watermark` and `X-Developer` headers. Streaming responses also emit it as an SSE comment before the first chunk, so normal OpenAI clients continue to work unchanged.

The standard-tier counter is in memory and resets if the process restarts. For multi-instance production use, replace it with Redis or a database.

## Run locally

Set the upstream secret in your shell, then start FastAPI:

```bash
export HMS_API_KEY='your-upstream-key'
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Test streaming:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sasuke' \
  -d '{"model":"deepseek-v4.1-flash-free","messages":[{"role":"user","content":"Explain SQLite WAL mode in 3 lines"}],"stream":true}'
```

## OpenAI Python client usage

```python
from openai import OpenAI

client = OpenAI(
    api_key="sasuke",
    base_url="https://YOUR_FADEHOST_ADDRESS/v1",
)

stream = client.chat.completions.create(
    model="deepseek-v4.1-flash-free",
    messages=[{"role": "user", "content": "Explain SQLite WAL mode in 3 lines"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="")
```

## FadeHost deployment

1. Create a FadeHost app from this GitHub repository.
2. Set the start command to:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

3. Add `HMS_API_KEY` in FadeHost's private environment-variable settings. Do not commit the upstream key to GitHub.
4. Assign the app a public address. The API will be available at `https://YOUR_FADEHOST_ADDRESS/v1/chat/completions`.

You can optionally override `HMS_MODEL`, `HMS_BASE_URL`, `SASUKE_PREMIUM_KEY`, `SASUKE_STANDARD_KEY`, and `SASUKE_DAILY_LIMIT` in the hosting panel.
