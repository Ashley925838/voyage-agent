# Voyage Agent

An intelligent 2-day trip-planning agent built for the Hipster Tech Intern assessment (Task 2).

## Stack

- **LLM**: Groq (Llama 3.3 70B) via OpenAI-compatible API
- **Web framework**: FastAPI
- **Agent**: hand-written tool-calling loop (no LangChain dependency)
- **Memory**: in-process conversation buffer (long-term FAISS memory coming Day 2)
- **Tools** (mocked on Day 1, real APIs Day 2): weather, web search, attractions

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your Groq API key to .env
```

## Run locally

```bash
uvicorn app.main:app --reload --port 8000
```

Swagger UI: http://127.0.0.1:8000/docs

## Example call

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Plan 2 days in Tokyo. I love ramen and photography."}'
```

## Architecture

See `ARCHITECTURE.md` (coming Day 3).
