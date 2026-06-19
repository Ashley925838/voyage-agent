---
title: Voyage Agent
emoji: 🗺️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Voyage Agent

An intelligent 2-day trip-planning AI agent. Plans personalised itineraries by combining live weather, web search, and Wikipedia-grounded retrieval, with short- and long-term memory that adapts to each user across sessions.

Built for the **Hipster Pte. Ltd. Tech Intern assessment (Task 2)**.

## Live demo

- **API**: <https://ashley925838-voyage-agent.hf.space>
- **Interactive Swagger UI**: <https://ashley925838-voyage-agent.hf.space/docs>
- **Source code**: <https://github.com/Ashley925838/voyage-agent>

> The free Hugging Face Space sleeps after inactivity. The first request after a sleep takes ~30 seconds to wake the container; subsequent requests are fast.

## What it does

Given a request like *"Plan 2 days in Tokyo. I love ramen and street photography. Should I pack an umbrella?"*, the agent:

1. Produces a numbered plan listing the sub-tasks and which tool serves each.
2. Calls the relevant tools — weather, Wikipedia RAG, web search — possibly several times in parallel with different queries.
3. Synthesises a 2-day morning / afternoon / evening itinerary grounded in the retrieved facts.
4. Quietly extracts stable preferences (e.g. "vegetarian", "tight budget", "slow travel") and remembers them for next time, without storing one-off requests.

## Try it

```bash
curl -X POST https://ashley925838-voyage-agent.hf.space/chat \\
  -H "Content-Type: application/json" \\
  -d '{
    "message": "Plan 2 days in Singapore. I love architecture and good coffee."
  }'
```

Or visit the Swagger UI link above and click **Try it out** on `POST /chat`.

To keep context across turns, pass back the `session_id` returned by the previous response:

```bash
curl -X POST https://ashley925838-voyage-agent.hf.space/chat \\
  -H "Content-Type: application/json" \\
  -d '{
    "message": "Actually budget around 200 SGD per day. Adjust.",
    "session_id": "<paste-id-from-previous-response>"
  }'
```

## How it works

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for design decisions and trade-offs.

Quick summary:

| Layer | Choice |
|---|---|
| LLM | Groq Cloud · Llama 3.3 70B (OpenAI-compatible API) |
| Web framework | FastAPI |
| Agent | Hand-written tool-calling loop, no LangChain |
| Tools | OpenWeatherMap, Tavily, Wikipedia RAG |
| Vector store | FAISS, per-city + per-user indices |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Container | Single Dockerfile, CPU-only torch wheel |
| Hosting | Hugging Face Spaces, free CPU tier |

## Run locally

```bash
git clone https://github.com/Ashley925838/voyage-agent
cd voyage-agent

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in your API keys: GROQ_API_KEY, OPENWEATHER_API_KEY, TAVILY_API_KEY

uvicorn app.main:app --reload --port 8000
```

Then open <http://127.0.0.1:8000/docs>.

## Project layout

```text
app/
├── main.py                      # FastAPI app and routes
├── config.py                    # Settings (env-var driven)
├── llm/
│   ├── base.py                  # LLMProvider abstract base (Strategy pattern)
│   └── groq_provider.py         # Groq implementation
├── agent/
│   ├── loop.py                  # Hand-written agent reasoning loop
│   ├── prompts.py               # System prompts (plan-and-execute)
│   └── preference_extractor.py  # Long-term memory extraction
├── tools/
│   ├── base.py                  # Tool ABC + OpenAI schema export
│   ├── registry.py              # Tool registration
│   ├── weather.py               # OpenWeatherMap
│   ├── web_search.py            # Tavily
│   ├── attractions.py           # Wikipedia + FAISS RAG
│   └── _dates.py                # Defensive date parsing
├── memory/
│   ├── short_term.py            # In-session conversation buffer
│   └── long_term.py             # Per-user preference FAISS index
└── rag/
    ├── wiki_fetcher.py          # Wikipedia API client
    ├── chunker.py               # Sentence-aware chunking
    ├── embedder.py              # Sentence-transformers wrapper
    └── store.py                 # Per-city FAISS persistence
tests/
└── test_smoke.py                # Pytest smoke tests
```

## Tests

```bash
pytest tests/ -v
```

## License

MIT.
