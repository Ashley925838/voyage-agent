# Voyage Agent — Architecture

## Overview

Voyage Agent is a 2-day trip-planning AI agent that combines tool use, retrieval-augmented generation, and a tiered memory system to produce grounded, personalised itineraries. The reasoning loop is hand-written (no LangChain) for explicit control over each step; the LLM backend, tools, and memory layers are kept independently swappable.

## High-level Flow

```text
                   +--------------------------------+
   User message -->|            FastAPI             |
                   |     POST /chat {session_id}    |
                   +---------------+----------------+
                                   |
                                   v
   +--------------------------------------------------------+
   |                       AgentLoop                        |
   |                                                        |
   |  1. Retrieve relevant long-term preferences (FAISS)    |
   |  2. Append user message to ConversationBuffer          |
   |  3. Loop (max 6 iters):                                |
   |       LLM.chat(messages, tools)                        |
   |         |                                              |
   |         +-- no tool calls  --> return final answer     |
   |         +-- tool calls     --> execute, append, loop   |
   |  4. Extract and persist new stable preferences         |
   +-------------+------------------------------------------+
                 |
       +---------+---------+----------------+------------------+
       v         v                          v                  v
   +--------+ +--------+ +------------------+ +------------------+
   | get_   | | web_   | | search_          | |  Long-term       |
   | weather| | search | | attractions      | |  preference      |
   |        | |        | | (RAG)            | |  store           |
   | Open   | | Tavily | | Wikipedia +      | |  FAISS per user  |
   | Weather| |        | | FAISS per city   | |                  |
   | Map    | |        | | bge-small-en     | |                  |
   +--------+ +--------+ +------------------+ +------------------+
```

## LLM Provider

**Decision:** Groq Cloud API serving Llama 3.3 70B, accessed via the OpenAI-compatible endpoint.

**Why:** Groq's LPU inference is fast enough that multi-iteration tool-calling loops feel snappy in demos (a typical 3-tool plan returns in 4-6 seconds). The free tier is generous enough for the assessment without quota anxiety, and the OpenAI-compatible API means swapping providers is one line of code.

**Alternatives considered:**

- *OpenAI / Anthropic:* better reasoning at higher cost; not chosen because Llama 3.3 70B is sufficient for tool selection and the latency penalty hurts demo experience.
- *Gemini Flash free tier:* viable, kept as a documented fallback in the `LLMProvider` abstraction.

**Swap cost:** Implement `LLMProvider` once (`app/llm/base.py`); swap by changing one environment variable and one `__init__`. No agent logic touches provider details.

## Agent Loop

**Decision:** A hand-written tool-calling loop (`app/agent/loop.py`) capped at 6 iterations.

**Why not LangChain:** LangChain's `AgentExecutor` and Plan-and-Execute abstractions add indirection that hurts debugging on a small project. With a hand-written loop:

- Every state transition is one Python line, easy to step through.
- Tool result formatting, error handling, and the iteration cap are explicit.
- The plan-and-execute discipline is enforced by prompt engineering instead of a separate framework, keeping one source of truth.

**Why max 6 iterations:** Empirically, well-formed tool plans complete in 1-3 iterations; 6 is generous enough for genuine multi-step reasoning but short enough to bound cost on misbehaving runs.

**Graceful degradation:** Tool exceptions are caught and surfaced to the LLM as `"ERROR: ..."` strings, so the model can decide whether to retry, switch tools, or proceed without that data. This was observed working when the OpenWeather API briefly 401'd during development — the model continued with a caveat instead of crashing.

## Tools

| Tool | Backed by | When the LLM calls it |
|---|---|---|
| `get_weather` | OpenWeatherMap 5-day forecast + geocoding | Packing decisions, outdoor planning |
| `web_search` | Tavily | Current events, opening hours, prices |
| `search_attractions` | Wikipedia + FAISS RAG (per city) | Sightseeing, food, neighbourhoods, culture |

**Tool abstraction:** Every tool inherits from `app.tools.base.Tool` and exposes a JSON Schema via `to_openai_schema()`. Adding a tool is one new file plus one line in `app/tools/registry.py` — no changes to the agent loop.

**Date parsing:** During development the LLM passed loose date strings like `"next weekend"` directly to `get_weather`. Rather than rely on prompt engineering alone, the tool defensively parses these through `app/tools/_dates.py` before calling the upstream API. This is a deliberate defence-in-depth choice: prompts can drift; the tool boundary cannot.

## Retrieval-Augmented Generation

**Decision:** Per-city FAISS indices over Wikipedia content, built lazily on first request and persisted to disk.

**Pipeline:**

1. Fetch a small fixed set of pages per city (`{city}`, `Tourism in {city}`, `Culture of {city}`) — hand-picked to ensure travel-relevant grounding without freeform search.
2. Sentence-aware chunking (~600 chars, 80-char overlap) — avoids cutting mid-sentence, which hurts embedding quality.
3. Embed with `BAAI/bge-small-en-v1.5` (384 dims, normalised) — small enough for free-tier RAM, strong enough for retrieval.
4. Store in `faiss.IndexFlatIP` (exact cosine on normalised vectors) — at ~200 chunks per city, brute force is plenty fast and avoids the complexity of IVF/HNSW tuning.

**Multi-hop pattern:** The agent is encouraged via the system prompt to call `search_attractions` multiple times with different `interest` queries (e.g. once for "ramen", once for "photography spots") rather than one broad query. Each call is an independent retrieval — true multi-hop RAG. In practice the LLM does this naturally; one observed Singapore run made three retrievals (hawker centres / museums / photography) within a single user turn.

**Persistence:** Indices are written to `data/faiss/{city}.faiss` and reloaded on subsequent requests. First Tokyo query: ~30s build. Second Tokyo query: <1s.

## Memory

**Short-term** (`app/memory/short_term.py`): A `ConversationBuffer` per session holding the running list of OpenAI-format messages, capped at 30 entries to bound token cost on long sessions.

**Long-term** (`app/memory/long_term.py`): Per-user FAISS index of extracted preference strings. Two operations:

- *Inject:* On every turn, embed the current user message, retrieve the top 3 most relevant stored preferences (cosine > 0.3), and prepend them to the conversation as a system note.
- *Extract:* After each turn, a separate small LLM call (`app/agent/preference_extractor.py`) extracts only *stable* preferences (e.g. "vegetarian", "budget around 200 SGD per day") — never one-off requests like "plan Tokyo".

**Why separate the extractor:** Mixing it into the main loop would make every reply slower and harder to evolve. As its own component, the extractor can be batched, swapped for a cheaper model, or run asynchronously without touching agent code.

**Why "stable only":** Storing every utterance bloats memory and pollutes future sessions with irrelevant past requests. The conservative bias (extractor prompt explicitly says "better to miss than invent") trades recall for precision.

## Planning

**Decision:** Plan-and-Execute encoded via the system prompt, not a separate orchestrator.

The system prompt instructs the model to emit a numbered plan (one tool per step) before any tool call. In runtime this surfaces as a visible `PLAN:` block in the assistant's first message, which doubles as a debugging aid (the plan is logged alongside the actual tool calls so we can see when the model deviated from its own plan and why).

**Why prompt over framework:** A 50-line prompt instruction achieves what a LangChain `PlanAndExecuteAgent` would do, without the dependency footprint or the need to wire a separate planner LLM.

## Deployment

- **Container:** Single Dockerfile based on `python:3.11-slim`, ~1.2 GB image. Embedding model is pre-downloaded into the image during build so the first user request does not stall.
- **Hosting:** Hugging Face Spaces (free CPU tier, 16 GB RAM, 2 vCPU). Render was considered first but its 512 MB free tier is too tight for sentence-transformers + torch.
- **CI/CD:** Push to GitHub `main` -> push to HF Space remote `main` -> HF auto-builds and deploys.
- **Secrets:** Five environment variables (LLM and tool API keys) injected via HF Space's Secrets UI, never committed.

## Known Limitations

- **No request authentication:** `/chat` is open. For production this would need an API key gateway.
- **Wikipedia coverage bias:** Cities with sparse English-language Wikipedia content (e.g. smaller Chinese cities) get weaker retrieval. A production version would supplement with a curated source like TripAdvisor or Google Places.
- **Geocoding ambiguity:** OpenWeatherMap's geocoder will silently fall back to same-named cities (during development, "Atlantis" resolved to Atlantis, Florida). Production would require country-code disambiguation.
- **Cold start:** HF Spaces' free tier sleeps after inactivity; the first request after a sleep takes ~30 seconds to wake. Acceptable for assessment; a paid tier or warm-ping cron would fix this.
- **Stateless sessions:** The in-memory session dict is wiped on container restart. Production would persist sessions to Redis or a database. Long-term preferences *are* persisted to disk and survive restarts.
- **No streaming:** Responses are returned only when the full agent loop finishes. Streaming partial output would improve perceived latency.

## Future Work

- Add `book_hotel` / `find_flight` tools backed by real provider APIs to extend the agent beyond planning.
- Replace `IndexFlatIP` with `IndexHNSWFlat` once any single city's chunk count exceeds ~10K.
- Add evaluation harness: a fixed set of `(user_message, expected_tool_sequence)` pairs run on each commit to detect regressions in tool selection.
- Add LLM-as-judge scoring on a few golden itineraries for end-to-end quality tracking.
