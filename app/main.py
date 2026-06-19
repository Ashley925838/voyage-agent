"""FastAPI application exposing the travel agent over HTTP."""
import logging
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from app.agent.loop import AgentLoop
from app.llm.groq_provider import GroqProvider
from app.tools.registry import build_registry

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voyage Agent",
    description=(
        "Intelligent 2-day trip-planning AI agent. "
        "Built for the Hipster Tech Intern assessment (Task 2)."
    ),
    version="0.1.0",
)

# In-memory session store. Production would back this with Redis or similar.
_sessions: dict[str, AgentLoop] = {}


def _get_or_create_session(session_id: str) -> AgentLoop:
    if session_id not in _sessions:
        logger.info("Creating new session %s", session_id)
        _sessions[session_id] = AgentLoop(
            provider=GroqProvider(),
            tools=build_registry(),
            user_id=session_id,
        )
    return _sessions[session_id]


# ---------- schemas ----------


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(
        default=None,
        description="Reuse the same session_id across turns to keep context.",
    )


class ChatResponse(BaseModel):
    session_id: str
    reply: str


# ---------- routes ----------


@app.get("/", include_in_schema=False)
def root():
    """Redirect bare visits to the interactive Swagger UI."""
    return RedirectResponse(url="/docs")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or str(uuid4())
    try:
        agent = _get_or_create_session(session_id)
        reply = agent.chat(req.message)
    except Exception as e:  # noqa: BLE001
        logger.exception("Agent failure for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Agent error: {e}") from e

    return ChatResponse(session_id=session_id, reply=reply)
