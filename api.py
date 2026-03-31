"""
FastAPI streaming backend for the Protein Design Agent.

Run with:
    uvicorn api:app --reload --port 8001
"""
import asyncio
import json
import logging
import threading
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.agent import build_agent
from agent.cache import get_cache, get_query_cache

logger = logging.getLogger(__name__)

app = FastAPI(title="Protein Design Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent = None
_agent_lock = threading.Lock()
_query_cache_layer = None
# Track active streams so they can be cancelled by thread_id
_active_cancels: dict[str, threading.Event] = {}


def get_agent():
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = build_agent()
    return _agent


def _get_query_cache_layer():
    """Get or create the singleton QueryCacheLayer with embedding support."""
    global _query_cache_layer
    if _query_cache_layer is None:
        embed_fn = None
        try:
            from agent.rag.pdf_processing import get_embeddings

            def embed_fn(text: str) -> list[float]:
                result = get_embeddings([text])
                return result[0] if result else []
        except Exception:
            pass
        _query_cache_layer = get_query_cache(embed_fn)
    return _query_cache_layer


class ChatRequest(BaseModel):
    message: str
    thread_id: str


class CancelRequest(BaseModel):
    thread_id: str


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# ── Query-level cache helpers ─────────────────────────

def _try_query_cache(message: str) -> tuple[str, str] | None:
    """
    Check if we have a cached response for this query.

    Returns (response_text, query_type) or None.
    """
    return _get_query_cache_layer().lookup(message)


def _store_query_cache(message: str, query_type: str, response: str, tools_used: list[str]) -> None:
    """Store a completed response in the query cache (background, best-effort)."""
    try:
        _get_query_cache_layer().store(message, query_type, response, tools_used)
    except Exception as exc:
        logger.warning("Failed to cache query response: %s", exc)


async def cached_response_stream(response_text: str, query_type: str) -> AsyncGenerator[str, None]:
    """Emit SSE events for a cached response, mimicking the normal agent flow."""
    yield ": ping\n\n"
    yield _sse("route", {"query_type": query_type, "skill_name": None})
    yield _sse("status", {"message": "Retrieved from cache"})
    yield _sse("response", {"content": response_text})
    yield _sse("done", {})


# ── Agent streaming ───────────────────────────────────

async def agent_event_stream(message: str, thread_id: str) -> AsyncGenerator[str, None]:
    """Run the LangGraph agent in a thread and yield SSE events."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    # Cancellation signal for this stream
    cancel_event = threading.Event()
    _active_cancels[thread_id] = cancel_event

    # Track response content and metadata for post-stream caching
    captured_response: list[str] = []
    captured_query_type: list[str] = ["detailed"]
    captured_tools: list[str] = []

    def run_sync():
        try:
            agent = get_agent()
            for event in agent.stream(
                {"messages": [("human", message)]},
                {"configurable": {"thread_id": thread_id}},
            ):
                if cancel_event.is_set():
                    break
                asyncio.run_coroutine_threadsafe(queue.put(("event", event)), loop)
        except Exception as exc:
            if not cancel_event.is_set():
                asyncio.run_coroutine_threadsafe(queue.put(("error", str(exc))), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)

    thread = threading.Thread(target=run_sync, daemon=True)
    thread.start()

    # Keep connection alive while waiting
    yield ": ping\n\n"

    try:
        while True:
            item = await queue.get()
            kind, payload = item

            if kind == "done":
                if cancel_event.is_set():
                    yield _sse("cancelled", {"message": "Query cancelled"})
                else:
                    yield _sse("done", {})
                    # Store in query cache (background thread, non-blocking)
                    if captured_response:
                        full_response = captured_response[-1]
                        threading.Thread(
                            target=_store_query_cache,
                            args=(message, captured_query_type[0], full_response, captured_tools),
                            daemon=True,
                        ).start()
                break

            if kind == "error":
                yield _sse("error", {"message": payload})
                break

            event = payload

            if "route_query" in event:
                q = event["route_query"]
                query_type = q.get("query_type", "unknown")
                captured_query_type[0] = query_type
                yield _sse("route", {
                    "query_type": query_type,
                    "skill_name": q.get("skill_name"),
                })

            elif "rag_handler" in event:
                yield _sse("status", {"message": "Searching research papers…"})

            elif "tools" in event:
                msgs = event["tools"].get("messages", [])
                if msgs:
                    last = msgs[-1]
                    tool_name = last.name if hasattr(last, "name") else "tool"
                    captured_tools.append(tool_name)
                    yield _sse("tool", {"tool_name": tool_name})

            elif "simple_handler" in event or "detailed_handler" in event:
                node = "simple_handler" if "simple_handler" in event else "detailed_handler"
                msgs = event[node].get("messages", [])
                if msgs:
                    last = msgs[-1]
                    has_content = hasattr(last, "content") and last.content
                    no_tool_calls = not (hasattr(last, "tool_calls") and last.tool_calls)
                    if has_content and no_tool_calls:
                        captured_response.append(last.content)
                        yield _sse("response", {"content": last.content})
    finally:
        _active_cancels.pop(thread_id, None)


# ── Routes ────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    # Check query-level cache first
    cached = _try_query_cache(req.message)
    if cached is not None:
        response_text, query_type = cached
        return StreamingResponse(
            cached_response_stream(response_text, query_type),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    return StreamingResponse(
        agent_event_stream(req.message, req.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/chat/cancel")
async def cancel_chat(req: CancelRequest):
    """Cancel an in-flight agent stream for the given thread_id."""
    cancel_event = _active_cancels.get(req.thread_id)
    if cancel_event is not None:
        cancel_event.set()
        return {"cancelled": True}
    return {"cancelled": False, "reason": "no active stream for this thread"}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/cache/stats")
async def cache_stats():
    return get_cache().stats()


@app.post("/api/cache/clear")
async def cache_clear(cache_type: Optional[str] = Query(default="all", pattern="^(tool|query|all)$")):
    count = get_cache().clear_all(cache_type)
    return {"cleared": count, "type": cache_type}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
