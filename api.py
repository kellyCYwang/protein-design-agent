"""
FastAPI streaming backend for the Protein Design Agent.

Run with:
    uvicorn api:app --reload --port 8001
"""
import asyncio
import json
import threading
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.agent import build_agent

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


def get_agent():
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = build_agent()
    return _agent


class ChatRequest(BaseModel):
    message: str
    thread_id: str


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def agent_event_stream(message: str, thread_id: str) -> AsyncGenerator[str, None]:
    """Run the LangGraph agent in a thread and yield SSE events."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def run_sync():
        try:
            agent = get_agent()
            for event in agent.stream(
                {"messages": [("human", message)]},
                {"configurable": {"thread_id": thread_id}},
            ):
                asyncio.run_coroutine_threadsafe(queue.put(("event", event)), loop)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(queue.put(("error", str(exc))), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)

    thread = threading.Thread(target=run_sync, daemon=True)
    thread.start()

    # Keep connection alive while waiting
    yield ": ping\n\n"

    while True:
        item = await queue.get()
        kind, payload = item

        if kind == "done":
            yield _sse("done", {})
            break

        if kind == "error":
            yield _sse("error", {"message": payload})
            break

        event = payload

        if "route_query" in event:
            q = event["route_query"]
            yield _sse("route", {
                "query_type": q.get("query_type", "unknown"),
                "skill_name": q.get("skill_name"),
            })

        elif "rag_handler" in event:
            yield _sse("status", {"message": "Searching research papers…"})

        elif "tools" in event:
            msgs = event["tools"].get("messages", [])
            if msgs:
                last = msgs[-1]
                tool_name = last.name if hasattr(last, "name") else "tool"
                yield _sse("tool", {"tool_name": tool_name})

        elif "simple_handler" in event or "detailed_handler" in event:
            node = "simple_handler" if "simple_handler" in event else "detailed_handler"
            msgs = event[node].get("messages", [])
            if msgs:
                last = msgs[-1]
                has_content = hasattr(last, "content") and last.content
                no_tool_calls = not (hasattr(last, "tool_calls") and last.tool_calls)
                if has_content and no_tool_calls:
                    yield _sse("response", {"content": last.content})


@app.post("/api/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        agent_event_stream(req.message, req.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
