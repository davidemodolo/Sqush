from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from .config import QuenStarConfig
from .engine import InferenceEngine

log = logging.getLogger(__name__)

ENGINE: Optional[InferenceEngine] = None
CONFIG: Optional[QuenStarConfig] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_app(engine: InferenceEngine, config: QuenStarConfig) -> FastAPI:
    global ENGINE, CONFIG
    ENGINE = engine
    CONFIG = config

    app = FastAPI(title="QuenStar", version="2.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/health/vram")
    async def vram():
        return ENGINE.get_vram_info()

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": CONFIG.model.repo,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "quenstar",
                }
            ],
        }

    @app.get("/v1/models/{model_id}")
    async def get_model(model_id: str):
        if model_id != CONFIG.model.repo:
            raise HTTPException(status_code=404, detail="Model not found")
        return {
            "id": CONFIG.model.repo,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "quenstar",
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens")
        temperature = body.get("temperature")
        top_p = body.get("top_p")

        if temperature is not None:
            ENGINE.temperature = temperature
        if top_p is not None:
            ENGINE.top_p = top_p

        if stream:
            return EventSourceResponse(_stream_response(messages, max_tokens))
        else:
            return _sync_response(messages, max_tokens)

    return app


async def _stream_response(messages, max_tokens):
    import asyncio

    request_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    model = CONFIG.model.repo

    yield {
        "event": "message",
        "data": json.dumps({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }),
    }

    loop = asyncio.get_event_loop()

    def _next_token(gen):
        try:
            return next(gen)
        except StopIteration:
            return None

    gen = ENGINE.chat_completion_stream(messages, max_tokens)

    # State machine for reasoning vs content
    buffer = ""
    state = "pre"  # pre | think | post
    think_emitted = 0
    content_emitted = 0

    while True:
        text = await loop.run_in_executor(None, _next_token, gen)
        if text is None:
            break
        buffer += text

        if state == "pre":
            idx = buffer.find("<think>")
            if idx == -1:
                # No think tag yet — emit preceding whitespace/newlines as content
                new_content = buffer[content_emitted:]
                if new_content:
                    yield _delta_chunk(request_id, created, model, content=new_content)
                    content_emitted = len(buffer)
            else:
                # Found <think> — emit any preceding text as content
                pre_text = buffer[content_emitted:idx]
                if pre_text:
                    yield _delta_chunk(request_id, created, model, content=pre_text)
                content_emitted = idx
                state = "think"
                think_emitted = idx + len("<think>")
                # fall through to think handling

        if state == "think":
            end_idx = buffer.find("</think>")
            if end_idx == -1:
                # Still in think mode — emit incremental reasoning
                new_think = buffer[think_emitted:]
                if new_think:
                    yield _delta_chunk(request_id, created, model, reasoning_content=new_think)
                    think_emitted = len(buffer)
            else:
                # Found </think> — emit remaining reasoning
                new_think = buffer[think_emitted:end_idx]
                if new_think:
                    yield _delta_chunk(request_id, created, model, reasoning_content=new_think)
                think_emitted = len(buffer)
                content_emitted = end_idx + len("</think>")
                state = "post"
                # emit any text after </think> that's already in buffer
                post_text = buffer[content_emitted:]
                if post_text:
                    yield _delta_chunk(request_id, created, model, content=post_text)
                    content_emitted = len(buffer)
                continue

        if state == "post":
            new_content = buffer[content_emitted:]
            if new_content:
                yield _delta_chunk(request_id, created, model, content=new_content)
                content_emitted = len(buffer)

    yield {
        "event": "message",
        "data": json.dumps({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }),
    }

    yield {"event": "message", "data": "[DONE]"}


def _delta_chunk(request_id, created, model, content=None, reasoning_content=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    return {
        "event": "message",
        "data": json.dumps({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }),
    }


def _sync_response(messages, max_tokens):
    raw_text = ENGINE.chat_completion_sync(messages, max_tokens)

    content = raw_text
    reasoning_content = None

    think_start = raw_text.find("<think>")
    if think_start != -1:
        think_end = raw_text.find("</think>", think_start)
        if think_end != -1:
            reasoning_content = raw_text[think_start + len("<think>"):think_end].strip()
            content = raw_text[think_end + len("</think>"):].strip()

    message = {"role": "assistant", "content": content}
    if reasoning_content:
        message["reasoning_content"] = reasoning_content

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": CONFIG.model.repo,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
