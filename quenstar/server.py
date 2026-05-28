from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from .config import QuenStarConfig
from .engine import Engine
from .kvstore import KVCacheStore
from .session import SessionManager
from .types import ChatCompletionRequest

_log = logging.getLogger(__name__)


def create_app(config: QuenStarConfig) -> FastAPI:
    from contextlib import asynccontextmanager

    engine = Engine(config)
    kvstore = KVCacheStore(
        config=config.kv_cache,
        model_id=engine.model_id,
        n_ctx=config.model.n_ctx,
    )
    session = SessionManager(engine, kvstore, config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _log.info("QuenStar server started")
        yield
        _log.info("QuenStar shutting down: saving session to disk KV cache")
        try:
            session.save()
        except Exception as exc:
            _log.warning("Failed to save session on shutdown: %s", exc)

    app = FastAPI(title="QuenStar", version="0.1.0", lifespan=lifespan)

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "quenstar",
                }
            ],
        }

    @app.get("/v1/models/{model_id:path}")
    async def get_model(model_id: str):
        return {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "quenstar",
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        req = ChatCompletionRequest.from_dict(body)

        _log.info(
            "chat/completions: model=%s stream=%s msgs=%d tools=%d temp=%s",
            req.model or engine.model_id,
            req.stream,
            len(req.messages),
            len(req.tools) if req.tools else 0,
            req.temperature,
        )

        resumed = session.new_session(req.messages)
        if resumed:
            _log.info("Session resumed from disk KV cache (no re-prefill needed)")

        if req.stream:
            return EventSourceResponse(
                _stream_chat(engine, session, req),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return await _non_stream_chat(engine, session, req)

    @app.get("/health")
    async def health():
        files = kvstore.list_files()
        return {
            "status": "ok",
            "model": engine.model_path,
            "model_id": engine.model_id,
            "n_ctx": config.model.n_ctx,
            "n_tokens_current": engine.n_tokens,
            "kv_cache_files": len(files),
            "kv_cache_size_mb": round(kvstore.total_size_bytes() / (1024 * 1024), 1),
            "session_id": session.session_id,
            "session_resumed": session.is_resumed,
        }

    @app.get("/sessions")
    async def list_sessions():
        return {"sessions": session.list_sessions()}

    if config.server.cors:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    return app


async def _non_stream_chat(
    engine: Engine,
    session: SessionManager,
    request: ChatCompletionRequest,
) -> dict[str, Any]:
    chunks = list(engine.chat_completion(request))
    if not chunks:
        return _empty_response(engine.model_id)

    result = chunks[-1]

    if "choices" in result and len(result["choices"]) > 0:
        choice = result["choices"][0]
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason", "stop")
    else:
        message = {}
        finish_reason = "stop"

    content = message.get("content", "")
    tool_calls = message.get("tool_calls", [])

    usage = result.get("usage", {
        "prompt_tokens": engine.n_tokens,
        "completion_tokens": 0,
        "total_tokens": engine.n_tokens,
    })

    assistant_msg: dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
        assistant_msg["content"] = None

    session.update(request.messages + [assistant_msg])
    session.save_checkpoint()

    return {
        "id": _gen_id("chatcmpl-"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": engine.model_id,
        "choices": [
            {
                "index": 0,
                "message": assistant_msg,
                "finish_reason": finish_reason or "stop",
            }
        ],
        "usage": usage,
    }


async def _stream_chat(
    engine: Engine,
    session: SessionManager,
    request: ChatCompletionRequest,
) -> AsyncIterator[dict[str, Any]]:
    accumulated_content = ""
    accumulated_tool_calls: list[dict[str, Any]] = []
    final_finish = "stop"
    chunk_count = 0

    try:
        for chunk in engine.chat_completion(request):
            chunk_count += 1
            choices = chunk.get("choices", [])

            if choices:
                choice = choices[0]

                delta = choice.get("delta") or {}
                message = choice.get("message") or {}

                choice_data = delta if delta else message

                content_delta = choice_data.get("content", "")
                if content_delta:
                    accumulated_content += content_delta

                tool_delta = choice_data.get("tool_calls", [])
                if tool_delta:
                    accumulated_tool_calls.extend(tool_delta)

                if choice.get("finish_reason"):
                    final_finish = choice["finish_reason"]

            yield {"data": json.dumps(chunk)}

    except Exception as exc:
        _log.error("Stream error: %s", exc, exc_info=True)
        yield {"data": json.dumps({
            "id": _gen_id("chatcmpl-"),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": engine.model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error",
                }
            ],
        })}

    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": accumulated_content or None,
    }
    if accumulated_tool_calls:
        assistant_msg["tool_calls"] = accumulated_tool_calls
        assistant_msg["content"] = None

    session.update(request.messages + [assistant_msg])
    session.save_checkpoint()

    if final_finish != "stop":
        yield {"data": json.dumps({
            "id": _gen_id("chatcmpl-"),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": engine.model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": engine.n_tokens,
                "completion_tokens": 0,
                "total_tokens": engine.n_tokens,
            },
        })}


def _empty_response(model_id: str) -> dict[str, Any]:
    return {
        "id": _gen_id("chatcmpl-"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _gen_id(prefix: str = "", length: int = 29) -> str:
    import random
    import string

    chars = string.ascii_letters + string.digits
    return prefix + "".join(random.choices(chars, k=length))


def run_server(config: QuenStarConfig):
    app = create_app(config)
    server_cfg = config.server
    _log.info(
        "QuenStar server ready on http://%s:%d (model: %s, ctx: %d)",
        server_cfg.host,
        server_cfg.port,
        config.model.path,
        config.model.n_ctx,
    )
    uvicorn.run(
        app,
        host=server_cfg.host,
        port=server_cfg.port,
        log_level="warning",
        access_log=False,
    )
