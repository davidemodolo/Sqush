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
from .toolcall import (
    StreamingToolCallParser,
    ToolCallRegistry,
    replay_tool_calls,
    split_content_and_tool_calls,
)
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
    registry = ToolCallRegistry(max_entries=config.tool_calling.exact_replay_cache_size)

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
        try:
            return await _handle_chat(request, engine, session, registry, config)
        except Exception as exc:
            _log.error("Unhandled error in chat/completions: %s", exc, exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": str(exc),
                        "type": "server_error",
                        "code": "internal_error",
                    }
                },
            )


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

    @app.get("/health/vram")
    async def health_vram():
        info = _query_gpu_vram()
        info["model_path"] = engine.model_path
        info["n_ctx"] = config.model.n_ctx
        info["offload_kqv"] = config.model.offload_kqv
        info["n_gpu_layers"] = config.model.n_gpu_layers
        return info

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


async def _handle_chat(
    request: Request,
    engine: Engine,
    session: SessionManager,
    registry: ToolCallRegistry,
    config: QuenStarConfig,
):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Invalid JSON body",
                    "type": "invalid_request_error",
                    "code": "invalid_json",
                }
            },
        )

    raw_messages = body.get("messages", [])
    _log.debug("request body: %d bytes, %d messages, %d tools",
               len(json.dumps(body).encode()),
               len(raw_messages),
               len(body.get("tools", [])))

    for i, m in enumerate(raw_messages):
        role = m.get("role", "?")
        c = m.get("content", "")
        if isinstance(c, str):
            _log.debug("  msg[%d] role=%s str_len=%d end=%.80s", i, role, len(c), c[-80:])
        elif isinstance(c, list):
            parts = [p.get("type", "?") for p in c]
            _log.debug("  msg[%d] role=%s parts=%s", i, role, parts)

    if _has_image_content(raw_messages):
        _log.debug("image content detected in messages")
        if not _has_llava():
            return JSONResponse(
                status_code=501,
                content={
                    "error": {
                        "message": "Vision support not available in this llama-cpp-python build.",
                        "type": "server_error",
                        "code": "vision_not_available",
                    }
                },
            )
        if not config.model.mmproj_path:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Image input requires a vision encoder (mmproj GGUF). Set mmproj_path in config.yaml "
                                   "and place the mmproj file in models/.",
                        "type": "invalid_request_error",
                        "code": "mmproj_missing",
                    }
                },
            )

    replayed_messages = replay_tool_calls(raw_messages, registry)
    body["messages"] = replayed_messages
    req = ChatCompletionRequest.from_dict(body)

    if not req.messages:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "`messages` is required",
                    "type": "invalid_request_error",
                    "code": "missing_messages",
                }
            },
        )

    _log.info(
        "chat/completions: model=%s stream=%s msgs=%d tools=%d temp=%s max_tokens=%s seed=%s",
        req.model or engine.model_id,
        req.stream,
        len(req.messages),
        len(req.tools) if req.tools else 0,
        req.temperature,
        req.max_tokens,
        req.seed,
    )

    resumed = session.new_session(req.messages)
    if resumed:
        _log.info("Session resumed from disk KV cache (no re-prefill needed)")

    if req.stream:
        return EventSourceResponse(
            _stream_chat(engine, session, req, registry),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return await _non_stream_chat(engine, session, req, registry)


async def _non_stream_chat(
    engine: Engine,
    session: SessionManager,
    request: ChatCompletionRequest,
    registry: ToolCallRegistry,
) -> dict[str, Any]:
    try:
        prompt_tokens = engine.n_tokens
        chunks = list(engine.chat_completion(request))
    except Exception as exc:
        _log.error("Non-stream inference failed: %s", exc)
        raise

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

    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    # llama-cpp's auto-detected formatter returns tool calls as raw
    # "<tool_call>...</tool_call>" text in content, not structured tool_calls.
    if not tool_calls and content:
        content, parsed = split_content_and_tool_calls(content)
        tool_calls = _finalize_tool_calls(parsed, registry)
    if tool_calls:
        finish_reason = "tool_calls"

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": max(0, engine.n_tokens - prompt_tokens),
        "total_tokens": engine.n_tokens,
    }

    assistant_msg = _finalize_turn(content, tool_calls, registry, session, request)

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
    registry: ToolCallRegistry,
) -> AsyncIterator[dict[str, Any]]:
    stream_id = _gen_id("chatcmpl-")
    parser = StreamingToolCallParser()
    accumulated_content = ""
    emitted_tool_calls: list[dict[str, Any]] = []
    next_index = 0
    model_finish = "stop"
    errored = False
    prompt_tokens = engine.n_tokens

    def chunk(delta: dict[str, Any], finish_reason: Any = None,
              usage: dict[str, Any] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": engine.model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            body["usage"] = usage
        return {"data": json.dumps(body)}

    yield chunk({"role": "assistant"})

    try:
        for raw_chunk in engine.chat_completion(request):
            choices = raw_chunk.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or choice.get("message") or {}

            # Forward any structured tool calls the handler already produced.
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                yield chunk({"tool_calls": [dict(tc, index=idx)]})
                emitted_tool_calls.append({
                    "id": tc.get("id"), "type": "function",
                    "function": tc.get("function", {}),
                })

            content_piece = delta.get("content") or ""
            if content_piece:
                text, parsed = parser.feed(content_piece)
                if text:
                    accumulated_content += text
                    yield chunk({"content": text})
                for tc in parsed:
                    raw = tc.pop("_raw", None)
                    if raw:
                        registry.register(tc["id"], raw)
                    emitted_tool_calls.append(tc)
                    yield chunk({"tool_calls": [dict(tc, index=next_index)]})
                    next_index += 1

            if choice.get("finish_reason"):
                model_finish = choice["finish_reason"]

        text, parsed = parser.flush()
        if text:
            accumulated_content += text
            yield chunk({"content": text})
        for tc in parsed:
            raw = tc.pop("_raw", None)
            if raw:
                registry.register(tc["id"], raw)
            emitted_tool_calls.append(tc)
            yield chunk({"tool_calls": [dict(tc, index=next_index)]})
            next_index += 1

    except Exception as exc:
        errored = True
        _log.error("Stream error: %s", exc, exc_info=True)
        yield {"data": json.dumps({
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": engine.model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            "error": {"message": str(exc), "type": "server_error"},
        })}

    finish_reason = "error" if errored else ("tool_calls" if emitted_tool_calls else model_finish)
    yield chunk({}, finish_reason=finish_reason, usage={
        "prompt_tokens": prompt_tokens,
        "completion_tokens": max(0, engine.n_tokens - prompt_tokens),
        "total_tokens": engine.n_tokens,
    })

    if not errored:
        _finalize_turn(accumulated_content, emitted_tool_calls, registry, session, request)

    yield {"data": "[DONE]"}


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


def _finalize_turn(
    content: str,
    tool_calls: list[dict],
    registry: ToolCallRegistry,
    session: SessionManager,
    request: ChatCompletionRequest,
) -> dict[str, Any]:
    assistant_msg: dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
    session.update(request.messages + [assistant_msg])
    session.save_checkpoint()
    return assistant_msg


def _finalize_tool_calls(parsed: list[dict], registry: ToolCallRegistry) -> list[dict]:
    """Register each parsed tool call's exact raw bytes (for byte-exact replay)
    and strip the private ``_raw`` key before returning to the client."""
    clean = []
    for tc in parsed:
        raw = tc.pop("_raw", None)
        if raw and tc.get("id"):
            registry.register(tc["id"], raw)
        clean.append(tc)
    return clean


def _gen_id(prefix: str = "", length: int = 29) -> str:
    import random
    import string

    chars = string.ascii_letters + string.digits
    return prefix + "".join(random.choices(chars, k=length))


def _query_gpu_vram() -> dict[str, Any]:
    import subprocess

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"error": "nvidia-smi not available"}

    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            try:
                idx, name, total, used, free, util = parts
                total_mb = int(total)
                used_mb = int(used)
                free_mb = int(free)
                gpus.append({
                    "index": int(idx),
                    "name": name,
                    "memory_total_mb": total_mb,
                    "memory_used_mb": used_mb,
                    "memory_free_mb": free_mb,
                    "utilization_pct": int(util) if util != "[Not Supported]" else -1,
                })
            except (ValueError, IndexError):
                pass

    return {"gpus": gpus}


def _register_tool_content(registry: ToolCallRegistry, tool_calls: list[dict], raw_content: str | None):
    if not tool_calls or not raw_content:
        return
    for tc in tool_calls:
        tc_id = tc.get("id", "")
        if tc_id:
            registry.register(tc_id, raw_content)


def _has_image_content(messages: list[dict]) -> bool:
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _has_llava() -> bool:
    try:
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        return True
    except ImportError:
        return False


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
