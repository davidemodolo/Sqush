from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from .config import SqushConfig
from .engine import InferenceEngine

log = logging.getLogger(__name__)


def _is_small_task(messages: list[dict], max_tokens: Optional[int]) -> bool:
    """Detect title generation and other lightweight tasks that don't need thinking.

    OpenCode uses a small model for title generation. When our server is the small model,
    we detect title requests by their prompt content and disable reasoning to avoid
    leaking <think> tags into the title text.
    """
    if not messages:
        return False

    def _extract_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                p.get("text", "") if isinstance(p, dict) and p.get("type") == "text" else ""
                for p in content
            )
        return ""

    combined = " ".join(_extract_text(m.get("content", "")) for m in messages).lower()
    title_markers = [
        "generate a short title",
        "generate a title for",
        "descriptive title for this",
        "concise title for this conversation",
        "summarize this conversation in a title",
    ]
    for marker in title_markers:
        if marker in combined:
            return True
    return False


def create_app(engine: InferenceEngine, config: SqushConfig) -> FastAPI:
    app = FastAPI(title="Sqush", version="2.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/health/vram")
    async def vram():
        return engine.get_vram_info()

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model.repo,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "sqush",
                    "context_window": engine.max_context,
                    "max_output_tokens": engine.max_new_tokens,
                }
            ],
        }

    @app.get("/v1/models/{model_id}")
    async def get_model(model_id: str):
        if model_id != config.model.repo:
            raise HTTPException(status_code=404, detail="Model not found")
        return {
            "id": config.model.repo,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "sqush",
            "context_window": engine.max_context,
            "max_output_tokens": engine.max_new_tokens,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens")
        temperature = body.get("temperature")
        top_p = body.get("top_p")
        tools = body.get("tools")

        enable_thinking = not _is_small_task(messages, max_tokens)
        log.info("POST /v1/chat/completions stream=%s enable_thinking=%s tools=%s max_tokens=%s",
                 stream, enable_thinking, bool(tools), max_tokens)

        if stream:
            return EventSourceResponse(
                _stream_response(messages, max_tokens, enable_thinking, tools, engine, config, temperature, top_p)
            )
        else:
            # Generation is blocking and can take minutes; run it off the event
            # loop so other requests (and /health) stay responsive.
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, _sync_response, messages, max_tokens, enable_thinking,
                tools, engine, config, temperature, top_p,
            )

    return app


async def _stream_response(messages, max_tokens, enable_thinking, tools,
                           engine: InferenceEngine, config: SqushConfig,
                           temperature=None, top_p=None):
    import asyncio

    request_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    model = config.model.repo

    yield {
        "event": None,
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

    gen = engine.chat_completion_stream(
        messages, max_tokens, enable_thinking=enable_thinking, tools=tools,
        temperature=temperature, top_p=top_p,
    )

    buffer = ""
    state = "think" if enable_thinking else "post"
    think_emitted = 0
    content_emitted = 0
    tool_calls_emitted = 0
    has_tool_calls = False
    tool_call_count = 0
    tool_parser = None

    THINK_TAG = "<think>"
    THINK_CLOSE = "</think>"
    TOOL_TAG = "<tool_call>"
    TOOL_CLOSE = "</tool_call>"
    HOLD = max(len(THINK_TAG), len(THINK_CLOSE), len(TOOL_TAG), len(TOOL_CLOSE)) - 1

    stream_start = time.perf_counter()
    last_tps_log_tokens = 0
    last_tps_log_time = stream_start
    think_start = stream_start if state == "think" else None
    token_count = 0
    prompt_tokens = 0
    first_token = True

    while True:
        text = await loop.run_in_executor(None, _next_token, gen)
        if first_token:
            # _last_prompt_tokens is only populated once the generator body runs,
            # which first happens on the pull above — reading it before the loop
            # returns the previous request's value (or 0 on the first request).
            prompt_tokens = engine._last_prompt_tokens
            log.info("_stream_response start enable_thinking=%s state=%s tools=%s max_tokens=%s prompt_tokens=%d",
                     enable_thinking, state, bool(tools), max_tokens, prompt_tokens)
            first_token = False
        if text is None:
            break
        buffer += text

        # Count incrementally on the new chunk — re-encoding the whole buffer per
        # token is O(n²) and slows the very stream this meter measures.
        token_count += len(engine.tokenizer.encode(text, add_special_tokens=False))
        current_tokens = token_count
        tokens_since = current_tokens - last_tps_log_tokens
        tps_interval = getattr(config.logging, 'tps_interval_tokens', 50)
        if tokens_since >= tps_interval:
            now = time.perf_counter()
            elapsed_total = now - stream_start
            elapsed_recent = now - last_tps_log_time
            total_tps = current_tokens / elapsed_total if elapsed_total > 0 else 0
            recent_tps = tokens_since / elapsed_recent if elapsed_recent > 0 else 0
            log.info("stream TPS: %.1f total, %.1f recent (%d tok/%.2fs, state=%s)",
                     total_tps, recent_tps, tokens_since, elapsed_recent, state)
            last_tps_log_tokens = current_tokens
            last_tps_log_time = now

        if state == "think":
            end_idx = buffer.find(THINK_CLOSE, think_emitted)
            if end_idx == -1:
                safe_end = max(think_emitted, len(buffer) - HOLD)
                if safe_end > think_emitted:
                    text = buffer[think_emitted:safe_end]
                    yield _delta_chunk(request_id, created, model, reasoning_content=text)
                    think_emitted = safe_end
            else:
                if end_idx > think_emitted:
                    text = buffer[think_emitted:end_idx]
                    yield _delta_chunk(request_id, created, model, reasoning_content=text)
                think_emitted = len(buffer)
                content_emitted = end_idx + len(THINK_CLOSE)
                if think_start is not None:
                    think_tokens = len(engine.tokenizer.encode(buffer[:end_idx], add_special_tokens=False))
                    think_elapsed = time.perf_counter() - think_start
                    think_tps = think_tokens / think_elapsed if think_elapsed > 0 else 0
                    log.info("stream TPS think-phase: %d tokens in %.2fs (%.1f tok/s)",
                             think_tokens, think_elapsed, think_tps)
                state = "post"
                continue

        if state == "post":
            remaining = buffer[content_emitted:]
            tool_idx = remaining.find(TOOL_TAG)
            if tool_idx == -1:
                safe_end = max(content_emitted, len(buffer) - HOLD)
                if safe_end > content_emitted:
                    yield _delta_chunk(request_id, created, model, content=buffer[content_emitted:safe_end])
                    content_emitted = safe_end
            else:
                tool_abs = content_emitted + tool_idx
                if tool_abs > content_emitted:
                    yield _delta_chunk(request_id, created, model, content=buffer[content_emitted:tool_abs])
                content_emitted = tool_abs + len(TOOL_TAG)
                tool_calls_emitted = content_emitted
                state = "tool_call"
                has_tool_calls = True
                tool_parser = _make_tool_call_stream_parser(tool_index=tool_call_count)
                tool_call_count += 1
                log.info("_stream_response found <tool_call> #%d → state=tool_call", tool_call_count)

        if state == "tool_call":
            end_idx = buffer.find(TOOL_CLOSE, tool_calls_emitted)
            if end_idx == -1:
                new_text = buffer[tool_calls_emitted:]
                deltas = tool_parser(new_text)
                for tc in deltas:
                    yield _delta_chunk(request_id, created, model, tool_calls=tc)
                tool_calls_emitted = len(buffer)
            else:
                remaining_text = buffer[tool_calls_emitted:end_idx]
                deltas = tool_parser(remaining_text, finalize=True)
                for tc in deltas:
                    yield _delta_chunk(request_id, created, model, tool_calls=tc)
                content_emitted = end_idx + len(TOOL_CLOSE)
                state = "post"
                log.info("_stream_response found </tool_call> → state=post")
                continue

        if state == "post" and content_emitted < len(buffer):
            remaining = buffer[content_emitted:]
            if remaining:
                yield _delta_chunk(request_id, created, model, content=remaining)
                content_emitted = len(buffer)

    if state == "post" and content_emitted < len(buffer):
        remaining = buffer[content_emitted:]
        if remaining:
            yield _delta_chunk(request_id, created, model, content=remaining)
    if state == "think" and think_emitted < len(buffer):
        remaining = buffer[think_emitted:]
        if remaining:
            yield _delta_chunk(request_id, created, model, reasoning_content=remaining)
    if state == "tool_call" and tool_parser is not None:
        # Stream ended before </tool_call>; finalize so the arguments delta is
        # still emitted instead of a tool call with an empty argument list.
        deltas = tool_parser(buffer[tool_calls_emitted:], finalize=True)
        for tc in deltas:
            yield _delta_chunk(request_id, created, model, tool_calls=tc)

    completion_tokens = len(engine.tokenizer.encode(buffer, add_special_tokens=False))
    total_elapsed = time.perf_counter() - stream_start
    avg_tps = completion_tokens / total_elapsed if total_elapsed > 0 else 0

    log.info("_stream_response done prompt_tokens=%d completion_tokens=%d total_tokens=%d in %.2fs (%.1f tok/s)",
             prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
             total_elapsed, avg_tps)

    finish_reason = "tool_calls" if has_tool_calls else "stop"

    yield {
        "event": None,
        "data": json.dumps({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }),
    }

    yield {"event": None, "data": "[DONE]"}


def _make_tool_call_stream_parser(tool_index: int = 0):
    """Create a per-request incremental tool call XML parser.

    Returns a function `parse(new_text, finalize=False)` that returns a list of
    tool_call delta lists to emit.

    The function name is emitted as soon as it's found in the XML.
    Arguments are only emitted on finalize, as a single complete JSON delta.
    Streaming arguments character-by-character is not possible with XML parsing
    because JSON restructures when new keys are added (comma insertion breaks
    prefix-based concatenation).
    """
    import uuid as _uuid

    state = dict(
        buff="",
        emitted_name=False,
        call_id=None,
        func_name=None,
    )

    def parse(new_text: str, finalize: bool = False) -> list[list[dict]]:
        if not new_text and not finalize:
            return []

        state["buff"] += new_text
        buff = state["buff"]
        result = []

        if not state["emitted_name"]:
            m = re.search(r'<function=([^>]+)>', buff)
            if m:
                state["func_name"] = m.group(1)
                state["call_id"] = f"call_{_uuid.uuid4().hex[:12]}"
                state["emitted_name"] = True
                result.append([
                    {"index": tool_index, "id": state["call_id"], "type": "function",
                     "function": {"name": state["func_name"], "arguments": ""}},
                ])
                buff = buff[m.end():]
                state["buff"] = buff

        if finalize and state["emitted_name"]:
            param_pattern = re.compile(
                r'<parameter=([^>]+)>\s*(.*?)\s*</parameter>', re.DOTALL
            )
            params = {}
            for m in param_pattern.finditer(state["buff"]):
                pname = m.group(1)
                pval = m.group(2)
                try:
                    pval = json.loads(pval)
                except (json.JSONDecodeError, ValueError):
                    pass
                params[pname] = pval

            if params:
                args_str = json.dumps(params, ensure_ascii=False)
                result.append([
                    {"index": tool_index, "function": {"arguments": args_str}},
                ])

        return result

    return parse


def _delta_chunk(request_id, created, model, content=None, reasoning_content=None, tool_calls=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
        delta["reasoning_text"] = reasoning_content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {
        "event": None,
        "data": json.dumps({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }),
    }


def _sync_response(messages, max_tokens, enable_thinking, tools,
                   engine: InferenceEngine, config: SqushConfig,
                   temperature=None, top_p=None):
    raw_text, prompt_tokens, completion_tokens = engine.chat_completion_sync(
        messages, max_tokens, enable_thinking=enable_thinking, tools=tools,
        temperature=temperature, top_p=top_p,
    )
    log.info("_sync_response done prompt_tokens=%d completion_tokens=%d total_tokens=%d text_chars=%d",
             prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, len(raw_text))

    content = raw_text
    reasoning_content = None
    tool_calls = None

    # Mirror the streaming path: with add_generation_prompt the model emits the
    # closing </think> only (the opening tag lives in the prompt), so everything
    # up to </think> is reasoning even when no <think> tag is present.
    if enable_thinking:
        think_end = raw_text.find("</think>")
        if think_end != -1:
            reasoning = raw_text[:think_end]
            if reasoning.startswith("<think>"):
                reasoning = reasoning[len("<think>"):]
            reasoning_content = reasoning.strip()
            content = raw_text[think_end + len("</think>"):].strip()

    first_tool = content.find("<tool_call>") if content else -1
    if first_tool != -1:
        text_before = content[:first_tool].strip()
        tool_calls_list = []
        for idx, m in enumerate(re.finditer(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)):
            tool_xml = m.group(1)
            parser = _make_tool_call_stream_parser(tool_index=idx)
            deltas = parser(tool_xml, finalize=True)
            current = None
            for delta_list in deltas:
                for d in delta_list:
                    if "id" in d:
                        current = {
                            "id": d["id"],
                            "type": "function",
                            "function": {
                                "name": d["function"]["name"],
                                "arguments": d["function"]["arguments"],
                            },
                        }
                        tool_calls_list.append(current)
                    elif current is not None and "function" in d:
                        current["function"]["arguments"] += d["function"]["arguments"]
        content = text_before if text_before else None
        tool_calls = tool_calls_list if tool_calls_list else None

    message = {"role": "assistant", "content": content}
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
        message["reasoning_text"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = tool_calls

    finish_reason = "tool_calls" if tool_calls else "stop"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": config.model.repo,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
