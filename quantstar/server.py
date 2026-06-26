from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from .config import QuantStarConfig
from .engine import InferenceEngine, _safe_messages

log = logging.getLogger(__name__)

ENGINE: Optional[InferenceEngine] = None  # set by create_app()
CONFIG: Optional[QuantStarConfig] = None   # set by create_app()


def _is_small_task(messages: list[dict], max_tokens: Optional[int]) -> bool:
    """Detect title generation and other lightweight tasks that don't need thinking.
    
    OpenCode uses a small model for title generation. When our server is the small model,
    we detect title requests by their prompt content and disable reasoning to avoid
    leaking <think> tags into the title text.
    """
    if not messages:
        return False
    combined = " ".join(
        m.get("content", "") if isinstance(m.get("content"), str) else ""
        for m in messages
    ).lower()
    # OpenCode title generation typically contains these patterns
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


def create_app(engine: InferenceEngine, config: QuantStarConfig) -> FastAPI:
    global ENGINE, CONFIG
    ENGINE = engine
    CONFIG = config

    app = FastAPI(title="QuantStar", version="2.0.0")

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
                    "owned_by": "quantstar",
                    "context_window": ENGINE.max_context,
                    "max_output_tokens": ENGINE.max_new_tokens,
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
            "owned_by": "quantstar",
            "context_window": ENGINE.max_context,
            "max_output_tokens": ENGINE.max_new_tokens,
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

        if temperature is not None:
            ENGINE.temperature = temperature
        if top_p is not None:
            ENGINE.top_p = top_p

        enable_thinking = not _is_small_task(messages, max_tokens)
        log.info("POST /v1/chat/completions stream=%s enable_thinking=%s tools=%s max_tokens=%s",
                 stream, enable_thinking, bool(tools), max_tokens)

        if stream:
            return EventSourceResponse(_stream_response(messages, max_tokens, enable_thinking, tools))
        else:
            return _sync_response(messages, max_tokens, enable_thinking, tools)

    return app


async def _stream_response(messages, max_tokens, enable_thinking=True, tools=None):
    import asyncio

    request_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    model = CONFIG.model.repo

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

    gen = ENGINE.chat_completion_stream(messages, max_tokens, enable_thinking=enable_thinking, tools=tools)
    prompt_tokens = ENGINE._last_prompt_tokens

    buffer = ""
    # Qwen3.6 template always emits <think>\n before generation when thinking is
    # enabled. So we start in "think" state — the model's first tokens are the
    # reasoning content, followed by </think> and the actual response.
    state = "think" if enable_thinking else "post"
    think_emitted = 0
    content_emitted = 0
    tool_calls_emitted = 0
    has_tool_calls = False

    THINK_TAG = "<think>"
    THINK_CLOSE = "</think>"
    TOOL_TAG = "<tool_call>"
    TOOL_CLOSE = "</tool_call>"
    HOLD = max(len(THINK_TAG), len(THINK_CLOSE), len(TOOL_TAG), len(TOOL_CLOSE)) - 1  # chars held back to avoid emitting partial XML tags

    # Create per-request incremental tool call parser
    tool_parser = _make_tool_call_stream_parser(tool_index=0)

    log.info("_stream_response start enable_thinking=%s state=%s tools=%s max_tokens=%s prompt_tokens=%d",
             enable_thinking, state, bool(tools), max_tokens, prompt_tokens)

    while True:
        text = await loop.run_in_executor(None, _next_token, gen)
        if text is None:
            break
        buffer += text

        if state == "think":
            end_idx = buffer.find(THINK_CLOSE, think_emitted)
            if end_idx == -1:
                # Hold back HOLD chars to avoid leaking partial </think>
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
                log.info("_stream_response found <tool_call> → state=tool_call")

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

        # After all state-specific processing, if we're in post and
        # have hit end of stream, flush any remaining holdback
        if state == "post" and content_emitted < len(buffer):
            remaining = buffer[content_emitted:]
            if remaining:
                yield _delta_chunk(request_id, created, model, content=remaining)
                content_emitted = len(buffer)

    # Flush final holdback
    if state == "post" and content_emitted < len(buffer):
        remaining = buffer[content_emitted:]
        if remaining:
            yield _delta_chunk(request_id, created, model, content=remaining)
    if state == "think" and think_emitted < len(buffer):
        remaining = buffer[think_emitted:]
        if remaining:
            yield _delta_chunk(request_id, created, model, reasoning_content=remaining)

    completion_tokens = len(ENGINE.tokenizer.encode(buffer, add_special_tokens=False))

    log.info("_stream_response done prompt_tokens=%d completion_tokens=%d total_tokens=%d buffer_chars=%d",
             prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, len(buffer))

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
    import re
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

        # Extract function name as soon as we see <function=NAME>
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

        # On finalize, parse all complete <parameter> blocks and emit arguments
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


def _sync_response(messages, max_tokens, enable_thinking=True, tools=None):
    raw_text, prompt_tokens, completion_tokens = ENGINE.chat_completion_sync(
        messages, max_tokens, enable_thinking=enable_thinking, tools=tools
    )
    log.info("_sync_response done prompt_tokens=%d completion_tokens=%d total_tokens=%d text_chars=%d",
             prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, len(raw_text))

    content = raw_text
    reasoning_content = None
    tool_calls = None

    think_start = raw_text.find("<think>")
    if think_start != -1:
        think_end = raw_text.find("</think>", think_start)
        if think_end != -1:
            reasoning_content = raw_text[think_start + len("<think>"):think_end].strip()
            content = raw_text[think_end + len("</think>"):].strip()

    # Parse tool calls from content
    tool_start = content.find("<tool_call>")
    if tool_start != -1:
        tool_end = content.find("</tool_call>", tool_start)
        if tool_end != -1:
            tool_xml = content[tool_start + len("<tool_call>"):tool_end]
            text_before = content[:tool_start].strip()
            content = text_before if text_before else None
            # Create a fresh incremental parser for sync parsing
            tool_parser = _make_tool_call_stream_parser(tool_index=0)
            tool_call_deltas = tool_parser(tool_xml, finalize=True)
            if tool_call_deltas:
                # Merge deltas: first has id+name, subsequent have incremental args
                tool_calls = []
                current = None
                for delta_list in tool_call_deltas:
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
                            tool_calls.append(current)
                        elif current is not None and "function" in d:
                            current["function"]["arguments"] += d["function"]["arguments"]

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
        "model": CONFIG.model.repo,
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
