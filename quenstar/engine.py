from __future__ import annotations

import logging
import time
from typing import Any, Iterator, Optional

import torch

log = logging.getLogger(__name__)

PREFILL_CHUNK = 1024


def _safe_messages(messages: list[dict]) -> list[dict]:
    """Preprocess messages: Qwen3.6 template expects tool_call.arguments as dict,
    but OpenAI API sends them as JSON strings. Convert to dict to avoid Jinja2
    'Can only get item pairs from a mapping' errors."""
    import json as _json
    safe = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and "tool_calls" in m:
            tc_list = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = tc.get("function", {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = _json.loads(fn["arguments"])
                    except _json.JSONDecodeError:
                        pass
                tc["function"] = fn
                tc_list.append(tc)
            m["tool_calls"] = tc_list
        safe.append(m)
    return safe


class InferenceEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        cache_config: Any = None,
        max_context: int = 262144,
        max_new_tokens: int = 65536,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        presence_penalty: float = 1.5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cache_factory = cache_config
        self.max_context = max_context
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self._last_prompt_tokens = 0

    def _chunked_prefill(self, input_ids: torch.Tensor) -> tuple[object, torch.Tensor]:
        """Prefill long prompts in chunks to bound the fla linear-attention transient.

        Prefills all but the last token; generate() handles the last token + decode.
        Returns (cache, input_ids) — pass both to generate().
        """
        total_len = input_ids.shape[1]
        cache = self.cache_factory() if self.cache_factory is not None else None

        offset = 0
        while offset < total_len - 1:
            end = min(offset + PREFILL_CHUNK, total_len - 1)
            chunk = input_ids[:, offset:end]
            with torch.no_grad():
                out = self.model(
                    input_ids=chunk,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
            cache = out.past_key_values
            offset = end
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / (1024**3)
                log.info(f"prefill {offset}/{total_len} tokens, VRAM={alloc:.1f} GB")

        return cache, input_ids

    def _prepare_generation(self, input_ids: torch.Tensor, max_tokens: Optional[int] = None) -> tuple[dict, torch.Tensor]:
        """Build generate kwargs and determine the input for generate().

        For long prompts (>PREFILL_CHUNK), does chunked prefill first.
        Returns (generate_kwargs, generate_input_ids).
        """
        total_len = input_ids.shape[1]
        kwargs = {
            "max_new_tokens": max_tokens if max_tokens is not None else self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if total_len > PREFILL_CHUNK:
            cache, _ = self._chunked_prefill(input_ids)
            kwargs["past_key_values"] = cache
            self._prefilled = True
        else:
            if self.cache_factory is not None:
                kwargs["past_key_values"] = self.cache_factory()
            self._prefilled = False

        return kwargs, input_ids

    def _tokenize(self, messages: list[dict[str, str]], enable_thinking: bool = True,
                   tools: Optional[list] = None) -> torch.Tensor:
        kwargs = {"add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        kwargs["enable_thinking"] = enable_thinking
        text = self.tokenizer.apply_chat_template(_safe_messages(messages), tokenize=False, **kwargs)
        inputs = self.tokenizer(text, return_tensors="pt")
        return inputs["input_ids"].to(self.model.device)

    def chat_completion_sync(
        self,
        messages: list[dict[str, str]],
        max_tokens: Optional[int] = None,
        enable_thinking: bool = True,
        tools: Optional[list] = None,
    ) -> tuple[str, int, int]:
        input_ids = self._tokenize(messages, enable_thinking=enable_thinking, tools=tools)
        kwargs, generate_input = self._prepare_generation(input_ids, max_tokens)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(generate_input, **kwargs)
        elapsed = time.perf_counter() - t0

        n_input = input_ids.shape[1]
        generated_ids = outputs[0][n_input:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        n_tokens = len(generated_ids)
        log.info(f"Generated {n_tokens} tokens in {elapsed:.2f}s ({n_tokens / elapsed:.1f} tok/s)")
        return text, input_ids.shape[1], n_tokens

    def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: Optional[int] = None,
        enable_thinking: bool = True,
        tools: Optional[list] = None,
    ) -> Iterator[str]:
        from threading import Thread

        from transformers import TextIteratorStreamer

        input_ids = self._tokenize(messages, enable_thinking=enable_thinking, tools=tools)
        self._last_prompt_tokens = input_ids.shape[1]
        kwargs, generate_input = self._prepare_generation(input_ids, max_tokens)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        kwargs["streamer"] = streamer

        thread = Thread(target=self.model.generate, kwargs={**kwargs, "inputs": generate_input})
        thread.start()

        for text in streamer:
            yield text

        thread.join()

    def get_vram_info(self) -> dict:
        if not torch.cuda.is_available():
            return {"cuda_available": False}

        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        return {
            "cuda_available": True,
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - reserved, 2),
        }
