from __future__ import annotations

import logging
import time
from typing import Any, Iterator, Optional

import torch

log = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        cache_config: Any = None,
        max_context: int = 262144,
        max_new_tokens: int = 32768,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        presence_penalty: float = 1.5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache_config
        self.max_context = max_context
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty

    def _build_generate_kwargs(self) -> dict:
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.cache is not None:
            kwargs["past_key_values"] = self.cache
        return kwargs

    def _tokenize(self, messages: list[dict[str, str]]) -> torch.Tensor:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        return inputs["input_ids"].to(self.model.device)

    def chat_completion_sync(
        self,
        messages: list[dict[str, str]],
        max_tokens: Optional[int] = None,
    ) -> str:
        input_ids = self._tokenize(messages)
        kwargs = self._build_generate_kwargs()
        if max_tokens is not None:
            kwargs["max_new_tokens"] = max_tokens

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(input_ids, **kwargs)
        elapsed = time.perf_counter() - t0

        generated_ids = outputs[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        n_tokens = len(generated_ids)
        log.info(f"Generated {n_tokens} tokens in {elapsed:.2f}s ({n_tokens / elapsed:.1f} tok/s)")
        return text

    def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        from threading import Thread

        from transformers import TextIteratorStreamer

        input_ids = self._tokenize(messages)
        kwargs = self._build_generate_kwargs()
        if max_tokens is not None:
            kwargs["max_new_tokens"] = max_tokens

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        kwargs["streamer"] = streamer

        thread = Thread(target=self.model.generate, kwargs={**kwargs, "inputs": input_ids})
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
