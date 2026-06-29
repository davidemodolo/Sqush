from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Iterator, Optional

import torch
from PIL import Image

log = logging.getLogger(__name__)

PREFILL_CHUNK = 1024  # tokens per prefill chunk — bounds the FLA linear-attention transient


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


def _extract_images(messages: list[dict]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            image_url = part.get("image_url")
            if not isinstance(image_url, dict):
                continue
            url = image_url.get("url", "")
            if url.startswith("data:image/"):
                try:
                    b64 = url.split(",", 1)[1]
                    img_bytes = base64.b64decode(b64)
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    images.append(img)
                    log.info("_extract_images: decoded image %dx%d", img.width, img.height)
                except Exception as exc:
                    log.warning("_extract_images: failed to decode image data URL: %s", exc)
            elif url.startswith("http://") or url.startswith("https://"):
                log.warning("Remote image URLs are not yet supported: %s", url[:80])
            else:
                log.warning("_extract_images: unknown URL scheme: %s", url[:80])
    log.info("_extract_images: total images found: %d", len(images))
    return images


class InferenceEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        processor: Any = None,
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
        self.processor = processor
        self.cache_factory = cache_config
        self.max_context = max_context
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self._last_prompt_tokens = 0
        self._session_kv = None
        self._session_prompt_ids = None

    def _chunked_prefill(self, input_ids: torch.Tensor,
                         cache: object = None) -> tuple[object, torch.Tensor]:
        """Prefill long prompts in chunks to bound the fla linear-attention transient.

        Prefills all but the last token; generate() handles the last token + decode.
        If *cache* is given, prefill appends to it instead of creating a fresh cache.
        Returns (cache, input_ids) — pass both to generate().
        """
        total_len = input_ids.shape[1]
        if cache is None and self.cache_factory is not None:
            cache = self.cache_factory()

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

        Reuses the session KV cache when the new prompt extends a previous one
        (messages appended, not edited). Falls back to full prefill otherwise.
        Returns (generate_kwargs, generate_input_ids).
        """
        kwargs = {
            "max_new_tokens": max_tokens if max_tokens is not None else self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        total_len = input_ids.shape[1]

        # Reuse existing session cache when new prompt extends previous one
        if self._session_kv is not None and self._session_prompt_ids is not None:
            prompt_len = self._session_prompt_ids.shape[1]
            if total_len > prompt_len and torch.equal(input_ids[:, :prompt_len], self._session_prompt_ids):
                cache_seq_len = getattr(self._session_kv, "get_seq_length", lambda _: 0)(0)
                if total_len > cache_seq_len:
                    new_tokens = input_ids[:, cache_seq_len:]
                    if new_tokens.shape[1] > 1:
                        self._session_kv, _ = self._chunked_prefill(new_tokens, cache=self._session_kv)
                    kwargs["past_key_values"] = self._session_kv
                    self._session_prompt_ids = input_ids
                    return kwargs, input_ids

        if total_len > PREFILL_CHUNK:
            self._session_kv, _ = self._chunked_prefill(input_ids)
            kwargs["past_key_values"] = self._session_kv
        else:
            self._session_kv = self.cache_factory() if self.cache_factory is not None else None
            if self._session_kv is not None:
                kwargs["past_key_values"] = self._session_kv

        self._session_prompt_ids = input_ids
        return kwargs, input_ids

    def _tokenize(self, messages: list[dict[str, str]], images: Optional[list] = None,
                   enable_thinking: bool = True, tools: Optional[list] = None) -> tuple:
        kwargs = {"add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        kwargs["enable_thinking"] = enable_thinking
        kwargs["preserve_thinking"] = True
        text = self.tokenizer.apply_chat_template(_safe_messages(messages), tokenize=False, **kwargs)

        if images and self.processor is not None:
            t_proc = time.perf_counter()
            image_sizes = [(img.width, img.height) for img in images]
            log.info("_tokenize: processing %d image(s) sizes=%s", len(images), image_sizes)
            inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True)
            d = inputs.data if hasattr(inputs, "data") else inputs
            input_ids = d["input_ids"].to(self.model.device)
            pixel_values = d.get("pixel_values")
            image_grid_thw = d.get("image_grid_thw")
            mm_token_type_ids = d.get("mm_token_type_ids")
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.model.device)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(self.model.device)
            if mm_token_type_ids is not None:
                mm_token_type_ids = mm_token_type_ids.to(self.model.device)
            n_image_tokens = int((input_ids == self.model.config.image_token_id).sum()) if self.model.config.image_token_id else 0
            dt = time.perf_counter() - t_proc
            if torch.cuda.is_available():
                vram = torch.cuda.memory_allocated() / (1024**3)
                log.info("_tokenize: processor+move took %.2fs, input_ids=%s, pixel_values=%s, grid_thw=%s, image_tokens=%d, VRAM=%.1f GB",
                         dt, input_ids.shape,
                         pixel_values.shape if pixel_values is not None else None,
                         image_grid_thw.shape if image_grid_thw is not None else None,
                         n_image_tokens, vram)
            else:
                log.info("_tokenize: processor+move took %.2fs, input_ids=%s, pixel_values=%s, grid_thw=%s, image_tokens=%d",
                         dt, input_ids.shape,
                         pixel_values.shape if pixel_values is not None else None,
                         image_grid_thw.shape if image_grid_thw is not None else None,
                         n_image_tokens)
            return input_ids, pixel_values, image_grid_thw, mm_token_type_ids
        else:
            inputs = self.tokenizer(text, return_tensors="pt")
            return inputs["input_ids"].to(self.model.device), None, None, None

    def chat_completion_sync(
        self,
        messages: list[dict[str, str]],
        max_tokens: Optional[int] = None,
        enable_thinking: bool = True,
        tools: Optional[list] = None,
    ) -> tuple[str, int, int]:
        images = _extract_images(messages)
        input_ids, pixel_values, image_grid_thw, mm_token_type_ids = self._tokenize(
            messages, images=images, enable_thinking=enable_thinking, tools=tools,
        )

        if images:
            self._session_kv = None
            self._session_prompt_ids = None

            cache = self.cache_factory() if self.cache_factory is not None else None

            if torch.cuda.is_available():
                vram_before = torch.cuda.memory_allocated() / (1024**3)
                log.info("Vision prefill starting: %d tokens, int4 cache=%s, VRAM=%.1f GB",
                         input_ids.shape[1], cache is not None, vram_before)

            t0 = time.perf_counter()
            with torch.no_grad():
                prefill_out = self.model(
                    input_ids=input_ids,
                    past_key_values=cache,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    mm_token_type_ids=mm_token_type_ids,
                    use_cache=True,
                    logits_to_keep=1,
                )
            cache = prefill_out.past_key_values
            prefill_s = time.perf_counter() - t0

            if torch.cuda.is_available():
                vram_prefill = torch.cuda.memory_allocated() / (1024**3)
                log.info("Vision prefill done: %.2fs, VRAM=%.1f GB", prefill_s, vram_prefill)

            generate_kwargs = {
                "past_key_values": cache,
                "max_new_tokens": max_tokens if max_tokens is not None else self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "do_sample": self.temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = self.model.generate(input_ids, **generate_kwargs)
            elapsed = time.perf_counter() - t0

            if torch.cuda.is_available():
                vram_after = torch.cuda.memory_allocated() / (1024**3)
                peak = torch.cuda.max_memory_allocated() / (1024**3)

            n_input = input_ids.shape[1]
            generated_ids = outputs[0][n_input:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            n_tokens = len(generated_ids)
            log.info("Vision: %d tokens in %.2fs (prefill=%.2fs), VRAM before=%.1f after=%.1f peak=%.1f GB",
                     n_tokens, elapsed, prefill_s, vram_before, vram_after, peak)
            return text, n_input, n_tokens

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

        if "past_key_values" in kwargs:
            self._session_kv = kwargs["past_key_values"]
        self._session_prompt_ids = outputs[0].unsqueeze(0)

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

        images = _extract_images(messages)
        input_ids, pixel_values, image_grid_thw, mm_token_type_ids = self._tokenize(
            messages, images=images, enable_thinking=enable_thinking, tools=tools,
        )
        self._last_prompt_tokens = input_ids.shape[1]

        if images:
            self._session_kv = None
            self._session_prompt_ids = None

            cache = self.cache_factory() if self.cache_factory is not None else None

            if torch.cuda.is_available():
                vram_before = torch.cuda.memory_allocated() / (1024**3)
                log.info("Vision prefill starting (stream): %d tokens, int4 cache=%s, VRAM=%.1f GB",
                         input_ids.shape[1], cache is not None, vram_before)

            t0 = time.perf_counter()
            with torch.no_grad():
                prefill_out = self.model(
                    input_ids=input_ids,
                    past_key_values=cache,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    mm_token_type_ids=mm_token_type_ids,
                    use_cache=True,
                    logits_to_keep=1,
                )
            cache = prefill_out.past_key_values
            prefill_s = time.perf_counter() - t0

            if torch.cuda.is_available():
                vram_prefill = torch.cuda.memory_allocated() / (1024**3)
                log.info("Vision prefill done (stream): %.2fs, VRAM=%.1f GB", prefill_s, vram_prefill)

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            generate_kwargs = {
                "max_new_tokens": max_tokens if max_tokens is not None else self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "do_sample": self.temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "past_key_values": cache,
                "streamer": streamer,
            }

            thread = Thread(target=self.model.generate, kwargs={"inputs": input_ids, **generate_kwargs})
            thread.start()
            yield from streamer
            thread.join()

            if torch.cuda.is_available():
                vram_after = torch.cuda.memory_allocated() / (1024**3)
                peak = torch.cuda.max_memory_allocated() / (1024**3)
                log.info("Vision stream done: VRAM before=%.1f after=%.1f peak=%.1f GB", vram_before, vram_after, peak)
            return

        kwargs, generate_input = self._prepare_generation(input_ids, max_tokens)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        kwargs["streamer"] = streamer

        output_ids = {}
        def _generate():
            output_ids["data"] = self.model.generate(**{**kwargs, "inputs": generate_input})

        thread = Thread(target=_generate)
        thread.start()

        for text in streamer:
            yield text

        thread.join()

        if "past_key_values" in kwargs:
            self._session_kv = kwargs["past_key_values"]
        self._session_prompt_ids = output_ids["data"]

    def reset_session(self) -> None:
        self._session_kv = None
        self._session_prompt_ids = None

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
