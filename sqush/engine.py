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


def _vram_str() -> str:
    """GPU memory as 'alloc/reserved' — reserved is what nvidia-smi shows (minus
    the ~0.3 GB CUDA context); allocated alone understates real usage."""
    if not torch.cuda.is_available():
        return "n/a"
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    return f"{alloc:.1f} GB alloc / {reserved:.1f} GB reserved"


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


def _image_fingerprint(img: Image.Image) -> int:
    """Identity hash of an image's pixels — so a swapped image (same count) is
    detected as new rather than silently reusing a stale vision KV cache."""
    return hash((img.size, img.tobytes()))


def _images_extend(seen: list[Image.Image], current: list[Image.Image]) -> bool:
    """True when *current* is *seen* plus zero or more appended images, i.e. the
    conversation grew without editing earlier images. False means the history
    diverged (image replaced/removed) and the cache must be rebuilt."""
    if len(current) < len(seen):
        return False
    return all(_image_fingerprint(a) == _image_fingerprint(b)
               for a, b in zip(seen, current))


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
        max_image_pixels: Optional[int] = None,
        min_image_pixels: Optional[int] = None,
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
        self.max_image_pixels = max_image_pixels
        self.min_image_pixels = min_image_pixels
        self._last_prompt_tokens = 0
        self._session_kv = None
        self._session_num_messages = 0
        # Tokens the session KV cache was built from (full sequence of the last
        # turn). Reuse requires the new prompt to start with exactly these tokens
        # — length/message-count heuristics silently corrupt the cache when the
        # re-rendered history diverges (e.g. thinking blocks stripped).
        self._session_ids: Optional[torch.Tensor] = None
        # Messages whose content is already covered by the KV cache. Includes
        # the generated assistant response (+1) so new-message indexing skips
        # past the raw generated tokens in _session_ids when extending the cache.
        self._cached_msg_count: int = 0
        # Fingerprint of the input messages the KV cache was built from. Cache
        # reuse assumes the conversation is append-only; if an upstream client
        # edits/regenerates/truncates a prior message the prefix check in
        # _prepare_generation no longer protects us (splicing makes it a
        # tautology), so we validate the unchanged prefix explicitly here.
        self._cached_input_fp: tuple = ()
        self._last_prefill_s: float = 0.0

    @staticmethod
    def _fingerprint(messages: list[dict]) -> tuple:
        return tuple((m.get("role"), hash(str(m.get("content")))) for m in messages)

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
                log.info(f"prefill {offset}/{total_len} tokens, VRAM: {_vram_str()}")

        return cache, input_ids

    def _prepare_generation(self, input_ids: torch.Tensor, max_tokens: Optional[int] = None, num_messages: int = 0,
                            temperature: Optional[float] = None, top_p: Optional[float] = None) -> tuple[dict, torch.Tensor]:
        """Build generate kwargs and determine the input for generate().

        Reuses the session KV cache when the new prompt extends a previous one
        (messages appended, not edited). Falls back to full prefill otherwise.
        Returns (generate_kwargs, generate_input_ids).
        """
        _temperature = temperature if temperature is not None else self.temperature
        _top_p = top_p if top_p is not None else self.top_p
        kwargs = {
            "max_new_tokens": max_tokens if max_tokens is not None else self.max_new_tokens,
            "temperature": _temperature,
            "top_p": _top_p,
            "top_k": self.top_k,
            "do_sample": _temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        total_len = input_ids.shape[1]

        # Reuse the session cache only when the new prompt is an exact token-level
        # extension of the cached sequence. get_seq_length() is the true KV length
        # (one less than the last turn's full sequence — the final generated token
        # is never fed back), so slicing there re-prefills exactly the missing part.
        if self._session_kv is not None and self._session_ids is not None:
            cache_seq_len = self._session_kv.get_seq_length()
            if (0 < cache_seq_len < total_len
                    and cache_seq_len <= self._session_ids.shape[0]
                    and torch.equal(input_ids[0, :cache_seq_len],
                                    self._session_ids[:cache_seq_len].to(input_ids.device))):
                new_tokens = input_ids[:, cache_seq_len:]
                if new_tokens.shape[1] > 1:
                    self._session_kv, _ = self._chunked_prefill(new_tokens, cache=self._session_kv)
                kwargs["past_key_values"] = self._session_kv
                self._session_num_messages = num_messages
                self._last_prefill_s = 0.0
                log.info("Session KV reuse: %d cached + %d new tokens", cache_seq_len, total_len - cache_seq_len)
                return kwargs, input_ids
            log.info("Session KV miss: cached %d tokens do not prefix the new %d-token prompt — full prefill",
                     cache_seq_len, total_len)
            self._free_cache()

        if total_len > PREFILL_CHUNK:
            self._last_prefill_s = time.perf_counter()
            self._session_kv, _ = self._chunked_prefill(input_ids)
            self._last_prefill_s = time.perf_counter() - self._last_prefill_s
            kwargs["past_key_values"] = self._session_kv
        else:
            self._last_prefill_s = 0.0
            self._session_kv = self.cache_factory() if self.cache_factory is not None else None
            if self._session_kv is not None:
                kwargs["past_key_values"] = self._session_kv

        self._session_num_messages = num_messages
        return kwargs, input_ids

    def _tokenize(self, messages: list[dict[str, str]], images: Optional[list] = None,
                   enable_thinking: bool = True, tools: Optional[list] = None,
                   cached_input_ids: Optional[torch.Tensor] = None) -> tuple:
        if cached_input_ids is not None and self._cached_msg_count > 0 and not images:
            cache_seq_len = self._session_kv.get_seq_length() if self._session_kv is not None else cached_input_ids.shape[0]
            bridge = self._session_ids[cache_seq_len:cache_seq_len + 1] \
                if self._session_ids is not None and cache_seq_len < self._session_ids.shape[0] \
                else torch.empty(0, dtype=torch.long, device=self.model.device)
            new_msgs = messages[self._cached_msg_count:]
            kwargs = {"add_generation_prompt": True}
            kwargs["enable_thinking"] = enable_thinking
            kwargs["preserve_thinking"] = True
            new_text = self.tokenizer.apply_chat_template(
                _safe_messages(new_msgs), tokenize=False, **kwargs,
            )
            # Rendering the tail in isolation can make the template prepend a
            # default system block (present at conversation start, already in the
            # cached prefix). Splicing that in again would corrupt the sequence,
            # so drop a leading system turn the new messages didn't ask for.
            first_role = new_msgs[0].get("role") if new_msgs else None
            sys_header = "<|im_start|>system"
            stripped = new_text.lstrip()
            if first_role != "system" and stripped.startswith(sys_header):
                nxt = new_text.find("<|im_start|>", new_text.find(sys_header) + len(sys_header))
                if nxt != -1:
                    log.info("Tail render: dropped injected default system block before splice")
                    new_text = new_text[nxt:]
            new_ids = self.tokenizer.encode(new_text, add_special_tokens=False)
            new_ids_tensor = torch.tensor(new_ids, dtype=torch.long, device=self.model.device)
            combined = torch.cat([cached_input_ids, bridge, new_ids_tensor])
            return combined.unsqueeze(0), None, None, None

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
            proc_kwargs = {"return_tensors": "pt", "padding": True}
            if self.max_image_pixels is not None and self.min_image_pixels is not None:
                proc_kwargs["max_pixels"] = self.max_image_pixels
                proc_kwargs["min_pixels"] = self.min_image_pixels
            inputs = self.processor(text=[text], images=images, **proc_kwargs)
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
                log.info("_tokenize: processor+move took %.2fs, input_ids=%s, pixel_values=%s, grid_thw=%s, image_tokens=%d, VRAM: %s",
                         dt, input_ids.shape,
                         pixel_values.shape if pixel_values is not None else None,
                         image_grid_thw.shape if image_grid_thw is not None else None,
                         n_image_tokens, _vram_str())
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

    def _chunked_vision_prefill(self, input_ids, cache, pixel_values, image_grid_thw, mm_token_type_ids):
        total_len = input_ids.shape[1]
        model_inner = self.model.model

        with torch.no_grad():
            vram_log = lambda tag: log.info("vision vram %s: %s", tag,
                                            _vram_str()) if torch.cuda.is_available() else None

            inputs_embeds = model_inner.get_input_embeddings()(input_ids)

            image_outputs = model_inner.get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(device=input_ids.device, dtype=inputs_embeds.dtype)
            del image_outputs
            torch.cuda.empty_cache()
            vram_log("after vision encoder")

            image_mask, _ = model_inner.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            del image_embeds, image_mask
            vram_log("after merge")

            position_ids = model_inner.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )
            vram_log("after position ids")

            offset = 0
            while offset < total_len - 1:
                end = min(offset + PREFILL_CHUNK, total_len - 1)
                chunk_embeds = inputs_embeds[:, offset:end, :]
                chunk_pos = position_ids[:, :, offset:end] if position_ids is not None else None
                out = model_inner.language_model(
                    inputs_embeds=chunk_embeds,
                    past_key_values=cache,
                    position_ids=chunk_pos,
                    use_cache=True,
                )
                cache = out.past_key_values
                offset = end
                log.info("vision prefill %d/%d tokens, VRAM: %s", offset, total_len, _vram_str())

            if hasattr(model_inner, "rope_deltas"):
                model_inner.rope_deltas = None

        return cache

    def chat_completion_sync(
        self,
        messages: list[dict[str, str]],
        max_tokens: Optional[int] = None,
        enable_thinking: bool = True,
        tools: Optional[list] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> tuple[str, int, int]:
        all_images = _extract_images(messages)

        # Only run the vision encoder for images that aren't already in the session KV cache.
        # Images from earlier turns are still referenced in messages but their KV states
        # are already stored — re-running the vision encoder every turn is the bug.
        # Guard: if len(messages) <= _session_num_messages the history was trimmed/reset;
        # treat as a fresh turn so the stale cache isn't reused.
        if (self._session_kv is not None
                and self._session_num_messages > 0
                and len(messages) > self._session_num_messages):
            already_seen = _extract_images(messages[:self._session_num_messages])
            # New images if any were appended, OR the earlier images changed
            # (swap/removal) so the cached vision KV no longer matches.
            has_new_images = (len(all_images) > len(already_seen)
                              or not _images_extend(already_seen, all_images))
        else:
            has_new_images = bool(all_images)

        # Tokenize with the processor whenever images appear in the conversation so that
        # input_ids lengths are consistent with what the KV cache was built against.
        cached_ids = None
        if (not has_new_images and self._session_kv is not None
                and self._session_ids is not None and self._cached_msg_count > 0
                and len(messages) > self._cached_msg_count):
            # Only reuse the cache if the prior input messages are an unchanged
            # prefix of this turn — otherwise the spliced sequence would not match
            # the KV cache and produce corrupt output.
            prior = self._cached_msg_count - 1
            cache_seq_len = self._session_kv.get_seq_length()
            if (cache_seq_len < self._session_ids.shape[0]
                    and self._fingerprint(messages[:prior]) == self._cached_input_fp):
                cached_ids = self._session_ids[:cache_seq_len]

        input_ids, pixel_values, image_grid_thw, mm_token_type_ids = self._tokenize(
            messages, images=all_images or None, enable_thinking=enable_thinking, tools=tools,
            cached_input_ids=cached_ids,
        )

        if has_new_images:
            self._free_cache()
            _temperature = temperature if temperature is not None else self.temperature
            _top_p = top_p if top_p is not None else self.top_p

            cache = self.cache_factory() if self.cache_factory is not None else None

            vram_before = _vram_str()
            if torch.cuda.is_available():
                log.info("Vision prefill starting: %d tokens, int4 cache=%s, VRAM: %s",
                         input_ids.shape[1], cache is not None, vram_before)

            t0 = time.perf_counter()
            cache = self._chunked_vision_prefill(input_ids, cache, pixel_values, image_grid_thw, mm_token_type_ids)
            prefill_s = time.perf_counter() - t0

            if torch.cuda.is_available():
                log.info("Vision prefill done: %.2fs (%.1f tok/s), VRAM: %s", prefill_s,
                         input_ids.shape[1] / prefill_s if prefill_s > 0 else 0, _vram_str())

            generate_kwargs = {
                "past_key_values": cache,
                "max_new_tokens": max_tokens if max_tokens is not None else self.max_new_tokens,
                "temperature": _temperature,
                "top_p": _top_p,
                "top_k": self.top_k,
                "do_sample": _temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            if torch.cuda.is_available():
                log.info("Vision generate start: VRAM: %s", _vram_str())

            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    outputs = self.model.generate(input_ids, return_dict_in_generate=True, **generate_kwargs)
            except Exception:  # reclaim the freshly built vision cache on failure
                del cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
            elapsed = time.perf_counter() - t0

            peak = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

            n_input = input_ids.shape[1]
            generated_ids = outputs.sequences[0][n_input:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            n_tokens = len(generated_ids)
            decode_tps = n_tokens / (elapsed - prefill_s) if elapsed > prefill_s else 0
            log.info("Vision: %d tokens in %.2fs (prefill=%.2fs %.1f tok/s, decode=%.1f tok/s), VRAM before=%s after=%s peak_alloc=%.1f GB",
                     n_tokens, elapsed, prefill_s, input_ids.shape[1] / prefill_s if prefill_s > 0 else 0,
                     decode_tps, vram_before, _vram_str(), peak)
            # Save KV so follow-up text turns can extend it instead of full re-prefill.
            if outputs.past_key_values is not None:
                self._session_kv = outputs.past_key_values
                self._session_ids = outputs.sequences[0]
            self._session_num_messages = len(messages)
            self._cached_msg_count = len(messages) + 1
            self._cached_input_fp = self._fingerprint(messages)
            return text, n_input, n_tokens

        kwargs, generate_input = self._prepare_generation(input_ids, max_tokens, len(messages), temperature=temperature, top_p=top_p)

        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                outputs = self.model.generate(generate_input, return_dict_in_generate=True, **kwargs)
        except Exception:  # don't leave a half-built cache pinned in VRAM
            self._free_cache()
            raise
        elapsed = time.perf_counter() - t0

        n_input = input_ids.shape[1]
        generated_ids = outputs.sequences[0][n_input:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        n_tokens = len(generated_ids)
        decode_s = elapsed - self._last_prefill_s if elapsed > self._last_prefill_s else elapsed
        decode_tps = n_tokens / decode_s if decode_s > 0 else 0
        prefill_tps = n_input / self._last_prefill_s if self._last_prefill_s > 0 else 0
        if self._last_prefill_s > 0:
            log.info("Generated %d tokens in %.2fs (prefill: %.1f tok/s, decode: %.1f tok/s)",
                     n_tokens, elapsed, prefill_tps, decode_tps)
        else:
            log.info(f"Generated {n_tokens} tokens in {elapsed:.2f}s ({n_tokens / elapsed:.1f} tok/s)")

        if outputs.past_key_values is not None:
            self._session_kv = outputs.past_key_values
            self._session_ids = outputs.sequences[0]

        self._cached_msg_count = len(messages) + 1
        self._cached_input_fp = self._fingerprint(messages)
        return text, input_ids.shape[1], n_tokens

    def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: Optional[int] = None,
        enable_thinking: bool = True,
        tools: Optional[list] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Iterator[str]:
        from threading import Thread

        from transformers import TextIteratorStreamer

        all_images = _extract_images(messages)

        if (self._session_kv is not None
                and self._session_num_messages > 0
                and len(messages) > self._session_num_messages):
            already_seen = _extract_images(messages[:self._session_num_messages])
            has_new_images = (len(all_images) > len(already_seen)
                              or not _images_extend(already_seen, all_images))
        else:
            has_new_images = bool(all_images)

        cached_ids = None
        if (not has_new_images and self._session_kv is not None
                and self._session_ids is not None and self._cached_msg_count > 0
                and len(messages) > self._cached_msg_count):
            # Only reuse the cache if the prior input messages are an unchanged
            # prefix of this turn — otherwise the spliced sequence would not match
            # the KV cache and produce corrupt output.
            prior = self._cached_msg_count - 1
            cache_seq_len = self._session_kv.get_seq_length()
            if (cache_seq_len < self._session_ids.shape[0]
                    and self._fingerprint(messages[:prior]) == self._cached_input_fp):
                cached_ids = self._session_ids[:cache_seq_len]

        input_ids, pixel_values, image_grid_thw, mm_token_type_ids = self._tokenize(
            messages, images=all_images or None, enable_thinking=enable_thinking, tools=tools,
            cached_input_ids=cached_ids,
        )
        self._last_prompt_tokens = input_ids.shape[1]

        if has_new_images:
            self._free_cache()
            _temperature = temperature if temperature is not None else self.temperature
            _top_p = top_p if top_p is not None else self.top_p

            cache = self.cache_factory() if self.cache_factory is not None else None

            vram_before = _vram_str()
            if torch.cuda.is_available():
                log.info("Vision prefill starting (stream): %d tokens, int4 cache=%s, VRAM: %s",
                         input_ids.shape[1], cache is not None, vram_before)

            t0 = time.perf_counter()
            cache = self._chunked_vision_prefill(input_ids, cache, pixel_values, image_grid_thw, mm_token_type_ids)
            prefill_s = time.perf_counter() - t0

            if torch.cuda.is_available():
                log.info("Vision prefill done (stream): %.2fs (%.1f tok/s), VRAM: %s",
                         prefill_s, input_ids.shape[1] / prefill_s if prefill_s > 0 else 0, _vram_str())

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            generate_kwargs = {
                "max_new_tokens": max_tokens if max_tokens is not None else self.max_new_tokens,
                "temperature": _temperature,
                "top_p": _top_p,
                "top_k": self.top_k,
                "do_sample": _temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "past_key_values": cache,
                "streamer": streamer,
            }

            if torch.cuda.is_available():
                log.info("Vision generate start (stream): VRAM: %s", _vram_str())

            vision_out: dict = {}
            def _vision_generate():
                try:
                    vision_out["result"] = self.model.generate(
                        inputs=input_ids, return_dict_in_generate=True, **generate_kwargs
                    )
                except Exception as exc:
                    vision_out["error"] = exc
                finally:
                    streamer.end()

            thread = Thread(target=_vision_generate)
            thread.start()
            yield from streamer
            thread.join()

            if "error" in vision_out:
                self._free_cache()
                raise vision_out["error"]

            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / (1024**3)
                log.info("Vision stream done: VRAM before=%s after=%s peak_alloc=%.1f GB", vram_before, _vram_str(), peak)
            _out = vision_out.get("result")
            if _out is not None and _out.past_key_values is not None:
                self._session_kv = _out.past_key_values
                self._session_ids = _out.sequences[0]
            self._session_num_messages = len(messages)
            self._cached_msg_count = len(messages) + 1
            self._cached_input_fp = self._fingerprint(messages)
            return

        kwargs, generate_input = self._prepare_generation(input_ids, max_tokens, len(messages), temperature=temperature, top_p=top_p)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        kwargs["streamer"] = streamer

        output_ids = {}
        def _generate():
            try:
                output_ids["data"] = self.model.generate(**{**kwargs, "inputs": generate_input, "return_dict_in_generate": True})
            except Exception as exc:  # surface instead of dying silently → KeyError below
                output_ids["error"] = exc
            finally:
                # generate() only ends the streamer on success; end it here too so a
                # failed generation doesn't hang the consumer waiting for tokens.
                streamer.end()

        thread = Thread(target=_generate)
        thread.start()

        for text in streamer:
            yield text

        thread.join()

        if "error" in output_ids:
            self._free_cache()
            raise output_ids["error"]

        outputs = output_ids["data"]
        if outputs.past_key_values is not None:
            self._session_kv = outputs.past_key_values
            self._session_ids = outputs.sequences[0]
        self._cached_msg_count = len(messages) + 1
        self._cached_input_fp = self._fingerprint(messages)

    def _free_cache(self) -> None:
        if self._session_kv is None:
            return
        try:
            seq_len = self._session_kv.get_seq_length()
        except AttributeError:
            seq_len = 0
        del self._session_kv
        self._session_kv = None
        self._session_ids = None
        self._cached_msg_count = 0
        self._cached_input_fp = ()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Freed session KV cache (%d positions), VRAM: %s", seq_len, _vram_str())

    def reset_session(self) -> None:
        self._free_cache()
        self._session_num_messages = 0

    def get_vram_info(self) -> dict:
        if not torch.cuda.is_available():
            return {"cuda_available": False}

        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        try:
            import subprocess
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=2)
            parts = r.stdout.strip().split(",")
            nv_used = float(parts[0].strip()) / 1024
            nv_free = float(parts[1].strip()) / 1024
        except Exception:
            nv_used = reserved
            nv_free = total - reserved

        return {
            "cuda_available": True,
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(nv_free, 2),
            "used_gb": round(nv_used, 2),
        }
