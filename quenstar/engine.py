from __future__ import annotations

import logging
import threading
import time
from typing import Any, Iterator

from .config import QuenStarConfig
from .types import ChatCompletionRequest

_log = logging.getLogger(__name__)


class Engine:
    def __init__(self, config: QuenStarConfig):
        self.config = config
        self._llm = None
        self._chat_handler = None
        self._lock = threading.Lock()
        self._total_eval_tokens: int = 0
        self._load_model()

    def _load_model(self):
        import os

        import llama_cpp

        model_cfg = self.config.model

        if not os.path.isfile(model_cfg.path):
            _log.error("Model file not found: %s", model_cfg.path)
            raise FileNotFoundError(f"Model file not found: {model_cfg.path}")

        file_size = os.path.getsize(model_cfg.path)
        if file_size < 1024:
            _log.error("Model file is too small (%d bytes): %s", file_size, model_cfg.path)
            with open(model_cfg.path, "r", errors="replace") as f:
                snippet = f.read(200)
            _log.error("File contents: %s", snippet)
            _log.error("The file is not a valid GGUF model. Delete it and re-download.")
            raise RuntimeError(f"File is not a valid GGUF model (too small): {model_cfg.path}")

        with open(model_cfg.path, "rb") as f:
            magic = f.read(4)
        if magic != b"GGUF":
            _log.error("File is not a valid GGUF model (magic bytes: %r): %s", magic, model_cfg.path)
            _log.error("Expected 'GGUF', got %r. Delete this file and re-download.", magic.decode(errors="replace"))
            raise RuntimeError(f"Not a valid GGUF model: {model_cfg.path}")

        if model_cfg.n_gpu_layers != 0 and not llama_cpp.llama_supports_gpu_offload():
            _log.warning(
                "CUDA GPU offload not available in this llama-cpp-python build. "
                "GPU layers requested but only CPU will be used. "
                "Reinstall with: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python --force-reinstall"
            )

        _log.info(
            "Loading model %s (n_ctx=%d, n_gpu_layers=%d, offload_kqv=%s)",
            model_cfg.path,
            model_cfg.n_ctx,
            model_cfg.n_gpu_layers,
            model_cfg.offload_kqv,
        )
        try:
            self._llm = llama_cpp.Llama(
                model_path=model_cfg.path,
                n_gpu_layers=model_cfg.n_gpu_layers,
                n_ctx=model_cfg.n_ctx,
                n_batch=model_cfg.n_batch,
                n_ubatch=model_cfg.n_ubatch,
                offload_kqv=model_cfg.offload_kqv,
                flash_attn=model_cfg.flash_attn,
                use_mmap=model_cfg.use_mmap,
                use_mlock=False,
                verbose=False,
                seed=-1,
                type_k=model_cfg.type_k,
                type_v=model_cfg.type_v,
                yarn_ext_factor=model_cfg.yarn_ext_factor,
                yarn_attn_factor=model_cfg.yarn_attn_factor,
                yarn_beta_fast=model_cfg.yarn_beta_fast,
                yarn_beta_slow=model_cfg.yarn_beta_slow,
                chat_handler=self._chat_handler,
            )
        except ValueError as exc:
            _log.error("Failed to load model: %s", exc)
            if "ggml" in str(exc).lower() or "unsupported" in str(exc).lower():
                _log.error(
                    "This model may require a newer version of llama.cpp. "
                    "Try: pip install --upgrade llama-cpp-python"
                )
            raise

        _log.info(
            "Model loaded. VRAM: model ~%.1f GB, KV cache in %s (context: %d tokens)",
            self._estimate_model_size_gb(),
            "system RAM" if not model_cfg.offload_kqv else "GPU",
            model_cfg.n_ctx,
        )

        if model_cfg.mmproj_path and os.path.isfile(model_cfg.mmproj_path):
            self._load_mmproj(model_cfg.mmproj_path)

    def _load_mmproj(self, mmproj_path: str):
        from llama_cpp.llama_chat_format import Llava15ChatHandler

        try:
            self._chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path, verbose=False)
            self._llm.chat_handler = self._chat_handler
            _log.info("Vision encoder loaded: %s", mmproj_path)
        except Exception as exc:
            _log.warning("Failed to load vision encoder: %s", exc)

    def _estimate_model_size_gb(self) -> float:
        try:
            return self._llm.model_size() / (1024**3)
        except Exception:
            try:
                n = self._llm.n_params()
                return n * 2 / (1024**3)
            except Exception:
                import os
                return os.path.getsize(self.config.model.path) / (1024**3)

    @property
    def n_ctx(self) -> int:
        return self.config.model.n_ctx

    @property
    def n_tokens(self) -> int:
        if self._llm is None:
            return 0
        try:
            return len(self._llm.input_ids) if self._llm.input_ids else 0
        except Exception:
            return self._total_eval_tokens

    @property
    def model_path(self) -> str:
        return self.config.model.path

    @property
    def model_id(self) -> str:
        import hashlib

        return hashlib.sha1(self.model_path.encode()).hexdigest()[:16]

    def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> Iterator[dict[str, Any]]:
        with self._lock:
            yield from self._do_chat_completion(request)

    def manual_stream(
        self,
        request: ChatCompletionRequest,
    ) -> Iterator[dict[str, Any]]:
        with self._lock:
            yield from self._do_manual_stream(request)

    def _resolve_sampling_params(self, request: ChatCompletionRequest):
        s = self.config.sampling
        return (
            request.temperature if request.temperature is not None else s.default_temperature,
            request.top_p if request.top_p is not None else s.default_top_p,
            request.top_k if request.top_k is not None else s.default_top_k,
            s.default_min_p,
        )

    def _do_chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> Iterator[dict[str, Any]]:
        gen_cfg = self.config.generation

        if not self.config.tool_calling.enabled:
            has_tools = False
            tool_choice = None
        else:
            has_tools = bool(request.tools)
            tool_choice = request.tool_choice

        force_greedy = has_tools and tool_choice in ("required", "auto")
        logits_processor = None

        if force_greedy and self.config.tool_calling.manual_token_loop:
            yield from self._do_manual_stream(request)
            return

        if force_greedy and self.config.tool_calling.greedy_tool_syntax:
            # DS4-style: greedy ONLY on tool-call syntax via a per-token logits
            # processor; sample argument payloads + content at payload_temperature
            # (opencode pins temperature=0, so fall back to it for any sampling).
            if request.temperature is not None and request.temperature > 0:
                temperature = request.temperature
            else:
                temperature = self.config.tool_calling.payload_temperature
            _, top_p, top_k, min_p = self._resolve_sampling_params(request)
            try:
                from llama_cpp import LogitsProcessorList

                from .toolcall import ToolSyntaxGreedyProcessor

                logits_processor = LogitsProcessorList([ToolSyntaxGreedyProcessor(self._llm)])
                _log.debug("Tool-calling: greedy on syntax, payloads at temp=%.3f", temperature)
            except Exception as exc:
                _log.warning("greedy-syntax processor unavailable (%s); whole-gen greedy", exc)
                tc = self.config.sampling
                temperature, top_p, top_k, min_p = (
                    tc.tool_call_temperature, tc.tool_call_top_p,
                    tc.tool_call_top_k, tc.default_min_p,
                )
        elif force_greedy:
            tc = self.config.sampling
            temperature = tc.tool_call_temperature
            top_p = tc.tool_call_top_p
            top_k = tc.tool_call_top_k
            min_p = self.config.sampling.default_min_p
            _log.debug("Tool-calling mode: temperature=%.1f for deterministic syntax", temperature)
        else:
            temperature, top_p, top_k, min_p = self._resolve_sampling_params(request)

        if temperature > 0 and request.seed is not None:
            seed = request.seed
        elif temperature <= 0:
            seed = 0
        else:
            seed = int(time.time() * 1000) % (2**31)

        max_tokens = request.max_tokens if request.max_tokens is not None else gen_cfg.max_tokens
        stop = request.stop if request.stop else gen_cfg.stop_strings or None

        chat_kwargs: dict[str, Any] = {}
        if request.enable_thinking is not None:
            chat_kwargs["enable_thinking"] = request.enable_thinking

        _log.debug(
            "chat_completion: stream=%s temp=%.3f top_p=%.3f top_k=%d max_tokens=%d seed=%d tools=%d",
            request.stream,
            temperature,
            top_p,
            top_k,
            max_tokens,
            seed,
            len(request.tools) if request.tools else 0,
        )
        _log.info("inference start: %d tokens in context", self._llm.n_tokens)

        try:
            result = self._llm.create_chat_completion(
                messages=request.messages,
                tools=request.tools,
                tool_choice=tool_choice,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_tokens,
                stop=stop if stop else None,
                stream=request.stream,
                seed=seed,
                repeat_penalty=self.config.sampling.default_repeat_penalty,
                logits_processor=logits_processor,
                **chat_kwargs,
            )
        except Exception as exc:
            _log.error("Inference failed: %s", exc)
            raise

        if request.stream:
            for chunk in result:
                yield chunk
        else:
            yield result

        try:
            self._total_eval_tokens = self._llm.n_tokens
        except Exception:
            pass

        _log.info("inference done: %d tokens in context", self._total_eval_tokens)

    def save_state(self) -> bytes:
        import pickle

        state = self._llm.save_state()
        return pickle.dumps({
            "llama_state": state.llama_state,
            "input_ids": list(state.input_ids) if state.input_ids is not None else [],
            "scores": list(state.scores) if state.scores is not None else [],
            "n_tokens": state.n_tokens,
        })

    def load_state(self, raw_state: bytes):
        import pickle
        from llama_cpp import LlamaState

        data = pickle.loads(raw_state)
        state = LlamaState(
            input_ids=data["input_ids"],
            scores=data["scores"],
            n_tokens=data["n_tokens"],
            llama_state=data["llama_state"],
        )
        self._llm.load_state(state)
        self._total_eval_tokens = self._llm.n_tokens

    def reset_context(self):
        self._llm.reset()
        self._total_eval_tokens = 0

    def _do_manual_stream(
        self,
        request: ChatCompletionRequest,
    ) -> Iterator[dict[str, Any]]:
        from .toolcall import ToolCallDetector

        gen_cfg = self.config.generation

        temperature, top_p, top_k, min_p = self._resolve_sampling_params(request)

        max_tokens = request.max_tokens if request.max_tokens is not None else gen_cfg.max_tokens

        chat_kwargs: dict[str, Any] = {}
        if request.enable_thinking is not None:
            chat_kwargs["enable_thinking"] = request.enable_thinking

        _log.debug("Manual stream: per-token greedy inside tool-call syntax, max_tokens=%d", max_tokens)

        if temperature > 0 and request.seed is not None:
            seed = request.seed
        elif temperature <= 0:
            seed = 0
        else:
            seed = int(time.time() * 1000) % (2**31)

        result = self._llm.create_chat_completion(
            messages=request.messages,
            tools=request.tools,
            tool_choice=request.tool_choice,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=1,
            stream=False,
            seed=seed,
            repeat_penalty=self.config.sampling.default_repeat_penalty,
            **chat_kwargs,
        )

        first_content = ""
        if "choices" in result and result["choices"]:
            first_content = result["choices"][0].get("message", {}).get("content", "")

        detector = ToolCallDetector()
        if first_content:
            detector.feed(first_content)
            yield {
                "choices": [{
                    "index": 0,
                    "delta": {"content": first_content},
                    "finish_reason": None,
                }]
            }

        try:
            eos_token = self._llm.token_eos()
        except Exception:
            eos_token = -1

        remaining = max_tokens - 1 if max_tokens > 1 else 0
        for _ in range(remaining):
            try:
                if detector.is_in_tool_call():
                    next_token = self._llm.sample(temp=0.0, top_p=1.0, top_k=1)
                else:
                    next_token = self._llm.sample(temp=temperature, top_p=top_p, top_k=top_k)
            except Exception as exc:
                _log.error("Sampling failed: %s", exc)
                break

            if next_token == eos_token:
                yield {
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }]
                }
                break

            try:
                text = self._llm.detokenize([next_token])
                if isinstance(text, bytes):
                    text = text.decode("utf-8", errors="replace")
            except Exception:
                text = ""

            detector.feed(text)

            try:
                self._llm.eval([next_token])
            except Exception as exc:
                _log.error("Eval failed: %s", exc)
                break

            yield {
                "choices": [{
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": None,
                }]
            }

        try:
            self._total_eval_tokens = self._llm.n_tokens
        except Exception:
            pass

        _log.info("manual stream done: %d tokens in context", self._total_eval_tokens)
