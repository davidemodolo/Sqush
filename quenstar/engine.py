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
        self._lock = threading.Lock()
        self._total_eval_tokens: int = 0
        self._load_model()

    def _load_model(self):
        import os
        import sys

        import llama_cpp

        model_cfg = self.config.model

        if not os.path.isfile(model_cfg.path):
            _log.error("Model file not found: %s", model_cfg.path)
            sys.exit(1)

        file_size = os.path.getsize(model_cfg.path)
        if file_size < 1024:
            _log.error("Model file is too small (%d bytes): %s", file_size, model_cfg.path)
            with open(model_cfg.path, "r", errors="replace") as f:
                snippet = f.read(200)
            _log.error("File contents: %s", snippet)
            _log.error("The file is not a valid GGUF model. Delete it and re-download.")
            sys.exit(1)

        with open(model_cfg.path, "rb") as f:
            magic = f.read(4)
        if magic != b"GGUF":
            _log.error("File is not a valid GGUF model (magic bytes: %r): %s", magic, model_cfg.path)
            _log.error("Expected 'GGUF', got %r. Delete this file and re-download.", magic.decode(errors="replace"))
            sys.exit(1)

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
                offload_kqv=model_cfg.offload_kqv,
                flash_attn=model_cfg.flash_attn,
                use_mmap=model_cfg.use_mmap,
                use_mlock=False,
                verbose=False,
                seed=-1,
                n_batch=4096,
            )
        except ValueError as exc:
            _log.error("Failed to load model: %s", exc)
            if "ggml" in str(exc).lower() or "unsupported" in str(exc).lower():
                _log.error(
                    "This model may require a newer version of llama.cpp. "
                    "Try: pip install --upgrade llama-cpp-python"
                )
            sys.exit(1)

        _log.info(
            "Model loaded. VRAM: model ~%.1f GB, KV cache in %s (context: %d tokens)",
            self._estimate_model_size_gb(),
            "system RAM" if not model_cfg.offload_kqv else "GPU",
            model_cfg.n_ctx,
        )

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

    def _do_chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> Iterator[dict[str, Any]]:
        sampling = self.config.sampling
        gen_cfg = self.config.generation

        has_tools = bool(request.tools)
        tool_choice = request.tool_choice

        force_greedy = has_tools and (
            tool_choice == "required"
            or tool_choice == "auto"
            or (tool_choice is None and has_tools)
        )

        if force_greedy:
            temperature = 0.0
            top_p = 1.0
            top_k = 1
            min_p = 0.0
            _log.debug("Tool-calling mode: forcing temperature=0 for deterministic syntax")
        else:
            temperature = request.temperature if request.temperature is not None else sampling.default_temperature
            top_p = request.top_p if request.top_p is not None else sampling.default_top_p
            top_k = request.top_k if request.top_k is not None else sampling.default_top_k
            min_p = sampling.default_min_p

        if temperature > 0 and request.seed is not None:
            seed = request.seed
        elif temperature <= 0:
            seed = 0
        else:
            seed = int(time.time() * 1000) % (2**31)

        max_tokens = request.max_tokens if request.max_tokens is not None else gen_cfg.max_tokens
        stop = request.stop if request.stop else None

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
                repeat_penalty=sampling.default_repeat_penalty,
            )
        except Exception as exc:
            _log.error("Inference failed: %s", exc)
            raise

        if request.stream:
            for chunk in result:
                yield chunk
        else:
            yield result

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

        sampling = self.config.sampling
        gen_cfg = self.config.generation

        has_tools = bool(request.tools)
        tool_choice = request.tool_choice
        force_greedy = has_tools and (
            tool_choice == "required"
            or tool_choice == "auto"
            or (tool_choice is None and has_tools)
        )

        temperature = request.temperature if request.temperature is not None else sampling.default_temperature
        top_p = request.top_p if request.top_p is not None else sampling.default_top_p
        top_k = request.top_k if request.top_k is not None else sampling.default_top_k
        min_p = sampling.default_min_p

        if temperature > 0 and request.seed is not None:
            seed = request.seed
        elif temperature <= 0:
            seed = 0
        else:
            seed = int(time.time() * 1000) % (2**31)

        max_tokens = request.max_tokens if request.max_tokens is not None else gen_cfg.max_tokens
        stop = request.stop if request.stop else None

        if force_greedy:
            _log.debug("Tool-calling: using manual loop with per-token temp=0 for syntax")

        try:
            result = self._llm.create_chat_completion(
                messages=request.messages,
                tools=request.tools,
                tool_choice=tool_choice,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=1,
                stop=stop if stop else None,
                stream=False,
                seed=seed,
                repeat_penalty=sampling.default_repeat_penalty,
            )
        except Exception as exc:
            _log.error("Manual stream prefill failed: %s", exc)
            raise

        first_content = ""
        if "choices" in result and result["choices"]:
            msg = result["choices"][0].get("message", {})
            first_content = msg.get("content", "")

        if first_content:
            yield result

        detector = ToolCallDetector()
        if first_content:
            detector.feed(first_content)

        try:
            eos_token = self._llm.token_eos()
        except Exception:
            eos_token = -1

        for _ in range(max_tokens - 1 if max_tokens > 1 else max_tokens):
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

            chunk: dict[str, Any] = {
                "choices": [{
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": None,
                }]
            }
            yield chunk
