from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional


class KVSaveReason(IntEnum):
    UNKNOWN = 0
    COLD = 1
    CONTINUED = 2
    EVICT = 3
    SHUTDOWN = 4


@dataclass
class KVCacheHeader:
    magic: bytes = b"QSTK"
    version: int = 1
    quant_bits: int = 4
    save_reason: int = KVSaveReason.UNKNOWN
    flags: int = 0
    n_tokens: int = 0
    hit_count: int = 0
    context_size: int = 0
    created_at: float = 0.0
    last_used_at: float = 0.0
    payload_bytes: int = 0

    HEADER_SIZE = 64

    def pack(self) -> bytes:
        import struct

        return struct.pack(
            "<4sIBBBxxQIIxxddQ",
            self.magic,
            self.version,
            self.quant_bits,
            self.save_reason,
            self.flags,
            self.n_tokens,
            self.hit_count,
            self.context_size,
            self.created_at,
            self.last_used_at,
            self.payload_bytes,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "KVCacheHeader":
        import struct

        (
            magic,
            version,
            quant_bits,
            save_reason,
            flags,
            n_tokens,
            hit_count,
            context_size,
            created_at,
            last_used_at,
            payload_bytes,
        ) = struct.unpack("<4sIBBBxxQIIxxddQ", data[: cls.HEADER_SIZE])
        return cls(
            magic=magic,
            version=version,
            quant_bits=quant_bits,
            save_reason=save_reason,
            flags=flags,
            n_tokens=n_tokens,
            hit_count=hit_count,
            context_size=context_size,
            created_at=created_at,
            last_used_at=last_used_at,
            payload_bytes=payload_bytes,
        )


@dataclass
class Session:
    id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    n_tokens: int = 0
    created_at: float = 0.0
    last_used_at: float = 0.0


@dataclass
class ToolCall:
    id: str
    function_name: str
    arguments: str
    raw_dsml: Optional[str] = None


@dataclass
class ChatCompletionRequest:
    model: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None
    stream: bool = False
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Any = None
    stop: Optional[list[str]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatCompletionRequest":
        return cls(
            model=data.get("model", ""),
            messages=data.get("messages", []),
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            top_k=data.get("top_k"),
            seed=data.get("seed"),
            stream=data.get("stream", False),
            tools=data.get("tools"),
            tool_choice=data.get("tool_choice"),
            stop=data.get("stop"),
        )
