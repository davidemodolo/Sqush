# Shared helpers for the benchmarks: filler text and a chat request against a running llama-server.
import json, os, time, urllib.request


def _default_host():
    port = os.environ.get("PORT")
    config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.env")
    if port is None and os.path.exists(config):
        for line in open(config, encoding="utf-8"):
            if line.startswith("PORT="):
                port = line.strip().split("=", 1)[1]
    return os.environ.get("BENCH_HOST", f"http://127.0.0.1:{port or 8080}")


DEFAULT_HOST = _default_host()

PARAS = [
    "The architecture of large language models rests on a stack of engineering decisions, from tokenization strategy and positional encodings to the precise shape of the attention mechanism and the way normalization layers are placed between residual streams. Every one of these choices interacts with every other, and small differences in initialization, learning rate schedule, or the ordering of training data can compound into large differences in final behavior, which is why careful ablations remain the backbone of empirical research in this field.",
    "Quantization is the art of representing high-precision weights with far fewer bits while preserving as much of the original behavior as possible. Rounding errors, scale factors, and block sizes all matter, and the best schemes are those that adapt to the distribution of values in each tensor rather than applying a single global transform, since outlier channels can otherwise dominate the error budget and degrade generation quality in subtle ways.",
    "Attention mechanisms compute a weighted sum of values based on the similarity between a query and a set of keys, and the resulting complexity grows quadratically with sequence length, which is why long-context inference is dominated by the memory cost of the key-value cache rather than by the arithmetic itself. Quantizing that cache to lower precision is an effective way to fit hundreds of thousands of tokens into consumer hardware, at the price of a small quality penalty.",
    "Hybrid architectures interleave recurrent layers with ordinary attention layers so that the model gets the long-range efficiency of the former and the precise retrieval capabilities of the latter, and the ratio between the two is a design choice that trades memory against quality for different workloads.",
    "Memory bandwidth is frequently the limiting factor in token generation, since each generated token requires reading the full weight matrix, so quantization that shrinks weights without hurting accuracy translates almost directly into faster decoding on consumer graphics cards.",
]

# one repetition is ~360 tokens: 70 reps ~ 25k, 250 ~ 90k, 560 ~ 200k


def filler(reps):
    return ("\n\n".join(PARAS) + "\n\n") * reps


def chat(host, model, messages, max_tokens=24, cache_prompt=False):
    """Send a chat completion; returns (answer, response json, wall time in seconds)."""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "cache_prompt": cache_prompt,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()

    req = urllib.request.Request(host + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=7200))
    msg = r["choices"][0]["message"]
    answer = ((msg.get("content") or "") + (msg.get("reasoning_content") or "")).strip()
    return answer, r, time.time() - t0


def print_timings(r, wall):
    tm = r["timings"]
    print(f"prompt_tokens: {r['usage']['prompt_tokens']} (processed: {tm.get('prompt_n')})", flush=True)
    print(f"prefill: {tm['prompt_per_second']:.1f} t/s ({tm['prompt_ms']/1000:.1f}s)", flush=True)
    print(f"decode: {tm['predicted_per_second']:.1f} t/s", flush=True)
    print(f"MTP draft: {tm.get('draft_n_accepted')}/{tm.get('draft_n')} accepted", flush=True)
    print(f"total wall time: {wall:.1f}s", flush=True)
