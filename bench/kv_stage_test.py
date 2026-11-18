#!/usr/bin/env python3
# Sends a long prompt to a running llama-server and reports prefill/decode timings.
# Used to exercise the dynamic KV quantization stage transitions.
#
#   python3 bench/kv_stage_test.py --reps 70 --max-tokens 32
import argparse, uuid

from common import DEFAULT_HOST, chat, filler, print_timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--reps", type=int, default=70, help="number of paragraph block repetitions")
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    uid = "STAGE-TEST-" + str(uuid.uuid4())[:8]
    text = uid + "\n\n" + filler(args.reps) + "\n\nRepeat the identifier at the start of this message verbatim."

    answer, r, wall = chat(args.host, args.model, [{"role": "user", "content": text}], args.max_tokens)
    print_timings(r, wall)
    print(f"answer: {answer[:120]!r}", flush=True)
    if uid not in answer:
        print("WARNING: the prompt identifier was not repeated, check response coherence", flush=True)


if __name__ == "__main__":
    main()
