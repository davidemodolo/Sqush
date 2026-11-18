#!/usr/bin/env python3
# Long-context needle test for the dynamic KV quantization stages.
# Places a unique key near the start of a long filler prompt and asks for it at the end.
#
#   python3 bench/needle_test.py --reps 418          # ~150k tokens
#   python3 bench/needle_test.py --reps 200 --query-bottom
import argparse, sys, uuid

from common import DEFAULT_HOST, chat, filler, print_timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--reps", type=int, default=418)
    ap.add_argument("--key", default="")
    ap.add_argument("--query-bottom", action="store_true", help="put the question before the filler instead of after")
    args = ap.parse_args()

    key = args.key or ("SECRET-" + uuid.uuid4().hex[:10].upper())

    needle = f"Remember this secret key: {key}. It will be asked at the end."
    question = "What is the secret key? Reply with the key only."
    text = needle + "\n\n" + filler(args.reps) + "\n\n" + ("" if args.query_bottom else question)

    answer, r, wall = chat(args.host, args.model, [{"role": "user", "content": text}])
    print_timings(r, wall)
    print(f"answer: {answer[:120]!r}", flush=True)

    ok = key in answer
    print("RESULT: " + ("PASS" if ok else "FAIL") + f" (key={key})", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
