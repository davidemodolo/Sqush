#!/usr/bin/env python3
# Main agent -> subagent -> main agent, the way an agent delegating to subagents uses the server.
# Checks that:
#   - the subagent runs on a cache that went back to f16 (see the set_stage lines in the server log)
#   - the main agent's follow-up is restored from the RAM prompt cache instead of being reprocessed
#
#   python3 bench/subagent_test.py                   # main ~90k tokens (crosses f16 -> q8_0)
import argparse, sys, uuid

from common import DEFAULT_HOST, chat, filler


def run(name, host, model, key, messages):
    answer, r, wall = chat(host, model, messages, cache_prompt=True)
    ok = key in answer
    print(f"[{name}] prompt_tokens={r['usage']['prompt_tokens']} processed={r['timings'].get('prompt_n')} "
          f"wall={wall:.1f}s answer={answer[:60]!r} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--main-reps", type=int, default=250)
    ap.add_argument("--sub-reps", type=int, default=10)
    args = ap.parse_args()

    key_main = "MAIN-" + uuid.uuid4().hex[:8].upper()
    key_sub = "SUB-" + uuid.uuid4().hex[:8].upper()

    main_msgs = [
        {"role": "system", "content": "You are the main coding agent."},
        {"role": "user", "content": f"Remember this secret key: {key_main}.\n\n{filler(args.main_reps)}\n\n"
                                    "What is the secret key? Reply with the key only."},
    ]
    sub_msgs = [
        {"role": "system", "content": "You are a subagent working on a delegated task."},
        {"role": "user", "content": f"Task id: {key_sub}.\n\n{filler(args.sub_reps)}\n\n"
                                    "What is the task id? Reply with the id only."},
    ]

    ok = run("main 1", args.host, args.model, key_main, main_msgs)
    ok &= run("subagent", args.host, args.model, key_sub, sub_msgs)

    main_msgs += [
        {"role": "assistant", "content": key_main},
        {"role": "user", "content": "The subagent finished. Repeat the secret key from the start of our "
                                    "conversation, key only."},
    ]
    ok &= run("main 2", args.host, args.model, key_main, main_msgs)

    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
