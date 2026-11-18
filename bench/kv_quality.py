#!/usr/bin/env python3
# Measures how far a KV cache configuration moves the predictions from an f16 reference (KL divergence).
# A small CTX makes every stage convert within the 27k-token document; SPEC_TYPE=none is required.
#
#   KV_TYPE=f16 CTX=32768 SPEC_TYPE=none ./serve.sh --kv-static     # reference
#   python3 bench/kv_quality.py collect bench/f16.json
#   CTX=32768 SPEC_TYPE=none KV_MARGIN=512 ./serve.sh               # dynamic stages
#   python3 bench/kv_quality.py collect bench/dynamic.json
#   python3 bench/kv_quality.py compare bench/f16.json bench/dynamic.json ...
import argparse, glob, json, math, os, sys, urllib.request

from common import DEFAULT_HOST

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llama.cpp")
DOC_FILE = os.path.join(ROOT, "tools", "server", "README.md")
SNIPPET_GLOB = os.path.join(ROOT, "docs", "*.md")


def post(host, path, payload):
    req = urllib.request.Request(host + path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def tokenize(host, text):
    return post(host, "/tokenize", {"content": text})["tokens"]


def collect(args):
    if not os.path.isfile(DOC_FILE):
        sys.exit(f"missing {DOC_FILE}, run ./build.sh first (the texts come from the llama.cpp checkout)")

    doc = tokenize(args.host, open(DOC_FILE, encoding="utf-8").read())
    while len(doc) < args.tokens:
        doc = doc + doc
    doc = doc[:args.tokens]

    snippets = []
    for path in sorted(glob.glob(SNIPPET_GLOB)):
        toks = tokenize(args.host, open(path, encoding="utf-8").read())
        for i in range(0, len(toks) - args.snippet_tokens, args.snippet_tokens * 4):
            snippets.append(toks[i:i + args.snippet_tokens])
    step = max(1, len(snippets) // args.snippets)
    snippets = snippets[::step][:args.snippets]

    runs = []
    for i, snip in enumerate(snippets):
        r = post(args.host, "/completion", {
            "prompt": doc + snip,
            "n_predict": args.predict,
            "n_probs": args.probs,
            "temperature": -1,
            "cache_prompt": True,
        })
        steps = []
        for p in r["completion_probabilities"]:
            steps.append({"id": p["id"], "top": [[t["id"], t["logprob"]] for t in p["top_logprobs"]]})
        runs.append(steps)
        print(f"\r{i + 1}/{len(snippets)}", end="", flush=True)
    print()

    json.dump({"tokens": args.tokens, "runs": runs}, open(args.out, "w"))
    print(f"saved {args.out} ({len(doc)} document tokens, {len(runs)} snippets)")


def kl(ref_top, top):
    # KL(ref || other) over the reference's top tokens; a token missing from the other list
    # gets that list's lowest probability
    other = dict((t, lp) for t, lp in top)
    floor = min(other.values())
    return sum(math.exp(lp) * (lp - other.get(t, floor)) for t, lp in ref_top)


def compare(args):
    ref = json.load(open(args.ref))
    print(f"reference: {args.ref}")
    print(f"{'run':28} {'KL mean':>9} {'KL p90':>9} {'top-1 same':>11} {'positions':>10}")
    for path in args.runs:
        cur = json.load(open(path))
        if cur["tokens"] != ref["tokens"] or len(cur["runs"]) != len(ref["runs"]):
            print(f"{path:28} collected with different settings, skipped")
            continue
        kls, same = [], 0
        for a, b in zip(ref["runs"], cur["runs"]):
            # the continuations are only comparable while both runs produced the same tokens
            for sa, sb in zip(a, b):
                kls.append(kl(sa["top"], sb["top"]))
                if sa["id"] != sb["id"]:
                    break
                same += 1
        if not kls:
            print(f"{path:28} no positions to compare, skipped")
            continue
        kls.sort()
        print(f"{os.path.basename(path):28} {sum(kls) / len(kls):9.4f} {kls[int(len(kls) * 0.9)]:9.4f} "
              f"{100.0 * same / len(kls):10.1f}% {len(kls):10d}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="query a running server and save the token probabilities")
    c.add_argument("out")
    c.add_argument("--host", default=DEFAULT_HOST)
    c.add_argument("--tokens", type=int, default=27000, help="document length in tokens")
    c.add_argument("--snippets", type=int, default=40)
    c.add_argument("--snippet-tokens", type=int, default=96)
    c.add_argument("--predict", type=int, default=24, help="tokens generated per snippet")
    c.add_argument("--probs", type=int, default=40, help="top probabilities kept per token")
    c.set_defaults(fn=collect)

    c = sub.add_parser("compare", help="compare saved runs against a reference run")
    c.add_argument("ref")
    c.add_argument("runs", nargs="+")
    c.set_defaults(fn=compare)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
