"""Summarise what each Claude Code variant sent to the stub and how its turn ended."""
import json, os, sys

D = sys.argv[1]
for v in ("default", "updates_0", "updates_false", "flag_omitted", "summarized_0", "summarized"):
    p = os.path.join(D, v + ".jsonl")
    if not os.path.exists(p):
        print(f"== {v}: no stub log")
        continue
    reqs = [json.loads(l) for l in open(p) if l.strip()]
    msgs = [r for r in reqs if r["path"].startswith("/v1/messages") and "count_tokens" not in r["path"]]
    print(f"== {v}: {len(reqs)} requests, {len(msgs)} to /v1/messages")
    for r in msgs:
        th = r.get("thinking") or {}
        print(f"   stream={r.get('stream')} thinking={th.get('type')} display={th.get('display')!r} "
              f"updates_beta={'thinking-display-updates' in (r.get('beta') or '')} len={r['len']} -> {r['outcome']}")
    for r in reqs:
        if r not in msgs:
            print(f"   other: {r['method']} {r['path']} -> {r['outcome']}")
    res = None
    sp = os.path.join(D, v + ".stream")
    if os.path.exists(sp):
        for line in open(sp):
            try:
                j = json.loads(line)
            except ValueError:
                continue
            if j.get("type") == "result":
                res = j
    print("   result:", None if res is None else {k: res.get(k) for k in ("subtype", "is_error", "num_turns", "result")})
    # G7: did any assistant message in the stream carry thinking text?
    think_chars = 0; think_blocks = 0
    if os.path.exists(sp):
        for line in open(sp):
            try:
                j = json.loads(line)
            except ValueError:
                continue
            for c in ((j.get("message") or {}).get("content") or []) if j.get("type") == "assistant" else []:
                if isinstance(c, dict) and c.get("type") == "thinking":
                    think_blocks += 1; think_chars += len(c.get("thinking") or "")
    print(f"   thinking in stream: {think_blocks} block(s), {think_chars} chars")
print(open(os.path.join(D, "meta.txt")).read())
