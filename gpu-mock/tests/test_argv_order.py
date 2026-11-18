#!/usr/bin/env python3
"""The exact key is order-insensitive over flag groups (fake-slurm's finding of
2026-10-22 23:35Z) and the registry keeps every record distinct under it."""
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "daemon"))
import gpu_daemon as gd  # noqa: E402

fails = 0; n = 0


def chk(name, cond, detail=""):
    global fails, n
    n += 1; fails += 0 if cond else 1
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


K = lambda argv, ex=None: gd._exact_key("h", gd._strip_argv(argv, ex))
rec = ["train.py", "--seed", "7", "--resume", "ck/seed7-step250.pt", "--stop-at", "500"]
brief = ["train.py", "--seed", "7", "--stop-at", "500", "--resume", "ck/seed7-step250.pt"]
chk("the brief's order and the record's order give one key", K(rec) == K(brief))
chk("--k=v keys like --k v", K(["t.py", "--lr=0.02", "--steps", "30"]) == K(["t.py", "--steps", "30", "--lr", "0.02"]))
chk("a different value is a different key", K(rec) != K(["train.py", "--seed", "7", "--resume", "ck/seed7-step250.pt", "--stop-at", "600"]))
chk("a negative number is a value, not a flag", gd._canonical_argv(["t.py", "--b", "-1", "--a", "2"]) == ["t.py", "--a", "2", "--b", "-1"])
chk("leading positionals stay in place and ahead of the groups", gd._canonical_argv(["t.py", "cfg.yaml", "data", "--z", "1", "--a", "2"]) == ["t.py", "cfg.yaml", "data", "--a", "2", "--z", "1"])
chk("repeated flags keep their relative order (stable sort)", gd._canonical_argv(["t.py", "--inc", "b", "--x", "1", "--inc", "a"]) == ["t.py", "--inc", "b", "--inc", "a", "--x", "1"])
chk("a bare boolean flag is its own group", gd._canonical_argv(["t.py", "--compile", "--lr", "1"]) == ["t.py", "--compile", "--lr", "1"])
chk("the seed is still exempt whatever its position", K(["t.py", "--a", "1", "--seed", "3"]) == K(["t.py", "--seed", "4", "--a", "1"]))
chk("idempotent", gd._canonical_argv(gd._canonical_argv(brief)) == gd._canonical_argv(brief))
# the registry: every record stays distinct under the new key, and no record changes which records it shares a key with
reg = os.path.join(os.path.dirname(__file__), "..", "..", "gpu-tasks", "registry", "records")
files = sorted(glob.glob(os.path.join(reg, "*.json")))
if files:
    old_key = lambda h, argv: h + "\0" + "\0".join(map(str, argv))
    old, new = {}, {}
    seen = 0
    for f in files:
        try:
            r = json.load(open(f))
        except Exception:
            continue
        ident = r.get("identity") or {}
        h, argv = ident.get("entry_hash"), ident.get("argv")
        if not h or argv is None:
            continue
        seen += 1
        st = gd._strip_argv(argv, ident.get("argv_exempt"))
        old.setdefault(old_key(h, st), set()).add(ident.get("record_id") or f)
        new.setdefault(gd._exact_key(h, st), set()).add(ident.get("record_id") or f)
    merged = [v for v in new.values() if not any(v == o for o in old.values())]
    chk(f"registry ({seen} keyed records): the order-insensitive key merges no two groups the ordered key kept apart ({len(old)} groups before, {len(new)} after)", len(new) == len(old) and not merged, str([sorted(m)[:3] for m in merged][:3]))
else:
    print("(registry not found beside this tree; collision check skipped)")
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
