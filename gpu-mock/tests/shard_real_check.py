#!/usr/bin/env python3
"""The verdict of tests/test_shard_real_in_image.sh, run inside the box: every served
score against the record's vector at its position (gpu_daemon._score_vector)."""
import json, sys, re
sys.path.insert(0, "/gm/src/daemon")
import gpu_daemon as gd
recs = {}
import glob
for p in glob.glob("/tmp/reg/mls-llm-kv-*.json"):
    r = json.load(open(p)); recs[r["identity"]["record_id"]] = r
ev = [json.loads(l) for l in open("/tmp/out/occupancy.jsonl")]
look = {e["job"]: e for e in ev if e.get("kind") == "lookup"}
bye = {e["job"]: e for e in ev if e.get("kind") == "job_bye"}
mat = {}
for e in ev:
    if e.get("kind") == "materialise" and e.get("role") == "example_score":
        mat.setdefault(e["job"], []).append(e)
miss = [e for e in ev if e.get("kind") == "example_order_miss"]
fails = 0; n = 0
def chk(name, cond, detail=""):
    global fails, n
    n += 1; fails += 0 if cond else 1
    print("[" + ("ok" if cond else "FAIL") + "] " + name + ((": " + detail) if detail else ""))
def job_for(rid_part, first):
    for j, l in look.items():
        if rid_part in (l.get("record_id") or "") and ((bye.get(j) or {}).get("positions") or {}).get("first") == first:
            return j
    return None
def case(tag, rid_part, first, n_ex):
    j = job_for(rid_part, first)
    rc = open(f"/tmp/out/{tag}.rc").read().strip()
    got = json.load(open(f"/tmp/out/{tag}.json")) if rc == "0" else []
    if j is None:
        chk(f"{tag}: a job served from a {rid_part} record with positions from {first}", False,
            str([(l.get("match"), l.get("record_id"), (bye.get(k) or {}).get("positions")) for k, l in look.items()]))
        return
    l, b = look[j], bye.get(j) or {}
    rec = recs[l["record_id"]]; vec = gd._score_vector(rec)
    pos = [e.get("position") if e.get("position") is not None else e.get("example") for e in mat.get(j, [])]
    served = [e.get("value") for e in mat.get(j, [])]
    m = re.search(r"final_score=([0-9.]+)", open(f"/tmp/out/{tag}.out").read())
    printed = float(m.group(1)) if m else None
    want = [vec[p] for p in range(first, first + n_ex)]
    chk(tag + ": rc 0, match " + str(l.get("match")) + ", record " + str(l.get("record_id")),
        rc == "0" and str(l.get("match")).endswith("module") and l.get("example_order", "").startswith("example-order-"), str(l.get("match")))
    chk(f"{tag}: positions {first}..{first + n_ex - 1}, the wrapper saw the record vector there, printed mean agrees",
        pos == list(range(first, first + n_ex)) and served == want and [s for _, s in got] == want
        and printed is not None and abs(printed - sum(want) / n_ex) < 1e-4,
        f"pos={pos} served={served} want={want} printed={printed}")
    chk(f"{tag}: no example_order_miss; job_bye names the module", not [e for e in miss if e.get("job") == j] and b.get("entry_module"), str(b.get("example_order_misses")))
case("Q0", "kivi-overlap-4bit-gsm8k", 100, 4)
case("Q1", "kivi-overlap-4bit-gsm8k", 1315, 4)
case("S0", "full-attention-gsm8k", 0, 3)
case("NH", "kivi-overlap-4bit-needlebench", 3, 3)
print(f"{n - fails} passed, {fails} failed")
sys.exit(fails)
