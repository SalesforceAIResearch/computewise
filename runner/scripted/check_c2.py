"""C2 check: one tagged model call from inside the box on GCP (the study, 2026-10-20, STATE.md
"Checkpoints"). Run on the controller after the C2 run has ended, because the ledger is here.

usage, on the controller:  python3 -I check_c2.py <tag> [--run-dir /srv/runs/<tag>] [--gate-log ~/.p006-gate/gate.jsonl]

The four things C2 proves, each printed PASS or FAIL with its evidence:
  1. the model call is in the model router's ledger under the run's tag and session id
     (ledger turns with tag = <tag>; their session equals run.json's session_id);
  2. the forwarder logged its route time: the box's edge log (forwarder/edge.jsonl on the VM) has
     a request with rid and timings, the controller gate log has the same rid, and the hop
     (edge total_ms - gate total_ms) is computed;
  3. the transcript was copied out: transcript/stream.jsonl and the session tree are non-empty;
  4. the run directory is complete: every file isolation's contract lists for a claude-code run.
The model's answer does not matter. The ledger is opened read-only.
"""
import argparse, json, os, sqlite3, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
VM = os.path.normpath(os.path.join(HERE, "..", "..", "gcp", "host", "vm.sh"))
REQUIRED = ["status.json", "run.json", "brief.txt", "task.json", "occupancy.jsonl", "resources.jsonl",
            "transcript/stream.jsonl", "transcript/stderr.log", "forwarder/edge.jsonl", "workspace.lst"]

def vm(cmd):
    p = subprocess.run(["bash", VM, "ssh", cmd], capture_output=True, text=True, timeout=180)
    return p.returncode, p.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--gate-log", default=os.path.expanduser("~/.p006-gate/gate.jsonl"))
    ap.add_argument("--ledger", default=os.environ.get("CW_LEDGER"))
    a = ap.parse_args()
    rd = a.run_dir or f"/srv/runs/{a.tag}"
    results = []
    def report(n, ok, ev):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {n}: {ev}")

    # pull what is needed from the VM in one call
    script = ("import json,os,sys;d=sys.argv[1];o={};"
              "o['files']={f:(os.path.getsize(os.path.join(d,f)) if os.path.exists(os.path.join(d,f)) else None) for f in sys.argv[2:]};"
              "o['run']=json.load(open(os.path.join(d,'run.json'))) if os.path.exists(os.path.join(d,'run.json')) else {};"
              "o['status']=json.load(open(os.path.join(d,'status.json'))) if os.path.exists(os.path.join(d,'status.json')) else {};"
              "e=os.path.join(d,'forwarder','edge.jsonl');o['edge']=[json.loads(l) for l in open(e)] if os.path.exists(e) else [];"
              "s=os.path.join(d,'transcript','session');o['session_files']=sum(len(f) for _,_,f in os.walk(s)) if os.path.isdir(s) else 0;"
              "print(json.dumps(o))")
    rc, out = vm(f"python3 -I -c \"{script}\" {rd} " + " ".join(REQUIRED))
    if rc != 0:
        sys.exit(f"could not read {rd} on the VM (exit {rc})")
    o = json.loads(out.strip().splitlines()[-1])
    session = (o["run"] or {}).get("session_id")

    # 1. ledger
    c = sqlite3.connect(f"file:{a.ledger}?mode=ro", uri=True)
    rows = c.execute("select session, backend, model, coalesce(upstream_usd, usd), latency_ms, queued_ms, status "
                     "from turns where tag = ? order by id", (a.tag,)).fetchall()
    sess_ok = bool(rows) and all(r[0] == session for r in rows)
    report("1 ledger", bool(rows) and sess_ok,
           f"{len(rows)} turn(s) under tag; session {'matches' if sess_ok else 'MISMATCH'} run.json ({session}); "
           f"backend {sorted({r[1] for r in rows})}; model {sorted({r[2] for r in rows})}; "
           f"${sum(r[3] or 0 for r in rows):.4f}")

    # 2. route time: edge rid joined to gate rid
    edge = [r for r in o["edge"] if r.get("rid") and r.get("total_ms") is not None and not r.get("event")]
    gate = {}
    if os.path.exists(a.gate_log):
        for line in open(a.gate_log, errors="replace"):
            try:
                g = json.loads(line)
            except json.JSONDecodeError:
                continue
            if g.get("run") == a.tag and g.get("rid"):
                gate[g["rid"]] = g
    hops = [(r["rid"], r["total_ms"] - gate[r["rid"]]["total_ms"]) for r in edge if r["rid"] in gate and gate[r["rid"]].get("total_ms") is not None]
    report("2 route time", bool(hops),
           f"{len(edge)} edge request(s) with rid; {len(hops)} joined to the gate; hop ms {[round(h, 1) for _, h in hops][:6]}")

    # 3. transcript
    f = o["files"]
    report("3 transcript", (f.get("transcript/stream.jsonl") or 0) > 0 and o["session_files"] > 0,
           f"stream.jsonl {f.get('transcript/stream.jsonl')} bytes; session tree {o['session_files']} file(s)")

    # 4. completeness
    missing = [k for k, v in f.items() if v is None]
    st = o["status"]
    report("4 run directory", not missing and bool(st.get("end_utc")),
           f"missing {missing or 'none'}; status {st.get('status')} ({st.get('reason')}); "
           f"harness_result is_error {((st.get('harness_result') or {}).get('is_error'))}")
    print("C2 " + ("PASS" if all(results) else "FAIL"))
    sys.exit(0 if all(results) else 1)

if __name__ == "__main__":
    main()
