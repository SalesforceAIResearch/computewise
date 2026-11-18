# the runner: what Astra's `kill -STOP` did (the analysis 2026-10-28 09:49Z, G-kvgate-astracc-L0's scheduler stop
# ended its task in Claude Code). Per run: every tool call sending SIGSTOP (kill -STOP/-s STOP/-19, os.kill(..., SIGSTOP)) and
# SIGCONT, from the session records (Claude Code native, or Codex rollouts as score_all normalises them); the pids named; what each
# pid led (the box's process records: its cmdline, and how many task-job children were alive under it), its daemon rank when it is
# a job itself (job_hello pid), and every job_bye within 60 s after the stop. Read-only.
# usage: python3 -I sigstop_scan.py <run_id> ...   (kept in runner/score/ for the paper's harness claim, the analysis 2026-10-28 10:16Z; runs read from ~/computewise-runs)
import datetime, glob, gzip, json, os, re, sys
ep = lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
STOP = re.compile(r"kill\s+(?:-STOP|-s\s+STOP|-SIGSTOP|-19)\s+([\d\s$\-]+)|os\.kill\(\s*(\w+)\s*,\s*signal\.SIGSTOP|pkill\s+-STOP\s+(\S.*)")
CONT = re.compile(r"kill\s+(?:-CONT|-s\s+CONT|-SIGCONT|-18)\s+([\d\s$\-]+)|os\.kill\(\s*(\w+)\s*,\s*signal\.SIGCONT|pkill\s+-CONT\s+(\S.*)")
def opn(p):
    return gzip.open(p + ".gz", "rt", errors="replace") if not os.path.exists(p) and os.path.exists(p + ".gz") else open(p, errors="replace")
def calls(d):
    out = []
    for f in glob.glob(os.path.join(d, "transcript", "session", "**", "*.jsonl"), recursive=True):
        for l in open(f, errors="replace"):
            try: r = json.loads(l)
            except ValueError: continue
            if r.get("type") != "assistant" or not r.get("timestamp"): continue
            for b in (r.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    i = b.get("input") or {}
                    c = i.get("command") if isinstance(i, dict) else None
                    if not c:
                        c = json.dumps(i)
                    out.append((ep(r["timestamp"]), c, "/subagents/" in f))
    return sorted(out)
def pids_of(m):
    g = next((x for x in m.groups() if x), "") or ""
    return [int(x) for x in re.findall(r"\b(\d{2,7})\b", g)]
for rid in sys.argv[1:]:
    d = os.path.expanduser(f"~/computewise-runs/{rid}")
    st = json.load(open(os.path.join(d, "status.json"))); a0 = st["agent_start_wall"]
    hello, bye = {}, {}
    for l in opn(os.path.join(d, "occupancy.jsonl")):
        if '"job_hello"' not in l and '"job_bye"' not in l: continue
        try: e = json.loads(l)
        except ValueError: continue
        (hello if e["kind"] == "job_hello" else bye)[e["job"]] = e
    by_pid = {h.get("pid"): j for j, h in hello.items()}
    stops, conts = [], []
    for t, c, sub in calls(d):
        for m in STOP.finditer(c): stops.append((t, pids_of(m), c[:200].replace("\n", " "), sub))
        for m in CONT.finditer(c): conts.append((t, pids_of(m), c[:120].replace("\n", " ")))
    # the box's process records: cmdline and parent of each pid (host pid space) and the daemon's (box pid space) are the same
    # namespace only when the record says so; job_hello pids are the box's own, as the agent's kill names them
    procs = {}
    rp = os.path.join(d, "resources.jsonl")
    if os.path.exists(rp) or os.path.exists(rp + ".gz"):
        for l in opn(rp):
            if '"procs"' not in l: continue
            try: r = json.loads(l)
            except ValueError: continue
            for q in (r.get("procs") or []) + (r.get("started") or []):
                if isinstance(q, dict) and q.get("pid") is not None:
                    procs.setdefault(int(q["pid"]), q)
    print(f"== {rid}: {len(stops)} stop call(s), {len(conts)} cont call(s)")
    for t, ps, c, sub in stops:
        ends = sorted((round(b["t_wall"] - t, 1), j, " ".join(map(str, (hello.get(j) or {}).get("argv") or []))[:60])
                      for j, b in bye.items() if 0 <= b["t_wall"] - t <= 60)
        led = []
        for p in ps:
            if p in by_pid:
                led.append(f"pid {p} = rank {by_pid[p]} ({' '.join(map(str, hello[by_pid[p]].get('argv') or []))[:60]})")
            else:
                kids = [j for j, h in hello.items() if h.get("ppid") == p]
                led.append(f"pid {p}: not a daemon job" + (f" ({procs[p].get('cmdline','')[:80]})" if p in procs else ""))
        cont = [round(ct - t) for ct, cps, _ in conts if ct >= t and set(cps) & set(ps)]
        print(f"  stop at {t - a0:.0f} s by {'sub' if sub else 'main'}: {c[:140]}")
        print(f"    led: {'; '.join(led) or 'no pid parsed'}")
        print(f"    job_byes within 60 s: {ends or 'none'}")
        print(f"    resumed with CONT: {('after ' + str(cont[0]) + ' s') if cont else 'no'}")
