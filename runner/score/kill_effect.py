# the runner: did each kill an agent sent stop the task job it aimed at? (kill_missed_child, the analysis
# 2026-10-28 ~03:42Z.) Reads every Bash call in the main session and every subagent that sends a signal (kill/pkill/killall,
# not kill -0), resolves its targets: pids named literally, $(cat X.pid) pid files (the pid is read from the run's workspace copy
# when it is there, else matched by the file's job name), and pkill -f patterns. A target counts as the task job when the pid is
# a rank's own pid (job_hello) or the parent of one is not known: a wrapper's pid. It reports, per kill, the ranks running at
# the kill and whether each ended within 30 s; a rank that ran on after a kill that named its wrapper or its job is a miss.
# usage: python3 -I kill_effect.py <run_id> [...] [--json]   (a copy of /tmp/c1r/kill-effect2.py kept in runner/score/ for the paper, 2026-10-28)
import json, sys, glob, datetime, re, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import idle
ep = lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())
LIVE_TOL_S = 10.0
SIG = re.compile(r"(?:^|[;&|\s(])(kill|pkill|killall)\b((?:\s+-[A-Za-z0-9]+)*)\s+([^;&|\n]+)")
out = {}
for rid in [a for a in sys.argv[1:] if not a.startswith("--")]:
    d = os.path.expanduser(f"~/computewise-runs/{rid}")
    if not os.path.exists(d + "/occupancy.jsonl"): continue
    a0 = json.load(open(d + "/status.json"))["agent_start_wall"]; task = json.load(open(d + "/task.json"))
    hello, bye, look = {}, {}, {}
    for l in open(d + "/occupancy.jsonl", errors="replace"):
        if not any(k in l for k in ('"job_hello"', '"job_bye"', '"lookup"')): continue
        e = json.loads(l); k = e["kind"]
        if k == "job_hello": hello.setdefault(e["job"], e)
        elif k == "job_bye": bye[e["job"]] = e
        else: look.setdefault(e["job"], e)
    jobs = idle.tie_jobs(task); idle.attach_decisions(task, list(look.values()))
    tj = {j: idle.task_job_of(h, look.get(j), jobs) for j, h in hello.items()}
    pid_rank = {h.get("pid"): j for j, h in hello.items() if h.get("pid")}
    kills = []
    for f in glob.glob(d + "/transcript/session/**/*.jsonl", recursive=True):
        who = "main" if "/subagents/" not in f else "sub"
        for l in open(f, errors="replace"):
            try: r = json.loads(l)
            except ValueError: continue
            if r.get("type") != "assistant" or not r.get("timestamp"): continue
            for b in (r.get("message") or {}).get("content") or []:
                if not (isinstance(b, dict) and b.get("type") == "tool_use" and b["name"] == "Bash"): continue
                c = (b.get("input") or {}).get("command", "")
                # a loop over job names (G-kvgate-haiku-T1: `for j in a b c; do kill $(cat logs/$j.log.pid); done`): one kill per name
                for lm in re.finditer(r"for\s+(\w+)\s+in\s+([^;]+);\s*do(.*?)\bdone\b", c, re.S):
                    var, names, body = lm.group(1), lm.group(2).split(), lm.group(3)
                    if re.search(r"(?:^|[;&|\s(])kill\b(?!\s+-0)", body):
                        for nm in names:
                            c += "\n" + re.sub(r"\$\{?" + var + r"\}?", nm, body)
                # a pid file read into a variable first (gpu-team T4: `P=$(cat X.log.pid); kill -TERM -- -$P`): the variable stands
                # for the pid file wherever the command uses it
                # ... and through one more variable (`L=logs/X.log.pid; P=$(cat $L)`, Sonnet T4)
                # each assignment stands for its value only up to the next assignment of the same variable (two loop copies of
                # `P=$(cat X.pid)` in one command, G-kvgate-haiku-T3: the first copy's file had replaced the second copy's $P)
                def _bind(c, assign_re, use_of, value_of):
                    """replace each use of an assigned variable by its value, from the assignment up to the next assignment of the
                    same variable (or the end)"""
                    ms = list(re.finditer(assign_re, c))
                    if not ms:
                        return c
                    out, pos = [], 0
                    for k, m in enumerate(ms):
                        out.append(c[pos:m.end()])
                        stop = next((x.start() for x in ms[k + 1:] if x.group(1) == m.group(1)), len(c))
                        # the text up to this variable's next assignment, cut at the next assignment of any kind so the loop
                        # continues from there (that assignment substitutes its own variable in turn)
                        nxt_any = ms[k + 1].start() if k + 1 < len(ms) else len(c)
                        seg_end = min(stop, nxt_any)
                        out.append(re.sub(use_of(m.group(1)), value_of(m), c[m.end():seg_end]))
                        pos = seg_end
                        # beyond the next assignment of another variable, this one still holds until its own reassignment
                        if seg_end < stop:
                            rest = re.sub(use_of(m.group(1)), value_of(m), c[seg_end:stop])
                            c = c[:seg_end] + rest + c[stop:]
                            ms = list(re.finditer(assign_re, c))
                    out.append(c[pos:])
                    return "".join(out)
                c = _bind(c, r"\b(\w+)=(\S+?\.pid)(?=[;\s])", lambda v: r"\$\(cat\s+\$\{?" + v + r"\}?\)", lambda m: "$(cat " + m.group(2) + ")")
                c = _bind(c, r"\b(\w+)=\$\(cat\s+(\S+?\.pid)\)", lambda v: r"\$\{?" + v + r"\}?(?!\w)", lambda m: "$(cat " + m.group(2) + ")")
                for m in SIG.finditer(c):
                    tool, flags, arg = m.group(1), m.group(2) or "", m.group(3)
                    if re.search(r"(?:^|\s)-0\b", flags) or arg.strip().startswith("-0"): continue
                    # a pause or a resume is not a kill (G-kvgate-fable-L0: `kill -STOP 5924 6136 6342 7091`, resumed by its own
                    # scheduler; sigstop_scan.py covers stops)
                    if re.search(r"(?:^|\s)-(?:STOP|SIGSTOP|19|CONT|SIGCONT|18|TSTP|SIGTSTP|20)\b", flags) or \
                       re.match(r"-s\s+(?:STOP|CONT|TSTP)\b", arg.strip()): continue
                    kills.append((ep(r["timestamp"]), who, tool, flags.strip(), arg.strip()[:160]))
    rows = []
    for t, who, tool, flags, arg in sorted(kills):
        # a rank starting within LIVE_TOL_S after the kill call is live for it: the call's time is when the agent issued it, and a
        # kill racing the job's own start (G-kvgate-haiku-T3: gpus killed squat GSM8K's wrapper pid 3854 in the second its python
        # 3857 started) reads the pid file of the launch it races
        live = {j for j, x in tj.items() if x and idle.job_start(hello[j]) <= t + LIVE_TOL_S and t <= bye.get(j, {}).get("t_wall", 9e18) + 5}
        named = set()
        for p in re.findall(r"(?<![\w$])(\d{2,7})(?![\w.])", arg):
            if int(p) in pid_rank and pid_rank[int(p)] in live: named.add(pid_rank[int(p)])
        if tool == "pkill" or "-f" in flags.split():
            pat = norm(re.sub(r"^-f\s+", "", arg).strip("'\""))
            for j in live:
                if pat and len(pat) >= 6 and pat in norm(" ".join(hello[j].get("argv") or [])): named.add(j)
        for pf in re.findall(r"\$\(cat\s+(\S+?)\.pid\)", arg):
            key = norm(os.path.basename(pf).replace(".log", ""))
            for j in live:
                x = norm(tj[j]); m_, _, w_ = (tj[j] or "").partition("--")
                if norm(m_) in key and norm(w_).replace("longbench", "")[:8] in key.replace("longbench", ""): named.add(j)
        # a pid-file kill whose name ties to no task job (G4n's soap-s0 is task job soap-run1): a live rank that ended within 30 s
        # of it is the one it stopped
        if not named and re.search(r"\$\(cat\s", arg):
            near = [j for j in live if bye.get(j) and 0 <= bye[j]["t_wall"] - t < 30]
            named |= set(near[:1])
        res = []
        for j in sorted(named):
            b = bye.get(j) or {}; dt = (b["t_wall"] - t) if b else None
            res.append({"rank": j, "task_job": tj[j], "stopped": dt is not None and dt < 30, "ran_on_s": round(dt) if dt is not None else None})
        rows.append({"t_s": round(t - a0), "by": who, "kill": f"{tool} {flags} {arg}"[:140], "targets": res})
    # one entry per rank: a command that sends kill then kill -9 to the same pid is one attempt
    seen = set(); 
    for r in rows:
        r["targets"] = [x for x in r["targets"] if (x["rank"], r["t_s"]) not in seen and not seen.add((x["rank"], r["t_s"]))]
    hits = [x for r in rows for x in r["targets"]]
    out[rid] = {"kills": len(rows), "kills_naming_a_rank": sum(1 for r in rows if r["targets"]),
                "stopped": sum(1 for x in hits if x["stopped"]), "missed": [dict(x, at_s=r["t_s"], by=r["by"]) for r in rows for x in r["targets"] if not x["stopped"]]}
    out[rid]["missed_gpu_s"] = round(sum(x["ran_on_s"] or 0 for x in out[rid]["missed"]))
if "--json" in sys.argv: print(json.dumps(out, indent=1)); sys.exit()
for rid, o in out.items():
    print(f"{rid:24s} kills {o['kills']:3d} naming a rank {o['kills_naming_a_rank']:3d} | stopped {o['stopped']:3d} | MISSED {len(o['missed'])} ({o['missed_gpu_s']} GPU-s ran on)")
    for x in o["missed"]: print(f"     at {x['at_s']} s by {x['by']}: rank {x['rank']} {x['task_job']} ran on {x['ran_on_s']} s")
