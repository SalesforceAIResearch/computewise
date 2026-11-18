# the runner: the per-run report analysis wants for each gpu-team (T1) run, beside the bar figure (the analysis
# 2026-10-27 ~16:15Z). Reads a collected run on the controller (~/computewise-runs/<run_id>): transcript/stream.jsonl for who called what
# (parent_tool_use_id chains: main -> named teammate -> job subagent), the session jsonl files for each tool call's timestamp,
# occupancy.jsonl for job ends per GPU, score.json for the bar figure, gate timing and the answer.
# usage: python3 -I team_report.py <run_id> [--json]
# (in the repository from 2026-10-29: the /tmp/c1r/team-report.py copy was lost when the controller's reboot cleared /tmp; recovered from the runner's session transcript)
# Heuristics, stated in the output: a "job launch" is a Bash call whose command runs a task job (CUDA_VISIBLE_DEVICES=, sbatch,
# torchrun, or a python evaluator/train script); a "sleep" is a foreground sleep of 60 s or more; a "poll" is a loop over
# pgrep/ps/nvidia-smi/squeue/ls/cat/grep/tail, or a pgrep/squeue call repeated by the same agent.
import datetime, glob, json, os, re, statistics, sys

rid = sys.argv[1]; D = os.path.expanduser(f"~/computewise-runs/{rid}")
J = lambda p: json.load(open(p)) if os.path.exists(p) else {}
rj, st, sc = J(f"{D}/run.json"), J(f"{D}/status.json"), J(f"{D}/score.json")
a0 = st.get("agent_start_wall") or 0
ep = lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if s else None

# timestamps per tool_use id, from every session file (main and subagents)
ts = {}
for f in glob.glob(f"{D}/transcript/session/**/*.jsonl", recursive=True):
    for l in open(f, errors="replace"):
        try: r = json.loads(l)
        except ValueError: continue
        if r.get("type") == "assistant":
            for b in (r.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use": ts[b["id"]] = ep(r.get("timestamp"))

JOB_RE = re.compile(r"CUDA_VISIBLE_DEVICES=|\bsbatch\b|\btorchrun\b|python3?\s+\S*(eval|train)\S*\.py")
SLEEP_RE = re.compile(r"\bsleep\s+(\d+)")
POLL_RE = re.compile(r"(while|until)\b.*\b(pgrep|ps |nvidia-smi|squeue|ls |cat |grep |tail )|watch\s+-n")
who = {}               # tool_use id of an Agent call -> the agent it created
agents = {"main": {"kind": "main"}}
calls = []             # (t, agent, tool, input)
main_results, sub_results, main_peak_ctx, init_tools = 0, 0, 0, set()
sends = {"ok": 0, "not_ok": 0}
send_ids = set()
for l in open(f"{D}/transcript/stream.jsonl", errors="replace"):
    try: x = json.loads(l)
    except ValueError: continue
    par = x.get("parent_tool_use_id")
    me = who.get(par, "main") if par else "main"
    if x.get("type") == "system" and x.get("subtype") == "init" and not par: init_tools |= set(x.get("tools") or [])
    if x.get("type") == "result":
        if par: sub_results += 1
        else: main_results += 1
    if x.get("type") == "assistant":
        u = (x.get("message") or {}).get("usage") or {}
        if me == "main":
            main_peak_ctx = max(main_peak_ctx, (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0))
        for b in (x.get("message") or {}).get("content") or []:
            if not isinstance(b, dict) or b.get("type") != "tool_use": continue
            i = b.get("input") or {}
            calls.append((ts.get(b["id"]), me, b["name"], i, b["id"]))
            if b["name"] in ("Agent", "Task"):
                if me == "main" and i.get("name"):
                    who[b["id"]] = i["name"]; agents[i["name"]] = {"kind": "teammate", "t": ts.get(b["id"])}
                else:
                    who[b["id"]] = f"sub<{me}"; agents.setdefault(f"sub<{me}", {"kind": "subagents", "n": 0})["n"] = agents.get(f"sub<{me}", {}).get("n", 0) + 1
            if b["name"] == "SendMessage": send_ids.add(b["id"])
    if x.get("type") == "user":
        for b in (x.get("message") or {}).get("content") or []:
            if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") in send_ids:
                c = b.get("content"); c = c if isinstance(c, str) else json.dumps(c)
                sends["ok" if '"success":true' in c.replace(" ", "").replace('\\"', '"') else "not_ok"] += 1

def role(agent):
    if agent == "main": return "main"
    if agent == "gpus": return "gpus"
    if agent.startswith("sub<"): return "subagents of " + agent[4:]
    return "branch " + agent

# 1. structure: job launches by agent
launch = {}
for t, me, tool, i, _ in calls:
    if tool == "Bash" and JOB_RE.search(i.get("command") or ""):
        launch[role(me)] = launch.get(role(me), 0) + 1
teammates = sorted(k for k, v in agents.items() if v.get("kind") == "teammate")

# 3. the grant gap: NEED (a SendMessage to gpus starting "NEED <job>") to the gpus Agent call whose prompt names the job;
# a GPU's job end (occupancy job_bye on that GPU) to the next gpus Agent call that sets CUDA_VISIBLE_DEVICES to it
need, grant = {}, []
for t, me, tool, i, _ in calls:
    if tool == "SendMessage" and str(i.get("to") or i.get("recipient")) == "gpus":
        m = re.match(r"\s*NEED\s+(\S+)", str(i.get("message") or i.get("content") or ""))
        if m and t: need.setdefault(m.group(1), t)
    if tool in ("Agent", "Task") and me == "gpus" and t:
        p = i.get("prompt") or ""
        jm = re.search(r"(?:DONE|FAILED)\s+\{?(\S+?)\}?\s+GPU", p); gm = re.search(r"CUDA_VISIBLE_DEVICES=(\d+)", p)
        grant.append((t, jm.group(1) if jm else None, int(gm.group(1)) if gm else None))
need_to_start = [round(g[0] - need[g[1]], 1) for g in grant if g[1] in need and g[0] >= need[g[1]]]
ends = []
occ = f"{D}/occupancy.jsonl"
if os.path.exists(occ):
    hello = {}
    for l in open(occ, errors="replace"):
        if '"job_hello"' not in l and '"job_bye"' not in l and '"busy_start"' not in l: continue
        try: e = json.loads(l)
        except ValueError: continue
        if e.get("kind") == "busy_start" and e.get("gpu", -1) >= 0: hello.setdefault(e["job"], e["gpu"])
        if e.get("kind") == "job_bye" and e["job"] in hello: ends.append((e["t_wall"], hello[e["job"]]))
end_to_grant = []
for te, g in ends:
    nxt = [t for t, _, gg in grant if gg == g and t >= te]
    if nxt: end_to_grant.append(round(min(nxt) - te, 1))
stat = lambda v: {"n": len(v), "median": round(statistics.median(v), 1), "max": max(v)} if v else {"n": 0}

# 4. waiting compliance by agent
wait = {}
seen_pgrep = {}
for t, me, tool, i, _ in calls:
    if tool != "Bash": continue
    c, r = i.get("command") or "", role(me)
    w = wait.setdefault(r, {"sleeps_ge_60s": 0, "polls": 0, "pgrep_or_squeue_repeats": 0})
    if not i.get("run_in_background") and any(int(s) >= 60 for s in SLEEP_RE.findall(c)): w["sleeps_ge_60s"] += 1
    if POLL_RE.search(c): w["polls"] += 1
    if re.search(r"\b(pgrep|squeue)\b", c):
        seen_pgrep[r] = seen_pgrep.get(r, 0) + 1
        if seen_pgrep[r] > 1: w["pgrep_or_squeue_repeats"] += 1

h = sc.get("headline") or {}; m = sc.get("mlskv_process") or {}; g4 = sc.get("g4n_answer") or {}
gt = sc.get("gate_timing") or {}
cc = rj.get("claude_code") or {}
out = {"run": rid, "task": h.get("task"), "model": h.get("model"), "validity": (sc.get("validity") or {}).get("status"),
       "teams_flag": {"run.json claude_code.agent_teams": cc.get("agent_teams"), "named Agent calls from main": len(teammates),
                      "init tools": sorted(x for x in init_tools if x in ("Agent", "Task", "SendMessage", "TeamCreate"))},
       "1_structure": {"teammates": teammates, "job_launches_by": launch, "job_subagents_by": {k[4:]: v["n"] for k, v in agents.items() if v.get("kind") == "subagents"},
                       "sendmessage_results": sends},
       "2_main_load": {"main_result_events": main_results, "subagent_result_events": sub_results, "main_peak_context_tokens": main_peak_ctx},
       "3_grant_gap_s": {"need_to_subagent_start": stat(need_to_start), "job_end_to_next_grant_same_gpu": stat(end_to_grant),
                         "grants_seen": len(grant), "needs_seen": len(need)},
       "4_waiting": wait,
       "5_gate_timing": {k: (len(v) if isinstance(v, list) else v) for k, v in gt.items() if k not in ("rejections", "how")},
       "6_answer": {"pick": m.get("pick") or g4.get("pick") or g4.get("adopted"), "supported": m.get("pick_supported") if m else g4.get("supported")},
       "heuristics": "job launch = Bash with CUDA_VISIBLE_DEVICES=/sbatch/torchrun/python *eval|train*.py; sleep = foreground sleep >= 60 s; poll = while/until loop over pgrep/ps/nvidia-smi/squeue/ls/cat/grep/tail; grant gap from gpus' Agent prompts (job from 'DONE <job> GPU', GPU from CUDA_VISIBLE_DEVICES)"}
print(json.dumps(out, indent=1, default=str) if "--json" in sys.argv else json.dumps(out, default=str))
