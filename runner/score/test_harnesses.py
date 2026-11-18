"""Known-answer tests for grok_updates.py and opencode_db.py. Standard library only.

usage: python3 -I test_harnesses.py

Each of the six real controller sessions in experiments/harnesses/known-answer/ is copied into a fresh run directory laid out
as the runner keeps it (transcript/grok-home/ or transcript/opencode-data/, and the session's ledger.tsv beside them, which
stands in for the runner's ledger.jsonl), normalised, and checked against values read from its own truth.md:
  - every tool call: its name (Grok's or OpenCode's own, as truth.md gives it), its input, and its start to within 1 s
    (truth.md's time is when the call reached the driver on stdout; the normalised time is the harness's own clock)
  - every tool call's mapped Claude Code name and input fields (Bash command, run_in_background, timeout; Read file_path)
  - each turn end (truth.md's "harness: turn completed" / "step_finish reason stop") to within 1 s, and its count
  - each harness wake (Grok), and the background launches: Grok's "backgrounded" tasks with their start and end, OpenCode's
    nohup ... & launches
  - the side calls: truth.md's ledger lines, by id, against the rows the normaliser marks as title or turn summary
Then the readers that will consume the output: idle.py's (transcript_records, activity), tells.py's (scan), prep.py's
(calls_with_inputs) and experiments/codex/wait_style.py's Claude Code reader (claude_threads), on every session, and the five
requirements the Codex driver set for wait_style.py and polling_table.py (2026-10-26): Claude Code's subagent layout,
exactly one end_turn record per turn end, a Grok wake as a "<task-notification>" user record, a failed-request turn end
marked is_error, and a side_calls list. The subagent and failed-request cases have no real session (none was recorded), so
they are built in the temporary directory from a real session with one field changed, and say so.
"""
import json, os, re, shutil, sqlite3, sys, tempfile
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "..", "codex"))
import grok_updates, opencode_db, idle, prep, tells
import wait_style

KA = os.path.normpath(os.path.join(HERE, "..", "..", "harnesses", "known-answer"))
FAILS, OKS = [], []

def check(name, got, want, tol=None):
    ok = abs(got - want) <= tol if tol is not None and isinstance(got, (int, float)) and isinstance(want, (int, float)) else got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: got {got}, want {want}")
    (OKS if ok else FAILS).append(name)

def wall(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

def hms(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M:%S.%f")[:12]

ROW_RE = re.compile(r"^\|\s*([\d.]+)\s*\|\s*(\S+)\s*\|\s*(.*)\s*\|\s*$")
CALL_RE = re.compile(r"^tool call (\S+) (\{.*\})(?: \(from the database.*)?$")

def truth(path):
    """The events of one truth.md: [(utc epoch, text)]."""
    out = []
    for line in open(path):
        m = ROW_RE.match(line.rstrip("\n"))
        if m and m.group(2) != "UTC" and not m.group(2).startswith("-"):
            out.append((wall(m.group(2)), m.group(3).strip()))
    return out

def run_dir(root, harness, name):
    """A run directory laid out as the runner keeps it, from one known-answer session."""
    src = os.path.join(KA, harness, name)
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "transcript"))
    sub = "grok-home" if harness == "grok-build" else "opencode-data"
    shutil.copytree(os.path.join(src, sub), os.path.join(d, "transcript", sub))
    shutil.copy(os.path.join(src, "ledger.tsv"), d)
    shutil.copy(os.path.join(src, "brief.txt"), d)
    return d

def tool_uses(d, harness):
    """(time, name, input) of every tool_use in the written records, and the records."""
    recs = []
    for r_, _, fs in os.walk(os.path.join(d, "transcript", "session", harness)):
        for f in fs:
            recs += [json.loads(l) for l in open(os.path.join(r_, f))]
    recs.sort(key=lambda r: wall(r["timestamp"]))
    uses = [(wall(r["timestamp"]), b["name"], b["input"], b["id"]) for r in recs if r["type"] == "assistant"
            for b in r["message"]["content"] if b.get("type") == "tool_use"]
    return uses, recs

def raw_of(harness, d):
    """The harness's own (time, tool name, input) per call, for comparing names and inputs with truth.md."""
    if harness == "grok-build":
        out = []
        for r_, _, fs in os.walk(os.path.join(d, "transcript", "grok-home", "sessions")):
            if "updates.jsonl" in fs:
                for l in open(os.path.join(r_, "updates.jsonl")):
                    u = json.loads(l)["params"]["update"]
                    if u.get("sessionUpdate") == "tool_call":
                        out.append((u["toolCallId"], u["title"], u["rawInput"]))
        return {cid: (n, i) for cid, n, i in out}
    con = sqlite3.connect(f"file:{os.path.join(d, 'transcript', 'opencode-data', 'opencode.db')}?mode=ro", uri=True)
    out = {}
    for (data,) in con.execute("SELECT data FROM part"):
        p = json.loads(data)
        if p.get("type") == "tool":
            out[p.get("callID")] = (p["tool"], p["state"].get("input"))
    con.close()
    return out

def readers(name, d, harness):
    """idle.py's, prep.py's, tells.py's and wait_style.py's readers on the written records: (records, tool intervals,
    calls, tells keys, wait_style threads)."""
    recs = idle.transcript_records(d)
    turns, tools, end = idle.activity(recs)
    calls = prep.calls_with_inputs(recs)
    sc = tells.scan(d)
    th = wait_style.claude_threads(d)
    return recs, turns, tools, calls, sc, th

def grok_case(root, name, want_wakes, want_bg):
    print(f"case {name} (Grok Build, experiments/harnesses/known-answer/grok-build/{name})")
    d = run_dir(root, "grok-build", name)
    ev = truth(os.path.join(KA, "grok-build", name, "truth.md"))
    r = grok_updates.normalise(d)
    uses, recs = tool_uses(d, "grok")
    raw = raw_of("grok-build", d)
    tcalls = [(t, *CALL_RE.match(x).groups()) for t, x in ev if CALL_RE.match(x)]
    check(f"{name}: tool calls, as many as truth.md lists", len(uses), len(tcalls))
    for (tt, tname, tinp), (t, nm, inp, cid) in zip(tcalls, uses):
        rn, ri = raw[cid]
        check(f"{name}: {tname} at {hms(tt)}: the harness's name and input", (rn, json.dumps(ri)[:240]), (tname, tinp))
        check(f"{name}: {tname} at {hms(tt)}: start within 1 s (harness clock vs stdout receipt)", round(abs(t - tt), 3), 0.0, tol=1.0)
        want_name = {"run_terminal_command": "Bash", "read_file": "Read", "get_command_or_subagent_output": "TaskOutput"}[tname]
        if want_name == "Bash":
            want_inp = (ri["command"], ri.get("block_until_ms") == 0, ri["block_until_ms"] if ri.get("block_until_ms") else None)
            got_inp = (inp.get("command"), bool(inp.get("run_in_background")), inp.get("timeout"))
        elif want_name == "Read":
            want_inp, got_inp = ri["target_file"], inp.get("file_path")
        else:
            want_inp, got_inp = (ri["task_ids"][0], ri["timeout_ms"], True), (inp.get("task_id"), inp.get("timeout"), inp.get("block"))
        check(f"{name}: {tname} at {hms(tt)} written as {want_name}", (nm, got_inp), (want_name, want_inp))
    ends = [t for t, x in ev if x.startswith("harness: turn completed")]
    got_ends = [wall(x) for x in r["turn_ends"]]
    check(f"{name}: turn ends, as many as truth.md's turn_completed", len(got_ends), len(ends))
    for te, g in zip(ends, got_ends):
        check(f"{name}: turn end at {hms(te)} within 1 s (events.jsonl turn_ended, ms)", round(abs(g - te), 3), 0.0, tol=1.0)
    et = [(wall(x["timestamp"]), x) for x in recs if x["type"] == "assistant" and x["message"].get("stop_reason") == "end_turn"]
    check(f"{name}: exactly one end_turn record per turn end, each within 1 s of it",
          (len(et), all(any(abs(t - g) <= 1 for g in got_ends) for t, _ in et)), (len(ends), True))
    wakes = [t for t, x in ev if x.startswith("harness wake")]
    check(f"{name}: harness wakes", len(r["wakes"]), want_wakes)
    check(f"{name}: wakes as truth.md has them", len(r["wakes"]), len(wakes))
    for tw, w in zip(wakes, r["wakes"]):
        check(f"{name}: wake at {hms(tw)} within 1 s", round(abs(wall(w["utc"]) - tw), 3), 0.0, tol=1.0)
    notes = [x for x in recs if x["type"] == "user" and isinstance(x["message"]["content"], list)
             and x["message"]["content"][0].get("type") == "text"]
    check(f"{name}: each wake is a user record whose first text block starts <task-notification>, Grok's text kept after it",
          [(n["message"]["content"][0]["text"].startswith("<task-notification>"), "<system-reminder>" in n["message"]["content"][0]["text"])
           for n in notes], [(True, True)] * want_wakes)
    bg = [(t, re.search(r"task (\S+) backgrounded", x).group(1)) for t, x in ev if " backgrounded (" in x]
    bend = {re.search(r"task (\S+) ended", x).group(1): (t, x) for t, x in ev if x.startswith("harness: task ") and " ended," in x}
    check(f"{name}: background tasks", len(r["background_tasks"]), want_bg)
    for (tb, tid), b in zip(bg, r["background_tasks"]):
        check(f"{name}: background task {tid[:8]} started within 1 s of truth.md's backgrounded line",
              (b["task_id"], abs(wall(b["backgrounded_utc"]) - tb) <= 1.0), (tid, True))
        te, tx = bend[tid]
        m = re.search(r"exit (\S+), signal (\S+), explicitly_killed (\S+)", tx)
        check(f"{name}: background task {tid[:8]} ended within 1 s, exit/signal/killed as truth.md",
              (abs(wall(b["end_utc"]) - te) <= 1.0, str(b["exit_code"]), str(b["signal"]), str(b["explicitly_killed"])),
              (True, *m.groups()))
    side = [(int(re.search(r"ledger id (\d+)", x).group(1)), x) for t, x in ev if x.startswith("model call")]
    titles = [i for i, x in side if "grok-4.6" in x]
    check(f"{name}: title side calls are exactly truth.md's grok-4.6 rows", sorted(s["ledger_id"] for s in r["side_calls"] if s["kind"] == "title"), titles)
    nsum = len([s for s in r["side_calls"] if s["kind"] == "turn_summary"])
    check(f"{name}: one turn-summary side call per turn end, nothing unmatched",
          (nsum, len([s for s in r["side_calls"] if s["kind"] == "unmatched"])), (len(ends), 0))
    check(f"{name}: agent calls in the ledger = the harness's own modelCalls = ledger rows - side calls",
          (len(r["agent_calls"]), r["agent_calls_check"]["harness_model_calls"]), (len(side) - len(r["side_calls"]),) * 2)
    check(f"{name}: side_calls entries carry ledger_id, ts, model, why",
          all(set(("ledger_id", "ts", "model", "why")) <= set(s) for s in r["side_calls"]), True)
    check(f"{name}: models", r["models"], ["grok-4.7"])
    return d, r

def opencode_case(root, name, want_ends, want_nohup, want_running):
    print(f"case {name} (OpenCode, experiments/harnesses/known-answer/opencode/{name})")
    d = run_dir(root, "opencode", name)
    ev = truth(os.path.join(KA, "opencode", name, "truth.md"))
    r = opencode_db.normalise(d)
    uses, recs = tool_uses(d, "opencode")
    raw = raw_of("opencode", d)
    tcalls = [(t, *CALL_RE.match(x).groups()) for t, x in ev if CALL_RE.match(x)]
    check(f"{name}: tool calls, as many as truth.md lists", len(uses), len(tcalls))
    ok_names = ok_times = ok_map = True
    for (tt, tname, tinp), (t, nm, inp, cid) in zip(tcalls, uses):
        rn, ri = raw[cid]
        # truth.md writes a | as / (it is a table cell)
        if (rn, json.dumps(ri)[:240].replace("|", "/")) != (tname, tinp):
            ok_names = False; print(f"    differs at {hms(tt)}: {rn} {json.dumps(ri)[:120]} vs {tname} {tinp[:120]}")
        if abs(t - tt) > 1.0:
            ok_times = False; print(f"    late at {hms(tt)}: {t - tt:+.3f} s")
        want = {"bash": "Bash", "read": "Read", "write": "Write", "edit": "Edit"}.get(rn, rn)
        got = (nm, inp.get("command") if nm == "Bash" else inp.get("file_path"), inp.get("timeout"))
        if got != (want, ri.get("command") if rn == "bash" else ri.get("filePath"), ri.get("timeout")):
            ok_map = False; print(f"    mapped wrong at {hms(tt)}: {got}")
    many = len(tcalls) > 4
    if many:
        check(f"{name}: all {len(tcalls)} tool calls: OpenCode's name and input as truth.md", ok_names, True)
        check(f"{name}: all {len(tcalls)} tool calls start within 1 s (state.time.start)", ok_times, True)
        check(f"{name}: all {len(tcalls)} written as Bash (command, timeout) or Read (file_path)", ok_map, True)
    else:
        for (tt, tname, tinp), (t, nm, inp, cid) in zip(tcalls, uses):
            check(f"{name}: {tname} at {hms(tt)}: name, input and start within 1 s, written as {nm}",
                  (raw[cid][0], json.dumps(raw[cid][1])[:240], abs(t - tt) <= 1.0), (tname, tinp, True))
    ends = [t for t, x in ev if x.startswith("step_finish reason stop")]
    got_ends = [wall(x) for x in r["turn_ends"]]
    check(f"{name}: turn ends (step-finish reason stop)", (len(got_ends), len(ends)), (want_ends, want_ends))
    for te, g in zip(ends, got_ends):
        check(f"{name}: turn end at {hms(te)} within 1 s, and it is the agent's end (no wake)",
              (abs(g - te) <= 1.0, r["agent_end"] == r["turn_ends"][-1], r["wakes"]), (True, True, []))
    et = [x for x in recs if x["type"] == "assistant" and x["message"].get("stop_reason") == "end_turn"]
    check(f"{name}: exactly one end_turn record per turn end", len(et), want_ends)
    nohup = [(t, x) for t, x in ev if CALL_RE.match(x) and "nohup" in x]
    check(f"{name}: nohup ... & launches (the background launches OpenCode has)",
          (len(r["ampersand_launches"]), len(nohup)), (want_nohup, want_nohup))
    for (tn, x), a in zip(nohup, r["ampersand_launches"]):
        check(f"{name}: nohup launch at {hms(tn)} within 1 s", abs(wall(a["utc"]) - tn) <= 1.0, True)
    still = [t for t, x in ev if "still running when the session ended" in x]
    check(f"{name}: tool calls still running at the end (truth.md's database line)", (len(r["running_at_end"]), len(still)),
          (want_running, want_running))
    side = [int(re.search(r"ledger id (\d+)", x).group(1)) for t, x in ev if x.startswith("model call")]
    check(f"{name}: one title side call, the first row, on the run's own model; every other row an agent step",
          ([(s["ledger_id"], s["kind"], s["model"]) for s in r["side_calls"]], len(r["agent_calls"])),
          ([(min(side), "title", "gemma-4-31b-bedrock")], len(side) - 1))
    check(f"{name}: steps matched to ledger rows by their own tokens (all but a stopped step)",
          (r["agent_calls_check"]["steps_matched"], r["agent_calls_check"]["title_logged"]), (r["agent_calls_check"]["steps"], True))
    check(f"{name}: models", r["models"], ["gemma-4-31b-bedrock"])
    return d, r

def reader_checks(name, d, harness, want_calls, want_tells_ok=True):
    recs, turns, tools, calls, sc, th = readers(name, d, harness)
    check(f"{name}: idle.py reads the records (transcript_records, activity): tool intervals and model turns",
          (len(tools), len(turns) > 0), (want_calls, True))
    check(f"{name}: prep.py calls_with_inputs reads every finished call", len(calls), want_calls)
    check(f"{name}: tells.py scan parses them (no tell, no own-name hit in a controller session)",
          (sc.get("saw_tell"), bool(sc.get("noticed"))), (False, False))
    return recs, th

def main():
    root = tempfile.mkdtemp(prefix="p006-harn-score-")
    try:
        k1, r1 = grok_case(root, "k1-background-then-one-blocking-wait", 0, 1)
        recs, th = reader_checks("k1", k1, "grok", 3)
        c = th[0]["calls"]
        check("k1: wait_style reads the moved-to-background Bash as tool_timeout at 30 s and the 1,500 s TaskOutput as a blocking read",
              (wait_style.ended_by(c[0], [])[0], c[0]["result"]["moved_to_background"] is not None, c[1]["name"], c[1]["timeout_arg_ms"],
               round(c[1]["end"] - c[1]["start"])), ("tool_timeout", True, "TaskOutput", 1500000, 1165))
        check("k1: wait_style sees one root turn end and no wake", (len(th[0]["turn_ends"]), [s[1] for s in th[0]["turn_starts"]]), (1, ["user"]))

        k2, r2 = grok_case(root, "k2-turn-end-woken-by-harness", 1, 1)
        recs, th = reader_checks("k2", k2, "grok", 2)
        check("k2: wait_style reads the wake as a notification turn start after the first turn end (woken_by_harness)",
              ([s[1] for s in th[0]["turn_starts"]], len(th[0]["turn_ends"]), th[0]["turn_starts"][1][0] > th[0]["turn_ends"][0]),
              (["user", "notification"], 2, True))
        check("k2: the background launch is a run_in_background Bash call with its task id",
              [(x["name"], x["background"]) for x in th[0]["calls"]], [("Bash", True), ("Read", False)])

        k3, r3 = grok_case(root, "k3-turn-end-job-killed-at-600s", 0, 1)
        reader_checks("k3", k3, "grok", 1)
        check("k3: the job grok -p killed 600 s after the turn end: killed, no wake, one turn end",
              (r3["background_tasks"][0]["explicitly_killed"], r3["background_tasks"][0]["will_wake"], len(r3["turn_ends"]),
               round(wall(r3["background_tasks"][0]["end_utc"]) - wall(r3["turn_ends"][0]))), (True, False, 1, 600))

        o1, q1 = opencode_case(root, "o1-one-foreground-block", 1, 0, 0)
        recs, th = reader_checks("o1", o1, "opencode", 2)
        check("o1: wait_style reads one 1,201 s foreground Bash call and one turn end",
              (round(th[0]["calls"][0]["end"] - th[0]["calls"][0]["start"]), th[0]["calls"][0]["background"], len(th[0]["turn_ends"])),
              (1201, False, 1))
        o2, q2 = opencode_case(root, "o2-killed-twice-then-nohup-and-busy-poll", 0, 1, 1)
        recs, th = reader_checks("o2", o2, "opencode", 70)
        c = th[0]["calls"]
        check("o2: the two foreground runs OpenCode's timeout ended read as tool_timeout: 120 s (default) and 1,200 s (the agent's own)",
              [(wait_style.ended_by(x, [])[0], wait_style.ended_by(x, [])[1].split(",")[0]) for x in c[:2]],
              [("tool_timeout", "after 120 s"), ("tool_timeout", "after 1200 s")])
        check("o2: the call in flight at the stop has no result (idle.py drops it unless open_until is given)",
              (c[-1]["end"], len(idle.activity(recs, open_until=wall("2026-10-26T19:18:53.496Z"))[1])), (None, 71))
        check("o2: no turn end (the supervisor stopped it mid-step)", len(th[0]["turn_ends"]), 0)
        o3, q3 = opencode_case(root, "o3-nohup-then-turn-end-no-wake", 1, 1, 0)
        reader_checks("o3", o3, "opencode", 1)

        print("case CX (the Codex driver's five requirements for wait_style.py and polling_table.py)")
        # 1. subagent layout: a session Grok marks as not primary, and an OpenCode session with a parent_id (none recorded:
        # built from k1 and o3 with that one field changed)
        sd = os.path.join(root, "sub-grok"); shutil.copytree(k1, sd)
        shutil.rmtree(os.path.join(sd, "transcript", "session"))
        src = next(os.path.join(a, x) for a, ds, _ in os.walk(os.path.join(sd, "transcript", "grok-home", "sessions")) for x in ds
                   if x.startswith("01a17943"))
        dst = src.replace("01a17943-41d1-7392-9b8d-8e3323ca60f9", "01a17943-0000-0000-0000-00000000c0de")
        shutil.copytree(src, dst)
        ev = [json.loads(l) for l in open(os.path.join(dst, "events.jsonl"))]
        for e in ev:
            if e.get("type") == "turn_started":
                e["session_relationship"] = "subagent"
        open(os.path.join(dst, "events.jsonl"), "w").write("".join(json.dumps(e) + "\n" for e in ev))
        os.remove(os.path.join(dst, "summary.json"))
        rs = grok_updates.normalise(sd)
        check("Grok subagent (synthetic: k1's session copied with session_relationship subagent) lands at "
              "session/grok/<root>/subagents/<id>.jsonl",
              sorted(t["file"] for t in rs["threads"]),
              ["transcript/session/grok/01a17943-41d1-7392-9b8d-8e3323ca60f9.jsonl",
               "transcript/session/grok/01a17943-41d1-7392-9b8d-8e3323ca60f9/subagents/01a17943-0000-0000-0000-00000000c0de.jsonl"])
        check("wait_style's claude_threads finds it as a subagent thread", sorted(t["subagent"] for t in wait_style.claude_threads(sd)),
              [False, True])
        od = os.path.join(root, "sub-oc"); shutil.copytree(o3, od)
        shutil.rmtree(os.path.join(od, "transcript", "session"))
        db = os.path.join(od, "transcript", "opencode-data", "opencode.db")
        for x in ("-wal", "-shm"):
            if os.path.exists(db + x):
                os.remove(db + x)
        con = sqlite3.connect(db)      # the temporary copy only
        con.execute("PRAGMA journal_mode=DELETE")
        con.execute("INSERT INTO session SELECT 'ses_child', project_id, workspace_id, id, 'child', directory, path, 'child', "
                    "version, share_url, summary_additions, summary_deletions, summary_files, summary_diffs, metadata, cost, "
                    "tokens_input, tokens_output, tokens_reasoning, tokens_cache_read, tokens_cache_write, revert, permission, "
                    "agent, model, time_created + 1, time_updated, time_compacting, time_archived FROM session")
        con.execute("INSERT INTO message SELECT id || '_c', 'ses_child', time_created + 1, time_updated, data FROM message")
        con.execute("INSERT INTO part SELECT id || '_c', message_id || '_c', 'ses_child', time_created + 1, time_updated, data FROM part")
        con.commit(); con.close()
        ro = opencode_db.normalise(od)
        check("OpenCode subagent (synthetic: o3's session copied with a parent_id) lands at session/opencode/<root>/subagents/<id>.jsonl",
              sorted(t["file"] for t in ro["threads"]),
              ["transcript/session/opencode/ses_e86426085ffeBb1ksqkPihDYA1.jsonl",
               "transcript/session/opencode/ses_e86426085ffeBb1ksqkPihDYA1/subagents/ses_child.jsonl"])
        check("and the root's turn end is still the only agent end", (ro["turn_ends"], ro["agent_end"]),
              (q3["turn_ends"], q3["agent_end"]))
        # 2. exactly one end_turn per turn end: checked per session above; here across all six
        n_ends = sum(len(x["turn_ends"]) for x in (r1, r2, r3, q1, q2, q3))
        n_rec = sum(len(t["turn_ends"]) for d_ in (k1, k2, k3, o1, o2, o3) for t in wait_style.claude_threads(d_))
        check("all six sessions: end_turn records wait_style counts = turn ends the normalisers return (k1 1, k2 2, k3 1, o1 1, o2 0, o3 1)",
              (n_rec, n_ends), (6, 6))
        # 3. the wake: checked in k2 above (first text block starts <task-notification>, read as a notification turn)
        # 4. a failed model request ends the turn with is_error (no recorded session has one: k3 with error_kind on its
        # turn_completed, and o3 with an APIError on its last message)
        ed = os.path.join(root, "err-grok"); shutil.copytree(k3, ed); shutil.rmtree(os.path.join(ed, "transcript", "session"))
        up = next(os.path.join(a, "updates.jsonl") for a, _, fs in os.walk(os.path.join(ed, "transcript", "grok-home")) if "updates.jsonl" in fs)
        rows = [json.loads(l) for l in open(up)]
        for x in rows:
            if x["params"]["update"].get("sessionUpdate") == "turn_completed":
                x["params"]["update"]["error_kind"] = "api_error"
        open(up, "w").write("".join(json.dumps(x) + "\n" for x in rows))
        re_ = grok_updates.normalise(ed)
        _, recs = tool_uses(ed, "grok")
        et = [x for x in recs if x["type"] == "assistant" and x["message"].get("stop_reason") == "end_turn"]
        check("Grok turn ending on a failed request (synthetic: k3 with error_kind): one end_turn record, top-level is_error true",
              ([(x.get("is_error"), x.get("isApiErrorMessage")) for x in et], re_["threads"][0]["turns"][0]["is_error"]), ([(True, True)], True))
        check("and STARTED keeps no end_turn of its own", [x["message"]["content"][0].get("text") for x in et][0].startswith("API Error"), True)
        _, recs = tool_uses(k3, "grok")
        check("k3 as recorded: no is_error anywhere", any(x.get("is_error") for x in recs), False)
        ed = os.path.join(root, "err-oc"); shutil.copytree(o3, ed); shutil.rmtree(os.path.join(ed, "transcript", "session"))
        db = os.path.join(ed, "transcript", "opencode-data", "opencode.db")
        for x in ("-wal", "-shm"):
            if os.path.exists(db + x):
                os.remove(db + x)
        con = sqlite3.connect(db)
        con.execute("PRAGMA journal_mode=DELETE")
        mid, data = con.execute("SELECT id, data FROM message ORDER BY time_created DESC LIMIT 1").fetchone()
        m = json.loads(data); m.pop("finish", None)
        m["error"] = {"name": "APIError", "data": {"message": "upstream 503", "statusCode": 503, "isRetryable": True}}
        con.execute("UPDATE message SET data = ? WHERE id = ?", (json.dumps(m), mid))
        con.execute("UPDATE part SET data = json_set(data, '$.reason', 'error') WHERE message_id = ? AND data LIKE '%step-finish%'", (mid,))
        con.commit(); con.close()
        re_ = opencode_db.normalise(ed)
        _, recs = tool_uses(ed, "opencode")
        et = [x for x in recs if x["type"] == "assistant" and x["message"].get("stop_reason") == "end_turn"]
        check("OpenCode step failing on a model request (synthetic: o3's last message with an APIError): one end_turn, is_error true",
              ([(x.get("is_error"), x.get("isApiErrorMessage")) for x in et], re_["threads"][0]["turns"][0]["error"]), ([(True, True)], "APIError"))
        _, recs = tool_uses(o3, "opencode")
        check("o3 as recorded: no is_error on any end record", any(x.get("is_error") for x in recs if x["type"] == "assistant"), False)
        # 5. side_calls: checked per session; OpenCode's title call is on the run's own model
        check("OpenCode's title call uses the run's own model, so a model filter cannot separate it from the agent's rows",
              sorted({s["model"] for q in (q1, q2, q3) for s in q["side_calls"]}), ["gemma-4-31b-bedrock"])
        check("Grok's side calls: 6 title rows (grok-4.6) and 4 turn summaries (grok-4.7) across k1-k3",
              (sum(s["kind"] == "title" for x in (r1, r2, r3) for s in x["side_calls"]),
               sum(s["kind"] == "turn_summary" for x in (r1, r2, r3) for s in x["side_calls"])), (6, 4))
    finally:
        if not os.environ.get("KEEP_KNOWN_ANSWER"):
            shutil.rmtree(root, ignore_errors=True)
        else:
            print("kept", root)
    print(f"{len(OKS)} ok")
    print("PASS" if not FAILS else f"FAIL: {len(FAILS)} check(s): {FAILS}")
    sys.exit(1 if FAILS else 0)

if __name__ == "__main__":
    main()
