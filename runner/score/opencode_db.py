"""OpenCode sessions as the transcript records the scorers already read.

usage: python3 -I opencode_db.py <run_dir> [--db <opencode.db>] [--json]

OpenCode (1.19.4) keeps every session in one SQLite database, $HOME/.local/share/opencode/opencode.db (the run directory
keeps it as transcript/opencode-data/opencode.db, or opencode-data/opencode.db), opened here read-only (mode=ro; SQLite
may still touch the -shm file beside it, as any reader of a WAL database does). Tables session (parent_id set for a
`task` subagent), message (data: role, time created and completed, tokens, finish, error) and part (data.type step-start,
text, tool, step-finish, reasoning; a tool's state has status running, completed or error, input, output, metadata.exit
and time start and end in ms). This writes Claude Code's session records from them, as codex_rollout.py does for Codex,
so every scorer reads them unchanged:

  transcript/session/opencode/<session>.jsonl                           a root session (parent_id empty)
  transcript/session/opencode/<root session>/subagents/<session>.jsonl  a session with a parent_id (Claude Code's layout).
                                                                        No recorded session has one, so this is unexercised

The mapping:
  model response  each assistant message is one model request (one agent step): message.id the message's id, model its
                  modelID; each text, reasoning (thinking) and tool part is one assistant record at its own time, one per
                  block as Claude Code writes them; the message's last record carries stop_reason tool_use when its
                  finish is tool-calls
  tool names      bash is Bash (timeout kept; OpenCode has no background option); read is Read, write is Write, edit is
                  Edit (filePath as file_path, oldString and newString as old_string and new_string); task is Task;
                  anything else keeps its name and input
  times           a tool call starts at state.time.start and its result is at state.time.end, the tool's own ms clock
                  (stdout only reports a tool when it ends); a text part at its time.start, else the part row's time
  tool result     each tool part with status completed or error. A bash exit other than 0 is written "Exit code N" with
                  is_error, as Claude Code writes one. A command OpenCode's timeout ended (its <shell_metadata> "terminated
                  command after exceeding timeout N ms", exit null) has toolUseResult.timedOutAfterMs N with no
                  backgroundTaskId, as Claude Code's timed-out-and-killed Bash. A tool still running when the database was
                  copied (the run stopped with it in flight, o2) gets no result, as idle.py expects of a call in flight
                  at the stop
  user input      each user message (the brief; OpenCode joins its text parts)
  turn end        a turn ends at a step-finish with reason stop (the agent's final answer), and exactly one assistant record
                  per turn end carries stop_reason end_turn: the turn's last record. OpenCode has no wake: the run ends with
                  the turn and nothing starts a later one (o3: the job ran on 20 min after OpenCode exited), as codex_rollout
                  returns Codex's task_complete. A step whose message carries an error other than MessageAbortedError (an
                  APIError, a ProviderAuthError, a ContextOverflowError...) ends the turn on a failed model request: its end
                  record has top-level is_error true and isApiErrorMessage, as Claude Code's stream result marks
                  result.is_error. MessageAbortedError is a stop from outside, like a step with no finish (the o2 stop):
                  neither is a turn end. The error names are read from the 1.19.4 binary; no recorded session has one

Also returned, for labels and counting: each session's turns (start, end, stop or error, the steps in it), turn_ends,
wakes (always empty), tools still running at the end, Bash calls with a plain `&` (nohup or not: such a process outlives
OpenCode, o3), the session's models and its title. And the ledger's side calls (`side_calls`, from the run's ledger.jsonl or a
known answer's ledger.tsv): OpenCode makes one title request per root session, on the run's own model (with no small_model
configured it falls back to the main model, so a model filter cannot separate it), sent in parallel with the first step.
Every step is one ledger row with the step's own token counts, so a row is an agent step when its tokens equal a step's
(input + cache_read + cache_write = the message's input + cache read + cache write, output equal) and it starts within 2 s
of the step; the step cut off by a stop has zeros and takes the remaining row nearest its start. The title row is then
the one left that starts within 3 s of the session's start; the process log (log/opencode.log beside the database) names
its start ("message=stream ... agent=title small=true"), which is checked when it was kept.
"""
import argparse, datetime, json, os, re, sqlite3, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codex_rollout import _ampersand_launch
from grok_updates import read_ledger, utc

TOOLS = {"bash": "Bash", "read": "Read", "write": "Write", "edit": "Edit", "task": "Task"}
RENAME = {"filePath": "file_path", "oldString": "old_string", "newString": "new_string", "replaceAll": "replace_all"}
TIMEOUT_RE = re.compile(r"terminated command after exceeding timeout (\d+) ms")
ABORTS = ("MessageAbortedError",)
TITLE_LOG_RE = re.compile(r"^timestamp=(\S+) .*message=stream .*session\.id=(\S+) .*agent=title")

def map_tool(raw, inp):
    inp = dict(inp or {})
    name = TOOLS.get(raw, raw)
    if raw in ("read", "write", "edit"):
        return name, {RENAME.get(k, k): v for k, v in inp.items()}
    if raw == "task":
        return name, {"description": inp.get("description"), "prompt": inp.get("prompt"),
                      "subagent_type": inp.get("subagent_type") or "opencode"}
    return name, inp

def _result(t, cid, raw, st):
    meta = st.get("metadata") if isinstance(st.get("metadata"), dict) else {}
    body = st.get("output") if st.get("status") == "completed" else st.get("error")
    body, err, tur = str(body or ""), st.get("status") == "error", {}
    if raw == "bash":
        code = meta.get("exit")
        m = TIMEOUT_RE.search(body)
        if m:
            tur["timedOutAfterMs"] = int(m.group(1))
        if isinstance(code, int) and code != 0:
            body, err = f"Exit code {code}\n{body}", True
        tur.update(stdout=str(meta.get("output") or ""), stderr="", interrupted=False)
    rec = {"type": "user", "timestamp": utc(t), "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": cid, "content": body, "is_error": err}]}}
    if tur:
        rec["toolUseResult"] = tur
    return rec

def _assistant(t, mid, model, block):
    return {"type": "assistant", "timestamp": utc(t), "message": {"id": mid, "role": "assistant", "model": model,
                                                                 "stop_reason": None, "content": [block]}}

def normalise_session(con, s):
    sid = s["id"]
    msgs = [(mid, tc, json.loads(d)) for mid, tc, d in
            con.execute("SELECT id, time_created, data FROM message WHERE session_id = ? ORDER BY time_created, id", (sid,))]
    parts = {}
    for pid, mid, tc, d in con.execute("SELECT id, message_id, time_created, data FROM part WHERE session_id = ? "
                                       "ORDER BY time_created, id", (sid,)):
        parts.setdefault(mid, []).append((tc, json.loads(d)))
    out, steps, running, calls = [], [], [], {}
    for mid, tc, m in msgs:
        ps = parts.get(mid, [])
        if m.get("role") == "user":
            text = "\n".join(p.get("text") or "" for _, p in ps if p.get("type") == "text" and not p.get("synthetic"))
            out.append({"type": "user", "timestamp": utc(tc / 1000), "message": {"role": "user", "content": text}})
            continue
        model, idx = m.get("modelID"), []
        for ptc, p in ps:
            ty, tt = p.get("type"), p.get("time") if isinstance(p.get("time"), dict) else {}
            if ty in ("text", "reasoning") and (p.get("text") or "").strip():
                k = "text" if ty == "text" else "thinking"
                out.append(_assistant((tt.get("start") or ptc) / 1000, mid, model, {"type": k, k: p["text"]}))
                idx.append(len(out) - 1)
            elif ty == "tool":
                st, raw = p.get("state") or {}, p.get("tool")
                stt = st.get("time") if isinstance(st.get("time"), dict) else {}
                name, inp = map_tool(raw, st.get("input"))
                cid, t0 = p.get("callID") or p.get("id"), (stt.get("start") or ptc) / 1000
                calls[cid] = {"name": name, "input": inp, "t": t0, "end": None}
                out.append(_assistant(t0, mid, model, {"type": "tool_use", "id": cid, "name": name, "input": inp}))
                idx.append(len(out) - 1)
                if st.get("status") in ("completed", "error"):
                    t1 = (stt.get("end") or ptc) / 1000
                    calls[cid]["end"] = t1
                    out.append(_result(t1, cid, raw, st))
                else:
                    running.append({"call_id": cid, "tool": name, "input": inp, "start_utc": utc(t0)})
        sf = [p for _, p in ps if p.get("type") == "step-finish"]
        mt = m.get("time") or {}
        err = m.get("error") if isinstance(m.get("error"), dict) else None
        steps.append({"message_id": mid, "start": mt.get("created", tc) / 1000,
                      "end": (mt.get("completed") or (max(p[0] for p in ps) if ps else tc)) / 1000,
                      "finish": m.get("finish") or (sf[-1].get("reason") if sf else None), "model": model,
                      "tokens": m.get("tokens") or {}, "error": err, "records": idx})
    # turn ends: a step that finished with reason stop, or failed on a model request
    turn_ends, turns, first = [], [], None
    for st in steps:
        first = st if first is None else first
        failed = st["error"] and st["error"].get("name") not in ABORTS
        if st["finish"] == "stop" or failed:
            if failed:
                d = st["error"].get("data") if isinstance(st["error"].get("data"), dict) else {}
                out.append({"type": "assistant", "timestamp": utc(st["end"]), "is_error": True, "isApiErrorMessage": True,
                            "message": {"id": f"{st['message_id']}:error", "role": "assistant", "model": st["model"],
                                        "stop_reason": "end_turn", "content": [{"type": "text", "text":
                                            f"API Error: {st['error'].get('name')}: {d.get('message') or ''}"[:500]}]}})
            elif st["records"]:
                out[st["records"][-1]]["message"]["stop_reason"] = "end_turn"
            else:
                out.append({"type": "assistant", "timestamp": utc(st["end"]), "message": {
                    "id": f"{st['message_id']}:end", "role": "assistant", "model": st["model"], "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": ""}]}})
            turn_ends.append(utc(st["end"]))
            turns.append({"start_utc": utc(first["start"]), "end_utc": utc(st["end"]), "steps": steps.index(st) - steps.index(first) + 1,
                          "is_error": bool(failed), "error": st["error"].get("name") if failed else None})
            first = None
        elif st["finish"] == "tool-calls" and st["records"]:
            out[st["records"][-1]]["message"]["stop_reason"] = "tool_use"
    out.sort(key=lambda x: (datetime.datetime.fromisoformat(x["timestamp"].replace("Z", "+00:00")), x["type"] == "user"))
    amp = [{"call_id": cid, "utc": utc(c["t"]), "call_end_utc": utc(c["end"]) if c["end"] else None,
            "command": str(c["input"].get("command"))[:300]}
           for cid, c in calls.items() if c["name"] == "Bash" and _ampersand_launch(str(c["input"].get("command") or ""))]
    return {"session": sid, "subagent": bool(s.get("parent_id")), "parent_id": s.get("parent_id"), "title": s.get("title"),
            "start_epoch": s["time_created"] / 1000, "records": out, "steps": steps, "turns": turns, "turn_ends": turn_ends,
            "wakes": [], "running_at_end": running, "ampersand_launches": amp,
            "models": sorted({st["model"] for st in steps if st["model"]})}

def classify_ledger(ledger, threads, log_path=None):
    """Which ledger rows are OpenCode's agent steps and which its side calls (the module docstring's rule)."""
    ctx = lambda r: (r.get("input_tokens") or 0) + (r.get("cache_read") or 0) + (r.get("cache_write") or 0)
    used, agent, unmatched = set(), {}, []
    steps = [st for th in threads for st in th["steps"]]
    for st in steps:
        tk = st["tokens"]
        want = (tk.get("input") or 0) + ((tk.get("cache") or {}).get("read") or 0) + ((tk.get("cache") or {}).get("write") or 0)
        r = next((r for r in ledger if r["id"] not in used and abs(r["ts_epoch"] - st["start"]) <= 2 and want
                  and ctx(r) == want and r.get("output_tokens") == tk.get("output")), None)
        if r:
            used.add(r["id"]); agent[st["message_id"]] = r["id"]
        else:
            unmatched.append(st)
    for st in unmatched:      # the step cut off by a stop (no tokens), or a row whose tokens disagree: nearest start
        r = min((r for r in ledger if r["id"] not in used and abs(r["ts_epoch"] - st["start"]) <= 3),
                key=lambda r: abs(r["ts_epoch"] - st["start"]), default=None)
        if r:
            used.add(r["id"]); agent[st["message_id"]] = r["id"]
    logged = {}
    if log_path and os.path.exists(log_path):
        for line in open(log_path, errors="replace"):
            m = TITLE_LOG_RE.match(line)
            if m:
                logged.setdefault(m.group(2), datetime.datetime.fromisoformat(m.group(1).replace("Z", "+00:00")).timestamp())
    side = []
    for r in ledger:
        if r["id"] in used:
            continue
        th = min((th for th in threads if not th["subagent"]), key=lambda th: abs(r["ts_epoch"] - th["start_epoch"]), default=None)
        lt = logged.get(th["thread"]) if th else None
        if th and abs(r["ts_epoch"] - th["start_epoch"]) <= 3:
            why = (f"session title for {th['thread']} (\"{th['title']}\"): the one row left after every agent step took "
                   f"its own row by exact tokens, started {r['ts_epoch'] - th['start_epoch']:+.1f} s from the session's "
                   f"start, on the run's own model {r.get('requested_model') or r.get('model')} ({ctx(r)} input, "
                   f"{r.get('output_tokens')} output tokens)")
            why += (f"; the process log names the title stream at {utc(lt)}" if lt is not None and abs(lt - r["ts_epoch"]) <= 2
                    else "; no process-log title line matched" if log_path else "")
            side.append({"ledger_id": r["id"], "ts": r.get("ts"), "model": r.get("requested_model") or r.get("model"),
                         "kind": "title", "why": why})
        else:
            side.append({"ledger_id": r["id"], "ts": r.get("ts"), "model": r.get("requested_model") or r.get("model"),
                         "kind": "unmatched", "why": "a row matching no agent step, away from any session start"})
    return side, sorted(agent.values()), {"steps": len(steps), "steps_matched": len(agent),
                                           "rows": len(ledger), "title_logged": bool(logged)}

def db_path(run_dir):
    for p in (os.path.join(run_dir, "transcript", "opencode-data", "opencode.db"), os.path.join(run_dir, "opencode-data", "opencode.db")):
        if os.path.exists(p):
            return p
    return None

def normalise(run_dir, db=None):
    """Write transcript/session/opencode/ from the session database; return what the labels need. None when there is none."""
    db = db or db_path(run_dir)
    if not db:
        return None
    con = sqlite3.connect(f"file:{os.path.abspath(db)}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    sessions = [dict(r) for r in con.execute("SELECT id, parent_id, title, time_created FROM session ORDER BY time_created")]
    if not sessions:
        return None
    ns = [normalise_session(con, s) for s in sessions]
    con.close()
    by_id = {n["session"]: n for n in ns}
    def root_of(n):
        seen = set()
        while n.get("parent_id") and n["parent_id"] in by_id and n["session"] not in seen:
            seen.add(n["session"]); n = by_id[n["parent_id"]]
        return n["session"]
    base = os.path.join(run_dir, "transcript", "session", "opencode")
    threads = []
    for n in ns:
        dest = (os.path.join(base, root_of(n), "subagents", f"{n['session']}.jsonl") if n["subagent"]
                else os.path.join(base, f"{n['session']}.jsonl"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as fh:
            for r in n["records"]:
                fh.write(json.dumps(r) + "\n")
        threads.append({"thread": n["session"], **{k: n[k] for k in n if k not in ("records", "session")},
                        "records": len(n["records"]), "file": os.path.relpath(dest, run_dir)})
    ledger = read_ledger(run_dir)
    log = os.path.join(os.path.dirname(os.path.abspath(db)), "log", "opencode.log")
    side, agent, check = classify_ledger(ledger, threads, log) if ledger else ([], [], None)
    for th in threads:
        for st in th["steps"]:
            st["start_utc"], st["end_utc"] = utc(st.pop("start")), utc(st.pop("end"))
    return {"threads": threads, "models": sorted({m for th in threads for m in th["models"]}),
            "turn_ends": sorted(t for th in threads if not th["subagent"] for t in th["turn_ends"]),
            "agent_end": max((t for th in threads if not th["subagent"] for t in th["turn_ends"]), default=None),
            "wakes": [], "running_at_end": [dict(x, thread=th["thread"]) for th in threads for x in th["running_at_end"]],
            "ampersand_launches": [dict(a, thread=th["thread"]) for th in threads for a in th["ampersand_launches"]],
            "side_calls": side, "agent_calls": agent, "agent_calls_check": check,
            "how": "OpenCode's session database (transcript/opencode-data/opencode.db) written as Claude-shaped session "
                   "records under transcript/session/opencode/ (opencode_db.py), so the scorers read them unchanged"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--db", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = normalise(a.run_dir, a.db)
    if r is None:
        sys.exit("no OpenCode session database found")
    if a.json:
        print(json.dumps(r, indent=1)); return
    for th in r["threads"]:
        print(f"{'subagent' if th['subagent'] else 'root':8s} {th['thread']}  {len(th['steps'])} steps, {th['records']} records, "
              f"{len(th['turn_ends'])} turn ends, {len(th['running_at_end'])} running at end -> {th['file']}")
    print(f"models {r['models']}  agent end {r['agent_end']}  side calls {[(s['ledger_id'], s['kind']) for s in r['side_calls']]}  "
          f"agent calls {len(r['agent_calls'])}  launches with a plain &: {len(r['ampersand_launches'])}")

if __name__ == "__main__":
    main()
