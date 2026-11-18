"""Grok Build sessions as the transcript records the scorers already read.

usage: python3 -I grok_updates.py <run_dir> [--sessions <dir>] [--json]

Grok Build (xAI's `grok` CLI, 1.0.70) keeps one directory per session under $HOME/.grok/sessions/<url-encoded cwd>/<session
uuid>/ (the run directory keeps it as transcript/grok-home/sessions/, or grok-home/sessions/): updates.jsonl, the ACP session
updates, each with a whole-second `timestamp` and, in params._meta.agentTimestampMs, the harness's own ms time; events.jsonl,
phase events with ms ISO times (turn_started, tool_started, tool_completed with tool_call_id, turn_ended); terminal/<call
id>.log, each shell call's output. $HOME/.grok/logs/unified.jsonl (kept beside sessions/ as logs/unified.jsonl) is the
process log, with an inference start and end per agent model request. This writes Claude Code's session records from them,
as codex_rollout.py does for Codex, so every scorer reads them unchanged:

  transcript/session/grok/<session>.jsonl                           a primary session
  transcript/session/grok/<root session>/subagents/<session>.jsonl  a session whose events.jsonl turn_started has a
                                                                    session_relationship other than "primary" (Claude Code's
                                                                    layout). Grok recorded no subagent in any of the 14
                                                                    controller sessions, so this path is unexercised by real data

The mapping:
  model response  the updates of one model stream (params._meta promptId and streamStartMs) are one response, message.id
                  "<promptId>:<streamStartMs>", message.model the turn's model (turn_started model_id); a run of
                  agent_message_chunk is one text block, agent_thought_chunk a thinking block, tool_call a tool_use block,
                  one record per block as Claude Code writes them; a response's last record carries stop_reason tool_use
                  when it called a tool
  tool names      run_terminal_command is Bash, block_until_ms 0 as run_in_background true and block_until_ms > 0 as timeout
                  (in both harnesses a command still running at that limit goes to the background: k1); read_file is Read
                  (target_file as file_path), write is Write, search_replace is Edit, spawn_subagent is Task (background as
                  run_in_background); get_command_or_subagent_output is TaskOutput with timeout_ms as timeout and block true,
                  because that is Grok's own variant name for it and Claude Code's blocking read (wait_style counts its
                  timeout); anything else keeps its name and input
  tool result     each tool_call_update with status completed or failed. A shell result's leading "exit: N" line is written
                  as Claude Code writes an exit: dropped for 0, "Exit code N" with is_error for any other. toolUseResult has
                  backgroundTaskId for a call that put its command in the background, and timedOutAfterMs too when it went
                  there because it outlived block_until_ms (k1: the 30-s default), as Claude Code's timed-out-and-
                  backgrounded Bash does
  times           a tool call starts at its events.jsonl tool_started (the n-th of its tool name, when within 2 s) and its
                  result is at tool_completed (by tool_call_id); otherwise agentTimestampMs; the whole-second `timestamp`
                  only when neither exists. updates.jsonl's own timestamp runs up to 5 s late (k1), so it is never preferred
  user input      each user prompt (user_message_chunk runs, by promptIndex). A harness wake, a synthetic prompt whose turn
                  has prompt_id task-completed-<task id> (or bash-, monitor-, subagent-completed-), or whose text names a
                  finished background task, is written as Claude Code's task notification: a user record whose first text
                  block starts "<task-notification>" with the task id, the tool-use id that started the task and its status,
                  then Grok's own text unchanged, so wait_style reads it as woken_by_harness and not as a user turn
  turn end        each turn_completed is one turn end, and exactly one assistant record per turn end carries stop_reason
                  end_turn: the turn's last assistant record (the agent's answer). A turn that ended on a failed model
                  request gets instead a synthetic end record at the turn's end, with top-level is_error true and
                  isApiErrorMessage, as Claude Code's stream result marks result.is_error; it is a failed request when
                  turn_completed carries error_kind or error, events.jsonl turn_ended has outcome error or failed, or the
                  process log's last inference event in the turn is shell.turn.inference_failed. Those field names are read
                  from the 1.0.70 binary; no recorded session has a failed request, so the rule is untested on a real one.
                  A turn with stop_reason cancelled (stopped from outside) is no turn end, as an interrupted Claude Code
                  turn writes none

Also returned, for labels and counting: each session's turns (prompt id, start, end, stop_reason, the harness's own
modelCalls, model, woken), turn_ends, wakes, background tasks (task_backgrounded, background_tasks and task_completed:
start, end, exit, signal, explicitly_killed, will_wake, how it got there), Bash calls with a plain `&` (codex_rollout's
test), and the session's models. And the ledger's side calls (`side_calls`, from the run's ledger.jsonl, or a known
answer's ledger.tsv): Grok sends, besides the agent's own requests, session-title requests to its session_summary model
(grok-4.6: two at a session's first user prompt, none at a wake, k2) and one turn-summary request on the agent's model as
each turn ends (often a 499 at exit). A title call is any row not on the agent's model. Agent calls are the rows that match
the process log's inference calls, by exact tokens (prompt_tokens = input + cache_read + cache_write, completion_tokens =
output, start within 2 s) and then by start time; a remaining agent-model row that starts within 2 s before to 15 s after a
turn end is that turn's summary. Without the process log, each turn's first modelCalls agent-model rows are its agent
calls and the rest its summary.
"""
import argparse, datetime, glob, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codex_rollout import _ampersand_launch

TOOLS = {"run_terminal_command": "Bash", "read_file": "Read", "write": "Write", "search_replace": "Edit",
         "get_command_or_subagent_output": "TaskOutput", "spawn_subagent": "Task"}
WAKE_PROMPT_RE = re.compile(r"^(?:task|bash|monitor|subagent)-completed-(.+)$")
WAKE_TEXT_RE = re.compile(r'Background task "([^"]+)" (completed|failed|was killed|killed|timed out)')
BLOCKED_RE = re.compile(r"still running after (\d+(?:\.\d+)?)s \(block_until_ms\)")
EXIT_RE = re.compile(r"\Aexit: (-?\d+)\n?")
DEFAULT_BLOCK_MS = 30000           # run_terminal_command's block_until_ms default ("still running after 30s", k1)
INFER = ("shell.turn.inference_start", "shell.turn.inference_done", "shell.turn.inference_failed")

def utc(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

def wall(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

def _jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, errors="replace"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict):
            out.append(r)
    return out

def read_ledger(run_dir):
    """The run's the router ledger rows (the runner's ledger.jsonl, or a known answer's ledger.tsv), by request start."""
    rows = _jsonl(os.path.join(run_dir, "ledger.jsonl"))
    tsv = os.path.join(run_dir, "ledger.tsv")
    if not rows and os.path.exists(tsv):
        lines = open(tsv).read().splitlines()
        rows = [dict(zip(lines[0].split("\t"), ln.split("\t"))) for ln in lines[1:]] if lines else []
    for r in rows:
        if r.get("ts_epoch") in (None, ""):
            r["ts_epoch"] = datetime.datetime.strptime(r["ts"], "%Y-%m-%dT%H:%M:%S%z").timestamp()
        r["ts_epoch"] = float(r["ts_epoch"])
        for k in ("id", "input_tokens", "cache_read", "cache_write", "output_tokens"):
            try:
                r[k] = int(r.get(k) or 0)
            except (TypeError, ValueError):
                pass
    return sorted(rows, key=lambda r: (r["ts_epoch"], r["id"]))

def map_tool(raw, inp):
    """(Claude Code name, input) for one Grok tool call."""
    inp = dict(inp or {})
    name = TOOLS.get(raw, raw)
    if raw == "run_terminal_command":
        out = {k: v for k, v in inp.items() if k != "block_until_ms"}
        b = inp.get("block_until_ms")
        if b == 0:
            out["run_in_background"] = True
        elif isinstance(b, (int, float)) and b > 0:
            out["timeout"] = b
        return name, out
    if raw == "read_file":
        return name, {"file_path": inp.pop("target_file", None), **inp}
    if raw == "get_command_or_subagent_output":
        ids = inp.get("task_ids") or []
        out = {"task_id": ids[0] if ids else None, "task_ids": ids, "block": True}
        if inp.get("timeout_ms") is not None:
            out["timeout"] = inp["timeout_ms"]
        return name, out
    if raw == "spawn_subagent":
        return name, {"description": inp.get("description"), "prompt": inp.get("prompt"), "subagent_type": "grok",
                      "run_in_background": bool(inp.get("background"))}
    return name, inp

def _result(t, cid, u, call):
    """The tool_result user record for one finished tool call."""
    ro = u.get("rawOutput") if isinstance(u.get("rawOutput"), dict) else {}
    body = "\n".join(c.get("content", {}).get("text", "") for c in u.get("content") or []
                     if isinstance(c, dict) and isinstance(c.get("content"), dict))
    err, tur = u.get("status") == "failed", {}
    typ = ro.get("type")
    if typ == "Bash":
        body = ro.get("output_for_prompt") if isinstance(ro.get("output_for_prompt"), str) else body
        m = EXIT_RE.match(body or "")
        code = ro.get("exit_code") if ro.get("exit_code") is not None else (int(m.group(1)) if m else None)
        rest = body[m.end():] if m else body
        body = rest if not code else f"Exit code {code}\n{rest}"
        err = err or bool(code)
        tur = {"stdout": rest, "stderr": "", "interrupted": False}
        if ro.get("timed_out"):
            tur["timedOutAfterMs"] = ((call or {}).get("input") or {}).get("block_until_ms") or DEFAULT_BLOCK_MS
    elif typ == "BackgroundTaskStarted":
        tur = {"backgroundTaskId": ro.get("task_id")}
        if ((call or {}).get("input") or {}).get("block_until_ms") != 0:
            m = BLOCKED_RE.search(ro.get("summary") or body or "")
            tur["timedOutAfterMs"] = int(float(m.group(1)) * 1000) if m else (
                ((call or {}).get("input") or {}).get("block_until_ms") or DEFAULT_BLOCK_MS)
        body = body or ro.get("summary") or ""
    elif typ == "TaskOutput":
        res = ro.get("Result") if isinstance(ro.get("Result"), dict) else {}
        tur = {"retrieval_status": "success" if res.get("status") in ("completed", "failed", "killed") else res.get("status"),
               "task": {"task_id": res.get("task_id"), "status": res.get("status"), "exitCode": res.get("exit_code")}}
        body = body or str(res.get("output") or "")
    rec = {"type": "user", "timestamp": utc(t), "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": cid, "content": body or "", "is_error": err}]}}
    if tur:
        rec["toolUseResult"] = tur
    return rec

def _assistant(t, rid, model, block):
    return {"type": "assistant", "timestamp": utc(t), "message": {"id": rid, "role": "assistant", "model": model,
                                                                 "stop_reason": None, "content": [block]}}

def inference_calls(sessions):
    """The process log's agent model requests per session id: [{start, end, prompt_tokens, completion_tokens, failed}].
    None when the log was not kept."""
    p = os.path.join(os.path.dirname(os.path.abspath(sessions)), "logs", "unified.jsonl")
    if not os.path.exists(p):
        return None
    out = {}
    for r in _jsonl(p):
        if r.get("msg") not in INFER or not r.get("ts"):
            continue
        calls, t, ctx = out.setdefault(r.get("sid"), []), wall(r["ts"]), r.get("ctx") or {}
        if r["msg"] == "shell.turn.inference_start":
            calls.append({"start": t, "end": None, "prompt_tokens": None, "completion_tokens": None, "failed": False})
            continue
        if not calls or calls[-1]["end"] is not None:
            # a retry's end with no start of its own: a request from the previous end to this one
            calls.append({"start": calls[-1]["end"] if calls else t, "end": None, "prompt_tokens": None,
                          "completion_tokens": None, "failed": False})
        calls[-1].update(end=t, failed=r["msg"] == "shell.turn.inference_failed",
                         prompt_tokens=ctx.get("prompt_tokens"), completion_tokens=ctx.get("completion_tokens"))
    return out

def normalise_one(sd, infer=None):
    ups, evs = _jsonl(os.path.join(sd, "updates.jsonl")), _jsonl(os.path.join(sd, "events.jsonl"))
    try:
        summary = json.load(open(os.path.join(sd, "summary.json")))
    except (OSError, ValueError):
        summary = {}
    sid = (summary.get("info") or {}).get("id") or os.path.basename(os.path.normpath(sd))
    t_started = [e for e in evs if e.get("type") == "turn_started"]
    t_ended = [e for e in evs if e.get("type") == "turn_ended"]
    rel = next((e.get("session_relationship") for e in t_started if e.get("session_relationship")), "primary")
    started = {}
    for e in evs:
        if e.get("type") == "tool_started" and e.get("ts"):
            started.setdefault(e.get("tool_name"), []).append(wall(e["ts"]))
    finished = {e["tool_call_id"]: wall(e["ts"]) for e in evs if e.get("type") == "tool_completed" and e.get("tool_call_id")}

    def when(r):
        ms = ((r.get("params") or {}).get("_meta") or {}).get("agentTimestampMs")
        return ms / 1000.0 if isinstance(ms, (int, float)) else float(r.get("timestamp") or 0)
    out, prompts, completions, calls, done, tasks, nth = [], [], [], {}, set(), {}, {}
    last = None                     # (response id, block kind, index in out) of the last block, to join streamed chunks
    pid_of = {}                     # index in out -> prompt id, for each assistant record
    for r in ups:
        t, p = when(r), r.get("params") or {}
        u, pm = p.get("update") or {}, p.get("_meta") or {}
        k = u.get("sessionUpdate")
        if pm.get("promptId") and prompts and prompts[-1]["pid"] is None:
            prompts[-1]["pid"] = pm["promptId"]
        if k == "user_message_chunk":
            um = u.get("_meta") or {}
            idx, text = um.get("promptIndex", len(prompts)), (u.get("content") or {}).get("text") or ""
            if prompts and prompts[-1]["idx"] == idx:
                prompts[-1]["text"] += text
            else:
                prompts.append({"t": t, "idx": idx, "text": text, "synthetic": bool(um.get("syntheticPrompt")),
                                "model": um.get("modelId"), "pid": None})
            last = None
        elif k in ("agent_message_chunk", "agent_thought_chunk"):
            rid, kind = f"{pm.get('promptId')}:{pm.get('streamStartMs')}", "text" if k == "agent_message_chunk" else "thinking"
            text = (u.get("content") or {}).get("text") or ""
            if last and last[:2] == (rid, kind):
                out[last[2]]["message"]["content"][0][kind] += text
            else:
                out.append(_assistant(t, rid, None, {"type": kind, kind: text}))
                pid_of[len(out) - 1], last = pm.get("promptId"), (rid, kind, len(out) - 1)
        elif k == "tool_call":
            raw = ((u.get("_meta") or {}).get("x.ai/tool") or {}).get("name") or u.get("title")
            name, inp = map_tool(raw, u.get("rawInput"))
            q, i = started.get(raw) or [], nth.get(raw, 0)
            t0 = q[i] if i < len(q) and abs(q[i] - t) <= 2 else t
            nth[raw] = i + 1 if i < len(q) and t0 == q[i] else i
            cid = u.get("toolCallId")
            calls[cid] = {"raw": raw, "input": u.get("rawInput") or {}, "name": name, "mapped": inp, "t": t0}
            out.append(_assistant(t0, f"{pm.get('promptId')}:{pm.get('streamStartMs')}", None,
                                  {"type": "tool_use", "id": cid, "name": name, "input": inp}))
            pid_of[len(out) - 1], last = pm.get("promptId"), None
        elif k == "tool_call_update" and u.get("status") in ("completed", "failed") and u.get("toolCallId") not in done:
            cid = u["toolCallId"]
            done.add(cid)
            calls.setdefault(cid, {})["end"] = finished.get(cid, t)
            out.append(_result(calls[cid]["end"], cid, u, calls.get(cid)))
            last = None
        elif k == "task_backgrounded":
            tk = tasks.setdefault(u.get("task_id"), {"task_id": u.get("task_id")})
            tk.update(tool_call_id=u.get("tool_call_id"), command=u.get("command"), description=u.get("description"),
                      backgrounded_utc=utc(t))
        elif k == "background_tasks":
            for x in u.get("tasks") or []:
                tk = tasks.setdefault(x.get("task_id"), {"task_id": x.get("task_id")})
                tk.setdefault("command", x.get("command"))
                tk["kind"], tk["status"] = x.get("kind"), x.get("status")
                if x.get("started_at"):
                    tk.setdefault("start_utc", utc(wall(x["started_at"])))
                if x.get("ended_at"):
                    tk["end_utc"] = utc(wall(x["ended_at"]))
        elif k == "task_completed":
            s = u.get("task_snapshot") or {}
            tk = tasks.setdefault(s.get("task_id"), {"task_id": s.get("task_id")})
            for key, field in (("start_utc", "start_time"), ("end_utc", "end_time")):
                v = s.get(field) or {}
                if v.get("secs_since_epoch") is not None:
                    tk[key] = utc(v["secs_since_epoch"] + v.get("nanos_since_epoch", 0) / 1e9)
            tk.update(exit_code=s.get("exit_code"), signal=s.get("signal"), explicitly_killed=s.get("explicitly_killed"),
                      block_waited=s.get("block_waited"), will_wake=u.get("will_wake"), kind=s.get("kind") or tk.get("kind"))
        elif k == "turn_completed":
            completions.append((t, u))
            if prompts and prompts[min(len(completions), len(prompts)) - 1]["pid"] is None:
                prompts[min(len(completions), len(prompts)) - 1]["pid"] = u.get("prompt_id")

    # turns: the i-th user prompt starts the i-th turn (turn_started, turn_completed and turn_ended, in order)
    turns, sess_calls = [], (infer or {}).get(sid) or []
    for i, p in enumerate(prompts):
        tc_t, tc = completions[i] if i < len(completions) else (None, None)
        ts_, te = (t_started[i] if i < len(t_started) else {}), (t_ended[i] if i < len(t_ended) else {})
        start = wall(ts_["ts"]) if ts_.get("ts") else p["t"]
        end = wall(te["ts"]) if te.get("ts") else tc_t
        pid = (tc or {}).get("prompt_id") or p["pid"]
        err = (tc or {}).get("error_kind") or (tc or {}).get("error") or te.get("error_category") or (
            te.get("outcome") if te.get("outcome") in ("error", "failed") else None)
        if not err and end is not None:
            inside = [c for c in sess_calls if c["end"] is not None and start - 1 <= c["end"] <= end + 1]
            if inside and inside[-1]["failed"]:
                err = "inference_failed (process log)"
        turns.append({"prompt_id": pid, "start_utc": utc(start), "end_utc": utc(end) if end is not None else None,
                      "start_epoch": start, "end_epoch": end, "stop_reason": (tc or {}).get("stop_reason"),
                      "model_calls": ((tc or {}).get("usage") or {}).get("modelCalls"), "is_error": bool(err),
                      "error": str(err) if err else None, "model": ts_.get("model_id") or p["model"],
                      "woken": False, "ended": tc is not None})
    model_of = {t["prompt_id"]: t["model"] for t in turns}
    for idx, pid in pid_of.items():
        out[idx]["message"]["model"] = model_of.get(pid) or next((t["model"] for t in turns if t["model"]), None)
    # a response's last record says why it stopped: tool_use when it called a tool
    last_of, has_tool = {}, set()
    for idx in sorted(pid_of):
        rid = out[idx]["message"]["id"]
        last_of[rid] = idx
        if out[idx]["message"]["content"][0].get("type") == "tool_use":
            has_tool.add(rid)
    for rid, idx in last_of.items():
        if rid in has_tool:
            out[idx]["message"]["stop_reason"] = "tool_use"
    # user prompts and harness wakes
    wakes = []
    for i, p in enumerate(prompts):
        pid = turns[i]["prompt_id"] if i < len(turns) else p["pid"]
        m, mt = WAKE_PROMPT_RE.match(pid or ""), WAKE_TEXT_RE.search(p["text"])
        tid = (m.group(1) if m else None) or (mt.group(1) if mt else None)
        if p["synthetic"] and tid:
            tk = tasks.get(tid) or {}
            status = ("killed" if tk.get("explicitly_killed") or tk.get("signal") else
                      "completed" if tk.get("exit_code") == 0 else "failed" if tk.get("exit_code") is not None else
                      {"was killed": "killed", "timed out": "failed"}.get(mt.group(2), mt.group(2)) if mt else "completed")
            note = (f"<task-notification>\n<task-id>{tid}</task-id>\n<tool-use-id>{tk.get('tool_call_id') or ''}</tool-use-id>\n"
                    f"<status>{status}</status>\n</task-notification>\n")
            out.append({"type": "user", "timestamp": utc(p["t"]), "promptSource": "system",
                        "origin": {"kind": "task-notification", "producer": "grok-build"},
                        "message": {"role": "user", "content": [{"type": "text", "text": note + p["text"]}]}})
            wakes.append({"utc": utc(p["t"]), "task_id": tid, "prompt_id": pid, "status": status})
            if i < len(turns):
                turns[i]["woken"] = True
        else:
            rec = {"type": "user", "timestamp": utc(p["t"]), "message": {"role": "user", "content": p["text"]}}
            if p["synthetic"]:
                rec["isMeta"] = True
            out.append(rec)
    # turn ends: exactly one end_turn record each
    turn_ends = []
    for tu in turns:
        if not tu["ended"] or tu["stop_reason"] == "cancelled" or tu["end_epoch"] is None:
            continue
        mine = [idx for idx, pid in pid_of.items() if pid == tu["prompt_id"]]
        if tu["is_error"]:
            out.append({"type": "assistant", "timestamp": tu["end_utc"], "is_error": True, "isApiErrorMessage": True,
                        "message": {"id": f"{tu['prompt_id']}:error", "role": "assistant", "model": tu["model"],
                                    "stop_reason": "end_turn",
                                    "content": [{"type": "text", "text": f"API Error: {tu['error']}"}]}})
        elif mine:
            out[max(mine)]["message"]["stop_reason"] = "end_turn"
        else:
            out.append({"type": "assistant", "timestamp": tu["end_utc"], "message": {
                "id": f"{tu['prompt_id']}:end", "role": "assistant", "model": tu["model"], "stop_reason": "end_turn",
                "content": [{"type": "text", "text": ""}]}})
        turn_ends.append(tu["end_utc"])
    out.sort(key=lambda x: wall(x["timestamp"]))
    for tk in tasks.values():
        c = calls.get(tk.get("tool_call_id")) or {}
        tk["how"] = ("block_until_ms 0" if (c.get("input") or {}).get("block_until_ms") == 0 else
                     "moved to the background after block_until_ms" if c else None)
    amp = [{"call_id": cid, "utc": utc(c["t"]), "call_end_utc": utc(c["end"]) if c.get("end") else None,
            "command": str(c["mapped"].get("command"))[:300]}
           for cid, c in calls.items() if c.get("name") == "Bash" and _ampersand_launch(str(c["mapped"].get("command") or ""))]
    return {"session": sid, "subagent": rel not in (None, "primary"), "relationship": rel, "cwd": (summary.get("info") or {}).get("cwd"),
            "records": out, "turns": turns, "turn_ends": turn_ends, "wakes": wakes,
            "background_tasks": sorted(tasks.values(), key=lambda x: x.get("backgrounded_utc") or x.get("start_utc") or ""),
            "ampersand_launches": amp, "models": sorted({t["model"] for t in turns if t["model"]}),
            "responses": len(last_of)}

def classify_ledger(ledger, threads, infer):
    """Which ledger rows are the agent's model requests and which are Grok's side calls (the module docstring's rule)."""
    agent_models = {t["model"] for th in threads for t in th["turns"] if t.get("model")}
    model = lambda r: r.get("requested_model") or r.get("model")
    am = [r for r in ledger if model(r) in agent_models]
    side, used = [], set()
    for r in ledger:
        if model(r) not in agent_models:
            side.append({"ledger_id": r["id"], "ts": r.get("ts"), "model": model(r), "kind": "title",
                         "why": f"session title: {model(r)} is Grok's session_summary model, not the agent's "
                                f"({', '.join(sorted(agent_models))}); title requests are capped at 100 output tokens "
                                f"(this one {r.get('output_tokens')}, status {r.get('status')})"})
    ctx = lambda r: (r.get("input_tokens") or 0) + (r.get("cache_read") or 0) + (r.get("cache_write") or 0)
    if infer is not None:
        how = "process log (logs/unified.jsonl): exact tokens, then start time"
        calls = [c for th in threads for c in infer.get(th["thread"]) or []]
        for c in calls:
            if c["prompt_tokens"] is None:
                continue
            r = next((r for r in am if r["id"] not in used and abs(r["ts_epoch"] - math.floor(c["start"])) <= 2
                      and ctx(r) == c["prompt_tokens"] and r.get("output_tokens") == c["completion_tokens"]), None)
            if r:
                used.add(r["id"]); c["ledger_id"] = r["id"]
        for c in calls:
            if c.get("ledger_id"):
                continue
            r = min((r for r in am if r["id"] not in used and abs(r["ts_epoch"] - c["start"]) <= 2),
                    key=lambda r: abs(r["ts_epoch"] - c["start"]), default=None)
            if r:
                used.add(r["id"]); c["ledger_id"] = r["id"]
    else:
        how = "no process log: each turn's first modelCalls agent-model rows, by request start"
        for th in threads:
            tu = th["turns"]
            for i, t in enumerate(tu):
                hi = tu[i + 1]["start_epoch"] - 1 if i + 1 < len(tu) else float("inf")
                lo = t["start_epoch"] - 1 if i else float("-inf")
                mine = [r for r in am if r["id"] not in used and lo <= r["ts_epoch"] < hi]
                used.update(r["id"] for r in mine[:t.get("model_calls") or 0])
    ends = sorted(t["end_epoch"] for th in threads for t in th["turns"] if t.get("end_epoch") is not None)
    for r in am:
        if r["id"] in used:
            continue
        te = max((e for e in ends if e - 2 <= r["ts_epoch"]), default=None)
        if te is not None and r["ts_epoch"] <= te + 15:
            side.append({"ledger_id": r["id"], "ts": r.get("ts"), "model": model(r), "kind": "turn_summary",
                         "why": f"turn summary for the turn that ended {utc(te)}: an agent-model request that is none of "
                                f"the agent's own ({how}), started as the turn ended (status {r.get('status')}, "
                                f"{ctx(r)} context tokens)"})
        else:
            side.append({"ledger_id": r["id"], "ts": r.get("ts"), "model": model(r), "kind": "unmatched",
                         "why": f"an agent-model request that is none of the agent's own ({how}) and starts at no turn end"})
    side.sort(key=lambda s: s["ledger_id"])
    agent = [r["id"] for r in am if r["id"] in used]
    n_harness = sum(t.get("model_calls") or 0 for th in threads for t in th["turns"])
    return side, agent, {"ledger_agent_calls": len(agent), "harness_model_calls": n_harness, "how": how}

def sessions_dir(run_dir):
    for p in (os.path.join(run_dir, "transcript", "grok-home", "sessions"), os.path.join(run_dir, "grok-home", "sessions")):
        if os.path.isdir(p):
            return p
    return None

def normalise(run_dir, sessions=None):
    """Write transcript/session/grok/ from the session directories; return what the labels need. None when there are none."""
    sessions = sessions or sessions_dir(run_dir)
    dirs = sorted({os.path.dirname(f) for f in glob.glob(os.path.join(sessions, "**", "updates.jsonl"), recursive=True)}) \
        if sessions else []
    if not dirs:
        return None
    infer = inference_calls(sessions)
    base = os.path.join(run_dir, "transcript", "session", "grok")
    ns = [normalise_one(d, infer) for d in dirs]
    roots = sorted((n for n in ns if not n["subagent"]), key=lambda n: n["turns"][0]["start_epoch"] if n["turns"] else 0)
    threads = []
    for n in ns:
        if n["subagent"]:
            # the primary session of the same working directory that started before it (Grok's own link is unrecorded)
            t0 = n["turns"][0]["start_epoch"] if n["turns"] else float("inf")
            root = next((r for r in reversed(roots) if r["cwd"] == n["cwd"] and (not r["turns"] or r["turns"][0]["start_epoch"] <= t0)),
                        roots[0] if roots else n)
            dest = os.path.join(base, root["session"], "subagents", f"{n['session']}.jsonl")
        else:
            dest = os.path.join(base, f"{n['session']}.jsonl")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as fh:
            for r in n["records"]:
                fh.write(json.dumps(r) + "\n")
        threads.append({"thread": n["session"], **{k: n[k] for k in n if k not in ("records", "session")},
                        "records": len(n["records"]), "file": os.path.relpath(dest, run_dir)})
    ledger = read_ledger(run_dir)
    side, agent, check = classify_ledger(ledger, threads, infer) if ledger else ([], [], None)
    return {"threads": threads, "models": sorted({m for th in threads for m in th["models"]}),
            "turn_ends": sorted(t for th in threads if not th["subagent"] for t in th["turn_ends"]),
            "wakes": [dict(w, thread=th["thread"]) for th in threads for w in th["wakes"]],
            "background_tasks": [dict(b, thread=th["thread"]) for th in threads for b in th["background_tasks"]],
            "ampersand_launches": [dict(a, thread=th["thread"]) for th in threads for a in th["ampersand_launches"]],
            "side_calls": side, "agent_calls": agent, "agent_calls_check": check,
            "how": "Grok Build sessions (transcript/grok-home/sessions) written as Claude-shaped session records under "
                   "transcript/session/grok/ (grok_updates.py), so the scorers read them unchanged"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--sessions", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = normalise(a.run_dir, a.sessions)
    if r is None:
        sys.exit("no Grok Build sessions found")
    if a.json:
        print(json.dumps(r, indent=1)); return
    for th in r["threads"]:
        print(f"{'subagent' if th['subagent'] else 'root':8s} {th['thread']}  {th['responses']} responses, {th['records']} "
              f"records, {len(th['turn_ends'])} turn ends, {len(th['wakes'])} wakes -> {th['file']}")
    print(f"models {r['models']}  background tasks {len(r['background_tasks'])}  side calls "
          f"{[(s['ledger_id'], s['kind']) for s in r['side_calls']]}  agent calls {len(r['agent_calls'])}")

if __name__ == "__main__":
    main()
