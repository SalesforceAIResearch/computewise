"""Codex rollouts as the transcript records the scorers already read.

usage: python3 -I codex_rollout.py <run_dir> [--sessions <dir>] [--json]

Codex writes one rollout per thread under $CODEX_HOME/sessions/ (the run directory keeps it as
transcript/codex-home/sessions/): timestamped records of the model's output items, the tool calls and their outputs,
and the events around them. The scorers (idle.py's activity, prep.py, tells.py, launches.py) read Claude Code's session
records. This writes the same records from the rollouts, so every scorer and its known answers apply unchanged:

  transcript/session/codex/<thread>.jsonl              the root agent
  transcript/session/codex/subagents/<thread>.jsonl    each subagent (session_meta thread_source "subagent"); the
                                                       parent history a subagent's rollout starts with (its records
                                                       before subagent_history_start_ordinal) is skipped

The mapping:
  model response    one per token_usage_record (response_id); each of its output items (an assistant message, a
                    reasoning summary, a tool call) is one assistant record at the item's time, all with message.id =
                    the response id, as Claude Code writes one record per content block
  exec cell         custom_tool_call "exec" (the code-mode host's JavaScript) is one Bash call whose command is what it
                    ran: each exec_command's cmd, "apply_patch <path>" per file a patch writes, and "wait session <id>"
                    for a write_stdin that sends nothing (a poll on a running command). Its result is the matching
                    custom_tool_call_output. It is run_in_background when a command it started (a CommandExecution item,
                    matched by its cmd) completed after the cell returned: a background terminal
  function call     spawn_agent is Task (a subagent); wait_agent and wait are waits (Bash "wait agent" / "wait cell <id>",
                    which idle.py's WAIT_RE reads as waiting); clock.sleep({duration_ms}) is `sleep <s>`; anything else
                    keeps its name and arguments
  user input        each UserMessage item (the brief, and any later user turn)

Also returned, for labels: the effort and model of every turn (turn_context), each turn's end (event_msg task_complete,
the time the agent handed back an answer: score_all's report_latency), and the exec cells that launched a process
with a plain `&` (nohup or not) outside a background terminal: in Codex such a process dies when its tool call ends
(the Codex driver's lifecycle note, 2026-10-21), so each is a launch killed at tool end.
"""
import argparse, glob, json, os, re, sys

JS_STR = r'"(?:[^"\\]|\\.)*"'
EXEC_RE = re.compile(r'exec_command\(\s*\{[^{}]*?"?cmd"?\s*:\s*(' + JS_STR + ')')
PATCH_RE = re.compile(r'apply_patch\(\s*(' + JS_STR + ')')
STDIN_RE = re.compile(r'write_stdin\(\s*\{[^{}]*?"?session_id"?\s*:\s*(\d+)[^{}]*?"?chars"?\s*:\s*(' + JS_STR + ')')
PATCH_FILE_RE = re.compile(r'^\*\*\* (?:Add|Update|Delete) File: (.+)$', re.M)
# a process put in the background with a plain & (not &&), the way a shell launch outlives nothing in Codex
AMP_RE = re.compile(r'(?<![&|>])&(?![&>])\s*(?:$|;|\n|\)|\s*wait\b)?')

def _js(s):
    try:
        return json.loads(s)
    except ValueError:
        return s.strip('"')

def cell_commands(js):
    """What one exec cell ran: [(kind, text)] with kind cmd, patch, poll."""
    out = []
    for m in EXEC_RE.finditer(js):
        out.append(("cmd", _js(m.group(1))))
    for m in PATCH_RE.finditer(js):
        for p in PATCH_FILE_RE.findall(_js(m.group(1))):
            out.append(("patch", p.strip()))
    for m in STDIN_RE.finditer(js):
        if _js(m.group(2)) == "":
            out.append(("poll", m.group(1)))
    return out

def _text(content):
    if isinstance(content, str):
        return content
    return "\n".join(str(x.get("text") or "") for x in content or [] if isinstance(x, dict))

def _ampersand_launch(cmd):
    """A command that puts a process in the background with a plain & (Codex kills it when the call ends)."""
    for line in cmd.splitlines():
        line = re.sub(r"'[^']*'|\"[^\"]*\"", "''", line)       # not an & inside quotes
        if re.search(r'(?<![&|>2])&(?![&>])', line.replace("&&", "")):
            return True
    return False

def normalise_one(path):
    rows = []
    for line in open(path, errors="replace"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and r.get("timestamp"):
            rows.append(r)
    meta = next((r["payload"] for r in rows if r.get("type") == "session_meta"), {}) or {}
    start = int(meta.get("subagent_history_start_ordinal") or 0)
    rows = [r for r in rows if int(r.get("ordinal") or 0) >= start or r.get("type") == "session_meta"]
    sub = meta.get("thread_source") == "subagent" or bool(meta.get("parent_thread_id"))
    model = None
    turns = []                       # (utc, model, effort)
    # a command's completion, from its CommandExecution item: the cmd it ran -> latest completion time
    cmd_done = {}
    for r in rows:
        p = r.get("payload") or {}
        if r["type"] == "event_msg" and p.get("type") == "item_completed" and (p.get("item") or {}).get("type") == "CommandExecution":
            it = p["item"]
            c = it.get("command")
            c = c[-1] if isinstance(c, list) and c else c
            if isinstance(c, str):
                cmd_done.setdefault(c, []).append(r["timestamp"])
    out, pending, outputs_at, calls = [], [], {}, {}
    for r in rows:
        if (r.get("payload") or {}).get("type") in ("custom_tool_call_output", "function_call_output"):
            outputs_at[r["payload"].get("call_id")] = r["timestamp"]
    amp, n_resp = [], 0
    # the thread's turn ends (event_msg task_complete): when the agent handed back an answer
    turn_ends = [r["timestamp"] for r in rows if r["type"] == "event_msg" and (r.get("payload") or {}).get("type") == "task_complete"]
    def flush(rid):
        nonlocal pending
        for t, block in pending:
            out.append({"type": "assistant", "timestamp": t, "message": {"id": rid, "role": "assistant", "model": model,
                                                                         "content": [block]}})
        pending = []
    for r in rows:
        t, p, typ = r["timestamp"], r.get("payload") or {}, r["type"]
        pt = p.get("type")
        if typ == "turn_context":
            model = p.get("model") or model
            turns.append({"utc": t, "model": p.get("model"), "effort": p.get("effort") or p.get("reasoning_effort")})
        elif typ == "event_msg" and pt == "item_completed" and (p.get("item") or {}).get("type") == "UserMessage":
            out.append({"type": "user", "timestamp": t, "message": {"role": "user", "content": _text(p["item"].get("content"))}})
        elif typ == "response_item" and pt == "message" and p.get("role") == "assistant":
            pending.append((t, {"type": "text", "text": _text(p.get("content"))}))
        elif typ == "response_item" and pt == "reasoning":
            s = "\n".join(x.get("text", "") for x in p.get("summary") or [] if isinstance(x, dict))
            if s:
                pending.append((t, {"type": "thinking", "thinking": s}))
        elif typ == "response_item" and pt == "custom_tool_call":
            js = p.get("input") or ""
            cmds = cell_commands(js) if p.get("name") == "exec" else []
            lines = [c if k == "cmd" else f"apply_patch {c}" if k == "patch" else f"wait session {c}" for k, c in cmds]
            end = outputs_at.get(p.get("call_id"))
            bg = any(k == "cmd" and any(d > end for d in cmd_done.get(c, [])) for k, c in cmds) if end else False
            for k, c in cmds:
                if k == "cmd" and _ampersand_launch(c) and not any(end and d > end for d in cmd_done.get(c, [])):
                    amp.append({"call_id": p.get("call_id"), "utc": t, "call_end_utc": end, "command": c[:300]})
            inp = {"command": "\n".join(lines) if lines else js[:2000]}
            if bg:
                inp["run_in_background"] = True
            calls[p.get("call_id")] = inp
            pending.append((t, {"type": "tool_use", "id": p.get("call_id"), "name": "Bash" if p.get("name") == "exec" else p.get("name"),
                                "input": inp}))
        elif typ == "response_item" and pt == "function_call":
            name = p.get("name")
            try:
                args = json.loads(p.get("arguments") or "{}")
            except ValueError:
                args = {"arguments": p.get("arguments")}
            if name == "spawn_agent":
                block = {"type": "tool_use", "id": p.get("call_id"), "name": "Task",
                         "input": {"description": args.get("task_name"), "subagent_type": "codex"}}
            elif name in ("wait_agent", "wait"):
                block = {"type": "tool_use", "id": p.get("call_id"), "name": "Bash",
                         "input": {"command": "wait agent" if name == "wait_agent" else f"wait cell {args.get('cell_id')}"}}
            elif name == "sleep" and (p.get("namespace") in (None, "clock")):
                # Codex's clock.sleep({duration_ms}) is a blind sleep, as `sleep N` is in a shell (X-prepP6-astra-L0: 16 of them)
                ms = args.get("duration_ms") or 0
                block = {"type": "tool_use", "id": p.get("call_id"), "name": "Bash",
                         "input": {"command": f"sleep {round(float(ms) / 1000.0, 3)}"}}
            else:
                block = {"type": "tool_use", "id": p.get("call_id"), "name": name, "input": args}
            pending.append((t, block))
        elif typ == "token_usage_record":
            n_resp += 1
            flush(p.get("response_id") or f"resp-{n_resp}")
        elif typ == "response_item" and pt in ("custom_tool_call_output", "function_call_output"):
            o = p.get("output")
            out.append({"type": "user", "timestamp": t, "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": p.get("call_id"), "content": _text(o) if not isinstance(o, str) else o}]}})
    flush(f"resp-{n_resp + 1}-unfinished")
    out.sort(key=lambda x: x["timestamp"])
    return {"thread": meta.get("id") or os.path.basename(path), "subagent": sub, "agent_path": meta.get("agent_path"),
            "records": out, "turns": turns, "turn_ends": turn_ends, "ampersand_launches": amp, "responses": n_resp}

def sessions_dir(run_dir):
    for p in (os.path.join(run_dir, "transcript", "codex-home", "sessions"), os.path.join(run_dir, "codex-home", "sessions")):
        if os.path.isdir(p):
            return p
    return None

def normalise(run_dir, sessions=None):
    """Write transcript/session/codex/ from the rollouts; return what the labels need. None when there are none."""
    sessions = sessions or sessions_dir(run_dir)
    files = sorted(glob.glob(os.path.join(sessions, "**", "rollout-*.jsonl"), recursive=True)) if sessions else []
    if not files:
        return None
    base = os.path.join(run_dir, "transcript", "session", "codex")
    os.makedirs(os.path.join(base, "subagents"), exist_ok=True)
    threads = []
    for f in files:
        n = normalise_one(f)
        dest = os.path.join(base, "subagents" if n["subagent"] else "", f"{n['thread']}.jsonl")
        with open(dest, "w") as fh:
            for r in n["records"]:
                fh.write(json.dumps(r) + "\n")
        threads.append({k: n[k] for k in ("thread", "subagent", "agent_path", "turns", "turn_ends", "ampersand_launches", "responses")}
                       | {"records": len(n["records"]), "file": os.path.relpath(dest, run_dir)})
    efforts = sorted({t["effort"] for th in threads for t in th["turns"] if t.get("effort")})
    models = sorted({t["model"] for th in threads for t in th["turns"] if t.get("model")})
    return {"threads": threads, "efforts": efforts, "models": models,
            "ampersand_launches": [dict(a, thread=th["thread"]) for th in threads for a in th["ampersand_launches"]],
            "how": "Codex rollouts (transcript/codex-home/sessions) written as Claude-shaped session records under "
                   "transcript/session/codex/ (codex_rollout.py), so the scorers read them unchanged"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--sessions", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = normalise(a.run_dir, a.sessions)
    if r is None:
        sys.exit("no Codex rollouts found")
    if a.json:
        print(json.dumps(r, indent=1)); return
    for th in r["threads"]:
        print(f"{'subagent' if th['subagent'] else 'root':8s} {th['thread']}  {th['responses']} responses, {th['records']} records -> {th['file']}")
    print(f"efforts {r['efforts']}  models {r['models']}  launches with a plain &: {len(r['ampersand_launches'])}")

if __name__ == "__main__":
    main()
