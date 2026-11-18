"""The Codex side of a ComputeWise box run: the app-server conversation and the rule for ending it (the Codex driver).

isolation's run_one.py runs `codex app-server` as the box's harness, with the box's stdin FIFO as its stdin. It writes
the lines this module gives it into the FIFO, and feeds back every line Codex writes to /run/box/harness.stdout. This
is the same shape as the runner's harness/lifecycle.py for Claude Code, so it plugs into run_one's monitor loop the same
way:

    s = CodexSession(model, cwd, brief)
    write(s.start_lines())                 # initialize, initialized, thread/start
    for each stdout line: write(s.feed(line, now))    # turn/start once the thread exists; replies to server requests
    on every monitor tick: write(s.poll(now))          # a resume turn whose backoff has passed (see below); without
                                                       # this call a capacity-failed run never resumes and never closes
    if s.should_close(now): close the FIFO             # Codex then exits, and the box stays up for running jobs
                                                       # (s.failed says why when Codex refused to start the session)

The rule (the study's ruling, 2026-10-21, on experiments/codex/notes/lifecycle-2026-10-21.md): send the brief as the
one user turn and nothing after it, because interactive Codex does not wake the agent when background work ends. Keep
the session while background work is alive, because the product does, and close it once the root turn is complete, no
other thread has a turn running or is active, and no command is still running (a background terminal's
commandExecution item completes only when its process exits), all for GRACE_S. The horizon is run_one's, as for Claude.

One exception to "nothing after the brief" (the analysis, 2026-10-28, after 7 of about 10 GPT-6.1 Sol runs lost
their only turn to a provider capacity error): when the root turn ends `failed` with codexErrorInfo serverOverloaded
("Selected model is at capacity"), the provider ended the turn, not the agent. A user at the terminal would type
"continue". So the session sends RESUME_TEXT as a new turn on the same thread after a backoff (RESUME_BACKOFF_S, doubled for
each consecutive failure), at most RESUME_MAX times per run. It is opt-in (`resume=True`), because the caller must
then call poll() on every tick; without the flag the session behaves exactly as before. The thread keeps its whole history (checked against Codex 0.163.0 on 2026-10-28:
notes/codex-capacity-resume-2026-10-28.md), and the box's jobs keep running meanwhile. Any other failed turn ends the
session as before. Each resume is recorded in `resumes` (time of the failure, time sent, gap) for run.json.

Standard library only. `python3 -I codex_session.py` runs the self-test against the known-answer runs in
experiments/codex/known-answer/.
"""
import json
import time

GRACE_S = 3.0
RESUME_ERRORS = ("serverOverloaded",)   # codexErrorInfo values that are the provider's capacity, not the agent's turn end
RESUME_BACKOFF_S = 30.0             # first wait before resuming (Codex's own reconnects back off over about 25 s),
                                    #   doubled for each consecutive failure: 30, 60, 120, 240, 480 s
RESUME_RESET_S = 600.0              # a failure this long after the last resume starts the doubling again
RESUME_MAX = 5                      # resumes per run; the next capacity failure after these ends the session
RESUME_TEXT = "continue"            # what a user types after an outage, and nothing else
CLIENT_NAME = "codex_cli_rs"        # the client name the CLI itself uses; reaches the model gateway, not the agent


# The context window the product's model catalogue gives each model (the study's Codex receives it from the router, its
# provider, 2026-10-21). Without the catalogue Codex assumes 272000, so a box run would compact at about half the
# product's point. Codex applies the catalogue's effective_context_window_percent, 95 for all of these, either way, and
# derives the auto-compact limit from the window, as the product does (the catalogue sets no explicit limit), so only
# the window is set here (the study's ruling, 2026-10-21).
CONTEXT_WINDOW = {"gpt-6-astra": 500000, "gpt-6.1-sol": 500000}


def config_toml(model, base_url, effort="medium"):
    """The agent's ~/.codex/config.toml. The provider is named neutrally. effort medium is the default the product's
    model catalogue gives every gpt-6 model; without the catalogue (a custom provider) Codex would send low. The
    context window is the catalogue's, where it differs from Codex's own default."""
    cw = CONTEXT_WINDOW.get(model)
    return (f'model = "{model}"\n'
            f'model_reasoning_effort = "{effort}"\n'
            + (f'model_context_window = {cw}\n' if cw else '')
            + 'model_provider = "llm"\n\n'
            '[model_providers.llm]\n'
            'name = "LLM API"\n'
            f'base_url = "{base_url}"\n'
            'wire_api = "responses"\n')


class CodexSession:
    def __init__(self, model, cwd, brief, grace_s=GRACE_S, resume=False):
        self.model, self.cwd, self.brief, self.grace_s = model, cwd, brief, grace_s
        self.resume = resume       # resume after a capacity failure; the caller must then call poll() every tick
        self.next_id = 0
        self.waiting = {}          # request id -> method
        self.root = None           # the brief's thread
        self.turn_sent = False
        self.root_turn_active = False
        self.results = 0           # root turns completed
        self.active_turns = set()  # thread ids with a turn started and not completed
        self.thread_status = {}    # thread id -> last status type
        self.open_cmds = {}        # commandExecution item id -> command
        self.idle_since = None
        self.errors = []           # JSON-RPC errors answered to our requests
        self.failed = None         # set when initialize, thread/start or turn/start is refused: the session cannot run
        self.sent = []             # (time, line) of everything written to Codex's stdin
        self.log = []
        self.resumes = []          # {"failed_at", "sent_at", "gap_s", "error"} per resume after a capacity failure
        self.resume_due = None     # (time due, failure time, error) while a resume is waiting for its backoff
        self.resume_max, self.resume_backoff_s = RESUME_MAX, RESUME_BACKOFF_S

    def _req(self, method, params, now):
        self.next_id += 1
        self.waiting[self.next_id] = method
        return self._line({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params}, now)

    def _line(self, msg, now):
        line = json.dumps(msg) + "\n"
        self.sent.append((now, line))
        return line

    def start_lines(self, now=None):
        now = time.time() if now is None else now
        return [self._req("initialize", {"clientInfo": {"name": CLIENT_NAME, "version": "0.0.0"}}, now),
                self._line({"jsonrpc": "2.0", "method": "initialized"}, now),
                self._req("thread/start", {"cwd": self.cwd, "model": self.model, "approvalPolicy": "never",
                                           "sandbox": "danger-full-access"}, now)]

    def feed(self, line, now=None):
        """One line Codex wrote; returns the lines to write back (usually none)."""
        now = time.time() if now is None else now
        try:
            m = json.loads(line)
        except (ValueError, TypeError):
            return []
        if not isinstance(m, dict):
            return []
        out = []
        if "id" in m and ("result" in m or "error" in m):
            method = self.waiting.pop(m["id"], None)
            if "error" in m:
                self.errors.append((now, method, m["error"]))
                self.log.append((now, "error", method))
                if method in ("initialize", "thread/start", "turn/start") and self.failed is None:
                    self.failed = f"codex: {method} failed: {json.dumps(m['error'])[:300]}"
            elif method == "thread/start":
                r = m.get("result") or {}
                self.root = (r.get("thread") or {}).get("id") or r.get("threadId")
                if self.root and not self.turn_sent:
                    self.turn_sent = True
                    self.root_turn_active = True       # from the moment the brief is sent
                    out.append(self._req("turn/start", {"threadId": self.root,
                                                        "input": [{"type": "text", "text": self.brief}]}, now))
                    self.log.append((now, "brief sent", self.root))
            return out
        if "id" in m and "method" in m:
            # A server request. With approvalPolicy=never and a full-access sandbox none is expected; answer it so
            # Codex is never left waiting, and record it.
            self.log.append((now, "server request", m.get("method")))
            return [self._line({"jsonrpc": "2.0", "id": m["id"],
                                "error": {"code": -32601, "message": "not supported"}}, now)]
        meth, p = m.get("method"), m.get("params") or {}
        if meth == "thread/status/changed":
            self.thread_status[p.get("threadId")] = (p.get("status") or {}).get("type")
        elif meth == "turn/started":
            self.active_turns.add(p.get("threadId"))
            if p.get("threadId") == self.root:
                self.root_turn_active = True
            self.idle_since = None
        elif meth == "turn/completed":
            self.active_turns.discard(p.get("threadId"))
            if p.get("threadId") == self.root:
                self.root_turn_active = False
                t = p.get("turn") or {}
                err = t.get("error") or {}
                if self.resume and t.get("status") == "failed" and err.get("codexErrorInfo") in RESUME_ERRORS \
                        and len(self.resumes) < self.resume_max:
                    # the provider ended the turn: resume after the backoff instead of counting it as the answer
                    consec, last = 0, now
                    for r in reversed(self.resumes):        # consecutive: each within RESUME_RESET_S of the next
                        if last - r["sent_at"] > RESUME_RESET_S:
                            break
                        consec, last = consec + 1, r["failed_at"]
                    wait = self.resume_backoff_s * 2 ** consec
                    self.resume_due = (now + wait, now, (err.get("message") or "")[:200])
                    self.log.append((now, "root turn failed on capacity: resume due", len(self.resumes) + 1))
                else:
                    self.results += 1
                    self.log.append((now, "root turn completed", self.results))
        elif meth in ("item/started", "item/completed"):
            it = p.get("item") or {}
            if it.get("type") == "commandExecution":
                if meth == "item/started":
                    self.open_cmds[it.get("id")] = it.get("command")
                    self.idle_since = None
                else:
                    self.open_cmds.pop(it.get("id"), None)
        return out

    def poll(self, now=None):
        """Lines to write that are due by time, not by a message: a resume whose backoff has passed. run_one calls it
        on every monitor tick, beside feed()."""
        now = time.time() if now is None else now
        if self.resume_due is None or now < self.resume_due[0] or self.root is None:
            return []
        _, failed_at, err = self.resume_due
        self.resume_due = None
        self.root_turn_active = True
        self.resumes.append({"failed_at": failed_at, "sent_at": now, "gap_s": round(now - failed_at, 1), "error": err})
        self.log.append((now, "resume sent", len(self.resumes)))
        return [self._req("turn/start", {"threadId": self.root, "input": [{"type": "text", "text": RESUME_TEXT}]}, now)]

    def busy(self):
        others = {t for t in self.active_turns if t != self.root} | \
                 {t for t, s in self.thread_status.items() if t != self.root and s == "active"}
        return sorted(others), dict(self.open_cmds)

    def should_close(self, now=None):
        """True when the session should end: the close rule holds, or Codex refused to start it (see `failed`)."""
        now = time.time() if now is None else now
        if self.failed:
            return True
        if not self.turn_sent or self.root_turn_active or self.resume_due is not None:
            self.idle_since = None
            return False
        others, cmds = self.busy()
        if others or cmds:
            self.idle_since = None
            return False
        if self.idle_since is None:
            self.idle_since = now
        return now - self.idle_since >= self.grace_s


def _selftest():
    """Replay each known-answer run's inbound messages through a session and check where it says to close."""
    import glob
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    ka = os.path.join(os.path.dirname(here), "known-answer")
    ok = True
    for d in sorted(glob.glob(os.path.join(ka, "s*"))):
        recs = [json.loads(l) for l in open(os.path.join(d, "codex-rpc.jsonl"))]
        if not any(r["dir"] == "driver" for r in recs):
            continue                # a box excerpt with no driver end (s6): checked by its own case below
        drv_end = next(r["t"] for r in recs if r["dir"] == "driver")
        s = CodexSession("gpt-6-astra", "/w", "brief")
        s.start_lines(recs[0]["t"])
        closed_at = None
        for r in recs:
            if r["dir"] != "in":
                continue
            msg = r["msg"]          # the prototype numbered its requests as this session does: 1, 2, 3
            s.feed(json.dumps(msg), r["t"])
            t = r["t"]
            if closed_at is None and s.should_close(t):
                closed_at = t
        t = recs[-1]["t"]
        while closed_at is None and t < drv_end + 30:
            t += 0.25
            if s.should_close(t):
                closed_at = t
        good = closed_at is not None and abs(closed_at - drv_end) < 2.0 and s.results == 1
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} {os.path.basename(d)}: closes at +{closed_at - recs[0]['t']:.1f} s, "
              f"prototype ended at +{drv_end - recs[0]['t']:.1f} s, root turns {s.results}")
    # s6: a real root turn ended by a capacity error (G-kvgate-sol-L0, 2026-10-28). The session must not close; after
    # the backoff it sends one "continue" turn on the same thread; when that turn completes it closes as usual
    s6 = os.path.join(ka, "s6-sol-capacity-failed-turn", "codex-rpc.jsonl")
    if os.path.exists(s6):
        recs = [json.loads(l) for l in open(s6)]
        s = CodexSession("gpt-6.1-sol", "/w", "brief", resume=True)
        s.start_lines(recs[0]["t"])
        for r in recs:
            if r["dir"] == "in":
                s.feed(json.dumps(r["msg"]), r["t"])
        t_fail = recs[-1]["t"]
        early = s.poll(t_fail + RESUME_BACKOFF_S - 1)
        closed_early = s.should_close(t_fail + RESUME_BACKOFF_S - 1)
        sent = s.poll(t_fail + RESUME_BACKOFF_S)
        msg = json.loads(sent[0]) if sent else {}
        good = (not early and not closed_early and len(sent) == 1 and msg.get("method") == "turn/start"
                and msg["params"]["threadId"] == s.root and msg["params"]["input"][0]["text"] == RESUME_TEXT
                and len(s.resumes) == 1 and s.resumes[0]["gap_s"] == RESUME_BACKOFF_S and s.results == 0)
        # the resumed turn completes: one answer, and the session closes after the grace
        tid = s.root
        s.feed(json.dumps({"method": "turn/started", "params": {"threadId": tid}}), t_fail + 31)
        s.feed(json.dumps({"method": "turn/completed", "params": {"threadId": tid, "turn": {"status": "completed"}}}),
               t_fail + 40)
        good &= not s.should_close(t_fail + 40) and s.should_close(t_fail + 40 + GRACE_S) and s.results == 1
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} s6-sol-capacity-failed-turn: no close at the failure, 'continue' sent at "
              f"+{RESUME_BACKOFF_S:g} s on the same thread, closes after the resumed turn; resumes {len(s.resumes)}")
    # the cap: after RESUME_MAX capacity failures, the next one ends the session as before
    s = CodexSession("gpt-6.1-sol", "/w", "brief", resume=True)
    s.start_lines(0.0)
    s.feed(json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"thread": {"id": "T"}}}), 0.1)
    t = 1.0
    fail = lambda t: s.feed(json.dumps({"method": "turn/completed", "params": {"threadId": "T", "turn": {
        "status": "failed", "error": {"message": "Selected model is at capacity.", "codexErrorInfo": "serverOverloaded"}}}}), t)
    for k in range(RESUME_MAX):
        fail(t); t += RESUME_BACKOFF_S * 2 ** k; s.poll(t); t += 5
    fail(t)
    gaps = [r["gap_s"] for r in s.resumes]
    good = (len(s.resumes) == RESUME_MAX and gaps == [RESUME_BACKOFF_S * 2 ** k for k in range(RESUME_MAX)]
            and s.resume_due is None and s.results == 1
            and not s.should_close(t) and s.should_close(t + GRACE_S))     # the grace starts at the first call
    # and a failure that is not capacity is not resumed
    s2 = CodexSession("gpt-6.1-sol", "/w", "brief", resume=True)
    s2.start_lines(0.0)
    s2.feed(json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"thread": {"id": "T"}}}), 0.1)
    s2.feed(json.dumps({"method": "turn/completed", "params": {"threadId": "T", "turn": {
        "status": "failed", "error": {"message": "stream error", "codexErrorInfo": "other"}}}}), 1.0)
    good &= s2.resume_due is None and not s2.resumes and not s2.should_close(1.0) and s2.should_close(1.0 + GRACE_S)
    ok &= good
    # a failure long after the last resume starts the doubling again
    s3 = CodexSession("gpt-6.1-sol", "/w", "brief", resume=True)
    s3.start_lines(0.0)
    s3.feed(json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"thread": {"id": "T"}}}), 0.1)
    for t0 in (1.0, 5000.0):
        s3.feed(json.dumps({"method": "turn/completed", "params": {"threadId": "T", "turn": {
            "status": "failed", "error": {"codexErrorInfo": "serverOverloaded"}}}}), t0)
        s3.poll(t0 + RESUME_BACKOFF_S)
    good &= [r["gap_s"] for r in s3.resumes] == [RESUME_BACKOFF_S, RESUME_BACKOFF_S]
    # without resume=True a capacity failure ends the session exactly as before the resume was added
    s4 = CodexSession("gpt-6.1-sol", "/w", "brief")
    s4.start_lines(0.0)
    s4.feed(json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"thread": {"id": "T"}}}), 0.1)
    s4.feed(json.dumps({"method": "turn/completed", "params": {"threadId": "T", "turn": {
        "status": "failed", "error": {"codexErrorInfo": "serverOverloaded"}}}}), 1.0)
    good &= s4.results == 1 and not s4.poll(100.0) and not s4.should_close(1.0) and s4.should_close(1.0 + GRACE_S)
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} resume rules: backoff {', '.join(f'{g:g}' for g in gaps)} s, then the next "
          f"capacity failure closes; a non-capacity failure is not resumed; a failure after {RESUME_RESET_S:g} s starts "
          f"the doubling again; without resume=True nothing changes")
    s = CodexSession("gpt-6-astra", "/w", "brief")
    s.start_lines(0.0)
    s.feed(json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}), 0.1)
    s.feed(json.dumps({"jsonrpc": "2.0", "id": 2, "error": {"code": -32600, "message": "bad model"}}), 0.2)
    good = s.should_close(0.3) and s.failed.startswith("codex: thread/start failed") and not s.turn_sent
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} refused thread/start: closes at once, failed = {s.failed!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
