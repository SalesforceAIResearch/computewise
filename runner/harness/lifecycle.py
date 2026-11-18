"""When to end a Claude Code session the way the interactive product would.

`claude -p` with the brief as its prompt ends the session when the agent's turn ends; background work (a background
Bash task, a subagent) then gets a bounded wait and is killed, and the agent never hears that it finished
(Q-prepP6-haiku-L0: six background subagents, three jobs killed at 703.5 s). In the interactive product a background
task's completion starts a new turn. Claude Code does the same in stream-json input mode while its stdin stays open:
probed on the controller with 2.1.303, the box's version (runner/notes/background-lifecycle-2026-10-21.md): the agent
started `sleep 25` in the background and ended its turn ("WAITING", result at 12.5 s); at 34.5 s Claude Code emitted
task_notification and background_tasks_changed [] and, with no input from us, ran a new turn ("FINISHED", result at
40.3 s). So the fix is only in when stdin closes:

  - the brief goes in as one stream-json user message, and stdin stays open;
  - stdin closes when a turn has ended (a `result` event) while no background task is alive (the last
    `background_tasks_changed` listed none) and nothing new started within GRACE_S after that result;
  - the horizon and every stop rule stay as they are: this decides only the agent's own end.

Nothing reaches the agent that Claude Code itself does not produce: the notifications and the new turn are its own.

One more close (the study, 2026-10-21, after Q-prepP6-sonnet-L0-nosub: the agent's final answer came with a background
`while pgrep -f "quick/.*train_gpt"` poll that matched its own shell, and the session sat three hours to the horizon):
when the agent's turn has ended with a result and every task job is completed, stdin closes after GRACE_S even with
background tasks alive. The scoring window is already cut at the last job's completion, so nothing after it changes a
score; it only gives the machine back. While any job is unfinished the rule above holds: a notification could wake the
agent to launch more. "Completed" is the scorer's own work_remaining (runner/score/idle.py) on the run's occupancy log,
so the two cannot disagree; run_one.py passes the run directory (Lifecycle(run_dir=...)). Any error reading it counts as
work remaining, so the session stays open.

Scheduler jobs count as background work (the study, 2026-10-24 08:09Z, after G-h1-haiku-S1: the agent submitted 15
screens to the box's scheduler and ended its turn saying it would read them when they finished, and the first rule closed
the session at 319 s while the jobs ran to 9,068 s). A job the agent hands to the box's slurmctld is not a Claude Code
background task, so background_tasks_changed never lists it. While the controller's count of jobs not yet ended
(slurmctld --active-file, /run/box/sched-active, which the host reads as <run_dir>/box/sched-active) is above 0, the first
rule does not close; it closes GRACE_S after the count reaches 0, if no turn is running and no background task is alive.
A run with no scheduler has no such file and is unaffected. A file that exists but cannot be read counts as jobs live.
Claude Code starts no turn when a scheduler job ends (only a user message or its own task notification does), so this
keeps the session open; it does not wake the agent.

The work-done close waits for the agent's next turn (the study, 2026-10-27 00:51Z, after G-g4nb-opus-L0-r2: the close
came 26 s after the last job, while the agent's 300-s `sleep 300 && echo ALLDONE` wait still ran as a background task, and
its sound final answer was lost; analysis labelled that cell ours, not the agent's). When every task job is completed and
the agent's own background task holds the session, stdin stays open up to WORK_DONE_GRACE_S for the next turn, which
that task's completion starts. A turn in that time is answered_in_grace, and the session then ends by the first rule; a
grace that runs out is closed_after_grace, and the session is ended as before; a task that ends with no turn following
is task_ended_no_answer. The outcome and the seconds waited are in close_reason, as
`[work_done_grace outcome=<outcome> waited_s=<s> grace_s=<s>]`, which run_one.py writes to status.json's
stdin_close_reason. Only Claude Code's own background tasks get the grace, because only their completion starts a turn:
work done with only a box scheduler job live closes after GRACE_S, as before.

A live CronCreate holds the session like a pending ScheduleWakeup (the analysis, ruled 2026-10-30 07:14 MDT, after
G-kvreph-sonnet-L3-m2b: the agent made a `*/20 * * * *` check and ended its turn at 106 s, and the first rule closed stdin at
110 s; all 40 jobs then ran with no agent). Probed on the controller with 2.1.303 (isolation, 2026-10-30 13:27Z, Haiku, one
session): in this stream-json mode with stdin open, a `* * * * *` CronCreate fired 20 s after the agent's result and Claude
Code ran a new turn with no input from us, so here too the fix is only in when stdin closes. A cron is live from its
CronCreate result until a CronDelete of its job id succeeds, or, for a one-shot (`recurring: false`), until it fires. While
one is live, the first rule does not close before its next match (the same rule as runner/score/score_all.py
cron_next_fire, in UTC, the box's clock) plus the product's lateness for a recurring job (up to half its period, at most
30 min) plus WAKE_GRACE_S. Its turn re-arms a recurring cron while work remains; once every task job is completed, the
next fire is the last the session waits for, so a cron never deleted cannot hold a finished run to the horizon.

usage (a known answer, no model): python3 -I lifecycle.py --selftest
"""
import json, re, sys, time

GRACE_S = 3.0   # after a result with no background task alive, how long a new turn may take to start
NOTIF_WAIT_S = 120.0  # a notification of the agent's own background task opens a turn; if no assistant event comes
                      # within this, the turn is taken as not coming (the horizon bounds the run regardless)


WORK_CHECK_S = 15.0   # how often, while a result waits on background tasks, the run's work is checked
WAKE_GRACE_S = 120.0  # a pending ScheduleWakeup keeps the session to its fire time plus this; the horizon bounds it
WORK_DONE_GRACE_S = 1800.0  # work done, the agent's background task alive: how long its next turn may take to come
CRON_LATE_MAX_S = 1800.0    # the product runs a recurring cron up to half its period late, at most 30 min (CronCreate's
CRON_EARLY_S = 90.0         # own description); a one-shot on :00 or :30 up to 90 s early

def cron_next_fire(expr, after):
    """The first wall time after `after` (epoch s, UTC) that a five-field cron expression matches, by minute; None when the
    expression does not parse or matches nothing within 8 days (the scorer's rule, runner/score/score_all.py)."""
    import datetime as _dt
    f = (expr or "").split()
    if len(f) != 5:
        return None
    def field(x, lo, hi):
        vals = set()
        for part in x.split(","):
            stp = 1
            if "/" in part:
                part, s_ = part.split("/", 1)
                if not s_.isdigit() or int(s_) < 1:
                    return None
                stp = int(s_)
            if part == "*":
                a, b = lo, hi
            elif "-" in part:
                a_, b_ = part.split("-", 1)
                if not (a_.isdigit() and b_.isdigit()):
                    return None
                a, b = int(a_), int(b_)
            elif part.isdigit():
                a = b = int(part)
            else:
                return None
            vals.update(range(a, b + 1, stp))
        return vals
    sets = [field(x, lo, hi) for x, (lo, hi) in zip(f, ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6)))]
    if any(v is None for v in sets):
        return None
    mi, ho, dom, mon, dow = sets
    t = _dt.datetime.fromtimestamp(int(after // 60 + 1) * 60, _dt.timezone.utc)
    for _ in range(8 * 24 * 60):
        if (t.minute in mi and t.hour in ho and t.day in dom and t.month in mon and (t.isoweekday() % 7) in dow):
            return t.timestamp()
        t += _dt.timedelta(minutes=1)
    return None

def work_done_in(run_dir, score_dir=None):
    """A callable: True when every task job of the run in run_dir is completed, by the scorer's own rule."""
    import importlib.util, os
    score = score_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "score")
    def load(name):
        spec = importlib.util.spec_from_file_location(f"p006_{name}", os.path.join(score, f"{name}.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
    state = {}
    def done():
        try:
            if "idle" not in state:
                import sys
                sys.path.insert(0, score)
                state["idle"], state["score_run"] = load("idle"), load("score_run")
            task = json.load(open(os.path.join(run_dir, "task.json")))
            ev = state["score_run"].load_events(os.path.join(run_dir, "occupancy.jsonl"))
            rem = state["idle"].work_remaining(ev, task)
            # an open pool (goal packages: the agent decides how many runs) is never done by the job list: its min is met
            # from the first completed run while more are running or yet to be launched (G-passbar-sonnet-L0-v2, killed
            # at 19:57Z with 4 of 16 runs done, the study 2026-10-21). Only the agent's final answer with no background
            # task alive (the first rule) or the horizon ends such a session
            if rem.get("open_pool"):
                return False
            # ... and no GPU work live on the box (the analysis 2026-11-05 21:05 MST, G-k-haiku-A3: every task job had ended
            # at 37,483 s while 4 of the agent's own gpu-harness runs, candidates the task file does not list, were still training;
            # the grace counted from the task jobs' end and its close killed them). Live = a rank with a job_hello and no job_bye yet,
            # for every arm, not the harness alone
            hello, bye = set(), set()
            for e in ev:
                if e.get("kind") == "job_hello":
                    hello.add(e.get("job"))
                elif e.get("kind") == "job_bye":
                    bye.add(e.get("job"))
            if hello - bye:
                return False
            # ... and no job running on the agent's own gpu-harness board, GPU or CPU (the analysis 2026-11-05 22:59 MST,
            # G-fyld-haiku-A3: a CPU prep ran on a GPU slot of the harness with no occupancy rank, and the close came while the
            # agent's wait on the harness's ALL_DECIDED would have returned). The box's home is the run's disk mount, run.json's
            # disk_image without '.img'; the board is the latest .gpuh/runs/<stamp>/board.json there. An unreadable board is no hold
            if board_busy(run_dir):
                return False
            return rem.get("remaining") == 0 and rem.get("all_done_wall") is not None
        except Exception:
            return False
    return done

def board_busy(run_dir, home=None):
    """True when the agent's gpu-harness board (the latest <home>/.gpuh/runs/*/board.json) has a job running: a GPU slot held
    (gpu_job non-null) or any job in state running, CPU jobs included. home defaults to the run's disk mount (run.json
    disk_image without '.img', then /home). False when there is no board or it cannot be read."""
    import glob, os
    try:
        if home is None:
            img = (json.load(open(os.path.join(run_dir, "run.json"))) or {}).get("disk_image") or ""
            if not img.endswith(".img"):
                return False
            home = os.path.join(img[:-4], "home")
        boards = sorted(glob.glob(os.path.join(home, ".gpuh", "runs", "*", "board.json")))
        if not boards:
            return False
        b = json.load(open(boards[-1]))
        if any(v for v in (b.get("gpu_job") or {}).values()):
            return True
        jobs = b.get("jobs") or {}
        jobs = jobs.values() if isinstance(jobs, dict) else jobs
        return any(isinstance(j, dict) and j.get("state") == "running" for j in jobs)
    except Exception:
        return False

def sched_active_in(run_dir):
    """A callable: the box scheduler's count of jobs not yet ended, read from <run_dir>/box/sched-active (run_one.py links
    <run_dir>/box to the box's /run/box). 0 when the file is absent: the run has no scheduler. Any other error raises,
    and the caller counts it as jobs live."""
    import os
    path = os.path.join(run_dir, "box", "sched-active")
    def n():
        try:
            with open(path) as fh:
                return int(fh.read().strip() or "0")
        except FileNotFoundError:
            return 0
    return n

class Lifecycle:
    """Feed it each stream-json line Claude Code writes (feed), and ask it whether stdin should close now
    (should_close). A result while background work is alive keeps the session; Claude Code's own notification then
    starts the next turn, whose result is checked the same way."""

    def __init__(self, grace_s=GRACE_S, run_dir=None, work_done=None, sched_active=None, work_done_grace_s=WORK_DONE_GRACE_S):
        self.grace_s = grace_s
        self.work_done_grace_s = work_done_grace_s
        self.grace_from = None     # when the current work-done grace began (work done, the agent's background task alive)
        self.grace_outcome = None  # answered_in_grace, closed_after_grace or task_ended_no_answer, once it ends
        self.grace_waited = None   # seconds from the grace's start to its outcome
        self.grace_episodes = []   # (outcome, waited) of earlier graces in this session
        self.work_done = work_done or (work_done_in(run_dir) if run_dir else None)
        self.sched_active = sched_active or (sched_active_in(run_dir) if run_dir else None)
        self.sched_seen_at = None  # the last time the box scheduler held a job not yet ended
        self.sched_jobs_max = 0    # the most scheduler jobs seen live while a result waited on them
        self.done_since = None     # when the turn had ended and every task job was completed (background tasks alive)
        self.last_check = None
        self.close_reason = None
        self.tasks = {}            # task_id -> description, from the last background_tasks_changed
        self.bash_timeouts = {}    # Bash tool_use id -> its timeout (s), for a background call that names one
        self.task_bound = {}       # task_id -> wall time its own timeout ends it (task_started time + the call's timeout)
        self.results = 0
        self.idle_since = None     # when the last result arrived with no background task alive
        self.turn_active = True    # the brief starts the first turn
        self.owned_by_subagent = {}  # task_id -> True for a subagent's own task (its notification goes to the subagent)
        self.pending_since = None  # a notification opened a turn whose first assistant event has not come yet
        self.log = []
        self.wake_at = None        # a pending ScheduleWakeup's fire time (wall), from its tool_use and our clock
        self.wake_calls = {}       # tool_use id -> delaySeconds, until its result confirms it
        self.crons = {}            # job id -> {expr, recurring, next, late}: live CronCreate jobs
        self.cron_calls = {}       # CronCreate tool_use id -> (expr, recurring), until its result gives the job id
        self.cron_deletes = {}     # CronDelete tool_use id -> job id, until its result confirms it
        self.cron_seen = False     # a cron was live at some point (for close_reason)

    def _cron_arm(self, jid, expr, recurring, after, now):
        nxt = cron_next_fire(expr, after)
        if nxt is None:
            self.crons.pop(jid, None)
            return
        after_that = cron_next_fire(expr, nxt)
        late = min(0.5 * (after_that - nxt), CRON_LATE_MAX_S) if (recurring and after_that) else 0.0
        self.crons[jid] = {"expr": expr, "recurring": recurring, "next": nxt, "late": late}
        self.log.append((now, "cron-pending", round(nxt - now)))

    def cron_wake(self):
        """The latest time the earliest live cron can still fire (its next match plus its lateness), or None."""
        return min((c["next"] + c["late"] for c in self.crons.values()), default=None)

    def feed(self, line, now=None):
        now = time.time() if now is None else now
        try:
            r = json.loads(line)
        except (ValueError, TypeError):
            return
        if not isinstance(r, dict):
            return
        typ, sub = r.get("type"), r.get("subtype")
        # a record's message is a dict for assistant and user records, but a string on some system records (2026-10-22
        # 19:23Z: system/permission_denied, when Claude Code's safety check refused an rm of a glob, carries
        # "message": "Dangerous rm operation ...", and feed() raised on it, ending Q-p20tail-sonnet-L0-bashdefault-r2
        # 73 s in as a harness-error)
        msg = r.get("message") if isinstance(r.get("message"), dict) else {}
        # a ScheduleWakeup: its call, and
        # its result, which says when the harness will re-invoke the agent
        for b in (msg.get("content") or []) if isinstance(msg.get("content"), list) else []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use" and b.get("name") == "Bash":
                try:
                    _to = (b.get("input") or {}).get("timeout")
                    if _to:
                        self.bash_timeouts[b.get("id")] = float(_to) / 1000.0
                except (TypeError, ValueError):
                    pass
            if b.get("type") == "tool_use" and b.get("name") == "ScheduleWakeup":
                try:
                    self.wake_calls[b.get("id")] = float((b.get("input") or {}).get("delaySeconds") or 0)
                except (TypeError, ValueError):
                    pass
            elif b.get("type") == "tool_result" and b.get("tool_use_id") in self.wake_calls and not b.get("is_error"):
                self.wake_at = max(self.wake_at or 0.0, now + self.wake_calls.pop(b.get("tool_use_id")))
                self.log.append((now, "wake-pending", round(self.wake_at - now)))
            elif b.get("type") == "tool_use" and b.get("name") == "CronCreate":
                inp = b.get("input") or {}
                self.cron_calls[b.get("id")] = (inp.get("cron"), inp.get("recurring", True) is not False)
            elif b.get("type") == "tool_use" and b.get("name") == "CronDelete":
                self.cron_deletes[b.get("id")] = str((b.get("input") or {}).get("id") or "")
            elif b.get("type") == "tool_result" and b.get("tool_use_id") in self.cron_calls:
                expr, recurring = self.cron_calls.pop(b.get("tool_use_id"))
                if not b.get("is_error"):
                    c = b.get("content")
                    text = c if isinstance(c, str) else " ".join(x.get("text", "") for x in c or [] if isinstance(x, dict))
                    m = re.search(r"\bjob ([0-9A-Za-z_-]+)", text or "")
                    self._cron_arm(m.group(1) if m else b.get("tool_use_id"), expr, recurring, now, now)
                    self.cron_seen = True
            elif b.get("type") == "tool_result" and b.get("tool_use_id") in self.cron_deletes:
                jid = self.cron_deletes.pop(b.get("tool_use_id"))
                if not b.get("is_error") and jid in self.crons:
                    del self.crons[jid]
                    self.log.append((now, "cron-deleted", 0))
        # the agent's next turn during a work-done grace: no turn is active then, so a main-thread assistant event is one.
        # A background subagent still working writes assistant events too, marked parent_tool_use_id; those are not.
        if typ == "assistant" and self.grace_from is not None and not r.get("parent_tool_use_id"):
            self._grace_end("answered_in_grace", now)
        if typ == "system" and sub == "background_tasks_changed":
            self.tasks = {t.get("task_id"): t.get("description") for t in r.get("tasks") or []}
            self.log.append((now, "tasks", len(self.tasks)))
        elif typ == "result":
            self.results += 1
            self.turn_active = False
            # the product fires a cron only while idle: a match that came and went during this turn fires now, so its
            # window runs from this result
            for c in self.crons.values():
                if c["next"] + c["late"] < now:
                    c["next"], c["late"] = now, 60.0
            self.idle_since = now if not self.tasks else None
            self.log.append((now, "result", len(self.tasks)))
        elif typ == "system" and sub == "task_notification":
            # The agent's own background task finished: Claude Code delivers the notification as a new turn, but the
            # turn's first assistant event can take longer than GRACE_S (time to first token with thinking, a slower
            # model), so the notification itself opens the turn (isolation's review, 2026-10-21). A subagent's own
            # task notifies the subagent, not the agent.
            if not self.owned_by_subagent.get(r.get("task_id")) and not self.turn_active:
                self.turn_active = True
                self.pending_since = now
                self.idle_since = None
                self.log.append((now, "notification-turn", len(self.tasks)))
        elif typ in ("assistant", "user") or (typ == "system" and sub in ("task_started", "init")):
            if typ == "system" and sub == "task_started":
                self.owned_by_subagent[r.get("task_id")] = bool(r.get("owned_by_subagent"))
            if typ == "assistant":
                self.pending_since = None
            # anything the agent does after a result means a new turn has started (a notification's, most often). In a
            # work-done grace a background subagent's own message is not the agent's turn: it would hold the session to
            # the horizon instead of the grace's end
            if not self.turn_active and typ == "assistant" and not (self.grace_from is not None and r.get("parent_tool_use_id")):
                self.turn_active = True
                self.idle_since = None
                if self.wake_at is not None and now >= self.wake_at - 5:
                    self.wake_at = None          # the wakeup fired: the harness re-invoked the agent
                    self.log.append((now, "wake-fired", 0))
                # a cron whose match has come fired this turn: a one-shot is gone, a recurring one re-arms from now while
                # work remains; once every task job is completed this was its last fire (unknown counts as work left)
                # recurring jobs only run late; a one-shot on :00 or :30 may run up to CRON_EARLY_S early
                fired = [j for j, c in self.crons.items()
                         if now >= c["next"] - (CRON_EARLY_S if (not c["recurring"] and int(c["next"] // 60) % 30 == 0) else 5.0)]
                if fired:
                    try:
                        done = bool(self.work_done()) if self.work_done else False
                    except Exception:
                        done = False
                    for j in fired:
                        c = self.crons[j]
                        if c["recurring"] and not done:
                            self._cron_arm(j, c["expr"], True, now, now)
                        else:
                            del self.crons[j]
                            self.log.append((now, "cron-spent" if c["recurring"] else "cron-fired-once", 0))
                self.log.append((now, "turn", len(self.tasks)))
            if typ == "system" and sub == "task_started":
                self.idle_since = None
                if r.get("tool_use_id") in self.bash_timeouts and r.get("task_id"):
                    self.task_bound[r.get("task_id")] = now + self.bash_timeouts[r.get("tool_use_id")]

    def tick(self, now=None):
        """Call periodically: a result that arrived while tasks were alive becomes idle once they are all gone."""
        now = time.time() if now is None else now
        if self.turn_active and self.pending_since is not None and now - self.pending_since > NOTIF_WAIT_S:
            self.turn_active, self.pending_since = False, None      # the notification's turn never came
            self.log.append((now, "notification-turn-lapsed", len(self.tasks)))
        if not self.turn_active and not self.tasks and self.idle_since is None and self.results:
            self.idle_since = now

    def sched_live(self, now):
        """True while the box scheduler holds a job not yet ended, or its count cannot be read."""
        if self.sched_active is None:
            return False
        try:
            n = int(self.sched_active())
        except Exception:          # unknown means jobs remain, as for the work check
            n = None
        if n == 0:
            return False
        if self.sched_seen_at is None:
            self.log.append((now, "sched-jobs-live", n))
        self.sched_seen_at = now
        self.sched_jobs_max = max(self.sched_jobs_max, n or 0)
        return True

    def should_close(self, now=None):
        now = time.time() if now is None else now
        self.tick(now)
        # scheduler jobs are background work for both rules below, as Claude Code's own background tasks are
        sched = bool(self.results) and not self.turn_active and self.sched_live(now)
        # the grace runs from the result, or from the last scheduler job's end if that came later
        since = max(self.idle_since, self.sched_seen_at or 0.0) if self.idle_since is not None else None
        if (not self.turn_active and not self.tasks and not sched and since is not None
                and now - since >= self.grace_s):
            # a pending wakeup keeps the session until its fire time plus WAKE_GRACE_S: the harness re-invokes the agent
            # then, as the product does; past that the wakeup did not come, and the session closes as before
            if self.wake_at is not None and now < self.wake_at + WAKE_GRACE_S:
                return False
            # a live cron keeps it the same way, to its next match plus the product's lateness plus WAKE_GRACE_S
            cw = self.cron_wake()
            if cw is not None and now < cw + WAKE_GRACE_S:
                return False
            if self.grace_from is not None:          # the task ended and its notification's turn never came
                self._grace_end("task_ended_no_answer", now)
            self.close_reason = (f"lifecycle: a result with no background task alive for {self.grace_s:g} s"
                                 + (" (a ScheduleWakeup was pending and did not fire)" if self.wake_at is not None else "")
                                 + (" (a CronCreate job was live and did not fire)" if cw is not None else "")
                                 + (f" (after the box scheduler's jobs ended, at most {self.sched_jobs_max} live)"
                                    if self.sched_seen_at is not None else "") + self._grace_tag())
            return True
        # the turn ended with a result, background tasks or scheduler jobs are alive, and every task job is completed
        if self.work_done and self.results and not self.turn_active and (self.tasks or sched):
            if self.done_since is None and (self.last_check is None or now - self.last_check >= WORK_CHECK_S):
                self.last_check = now
                try:
                    ok = bool(self.work_done())
                except Exception:          # the harness must never fail on this check: unknown means work remains
                    ok = False
                if ok:
                    self.done_since = now
            if self.done_since is not None:
                alive = ", ".join(sorted(str(k) for k in self.tasks)) + \
                    ((", " if self.tasks else "") + "box scheduler job(s)" if sched else "")
                if self.tasks:
                    # the agent's own background task holds the session, and its completion starts the agent's next
                    # turn: wait up to work_done_grace_s for that turn (a turn ends the grace in feed)
                    if self.grace_from is None:
                        self.grace_from = self.done_since
                        self.log.append((now, "work-done-grace", len(self.tasks)))
                    # a live background task with its own timeout holds the grace to that end plus WAKE_GRACE_S, when later than the
                    # usual grace (the analysis 2026-11-05 21:16 MST, G-k2p-sonnet-L0-v2-r2: a 10-h-bounded wait closed 1,800 s after
                    # the work, about 25 min before its bound; a bounded wait is the agent's own measured idle). The horizon bounds it
                    bounds = [self.task_bound[t] for t in self.tasks if t in self.task_bound]
                    hold_to = max(self.grace_from + self.work_done_grace_s, (max(bounds) + WAKE_GRACE_S) if bounds else 0.0)
                    if now >= hold_to:
                        self._grace_end("closed_after_grace", now)
                        self.close_reason = (f"lifecycle: every task job completed and the turn ended, background task(s) "
                                             f"still alive ({alive}), and no turn came in the {self.work_done_grace_s:g}-s "
                                             f"grace" + self._grace_tag())
                        return True
                elif now - self.done_since >= self.grace_s:
                    self.close_reason = (f"lifecycle: every task job completed and the turn ended, background task(s) still "
                                         f"alive ({alive}), for {self.grace_s:g} s" + self._grace_tag())
                    return True
        else:
            self.done_since = None
        return False

    def _grace_end(self, outcome, now):
        self.grace_outcome, self.grace_waited = outcome, now - self.grace_from
        self.grace_episodes.append((outcome, round(self.grace_waited, 1)))
        self.log.append((now, outcome, round(self.grace_waited)))
        self.grace_from = None

    def _grace_tag(self):
        """The work-done grace's outcome for close_reason (run_one.py writes it to status.json), or "" when none ran."""
        if not self.grace_episodes:
            return ""
        return (f" [work_done_grace outcome={self.grace_outcome} waited_s={self.grace_waited:.1f} "
                f"grace_s={self.work_done_grace_s:g}"
                + (f" episodes={len(self.grace_episodes)}" if len(self.grace_episodes) > 1 else "") + "]")


def brief_message(text):
    """The brief as the one stream-json user message the session starts with."""
    return json.dumps({"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": text}]}}) + "\n"


def _selftest():
    ok = True
    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(("  ok   " if good else "  FAIL ") + f"{name}: got {got}, want {want}")
    ev = lambda **r: json.dumps(r)
    # 1. The failing case: the agent backgrounds work and ends its turn. stdin must stay open until the work is done
    # and Claude Code's own notification turn has ended.
    lc = Lifecycle()
    lc.feed(ev(type="assistant", message={"content": [{"type": "tool_use"}]}), 5)
    lc.feed(ev(type="system", subtype="background_tasks_changed", tasks=[{"task_id": "a", "description": "job"}]), 6)
    lc.feed(ev(type="result", subtype="success"), 10)
    check("turn ended with a background task alive: stays open", lc.should_close(20), False)
    check("still open long after, while the task runs", lc.should_close(600), False)
    lc.feed(ev(type="system", subtype="task_notification", status="completed"), 700)
    lc.feed(ev(type="system", subtype="background_tasks_changed", tasks=[]), 700)
    check("task done, notification turn not yet started: grace", lc.should_close(701), False)
    lc.feed(ev(type="assistant", message={"content": [{"type": "text", "text": "FINISHED"}]}), 702)
    check("the agent's re-invoked turn is running: open", lc.should_close(710), False)
    lc.feed(ev(type="result", subtype="success"), 712)
    check("its turn ended with nothing alive: open within the grace", lc.should_close(713), False)
    check("then closes", lc.should_close(716), True)
    # 1b. A slow model: the notification's turn makes its first assistant event 10 s after the notification. The
    # notification itself opened the turn, so stdin stays open (isolation's review).
    lc = Lifecycle()
    lc.feed(ev(type="system", subtype="task_started", task_id="a", owned_by_subagent=False), 5)
    lc.feed(ev(type="system", subtype="background_tasks_changed", tasks=[{"task_id": "a", "description": "job"}]), 6)
    lc.feed(ev(type="result", subtype="success"), 10)
    lc.feed(ev(type="system", subtype="task_notification", task_id="a", status="completed"), 700)
    lc.feed(ev(type="system", subtype="background_tasks_changed", tasks=[]), 700)
    check("slow first token: open 8 s after the notification", lc.should_close(708), False)
    lc.feed(ev(type="assistant", message={"content": [{"type": "text", "text": "FINISHED"}]}), 710)
    lc.feed(ev(type="result", subtype="success"), 730)
    check("slow first token: closes after its turn's result and the grace", (lc.should_close(731), lc.should_close(734)), (False, True))
    # 1c. A notification whose turn never comes lapses after NOTIF_WAIT_S, so the session cannot hang on it.
    lc = Lifecycle()
    lc.feed(ev(type="system", subtype="task_started", task_id="a"), 5)
    lc.feed(ev(type="system", subtype="background_tasks_changed", tasks=[{"task_id": "a"}]), 6)
    lc.feed(ev(type="result", subtype="success"), 10)
    lc.feed(ev(type="system", subtype="task_notification", task_id="a", status="completed"), 100)
    lc.feed(ev(type="system", subtype="background_tasks_changed", tasks=[]), 100)
    check("a notification's turn that never comes: open at 200 s", lc.should_close(200), False)
    check("then lapses and closes after the grace", (lc.should_close(221), lc.should_close(225)), (False, True))
    # 1d. A subagent's own task notifies the subagent: no turn of the agent's is opened by it.
    lc = Lifecycle()
    lc.feed(ev(type="system", subtype="task_started", task_id="sb", owned_by_subagent=True), 5)
    lc.feed(ev(type="result", subtype="success"), 10)
    lc.feed(ev(type="system", subtype="task_notification", task_id="sb", status="completed"), 20)
    check("a subagent's task notification opens no turn of the agent's", lc.should_close(24), True)
    # 2. The plain case: no background work; close right after the grace.
    lc = Lifecycle()
    lc.feed(ev(type="assistant", message={}), 1)
    lc.feed(ev(type="result", subtype="success"), 50)
    check("no background work: closes after the grace", (lc.should_close(51), lc.should_close(54)), (False, True))
    # 4. the study's narrower close (2026-10-21): every task job completed and the turn ended with a result, a
    # background poll still alive (Q-prepP6-sonnet-L0-nosub's self-matching pgrep) -> close after the grace; one job
    # left -> stay open, as before.
    # Since the work-done grace the stuck poll is closed after WORK_DONE_GRACE_S, not
    # GRACE_S: the work check at 11 s starts the grace, so 1810 s is still open and 1811 s closes.
    for done, want, name in ((True, (False, False, True), "all jobs done, a stuck poll alive: closes after the work-done grace"),
                             (False, (False, False, False), "one job left, a stuck poll alive: stays open")):
        lc = Lifecycle(work_done=lambda d=done: d)
        lc.feed(ev(type="system", subtype="task_started", task_id="poll"), 5)
        lc.feed(ev(type="system", subtype="background_tasks_changed", tasks=[{"task_id": "poll"}]), 6)
        lc.feed(ev(type="result", subtype="success"), 10)
        check(name, (lc.should_close(11), lc.should_close(1810), lc.should_close(1811)), want)
    # 4b. The known answers for the grace. (i) G-g4nb-opus-L0-r2's case: work done, the agent's `sleep 300 && echo ALLDONE`
    # alive as its background task; the task ends at 310 s, its notification opens a turn, the agent answers and ends
    # it: answered_in_grace, then the session closes by the first rule after GRACE_S.
    lc = Lifecycle(work_done=lambda: True)
    lc.feed(ev(type="system", subtype="task_started", task_id="w", owned_by_subagent=False), 5)
    lc.feed(ev(type="system", subtype="background_tasks_changed", tasks=[{"task_id": "w", "description": "sleep 300 && echo ALLDONE"}]), 6)
    lc.feed(ev(type="result", subtype="success"), 10)
    check("work done, the agent's 300-s wait alive: open past GRACE_S and through the wait",
          (lc.should_close(11), lc.should_close(30), lc.should_close(300)), (False, False, False))
    lc.feed(ev(type="system", subtype="task_notification", task_id="w", status="completed"), 310)
    lc.feed(ev(type="system", subtype="background_tasks_changed", tasks=[]), 310)
    check("its notification's turn opening: open", lc.should_close(312), False)
    lc.feed(ev(type="assistant", message={"content": [{"type": "text", "text": "ALLDONE: the best config is ..."}]}), 315)
    lc.feed(ev(type="result", subtype="success"), 320)
    check("the agent answered in the grace: closes after GRACE_S, saying answered_in_grace with the seconds waited",
          (lc.should_close(321), lc.should_close(324), "outcome=answered_in_grace waited_s=304.0" in (lc.close_reason or ""),
           (lc.close_reason or "").startswith("lifecycle: a result with no background task alive")), (False, True, True, True))
    # (ii) A never-ending background loop: no turn comes, so closed_after_grace at 1,800 s, by the work-done close that
    # run_one.py follows with SIGTERM (its WORK_DONE_PREFIX)
    lc = Lifecycle(work_done=lambda: True)
    lc.feed(ev(type="system", subtype="background_tasks_changed", tasks=[{"task_id": "loop", "description": "while true; do sleep 60; done"}]), 6)
    lc.feed(ev(type="result", subtype="success"), 10)
    check("a never-ending loop: open at 1810 s, closed at 1811 s, saying closed_after_grace",
          (lc.should_close(11), lc.should_close(1810), lc.should_close(1811),
           "outcome=closed_after_grace waited_s=1800.0 grace_s=1800" in (lc.close_reason or ""),
           (lc.close_reason or "").startswith("lifecycle: every task job completed")), (False, False, True, True, True))
    # (iii) A background subagent's own messages during the grace are not the agent's answer
    lc = Lifecycle(work_done=lambda: True)
    lc.feed(ev(type="system", subtype="background_tasks_changed", tasks=[{"task_id": "s1", "task_type": "local_agent"}]), 6)
    lc.feed(ev(type="result", subtype="success"), 10)
    lc.should_close(11)
    lc.feed(ev(type="assistant", parent_tool_use_id="toolu_x", message={"content": [{"type": "text", "text": "subagent"}]}), 20)
    check("a subagent's message in the grace is not answered_in_grace, and opens no turn",
          (lc.grace_outcome, lc.turn_active), (None, False))
    check("so the grace still ends at 1,800 s: closed_after_grace",
          (lc.should_close(1810), lc.should_close(1811), "outcome=closed_after_grace" in (lc.close_reason or "")),
          (False, True, True))
    # (iv) The task ends but its notification's turn never comes: task_ended_no_answer, after NOTIF_WAIT_S and GRACE_S
    lc = Lifecycle(work_done=lambda: True)
    lc.feed(ev(type="system", subtype="task_started", task_id="w", owned_by_subagent=False), 5)
    lc.feed(ev(type="system", subtype="background_tasks_changed", tasks=[{"task_id": "w"}]), 6)
    lc.feed(ev(type="result", subtype="success"), 10)
    lc.should_close(11)
    lc.feed(ev(type="system", subtype="task_notification", task_id="w", status="completed"), 100)
    lc.feed(ev(type="system", subtype="background_tasks_changed", tasks=[]), 100)
    check("a task that ends with no turn after it: open until the notification lapses, then closes, saying so",
          (lc.should_close(200), lc.should_close(221), lc.should_close(225),
           "outcome=task_ended_no_answer" in (lc.close_reason or "")), (False, False, True, True))
    # (v) No grace ran: the close reason is exactly as before (every run without the case reads unchanged)
    lc = Lifecycle(work_done=lambda: True)
    lc.feed(ev(type="result", subtype="success"), 10)
    check("no background task: the close reason carries no grace tag",
          (lc.should_close(14), "work_done_grace" in (lc.close_reason or "")), (True, False))
    lc = Lifecycle(work_done=lambda: True)
    lc.feed(ev(type="system", subtype="background_tasks_changed", tasks=[{"task_id": "poll"}]), 6)
    check("all jobs done but the turn still running: stays open", lc.should_close(40), False)
    lc = Lifecycle(work_done=lambda: (_ for _ in ()).throw(OSError("unreadable")))
    lc.feed(ev(type="system", subtype="background_tasks_changed", tasks=[{"task_id": "poll"}]), 6)
    lc.feed(ev(type="result", subtype="success"), 10)
    check("no work check without run_dir, and an error is no 'done': stays open", (Lifecycle().should_close(99), lc.should_close(60)), (False, False))
    # 5. An open pool (goal packages): the real G-passbar-sonnet-L0-v2 run dir, closed at 19:57:19Z with 4 of 16 runs
    # done and contra at step 2387 of 3040. The job-list rule reads it done (min 0 met); the open-pool rule never does,
    # so only the agent's final answer with nothing alive, or the horizon, ends it
    import os as _os
    real = _os.path.expanduser("~/computewise-runs/G-passbar-sonnet-L0-v2")
    if _os.path.exists(_os.path.join(real, "occupancy.jsonl")):
        check("an open pool with an instance running is never 'work done'", work_done_in(real)(), False)
    else:
        print("  skip an open pool (no ~/computewise-runs/G-passbar-sonnet-L0-v2 on this machine)")
    # 6. A pending ScheduleWakeup: its result at 100 s with delaySeconds 1500 keeps
    # the session past the agent's result until the wakeup's turn; a wakeup that never fires closes at 1600 + 120 s
    def wake(lc, t):
        lc.feed(ev(type="assistant", message={"content": [{"type": "tool_use", "id": "w1", "name": "ScheduleWakeup",
                                                           "input": {"delaySeconds": 1500, "prompt": "check"}}]}), t)
        lc.feed(ev(type="user", message={"content": [{"type": "tool_result", "tool_use_id": "w1", "content": "Next wakeup scheduled"}]}), t)
        lc.feed(ev(type="result", subtype="success"), t + 5)
    lc = Lifecycle(); wake(lc, 100)
    check("a pending wakeup keeps the session past the grace", (lc.should_close(110), lc.should_close(1500)), (False, False))
    lc.feed(ev(type="assistant", message={"content": [{"type": "text", "text": "checking"}]}), 1601)
    lc.feed(ev(type="result", subtype="success"), 1610)
    check("the wakeup fired: its turn's result then closes after the grace", (lc.should_close(1611), lc.should_close(1614)), (False, True))
    lc = Lifecycle(); wake(lc, 100)
    check("a wakeup that never fires: open until its time plus the grace, then closes, saying so",
          (lc.should_close(1700), lc.should_close(1721), "did not fire" in (lc.close_reason or "")), (False, True, True))
    # 6b. A live CronCreate. The run's
    # own times: CronCreate `*/20 * * * *` at 18:51:52Z, its result names job b84ca272, the turn ends 4 s later; the old rule
    # closed 3 s after that. Now the session stays open to the 19:00Z match, plus up to 10 min (half the period) of the
    # product's lateness, plus the grace; its fired turn re-arms it while work remains; a CronDelete or work done ends it.
    import datetime as _dtt
    T = _dtt.datetime(2026, 10, 29, 18, 51, 52, tzinfo=_dtt.timezone.utc).timestamp()
    F = _dtt.datetime(2026, 10, 29, 19, 0, 0, tzinfo=_dtt.timezone.utc).timestamp()
    def cron(lc, t, expr="*/20 * * * *", jid="b84ca272", recurring=None):
        inp = {"cron": expr, "prompt": "check the queue"}
        if recurring is not None:
            inp["recurring"] = recurring
        lc.feed(ev(type="assistant", message={"content": [{"type": "tool_use", "id": "c1", "name": "CronCreate", "input": inp}]}), t)
        lc.feed(ev(type="user", message={"content": [{"type": "tool_result", "tool_use_id": "c1", "content":
            f"Scheduled recurring job {jid} (Every 20 minutes). Next run: Oct 29 7:00pm. Session-only"}]}), t)
        lc.feed(ev(type="result", subtype="success"), t + 4)
    check("cron_next_fire matches the scorer's rule: */20 after 18:51:52Z is 19:00:00Z", cron_next_fire("*/20 * * * *", T), F)
    lc = Lifecycle(work_done=lambda: False); cron(lc, T)
    check("the failing case: open 3 s after the result, and at the 19:00 match", (lc.should_close(T + 8), lc.should_close(F)), (False, False))
    check("open through the product's lateness (half of 20 min) and the grace", lc.should_close(F + 600 + 119), False)
    check("a cron that never fires: closes after its match + lateness + grace, saying so",
          (lc.should_close(F + 600 + 121), "CronCreate job was live" in (lc.close_reason or "")), (True, True))
    lc = Lifecycle(work_done=lambda: False); cron(lc, T)
    lc.feed(ev(type="assistant", message={"content": [{"type": "text", "text": "checking"}]}), F + 40)
    lc.feed(ev(type="result", subtype="success"), F + 60)
    check("it fired with work left: re-armed, open to the 19:20 match", (lc.should_close(F + 70), lc.should_close(F + 1200)), (False, False))
    lc.feed(ev(type="assistant", message={"content": [{"type": "tool_use", "id": "d1", "name": "CronDelete", "input": {"id": "b84ca272"}}]}), F + 1210)
    lc.feed(ev(type="user", message={"content": [{"type": "tool_result", "tool_use_id": "d1", "content": "Cancelled job b84ca272."}]}), F + 1210)
    lc.feed(ev(type="result", subtype="success"), F + 1215)
    check("CronDelete succeeded: closes after the grace", (lc.should_close(F + 1216), lc.should_close(F + 1219)), (False, True))
    lc = Lifecycle(work_done=lambda: True); cron(lc, T)
    lc.feed(ev(type="assistant", message={"content": [{"type": "text", "text": "all 40 done, table"}]}), F + 40)
    lc.feed(ev(type="result", subtype="success"), F + 60)
    check("its fire found every task job done: the cron is spent, closes after the grace", (lc.should_close(F + 61), lc.should_close(F + 64)), (False, True))
    lc = Lifecycle(work_done=lambda: False); cron(lc, T, expr="0 19 29 10 *", recurring=False)
    lc.feed(ev(type="assistant", message={"content": [{"type": "text", "text": "reminder"}]}), F + 10)
    lc.feed(ev(type="result", subtype="success"), F + 20)
    check("a one-shot fired: gone, closes after the grace", (lc.should_close(F + 21), lc.should_close(F + 24)), (False, True))
    lc = Lifecycle(work_done=lambda: False)
    lc.feed(ev(type="assistant", message={"content": [{"type": "tool_use", "id": "c1", "name": "CronCreate", "input": {"cron": "*/20 * * * *", "prompt": "check the queue"}}]}), T)
    lc.feed(ev(type="user", message={"content": [{"type": "tool_result", "tool_use_id": "c1", "content": "Scheduled recurring job b84ca272 (Every 20 minutes)."}]}), T)
    lc.feed(ev(type="assistant", message={"content": [{"type": "text", "text": "still working"}]}), T + 10)
    lc.feed(ev(type="result", subtype="success"), F + 900)
    check("a match that passed during a long turn: open after the turn's result, for the overdue fire",
          (lc.should_close(F + 904), lc.should_close(F + 900 + 179)), (False, False))
    check("and closes if even the overdue fire never comes", lc.should_close(F + 900 + 181), True)
    lc = Lifecycle(work_done=lambda: False); cron(lc, T)
    lc.feed(ev(type="assistant", message={"content": [{"type": "text", "text": "a task notification's turn"}]}), F - 60)
    lc.feed(ev(type="result", subtype="success"), F - 50)
    check("a turn 60 s before a recurring match is not its fire: still open for 19:00", lc.should_close(F + 30), False)
    lc = Lifecycle(work_done=lambda: False)
    lc.feed(ev(type="assistant", message={"content": [{"type": "tool_use", "id": "c1", "name": "CronCreate", "input": {"cron": "*/20 * * * *", "prompt": "x"}}]}), T)
    lc.feed(ev(type="user", message={"content": [{"type": "tool_result", "tool_use_id": "c1", "is_error": True, "content": "Invalid cron"}]}), T)
    lc.feed(ev(type="result", subtype="success"), T + 4)
    check("a CronCreate that errored holds nothing: closes after the grace", lc.should_close(T + 8), True)
    lc = Lifecycle(work_done=lambda: False); cron(lc, T, expr="not a cron")
    check("an expression that does not parse holds nothing: closes after the grace", lc.should_close(T + 8), True)
    # 3. A subagent counts as background work (task_type local_agent), as in Q-prepP6-haiku-L0.
    lc = Lifecycle()
    lc.feed(ev(type="system", subtype="background_tasks_changed",
               tasks=[{"task_id": "s1", "task_type": "local_agent", "description": "Build and run contra"}]), 30)
    lc.feed(ev(type="result", subtype="success"), 102)
    check("background subagents alive at the result: stays open past 600 s", lc.should_close(703), False)
    # 4. A system record whose message is a string (Q-p20tail-sonnet-L0-bashdefault-r2, 2026-10-22 19:23Z: Claude Code's
    # safety check refused an rm, and system/permission_denied carries "message": "Dangerous rm operation ..."): fed
    # without raising, and the turn goes on as before
    lc = Lifecycle()
    lc.feed(ev(type="assistant", message={"content": [{"type": "tool_use", "id": "t1", "name": "Bash"}]}), 5)
    try:
        lc.feed(ev(type="system", subtype="permission_denied", tool_name="Bash", message="Dangerous rm operation detected"), 6)
        lc.feed("[1, 2]", 6)
        raised = False
    except Exception:
        raised = True
    check("a string message and a non-object line are fed without raising; the turn stays open", (raised, lc.should_close(30)), (False, False))
    # 7. Box scheduler jobs. The known answer: the agent submits 2 sbatch
    # jobs of 60 s and ends its turn; the session stays open until both end, then closes after the grace.
    live = {"n": 0}
    lc = Lifecycle(sched_active=lambda: live["n"])
    lc.feed(ev(type="assistant", message={"content": [{"type": "tool_use", "id": "b1", "name": "Bash"}]}), 5)
    live["n"] = 2
    lc.feed(ev(type="result", subtype="success"), 10)
    check("2 scheduler jobs live after the result: open past the grace and to 60 s",
          (lc.should_close(14), lc.should_close(40), lc.should_close(69)), (False, False, False))
    live["n"] = 1
    check("one job left: open", lc.should_close(70), False)
    live["n"] = 0
    check("both jobs ended: open within the grace after the last one", lc.should_close(71), False)
    check("then closes, saying the scheduler's jobs ended", (lc.should_close(74), "scheduler" in (lc.close_reason or "")),
          (True, True))
    # 7b. A turn that starts while jobs run (the agent polls with its own Bash) is not cut, and its result is checked again
    live["n"] = 1
    lc = Lifecycle(sched_active=lambda: live["n"])
    lc.feed(ev(type="result", subtype="success"), 10)
    lc.feed(ev(type="assistant", message={"content": [{"type": "text", "text": "checking"}]}), 30)
    check("a new turn while a job runs: open", lc.should_close(40), False)
    lc.feed(ev(type="result", subtype="success"), 45)
    check("its result with the job still live: open", lc.should_close(60), False)
    live["n"] = 0
    check("job ended: closes after the grace", (lc.should_close(61), lc.should_close(65)), (False, True))
    # 7c. A count that cannot be read keeps the session open; no scheduler (the file absent) changes nothing
    lc = Lifecycle(sched_active=lambda: int("x"))
    lc.feed(ev(type="result", subtype="success"), 10)
    check("an unreadable scheduler count: open", lc.should_close(60), False)
    import tempfile, os as _os
    with tempfile.TemporaryDirectory() as d:
        lc = Lifecycle(run_dir=d, work_done=lambda: False)
        lc.feed(ev(type="result", subtype="success"), 10)
        check("a run dir with no box/sched-active (no scheduler): closes after the grace as before",
              (lc.should_close(11), lc.should_close(14)), (False, True))
        _os.makedirs(_os.path.join(d, "box"))
        open(_os.path.join(d, "box", "sched-active"), "w").write("3\n")
        lc = Lifecycle(run_dir=d, work_done=lambda: False)
        lc.feed(ev(type="result", subtype="success"), 10)
        check("a run dir whose box/sched-active reads 3: open", lc.should_close(30), False)
        open(_os.path.join(d, "box", "sched-active"), "w").write("0\n")
        check("it reads 0: closes after the grace", (lc.should_close(31), lc.should_close(34)), (False, True))
    # 7d. Every task job completed while a scheduler job is still live (a stuck job): the second rule closes, as for a
    # stuck background poll, and says which work was still alive
    lc = Lifecycle(work_done=lambda: True, sched_active=lambda: 1)
    lc.feed(ev(type="result", subtype="success"), 10)
    check("all task jobs done, a scheduler job still live: closes after the grace",
          (lc.should_close(11), lc.should_close(30), "box scheduler" in (lc.close_reason or "")), (False, True, True))
    # 7e. A background wait with its own timeout holds the work-done grace to that bound plus WAKE_GRACE_S (the analysis 2026-11-05
    # 21:16 MST, G-k2p-sonnet-L0-v2-r2's shape: a 36,000-s-bounded wait started at 12,661 s, the work done at 45,342 s; the old grace closed
    # at 47,142 s, about 25 min before the bound at 48,661 s; now it holds to 48,781 s). A task with no timeout keeps the 1,800-s grace
    lc = Lifecycle(work_done=lambda: True)
    lc.feed(ev(type="assistant", message={"content": [{"type": "tool_use", "id": "w", "name": "Bash",
                                                        "input": {"command": "while pgrep -f x; do sleep 120; done", "run_in_background": True, "timeout": 36000000}}]}), 12660)
    lc.feed(ev(type="system", subtype="task_started", task_id="b3q", tool_use_id="w"), 12661)
    lc.feed(ev(type="system", subtype="background_tasks_changed", tasks=[{"task_id": "b3q", "description": "wait"}]), 12661)
    lc.feed(ev(type="result", subtype="success"), 12663)
    lc.done_since = None
    check("a 36,000-s-bounded background wait, the work done at 45,342 s: open at the old 1,800-s close (47,142 s) and at the bound (48,661 s)",
          (lc.should_close(45342), lc.should_close(47142 + 15), lc.should_close(48661 + 60)), (False, False, False))
    check("... closes once the bound plus WAKE_GRACE_S has passed (48,781 s), and says the grace closed it",
          (lc.should_close(48661 + WAKE_GRACE_S + 5), "closed_after_grace" in (lc.close_reason or "")), (True, True))
    lc = Lifecycle(work_done=lambda: True)
    lc.feed(ev(type="assistant", message={"content": [{"type": "tool_use", "id": "u", "name": "Bash", "input": {"command": "while pgrep -f x; do sleep 60; done", "run_in_background": True}}]}), 100)
    lc.feed(ev(type="system", subtype="task_started", task_id="u1", tool_use_id="u"), 101)
    lc.feed(ev(type="system", subtype="background_tasks_changed", tasks=[{"task_id": "u1", "description": "wait"}]), 101)
    lc.feed(ev(type="result", subtype="success"), 102)
    check("an unbounded background wait keeps the 1,800-s grace (G-sfck-sonnet-L0's shape)",
          (lc.should_close(200), lc.should_close(200 + 1800 - 20), lc.should_close(200 + 1800 + 20)), (False, False, True))
    # 7f. work_done_in reads a rank with a job_hello and no job_bye as GPU work live
    import tempfile as _tf, os as _os2
    with _tf.TemporaryDirectory() as d2:
        json.dump({"task_id": "t", "jobs": [{"job_id": "a", "command": "x", "records": []}]}, open(_os2.path.join(d2, "task.json"), "w"))
        open(_os2.path.join(d2, "occupancy.jsonl"), "w").write("")
        check("work_done_in on an empty run is not done (no task job completed)", work_done_in(d2)(), False)
    # 7g. The agent's gpu-harness board holds the grace while a job runs on it, GPU or CPU (the analysis 2026-11-05 22:59 MST,
    # G-fyld-haiku-A3: fineweb_prep, a CPU prep on GPU slot 1, ran with no occupancy rank; the close came before its wait returned)
    with _tf.TemporaryDirectory() as d3:
        home = _os2.path.join(d3, "home"); runs = _os2.path.join(home, ".gpuh", "runs", "20261105T212302"); _os2.makedirs(runs)
        bp = _os2.path.join(runs, "board.json")
        json.dump({"gpu_job": {"0": None, "1": "fineweb_prep", "2": None, "3": None},
                   "jobs": {"fineweb_prep": {"state": "running", "gpu": 1, "kind": "prep"}, "fineweb_s0": {"state": "waiting"}}}, open(bp, "w"))
        a = board_busy(None, home=home)
        json.dump({"gpu_job": {"0": None, "1": None}, "jobs": {"x_prep": {"state": "running", "gpu": None, "kind": "prep"}}}, open(bp, "w"))
        b = board_busy(None, home=home)
        json.dump({"gpu_job": {"0": None, "1": None}, "jobs": {"x": {"state": "done"}, "y": {"state": "cancelled"}}}, open(bp, "w"))
        c = board_busy(None, home=home)
        open(bp, "w").write("{not json")
        d_ = board_busy(None, home=home)
        check("board: a prep on a GPU slot holds; a running CPU job holds; all done or cancelled does not; an unreadable board does not",
              (a, b, c, d_, board_busy(None, home=_os2.path.join(d3, "nohome"))), (True, True, False, False, False))
        # through run.json's disk_image, as run_one writes it
        run_dir = _os2.path.join(d3, "run"); _os2.makedirs(run_dir)
        json.dump({"disk_image": d3 + ".img"}, open(_os2.path.join(run_dir, "run.json"), "w"))
        json.dump({"gpu_job": {"1": "fineweb_prep"}, "jobs": {}}, open(bp, "w"))
        check("board_busy finds the board from run.json's disk_image (<disk>.img -> <disk>/home)", board_busy(run_dir), True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    sys.exit(__doc__)
