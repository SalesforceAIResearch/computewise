"""The ComputeWise campaign driver: takes runs from the study's backlog and runs each one end to end on a fleet
machine, then keeps everything it produced. Runs on the controller; every run itself happens on the machine.

usage (from the controller, light work only):
  python3 -I driver.py status                         claims, and what each machine could take next
  python3 -I driver.py next [--machine M]             the line that would be taken next on M, with why others wait
  python3 -I driver.py take --machine M [--dry-run]   claim the next line for M and run it to the end (blocks)
  python3 -I driver.py loop --machines M1,M2 [--once] one worker per machine, each taking lines until none can be taken
  python3 -I driver.py collect <run_id>               redo a claimed run's collection (ledger, score, controller copy);
                                                      never starts a run

Inputs (read-only; the study owns them): experiments/campaign/campaign.json, backlog-*.jsonl, AGENTS.md there.
Outputs: experiments/campaign/claims/<run_id>.json (the runner owns claims/), the run directory on the machine,
~/computewise-runs/<run_id>/ on the controller, and this directory's driver-log.jsonl.

A line is taken when its wave is open, it has no claim, and every `requires` entry is met (requirement_met below).
The claim is created atomically (O_EXCL) before anything starts, so two workers can never take one line. A claim
moves claimed -> running -> done | void | failed. If nothing was started (a machine check failed first), the claim
file is removed and the event logged, because the line was never run.

Per run, in order:
  1. checks on the machine: no other box up (one box per machine), health.sh (--need-tunnel for model runs), and the
     60 s memory baseline (gcp's --mem-baseline --mem-json), kept in the controller copy as health.txt;
  2. a model run gets its ceiling (score/set_ceiling.py) and a stop relay (score/stop_relay.py) for its tag;
  3. run_smoke.py on the machine under systemd-run, with the line's fields; the driver waits for the unit to end;
  4. G2: the tag's ledger turns are dumped (score/ledger.py, read-only) and copied into the run directory as ledger.jsonl;
  5. the run is re-scored on the machine (score/score_all.py): served model (G4), validity, provenance (G8);
  6. G3: the run directory is copied to ~/computewise-runs/<run_id>/ and checked file by file against a sha256 manifest made
     on the machine; the balloon view is added if the machine is an E2;
  7. the claim becomes done (valid), void (served model or a voiding check), or failed (needs review), and the lease
     is released.
"""
import argparse, glob, datetime, hashlib, json, os, re, shutil, subprocess, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.dirname(HERE)
EXP = os.path.dirname(RUNNER)
CAMPAIGN = os.path.join(EXP, "campaign")
CLAIMS = os.path.join(CAMPAIGN, "claims")
VM = os.path.join(EXP, "gcp", "host", "vm.sh")
HEALTH = os.path.join(EXP, "gcp", "host", "health.sh")
FLEET = os.path.join(EXP, "gcp", "fleet", "fleet.py")
BALLOON = os.path.join(EXP, "gcp", "host", "balloon.py")
MET = os.path.join(HERE, "requirements-met.json")
LOG = os.path.join(HERE, "driver-log.jsonl")
LAPTOP_RUNS = os.path.expanduser("~/computewise-runs")
# per task, the find -path pattern (relative to the run dir, "./" first) of data the copy leaves on the machine; its listing
# (sha256 and path) comes to the controller as workspace.excluded.lst (the study, 2026-10-24 21:12Z: H4's filtered/ shards)
# pkg-filter-yield: its recipes' filtered/ jsonl, 1,536 files and 11 GB of W0-fyld-check10's 11 GB on
# c3-18 (60 MB left), failed the copy; prep_observed reads the data/fineweb10B shards' sizes, which stay in the copy
COLLECT_EXCLUDE = {"pkg-h4-filter-ablation": "./workspace/home/project/runs/*/filtered/*",
                   "pkg-filter-yield": "./workspace/home/project/yieldlab/*/filtered/*"}
REMOTE_EXP = "/opt/computewise"
KEY = os.path.expanduser("~/.ssh/<key>")
PROJECT = "<your-gcp-project>"
NOISE = re.compile(r"environment\. tag|resource-manager tags|designate_project|NumPy|increasing_the_tcp|^WARNING: *$|^\s*$")
LOCK = threading.Lock()

def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def log(event, **f):
    rec = {"utc": utc(), "event": event, **f}
    with LOCK:
        open(LOG, "a").write(json.dumps(rec) + "\n")
    print(f"[driver {rec['utc']}] {event} " + " ".join(f"{k}={v}" for k, v in f.items() if k not in ("detail",)), flush=True)

# ---- machines ----------------------------------------------------------------------------------------------------

_FLEET_LAST = {}

def fleet():
    """The fleet's state from fleet.py. When fleet.py cannot answer (gcloud's reauth lapse, 2026-10-22 05:22Z: it printed
    an error, json.loads raised, and seven loops died), the last good answer is returned, marked stale, so a loop waits
    instead of dying; with no good answer yet, every machine reads as unknown and nothing is claimed."""
    try:
        p = subprocess.run([sys.executable, "-I", FLEET, "status", "--json"], capture_output=True, text=True, timeout=180)
        got = {m["name"]: m for m in json.loads(p.stdout)["machines"]}
        _FLEET_LAST.clear(); _FLEET_LAST.update(got)
        return got
    except (ValueError, KeyError, TypeError, subprocess.TimeoutExpired, OSError):
        return {k: dict(v, stale=True) for k, v in _FLEET_LAST.items()}

def fleet_unanswered(fl, machine):
    """True when fleet() did not answer for this machine: its row is stale (fleet.py failed and the last good answer came
    back), or there is no row while the answer is empty or stale (fleet.py failed before any good answer). A row missing
    from a fresh, non-empty answer is a real absence."""
    row = (fl or {}).get(machine)
    if row:
        return bool(row.get("stale"))
    return not fl or any((v or {}).get("stale") for v in fl.values())

def env_for(machine, zone=None):
    e = dict(os.environ, P006_VM=machine)
    if zone:
        e["P006_ZONE"] = zone
    return e

def ssh(machine, cmd, zone=None, timeout=600):
    p = subprocess.run(["bash", VM, "ssh", cmd], capture_output=True, text=True, timeout=timeout, env=env_for(machine, zone))
    return p.returncode, "\n".join(l for l in p.stdout.splitlines() if not NOISE.search(l))

def gate_rtt(machine, zone, n=15):
    """The model route's round trip from the machine (the study, 2026-10-20: c3-5 in us-east1 is 98 ms against
    30 ms in us-central1): n keyless count_tokens calls to the gate's tunnel address, each refused (403) by the controller
    gate, timed by curl, as the study measured it. A label on the run, never a check."""
    rc, out = ssh(machine, f"for i in $(seq {n}); do curl -s -o /dev/null -m 10 -w '%{{http_code}} %{{time_total}}\\n' "
                           "-X POST http://10.200.0.1:8790/v1/messages/count_tokens -H 'content-type: application/json' "
                           "-d '{}'; done; true", zone, timeout=300)
    ms, codes = [], {}
    for l in out.splitlines():
        f = l.split()
        if len(f) == 2 and f[0].isdigit():
            codes[f[0]] = codes.get(f[0], 0) + 1
            try:
                ms.append(1000.0 * float(f[1]))
            except ValueError:
                pass
    ms.sort()
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "calls": len(ms), "http_codes": codes,
            "median_ms": round(ms[len(ms) // 2], 1) if ms else None, "min_ms": round(ms[0], 1) if ms else None,
            "max_ms": round(ms[-1], 1) if ms else None,
            "how": f"{n} keyless count_tokens calls from the machine to http://10.200.0.1:8790 (refused 403 by the gate)"}

def upload(machine, local, remote, zone=None):
    subprocess.run(["bash", VM, "scp", local, remote], capture_output=True, text=True, timeout=600, env=env_for(machine, zone))

def download(machine, zone, remote, local, tries=2):
    """Remote to local, binary-safe: gcloud's own scp, not vm.sh ssh (whose noise filter would cut binary output). Tried
    twice, 30 s apart (the study, 2026-10-23 14:16Z: Q-stagger-haiku-L0-bashdefault-r5's copy failed at 14:11:36Z while
    isolation's 2b8f staging was writing to the same machine; its tarball and manifest were intact on the machine); the
    stderr of a failure is logged."""
    for i in range(tries):
        p = subprocess.run(["gcloud", "compute", "scp", f"{machine}:{remote}", local, f"--project={PROJECT}", f"--zone={zone}",
                            "--tunnel-through-iap", f"--ssh-key-file={KEY}", "--strict-host-key-checking=no"],
                           capture_output=True, text=True, timeout=1800)
        if p.returncode == 0:
            return 0
        log("download-retry" if i + 1 < tries else "download-failed", machine=machine, remote=remote, rc=p.returncode,
            detail=(p.stderr or "")[-300:])
        if i + 1 < tries:
            time.sleep(30)
    return p.returncode

# ---- requirements ------------------------------------------------------------------------------------------------

def superseded(lines):
    """A run_id's last line supersedes its earlier ones, in the first one's place in the backlog's order (2026-11-03
    10:08Z: the study's supersede lines added fix:plan-isolate-exempt to the 16 TTC adoption lines, but pick tried
    the older line first, met without the new key, and c3-19 took W0-tsci-check10 and W0-tops-check10 onto a plan.py that
    refuses them). Kept together, a supersede line could drop a requirement but never add one."""
    last = {l["run_id"]: l for l in lines}
    out, seen = [], set()
    for l in lines:
        if l["run_id"] not in seen:
            seen.add(l["run_id"]); out.append(last[l["run_id"]])
    return out

def load_campaign():
    c = json.load(open(os.path.join(CAMPAIGN, "campaign.json")))
    backlogs = sorted(f for f in os.listdir(CAMPAIGN) if f.startswith("backlog-") and f.endswith(".jsonl"))
    lines = [json.loads(l) for l in open(os.path.join(CAMPAIGN, backlogs[-1])) if l.strip()]
    withdrawn = {l["run_id"] for l in lines if l.get("withdrawn")}
    return c, [l for l in superseded(lines) if not l.get("withdrawn") and l["run_id"] not in withdrawn], backlogs[-1]

def claims():
    os.makedirs(CLAIMS, exist_ok=True)
    return {f[:-5]: json.load(open(os.path.join(CLAIMS, f))) for f in os.listdir(CLAIMS) if f.endswith(".json")}

def declared():
    return json.load(open(MET)).get("met", {}) if os.path.exists(MET) else {}

def requirement_met(req, machine_row, cl, images_on_machine):
    """(met, why). Checked here: machine:no-balloon (the machine type is not E2), image:<label> (an image with that
    p006.task-env label is on the machine), run:<id> (its claim is done), task:<id> (the task file is in the
    registry), model-runs:<task> (the task file's model_runs.allowed is true; added to every model line), task-file:<task>
    (the machine's copy of the task file is the controller registry's; added to every line), gated-image:<label> (the
    machine's images with that label are exactly the declared gate's id; added to every line). Everything else is met only when declared in requirements-met.json, with who declared it and when."""
    kind, _, arg = req.partition(":")
    if kind == "machine" and arg == "no-balloon":
        mt = (machine_row or {}).get("machine_type", "")
        return (bool(mt) and not mt.startswith("e2-")), f"machine type {mt or 'unknown'}"
    if kind == "image":
        return (arg in images_on_machine), f"images on the machine: {sorted(images_on_machine) or 'none'}"
    if kind == "run":
        st = (cl.get(arg) or {}).get("status")
        return st == "done", f"{arg} is {st or 'unclaimed'}"
    if kind == "task":
        p = os.path.join(EXP, "gpu-tasks", "registry", "tasks", f"{arg}.json")
        return os.path.exists(p), f"{arg}.json {'in' if os.path.exists(p) else 'not in'} the registry"
    if kind == "runone-flags":
        # a line with harness_flags needs a run_one.py that takes --claude-arg (isolation's be1c73e8 and later); the
        # machine's pinned run_one.py says which it has
        pins = json.load(open(PINS)) if os.path.exists(PINS) else {}
        h = (((pins.get("by_machine") or {}).get(machine_row.get("name") or "") or {}).get("sha256") or {}).get("isolation/run_one.py", "")
        return bool(h) and any(h.startswith(x) for x in CLAUDE_ARG_RUN_ONE), f"the machine's pinned run_one.py is {h[:16] or 'unknown'}"
    if kind == "fifo-image":
        # a model line runs only where its image carries isolation's p006.box-stdin=fifo label, the images whose
        # run_one.py keeps the session open while background work runs (runner/harness/lifecycle.py)
        return f"fifo:{arg}" in images_on_machine, f"the machine's {arg} image {'carries' if f'fifo:{arg}' in images_on_machine else 'lacks'} p006.box-stdin=fifo"
    if kind == "gated-image":
        # the machine's images with this label must be exactly the declared gate's id, so a machine whose image moved
        # before the gate did waits for the gate instead of being refused at run time (which stops its loop)
        gated = {g["image_id"].replace("sha256:", "")[:12] for g in declared().values()
                 if isinstance(g, dict) and g.get("label") == arg and g.get("image_id")}
        ids = {x.split(":", 2)[2] for x in images_on_machine if x.startswith(f"imageid:{arg}:")}
        # a withdrawn gate's image is refused even when no other gate names the label (2026-10-22 19:58Z: the 922d KV box
        # 3469eda0, withdrawn for its 101 s CPU-built start-up, would otherwise pass as "no gate declared")
        wd = json.load(open(MET)).get("withdrawn", {}) if os.path.exists(MET) else {}
        bad = {g["image_id"].replace("sha256:", "")[:12] for g in wd.values()
               if isinstance(g, dict) and g.get("label") == arg and g.get("image_id")} - gated
        if ids & bad:
            return False, f"the machine's {arg} image(s) {sorted(ids & bad)} are a withdrawn gate's; waiting for a declared one"
        if not gated:
            return True, "no image gate declared for this label"
        # every image the machine holds with this label is a gated one; several gated ids at once let a rolling ship
        # run on shipped and unshipped machines alike
        return bool(ids) and ids <= gated, f"the machine's {arg} image(s) {sorted(ids)}; the gate names {sorted(gated)}"
    if kind == "codex-image":
        # a Codex line runs only on an image that carries Codex (isolation's p006.codex label)
        return f"codex:{arg}" in images_on_machine, f"the machine's {arg} image {'carries' if f'codex:{arg}' in images_on_machine else 'lacks'} p006.codex"
    if kind == "fix" and arg == "open-pool-lifecycle":
        # per machine: met where the machine's pinned runner/harness/lifecycle.py is the
        # open-pool fix, so the first shipped machine can take a goal line without waiting for the other nine
        pins = json.load(open(PINS)) if os.path.exists(PINS) else {}
        h = (((pins.get("by_machine") or {}).get(machine_row.get("name") or "") or {}).get("sha256") or {}).get("runner/harness/lifecycle.py", "")
        return h[:16] in OPEN_POOL_LIFECYCLE, f"the machine's pinned lifecycle.py is {h[:16] or 'unknown'}"
    if kind == "fix" and arg == "bash-timeout-line":
        # per machine: the pinned run_one.py takes --bash-timeout-ms, so a line's bash_timeouts_ms
        # reaches the box's BASH_DEFAULT_TIMEOUT_MS / BASH_MAX_TIMEOUT_MS
        h = machine_pinned(machine_row.get("name") or "", "isolation/run_one.py")
        return bool(h) and any(h.startswith(x) for x in BASH_TIMEOUT_RUN_ONE), f"the machine's pinned run_one.py is {h[:16] or 'unknown'}"
    if kind == "fix" and arg == "lifecycle-scheduler-jobs":
        # per machine: met where the pinned lifecycle.py keeps a session open
        # while the box scheduler holds a job not yet ended, so a session whose agent polls or waits is not cut off
        pins = json.load(open(PINS)) if os.path.exists(PINS) else {}
        h = (((pins.get("by_machine") or {}).get(machine_row.get("name") or "") or {}).get("sha256") or {}).get("runner/harness/lifecycle.py", "")
        return h[:16] in SCHED_LIFECYCLE, f"the machine's pinned lifecycle.py is {h[:16] or 'unknown'}"
    if kind == "fix" and arg == "wakeup-lifecycle":
        # per machine: met where the pinned lifecycle.py keeps a session open while a
        # ScheduleWakeup is pending (G-goalq-sonnet-L0 closed under one and never saw its results)
        pins = json.load(open(PINS)) if os.path.exists(PINS) else {}
        h = (((pins.get("by_machine") or {}).get(machine_row.get("name") or "") or {}).get("sha256") or {}).get("runner/harness/lifecycle.py", "")
        return h[:16] in WAKEUP_LIFECYCLE, f"the machine's pinned lifecycle.py is {h[:16] or 'unknown'}"
    if kind == "fix" and arg == "cron-lifecycle":
        # per machine: met where the pinned lifecycle.py keeps a session open while a
        # CronCreate job is live (G-kvreph-sonnet-L3-m2b closed 4 s after its */20 check and never saw its results)
        pins = json.load(open(PINS)) if os.path.exists(PINS) else {}
        h = (((pins.get("by_machine") or {}).get(machine_row.get("name") or "") or {}).get("sha256") or {}).get("runner/harness/lifecycle.py", "")
        return h[:16] in CRON_LIFECYCLE, f"the machine's pinned lifecycle.py is {h[:16] or 'unknown'}"
    if kind in ("grok-image", "opencode-image"):
        h_ = kind.split("-")[0]
        # the machine holds this label's harness image (HARN_IMAGE), by id, and it carries the harness (p006.<harness>)
        got = harness_image_for(arg, images_on_machine)
        held = bool(got)
        want = (got or "/".join(i.replace("sha256:", "")[:12] for i in HARN_IMAGES.get(arg) or []) or "(none named)").replace("sha256:", "")[:40]
        return held and f"{h_}:{arg}" in images_on_machine, (f"the machine {'holds' if held else 'lacks'} {arg}'s harness image "
                                                            f"{want}; p006.{h_} {'present' if f'{h_}:{arg}' in images_on_machine else 'absent'}")
    if kind in ("grok-runone", "opencode-runone"):
        h_ = kind.split("-")[0]
        h = machine_pinned(machine_row.get("name") or "", "isolation/run_one.py")
        return bool(h) and any(h.startswith(x) for x in HARN_RUN_ONE[h_]), f"the machine's pinned run_one.py is {h[:16] or 'unknown'}"
    if kind == "astracc-runone":
        h = machine_pinned(machine_row.get("name") or "", "isolation/run_one.py")
        return bool(h) and any(h.startswith(x) for x in ASTRACC_RUN_ONE), f"the machine's pinned run_one.py is {h[:16] or 'unknown'}"
    if kind == "agent-teams-runone":
        h = machine_pinned(machine_row.get("name") or "", "isolation/run_one.py")
        return bool(h) and any(h.startswith(x) for x in AGENT_TEAMS_RUN_ONE), f"the machine's pinned run_one.py is {h[:16] or 'unknown'}"
    if kind == "codex-runone":
        pins = json.load(open(PINS)) if os.path.exists(PINS) else {}
        h = (((pins.get("by_machine") or {}).get(machine_row.get("name") or "") or {}).get("sha256") or {}).get("isolation/run_one.py", "")
        if not (bool(h) and any(h.startswith(x) for x in CODEX_RUN_ONE)):
            return False, f"the machine's pinned run_one.py is {h[:16] or 'unknown'}"
        # a run_one that resumes every Codex run needs a codex_session with the resume rule beside it, or it refuses every
        # Codex line; read from the machine itself
        need = next((v for k, v in CODEX_SESSION_FOR.items() if h.startswith(k)), None)
        if need:
            got = next((x.split(":", 1)[1] for x in images_on_machine if x.startswith("codexsession:")), "")
            if not any(got.startswith(n) for n in need):
                return False, f"the machine's run_one.py {h[:8]} needs codex_session {'/'.join(sorted(need))}; it has {got[:8] or 'none read'}"
        return True, f"the machine's pinned run_one.py is {h[:16]}"
    if kind == "campaign-level":
        # the machine's pinned campaign.json carries this nudge level: known from the hashes of controller copies seen with it
        h = machine_pinned(machine_row.get("name") or "", "campaign/campaign.json")
        lv = campaign_levels().get(h, [])
        return arg in lv, f"the machine's pinned campaign.json {h[:8] or 'unknown'} {'has' if arg in lv else 'is not known to have'} level {arg}"
    if kind == "horizon-at-least":
        # horizon-at-least:<task>:<seconds> (the study, 2026-10-22 21:20Z: G-passbar-astra-L0-v3 waits for goal-pass-bar's
        # horizon to be raised, Astra's v2 having run past 16 h): the controller registry's task file has horizon_s at least
        # that; the task-file check makes the machine's copy the same
        task_, _, n = arg.rpartition(":")
        p = os.path.join(EXP, "gpu-tasks", "registry", "tasks", f"{task_}.json")
        hz = (json.load(open(p)) if os.path.exists(p) else {}).get("horizon_s")
        try:
            want = float(n)
        except ValueError:
            return False, f"horizon-at-least needs a number, got {n!r}"
        return (hz is not None and float(hz) >= want), f"{task_}.json horizon_s {hz} against {want:.0f}"
    if kind == "task-artifacts":
        # every registry artifact the task's stage reads (stage_task.py: files[] entries `from: artifacts/...` and the
        # install constraints) is on the machine with the controller's bytes (the study, 2026-10-23 02:16Z: three G-mlskv
        # lines failed at staging, artifacts/mls/env/mlsbench-kv/constraints.txt never synced)
        bad = [x for x, h in task_artifacts(arg) if f"artifact:{x}:{h}" not in images_on_machine]
        return not bad, (f"all {len(task_artifacts(arg))} staged artifact(s) are the controller's" if not bad else
                         f"{len(bad)} staged artifact(s) missing or different on the machine, e.g. {bad[0]}: isolation syncs registry/artifacts")
    if kind == "gt-common":
        p = os.path.join(EXP, "gpu-tasks", "build", "common.py")
        want = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
        ok = want is not None and f"gtcommon:{want}" in images_on_machine
        return ok, ("the machine's gpu-tasks/build/common.py is the controller's" if ok else
                    f"the machine's gpu-tasks/build/common.py differs from the controller's {(want or '')[:16]}: isolation deploys gpu-tasks/build")
    if kind == "skill-frozen":
        # skill-frozen:<name>: the machine holds the SKILL.md whose sha solution sent
        # analysis as frozen; the declaration names it (`sha256`), and images() reads the machine's skill hashes
        d0 = declared().get(req) or {}
        want = (d0.get("sha256") or "").lower()
        ok = bool(want) and any(x.startswith(f"skill:{arg}:") and x.split(":", 2)[2].startswith(want) for x in images_on_machine)
        return ok, (f"no frozen sha declared for {arg}" if not want else
                    f"the machine's {arg}/SKILL.md is the frozen {want[:16]}" if ok else f"the machine lacks the frozen {arg}/SKILL.md {want[:16]}")
    if kind == "scheduler-image":
        # scheduler-image:<label>: fake-slurm's check that the box image with that label
        # carries the scheduler, recorded by box id in scheduler:box's `label_images`
        sb = declared().get("scheduler:box") or {}
        want = {x.replace("sha256:", "")[:12] for x in (sb.get("label_images") or {}).get(arg, [])}
        have = {x.split(":", 2)[2] for x in images_on_machine if x.startswith(f"imageid:{arg}:")}
        return bool(want & have), f"scheduler checked on {arg} box(es) {sorted(want) or 'none'}; the machine has {sorted(have) or 'none'}"
    if kind == "brief-variant":
        # brief-variant:<task>:<variant>: the registry's task file carries that variant;
        # gpu-tasks writes it, and the task-file check makes the machine's copy the same
        task_, _, var = arg.partition(":")
        p = os.path.join(EXP, "gpu-tasks", "registry", "tasks", f"{task_}.json")
        v = ((json.load(open(p)) if os.path.exists(p) else {}).get("brief_variants") or {}).get(var) or {}
        return bool(v.get("text")), f"{task_}.json {'has' if v.get('text') else 'lacks'} brief_variants.{var}"
    if kind == "skill-synced":
        # skill-synced:<name>: the machine's solution/skills/<name>/SKILL.md is the
        # controller's, byte for byte (isolation syncs it); images() reads the machine's skill hashes
        p = os.path.join(EXP, "solution", "skills", arg, "SKILL.md")
        want = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
        ok = want is not None and f"skill:{arg}:{want}" in images_on_machine
        return ok, (f"no {arg}/SKILL.md on the controller" if want is None else
                    "the machine's SKILL.md is the controller's" if ok else f"the machine lacks {arg}/SKILL.md {want[:16]}")
    if kind == "task-file":
        p = os.path.join(EXP, "gpu-tasks", "registry", "tasks", f"{arg}.json")
        want = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
        ok = want is not None and f"taskfile:{arg}:{want}" in images_on_machine
        return ok, (f"{arg}.json not in the registry" if want is None else
                    "the machine's task file is the registry's" if ok else
                    "the machine's task file differs from the controller registry: deploy gpu-tasks' registry")
    if kind == "model-runs":
        # no task that can repeat a record goes to a model run: an agent that reads its
        # logs sees two seeds with one loss curve (P-pilotp6-haiku-L0). gpu-tasks computes it into every task file;
        # a task without the field is refused, not assumed
        p = os.path.join(EXP, "gpu-tasks", "registry", "tasks", f"{arg}.json")
        mr = (json.load(open(p)) if os.path.exists(p) else {}).get("model_runs")
        if not isinstance(mr, dict):
            return False, f"{arg}.json has no model_runs field"
        return mr.get("allowed") is True, f"model_runs.allowed {mr.get('allowed')}: {str(mr.get('problems') or '')[:160]}"
    d = declared().get(req)
    if d and d.get("machines") is not None:
        # a declaration made per machine (fix:mlskv-hf-cache-fill, the study 2026-10-22 21:46Z: isolation verifies each
        # machine's KV cache in turn): met only on the machines listed, so a line starts where its fix has landed
        m = machine_row.get("name") or ""
        if m not in d["machines"]:
            return False, f"declared by {d.get('by')} for {len(d['machines'])} machine(s); {m or 'this machine'} is not among them yet"
    if d and d.get("requires_run_one"):
        # a declaration bound to a run_one version too (scheduler:box, fake-slurm 2026-10-22 17:51Z: on a9d06a04 every
        # scheduler job fails at exec, the spool tmpfs being noexec; 9f7a7e30 mounts it exec): met only where the
        # machine's pinned run_one.py is one of them, and then the image rule below as well
        h = machine_pinned(machine_row.get("name") or "", "isolation/run_one.py")
        if not (h and any(h.startswith(x) for x in d["requires_run_one"])):
            return False, f"declared by {d.get('by')} for run_one {d['requires_run_one']}; the machine's pinned run_one.py is {h[:16] or 'unknown'}"
    if d and d.get("requires_image_ids"):
        # a gpu-mock feature that only a new image carries: met
        # only where the machine holds one of the images it was gated on, so a declaration made before the ship never
        # sends its lines to a machine still on the old image
        want = {x.replace("sha256:", "")[:12] for x in d["requires_image_ids"]}
        have = {x.split(":", 2)[2] for x in images_on_machine if x.startswith("imageid:")}
        return bool(want & have), (f"declared by {d.get('by')} at {d.get('utc')}; the machine "
                                   + ("holds" if want & have else "lacks") + f" its image {sorted(want)}")
    return bool(d), (f"declared by {d.get('by')} at {d.get('utc')}" if d else "not declared in requirements-met.json")

def images(machine, zone):
    """What the machine holds, in one ssh: the p006.task-env labels of its box images (isolation labels each box
    image with its task), and "taskfile:<task>:<sha256>" for each registry task file, so a line whose task file on
    the machine differs from the controller registry waits for a deploy instead of being refused at run time."""
    rc, out = ssh(machine, "sudo docker image ls --filter dangling=false -q --no-trunc | sort -u | xargs -r sudo docker image inspect --format "
                           "'LABEL {{index .Config.Labels \"p006.task-env\"}} {{.Id}} {{index .Config.Labels \"p006.box-stdin\"}} {{index .Config.Labels \"p006.role\"}}' 2>/dev/null; "
                           f"cd {REMOTE_EXP}/gpu-tasks/registry/tasks 2>/dev/null && sha256sum *.json | sed 's/^/TASKFILE /'; "
                           f"cd {REMOTE_EXP}/solution/skills 2>/dev/null && sha256sum */SKILL.md | sed 's/^/SKILLFILE /'; "
                           f"sha256sum {REMOTE_EXP}/gpu-tasks/build/common.py 2>/dev/null | sed 's/^/GTCOMMON /'; "
                           f"sha256sum {REMOTE_EXP}/codex/harness/codex_session.py 2>/dev/null | sed 's/^/CODEXSESSION /'; "
                           f"cd {REMOTE_EXP}/gpu-tasks/registry 2>/dev/null && echo ARTIFACTSREAD && find artifacts -type f -exec sha256sum {{}} + | sed 's/^/ARTIFACT /'; "
                           "sudo docker image ls --filter dangling=false -q --no-trunc | sort -u | xargs -r sudo docker image inspect --format "
                           "'CODEXLABEL {{index .Config.Labels \"p006.task-env\"}}={{index .Config.Labels \"p006.codex\"}}' 2>/dev/null; "
                           "sudo docker image ls --filter dangling=false -q --no-trunc | sort -u | xargs -r sudo docker image inspect --format "
                           "'HARNLABEL {{index .Config.Labels \"p006.task-env\"}} {{index .Config.Labels \"p006.grok\"}} {{index .Config.Labels \"p006.opencode\"}}' 2>/dev/null; true", zone)
    have = set()
    for l in out.splitlines():
        f = l.split()
        if len(f) >= 3 and f[0] == "LABEL" and f[1] not in ("", "<no value>"):
            # a by-id image (isolation's Grok/OpenCode images, p006.role=box-image-by-id, 2026-10-26 22:36Z): run only with
            # --image, so it is no box image for its label's gates, label-bound fixes or run_one's own lookup
            if "box-image-by-id" in f[3:]:
                have.add(f"byid:{f[1]}:{f[2].replace('sha256:', '')[:12]}")
                continue
            have.add(f[1])
            have.add(f"imageid:{f[1]}:{f[2].replace('sha256:', '')[:12]}")
            if len(f) >= 4 and f[3] == "fifo":
                have.add(f"fifo:{f[1]}")       # isolation's lifecycle images: the brief through a pipe kept open
    for l in out.splitlines():
        f = l.split()
        if len(f) == 3 and f[0] == "TASKFILE" and f[2].endswith(".json"):
            have.add(f"taskfile:{f[2][:-5]}:{f[1]}")
        if f == ["ARTIFACTSREAD"]:
            have.add("artifacts-read")
        if len(f) == 3 and f[0] == "ARTIFACT" and f[2].startswith("artifacts/"):
            have.add(f"artifact:{f[2]}:{f[1]}")
        if len(f) == 3 and f[0] == "GTCOMMON":
            have.add(f"gtcommon:{f[1]}")
        if len(f) == 3 and f[0] == "CODEXSESSION":
            have.add(f"codexsession:{f[1]}")
        if len(f) == 3 and f[0] == "SKILLFILE" and f[2].endswith("/SKILL.md"):
            have.add(f"skill:{f[2][:-len('/SKILL.md')]}:{f[1]}")
        # an image that carries Codex (isolation's p006.codex label, e.g. 0.163.0): "codex:<task-env label>"
        if len(f) == 2 and f[0] == "CODEXLABEL" and "=" in f[1]:
            lab, cv = f[1].split("=", 1)
            if lab not in ("", "<no") and cv not in ("", "<no value>") and not cv.startswith("<no"):
                have.add(f"codex:{lab}")
        # an image that carries Grok Build or OpenCode (isolation's p006.grok / p006.opencode labels): "<harness>:<label>"
        if len(f) >= 2 and f[0] == "HARNLABEL" and f[1] not in ("", "<no"):
            for h_, v in zip(("grok", "opencode"), f[2:4]):
                if v and not v.startswith("<no"):
                    have.add(f"{h_}:{f[1]}")
    return have

def horizon_of(task_id):
    """The task's hidden horizon_s (recorded seconds), else its deadline_s (earlier design), from the registry."""
    p = os.path.join(EXP, "gpu-tasks", "registry", "tasks", f"{task_id}.json")
    t = json.load(open(p)) if os.path.exists(p) else {}
    h = t.get("horizon_s") or t.get("deadline_s")
    return float(h) if isinstance(h, (int, float)) else None

def image_label(task_id):
    """The p006.task-env label of the box image a task runs in: the task's environment.image_label.value (gpu-tasks),
    else the task id, which is how run_one.py chooses the image."""
    p = os.path.join(EXP, "gpu-tasks", "registry", "tasks", f"{task_id}.json")
    t = json.load(open(p)) if os.path.exists(p) else {}
    v = ((t.get("environment") or {}).get("image_label") or {})
    return (v.get("value") if isinstance(v, dict) else v) or task_id

PICK_WAIT_S = 60.0         # a free machine's wait between picks (was 300 s until 2026-10-23 15:00Z)
LONG_S = 3600.0            # a line expected to run longer than this is long
SOON_RUN_S = 900.0         # a non-Q run in flight counts as "soon" only if expected to end within this
SOON = ("task-file:", "task-artifacts:", "task-ready:", "fix:background-lifecycle", "fifo-image:", "gated-image:", "runone-flags",
        "fix:open-pool-lifecycle", "fix:wakeup-lifecycle", "fix:lifecycle-scheduler-jobs")      # isolation's ship of lifecycle.py 99e6cb1c, then ff9cd5e8
# the last three: isolation's lifecycle ship, image and gate (the study, 2026-10-21: no long line just before the Q
# model lines come back)

def run_ends_soon(dep, rec, est, now, speed=None):
    """Does a run in flight clear "soon": a Q or W0 line always, any other only when its expected end (its start plus its
    expected seconds) is still ahead of now and within SOON_RUN_S (the study, 2026-10-22 01:11Z: c3-7 idled on an A twin
    whose Haiku run takes 25+ min). A run already past its expected end is overrunning, so its end is unknown, not near: it
    is not soon (the study, 2026-10-27 05:33Z: G-kvgate-gemma-L0, 978 s past its 16,456-s estimate, read as soon and
    held every fill line on five idle machines as long-line-held)."""
    if rec.get("status") not in ("claimed", "running"):
        return False
    # a Q line, or a W0 check at speed > 1 (minutes), always clears soon. A speed-1 W0 reference (a controller or measure: 8.6 h for
    # W0-sfz-reference-r2) clears soon only on the same expected-end rule as any line (the study 2026-11-02 22:22Z: with it in flight,
    # c3-21 and c3-6 held every long fill line as long-line-held for 35 min on 4-GPU machines with nothing else to run)
    if dep.startswith("Q-") or (dep.startswith("W0-") and float(speed or 0) > 1.0):
        return True
    try:
        started = datetime.datetime.fromisoformat((rec.get("started_utc") or rec.get("claimed_utc") or "").replace("Z", "+00:00")).timestamp()
    except ValueError:
        started = None
    if not (est and started):
        return False
    left = started + est - now
    return 0 < left <= SOON_RUN_S

def expected_s(line):
    """A line's expected wall seconds: gpu-tasks' expected makespan for its task under its policy (packed for a model
    line, as a proxy for a competent agent), divided by its speed; else the task's horizon over its speed."""
    p = os.path.join(EXP, "gpu-tasks", "registry", "expected", f"{line['task']}.json")
    sp = float(line.get("speed") or 1.0)
    try:
        e = json.load(open(p))
        pol = (e.get("policies") or {}).get(line.get("policy") or "packed") or (e.get("policies") or {}).get("packed") or {}
        m = (pol.get("score_run") or {}).get("makespan_from_agent_start_wall_s")
        if m:
            return float(m) * float(e.get("speed") or 1.0) / sp
    except (OSError, ValueError):
        pass
    h = horizon_of(line["task"])
    return h / sp if h else None

# run_one.py versions that run Grok Build / OpenCode: filled from isolation's ship; empty means no
# machine qualifies yet, so the lines wait
HARN_RUN_ONE = {"grok": {"a76491ef", "792b27a1", "c3855199", "5761a0dd", "f513c811", "88be08e5", "27f9dc27", "cbe91ce0", "89f7c038", "85126860"}, "opencode": {"a76491ef", "792b27a1", "c3855199", "5761a0dd", "f513c811", "88be08e5", "27f9dc27", "cbe91ce0", "89f7c038", "85126860"}}
# run_one.py versions that take --agent-teams (CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 in the box, so Claude Code's Agent tool
# has its name parameter and a skill's named teammates can exist): 792b27a1 = a76491ef + the flag, its refusal for other
# harnesses, one env line and claude_code.agent_teams in run.json (isolation 2026-10-27 16:00Z; shipped per gap when the first
# gpu-team lines are queued)
AGENT_TEAMS_RUN_ONE = {"792b27a1", "c3855199", "5761a0dd", "f513c811", "88be08e5", "27f9dc27", "cbe91ce0", "89f7c038", "85126860"}
# run_one.py versions for Astra in Claude Code: c3855199 = 792b27a1 + edge_models() (the edge's
# allowed list without [1m], which Claude Code strips before sending: on a76491ef/792b27a1 a gpt-6-astra:medium[1m] line is
# refused 404 on every turn), model_parts(), --auto-compact-window and --effort-level (claude-code only) and three run.json fields
ASTRACC_RUN_ONE = {"c3855199", "5761a0dd", "f513c811", "88be08e5", "27f9dc27", "cbe91ce0", "89f7c038", "85126860"}   # 5761a0dd = c3855199 + run.json example_order_sha256
# the one image each label runs Grok Build and OpenCode in (isolation 22:36Z: the frozen 3ec0 image plus the two binaries,
# p006.role=box-image-by-id so run_one's own lookup never sees it): the run line passes it as --image
# Newest first, per label (isolation 2026-10-27 17:31Z, option (a)): a machine runs the first one it holds, so during a rolling
# release ship a shipped machine runs Grok/OpenCode on the new build and an unshipped one on the old, never a mixed cell; the
# run's own run.json records which. 07aa166d / ab2ada11: r2b (89c4) builds on fa92b7561c1f / bac931d95f39, sidecar 587227e76293
HARN_IMAGES = {"mng3-packages": ["sha256:b58b12e2dbf1823b0192fc5b53660a7506df85cf435c85923e9bbef4279a1c41",   # r7.1
                                 "sha256:cc441afadc74f3ab311c84b0783b487b40cdc2bbbe5cfb801acb73a838f6084e",   # r7
                                 "sha256:2a185b80e89321fffbb6227d6db06ff5dfa34e4eee87a72286ba22457ac67ca8",   # r6
                                 "sha256:5619e3b347ff06d99ae7ac42079abffcd3f4fb9c1d83842dacb75d344ebe52c0",   # r4
                                 "sha256:07aa166d8bf28a609d77d619c03c26355e02ab225eac1dc20beafc87f3e91a35",   # r2b
                                 "sha256:067e72d0f7800a14a8f866d86c281b1b6295794b1b63747623278dfa3e204cff"],  # 3ec0
               "mlsbench-kv": ["sha256:0c526e81d28c6a112460f05d9ff9d4375f710006f38e5fb92e92af3fd952a0ab",     # r7.1 (on KV box 5994258c9617; KV harness key 59e906003268)
                               "sha256:0e2f5f3b730af1c46609f58b6191ea475597bbf51dc5ca19a408d8b8c9fe4205",     # r7 (on KV box bc564e9251d1; KV cache key 7d32c5497788, linked to aa8e7253)
                               "sha256:de359232516c20e10eb53458b589739c59e5dcdc455c49680a1a79f7cfbdc560",     # r6 (on KV box 19fd91e11c0c; KV cache key f5b0c45b8f61)
                               "sha256:a1c1dd7d353d538a709e0fa03b026acb5d4a1baaf294e0a64e1be1849362ed47",     # r4
                               "sha256:ab2ada1178e2851c3ae22cf0923a13f3c433e774b8081c996b6c5e8337ff4726",     # r2b
                               "sha256:4040cdc7f1ef492bac0b529a4843c41fbaf5ab796a4ea29baad09d2d5a4c1bcf"]}    # 3ec0
HARN_IMAGE = {k: v[-1] for k, v in HARN_IMAGES.items()}   # the 3ec0 harness images (selftests that name one id)

def harness_image_for(label, images_on_machine):
    """The label's harness image this machine holds by id, newest first; None when it holds none."""
    for i in HARN_IMAGES.get(label) or []:
        if f"byid:{label}:{i.replace('sha256:', '')[:12]}" in (images_on_machine or set()):
            return i
    return None   # a76491ef = a66a5302 + the two harnesses
CODEX_RUN_ONE = {"82d28740c0a25efa", "a96f623676f858f4", "ec0e026dc7f91aca", "16f9cfec", "a9d06a04", "9f7a7e30", "9114e1c9", "a66a5302", "a76491ef", "792b27a1", "c3855199", "5761a0dd", "f513c811", "88be08e5", "27f9dc27", "cbe91ce0", "89f7c038", "85126860"}     # run_one.py versions with the codex harness (isolation, 2026-10-21; ec0e026d the (b') ship, 17:45Z)
CODEX_SESSION_FOR = {"f513c811": {"db728687"}, "88be08e5": {"db728687"}, "27f9dc27": {"db728687"}, "cbe91ce0": {"db728687"}, "89f7c038": {"db728687"}, "85126860": {"db728687"}}   # run_one -> the codex_session.py versions it runs with
# 88be08e5 = f513c811 + the weights-digest teardown (isolation 2026-11-02 17:34Z, fix:capture-weights-digest: tools/weights_digest.py c865bc51
# loaded as a module, run.json code.weights_digest_sha256, and one block after capture_workspace that acts only on a task with
# capture.weights_index); every other line unchanged, so it joins each set f513c811 is in, with the same codex_session.
# cbe91ce0 = 27f9dc27 + the per-environment Hugging Face caches and tree preparation for a task with environment.python_environments
# (pkg-k2-latepole only: stage_env_hf, prepare_tree, run.json hf_caches and prepared; isolation 2026-11-03 01:29 MST); every other task
# stages byte-identically (isolation/tools/stage_ka.py same), so it joins each set 27f9dc27 is in, with the same codex_session
# 85126860 = 89f7c038 + --skill for the codex harness, mounted at ~/.codex/skills/<name> (SKILL_ROOT by harness) and run.json
# skills[*].mounted_at (isolation 2026-11-11 16:35:45Z, frozen after stage_ka same on c3-20 and a Codex skills/list box check;
# the analysis 09:20 MST: the native Codex harness lines); codex_session stays db728687
# 89f7c038 = cbe91ce0 + the scheduler library mount for a --scheduler on run (isolation 2026-11-04 07:36 MST, fake-slurm's self-pipe controller
# ctld.py 9e2a3446, route (a)): a start check that the four scheduler/slurm files exist, a copy to <box dir>.l, a read-only bind at
# /usr/lib/slurm, run.json scheduler_lib with per-file sha256, removal at teardown; 27 added lines in 4 hunks, nothing removed (read by the
# runner in ~/.cache/p006-ship/sched-selfpipe/run_one.cbe91ce0-89f7c038.diff); E2 known answer 40 rounds ok against the control's hang at
# round 4. Runs other than --scheduler on stage byte-identically (stage_ka same), so it joins each set cbe91ce0 is in, with the same codex_session
# 27f9dc27 = 88be08e5 + the project root passed to the capture (isolation 2026-11-02 ~19:26Z, on the runner's 19:22Z finding: the
# index and relative member paths resolve from /home/researcher/project, where the task stages ensemble.json)
CLAUDE_ARG_RUN_ONE = {"be1c73e88e2986b9", "473ea341194e05da", "82d28740c0a25efa", "a96f623676f858f4", "ec0e026dc7f91aca", "16f9cfec", "a9d06a04", "9f7a7e30", "9114e1c9", "a66a5302", "a76491ef", "792b27a1", "c3855199", "5761a0dd", "f513c811", "88be08e5", "27f9dc27", "cbe91ce0", "89f7c038", "85126860"}   # run_one.py versions that pass --claude-arg to claude: be1c73e8 in service, 473ea341 the Codex build
OPEN_POOL_LIFECYCLE = {"99e6cb1c7a47b50f", "ff9cd5e856554f47", "52f13ee5fa68a4d4", "645f6eac546b9729", "8486c765e6a54a09", "91085137933cbf35", "b15d03468d98f5d4", "d264366468b3df78"}   # lifecycle.py versions that never close an open pool on work done
WAKEUP_LIFECYCLE = {"ff9cd5e856554f47", "52f13ee5fa68a4d4", "645f6eac546b9729", "8486c765e6a54a09", "91085137933cbf35", "b15d03468d98f5d4", "d264366468b3df78"}
# d2643664: b15d0346 + the grace also holds while the agent's gpu-harness board has a job running, GPU or CPU (the analysis
# 2026-11-05 22:59 MST, G-fyld-haiku-A3: a CPU prep on a harness GPU slot, no occupancy rank); only adds a hold
# b15d0346: + the work-done grace waits while GPU work is live (no occupancy rank between hello and bye) and holds to a live bounded
# background task's timeout plus 120 s; only adds
# holds to 91085137, so it meets every key 91085137 meets
# 91085137: + a live CronCreate holds the session to its next match, its lateness and 120 s (isolation 2026-10-30 13:35Z, on
# the analysis 07:14 MDT, G-kvreph-sonnet-L3-m2b); every 8486c765 behaviour kept (48 of 49 replayed streams close at the
# same second; the 49th is m2b, held to its cron); probed on Claude Code 2.1.303: a session cron fires headless while stdin is open
CRON_LIFECYCLE = {"91085137933cbf35", "b15d03468d98f5d4", "d264366468b3df78"}
SCHED_LIFECYCLE = {"645f6eac546b9729", "8486c765e6a54a09", "91085137933cbf35", "b15d03468d98f5d4", "d264366468b3df78"}   # 8486c765: + the work-done grace (up to 1,800 s for the agent's next turn while its own background task is alive; isolation 2026-10-27 01:48Z, the study 01:48Z); selftest 42 ok; known answer P006-LCGRACE-KA-20261027T010548Z (~/computewise-runs/_probes/): answered_in_grace waited_s=301.5 grace_s=1800   # lifecycle.py versions that count the box scheduler's live jobs as background work; known answer P006-LC645-KA-20261024T085201Z on c3-15 (~/computewise-runs/_probes/): 2 sbatch sleep-60 jobs, one result at 8.7 s, stdin closed at +67.2 s, 2.2 s after the second job ended, "(after the box scheduler's jobs ended, at most 2 live)"   # 52f13ee5: feed() survives a string message (2026-10-22 19:23Z)
BASH_TIMEOUT_RUN_ONE = {"a9d06a04", "9f7a7e30", "9114e1c9", "a66a5302", "a76491ef", "792b27a1", "c3855199", "5761a0dd", "f513c811", "88be08e5", "27f9dc27", "cbe91ce0", "89f7c038", "85126860"}   # a66a5302 = 9114e1c9 with the staged-tree cache keyed on the mapped artifacts too          # run_one.py versions that take --bash-timeout-ms       # lifecycle.py versions that keep a session open while a ScheduleWakeup is pending
DECLARE_WAIT_S = 600.0     # how long a freed machine waits for a Q line's task-ready declaration

_OPEN_POOL = {}
def open_pool_task(task):
    """True when the task's jobs are an open pool (draw "open": the agent decides how many runs)."""
    if task not in _OPEN_POOL:
        p = os.path.join(EXP, "gpu-tasks", "registry", "tasks", f"{task}.json")
        try:
            _OPEN_POOL[task] = any(j.get("draw") == "open" for j in json.load(open(p)).get("jobs") or [])
        except (OSError, ValueError):
            _OPEN_POOL[task] = False
    return _OPEN_POOL[task]

NIGHT_BAND = os.path.join(HERE, "night-band.json")
MODEL_OVER_PACKED = 1.22   # median wall of 186 completed model runs over their task's packed expectation (2026-10-23 01:10Z)

def night_band(now=None):
    """The night band (the study, 2026-10-23 01:06Z: a possible gcloud lapse at about 06:35Z, the study's 00:59Z login
    may not have restarted the 16-h clock): from from_utc until until_utc, lines expected to run min_hours or more rank above
    shorter ones, the priority list's order kept inside each band, so a machine freeing near the lapse takes work that lasts
    the night. night-band.json, read at every pick, so it is set and lifted with no restart. None when not in force."""
    try:
        b = json.load(open(NIGHT_BAND))
    except (OSError, ValueError):
        return None
    now = time.time() if now is None else now
    ts = lambda k: datetime.datetime.fromisoformat(b[k].replace("Z", "+00:00")).timestamp() if b.get(k) else None
    try:
        lo, hi = ts("from_utc"), ts("until_utc")
    except ValueError:
        return None
    if b.get("lifted") or (lo is not None and now < lo) or (hi is not None and now >= hi):
        return None
    return float(b.get("min_hours") or 4.0) * 3600.0

def night_force():
    """run_id patterns the study puts in the night band's long group whatever their estimate (night-band.json
    `force_long`; the study 2026-10-23 04:39Z: A-p6parL-haiku-L0-bashdefault and -r2, about 10 h each on a task whose
    model runs' median is 2.8 h)."""
    try:
        return list(json.load(open(NIGHT_BAND)).get("force_long") or [])
    except (OSError, ValueError):
        return []

def night_est(line, lines, cl):
    """A line's expected wall seconds for the night band. A scripted line: expected_s. A model line: the median wall of the
    completed model runs on its task where there are two or more, else expected_s x MODEL_OVER_PACKED; never below
    expected_s. goal-choose-optimizer-q's packed 3.1 h is a 2.99 h and an 8.64 h run (G-goalq), pkg-stagger's 2.7 h a 3.2,
    3.7 and 9.8 h (A-stagger)."""
    e = expected_s(line)
    if e is None or line.get("harness") in (None, "scripted"):
        return e
    by = {x["run_id"]: x for x in lines}
    durs = []
    for rid, r in cl.items():
        x = by.get(rid)
        if not x or x["task"] != line["task"] or x.get("harness") in (None, "scripted") or r.get("status") != "done":
            continue
        try:
            a = datetime.datetime.fromisoformat(r["started_utc"].replace("Z", "+00:00")).timestamp()
            z = datetime.datetime.fromisoformat(r["ended_utc"].replace("Z", "+00:00")).timestamp()
        except (KeyError, AttributeError, ValueError):
            continue
        durs.append(z - a)
    if len(durs) >= 2:
        durs.sort()
        m = durs[len(durs) // 2] if len(durs) % 2 else (durs[len(durs) // 2 - 1] + durs[len(durs) // 2]) / 2
        return max(e, m)
    return e * MODEL_OVER_PACKED

def label_bound(req, ok, why, line, imgs):
    """A fix bound to image ids and to the image the line runs in (`label_bound`, the ba36 pair 2026-10-23: the host
    back-dating names the packages and the KV box): met only where the machine's images with the line's label are all
    among the declared ids, so an mls-kv line never takes a machine whose packages box moved and whose KV box did not."""
    d = declared().get(req) or {}
    if not ok or not d.get("label_bound") or not d.get("requires_image_ids"):
        return req, ok, why
    lab = image_label(line["task"])
    want = {x.replace("sha256:", "")[:12] for x in d["requires_image_ids"]}
    have = {x.split(":", 2)[2] for x in imgs if x.startswith(f"imageid:{lab}:")}
    if have and have <= want:
        return req, True, why
    return req, False, f"the machine's {lab} image(s) {sorted(have) or 'none'} are not among {req}'s {sorted(want)}"

def read_only_file(path):
    """The --only list from a file: comma- or newline-separated patterns, blank entries dropped."""
    return [x.strip() for x in open(path).read().replace("\n", ",").split(",") if x.strip()]

def only_match(run_id, pat):
    """An --only pattern: fnmatch on its first part, and on none of its `!`-separated exclusions."""
    import fnmatch
    inc, *exc = pat.split("!")
    return fnmatch.fnmatch(run_id, inc) and not any(fnmatch.fnmatch(run_id, x) for x in exc)

def pick(machine, row, c, lines, cl, imgs, explain=False, only=None, hold_since=None):
    """only: run_id patterns (fnmatch) this worker may take, in priority order: the first pattern's runnable lines
    are taken before the second's, and within a pattern the backlog's order holds (the study's machine
    assignments, e.g. a chain measuring run before a parallel one on the same machine). A pattern may carry exclusions,
    `Q-*!Q-stagger-*-L0-r[34]`: it matches what its first part matches and no `!` part does, so a line can be ranked
    after a broad pattern that would otherwise take it (the study, 2026-10-22 15:15Z: the r3/r4 lines last)."""
    import fnmatch
    waves = set(c.get("waves_open") or [])
    why = []
    rank = lambda l: next((k for k, pat in enumerate(only) if only_match(l["run_id"], pat)), None) if only else 0
    ordered = sorted((l for l in lines if rank(l) is not None), key=lambda l: rank(l))   # stable: backlog order within
    band = night_band()
    long_band = set()
    if band is not None:
        # the night band: lines expected to run band seconds or more first, the list's order kept inside each band
        force = night_force()
        long_band = {l["run_id"] for l in ordered if (night_est(l, lines, cl) or 0.0) >= band
                     or any(only_match(l["run_id"], f) for f in force)}
        ordered = sorted(ordered, key=lambda l: 0 if l["run_id"] in long_band else 1)
    held_soon = []          # the first higher-priority line waiting only on something that clears soon
    held_ready = []         # the first higher-priority Q line waiting only on its declaration or the registry sync
    hold_since = hold_since if hold_since is not None else {}
    for l in ordered:
        if l["wave"] not in waves or l["run_id"] in cl:
            continue
        # every line also needs its task's box image on the machine, whether or not its requires names it
        reqs = list(l.get("requires") or [])
        if f"image:{image_label(l['task'])}" not in reqs:
            reqs.append(f"image:{image_label(l['task'])}")
        if l.get("harness") == "codex":
            # a Codex line: its own box side (the codex harness in run_one.py and the
            # image), declared harness:codex once checked in a box; Claude Code's lifecycle and fifo gates don't apply
            reqs.append(f"model-runs:{l['task']}")
            reqs.append("harness:codex")
            reqs.append(f"codex-image:{image_label(l['task'])}")     # per machine: the image carries Codex
            reqs.append("codex-runone")                               # and its run_one.py has the codex harness
            if l.get("harness_flags"):
                reqs.append("harness:codex-flags")    # -c key=value / --disable <feature>, checked in a box
        elif l.get("harness") in HARN_RUN_ONE:
            # a Grok Build or OpenCode line (the analysis 2026-10-26 20:09Z; experiments/harnesses/): its own box
            # side, as for Codex, declared harness:<name> once its one-turn box check has gone through the scoring chain;
            # Claude Code's lifecycle and fifo gates don't apply (each harness ends a session on its own rule)
            reqs.append(f"model-runs:{l['task']}")
            reqs.append(f"harness:{l['harness']}")
            reqs.append(f"{l['harness']}-image:{image_label(l['task'])}")    # per machine: the image carries the harness
            reqs.append(f"{l['harness']}-runone")                            # and its pinned run_one.py runs it
        elif l.get("harness") not in (None, "scripted"):
            reqs.append(f"model-runs:{l['task']}")
            # claude -p ends the session when the agent's turn ends, and a background task or subagent then dies with
            # it (Q-prepP6-haiku-L0); every model line waits until the session lifecycle is fixed
            reqs.append("fix:background-lifecycle")
            reqs.append(f"fifo-image:{image_label(l['task'])}")
            if l.get("harness_flags"):
                reqs.append("fix:harness-flags")       # run_one.py passes them to claude, checked in a box
                reqs.append("runone-flags")
            # an open pool (goal packages) under Claude Code needs the lifecycle that never closes it on "work done": the
            # older one ended G-passbar-sonnet-L0-v2 with 4 of 16 runs done. Per machine, so an
            # unshipped machine never takes such a line; Codex lines close on codex_session's own rule
            if open_pool_task(l["task"]):
                reqs.append("fix:open-pool-lifecycle")
            # a gpu-team line (solution's option 3, isolation 2026-10-27 16:00Z): "agent_teams": true needs a pinned run_one
            # that takes --agent-teams, per machine, as <h>-runone does for Grok
            if l.get("agent_teams"):
                reqs.append("agent-teams-runone")
            # a claude-code line on a non-Claude model,
            # or one with effort_level / auto_compact_window, needs the run_one whose edge allows the model without [1m]
            if not str(l.get("model") or "").startswith("claude-") or l.get("effort_level") or l.get("auto_compact_window"):
                reqs.append("astracc-runone")
        if l.get("bash_timeouts_ms"):
            reqs.append("fix:bash-timeout-line")     # per machine: the pinned run_one takes --bash-timeout-ms
        if l.get("nudge") and l["nudge"] not in BASE_NUDGE_LEVELS and l.get("harness") != "scripted":
            # a nudge level newer than every pinned campaign.json (P, 2026-10-22 16:46:59Z): run_one refuses a level its
            # campaign.json lacks, so the line waits for the machine's pinned copy to carry it
            reqs.append(f"campaign-level:{l['nudge']}")
        if l.get("rung") in ("S0", "S1", "S2", "S3"):     # every scheduler rung (analysis' fix 1, 2026-10-22 23:53Z)
            # a scheduler rung: the box's sbatch scheduler, started by box-init and
            # its event log copied to <run>/scheduler.jsonl; isolation declares scheduler:box once the image carries it
            reqs.append("scheduler:box")
        if l.get("harness") == "scripted" and plan_imports_gt_common(l["task"]) and any(x.startswith("taskfile:") for x in imgs):
            # plan.py plans an open task's reference plan through gpu-tasks' build/common.py (2026-10-23 00:42Z: c3-6's older
            # common.py had no expand_reference_plan and W0-mlskv-check10c failed before any box): the machine's copy must be
            # the controller's
            reqs.append("gt-common")
        if any(x.startswith("taskfile:") for x in imgs):     # only when the machine's task files were read
            reqs.append(f"task-file:{l['task']}")
        if "artifacts-read" in imgs and task_artifacts(l["task"]):    # only when the machine's artifacts were read
            reqs.append(f"task-artifacts:{l['task']}")
        if any(x.startswith("imageid:") for x in imgs):
            reqs.append(f"gated-image:{image_label(l['task'])}")
        # a screen line (the study, 2026-10-23 15:02Z: `screen: true` on the package-search lines) starts without its task's
        # task-ready declaration; its W0 runs alongside it, and a serving defect the W0 later shows voids the screen's runs
        if l.get("screen"):
            reqs = [r for r in reqs if not r.startswith("task-ready:")]
        res = [(r, *requirement_met(r, row, cl, imgs)) for r in reqs]
        res = [label_bound(r, ok, w, l, imgs) for r, ok, w in res]
        unmet = [(r, w) for r, ok, w in res if not ok]
        if not unmet:
            # a Q line whose measure is done and which waits only on gpu-tasks' task-ready, or on the sync of the task
            # file that declaration rewrote, keeps this machine for up to
            # DECLARE_WAIT_S from when this machine first saw it so, rather than let it start a 55-min measure
            # (the study, 2026-10-21: the key prep/control pairs run measure, Haiku, Sonnet end to end)
            if held_ready and time.time() - hold_since.get(held_ready[0], time.time()) < DECLARE_WAIT_S:
                why.append((l["run_id"], [("waiting-for-declaration", f"{held_ready[0]} waits only on task-ready, held "
                                                                      f"up to {DECLARE_WAIT_S / 60:.0f} min")]))
                continue
            # a higher-priority line held only by what clears soon (a registry sync, a task-ready declaration, a run
            # already in flight) keeps this machine off long lines, so the wave-Q lines never sit behind a 4-h run
            #; a short line may go, or the machine waits for the next poll
            est = expected_s(l)
            if held_soon and est and est > LONG_S and l["run_id"] not in long_band:   # the night band wants the long line
                why.append((l["run_id"], [("long-line-held", f"about {est / 3600:.1f} h while {held_soon[0]} waits only "
                                                               f"on a sync, a declaration or a run in flight")]))
                continue
            return l, why
        why.append((l["run_id"], unmet))
        def run_soon(dep):
            dl = next((x for x in lines if x["run_id"] == dep), None)
            return run_ends_soon(dep, cl.get(dep) or {}, expected_s(dl) if dl else None, time.time(), (dl or {}).get("speed"))
        soon = all(r.startswith(SOON) or (r.startswith("run:") and run_soon(r[4:])) for r, _ in unmet)
        if soon and not held_soon:
            held_soon.append(l["run_id"])
        if l["run_id"].startswith("Q-") and all(r.startswith(("task-ready:", "task-file:")) for r, _ in unmet) and not held_ready:
            held_ready.append(l["run_id"])
            hold_since.setdefault(l["run_id"], time.time())
    return None, why

def claim(line, machine):
    os.makedirs(CLAIMS, exist_ok=True)
    p = os.path.join(CLAIMS, f"{line['run_id']}.json")
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return None
    rec = {"run_id": line["run_id"], "claimed_utc": utc(), "machine": machine, "run_dir": None, "status": "claimed"}
    os.write(fd, (json.dumps(rec, indent=1) + "\n").encode()); os.close(fd)
    return rec

def set_claim(run_id, **f):
    p = os.path.join(CLAIMS, f"{run_id}.json")
    with LOCK:
        rec = json.load(open(p)); rec.update(f)
        tmp = p + ".tmp"; open(tmp, "w").write(json.dumps(rec, indent=1) + "\n"); os.replace(tmp, p)
    return rec

# ---- one run -----------------------------------------------------------------------------------------------------

PINNED = ["isolation/run_one.py", "isolation/tools/cgroup_sampler.py", "runner/run_smoke.py", "runner/score/idle.py",
          "runner/score/prep.py", "runner/score/tells.py", "runner/score/launches.py", "runner/score/score_all.py", "runner/score/served_model.py", "runner/score/score_run.py", "runner/score/overshoot.py",
          "runner/scripted/agent.py", "runner/scripted/plan.py", "campaign/campaign.json",
          "gpu-tasks/build/stage_task.py",          # stages every run's task tree
          "runner/harness/lifecycle.py",            # run_one.py imports it to end a session
          "gpu-tasks/build/prep_fidelity.py",       # score_all runs it for the prep_fidelity label
          "gpu-tasks/build/h4_prep_parallelism.py"]  # and this for H4's prep_default_parallelism label

PINS = os.path.join(HERE, "pins.json")

def machine_hashes(machine, zone):
    rc, out = ssh(machine, "cd " + REMOTE_EXP + " && sha256sum " + " ".join(PINNED) + " 2>&1; true", zone)
    got = {}
    for l in out.splitlines():
        parts = l.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            got[parts[1]] = parts[0]
    return got

KEPT_DIR = os.path.expanduser("~/.cache")

def kept_run_one():
    """isolation's frozen copies of each run_one.py it ships (~/.cache/run_one.<first 8 hex>.py, the file a ship manifest
    names): their full sha256s. A ship installs one of these, not the controller's working file, which can move on before the
    ship's gap comes (2026-10-27 17:39Z: the controller went to c3855199 while 18 machines waited to get 792b27a1 from the kept
    copy, and the re-pin would have refused every one). A copy counts only when its own hash starts with its name."""
    out = set()
    for p in glob.glob(os.path.join(KEPT_DIR, "run_one.*.py")):
        try:
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        except OSError:
            continue
        if h.startswith(os.path.basename(p).split(".")[1]):
            out.add(h)
    return out

def repin(machine, why):
    """Re-pin a machine from its own files, only when every pinned file there is either the controller's copy (the reviewed
    tree, which a ship puts on) or the version the machine was already pinned to (a file the ship left alone; 2026-10-22
    18:30Z: c3-8's score_run.py was an older accepted version the gap ship never touched). Anything else is left to the
    tree check, which refuses the next start until someone deploys and pins by hand."""
    zone = (fleet().get(machine) or {}).get("zone")
    got = machine_hashes(machine, zone)
    lap = {f: hashlib.sha256(open(os.path.join(EXP, f), "rb").read()).hexdigest() for f in PINNED if os.path.exists(os.path.join(EXP, f))}
    was = ((json.load(open(PINS)).get("by_machine") or {}).get(machine) or {}).get("sha256") or {} if os.path.exists(PINS) else {}
    kept = kept_run_one()
    off = [f for f in PINNED if (got.get(f) not in (lap.get(f), was.get(f)) and not (f == "isolation/run_one.py" and got.get(f) in kept))
           or got.get(f) is None]
    if off:
        log("repin-refused", machine=machine, why="pinned files differ from the controller's: " + ", ".join(off))
        return False
    with LOCK:
        pins = json.load(open(PINS)) if os.path.exists(PINS) else {}
        bm = pins.setdefault("by_machine", {})
        old = (bm.get(machine) or {}).get("sha256") or {}
        bm[machine] = {"utc": utc(), "why": why, "sha256": got}
        tmp = PINS + ".tmp"; json.dump(pins, open(tmp, "w"), indent=1); os.replace(tmp, PINS)
    log("pinned", machine=machine, why=why, changed=",".join(f for f in PINNED if old.get(f) != got[f]))
    return True

def tree_differs(machine, zone):
    """The pinned files on the machine against pins.json, the versions accepted for campaign runs (driver.py pin).
    Without pins.json, against this controller's working tree. Pins, not the controller, are the reference, because peers edit
    their files on the controller before they deploy: a series (the five C1 repeats) must keep one code version."""
    if os.path.exists(PINS):
        pins = json.load(open(PINS))
        want = ((pins.get("by_machine") or {}).get(machine) or {}).get("sha256") or pins["sha256"]
        got = machine_hashes(machine, zone)
        return [f for f in PINNED if got.get(f) != want.get(f)]
    want = {f: hashlib.sha256(open(os.path.join(EXP, f), "rb").read()).hexdigest() for f in PINNED}
    rc, out = ssh(machine, "cd " + REMOTE_EXP + " && sha256sum " + " ".join(PINNED) + " 2>&1; true", zone)
    got = {}
    for l in out.splitlines():
        parts = l.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            got[parts[1]] = parts[0]
    return [f for f in PINNED if got.get(f) != want[f]]

def host_terms_for(task, machine_type):
    """packed_overlap needs host terms measured on this machine type (the C1 note: E2 terms don't carry to c3)."""
    for name in (f"{task}.{machine_type}.json",):
        p = os.path.join(EXP, "gpu-tasks", "survey", "host-terms", name)
        if os.path.exists(p):
            return f"{REMOTE_EXP}/gpu-tasks/survey/host-terms/{name}"
    return None

HARNESS_ARG = {"claude-code": "--claude-arg", "codex": "--codex-arg"}   # run_one.py's pass-through flag per harness

WAIT_STYLE = os.path.join(EXP, "codex", "wait_style.py")
WAIT_TABLE = os.path.join(RUNNER, "reports", "wait-style.md")
POLLING = os.path.join(EXP, "codex", "polling_table.py")      # claim 1b's table on the five packages
POLLING_TABLE = os.path.join(RUNNER, "reports", "polling.md")

TABLES_DUE = os.path.join(HERE, "tables-due")      # touched after every collect; launchd/tables.sh rebuilds when it moves

def wait_table():
    """After a collect: mark the two tables due, and return at once. The rebuild is off the claim path (the study,
    2026-10-28 22:28Z: five loops each rebuilt both over all 538 controller copies at once, every one hit the 600-s timeout, wrote
    nothing, and held its machine 20 min before claiming). launchd/tables.sh, under launchd as com.p006.tables, rebuilds
    both whenever this file is newer than its last build, one build at a time."""
    try:
        with open(TABLES_DUE, "a"):
            os.utime(TABLES_DUE, None)
    except OSError as e:
        log("wait-table-failed", table="tables-due", detail=repr(e)[:300])

def build_tables():
    """Regenerate the "how the agent waited" table over every controller copy (the Codex driver's wait_style.py, at
    the analysis' request, 2026-10-21), and the polling table beside it (2026-10-26 16:15Z). Written to a temporary
    file and moved into place; a failure is logged, never raised. Run by launchd/tables.sh, never by a loop."""
    for script, table, args in ((WAIT_STYLE, WAIT_TABLE, ["--runs", LAPTOP_RUNS]), (POLLING, POLLING_TABLE, [])):
        tmp = f"{table}.{os.getpid()}.tmp"
        try:
            p = subprocess.run([sys.executable, "-I", script] + args + ["--out", tmp],
                               capture_output=True, text=True, timeout=4 * 3600, cwd=tempfile.gettempdir())   # one build over every controller copy took > 1 h at load 15 (2026-10-28 23:28Z)
            if p.returncode == 0 and os.path.exists(tmp):
                os.replace(tmp, table)
            else:
                log("wait-table-failed", table=os.path.basename(table), detail=(p.stderr or p.stdout)[-300:])
        except Exception as e:
            log("wait-table-failed", table=os.path.basename(table), detail=repr(e)[:300])
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

def token_watch(tag, line, log_path, env=None):
    """A run's token ceiling: ledger.py watch-tokens drops the
    gate's stop file for the tag once either sum crosses its cap, like the dollar ceiling's watcher."""
    caps = line.get("token_caps") or {}
    if not caps:
        return None
    stop = os.path.join(os.path.expanduser(os.environ.get("P006_GATE_HOME", "~/.p006-gate")), "stop", tag)
    return subprocess.Popen([sys.executable, "-I", os.path.join(RUNNER, "score", "ledger.py"), "watch-tokens", tag, stop,
                             "--input", str(caps.get("input", 4e6)), "--output", str(caps.get("output", 2e5)),
                             "--json", os.path.join(HERE, f"tokens-{line['run_id']}.json")],
                            stdout=open(log_path, "a"), stderr=subprocess.STDOUT, env=env)

NO_SCORE_RUN_SMOKE = {"f3e5702b"}      # run_smoke.py versions that take --no-score

def machine_pinned(machine, path):
    pins = json.load(open(PINS)) if os.path.exists(PINS) else {}
    return (((pins.get("by_machine") or {}).get(machine or "") or {}).get("sha256") or {}).get(path, "")

def smoke_cmd(line, c, tag, machine_type, machine_name=None, imgs=None):
    box = c["box"]
    horizon = horizon_of(line["task"])
    if horizon is None:
        return None, f"{line['task']} has no horizon_s or deadline_s in the registry"
    max_h = horizon / float(line.get("speed", 1.0)) / 3600.0 + 0.5
    cmd = ["/usr/bin/python3", "-I", "run_smoke.py", "--task", line["task"], "--speed", str(line.get("speed", 1.0)),
           "--max-hours", f"{max_h:.2f}", "--tag", tag]
    if line["harness"] == "scripted":
        cmd += ["--harness", "scripted", "--policy", line["policy"]]
        if line["policy"] == "packed_overlap":
            ht = host_terms_for(line["task"], machine_type)
            if not ht:
                return None, f"no host terms for {line['task']} on {machine_type} (gpu-tasks/survey/host-terms/{line['task']}.{machine_type}.json)"
            cmd += ["--host-terms", ht]
    else:
        if line["harness"] in ("codex", "grok", "opencode"):
            cmd += ["--harness", line["harness"]]
        cmd += ["--model", line["model"], "--budget-usd", str(line["ceiling_usd"]), "--ceiling-confirmed"]
        # --no-score only where the machine's pinned run_smoke.py takes it (f3e5702b and later): an older one exits 2 on
        # the flag before any box
        if machine_pinned(machine_name, "runner/run_smoke.py")[:8] in NO_SCORE_RUN_SMOKE:
            cmd += ["--no-score"]
        if line.get("nudge"):
            cmd += ["--nudge", line["nudge"], "--campaign", f"{REMOTE_EXP}/campaign/campaign.json"]
    # the line's Bash timeouts (the study, 2026-10-22: the product default, 120 s / 600 s, beside run_one's 10 h),
    # passed after "--" to run_one; the line also requires fix:bash-timeout-line, met where the pinned run_one takes it
    bt = line.get("bash_timeouts_ms")
    extra_run_one = [f"--bash-timeout-ms={int(bt['default'])},{int(bt['max'])}"] if bt else []
    if bt and line["harness"] != "claude-code":
        return None, "bash_timeouts_ms applies to claude-code lines only"
    # harness_flags (the study, 2026-10-21: the no-subagent arm, ["--disallowedTools", "Task"]): passed to claude
    # unchanged, one --claude-arg=<token> each to run_one.py (isolation), so no token is split by the shell command
    flags = list(line.get("harness_flags") or [])
    if flags and line["harness"] == "scripted":
        return None, "harness_flags apply to model lines only"
    bad = [f for f in flags if not re.fullmatch(r"[A-Za-z0-9_.,:=/@+-]+", str(f))]
    if bad:
        return None, f"harness_flags must be single shell-safe tokens: {bad}"
    # The box's caps come from campaign.json's box block (the study's). "machine", or no value, leaves them to
    # run_one.py, whose default is the whole machine less the sidecar's cores and 8 GB (the study, 2026-10-20: the
    # machine is the isolation boundary, one box per machine). run.json records the caps applied and caps.source.
    caps = []
    # a task may declare its own box disk (cpu-tasks, 2026-10-23 15:50Z: H4's filter ablation keeps every recipe's tokenised
    # data, up to about 35 GB, so its task file says box.disk 80g): the task's value wins over campaign.json's for that key
    tbox = {}
    try:
        tbox = (json.load(open(os.path.join(EXP, "gpu-tasks", "registry", "tasks", f"{line['task']}.json"))).get("box") or {})
    except (OSError, ValueError):
        pass
    for flag, key in (("--cpus", "cpus"), ("--memory", "memory"), ("--disk", "disk")):
        v = tbox.get(key) if key == "disk" and tbox.get(key) else box.get(key)
        if v not in (None, "", "machine"):
            caps += [flag, str(v)]
    # the scheduler rungs: S0 and up start the box's scheduler; S1 and
    # up replace the brief with the task's brief_variants[<brief_variant>] then a blank line then the line's brief_append
    # (Text S1), written on the machine by run_line before the start (the brief file path is fixed by the tag)
    if line.get("rung") in ("S0", "S1", "S2", "S3"):
        if line["harness"] == "scripted":
            return None, "a scheduler rung applies to model lines only"
        extra_run_one += ["--scheduler", "on"]
    if line.get("brief_variant") or line.get("brief_append"):
        text, err = brief_text(line)
        if err:
            return None, err
        cmd += ["--brief-file", brief_path(tag)]
    # a skill (rung P, later S2/S3): mounted by run_one from the machine's solution/skills/<name>, on claude-code and codex lines
    # (the analysis 09:20 MST 2026-11-11, the native Codex harness lines: isolation's run_one mounts a codex line's skill at
    # ~/.codex/skills/<name>); the other harnesses still refuse one
    skills = line.get("skills") or ([line["skill"]] if line.get("skill") else [])
    if skills and line["harness"] not in ("claude-code", "codex"):
        return None, "skills apply to claude-code and codex lines only"
    extra_run_one += [f"--skill={REMOTE_EXP}/solution/skills/{s}" for s in skills]
    # effort_level / auto_compact_window: --effort-level <v> / --auto-compact-window <n>
    # after "--", claude-code only, as run_one c3855199 takes them
    for fld, flag in (("effort_level", "--effort-level"), ("auto_compact_window", "--auto-compact-window")):
        if line.get(fld) is not None:
            if line["harness"] != "claude-code":
                return None, f"{fld} applies to claude-code lines only"
            if fld == "effort_level" and line[fld] not in ("low", "medium", "high", "xhigh"):
                return None, f"effort_level must be low, medium, high or xhigh: {line[fld]!r}"
            if fld == "auto_compact_window" and not (isinstance(line[fld], int) and line[fld] > 0):
                return None, f"auto_compact_window must be a positive number of tokens: {line[fld]!r}"
            extra_run_one += [flag, str(line[fld])]
    # "agent_teams": true (a gpu-team line): --agent-teams after "--", claude-code only (run_one refuses it for any other harness)
    if line.get("agent_teams"):
        if line["harness"] != "claude-code":
            return None, "agent_teams applies to claude-code lines only"
        extra_run_one += ["--agent-teams"]
    # a Grok Build or OpenCode line runs in its label's harness image, by id
    if line["harness"] in HARN_RUN_ONE:
        # the machine's own, newest first (imgs given: a real start); without imgs (a dry run, a selftest) the 3ec0 one
        img = harness_image_for(image_label(line["task"]), imgs) if imgs is not None else HARN_IMAGE.get(image_label(line["task"]))
        if not img:
            return None, f"no harness image for {image_label(line['task'])} on this machine (HARN_IMAGES)"
        extra_run_one += ["--image", img]
    # a Codex line's flags (-c key=value, --disable <feature>) go to codex the same way (HARNESS_ARG; isolation's run_one)
    cmd += ["--", *caps, "--shm", "8g", "--pids", "8192", "--oom-protect", "on",
            "--after-deadline", box.get("after_deadline", "stop"), *extra_run_one,
            *[f"{HARNESS_ARG.get(line['harness'], '--claude-arg')}={f}" for f in flags]]
    return cmd, None

BASE_NUDGE_LEVELS = {"L0", "L1", "L2", "L3", "M"}       # in every campaign.json pinned since 2026-10-21 15:57Z
LEVELS_SEEN = os.path.join(HERE, "campaign-levels.json")

def campaign_levels():
    """campaign.json sha256 -> its nudge levels, for every controller copy the driver has read (recorded as it goes, since a
    machine's pin names only a hash). The current controller copy is added on each call."""
    seen = json.load(open(LEVELS_SEEN)) if os.path.exists(LEVELS_SEEN) else {}
    p = os.path.join(EXP, "campaign", "campaign.json")
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if h not in seen:
        seen[h] = sorted(((json.load(open(p)).get("nudges") or {}).get("levels") or {}).keys())
        tmp = LEVELS_SEEN + ".tmp"; json.dump(seen, open(tmp, "w"), indent=1); os.replace(tmp, LEVELS_SEEN)
    return seen

# a task whose host terms come from a W0 measure still to land: for mls-kv-deploy, measure-c (W0-mlskv-measure never ran,
# measure-b is labelled inflated by the host double count, the study 2026-10-22 23:53Z; measure-c runs on the ba36 KV box
# with the back-dating, started 2026-10-23 01:53Z)
MEASURE_OF = {"mls-kv-deploy": "W0-mlskv-measure-c"}

def host_terms_status(line, cl=None):
    """Which host terms a model run was scored before: "provisional (check10)" when the
    task's W0 measure is not done at claim time, "final (measure)" when it is; None for tasks outside MEASURE_OF and
    scripted lines. A label on the claim; analysis reads the terms themselves from the measure."""
    m = MEASURE_OF.get(line.get("task"))
    if not m or line.get("harness") == "scripted":
        return None
    st = ((cl if cl is not None else claims()).get(m) or {}).get("status")
    return "final (measure)" if st == "done" else "provisional (check10)"

def regression_alert(run_id):
    """A score that flags a regression (score_all's regression_alert: a seed recycle on a machine whose fix was declared,
    2026-10-23 00:43Z): logged loudly in the driver log, for the study's tick to see."""
    p = os.path.join(LAPTOP_RUNS, run_id, "score.json")
    try:
        ra = json.load(open(p)).get("regression_alert")
    except (OSError, ValueError):
        return
    if ra:
        log("REGRESSION-ALERT", run_id=run_id, alert=ra, why="the scorer voided this run: a defect declared fixed on its machine recurred")
    # gpu-mock's files showing to the agent (the study, 2026-10-23 03:46Z: any mock-path traceback the agent sees is an
    # alert, learned at collect and not from analysis afterwards)
    try:
        mp = json.load(open(p)).get("mock_path_alert")
    except (OSError, ValueError):
        mp = None
    if mp:
        log("MOCK-PATH-ALERT", run_id=run_id, alert=mp, why="a traceback through gpu-mock's files or its structseq-tuple error reached the agent")
    # any tell on a screen run (the study, 2026-10-23 16:32Z: read saw_tell on each mls-kv-report and mls-kv-gate run as it
    # is collected, and say so at once): the classes, logged for the study's and analysis' tick
    try:
        hd = json.load(open(p)).get("headline") or {}
    except (OSError, ValueError):
        hd = {}
    # the 0 MiB reading on a busy GPU and any kill because of it: flagged at collect too
    if hd.get("saw_zero_mib"):
        log("ZERO-MIB-ALERT", run_id=run_id, task=hd.get("task"), killed_because=hd.get("killed_because_of_zero_mib"),
            killed_after=hd.get("killed_after_zero_mib"), why="nvidia-smi showed 0 MiB used on a busy GPU (records without a memory figure)")
    if hd.get("saw_tell") or hd.get("probed_runtime") or hd.get("inspected_box_tooling"):
        log("TELL-ALERT", run_id=run_id, task=hd.get("task"), saw_tell=hd.get("saw_tell"),
            traceback_through_patch=hd.get("traceback_through_patch"), probed_runtime=hd.get("probed_runtime"),
            inspected_box_tooling=hd.get("inspected_box_tooling"), first_tell=str(hd.get("first_tell"))[:200])

_ART = {}
def task_artifacts(task_id):
    """[(registry-relative path, controller sha256)] for every artifact the task's stage reads: files[] `from: artifacts/...`
    and environment.install.constraints, as stage_task.py reads them, and the task's scripted_reference `from`, which
    plan.py's controller policy reads from the machine's registry (2026-11-03 13:18Z: W0-k2lp-reference was refused on
    c3-8, artifacts/k2-latepole/reference.py missing there, with task-artifacts met; 7 tasks' references sit outside files[])."""
    p = os.path.join(EXP, "gpu-tasks", "registry", "tasks", f"{task_id}.json")
    try:
        key = (task_id, os.path.getmtime(p))
    except OSError:
        return []
    if key not in _ART:
        t = json.load(open(p))
        paths = [f["from"] for f in t.get("files") or [] if str(f.get("from", "")).startswith("artifacts/")]
        c = ((t.get("environment") or {}).get("install") or {}).get("constraints")
        if c:
            paths.append(c)
        sr = t.get("scripted_reference")
        if isinstance(sr, dict) and str(sr.get("from", "")).startswith("artifacts/"):
            paths.append(sr["from"])
        out = []
        for x in sorted(set(paths)):
            fp = os.path.join(EXP, "gpu-tasks", "registry", x)
            out.append((x, hashlib.sha256(open(fp, "rb").read()).hexdigest() if os.path.exists(fp) else "absent-on-controller"))
        _ART[key] = out
    return _ART[key]

def plan_imports_gt_common(task_id):
    """plan.py imports gpu-tasks' build/common.py for an open task with a reference plan (a goal package's scripted
    reference): mirrors plan.plan's own condition."""
    p = os.path.join(EXP, "gpu-tasks", "registry", "tasks", f"{task_id}.json")
    try:
        t = json.load(open(p))
    except (OSError, ValueError):
        return False
    return bool(t.get("reference_plan")) and any(j.get("draw") == "open" for j in t.get("jobs") or [])

def brief_path(tag):
    return f"/srv/work/runner/{tag}.brief.txt"

def brief_text(line):
    """An S line's brief (SOLUTION-LADDER.md, 'Brief variant for S1-S3'): the task's brief_variants[<brief_variant>].text
    (else its brief.text), then a blank line, then the line's brief_append, verbatim. From the controller's registry, which the
    driver has checked equals the machine's task file."""
    p = os.path.join(EXP, "gpu-tasks", "registry", "tasks", f"{line['task']}.json")
    t = json.load(open(p))
    if line.get("brief_variant"):
        v = (t.get("brief_variants") or {}).get(line["brief_variant"])
        if not v or not v.get("text"):
            return None, f"{line['task']} has no brief_variants.{line['brief_variant']}"
        base = v["text"]
    else:
        base = t["brief"]["text"]
    return base.rstrip("\n") + ("\n\n" + line["brief_append"].strip("\n") if line.get("brief_append") else ""), None

IDLE_ALERT_S = 1800.0      # a goal run's GPUs all idle this long with its session up: logged for the study's tick

IDLE_PROBE = r"""
import json, os, sys, time
d = sys.argv[1]; last = {}
# the whole file, line by line (the study, 2026-10-26 18:48Z: a 2 MB tail missed every busy event on long Opus runs,
# whose enqueue lines filled it, and the alert counted idle from agent_start); only busy lines are parsed
try:
    with open(os.path.join(d, "occupancy.jsonl"), "rb") as fh:
        for raw in fh:
            if b'"busy_' not in raw:
                continue
            line = raw.decode("utf-8", "replace")
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("kind") in ("busy_start", "busy_end"):
                last[str(e.get("gpu"))] = [e["kind"], e["t_wall"]]
except OSError:
    pass
try:
    a0 = float(open(os.path.join(d, "box", "agent_start")).read().strip())
except (OSError, ValueError):
    a0 = None
print(json.dumps({"last": last, "agent_start": a0, "now": time.time(),
                  "harness_ended": os.path.exists(os.path.join(d, "box", "harness_exit"))}))
"""
import base64 as _b64
IDLE_PROBE_B64 = _b64.b64encode(IDLE_PROBE.encode()).decode()

def goal_idle(last_by_gpu, agent_start, now):
    """(GPUs busy, seconds every GPU has been idle) from each GPU's last busy event, kind and time. Never busy yet counts
    from the agent's start, not from 0 (the study, 2026-10-21 20:40Z: G-passbar-sonnet-L0-v3 alerted with idle_s equal
    to the epoch before its first launch); no start known gives None, so no alert."""
    busy = sum(1 for k, _ in last_by_gpu.values() if k == "busy_start")
    if busy:
        return busy, None
    ends = [t for k, t in last_by_gpu.values() if k == "busy_end"]
    since = max(ends + ([agent_start] if agent_start else [])) if (ends or agent_start) else None
    return 0, (None if since is None else max(0.0, now - since))

# the running-unit poll while the controller is short of memory (the study 2026-11-04 10:46 MST: swap full, AnyConnect's socket
# extension leaking per DNS lookup, and each poll a fresh gcloud ssh). REVERT TO 60.0 once the controller's swap has room again, after
# the study's AnyConnect restart; the study says when. Costs a collect about 45 s on average (90 s at worst)
UNIT_POLL_BASE_S = 150.0

def unit_poll_s():
    """The running-unit poll interval: UNIT_POLL_BASE_S (60 s normally), or 150 s while the controller's 1-min load is over 40
   ."""
    try:
        return max(UNIT_POLL_BASE_S, 150.0) if os.getloadavg()[0] > 40 else UNIT_POLL_BASE_S
    except OSError:
        return UNIT_POLL_BASE_S

def wait_unit(machine, zone, unit, limit_s, run_id=None, run_dir=None):
    """Wait for the run's unit to end. For a goal line (G-*), also watch the box: once every GPU has been idle for
    IDLE_ALERT_S while the harness is still up, log gpus-idle-session-up (once per idle stretch). With the open-pool
    lifecycle fix a session whose agent left a never-ending background task after its answer is held to the horizon;
    the study decides whether to stop it (.stop with a reason), no rule here guesses."""
    t0, alerted, last_check = time.time(), None, 0.0
    ssh_failures = {}
    goal = bool(run_id and run_id.startswith("G-") and run_dir)
    while time.time() - t0 < limit_s:
        probe = f"systemctl is-active {unit} 2>/dev/null; true"
        if goal and time.time() - last_check >= 300:
            last_check = time.time()
            # each GPU's last busy event, the agent's start, the machine's clock and whether the harness has ended, read on
            # the machine by a short script sent as base64 (no shell quoting inside it)
            probe += "; echo @@; echo " + IDLE_PROBE_B64 + " | base64 -d | sudo python3 - " + run_dir
        rc, out = ssh(machine, probe, zone, timeout=120)
        state = (out.strip().split("@@")[0].strip().splitlines() or ["?"])[-1]
        # only systemd's own answer ends the wait: a failed ssh (gcloud's reauth lapse, 2026-10-22 05:22Z: every loop read
        # the error text as "not active" and went on to collect runs still going) says nothing about the unit
        if state not in ("active", "activating", "reloading", "inactive", "failed", "deactivating"):
            if not ssh_failures.get("since"):
                ssh_failures["since"] = time.time()
                log("ssh-unreachable", machine=machine, unit=unit, detail=out.strip()[-200:])
            time.sleep(60)
            continue
        if ssh_failures.get("since"):
            log("ssh-back", machine=machine, unit=unit, down_s=round(time.time() - ssh_failures.pop("since")))
        if state not in ("active", "activating", "reloading"):
            return state
        if goal and "@@" in out:
            try:
                pr = json.loads(next(l for l in out.split("@@", 1)[1].splitlines() if l.strip().startswith("{")))
            except (StopIteration, ValueError):
                pr = None
            if pr:
                busy, idle_s = goal_idle(pr.get("last") or {}, pr.get("agent_start"), pr.get("now") or time.time())
                if busy == 0 and not pr.get("harness_ended") and idle_s is not None and idle_s >= IDLE_ALERT_S:
                    if alerted is None:
                        alerted = time.time()
                        log("gpus-idle-session-up", run_id=run_id, machine=machine, idle_s=round(idle_s),
                            why=f"every GPU idle {idle_s:.0f} s with the harness still up: the study decides whether to stop it")
                elif busy:
                    alerted = None
        time.sleep(unit_poll_s())
    return "driver-timeout"

def run_stop_record(tag, run_id):
    """What our own ceilings left for one run on the controller: the gate's stop file (budget_watch.py writes its spend into
    it as JSON; ledger.py watch-tokens and watch leave it empty), the token watcher's record (tokens-<run_id>.json),
    and budget_watch's stop event (budget.jsonl). Read only; score_all decides stopped_by from it."""
    g = os.path.expanduser(os.environ.get("P006_GATE_HOME", "~/.p006-gate"))
    sf = os.path.join(g, "stop", tag)
    out = {"tag": tag, "stop_file": sf, "stop_file_exists": os.path.exists(sf), "stop_file_mtime": None,
           "stop_file_body": None, "token_watch": None, "budget_events": []}
    if out["stop_file_exists"]:
        out["stop_file_mtime"] = os.path.getmtime(sf)
        try:
            out["stop_file_body"] = json.load(open(sf))
        except (ValueError, OSError):
            pass
    tw = os.path.join(HERE, f"tokens-{run_id}.json")
    if os.path.exists(tw):
        try:
            out["token_watch"] = json.load(open(tw))
        except (ValueError, OSError):
            pass
    try:
        bl = (json.load(open(os.path.join(g, "budget.json"))).get("log_path")) or os.path.join(g, "budget.jsonl")
        for l in (open(bl) if os.path.exists(bl) else []):
            if f'"{tag}"' in l:
                out["budget_events"].append(json.loads(l))
    except (ValueError, OSError):
        pass
    # the gate's own first refusal of this tag as stopped, which no watcher can overwrite (the Codex driver, 2026-10-22
    # 16:51Z: a watcher restarted at adoption with an older cap rewrote the stop file and tokens-*.json after the real
    # stop, and G-passbar-astra-L0-v2 read as stopped by nothing). The log is append-only and the gate's
    out["gate_first_stopped"] = None
    gl = os.path.join(g, "gate.jsonl")
    try:
        for l in (open(gl, errors="replace") if os.path.exists(gl) else []):
            if '"stopped"' in l and f'"{tag}"' in l:
                e = json.loads(l)
                if e.get("run") == tag and e.get("refused") == "stopped":
                    out["gate_first_stopped"] = {"ts": e.get("ts"), "ts_epoch": e.get("ts_epoch"), "status": e.get("status"),
                                                 "path": e.get("path")}
                    break
    except (ValueError, OSError):
        pass
    return out

def collect(run_id, rec=None, line=None):
    """G2, the re-score (G4, G8) and G3, for a run that has ended. Idempotent; never starts anything."""
    rec = rec or json.load(open(os.path.join(CLAIMS, f"{run_id}.json")))
    machine, tag, rd = rec["machine"], rec["tag"], rec["run_dir"]
    row = fleet().get(machine) or {}
    zone = row.get("zone")
    local = os.path.join(LAPTOP_RUNS, run_id)
    os.makedirs(local, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"p006-collect-{run_id}-")
    # G2: the ledger turns, into the run directory on the machine
    led = os.path.join(work, "ledger.jsonl")
    p = subprocess.run([sys.executable, "-I", os.path.join(RUNNER, "score", "ledger.py"), "dump", tag, led],
                       capture_output=True, text=True, timeout=300)
    if p.returncode != 0 or not os.path.exists(led):
        log("ledger-dump-failed", run_id=run_id, detail=p.stderr[-500:])
        return "failed", ["ledger dump failed"]
    upload(machine, led, f"/tmp/{tag}.ledger.jsonl", zone)
    ssh(machine, f"sudo install -m 644 /tmp/{tag}.ledger.jsonl {rd}/ledger.jsonl && rm -f /tmp/{tag}.ledger.jsonl", zone)
    # the model route's round trip, before (from the claim) and after the run, into the run directory: a label that
    # score_all carries into score.json, so a per-machine latency offset can be seen
    if rec.get("gate_rtt_before") or (line or {}).get("harness") not in (None, "scripted"):
        rt = {"machine": machine, "zone": zone, "before": rec.get("gate_rtt_before"), "after": gate_rtt(machine, zone)}
        open(os.path.join(work, "gate_rtt.json"), "w").write(json.dumps(rt, indent=1) + "\n")
        upload(machine, os.path.join(work, "gate_rtt.json"), f"/tmp/{tag}.gate_rtt.json", zone)
        ssh(machine, f"sudo install -m 644 /tmp/{tag}.gate_rtt.json {rd}/gate_rtt.json && rm -f /tmp/{tag}.gate_rtt.json", zone)
    # our own ceilings: what the controller's watchers left for this tag, into the run
    # directory, so score_all can say stopped_by token_cap | budget and a cap stop never reads as the agent ending
    rs = run_stop_record(tag, (line or {}).get("run_id") or run_id)
    # host work on the same machine during the run, as the claim records it (isolation's cache fill, 2026-10-22 21:57Z)
    chl = (claims().get((line or {}).get("run_id") or run_id) or {}).get("concurrent_host_load")
    if chl:
        rs["concurrent_host_load"] = chl
    # the backlog line's own mark of a testbed defect (the study's `edited` note naming one): void at scoring
    td = [e for e in ((line or {}).get("edited") or []) if "testbed defect" in str(e)]
    if td:
        rs["testbed_defect"] = td[-1]
    # whether gpu-mock's seeded-recycle fix was declared on this machine when the run started (the study, 2026-10-23
    # 00:43Z: from then on a seed_recycled_from_other_job label means a regression): the scorer voids and alerts on it
    rec_ = claims().get((line or {}).get("run_id") or run_id) or {}
    rs["seeded_recycle_fixed"] = bool(rec_.get("seeded_recycle_fixed"))
    # a run stopped by the KV miss watcher (kv_miss_watch.py, the study 2026-10-23 04:28Z): a task-script launch missed its
    # record on literal-argument keys, so the testbed paced it on the CPU; void at scoring like a line's testbed-defect note
    if rec_.get("testbed_stop") and not rs.get("testbed_defect"):
        rs["testbed_defect"] = rec_["testbed_stop"]
    # labels set on the claim for analysis (the study, 2026-10-23 03:30Z: brief_version on every G-mlskv run, so a v2
    # run is never pooled with v1 ones), carried into run_stop.json's analysis_labels, which score_all puts in score.json
    if rec_.get("analysis_labels"):
        rs["analysis_labels"] = dict(rs.get("analysis_labels") or {}, **rec_["analysis_labels"])
    open(os.path.join(work, "run_stop.json"), "w").write(json.dumps(rs, indent=1) + "\n")
    upload(machine, os.path.join(work, "run_stop.json"), f"/tmp/{tag}.run_stop.json", zone)
    ssh(machine, f"sudo install -m 644 /tmp/{tag}.run_stop.json {rd}/run_stop.json && rm -f /tmp/{tag}.run_stop.json", zone)
    # no machine score: the controller re-scores after the copy (below) with its current scorer, so the machine is not held
    # scoring a run it has finished. score.collected.* is then absent unless run_smoke scored
    # (scripted lines still do, to keep check.py's plan check beside them)
    # G3: the whole run directory, and a manifest made on the machine to check the copy against. A task's data shards the
    # copy leaves on the machine (COLLECT_EXCLUDE, the study 2026-10-24 21:12Z: H4's per-recipe filtered/ shards, 21 GB
    # per run, timed out the copy of W0-h4-measure twice) come to the controller as a listing only: name, size and sha256 of
    # each, in workspace.excluded.lst; the manifest covers every other file
    man = f"/tmp/{tag}.manifest"
    task_id = (line or {}).get("task") or rec.get("task") or next((l.get("task") for l in load_campaign()[1] if l.get("run_id") == run_id), None)
    exc = COLLECT_EXCLUDE.get(task_id, "")
    if exc:
        ssh(machine, f"cd {rd} && sudo find . -type f ! -path '{exc}' -print0 | sort -z | sudo xargs -0 sha256sum > {man}; "
                     f"sudo find . -type f -path '{exc}' -print0 | sort -z | sudo xargs -0 -r sha256sum | "
                     f"sudo tee workspace.excluded.lst >/dev/null; "
                     f"sudo sha256sum workspace.excluded.lst | sudo tee -a {man} >/dev/null; "
                     f"sudo tar czf /tmp/{tag}.tgz --exclude='{os.path.basename(rd)}/{exc[2:]}' -C {os.path.dirname(rd)} "
                     f"{os.path.basename(rd)}; sudo chmod 644 /tmp/{tag}.tgz {man}", zone, timeout=1800)
    else:
        ssh(machine, f"cd {rd} && sudo find . -type f -print0 | sort -z | sudo xargs -0 sha256sum > {man}; "
                     f"sudo tar czf /tmp/{tag}.tgz -C {os.path.dirname(rd)} {os.path.basename(rd)}; "
                     f"sudo chmod 644 /tmp/{tag}.tgz {man}", zone, timeout=1800)
    if download(machine, zone, f"/tmp/{tag}.tgz", os.path.join(work, "run.tgz")) or download(machine, zone, man, os.path.join(work, "manifest")):
        log("copy-failed", run_id=run_id); return "failed", ["copy to the controller failed"]
    subprocess.run(["tar", "xzf", os.path.join(work, "run.tgz"), "-C", work], check=True)
    src = os.path.join(work, os.path.basename(rd))
    subprocess.run(["rsync", "-a", src + "/", local + "/"], check=True)
    bad = []
    for l in open(os.path.join(work, "manifest")):
        h, f = l.rstrip("\n").split("  ", 1)
        q = os.path.join(local, f[2:] if f.startswith("./") else f)
        if not os.path.exists(q) or hashlib.sha256(open(q, "rb").read()).hexdigest() != h:
            bad.append(f)
    ssh(machine, f"sudo rm -f /tmp/{tag}.tgz {man}", zone)
    if bad:
        log("copy-mismatch", run_id=run_id, files=len(bad)); return "failed", [f"controller copy differs in {len(bad)} file(s)"]
    if (row.get("machine_type") or "").startswith("e2-"):
        b = subprocess.run([sys.executable, "-I", BALLOON, "--run", f"vm:{rd}", "--json"], capture_output=True, text=True,
                           timeout=300, env=env_for(machine, zone))
        open(os.path.join(local, "balloon.json"), "w").write(b.stdout)
    # the controller's re-score with its current scorer (the machine's pinned scorer can be older): the machine's score is
    # kept as score.collected.*, and the controller's is the one the claim and the notes read; a failure keeps the machine's
    for f in ("score.json", "score.txt"):
        if os.path.exists(os.path.join(local, f)):
            shutil.copy(os.path.join(local, f), os.path.join(local, f.replace("score.", "score.collected.")))
    # the controller's score is now the only one for a model run: if it fails there is nothing to restore, so it is logged
    # and the claim reads failed with the reason (below)
    rs_ = subprocess.run([sys.executable, "-I", os.path.join(RUNNER, "score", "score_all.py"), local], capture_output=True,
                         text=True, timeout=900, cwd=tempfile.gettempdir())
    if rs_.returncode != 0:
        log("controller-rescore-failed", run_id=run_id, detail=rs_.stderr[-300:])
        for f in ("score.json", "score.txt"):
            if os.path.exists(os.path.join(local, f.replace("score.", "score.collected."))):
                shutil.copy(os.path.join(local, f.replace("score.", "score.collected.")), os.path.join(local, f))
    if not os.path.exists(os.path.join(local, "score.json")):
        log("no-score", run_id=run_id, detail=(rs_.stderr or "")[-300:])
        return "failed", ["the controller's score_all produced no score.json: " + (rs_.stderr or "")[-200:]]
    sj = json.load(open(os.path.join(local, "score.json")))
    v = sj.get("validity") or {}
    # our own ceiling stopping the run is on the claim too, so the backlog never reads a cap stop as the agent's end
    set_claim(run_id, stopped_by=(sj.get("stopped_by") or {}).get("stopped_by"),
              report_latency_s=(sj.get("headline") or {}).get("report_latency_s"))
    # invalid (a job crashed in a way an H100 would not, or a scripted job did not complete) voids the run like a
    # served-model mismatch: it needs a -v2 line, and a void claim never satisfies a run: requirement
    status = {"valid": "done", "void": "void", "invalid": "void"}.get(v.get("status"), "failed")
    # a run someone stopped (run_one's external .stop, the study's or a person's) is a record, not a result: its
    # claim says stopped, so it never satisfies a run: requirement and never reads as done
    stj = json.load(open(os.path.join(local, "status.json"))) if os.path.exists(os.path.join(local, "status.json")) else {}
    if stj.get("status") == "stopped":
        # unless the agent had already finished: its final answer came, no task job ran and nothing remained, and the
        # .stop only freed a machine its session was holding (the study, 2026-10-21 23:44Z: G-passbar-haiku-L0-v2's
        # self-matching pgrep loop). That run is done, scored to the final answer, with the stop beside it
        if finished_before_stop(sj, stj):
            rl = sj.get("report_latency") or {}
            return status, [f"done: the agent's final answer at {rl.get('report_s') or 0:.0f} s from its start, then its "
                            f"session was held by its own background task until an external stop freed the machine"] + (v.get("reasons") or [])
        return "stopped", [f"stopped: {stj.get('reason') or 'external stop'}"] + (v.get("reasons") or [])
    # the work-done close ended the session while task work ran (an open pool, before lifecycle 99e6cb1c): the
    # environment ended it, so it is a record, not the agent's failure
    hk = ((sj.get("idle") or {}).get("work_remaining_at_window_end") or {}).get("killed_at_harness_exit") or []
    # unless the scorer found the grace close took them with nothing owed (killed_after_grace_close; the analysis
    # 2026-10-28 05:02Z, G-g4na-astracc-L0-r2's fourth soap run): the agent gave no answer in the grace, which is its doing
    if sj.get("killed_after_grace_close"):
        hk = []
    if hk and str(stj.get("reason") or "").startswith("lifecycle: work done"):
        return "stopped", [f"environment-ended: the work-done close ended the session with {len(hk)} rank(s) running "
                           f"({hk[0]['task_job']} at step {hk[0]['reached_step']} of {hk[0]['final_step']}); not the agent's"] + (v.get("reasons") or [])
    # our own ceiling (a token cap or the dollar budget) ended the run: truncated, so a record and never done
    #
    sb = (sj.get("stopped_by") or {})
    if sb.get("stopped_by"):
        return "stopped", [f"stopped by our {sb['stopped_by']} ceiling at {sb.get('stop_s') or 0:.0f} s: {sb.get('detail') or ''}"] + (v.get("reasons") or [])
    return status, v.get("reasons") or []

def recollected(run_id, was, status, how):
    """A claim corrected after its collect: logged beside the original `finished`, which the log keeps (it is append-only;
    the claim is the truth), so a reader of the log alone does not take the old status."""
    log("recollected", run_id=run_id, was=was, status=status, how=how[:300])

def finished_before_stop(sj, stj):
    """An externally stopped run whose agent had already finished: the harness's last result came before the stop
    (status.json harness_end_wall, from stdin's close on 'a result with no background task alive'... or the last turn
    end), no task job or open-pool instance was running or remained, and the report came after the work."""
    wr = ((sj.get("idle") or {}).get("work_remaining_at_window_end") or {})
    op = ((sj.get("idle") or {}).get("open_pool") or {})
    rl = sj.get("report_latency") or {}
    running = [u for u in op.get("unfinished") or [] if u.get("cause") == "still running at the window's end"]
    return (wr.get("remaining") == 0 and not running and rl.get("report_latency_s") is not None
            and rl["report_latency_s"] >= 0 and not wr.get("killed_at_harness_exit"))

REFUSALS = os.path.join(CAMPAIGN, "refusals.jsonl")

def refusal_log(run_id, history):
    """A line released after a launch refusal keeps its history: no agent ran, nothing was spent; the claim file is
    removed so the line is claimable under its own id (twins depend on it by run: id)."""
    with LOCK:
        with open(REFUSALS, "a") as fh:
            for h in history:
                fh.write(json.dumps({"run_id": run_id, **h}) + "\n")

def run_line(line, machine, c, row, dry=False, imgs=None):
    zone, mt = row.get("zone"), row.get("machine_type", "")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tag = f"P006-{line['run_id']}-{stamp}"[:128]
    rd = f"/srv/runs/{tag}"
    if imgs is None and line.get("harness") in HARN_RUN_ONE:
        imgs = images(machine, zone)
    cmd, err = smoke_cmd(line, c, tag, mt, machine, imgs=imgs)
    if err:
        return None, err
    if dry:
        return {"tag": tag, "cmd": " ".join(cmd)}, None
    rec = claim(line, machine)
    if not rec:
        return None, "claimed by another worker"
    log("claimed", run_id=line["run_id"], machine=machine, tag=tag)
    set_claim(line["run_id"], tag=tag, run_dir=rd, machine_type=mt, backlog=load_campaign()[2])
    # the task file the run starts on (the study, 2026-10-22 23:22Z: mls-kv-deploy's NeedleBench records regenerate
    # under live runs, so each claim says which version it ran); the run's own task.json holds the same bytes
    tfp = os.path.join(EXP, "gpu-tasks", "registry", "tasks", f"{line['task']}.json")
    if os.path.exists(tfp):
        set_claim(line["run_id"], task_file_sha256=hashlib.sha256(open(tfp, "rb").read()).hexdigest())
    hts = host_terms_status(line)
    if hts:
        set_claim(line["run_id"], host_terms_status=hts)
    # the machine's images, as pick saw them: a fix bound to image ids (requires_image_ids, the ba36 pair 2026-10-23) is met
    # only where the machine holds one, so an empty set would mark every claim unfixed
    imgs = imgs if imgs is not None else images(machine, zone)
    set_claim(line["run_id"], seeded_recycle_fixed=requirement_met("fix:gpumock-seeded-recycle", dict(row, name=machine), {}, imgs)[0])
    # the mls-kv brief version a model run starts on: 1 until analysis's decision:mlskv-brief
    # names another (its declaration's brief_version), so runs on different briefs are never pooled
    if line.get("task") == "mls-kv-deploy" and line.get("harness") != "scripted":     # mls-kv-report has its own brief
        bv = (declared().get("decision:mlskv-brief") or {}).get("brief_version") or 1
        set_claim(line["run_id"], analysis_labels=dict(claims().get(line["run_id"], {}).get("analysis_labels") or {}, brief_version=bv))
    # an MLS-KV run claimed before gpu-mock's exact passage-retrieval serving reached its machine (the study, 2026-10-22
    # 23:26Z: the first batch runs on the current image; its passage-retrieval scores are served off by rounding)
    if line.get("task") == "mls-kv-deploy" and line.get("harness") != "scripted" and \
            not requirement_met("fix:mlskv-retrieval-exact", dict(row, name=machine), {}, imgs)[0]:
        set_claim(line["run_id"], served_score_off_by_rounding="passage retrieval: served before gpu-mock's exact retrieval serving reached this machine")
    def unstart(why):
        os.remove(os.path.join(CLAIMS, f"{line['run_id']}.json"))
        log("released-unstarted", run_id=line["run_id"], machine=machine, why=why)
        return None, why
    rc, out = ssh(machine, "sudo docker ps --format '{{.Names}}' | grep -c '^p6-box-' ; true", zone)
    if (out.strip().splitlines() or ["0"])[-1] != "0":
        return unstart("another box is up on this machine")
    # The code a run uses must be the code on this controller, reviewed here: run_one.py and the runner's run and score
    # files, byte for byte. A machine with an older tree (one without the capture fixes, say) is refused until deployed.
    diff = tree_differs(machine, zone)
    if diff:
        return unstart("the machine's tree differs from its pins, deploy and re-pin first: " + ", ".join(diff))
    # an image gate names the image id it passed on (requirements-met.json `image_id` beside `label`): the machine's
    # image with that label must be that id, so a run never starts on an image the gate did not test
    lab = image_label(line["task"])
    gated = [g for g in declared().values() if isinstance(g, dict) and g.get("label") == lab and g.get("image_id")]
    if gated:
        # tagged images only (run_one.py ignores dangling ones; isolation's Codex ship leaves the old bases untagged),
        # and each must be a gated id: several gated ids at once carry a rolling ship (the requirement check agrees)
        # an empty read is a dropped ssh, not a machine without the image (c3-9 at 20:28:18Z: the gated box was there):
        # read again, up to three times, before refusing
        for _ in range(3):
            rc, out = ssh(machine, "sudo docker image ls --filter dangling=false -q --no-trunc | sort -u | xargs -r sudo docker image inspect --format "
                                   "'{{.Id}} {{index .Config.Labels \"p006.task-env\"}} {{index .Config.Labels \"p006.role\"}}' 2>/dev/null; echo READ-OK; true", zone)
            if "READ-OK" in out:
                break
            time.sleep(10)
        if "READ-OK" not in out:
            return unstart(f"could not read the machine's images over ssh (three tries)")
        # a by-id image (p006.role=box-image-by-id: isolation's Grok/OpenCode images, run only with --image) is no box image
        # for the label's gate, as in images(); a line that runs in one checks it by id in grok-image/opencode-image
        ids = {l.split()[0].replace("sha256:", "")[:12] for l in out.splitlines()
               if len(l.split()) >= 2 and l.split()[1] == lab and "box-image-by-id" not in l.split()[2:]}
        want = {g["image_id"].replace("sha256:", "")[:12] for g in gated}
        if not ids or not ids <= want:
            return unstart(f"the machine's {lab} image(s) {sorted(ids)} are not among the gated ids {sorted(want)}")
    # the task file the box stages from is the machine's copy: it must be the registry's current version
    tf = f"gpu-tasks/registry/tasks/{line['task']}.json"
    want_t = hashlib.sha256(open(os.path.join(EXP, tf), "rb").read()).hexdigest()
    # an empty read is a dropped ssh, not a stale file (isolation, 2026-10-21: c3-12's 17:14Z refusal, its file the
    # controller's exactly): read again, up to three times, before calling the file missing or different
    for _ in range(3):
        rc, out = ssh(machine, f"sha256sum {REMOTE_EXP}/{tf} 2>/dev/null | cut -c1-64; echo READ-OK; true", zone)
        if "READ-OK" in out:
            break
        time.sleep(10)
    got_t = [l.strip() for l in out.splitlines() if re.fullmatch(r"[0-9a-f]{64}", l.strip())]
    if "READ-OK" not in out:
        return unstart(f"could not read the machine's {tf} over ssh (three tries)")
    if (got_t or [""])[-1] != want_t:
        return unstart(f"the machine's {tf} is missing or differs from the registry's; deploy gpu-tasks' registry first")
    model = line["harness"] != "scripted"
    if model:
        # the machine's zone and region beside the round trip
        set_claim(line["run_id"], gate_rtt_before=gate_rtt(machine, zone), zone=zone, region=(zone or "").rsplit("-", 1)[0] or None)
    h = subprocess.run(["bash", HEALTH, *(["--need-tunnel"] if model else []), "--mem-baseline", "--mem-json"],
                       capture_output=True, text=True, timeout=600, env=env_for(machine, zone))
    if re.search(r"summary: [1-9]\d* failed", h.stdout):
        return unstart("health.sh failed: " + "; ".join(l.strip() for l in h.stdout.splitlines() if l.startswith("FAIL"))[:300])
    hours = int(horizon_of(line["task"]) / 3600) + 2
    ssh(machine, "sudo mkdir -p /srv/work/runner", zone)
    if line.get("brief_variant") or line.get("brief_append"):
        # the S line's brief, written whole on the machine and read back by hash before the start
        text, _ = brief_text(line)
        want_b = hashlib.sha256(text.encode()).hexdigest()
        b64 = _b64.b64encode(text.encode()).decode()
        rc, out = ssh(machine, f"echo {b64} | base64 -d | sudo tee {brief_path(tag)} >/dev/null && sudo chmod 644 {brief_path(tag)} && "
                               f"sha256sum {brief_path(tag)} | cut -c1-64; true", zone)
        if want_b not in out:
            return unstart(f"the brief file did not read back on the machine (want {want_b[:12]})")
        set_claim(line["run_id"], brief_file=brief_path(tag), brief_sha256=want_b)
    subprocess.run(["bash", VM, "lease", f"run-{line['run_id']}", str(hours)], capture_output=True, text=True, env=env_for(machine, zone))
    relay = None
    if model:
        subprocess.run([sys.executable, "-I", os.path.join(RUNNER, "score", "set_ceiling.py"), tag, str(line["ceiling_usd"])],
                       check=True, capture_output=True, text=True)
        relay = subprocess.Popen([sys.executable, "-I", os.path.join(RUNNER, "score", "stop_relay.py"), tag],
                                 stdout=open(os.path.join(HERE, f"relay-{line['run_id']}.log"), "w"), stderr=subprocess.STDOUT,
                                 env=env_for(machine, zone))
        token_watch(tag, line, os.path.join(HERE, f"relay-{line['run_id']}.log"))
    unit = f"box-run-{line['run_id']}"
    rc, out = ssh(machine, f"sudo systemd-run --unit {unit} --collect --property=StandardOutput=file:/srv/work/runner/{tag}.log "
                           f"--property=StandardError=file:/srv/work/runner/{tag}.log --working-directory={REMOTE_EXP}/runner "
                           + " ".join(cmd), zone)
    if "Running as unit" not in out:
        if relay:
            relay.terminate()
        set_claim(line["run_id"], status="failed", why="systemd-run did not start the run: " + out[-300:])
        log("start-failed", run_id=line["run_id"], detail=out[-300:])
        return None, "start failed"
    set_claim(line["run_id"], status="running", started_utc=utc(), unit=unit)
    log("running", run_id=line["run_id"], machine=machine, unit=unit)
    state = wait_unit(machine, zone, unit, hours * 3600, run_id=line["run_id"], run_dir=rd)
    if relay:
        try:
            relay.wait(timeout=120)
        except subprocess.TimeoutExpired:
            relay.terminate()
    rec = set_claim(line["run_id"], ended_utc=utc(), unit_state=state)
    # the run never started a box (run_smoke or run_one refused before it: an unknown flag, a missing input): a launch
    # refusal, released for re-claim under its own id and set aside here, never "copy to the controller failed"
    #
    rc, out = ssh(machine, f"test -d {rd} && test -e {rd}/status.json && echo HAS_STATUS || echo NO_STATUS; "
                           f"sudo tail -5 /srv/work/runner/{tag}.log 2>/dev/null; true", zone)
    if "NO_STATUS" in out:
        why = "launch refused before any box: " + " | ".join(l.strip() for l in out.splitlines()[1:] if l.strip())[-300:]
        if relay:
            relay.terminate()
        history = (rec.get("refusals") or []) + [{"utc": utc(), "machine": machine, "tag": tag, "why": why}]
        os.remove(os.path.join(CLAIMS, f"{line['run_id']}.json"))
        refusal_log(line["run_id"], history)
        subprocess.run(["bash", VM, "lease", f"run-{line['run_id']}", "0"], capture_output=True, text=True, env=env_for(machine, zone))
        log("released-unstarted", run_id=line["run_id"], machine=machine, why=why)
        return None, why
    status, reasons = collect(line["run_id"], rec, line)
    open(os.path.join(LAPTOP_RUNS, line["run_id"], "health.txt"), "w").write(h.stdout)
    set_claim(line["run_id"], status=status, reasons=reasons, collected_utc=utc(), laptop_copy=os.path.join(LAPTOP_RUNS, line["run_id"]))
    subprocess.run(["bash", VM, "lease", f"run-{line['run_id']}", "0"], capture_output=True, text=True, env=env_for(machine, zone))
    log("finished", run_id=line["run_id"], status=status, reasons="; ".join(reasons))
    regression_alert(line["run_id"])
    wait_table()
    return status, None

def adopt(machine):
    """Finish what an earlier driver process left running on this machine: wait for its unit, collect it, and set
    its claim. A driver restart then never orphans a run, and never starts a second box beside it."""
    for rid, rec in sorted(claims().items()):
        if rec.get("machine") != machine or rec.get("status") != "running" or not rec.get("unit"):
            continue
        zone = (fleet().get(machine) or {}).get("zone")
        log("adopting", run_id=rid, machine=machine, unit=rec["unit"])
        relay = None
        line = next((l for l in load_campaign()[1] if l["run_id"] == rid), {})
        # a relay an earlier driver started for this run is still running (it outlives its parent): don't add another
        running_relay = subprocess.run(["pgrep", "-f", f"stop_relay.py {rec.get('tag')}"], capture_output=True).returncode == 0
        if rec.get("tag") and line.get("harness") not in (None, "scripted") and not running_relay:
            relay = subprocess.Popen([sys.executable, "-I", os.path.join(RUNNER, "score", "stop_relay.py"), rec["tag"]],
                                     stdout=open(os.path.join(HERE, f"relay-{rid}.log"), "a"), stderr=subprocess.STDOUT,
                                     env=env_for(machine, zone))
        # a token watcher only for a run whose harness is still up: one started on a finished run, with the backlog's cap
        # rather than the one it ran under, crosses at once and rewrites the stop record (G-passbar-astra-L0-v2, adopted
        # 14:47:32Z with 90M after a live 100M watcher had stopped it at 08:28Z; the Codex driver, 2026-10-22 16:51Z)
        rc_, out_ = ssh(machine, f"test -e {rec.get('run_dir')}/status.json && grep -q '\"end_utc\"' {rec.get('run_dir')}/status.json "
                                 f"&& echo HARNESS_ENDED; true", zone) if rec.get("run_dir") else (1, "")
        if rec.get("tag") and line.get("token_caps") and "HARNESS_ENDED" not in out_ and subprocess.run(
                ["pgrep", "-f", f"watch-tokens {rec['tag']}"], capture_output=True).returncode != 0:
            token_watch(rec["tag"], line, os.path.join(HERE, f"relay-{rid}.log"))
        # as long as a fresh run would wait: its task's horizon plus 2 h, at least 24 h (an adopted G-mlskv run has a 46-h
        # horizon, goal-pass-bar 28 h; a fixed 24 h would return driver-timeout with the run still going)
        hz = horizon_of(line["task"]) if line.get("task") else None
        state = wait_unit(machine, zone, rec["unit"], max(24 * 3600, (int(hz / 3600) + 2) * 3600 if hz else 0),
                          run_id=rid, run_dir=rec.get("run_dir"))
        if relay:
            try:
                relay.wait(timeout=120)
            except subprocess.TimeoutExpired:
                relay.terminate()
        rec = set_claim(rid, ended_utc=utc(), unit_state=state)
        status, reasons = collect(rid, rec)
        set_claim(rid, status=status, reasons=reasons, collected_utc=utc(), laptop_copy=os.path.join(LAPTOP_RUNS, rid))
        subprocess.run(["bash", VM, "lease", f"run-{rid}", "0"], capture_output=True, text=True, env=env_for(machine, zone))
        log("finished", run_id=rid, status=status, reasons="; ".join(reasons), adopted=True)
        regression_alert(rid)
        wait_table()

def loop_locks(machines):
    """One loop per machine: an exclusive flock on
    $P006_GATE_HOME/loop-<machine>.lock, held for the process's life and released by the kernel however it ends. A second
    loop for a machine (a hand-started copy beside launchd's, or KeepAlive's respawn racing a helper's pkill and restart)
    finds it held and exits before adopt, so it never touches a claim; launchd retries it every ThrottleInterval, and it
    takes over when the holder ends. The fd is not inherited (PEP 446), so a relay that outlives the loop never holds it.
    Returns (held fds, None), or (None, (machine, its holder's "pid utc")) with nothing held."""
    import fcntl
    g = os.path.expanduser(os.environ.get("P006_GATE_HOME", "~/.p006-gate"))
    fds = []
    for m in machines:
        fd = os.open(os.path.join(g, f"loop-{m}.lock"), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = os.pread(fd, 200, 0).decode(errors="replace").strip()
            for x in fds + [fd]:
                os.close(x)
            return None, (m, holder)
        os.ftruncate(fd, 0)
        os.pwrite(fd, f"{os.getpid()} {utc()}\n".encode(), 0)
        fds.append(fd)
    return fds, None

def worker(machine, once=False, only=None, wait_hours=0.0, only_file=None):
    while True:
        try:
            adopt(machine)
            break
        except Exception as ex:      # a loop started during a lapse waits for it to end rather than dying
            log("adopt-error", machine=machine, error=repr(ex)[:300])
            time.sleep(300)
    hold_since = {}         # Q line -> when this worker first saw it waiting only on its declaration
    t_end = time.time() + 3600 * wait_hours
    last_wait = None
    held_logged = False
    skip = {}               # run_id -> until: a line this machine refused to start (its task file, its image), set aside
    tree_repinned = False   # a re-pin already tried on this refusal of the tree
    while True:
        # a hold file (runner/campaign/hold-<machine>) keeps this worker from taking a line, between runs only: a run
        # in progress finishes and is collected first. For deploys and image loads in a machine's gap (isolation's
        # scrub ship, 2026-10-21); deleting the file releases the machine at the next 30 s check
        hold = os.path.join(HERE, f"hold-{machine}")
        # a hold file whose every line is gone (isolation's ship_gap removes only its own line; the releaser that deleted
        # empty files is retired, 2026-10-22 17:49Z) holds nothing: it is removed, and the machine is released. Before
        # that, this worker re-pins from the machine, since a ship in the gap changed its tree (the re-pin start-machine.sh
        # did at release)
        if os.path.exists(hold) and not open(hold).read().strip():
            try:
                os.remove(hold)
            except FileNotFoundError:
                pass
        if os.path.exists(hold):
            if not held_logged:
                log("held", machine=machine, why=open(hold).read().strip()[:200])
                held_logged = True
            time.sleep(30)
            continue
        if held_logged:
            log("released", machine=machine)
            held_logged = False
            # a ship in the gap changes the machine's tree, whoever then removed the hold file (isolation's ship_gap deletes
            # it outright; 2026-10-22 18:01Z, c3-9, -11 and -20 then sat 20 min refused on the tree): re-pin from the
            # machine, accepted only when every pinned file there equals the controller's
            try:
                if tree_differs(machine, (fleet().get(machine) or {}).get("zone")):
                    repin(machine, "re-pinned at release from a hold: the ship in the gap put the controller's files on")
            except Exception as ex:      # the tree check still guards the next start
                log("repin-error", machine=machine, error=repr(ex)[:200])
        c, lines, _ = load_campaign()
        fl_ = fleet()
        row = fl_.get(machine)
        # fleet.py could not answer (a lapse): its rows come back stale, or none at all when it never answered in this
        # loop's life (the study 2026-10-24 11:33Z: c3-23 released during the 11:09Z lapse and its loop exited on a
        # missing row). Either way wait and re-check; never conclude the machine is gone from a fleet.py that did not answer
        if fleet_unanswered(fl_, machine):
            if not held_logged:
                log("fleet-unreachable", machine=machine)
            time.sleep(300)
            continue
        if not row or row.get("status") != "RUNNING":
            log("machine-not-running", machine=machine); return
        imgs = images(machine, row.get("zone"))
        if only_file:
            # the priority list re-read before every pick, so an edit reaches every loop at its next gap with no restart
            # (2026-10-22 15:56Z: c3-12's loop, started on the list before orders 329-330, took A-stagger-haiku-L3 first)
            try:
                only = read_only_file(only_file)
            except OSError as ex:
                log("only-file-unreadable", machine=machine, error=repr(ex)[:200])
        now = time.time()
        lines = [l for l in lines if skip.get(l["run_id"], 0) <= now]
        line, why = pick(machine, row, c, lines, claims(), imgs, only=only, hold_since=hold_since)
        if not line:
            if time.time() < t_end:
                w = "; ".join(f"{rid}: {[r for r, _ in u]}" for rid, u in why[:3]) or "no open, unclaimed line matches"
                if w != last_wait:
                    log("waiting", machine=machine, only=",".join(only or []), why=w)
                    last_wait = w
                # a free machine re-picks every minute (forum 2026-10-23: the package screen, where a line's last requirement
                # landing should start it within a minute); each pick is one ssh to this machine
                time.sleep(PICK_WAIT_S)
                continue
            log("nothing-to-take", machine=machine, only=",".join(only or [])); return
        last_wait = None
        try:
            status, err = run_line(line, machine, c, row, imgs=imgs)
        except Exception as ex:
            # one line's error never ends the loop (c3-11, 2026-10-22: an ssh timeout in collect killed it at 00:14Z and its
            # finished run sat uncollected); its claim stays as it was, and adopt collects it on the next pass. adopt's own
            # failure (the reauth lapse made fleet() raise inside it, 05:30Z) is caught too, and retried after 5 minutes
            log("run-line-error", machine=machine, run_id=line["run_id"], error=repr(ex)[:300])
            while True:
                time.sleep(300)
                try:
                    adopt(machine)
                    break
                except Exception as ex2:
                    log("adopt-error", machine=machine, error=repr(ex2)[:300])
            continue
        if err:
            log("not-run", machine=machine, run_id=line["run_id"], why=err)
            # a refusal never ends the loop. One
            # about this line (its task file, its image, its command) sets the line aside for 15 min and the loop takes
            # the next; one about the machine (another box up, its tree, health, the start) waits 5 min and retries
            # the tree changed with no hold seen (a ship's release fell between two loops, 2026-10-22 23:58Z: c3-22): re-pin from
            # the machine once, on the same rule as at release, and retry at once rather than after the 5-min wait
            if err.startswith("the machine's tree") and not tree_repinned:
                tree_repinned = True
                try:
                    if repin(machine, "re-pinned after a start refused on the tree: the machine's files are the controller's (or as pinned)"):
                        continue
                except Exception as ex:
                    log("repin-error", machine=machine, error=repr(ex)[:200])
            elif not err.startswith("the machine's tree"):
                tree_repinned = False
            machine_level = err.startswith(("another box", "the machine's tree", "health.sh", "start failed"))
            if machine_level:
                time.sleep(300)
            elif err != "claimed by another worker":
                skip[line["run_id"]] = time.time() + 900
        if once:
            return

def selftest():
    # a release in a lapse must not end the loop: fleet() answers {} or stale rows
    assert fleet_unanswered({}, "worker-23") is True
    assert fleet_unanswered({"worker-9": {"status": "RUNNING", "stale": True}}, "worker-23") is True
    assert fleet_unanswered({"worker-23": {"status": "RUNNING", "stale": True}}, "worker-23") is True
    assert fleet_unanswered({"worker-23": {"status": "RUNNING"}}, "worker-23") is False
    assert fleet_unanswered({"worker-9": {"status": "RUNNING"}}, "worker-23") is False     # a fresh answer without it: truly gone
    assert fleet_unanswered({"worker-23": {"status": "TERMINATED"}}, "worker-23") is False
    """Known answers for rules the driver applies itself (no machine, no claim touched)."""
    ok = True
    def check(name, got, want):
        nonlocal ok
        good = got == want; ok &= good
        print(("  ok   " if good else "  FAIL ") + f"{name}: got {got}, want {want}")
    A = 1792614700.0
    # a run in flight counts as soon only while its expected end is ahead and within SOON_RUN_S
    _st = "2026-10-27T00:41:13Z"; _t0 = datetime.datetime.fromisoformat(_st.replace("Z", "+00:00")).timestamp()
    _rec = {"status": "running", "started_utc": _st}
    check("G-kvgate-gemma-L0, 978 s past its 16,456-s estimate: not soon (it held every fill line on five machines)",
          run_ends_soon("G-kvgate-gemma-L0", _rec, 16456.3, _t0 + 16456.3 + 978), False)
    check("the same run 600 s before its expected end: soon; 1,200 s before: not soon (beyond SOON_RUN_S)",
          (run_ends_soon("G-kvgate-gemma-L0", _rec, 16456.3, _t0 + 16456.3 - 600),
           run_ends_soon("G-kvgate-gemma-L0", _rec, 16456.3, _t0 + 16456.3 - 1200)), (True, False))
    check("a Q line or a sped-up W0 check in flight is soon whatever its estimate; a finished or unclaimed run never is",
          (run_ends_soon("Q-x", _rec, 99999, _t0 + 99999 * 2), run_ends_soon("W0-x", _rec, None, _t0, 10.0),
           run_ends_soon("G-x", {"status": "done", "started_utc": _st}, 100, _t0 + 50), run_ends_soon("G-x", {}, 100, _t0 + 50)),
          (True, True, False, False))
    # a speed-1 W0 reference reads as any line (the study 2026-11-02 22:22Z: W0-sfz-reference-r2, 30,927 s expected, held c3-21 and
    # c3-6 off every long fill line for 35 min): soon only within SOON_RUN_S of its expected end
    check("a speed-1 W0 reference 16 min into its 30,927-s run is not soon; 600 s before its expected end it is",
          (run_ends_soon("W0-sfz-reference-r2", _rec, 30927.1, _t0 + 1000, 1.0), run_ends_soon("W0-sfz-reference-r2", _rec, 30927.1, _t0 + 30927.1 - 600, 1.0)),
          (False, True))
    # --only exclusions (the study, 2026-10-22 15:15Z: the r3/r4 lines rank after every other Q line)
    R34 = "!Q-stagger-*-L0-r[34]!Q-p20tail-*-L0-r[34]"
    check("Q-*-haiku-* with the r3/r4 exclusion skips Q-stagger-haiku-L0-r3", only_match("Q-stagger-haiku-L0-r3", "Q-*-haiku-*" + R34), False)
    check("... and still takes Q-dataprepP6-haiku-L0", only_match("Q-dataprepP6-haiku-L0", "Q-*-haiku-*" + R34), True)
    check("... and an r3 of an earlier order (Q-p6par-sonnet-L0-r3, 294) is still a Q-* line", only_match("Q-p6par-sonnet-L0-r3", "Q-*" + R34), True)
    check("a plain pattern is plain fnmatch", only_match("Q-p20tail-sonnet-L0-r4", "Q-*-L0-r[34]"), True)
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write("G-*,A-stagger-*!A-stagger-*-L3\nA-stagger-*-L3\n\n")
    # the S rungs and P: --scheduler on, the brief file, the skill mount
    s1 = {"task": "pkg-stagger-q", "harness": "claude-code", "model": "m", "ceiling_usd": 1, "speed": 1.0, "rung": "S1",
          "brief_variant": "no_launch_line", "brief_append": "This machine runs SLURM."}
    cmd_, err_ = smoke_cmd(s1, {"box": {}}, "P006-T", "c3-highmem-44")
    ra = cmd_[cmd_.index("--") + 1:] if cmd_ else []
    check("an S1 line: --scheduler on after --, and --brief-file before it", (err_, "--scheduler" in ra and ra[ra.index("--scheduler") + 1] == "on",
          cmd_ and cmd_.index("--brief-file") < cmd_.index("--")), (None, True, True))
    bt_, _ = brief_text(s1)
    check("the S1 brief: the variant's text, a blank line, Text S1", (bt_.endswith("\n\nThis machine runs SLURM."), "Run, each on one GPU" not in bt_), (True, True))
    s0 = dict(s1, rung="S0"); s0.pop("brief_variant"); s0.pop("brief_append")
    cmd_, _ = smoke_cmd(s0, {"box": {}}, "P006-T", "c3-highmem-44")
    check("an S0 line: the scheduler, the task's own brief", ("--scheduler" in cmd_, "--brief-file" in cmd_), (True, False))
    pl = {"task": "pkg-stagger-q", "harness": "claude-code", "model": "m", "ceiling_usd": 1, "speed": 1.0, "skill": "parallel-attention", "nudge": "P"}
    cmd_, _ = smoke_cmd(pl, {"box": {}}, "P006-T", "c3-highmem-44")
    check("a P line: --skill <REMOTE_EXP>/solution/skills/parallel-attention after --, no scheduler",
          (f"--skill={REMOTE_EXP}/solution/skills/parallel-attention" in cmd_[cmd_.index("--"):], "--scheduler" in cmd_), (True, False))
    # a Codex line with a skill: the same --skill argument after --, as for Claude Code;
    # the claude-code line's argv unchanged by it; another harness (grok) still refused
    _cx_cmd, _cx_err = smoke_cmd(dict(pl, harness="codex", skill="gpu-harness-a5-codex"), {"box": {}}, "P006-T", "c3-highmem-44")
    check("a Codex line with a skill: no refusal, --skill <REMOTE_EXP>/solution/skills/gpu-harness-a5-codex after --",
          (_cx_err, _cx_cmd is not None and f"--skill={REMOTE_EXP}/solution/skills/gpu-harness-a5-codex" in _cx_cmd[_cx_cmd.index("--"):]), (None, True))
    _cc_cmd, _cc_err = smoke_cmd(dict(pl, skill="gpu-harness-a5-codex"), {"box": {}}, "P006-T", "c3-highmem-44")
    _hi = _cx_cmd.index("--harness") if "--harness" in _cx_cmd else None
    check("the same line on claude-code: no refusal, the same --skill argument, and the codex argv is it plus `--harness codex` alone",
          (_cc_err, [x for x in _cc_cmd if "--skill=" in x], _hi is not None and _cx_cmd[_hi:_hi + 2] == ["--harness", "codex"]
           and _cx_cmd[:_hi] + _cx_cmd[_hi + 2:] == _cc_cmd),
          (None, [f"--skill={REMOTE_EXP}/solution/skills/gpu-harness-a5-codex"], True))
    check("a Grok line with a skill is still refused", smoke_cmd(dict(pl, harness="grok"), {"box": {}}, "P006-T", "c3-highmem-44")[1],
          "skills apply to claude-code and codex lines only")
    # Astra in Claude Code: --effort-level after --, astracc-runone on a non-Claude model
    ac = {"run_id": "G-x-astracc-L0", "wave": "G", "task": "pkg-p6-par-q", "harness": "claude-code", "model": "gpt-6-astra:medium[1m]",
          "ceiling_usd": 1, "speed": 1.0, "effort_level": "medium", "requires": []}
    cmd_, err_ = smoke_cmd(ac, {"box": {}}, "P006-T", "c3-highmem-44")
    _ra = cmd_[cmd_.index("--") + 1:] if cmd_ else []
    _ap = pick("no-such-machine", {"name": "no-such-machine"}, {"waves_open": ["G"], "box": {}}, [ac], {}, set(), only=["G-x-astracc-L0"])[1]
    _areq = sorted(r for _, u in _ap for r, _w in u)
    _hp = pick("no-such-machine", {"name": "no-such-machine"}, {"waves_open": ["G"], "box": {}}, [dict(ac, model="claude-haiku-5-5", effort_level=None, run_id="G-x-h-L0")], {}, set(), only=["G-x-h-L0"])[1]
    check("an astracc line: --effort-level medium after --, waits on astracc-runone; a plain Claude line does not; effort_level refused for Codex; c3855199 in every set",
          (err_, "--effort-level" in _ra and _ra[_ra.index("--effort-level") + 1] == "medium", "astracc-runone" in _areq,
           "astracc-runone" in sorted(r for _, u in _hp for r, _w in u),
           smoke_cmd(dict(ac, harness="codex"), {"box": {}}, "P006-T", "c3-highmem-44")[1],
           all("c3855199" in x and "5761a0dd" in x and "f513c811" in x and "88be08e5" in x and "27f9dc27" in x and "cbe91ce0" in x and "89f7c038" in x and "85126860" in x for x in (CODEX_RUN_ONE, CLAUDE_ARG_RUN_ONE, BASH_TIMEOUT_RUN_ONE, HARN_RUN_ONE["grok"], HARN_RUN_ONE["opencode"], AGENT_TEAMS_RUN_ONE, ASTRACC_RUN_ONE))),
          (None, True, True, False, "effort_level applies to claude-code lines only", True))
    # resume by default: run_one f513c811 takes Codex lines only beside codex_session db728687,
    # read from the machine; 5761a0dd needs no particular module. A scratch pins file, so the live one is never touched
    _pins_live, _tp = PINS, tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"sha256": {}, "by_machine": {"m-new": {"sha256": {"isolation/run_one.py": "f513c811" + "0" * 56}},
                                             "m-old": {"sha256": {"isolation/run_one.py": "5761a0dd" + "0" * 56}},
                                             "m-wd": {"sha256": {"isolation/run_one.py": "88be08e5" + "0" * 56}}}}, _tp); _tp.close()
    globals()["PINS"] = _tp.name
    try:
        _cx = lambda m, have: requirement_met("codex-runone", {"name": m}, {}, have)[0]
        check("codex-runone: f513c811 with codex_session db728687 met; with a3fd2093 or none read, not met; 5761a0dd met with any",
              (_cx("m-new", {"codexsession:db7286872dbeeec9"}), _cx("m-new", {"codexsession:a3fd2093aaaa"}), _cx("m-new", set()),
               _cx("m-old", {"codexsession:a3fd2093aaaa"})), (True, False, False, True))
        check("codex-runone: 88be08e5 (f513c811 + the weights digest) is met beside db728687 and not beside a3fd2093",
              (_cx("m-wd", {"codexsession:db7286872dbeeec9"}), _cx("m-wd", {"codexsession:a3fd2093aaaa"})), (True, False))
    finally:
        globals()["PINS"] = _pins_live
        os.unlink(_tp.name)
    _gl, _base = os.getloadavg, UNIT_POLL_BASE_S
    try:
        # the rule at the normal base (60 s); the base itself is the memory-short setting, 150 s since 2026-11-04 10:46 MST
        globals()["UNIT_POLL_BASE_S"] = 60.0
        os.getloadavg = lambda: (119.0, 115.0, 88.0); _hi = unit_poll_s()
        os.getloadavg = lambda: (12.0, 15.0, 20.0); _lo = unit_poll_s()
        globals()["UNIT_POLL_BASE_S"] = 150.0
        _mem = unit_poll_s()
    finally:
        os.getloadavg = _gl
        globals()["UNIT_POLL_BASE_S"] = _base
    check("unit poll: 150 s while the 1-min load is over 40, else 60 s; at the memory-short base, 150 s whatever the load",
          (_hi, _lo, _mem), (150.0, 60.0, 150.0))
    # the re-pin takes a run_one that is one of isolation's kept copies, whatever the controller's working file is now (17:39Z)
    _kh = kept_run_one()
    check("kept_run_one: isolation's frozen 792b27a1 and a76491ef copies count; each one's hash starts with its name",
          (any(h.startswith("792b27a1c96774c9") for h in _kh), any(h.startswith("a76491efb7da27c0") for h in _kh)), (True, True))
    # a gpu-team line: --agent-teams after --, refused off Claude Code, needs 792b27a1 pinned
    gt = dict(pl, skill="gpu-team", agent_teams=True)
    cmd_, err_ = smoke_cmd(gt, {"box": {}}, "P006-T", "c3-highmem-44")
    check("an agent_teams line: --agent-teams after --, with its skill; none on a line without the field; refused for Codex",
          (err_, "--agent-teams" in cmd_[cmd_.index("--"):], "--agent-teams" in smoke_cmd(pl, {"box": {}}, "P006-T", "c3-highmem-44")[0],
           smoke_cmd(dict(gt, harness="codex", skill=None), {"box": {}}, "P006-T", "c3-highmem-44")[1]),
          (None, True, False, "agent_teams applies to claude-code lines only"))
    check("agent-teams-runone is unmet on a machine with no pinned run_one; 792b27a1 also counts for codex, claude-arg, bash-timeout and grok",
          (requirement_met("agent-teams-runone", {"name": "p006-no-pin"}, {}, set())[0], "792b27a1" in CODEX_RUN_ONE,
           "792b27a1" in CLAUDE_ARG_RUN_ONE, "792b27a1" in BASH_TIMEOUT_RUN_ONE, "792b27a1" in HARN_RUN_ONE["grok"]), (False, True, True, True, True))
    # Grok Build and OpenCode lines: their own branch, as Codex's, not Claude Code's
    _gl = {"run_id": "G-x-grok-L0", "wave": "G", "task": "pkg-g4n-a", "harness": "grok", "model": "grok-4.7", "ceiling_usd": 1,
           "speed": 1.0, "requires": []}
    _gp = pick("no-such-machine", {"name": "no-such-machine"}, {"waves_open": ["G"], "box": {}}, [_gl], {}, set(), only=["G-x-grok-L0"])[1]
    _greq = sorted(r for _, u in _gp for r, _w in u)
    check("a Grok line waits on grok-image and grok-runone, and on harness:grok until it is declared (pick lists the unmet), and on none of Claude Code's lifecycle or fifo gates",
          (all(x in _greq for x in ("grok-image:mng3-packages", "grok-runone")) and ("harness:grok" in _greq or bool(declared().get("harness:grok"))),
           any(x.startswith(("fix:background-lifecycle", "fifo-image:", "fix:open-pool-lifecycle")) for x in _greq)), (True, False))
    _hid = HARN_IMAGE["mng3-packages"].replace("sha256:", "")[:12]
    check("grok-image needs the label's harness image held by id and labelled p006.grok; either alone does not meet it",
          (requirement_met("grok-image:mng3-packages", {}, {}, {"grok:mng3-packages", f"byid:mng3-packages:{_hid}"})[0],
           requirement_met("grok-image:mng3-packages", {}, {}, {"grok:mng3-packages"})[0],
           requirement_met("grok-image:mng3-packages", {}, {}, {f"byid:mng3-packages:{_hid}"})[0]), (True, False, False))
    check("a by-id harness image is no box image for its label: gated-image sees only the frozen id",
          requirement_met("gated-image:mng3-packages", {}, {}, {"imageid:mng3-packages:04827d41a7e0", f"byid:mng3-packages:{_hid}"})[0], True)
    check("grok-runone is unmet on a machine with no pinned run_one, and a76491ef is a Grok and OpenCode run_one",
          (requirement_met("grok-runone", {"name": "p006-no-pin"}, {}, set())[0], "a76491ef" in HARN_RUN_ONE["grok"],
           "a76491ef" in HARN_RUN_ONE["opencode"]), (False, True, True))
    _gc = smoke_cmd(dict(_gl, task="pkg-p6-par-q"), {"box": {}}, "P006-T", "c3-highmem-44")[0] or []
    check("a Grok line's run command passes --harness grok, its model, and --image <its label's harness image>",
          ("--harness" in _gc and _gc[_gc.index("--harness") + 1] == "grok", "--model" in _gc and _gc[_gc.index("--model") + 1] == "grok-4.7",
           "--image" in _gc and _gc[_gc.index("--image") + 1] == HARN_IMAGE["mng3-packages"]), (True, True, True))
    # a rolling release ship: a machine holding both harness images runs the newest, one holding only the
    # old runs the old, one holding neither refuses at command build (and grok-image is unmet there anyway)
    _r71, _r7, _r6, _r4, _new, _old = (x.replace("sha256:", "")[:12] for x in HARN_IMAGES["mng3-packages"])
    _seven = {f"byid:mng3-packages:{_r7}", f"byid:mng3-packages:{_r6}", "grok:mng3-packages"}
    _six = {f"byid:mng3-packages:{_r6}", f"byid:mng3-packages:{_r4}", "grok:mng3-packages"}
    _all = {f"byid:mng3-packages:{_r4}", f"byid:mng3-packages:{_new}", f"byid:mng3-packages:{_old}", "grok:mng3-packages"}
    _both = {f"byid:mng3-packages:{_new}", f"byid:mng3-packages:{_old}", "grok:mng3-packages"}
    _oldonly = {f"byid:mng3-packages:{_old}", "grok:mng3-packages"}
    _img = lambda im: (lambda c_: c_[0][c_[0].index("--image") + 1] if c_[0] else c_[1])(smoke_cmd(dict(_gl, task="pkg-p6-par-q"), {"box": {}}, "P006-T", "c3-highmem-44", imgs=im))
    check("per machine: r6's and r4's harness images held -> r6's 2a185b80; r4's (with older) -> 5619e3b3; r2b's and 3ec0's -> r2b's 07aa166d; "
          "only 3ec0's -> 067e72d0; none -> refused; grok-image met on each",
          (_img(_six)[:19], _img(_all)[:19], _img(_both)[:19], _img(_oldonly)[:19], _img(set()).startswith("no harness image"),
           requirement_met("grok-image:mng3-packages", {}, {}, _six)[0], requirement_met("grok-image:mng3-packages", {}, {}, _all)[0],
           requirement_met("grok-image:mng3-packages", {}, {}, _both)[0], requirement_met("grok-image:mng3-packages", {}, {}, _oldonly)[0]),
          ("sha256:2a185b80e893", "sha256:5619e3b347ff", "sha256:07aa166d8bf2", "sha256:067e72d0f780", True, True, True, True, True))
    check("per machine: r7's and r6's harness images held (mid-ship) -> r7's cc441afa; grok-image met",
          (_img(_seven)[:19], requirement_met("grok-image:mng3-packages", {}, {}, _seven)[0]), ("sha256:cc441afadc74", True))
    # r7.1's rolling ship: a machine holding r7.1's and r7's harness images runs r7.1's b58b12e2
    _sevenone = {f"byid:mng3-packages:{_r71}", f"byid:mng3-packages:{_r7}", "grok:mng3-packages"}
    check("per machine: r7.1's and r7's harness images held (mid-ship) -> r7.1's b58b12e2; grok-image met",
          (_img(_sevenone)[:19], requirement_met("grok-image:mng3-packages", {}, {}, _sevenone)[0]), ("sha256:b58b12e2dbf1", True))
    _cc = smoke_cmd(dict(_gl, task="pkg-p6-par-q", harness="claude-code", model="claude-haiku-5-5"), {"box": {}}, "P006-T", "c3-highmem-44")[0] or []
    check("a Claude Code line passes no --image", "--image" in _cc, False)
    check("brief-variant: pkg-stagger-q has no_launch_line; no task has a variant named no-such-variant",
          (requirement_met("brief-variant:pkg-stagger-q:no_launch_line", {}, {}, set())[0], requirement_met("brief-variant:pkg-stagger-q:no-such-variant", {}, {}, set())[0]), (True, False))
    cur = hashlib.sha256(open(os.path.join(EXP, "campaign", "campaign.json"), "rb").read()).hexdigest()
    check("campaign-level:P: the controller's campaign.json carries P; an unknown pinned hash does not",
          ("P" in campaign_levels()[cur], requirement_met("campaign-level:P", {"name": "no-such-machine"}, {}, set())[0]), (True, False))
    check("skill-synced: met only when the machine's SKILL.md hash is the controller's",
          (requirement_met("skill-synced:parallel-attention", {}, {}, set())[0],
           requirement_met("skill-synced:parallel-attention", {}, {}, {"skill:parallel-attention:" + hashlib.sha256(open(os.path.join(EXP, "solution", "skills", "parallel-attention", "SKILL.md"), "rb").read()).hexdigest()})[0]), (False, True))
    check("an --only file: commas and newlines both separate, blanks dropped", read_only_file(tf.name), ["G-*", "A-stagger-*!A-stagger-*-L3", "A-stagger-*-L3"])
    os.unlink(tf.name)
    # gpus-idle-session-up
    check("never busy, agent started 300 s ago: idle 300 s, not the epoch", goal_idle({}, A, A + 300), (0, 300.0))
    check("never busy, no agent start known: no idle figure, so no alert", goal_idle({}, None, A + 300), (0, None))
    check("a GPU busy: no idle", goal_idle({"0": ["busy_start", A + 10], "1": ["busy_end", A + 20]}, A, A + 3000)[0], 1)
    check("all idle, last end 2000 s ago: idle 2000 s", goal_idle({"0": ["busy_end", A + 1000], "1": ["busy_end", A + 900]}, A, A + 3000), (0, 2000.0))
    check("the agent started after the last end: idle from its start", goal_idle({"0": ["busy_end", A - 50]}, A, A + 100), (0, 100.0))
    # the probe reads the whole occupancy file (the study, 2026-10-26 18:48Z: G-kvgate-opus-L0's 4 MB file had no busy
    # line in its last 2 MB, so a GPU busy all along read idle since agent_start): a busy_start, then 2.5 MB of enqueues
    _ip = tempfile.mkdtemp(prefix="idleprobe-")
    os.makedirs(os.path.join(_ip, "box"))
    open(os.path.join(_ip, "box", "agent_start"), "w").write(str(A))
    with open(os.path.join(_ip, "occupancy.jsonl"), "w") as fh:
        fh.write(json.dumps({"kind": "busy_start", "gpu": 2, "t_wall": A + 60}) + "\n")
        _enq = json.dumps({"kind": "enqueue", "gpu": 2, "job": 7, "t_wall": A + 61, "wall_cost": 0.5, "pad": "x" * 200}) + "\n"
        fh.write(_enq * (2_500_000 // len(_enq) + 1))
    _pr = json.loads(subprocess.run([sys.executable, "-I", "-c", IDLE_PROBE, _ip], capture_output=True, text=True).stdout)
    check("a busy_start behind 2.5 MB of enqueue lines is found: GPU 2 busy, no idle figure",
          (_pr["last"].get("2", [None])[0], goal_idle(_pr["last"], _pr["agent_start"], A + 9000)), ("busy_start", (1, None)))
    shutil.rmtree(_ip, ignore_errors=True)
    check("the first 30 min of a new goal run never alerts", goal_idle({}, A, A + IDLE_ALERT_S - 1)[1] < IDLE_ALERT_S, True)
    # an open-pool task is one whose jobs draw "open"; a fixed job list is not
    check("goal-pass-bar is an open pool; pkg-p6-par-q is not", (open_pool_task("goal-pass-bar"), open_pool_task("pkg-p6-par-q")), (True, False))
    cmd, err = smoke_cmd({"task": "pkg-stagger-q", "harness": "claude-code", "model": "m", "ceiling_usd": 1, "speed": 1.0,
                          "bash_timeouts_ms": {"default": 120000, "max": 600000}}, {"box": {}}, "T", "c3-highmem-44", None)
    check("bash_timeouts_ms reaches run_one as --bash-timeout-ms=120000,600000 after --", (err, cmd and cmd[cmd.index("--") + 1:].count("--bash-timeout-ms=120000,600000")), (None, 1))
    check("fix:wakeup-lifecycle is met only on a machine pinned to ff9cd5e8", (requirement_met("fix:wakeup-lifecycle", {"name": "no-such-machine"}, {}, set())[0],), (False,))
    # lifecycle 91085137: a machine pinned to it meets every lifecycle key 8486c765 meets, and
    # fix:cron-lifecycle, which 8486c765 does not; read against a temporary pins file so the live one is untouched
    _mod = sys.modules[__name__]
    _pins_live = _mod.PINS
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as _td:
            _mod.PINS = os.path.join(_td, "pins.json")
            L = "runner/harness/lifecycle.py"
            json.dump({"by_machine": {"m-new": {"sha256": {L: "91085137933cbf35" + "0" * 48}},
                                      "m-old": {"sha256": {L: "8486c765e6a54a09" + "0" * 48}},
                                      "m-b15": {"sha256": {L: "b15d03468d98f5d4" + "0" * 48}},
                                      "m-d26": {"sha256": {L: "d264366468b3df78" + "0" * 48}}}}, open(_mod.PINS, "w"))
            keys = ("fix:open-pool-lifecycle", "fix:wakeup-lifecycle", "fix:lifecycle-scheduler-jobs", "fix:cron-lifecycle")
            got = {m: tuple(requirement_met(k, {"name": m}, {}, set())[0] for k in keys) for m in ("m-new", "m-old", "m-b15", "m-d26")}
    finally:
        _mod.PINS = _pins_live
    check("91085137 meets open-pool, wakeup, scheduler and cron lifecycle keys; 8486c765 all but cron",
          (got["m-new"], got["m-old"]), ((True, True, True, True), (True, True, True, False)))
    check("b15d0346 (the GPU-live and bounded-wait holds) meets every lifecycle key 91085137 meets", got["m-b15"], (True, True, True, True))
    check("d2643664 (b15d0346 + the harness-board hold) meets every lifecycle key 91085137 meets", got["m-d26"], (True, True, True, True))
    # a run_one named by its first 8 hex (isolation's 16f9cfec, the study 2026-10-22 00:14Z) matches its full hash
    wd_ = json.load(open(MET)).get("withdrawn", {}) if os.path.exists(MET) else {}
    if "gate:mlsbench-kv-image-922d" in wd_:
        check("a withdrawn gate's image (the 922d KV box) is refused under gated-image",
              requirement_met("gated-image:mlsbench-kv", {"name": "x"}, {}, {"imageid:mlsbench-kv:3469eda044ae"})[0], False)
    gp = json.load(open(os.path.join(EXP, "gpu-tasks", "registry", "tasks", "goal-pass-bar.json"))).get("horizon_s")
    check("horizon-at-least: met at the task's own horizon, not one second above it",
          (requirement_met(f"horizon-at-least:goal-pass-bar:{gp}", {}, {}, set())[0], requirement_met(f"horizon-at-least:goal-pass-bar:{float(gp) + 1}", {}, {}, set())[0]), (True, False))
    g = {"task": "mls-kv-deploy", "harness": "claude-code"}
    check("host_terms_status: provisional before W0-mlskv-measure-c is done, final after, none for other tasks or scripted lines",
          (host_terms_status(g, {}), host_terms_status(g, {"W0-mlskv-measure-c": {"status": "done"}}),
           host_terms_status(dict(g, task="pkg-stagger-q"), {}), host_terms_status(dict(g, harness="scripted"), {})),
          ("provisional (check10)", "final (measure)", None, None))
    _real_declared = declared
    try:
        globals()["declared"] = lambda: {"fix:per-machine-x": {"by": "t", "utc": "u", "machines": ["worker-17"]}}
        check("a per-machine declaration is met only on its listed machines",
              (requirement_met("fix:per-machine-x", {"name": "worker-17"}, {}, set())[0],
               requirement_met("fix:per-machine-x", {"name": "worker-9"}, {}, set())[0]), (True, False))
    finally:
        globals()["declared"] = _real_declared
    # a supersede line that adds a requirement (W0-tsci-check10, 2026-11-03 10:08Z): its run_id's last line is the one
    # picked, in the first one's place, so the line waits where the new key is unmet
    _a1 = {"run_id": "G-x-sup", "wave": "G", "task": "pkg-stagger-q", "harness": "scripted", "speed": 1.0, "requires": []}
    _b1 = dict(_a1, run_id="G-x-other")
    _a2 = dict(_a1, requires=["fix:no-such-supersede-key"])
    _sp = superseded([_a1, _b1, _a2])
    _ws = pick("no-such-machine", {"name": "no-such-machine"}, {"waves_open": ["G"], "box": {}}, _sp, {}, set(), only=["G-x-sup"])
    check("a supersede line replaces its run_id's earlier lines in their place, so a requirement it adds holds",
          ([l["run_id"] for l in _sp], _sp[0]["requires"], _ws[0], any(r == "fix:no-such-supersede-key" for _, u in _ws[1] for r, _ in u)),
          (["G-x-sup", "G-x-other"], ["fix:no-such-supersede-key"], None, True))
    # an S3 line that does not list scheduler:box itself: pick adds it, so on a machine where it is not met the line waits
    s3 = {"run_id": "Q-x-S3", "task": "pkg-stagger-q", "harness": "claude-code", "model": "m", "rung": "S3", "wave": "Q",
          "requires": [], "ceiling_usd": 1, "speed": 1.0}
    _, why3 = pick("no-such-machine", {"name": "no-such-machine", "machine_type": "c3-highmem-44"}, {"waves_open": ["Q"], "box": {}},
                   [s3], {}, set(), only=["Q-x-S3"], hold_since={})
    check("an S3 line not listing scheduler:box gets it from pick", any(r == "scheduler:box" for _, u in why3 for r, _w in u), True)
    _real_declared2 = declared
    try:
        globals()["declared"] = lambda: {"skill-frozen:scheduler-playbook": {"by": "t", "utc": "u", "sha256": "82790759"},
                                         "scheduler:box": {"by": "t", "utc": "u", "label_images": {"mlsbench-kv": ["sha256:64906350fcb456d8"]}}}
        check("skill-frozen: met only when the machine's SKILL.md hash starts with the frozen sha",
              (requirement_met("skill-frozen:scheduler-playbook", {"name": "x"}, {}, {"skill:scheduler-playbook:82790759aaaa"})[0],
               requirement_met("skill-frozen:scheduler-playbook", {"name": "x"}, {}, {"skill:scheduler-playbook:35e1d410aaaa"})[0]), (True, False))
        check("scheduler-image: met only where the machine holds a KV box fake-slurm checked",
              (requirement_met("scheduler-image:mlsbench-kv", {"name": "x"}, {}, {"imageid:mlsbench-kv:64906350fcb4"})[0],
               requirement_met("scheduler-image:mlsbench-kv", {"name": "x"}, {}, {"imageid:mlsbench-kv:a15c3246322b"})[0]), (True, False))
    finally:
        globals()["declared"] = _real_declared2
    # a task's scripted_reference is an artifact too (W0-k2lp-reference, 2026-11-03 13:18Z): pkg-k2-latepole's line waits
    # where artifacts/k2-latepole/reference.py is missing, and is met with it
    _tk = task_artifacts("pkg-k2-latepole")
    _ref = next(((x, h) for x, h in _tk if x == "artifacts/k2-latepole/reference.py"), None)
    _allk = {f"artifact:{x}:{h}" for x, h in _tk} | {"artifacts-read"}
    check("task-artifacts: a task's scripted_reference is among its artifacts; the line waits without it",
          (_ref is not None and _ref[1][:16] == "4f7b163ec4811c29", requirement_met("task-artifacts:pkg-k2-latepole", {}, {}, _allk)[0],
           requirement_met("task-artifacts:pkg-k2-latepole", {}, {}, {x for x in _allk if "k2-latepole/reference.py" not in x})[0]),
          (True, True, False))
    _ta = task_artifacts("mls-kv-deploy")
    _all = {f"artifact:{x}:{h}" for x, h in _ta} | {"artifacts-read"}
    check("task-artifacts: met with every staged artifact's controller bytes; not without the KV constraints file",
          (len(_ta) >= 9, requirement_met("task-artifacts:mls-kv-deploy", {}, {}, _all)[0],
           requirement_met("task-artifacts:mls-kv-deploy", {}, {}, {x for x in _all if "env/mlsbench-kv/constraints.txt" not in x})[0]),
          (True, True, False))
    _scr = {"run_id": "G-x-haiku-L0", "wave": "Q", "task": "mls-kv-report", "harness": "claude-code", "model": "m", "screen": True,
            "requires": ["task-ready:no-such-task"]}
    _w = pick("no-such-machine", {"name": "no-such-machine"}, {"waves_open": ["Q"]}, [_scr], {}, set(), only=["G-x-haiku-L0"])[1]
    _w2 = pick("no-such-machine", {"name": "no-such-machine"}, {"waves_open": ["Q"]}, [dict(_scr, screen=False)], {}, set(), only=["G-x-haiku-L0"])[1]
    check("a screen line drops its task-ready requirement; the same line without screen: true keeps it",
          (any(r == "task-ready:no-such-task" for _, u in _w for r, _ in u), any(r == "task-ready:no-such-task" for _, u in _w2 for r, _ in u)),
          (False, True))
    _c0 = load_campaign()[0]
    _l = {"run_id": "x", "task": "pkg-stagger-q", "harness": "scripted", "policy": "packed", "speed": 1.0}
    _cmd, _ = smoke_cmd(_l, _c0, "P006-x", "c3-highmem-44")
    _tf = os.path.join(EXP, "gpu-tasks", "registry", "tasks", "pkg-stagger-q.json")
    check("box disk: the campaign's 40g unless the task file declares box.disk",
          (_cmd[_cmd.index("--disk") + 1] if _cmd and "--disk" in _cmd else None,
           (json.load(open(_tf)).get("box") or {}).get("disk")), ("40g", None))
    _met = declared()
    if all((_met.get(k) or {}).get("label_bound") for k in ("fix:gpumock-host-backdate",)):
        _kv = {"run_id": "x", "task": "mls-kv-deploy"}
        _p20 = {"run_id": "y", "task": "pkg-p20-tail-q"}
        check("label-bound fix: met where the line's own box is a declared id (ba36 KV yes, d1da KV no; ba36 packages for a packages line)",
              (label_bound("fix:gpumock-host-backdate", True, "", _kv, {"imageid:mlsbench-kv:cfd4289898a9", "imageid:mng3-packages:e5655529c6c7"})[1],
               label_bound("fix:gpumock-host-backdate", True, "", _kv, {"imageid:mlsbench-kv:64906350fcb4", "imageid:mng3-packages:e5655529c6c7"})[1],
               label_bound("fix:gpumock-host-backdate", True, "", _p20, {"imageid:mlsbench-kv:64906350fcb4", "imageid:mng3-packages:e5655529c6c7"})[1],
               label_bound("fix:gpumock-host-backdate", True, "", _p20, {"imageid:mng3-packages:09c2e35e3abb"})[1]), (True, False, True, False))
    nb_tmp = NIGHT_BAND
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as nbf:
        json.dump({"from_utc": "2026-10-23T04:30:00Z", "until_utc": "2026-10-23T07:30:00Z", "min_hours": 4.0}, nbf)
    globals()["NIGHT_BAND"] = nbf.name
    t0 = datetime.datetime(2026, 10, 23, 4, 30, tzinfo=datetime.timezone.utc).timestamp()
    check("night band: in force 04:30-07:30Z only, at min_hours", (night_band(t0 - 1), night_band(t0), night_band(t0 + 3 * 3600 - 1),
          night_band(t0 + 3 * 3600)), (None, 14400.0, 14400.0, None))
    with open(nbf.name, "w") as _f:
        json.dump({"from_utc": "2026-10-23T04:30:00Z", "force_long": ["A-p6parL-haiku-L0-bashdefault*"]}, _f)
    check("night band: force_long names lines whatever their estimate", night_force(), ["A-p6parL-haiku-L0-bashdefault*"])
    globals()["NIGHT_BAND"] = nb_tmp; os.remove(nbf.name)
    _c, _lines, _ = load_campaign()
    _cl = claims()
    _by = {l["run_id"]: l for l in _lines}
    _l = lambda rid: _by.get(rid) or {"run_id": rid, "task": "goal-choose-optimizer-q", "harness": "claude-code", "speed": 1.0}
    check("night estimate: a model line is never below its packed expectation; a scripted line is its expectation",
          (night_est(_l("G-goalq-sonnet-L0-v4"), _lines, _cl) >= expected_s(_l("G-goalq-sonnet-L0-v4")),
           night_est({"run_id": "x", "task": "pkg-p6-par-q", "harness": "scripted", "policy": "packed"}, _lines, _cl)
           == expected_s({"run_id": "x", "task": "pkg-p6-par-q", "harness": "scripted", "policy": "packed"})), (True, True))
    gtc = hashlib.sha256(open(os.path.join(EXP, "gpu-tasks", "build", "common.py"), "rb").read()).hexdigest()
    check("gt-common: met only where the machine's gpu-tasks/build/common.py is the controller's; asked for on mls-kv-deploy's scripted lines",
          (requirement_met("gt-common", {}, {}, {f"gtcommon:{gtc}"})[0], requirement_met("gt-common", {}, {}, {"gtcommon:211fc594"})[0],
           plan_imports_gt_common("mls-kv-deploy"), plan_imports_gt_common("pkg-stagger-q")), (True, False, True, False))
    sb_ok, sb_why = requirement_met("scheduler:box", {"name": "no-such-machine"}, {}, {"imageid:mng3-packages:cc646022fce5"})
    check("scheduler:box is not met on a machine whose pinned run_one is not 9f7a7e30, whatever its images", sb_ok, False)
    h9 = "9f7a7e306e0cf866598b0ffe499e025b1f7d356b490cc95426986b7ce7b0bd59"
    check("run_one 9f7a7e30 (the S rungs' exec spool) is accepted for Claude, Codex and bash-timeout lines",
          (any(h9.startswith(x) for x in CLAUDE_ARG_RUN_ONE), any(h9.startswith(x) for x in CODEX_RUN_ONE),
           any(h9.startswith(x) for x in BASH_TIMEOUT_RUN_ONE)), (True, True, True))
    check("run_one 16f9cfec... is accepted for Claude and Codex lines", (any(("16f9cfec" + "0" * 56).startswith(x) for x in CLAUDE_ARG_RUN_ONE),
                                                                       any(("16f9cfec" + "0" * 56).startswith(x) for x in CODEX_RUN_ONE)), (True, True))
    # one loop per machine: a second is refused while the first holds the lock, and
    # takes it once the first ends. A scratch lock directory, so the live loops' locks are never touched
    tg, old_g = tempfile.mkdtemp(prefix="looplock-"), os.environ.get("P006_GATE_HOME")
    os.environ["P006_GATE_HOME"] = tg
    try:
        h1, b1 = loop_locks(["selftest-m"])
        h2, b2 = loop_locks(["selftest-m"])
        by_me = bool(b2) and b2[1].split()[0] == str(os.getpid())
        for x in h1 or []:
            os.close(x)
        h3, b3 = loop_locks(["selftest-m"])
        for x in h3 or []:
            os.close(x)
    finally:
        if old_g is None:
            os.environ.pop("P006_GATE_HOME", None)
        else:
            os.environ["P006_GATE_HOME"] = old_g
        shutil.rmtree(tg, ignore_errors=True)
    check("loop lock: the first loop takes it, a second is refused naming the holder's pid, a third takes it after the first ends",
          (b1, h2, by_me, b3), (None, None, True, None))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("selftest")      # known answers for the driver's own rules, no machine needed
    sub.add_parser("tables")        # rebuild wait-style.md and polling.md once (launchd/tables.sh calls it)
    n = sub.add_parser("next"); n.add_argument("--machine")
    t = sub.add_parser("take"); t.add_argument("--machine", required=True); t.add_argument("--dry-run", action="store_true")
    t.add_argument("--wait-hours", type=float, default=0.0,
                   help="if nothing can be taken yet, re-check every 5 min for up to this long, then take the first that can")
    lp = sub.add_parser("loop"); lp.add_argument("--machines", required=True); lp.add_argument("--once", action="store_true")
    lp.add_argument("--only", default=None, help="comma-separated run_id patterns this loop may take (fnmatch)")
    lp.add_argument("--only-file", default=None, help="a file holding the --only list, re-read before every pick")
    lp.add_argument("--wait-hours", type=float, default=0.0,
                    help="when nothing can be taken, re-check every 5 min for up to this long before exiting")
    co = sub.add_parser("collect"); co.add_argument("run_id")
    pn = sub.add_parser("pin"); pn.add_argument("--from-machine", required=True); pn.add_argument("--why", required=True)
    a = ap.parse_args()
    if a.cmd == "selftest":
        sys.exit(selftest())
    if a.cmd == "tables":
        build_tables()
        return
    if a.cmd == "status":
        cl = claims()
        for k, v in sorted(cl.items()):
            print(f"{k:32s} {v['status']:8s} {v.get('machine','')}  {v.get('tag','')}")
        print(f"{len(cl)} claims")
        return
    if a.cmd in ("next", "take"):
        c, lines, bl = load_campaign()
        fl = fleet()
        machine = a.machine or next(iter(fl))
        row = fl.get(machine) or {}
        imgs = images(machine, row.get("zone")) if row.get("status") == "RUNNING" else set()
        line, why = pick(machine, row, c, lines, claims(), imgs)
        t_end = time.time() + 3600 * getattr(a, "wait_hours", 0.0)
        while a.cmd == "take" and not a.dry_run and not line and time.time() < t_end:
            log("waiting", machine=machine, first_wait="; ".join(f"{rid}: {[r for r, _ in u]}" for rid, u in why[:2]))
            time.sleep(300)
            c, lines, bl = load_campaign()
            row = fleet().get(machine) or {}
            imgs = images(machine, row.get("zone")) if row.get("status") == "RUNNING" else set()
            line, why = pick(machine, row, c, lines, claims(), imgs)
        if a.cmd == "next" or (a.cmd == "take" and a.dry_run):
            print(f"backlog {bl}, waves open {c.get('waves_open')}, machine {machine} ({row.get('machine_type')}, {row.get('status')})")
            for rid, unmet in why[:12]:
                print(f"  waits  {rid}: " + "; ".join(f"{r} ({w})" for r, w in unmet))
            print(f"  next   {line['run_id'] if line else 'nothing can be taken'}")
            if line and a.cmd == "take":
                print(run_line(line, machine, c, row, dry=True))
            return
        if not line:
            sys.exit("nothing can be taken on " + machine)
        status, err = run_line(line, machine, c, row)
        print(status or err)
        return
    if a.cmd == "loop":
        ms = [m.strip() for m in a.machines.split(",")]
        held, busy = loop_locks(ms)
        if busy:
            print(f"loop-lock: {busy[0]} already has a loop (holder: {busy[1] or '?'}); exiting 75", file=sys.stderr, flush=True)
            sys.exit(75)
        only = read_only_file(a.only_file) if a.only_file else [x.strip() for x in a.only.split(",")] if a.only else None
        crashed = []
        def run(m):
            try:
                worker(m, a.once, only, a.wait_hours, a.only_file)
            except BaseException:
                crashed.append(m)
                raise
        ts = [threading.Thread(target=run, args=(m,), daemon=False) for m in ms]
        for x in ts:
            x.start()
        for x in ts:
            x.join()
        # a worker that died on an exception ends the loop non-zero, so launchd's KeepAlive (SuccessfulExit false) restarts
        # it; its clean ends (machine-not-running, nothing-to-take) exit 0 and stay down, as before launchd
        if crashed:
            sys.exit(1)
        return
    if a.cmd == "pin":
        row = fleet().get(a.from_machine) or {}
        got = machine_hashes(a.from_machine, row.get("zone"))
        missing = [f for f in PINNED if f not in got]
        if missing:
            sys.exit(f"cannot pin: {a.from_machine} lacks {missing}")
        pins = json.load(open(PINS)) if os.path.exists(PINS) else {}
        bm = pins.setdefault("by_machine", {})
        old = (bm.get(a.from_machine) or {}).get("sha256") or pins.get("sha256") or {}
        bm[a.from_machine] = {"utc": utc(), "why": a.why, "sha256": got}
        pins.setdefault("how", "Each machine is checked against its own pin set (by_machine), else the default sha256. A pin "
                               "records what a machine runs when a series starts, so it cannot change under the series; "
                               "re-pinning is an explicit act after a deploy, with its reason.")
        tmp = PINS + ".tmp"; json.dump(pins, open(tmp, "w"), indent=1); os.replace(tmp, PINS)
        log("pinned", machine=a.from_machine, why=a.why, changed=",".join(f for f in PINNED if old.get(f) != got[f]))
        for f in PINNED:
            print(f"{got[f][:16]}  {f}")
        return
    if a.cmd == "collect":
        status, reasons = collect(a.run_id)
        set_claim(a.run_id, status=status, reasons=reasons, collected_utc=utc())
        print(status, reasons)

if __name__ == "__main__":
    main()
