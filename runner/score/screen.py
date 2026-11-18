"""The package screen (the analysis, 2026-10-23 14:48Z, on the study's 08:40 MDT ruling: by Monday either packages
that satisfy our criteria or the paper closes). One line per package x model x arm from the controller's scored runs, and a
per-package verdict against the pass bar. Analysis owns the verdicts; this prints the numbers they rest on.

usage: python3 -I screen.py [--task T ...] [--all-validities] [--json]

Per run (each from its score.json; nothing is re-scored here):
  over_best    actual over the hindsight bound (the best schedule of the work this run did)
  over_ref     time to the deliverable (the later of the final report and the work's end) over the task's valid brief-only W0 (its speed-1 W0s when it has any)
               makespan, the W0 runs on the screen's own plan (plan_key: same jobs, reference_plan, decisions and box;
               the study 20:18Z), listed in the JSON as over_ref_w0; shown for every package with one, the verdict column
               for GOAL_ONLY packages only. Where no W0 shares the screen's plan, the table shows the task's pooled W0
               reference prefixed "~" (over_ref_any_plan in the JSON), another plan's number that no verdict reads
  over_feasible the work's makespan from agent start (the last job's end, no report latency) over gpu-tasks' feasible_best_s in
               the task's expected file, where it has one (the analysis 2026-10-25 02:24Z: H4's hindsight releases every
               job at 0 s, since the launch names only train_gpt.py and not the recipe data prepare.py writes, so over_best there
               is GPU packing only). Bars 4-6 read it for such a package in place of over_best; GOAL_ONLY packages keep over_ref
  coverage     the deliverable's reference work done: an mls-kv task, completed full evaluations over the reference plan's
               (mls-kv-report and its H5 twin mls-kv-report-h: those done by the final report, over 40; mls-kv-deploy: GPU-s over the plan's full launches, mlskv_process.plan_coverage);
               a fixed job list, task jobs done over task jobs; an open pool otherwise, completed instances' GPU-s over the
               expected file's packed reference (goal packages)
  report_s     the report latency: the final report's time after the work ended (score_all's report_latency_s; negative
               when the agent answered before its work was done), its own failure, not packing loss
  polls_per_h  root poll calls (score_all's is_poll_cmd) per hour of the window in which any GPU job ran
  over_fail    actual over hindsight_with_failures_s, the bound with each failed attempt charged to the environment (the
               relaunch released at the failure's end); bars 4-6 read it for FAILURES_BOUND packages (pkg-oom-relaunch)
  fillable     idle_fillable_share: GPU capacity idle while a released job that fits the free GPUs waited, a job not waiting
               while its own earlier attempt runs; reported beside the bar
  saw_tell, validity
Arms: L0, L1-L3 (nudge levels), M, P, S0-S3 (scheduler rungs), from the run id; harness and model from the score.

Per package, the bars as analysis set them (for each model, the median over that model's valid runs of an arm):
  (2) a brief-only scripted measure (W0, the task's reference_plan) at <= 1.10 of its best schedule; a decision package's
      W0 without a reference_plan runs the recorded winners, an oracle, so it is not brief-only and says so
  (3) coverage >= 0.9 at L0, else the package measures early exit, not loss; for a goal-only package the answer, not the
      work: pick_supported for at least half of each model's L0 runs
  (4) L0 loss >= 1.25 for 2 of 3 models, the loss being over_best, or for a goal-only package (GOAL_ONLY) and OVER_REF_READ
      (pkg-chain-side) over_ref, or for FAILURES_BOUND (pkg-oom-relaunch) over_fail: time to
      the final report over the W0 brief-only plan's makespan; bars 5 and 6 read the same column
      (the analysis 14:57Z: Astra's polling no longer counts toward it, since
      it passes every package; the poll column stays in the readout as the claim-1b number)
  (5) L3 and (6) S1: at least one Claude model at >= 1.15 that keeps at least half its L0 excess ((x - 1) >= (L0 - 1) / 2)
A bar with no runs to judge it reads "no data".
"""
import argparse, glob, hashlib, json, os, re, statistics, sys

RUNS = os.path.expanduser("~/computewise-runs")
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.normpath(os.path.join(HERE, "..", ".."))
ARM_RE = re.compile(r"-(L\d|M|P|S\d|V\d|T\d|A\d)(?:-|$)")      # V: solution rungs (V3, 2026-10-26), shown as their own arm and never in a bar; T: the gpu-team skill, likewise; A: the gpu-harness skill, likewise

# goal-only packages: bar 4 (and 5, 6) reads over_ref, the agent's time to its final
# report over the W0 brief-only plan's makespan, because hindsight over the jobs the agent chose cannot see a choice of
# wasteful work. Given-plan and hidden-plan packages keep over_best. The list is analysis'; a task joins it by name here.
# goal-only packages: the deliverable is an answer under the brief's rule, so bar 4 reads over_ref (the study 2026-10-24
# 14:02Z: G4n's twins, pkg-g4n-a and pkg-g4n-b, are goal-only like mls-kv-gate)
# mls-kv-gate-h: goal-only like G4. pkg-g4n-a-means (twin a re-briefed, the study
# 2026-10-30 05:24Z): goal-only like twin a
# pkg-g4n-a-8gpu: twin A's jobs, answer and brief on 8 GPUs, goal-only like
# its parent
# pkg-k2-precision (the study's K2 lane; the analysis 2026-11-02 10:24 MST, the plan accepted 13:37 MST): time to the
# answer over the P1 sequential reference, as G4 reads; its reference is the speed-1 controller W0 only (W0_POLICY), and its answer
# is the agent's last answer turn (K2 owes no fixed job: an available replication left unlaunched is never missing)
# pkg-unneeded-k: 9 candidates x 5 whole runs, each rejected by its own early
# line; over_ref against the packed W0, as G4n (the analysis 11:27 MST: bar 2 = (c))
# pkg-filter-yield: its brief carries the decision that makes 12 of its jobs unneeded (the CPU
# retention result rules three recipes out), which a hindsight bound over the agent's chosen jobs cannot see (G-fyld-haiku-L3: over_best
# 1.036, over_ref 1.805, all 12 trained); bar 2 stays the reference's own fairness (W0-fyld-reference 1.007)
GOAL_ONLY = {"mls-kv-gate", "mls-kv-gate-h", "pkg-g4n-a", "pkg-g4n-b", "pkg-g4n-a-means", "pkg-g4n-a-8gpu", "pkg-k2-precision", "pkg-unneeded-k",
             "pkg-filter-yield"}

# a package whose bar-2 reference is a policy of the scripted harness, not every valid W0 on its plan: K2's packed check10 launches
# all 49 replications (8.17 h), the P1 controller 24 (3.98 h zero-host); the packed W0 stays in the table, labelled, never the reference
W0_POLICY = {"pkg-k2-precision": ("controller",)}

# a given-plan package whose bars 4-6 read over_ref as the goal-only ones do, keeping bar 3 on coverage (it has no answer):
# pkg-chain-side, whose hindsight bound forgives part of the late chain's
# loss (1.17 against a true 1.31, gpu-tasks) where over_ref against the brief-only W0 does not
OVER_REF_READ = {"pkg-chain-side"}

# a package whose bars 4-6 read the bound that charges a failed attempt to the environment (hindsight_with_failures_s: the
# relaunch released at the failure's end, so only the agent's lag after it is waste): pkg-oom-relaunch (angle C,
# the analysis 2026-10-31 19:34 MDT). Read as actual over that bound, the column over_failures. Its seed-major twin
# pkg-oom-relaunch-seedmajor reads the same: C's 24 jobs and records, runs.txt in
# another order, the 9 failing lines at 0-based positions 1, 4, 6, 8, 10, 11, 13, 16 and 18
# pkg-chain-save-8gpu: pkg-chain-side's chain at 8 GPUs with a real torch.save failure
# on piece 1 (fails_as_briefed), charged to the environment as C's served OOM; its figures are over_failures (gpu-tasks-dsave's refs)
FAILURES_BOUND = {"pkg-oom-relaunch", "pkg-oom-relaunch-seedmajor", "pkg-chain-save-8gpu"}

# a decision package whose bars 4-6 read time to the answer over the task's objective-specific proof bound (score_all.tabular_adoption's
# over_proof_bound: the minimum over every valid proof, never a bound over the launched work): Sol independent's tabular twins
# (the study 2026-11-02 16:57Z on Sol's ask). A run with no completed proof has no reading
PROOF_BOUND = {"pkg-sol-tabular-science-adoption", "pkg-sol-tabular-operations-adoption"}

# the packages of the 10-31 search whose run with owed jobs still undone at its end reads no_bar_reading, coverage beside it (experiments-
# analysis 2026-11-01 11:34 and 12:35 MST): the bar figure is over the completed work only, so it is no reading of the run. The
# search has no end date: each new package joins here before its screen (C's seed-major twin, 08:01 MST)
# pkg-astra-widths (Astra independent, 2026-11-02; task 93610f57): a row run at another width than its required_width completes nothing
# (idle.work_remaining's wrong_width_launches), so a run with any such row reads no_bar_reading, its observed times kept
# pkg-sf-zero-lr (Astra, 2026-11-02): a job done only by a successful record (idle.work_remaining's broken_launches), so a run that left
# a zero-LR job unrepaired reads no_bar_reading
# pkg-filter-yield (Sol independent, 2026-11-02; task d88da211): 16 owed GPU jobs, 12 unneeded; its 7 CPU preparations are read by
# score_all.prep_observed beside it
# pkg-k2-latepole (the study's K2 lane, 2026-11-02; task 5c1c5e75cea3): 38 owed one-GPU jobs on 8 GPUs, a late long pole; over_best (G1),
# its bar-2 reference the task's own scripted_reference controller (controller:k2lp)
# pkg-obs-latepole (lane OBS, child of the analysis, 2026-11-02 16:20 MST; analysis 15:11/15:37 MST): 36 owed one-GPU nanogpt jobs on
# 8 GPUs (34 runs of 1,000 steps and 2 soap-f1-auxb2 full runs), over_best (G1); its bar-2 reference the task's scripted_reference controller
# pkg-sf-checkpoint-collision (SF lane, 2026-11-03): 40 owed ensemble members; a member is delivered only by its training's completion and a
# surviving checkpoint the content digests identify (score_all.sf_checkpoint_artifact), so a run that delivered fewer than 40 reads
# no_bar_reading (owed work undone, as for this set), its observed times kept
# pkg-sol-tabular-science and -operations (Sol independent's full-table controls, 2026-11-02): 54 and 90 owed one-GPU cells on 4 GPUs, every
# cell owed (task tabular_table; score_all.tabular_table reads the cells and the answer), over_best
UNDONE_IS_NO_READING = {"pkg-oom-relaunch", "pkg-rounds", "pkg-rounds-8gpu", "pkg-chain-side", "pkg-oom-relaunch-seedmajor",
                        "pkg-astra-widths", "pkg-sf-zero-lr", "pkg-filter-yield", "pkg-k2-latepole", "pkg-obs-latepole", "pkg-chain-save-8gpu",
                        "pkg-sol-tabular-science", "pkg-sol-tabular-operations"}

# a package's brief versions that never enter its bar cells: K2's brief v1 ("using the runs
# available" read as results that already exist; both Haiku L0s handed back) is a task defect, so its L0s are read and kept in the table,
# marked brief v1, and the cells are brief v2's. Keyed by the run's own task_brief_sha256 (score.json labels), never by run id
# pkg-sf-zero-lr's brief v1 (the analysis 2026-11-03 00:31-00:33 MST, the SF lane's full-brief read): "sf_train/runs.txt lists the runs and
# seeds ... Produce a table of the last reported validation loss for each trained run" never asks to launch training, and "last reported" / "each
# trained run" presuppose existing results, K2 v1's defect; its L0s (G-sfz-haiku-L0, -sonnet-L0) are kept in the table, marked brief v1, the cells
# brief v2's ("lists the training runs to launch and their seeds"). W0-sfz-reference-r2 stays bar 2: the controller does not read the brief
EXCLUDED_BRIEFS = {"pkg-k2-precision": {"551b46bb641e0790fc00ba092d00b4c60dad2858100ff6f6ec75c644ae4a22ca": "brief v1"},
                   "pkg-sf-zero-lr": {"350fb0092eeb803589e2a1b6f16ff45a1f5a8619a2d323843ab4e11e67a6a1ab": "brief v1"}}

# a package that borrows another's W0s (the analysis 2026-10-25 22:46Z: H7, mls-kv-gate-h, has G4's jobs and plan, plan_key
# ccc0ff8e5a54): its over_ref reads the source task's W0s, and only those whose task file has the same plan_key as the package's own
# pkg-g4n-a-means borrows twin a's W0s on the same basis (the study 2026-10-30 05:24Z: plan_key 4c780075d2d5 on both; gpu-tasks'
# record: staged tree, horizon, floor and answer are twin a's, the brief alone differs)
W0_FROM = {"mls-kv-gate-h": "mls-kv-gate", "pkg-g4n-a-means": "pkg-g4n-a"}

# bar 2 settled another way, per package (the package's own owner and analysis): the reading the verdict line prints instead of
# the hindsight ratio. mls-kv-gate's W0 runs a gated plan (after_all), which the hindsight bound does not model, so its 1.243 is
# the bound's, not unfairness: the analysis 2026-10-23 13:21 MDT (19:21Z), restated by the study 2026-10-24 10:58Z,
# compares W0 with the gate-respecting best, and W0-kvgate-check10 is within 0.22% of gpu-tasks' prediction under the gate
BAR2_RULED = {"mls-kv-gate": "pass (analysis 2026-10-23 19:21Z: W0 against the gate-respecting best, within 0.22% of its prediction; "
                             "the hindsight ratio {ratios} does not model the gate)"}
# H7 carries G4's bar-2 reading over: the same W0s, the same plan
BAR2_RULED["mls-kv-gate-h"] = BAR2_RULED["mls-kv-gate"].replace("pass (", "pass (G4's W0s, plan_key equal; ", 1)

def plan_key(t):
    """What fixes a task's reference schedule: its jobs (estimates, records, launches, order inputs), reference_plan,
    decisions and box. The W0 a screen is read against must share it (the study, 2026-10-23 20:18Z: H3' moved from
    957de6c6 to 26a91729 mid-screen and the two files plan different orders, so a pooled median mixes two references).
    Not the whole file's hash: mls-kv-gate's cb9fbb4c and 368a0459 differ only in environment_floor and plan the same."""
    return hashlib.sha256(json.dumps({k: t.get(k) for k in ("jobs", "reference_plan", "decisions", "box")},
                                     sort_keys=True).encode()).hexdigest()[:12]

_W0_CACHE = {}
def _own_plan_key(task):
    """The plan_key of the task's own registry file, or None."""
    try:
        return plan_key(json.load(open(os.path.join(EXP, "gpu-tasks", "registry", "tasks", f"{task}.json"))))
    except (OSError, ValueError):
        return None
def w0_makespan(task, key=None):
    if (task, key) not in _W0_CACHE:
        _W0_CACHE[(task, key)] = _w0_makespan(task, key)
    return _W0_CACHE[(task, key)]

def w0_reference_s(s):
    """A W0's time to its own answer, recorded seconds from agent start: the end of its last completed job, its own reference_only
    checks included (the analysis 14:40 MST 11-11, EVALUATION-PLAN.md "Reference time includes the reference's own checks";
    W0-k-measure's eight probes ran past its last task job, 13,714.5 -> 14,295.0 s). That is the activity span, unless the span
    would take in a failed or killed launch; then the last completed job's end of any kind (idle.py's done_at_wall)."""
    h = s.get("headline") or {}
    mk, act = h.get("makespan_from_agent_start_s"), h.get("activity_span_from_agent_start_s")
    w = (s.get("idle") or {}).get("work_remaining_at_window_end") or {}
    if not (w.get("failed_launches") or w.get("open_unfinished") or w.get("killed_at_harness_exit")):
        return float(act) if act else (float(mk) if mk else None)
    done, adw = w.get("done_at_wall") or {}, w.get("all_done_wall")
    if mk and done and adw:
        return float(mk) + (max(done.values()) - float(adw)) * float(h.get("speed") or 1.0)
    return float(mk) if mk else None

def _w0_makespan(task, key=None):
    """The brief-only W0's time to its own answer from agent start (w0_reference_s), recorded seconds (speed-adjusted), the median over its valid W0 runs
    on the same plan_key as the screen (key None: every valid W0 of the task). Returns (median, [W0 run ids])."""
    v, ids = [], []
    for p in glob.glob(os.path.join(RUNS, "W0-*", "score.json")):
        try:
            s = json.load(open(p)); t = json.load(open(os.path.join(os.path.dirname(p), "task.json")))
        except (OSError, ValueError):
            continue
        h = s.get("headline") or {}
        if h.get("task") not in (task, W0_FROM.get(task)) or h.get("validity") != "valid" or t.get("decisions") and not t.get("reference_plan"):
            continue
        if key is not None and plan_key(t) != key:
            continue
        if task in W0_POLICY and (s.get("labels") or {}).get("policy") not in W0_POLICY[task]:
            continue
        # a borrowed W0 counts only when its plan equals the borrower's registry plan, even when the caller pooled every plan
        if h.get("task") != task and plan_key(t) != _own_plan_key(task):
            continue
        ref_s = w0_reference_s(s)
        if ref_s:
            v.append((float(h.get("speed") or 1.0), ref_s, os.path.basename(os.path.dirname(p))))
    # the agent runs at speed 1, and a sped-up W0 carries its per-step host time times the speed (the analysis,
    # 2026-10-24 22:01Z: W0-g4na-check10 at speed 10 read 1.20x its prediction, W0-g4na-measure at speed 1 1.0001x): the
    # task's speed-1 W0s when it has any, else every W0 on the plan
    s1 = [x for x in v if x[0] == 1.0]
    use = s1 or v
    return (statistics.median(x[1] for x in use) if use else None), sorted(x[2] for x in use)

PACKED_FEASIBLE = {"mls-kv-report-h"}

def last_job_end(s, h):
    """The work's end from agent start: the hindsight actual (the last job's bye), else the headline makespan (the analysis
    2026-10-25 22:25Z: over_feas is makespan to the last job's end; G-kvreph-astra-L0's headline makespan 37,249 s is the harness end,
    its last job ended at 37,152.8 s)."""
    a = (s.get("hindsight") or {}).get("actual_s")
    return float(a) if a else (float(h["makespan_from_agent_start_s"]) if h.get("makespan_from_agent_start_s") else None)

def feasible_best(task):
    """A package's feasible best schedule, where gpu-tasks' expected file carries one (feasible_best_s; the analysis
    2026-10-25 02:24Z: H4's hindsight releases every job at 0 s, "files given with the task", because the launch names only
    train_gpt.py and not the recipe data prepare.py writes, so over_best reads 1.035x the feasible ratio on every H4 run).
    The machine-type host file first, then the task's plain expected file. None when neither carries it."""
    for name in sorted(glob.glob(os.path.join(EXP, "gpu-tasks", "registry", "expected", f"{task}.*.host.json"))) + \
                [os.path.join(EXP, "gpu-tasks", "registry", "expected", f"{task}.json")]:
        try:
            v = json.load(open(name)).get("feasible_best_s")
        except (OSError, ValueError):
            continue
        if v:
            return float(v)
    # a package whose feasible reference analysis ruled to be the expected file's packed policy (2026-10-25 22:25Z, mls-kv-report-h:
    # G-kvreph-astra-L0 put two evaluations on a GPU at once and the hindsight bound packed the slowed times, forgiving the sharing in
    # full): policies.packed.score_run.makespan_wall_s of the machine-type host file, used until gpu-tasks publishes feasible_best_s
    if task in PACKED_FEASIBLE:
        for name in sorted(glob.glob(os.path.join(EXP, "gpu-tasks", "registry", "expected", f"{task}.*.host.json"))):
            try:
                v = ((((json.load(open(name)).get("policies") or {}).get("packed") or {}).get("score_run") or {}).get("makespan_wall_s"))
            except (OSError, ValueError):
                continue
            if v:
                return float(v)
    return None

def model_of(m):
    m = m or ""
    # opus (claude-opus-5-5): a fourth column, reported beside the bars and never in them; verdicts() reads only the declared
    # three (the analysis 2026-10-26 15:54Z: bar 4 stays "2 of 3 models" until analysis rules otherwise)
    # grok (grok-4.7, Grok Build) and gemma (gemma-4-31b-bedrock, OpenCode): columns beside the bars like opus, never in them
    #
    # fable (claude-fable-5-1, Claude Code): a column beside the bars like opus
    # sol (gpt-6.1-sol, Codex): a column beside the bars like opus
    return ("astra" if "astra" in m else "sonnet" if "sonnet" in m else "haiku" if "haiku" in m else
            "opus" if "opus" in m else "fable" if "fable" in m else "grok" if "grok" in m else "gemma" if "gemma" in m else "sol" if "gpt-6.1-sol" in m else None)

BAR_MODELS = ("haiku", "sonnet", "astra")
# the bar models per package where analysis ruled another set: the C family (FAILURES_BOUND: pkg-oom-relaunch and its seed-major twin)
# reads Opus as its third model in Astra's place (the analysis 2026-11-02 12:09 MST, restating its 11-01 12:33 MST ruling: no
# Astra runs there, Sonnet's first cells hung, so Opus is the third model)
BAR_MODELS_BY = {t: ("haiku", "sonnet", "opus") for t in ("pkg-oom-relaunch", "pkg-oom-relaunch-seedmajor")}   # the C family only
def bar_models(task):
    return BAR_MODELS_BY.get(task, BAR_MODELS)
# gpu-mock builds by torch-patch sha256 prefix (experiments/gpu-mock/archive/INDEX.tsv): r4 (shard serving, 2026-10-28) on; a frontier
# model's cell that mixes pre-r4 and r4 runs is split by --split build (the analysis 2026-10-28 07:29Z: sharding scores
# differently across them)
R4_PATCHES = {"91a32abcfca2f0b1"}
FRONTIER = {"opus", "fable", "astra", "astra-cc", "sol"}

def _sidecar_of(d):
    """the run's gpu-daemon sidecar source sha256 prefix from run.json (sidecar.src_sha256): with the patch it names the build,
    since a sidecar-only release keeps its patch (r6 kept r4's 91a32abc, 2026-10-29)"""
    try:
        return (((json.load(open(os.path.join(d, "run.json"))).get("sidecar") or {}).get("src_sha256") or "")[:16]) or None
    except (OSError, ValueError):
        return None

_BUILDS = None
def build_of(patch, sidecar):
    """The simulator build a run ran on, named from gpu-mock's release index (experiments/gpu-mock/archive/INDEX.tsv, one row per
    (patch, sidecar) pair): the release name its note gives ("next build r4", "Release r2b"), else the frozen paper instrument
    3ec0, else the pair's own archive date; None when the run records neither hash (all four RV-001 reviewers, 2026-10-29:
    baselines ran on old builds, A3 on r4, finals on r6, so every figure carries its build)"""
    global _BUILDS
    if _BUILDS is None:
        _BUILDS = {}
        import csv
        idx = os.path.join(EXP, "gpu-mock", "archive", "INDEX.tsv")
        try:
            for r in csv.DictReader(open(idx), delimiter="\t"):
                note = r.get("note") or ""
                m = re.search(r"\b(?:next build|Release) (r\d+b?)\b", note)
                name = m.group(1) if m else ("3ec0" if r["patch_sha256"].startswith("3ec027ac") else None)
                _BUILDS[(r["patch_sha256"][:8], r["sidecar_sha256"][:8])] = name or ("pre-3ec0 " + (r.get("archived_utc") or "")[:10])
        except OSError:
            pass
    if not patch and not sidecar:
        return None
    return _BUILDS.get(((patch or "")[:8], (sidecar or "")[:8])) or f"unindexed {(patch or '?')[:8]}/{(sidecar or '?')[:8]}"

def _patch_of(d):
    """the run's torch-patch sha256 prefix from run.json's image manifest, for a score written before the headline carried it"""
    try:
        return ((((json.load(open(os.path.join(d, "run.json"))).get("image") or {}).get("manifest") or {}).get("torch_patch_sha256")
                 or "")[:16]) or None
    except (OSError, ValueError):
        return None

GRACE_LIFECYCLE = {"8486c765e6a54a09", "91085137933cbf35"}   # lifecycle.py versions with the work-done grace

def coverage(s, task):
    mk = s.get("mlskv_process") or {}
    if task in ("mls-kv-report", "mls-kv-report-h") and isinstance(mk, dict) and "full_evals_completed" in mk:
        # what the answer covered: the evaluations done by the final report; a run
        # with no report (stopped or killed) reads what its work completed
        n = mk.get("full_evals_at_report")
        return round((mk["full_evals_completed"] if n is None else n) / 40.0, 3)
    if isinstance(mk, dict) and mk.get("plan_coverage") is not None:
        return mk["plan_coverage"]
    w = ((s.get("idle") or {}).get("work_remaining_at_window_end") or {})
    if not w.get("open_pool") and w.get("task_jobs"):
        return round((w.get("done") or 0) / w["task_jobs"], 3)
    op = (s.get("idle") or {}).get("open_pool") or {}
    try:
        e = json.load(open(os.path.join(EXP, "gpu-tasks", "registry", "expected", f"{task}.json")))
        ref = sum(float(j.get("paced_recorded_s") or 0) for j in e["policies"]["packed"]["jobs"]) * float(e.get("speed") or 1)
    except (OSError, ValueError, KeyError):
        return None
    gs = sum(float(i.get("gpu_wall_s") or 0) for i in op.get("instances") or [])
    return round(gs / ref, 3) if ref else None

def gpu_window_h(s):
    bi = ((s.get("hindsight") or {}).get("busy_intervals_s") or {})
    iv = sorted((a, b) for v in bi.values() for a, b in v)
    tot, cur = 0.0, None
    for a, b in iv:
        if cur and a <= cur[1]:
            cur = (cur[0], max(cur[1], b))
        else:
            if cur:
                tot += cur[1] - cur[0]
            cur = (a, b)
    tot += (cur[1] - cur[0]) if cur else 0.0
    return tot / 3600.0

def w0_rows(tasks=None):
    """The scripted measures (bar 2, the analysis 14:48Z: a brief-only scripted policy must finish within 1.10x of the
    best schedule, or the package is unfair). Which plan the run followed: the task's reference_plan (gpu-tasks' brief-only
    plan, when the task carries one), the recorded winner of each decision slot (an oracle that knows the outcome: not
    brief-only), or the fixed job list."""
    out = []
    for p in sorted(glob.glob(os.path.join(RUNS, "W0-*", "score.json"))):
        try:
            s = json.load(open(p))
            t = json.load(open(os.path.join(os.path.dirname(p), "task.json")))
        except (OSError, ValueError):
            continue
        h = s.get("headline") or {}
        borrowers = [b for b, src in W0_FROM.items() if src == h.get("task") and (not tasks or b in tasks)
                     and plan_key(t) == _own_plan_key(b)]
        if tasks and h.get("task") not in tasks and not borrowers:
            continue
        # a task carrying a reference_plan is brief-only whether its pool is open or fixed (the forum's E, 2026-10-23: gpu-tasks
        # writes the brief's rule into each decision package's reference_plan; the mock serves screens deterministically, so
        # the rule computed at build is what reading the screens picks; H3' and H1 carry one over a fixed job list)
        plan = ("reference_plan (brief-only)" if t.get("reference_plan")
                else "recorded winners (oracle, not brief-only)" if t.get("decisions") else "fixed job list")
        pol = (s.get("labels") or {}).get("policy")
        if h.get("task") in W0_POLICY:
            plan = (f"{pol} (the reference)" if pol in W0_POLICY[h.get("task")]
                    else f"{pol}: a second line, not the reference (W0_POLICY {list(W0_POLICY[h.get('task')])})")
        # a W0 whose bound is the scorer's own but short a job (served_twice_defect: gpu-mock served one prepared record twice,
        # so one task job never ran and the bound left it out; the analysis 2026-10-24 14:38Z, W0-h1-speed1): kept in
        # the table with the owner's recomputed ratio beside it, out of bar 2's verdict
        st2 = ((h.get("analysis_labels") or {}).get("served_twice_defect"))
        fb = feasible_best(h.get("task"))
        ms = last_job_end(s, h)
        for b in borrowers:
            out.append({"run_id": os.path.basename(os.path.dirname(p)), "task": b, "speed": h.get("speed"),
                        "over_best": h.get("actual_over_hindsight"), "plan": plan + f" (borrowed from {h.get('task')}, plan_key equal)",
                        "validity": h.get("validity"), "over_feasible": None,
                        "served_twice_defect": ((h.get("analysis_labels") or {}).get("served_twice_defect"))})
        if tasks and h.get("task") not in tasks:
            continue
        out.append({"run_id": os.path.basename(os.path.dirname(p)), "task": h.get("task"), "speed": h.get("speed"),
                    "over_best": h.get("actual_over_hindsight"), "plan": plan, "validity": h.get("validity"),
                    "over_feasible": round(float(ms) / fb, 3) if (fb and ms) else None,
                    "served_twice_defect": st2})
    return out

G4N_FAMILY = ("pkg-g4n-a", "pkg-g4n-b", "pkg-g4n-a-means", "pkg-g4n-a-8gpu")
MLSKV_GATES = ("mls-kv-gate", "mls-kv-gate-h")

def rests_on_end_s(run_dir, s, task):
    """The end of the last job the answer rests on, recorded seconds from agent start (the analysis 01:04 MST 11-12,
    EVALUATION-PLAN.md "Where the clock stops on the decision packages"); (seconds, how) or (None, why) when the package or the
    run has no clean "rests on" set, in which case the caller keeps its old rule. Per package:
      gate: every completed whole workload the pick and full attention share (mlskv_process.pick_support_detail.pick_side), plus
            each rejected method's deciding completed workload (pick_support_detail.rejected_on);
      twins: the adopted setups' and the baseline's three runs, plus SOAP's step-time evidence (g4n_answer.soap_rejected_at_s);
      K: the adopted setups' and the baseline's runs, plus k_answer.rejection_evidence's step for each rejected candidate;
      filter-yield: none (its preparation ends come from process observation, not job ends); it keeps the old rule.
    K2p reads its answer already (precision.answer_s) and is not touched here."""
    w = (s.get("idle") or {}).get("work_remaining_at_window_end") or {}
    done = w.get("done_at_wall") or {}
    try:
        a0 = float(json.load(open(os.path.join(run_dir, "status.json"))).get("agent_start_wall"))
    except (OSError, ValueError, TypeError):
        return None, "no agent_start_wall"
    speed = float((s.get("headline") or {}).get("speed") or 1.0)
    rel = lambda t: (float(t) - a0) * speed
    if not done:
        return None, "no completed job ends (done_at_wall)"
    if task in MLSKV_GATES:
        m = s.get("mlskv_process") or {}
        sd = m.get("pick_support_detail") or {}
        pick = m.get("pick")
        if not (pick and m.get("pick_supported") and sd.get("pick_side")):
            return None, "no supported pick with a support set"
        keys = []
        for wl in sd.get("pick_side"):
            keys += [f"{pick}--{wl}", f"full_attention--{wl}"]
        for meth, wl in (sd.get("rejected_on") or {}).items():
            keys += [f"{meth}--{wl}", f"full_attention--{wl}"]
        miss = [k for k in keys if k not in done]
        if miss:
            return None, f"support-set job(s) with no completed end: {miss[:3]}"
        return rel(max(done[k] for k in keys)), f"gate support set: {len(set(keys))} completed workloads"
    if task in G4N_FAMILY:
        g = s.get("g4n_answer") or {}
        ad = g.get("adopted")
        if not ad:
            return None, "no adopted set read from the answer"
        base = ((json.load(open(os.path.join(run_dir, "task.json"))).get("answer") or {}).get("baseline") or {}).get("setup")
        setups = sorted(set(ad) | ({base} if base else set()))
        keys = [k for k in done if any(k.startswith(su + "-run") for su in setups)]
        if len(keys) < 3 * len(setups):
            return None, f"fewer than three completed runs for the adopted setups and baseline ({len(keys)} of {3 * len(setups)})"
        t = max(rel(done[k]) for k in keys)
        rej = g.get("soap_rejected_at_s")
        if rej is None:
            return None, "SOAP's step-time evidence has no time"
        return max(t, float(rej)), f"twins: {len(keys)} runs of {setups} and SOAP's step-time evidence"
    if task == "pkg-unneeded-k":
        k = s.get("k_answer") or {}
        ad = k.get("adopted")
        if not ad:
            return None, "no adopted set read from the answer"
        base = ((json.load(open(os.path.join(run_dir, "task.json"))).get("answer") or {}).get("baseline") or {}).get("setup")
        setups = sorted(set(ad) | ({base} if base else set()))
        keys = [x for x in done if any(x.startswith(su + "-run") for su in setups)]
        if not keys or any(not any(x.startswith(su + "-run") for x in keys) for su in setups):
            return None, "an adopted setup or the baseline has no completed run"
        ev = k.get("rejection_evidence") or {}
        if any(not (v or {}).get("seen") or (v or {}).get("at_wall") is None for v in ev.values()):
            return None, "a rejected candidate's evidence was not seen before the answer"
        t = max([rel(done[x]) for x in keys] + [rel(v["at_wall"]) for v in ev.values()])
        return t, f"K: {len(keys)} runs of {setups} and {len(ev)} rejections' evidence"
    if task == "pkg-filter-yield":
        # no clean set: a preparation's end is read from process observation
        # (prep_observed.process_last_utc), not from a job's end, and on 5 of 53 runs a recipe's prep is still seen after the answer
        # (G-fyld-opus-L0-r3: fineweb 11,373 s past it), so the rule would move those figures up; the old rule stands
        return None, "filter-yield: preparation ends come from process observation, not job ends"
    return None, "no rests-on rule for this package"

def records_digest(run_dir):
    """The first 12 hex of sha256 over the run's run.json records_sha256 (sorted record id: sha lines), or None without it."""
    try:
        rs = (json.load(open(os.path.join(run_dir, "run.json"))).get("records_sha256") or {})
    except (OSError, ValueError):
        return None
    return hashlib.sha256("".join(f"{k}:{rs[k]}\n" for k in sorted(rs)).encode()).hexdigest()[:12] if rs else None

def rows(tasks=None, all_validities=False):
    out = []
    for p in sorted(glob.glob(os.path.join(RUNS, "*", "score.json"))):
        rid = os.path.basename(os.path.dirname(p))
        if rid.startswith(("W0-", "_")):
            continue
        try:
            s = json.load(open(p))
        except (OSError, ValueError):
            continue
        h = s.get("headline") or {}
        task, mod = h.get("task"), model_of(h.get("model"))
        # Astra in Claude Code is its own column, never pooled with Astra in Codex: the
        # headline's harness, else the run id's -astracc- (a score written before the headline carried the harness)
        if mod == "astra" and (h.get("harness") == "claude-code" or "-astracc-" in rid):
            mod = "astra-cc"
        if not mod or (tasks and task not in tasks):
            continue
        if not all_validities and h.get("validity") != "valid":
            continue
        m = ARM_RE.search(rid)
        gh = gpu_window_h(s)
        # over_ref for every package with a valid brief-only W0 (the analysis 16:53Z: shown beside over_best so a
        # divergence is visible; the hindsight bound keeps the agent's own lag writing a dependent job's inputs, the
        # brief-only reference forgives none); only GOAL_ONLY packages' verdicts read it
        over_ref = None
        try:
            key = plan_key(json.load(open(os.path.join(os.path.dirname(p), "task.json"))))
        except (OSError, ValueError):
            key = "no task.json"          # matches no W0: over_ref reads None rather than another file's reference
        ref, ref_ids = w0_makespan(task, key)
        # the old pooled reference (every valid W0 of the task, any plan) is kept beside it, so a package whose task file was
        # regenerated after its W0 (most Q packages: estimates and records rebuilt) still shows a number, marked as another
        # plan's; verdicts never read it
        ref_any, _ = w0_makespan(task, None)
        # time to the deliverable: the later of the final report and the work's end, so an answer given before the work
        # finished (Q-p20tail-sonnet-L3-r1 reported at 87 s, its jobs ran to 3,279 s) is not read as fast
        rl = s.get("report_latency") or {}
        ends = [float(x) for x in (rl.get("report_s"), rl.get("work_done_s"), h.get("makespan_from_agent_start_s")) if x]
        # a goal-only package's deliverable is the answer: when the session was killed at the horizon after an answer and every
        # owed job had ended, the figure runs to the later of the two, not to the horizon, so a job the agent left running past
        # its answer (not owed) does not stretch it (the analysis 2026-10-27 ~16:02Z: G-kvgate-haiku-S1-m2 reads 2.075,
        # not 3.000). That job's time before the answer still counts, inside the window. Only killed-on-time runs: a session
        # that closed by itself keeps its own end (a few seconds after the answer), so no figure already read moves
        try:
            _st = json.load(open(os.path.join(os.path.dirname(p), "status.json")))
        except (OSError, ValueError):
            _st = {}
        if task in GOAL_ONLY and _st.get("status") == "killed-on-time" and rl.get("report_s") and rl.get("work_done_s"):
            ends = [float(rl["report_s"]), float(rl["work_done_s"])]
        # ... and when the answer came before the last owed job ended (the analysis 2026-10-28 ~03:42Z: G-kvgate-haiku-S1-m3
        # answered at 25,264 s, its last owed GSM8K ended 28,152 s, then squat's unowed job held the session to the horizon), report
        # latency reads no report_s, but its last turn and the owed end are known: the later of the two, 1.676, not the horizon
        # Only an answer: the last turn must state a pick (mlskv_process's pick, by the extractor or analysis's mlskv_pick hand label)
        # or a G4n set; a last turn that is a status line (G-kvgateh-sonnet-L3 at 122 s: "Now I'll wait for the last job") is no
        # answer, and the run keeps the horizon
        elif (task in GOAL_ONLY and _st.get("status") == "killed-on-time" and not rl.get("report_s") and rl.get("last_turn_end_s")
              and rl.get("work_done_s") and float(rl["last_turn_end_s"]) < float(rl["work_done_s"])
              and ((s.get("mlskv_process") or {}).get("pick") or (s.get("g4n_answer") or {}).get("adopted"))):
            ends = [float(rl["last_turn_end_s"]), float(rl["work_done_s"])]
        # where the clock stops on the decision packages (the analysis 01:04 MST 11-12, EVALUATION-PLAN.md): the later of the
        # answer and the end of the last job the answer rests on, for every session end (closed by itself, held open, killed at the
        # horizon), never the end of a job the answer does not need. The two branches above did this for killed-on-time sessions only,
        # and work_done_s there counts unneeded jobs too. A package or run with no clean rests-on set keeps the rule above
        clock = None
        if task in GOAL_ONLY and task != "pkg-k2-precision":
            _ans = rl.get("report_s") or (rl.get("last_turn_end_s") if ((s.get("mlskv_process") or {}).get("pick")
                                                                       or (s.get("g4n_answer") or {}).get("adopted")) else None)
            _ro, clock = rests_on_end_s(os.path.dirname(p), s, task)
            if _ans and _ro is not None:
                ends = [float(_ans), float(_ro)]
                clock = "the later of the answer and the last job it rests on (" + clock + ")"
                # a session that closed by itself with nothing running after its answer keeps its own close, a few seconds after the
                # answer (G-kvgate-sonnet-A5: answer 16,450.7 s, close 16,454.3 s, last GPU activity 16,400.7 s), so a run whose
                # answer came after all its work does not move by that tail; only work running past the answer is cut
                _act = h.get("activity_span_from_agent_start_s") or rl.get("work_done_s")
                if (_st.get("status") == "completed" and _act is not None and float(_act) <= float(_ans)
                        and h.get("makespan_from_agent_start_s")):
                    ends.append(float(h["makespan_from_agent_start_s"]))
                    clock += "; nothing ran after the answer, so the session's own close counts"
            elif _ans is None:
                clock = "kept: no answer time"
            else:
                clock = "kept: " + clock
        # K2: the deliverable is the answer, and the run owes no fixed job, so time to the answer is the agent's last answer turn
        # (score_all.precision_comparison's answer_s), not the later of it and the last job's end
        _pa = (s.get("precision") or {}).get("answer_s") if isinstance(s.get("precision"), dict) else None
        if task == "pkg-k2-precision" and _pa:
            ends = [float(_pa)]
        rep = max(ends) if ends else None
        over_ref = round(rep / ref, 3) if (ref and rep) else None
        over_ref_any = round(rep / ref_any, 3) if (ref_any and rep) else None
        # a cell analysis ruled to have no reading on the bars (no deliverable: G-kvgate-sonnet-L0, the analysis
        # 2026-10-24 10:55Z, never answered and was cut at the horizon): kept in the table, out of every verdict
        # a run whose work never reached a GPU before the horizon:
        # valid with no deliverable; its ratio is a lower bound capped by the horizon, so it is counted +Knr, never read as a number
        no_reading = bool(((h.get("analysis_labels") or {}).get("no_bar_reading"))) or bool(h.get("no_gpu_job_by_horizon"))
        _bv = (EXCLUDED_BRIEFS.get(task) or {}).get((s.get("labels") or {}).get("task_brief_sha256") or "")
        if _bv:
            no_reading = True
        # a FAILURES_BOUND package's run with owed jobs still undone at its end: its
        # hindsight schedules only the completed launches, so over_failures over them is no reading of the run; coverage stays beside it
        # (G-oom-sonnet-L0-r2: 15 of 24 done, coverage 0.625, over_failures 1.003 over the 15)
        # every package of the 10-31 search, not FAILURES_BOUND's alone: G-rnd8-sonnet-L0 did
        # 30 of 63 (coverage 0.476) and its over_best 1.134 is over those 30. The five frozen packages keep their own rules
        owed_left = ((s.get("idle") or {}).get("work_remaining_at_window_end") or {}).get("remaining")
        if task in UNDONE_IS_NO_READING and owed_left:
            no_reading = True
        # the checkpoint-collision ensemble: owed members are the checkpoints the digests identify, not the trainings that ran
        _sfc = s.get("sf_checkpoint")
        if task == "pkg-sf-checkpoint-collision" and (not isinstance(_sfc, dict) or _sfc.get("pending")
                                                      or (_sfc.get("delivered") or 0) < (_sfc.get("required") or 40)):
            no_reading = True
        # pkg-filter-yield's preparations are owed work too (Sol independent 2026-11-02 18:18Z): a run that left a required preparation
        # undone has no reading, its GPU jobs' figure kept beside it (score_all.filter_yield_answer's all_preparations_done)
        _fy = s.get("filter_yield_answer")
        if task == "pkg-filter-yield" and isinstance(_fy, dict) and _fy.get("all_preparations_done") is False:
            no_reading = True
        # a PROOF_BOUND package's run that completed no valid proof (its open cells owe nothing, so owed_left reads 0): no reading
        if task in PROOF_BOUND and (s.get("tabular_adoption") or {}).get("over_proof_bound") is None:
            no_reading = True
        # over_feasible: the work's makespan from agent start (the last job's end, no
        # report latency) over the expected file's feasible_best_s, for a package whose hindsight bound cannot see a dependency
        fb = feasible_best(task)
        ms = last_job_end(s, h)
        over_feasible = round(float(ms) / fb, 3) if (fb and ms) else None
        out.append({"run_id": rid, "task": task, "model": mod, "arm": m.group(1) if m else "?",
                    "over_best": h.get("actual_over_hindsight"), "over_ref": over_ref, "over_ref_clock": clock, "over_ref_w0": ref_ids, "over_ref_any_plan": over_ref_any, "coverage": coverage(s, task),
                    # a goal-only package's answer support: mls-kv's pick_supported, or G4n's g4n_answer.supported (score_all)
                    "pick_supported": ((s.get("mlskv_process") or {}).get("pick_supported") if isinstance(s.get("mlskv_process"), dict)
                                       else (s.get("g4n_answer") or {}).get("supported") if isinstance(s.get("g4n_answer"), dict)
                                       else (s.get("k_answer") or {}).get("supported") if isinstance(s.get("k_answer"), dict) and "supported" in (s.get("k_answer") or {})
                                       else (s.get("filter_yield_answer") or {}).get("supported") if isinstance(s.get("filter_yield_answer"), dict)
                                       else (s.get("tabular_table") or {}).get("supported") if isinstance(s.get("tabular_table"), dict)
                                       else ((s.get("sf_checkpoint") or {}).get("delivered") == (s.get("sf_checkpoint") or {}).get("required"))
                                       if isinstance(s.get("sf_checkpoint"), dict) and not s["sf_checkpoint"].get("pending")
                                       else (s.get("precision") or {}).get("answer_matches_key") if isinstance(s.get("precision"), dict) else None),
                    # the GPU-seconds the run spent on its task's unneeded_jobs (idle.work_remaining's unneeded_work), on every package
                    # with a decision; None where the task lists none
                    "unneeded_gpu_s": (round(sum(u.get("gpu_wall_s") or 0.0 for u in
                                                 (((s.get("idle") or {}).get("work_remaining_at_window_end") or {}).get("unneeded_work") or [])), 1)
                                       if unneeded_listed(task) else None),
                    # the records the run's daemon read (run.json records_sha256 at its start): a records-only fix keeps the task hash
                    # (TTC sync 9f's memory.peak, the study 2026-11-03 08:50 MST), so this separates the cohorts the task hash cannot
                    "records_digest": records_digest(os.path.dirname(p)),
                    "polls_per_h": round((h.get("poll_count") or 0) / gh, 1) if gh > 0.05 else None,
                    "report_latency_s": h.get("report_latency_s"),
                    "over_feasible": over_feasible,
                    # the T-score fields, beside the bar for every package:
                    # actual over the bound with failed attempts charged to the environment, and the GPU-capacity share idle while
                    # a released job that fits waited (width- and attempt-aware); score.json's hindsight, None before they existed
                    "over_failures": _r3((s.get("hindsight") or {}).get("actual_over_hindsight_with_failures")),
                    "over_proof": _r3((s.get("tabular_adoption") or {}).get("over_proof_bound")),
                    # the first supported answer's ratio, beside the final one (a written label; no verdict reads it)
                    "over_proof_first": _r3((s.get("tabular_adoption") or {}).get("over_proof_first")),
                    # over_answer: on pkg-sol-tabular-operations, whose bar over_best is jobs-only
                    # (hindsight's actual is the last job's end), the time to the answer over the same bound, shown beside it and never as
                    # the bar: (actual + report_latency) / bound, for a run whose answer gave a recommendation (tabular_table); None for a
                    # hand-back before the table (its latency is negative) and on every other package
                    "over_answer": (_r3(((s.get("hindsight") or {}).get("actual_s") + h.get("report_latency_s")) / (s.get("hindsight") or {}).get("bound_s"))
                                    if (task == "pkg-sol-tabular-operations" and (s.get("hindsight") or {}).get("bound_s")
                                        and (s.get("hindsight") or {}).get("actual_s") is not None
                                        and isinstance(h.get("report_latency_s"), (int, float)) and h["report_latency_s"] >= 0
                                        and any(v is not None for v in ((s.get("tabular_table") or {}).get("answer_recommendations") or {}).values()))
                                    else None),
                    "idle_fillable_share": _r3((s.get("hindsight") or {}).get("idle_fillable_share")),
                    "saw_tell": h.get("saw_tell"), "validity": h.get("validity"), "no_bar_reading": no_reading, "excluded_brief": _bv,
                    # the work-done grace (lifecycle 8486c765): pooled by default; --split grace shows the cells by it
                    "lifecycle": h.get("lifecycle_sha256"), "grace_outcome": h.get("grace_outcome"),
                    "gpumock_patch": h.get("gpumock_patch") or _patch_of(os.path.dirname(p)),
                    "gpumock_sidecar": _sidecar_of(os.path.dirname(p)),
                    "gpumock_build": build_of(h.get("gpumock_patch") or _patch_of(os.path.dirname(p)), _sidecar_of(os.path.dirname(p))),
                    # detected_mock_disclosed: the final answer says the hardware may be
                    # emulated and the run still delivered; kept in the cell's median, marked with a dagger wherever it is quoted
                    "disclosed": bool((h.get("analysis_labels") or {}).get("detected_mock_disclosed"))})
    return out

def _r3(x):
    return round(float(x), 3) if isinstance(x, (int, float)) else None

_UNNEEDED = {}


def unneeded_listed(task):
    """Whether the task's registry file lists unneeded_jobs (the work a decision in its brief makes unnecessary)."""
    if task not in _UNNEEDED:
        try:
            t = json.load(open(os.path.join(EXP, "gpu-tasks", "registry", "tasks", f"{task}.json")))
            _UNNEEDED[task] = bool(t.get("unneeded_jobs"))
        except (OSError, ValueError):
            _UNNEEDED[task] = False
    return _UNNEEDED[task]


def loss_column(task):
    """The column bars 4-6 read for a package: over_ref (goal-only, and OVER_REF_READ), over_failures (FAILURES_BOUND), over_proof
    (PROOF_BOUND), over_feasible (an expected file with feasible_best_s), else over_best."""
    return ("over_ref" if task in GOAL_ONLY or task in OVER_REF_READ else "over_failures" if task in FAILURES_BOUND else
            "over_proof" if task in PROOF_BOUND else
            "over_feasible" if feasible_best(task) is not None else "over_best")

def med(v):
    v = [x for x in v if x is not None]
    return round(statistics.median(v), 3) if v else None

def cell_runs(va):
    """The runs a table cell's medians read, and its n: a no_bar_reading run is left out and counted as +Knr."""
    v = [r for r in va if not r.get("no_bar_reading")]
    nr = len(va) - len(v)
    return v, f"{len(v)}" + (f"+{nr}nr" if nr else "")

def verdicts(rs):
    by = {}
    for r in rs:
        if r["validity"] == "valid" and not r.get("no_bar_reading"):
            by.setdefault(r["task"], {}).setdefault((r["model"], r["arm"]), []).append(r)
    out = {}
    for task, g in sorted(by.items()):
        cell = lambda mod, arm, k: med([r[k] for r in g.get((mod, arm), [])])
        loss = loss_column(task)
        bm = bar_models(task)
        l0 = {mod: cell(mod, "L0", loss) for mod in bm}
        cov = {mod: cell(mod, "L0", "coverage") for mod in bm}
        have_cov = [c for c in cov.values() if c is not None]
        if task in GOAL_ONLY:
            # a goal-only package's bar 3 reads the answer: supported (pick_supported true) for
            # every model with L0 runs, its share of supported runs at least half; a hand-back or an unsupported pick is early exit
            sup = {mod: [r["pick_supported"] for r in g.get((mod, "L0"), [])] for mod in bm}
            sup = {m: v for m, v in sup.items() if v}
            b3 = ("no data" if not sup else "pass" if all(sum(1 for x in v if x) * 2 >= len(v) for v in sup.values())
                  else f"FAIL (early exit: supported {dict((m, f'{sum(1 for x in v if x)}/{len(v)}') for m, v in sup.items())})")
        else:
            b3 = "no data" if not have_cov else ("pass" if all(c >= 0.9 for c in have_cov) else f"FAIL (early exit: {cov})")
        n125 = sum(1 for v in l0.values() if v is not None and v >= 1.25)
        ap = cell("astra", "L0", "polls_per_h")
        b4 = ("no data" if all(v is None for v in l0.values()) else
              "pass" if n125 >= 2 else f"FAIL ({n125} of 3 at >= 1.25: {l0})")
        def keeps(arm):
            res = []
            for mod in ("haiku", "sonnet"):
                x, b = cell(mod, arm, loss), l0.get(mod)
                if x is None or b is None:
                    continue
                res.append((mod, x, b, x >= 1.15 and (x - 1) >= (b - 1) / 2))
            return ("no data" if not res else "pass" if any(r[3] for r in res) else
                    "FAIL (" + "; ".join(f"{m} {x} vs L0 {b}" for m, x, b, _ in res) + ")")
        out[task] = {"bar2_fair": None, "loss_column": loss, "L0_loss": l0, "L0_coverage": cov, "astra_L0_polls_per_h": ap,
                     "L0_idle_fillable_share": {mod: med([r.get("idle_fillable_share") for r in g.get((mod, "L0"), [])]) for mod in bm},
                     "bar3_coverage": b3, "bar4_loss": b4, "bar5_L3": keeps("L3"), "bar6_S1": keeps("S1")}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append")
    ap.add_argument("--all-validities", action="store_true")
    ap.add_argument("--json", action="store_true")
    # analysis 2026-10-27 00:58Z: a cell split into runs closed under the work-done grace lifecycle and runs before it, on request;
    # the default pools them (no bar reads the tail) and verdicts() always pools
    ap.add_argument("--split", choices=("grace", "build"), default=None,
                    help="grace: split each cell's arm into <arm>/pre-grace (lifecycle without the work-done grace) and "
                         "<arm>/grace (lifecycle 8486c765 and later), in the table only; build: split a frontier model's cell "
                         "into <arm>/pre-r4 and <arm>/r4 (gpu-mock's shard serving) when it holds both, in the table only")
    a = ap.parse_args()
    rs = rows(a.task, a.all_validities)
    vd = verdicts(rs)
    w0 = w0_rows(a.task)
    for task in vd:
        ws = [w for w in w0 if w["task"] == task and w["validity"] == "valid" and w["over_best"] is not None
              and not w.get("served_twice_defect")]
        st2 = [w for w in w0 if w["task"] == task and w.get("served_twice_defect")]
        bo = [w for w in ws if not w["plan"].startswith("recorded winners") and "a second line" not in w["plan"]]
        vd[task]["bar2_fair"] = ("no brief-only measure" + (f" (oracle measures: {[round(w['over_best'], 3) for w in ws]})" if ws else "")
                                 if not bo else "pass" if min(w["over_best"] for w in bo) <= 1.10
                                 else f"FAIL ({[round(w['over_best'], 3) for w in bo]})")
        if task in BAR2_RULED:
            vd[task]["bar2_fair"] = BAR2_RULED[task].format(ratios=[round(w["over_best"], 3) for w in bo])
        for w in st2:
            vd[task]["bar2_fair"] += f"; {w['run_id']}: {w['served_twice_defect'].get('bar2_reading')}"
    if a.json:
        print(json.dumps({"rows": rs, "w0": w0, "verdicts": vd}, indent=1))
        return
    aw = 12 if a.split else 3   # build: "L0/pre-r4" fits
    print(f"{'task':24s} {'model':6s} {'arm':{aw}s} {'n':>5s} {'over_best':>9s} {'over_feas':>9s} {'over_ref':>8s} {'over_fail':>9s} {'report_s':>8s} {'coverage':>8s} {'polls/h':>7s} {'fillable':>8s} {'tell':>4s}  runs")
    g = {}
    for r in rs:
        arm = r["arm"]
        if a.split == "grace":
            arm = f"{arm}/{'grace' if (r.get('lifecycle') or '') in GRACE_LIFECYCLE else 'pre-grace'}"
        g.setdefault((r["task"], r["model"], arm), []).append(r)
    if a.split == "build":
        # only a frontier cell holding both builds is split; every other cell stays pooled
        g2 = {}
        for (task, mod, arm), va in g.items():
            kinds = {(r.get("gpumock_patch") or "") in R4_PATCHES for r in va}
            if mod in FRONTIER and len(kinds) == 2:
                for r in va:
                    g2.setdefault((task, mod, f"{arm}/{'r4' if (r.get('gpumock_patch') or '') in R4_PATCHES else 'pre-r4'}"), []).append(r)
            else:
                g2[(task, mod, arm)] = va
        g = g2
    for (task, mod, arm), va in sorted(g.items()):
        # a no_bar_reading run (analysis's ruling: valid, no answer) is left out of the cell's medians, as verdicts() leaves it
        # out of the bars, and shown beside n (the analysis 2026-10-25 01:11Z: mls-kv-gate Sonnet L0 read n 2 at 2.536,
        # the median of r2's 2.072 and the no-reading run's 3.0)
        v, _n = cell_runs(va)
        _rl = med([r['report_latency_s'] for r in v])
        _or = med([r['over_ref'] for r in v])
        _or = str(_or) if _or is not None else ("~" + str(med([r['over_ref_any_plan'] for r in v])) if med([r['over_ref_any_plan'] for r in v]) is not None else "None")
        print(f"{task:24s} {mod:6s} {arm:{aw}s} {_n:>5s} {str(med([r['over_best'] for r in v])):>9s} "
              f"{str(med([r['over_feasible'] for r in v])):>9s} {_or:>8s} {str(med([r.get('over_failures') for r in v])):>9s} "
              f"{(str(round(_rl)) if _rl is not None else 'None'):>8s} {str(med([r['coverage'] for r in v])):>8s} "
              f"{str(med([r['polls_per_h'] for r in v])):>7s} {str(med([r.get('idle_fillable_share') for r in v])):>8s} {sum(1 for r in v if r['saw_tell']):4d}  {','.join(r['run_id'] + ('*' if r.get('no_bar_reading') else '') + ('\u2020' if r.get('disclosed') else '') for r in va)[:90]}")
    if any(r.get("no_bar_reading") for r in rs):
        print("n: runs with a reading; +Knr: K more valid runs ruled no_bar_reading (marked * in runs), left out of the medians and the bars")
    if any(r.get("disclosed") for r in rs):
        print("\u2020: detected_mock_disclosed (its answer says the hardware may be emulated; it still delivered): kept in the median, marked")
    print()
    for w in w0:
        print(f"{w['task']:24s} W0 {w['run_id']:32s} speed {w['speed']} over_best {w['over_best']}"
              + (f" (GPU packing only) over_feasible {w['over_feasible']}" if w.get("over_feasible") is not None else "")
              + f" {w['plan']} {w['validity']}")
    if any(r.get("over_feasible") is not None for r in rs) or any(w.get("over_feasible") is not None for w in w0):
        print("over_feas: makespan to the last job's end over the expected file's feasible_best_s; where a package has it, its "
              "bars 4-6 read it and over_best is GPU packing only")
    print()
    for task, x in vd.items():
        print(f"{task} [{x['loss_column']}]: (2) {x['bar2_fair']} | (3) {x['bar3_coverage']} | (4) {x['bar4_loss']} | (5) {x['bar5_L3']} | (6) {x['bar6_S1']}")

if __name__ == "__main__":
    main()
