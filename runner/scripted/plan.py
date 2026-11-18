"""Write a scripted agent's plan for one task and one policy.

usage: python3 -I plan.py <task.json> <policy> [--think-s 20] [--delay-s 300] [--out plan.json]

Policies (design: the C1 note to the study, 2026-10-20; approved in design/IDEA.md, "Idle time"):
  serial              one job at a time, in the task's job order, with a think before each job
  packed              list scheduling: jobs in longest-recorded-first order, each launched the
                      moment enough GPUs are free and its `after` jobs have exited, on the
                      lowest-numbered free GPUs; the first job in order that is ready goes next.
                      Meets the lower bound on both smoke tasks
  delayed             packed, after a think of --delay-s before the first launch
  quit_at_first_completion  packed, but the session ends when the first job completes; jobs
                      already running keep going, nothing more is launched
  quit_after_first    run the first job in the foreground, then end the session
  quit_while_running  launch the first job in the background and end the session at once
  packed_cp           packed, with jobs taken by their remaining critical path (estimate plus the longest chain of
                      `after` descendants; ties by the longest estimate, then file order): the scripted reference for
                      every shape with dependencies; on a task without
                      dependencies it is packed exactly
  packed_overlap      the reference for the environment floor: packed, with each job's CPU start-up
                      overlapped with the previous job's GPU work (overlap_lanes). Needs
                      --host-terms from a measuring run; the floor is then paid once per lane
  controller          a task's own scripted reference controller (task.scripted_reference: Sol independent, 2026-11-02,
                      pkg-filter-yield; the study, pkg-sf-zero-lr): a Write of the controller to its path, then one
                      foreground shell step that runs it. The controller makes every preparation, launch and repair itself;
                      the plan carries it, so only the scripted run's box receives it (a model-agent run's tree never holds it).
                      filter_yield is the same policy under its first name (order 1014 is queued with it)

Preparation packages (jobs that carry `prep`; gpu-tasks' Sim.prepare in expected_scores.py, 2026-10-21):
  serial              per job in task order: a think, its preparation, then the job in the foreground
  packed              prepare the jobs one at a time in the packed order (longest disclosed estimate first); after
                      each preparation, launch every prepared job that fits on free GPUs (its `after` done), then
                      prepare the next; a job's exit is noticed only between preparations; once all are prepared,
                      wait as packed does (agent.run_prep_pool)
  Other policies refuse a preparation package.
A data-preparation package (task.data_prep: one shared step makes every job's data) starts with that step: a Write of
data_prep.reference to data_prep.scripted_write, then data_prep.scripted_command in the foreground; the task's policy
then runs as for its control package.
A preparation takes the job's prep.scripted_s (else --prep-s) of the agent's time, then writes the job's reference
script (prep.reference, else records[0] agent_view.script, registry-relative) to prep.produces[0]. The box's tree does
not hold the references, so the planner embeds the script on the host side, and the agent writes it with a Write call
after spending the time as a model turn, as a model would.

Commands come from the task's own launch lines (jobs[].launch). The agent sees only those, as
any agent would. Per-job output directories are made distinct by appending the job id to
--output_dir, which the records exempt from matching (task shared_state.isolation), so no
policy collides on shared state; a task whose shared_state.isolation is null shares nothing, and its
lines are used unchanged. Every step carries the task job's id as its label, which expect.py and the
agent's transcript use to tie a launch to its job. The agent pins each job with CUDA_VISIBLE_DEVICES to the GPUs
it picks. Job order uses the recorded durations, which only the planner sees: a scripted agent
is an oracle policy, not a model.

The planner never reads anything the scorer computes. The expected scores come from expect.py,
which reads this plan and the registry.
"""
import argparse, json, os, re, shlex, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.normpath(os.path.join(HERE, "..", "..", "gpu-tasks", "registry"))
sys.path.insert(0, HERE)
import expect                      # the paced-GPU-time rule, shared so plan timing and expectations agree

def record_train_s(record_id):
    r = json.load(open(os.path.join(REGISTRY, "records", record_id + ".json")))
    return float(r["timing"]["train_time_s"]["value"] or 0.0)

ISOLATE = True    # False when the task's shared_state.isolation is null: its jobs share nothing (main sets it)
ISOLATE_ENV = None   # the env var a job script takes its output path from (shared_state.isolation.flag), when it has one
ISOLATE_EXEMPT = None   # shared_state.isolation.exempt: each job already writes its own unique output (the TTC adoption twins' wrappers)
ISOLATE_FLAG = None     # shared_state.isolation.flag as written

def isolated(cmd, job_id):
    """Append the job id to --output_dir so jobs that share a default path don't collide. A task whose
    shared_state.isolation is null shares no state between jobs (the modded-nanogpt packages write logs/<uuid>.txt and
    read data/ only; gpu-tasks, 2026-10-20), so its launch lines are used unchanged. A task whose isolation names an
    environment variable (MLS-Bench KV's job scripts read ${OUTPUT_DIR:-./output}; gpu-tasks, 2026-10-22) gets that
    variable set to a per-job directory in front of the launch line instead."""
    if not ISOLATE:
        return cmd
    if ISOLATE_ENV:
        if re.search(rf"(^|\s){re.escape(ISOLATE_ENV)}=", cmd):
            return cmd
        return f"{ISOLATE_ENV}=./output-{job_id} {cmd}"
    out, n = re.subn(r"(--output_dir[= ])(\S+)", lambda m: f"{m.group(1)}{m.group(2)}-{job_id}", cmd, count=1)
    if n != 1:
        # a task whose isolation flag is an argument the launch line does not carry, and whose `exempt` says each job already writes
        # its own unique output (the TTC adoption twins: flag --output, "each wrapper already writes its unique results/<cell>.json;
        # --output is argv-exempt"; launch lines `bash jobs/<cell>.sh`; the study 2026-11-03 02:06 MST): the line is used unchanged,
        # as a null isolation's is. An env-var flag (MLS-KV's OUTPUT_DIR, also exempt) is prefixed above and never reaches here
        if ISOLATE_EXEMPT and ISOLATE_FLAG and not re.search(rf"(^|\s){re.escape(ISOLATE_FLAG)}([= ]|$)", cmd):
            return cmd
        sys.exit(f"job {job_id}: no --output_dir in its launch line, so it cannot be isolated")
    return out

def job_cmd(job, gpu=None):
    cmd = isolated(job["launch"], job["job_id"])
    if gpu is not None:
        cmd = f"CUDA_VISIBLE_DEVICES={gpu} {cmd}"
    return cmd

def job_estimate_s(j):
    """A job's duration for ordering: the estimate the task discloses to the agent (disclosure.estimate.median_s,
    gpu-tasks: the paced duration at the job's launched width, so an 8-GPU record run on one GPU counts 8x), else its
    record's recorded train time. gpu-tasks' expected_scores.py orders packed the same way."""
    est = ((j.get("disclosure") or {}).get("estimate") or {}).get("median_s")
    return float(est) if est else record_train_s(j["records"][0])

def critical_path_order(jobs):
    """packed_cp's order: each job by its remaining critical path,
    its disclosed estimate plus the longest chain of its `after` descendants; ties by the longest estimate, then file
    order. On a task without dependencies it is packed's order exactly."""
    idx = {j["job_id"]: k for k, j in enumerate(jobs)}
    kids = {k: [] for k in range(len(jobs))}
    for k, j in enumerate(jobs):
        for d in j.get("after") or []:
            if d in idx:
                kids[idx[d]].append(k)
    tail = {}
    def cp(i):
        if i not in tail:
            tail[i] = job_estimate_s(jobs[i]) + max([cp(k) for k in kids[i]], default=0.0)
        return tail[i]
    return [jobs[i] for i in sorted(range(len(jobs)), key=lambda i: (-cp(i), -job_estimate_s(jobs[i]), i))]

def pool_jobs(jobs, by_critical_path=False):
    """Jobs for a pool step, longest disclosed estimate first, ties in task-file order (stable
    sort); with by_critical_path, packed_cp's order. The agent assigns GPUs itself as they free up; the planner only
    fixes the order."""
    order = critical_path_order(jobs) if by_critical_path else sorted(jobs, key=lambda j: -job_estimate_s(j))
    return [{"cmd": isolated(j["launch"], j["job_id"]), "label": j["job_id"],
             "width": max(j["gpu_options"]), "after": list(j.get("after") or [])} for j in order]

def prep_step(j, prep_s, done_groups):
    """The preparation that makes job j launchable. A code preparation (the preparation packages) is a write of its
    reference script. A data preparation that runs a real command (cpu-tasks' H4, 2026-10-23: each filter recipe's
    datatrove filters and tokenisation, real CPU work in the box) is a foreground shell step, its scripted_s the expected
    wall time; prep.shared_by names the jobs one command prepares, so the command runs for the first of them and the rest
    are prepared by it (None for them). A prep with both a reference and a command writes first, then runs the command."""
    pr = j.get("prep") or {}
    if not pr.get("scripted_command"):
        return prep_write(j, prep_s)
    grp = tuple(sorted(pr.get("shared_by") or [j["job_id"]]))
    if grp in done_groups:
        return {"op": "none", "label": j["job_id"], "prepared_by": done_groups[grp]}
    done_groups[grp] = j["job_id"]
    steps = []
    if pr.get("reference"):
        steps.append(prep_write(j, 0.0))
    steps.append({"op": "shell", "cmd": pr["scripted_command"], "label": f"prep-{j['job_id']}",
                  "expect_s": float(pr.get("scripted_s") or 0.0)})
    return {"op": "seq", "steps": steps, "label": j["job_id"], "group": list(grp)}

def prep_write(j, prep_s):
    """The write step that prepares job j: after prep_s of the agent's time, its reference script (records[0]
    agent_view.script, registry-relative) goes to prep.produces[0], relative to the task root."""
    pr = j.get("prep") or {}
    ref = pr.get("reference")
    if not ref:
        rec = json.load(open(os.path.join(REGISTRY, "records", j["records"][0] + ".json")))
        ref = (rec.get("agent_view") or {}).get("script") or ""
    src = os.path.join(REGISTRY, ref)
    if not os.path.isfile(src):
        sys.exit(f"job {j['job_id']}: no reference script for its preparation ({src})")
    produces = (j.get("prep") or {}).get("produces") or []
    if not produces:
        sys.exit(f"job {j['job_id']}: prep has no produces path to write the script to")
    sec = pr.get("scripted_s")
    return {"op": "write", "path": produces[0], "content": open(src).read(),
            "seconds": float(sec) if sec is not None else prep_s, "label": j["job_id"]}

def own_port(cmd, port):
    """Give a multi-process launch its own rendezvous port (accelerate --main_process_port, torchrun
    --master_port). A single-process launch is returned unchanged."""
    if " accelerate launch " in f" {cmd} " and "--main_process_port" not in cmd:
        return cmd.replace("accelerate launch ", f"accelerate launch --main_process_port {port} ", 1)
    if " torchrun " in f" {cmd} " and "--master_port" not in cmd and "--master-port" not in cmd:
        return cmd.replace("torchrun ", f"torchrun --master_port {port} ", 1)
    return cmd

def script_of(launch):
    """The job's script from its launch line, as a normalised path relative to the task tree, e.g.
    "examples/scripts/dpo.py". Two different scripts can share a basename (contra-soft-muon/ and soda-fade/ both hold
    contra-muon-to-soft-muon.py), so jobs are told apart by path; host terms are looked up by terms_of. A job script run
    with bash (MLS-Bench KV's methods/<m>/scripts/<w>.sh; gpu-tasks, 2026-10-22) is its own key when the line names no .py."""
    for tok in launch.split():
        if tok.endswith(".py"):
            return os.path.normpath(tok)
    for tok in launch.split():
        if tok.endswith(".sh"):
            return os.path.normpath(tok)
    return None

def terms_of(host_terms, script):
    """A script's host terms: by its path when the terms file keys it so (two scripts of the task share a basename),
    else by basename (gpu-tasks' host_terms_of and derive_host_terms.py, agreed 2026-10-20)."""
    if not script:
        return {}
    h = host_terms or {}
    return h.get(os.path.normpath(script)) or h.get(os.path.basename(script)) or {}

def overlap_lanes(jobs, gpus, host_terms):
    """packed_overlap (the study, 2026-10-20, design/IDEA.md "The environment floor"): jobs packed
    into GPU lanes longest-first, and within a lane each successor launched early enough that its
    CPU start-up (host hello_s + pre_s) ends as the predecessor's GPU work ends. So the GPUs never
    wait on a start-up after the first, and the floor is paid once per lane.
    Rule, shared with gpu-tasks' expected_scores.py so both paths mean the same thing:
        launch(k)    = max(launch(k-1), gpu_free(lane) - pre(k))        (first job: 0)
        gpu_start(k) = max(launch(k) + pre(k), gpu_free(lane))
        gpu_free     = gpu_start(k) + gpu_s(k)
    Returns [(launch_at_s, job, gpu_list)] in launch order."""
    widths = {max(j["gpu_options"]) for j in jobs}
    def lane_key(j):
        # median paced GPU time over the job's records at its width (gpu-tasks' key, 2026-10-20), so
        # a seed whose draw depends on launch order does not move the lanes
        vals = []
        for rid in j.get("records") or []:
            try:
                vals.append(expect.gpu_s(dict(j, records=[rid])))
            except SystemExit:
                pass
        return statistics.median(vals) if vals else 0.0
    order = sorted(jobs, key=lambda j: -lane_key(j))                 # stable: equal keys keep file order
    if widths == {gpus}:
        lanes = [order]                                   # every job takes the whole box
        lane_gpus = [None]
    elif widths == {1}:
        lanes, load = [[] for _ in range(gpus)], [0.0] * gpus
        for j in order:
            i = min(range(gpus), key=lambda k: load[k]); lanes[i].append(j); load[i] += lane_key(j)
        lane_gpus = list(range(gpus))
    else:
        sys.exit(f"packed_overlap: mixed job widths {sorted(widths)} not handled yet")
    # Cold or warm start-up is decided PER SCRIPT, on the datasets cache's own keying (it is keyed
    # by the dataset and the map function, which the script fixes; gpu-tasks, 2026-10-20): a job is
    # cold when no earlier job of the same script had reached its first paced unit by its launch,
    # else warm. Lanes are filled together in launch-time order, ties in lane order, so a job is
    # judged against every job already placed, in any lane.
    def pre(j, cold):
        t = terms_of(host_terms, script_of(j["launch"]))
        p = t.get("pre_s_first") if cold and t.get("pre_s_first") is not None else \
            (t.get("pre_s_rest") if not cold and t.get("pre_s_rest") is not None else t.get("pre_s", 0.0))
        return float(t.get("hello_s", 0.0)) + float(p or 0.0)
    state = [{"lane": lane, "g": g, "k": 0, "launch_prev": 0.0, "gfree": 0.0} for lane, g in zip(lanes, lane_gpus)]
    started = {}                       # script -> GPU-start times of the jobs placed so far
    def tentative(st):
        j = st["lane"][st["k"]]
        def launch_for(cold):
            pk = pre(j, cold)
            return (0.0 if st["k"] == 0 else max(st["launch_prev"], st["gfree"] - pk)), pk
        lw, pw = launch_for(False)
        if any(gs <= lw for gs in started.get(script_of(j["launch"]), [])):
            return lw, pw, j           # a same-script job had begun its GPU work by then: warm
        lc, pc = launch_for(True)      # earlier than lw, so nothing had begun by lc either: cold
        return lc, pc, j
    out = []
    while any(st["k"] < len(st["lane"]) for st in state):
        cands = [(tentative(st), i) for i, st in enumerate(state) if st["k"] < len(st["lane"])]
        (launch, pk, j), i = min(cands, key=lambda c: (c[0][0], c[1]))
        st = state[i]
        gstart = max(launch + pk, st["gfree"])
        # the GPU frees when the job's last paced unit ends, its in-run gaps included
        _, gfree = expect.busy_pieces(gstart, expect.gpu_s(j), expect.gap_list(j, host_terms), 1.0)
        out.append((launch, j, st["g"]))
        started.setdefault(script_of(j["launch"]), []).append(gstart)
        st["launch_prev"], st["gfree"], st["k"] = launch, gfree, st["k"] + 1
    return sorted(out, key=lambda x: x[0])

def plan(task, policy, think_s, delay_s, host_terms=None, prep_s=60.0, brief_only=False):
    global ISOLATE, ISOLATE_ENV, ISOLATE_EXEMPT, ISOLATE_FLAG
    ss = task.get("shared_state")
    ISOLATE = not (isinstance(ss, dict) and "isolation" in ss and ss["isolation"] is None)
    iso = (ss or {}).get("isolation") if isinstance(ss, dict) else None
    flag = iso.get("flag") if isinstance(iso, dict) else None
    ISOLATE_ENV = flag if flag and re.fullmatch(r"[A-Z_][A-Z0-9_]*", flag) else None
    ISOLATE_EXEMPT = (iso.get("exempt") or None) if isinstance(iso, dict) else None
    ISOLATE_FLAG = flag
    if task.get("reference_plan") and any(j.get("draw") == "open" for j in task["jobs"]):
        # an open task (a goal package): the scripted agent runs its reference plan, each entry that many instances of
        # the candidate at the entry's width, seeded where the script takes a seed; gpu-tasks' expand_reference_plan
        # (build/common.py) is the one definition, imported so the two cannot drift
        sys.path.insert(0, os.path.join(REGISTRY, "..", "build"))
        import common as _gt_common
        task = _gt_common.expand_reference_plan(task)
    jobs = task["jobs"]
    gpus = task["box"]["gpus"]
    if policy in ("controller", "filter_yield"):
        return controller_plan(task, think_s, delay_s, policy)
    if task.get("decisions") and brief_only and not task.get("reference_plan"):
        # bar 2 (the analysis 2026-10-23 14:48Z; the forum's E): a brief-only measure of a decision package runs the
        # rule gpu-tasks writes into its reference_plan, never the recorded winners, which only an oracle knows
        sys.exit(f"{task['task_id']}: a brief-only plan of a decision package needs a reference_plan carrying the brief's rule; "
                 "the recorded winners are an oracle's")
    if task.get("decisions"):
        # a decision package: the scripted agent is the
        # oracle that knows the outcome, so each decision slot (records null, bind {decision, slot}) runs its
        # recorded_winner's launch on its records, and the slots' `after` (the screens, or the previous round's slots)
        # orders it like any dependency. A slot without a recorded winner cannot be planned.
        bound = [j for j in jobs if j.get("bind")]
        missing = [j["job_id"] for j in bound if not (j.get("recorded_winner") or {}).get("launch")]
        if missing:
            sys.exit(f"{task['task_id']}: decision slot(s) without a recorded_winner launch: {missing}")
        jobs = [dict(j, launch=j["recorded_winner"]["launch"], records=j["recorded_winner"]["records"]) if j.get("bind") else j
                for j in jobs]
        task = dict(task, jobs=jobs)
    if any(j.get("after") for j in jobs):
        # serial in task-file order respects every `after` when each job's dependencies come earlier in the file (a
        # chain's measuring run, gpu-tasks 2026-10-20); the parallel policies would need a dependency-aware pool
        seen, ok = set(), True
        for j in jobs:
            ok = ok and all(d in seen for d in j.get("after") or [])
            seen.add(j["job_id"])
        if policy == "serial" and not ok:
            sys.exit(f"{task['task_id']}: serial runs the task-file order, which here does not respect every `after`")
        if policy == "packed_overlap":
            sys.exit(f"{task['task_id']}: packed_overlap does not plan jobs with dependencies")
        # packed, delayed and quit_at_first_completion: the pool launches a job only once its `after` jobs have exited
        # cleanly (agent.run_pool, expect.simulate), as gpu-tasks' packed with deps_done does
    widths = {max(j["gpu_options"]) for j in jobs}
    steps = []
    dp = task.get("data_prep")
    if isinstance(dp, dict):
        # a shared data step (gpu-tasks' data-preparation packages, Sim in expected_scores.py): at the start, write the
        # reference preparation script, run it in the foreground (real CPU work in the box), then the control policy
        # from when it exits. The jobs' `prep` (kind data, shared) is that one step, not a per-job preparation.
        src = os.path.join(REGISTRY, dp["reference"])
        if not os.path.isfile(src):
            sys.exit(f"{task['task_id']}: data_prep reference {src} is missing")
        steps.append({"op": "write", "path": dp["scripted_write"], "content": open(src).read(), "seconds": 0.0,
                      "label": "data-prep"})
        steps.append({"op": "shell", "cmd": dp["scripted_command"], "label": "data-prep",
                      "expect_s": float(dp.get("scripted_s") or 0.0)})
    if any(isinstance(j.get("prep"), dict) and not j["prep"].get("shared") for j in jobs):
        if policy == "serial":
            done_groups = {}
            for j in jobs:
                steps.append({"op": "think", "seconds": think_s})
                if isinstance(j.get("prep"), dict):
                    st = prep_step(j, prep_s, done_groups)
                    steps.extend(st["steps"] if st["op"] == "seq" else [] if st["op"] == "none" else [st])
                steps.append({"op": "run", "cmd": job_cmd(j, 0 if max(j["gpu_options"]) == 1 else None), "label": j["job_id"]})
        elif policy == "packed":
            pj = pool_jobs(jobs)
            by_id = {j["job_id"]: j for j in jobs}
            # a data command shared by a group runs for the group's first job in the pool's order; its siblings follow
            # it at once, so the agent launches the whole group as soon as the command exits (H4: prep one recipe, launch
            # its two jobs, prep the next while they train)
            if any((by_id[x["label"]].get("prep") or {}).get("shared_by") for x in pj):
                first = {}
                for k, x in enumerate(pj):
                    g = tuple(sorted((by_id[x["label"]].get("prep") or {}).get("shared_by") or [x["label"]]))
                    first.setdefault(g, k)
                pj = sorted(pj, key=lambda x: (first[tuple(sorted((by_id[x["label"]].get("prep") or {}).get("shared_by") or [x["label"]]))],
                                               pj.index(x)))
            done_groups = {}
            for x in pj:
                j = by_id[x["label"]]
                x["write"] = prep_step(j, prep_s, done_groups) if isinstance(j.get("prep"), dict) else None
            steps.append({"op": "prep_pool", "gpus": gpus, "jobs": pj})
        else:
            sys.exit(f"{task['task_id']}: a preparation package is planned as serial or packed only, not {policy}")
        return {"policy": policy, "task_id": task["task_id"], "think_s": think_s, "delay_s": delay_s, "prep_s": prep_s,
                "host_terms": host_terms or None, "steps": steps}
    if policy == "serial":
        for j in jobs:
            steps.append({"op": "think", "seconds": think_s})
            steps.append({"op": "run", "cmd": job_cmd(j, 0 if max(j["gpu_options"]) == 1 else None), "label": j["job_id"]})
    elif policy in ("packed", "packed_cp", "delayed", "quit_at_first_completion"):
        # list scheduling: launch the next job in order whenever enough GPUs are free
        if policy == "delayed":
            steps.append({"op": "think", "seconds": delay_s})
        steps.append({"op": "pool", "gpus": gpus, "jobs": pool_jobs(jobs, by_critical_path=policy == "packed_cp"),
                      "stop_at_first_completion": policy == "quit_at_first_completion"})
        if policy == "quit_at_first_completion":
            steps.append({"op": "exit"})
    elif policy == "packed_overlap":
        # Overlapping multi-rank launches run at the same time, and accelerate and torchrun both
        # rendezvous on port 29500 by default, so the second would fail to bind. Each launch gets
        # its own port. The flag goes to the launcher, before the script, so the script's own
        # argv, which replay matches on, is unchanged.
        for k, (at, j, g) in enumerate(overlap_lanes(jobs, gpus, host_terms or {})):
            steps.append({"op": "launch_at", "at_s": round(at, 3), "cmd": own_port(job_cmd(j, g), 29500 + 11 * (k + 1)),
                          "label": j["job_id"]})
        steps.append({"op": "wait", "labels": "all"})
    elif policy == "quit_after_first":
        j = jobs[0]
        steps.append({"op": "run", "cmd": job_cmd(j, 0 if max(j["gpu_options"]) == 1 else None), "label": j["job_id"]})
        steps.append({"op": "exit"})
    elif policy == "quit_while_running":
        j = jobs[0]
        steps.append({"op": "launch", "cmd": job_cmd(j, 0 if max(j["gpu_options"]) == 1 else None), "label": j["job_id"]})
        steps.append({"op": "exit"})
    else:
        sys.exit(f"unknown policy {policy!r}")
    return {"policy": policy, "task_id": task["task_id"], "think_s": think_s, "delay_s": delay_s,
            "host_terms": host_terms or None, "steps": steps}

# a package's scripted reference kept in the runner's tree, for a task file that carries none (its hash frozen by its owners):
# pkg-k2-precision's P1 sequential controller (the study's K2 lane, package-search/lanes/unneeded-k2/precision_reference.py
# cd70edcd, 2026-11-02 12:58 MST; the analysis 10:24 MST: the W0 runs the P1 procedure, looks at 4 and 8 per arm in RUNS.tsv's
# seed-major order, not the packed plan). `runner_from` is relative to this directory; the task's own scripted_reference wins
RUNNER_REFERENCES = {"pkg-k2-precision": {"path": "reference/precision_reference.py",
                                          "runner_from": "references/pkg-k2-precision/precision_reference.py"}}

def controller_plan(task, think_s, delay_s, policy="controller"):
    """filter_yield: the task's scripted_reference (path in the task root, from registry-relative) written by the agent, then
    run in the foreground as `python <path> --root . --gpus <box gpus>` (scripted_reference.args when the task gives them).
    A task with none takes RUNNER_REFERENCES' entry. Refuses a task with neither: the controller is the reference, never a
    stand-in the planner improvises. scripted_reference.python, when given, is the interpreter it runs under (pkg-k2-latepole,
    K2 2026-11-03 06:5x MST: /opt/envs/b/bin/python, whose pyarrow the reference reads parquet sizes with; the box's bare
    `python` has none, and W0-k2lp-reference died on its import); without it, `python` as before."""
    sr = task.get("scripted_reference") or RUNNER_REFERENCES.get(task.get("task_id")) or {}
    if not sr.get("path") or not (sr.get("from") or sr.get("runner_from")):
        sys.exit(f"{task['task_id']}: filter_yield needs task.scripted_reference {{path, from}}")
    src = os.path.join(REGISTRY, sr["from"]) if sr.get("from") else os.path.join(HERE, sr["runner_from"])
    if not os.path.isfile(src):
        sys.exit(f"{task['task_id']}: scripted_reference {src} is missing")
    args = sr.get("args") or ["--root", ".", "--gpus", str(task["box"]["gpus"])]
    steps = [{"op": "write", "path": sr["path"], "content": open(src).read(), "seconds": 0.0, "label": "controller"},
             {"op": "shell", "cmd": " ".join([str(sr.get("python") or "python"), sr["path"]] + [str(x) for x in args]),
              "label": "controller", "expect_s": float(sr.get("expect_s") or 0.0)}]
    return {"policy": policy, "task_id": task["task_id"], "think_s": think_s, "delay_s": delay_s,
            "host_terms": None, "steps": steps}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("policy")
    ap.add_argument("--think-s", type=float, default=20.0)
    ap.add_argument("--delay-s", type=float, default=300.0)
    ap.add_argument("--prep-s", type=float, default=300.0,
                    help="preparation packages: a job's preparation time when its prep has no scripted_s")
    ap.add_argument("--out", default=None)
    ap.add_argument("--host-terms", default=None,
                    help="packed_overlap: host_terms.py output; its per-script hello_s + pre_s time each launch")
    ap.add_argument("--brief-only", action="store_true",
                    help="refuse a decision package that has no reference_plan: its recorded winners are an oracle's (bar 2)")
    a = ap.parse_args()
    ht = None
    if a.host_terms:
        d = json.load(open(a.host_terms))
        ht = d.get("terms", d)
    if a.policy == "packed_overlap" and not ht:
        print("packed_overlap without --host-terms: start-up taken as 0, so launches are not ahead of time", file=sys.stderr)
    p = plan(json.load(open(a.task)), a.policy, a.think_s, a.delay_s, ht, a.prep_s, brief_only=a.brief_only)
    s = json.dumps(p, indent=1)
    if a.out:
        open(a.out, "w").write(s + "\n")
    print(s)

if __name__ == "__main__":
    main()
