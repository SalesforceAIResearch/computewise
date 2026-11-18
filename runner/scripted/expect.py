"""Expected scores for a scripted agent's run, computed from its plan and the registry alone.

usage: python3 -I expect.py <task.json> <plan.json> [--speed 1] [--host-s <per-job host time>]
                            [--host-s-file calib.json] [--json]

This is the independent path for checkpoint C1 (design/IDEA.md, "Idle time"; the study's
ruling: keep it independent). It must never import or call anything under runner/score/. It
simulates the plan on an idealised box: each job takes its host time (start-up and teardown on
the CPU, GPUs idle) plus its paced GPU time from the record, divided by the speed knob. Host time
(start-up, in-run gaps, teardown) is real CPU work in wall seconds and is not divided by the speed
knob, which paces GPU work only (design/IDEA.md, "Reported runs are at speed 1"; gpu-tasks' host
terms are wall seconds too). Think steps take their stated time. Launches and process starts take no time.

Two kinds of output, kept apart:
  exact       follows from the plan and the records alone. Busy GPU-seconds per job and per GPU,
              which GPUs each job used, work remaining at the deadline, agent-attributed think
              time, the agent's end relative to the last launch.
  predicted   also needs each job's host time, which only a calibration run measures. Makespan,
              whole-box and partial idle shares, time to first launch, idle after the agent stopped.
              With no --host-s the host time is 0 and the predicted figures are the ideal.

Paced GPU time for a job is what gpu-mock paces from the record (see gpu_s), at the launched GPU count.
Every smoke-task job runs at its recorded count, so the conversion factor is 1. A job whose
launched count differs is refused, not guessed.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.normpath(os.path.join(HERE, "..", "..", "gpu-tasks", "registry"))

def record(rid):
    return json.load(open(os.path.join(REGISTRY, "records", rid + ".json")))

def _v(x):
    return x.get("value") if isinstance(x, dict) else x

def gpu_s(job, width=None):
    if "bench_gpu_s" in job:          # the bench's synthetic jobs carry their paced GPU time directly
        return float(job["bench_gpu_s"])
    return _gpu_s_record(job, width)

STARTUP_S = {}       # job -> its measured start-up (host terms' hello_s + pre_s), for per-example pacing; set by simulate
GROW_MICROBATCH_PER_DOUBLING = 1.25   # gpu-mock's convert.py constant, written out here, not imported

def time_factor(batch_policy, recorded, launched):
    """The factor on a job's step time at `launched` GPUs instead of `recorded`, by gpu-mock's convert() rule
    (experiments/gpu-mock/src/convert/convert.py), written out independently: more GPUs than recorded is clamped to 1;
    fixed_per_device does not apply off the recorded count (None); fixed_global_grow_microbatch is 1.25 per doubling;
    otherwise (fixed_global_grow_accumulation, the default) recorded / launched."""
    if launched >= recorded:
        return 1.0
    ratio = recorded / launched
    if batch_policy == "fixed_per_device":
        return None
    if batch_policy == "fixed_global_grow_microbatch":
        import math
        return GROW_MICROBATCH_PER_DOUBLING ** math.log2(ratio)
    return ratio

def _gpu_s_record(job, width=None):
    """The GPU time gpu-mock paces for a replayed job, from the record, by the rule its patch
    applies (experiments/gpu-mock/src/torch_patch/replay.py: _step_time, on_optimizer_step,
    on_eval_phase_begin, and the startup pacing in _engage):
        first_step_s (or steady_step_s) + (steps - 1) * steady_step_s
        + startup_in_train_s, only when the record has no first_step_s
        + eval_s, spread over the validations
    This is NOT timing.train_time_s. Recorded wall time includes host work that replay does not
    pace; on the DPO smoke task the paced figure is about 12 s shorter per job (gpu-tasks and the
    runner, 2026-10-19). Written out independently here, not imported, so that a change to the
    patch shows up as a C1 mismatch and is not silently absorbed."""
    r = record(job["records"][0])
    recorded = int(r["reference"]["gpu_count"])
    launched = int(width or max(job["gpu_options"]))
    f = time_factor(((r.get("launch") or {}).get("batch_policy") or {}).get("value"), recorded, launched)
    if f is None:
        sys.exit(f"{job['job_id']}: fixed_per_device record launched on {launched} of {recorded} GPUs: the recorded curve does not apply")
    tm = r.get("timing") or {}
    sv = (r.get("materialisation") or {}).get("serving") or {}
    if sv.get("mode") == "per_example":
        # an evaluation served per example (MLS-Bench KV; gpu-mock's docs/MLS-KV-SERVING.md §1.3 and §2, written out
        # here, not imported): the launch's start-up is host time, then each example paces max(0, wall_total_s - L) /
        # n_examples_full on the GPU. L is the launch's own real start-up on the box (gpu-mock, 2026-10-22 19:25Z: the
        # daemon subtracts what it measured, about 101 s on a c3, so a full launch prints the recorded runtime), which
        # expect.py takes from the plan's host terms (STARTUP_S, set by simulate) and else from the record's load_s.
        # A --max-examples k launch (the job's max_examples, from the reference plan's SELECTION_KV_MAX_EXAMPLES /
        # ADAPTIVE_KV_MAX_EXAMPLES) paces k examples; k 0, absent or >= n is the whole record
        wall = float(_v(tm.get("wall_total_s")) or 0.0)
        load = STARTUP_S.get(job["job_id"])
        load = float(_v(sv.get("load_s")) or 0.0) if load is None else float(load)
        n = int(_v(sv.get("n_examples_full")) or job.get("n_examples_full") or 0)
        if wall <= 0 or n <= 0:
            sys.exit(f"{job['job_id']}: a per_example record without wall_total_s or n_examples_full to pace from")
        k = int(job.get("max_examples") or 0)
        k = n if k <= 0 or k >= n else k
        return k * max(0.0, wall - load) / n
    steps = int(r.get("steps") or 0)
    steady = float(_v(tm.get("steady_step_s")) or r.get("per_step_gpu_s") or 0.0)
    first = _v(tm.get("first_step_s"))
    startup = 0.0 if first else float(_v(tm.get("startup_in_train_s")) or 0.0)
    ev = float(_v(tm.get("eval_s")) or 0.0)
    if steps <= 0 or steady <= 0:
        sys.exit(f"{job['job_id']}: record has no steps or step time to pace from")
    # convert(): the steady step and its share of the first step scale by f; the first step's one-off excess and the
    # start-up in training don't; validations scale by f
    first_unit = (max(0.0, float(first) - steady) + steady * f) if first else steady * f
    return first_unit + (steps - 1) * steady * f + startup + ev * f

def gap_list(job, terms):
    """A job's in-run GPU-idle gaps (checkpoint saves, blocking eval), from host terms by script:
    [(fraction of its paced GPU work after which the gap falls, gap seconds)]. The job holds its
    GPUs through a gap, so it ends later by the sum of its gaps (gpu-tasks' rule, 2026-10-20)."""
    sc = None
    for tok in job.get("launch", "").split():
        if tok.endswith(".py"):
            sc = os.path.normpath(tok)
            break
    # by path where the terms file keys it so (two scripts share a basename), else by basename (gpu-tasks' rule)
    t = ((terms or {}).get(sc) or (terms or {}).get(os.path.basename(sc))) if sc else {}
    t = t or {}
    if not t.get("gaps"):
        return []
    try:
        steps = int(record(job["records"][0]).get("steps") or 0)
    except (KeyError, IndexError, OSError):
        steps = 0
    out = []
    for after, g in t["gaps"]:
        frac = min(1.0, float(after) / steps) if steps else 1.0
        out.append((frac, float(g)))
    return sorted(out)

def busy_pieces(gstart, gpu, gaps, speed):
    """The job's paced GPU work as busy intervals from gstart, with its gaps as idle holes inside.
    Returns (pieces, gpu_free) where gpu_free is when the last paced unit ends."""
    pieces, t, done = [], gstart, 0.0
    for frac, g in gaps:
        upto = gpu * frac
        if upto > done:
            pieces.append((t, t + (upto - done)))
            t += upto - done
            done = upto
        t += g                            # an in-run gap is host time: wall seconds, not sped up
    if gpu > done:
        pieces.append((t, t + (gpu - done)))
        t += gpu - done
    return pieces, t

def parse_chain(cmd, jobs_by_cmd):
    """Map a launch line (one command, or `a; b; c`) back to the task jobs it runs, in order."""
    out = []
    for part in cmd.split("; "):
        m = [jid for frag, jid in jobs_by_cmd if frag in part]
        if len(m) != 1:
            sys.exit(f"cannot tie launch fragment to exactly one job: {part[:120]}")
        out.append(m[0])
    return out

def simulate(task, plan, speed, host_s):
    jobs = {j["job_id"]: j for j in task["jobs"]}
    gpus = task["box"]["gpus"]
    # a job is recognised by its step's label, which plan.py sets to the task job's id; a chain step (`a; b`, or a label
    # that is not a job id) falls back to the isolated output dir in each part of its command line
    keyfrag = [(f"out-{jid}" if "bench_gpu_s" in jobs[jid] else f"-{jid}", jid) for jid in jobs]
    def jobs_in(step):
        lab = step.get("label")
        if lab in jobs and "; " not in step["cmd"]:
            return [lab]
        return parse_chain(step["cmd"], keyfrag)
    gpu_free = {}              # GPU -> when its last queued work ends (launch_at only)
    t = 0.0
    lanes_free = {}            # label -> time its chain ends
    busy = []                  # (start, end, [gpu indices], job_id): the job's paced GPU work
    done = {}                  # job_id -> when the job exits: GPU work end + its post_s
    think = 0.0
    first_launch = None
    def hello_of(jid):
        # the scorer's first launch is the job's process start (job_hello's t_start), which is the
        # launch plus hello_s; so is gpu-tasks' (its time_to_first_launch is hello_s on a launch at 0)
        if ht is None or not speed:
            return 0.0
        return float(_plan.terms_of(ht, _plan.script_of(jobs[jid]["launch"])).get("hello_s", 0.0))
    def place(jid, start, gpu_list):
        nonlocal first_launch
        g = gpu_s(jobs[jid], len(gpu_list)) / speed
        h = host_s.get(jid, host_s.get("*", 0.0))       # host time is wall seconds: the speed knob paces GPU work only
        # host time first (start-up), then the paced GPU work; teardown is folded into host time
        fl = start + hello_of(jid)
        first_launch = fl if first_launch is None else min(first_launch, fl)
        pieces, gend = busy_pieces(start + h, g, gap_list(jobs[jid], terms), speed)
        for a, b in pieces:
            busy.append((a, b, gpu_list, jid))
        done[jid] = gend + post_s.get(jid, 0.0)
        return done[jid]
    def gpus_for(cmd, jid):
        if cmd.strip().startswith("CUDA_VISIBLE_DEVICES="):
            return [int(cmd.split("=", 1)[1].split()[0])]
        return list(range(max(jobs[jid]["gpu_options"])))
    post_s = {}                # job -> its shutdown after GPU work (final save, exit); ends the job, not the GPUs
    terms = plan.get("host_terms") or {}
    ht = None
    script_started = {}        # script -> GPU-start times of the launch_at jobs placed so far
    if not host_s and plan.get("host_terms"):
        # packed_overlap carries the host terms its launches were timed from; use the same ones,
        # per job by its script, so the plan and the expectation mean the same thing
        sys.path.insert(0, HERE)
        import plan as _plan
        ht = plan["host_terms"]
        # The same start-up rule plan.overlap_lanes timed the launches with, decided per script:
        # a launch_at job is cold when no earlier job of its script had begun its GPU work by its
        # launch (applied in the launch_at branch below). Other ops use the warm figure.
        def start_up(j, cold):
            t = _plan.terms_of(ht, _plan.script_of(j["launch"]))
            p = t.get("pre_s_first") if cold and t.get("pre_s_first") is not None else \
                (t.get("pre_s_rest") if not cold and t.get("pre_s_rest") is not None else t.get("pre_s", 0.0))
            return float(t.get("hello_s", 0.0)) + float(p or 0.0)
        host_s = {jid: start_up(j, False) for jid, j in jobs.items()}
        post_s = {jid: float(_plan.terms_of(ht, _plan.script_of(j["launch"])).get("post_s", 0.0)) for jid, j in jobs.items()}
        # a per-example job's pace subtracts its own start-up (gpu-mock's daemon does), so the GPU work and the host phase
        # add up to the record's wall time
        STARTUP_S.clear()
        STARTUP_S.update(host_s)
    agent_end = None
    for s in plan["steps"]:
        op = s["op"]
        if op == "think":
            t += float(s["seconds"])          # thinking is wall time, not divided by the speed knob
            think += float(s["seconds"])
        elif op == "run":
            for jid in jobs_in(s):
                t = place(jid, t, gpus_for(s["cmd"], jid))
        elif op == "launch_at":
            # a background launch at a fixed time after the agent started: its host phase runs
            # from then, and its GPU work starts when both the host phase is done and its GPUs are
            # free (the daemon serialises work on a GPU's cursor), as plan.overlap_lanes assumes
            jid = jobs_in(s)[0]
            at = float(s["at_s"])
            gl = gpus_for(s["cmd"], jid)
            sc = None
            if ht is not None:
                sc = _plan.script_of(jobs[jid]["launch"])
                cold = not any(gs <= at for gs in script_started.get(sc, []))
                h = start_up(jobs[jid], cold)
            else:
                h = host_s.get(jid, host_s.get("*", 0.0))
            g = gpu_s(jobs[jid], len(gl)) / speed
            fl = at + hello_of(jid)
            first_launch = fl if first_launch is None else min(first_launch, fl)
            gstart = max(at + h, max((gpu_free.get(x, 0.0) for x in gl), default=0.0))
            pieces, gend = busy_pieces(gstart, g, gap_list(jobs[jid], terms), speed)
            for a, b in pieces:
                busy.append((a, b, gl, jid))
            if sc is not None:
                script_started.setdefault(sc, []).append(gstart)
            done[jid] = gend + post_s.get(jid, 0.0)
            for x in gl:
                gpu_free[x] = gend
            lanes_free[s["label"]] = done[jid]
            t = max(t, at)
        elif op == "launch":
            tl = t
            for part, jid in zip(s["cmd"].split("; "), parse_chain(s["cmd"], keyfrag)):
                tl = place(jid, tl, gpus_for(part, jid))
            lanes_free[s["label"]] = tl
        elif op == "wait":
            labels = list(lanes_free) if s["labels"] == "all" else s["labels"]
            t = max([t] + [lanes_free[l] for l in labels])
        elif op == "pool":
            # List scheduling, simulated from the step's own definition: the next job in order
            # starts as soon as `width` GPUs are free, on the lowest-numbered free ones. With
            # stop_at_first_completion the agent launches nothing once any job has completed,
            # and its session ends at that first completion; jobs already running run on.
            # With `after` dependencies it is an event simulation of agent.run_pool: at each moment the first job
            # in order whose dependencies have exited goes next, once enough GPUs are free; otherwise time
            # advances to the next exit. A job's GPUs free when it exits (its process ends), as the agent sees it.
            gfree = {g: t for g in range(int(s["gpus"]))}       # GPU -> time it frees
            pend = list(s["jobs"])
            ends = []
            stop = bool(s.get("stop_at_first_completion"))
            now = t
            while pend:
                ready = [x for x in pend if all(d in done and done[d] <= now for d in x.get("after") or [])]
                x = ready[0] if ready else None
                w = int(x["width"]) if x else None
                if x is not None and sum(1 for g in gfree if gfree[g] <= now) >= w:
                    if stop and ends and now >= min(ends):
                        break                                   # a job finished first: stop launching
                    avail = sorted(g for g in gfree if gfree[g] <= now)[:w]
                    pend.remove(x)
                    e = place(jobs_in(x)[0], now, avail)
                    for g in avail:
                        gfree[g] = e
                    ends.append(e)
                    continue
                later = sorted(v for v in list(gfree.values()) + ends if v > now)
                if not later:
                    break                                       # nothing ready and nothing running
                now = later[0]
                if stop and ends and now >= min(ends):
                    break
            t = (min(ends) if stop else max(ends)) if ends else t
            lanes_free["pool"] = max(ends) if ends else t
        elif op == "shell":
            t += float(s.get("expect_s") or 0.0)    # a foreground data step: real CPU time, wall seconds, unscaled
        elif op == "write":
            t += float(s.get("seconds") or 0.0)     # a preparation is the agent's wall time, like a think
        elif op == "prep_pool":
            # agent.run_prep_pool: at each moment, launch the first prepared job (in order) whose `after` jobs have
            # exited and that has GPUs free; else prepare the next job (its seconds of the agent's time, launching
            # nothing meanwhile); else wait for the next exit. A job's GPUs free when it exits.
            gfree = {g: t for g in range(int(s["gpus"]))}
            unprepared = [x for x in s["jobs"] if x.get("write")]
            prepared = [x for x in s["jobs"] if not x.get("write")]
            ends, now = [], t
            while unprepared or prepared:
                x = next((x for x in prepared if all(d in done and done[d] <= now for d in x.get("after") or [])
                          and sum(1 for g in gfree if gfree[g] <= now) >= int(x["width"])), None)
                if x is not None:
                    avail = sorted(g for g in gfree if gfree[g] <= now)[:int(x["width"])]
                    prepared.remove(x)
                    e = place(jobs_in(x)[0], now, avail)
                    for g in avail:
                        gfree[g] = e
                    ends.append(e)
                    continue
                if unprepared:
                    x = unprepared.pop(0)
                    w = x["write"]
                    # a write takes its seconds; a data preparation's foreground command its expect_s (wall, unscaled);
                    # a sibling the group's command already prepared takes nothing
                    now += (float(w.get("seconds") or 0.0) if w["op"] == "write" else
                            sum(float(st.get("seconds") or 0.0) if st["op"] == "write" else float(st.get("expect_s") or 0.0)
                                for st in w.get("steps") or []))
                    prepared.append(x)
                    continue
                later = sorted(v for v in list(gfree.values()) + ends if v > now)
                if not later:
                    break
                now = later[0]
            t = max([now] + ends)
            lanes_free["pool"] = t
        elif op == "exit":
            agent_end = t
            break
    if agent_end is None:
        agent_end = t
    return busy, think, first_launch, agent_end, gpus, done

def score(task, plan, speed, host_s):
    # an open task runs its reference plan (gpu-tasks' expand_reference_plan, the same as plan.py)
    if task.get("reference_plan") and any(j.get("draw") == "open" for j in task["jobs"]):
        import os as _os
        sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "gpu-tasks", "build"))
        import common as _gt_common
        task = _gt_common.expand_reference_plan(task)
    # a decision slot (records null, bind {decision, slot}) is run by the scripted oracle as its recorded winner
    # (plan.py does the same; this path stays independent of the scorer, not of the plan)
    if task.get("decisions"):
        task = dict(task, jobs=[dict(j, launch=j["recorded_winner"]["launch"], records=j["recorded_winner"]["records"])
                                if j.get("bind") and (j.get("recorded_winner") or {}).get("records") else j
                                for j in task["jobs"]])
    busy, think, first_launch, agent_end, gpus, done_at = simulate(task, plan, speed, host_s)
    if task.get("horizon_s"):
        deadline, deadline_how = float(task["horizon_s"]) / speed, "the task's horizon_s"
    elif task.get("deadline_s"):
        deadline, deadline_how = float(task["deadline_s"]) / speed, "the task's deadline_s (earlier design)"
    else:
        deadline = 1.5 * float(task["stats"]["lower_bound_makespan_s"]) / speed
        deadline_how = "default 1.5 x lower bound, unrounded (the task has no deadline_s)"
    # a job is done when it EXITS by the deadline (its GPU work plus post_s), as the scorer counts it
    remaining = len(task["jobs"]) - sum(1 for j, t in done_at.items() if t <= deadline)
    last_end = max(done_at.values()) if done_at else 0.0
    w1 = last_end if remaining == 0 and last_end < deadline else deadline
    # sweep: count busy GPUs over [0, w1)
    pts = []
    for s, e, gl, _ in busy:
        s, e = max(0.0, s), min(e, w1)
        if e > s:
            pts += [(s, len(gl)), (e, -len(gl))]
    pts.sort()
    whole = partial_gs = busy_gs = 0.0
    n, prev = 0, 0.0
    for x, d in pts + [(w1, 0)]:
        if x > prev:
            if n == 0:
                whole += x - prev
            elif n < gpus:
                partial_gs += (gpus - n) * (x - prev)
            busy_gs += n * (x - prev)
        n += d
        prev = max(prev, x)
    # busy GPU-seconds inside the scoring window [0, w1): what the scorer measures
    per_gpu, per_gpu_all = {}, {}
    for s, e, gl, _ in busy:
        for g in gl:
            per_gpu[g] = per_gpu.get(g, 0.0) + max(0.0, min(e, w1) - max(s, 0.0))
            per_gpu_all[g] = per_gpu_all.get(g, 0.0) + (e - s)
    # all whole-box idle after the agent ended its session, inside the window, host phases
    # included. Compare with the scorer's whole_box_idle_after_agent_end, not its after_agent_ended
    # cause, which excludes host phases by the cause priority
    after_stop = 0.0
    if agent_end < w1:
        n2, prev2, pts2 = 0, agent_end, []
        for s, e, gl, _ in busy:
            s, e = max(agent_end, s), min(e, w1)
            if e > s:
                pts2 += [(s, 1), (e, -1)]
        for x, d in sorted(pts2) + [(w1, 0)]:
            if x > prev2 and n2 == 0:
                after_stop += x - prev2
            n2 += d
            prev2 = max(prev2, x)
    return {
        "exact": {
            "busy_gpu_wall_s_total": sum(per_gpu.values()),
            "busy_gpu_wall_s_per_gpu": {str(k): v for k, v in sorted(per_gpu.items())},
            "busy_gpu_wall_s_total_unclipped": sum(per_gpu_all.values()),
            "gpus_per_job": {b[3]: b[2] for b in busy},
            "work_remaining_at_deadline": remaining,
            "think_s": think,
            "deadline_wall_s": deadline,
            "deadline_how": deadline_how,
        },
        "predicted": {
            "window_wall_s": w1,
            "makespan_from_agent_start_wall_s": last_end,
            "time_to_first_launch_wall_s": first_launch,
            "whole_box_idle_share_of_wall": whole / w1 if w1 else None,
            "partial_idle_share_of_gpu_seconds": partial_gs / (gpus * w1) if w1 else None,
            "utilisation_over_window": busy_gs / (gpus * w1) if w1 else None,
            "agent_end_wall_s": agent_end,
            "whole_box_idle_after_agent_end_wall_s": after_stop,
        },
        "assumptions": {"speed": speed, "host_s": host_s,
                        "note": "host time 0 gives the ideal; a calibration run supplies measured host time"},
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task"); ap.add_argument("plan")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--host-s", type=float, default=None, help="host time per job, wall seconds")
    ap.add_argument("--host-s-file", default=None, help="JSON {job_id or '*': seconds}")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    host = json.load(open(a.host_s_file)) if a.host_s_file else ({"*": a.host_s} if a.host_s is not None else {})
    r = score(json.load(open(a.task)), json.load(open(a.plan)), a.speed, host)
    print(json.dumps(r, indent=1))

if __name__ == "__main__":
    main()
