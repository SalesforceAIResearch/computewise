"""Would a shared cluster with a good scheduler recover the GPU time our agents lose? A replay, not a multi-tenant
experiment. Recorded L0 runs become
tenants of one shared GPU pool, and a standard batch scheduler places their jobs. The hypothesis it tests: a pool
recovers the GPU time the agents left idle, but not the unneeded work, because the scheduler can't tell that a job's
result won't be used.

usage: python3 -I shared_pool.py [--runs DIR] [--models haiku,sonnet,opus,astra,sol] [--per-tenant 4,3,2] [--json OUT]
       python3 -I shared_pool.py --selftest      (known answers; needs no runs)

Tenants. These are the valid L0 runs of the five packages (G1 mls-kv-report, H5 mls-kv-report-h, G4 mls-kv-gate, G4n
pkg-g4n-a and pkg-g4n-b), taken from screen.py's rows with its no_bar_reading runs left out. The jobs come from
occupancy.jsonl (jobs_of below): each job's wall span, hello to bye, and its one GPU from the first busy_start, hold or
enqueue that names one (job_gpu_map; busy_start alone drops a job stacked onto a busy GPU, fixed 2026-11-02). A job is
needed when its record is in reference_records(task), the expected file's brief-only plan, and unneeded otherwise.

The tenant's own box. Four GPUs held for W = max(D, the last job's end), where D is the time to the deliverable:
screen.py's over_ref times its W0 reference, the later of the final report and the work's end. Over 4·W, the box's
GPU-seconds split exactly three ways:
  needed / unneeded   each job's share of its GPU's time. A job that ran alone gets its wall time; a stretch of a GPU
                      several jobs shared is divided by each job's own work (job_work: the daemon's unit costs, a full
                      job's recorded solo length to a median 0.1%), since gpu-mock serves a shared GPU first in, first
                      out. Equal slices (--share equal, the defect fixed 2026-11-02) misattribute it badly
  idle                4·W less the busy time. Of it, "tail" is the stretch after the last job ended (report latency)

The pool. P = c·N GPUs for N tenants: c = 4 gives each tenant a box's worth, and c < 4 tests contention. Every tenant
starts at 0.
  - Every job is 1 GPU wide (checked: no job's occupancy records name two GPUs). Its length is its share above: its
    wall time, or its own work for a job that shared a GPU in its box, which in the pool gets a GPU of its own.
  - Submission. A job is submitted at its recorded start time. If the pool delayed the job whose end most recently
    preceded that start (its trigger: the result the agent last saw), the delay carries over. Only delays carry: an
    agent that slept or polled doesn't react sooner because a job ended earlier, so the agent's own think time and sleeps
    stay as recorded. A delay carries in full, so under contention the deliverables are an upper estimate: an agent busy
    thinking when a job ended would have absorbed some of it. The deliverable reports on every result before it, so it
    moves by the largest delay among the jobs that had ended by it.
  - Scheduler. EASY backfill: priority by submit time; the queue's head gets a reservation; a later job starts early
    only if it doesn't delay that reservation. FCFS is the floor. With every job 1 GPU wide, nothing ever blocks the head
    while a GPU is free, so the two schedules are identical. The selftest checks both on a wide job, where they differ.
  - Backlog tenant. An always-ready queue of preemptible filler jobs takes every GPU-second the tenants leave free, and
    yields at once to a tenant's job. That is the most any pool can hand out: a filler that couldn't be preempted would
    recover less and delay tenants. It is counted over the tenants' own windows, c GPUs per tenant from 0 to its end in
    the pool. Every tenant arrives at 0 and none after, so the pool also holds the GPUs of tenants that have left, up to
    H, the last tenant's end. That is no agent's idle, and it is shown apart.
  - With c = 4, a tenant queues only where its box ran more than 4 jobs at once by sharing GPUs, since each gets a GPU of
    its own in the pool. The backlog is then the tenants' own idle plus the capacity of tenants that left before H; that
    accounting identity holds at any c = 4 schedule, and the code prints its residual as a check on its bookkeeping.
Counterfactual. The same pool with each tenant's unneeded jobs removed. Needed jobs keep their recorded launch times, so
this credits judgement with the GPU-seconds it saves, not with an earlier finish: a bound on GPU time, a lower bound on
time to the deliverable. Triggers are re-derived among the needed jobs, so under contention one tenant's deliverable
can come later than with its unneeded jobs in, when its new trigger is a needed job that queued longer.

Not modelled: agents on a shared cluster might launch differently (queue waits, other tenants' load); and arrivals
are simultaneous, the worst alignment for contention.
"""
import argparse, heapq, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.normpath(os.path.join(HERE, "..", ".."))
PACKAGES = {"mls-kv-report": "G1", "mls-kv-report-h": "H5", "mls-kv-gate": "G4", "pkg-g4n-a": "G4n A", "pkg-g4n-b": "G4n B"}
MODELS = ("haiku", "sonnet", "opus", "astra", "sol")
CLAUDE = ("haiku", "sonnet", "opus")
BOX = 4


def _mods():
    sys.path.insert(0, os.path.join(EXP, "codex"))
    sys.path.insert(0, os.path.join(EXP, "runner", "score"))
    import tick_decomposition as T, screen as S
    return T, S


# A job's GPU, from the first occupancy event that names one. gpu-mock logs busy_start only when a GPU goes from idle to
# busy, so a job that joins a GPU another job holds has no busy_start of its own; its hold (the memory it took) and its
# enqueue (each unit it ran) name the GPU. Over the 91 runs of 2026-11-02, hold and busy_start agree for every job that
# has both (1,768 of 1,768) and no job names two GPUs. busy_start alone dropped 81 jobs in 8 runs (lane OBS, analysis
# 10:13 MST): --gpu-from busy_start reproduces that defect for comparison.
GPU_FROM = ("busy_start", "hold", "enqueue")


def job_gpu_map(occ, kinds=None):
    """{job: gpu} from the first event of `kinds` naming the job on a GPU, and the set of jobs seen on two GPUs."""
    kinds = kinds or GPU_FROM
    gpu, wide = {}, set()
    for o in occ:
        if o.get("kind") in kinds and o.get("job") is not None and (o.get("gpu") if o.get("gpu") is not None else -1) >= 0:
            if o["job"] not in gpu:
                gpu[o["job"]] = o["gpu"]
            elif gpu[o["job"]] != o["gpu"]:
                wide.add(o["job"])
    return gpu, wide


def jobs_of(run_dir, kinds=None):
    """[(job, gpu, start, end, record, work)] for the run's GPU jobs, seconds from agent start: tick_decomposition.jobs_of's
    reading (hello to bye, the bye else the teardown; records from score.json), with each job's GPU from job_gpu_map."""
    T, _ = _mods()
    occ = T.W._jsonl(os.path.join(run_dir, "occupancy.jsonl"))
    st = json.load(open(os.path.join(run_dir, "status.json")))
    t0, tend = st["agent_start_wall"], st.get("teardown_wall")
    hello = {d["job"]: d["t_wall"] for d in occ if d.get("kind") == "job_hello"}
    bye = {d["job"]: d["t_wall"] for d in occ if d.get("kind") == "job_bye"}
    gpu, wide = job_gpu_map(occ, kinds)
    sc = json.load(open(os.path.join(run_dir, "score.json")))
    rec = {x["job"]: x.get("record_id") for x in (sc.get("score") or {}).get("jobs") or []}
    work = job_work(occ)
    out = [(j, gpu[j], h - t0, (bye.get(j, tend) or h) - t0, rec.get(j), work.get(j)) for j, h in hello.items() if j in gpu]
    return sorted(out, key=lambda x: x[2]), len(wide)


SHARE = "work"          # how a shared GPU's time is divided: "work" (default) or "equal" (the defect, for comparison)


def job_work(occ):
    """{job: its solo length}: the daemon's own costs of the units it ran (modelled_s + host_gap_s per enqueue) plus its
    start-up (job_bye's startup_real_s). Over the 91 runs of 2026-11-02 this is a full job's recorded paced duration
    to a median 0.1%, alone or shared, and on a stretch of a GPU that several jobs shared it sums to that stretch's
    busy time within 1.5%."""
    w = {}
    for o in occ:
        if o.get("kind") == "enqueue" and o.get("job") is not None:
            w[o["job"]] = w.get(o["job"], 0.0) + float(o.get("modelled_s") or 0) + float(o.get("host_gap_s") or 0)
    for o in occ:
        if o.get("kind") == "job_bye" and o.get("job") in w:
            w[o["job"]] += float(o.get("startup_real_s") or 0)
    return w


def attribute(jobs, share=None):
    """{job id: its share of its GPU's time}, and the busy union over all GPUs. jobs: dicts with id, gpu, s, e, and
    optionally work (job_work's solo length).
    On each GPU the jobs fall into stretches of overlapping intervals. A job alone in its stretch gets its wall time.
    In a stretch several jobs shared, the stretch's busy time is divided in proportion to each job's work: gpu-mock
    serves a shared GPU's units first in, first out, not in equal slices, so an equal split (share="equal", or a
    stretch where a job has no work) gives a long job stacked with short ones far too little and the short ones too
    much (2026-11-02: -58% to +393% against the recorded solo lengths). Either way each GPU's total is its busy time."""
    share = share or SHARE
    out, union, by = {j["id"]: 0.0 for j in jobs}, 0.0, {}
    for j in jobs:
        by.setdefault(j["gpu"], []).append(j)
    for js in by.values():
        js = sorted(js, key=lambda j: (j["s"], j["e"]))
        comps = []
        for j in js:
            if comps and j["s"] < comps[-1][1]:
                comps[-1][1] = max(comps[-1][1], j["e"]); comps[-1][2].append(j)
            else:
                comps.append([j["s"], j["e"], [j]])
        for s0, e0, cj in comps:
            union += e0 - s0
            w = [float(j.get("work") or 0) for j in cj]
            if len(cj) == 1:
                out[cj[0]["id"]] += e0 - s0
            elif share == "work" and all(x > 0 for x in w):
                for j, x in zip(cj, w):
                    out[j["id"]] += (e0 - s0) * x / sum(w)
            else:
                pts = sorted({x for j in cj for x in (j["s"], j["e"])})
                for a, b in zip(pts, pts[1:]):
                    act = [j for j in cj if j["s"] <= a and j["e"] >= b]
                    for j in act:
                        out[j["id"]] += (b - a) / len(act)
    return out, union


def make_tenant(name, jobs, D, **meta):
    """jobs: dicts with id, gpu, s, e, needed. Adds each job's len (its share) and the box's exact split."""
    share, union = attribute(jobs)
    for j in jobs:
        j["len"] = share[j["id"]]
    W = max([D] + [j["e"] for j in jobs])
    last = max((j["e"] for j in jobs), default=0.0)
    need = sum(j["len"] for j in jobs if j["needed"])
    un = sum(j["len"] for j in jobs if not j["needed"])
    return dict(meta, name=name, D=D, W=W, jobs=jobs, cap=BOX * W, needed_s=need, unneeded_s=un, idle_s=BOX * W - union,
                tail_idle_s=BOX * max(0.0, W - last), overlap_s=sum(j["e"] - j["s"] for j in jobs) - union)


def trigger(js, t, skip=None):
    """The job among js whose end most recently preceded time t (the result the agent last saw), or None."""
    c = [k for k in js if k is not skip and k["e"] <= t]
    return max(c, key=lambda k: k["e"]) if c else None


def simulate(tenants, P, backfill=True, drop_unneeded=False):
    """Replay the tenants' jobs on a P-GPU pool. Returns (per-tenant results, pool totals)."""
    jobs, deps, trig_of = [], {}, {}
    for ti, t in enumerate(tenants):
        js = [j for j in t["jobs"] if not (drop_unneeded and not j["needed"])]
        for j in js:
            u = (ti, j["id"])
            jobs.append(dict(j, uid=u, ti=ti, w=j.get("w", 1)))
            k = trigger(js, j["s"], skip=j)
            trig_of[u] = (ti, k["id"]) if k else None
            if k:
                deps.setdefault((ti, k["id"]), []).append(u)
        t["_js"] = js
    by = {j["uid"]: j for j in jobs}
    ev, seq = [], 0
    for j in jobs:
        if trig_of[j["uid"]] is None:
            heapq.heappush(ev, (j["s"], 1, seq, j["uid"])); seq += 1
    queue, running, free, end_at = [], [], P, {}

    def schedule(now):
        nonlocal free
        queue.sort(key=lambda u: (by[u]["submit"], u))
        while queue and by[queue[0]]["w"] <= free:
            start(queue.pop(0), now)
        if not queue or not backfill:
            return
        head = by[queue[0]]
        avail, shadow, extra = free, float("inf"), 0
        for e, w in sorted((by[u]["end"], by[u]["w"]) for u in running):
            avail += w
            if avail >= head["w"]:
                shadow, extra = e, avail - head["w"]
                break
        for u in list(queue[1:]):
            j = by[u]
            if j["w"] <= free and (now + j["len"] <= shadow or j["w"] <= extra):
                if now + j["len"] > shadow:
                    extra -= j["w"]
                queue.remove(u)
                start(u, now)

    def start(u, now):
        nonlocal free, seq
        j = by[u]
        j["start"], j["end"] = now, now + j["len"]
        free -= j["w"]
        running.append(u)
        heapq.heappush(ev, (j["end"], 0, seq, u)); seq += 1

    while ev:
        now = ev[0][0]
        while ev and ev[0][0] == now:
            _, kind, _, u = heapq.heappop(ev)
            j = by[u]
            if kind == 0:                                   # an end: its delay carries to the jobs it triggered
                running.remove(u)
                free += j["w"]
                end_at[u] = now
                shift = max(0.0, now - j["e"])
                for d in deps.get(u, []):
                    heapq.heappush(ev, (by[d]["s"] + shift, 1, seq, d)); seq += 1
            else:
                j["submit"] = now
                queue.append(u)
        schedule(now)
    res = []
    for ti, t in enumerate(tenants):
        D2 = t["D"] + max([0.0] + [end_at[(ti, j["id"])] - j["e"] for j in t["_js"] if j["e"] <= t["D"]])
        ends = [end_at[(ti, j["id"])] for j in t["_js"]]
        res.append(dict(name=t["name"], D=t["D"], D_pool=D2, ratio=D2 / t["D"] if t["D"] else None,
                        W_pool=max([D2] + ends), wait_s=sum(by[(ti, j["id"])]["start"] - by[(ti, j["id"])]["submit"]
                                                            for j in t["_js"])))
    H = max(r["W_pool"] for r in res) if res else 0.0
    busy = sum(j["len"] for j in jobs)
    tot = dict(P=P, H=H, busy_s=busy, needed_s=sum(j["len"] for j in jobs if j["needed"]),
               unneeded_s=sum(j["len"] for j in jobs if not j["needed"]), backlog_s=P * H - busy,
               util_real=busy / (P * H) if H else None, queued_s=sum(r["wait_s"] for r in res),
               starts={u: j["start"] for u, j in by.items()})
    return res, tot


def load(base, models):
    T, S = _mods()
    rows, _ = T.screen()
    sel = [r for r in rows if r["task"] in PACKAGES and r["arm"] == "L0" and r["validity"] == "valid"
           and not r.get("no_bar_reading") and r["model"] in models]
    out, wide = [], 0
    for r in sorted(sel, key=lambda r: (list(PACKAGES).index(r["task"]), models.index(r["model"]), r["run_id"])):
        d = os.path.join(base, r["run_id"])
        ref = T.reference_records(r["task"])
        raw, w = jobs_of(d)
        wide += w
        w0, _ = S.w0_makespan(r["task"], S.plan_key(json.load(open(os.path.join(d, "task.json")))))
        jobs = [dict(id=j, gpu=g, s=s, e=e, needed=rec in ref, work=wk) for j, g, s, e, rec, wk in raw]
        wall = sum(j["e"] - j["s"] for j in jobs)
        t = make_tenant(r["run_id"], jobs, r["over_ref"] * w0, task=r["task"], package=PACKAGES[r["task"]],
                        model=r["model"],
                        wall_unneeded_share=sum(j["e"] - j["s"] for j in jobs if not j["needed"]) / wall if wall else None)
        out.append(t)
    return out, wide


def h(x):
    return f"{x / 3600:,.1f}"


def main():
    global GPU_FROM, SHARE
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.expanduser("~/computewise-runs"))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--per-tenant", default="4,3,2")
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gpu-from", default=",".join(GPU_FROM),
                    help="occupancy events that name a job's GPU (default busy_start,hold,enqueue; busy_start alone is the "
                         "defect fixed 2026-11-02, kept for comparison)")
    ap.add_argument("--share", choices=("work", "equal"), default=SHARE,
                    help="a shared GPU's time: in proportion to each job's own work (default), or equal slices (the defect "
                         "fixed 2026-11-02, for comparison)")
    a = ap.parse_args()
    GPU_FROM, SHARE = tuple(a.gpu_from.split(",")), a.share
    if a.selftest:
        sys.exit(selftest())
    models = tuple(a.models.split(","))
    cs = [float(x) for x in a.per_tenant.split(",")]
    ten, wide = load(a.runs, models)
    N = len(ten)
    print(f"{N} tenants; jobs naming two GPUs: {wide}")
    print("\nCheck (tick_decomposition's split, wall GPU-s): G4 unneeded share, "
          + "; ".join(f"{t['model']} {t['wall_unneeded_share']:.3f}" for t in ten if t["package"] == "G4"))
    runs = {}
    for c in cs:
        for drop in (False, True):
            for bf in ((True, False) if (c == cs[-1] and not drop) else (True,)):
                runs[(c, drop, bf)] = simulate(ten, int(round(c * N)), backfill=bf, drop_unneeded=drop)
    c4 = runs.get((4.0, False, True))
    if c4:
        res, tot = c4
        ident = tot["backlog_s"] - (sum(t["idle_s"] for t in ten) + sum(BOX * (tot["H"] - t["W"]) for t in ten))
        print(f"c = 4 identity: backlog - (tenants' idle + capacity after they left) = {ident:.3f} GPU-s; "
              f"time queued {tot['queued_s']:.1f} s; deliverables moved: {sum(1 for r in res if abs(r['D_pool'] - r['D']) > 1e-6)}")
    lo = cs[-1]
    eq = [abs(x["D_pool"] - y["D_pool"]) < 1e-6 for x, y in zip(runs[(lo, False, True)][0], runs[(lo, False, False)][0])]
    print(f"EASY vs FCFS at {lo:g} GPUs per tenant: identical deliverables for {sum(eq)} of {len(eq)} tenants")

    # the loss split: of each tenant's own box, what a pool recovers (idle) and what it can't (unneeded work)
    fam = (("Claude", CLAUDE), ("Astra and Sol", ("astra", "sol")))
    lg = [(f"{pk}, {lab}", [t for t in ten if t["package"] == pk and t["model"] in ms]) for pk in PACKAGES.values() for lab, ms in fam]
    lg += [(f"all, {lab}", [t for t in ten if t["model"] in ms]) for lab, ms in fam] + [("all", ten)]
    print("\n| tenants | n | lost (idle + unneeded) | idle: a pool recovers it | unneeded: a pool can't | unneeded share of the loss |\n"
          + "|---" * 6 + "|")
    for name, g in lg:
        if not g:
            continue
        i, u, c = (sum(t[k] for t in g) for k in ("idle_s", "unneeded_s", "cap"))
        print(f"| {name} | {len(g)} | {(i + u) / c:.2f} | {i / c:.2f} | {u / c:.2f} | {u / (i + u):.2f} |")
    # the same split by model on the packages with unneeded work: within Claude,
    # and Astra in Claude Code (astra-cc, added with --models) on G4 only, as a harness contrast
    print("\n| package | model | n | lost (idle + unneeded) | idle: a pool recovers it | unneeded: a pool can't | "
          "unneeded share of the loss |\n" + "|---" * 7 + "|")
    for pk in ("G4", "G4n A", "G4n B"):
        for m in models:
            g = [t for t in ten if t["package"] == pk and t["model"] == m]
            if not g or (m == "astra-cc" and pk != "G4"):
                continue
            i, u, c = (sum(t[k] for t in g) for k in ("idle_s", "unneeded_s", "cap"))
            print(f"| {pk} | {m} | {len(g)} | {(i + u) / c:.2f} | {i / c:.2f} | {u / c:.2f} | {u / (i + u):.2f} |")
    groups = [(f"{pk} {m}", [i for i, t in enumerate(ten) if t["package"] == pk and t["model"] == m])
              for pk in PACKAGES.values() for m in models]
    groups = [g for g in groups if g[1]]
    groups += [("pooled, Claude", [i for i, t in enumerate(ten) if t["model"] in CLAUDE]),
               ("pooled, Astra and Sol", [i for i, t in enumerate(ten) if t["model"] in ("astra", "sol")]),
               ("pooled, all", list(range(N)))]
    hdr = ("| tenants | n | box GPU-h (4·W) | needed | unneeded (share of busy) | idle (share of box; of it, tail) | "
           + " | ".join(f"deliverable at {c:g}/tenant: median [max]" for c in cs)
           + f" | at {lo:g}/tenant, unneeded removed |")
    print("\n" + hdr + "\n" + "|---" * (7 + len(cs)) + "|")
    for name, ix in groups:
        g = [ten[i] for i in ix]
        cap, nd, un, idl, tail = (sum(t[k] for t in g) for k in ("cap", "needed_s", "unneeded_s", "idle_s", "tail_idle_s"))
        cells = []
        for c in cs + [lo]:
            rr = [runs[(c, len(cells) == len(cs), True)][0][i]["ratio"] for i in ix]
            cells.append(f"{statistics.median(rr):.3f} [{max(rr):.3f}]")
        print(f"| {name} | {len(g)} | {h(cap)} | {h(nd)} | {h(un)} ({un / (nd + un):.2f}) | {h(idl)} ({idl / cap:.2f}; "
              f"{h(tail)}) | " + " | ".join(cells) + " |")
    # the pool over the tenants' own windows: c GPUs per tenant until it leaves. Over [0, H] the pool also holds the GPUs
    # of tenants that already left (arrivals all at 0, none after), which is no agent's idle, so it is shown apart
    print("\n| pool | GPUs | tenants' capacity, c x their windows (GPU-h) | needed | unneeded | backlog in those windows | "
          "used by tenants | useful (needed + backlog) | tenant jobs' queue wait (GPU-h) | free after tenants left (GPU-h) |\n"
          + "|---" * 10 + "|")
    for (c, drop, bf), (res, tot) in runs.items():
        if not bf:
            continue
        win = c * sum(r["W_pool"] for r in res)
        bl = win - tot["busy_s"]
        print(f"| {c:g}/tenant{', unneeded removed' if drop else ''} | {tot['P']} | {h(win)} | {h(tot['needed_s'])} "
              f"({tot['needed_s'] / win:.3f}) | {h(tot['unneeded_s'])} ({tot['unneeded_s'] / win:.3f}) | {h(bl)} ({bl / win:.3f}) | "
              f"{tot['busy_s'] / win:.3f} | {(tot['needed_s'] + bl) / win:.3f} | {h(tot['queued_s'])} | {h(tot['backlog_s'] - bl)} |")
    if a.json:
        json.dump(dict(tenants=[{k: v for k, v in t.items() if not k.startswith("_") and k != "jobs"} for t in ten],
                       pools={f"{c:g}{'-cf' if d else ''}{'' if b else '-fcfs'}": dict({k: v for k, v in tot.items() if k != "starts"}, tenants=res)
                              for (c, d, b), (res, tot) in runs.items()}), open(a.json, "w"), indent=1)


def selftest():
    fails = []
    def ok(name, got, want, tol=1e-6):
        good = abs(got - want) <= tol
        print(f"{'ok  ' if good else 'FAIL'} {name}: {got:g} (want {want:g})")
        if not good:
            fails.append(name)
    # attribution: two jobs overlapping on one GPU split the shared 50 s
    sh, un = attribute([dict(id=1, gpu=0, s=0, e=100), dict(id=2, gpu=0, s=50, e=150)])
    ok("shared GPU: job 1 share", sh[1], 75); ok("shared GPU: job 2 share", sh[2], 75); ok("shared GPU: busy union", un, 150)
    # two tenants with known idle gaps on a pool of 4 GPUs each
    A = make_tenant("A", [dict(id=1, gpu=0, s=0, e=400, needed=True), dict(id=2, gpu=0, s=500, e=1000, needed=False),
                          dict(id=3, gpu=1, s=0, e=1000, needed=True), dict(id=4, gpu=2, s=0, e=600, needed=True)], 1000.0)
    B = make_tenant("B", [dict(id=1, gpu=0, s=0, e=800, needed=True), dict(id=2, gpu=1, s=100, e=300, needed=True),
                          dict(id=3, gpu=1, s=400, e=800, needed=True), dict(id=4, gpu=2, s=0, e=200, needed=False)], 800.0)
    ok("A idle (a gap of 100, GPU 2 free after 600 s for 400, an unused GPU 1000)", A["idle_s"], 1500)
    ok("B idle (gaps 100 + 100, 600, an unused GPU 800)", B["idle_s"], 1600)
    ok("A unneeded", A["unneeded_s"], 500); ok("B unneeded", B["unneeded_s"], 200)
    res, tot = simulate([A, B], 8)
    ok("pool of 8: backlog = idle 1500 + 1600 + B's 4 GPUs after it left (4 x 200)", tot["backlog_s"], 3900)
    ok("pool of 8: A's deliverable unchanged", res[0]["D_pool"], 1000); ok("pool of 8: B's deliverable unchanged", res[1]["D_pool"], 800)
    res, tot = simulate([A, B], 8, drop_unneeded=True)
    ok("unneeded removed: backlog gains exactly 500 + 200", tot["backlog_s"], 4600)
    # contention: two tenants, each 4 one-GPU jobs of 100 s at 0, the second with a dependent 50 s after its jobs end
    C = make_tenant("C", [dict(id=i, gpu=i, s=0, e=100, needed=True) for i in range(4)], 100.0)
    Dt = make_tenant("D", [dict(id=i, gpu=i, s=0, e=100, needed=True) for i in range(4)]
                     + [dict(id=9, gpu=0, s=150, e=160, needed=True)], 160.0)
    res, tot = simulate([C, Dt], 4)
    ok("pool of 4: C first, unchanged", res[0]["D_pool"], 100)
    ok("pool of 4: D's jobs wait 100 s, its dependent submits at 150 + 100, deliverable 260", res[1]["D_pool"], 260)
    ok("pool of 4: backlog = 4 x 260 - 810", tot["backlog_s"], 4 * 260 - 810)
    # EASY against FCFS on wide jobs: J1 holds 2 of 4 GPUs to 100 s, J2 wants all 4 from 1 s and is the queue's head, and
    # J3 (1 GPU, 50 s, from 2 s) fits before J2's reservation at 100 s, so EASY starts it at 2 s and FCFS at 110 s
    for bf, want in ((True, 2.0), (False, 110.0)):
        E = make_tenant("E", [dict(id=1, gpu=0, s=0, e=100, needed=True), dict(id=2, gpu=1, s=1, e=11, needed=True),
                              dict(id=3, gpu=2, s=2, e=52, needed=True)], 200.0)
        E["jobs"][0]["w"], E["jobs"][1]["w"] = 2, 4
        _, tot = simulate([E], 4, backfill=bf)
        ok(f"{'EASY' if bf else 'FCFS'}: the 1-GPU job starts at", tot["starts"][(0, 3)], want)
    # a job stacked onto a busy GPU: B joins GPU 0 while A holds it, so gpu-mock logs no busy_start for B
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ev = [dict(kind="job_hello", job=1, t_wall=1010, gpu=-1), dict(kind="busy_start", job=1, gpu=0, t_wall=1010),
              dict(kind="hold", job=1, gpu=0, t_wall=1010), dict(kind="enqueue", job=1, gpu=0, t_wall=1011),
              dict(kind="job_hello", job=2, t_wall=1050, gpu=-1), dict(kind="hold", job=2, gpu=0, t_wall=1050),
              dict(kind="enqueue", job=2, gpu=0, t_wall=1051), dict(kind="job_bye", job=1, t_wall=1110, gpu=-1),
              dict(kind="job_bye", job=2, t_wall=1150, gpu=-1)]
        open(os.path.join(td, "occupancy.jsonl"), "w").write("\n".join(json.dumps(e) for e in ev) + "\n")
        json.dump(dict(agent_start_wall=1000, teardown_wall=1200), open(os.path.join(td, "status.json"), "w"))
        json.dump(dict(score=dict(jobs=[dict(job=1, record_id="a"), dict(job=2, record_id="b")])),
                  open(os.path.join(td, "score.json"), "w"))
        got, _ = jobs_of(td)
        ok("stacked job: both jobs read", len(got), 2)
        ok("stacked job: B on GPU 0, 50 s to 150 s", sum(1 for j in got if j[0] == 2 and j[1] == 0 and j[2] == 50 and j[3] == 150), 1)
        legacy, _ = jobs_of(td, ("busy_start",))
        ok("busy_start alone drops B (the defect)", len(legacy), 1)
        sh, un = attribute([dict(id=j, gpu=g, s=s0, e=e0) for j, g, s0, e0, _, _ in got], share="equal")
        ok("stacked job, equal split: A 40 alone + 60 / 2", sh[1], 70); ok("stacked job, equal split: B 60 / 2 + 40 alone", sh[2], 70)
        ok("stacked job: GPU 0 busy 10 s to 150 s", un, 140)
    # the same stretch divided by each job's own work: A did 30 s of units, B 90 s, so of 140 s busy A gets 35, B 105
    sh, un = attribute([dict(id=1, gpu=0, s=10, e=110, work=30), dict(id=2, gpu=0, s=50, e=150, work=90)])
    ok("work split: A gets 140 x 30/120", sh[1], 35); ok("work split: B gets 140 x 90/120", sh[2], 105)
    ok("work split: the GPU's total is its busy time", sh[1] + sh[2], un)
    sh, _ = attribute([dict(id=1, gpu=0, s=0, e=100, work=40)])
    ok("a job alone keeps its wall time, whatever its work", sh[1], 100)
    sh, _ = attribute([dict(id=1, gpu=0, s=10, e=110, work=30), dict(id=2, gpu=0, s=50, e=150)])
    ok("a stretch with a job of unknown work falls back to equal split", sh[1], 70)
    print(f"{'PASS' if not fails else 'FAIL'}: {len(fails)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    main()
