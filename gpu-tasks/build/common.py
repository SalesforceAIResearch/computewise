"""Shared helpers for the gpu-tasks registry generators. Stdlib only.

The contract these helpers serve is ../RECORD-FORMAT.md. validate.py enforces it.
"""
import hashlib
import json
import math
import os
import re

SCHEMA_RECORD = "p006-gpu-record/0.1"
SCHEMA_TASK = "p006-gpu-task/0.1"
READ_ON = "2026-10-19"

# How a value was obtained. RECORD-FORMAT.md §3 defines each term.
PROVENANCE = ("recorded", "derived", "inferred", "interpolated", "modelled", "absent")
BATCH_POLICIES = ("fixed_global_grow_microbatch", "fixed_global_grow_accumulation", "fixed_per_device")
FAILURE_MODES = ("oom", "xid", "nan", "hang", "crash", "killed")
MODES = ("replay", "model", "refuse")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def tv(value, provenance, how=None):
    """A value with its provenance."""
    assert provenance in PROVENANCE, provenance
    d = {"value": value, "provenance": provenance}
    if how:
        d["how"] = how
    return d


def absent(how=None):
    return tv(None, "absent", how)


def slug(s: str) -> str:
    s = s.lower().replace("_", "-").replace(".", "-")
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


GPU_SLUGS = {
    "NVIDIA H100 80GB HBM3": "h100-sxm",
    "NVIDIA H100 NVL": "h100-nvl",
    "NVIDIA H200": "h200",
    "NVIDIA A100-SXM4-80GB": "a100-sxm-80g",
    "NVIDIA L40S": "l40s",
    "NVIDIA GH200 480GB": "gh200-480g",
    "NVIDIA GH200 120GB": "gh200-120g",
    "NVIDIA B200": "b200",
    "NVIDIA A40": "a40",
    "NVIDIA H800": "h800",
}


def gpu_slug(name):
    if not name:
        return "unknown-gpu"
    return GPU_SLUGS.get(name, slug(name.replace("NVIDIA ", "")))


_SCALAR_LIST = re.compile(r"\[\s*\n([^\[\]{}]*?)\n\s*\]")


def expand_reference_plan(task):
    """An open task (goal-level, jobs with draw "open") as its scripted reference runs it: each reference_plan entry becomes
    that many instance jobs of the candidate's pool, at the entry's width, seeded where the script takes a seed. A task without
    a reference_plan is returned as it is."""
    plan = task.get("reference_plan")
    if not plan or not any(j.get("draw") == "open" for j in task["jobs"]):
        return task                              # not a goal-level task, or already expanded
    t = json.loads(json.dumps(task))
    by = {j["job_id"]: j for j in t["jobs"]}
    jobs, seen = [], {}
    # A plan of record (task.reference_plan.instances) or, with `which`, one of its named alternatives. An entry may carry
    # max_examples (an MLS-KV --max-examples k screen of the pool's record: paced at the load plus k of its n examples) and
    # after (the "<job_id> screen" entry it waits for: a confirm released when its own screen ends).
    for e in plan["instances"]:
        j0 = by[e["job_id"]]
        seeds = e.get("seeds") or [None] * e["count"]
        for sd in seeds:
            seen[e["job_id"]] = seen.get(e["job_id"], 0) + 1
            i = seen[e["job_id"]]
            j = {k: v for k, v in j0.items() if k not in ("instances", "launch")}   # disclosure kept: packed orders by it
            j["job_id"] = f"{e['job_id']}-{i}"
            if j0.get("launch", "").startswith("bash "):
                # a job script (MLS-Bench KV), not a torchrun line: the script itself, with the subset limit in front
                j["launch"] = (f"SELECTION_KV_MAX_EXAMPLES={e['max_examples']} ADAPTIVE_KV_MAX_EXAMPLES={e['max_examples']} "
                               if e.get("max_examples") else "") + j0["launch"]
                if e.get("max_examples"):
                    j["max_examples"] = e["max_examples"]
                j["phase"] = e.get("phase")
                j["gpu_options"] = [e["width"]]
                j["draw"] = "without_replacement"   # an expanded instance, not an open pool (so it is not expanded again)
                j["records"] = j0.get("reference_records") or j0["records"]
                if e.get("after"):
                    j["after"] = [f"{e['after'].split(' ')[0]}-1"]       # the pool's first instance is its screen
                jobs.append(j)
                continue
            j["launch"] = (f"torchrun --standalone --nproc_per_node={e['width']} {j0['script_path']}"
                           + (f" --seed {sd}" if sd is not None else ""))
            j["gpu_options"] = [e["width"]]
            j["draw"] = "exact_seed" if sd is not None else "without_replacement"
            j["seed"] = sd
            ref = j0.get("reference_records") or j0["records"]
            j["records"] = ([r for r in ref if j0.get("seeds_of", {}).get(r) == sd] or ref) if sd is not None else ref
            jobs.append(j)
    t["jobs"] = jobs
    return t


def write_text(path, text):
    """Write text to path unless it already holds exactly that; a changed file is replaced whole (temp file, then rename)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        with open(path) as f:
            if f.read() == text:
                return
    except (FileNotFoundError, UnicodeDecodeError):
        pass
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def write_json(path, obj):
    """Indented JSON with lists of scalars on one line, so a series is one line, not thousands. A file whose content would not
    change is left alone, mtime and all, and a changed one is replaced whole (temp file, then rename), so a peer copying the
    registry never reads a half-written file or sees an unchanged one move."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False)
    write_text(path, _SCALAR_LIST.sub(lambda m: "[" + re.sub(r",\n\s*", ", ", m.group(1).strip()) + "]", text) + "\n")


# Files in records/ that are not records (gpu-mock's r4, 2026-10-27): the KV example-order tables live beside the records because
# run_one mounts only records/ into the daemon's sidecar; they carry this schema and no identity, and every reader here skips them
NON_RECORD_SCHEMAS = ("p006-example-order/",)


def is_non_record(obj):
    return isinstance(obj, dict) and str(obj.get("schema", "")).startswith(NON_RECORD_SCHEMAS)


def load_records(records_dir):
    out = {}
    for name in sorted(os.listdir(records_dir)):
        if name.endswith(".json"):
            with open(os.path.join(records_dir, name)) as f:
                rec = json.load(f)
            if is_non_record(rec):
                continue
            out[rec["identity"]["record_id"]] = rec
    return out


# NVML total memory of an H100 80GB HBM3 (81559 MiB), the denominator of W&B's memoryAllocated %.
H100_80GB_TOTAL_BYTES = 81559 * 2**20


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def readiness(rec):
    """Whether gpu-mock can replay this record as it stands, and what is missing. RECORD-FORMAT.md §9."""
    miss = []
    ident, ref, t = rec["identity"], rec["reference"], rec["timing"]
    if not ident.get("entry_hash"):
        miss.append("identity.entry_hash: the exact script bytes are not materialised")
    if not rec["agent_view"].get("script"):
        miss.append("agent_view.script: the registry holds no script file for this job")
    if not ref.get("gpu") or ref.get("gpu_provenance") == "absent":
        miss.append("reference.gpu: the recorded hardware is unknown")
    if not ref.get("gpu_count"):
        miss.append("reference.gpu_count: unknown")
    if t["pacing"] in ("per_step", "per_logged_interval") and not rec.get("per_step_gpu_s"):
        miss.append("per_step_gpu_s: no step time")
    if t["pacing"] == "total" and t["wall_total_s"]["value"] is None:
        miss.append("timing.wall_total_s: no duration")
    for need in rec["materialisation"].get("needs_series", []):
        if need not in rec["series"]:
            miss.append(f"series.{need}: the job reads it and the source did not record it")
    miss.extend(rec["materialisation"].get("blockers", []))
    return {"replay": not miss, "missing": miss}


def add_disclosure(records, key="replicate_group"):
    """Per record: the true duration and the estimates an agent could plausibly be given (RECORD-FORMAT.md §8)."""
    groups = {}
    for r in records:
        groups.setdefault(r["identity"][key], []).append(r)
    for g, rs in groups.items():
        durs = [job_duration(r) for r in rs]
        durs = [d for d in durs if d is not None]
        for r in rs:
            d = job_duration(r)
            r.setdefault("disclosure", {})
            r["disclosure"]["duration"] = {
                "true_s": d,
                "true_how": job_duration_how(r),
                "group_median_s": round(median(durs), 3) if durs else None,
                "group_range_s": [round(min(durs), 3), round(max(durs), 3)] if durs else None,
                "group_n": len(durs),
                "group_how": "the same configuration's recorded runs, this one included: what a researcher who had run it before would expect",
            }


def job_duration(r):
    t = r["timing"]
    if t["wall_total_s"]["value"] is not None:
        return t["wall_total_s"]["value"]
    if t.get("train_time_s", {}).get("value") is not None:
        extra = sum((t.get(k, {}) or {}).get("value") or 0 for k in ("eval_s",))
        return round(t["train_time_s"]["value"] + extra, 3)
    return None


def subset_duration(r, k=None, gpu_count=None):
    """paced_duration for a launch of record r that evaluates only its first k examples (MLS-Bench KV, --max-examples k): the
    model load once, then k of the record's n_examples_full examples at (wall - load) / n each, as gpu-mock's per_example
    serving paces them (experiments/gpu-mock/docs/MLS-KV-SERVING.md §1.3). k None, 0 or at least n is the whole record."""
    d = paced_duration(r, gpu_count)
    sv = (r.get("materialisation") or {}).get("serving") or {}
    n = (sv.get("n_examples_full") or {}).get("value")
    if d is None or not k or not n or k >= n:
        return d
    load = (sv.get("load_s") or {}).get("value") or 0.0
    return load + k * max(0.0, d - load) / n


def paced_duration(r, gpu_count=None):
    """Recorded seconds gpu-mock's replay paces for record r on gpu_count GPUs (default: the recorded count), or None when the
    recorded curve does not apply at that count (gpu-mock's convert(): fixed_per_device off the recorded count).

    It follows experiments/gpu-mock/src/torch_patch/replay.py and gpu_daemon._serving_info, 2026-10-20: one unit per optimizer
    step, the first of them first_step_s when the record has one; startup_in_train_s before the first step only when there is
    no first_step_s; one unit per validation, eval_s split over the val_loss series' steps, paced only when both are known.
    Off the recorded count, convert() scales the steady step, the steady share of the first step and the validations, and
    leaves the one-off startup and the first step's excess as recorded. Process start, model loading, checkpoint writes and
    exit are real host time that replay does not pace, so this is shorter than the recorded wall time and is a floor under
    a replayed job's duration. Records without per-step pacing ("total") pace their recorded duration, scaled by the factor."""
    t = r["timing"]
    v = lambda k: t.get(k, {}).get("value") if isinstance(t.get(k), dict) else t.get(k)
    g0 = r["reference"].get("gpu_count")
    n = gpu_count or g0
    f, _ = time_factor((r["launch"].get("batch_policy") or {}).get("value"), g0, n) if g0 else (1.0, "identity")
    if f is None:
        return None
    steps, steady = r.get("steps"), v("steady_step_s") or r.get("per_step_gpu_s")
    if t.get("pacing") in ("per_step", "per_logged_interval") and steps and steady:
        first, su, ev = v("first_step_s"), v("startup_in_train_s") or 0.0, v("eval_s")
        vl = (r.get("series") or {}).get("val_loss")
        n_val = len(vl.get("steps") or []) if isinstance(vl, dict) else (len(vl) if isinstance(vl, list) else 0)
        d = 0.0 if first else su
        d += (max(0.0, first - steady) + steady * f) if first else steady * f
        d += (steps - 1) * steady * f
        d += ev * f if (ev and n_val) else 0.0
        return d
    d = job_duration(r)
    return None if d is None else d * f


def shard_load_steps(rec):
    """Optimizer steps (1-based, as gpu-mock counts them) before which a modded-nanogpt job loads its next training shard, after
    the first. The scripts' distributed_data_generator reloads when pos + batch + 1 >= the shard's tokens, pos advancing by the
    global batch, and step s (0-based) draws one batch before optimizer step s+1. [] for a record with no shard profile."""
    td = (rec.get("host") or {}).get("train_data") or {}
    sb = td.get("shard_bytes")
    b = (rec.get("launch") or {}).get("global_batch_tokens") or {}
    b = b.get("value") if isinstance(b, dict) else b
    if not sb or not b:
        return []
    # a resumed segment's loader first skips the batches earlier segments trained on (host.train_data.start_batch); loads
    # during that skip happen before its first paced unit, in its start-up
    first = int(td.get("start_batch") or 0)
    n = (int(sb) - 1024) // 2
    out, pos = [], 0
    for s in range(first + int(rec["steps"])):
        if pos + b + 1 >= n:
            if s > first:
                out.append(s - first + 1)
            pos = 0
        pos += b
    return out


GAP_CLASSES = ("once", "val_after", "val_before", "shard", "step")


def unit_kinds(values, val_flags):
    """A job's paced units as `step`, `val` or `other` (a one-off: the start-up unit, a first step of its own length). values are
    the units' modelled seconds; val_flags says which are validations where the caller knows it (the simulator), else None, and
    a validation is then the value that recurs as often as the record validates (build/derive_host_terms.py)."""
    import collections
    steps = [round(v, 9) for v, f in zip(values, val_flags or [False] * len(values)) if not f]
    step_v = collections.Counter(steps).most_common(1)[0][0] if steps else None
    return ["val" if (val_flags and f) else "step" if round(v, 9) == step_v else "other"
            for v, f in zip(values, val_flags or [False] * len(values))]


def gap_classes(kinds, shards):
    """The class of the gap before each unit i >= 1 (index 0 gets None), one rule for the measured runs and the simulator:
    beside a one-off unit `once`; after a validation `val_after`; before one `val_before`; before the optimizer step at which the
    loader reads its next shard `shard` (shards: common.shard_load_steps); else `step`."""
    out, step = [None], 1 if kinds and kinds[0] == "step" else 0
    for i in range(1, len(kinds)):
        prev, cur = kinds[i - 1], kinds[i]
        if cur == "step":
            step += 1
        if prev == "other" or cur == "other":
            out.append("once")
        elif prev == "val":
            out.append("val_after")
        elif cur == "val":
            out.append("val_before")
        elif cur == "step" and step in shards:
            out.append("shard")
        else:
            out.append("step")
    return out


def served_unit(r):
    """What a record serves, as (the recorded run it comes from, the parent steps whose values it shows, kind): a derived record's
    parent and range, else itself and its whole run. A resumed segment also shows the parent's validation at its start step (its
    own step 0). Two units overlap when they come from one run and their ranges meet, and an agent then sees the same losses
    twice; consecutive segments of one run are the exception (overlaps), since a real resume shows that boundary value again."""
    d = r["identity"].get("derived_from")
    if d:
        lo, hi = int(d["steps"][0]), int(d["steps"][1])
        if d.get("kind") == "segment" and lo > 1:
            lo -= 1
        return d["record"], (lo, hi), d.get("kind")
    return r["identity"]["record_id"], (1, int(r.get("steps") or 0)), None


def overlaps(ua, ub):
    """Whether two served units show the same values (served_unit). Two segments of one run overlap only if their trained steps
    do; sharing the boundary validation is a resume, not a repeat."""
    if ua[0] != ub[0]:
        return False
    (a0, a1), (b0, b1) = ua[1], ub[1]
    if ua[2] == ub[2] == "segment":
        a0, b0 = a0 + (1 if a0 > 1 else 0), b0 + (1 if b0 > 1 else 0)
    return a0 <= b1 and b0 <= a1


def repeated_records(task, recs):
    """Why a model run of this task could serve one recorded run, or part of one, to two jobs (the study, 2026-10-20, from the
    p6 pilot: an agent that reads its logs sees repeated losses, so no task that can repeat a record goes to a model run;
    design/IDEA.md, Detectability). [] when every job's draw is distinct from every other's:
    - a seed-substituted job draws another seed's record, so it repeats one;
    - two jobs whose possible records overlap (served_unit) can be served the same values, except jobs of one replicate group
      drawn without replacement from a pool at least as large as their number, which the daemon keeps apart."""
    out = []
    jobs = task["jobs"]
    for j in jobs:
        if j.get("draw") == "seed_substituted":
            out.append(f"{j['job_id']}: seed {j.get('seed')} has no recorded run, so it is served another seed's")
    pool = {j["job_id"]: [recs[x] for x in (j.get("records") or j.get("candidate_pool") or []) if x in recs] for j in jobs}
    by_group = {}
    for j in jobs:
        if j.get("draw") == "without_replacement":
            by_group.setdefault(j.get("replicate_group"), []).append(j["job_id"])
    for g, ids in by_group.items():
        n_pool = len({r["identity"]["record_id"] for i in ids for r in pool[i] if r["readiness"]["replay"]})
        if n_pool < len(ids):
            out.append(f"{', '.join(ids)}: {len(ids)} jobs drawn without replacement from {n_pool} records of {g}")
    for a in range(len(jobs)):
        for b in range(a + 1, len(jobs)):
            ja, jb = jobs[a], jobs[b]
            if (ja.get("draw") == jb.get("draw") == "without_replacement" and ja.get("replicate_group") == jb.get("replicate_group")):
                continue
            ba, bb = ja.get("bind") or {}, jb.get("bind") or {}
            if ba and bb and ba.get("decision") == bb.get("decision") and ba.get("slot") != bb.get("slot"):
                continue                       # two slots of one decision bind two different candidates by the rule
            if (ba and bb and ba.get("decision") == bb.get("decision") and ba.get("slot") == bb.get("slot")
                    and ba.get("run") and bb.get("run") and ba["run"] != bb["run"]):
                continue                       # one slot's runs k bind the bound candidate's k-th seed (build/gen_h1.py)
            hit = None
            for ra in pool[ja["job_id"]]:
                ua = served_unit(ra)
                for rb in pool[jb["job_id"]]:
                    ub = served_unit(rb)
                    if overlaps(ua, ub):
                        hit = (ra["identity"]["record_id"], rb["identity"]["record_id"])
                        break
                if hit:
                    break
            if hit:
                out.append(f"{ja['job_id']} and {jb['job_id']} can be served overlapping steps of one recorded run "
                           f"({hit[0]}, {hit[1]})")
    return out


def colleague_round(s):
    """Round a duration up to what a colleague would say: 5 minutes under an hour, 15 minutes under six hours, else whole hours."""
    m = s / 60
    step = 5 if m <= 60 else 15 if m <= 360 else 60
    return int(math.ceil(m / step - 1e-9) * step) * 60


def say_duration(s):
    h, m = divmod(int(s) // 60, 60)
    return " ".join(([f"{h} hour" + ("s" if h != 1 else "")] if h else []) + ([f"{m} minutes"] if m else []))


def job_duration_how(r):
    t = r["timing"]
    if t["wall_total_s"]["value"] is not None:
        return "timing.wall_total_s"
    return "timing.train_time_s + timing.eval_s; setup and process start are not included"


def quantile(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    i = q * (len(xs) - 1)
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


# gpu-mock's reference fingerprint (experiments/gpu-mock/docs/FINGERPRINT-CANONICALISATION.md). Loaded, never re-implemented,
# so a launched process and the record it matches hash identically.
GPU_MOCK_SRC = os.environ.get("GPU_TASKS_GPU_MOCK_SRC") or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gpu-mock", "src"))
# gpu-mock's code the build imports by path: the fingerprint, convert() and (for the expected scores) the Registry. Set
# GPU_TASKS_GPU_MOCK_SRC to a snapshot of gpu-mock's src/ when building away from the repository (the VM); the build
# records which fingerprint version it used, and the validator reports a version change.
# gpu-mock renamed its patch modules to neutral names on 2026-10-23 (fingerprint.py -> model_signature.py, byte-identical at
# fp256/0.3), so an agent listing the patch reads nothing telling; the build takes whichever name the snapshot has
FP_PATH = next((q for q in (os.path.join(GPU_MOCK_SRC, "torch_patch", n) for n in ("model_signature.py", "fingerprint.py"))
                if os.path.exists(q)), os.path.join(GPU_MOCK_SRC, "torch_patch", "model_signature.py"))
_FP = None


def fingerprint(inputs):
    global _FP
    if _FP is None:
        import importlib.util
        if not os.path.exists(FP_PATH):
            raise SystemExit(f"gpu-mock's fingerprint implementation is missing: {FP_PATH}")
        spec = importlib.util.spec_from_file_location("gpumock_fingerprint", FP_PATH)
        _FP = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_FP)
    return _FP.fingerprint(inputs), _FP.VERSION


CONVERT_PATH = os.path.join(GPU_MOCK_SRC, "convert", "convert.py")
_CV = None


def time_factor(batch_policy, recorded_gpus, gpu_count):
    """gpu-mock's convert(): the factor on a job's step time when it runs on gpu_count GPUs instead of recorded_gpus,
    and the provenance string. None means the recorded curve does not apply at that count (fixed_per_device), so the
    job would run on another record or the modelled path. Same function the daemon and the reference-schedule solver use."""
    global _CV
    if _CV is None:
        import importlib.util
        if not os.path.exists(CONVERT_PATH):
            raise SystemExit(f"gpu-mock's convert implementation is missing: {CONVERT_PATH}")
        spec = importlib.util.spec_from_file_location("gpumock_convert", CONVERT_PATH)
        _CV = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_CV)
    out = _CV.convert({"world_size_recorded": recorded_gpus, "batch_policy": batch_policy, "steady_step_s": 1.0}, gpu_count)
    return out["time_factor"], out["provenance"]


TASK_CLASSES = ("fan_out_required", "width_flexible", "control")


def task_class(jobs, box_gpus):
    """the study's three task classes (design/IDEA.md, "Reference schedules", 2026-10-19), read off the jobs' GPU options
    and, since the packages of 2026-10-20, their dependencies: a task whose jobs form one chain has no parallelism to find, so it
    is a control however narrow its jobs are."""
    lo = [min(j["gpu_options"]) for j in jobs]
    hi = [max(j["gpu_options"]) for j in jobs]
    ids = {j["job_id"] for j in jobs}
    heads = [j for j in jobs if not [d for d in j.get("after") or [] if d in ids]]
    succ = {}
    for j in jobs:
        for d in j.get("after") or []:
            succ.setdefault(d, []).append(j["job_id"])
    if len(jobs) > 1 and len(heads) == 1 and all(len(v) == 1 for v in succ.values()) and len(succ) == len(jobs) - 1:
        return "control", ("the jobs form one chain, each starting after the one before, so the best schedule runs them in order and "
                           f"{box_gpus - max(hi)} of the {box_gpus} GPUs idle by construction; what the task exposes is the gap at each "
                           "hand-off, and any job launched before its predecessor finished")
    if all(g == box_gpus for g in lo):
        return "control", f"every job needs all {box_gpus} GPUs, so the best schedule runs them one after another"
    if all(g < box_gpus for g in hi):
        return "fan_out_required", f"no job can use more than {max(hi)} of the box's {box_gpus} GPUs, so filling the box means running several jobs at once"
    return "width_flexible", (f"at least one job can take all {box_gpus} GPUs, so running one job at a time at full width is close to optimal "
                             "under linear conversion; the task can show idle GPUs within a job and gaps between launches, not a failure to fan out")


def get_dotted(d, dotted):
    for k in dotted.split("."):
        d = d.get(k) if isinstance(d, dict) else None
    return d
