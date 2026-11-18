"""Registry records from modded-nanogpt track 3 (the optimizer benchmark).

usage: python3 -I gen_track3.py <modded-nanogpt clone> <registry dir>

One record per recorded run. A run is a log under records/track_3_optimization/results/
that embeds its own script (the text before the first line of 100 '=' characters, as the
track's rules require) and prints validation lines `step:S/T val_loss:L train_time:Ts`.
Trustlight's seed runs are stdout captures without the script; their script is read from
the submission's reconstructed log. Duplicate logs (same script, same validation lines)
are kept once. Contract: ../RECORD-FORMAT.md.
"""
import collections
import datetime
import math
import os
import re
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

SEP = "\n" + "=" * 100
VAL = re.compile(r"^step:(\d+)/(\d+) val_loss:([\d.]+)(?: [a-z_]+:[\d.]+)* train_time:([\d.]+)s", re.M)
STEP1 = re.compile(r"^step:1/\d+ train_time:([\d.]+)s", re.M)
DEV = [re.compile(r"^Running on device_name=(.+?) with world_size=(\d+)\s*$", re.M),
       re.compile(r"^Running PyTorch .*? on (NVIDIA .+?) with world_size (\d+)\s*$", re.M)]
SEED_LINE = re.compile(r"^Using seed=(\d+)\s*$", re.M)
DATE_LINE = re.compile(r"^[A-Z][a-z]{2} [A-Z][a-z]{2} +\d+ \d\d:\d\d:\d\d UTC \d{4}$", re.M)
SAVE_LINE = re.compile(r"^(saved_checkpoint |saved_screening_checkpoint |\[fork\] dumped state|Saved checkpoint to )", re.M)
TARGET = 3.28
REPO = "https://github.com/KellerJordan/modded-nanogpt"
T3 = "records/track_3_optimization"
# track_1_short/2024-10-10_Muon logs print "peak memory consumption: 34239 MiB" for the same
# 12-layer, 768-wide GPT-2 at 64 x 1024 tokens per micro-step on one H100.
PEAK_MIB_T1 = 34239
SHARD_TOKENS = 100_000_000  # data/cached_fineweb10B.py: "Each chunk is 100M tokens"


def eval_int(expr, env):
    expr = expr.strip()
    m = re.fullmatch(r"""int\(os\.environ\.get\(\s*["'](\w+)["']\s*,\s*(?:str\((.+)\)|["'](\d+)["'])\s*\)\)""", expr)
    if m:
        return eval_int(m.group(2), env) if m.group(2) else int(m.group(3))
    out = []
    for p in re.split(r"(\W)", expr):
        if re.fullmatch(r"[A-Za-z_]\w*", p):
            if p not in env:
                return None
            out.append(str(env[p]))
        else:
            out.append(p)
    s = "".join(out)
    if "**" in s or not re.fullmatch(r"[\d\s*()+]+", s):
        return None
    try:
        return int(eval(s, {"__builtins__": {}}, {}))  # digits, *, +, () only
    except Exception:
        return None


def int_consts(code):
    env = {}
    for m in re.finditer(r"^([A-Za-z_]\w*)\s*=\s*([^#\n]+)", code, re.M):
        if m.group(1) in env:
            continue
        v = eval_int(m.group(2), env)
        if v is not None:
            env[m.group(1)] = v
    return env


def t3_artefacts(code, rest, saves, log_bytes):
    """Checkpoint and log disk use of one track-3 run, from what its log printed."""
    a = {}
    if not saves:
        a["checkpoint_saves"] = C.tv(0, "derived", "the script never calls torch.save")
        none = {"value": None, "provenance": "derived", "null_means": "the run writes no checkpoint", "how": "the script never calls torch.save"}
    else:
        n = len(SAVE_LINE.findall(rest))
        said = [l.strip() for l in rest.splitlines() if l.startswith("Using ") and re.search(r"(save_checkpoint\w*|checkpoint_save\w*)=", l)]
        how = (f"save lines the run printed: {n}. The script prints a line on every save; "
               + ("it printed its save settings as " + "; ".join(said) if said else
                  "it reads its save setting from an environment variable the log does not print"))
        a["checkpoint_saves"] = C.tv(n, "recorded" if n else "derived", how)
        none = {"value": None, "provenance": "derived", "null_means": "the run wrote no checkpoint", "how": how} if n == 0 else None
    if none:
        for k in ("checkpoint_bytes", "save_every_steps", "save_total_limit"):
            a[k] = dict(none)
    else:
        a["checkpoint_bytes"] = C.absent("the log does not give the checkpoint's size")
        a["save_every_steps"] = C.absent("the save step comes from an environment variable")
        a["save_total_limit"] = C.absent("the script keeps whatever it writes")
    a["log_bytes"] = (C.tv(log_bytes, "derived", "size of the published log; rank 0 writes the same text to logs/<uuid>.txt as it runs")
                      if log_bytes else C.absent("the published file is a console capture, not the run's log"))
    return a


def lr_setter(code):
    """The top-level function whose body assigns group["lr"], and its source. Line scan, no regex backtracking."""
    lines = code.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"def (\w+)\(", lines[i])
        if not m:
            i += 1
            continue
        j = i + 1
        while j < len(lines) and (not lines[j] or lines[j][0] in " \t#)"):
            j += 1
        body = "\n".join(lines[i:j])
        if re.search(r"""\[["']lr["']\]\s*=""", body):
            return m.group(1), body
        i = j
    return None, None


def script_name(code, dirpath, code_sha):
    m = re.match(r'\s*"""\s*\n\s*([\w.\-]+\.py)\s*\n', code)
    if m:
        return m.group(1), "inferred", "first line of the script's docstring"
    for f in sorted(os.listdir(dirpath)):
        if f.endswith(".py"):
            with open(os.path.join(dirpath, f), errors="ignore") as fh:
                if C.sha256_text(fh.read()) == code_sha:
                    return f, "recorded", "file of the same bytes in the submission directory"
    return "train_gpt_simple.py", "inferred", "the track's baseline script name; the log does not name its file"


def git_commit(clone):
    try:
        return subprocess.run(["git", "-C", clone, "log", "-1", "--format=%H"], capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def parse_val(rest):
    pts, seen = [], set()
    for st, tot, vl, tt in VAL.findall(rest):
        st = int(st)
        if st in seen:
            continue
        seen.add(st)
        pts.append((st, int(tot), float(vl), float(tt)))
    return pts


def wall_map(results_root):
    """(final val line) -> wall seconds, from stdout captures that print `date` before and after."""
    out = {}
    for dp, dn, fn in os.walk(results_root):
        for f in fn:
            p = os.path.join(dp, f)
            with open(p, errors="ignore") as fh:
                s = fh.read()
            if SEP in s:
                continue
            dates = DATE_LINE.findall(s)
            vals = re.findall(r"^step:\d+/\d+ val_loss:[\d.]+(?: [a-z_]+:[\d.]+)* train_time:[\d.]+s", s, re.M)
            if len(dates) >= 2 and vals:
                fmt = "%a %b %d %H:%M:%S UTC %Y"
                t0 = datetime.datetime.strptime(re.sub(r" +", " ", dates[0]), fmt)
                t1 = datetime.datetime.strptime(re.sub(r" +", " ", dates[-1]), fmt)
                out[vals[-1]] = ((t1 - t0).total_seconds(), os.path.relpath(p, results_root))
    return out


def unparseable(code):
    """A readiness blocker when the script's bytes do not parse: the recorded run cannot have executed them, so no launch may be
    served this record. Found 2026-10-21 by gpu-mock's probe: 20260506_trustlight's code, taken from the submitter's
    reconstructed log, ends with a log path the extraction ran into."""
    try:
        compile(code, "script", "exec")
        return []
    except SyntaxError as e:
        return [f"script does not parse: {e.msg} at line {e.lineno}; the recorded run cannot have executed these bytes"]


def build_record(*, code, rest, rel_log, dirpath, sub, commit, scripts_out, walls, extra=None):
    extra = extra or {}
    pts = parse_val(rest)
    code_sha = C.sha256_text(code)
    env = int_consts(code)
    dev = None
    for rx in DEV:
        m = rx.findall(rest)
        if m:
            dev = (m[-1][0].strip(), int(m[-1][1]))
            break
    if not dev and sub == "." and not extra:
        extra = dict(extra, gpu="NVIDIA H100 80GB HBM3", gpu_provenance="inferred", world_size=8, ws_provenance="inferred",
                     gpu_how="the track README's first results (2026/04/26, by its author) predate the device line; their steady "
                             "step time, 0.148-0.153 s, matches the 8x H100 logs of the same baseline script (0.147-0.155 s)")
    gpu, ws = (dev if dev else (extra.get("gpu"), extra.get("world_size")))
    gpu_prov = "recorded" if dev else extra.get("gpu_provenance", "absent")
    ws_prov = "recorded" if dev else extra.get("ws_provenance", "absent")
    base, base_prov, base_how = script_name(code, dirpath, code_sha)
    seed, seed_prov, seed_how = None, "absent", "the run does not print or fix a seed"
    m = SEED_LINE.search(rest)
    if m:
        seed, seed_prov, seed_how = int(m.group(1)), "recorded", "printed by the run: `Using seed=N`"
    else:
        m = re.search(r"^SEED\s*=\s*(\d+)\s*$", code, re.M)
        if m:
            seed, seed_prov, seed_how = int(m.group(1)), "recorded", "constant `SEED = N` in the script"
        elif "seed" in extra:
            seed, seed_prov, seed_how = extra["seed"], "recorded", extra["seed_how"]
    argparse_seed = bool(re.search(r"""add_argument\(\s*["']--seed["']""", code))
    env_seed = re.search(r"""os\.environ\.get\(\s*["'](\w*SEED\w*)["']""", code)
    argv = [base] + (["--seed", str(seed)] if argparse_seed and seed is not None else [])
    launch_env = {env_seed.group(1): str(seed)} if (env_seed and seed is not None and not argparse_seed) else {}
    # replicate group: same script once seed constants are blanked, same argv without the seed
    norm = re.sub(r"^SEED\s*=\s*\d+\s*$", "SEED = <seed>", code, flags=re.M)
    group_full = C.sha256_text(norm + "\0" + base)
    group = "mng3-cfg-" + group_full[:12]
    m = re.search(r"^Running PyTorch ([^\s]+)", rest, re.M)
    torch_ver = m.group(1) if m else None
    lr_fn, lr_body = lr_setter(code)
    if lr_fn:
        lr_id = "sha256:" + C.sha256_text(lr_body)[:16]
        lr_src = f"function `{lr_fn}` rewrites each param group's lr every step; its constants are in the script"
    else:
        lr_id, lr_src = None, "no function in the script assigns group['lr']; the optimizers' constructor LRs are constant"

    executed, planned = pts[-1][0], pts[-1][1]
    tt_final = pts[-1][3]
    later = [p for p in pts if p[0] >= 500] or pts[1:]
    start = later[0] if later and later[0][0] < executed else pts[1] if len(pts) > 2 else None
    steady = (tt_final - start[3]) / (executed - start[0]) if start and executed > start[0] else None
    first = next((p for p in pts if p[0] > 0), None)
    startup = max(0.0, first[3] - steady * first[0]) if (first and steady) else None
    s1 = STEP1.search(rest)
    batch_tokens = env.get("batch_size")
    mbs = env.get("mbs")
    m = re.search(r"def distributed_data_generator\([^)]*seq_len\s*=\s*(\d+)", code)
    seq_len = int(m.group(1)) if m else None
    accum = None
    if batch_tokens and mbs and seq_len and ws:
        local_seqs = batch_tokens // ws // seq_len
        accum = local_seqs // mbs if local_seqs % mbs == 0 else None
    val_tokens = env.get("val_tokens")
    n_eval = len(pts)
    eval_s = None
    if steady and val_tokens and batch_tokens:
        eval_s = n_eval * (val_tokens / batch_tokens) * steady / 3.0
    mm = re.search(r"^model\s*=\s*(GPT\([^\n]*?\))", code, re.M)
    model_ctor = mm.group(1) if mm else None
    nl = md = None
    if model_ctor:
        a = re.search(r"num_layers\s*=\s*(\w+)", model_ctor)
        b = re.search(r"model_dim\s*=\s*(\w+)", model_ctor)
        nl = (int(a.group(1)) if a.group(1).isdigit() else env.get(a.group(1))) if a else None
        md = (int(b.group(1)) if b.group(1).isdigit() else env.get(b.group(1))) if b else None
    optimizers = [c for _, c in re.findall(r"^(optimizer\w*)\s*=\s*([A-Za-z_][\w.]*)\(", code, re.M)]
    compiles_model = bool(re.search(r"torch\.compile\(\s*model\b", code))
    compiles_functions = "@torch.compile" in code
    saves = "torch.save" in code
    supported = [k for k in (1, 2, 4, 8) if 8 % k == 0] if "assert 8 % dist.get_world_size() == 0" in code else None

    script_rel = f"artifacts/mng3/scripts/{code_sha[:12]}/{base}"
    spath = os.path.join(scripts_out, code_sha[:12], base)
    if not os.path.exists(spath):
        os.makedirs(os.path.dirname(spath), exist_ok=True)
        with open(spath, "w") as fh:
            fh.write(code)

    stem = os.path.splitext(os.path.basename(rel_log))[0]
    if extra.get("stem"):
        stem = extra["stem"]
    rid = f"mng3-{C.slug(sub) if sub != '.' else 'root'}-{C.gpu_slug(gpu)}x{ws if ws else 'unk'}-{C.slug(stem)[:12]}"
    final_line = f"step:{executed}/{planned} val_loss:{rest_val_str(rest, executed)} train_time:"
    wall = extra.get("wall")
    if wall is None:
        for k, (w, src) in walls.items():
            if k.startswith(final_line) and k == final_val_line(rest):
                wall = (w, f"`date` before and after the run in {src}")
    peak = None
    if nl == 12 and md == 768 and mbs == 64 and seq_len == 1024:
        peak = C.tv(PEAK_MIB_T1 * 2**20, "inferred",
                    "track_1_short/2024-10-10_Muon logs report 34239 MiB peak allocated for the same 12-layer, "
                    "768-wide GPT-2 at 64 x 1024 tokens per micro-step on H100; this script does not log memory")
    shards = math.ceil(executed * batch_tokens / SHARD_TOKENS) if batch_tokens else None

    rec = {
        "schema": C.SCHEMA_RECORD,
        "identity": {
            "record_id": rid,
            "kind": "train",
            "entry_hash": code_sha,
            "argv": argv,
            "argv_provenance": "inferred",
            "argv_how": f"script name: {base_how}; --seed added when the script parses one and the run's seed is known",
            "env": launch_env,
            "seed": C.tv(seed, seed_prov, seed_how),
            "replicate_group": group,
            "match_policy": "exact_only",
            "match_policy_how": "the script's hyperparameters, schedule and data order are constants in its bytes, and it logs no LR trace, so only an exact byte match may replay",
            "trajectory": {
                "config_hash": group_full,
                "config_hash_how": "sha256 of the script bytes with any `SEED = N` line blanked, plus the script name",
                "environment": {"torch": torch_ver},
                "lr_schedule": {"identity": lr_id, "source": lr_src, "trace": None,
                                "trace_how": "absent: the script never prints its learning rate"},
                "data": {"dataset": "kjj0/fineweb10B-gpt2 (GPT-2 tokens of FineWeb 10B)", "train_glob": "data/fineweb10B/fineweb_train_*.bin",
                         "order": "files sorted by name, read sequentially, rank r takes its slice of each global batch",
                         "train_tokens_consumed": (executed * batch_tokens) if batch_tokens else None,
                         "val_file": "data/fineweb10B/fineweb_val_000000.bin", "val_tokens": val_tokens},
                "precision": C.tv("bf16", "inferred", "the embedding is cast to bfloat16 and the logits to float32; no autocast"),
                "global_batch_tokens": batch_tokens,
            },
            "fingerprint": None,
            "fingerprint_inputs": {
                "model": (re.match(r"(\w+)\(", model_ctor).group(1).lower() if model_ctor else None),
                "num_layers": nl, "model_dim": md, "vocab_size": 50304 if model_ctor and "50304" in model_ctor else None,
                "optimizers": sorted({o.split(".")[-1] for o in optimizers}) or None,
                "micro_batch_shape": [mbs, seq_len] if mbs and seq_len else None,
                "world_size": ws, "global_batch_tokens": batch_tokens, "precision": "fp32", "torch_compile": compiles_model,
                "model_constructor": model_ctor, "compiles_functions": compiles_functions,
                "num_parameters": None,
                "num_parameters_how": "fp256/0.3 field 11, null: each track-3 script defines its own GPT class, with variants per script, "
                                      "and no count can be vouched for from the source alone. These records are exact_only, so the "
                                      "structural index never serves them, and a null here matches no runtime count",
            },
        },
        "mode": "replay",
        "source": {
            "name": "modded-nanogpt track 3", "repo": REPO, "commit": commit, "path": f"{T3}/results/{rel_log}",
            "submission": sub, "licence": "MIT", "read_on": C.READ_ON,
            "date": (re.match(r"(\d{8})", sub).group(1) if re.match(r"\d{8}", sub) else None),
        },
        "reference": {"gpu": gpu, "gpu_provenance": gpu_prov, "gpu_count": ws, "gpu_count_provenance": ws_prov,
                      "interconnect": None, "precision": "bf16", **({"gpu_how": extra["gpu_how"]} if extra.get("gpu_how") else {})},
        "launch": {
            "launcher": "torchrun", "world_size": C.tv(ws, ws_prov),
            "per_device_microbatch": C.tv(mbs, "recorded" if mbs else "absent", "sequences per micro-step, constant `mbs` in the script"),
            "sequence_length": seq_len,
            "accumulation_steps": C.tv(accum, "derived" if accum else "absent", "global batch / world size / sequence length / mbs"),
            "global_batch_tokens": C.tv(batch_tokens, "recorded" if batch_tokens else "absent", "constant `batch_size` in the script"),
            "batch_policy": C.tv("fixed_global_grow_accumulation", "derived",
                                 "global batch and micro-batch are constants; the loop accumulates len(inputs)//mbs micro-steps per step"),
            "gpu_counts_supported": supported,
        },
        "steps": executed,
        "planned_steps": planned,
        "per_step_gpu_s": round(steady, 6) if steady else None,
        "timing": {
            "pacing": "per_step",
            "train_time_s": C.tv(tt_final, "recorded", "the script's own clock at the last validation; it stops during validation"),
            "steady_step_s": C.tv(round(steady, 6) if steady else None, "derived" if steady else "absent",
                                  "slope of train_time between the first validation at or after step 500 and the last"),
            "startup_in_train_s": C.tv(round(startup, 3) if startup is not None else None, "derived" if startup is not None else "absent",
                                       "train_time at the first validation after step 0, minus steady step time x steps; mostly torch.compile"),
            "first_step_s": C.tv(float(s1.group(1)), "recorded", "per-step line for step 1") if s1 else C.absent("the log has no per-step lines"),
            "eval_s": C.tv(round(eval_s, 3) if eval_s else None, "modelled" if eval_s else "absent",
                           f"{n_eval} validations x (val_tokens / batch tokens) x steady step / 3, forward only"),
            "setup_s": C.absent("process start to training start is not logged"),
            "checkpoint_s": C.tv(0, "derived", "the script never calls torch.save") if not saves else C.absent("the script saves; duration not logged"),
            "wall_total_s": C.tv(round(wall[0], 1), "recorded", wall[1]) if wall else C.absent("the log has no wall-clock start and end"),
        },
        "memory": {"peak_allocated_bytes": (peak or C.absent())["value"], "peak": peak or C.absent("no memory line in the log")},
        "series": {
            "val_loss": {"steps": [p[0] for p in pts], "values": [p[2] for p in pts], "provenance": "recorded",
                         "how": "every validation line; the reduced value every rank formats"},
            "train_time_s": {"steps": [p[0] for p in pts], "values": [p[3] for p in pts], "provenance": "recorded"},
        },
        "materialisation": {
            "serving": {"mode": "per_op", "how": "the job reads only val_loss, which every log records"},
            "needs_series": ["val_loss"],
            "blockers": unparseable(code),
            "reads": ["val_loss every validation, all ranks (f-string before the rank check)"]
                     + (["isfinite(loss) per micro-step, all ranks"] if "isfinite(loss)" in code else []),
            "train_loss": C.absent("the script never prints the train loss; serve a finite value"),
        },
        "result": {
            "final_val_loss": C.tv(pts[-1][2], "recorded"),
            "target_val_loss": TARGET,
            "reached_target": C.tv(pts[-1][2] < TARGET, "derived", "final val loss below 3.28, the track's target"),
        },
        "outcome": {"status": "completed", "provenance": "derived",
                    "how": "the log reaches its last scheduled validation" if executed == planned else
                           "the run stops at its final scheduled validation before the nominal step count, by design of the script"},
        "failures": [],
        "artefacts": t3_artefacts(code, rest, saves, None if extra else len((code + SEP + rest).encode())),
        "host": {
            "processes": C.tv(ws, ws_prov, "torchrun starts one process per GPU"),
            "omp_threads_per_process": C.tv(1, "inferred", "torchrun's default OMP_NUM_THREADS=1, as trustlight's stderr shows"),
            "train_data": {"glob": "data/fineweb10B/fineweb_train_*.bin", "shard_bytes": 2 * SHARD_TOKENS + 1024,
                           "shards_read": shards, "provenance": "derived",
                           "how": "every rank reads each shard it reaches in full: ceil(steps x batch tokens / 100M)"},
            "val_data": {"file": "data/fineweb10B/fineweb_val_000000.bin", "tokens_read": val_tokens},
            "pinned_host_bytes_per_rank": C.tv(2 * SHARD_TOKENS, "derived", "one shard of uint16 tokens in pinned memory"),
        },
        "agent_view": {
            "script": script_rel,
            "launch_command": f"torchrun --standalone --nproc_per_node=<N> {' '.join(argv)}",
            "inputs": ["data/fineweb10B/ (download script data/cached_fineweb10B.py)"],
            "recognisability": "high: a public modded-nanogpt track-3 record, script and log online",
        },
    }
    return rec


def rest_val_str(rest, step):
    m = re.search(rf"^step:{step}/\d+ val_loss:([\d.]+)", rest, re.M)
    return m.group(1) if m else ""


def final_val_line(rest):
    v = re.findall(r"^step:\d+/\d+ val_loss:[\d.]+(?: [a-z_]+:[\d.]+)* train_time:[\d.]+s", rest, re.M)
    return v[-1] if v else None


def main():
    clone, reg = sys.argv[1], sys.argv[2]
    root = os.path.join(clone, T3, "results")
    commit = git_commit(clone)
    out_dir = os.path.join(reg, "records")
    scripts_out = os.path.join(reg, "artifacts", "mng3", "scripts")
    walls = wall_map(root)
    seen, ids = set(), set()
    counts = collections.Counter()
    recs = []
    for dp, dn, fn in sorted(os.walk(root)):
        dn.sort()
        for f in sorted(fn):
            if not f.endswith(".txt"):
                continue
            p = os.path.join(dp, f)
            with open(p, errors="ignore") as fh:
                s = fh.read()
            if SEP not in s:
                counts["skipped: no embedded script (stdout capture or summary)"] += 1
                continue
            code, rest = s.split(SEP, 1)
            if f.startswith("fake_log"):
                counts["skipped: reconstructed log (seed runs read instead)"] += 1
                continue
            if len(parse_val(rest)) < 3:
                counts["skipped: fewer than 3 validation lines"] += 1
                continue
            key = (C.sha256_text(code), tuple(VAL.findall(rest)))
            if key in seen:
                counts["skipped: duplicate of another log"] += 1
                continue
            seen.add(key)
            sub = os.path.relpath(dp, root)
            sub = sub.split(os.sep)[0] if sub != "." else "."
            recs.append(build_record(code=code, rest=rest, rel_log=os.path.relpath(p, root), dirpath=dp, sub=sub,
                                     commit=commit, scripts_out=scripts_out, walls=walls))
            counts["records from logs with embedded script"] += 1
    # trustlight: seed stdout captures + metadata; the script comes from the reconstructed log
    tl = os.path.join(root, "20260506_trustlight")
    fake = os.path.join(tl, "fake_log_from_seed0.txt")
    if os.path.exists(fake):
        with open(fake, errors="ignore") as fh:
            code = fh.read().split(SEP, 1)[0]
        base = os.path.join(tl, "idea_10e_attnproj_bounded_trust_floor_soapish")
        for sd in sorted(os.listdir(base)):
            d = os.path.join(base, sd)
            so, md = os.path.join(d, "stdout.txt"), os.path.join(d, "metadata.txt")
            if not (os.path.exists(so) and os.path.exists(md)):
                continue
            meta = dict(l.split("=", 1) for l in open(md).read().splitlines() if "=" in l)
            rest = open(so, errors="ignore").read()
            t0 = datetime.datetime.strptime(meta["started_at"], "%Y-%m-%dT%H:%M:%SZ")
            t1 = datetime.datetime.strptime(meta["finished_at"], "%Y-%m-%dT%H:%M:%SZ")
            extra = {"gpu": "NVIDIA H100 (variant not stated)", "gpu_provenance": "inferred",
                     "world_size": int(meta.get("nproc_per_node", 4)), "ws_provenance": "recorded",
                     "seed": int(meta["seed"]), "seed_how": "seed in the run's metadata.txt", "stem": sd,
                     "wall": ((t1 - t0).total_seconds(), "started_at and finished_at in the run's metadata.txt")}
            r = build_record(code=code, rest=rest, rel_log=os.path.relpath(so, root), dirpath=tl, sub="20260506_trustlight",
                             commit=commit, scripts_out=scripts_out, walls=walls, extra=extra)
            r["source"]["script_from"] = f"{T3}/results/20260506_trustlight/fake_log_from_seed0.txt (the submitter's reconstructed log)"
            r["reference"]["gpu_how"] = "the run's metadata says device=h100; PR #283 says 4x H100; the variant is not stated"
            recs.append(r)
            counts["records from trustlight seed captures"] += 1
    by_group = collections.defaultdict(list)
    for r in recs:
        by_group[r["identity"]["replicate_group"]].append(r)
    for g, rs in by_group.items():
        planned_reached = any(r["steps"] == r["planned_steps"] for r in rs)
        for r in rs:
            if r["steps"] < r["planned_steps"] and planned_reached:
                r["outcome"] = {"status": "truncated", "provenance": "derived",
                                "how": f"the log ends at step {r['steps']} of {r['planned_steps']} while other runs of the same script reach "
                                       f"{r['planned_steps']}; the cause is not stated"}
    C.add_disclosure(recs)
    for r in recs:
        fi = r["identity"]["fingerprint_inputs"]
        if fi.get("optimizers") and fi.get("model"):
            r["identity"]["fingerprint"], r["identity"]["fingerprint_version"] = C.fingerprint(fi)
            r["identity"]["fingerprint_how"] = ("gpu-mock's reference implementation over fingerprint_inputs. model is the root module's class "
                                                "name, lowercased; optimizers are the classes the script assigns to optimizer* variables; "
                                                "precision is fp32 because the script enables no autocast (it casts the embedding to bfloat16 "
                                                "by hand, which trajectory.precision records); torch_compile is whether the script compiles "
                                                "the model itself, not only helper functions")
        else:
            r["identity"]["fingerprint_how"] = ("not computed: the script's model or optimizer classes could not be read statically, so a hash "
                                                "would not equal the running job's. The record is exact_only, so the entry hash matches it")
        r["readiness"] = C.readiness(r)
    for r in recs:
        rid = r["identity"]["record_id"]
        n = 2
        while rid in ids:
            rid = f"{r['identity']['record_id']}-{n}"
            n += 1
        r["identity"]["record_id"] = rid
        ids.add(rid)
        C.write_json(os.path.join(out_dir, rid + ".json"), r)
    groups = collections.Counter(r["identity"]["replicate_group"] for r in recs)
    hw = collections.Counter((r["reference"]["gpu"], r["reference"]["gpu_count"]) for r in recs)
    print(f"track 3: {len(recs)} records, {len(groups)} replicate groups, commit {commit}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print("  hardware:", ", ".join(f"{g} x{w}: {n}" for (g, w), n in hw.most_common()))
    print("  reached target:", sum(r["result"]["reached_target"]["value"] for r in recs), "of", len(recs))
    print("  wall_total recorded:", sum(r["timing"]["wall_total_s"]["provenance"] == "recorded" for r in recs))
    print("  replay-ready:", sum(r["readiness"]["replay"] for r in recs), "| outcomes:",
          dict(collections.Counter(r["outcome"]["status"] for r in recs)))
    miss = collections.Counter(m.split(":")[0] for r in recs for m in r["readiness"]["missing"])
    print("  missing:", dict(miss))


if __name__ == "__main__":
    main()
