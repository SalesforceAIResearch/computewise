"""Registry records from Hugging Face TRL's public W&B project (huggingface/trl), H100 runs.

usage: python3 -I gen_trl.py <W&B snapshot dir> <TRL script dir> <registry dir>

<W&B snapshot dir> holds one JSON per run, as fetched by fetch_trl.py beside this file:
run metadata, config, summary, full history and system events from W&B's public GraphQL API, and,
for each run that did not finish, <run>.output.log (its console log) or <run>.output.log.status.
<TRL script dir> holds each run's entry script fetched from github.com/huggingface/trl at the
run's recorded git commit, named `<codePath with / as _>@<commit[:8]>`.
A run is kept when W&B records its program and the script exists in the TRL repository at that
commit. Contract: ../RECORD-FORMAT.md.
"""
import collections
import datetime
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import trl_models as T  # noqa: E402

ROLE = {"train/loss": "train_loss", "train/grad_norm": "grad_norm", "train/learning_rate": "learning_rate",
        "eval/loss": "eval_loss", "eval/accuracy": "eval_accuracy", "ppo/loss/total": "train_loss",
        "ppo/learning_rate": "learning_rate"}
SKIP_KEYS = {"_step", "_runtime", "_timestamp", "train/global_step", "train/epoch", "game_log", "completions",
             "train/train_runtime", "train/train_samples_per_second", "train/train_steps_per_second", "train/total_flos",
             "train/train_loss", "eval/runtime", "eval/samples_per_second", "eval/steps_per_second"}
GENERATES = ("ppo", "rloo", "online_dpo", "dpo_online")


def cv(c, k):
    v = c.get(k)
    return v.get("value") if isinstance(v, dict) and "value" in v else v


def parse_ts(s):
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ") if s else None


def arg_val(args, name):
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def strip_seed(args):
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a == "--seed":
            skip = True
            continue
        if a.startswith("--seed="):
            continue
        out.append(a)
    return out


def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


TRAINER_OPTIM = {"adamw_torch": "AdamW", "rmsprop": "RMSprop"}  # Trainer.get_optimizer_cls_and_kwargs, read at transformers 4.36.2 and 4.44.2
NO_CKPT_SIZE = ("W&B logs no checkpoint sizes: a run's files hold only metadata and requirements, and its disk metrics cover only the root "
                "filesystem `/`, which stayed flat at 61.1 GB through the gemma-2-27b runs' saves at steps 500 and 1000, so checkpoints "
                "went to another filesystem")
EXC = re.compile(r"^(?:\[rank\d+\]:\s*)?((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception|Interrupt))(?::\s*(.*))?$")


def merge_events(ev):
    """W&B 0.18 writes one sample as several rows milliseconds apart, one per asset; merge rows under 1 s apart, dropping nulls."""
    rows = sorted((e for e in ev if num(e.get("_runtime"))), key=lambda e: e["_runtime"])
    out = []
    for e in rows:
        e = {k: v for k, v in e.items() if v is not None}
        if out and e["_runtime"] - out[-1]["_runtime"] < 1.0:
            for k, v in e.items():
                out[-1].setdefault(k, v)
        else:
            out.append(e)
    return out


def sdk_version(c):
    w = c.get("_wandb")
    w = w.get("value") if isinstance(w, dict) and "value" in w else w
    return w.get("cli_version") if isinstance(w, dict) else None


def vtuple(v):
    parts = re.findall(r"\d+", v or "")[:3]
    return tuple(int(x) for x in parts) if parts else (0,)


def failure_cause(snap, name):
    """The last exception line of the run's console log, as fetch_trl.py saved it."""
    if not snap:
        return None, "absent", "no snapshot directory given"
    p = os.path.join(snap, name + ".output.log")
    if os.path.exists(p):
        lines = open(p, errors="ignore").read().splitlines()
        hits = []
        for i, line in enumerate(lines):
            m = EXC.match(line.strip())
            if m:
                msg = (m.group(2) or "").strip()
                if not msg or re.fullmatch(r"\(Request ID:[^)]*\)", msg):
                    # the exception carries no message of its own; the server's explanation follows on the next lines
                    tail = []
                    for l in lines[i + 1:i + 4]:
                        if not l.strip() or l.startswith(("Traceback", "  File", "[rank")):
                            break
                        tail.append(l.strip())
                    msg = (msg + " " + " ".join(tail)).strip()
                hits.append(m.group(1) + (": " + msg if msg else ""))
        if hits:
            return hits[-1][:300], "recorded", "the last exception line in the run's console log (W&B output.log), with the line after it when the exception carried no message"
        return None, "absent", "the run's console log holds no exception line"
    if os.path.exists(p + ".status"):
        return None, "absent", f"W&B refuses the run's console log ({open(p + '.status').read().strip()})"
    return None, "absent", "the run's console log was not fetched"


def hub_target(c, ri):
    """Where a push_to_hub run commits: hub_model_id, else the Trainer default of the output_dir's basename under the author's namespace."""
    if not cv(c, "push_to_hub"):
        return None
    hid = cv(c, "hub_model_id")
    author = ((ri.get("program") or "").split("/") + ["", "", ""])[2]
    return hid if hid and "/" in hid else f"{author}:{hid or os.path.basename((cv(c, 'output_dir') or '').rstrip('/'))}"


def concurrent_runs(runs):
    """For each run: other runs whose createdAt-to-heartbeatAt span overlaps, by output_dir and by Hub target."""
    info = []
    for r in runs:
        c = json.loads(r.get("config") or "{}")
        a, b = parse_ts(r.get("createdAt")), parse_ts(r.get("heartbeatAt"))
        if a and b:
            info.append((r["name"], cv(c, "output_dir"), hub_target(c, r.get("runInfo") or {}), a, b))
    out = {}
    for n, od, hub, a, b in info:
        same = sorted(m for m, od2, _, a2, b2 in info if m != n and od and od2 == od and a2 <= b and a <= b2)
        hubs = sorted(m for m, _, h2, a2, b2 in info if m != n and hub and h2 == hub and a2 <= b and a <= b2)
        if same or hubs:
            out[n] = {"output_dir": same, "hub": hubs}
    return out


def classify_failure(cause, at, same_dir, same_hub):
    """the study's rule (design/IDEA.md): a collision counts only when the failure's cause matches the shared path."""
    c = cause or ""
    if re.search(r"tmp-checkpoint-\d+ does not exist", c) and same_dir:
        return {"class": "local_output_dir_race", "matched": True,
                "how": "the save failed because its temporary checkpoint directory vanished, and another run was writing the same output_dir"}
    if "412 Client Error: Precondition Failed" in c and "/commit/" in c and same_hub:
        return {"class": "hub_repo_push_race", "matched": True,
                "how": "the Hub rejected the commit because another commit landed first, and another run was pushing to the same repository"}
    if "LFS pointer pointed to a file that does not exist" in c and same_hub:
        return {"class": "hub_repo_push_race", "matched": False,
                "how": "the Hub rejected a push whose LFS pointer had no file; another run was pushing to the same repository, which fits a "
                       "race but does not prove one"}
    if same_dir or same_hub:
        return {"class": "unknown", "matched": False,
                "how": "another run shared this run's output_dir or Hub repository at the time, but the failure's cause is not logged or does "
                       "not mention the shared path, so the overlap proves nothing"}
    return None


def host_profile(c, ri, evm, runtime_s, n, args):
    sdk = sdk_version(c)
    core = vtuple(sdk) >= (0, 18)
    window = ("one reading every 10 s (wandb-core, the Go monitor from SDK 0.18)" if core else
              "the mean of 15 readings taken 2 s apart, published every 30 s (the Python monitor before SDK 0.18)")
    who = ("the process that called wandb.init, which is rank 0: HF's WandbCallback and accelerate's trackers log only from the main "
           "process, so the other ranks are not counted" if (n or 0) > 1 else "the training process")
    cpus = ri.get("cpuCount")
    h = {"wandb_sdk": C.tv(sdk, "recorded", "the W&B client version in the run's config") if sdk else C.absent(),
         "sample_window": window if evm else None,
         "cpus": C.tv(cpus, "recorded", "W&B runInfo.cpuCount: logical CPUs of the host") if cpus else C.absent()}
    rss = [(e["_runtime"], e["system.proc.memory.rssMB"]) for e in evm if num(e.get("system.proc.memory.rssMB"))]
    if rss and runtime_s:
        pts = [min(rss, key=lambda x: abs(x[0] - k / 10 * runtime_s)) for k in range(11)]
        h["rss_bytes_at_fraction"] = {
            "values": [int(v * 2 ** 20) for _, v in pts], "t_s": [round(t, 1) for t, _ in pts], "provenance": "recorded",
            "how": (f"W&B system.proc.memory.rssMB of {who}. Entry k is the sample nearest k x 10% of the run's {runtime_s:.0f} s wall "
                    f"time; t_s is when it was taken, and the first sample arrives {rss[0][0]:.0f} s in. Each sample is {window}")}
        t, v = max(rss, key=lambda x: x[1])
        h["process_rss_peak_bytes"] = C.tv(int(v * 2 ** 20), "recorded", f"the maximum of the same series, at {t:.0f} s"
                                           + ("" if core else "; a spike shorter than the 30-s window is averaged down"))
    else:
        h["rss_bytes_at_fraction"] = C.absent("no RSS samples in the run's system metrics")
        h["process_rss_peak_bytes"] = C.absent("no RSS samples")
    memk = "system.memory_percent" if any("system.memory_percent" in e for e in evm) else "system.memory"
    used, tot = [], []
    for e in evm:
        if core:
            r, pp, up = e.get("system.proc.memory.rssMB"), e.get("system.proc.memory.percent"), e.get(memk)
            if num(r) and num(pp) and num(up) and pp > 0:
                T = r / (pp / 100)
                tot.append(T)
                used.append((e["_runtime"], T * up / 100))
        else:
            a, p = e.get("system.proc.memory.availableMB"), e.get(memk)
            if num(a) and num(p) and 0 <= p < 100:
                tot.append(a / (1 - p / 100))
                used.append((e["_runtime"], a * p / (100 - p)))
    if used:
        T = C.median(tot)
        t, u = max(used, key=lambda x: x[1])
        if core:
            how = (f"host memory in use = total x {memk} / 100, with total = rssMB / proc.memory.percent, both at full precision; "
                   "gopsutil counts used memory as total - free - buffers - cached")
        else:
            how = (f"host memory in use = availableMB x p / (100 - p), where p is W&B {memk}: psutil's (total - available) / total, "
                   f"rounded to 0.1% before averaging, which leaves about +-{0.0005 * T / 1024:.0f} GiB")
        h["system_mem_used_peak_bytes"] = C.tv(int(u * 2 ** 20), "derived", how + f". Peak at {t:.0f} s. Every process on the host "
                                               "counts: all of this job's ranks and loader workers, and any other job on a shared host")
        h["host_mem_total_bytes"] = C.tv(int(T * 2 ** 20), "derived", "the median over samples of the total the same fields imply")
    else:
        h["system_mem_used_peak_bytes"] = C.absent("no host memory samples")
        h["host_mem_total_bytes"] = C.absent("no host memory samples")
    pc = [v for e in evm for v in [e.get("system.cpu")] if num(v)]
    one = f"; one fully busy core is {100 / cpus:.2f}%" if cpus else ""
    if pc and core:
        why = ("wandb-core reports gopsutil's CPUPercent, which is the process's CPU time since it started divided by its age: a lifetime "
               "average, so a median over samples is not a utilisation distribution")
        h["main_proc_cpu_pct_of_all_cores_median"] = C.absent(why)
        h["main_proc_cpu_pct_of_all_cores_p90"] = C.absent(why)
        h["main_proc_cpu_pct_of_all_cores_lifetime_mean"] = C.tv(round(pc[-1], 4), "recorded", f"the last W&B system.cpu sample of {who}: "
                                                                 f"CPU time since start / age / logical CPUs{one}. It is not the host's CPU use")
    elif pc:
        base = (f"W&B system.cpu of {who}: psutil Process.cpu_percent() over each 2-s interval divided by logical CPUs, averaged over "
                f"30 s{one}. It is not the host's CPU use")
        h["main_proc_cpu_pct_of_all_cores_median"] = C.tv(round(C.quantile(pc, 0.5), 4), "derived", "the median over samples of " + base)
        h["main_proc_cpu_pct_of_all_cores_p90"] = C.tv(round(C.quantile(pc, 0.9), 4), "derived", "the 90th percentile over samples of " + base)
        h["main_proc_cpu_pct_of_all_cores_lifetime_mean"] = C.tv(round(sum(pc) / len(pc), 4), "derived", "the mean over samples of " + base)
    else:
        for k in ("median", "p90", "lifetime_mean"):
            h[f"main_proc_cpu_pct_of_all_cores_{k}"] = C.absent("no CPU samples")
    hc = []
    for e in evm:
        per = [v for k, v in e.items() if k.startswith("system.cpu.") and k.endswith(".cpu_percent") and num(v)]
        if per:
            hc.append(sum(per) / len(per))
    if hc:
        how = ("mean over the host's logical CPUs of W&B system.cpu.<i>.cpu_percent, each CPU's busy share since the previous reading, "
               + ("per 10-s sample" if core else "averaged over 30 s") + "; every process on the host counts")
        h["host_cpu_util_pct_median"] = C.tv(round(C.quantile(hc, 0.5), 3), "derived", "the median over samples of the " + how)
        h["host_cpu_util_pct_p90"] = C.tv(round(C.quantile(hc, 0.9), 3), "derived", "the 90th percentile over samples of the " + how)
    else:
        h["host_cpu_util_pct_median"] = C.absent("no per-CPU samples")
        h["host_cpu_util_pct_p90"] = C.absent("no per-CPU samples")
    nw = cv(c, "dataloader_num_workers")
    h["dataloader_num_workers"] = (C.tv(nw, "recorded", "TrainingArguments.dataloader_num_workers; at 0 the training process loads data "
                                        "itself, so its RSS includes the loader") if nw is not None else C.absent("no Trainer loader setting"))
    dnp = arg_val(args, "--dataset_num_proc")
    h["dataset_num_proc"] = (C.tv(int(dnp), "recorded", "`--dataset_num_proc` in argv: processes for dataset preprocessing before training")
                             if dnp and dnp.isdigit() else C.absent("not set"))
    return h


def size_profile(c, stem, args):
    """Parameter counts and bytes per save, from the run's HF config, its optimizer and its precision flags."""
    out = {}
    base, how = T.causal_lm_params(c)
    model = cv(c, "model_name_or_path") or cv(c, "_name_or_path") or cv(c, "model_name")
    if base is None:
        out["model_parameters"] = C.absent("the run's config names no model" if not cv(c, "model_type") else how)
        return out, None, None
    hub = T.HUB.get(model)
    cross = f"; the Hub's safetensors count for {model} is {hub[0]:,}: {hub[1]}" if hub and hub[0] else (f"; {hub[1]}" if hub else "")
    n, hhow = T.head_params(stem, c, base)
    out["model_parameters"] = C.tv(n, "derived", f"{how}; {hhow}{cross}")
    pc = cv(c, "peft_config")
    pc = pc.get("default") if isinstance(pc, dict) and "default" in pc else pc
    td = arg_val(args, "--torch_dtype")
    wbytes = 2 if td in ("float16", "bfloat16") else 4
    wd_how = (f"weights in {td}, 2 bytes each, from --torch_dtype in argv" if wbytes == 2 else
              "weights in fp32, 4 bytes each: no --torch_dtype in argv, so the model loads in full precision and bf16/fp16 only autocast")
    optim = cv(c, "optim") or "adamw_torch"
    moments = {"adamw_torch": 2, "adamw_torch_fused": 2, "adamw_hf": 2, "rmsprop": 1, "sgd": 0, "adafactor": None}.get(optim)
    if moments is None:
        out["trainable_parameters"] = C.absent(f"optimizer {optim} has no state-size rule here")
        return out, None, None
    opt_how = {2: "AdamW keeps two moments per trainable parameter", 1: "RMSprop keeps one running average per trainable parameter",
               0: "SGD without momentum keeps no state"}[moments]
    if pc and isinstance(pc, dict) and pc.get("peft_type") == "LORA":
        r, targets = pc.get("r"), pc.get("target_modules") or []
        nt, lhow = T.lora_params(c, r, targets)
        if nt is None:
            out["trainable_parameters"] = C.absent(lhow)
            return out, None, None
        out["trainable_parameters"] = C.tv(nt, "derived", f"LoRA r={r}: {lhow}")
        abytes = 4
        weights = nt * abytes
        opt = nt * abytes * moments
        wd_how = ("the adapter alone is saved, in fp32: PEFT keeps LoRA weights in fp32 on a quantised or half-precision base")
    else:
        out["trainable_parameters"] = C.tv(n, "derived", "every parameter is trained")
        weights = n * wbytes
        opt = n * wbytes * moments
    small = "tokenizer, scheduler, RNG and trainer-state files, a few MB, are not counted"
    ckpt = C.tv(weights + opt, "derived", f"saved weights {weights:,} B ({wd_how}) + optimizer state {opt:,} B ({opt_how}, in the "
                                          f"weights' dtype); {small}")
    final = C.tv(weights, "derived", f"trainer.save_model writes the weights only: {wd_how}; {small}")
    return out, ckpt, final


def checkpoint_profile(c, code, steps, planned, status, stem=None, args=()):
    ss, st, lim, som, tfv = (cv(c, k) for k in ("save_strategy", "save_steps", "save_total_limit", "save_only_model", "transformers_version"))
    final = bool(re.search(r"trainer\.save_model\(", code))
    sizes, ckpt_b, final_b = size_profile(c, stem, args)
    a = {"output_dir": cv(c, "output_dir"), **sizes,
         "checkpoint_bytes": ckpt_b or C.absent("no parameter count, so no size"),
         "final_model_save": C.tv(final, "derived", "the script calls trainer.save_model(output_dir) after training, on top of any "
                                  "checkpoints" if final else "the script never calls trainer.save_model"),
         "final_model_bytes": (final_b if (final and final_b) else C.absent("no final save" if not final else "no parameter count"))}
    if ss is None:
        for k in ("save_strategy", "save_only_model", "save_every_steps", "save_total_limit", "checkpoint_saves"):
            a[k] = C.absent("no Trainer save settings in the run's config")
        return disk_at_end(a, final and status == "completed")
    a["save_strategy"] = C.tv(ss, "recorded", "TrainingArguments.save_strategy")
    a["save_only_model"] = (C.tv(som, "recorded", "TrainingArguments.save_only_model; false means each checkpoint also holds the "
                                 "optimizer and scheduler state") if som is not None else C.absent())
    if ss == "no":
        a["save_every_steps"] = {"value": None, "provenance": "recorded", "null_means": "no periodic checkpoints", "how": "save_strategy is `no`"}
        a["save_total_limit"] = C.absent("no periodic checkpoints")
        a["checkpoint_saves"] = C.tv(0, "derived", "save_strategy is `no`")
        return disk_at_end(a, final and status == "completed")
    if ss != "steps" or not num(st) or st <= 0:
        for k in ("save_every_steps", "save_total_limit", "checkpoint_saves"):
            a[k] = C.absent(f"save_strategy `{ss}` with save_steps {st} is not handled")
        return disk_at_end(a, final and status == "completed")
    every = None
    if st >= 1:
        every = int(st)
        a["save_every_steps"] = C.tv(every, "recorded", "TrainingArguments.save_steps")
    elif planned:
        every = math.ceil(planned * st)
        a["save_every_steps"] = C.tv(every, "derived", f"save_steps {st} is below 1, so the HF Trainer reads it as a fraction of the run's "
                                     f"steps: ceil({planned} x {st})")
    else:
        a["save_every_steps"] = C.absent(f"save_steps {st} is a fraction of an unknown step count")
    a["save_total_limit"] = (C.tv(lim, "recorded", "TrainingArguments.save_total_limit: older checkpoints beyond it are deleted")
                             if lim is not None else {"value": None, "provenance": "recorded", "null_means": "every checkpoint is kept",
                                                      "how": "TrainingArguments.save_total_limit is unset"})
    if every and steps:
        k = steps // every
        end = vtuple(tfv) >= (4, 42) and status == "completed" and steps % every != 0
        mid = status != "completed" and steps % every == 0
        a["checkpoint_saves"] = C.tv(k + int(end), "derived", f"{steps} steps // {every}"
                                     + (" + 1, because transformers 4.42 and later also save when training ends" if end else "")
                                     + (f"; the run ended on save step {steps}, so the count includes a save that may not have completed" if mid else "")
                                     + f"; transformers {tfv}")
    else:
        a["checkpoint_saves"] = C.absent("the step count or the save interval is unknown")
    return disk_at_end(a, final and status == "completed")


def disk_at_end(a, final_saved):
    saves, ckpt, lim, fin = a["checkpoint_saves"].get("value"), a["checkpoint_bytes"].get("value"), a.get("save_total_limit", {}), a.get("final_model_bytes", {})
    if saves is None or ckpt is None:
        a["disk_at_end_bytes"] = C.absent("no save count or no checkpoint size")
        return a
    kept = saves if lim.get("value") is None else min(saves, lim["value"])
    total = kept * ckpt + ((fin.get("value") or 0) if final_saved else 0)
    a["disk_at_end_bytes"] = C.tv(total, "derived", f"{kept} kept checkpoints x checkpoint_bytes"
                                  + (f" + final_model_bytes" if final_saved and fin.get("value") else "")
                                  + ("; save_total_limit deletes older checkpoints" if lim.get("value") is not None else "; no save_total_limit, so every checkpoint stays")
                                  + "; the peak on disk, since nothing else is deleted")
    return a


def build(run, script_dir, scripts_out, snap=None, overlaps=None):
    ri = run.get("runInfo") or {}
    cp, commit = ri.get("codePath"), (ri.get("git") or {}).get("commit")
    if not cp or not commit:
        return None, "no program or commit recorded"
    sp = os.path.join(script_dir, f"{cp.replace('/', '_')}@{commit[:8]}")
    if not os.path.exists(sp) or os.path.getsize(sp) == 0:
        return None, "script not in the TRL repository at the recorded commit"
    code = open(sp, errors="ignore").read()
    sha = C.sha256_text(code)
    base = os.path.basename(cp)
    stem = base[:-3]
    c = json.loads(run.get("config") or "{}")
    s = json.loads(run.get("summaryMetrics") or "{}")
    hist = [json.loads(x) if isinstance(x, str) else x for x in (run.get("history") or [])]
    ev = [json.loads(x) if isinstance(x, str) else x for x in (run.get("events") or [])]
    evm = merge_events(ev)
    args = list(ri.get("args") or [])
    n = ri.get("gpuCount")
    gpu = ri.get("gpu")
    if isinstance(gpu, str) and gpu.startswith("["):
        names = sorted({x.strip() for x in gpu.strip("[]").split(",") if x.strip()})
        gpu = names[0] if len(names) == 1 else gpu
    seed = cv(c, "seed")
    seed_how = "TrainingArguments.seed in the W&B config"
    if seed is None and arg_val(args, "--seed") is not None:
        seed, seed_how = int(arg_val(args, "--seed")), "`--seed` in the recorded argv"
    pdb = cv(c, "per_device_train_batch_size") or cv(c, "mini_batch_size")
    acc = cv(c, "gradient_accumulation_steps")
    gb = pdb * acc * n if (pdb and acc and n) else None
    lr = cv(c, "learning_rate")
    sched = cv(c, "lr_scheduler_type")
    warm = cv(c, "warmup_steps") or cv(c, "warmup_ratio")
    precision = "bf16" if cv(c, "bf16") else ("fp16" if cv(c, "fp16") else None)
    dataset = cv(c, "dataset_name") or arg_val(args, "--dataset_name")
    model = cv(c, "model_name_or_path") or cv(c, "_name_or_path") or cv(c, "model_name") or arg_val(args, "--model_name")
    xkey = "train/global_step" if any("train/global_step" in h for h in hist) else "_step"

    def series_of(key):
        pts = {}
        for h in hist:
            x, v = h.get(xkey), h.get(key)
            if num(x) and num(v):
                pts[int(x)] = float(v)
        if not pts:
            return None
        ks = sorted(pts)
        return {"steps": ks, "values": [pts[k] for k in ks], "provenance": "recorded", "how": f"W&B history key `{key}`, by `{xkey}`"}

    series = {}
    keys = sorted({k for h in hist for k in h if k not in SKIP_KEYS and not k.startswith("_")})
    for k in keys:
        sr = series_of(k)
        if sr:
            series[ROLE.get(k, k.replace("/", "."))] = sr
    wall = series_of("_runtime")
    if len(hist) >= 5000 and (run.get("historyLineCount") or 0) > len(hist):
        for v in series.values():
            v["how"] += f"; {len(hist)} of {run['historyLineCount']} rows, sampled by the W&B API"
    # pacing
    per_step = None
    if wall and len(wall["steps"]) >= 3:
        st, ws_ = wall["steps"], wall["values"]
        slopes = [(ws_[i + 1] - ws_[i]) / (st[i + 1] - st[i]) for i in range(1, len(st) - 1) if st[i + 1] > st[i]]
        per_step = C.median(slopes)
    steps = (wall["steps"][-1] if wall else None)
    if xkey == "train/global_step":
        gsv = [h.get("train/global_step") for h in hist if num(h.get("train/global_step"))]
        steps = int(max(gsv)) if gsv else steps
    max_steps = cv(c, "max_steps")
    planned, planned_how = None, None
    if max_steps and max_steps > 0:
        planned, planned_how = max_steps, "max_steps in the config"
    else:
        ep = [h.get("train/epoch") for h in hist if num(h.get("train/epoch"))]
        nep = cv(c, "num_train_epochs")
        if ep and nep and steps and max(ep) > 0:
            planned, planned_how = int(round(steps * nep / max(ep))), "steps x num_train_epochs / last logged epoch"
    created, hb = parse_ts(run.get("createdAt")), parse_ts(run.get("heartbeatAt"))
    rt = s.get("_runtime")
    wall_total = C.tv(round(rt, 3), "recorded", "W&B `_runtime`: process wall time from wandb.init to exit") if num(rt) and rt > 0 else (
        C.tv((hb - created).total_seconds(), "derived", "W&B heartbeatAt minus createdAt; `_runtime` is 0") if created and hb else C.absent())
    tr_rt = s.get("train_runtime") or s.get("train/train_runtime")
    if tr_rt is None:
        trs = [h.get("train/train_runtime") for h in hist if num(h.get("train/train_runtime"))]
        tr_rt = trs[-1] if trs else None
    first_x = wall["steps"][0] if wall else None
    setup = None
    if wall and per_step:
        setup = max(0.0, wall["values"][0] - per_step * first_x)
    # memory, utilisation, host
    def ev_max(k):
        v = [e[k] for e in ev if num(e.get(k))]
        return max(v) if v else None
    mem_pct = {g: ev_max(f"system.gpu.{g}.memoryAllocated") for g in range(n or 0)}
    mem_pct = {g: v for g, v in mem_pct.items() if v is not None}
    total = C.H100_80GB_TOTAL_BYTES if gpu and "H100 80GB" in str(gpu) else None
    peak_bytes = {g: int(v / 100 * total) for g, v in mem_pct.items()} if total else {}
    util = {}
    for g in range(n or 0):
        pts = [(e.get("_runtime"), e.get(f"system.gpu.{g}.gpu")) for e in ev if num(e.get(f"system.gpu.{g}.gpu")) and num(e.get("_runtime"))]
        if pts:
            util[str(g)] = {"t_s": [round(p[0], 1) for p in pts], "util_pct": [p[1] for p in pts]}
    state = run.get("state")
    status = {"finished": "completed", "crashed": "crashed", "failed": "failed", "killed": "killed"}.get(state, state)
    failures, blockers = [], []
    if status != "completed":
        last_step = steps
        at = "during_setup" if not hist else ("after_training" if tr_rt is not None else "during_training")
        cause, cprov, chow = failure_cause(snap, run["name"])
        f = {"mode": "killed" if (status == "killed" or (cause or "").startswith("KeyboardInterrupt")) else "crash",
             "step": last_step, "time_s": wall_total["value"], "at": at, "provenance": "recorded",
             "how": f"W&B run state `{state}` "
                    + {"crashed": "(the heartbeat stopped)", "failed": "(the process exited non-zero)", "killed": "(the run was interrupted)"}.get(state, "")
                    + f"; the last logged row is step {last_step}"
                    + ("; the trainer had logged train_runtime, so training itself finished" if at == "after_training" else ""),
             "cause": C.tv(cause, cprov, chow)}
        ov = (overlaps or {}).get(run["name"]) or {}
        if ov.get("output_dir"):
            f["concurrent_same_output_dir"] = {"runs": ov["output_dir"], "provenance": "derived",
                                               "how": "other runs in the project with the same output_dir whose createdAt-to-heartbeatAt spans overlap this run's"}
        if ov.get("hub"):
            f["concurrent_same_hub_repo"] = {"runs": ov["hub"], "provenance": "derived",
                                             "how": "other push_to_hub runs by the same author targeting the same repository (hub_model_id, or the "
                                                    "Trainer's default of the output_dir's basename) whose spans overlap this run's"}
        f["collision"] = classify_failure(cause, at, ov.get("output_dir"), ov.get("hub"))
        push = bool(re.search(r"huggingface_hub\.|huggingface\.co/api/|push was rejected|commit endpoint|push_to_hub", cause or ""))
        if (f["collision"] or {}).get("matched"):
            f["replay"] = {"value": "suppress", "how": "a collision in the original project is not the job's failure (design/IDEA.md, "
                                                       "\"Recorded collisions are not job failures\"); replay must not fire it"}
        elif at == "after_training" and push:
            f["replay"] = {"value": "suppress", "how": "the failure is in the Hub push after training finished; tasks never push to a Hub "
                                                       "(design/IDEA.md), so the push and its failure never happen in replay"}
        elif (f["collision"] or {}).get("class") == "unknown" and not cause:
            f["replay"] = {"value": "fire", "how": "the cause is unknown and the overlap proves nothing, so the failure stays the job's; "
                                                   "see group_pattern"}
        else:
            f["replay"] = {"value": "fire", "how": "the recorded failure is the job's own"}
        if f["replay"]["value"] == "suppress":
            if at == "after_training":
                f["replay"]["continuation"] = "complete"
                f["replay"]["continuation_how"] = "training had finished, so the record's series cover the whole job and replay runs it to completion"
            else:
                f["replay"]["continuation"] = "sibling_required"
                f["replay"]["continuation_how"] = (f"the run died at step {last_step} of {planned}, so the record holds no values past that point; "
                                                   "a job must be served by a sibling record of the same configuration instead")
                blockers.append(f"truncated by a suppressed collision at step {last_step} of {planned}: not drawable on its own; "
                                "the daemon serves a sibling of the same replicate group")
        failures.append(f)
    gen_based = stem in GENERATES or stem.startswith(GENERATES)
    needs = [r for r in ("train_loss", "grad_norm", "eval_loss") if r in series]
    serving = ({"mode": "results_at_end", "how": "the job samples text and scores it. Per-token values were not recorded, so the decode loop's "
                "reads (sampled tokens, end-of-sequence checks) must be answered so that each generation runs to its recorded length; the "
                "logged per-step series and result.* are then served as recorded"} if gen_based else
               {"mode": "per_op", "how": "every value the job reads has a recorded series or a derivation from one"})
    if not hist:
        needs = []
    config_basis = json.dumps({"script": sha, "argv": strip_seed(args), "trl_commit": commit, "gpus": n}, sort_keys=True)
    cfg_hash = C.sha256_text(config_basis)
    rid = f"trl-{C.slug(stem)}-{C.gpu_slug(gpu)}x{n}-{commit[:8]}-s{seed if seed is not None else 'na'}-{run['name']}"
    script_rel = f"artifacts/trl/scripts/{sha[:12]}/{base}"
    out = os.path.join(scripts_out, sha[:12], base)
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "w").write(code)
    lr_trace = series.get("learning_rate")
    trainer_lr = lr_trace is not None and any("train/learning_rate" in h for h in hist)
    align, align_prov, align_how = None, "absent", "no learning-rate trace"
    if trainer_lr:
        align, align_prov, align_how = "after_step", "inferred", (
            "HF Trainer logs learning_rate after lr_scheduler.step(): the value at global_step k is the LR optimizer step k+1 uses")
        w = cv(c, "warmup_steps")
        if w and w > 0 and lr and lr_trace["steps"][0] < w:
            k, v = lr_trace["steps"][0], lr_trace["values"][0]
            after, at = lr * k / w, lr * (k - 1) / w
            if abs(v - after) <= 1e-6 * abs(after) + 1e-12:
                align, align_prov = "after_step", "derived"
                align_how = f"first logged LR {v:.6g} at global_step {k} equals peak {lr:g} x {k}/{w} warmup steps"
            elif abs(v - at) <= 1e-6 * max(abs(at), 1e-30) + 1e-12:
                align, align_prov = "at_step", "derived"
                align_how = f"first logged LR {v:.6g} at global_step {k} equals peak {lr:g} x {k - 1}/{w}"
    elif lr_trace is not None:
        align, align_prov, align_how = None, "absent", "the trace is not the HF Trainer's; its alignment to optimizer steps is unverified"
    result = {}
    for k in ("train_loss", "eval_loss", "eval_accuracy", "train.rewards.accuracies", "train.rewards.margins", "env.reward_mean"):
        if k in series:
            result[k] = C.tv(series[k]["values"][-1], "recorded", f"last logged value, step {series[k]['steps'][-1]}"
                             + ("; the Trainer logs each value as the mean over its logging interval. The end-of-run mean over "
                                "every step, which the Trainer prints as train_loss, is result.trainer_train_loss"
                                if k == "train_loss" else ""))
    tl_mean = s.get("train/train_loss", s.get("train_loss"))
    if isinstance(tl_mean, (int, float)) and not isinstance(tl_mean, bool):
        result["trainer_train_loss"] = C.tv(tl_mean, "recorded", "the Trainer's end-of-run train_loss: the mean of the loss over "
                                            "every optimizer step, from the W&B summary (`train/train_loss`, or `train_loss` in later "
                                            "TRL versions). Not the last logged value, which is result.train_loss")
    tcfg = cv(c, "text_config")

    def cfgv(*ks):
        for k in ks:
            if cv(c, k) is not None:
                return cv(c, k)
        if isinstance(tcfg, dict):
            for k in ks:
                if tcfg.get(k) is not None:
                    return tcfg[k]
        return None
    optim = cv(c, "optim")
    arte = checkpoint_profile(c, code, steps, planned, status, stem, args)
    fi = {"model": cv(c, "model_type"), "num_layers": cfgv("num_hidden_layers", "n_layer"), "model_dim": cfgv("hidden_size", "n_embd"),
          "vocab_size": cfgv("vocab_size"), "optimizers": [TRAINER_OPTIM[optim]] if optim in TRAINER_OPTIM else None,
          "world_size": n, "per_device_batch": pdb, "accumulation": acc, "precision": precision or "fp32",
          "torch_compile": bool(cv(c, "torch_compile")), "model_name_or_path": model,
          "num_parameters": (arte.get("model_parameters") or {}).get("value"),
          "num_parameters_how": "fp256/0.3 field 11: artefacts.model_parameters, the root module's count derived from the architecture "
                                "(the base model for LoRA runs, before the peft wrapper)"}
    fp = fpv = None
    if fi["model"] and pdb and acc and n:
        fp, fpv = C.fingerprint(fi)
        fp_how = ("gpu-mock's reference implementation over fingerprint_inputs, read from the run's W&B config: the HF config's model_type "
                  "and its layer, width and vocabulary sizes, the Trainer's optimizer class, batch shape, world size, mixed-precision flag, "
                  "torch_compile, and (fp256/0.3) the model's parameter count from artefacts.model_parameters")
    else:
        fp_how = "not computed: the run's config lacks the model type or the batch settings"
    rec = {
        "schema": C.SCHEMA_RECORD,
        "identity": {
            "record_id": rid, "kind": "train",
            "entry_hash": sha, "argv": [base] + args, "argv_provenance": "recorded",
            "argv_how": "W&B runInfo.args, with the program's basename in front",
            "env": {}, "seed": C.tv(seed, "recorded" if seed is not None else "absent", seed_how),
            "replicate_group": "trl-cfg-" + cfg_hash[:12],
            "match_policy": "structural_verified" if align else "exact_only",
            "match_policy_how": ("the run logged its learning rate through the HF Trainer, so a structural match can be checked against the trace at run time"
                                 if align else "no learning-rate trace with a known alignment was logged, so only an exact match of script, argv and environment may replay"),
            "argv_exempt": ["--logging_steps", "--logging_first_step", "--report_to", "--output_dir", "--run_name", "--log_with",
                            "--push_to_hub", "--hub_model_id"],
            "argv_exempt_how": "arguments that change logging, output paths or the Hub push and not the trajectory; any other difference sends "
                               "the run to the modelled path. Tasks never push to a Hub (the study's ruling, design/IDEA.md), so the push flags are dropped",
            "trajectory": {
                "config_hash": cfg_hash, "config_hash_how": "sha256 of script bytes, argv without --seed, TRL commit and GPU count",
                "environment": {"trl_commit": commit, "python": ri.get("python"), "transformers": cv(c, "transformers_version"), "os": ri.get("os")},
                "lr_schedule": {"identity": f"{sched or 'n/a'}:lr={lr}:warmup={warm}", "source": "TrainingArguments in the W&B config",
                                "trace": {"steps": lr_trace["steps"], "values": lr_trace["values"],
                                          "axis": xkey, "alignment": align, "alignment_provenance": align_prov, "alignment_how": align_how,
                                          "per_group": None, "per_group_how": "one value applies to every param group",
                                          "tolerance": {"rel": 1e-6, "abs": 1e-12}} if lr_trace else None},
                "data": {"dataset": dataset, "how": "dataset_name in the config or argv; downloaded from the Hugging Face Hub"},
                "precision": C.tv(precision or "fp32", "recorded" if precision else "inferred",
                                  "bf16/fp16 flags in TrainingArguments" if precision else "neither bf16 nor fp16 set: the Trainer default"),
                "global_batch_sequences": gb,
                "model": model,
            },
            "fingerprint": fp, "fingerprint_version": fpv, "fingerprint_how": fp_how,
            "fingerprint_inputs": fi,
        },
        "mode": "replay",
        "source": {"name": "Hugging Face TRL W&B project", "project": "huggingface/trl", "run": run["name"], "display_name": run.get("displayName"),
                   "url": f"https://wandb.ai/huggingface/trl/runs/{run['name']}", "repo": "https://github.com/huggingface/trl",
                   "commit": commit, "program_on_author_cluster": ri.get("program"), "created_at": run.get("createdAt"),
                   "licence": "logs: none stated; code: Apache-2.0", "read_on": C.READ_ON,
                   "script_how": "fetched from the TRL repository at the run's recorded commit; uncommitted local changes, if any, are unknown"},
        "reference": {"gpu": gpu, "gpu_provenance": "recorded", "gpu_count": n, "gpu_count_provenance": "recorded",
                      "interconnect": None, "precision": precision or "fp32", "host_cpus": ri.get("cpuCount")},
        "launch": {
            "launcher": "python" if n == 1 else "accelerate launch",
            "launcher_provenance": "recorded" if n == 1 else "inferred",
            "world_size": C.tv(n, "recorded"),
            "per_device_microbatch": C.tv(pdb, "recorded" if pdb else "absent", "per_device_train_batch_size in the config"),
            "accumulation_steps": C.tv(acc, "recorded" if acc else "absent", "gradient_accumulation_steps in the config"),
            "global_batch_sequences": C.tv(gb, "derived" if gb else "absent", "per-device batch x accumulation x world size"),
            "batch_policy": C.tv("fixed_per_device", "derived", "HF Trainer multiplies the per-device batch by the world size; "
                                 "launching on fewer GPUs shrinks the global batch and lengthens the epoch in steps"),
            "gpu_counts_supported": None,
        },
        "steps": steps, "planned_steps": planned, "planned_steps_how": planned_how,
        "per_step_gpu_s": round(per_step, 6) if per_step else None,
        "timing": {
            "pacing": "per_logged_interval",
            "train_time_s": C.tv(round(tr_rt, 3), "recorded", "the Trainer's train_runtime") if num(tr_rt) else C.absent("no train_runtime logged"),
            "steady_step_s": C.tv(round(per_step, 6), "derived", f"median of wall-time slopes between logged rows, by `{xkey}`") if per_step else C.absent(),
            "setup_s": C.tv(round(setup, 3), "derived", "wall time at the first logged row minus steady step time x its step") if setup is not None else C.absent(),
            "eval_s": C.absent("evaluation time is inside the wall-time series"),
            "checkpoint_s": C.absent("inside the wall-time series"),
            "wall_total_s": wall_total,
            "wall_series": {"steps": wall["steps"], "values": wall["values"], "provenance": "recorded",
                            "how": f"W&B `_runtime` at each logged row, by `{xkey}`"} if wall else None,
        },
        "memory": {
            "peak_allocated_bytes": max(peak_bytes.values()) if peak_bytes else None,
            "peak": C.tv(max(peak_bytes.values()), "derived", "maximum over GPUs of W&B's NVML memoryAllocated % x 81559 MiB; "
                         "NVML counts the CUDA context and the allocator's reserve, so this is nearer reserved than allocated")
                    if peak_bytes else C.absent("no GPU memory samples"),
            "per_gpu_peak_pct": {str(g): v for g, v in mem_pct.items()},
        },
        "series": series,
        "observed": {"nvml_util": util, "how": "W&B system metrics, sampled every few seconds; calibration only, never served"},
        "materialisation": {"needs_series": needs, "blockers": blockers, "serving": serving,
                            "reads": ["the Trainer calls .item() on the accumulated loss and the grad norm at every logging step"]},
        "result": result,
        "outcome": {"status": status, "provenance": "recorded", "how": f"W&B run state `{state}`"},
        "failures": failures,
        "artefacts": arte,
        "host": host_profile(c, ri, evm, wall_total["value"], n, args),
        "agent_view": {"script": script_rel,
                       "launch_command": ("python " if n == 1 else f"accelerate launch --num_processes {n} ") + " ".join([base] + args),
                       "inputs": [x for x in (model, dataset) if x],
                       "recognisability": "medium: a public TRL example script and a public W&B run"},
    }
    return rec, None


def main():
    snap, script_dir, reg = sys.argv[1], sys.argv[2], sys.argv[3]
    out_dir = os.path.join(reg, "records")
    scripts_out = os.path.join(reg, "artifacts", "trl", "scripts")
    recs, skipped, runs = [], collections.Counter(), []
    for f in sorted(os.listdir(snap)):
        if not f.endswith(".json") or f.count(".") > 1:
            continue
        d = json.load(open(os.path.join(snap, f)))
        run = ((d.get("data") or {}).get("project") or {}).get("run")
        if not run:
            skipped["no run in snapshot"] += 1
            continue
        runs.append(run)
    overlaps = concurrent_runs(runs)
    for run in runs:
        rec, why = build(run, script_dir, scripts_out, snap, overlaps)
        if rec is None:
            skipped[why] += 1
            continue
        recs.append(rec)
    C.add_disclosure(recs)
    groups = collections.defaultdict(list)
    for r in recs:
        groups[r["identity"]["replicate_group"]].append(r)
    for r in recs:
        for f in r["failures"]:
            if (f["collision"] or {}).get("class") != "unknown" or f["cause"]["value"]:
                continue
            g = groups[r["identity"]["replicate_group"]]
            st = collections.Counter(x["outcome"]["status"] for x in g)
            steps = sorted({x["steps"] for x in g})
            f["group_pattern"] = {"runs": len(g), "statuses": dict(st), "last_steps": steps, "provenance": "derived",
                                  "how": ("every run of this configuration ended the same way at the same step, which fits a defect in the "
                                          "code at this commit rather than a collision" if len(g) > 1 and len(st) == 1 and len(steps) == 1 else
                                          "the configuration's other runs differ, so the pattern does not say what failed")}
    for r in recs:
        r["readiness"] = C.readiness(r)
        C.write_json(os.path.join(out_dir, r["identity"]["record_id"] + ".json"), r)
    print(f"trl: {len(recs)} records, {len({r['identity']['replicate_group'] for r in recs})} replicate groups")
    for k, v in skipped.items():
        print(f"  skipped: {k}: {v}")
    print("  by script x GPUs:", dict(collections.Counter((r['identity']['argv'][0], r['reference']['gpu_count']) for r in recs)))
    print("  outcomes:", dict(collections.Counter(r["outcome"]["status"] for r in recs)))
    print("  replay-ready:", sum(r["readiness"]["replay"] for r in recs))
    print("  missing:", dict(collections.Counter(m.split(":")[0] for r in recs for m in r["readiness"]["missing"])))
    print("  fingerprints:", sum(1 for r in recs if r["identity"]["fingerprint"]), "| failure causes from console logs:",
          sum(1 for r in recs for f in r["failures"] if f["cause"]["value"]), "of", sum(len(r["failures"]) for r in recs))
    print("  collisions:", dict(collections.Counter(((f["collision"] or {}).get("class"), (f["collision"] or {}).get("matched")) for r in recs for f in r["failures"])),
          "| replay:", dict(collections.Counter((f["replay"]["value"], f["replay"].get("continuation")) for r in recs for f in r["failures"])))
    print("  checkpoint_bytes:", dict(collections.Counter(r["artefacts"]["checkpoint_bytes"]["provenance"] for r in recs)))


if __name__ == "__main__":
    main()
