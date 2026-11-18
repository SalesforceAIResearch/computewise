"""Tasks for ComputeWise's GPU registry, assembled from the records gen_track3/gen_trl/gen_mlsbench wrote.

usage: python3 -I gen_tasks.py <registry dir>

A task is what one agent run is handed: a brief, the files it may read, and the jobs the brief
implies. Each job points at the records that can serve it. Durations, curves and results stay in
the records; a task adds the job graph, the GPU counts each job may use on the box, the decision
rule for rung-2 follow-ups, and what the agent may be told about durations. Contract: ../RECORD-FORMAT.md §9.
Also writes <registry dir>/CATALOG.md, a generated index.
"""
import collections
import glob
import json
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

BOX = {"gpus": 4, "gpu": "NVIDIA H100 80GB HBM3", "interconnect": "NVLink",
       "source": "gpu-mock's default box: experiments/gpu-mock/GPU-COUNT-CONVERSION-NOTE-2026-10-19.md §3"}
# package-search angle F: an 8-GPU box. A builder asks for it by returning
# "box": BOX8; run_one passes --gpus {box.gpus} to the daemon, and the task names its own image label so it runs only on the
# 8-GPU image
BOX8 = dict(BOX, gpus=8, source=("package-search angle F, 8 GPUs (the analysis 2026-10-31 13:58 MDT): gpu-mock's "
                                 "device-count release and isolation's 8-GPU box image, built from the 4-GPU one with GPUS=8"))
IMAGE_8GPU = {"mng3-packages": "mng3-packages-8gpu", "mlsbench-kv": "mlsbench-kv-8gpu", "k2-latepole": "k2-latepole-8gpu"}
# images built only at 8 GPUs, with no 4-GPU stack to be the twin of (K2's late pole, 2026-11-02: two Python environments in one image)
IMAGE_8GPU_ONLY = {"k2-latepole"}


def image_for_box(task):
    """An 8-GPU task's environment names the 8-GPU image's label, never a 4-GPU one."""
    if task["box"]["gpus"] == 8:
        lab = task["environment"]["image_label"]
        if lab["value"].endswith("-8gpu"):
            # a builder that already names the 8-GPU label; kept as it is
            return
        if lab["value"] not in IMAGE_8GPU:
            raise SystemExit(f"{task['task_id']}: no 8-GPU image for label {lab['value']!r}")
        if lab["value"] in IMAGE_8GPU_ONLY:
            how = (lab["how"] + "; its 8-GPU label (gpu-tasks, 2026-11-02, for K2's late pole): the image is built only at 8 GPUs, "
                   "and the label says so, so no 4-GPU line ever draws it")
        else:
            how = (lab["how"] + "; the 8-GPU image of the same stack (isolation, 2026-10-31 20:27Z): same packages, built with "
                   "GPUS=8, a label of its own so a 4-GPU line never draws it")
        task["environment"]["image_label"] = {"value": IMAGE_8GPU[lab["value"]], "how": how}
LINEAR_NOTE = ("Under gpu-mock's convert() for grow-accumulation jobs (step time scales with recorded GPUs / launched GPUs), running "
               "each job on all four GPUs one after another and running four 1-GPU jobs side by side are both close to the best schedule. "
               "What this task exposes is a job left on one GPU while three idle, and idle time between launches.")


ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REG_DIR = os.path.join(ROOT, "registry")
HOST_TERMS_DIR = os.environ.get("GPU_TASKS_HOST_TERMS_DIR") or os.path.join(ROOT, "survey", "host-terms")
def fleet_machine_type():
    """The GPU-task boxes' machine type: GPU_TASKS_MACHINE_TYPE, else the "gpu" entry of MACHINE_TYPES in
    experiments/gcp/fleet/fleet.py (the study's types by role, since 2026-10-20), else the older single FLEET_SPEC there.
    Read as text so nothing in fleet.py is imported or run."""
    env = os.environ.get("GPU_TASKS_MACHINE_TYPE")
    if env:
        return env
    p = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gcp", "fleet", "fleet.py"))
    if os.path.exists(p):
        text = open(p).read()
        for pat in (r'MACHINE_TYPES\s*=\s*\{.*?"gpu"\s*:\s*\{\s*"machine_type"\s*:\s*"([^"]+)"',
                    r'FLEET_SPEC\s*=\s*\{\s*"machine_type":\s*"([^"]+)"'):
            m = re.search(pat, text, re.S)
            if m:
                return m.group(1)
    raise SystemExit("no machine type: set GPU_TASKS_MACHINE_TYPE, or keep a \"gpu\" entry in MACHINE_TYPES in "
                     "experiments/gcp/fleet/fleet.py")


FLEET_TYPE = fleet_machine_type()
HUB_REVISIONS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "survey", "hub-revisions-2026-10-19.json"))
REQ_DROP = re.compile(r" @ |^-|^#|^trl[=<>]|^nvidia|^triton|^bitsandbytes|^deepspeed|^flash|^xformers|^vllm|^apex|^torchvision|^torchaudio|^llm-blender|^liger")
MODEL_FILES = ("config.json", "generation_config.json", "merges.txt", "vocab.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json")
TRL_ENV = {
    "trl-dpo-loss-variants-4gpu": {
        "requirements": "survey/trl-requirements/e5295792-run-ig7dh8vc.requirements.txt",
        "packages": ("torch", "transformers", "accelerate", "datasets", "peft", "tyro", "rich"),
        "hub": (("model", "gpt2"), ("dataset", "trl-internal-testing/hh-rlhf-helpful-base-trl-style"))},
    "trl-regression-battery": {
        "requirements": "survey/trl-requirements/0022af66-run-2c7oeleh.requirements.txt",
        "packages": ("torch", "transformers", "accelerate", "datasets", "peft", "tyro", "rich", "wandb"),
        "hub": (("model", "gpt2"), ("model", "facebook/opt-350m"), ("dataset", "Anthropic/hh-rlhf"),
                ("dataset", "timdettmers/openassistant-guanaco"))},
}
# The modded-nanogpt environment for the package tasks (the study's packages ruling, 2026-10-20, PACKAGES-2026-10-20.md).
# The records carry torch but no Python and no other pins, so these are chosen, each with its reason, not recorded:
#   torch 2.11.0: 180 of the 349 ready track-3 records ran 2.11.0 (cu130/cu128/cu129), the track's README quickstart says
#     `pip install torch==2.11`, and the packages draw only from those records.
#   Python 3.11.9: one of the two Pythons isolation's build.sh pins by digest; torch 2.11.0 has a CPU wheel for it. The repo's
#     Dockerfile uses 3.12.7 (for its own torch 2.10 stack); nothing in the scripts depends on the minor version.
#   numpy and torch's runtime dependencies: the newest final release before 2026-05-01, the records' earliest date, that
#     installs on Python 3.11 (PyPI, read 2026-10-20), except where a pinned package bounds it: setuptools under 82, because
#     torch 2.11.0 requires it, and mpmath 1.3.0, because sympy 1.14.0 requires mpmath<1.4 (isolation's image build hit the
#     conflict on 2026-10-20 with mpmath 1.4.1; the staging check had missed it, see cpu_lines below).
# The scripts import torch and the standard library only; numpy backs the loader's tokens.numpy(). The data is the Hub dataset
# the records trained on, at its last revision (2024-09-28): the shards are files in the agent's tree, fetched by
# build/stage_task.py and checked against the sha256 the Hub gives (survey/hub-files-kjj0-fineweb10B-gpt2-889765ea-2026-10-20.json),
# so the image needs no Hub client and the box no Hugging Face cache.
MNG3_ENV = {
    "python": "3.11.9",
    "packages": ("torch==2.11.0", "numpy==2.4.4", "setuptools==81.0.0"),
    "constraints": ("torch==2.11.0", "numpy==2.4.4", "setuptools==81.0.0", "filelock==3.29.0", "typing-extensions==4.15.0",
                    "sympy==1.14.0", "mpmath==1.3.0", "networkx==3.6.1", "jinja2==3.1.6", "markupsafe==3.0.3", "fsspec==2026.4.0"),
    "hub": {"repo": "kjj0/fineweb10B-gpt2", "revision": "889765ea1f903759787add96995d81171b632d0c", "kind": "dataset",
            "revision_how": "the dataset's last commit (2024-09-28, \"Upload fineweb_train_000084.bin\"); every recorded run is later",
            "shard_bytes": 200001024, "val_file": "fineweb_val_000000.bin"},
    "image_label": "mng3-packages",
}
# torch 2.11.0's runtime requirements (its PyPI metadata, Linux CUDA wheels excepted), named because torch is installed with
# --no-deps; the image's Dockerfile names the same six.
MNG3_TORCH_DEPS = ("sympy", "networkx", "jinja2", "filelock", "typing-extensions", "fsspec")
# The quick task is the battery without reward modelling: the same scripts, pins, cache and box image (the study, 2026-10-20,
# experiments/campaign/campaign-2026-10-20.html §4). Its environment is the battery's, and its image label is the battery's.
TRL_ENV["trl-regression-battery-quick"] = dict(TRL_ENV["trl-regression-battery"], image_label="trl-regression-battery")
# The SFT pilot packages run the battery's sft.py on the battery's image.
for _n in (3, 6):
    TRL_ENV[f"pilot-sft-p{_n}-par"] = dict(TRL_ENV["trl-regression-battery"], image_label="trl-regression-battery")
# What was run on 2026-10-19, and what it showed. Written only after the run; a task without an entry says so.
TRL_VERIFIED = {
    "trl-dpo-loss-variants-4gpu": (
        "2026-10-20 on the ComputeWise VM (build-host: x86_64, Ubuntu 24.04, no GPU), by build/check_staging.sh from the staging "
        "manifest alone: Python 3.11.9 (uv) installed with environment.install.cpu_lines, torch 2.4.0+cpu, trl not installed in the "
        "venv and imported from the task tree; build/stage_hub.py filled a fresh cache (678 MB) and re-opened both entries offline. "
        "With hub/ and modules/ read-only, datasets/ writable, the three offline flags and PYTHONPATH=<task root>: (start) the first "
        "line of runs.sh as one process with --use_cpu --max_steps 1, batch 2, accumulation 1 exited 0 in 814 s; (steps) each of the "
        "three lines in full under torchrun --nproc_per_node 4, with compute stubbed by build/stepcount, took 513 optimizer steps on "
        "every rank, the records' `steps`. Not run: gpu-mock's patched torch, the box's caps. accelerate launch --cpu "
        "--num_processes 4 started one process, not four, so the 4-rank checks used torchrun. On 2026-10-19 the start check also "
        "passed on the controller (macOS arm64, 284 s), after a wholly read-only HF_HOME had failed in load_dataset (see env_how)"),
    "trl-regression-battery": (
        "2026-10-20 on the ComputeWise VM (build-host: x86_64, Ubuntu 24.04, no GPU), by build/check_staging.sh from the staging "
        "manifest alone: Python 3.10.13 (uv) installed with environment.install.cpu_lines, torch 2.1.2+cpu, trl not installed in the "
        "venv and imported from the task tree; build/stage_hub.py filled a fresh cache (2.0 GB) and re-opened all four entries "
        "offline. With hub/ and modules/ read-only, datasets/ writable, the offline flags, WANDB_MODE=offline and "
        "PYTHONPATH=<task root>: (start) seed 1 of each script as one process with --use_cpu --max_steps 1, batch 2, accumulation 1 "
        "exited 0: dpo.py in 1718 s, reward_modeling.py in 150 s, sft.py in 23 s, while the other task's check shared the VM; "
        "(steps) each distinct command in full with compute stubbed by build/stepcount took the records' `steps`: dpo.py 1000, "
        "reward_modeling.py 143, sft.py 30 optimizer steps. Not run: seeds 2 and 3 (they differ only in --seed), gpu-mock's "
        "patched torch, the box's caps. On 2026-10-19 the start check also passed on the controller (macOS arm64)"),
}
TRL_VERIFIED["trl-regression-battery-quick"] = (
    "not staged or checked on its own. Its scripts, launch lines, pins, cache and image are the battery's, whose check of "
    "2026-10-20 covered dpo.py and sft.py (see trl-regression-battery's environment.verified); only reward_modeling.py is left "
    "out. gpu-mock replayed one seed-1 job of each script on torch 2.1.2 the same day and matched the records "
    "(SMOKE-TASKS-HANDOFF-2026-10-19.md, the battery's host time). A run of this task through run_one.py is still owed")


colleague_round = C.colleague_round


def host_measurements(task):
    """Every host-terms measurement for this task, by machine type: survey/host-terms/<task_id>.<machine type>.json, written by
    build/derive_host_terms.py from a speed-1 scripted run at the box's caps. The machine is the isolation boundary and the box
    takes the whole machine, so terms, F and start-ups hold for one
    machine type, its caps and its image; a measurement is never used for another type."""
    out = {}
    for path in sorted(glob.glob(os.path.join(HOST_TERMS_DIR, f"{task['task_id']}.*.json"))):
        mt = os.path.basename(path)[len(task["task_id"]) + 1:-5]
        m = json.load(open(path))
        for k in ("terms", "speed", "caps", "image", "run_dir", "measured_on", "source", "machine_type"):
            if k not in m:
                raise SystemExit(f"{path}: missing {k!r}")
        if m["machine_type"] != mt:
            raise SystemExit(f"{path}: machine_type {m['machine_type']!r} does not match its file name")
        if float(m["speed"]) != 1.0:
            raise SystemExit(f"{path}: host terms must come from a speed-1 run (design/IDEA.md: reported runs are at speed 1)")
        if not (m["image"] or {}).get("id"):
            raise SystemExit(f"{path}: the image digest the terms were measured on is missing")
        m["_rel"] = os.path.relpath(path, ROOT) if path.startswith(ROOT) else path
        out[mt] = m
    return out


def simulate(task, reg, terms, policy):
    """The scripted policy's runs with these host terms at speed 1, under a horizon no run reaches."""
    import types
    import expected_scores as X
    probe = json.loads(json.dumps(task))
    if probe.get("decisions") and all((d.get("recorded_outcome") or {}).get("winners") for d in probe["decisions"]):
        probe = X.bind_recorded_winners(probe, reg)      # each slot as the scripted measure runs it
    probe = C.expand_reference_plan(probe)
    probe["horizon_s"] = 100 * task["stats"]["serial_makespan_s"] + 86400
    sim = X.Sim(probe, reg, types.SimpleNamespace(speed=1.0, host_terms=terms), policy)
    end, killed = sim.run()
    return sim, X.score(sim, end, killed)[0]


def measured_fields(m):
    return {k: m[k] for k in ("machine_type", "run_dir", "measured_on", "caps", "image", "source") if k in m}


def environment_floor(task, reg, ms):
    """F per machine type, the environment floor (design/IDEA.md, "The environment floor", the study, 2026-10-20): the
    overlapping packed scripted policy's (packed_overlap, the runner's rule; a script's first job pays its cold start-up
    pre_s_first, later ones the warm pre_s_rest, when the terms measured them apart) makespan with that type's host terms at its
    box's caps, minus the paced lower bound. It sets nothing; it is the start-up
    no agent avoids on that machine, reported beside the scores, and a run's scores are read against its own type's floor."""
    lb = task["stats"]["lower_bound_makespan_s"]
    by = {}
    for mt, m in sorted(ms.items()):
        _, sr = simulate(task, reg, m["terms"], "packed_overlap")
        packed = sr["makespan_from_agent_start_recorded_s"]
        by[mt] = {"value": round(max(0.0, packed - lb), 1), "provenance": "derived", "host_terms": m["_rel"],
                  "packed_makespan_s": round(packed, 3), "measured": measured_fields(m)}
    return {"by_machine_type": by, "provenance": "derived" if by else "absent",
            "fleet_machine_type": FLEET_TYPE, "fleet_measured": FLEET_TYPE in by,
            "how": ("per machine type: the overlapping packed policy's makespan from agent start (packed_overlap: each successor's "
                    "start-up runs during its predecessor's GPU work, the runner's rule; build/expected_scores.py, speed 1) "
                    "with the host terms measured on that type at its box's caps and image (`measured`), minus "
                    "stats.lower_bound_makespan_s. The machine is the isolation boundary and the box takes the whole machine, so a "
                    "type's floor holds for that type alone; a run on a type with no entry here has no floor to be read against until "
                    "its measuring run lands (the study, 2026-10-20, relayed by the sandbox). host-terms files are "
                    f"survey/host-terms/<task_id>.<machine type>.json; the fleet's type today is {FLEET_TYPE}"
                    + ("" if by else ". Nothing is measured yet"))}


def startup_totals(task, reg, ms):
    """Every job's start-up as a serial agent pays it, per measured machine type: launch to first paced step, each script's
    first job cold and the rest warm, summed over the serial policy's runs with that type's terms."""
    out = {}
    for mt, m in sorted(ms.items()):
        sim, _ = simulate(task, reg, m["terms"], "serial")
        out[mt] = round(sum(r["units"][0]["start"] - r["launch"] for r in sim.runs.values() if r["units"]), 1)
    return out


def add_horizon(task, reg):
    """horizon_s = colleague_round(1.5 x (the serial makespan + every job's start-up)): the study's rule (design/IDEA.md,
    "No deadline in the brief", the study, 2026-10-20). Hidden from the agent: the brief states no deadline or time limit. It is
    the run's hard stop and the end of the scoring window when work is left, set so that a serial agent with no overlap finishes
    well inside it. With several machine types in the fleet, the start-up term is the largest total among the types measured, so
    the one stop is generous on every type; each type's own total is recorded. The earlier rule, deadline_s =
    colleague_round(1.5 x lower bound + F) stated in the brief, is retired."""
    st = task["stats"]
    if not st.get("lower_bound_makespan_s"):
        return
    ms = host_measurements(task)
    totals = startup_totals(task, reg, ms)
    serial = st["serial_makespan_s"]
    src = max(totals, key=totals.get) if totals else None
    su = totals[src] if src else 0.0
    provisional = FLEET_TYPE not in ms
    # The preparation arm: a job the agent must prepare first adds its preparation to what a serial agent does before the
    # job can run. The allowance is prep.horizon_s per job (PREP_HORIZON_S): generous for writing one script, since the
    # horizon is a hard stop and a too-short one would kill runs, not measure them.
    prep_total = (sum(float((j.get("prep") or {}).get("horizon_s") or 0) for j in task["jobs"])
                  + float((task.get("data_prep") or {}).get("horizon_s") or 0))
    task["horizon_s"] = colleague_round(1.5 * (serial + su + prep_total))
    task["horizon_terms"] = {
        "serial_makespan_s": serial, "startup_total_s": su,
        **({"prep_total_s": prep_total, "prep_how": "the sum of every job's prep.horizon_s, the allowance for preparing it"
                                                    + (", and data_prep.horizon_s, the shared data step's" if task.get("data_prep")
                                                       else "")}
           if prep_total else {}),
        "startup_provenance": "derived" if src else "absent",
        "startup_source_machine_type": src, "startup_by_machine_type": totals,
        "fleet_machine_type": FLEET_TYPE, "provisional": provisional,
        "startup_how": ("the serial scripted policy's start-ups (launch to the job's first paced step; a script's first job cold, later "
                        "ones warm), summed over every job, with each measured machine type's host terms (build/expected_scores.py, "
                        "speed 1); the term used is the largest among the types measured, so the horizon is generous on every type"
                        + (f". The fleet's type, {FLEET_TYPE}, is not measured yet, so this is provisional and expected to move "
                           "when its measuring run lands" if provisional else "")
                        + (". No type is measured, so the term is 0" if not src else "")),
        **({"measured": measured_fields(ms[src])} if src else {})}
    task["environment_floor"] = environment_floor(task, reg, ms)
    task["horizon_how"] = ("colleague_round(1.5 x (stats.serial_makespan_s + horizon_terms.startup_total_s)), the study's rule "
                           "(design/IDEA.md, \"No deadline in the brief\", the study, 2026-10-20): a serial agent with no overlap "
                           "finishes well inside it. Rounded up to 5 minutes under an hour, 15 minutes under six hours, whole hours "
                           "above. Never shown to the agent: it is the run's hard stop and the end of the scoring window when work "
                           "remains. In recorded seconds, like stats: a run with the speed knob divides it by the same factor. "
                           "One value for every machine type, from the largest measured start-up total (horizon_terms). "
                           + (f"Provisional: {FLEET_TYPE} is not measured yet. " if provisional else
                              f"{FLEET_TYPE} is measured; a new machine type, caps or image that moves it makes another task version. ")
                           + "Replaced deadline_s, which the brief stated, on 2026-10-20; runs under a brief that stated a deadline "
                             "are not pooled with runs under this one")
    task["work_remaining"] = {"rule": "jobs_not_completed",
                              "how": "at the horizon, every job without a completed run counts as remaining: not launched, still "
                                     "running, or failed and not rerun to completion. A decision-bound job counts once its slot is known; "
                                     "before that, each unfilled slot counts as one job"}
    if DEADLINE_WORDS.search(task["brief"]["text"]):
        raise SystemExit(f"{task['task_id']}: the brief states a deadline or time limit, which no brief may (the study, 2026-10-20)")


DEADLINE_WORDS = re.compile(r"\bwithin\s+(\d+|an?|one|two|three|four|five|six|ten|twelve|half)\b[^.]*\b(hour|minute|second|day)s?\b"
                            r"|\bdeadline\b|\btime limit\b|\bby (the )?end of\b|\bas (fast|quickly|soon) as\b|\bno more than \d+\s*(h|min|hour|minute)",
                            re.I)


def constraints_from(text):
    return [l for l in text.splitlines() if "==" in l and not REQ_DROP.search(l)]


def add_mng3_environment(task, recs):
    """environment for a modded-nanogpt package task: chosen pins (MNG3_ENV, with reasons) rather than recorded ones, no Hub
    cache (the shards are files in the tree), and the package image's label."""
    e = MNG3_ENV
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    cons_rel = "artifacts/mng3/env/torch-2.11.0-py3.11/constraints.txt"
    cons_abs = os.path.join(root, "registry", cons_rel)
    os.makedirs(os.path.dirname(cons_abs), exist_ok=True)
    C.write_text(cons_abs, "\n".join(e["constraints"]) + "\n")
    torches = sorted({(recs[x]["identity"]["trajectory"].get("environment") or {}).get("torch") for j in task["jobs"]
                      for x in (j.get("records") or [])})
    task["environment"] = {
        "python": {"value": e["python"], "provenance": "inferred",
                   "how": "the records carry no Python version; 3.11.9 is one of the two isolation's build.sh pins, torch 2.11.0 has a "
                          "CPU wheel for it, and the scripts use no version-specific feature (the repo's Dockerfile uses 3.12.7)"},
        "install": {"packages": list(e["packages"]), "constraints": cons_rel,
                    "line": "pip install -c constraints.txt " + " ".join(e["packages"]),
                    # The same two steps as the image's Dockerfile: torch's CPU wheel with --no-deps, then torch's runtime
                    # dependencies by name with the rest, all under the constraints. Until 2026-10-20 the first line let the
                    # installer choose torch's dependencies outside the constraints, so the staging check passed on versions
                    # the image would never hold and missed the sympy/mpmath conflict the image build then hit.
                    "cpu_lines": ["pip install --index-url https://download.pytorch.org/whl/cpu --no-deps torch==2.11.0",
                                  "pip install -c constraints.txt " + " ".join(p for p in e["packages"] if not p.startswith("torch=="))
                                  + " " + " ".join(MNG3_TORCH_DEPS)],
                    "how": f"torch 2.11.0 matches the records this task draws (their torch: {', '.join(t for t in torches if t)}) and "
                           "the track's README quickstart (`pip install torch==2.11`). Its runtime dependencies and numpy are pinned in "
                           "the constraints at their newest final release before 2026-05-01, the records' earliest date, that installs on "
                           "Python 3.11 (PyPI, read 2026-10-20), except setuptools under 82, because torch 2.11.0 requires it, and "
                           "mpmath 1.3.0, because sympy 1.14.0 requires mpmath<1.4. Chosen, not recorded: the records hold no "
                           "package list"},
        "source": None,
        "hub": [], "hub_bytes": 0,
        "hub_how": "none: the training data are the Hub dataset's files, placed in the tree by build/stage_task.py (files[] git_lfs entry)",
        "env": {},
        "env_how": "nothing: the scripts read data/fineweb10B/ relative to the working directory and write logs/<uuid>.txt there",
        "verified": "pending: the staging check on the VM (build/check_staging.sh) under these pins; gpu-mock's replay of a track-3 "
                    "script on torch 2.11 is gpu-mock's own check",
        "image_label": {"value": e["image_label"],
                        "how": "one p006.task-env label for every package task drawn from this image (the study, 2026-10-20), so one "
                               "image serves the whole package family"},
    }


# What the data-preparation arm adds to the mng3 image: fineweb.py's tokeniser and progress bar, and GPT-2's encoding files in a
# cache tiktoken reads offline. Pinned at what installed on the VM with the mng3 pins on 2026-10-21 (build/check_dataprep.sh).
MNG3_DATAPREP_PACKAGES = ("tiktoken==0.14.0", "tqdm==4.70.1")
MNG3_DATAPREP_CONSTRAINTS = ("tiktoken==0.14.0", "tqdm==4.70.1", "regex==2026.9.29", "requests==2.34.2", "certifi==2026.10.19",
                             "charset-normalizer==3.5.2", "idna==3.20", "urllib3==2.8.0")


def add_mng3_dataprep_environment(task, recs):
    """The mng3 environment plus what data/fineweb.py needs offline, on an image of its own (mng3-dataprep)."""
    add_mng3_environment(task, recs)
    env = task["environment"]
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    cons_rel = "artifacts/mng3/env/torch-2.11.0-py3.11-dataprep/constraints.txt"
    cons_abs = os.path.join(root, "registry", cons_rel)
    os.makedirs(os.path.dirname(cons_abs), exist_ok=True)
    C.write_text(cons_abs, "\n".join(MNG3_ENV["constraints"] + MNG3_DATAPREP_CONSTRAINTS) + "\n")
    inst = env["install"]
    inst["packages"] = inst["packages"] + list(MNG3_DATAPREP_PACKAGES)
    inst["constraints"] = cons_rel
    inst["line"] = "pip install -c constraints.txt " + " ".join(inst["packages"])
    inst["cpu_lines"] = inst["cpu_lines"][:1] + [inst["cpu_lines"][1] + " " + " ".join(MNG3_DATAPREP_PACKAGES)]
    inst["how"] += ("; tiktoken and tqdm, which data/fineweb.py imports, at the versions that installed beside these pins on the "
                    "VM on 2026-10-21. datasets, its third import, is left out: it loads from the Hub, which the box cannot "
                    "reach, and data/raw/ is plain JSONL")
    env["env"] = {"TIKTOKEN_CACHE_DIR": "/usr/local/share/tiktoken"}   # where the mng3-dataprep image's build fills it (isolation)
    env["env_how"] = ("tiktoken downloads GPT-2's vocab.bpe and encoder.json on first use; the image fetches them at build time "
                      "(python -c \"import tiktoken; tiktoken.get_encoding('gpt2')\" with TIKTOKEN_CACHE_DIR set), world-readable, and "
                      "its build.sh checks that get_encoding('gpt2') loads there with no network (the sandbox, "
                      "2026-10-21). The scripts still read data/fineweb10B/ relative to the working directory")
    env["image_label"] = {"value": "mng3-dataprep",
                          "how": "the mng3-packages image plus tiktoken, tqdm and the gpt2 cache: its own label, so the packages "
                                 "already measured on mng3-packages keep their image"}


def add_environment(task, recs):
    """environment: what the box must hold for the task's scripts to start offline."""
    if task.get("env_kind") == "mng3":
        task.pop("env_kind")
        return add_mng3_environment(task, recs)
    if task.get("env_kind") == "mng3-dataprep":
        task.pop("env_kind")
        return add_mng3_dataprep_environment(task, recs)
    spec = TRL_ENV.get(task["task_id"])
    if not spec:
        return
    rs = [recs[x] for j in task["jobs"] for x in (j.get("records") or [])]
    commits = {r["source"]["commit"] for r in rs}
    pys = {r["identity"]["trajectory"]["environment"].get("python") for r in rs}
    assert len(commits) == 1 and len(pys) == 1, (task["task_id"], commits, pys)
    commit, py = commits.pop(), pys.pop()
    since = min(r["source"]["created_at"] for r in rs)
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    req = open(os.path.join(root, spec["requirements"])).read()
    pins = dict(l.split("==", 1) for l in constraints_from(req))
    cons_rel = f"artifacts/trl/env/{commit[:12]}/constraints.txt"
    cons_abs = os.path.join(root, "registry", cons_rel)
    os.makedirs(os.path.dirname(cons_abs), exist_ok=True)
    C.write_text(cons_abs, "\n".join(constraints_from(req)) + "\n")
    install = [f"{p}=={pins[p]}" for p in spec["packages"]]
    revs = json.load(open(HUB_REVISIONS))["repos"]
    hub = []
    for kind, repo in spec["hub"]:
        r = revs[repo]
        before = [c for c in r["commits"] if c["date"] < since]
        rev = before[0]["id"]
        tree = r["trees"][rev]
        if kind == "model":
            weights = "model.safetensors" if "model.safetensors" in tree else "pytorch_model.bin"
            files = {f: tree[f] for f in MODEL_FILES + (weights,) if f in tree}
        else:
            files = dict(tree)
        hub.append({"kind": kind, "repo": repo, "revision": rev,
                    "revision_how": f"the last commit on main before the task's earliest recorded run ({since}): {before[0]['date'][:10]}, "
                                    f"\"{before[0]['title']}\" (survey/hub-revisions-2026-10-19.json)",
                    "files": files, "bytes": sum(files.values())})
    wandb = any("--report_to" in (j.get("launch") or "") and "wandb" in (j.get("launch") or "") for j in task["jobs"])
    task["environment"] = {
        "python": {"value": py, "provenance": "recorded", "how": "runInfo.python of the task's W&B runs"},
        "install": {"packages": install, "constraints": cons_rel,
                    "line": "pip install -c constraints.txt " + " ".join(install),
                    "cpu_lines": [f"pip install --index-url https://download.pytorch.org/whl/cpu torch=={pins['torch']}",
                                  "pip install -c constraints.txt " + " ".join(install)],
                    "cpu_how": "the same pins on a host without CUDA: torch's CPU wheel first, from PyTorch's CPU index (its +cpu "
                               "local version satisfies the ==pin), then the rest from PyPI, which keeps the installed torch. One "
                               "command with the CPU index as an extra index timed out on the ComputeWise VM on 2026-10-20, because it "
                               "queries that index for every package. build/check_staging.sh runs these lines",
                    "how": f"top-level pins and constraints from {spec['requirements']} (the recorded runs' requirements.txt), "
                           "minus trl (loaded from the task tree), the CUDA-only and x86-only wheels (nvidia-*, triton, bitsandbytes, "
                           "deepspeed, flash-attn, xformers, vllm, apex, liger), torchvision and torchaudio, and editable or URL lines. "
                           "The torch pin is the recorded one; whether gpu-mock's patch runs on it is gpu-mock's call"},
        "source": {"path": "trl/", "repo": "huggingface/trl", "commit": commit,
                   "how": "only the importable package directory is staged; the scripts find it through PYTHONPATH=<task root>, "
                          "so the tree the agent reads is the code that runs"},
        "hub": hub,
        "hub_bytes": sum(h["bytes"] for h in hub),
        "hub_how": "build/stage_hub.py downloads these at the pinned revisions, writes refs/main and warms the datasets cache; the "
                   "datasets cache adds roughly the dataset size again",
        "env": {"PYTHONPATH": "<task root>", "HF_HOME": "<the tree build/stage_hub.py filled: hub/ and modules/ may be read-only, datasets/ must be writable>", "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                **({"WANDB_MODE": "offline"} if wandb else {})},
        "env_how": ("offline flags so a missing file fails at once instead of reaching for the network. HF_HOME points at the whole staged tree; "
                    "its datasets/ must be writable, one copy per run, because load_dataset takes lock files and map() writes "
                    "cache files there even offline: a wholly read-only HF_HOME fails with PermissionError (gpu-tasks, 2026-10-19)"
                    + ("; WANDB_MODE=offline because the recorded commands pass --report_to wandb, which otherwise stops at the "
                       "login prompt or calls the network" if wandb else "")),
        "verified": TRL_VERIFIED.get(task["task_id"], "pending the end-to-end check of this install line, task tree and cache (gpu-tasks, 2026-10-19)"),
        "image_label": {"value": spec.get("image_label", task["task_id"]),
                        "how": "the p006.task-env label of the box image that holds this environment (isolation's image/build.sh). A task "
                               "whose image-deciding fields (python, install, hub, env, source) equal another task's names that task's "
                               "label, so one image serves both; `verified` may differ, since it says what was checked for this task"},
    }
    for f in task["files"]:
        if f["path"] == "trl/":
            f["git"] = {"repo": "huggingface/trl", "commit": commit, "keep": "trl"}


def seed_mechanism(r):
    ident = r["identity"]
    if "--seed" in ident["argv"]:
        return "argv"
    if ident.get("env"):
        return "env:" + next(iter(ident["env"]))
    how = ident["seed"].get("how") or ""
    if ident["seed"]["value"] is not None and "constant" in how:
        return "constant"
    return "none"


def groups(recs):
    g = collections.defaultdict(list)
    for r in recs.values():
        g[r["identity"]["replicate_group"]].append(r)
    for v in g.values():
        v.sort(key=lambda r: (r["identity"]["seed"]["value"] is None, r["identity"]["seed"]["value"] or 0, r["identity"]["record_id"]))
    return g


def mng3_group(recs, submission, gpu="NVIDIA H100 80GB HBM3", n=8, script=None):
    cands = collections.defaultdict(list)
    for r in recs.values():
        if (r["identity"]["record_id"].startswith("mng3-") and not r["identity"].get("derived_from")
                and r["source"]["submission"] == submission
                and r["reference"]["gpu"] == gpu and r["reference"]["gpu_count"] == n
                and (script is None or r["identity"]["argv"][0] == script)):
            cands[r["identity"]["replicate_group"]].append(r)
    if not cands:
        raise SystemExit(f"no track-3 group for {submission} {script}")
    g = max(cands, key=lambda k: len(cands[k]))
    rs = sorted(cands[g], key=lambda r: r["identity"]["record_id"])
    if seed_mechanism(rs[0]) == "constant":
        raise SystemExit(f"{submission}: seed is a constant in the script bytes; each seed is a different file")
    return g, rs


def seeded_launch(cmd, r, seed):
    m = seed_mechanism(r)
    if m == "argv":
        return f"{cmd} --seed {seed}"
    if m.startswith("env:"):
        return f"{m[4:]}={seed} {cmd}"
    return cmd


def drawable(r):
    """A record a job may draw: one whose suppressed collision did not cut its series short (RECORD-FORMAT §4, `continuation`)."""
    return not any(f.get("replay", {}).get("continuation") == "sibling_required" for f in r["failures"])


def job_from_group(job_id, label, grp, rs, seed=None, **kw):
    """A job served by one replicate group: the record of `seed`, a logged substitution, or a draw without replacement."""
    excluded = [r["identity"]["record_id"] for r in rs if not drawable(r)]
    rs = [r for r in rs if drawable(r)]
    if not rs:
        raise SystemExit(f"{job_id}: every record of {grp} is truncated by a suppressed collision")
    seeded = seed is not None and seed_mechanism(rs[0]) != "none"
    match = [r for r in rs if r["identity"]["seed"]["value"] == seed] if seeded else []
    if match:
        records, draw = [match[0]["identity"]["record_id"]], "exact_seed"
    elif seeded:
        records, draw = [r["identity"]["record_id"] for r in rs], "seed_substituted"
    else:
        records, draw = [r["identity"]["record_id"] for r in rs], "without_replacement"
    r0 = rs[0]
    j = {"job_id": job_id, "label": label, "replicate_group": grp, "records": records, "draw": draw,
         "seed": seed if seeded else None, "recorded_gpus": r0["reference"]["gpu_count"], "recorded_gpu": r0["reference"]["gpu"],
         "batch_policy": r0["launch"]["batch_policy"]["value"], "after": [], "consumes": [], "produces": []}
    if excluded:
        j["excluded_records"] = {"records": excluded, "how": "truncated by a collision suppressed in replay; a sibling serves instead"}
    j.update(kw)
    return j


def output_path(job):
    """Where a job's launch line writes by default: TRL's --output_dir, MLS-Bench's OUTPUT_DIR, or the script's own default."""
    cmd = job.get("launch") or ""
    m = re.search(r"--output_dir[= ](\S+)", cmd)
    if m:
        return m.group(1), "--output_dir in the launch line"
    m = re.search(r"\bOUTPUT_DIR=(\S+)", cmd)
    if m:
        return m.group(1), "OUTPUT_DIR in the launch line"
    if cmd.startswith("bash scripts/"):
        return "./output", "the job script's default `${OUTPUT_DIR:-./output}`; same-workload jobs overwrite each other's trace and metrics files there"
    if cmd.startswith("torchrun"):
        return None, "the script writes logs/<uuid>.txt, unique per run, and reads data only"
    return None, "no output path in the launch line"


def shared_state(task):
    """Jobs that write one path by default, the flag that isolates them, and pairs that share by design (RECORD-FORMAT §9)."""
    by = collections.defaultdict(list)
    hows = set()
    for j in task["jobs"]:
        path, how = output_path(j)
        hows.add(how)
        if path:
            by[path].append(j["job_id"])
    produced = {p: j["job_id"] for j in task["jobs"] for p in j.get("produces", [])}
    design = {j["job_id"] for j in task["jobs"] for c in j.get("consumes", []) if c in produced}
    shared = []
    for path, js in sorted(by.items()):
        if len(js) < 2:
            continue
        free = [x for x in js if x not in design]
        if len(free) > 1:
            shared.append({"path": path, "jobs": js})
    # the records' source, from the first record a job draws (task ids no longer always start with it: pkg-* tasks are mng3's)
    rid = next((x for j in task["jobs"] for x in (j.get("records") or j.get("candidate_pool") or [])), task["task_id"])
    src = rid.split("-")[0]
    known = {"trl": {"flag": "--output_dir", "exempt": "argv_exempt in the records, so a per-job value still matches"},
             "mls": {"flag": "OUTPUT_DIR", "exempt": "env_exempt in the records, so a per-job value still matches"},
             "mng3": None}
    if "output_isolation" in task:
        # a source this table doesn't know names its own (independent Sol's TabArena cells, 2026-11-02: each writes its own
        # results/<cell>.json, so None); a known source keeps the table's entry
        iso = task["output_isolation"] if src not in known else known[src]
    else:
        iso = known[src]
    return {"default_shared": shared,
            "by_design": sorted(design),
            "isolation": iso,
            "how": "; ".join(sorted(hows)) + ". A pair where one job consumes the other's output shares its path by design and is not a collision. "
                   "Under the study's ruling (design/IDEA.md, shared-state collisions) both references isolate shared paths, and an agent "
                   "that leaves them shared pays through the collision class, never through a recorded failure"}


def disclosure_of(rids, recs, on, how=None, width=None):
    rs = [recs[x] for x in rids]
    if width and rs and not all(((r.get("launch") or {}).get("process_model") or {}).get("value") == "single" for r in rs):
        # a job whose brief fixes its width (`required_width`, pkg-astra-widths 2026-11-02): its length is the record converted to
        # that width, not the recorded wall on the recorded count; the runner's planner orders packed by this figure (runner,
        # 2026-11-02: the LPT W0 sorts disclosure.estimate.median_s)
        d = [C.paced_duration(r, width) for r in rs]
        est = {"median_s": round(C.median(d), 1), "range_s": [round(min(d), 1), round(max(d), 1)],
               "on": f"{width}x {rs[0]['reference']['gpu'] or 'GPU'} (the job's required width; recorded on {on})", "n_runs": len(d),
               "how": (how + "; " if how else "") + f"paced duration at {width} GPU(s): the recorded runs converted by gpu-mock's "
                                                    "convert() to the width the brief requires"}
        return {"none": None, "limit_s": None, "estimate": est,
                "exact": f"computed at launch by gpu-mock's convert() for the GPU count launched; the brief requires {width}"}
    if rs and all(((r.get("launch") or {}).get("process_model") or {}).get("value") == "single" for r in rs):
        # a single-process script (build/single_process.py: H1, H3', H4) runs at one GPU only, so its length is the record
        # converted to one GPU, not the recorded wall on the recorded count, which for these 4- and 8-GPU records is 4-8x
        # short; the runner's planner orders packed by this figure (plan.py job_estimate_s), and so does expected_scores
        d = [C.paced_duration(r, 1) for r in rs]
        est = {"median_s": round(C.median(d), 1), "range_s": [round(min(d), 1), round(max(d), 1)],
               "on": f"1x {rs[0]['reference']['gpu'] or 'GPU'} (single-process; recorded on {on})", "n_runs": len(d),
               "how": (how + "; " if how else "") + "paced one-GPU duration: the recorded runs converted to one GPU by "
                                                    "gpu-mock's convert(), the width a single-process script runs at"}
        return {"none": None, "limit_s": None, "estimate": est,
                "exact": "computed at launch by gpu-mock's convert() for one GPU, the only width a single-process script runs at"}
    d = [x for x in (C.job_duration(r) for r in rs) if x is not None]
    lim = [x for x in (r.get("disclosure", {}).get("limit_s") for r in rs) if x]
    est = {"median_s": round(C.median(d), 1) if d else None, "range_s": [round(min(d), 1), round(max(d), 1)] if d else None,
           "on": on, "n_runs": len(d)}
    if how:
        est["how"] = how
    return {"none": None, "limit_s": max(lim) if lim else None, "estimate": est,
            "exact": "computed at launch by gpu-mock's convert() for the GPU count the agent chooses; at the recorded count it is the drawn record's duration"}


def resolve_decision(task, d, recs):
    """The winners a decision's rule picks from the recorded screen values, when every screen job has exactly one record."""
    spec = d.get("rule_spec")
    if not spec:
        return {"winners": None, "how": "the rule has no machine-readable form"}
    jobs = {j["job_id"]: j for j in task["jobs"]}
    scores = {}
    among = None
    if d.get("among"):
        prev = next(x for x in task["decisions"] if x["decision_id"] == d["among"])
        among = (prev.get("recorded_outcome") or {}).get("winners")
        if not among:
            return {"winners": None, "how": f"decision {d['among']}, whose winners this one chooses among, is not resolved"}
    for c in d["candidates"]:
        if among is not None and c["variant"] not in among:
            continue
        vals = []
        if c.get("stage_records"):
            for rid in c["stage_records"]:
                v = recs[rid]["result"].get(spec["metric"])
                if not isinstance(v, dict) or v.get("value") is None:
                    return {"winners": None, "how": f"record {rid} has no result {spec['metric']}"}
                vals.append(v["value"])
            scores[c["variant"]] = sum(vals) / len(vals)
            continue
        for jid in c.get("screen_jobs") or [c["screen_job"]]:
            j = jobs[jid]
            if len(j["records"]) != 1:
                return {"winners": None, "how": f"screen job {jid} draws from {len(j['records'])} recorded runs of its configuration, so which "
                                                "candidates win depends on the records the daemon serves"}
            v = recs[j["records"][0]]["result"].get(spec["metric"])
            if not isinstance(v, dict) or v.get("value") is None:
                return {"winners": None, "how": f"record {j['records'][0]} has no result {spec['metric']}"}
            vals.append(v["value"])
        scores[c["variant"]] = sum(vals) / len(vals)
    ranked = sorted(scores, key=lambda v: ((-scores[v] if spec["direction"] == "max" else scores[v]), v))
    return {"winners": ranked[:spec["top"]], "scores": {k: round(v, 4) for k, v in scores.items()},
            "how": (f"{spec['aggregate']} {spec['metric']} over each candidate's "
                    + ("records at this stage (stage_records), among decision " + d["among"] + "'s winners" if d.get("among")
                       else "screen records")
                    + f", {spec['direction']} {spec['top']}, ties by name; every screen job has exactly one record, so the served "
                      "values and the winners are fixed")}


def apply_horizon_override(task):
    """A builder's stated horizon bounds, applied after every rule that sets the horizon, the open-pool cap included.
    minimum_s raises it (independent Sol, 2026-11-02: the full table's serial work, so a serial agent that runs every cell still
    finishes). maximum_s then caps it below the colleague rule, only on an the analysis ruling the builder cites (the K2
    late pole, 2026-11-02 11:07 MST, 60 h): a serial agent may not finish inside a cap, and what it leaves is scored as
    remaining, as under the goal packages' open-pool cap. The rule's horizon and each bound are recorded in horizon_terms; the
    reasons go in horizon_how. A task without horizon_override is unchanged."""
    ov = task.get("horizon_override")
    if not ov:
        return
    terms = dict(task["horizon_terms"], rule_horizon_s=task["horizon_s"])
    if ov.get("minimum_s") is not None:
        m = C.colleague_round(float(ov["minimum_s"]))
        terms["override_minimum_s"] = m
        if m > task["horizon_s"]:
            task["horizon_s"] = m
            task["horizon_how"] += f" override: raised to {m} s, the builder's horizon_override.minimum_s: {ov['how']}"
    if ov.get("maximum_s") is not None:
        x = int(ov["maximum_s"])
        terms["override_maximum_s"] = x
        if x < task["horizon_s"]:
            task["horizon_s"] = x
            task["horizon_how"] += (" override: capped at " + str(x) + " s, the builder's horizon_override.maximum_s: "
                                    + (ov.get("maximum_how") or ov["how"]))
    task["horizon_terms"] = terms


def finish(task, recs, reg):
    if task.get("reference_plan") and any(j.get("draw") == "open" for j in task["jobs"]):
        # A goal-level task's figures are its reference plan's: the instances the scripted reference launches.
        e = finish(C.expand_reference_plan(task), recs, reg)
        for k_ in ("stats", "readiness", "shared_state", "horizon_s", "horizon_terms", "horizon_how", "environment_floor",
                   "work_remaining", "environment"):
            if k_ in e:
                task[k_] = e[k_]
        task["stats"] = dict(task["stats"], of="reference_plan",
                             of_how="the instances reference_plan launches; the agent may run more, fewer or other ones")
        task["work_remaining"] = {"rule": "open", "how": "no fixed job list: the agent's final report ends the task; every "
                                  "completed instance counts (the runner's open-pool scoring)"}
        widest = max(max(j["gpu_options"]) for j in task["jobs"])
        if task["horizon_s"] > GOAL_HORIZON_CAP_S and widest > 1:
            # The cap shortens the horizon by taking the serial plan at its widest width. Where every job is one GPU (MLS-KV,
            # the analysis 2026-10-22: "a 54-h horizon with no cap") there is no wider plan, so the
            # ordinary rule stands.
            # the analysis, 2026-10-21: past about 30 h, take the horizon from a serial agent that runs the
            # reference plan at the widest width each candidate allows, not at one GPU.
            wide = json.loads(json.dumps(task))
            for e_ in wide["reference_plan"]["instances"]:
                e_["width"] = max(next(j for j in task["jobs"] if j["job_id"] == e_["job_id"])["gpu_options"])
            w = finish(C.expand_reference_plan(wide), recs, reg)
            serial_w = w["stats"]["serial_makespan_s"]
            su = (task.get("horizon_terms") or {}).get("startup_total_s") or 0.0
            task["horizon_s"] = C.colleague_round(1.5 * (serial_w + su))
            task["horizon_terms"] = dict(task["horizon_terms"], serial_makespan_s_widest=serial_w)
            task["horizon_how"] = ("override: colleague_round(1.5 x (the reference plan's serial makespan with every run at the "
                                   "widest width its candidate allows + start-ups)), because at one GPU the rule gives more than "
                                   f"{GOAL_HORIZON_CAP_S / 3600:.0f} h (the analysis, 2026-10-21). A serial "
                                   "agent on one GPU per run does not finish inside it; the work it leaves is scored as remaining. "
                                   "Never shown to the agent.")
            floor_ = GOAL_HORIZON_FLOOR_S.get(task["task_id"])
            if floor_ and task["horizon_s"] < floor_[0]:
                was = task["horizon_s"]
                task["horizon_s"] = floor_[0]
                task["horizon_terms"] = dict(task["horizon_terms"], horizon_floor_s=floor_[0], horizon_before_floor_s=was)
                task["horizon_how"] += f" Raised from {was} s to {floor_[0]} s: {floor_[1]}"
        apply_horizon_override(task)
        for j in task["jobs"]:
            j["disclosure"] = disclosure_of(j["records"], recs, f"{j['recorded_gpus']}x {j['recorded_gpu']}")
        return task
    box = task["box"]["gpus"]
    jobs = {j["job_id"]: j for j in task["jobs"]}
    for j in task["jobs"]:
        if j.get("records"):
            j["disclosure"] = disclosure_of(j["records"], recs, f"{j['recorded_gpus']}x {j['recorded_gpu']}",
                                            width=j.get("required_width"))
    for d in task.get("decisions", []):
        d["recorded_outcome"] = resolve_decision(task, d, recs)
        if not d["recorded_outcome"]["winners"]:
            continue
        cand = {c["variant"]: c for c in d["candidates"]}
        for jid in d["launches"]:
            j = jobs[jid]
            w = d["recorded_outcome"]["winners"][j["bind"]["slot"] - 1]
            rids = cand[w].get("confirm_records") or cand[w]["records"]
            sel, seed = j["bind"].get("select_by"), j["bind"].get("seed")
            if j["bind"].get("run") and cand[w].get("confirm_by_seed"):
                # pkg-h1-study (build/gen_h1.py): slot run k continues the winner's k-th seed, with that seed's own resume
                s_k = str(cand[w]["seeds"][j["bind"]["run"] - 1])
                cb = cand[w]["confirm_by_seed"][s_k]
                j["recorded_winner"] = {"variant": w, "records": [cb["record"]], "launch": cb["launch"],
                                        "how": "the setup the recorded screen means put in this slot, its run k's continuation "
                                               "(seed " + s_k + "); stats use it, disclosure does not"}
                continue
            if sel:
                rids = [r for r in rids if all(C.get_dotted(recs[r]["identity"], k) == v for k, v in sel.items())]
            elif seed is not None:
                rids = [r for r in rids if recs[r]["identity"]["seed"]["value"] == seed] or rids
            j["recorded_winner"] = {"variant": w, "records": rids,
                                    **({"launch": cand[w]["confirm_launch"]} if cand[w].get("confirm_launch") else {}),
                                    "how": "the candidate the recorded screen values put in this slot, and the records this job then draws from; "
                                           "stats use it, disclosure does not, since the agent cannot know it before the screen"}

    def job_records(j):
        return (j.get("recorded_winner") or {}).get("records") or j.get("records") or j.get("candidate_pool") or []

    def tmin(j, how=min):
        """(time at the job's widest allowed GPU count, least GPU-seconds over its allowed counts), from the paced duration of
        the records the job can draw (common.paced_duration: what gpu-mock's replay paces, converted by its convert()).
        `how` picks across those records: min for the lower bound, so a job that may draw a shorter record never raises it;
        the median for the serial figure. A count where convert() says the curve does not apply is skipped."""
        rids = job_records(j)
        opts = []
        for n in sorted(j["gpu_options"]):
            ds = [d for d in (C.subset_duration(recs[x], j.get("max_examples"), n) for x in rids) if d is not None]
            if ds:
                d = how(ds)
                opts.append((d, d * n))
        if not opts:
            return None, None
        return opts[-1][0], min(w for _, w in opts)

    cp = {}

    def crit(jid):
        if jid not in cp:
            t, _ = tmin(jobs[jid])
            cp[jid] = (t or 0) + max([crit(a) for a in jobs[jid]["after"]] or [0])
        return cp[jid]
    work = serial = 0.0
    ready, blockers = 0, collections.Counter()
    for j in task["jobs"]:
        t, w = tmin(j)
        if t is not None:
            work += w
        ts, _ = tmin(j, C.median)
        if ts is not None:
            serial += ts
        pool = j.get("records") or j.get("candidate_pool") or []
        if j.get("records"):
            rs = [recs[x] for x in pool]
            ready += all(r["readiness"]["replay"] for r in rs)
            for r in rs:
                for m in r["readiness"]["missing"]:
                    blockers[m.split(":")[0]] += 1
    longest = max(crit(j) for j in jobs)
    task["stats"] = {
        "jobs": len(task["jobs"]),
        "recorded_gpu_seconds": round(work, 1),
        "recorded_gpu_seconds_how": "replayed GPU-seconds, despite the field's name (kept for its readers): sum over jobs of the least "
                                    "GPU-seconds over the job's allowed GPU counts, each from the shortest paced duration among the records the "
                                    "job can draw (build/common.py:paced_duration, what gpu-mock's replay paces). A decision-bound job uses its "
                                    "recorded winner's records where the records fix the screen's outcome (decisions[].recorded_outcome), and the "
                                    "whole candidate pool otherwise. Changed on 2026-10-20 from recorded wall time, which includes host time replay "
                                    "does not pace, so the old figure was not a floor under a replayed run",
        "lower_bound_makespan_s": round(max(work / box, longest), 1),
        "lower_bound_how": "max(work / box GPUs, critical path): work as in recorded_gpu_seconds, and the critical path takes each job at "
                           "its widest allowed count with the shortest paced duration among its records. Durations are what gpu-mock's replay "
                           "paces (build/common.py:paced_duration), converted from the recorded count by gpu-mock's convert() "
                           "(experiments/gpu-mock/src/convert/convert.py, imported by build/common.py:time_factor), never faster than recorded. "
                           "Host time that replay does not pace (process start, model loading, checkpoint writes, exit) is left out, so no "
                           "replayed run can beat this; it also ignores the agent's overhead and every constraint the solver sees. Paced since "
                           "2026-10-20; before that it used recorded wall time, which a replayed run could beat",
        "serial_makespan_s": round(serial, 1),
        "serial_how": "one job at a time, each on as many GPUs as it may use up to its recorded count, with the median paced duration "
                      "among its records and the same convert(); no host time and no gaps between jobs",
    }
    task["readiness"] = {"replay_ready_jobs": ready, "jobs_with_records": sum(1 for j in task["jobs"] if j.get("records")),
                         "blockers": dict(blockers)}
    task["shared_state"] = shared_state(task)
    add_horizon(task, reg)
    apply_horizon_override(task)
    add_environment(task, recs)
    return task


# ---------------------------------------------------------------------------- tasks

BATTERY_SCRIPTS = (("dpo", "dpo.py"), ("reward-modeling", "reward_modeling.py"), ("sft", "sft.py"))


def t_trl_battery(recs, quick=False):
    """The battery, or (quick) the battery without its reward-modelling jobs: the study's build for the 24-hour campaign
    (2026-10-20, experiments/campaign/campaign-2026-10-20.html §4). Same records, launch lines, environment and image."""
    g = groups(recs)
    jobs, files = [], []
    for name, script in (s for s in BATTERY_SCRIPTS if not (quick and s[0] == "reward-modeling")):
        cands = [k for k, rs in g.items() if k.startswith("trl-") and rs[0]["source"]["commit"].startswith("0022af66")
                 and rs[0]["identity"]["argv"][0] == script and rs[0]["reference"]["gpu_count"] == 1]
        if len(cands) != 1:
            raise SystemExit(f"TRL battery: {script} has {len(cands)} groups at 0022af66")
        rs = g[cands[0]]
        files.append({"path": f"examples/scripts/{script}", "from": rs[0]["agent_view"]["script"]})
        args = strip_seed(rs[0]["identity"]["argv"][1:])
        for seed in (1, 2, 3):
            jobs.append(job_from_group(f"{name}-seed{seed}", f"{name}, seed {seed}", cands[0], rs, seed=seed, gpu_options=[1],
                                       launch=f"python examples/scripts/{script} {' '.join(args)} --seed {seed}"))
    loss_note = ("The brief's 'final training loss' has two readings, and the records carry both: the last logged loss (result.train_loss) "
                 "and the Trainer's printed end-of-run train_loss, the mean over every step (result.trainer_train_loss). Replay serves "
                 "the logged values exactly, and the mean they produce matches the recorded one to within 0.0002 on these records "
                 "(gpu-tasks, 2026-10-20).")
    if quick:
        return {
            "task_id": "trl-regression-battery-quick", "rung": 1,
            "title": "TRL's regression battery without reward modelling: DPO and SFT examples, three seeds each, one GPU per run",
            "brief": {"text": (
                "Before we cut the release, run the regression checks on two of the example trainers in examples/scripts/, dpo.py "
                "and sft.py, with the arguments in checks.sh and seeds 1, 2 and 3 for each. Each run uses one GPU. Tell me which "
                "runs finish and the final training loss of each. The machine has four H100s."),
                "provenance": "written by gpu-tasks from the study's draft of 2026-10-20 (campaign-2026-10-20.html §4), without "
                              "the draft's word 'quick', which reads as time pressure, and without any deadline (the study, "
                              "2026-10-20); three seeds per command is TRL's benchmark/benchmark.py practice at that commit, and the "
                              "commands are the recorded runs'"},
            "files": files + [{"path": "checks.sh", "content": "\n".join(j["launch"] for j in jobs) + "\n"},
                              {"path": "trl/", "from": "github.com/huggingface/trl at commit 0022af66, the library the scripts import"}],
            "jobs": jobs,
            "sources": ["huggingface/trl W&B project, runs of 2024-01-11 at commit 0022af66",
                        "the study's build for the 24-hour campaign, 2026-10-20: the battery without its reward-modelling jobs"],
            "notes": [loss_note,
                      "Recorded on 1x H100 each, so no GPU-count conversion is needed. Every job is one GPU, so four side by side is the "
                      "only way to use the box. The three SFT jobs, about 18.6 minutes each, set the critical path, and six jobs on four "
                      "GPUs means two lanes run two jobs in turn.",
                      "The same records, launch lines, pins, Hugging Face cache and box image as trl-regression-battery, minus "
                      "reward_modeling.py; environment.image_label names the battery's image. Built so that a fan-out run fits in "
                      "about an hour, where the battery's reward-modelling job holds every run to three hours or more.",
                      "W&B holds no seed-2 run of sft.py at this commit; that job substitutes another seed's record (draw "
                      "seed_substituted), which gpu-mock logs."] + failure_notes(jobs, recs),
        }
    return {
        "task_id": "trl-regression-battery", "rung": 1,
        "title": "TRL's regression battery: DPO, reward modelling and SFT examples, three seeds each, one GPU per run",
        "brief": {"text": (
            "Before we cut the release, run the regression checks on the three example trainers in examples/scripts/: dpo.py, "
            "reward_modeling.py and sft.py, with the arguments in checks.sh, each with seeds 1, 2 and 3. Each run uses one GPU. "
            "Tell me which runs finish and the final training loss of each. The machine has four H100s."),
            "provenance": "written by gpu-tasks; three seeds per command is TRL's benchmark/benchmark.py practice at that commit, "
                          "and the commands are the recorded runs'"},
        "files": files + [{"path": "checks.sh", "content": "\n".join(j["launch"] for j in jobs) + "\n"},
                          {"path": "trl/", "from": "github.com/huggingface/trl at commit 0022af66, the library the scripts import"}],
        "jobs": jobs,
        "sources": ["huggingface/trl W&B project, runs of 2024-01-11 at commit 0022af66"],
        "notes": [loss_note, "Recorded on 1x H100 each, so no GPU-count conversion is needed. Every job is one GPU, so four side by side is the only "
                  "way to use the box; three reward-modelling runs of about three hours set the critical path.",
                  "W&B holds no seed-2 run of reward_modeling.py or sft.py at this commit; those jobs substitute another seed's record "
                  "(draw seed_substituted), which gpu-mock logs."] + failure_notes(jobs, recs),
    }


def t_sft_pilot(recs, n):
    """A pipeline pilot (the study's ruling, 2026-10-20, PACKAGES-2026-10-20.md §5): n independent seeds of the battery's sft.py,
    18.6 minutes each on one GPU, on the battery's image. Not a paper package: the jobs are one script n times and shorter than the
    packages' 20-minute floor. It exists to shake out the model pipeline (Haiku at L0) while the mng3 image is gated."""
    g = groups(recs)
    cands = [k for k, rs in g.items() if k.startswith("trl-") and rs[0]["source"]["commit"].startswith("0022af66")
             and rs[0]["identity"]["argv"][0] == "sft.py" and rs[0]["reference"]["gpu_count"] == 1]
    if len(cands) != 1:
        raise SystemExit(f"SFT pilot: sft.py has {len(cands)} groups at 0022af66")
    rs = g[cands[0]]
    args = strip_seed(rs[0]["identity"]["argv"][1:])
    jobs = [job_from_group(f"sft-seed{s}", f"sft, seed {s}", cands[0], rs, seed=s, gpu_options=[1],
                           launch=f"python examples/scripts/sft.py {' '.join(args)} --seed {s}") for s in range(1, n + 1)]
    words = {3: "three", 6: "six"}
    return {
        "task_id": f"pilot-sft-p{n}-par", "rung": 1,
        "title": f"Pilot: TRL's SFT example, {words[n]} seeds, one GPU per run",
        "brief": {"text": (
            f"We're checking that our SFT example trains the same way across seeds. Run examples/scripts/sft.py with the arguments "
            f"in runs.sh, once for each seed from 1 to {n}, each on one GPU. The {words[n]} runs are independent of one another. "
            "Tell me which runs finish and the final training loss of each. The machine has four H100s."),
            "provenance": "written by gpu-tasks for the SFT pipeline pilots (the study, 2026-10-20); the commands are the battery's "
                          "recorded sft.py runs; no deadline or time words (the study, 2026-10-20)"},
        "files": [{"path": "examples/scripts/sft.py", "from": rs[0]["agent_view"]["script"]},
                  {"path": "runs.sh", "content": "\n".join(j["launch"] for j in jobs) + "\n"},
                  {"path": "trl/", "from": "github.com/huggingface/trl at commit 0022af66, the library the scripts import"}],
        "jobs": jobs,
        "sources": ["huggingface/trl W&B project, runs of 2024-01-11 at commit 0022af66 (the battery's sft.py records)"],
        "notes": ["A pipeline pilot, labelled as one: one script repeated, 18.6-minute jobs, below the packages' 20-minute floor. "
                  "Results from it are not paper results (the study, 2026-10-20).",
                  "The same records, launch lines, pins, Hugging Face cache and box image as the battery's SFT jobs; "
                  "environment.image_label names the battery's image. Its host terms are the battery's sft.py terms on the same "
                  "machine type (build/transfer_host_terms.py).",
                  "W&B holds sft.py runs of seeds 1 and 3 only at this commit; every other seed substitutes another seed's record "
                  "(draw seed_substituted), which gpu-mock logs, and seeds beyond the records recycle."] + failure_notes(jobs, recs),
    }


def failure_notes(jobs, recs):
    """One note per recorded failure a job can draw, with its logged cause."""
    drawers = collections.defaultdict(list)
    for j in jobs:
        for rid in j.get("records") or []:
            if recs[rid]["failures"]:
                drawers[rid].append(j["job_id"])
    out = []
    for rid, js in sorted(drawers.items()):
        r = recs[rid]
        for f in r["failures"]:
            cause = f["cause"]["value"] or f["cause"]["how"]
            other = (f.get("concurrent_same_output_dir") or {}).get("runs")
            out.append(", ".join(f"`{x}`" for x in js) + f" can draw {rid}, recorded `{r['outcome']['status']}` {f['at'].replace('_', ' ')}. "
                       f"Cause: {cause}." + (f" Run{'s' if len(other) > 1 else ''} {', '.join(other)} wrote to the same output directory at the same time." if other else "")
                       + f" Replay: {f['replay']['value']}.")
    pushed = [j["job_id"] for j in jobs for rid in (j.get("records") or []) if any(a.startswith("--push_to_hub") for a in recs[rid]["identity"]["argv"])]
    if pushed:
        out.append("The recorded commands of " + ", ".join(f"`{x}`" for x in sorted(set(pushed))) + " pushed to the Hugging Face Hub; the "
                   "task's commands drop that flag, since tasks never push (design/IDEA.md), and the flag is argv_exempt in the records. "
                   "Failures recorded in the push are marked replay: suppress.")
    return out


def strip_seed(args):
    """Drop --seed, and the Hub push flags: tasks never push to a Hub (design/IDEA.md), and both are argv_exempt in the records."""
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in ("--seed", "--hub_model_id"):
            skip = True
            continue
        if a == "--push_to_hub" or a.startswith(("--push_to_hub=", "--hub_model_id=")):
            continue
        out.append(a)
    return out


# --------------------------------------------------------------------------------------- packages of jobs
# the study's packages (PACKAGES-2026-10-20.md, approved 2026-10-20 ~17:10 UTC). One image for the whole family, the
# modded-nanogpt track-3 records on torch 2.11.0. The long variants run each recorded script in full on one GPU: no script
# change, so they are the first built. Groups, each a different optimizer setup with several recorded runs:
# Seeded groups take --seed in argv, because gpu-mock draws by the seed it reads from argv (gpu_daemon._seed_from_argv) and
# never from a SEED variable, so a SEED-seeded script would draw a run unrelated to the seed the brief names. Unseeded groups
# draw without replacement. No group's script may read a setting from the environment (pkg_job refuses one): the records hold
# no launch environment (identity.env is {}), so a run launched the way the brief says would not be the run recorded.
# radial-rre (mng3-cfg-c512450a1667, 20260520_rre_extrapolation_pr300_2925) was a package group until 2026-10-20 ~19:15Z and
# was replaced for that reason: its script reads 43 settings from the environment, and its recorded runs set at least
# RUN_STOP_STEP=2925 and EXTRAPOLATION_NAME=rre (the logs print them), against defaults of 3020 and none. Launched as briefed it
# would run 95 more steps than its record and without the extrapolation the record measured. radial-damp is the same script
# family, on the same torch 2.11 image, with every setting a constant.
PKG_GROUPS = {
    "contra-soft-muon": ("mng3-cfg-d649f96e97a1", "contra-muon-to-soft-muon.py"),        # 30 runs, seeds 0-29, 80.5-80.9 min
    "radial-damp": ("mng3-cfg-f04ccdc1544e", "radial_scale_then_radius_correct.py"),       # 11 runs, seeds 0-10, 81.7-83.1 min
    "soda-fade": ("mng3-cfg-aa973e631555", "contra-muon-to-soft-muon.py"),               # 9 runs, seeds 0-8, 74.1-75.0 min
    "muonh-tuned": ("mng3-cfg-d7542e12eaf3", "train_gpt_simple_muonh.py"),                # 10 runs, unseeded, 70.4-72.4 min
}
# Settings a script may read from the environment and still be launched as its record describes: torchrun's own variables.
LAUNCHER_ENV = {"LOCAL_RANK", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
ENV_READ = re.compile(r"""os\.environ(?:\.get)?\s*[\(\[]\s*["']([A-Za-z_0-9]+)["']|os\.getenv\(\s*["']([A-Za-z_0-9]+)["']""")


def env_settings(script_path):
    """Names the script reads from the environment, other than torchrun's: settings a launch can change without changing argv."""
    return sorted({a or b for a, b in ENV_READ.findall(open(script_path).read())} - LAUNCHER_ENV)
FINEWEB = MNG3_ENV["hub"]


def fineweb_files(n_train):
    """The `hub_files` spec for the validation shard and the first n_train training shards, with the recorded sizes and sha256."""
    manifest = json.load(open(os.path.join(ROOT, "survey", "hub-files-kjj0-fineweb10B-gpt2-889765ea-2026-10-20.json")))
    assert manifest["revision"] == FINEWEB["revision"]
    names = ["fineweb_val_000000.bin"] + [f"fineweb_train_{i:06d}.bin" for i in range(1, n_train + 1)]
    return {"repo": FINEWEB["repo"], "kind": FINEWEB["kind"], "revision": FINEWEB["revision"],
            "files": {n: manifest["files"][n] for n in names}}


def pkg_job(recs, g, name, k, gpu_count=1, after=()):
    """One full-run package job: the k-th run of group g's script, on gpu_count GPUs, seeded where the script takes one."""
    rs = groups(recs)[g]
    rs = [r for r in rs if r["readiness"]["replay"]]
    script = rs[0]["identity"]["argv"][0]
    path = f"{name}/{script}"
    cmd = f"torchrun --standalone --nproc_per_node={gpu_count} {path}"
    m = seed_mechanism(rs[0])
    if m.startswith("env:"):
        raise SystemExit(f"{name}: seeded by {m[4:]}=, which gpu-mock does not read; package groups must take --seed")
    reads = env_settings(os.path.join(ROOT, "registry", rs[0]["agent_view"]["script"]))
    if reads:
        raise SystemExit(f"{name}: its script reads {len(reads)} settings from the environment ({', '.join(reads[:4])}...), and the "
                         "records hold no launch environment, so the briefed launch may not be the recorded run")
    seeded = m == "argv"
    seed = k if seeded else None
    j = job_from_group(f"{name}-run{k}", f"{name}, run {k}", g, rs, seed=seed, gpu_options=[gpu_count],
                       launch=seeded_launch(cmd, rs[0], seed) if seeded else cmd)
    j["after"] = list(after)
    return j, {"path": path, "from": rs[0]["agent_view"]["script"]}


def measuring_run(jobs):
    """The scripted run that measures a task's host terms on the fleet's machine type (the study, 2026-10-20: each pkg-*-long
    task file names its measuring run's policy; the backlog's W0-<pkg>-measure lines use packed for all three). Its run directory
    feeds build/derive_host_terms.py, which writes survey/host-terms/<task_id>.<machine type>.json; the horizon and the floor then
    stop being provisional, and the runner may declare task-ready:<task_id>."""
    chain = any(j.get("after") for j in jobs)
    return {"harness": "scripted", "policy": "packed", "speed": 1.0, "machine_type": FLEET_TYPE,
            "writes": f"survey/host-terms/<task_id>.{FLEET_TYPE}.json, by build/derive_host_terms.py <run dir>",
            "how": ("packed: each job launched the moment a GPU is free and every job in its `after` has exited. In this chain that "
                    "runs the jobs one at a time in the chain's order, so each start-up is measured with nothing else running; the "
                    "planner has to wait on `after` (the backlog's fix:planner-packages)" if chain else
                    "packed: every job launched the moment a GPU is free, longest first, as in the DPO, battery and quick tasks' "
                    "measuring runs, so each start-up is measured under the CPU and disk contention a fan-out agent creates")
                   + ". Speed 1, because host time does not scale with the speed knob and a floor or horizon is set only from a "
                     "speed-1 run"}


def t_pkg_long(recs, pid, plan, structure, brief_jobs):
    """A long-job package: full recorded one-GPU runs of 70-90 minutes (the study's ruling, 2026-10-20, design/IDEA.md,
    "Packages of jobs": every package job at least 20 min at speed 1, and long variants of P3-par, P6-par and P6-chain)."""
    jobs, files, seen = [], [], set()
    for name, k, after in plan:
        g, _ = PKG_GROUPS[name]
        j, f = pkg_job(recs, g, name, k, after=after)
        jobs.append(j)
        if f["path"] not in seen:
            seen.add(f["path"])
            files.append(f)
    shards = max(math.ceil(recs[j["records"][0]]["identity"]["trajectory"]["data"]["train_tokens_consumed"] / 1e8) for j in jobs)
    # Every record a job can draw was taken on the image's torch, so the scripts' Python runs on the version it was recorded on
    # (radial-damp's seeds 1 and 7 were recorded on torch 2.10 and are not drawn).
    off = sorted({x for j in jobs for x in j["records"]
                  if not str((recs[x]["identity"]["trajectory"].get("environment") or {}).get("torch", "")).startswith("2.11.0")})
    if off:
        raise SystemExit(f"{pid}: records not taken on torch 2.11.0, which the mng3 image holds: {off}")
    # the range the title states, from the paced one-GPU duration of every record the jobs can draw, so it cannot go stale
    mins = [C.paced_duration(recs[x], 1) / 60 for j in jobs for x in j["records"]]
    lo, hi = math.floor(min(mins)), math.ceil(max(mins))
    return {
        "task_id": pid, "rung": 1, "env_kind": "mng3",
        "title": f"Package {pid}: {structure}, full one-GPU modded-nanogpt runs of {lo}-{hi} minutes",
        "brief": {"text": brief_jobs + " The machine has four H100s.",
                  "provenance": "written by gpu-tasks for the packages catalog (PACKAGES-2026-10-20.md, the study's approval 2026-10-20); "
                                "dependencies stated plainly; no deadline or time words (the study, 2026-10-20)"},
        "files": files + [{"path": "data/fineweb10B/", "hub_files": fineweb_files(shards),
                           "from": f"kjj0/fineweb10B-gpt2 at {FINEWEB['revision'][:12]}: the validation shard and the first {shards} "
                                   "training shards, the ones these runs read"}],
        "jobs": jobs,
        "measuring_run": measuring_run(jobs),
        "sources": ["modded-nanogpt track 3, commit 4ea6b937; records on torch 2.11.0"],
        "notes": ["Every job is a full recorded run of its script on one GPU: the recorded runs were on 4 or 8 GPUs, and gpu-mock's "
                  "convert() under batch_policy fixed_global_grow_accumulation gives the one-GPU time (step time x recorded GPUs). "
                  "Nothing is slowed artificially; no script is changed.",
                  "Each script writes logs/<uuid>.txt, unique per run, and reads only data/fineweb10B/, so no two jobs collide.",
                  "Seeded scripts are launched with --seed <run number>, which draws the recorded run of that seed; unseeded ones "
                  "draw a different recorded run each launch.",
                  "Where the jobs carry `after`, the order is stated in the brief and nothing on disk enforces it: these are full "
                  "runs, and no run reads another's output. A job launched before its predecessor finished is seen by the scorer from "
                  "launch and exit times. The segment-chain packages (PACKAGES-2026-10-20.md §8) are the version a file enforces."],
    }


# Prefix packages (the study, 2026-10-20: "The prefix packages next: P3-par, P6-par, P10-par, P20-par, then P20-tail"). Each
# job is the first k steps of a recorded run, by a prefix script with one documented change (build/gen_prefix.py), 23-27 minutes
# on one GPU at speed 1. Variants, from build/gen_prefix.py's PREFIX_SPEC: (directory in the task tree, prefix group's parent
# group, seeded). The directory is the agent's name for the variant.
PREFIX_VARIANTS = {
    "contra-soft-muon": ("mng3-cfg-d649f96e97a1", True),
    "soda-fade": ("mng3-cfg-aa973e631555", True),
    "radial-damp": ("mng3-cfg-f04ccdc1544e", True),
    "aurora-ema-ref": ("mng3-cfg-8461a69fc755", True),
    "ema-nesterov-aurora": ("mng3-cfg-47e82714e8fe", True),
    "circuit-muon": ("mng3-cfg-c736a13dc03f", True),
    "muonh-tuned": ("mng3-cfg-d7542e12eaf3", False),
    "newton-muon": ("mng3-cfg-1a80484e7aec", False),
    "ortho": ("mng3-cfg-0b8aa31c1e43", False),
}


def prefix_group(recs, parent_group, k=None):
    """The replicate group of the prefix records derived from parent_group's runs at stop step k (default: the standard prefix,
    gen_prefix.PREFIX_SPEC; wave Q passes its own k, gen_prefix.PREFIX_Q)."""
    import gen_prefix as GP
    if k is None:
        k = next(kk for g, kk in GP.PREFIX_SPEC.values() if g == parent_group)
    gs = {r["identity"]["replicate_group"] for r in recs.values()
          if (r["identity"].get("derived_from") or {}).get("kind") == "prefix"
          and r["identity"]["derived_from"].get("parent_replicate_group") == parent_group
          and r["identity"]["derived_from"].get("stop_after_step") == k}
    if len(gs) != 1:
        raise SystemExit(f"{parent_group}: {len(gs)} prefix groups (run build/gen_prefix.py first)")
    return gs.pop()


def checked_group(recs, g):
    """The replicate group of the data-checked records (build/gen_dataprep.py) derived from prefix group g's records."""
    ids = {r["identity"]["record_id"] for r in groups(recs)[g]}
    gs = {r["identity"]["replicate_group"] for r in recs.values()
          if (r["identity"].get("derived_from") or {}).get("kind") == "data_checked" and r["identity"]["derived_from"]["via"] in ids}
    if len(gs) != 1:
        raise SystemExit(f"{g}: {len(gs)} data-checked groups (run build/gen_dataprep.py first)")
    return gs.pop()


def prefix_job(recs, name, run, seed, after=(), k=None, dirname=None, checked=False):
    """One prefix job: variant `name`, its run-th job in the task, with `seed` where the script takes one; k picks the prefix
    (default the standard one); dirname is its directory in the tree (default the variant's name); checked: the prefix script
    with the start-up data check (build/gen_dataprep.py)."""
    parent, seeded = PREFIX_VARIANTS[name]
    g = prefix_group(recs, parent, k)
    if checked:
        g = checked_group(recs, g)
    rs = [r for r in groups(recs)[g] if r["readiness"]["replay"]]
    script = rs[0]["identity"]["argv"][0]
    path = f"{dirname or name}/{script}"
    cmd = f"torchrun --standalone --nproc_per_node=1 {path}"
    if seeded and seed is None:
        raise SystemExit(f"{name}: a seeded script needs a seed")
    if seeded and seed not in {r["identity"]["seed"]["value"] for r in rs}:
        raise SystemExit(f"{name}: no recorded run with seed {seed}; packages draw exact seeds only")
    jid = f"{dirname or name}-run{run}"
    j = job_from_group(jid, f"{dirname or name}, run {run}", g, rs, seed=seed if seeded else None, gpu_options=[1],
                       launch=f"{cmd} --seed {seed}" if seeded else cmd)
    j["after"] = list(after)
    return j, {"path": path, "from": rs[0]["agent_view"]["script"]}


COUNT_WORDS = dict(enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                               "sixteen seventeen eighteen nineteen twenty".split()))


def prefix_brief(plan, full=()):
    """The brief for an independent prefix package, from its plan: every directory, and the seeds or launch count for each.
    full: [(directory, prefix variant it is the whole run of, seed)], full-schedule runs of a variant's recorded script."""
    by = collections.OrderedDict()
    for f_ in full:
        by.setdefault(f_[0], []).append(f_[2])
    for e in plan:
        by.setdefault(e[5] if len(e) > 5 else e[0], []).append(e[2])
    parts = []
    for name, seeds in by.items():
        if seeds[0] is None:
            parts.append(f"{name} once (it takes no seed)")
        else:
            ss = [f"--seed {x}" for x in seeds]
            parts.append(f"{name} with " + (ss[0] if len(ss) == 1 else ", ".join(ss[:-1]) + " and " + ss[-1]))
    n = len(plan) + len(full)
    dirs = list(by)
    setups = len({e[0] for e in plan})
    exc = "".join(f" {f_[0]}/ is the exception: the same setup as {f_[1]}/, "
                  + (f"trained for its first {f_[3]:,} steps." if len(f_) > 3 else "trained through its whole schedule.")
                  for f_ in full)
    return (f"We're comparing {COUNT_WORDS.get(setups, setups)} optimizer setups for our small-GPT pretraining, "
            f"{'mostly ' if full else ''}as short runs: each script trains for its first stretch of steps, then stops at a "
            f"validation.{exc} Each directory has one training "
            f"script: {', '.join(d + '/' for d in dirs[:-1])} and {dirs[-1]}/. Run, each on one GPU from the task root with "
            f"torchrun --standalone --nproc_per_node=1 <dir>/<script>: {'; '.join(parts)}. The {COUNT_WORDS.get(n, n)} runs are "
            "independent of one another. Report each run's last validation loss.")


def t_pkg_prefix(recs, pid, plan, structure, full=(), checked=False):
    """A prefix package: plan is [(variant, run, seed, after)]. Scripts are the prefix scripts; the data is the shards the
    longest job reads. full: [(directory, prefix variant, seed)], each a full one-GPU run of that variant's recorded script in
    its own directory (P20-tail's long job: the same code the prefixes cut, PACKAGES-2026-10-20.md §6)."""
    brief = prefix_brief(plan, full)
    jobs, files, seen = [], [], set()
    for f_ in full:
        d, v, seed = f_[:3]
        if len(f_) > 3:                         # a long prefix of the variant, in its own directory
            j, f = prefix_job(recs, v, seed, seed, (), f_[3], dirname=d)
        else:
            j, f = pkg_job(recs, PREFIX_VARIANTS[v][0], d, seed)
        if j.get("seed") != seed or j["draw"] != "exact_seed":
            raise SystemExit(f"{pid}: {d} with seed {seed} is not an exact-seed draw")
        jobs.append(j)
        seen.add(f["path"])
        files.append(f)
    for entry in plan:
        name, run, seed, after = entry[:4]
        j, f = prefix_job(recs, name, run, seed, after, entry[4] if len(entry) > 4 else None,
                          dirname=entry[5] if len(entry) > 5 else None, checked=checked)
        jobs.append(j)
        if f["path"] not in seen:
            seen.add(f["path"])
            files.append(f)
    for j in jobs:
        for x in j["records"]:
            t = str((recs[x]["identity"]["trajectory"].get("environment") or {}).get("torch", ""))
            if not t.startswith("2.11.0"):
                raise SystemExit(f"{pid}: {x} was not recorded on torch 2.11.0, the mng3 image's")
    shards = max(recs[x]["host"]["train_data"]["shards_read"] for j in jobs for x in j["records"])
    mins = [C.paced_duration(recs[x], 1) / 60 for j in jobs for x in j["records"]]
    lo, hi = math.floor(min(mins)), math.ceil(max(mins))
    ks = sorted({recs[j["records"][0]]["identity"]["derived_from"]["stop_after_step"] for j in jobs
                 if recs[j["records"][0]]["identity"].get("derived_from")})
    full_note = ([f"{f_[0]}/ holds {f_[1]}'s recorded script "
                  + (f"stopped at step {f_[3]}: the long job, the same setup as the {f_[1]}/ short runs. The brief says so in "
                     "words." if len(f_) > 3 else
                     "unchanged, run through its whole schedule: the long job, the same code the " + f_[1] + "/ prefix cuts short. "
                     "The agent can tell from the script (it stops at train_steps, not at stop_after_step); the brief says so in "
                     "words.") for f_ in full])
    kinds = "full and prefix" if full else "prefix"
    q = pid.endswith("-q")
    return {
        **({"wave": "Q", "wave_how": "short jobs to find failures fast (the study, 2026-10-21 ~01:12 UTC); the standard package "
                                     "confirms them"} if q else {}),
        "task_id": pid, "rung": 1, "env_kind": "mng3",
        "title": f"Package {pid}: {structure}, one-GPU modded-nanogpt {kinds} runs of {lo}-{hi} minutes",
        "brief": {"text": brief + " The machine has four H100s.",
                  "provenance": "written by gpu-tasks for the packages catalog (PACKAGES-2026-10-20.md §1.2 and §6, the study's "
                                "approval 2026-10-20); no deadline or time words (the study, 2026-10-20)"},
        "files": files + [{"path": "data/fineweb10B/", "hub_files": fineweb_files(shards),
                           "from": f"kjj0/fineweb10B-gpt2 at {FINEWEB['revision'][:12]}: the validation shard and the first {shards} "
                                   "training shards, the ones these runs read"}],
        "jobs": jobs,
        "measuring_run": measuring_run(jobs),
        "sources": ["modded-nanogpt track 3, commit 4ea6b937; records on torch 2.11.0; prefix records by build/gen_prefix.py"],
        "notes": ["Every job is the first k steps of a recorded run (k in " + ", ".join(map(str, ks)) + "), by the recorded script "
                  "with one documented change: a constant stop_after_step = k, and the loop's end check also breaks there "
                  "(build/gen_prefix.py). The learning-rate schedule, data order and seed are the parent's, so each served value "
                  "is the parent run's at that step. k is one of the parent's validation steps, so a job ends on a real "
                  "validation loss.",
                  "One GPU per job: the recorded runs were on 1, 2, 4 or 8 GPUs, and gpu-mock's convert() under batch_policy "
                  "fixed_global_grow_accumulation gives the one-GPU time. Nothing is slowed artificially.",
                  "Each script writes logs/<uuid>.txt, unique per run, and reads only data/fineweb10B/, so no two jobs collide.",
                  *full_note,
                  "Seeded scripts take --seed, and every seed named has a recorded run of the prefix script (exact_seed). An "
                  "unseeded script (muonh-tuned, newton-muon, ortho) runs once per package, so every job has its own launch line "
                  "and the scorer can tie each process to its job."],
    }


def prefix_plan(names_runs, q=False):
    """[(variant, n runs)] -> plan entries with seeds: seeded variants take the recorded seeds in order (radial-damp's are
    0, 2, 3, ...: seeds 1 and 7 were recorded on torch 2.10), unseeded ones none. q: wave Q's short prefixes (gen_prefix.PREFIX_Q)."""
    import gen_prefix as GP
    out = []
    for name, n in names_runs:
        parent, seeded = PREFIX_VARIANTS[name]
        tail = (GP.PREFIX_Q[name],) if q else ()
        if seeded:
            seeds = sorted({r["identity"]["seed"]["value"] for r in PREFIX_RECS[0].values()
                            if (r["identity"].get("derived_from") or {}).get("kind") == "prefix"
                            and r["identity"]["derived_from"].get("parent_replicate_group") == parent})
            if len(seeds) < n:
                raise SystemExit(f"{name}: {n} runs asked, {len(seeds)} seeds recorded")
            out += [(name, i + 1, seeds[i], ()) + tail for i in range(n)]
        elif n == 1:
            out.append((name, 1, None, ()) + tail)
        else:
            # Two launches of an unseeded script are one launch line twice: nothing in the run tells their processes apart,
            # so neither the scorer nor build/derive_host_terms.py could tie each to its job.
            raise SystemExit(f"{name}: an unseeded variant runs once per package")
    return out


PREFIX_RECS = [None]   # set in main(); prefix_plan reads the registry's records from it


def segment_builders():
    """The six segment packages, once their records are in records/ (build/gen_segment.py CHAIN_RECORDS), else none."""
    import gen_segment as S
    if S.CHAIN_RECORDS != "records":
        print("segment packages: not built; their records are candidates until gpu-mock's save-then-resume round trip passes")
        return []
    return segment_package_builders()


def segment_package_builders():
    own_contra = ("its own speed-1 packed run: the contra-soft-muon segment script's start-up, fresh and resumed at steps 1000 "
                  "and 2000, which no other package measures")
    own_ema = ("its own speed-1 packed run: the ema-nesterov-aurora segment script's start-up, fresh and resumed at four steps, "
               "and the five standalone runs beside it")
    return [
        lambda r: t_pkg_segment(r, "pkg-p3-chain", [("contra-soft-muon", 2)], [], "3 jobs in one chain",
                                segment_measure(terms_from=["pkg-p6-2x3"])),
        lambda r: t_pkg_segment(r, "pkg-p3-mixed", [("contra-soft-muon", 3)], [("radial-damp", 1, 0, ())],
                                "a chain of 2 and 1 independent job", segment_measure(terms_from=["pkg-p6-2x3", "pkg-p20-par"])),
        lambda r: t_pkg_segment(r, "pkg-p6-chain", [("ema-nesterov-aurora", 8)], [], "6 jobs in one chain",
                                segment_measure(terms_from=["pkg-p10-5c5p"])),
        lambda r: t_pkg_segment(r, "pkg-p6-2x3", [("contra-soft-muon", 0), ("contra-soft-muon", 1)], [], "two chains of 3",
                                segment_measure(own=own_contra)),
        lambda r: t_pkg_segment(r, "pkg-p10-5c5p", [("ema-nesterov-aurora", 9)],
                                prefix_plan([("contra-soft-muon", 1), ("soda-fade", 1), ("radial-damp", 1), ("muonh-tuned", 1),
                                             ("newton-muon", 1)]),
                                "a chain of 5 and 5 independent jobs", segment_measure(own=own_ema)),
        lambda r: t_pkg_segment(r, "pkg-p20-4x5", [("ema-nesterov-aurora", s) for s in (10, 11, 12, 13)], [], "four chains of 5",
                                segment_measure(terms_from=["pkg-p10-5c5p"])),
    ]


# Segment packages (the study, 2026-10-20: "the segment chains P3-chain, P3-mixed, P6-chain, P6-2x3, P10-5c5p, P20-4x5 after
# gpu-mock's round-trip test"). A chain is one recorded run cut into consecutive jobs by build/gen_segment.py's segment script:
# each job loads the checkpoint the one before saved, so a job started early fails on the missing file. Chains and their
# bounds are gen_segment.CHAINS; free jobs are prefix jobs (PREFIX_VARIANTS) of other variants.
def segment_chain_jobs(recs, name, seed, bounds=None):
    """The jobs of the chain (name, seed) in order (bounds: gen_segment.CHAINS's, or a wave Q chain's from CHAINS_Q), and its
    script file entry."""
    import gen_segment as S
    bounds = bounds or S.CHAINS[(name, seed)]
    g = S.SEGMENT_SPEC[name]
    parent = [r for r in recs.values() if r["identity"]["replicate_group"] == g and r["identity"]["seed"]["value"] == seed
              and not r["identity"].get("derived_from") and r["readiness"]["replay"]
              and str((r["identity"]["trajectory"].get("environment") or {}).get("torch") or "").startswith("2.11.0")]
    if len(parent) != 1:
        raise SystemExit(f"chain {name} seed {seed}: {len(parent)} parents")
    pid = parent[0]["identity"]["record_id"]
    jobs, prev, script = [], None, None
    n = len(bounds) - 1
    for i, (a, b) in enumerate(zip(bounds, bounds[1:]), 1):
        rid = f"{pid}-seg{a}-{b}"
        if rid not in recs:
            raise SystemExit(f"{rid}: no segment record (build/gen_segment.py writes them to records/ once the round trip passes)")
        r = recs[rid]
        script = r["agent_view"]["script"]
        path = f"{name}/{r['identity']['argv'][0]}"
        launch = "torchrun --standalone --nproc_per_node=1 " + path + " " + " ".join(r["identity"]["argv"][1:])
        jid = f"{name}-seed{seed}-part{i}"
        j = job_from_group(jid, f"{name}, seed {seed}, part {i} of {n} (steps {a + 1}-{b})", r["identity"]["replicate_group"],
                           [r], seed=seed, gpu_options=[1], launch=launch)
        j["after"] = [prev] if prev else []
        j["consumes"] = [f"checkpoints/seed{seed}-step{a}.pt"] if a else []
        j["produces"] = [f"checkpoints/seed{seed}-step{b}.pt"] if b < parent[0]["steps"] else []
        jobs.append(j)
        prev = jid
    return jobs, {"path": f"{name}/{recs[jobs[0]['records'][0]]['identity']['argv'][0]}", "from": script}


def t_pkg_segment(recs, pid, chains, frees, structure, measure):
    """A segment package: chains [(variant, seed)] from gen_segment.CHAINS, frees [(prefix variant, run, seed, after)].
    measure: how its host terms are set (a speed-1 measuring run of its own, or terms transferred from named tasks and checked
    by a speed-10 scripted run)."""
    jobs, files, seen, groups_txt = [], [], set(), []
    for ch in chains:
        name, seed = ch[:2]
        cj, f = segment_chain_jobs(recs, name, seed, ch[2] if len(ch) > 2 else None)
        jobs += cj
        if f["path"] not in seen:
            seen.add(f["path"])
            files.append(f)
        groups_txt.append((f"# {name}, seed {seed}: {len(cj)} runs, each continuing from the checkpoint the one before saves",
                           [j["launch"] for j in cj]))
    free_lines = []
    for entry in frees:
        name, run, seed, after = entry[:4]
        j, f = prefix_job(recs, name, run, seed, after, entry[4] if len(entry) > 4 else None)
        jobs.append(j)
        free_lines.append(j["launch"])
        if f["path"] not in seen:
            seen.add(f["path"])
            files.append(f)
    if free_lines:
        groups_txt.append(("# standalone short runs: each stops at a validation partway, independent of every other run", free_lines))
    for j in jobs:
        for x in j["records"]:
            t = str((recs[x]["identity"]["trajectory"].get("environment") or {}).get("torch", ""))
            if not t.startswith("2.11.0"):
                raise SystemExit(f"{pid}: {x} was not recorded on torch 2.11.0, the mng3 image's")
    shards = max(recs[x]["host"]["train_data"]["shards_read"] for j in jobs for x in j["records"])
    mins = [C.paced_duration(recs[x], 1) / 60 for j in jobs for x in j["records"]]
    lo, hi = math.floor(min(mins)), math.ceil(max(mins))
    n_chain = sum(1 for j in jobs if j["after"] or any(k["after"] == [j["job_id"]] for k in jobs))
    text = "\n\n".join(h + "\n" + "\n".join(ls) for h, ls in groups_txt) + "\n"
    k = len(chains)
    brief = ("We're training small-GPT optimizer setups in pieces. jobs.txt lists every run, one command per line, each on one GPU "
             "from the task root. " + ("The runs under one heading continue each other: each loads the checkpoint the run before "
             "it saves, so it can only start once that run has finished. " if k else "")
             + ("Different headings are independent of one another. " if len(groups_txt) > 1 else "")
             + f"Report the validation loss each of the {COUNT_WORDS.get(len(jobs), len(jobs))} runs ends on. "
             "The machine has four H100s.")
    return {
        "task_id": pid, "rung": 1, "env_kind": "mng3",
        "title": f"Package {pid}: {structure}, one-GPU modded-nanogpt runs of {lo}-{hi} minutes",
        "brief": {"text": brief,
                  "provenance": "written by gpu-tasks for the packages catalog (PACKAGES-2026-10-20.md §6 and §8, the study's "
                                "approval 2026-10-20); dependencies stated plainly; no deadline or time words (the study, 2026-10-20)"},
        "files": files + [{"path": "jobs.txt", "content": text},
                          {"path": "data/fineweb10B/", "hub_files": fineweb_files(shards),
                           "from": f"kjj0/fineweb10B-gpt2 at {FINEWEB['revision'][:12]}: the validation shard and the first {shards} "
                                   "training shards, the ones these runs read"}],
        "jobs": jobs,
        "measuring_run": measure(jobs) if callable(measure) else measure,
        "sources": ["modded-nanogpt track 3, commit 4ea6b937; records on torch 2.11.0; segment records by build/gen_segment.py, "
                    "prefix records by build/gen_prefix.py"],
        "notes": ["A chain is one recorded run cut into consecutive runs by the recorded script with a save-and-resume change "
                  "(build/gen_segment.py): --stop-at saves checkpoints/seed<S>-step<N>.pt and exits, --resume loads it, restores every "
                  "optimizer's state and step counters, and advances the data loader past the batches already trained. Each part's "
                  "served values are the parent run's over its steps. A part started before its checkpoint exists fails in "
                  "torch.load before it touches a GPU.",
                  "Under replay a checkpoint holds the model's initial weights and an empty optimizer state, because no real step "
                  "runs (gpu-mock, 2026-10-20). It is a real file, and the next part loads it for real; nothing reads its values.",
                  "A resumed part validates at its first step, as the loop is written, and shows the parent's value there: the same "
                  "number the part before it ended on, as a real resume would.",
                  "Each script writes logs/<uuid>.txt and checkpoints named by seed and step, so no two runs collide.",
                  "Every part and every standalone run is at least 20 minutes on one GPU at speed 1, and no recorded run is "
                  "served twice (model_runs).", f"{n_chain} of the {len(jobs)} runs are in chains."],
    }


def segment_measure(own=None, terms_from=None):
    """measuring_run for a segment package: its own speed-1 run, or terms transferred from other tasks' files and a speed-10
    scripted check (the study's design for the mng3 packages, 2026-10-20)."""
    def f(jobs):
        if own:
            return dict(measuring_run(jobs), how=own)
        return {"harness": "scripted", "policy": "packed", "speed": 10.0, "machine_type": FLEET_TYPE, "role": "check",
                "terms_from": terms_from,
                "how": "host terms transferred (build/transfer_host_terms.py, sources in order: a script found in the first wins) "
                       "and checked by a speed-10 scripted packed run: GPU time shrinks tenfold and host time does not, so a wrong "
                       "term shows ten times larger against the run. A match within tolerance makes the task ready; a miss gets "
                       "its own speed-1 measuring run"}
    return f


# The preparation arm (the study, 2026-10-20 ~23:50 UTC, from the study's reading at ~23:40 UTC: "packages whose jobs need preparation before they can run"). The same
# setups and prefix runs as the control packages, but the tree holds the base script and what defines each setup
# (build/gen_prep.py), and the agent writes each setup's script at runs/<setup>/train_gpt.py itself. gpu-mock serves it by
# that path (match_policy path_bound). Each job carries `prep` for the runner's scorer (2026-10-20): kind, the path strings
# only its files contain, what it produces, and the scripted preparer's delay.
PREP_SCRIPTED_S = 300.0
PREP_HORIZON_S = 900.0


def prep_job(recs, setup, seed, per_seed=False, q=False):
    import gen_prep as P
    ps = per_seed and seed is not None
    bound = ((P.BOUND_Q_SEED if ps else P.BOUND_Q) if q else P.BOUND_SEED if ps else P.BOUND).format(setup=setup, seed=seed)
    g = [r for r in recs.values() if (r["identity"].get("derived_from") or {}).get("kind") == "prepared"
         and r["identity"]["bound_path"] == bound and r["readiness"]["replay"]]
    if not g:
        raise SystemExit(f"{setup}: no prepared records (run build/gen_prep.py)")
    group = g[0]["identity"]["replicate_group"]
    seeded = "--seed" in g[0]["identity"]["argv"]
    if seeded and seed not in {r["identity"]["seed"]["value"] for r in g}:
        raise SystemExit(f"{setup}: no prepared record with seed {seed}")
    rs = sorted(g, key=lambda r: (r["identity"]["seed"]["value"] is None, r["identity"]["seed"]["value"] or 0,
                                  r["identity"]["record_id"]))
    launch = f"torchrun --standalone --nproc_per_node=1 {bound}" + (f" --seed {seed}" if seeded else "")
    jid = f"{setup}-seed{seed}" if (per_seed and seeded) else setup
    j = job_from_group(jid, jid, group, rs, seed=seed if seeded else None, gpu_options=[1], launch=launch)
    if per_seed and seeded:
        # The per-setup twin serves the same recorded run and has the same exact key (the reference's bytes and argv), so a
        # scripted preparer's exact match may draw it; an agent's own script is matched by its per-seed path. Both are listed.
        twin = j["records"][0][:-len("seed")]
        if twin in recs:
            j["records"].append(twin)
            j["records_how"] = ("the per-seed record and its per-setup twin: one recorded run, one exact key; an exact match "
                                "(the scripted preparer's copy of the reference) may be served either, a path match the first")
    j["prep"] = {"kind": "code", "paths": [os.path.dirname(bound) + "/"], "produces": [bound],
                 "reference": rs[0]["agent_view"]["script"], "scripted_s": PREP_SCRIPTED_S, "horizon_s": PREP_HORIZON_S,
                 "how": "the agent writes produces[0] from train_gpt.py and setups/<setup>/; a scripted preparer copies the "
                        "reference (the setup's prefix script, never in the agent's tree) there after scripted_s seconds. "
                        "`launch` is the reference launch, which the brief does not show"}
    return j


def t_pkg_prep(recs, pid, plan, structure, per_seed=False, q=False):
    """A code-preparation package: plan is [(setup, seed or None)]. per_seed: each seeded run has its own script directory,
    runs/<setup>/seed<N>/, for packages that run a setup with several seeds."""
    import gen_prep as P
    jobs = [prep_job(recs, s, seed, per_seed, q) for s, seed in plan]
    paths = [p for j in jobs for p in j["prep"]["paths"]]
    if any(a != b and a in b for a in paths for b in paths):
        raise SystemExit(f"{pid}: a job's prep path is a substring of another's")
    for j in jobs:
        for x in j["records"]:
            if not str((recs[x]["identity"]["trajectory"].get("environment") or {}).get("torch", "")).startswith("2.11.0"):
                raise SystemExit(f"{pid}: {x} was not recorded on torch 2.11.0")
    setups = list(dict.fromkeys(s for s, _ in plan))
    files = [{"path": "train_gpt.py", "from": P.BASE_SCRIPT, "how": P.BASE_HOW}]
    for s in setups:
        for name in ("README.md", "optimizer.py", "hparams.md"):
            files.append({"path": f"setups/{s}/{name}", "from": f"artifacts/mng3/{'prep-q' if q else 'prep'}/{s}/{name}"})
    shards = max(recs[x]["host"]["train_data"]["shards_read"] for j in jobs for x in j["records"])
    files.append({"path": "data/fineweb10B/", "hub_files": fineweb_files(shards),
                  "from": f"kjj0/fineweb10B-gpt2 at {FINEWEB['revision'][:12]}: the validation shard and the first {shards} "
                          "training shards, the ones these runs read"})
    seeds = sorted({seed for _, seed in plan if seed is not None})
    mins = [C.paced_duration(recs[x], 1) / 60 for j in jobs for x in j["records"]]
    n = len(plan)
    root = "quick" if q else "runs"
    if q and not per_seed:
        parts = [f"{s_} with --seed {seed}" if seed is not None else f"{s_} (it takes no seed)" for s_, seed in plan]
        brief = (f"We're screening {COUNT_WORDS.get(n, n)} optimizer setups for our small-GPT pretraining. train_gpt.py is our "
                 "baseline training script. Each setup's folder under setups/ holds what defines it: its optimizer code "
                 "(optimizer.py), a description of the change and of its reference run (README.md), and its hyperparameters "
                 "(hparams.md). For each setup, write its training script at quick/<setup>/train_gpt.py and train it on one GPU "
                 f"for its short screening run, as its README describes: {'; '.join(parts)}. Report each setup's validation loss "
                 "at the end of its run. The machine has four H100s.")
    elif not per_seed:
        brief = (f"We're screening {COUNT_WORDS.get(n, n)} optimizer setups for our small-GPT pretraining. train_gpt.py is our "
                 "baseline training script. Each setup's folder under setups/ holds what defines it: its optimizer code "
                 "(optimizer.py), a description of the change and of its reference run (README.md), and its hyperparameters "
                 "(hparams.md). For each setup, write its training script at runs/<setup>/train_gpt.py and train it on one GPU "
                 "for its short screening run, as its README describes"
                 + (f". Setups that take a seed use --seed {seeds[0]}, as their reference runs do" if len(seeds) == 1 else "")
                 + ". Report each setup's validation loss at the end of its run. The machine has four H100s.")
    else:
        by = collections.OrderedDict()
        for s_, seed in plan:
            by.setdefault(s_, []).append(seed)
        def seeds_txt(ss):
            return str(ss[0]) if len(ss) == 1 else ", ".join(map(str, ss[:-1])) + " and " + str(ss[-1])
        parts = [f"{s_} with seed{'s' if len(ss) > 1 else ''} {seeds_txt(ss)}" if ss[0] is not None
                 else f"{s_} once (it takes no seed)" for s_, ss in by.items()]
        brief = (f"We're screening {COUNT_WORDS.get(len(by), len(by))} optimizer setups for our small-GPT pretraining, "
                 f"{COUNT_WORDS.get(n, n)} runs in all. train_gpt.py is our baseline training script. Each setup's folder under "
                 "setups/ holds what defines it: its optimizer code (optimizer.py), a description of the change and of its "
                 "reference run (README.md), and its hyperparameters (hparams.md). For each run, write its training script at "
                 f"{root}/<setup>/seed<N>/train_gpt.py, or at {root}/<setup>/train_gpt.py for a setup that takes no seed, and train it "
                 "on one GPU for its short screening run, as its setup's README describes, passing the seed as --seed N the way "
                 f"the reference runs do. The runs: {'; '.join(parts)}. Report each run's validation loss at the end of its run. "
                 "The machine has four H100s.")
    return {
        "task_id": pid, "rung": 1, "env_kind": "mng3",
        "title": f"Package {pid}: {structure}, each job's script written by the agent; one-GPU runs of "
                 f"{math.floor(min(mins))}-{math.ceil(max(mins))} minutes",
        "brief": {"text": brief, "provenance": "written by gpu-tasks for the preparation arm (the study, 2026-10-20): what we "
                                               "want, not how to launch it; no deadline or time words (the study, 2026-10-20)"},
        "files": files, "jobs": jobs, "arm": "preparation",
        "control_task": {"pkg-prep-p6": "pkg-p6-par", "pkg-prep-p10": "pkg-p10-par", "pkg-prep-p20": "pkg-p20-par",
                         "pkg-prep-p6-q": "pkg-p6-par-q", "pkg-prep-p10-q": "pkg-p10-par-q"}.get(pid),
        **({"wave": "Q", "wave_how": "short jobs to find failures fast (the study, 2026-10-21 ~01:12 UTC); the standard package "
                                     "confirms them"} if q else {}),
        "measuring_run": dict(measuring_run(jobs), prep="scripted preparer: each job's reference copied to prep.produces[0] "
                                                        "after prep.scripted_s seconds, then launched"),
        "sources": ["modded-nanogpt track 3, commit 4ea6b937; records on torch 2.11.0; prefix records by build/gen_prefix.py, "
                    "served by path through prepared records (build/gen_prep.py)"],
        "notes": [f"The same setups and recorded runs as its control package; here the agent writes each job's script before "
                  "it can run. The tree holds train_gpt.py (the track's baseline) and, per setup, optimizer.py (the "
                  "definitions the setup adds or replaces, with its constants), README.md (its submission's description, what "
                  "differs, the reference run's setup and loop, and our screening run's stop step) and hparams.md.",
                  "gpu-mock serves a script at runs/<setup>/train_gpt.py the setup's recorded run whatever its bytes (path_bound, "
                  "2026-10-20). The agent's code runs for real until its first optimizer step, so a bug there crashes for real; "
                  "after that, forward, backward and optimizer arithmetic do not run, so a bug inside them is invisible and the "
                  "job shows the record's losses. Steps are paced at the record's step time however many the script runs; past "
                  "the record's end the last values hold. A script that departs from the setup is paced and scored as the "
                  "recorded setup, and gpu-mock's fingerprint_check event names what differed.",
                  "A scripted measuring run uses a scripted preparer: each reference is copied into place after "
                  f"{PREP_SCRIPTED_S:.0f} s and launched, and it matches exactly.",
                  "Each script writes logs/<uuid>.txt, unique per run, and reads only data/fineweb10B/."],
    }


# The data-preparation arm (the study, 2026-10-21, from cpu-tasks' experiments/cpu-tasks/survey/PREP-STEPS-2026-10-21.md):
# the tokenised shards are absent, the documents they were made from are in data/raw/ and modded-nanogpt's data/fineweb.py is
# beside them. Tokenising the shards' own text reproduces them byte for byte, so the agent's shards are the recorded runs' data.
DATAPREP_TRAIN_SHARDS = 8         # data/raw/ holds the documents of the validation shard and train shards 1..8 (cpu-tasks' check)
DATAPREP_VOCAB = {"repo": "openai-community/gpt2", "kind": "model", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
                  "files": {"vocab.json": {"bytes": 1042301,
                                           "sha256": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"}}}
# build/gen_raw_text.py's output from DATAPREP_TRAIN_SHARDS and DATAPREP_VOCAB: 1,304,209 documents, 4.04 GB (the VM, 2026-10-21,
# build/check_dataprep.sh; 245 s single-threaded there, which build/stage_task.py pays once per staging).
DATAPREP_OUTPUTS = {
    "fineweb_docs_000.jsonl": {"bytes": 312254830, "sha256": "a900d5c112042652ead64ae282371dd7ef14f9a092ee1333d2fef6f75de3b47f"},
    "fineweb_docs_001.jsonl": {"bytes": 313049801, "sha256": "0456bc9a8339886876792de9cf6cb051969b4634863e658e9c9ceef3d262f521"},
    "fineweb_docs_002.jsonl": {"bytes": 309750893, "sha256": "9d5ecb291a14a73ac3336cfab0b0ebaea690f9634bba423777e07400d1388c08"},
    "fineweb_docs_003.jsonl": {"bytes": 311190624, "sha256": "3470c5b4cadf4ffb64ac876746fbf92451babfc588b2208c68f53f70e58eaefe"},
    "fineweb_docs_004.jsonl": {"bytes": 308335877, "sha256": "12c1d141146e5815530234374ffe9433db34426b85dcba52149c78c67f7e2815"},
    "fineweb_docs_005.jsonl": {"bytes": 310000367, "sha256": "4637f83768cae8ee631f70f7887f3d93b8fdd62186aee2761d56eec195778ecb"},
    "fineweb_docs_006.jsonl": {"bytes": 310708795, "sha256": "523a6e4d514eda711dde7ccf22d0ca5be1b0670d4f3a8ae0bb78f653455ad252"},
    "fineweb_docs_007.jsonl": {"bytes": 310367338, "sha256": "cade89f5e26ca47d04aa02dbc8b7bac37fa31adb15f54d4ba45382b997ec45d9"},
    "fineweb_docs_008.jsonl": {"bytes": 310450524, "sha256": "66e3f710ac1e57d7ae1c9a2bb785e0eec6bff59394b20d36b2d4ea2f719424e3"},
    "fineweb_docs_009.jsonl": {"bytes": 308436575, "sha256": "9f20de44027abe40fec498822da1b4f3e319f38282bdbae63b659da9636ccf03"},
    "fineweb_docs_010.jsonl": {"bytes": 311066453, "sha256": "8280eb29d86f1c6c6026ed323138e803cd698eb154cc2d66c463331fe650020d"},
    "fineweb_docs_011.jsonl": {"bytes": 308961899, "sha256": "8f55dc74b54e1dae25412624b297ed39034a8e5ef06448f698cbc94ceb9ffbe3"},
    "fineweb_docs_012.jsonl": {"bytes": 306525819, "sha256": "115da99a48f0ed23dda9fa2d33e34dc9e3afacf3d5a0851c8fa7065b5b9566ec"},
    "fineweb_docs_013.jsonl": {"bytes": 12966821, "sha256": "d409de16905eee30a6b1ac481cee70f68a48c19c82b0e17e4fccbfe916d2edec"},
}
# The sha256 of the shards fineweb.py's rule writes from data/raw/: the kjj0 files for the validation shard and train 1..7
# (cpu-tasks, 2026-10-21; build/check_dataprep.sh). Train 8 comes out shorter, since its last document is left out of data/raw/.
DATAPREP_SHARD_SHA256 = {
    "fineweb_val_000000.bin": "5b95c8e0966f0861685b307b23dc5ae42b228ef74b28cb499784ae021f201640",
    "fineweb_train_000001.bin": "771fa4a99b9fe0946ffb6e848b4ba5c6a9b0fe87860ebf03bc2c1c7e45f8178e",
    "fineweb_train_000002.bin": "d4ba8d2d740905a836a4222d2aff01750560e7d49a51a745a1b3235520aec2cd",
    "fineweb_train_000003.bin": "f70fa272369730fd9e39f1848bfe4daa3289b6e490c1e7762777f00ea8b66246",
    "fineweb_train_000004.bin": "d8d88850f89fe2030e866380e90d544167467d407080403b830b903543925835",
    "fineweb_train_000005.bin": "36cef3fc41863eb6731b41e986edde38e368853ee6deb87b928990bf445a119c",
    "fineweb_train_000006.bin": "d80e5b209983e636655fc738fdf281f0132f0298bf4c0848f7e7a6bea6ca06f0",
    "fineweb_train_000007.bin": "fa9f94dce61686b5467bc6cd3078424203db23ac254d804a6d8d5d440afc1e11",
}
DATAPREP_SCRIPTED_S = 39.1        # the reference preparation on the fleet's c3-highmem-44 (42 workers), measured by
                                  # W0-dataprepP6-q-measure on worker-6, 2026-10-21 (its foreground call, 39.14 s). The VM's
                                  # 32 cores (30 workers) took 63.5 s (build/check_dataprep.sh, 2026-10-21)


def t_pkg_dataprep(recs, pid, plan, structure, control):
    """A data-preparation package: the prefix package `plan` with each script's start-up data check, no data/fineweb10B/, and
    the raw documents plus data/fineweb.py in its place. One shared step makes every job's input (cpu-tasks: prefix jobs all
    read shards 1..n)."""
    import gen_dataprep as DP
    if not DATAPREP_OUTPUTS:
        raise SystemExit(f"{pid}: DATAPREP_OUTPUTS is not set (build/check_dataprep.sh prints it)")
    t = t_pkg_prefix(recs, pid, plan, structure, checked=True)
    t["files"] = [f for f in t["files"] if f["path"] != "data/fineweb10B/"]
    t["files"] += [
        {"path": "data/fineweb.py", "from": "artifacts/mng3/dataprep/fineweb.py",
         "how": "modded-nanogpt's data/fineweb.py at commit 4ea6b937, unchanged: the script that made data/fineweb10B/ from the "
                "Hub's FineWeb sample-10BT"},
        {"path": "data/raw/", "generated": {
            "by": "build/gen_raw_text.py", "args": [str(DATAPREP_TRAIN_SHARDS)],
            "inputs": {"shards": fineweb_files(DATAPREP_TRAIN_SHARDS), "vocab": DATAPREP_VOCAB},
            "outputs": DATAPREP_OUTPUTS},
         "from": f"the documents of kjj0/fineweb10B-gpt2's validation shard and train shards 1..{DATAPREP_TRAIN_SHARDS}, decoded "
                 "with GPT-2's vocabulary, 100,000 a file, in order; build/stage_task.py makes them and checks each file's sha256"},
    ]
    needs = {}
    for j in t["jobs"]:
        files = list(DP.data_files(recs[j["records"][0]]))
        needs[j["job_id"]] = files
        j["prep"] = {"kind": "data", "paths": ["data/fineweb10B/"], "produces": files, "shared": True,
                     "how": "the data files this job's script reads and checks at start-up; one shared step (task.data_prep) "
                            "makes them all"}
    t["data_prep"] = {
        "kind": "tokenise", "raw": "data/raw/", "script": "data/fineweb.py", "produces": "data/fineweb10B/",
        "reference": "artifacts/mng3/dataprep/fineweb_local.py",
        "scripted_write": "data/fineweb_local.py", "scripted_command": "python data/fineweb_local.py",
        "scripted_s": DATAPREP_SCRIPTED_S,
        "scripted_how": ("the scripted preparer writes the reference (data/fineweb.py reading data/raw/*.jsonl in order instead of the "
                         "Hub, its one change) to data/fineweb_local.py, runs it in the foreground from the task root with "
                         "fineweb.py's own os.cpu_count() - 2 workers, and launches once it exits. scripted_s is that call's time on "
                         "the fleet's c3-highmem-44 (42 workers), measured by W0-dataprepP6-q-measure on worker-6, "
                         "2026-10-21"),
        "horizon_s": PREP_HORIZON_S,
        "shard_sha256": DATAPREP_SHARD_SHA256,
        "label": ("each run is labelled with the sha256 of every data file a job read, against shard_sha256: a label, not a void "
                  "(the study, 2026-10-21). gpu-mock serves the recorded run whatever the bytes, so the label is the only "
                  "record of whether the agent's data were the recorded runs' data"),
        "needs": needs,
    }
    t["arm"] = "data-preparation"
    t["control_task"] = control
    t["env_kind"] = "mng3-dataprep"
    t["brief"]["text"] = t["brief"]["text"].replace(
        " Report each run's last validation loss.",
        " The scripts read tokenised FineWeb from data/fineweb10B/, which isn't in the tree yet: data/raw/ holds the documents "
        "those files are made from, in order, and data/fineweb.py is the script that made them. Report each run's last "
        "validation loss.", 1)
    if "data/raw/" not in t["brief"]["text"]:
        raise SystemExit(f"{pid}: the brief has no place for the data sentence")
    t["brief"]["provenance"] += "; the data sentence by gpu-tasks for the data-preparation arm (the study, 2026-10-21)"
    t["sources"].append("data/fineweb.py from modded-nanogpt at commit 4ea6b937; data/raw/ decoded from kjj0/fineweb10B-gpt2 at "
                        f"{FINEWEB['revision'][:12]} (build/gen_raw_text.py)")
    t["notes"] = [n for n in t["notes"] if "reads only data/fineweb10B/" not in n] + [
        "Each script is its wave-Q prefix script with one more change: at start-up it checks that every data file it reads "
        "(the validation shard and train shards 1..shards_read) is present and complete, and stops with the file's name if not "
        "(build/gen_dataprep.py). So a job launched before its data are ready fails at once rather than mid-run.",
        "data/raw/ holds the documents of the validation shard and train shards 1..8. Tokenised in order with fineweb.py's rule "
        "they give back the validation shard and train 1..7 byte for byte (cpu-tasks, 2026-10-21; build/check_dataprep.sh on "
        "these files, 2026-10-21: the reference preparation took 63.5 s on the VM's 30 workers). "
        "The jobs read the validation shard and train 1..2, so more text is present than the runs need, as with a real corpus.",
        "fineweb.py loads FineWeb from the Hub, which the box cannot reach; reading data/raw/ instead is the agent's change to "
        "make. As written it tokenises with os.cpu_count() - 2 workers; a loop of the agent's own may use one core. On the VM "
        "one worker takes about a minute per shard (cpu-tasks), so the step is minutes single-threaded and under a minute in "
        "parallel, against 6-10 minute jobs.",
        "Each script writes logs/<uuid>.txt, unique per run, and reads only data/fineweb10B/, so no two jobs collide.",
    ]
    return t


def t_pkg_screen(recs, pid, chains, top):
    """Screen, then confirm, from segments (the study, 2026-10-20: "results drive the next stage"): each setup's first segment
    is its screen, which saves a checkpoint; the `top` setups with the lowest validation loss at the screen's end continue their
    own recorded runs from it. Every confirm job is the rest of one recorded run, so no record is served twice."""
    jobs, files, seen, cands = [], [], set(), []
    for ch in chains:
        name, seed = ch[:2]
        cj, f = segment_chain_jobs(recs, name, seed, ch[2] if len(ch) > 2 else None)
        if len(cj) != 2:
            raise SystemExit(f"{pid}: {name} seed {seed} is not a two-part chain")
        if f["path"] not in seen:
            seen.add(f["path"])
            files.append(f)
        screen, confirm = cj
        screen["job_id"], screen["label"] = f"screen-{name}", f"screen {name} (seed {seed}): its first steps, saving a checkpoint"
        jobs.append(screen)
        cands.append({"variant": name, "records": screen["records"], "screen_job": screen["job_id"],
                      "confirm_records": confirm["records"], "confirm_launch": confirm["launch"],
                      "replicate_group": confirm["replicate_group"], "seed": seed})
    k = recs[jobs[0]["records"][0]]["steps"]
    if any(recs[j["records"][0]]["steps"] != k for j in jobs):
        raise SystemExit(f"{pid}: the screens stop at different steps")
    # A wave-Q confirm runs one more stretch, to step k2, not to the end of its schedule.
    ends = {recs[r]["identity"]["derived_from"]["steps"][1] for c in cands for r in c["confirm_records"]}
    full_end = all(recs[r]["identity"]["derived_from"]["steps"][1] == recs[recs[r]["identity"]["derived_from"]["record"]]["steps"]
                   for c in cands for r in c["confirm_records"])
    if not full_end and len(ends) != 1:
        raise SystemExit(f"{pid}: the confirms stop at different steps")
    k2 = None if full_end else ends.pop()
    follow = []
    for slot in range(1, top + 1):
        jid = f"confirm-pick{slot}"
        follow.append(jid)
        jobs.append({"job_id": jid, "label": (f"continue the #{slot} screened setup " + (f"to step {k2}" if k2 else
                                                                                         "to the end of its schedule")),
                     "records": None,
                     "launch": "torchrun --standalone --nproc_per_node=1 <setup>/<script> --seed <its seed> --resume "
                               "checkpoints/seed<its seed>-step" + str(k) + ".pt" + (f" --stop-at {k2}" if k2 else ""),
                     "bind": {"decision": "pick", "slot": slot},
                     "draw": "the bound setup's continuation record", "seed": None, "recorded_gpus": 1,
                     "recorded_gpu": "NVIDIA H100 80GB HBM3", "batch_policy": "fixed_global_grow_accumulation", "gpu_options": [1],
                     "after": [c["screen_job"] for c in cands], "consumes": [], "produces": [],
                     "candidate_pool": [r for c in cands for r in c["confirm_records"]]})
    for j in jobs:
        if j.get("bind"):
            j["disclosure"] = disclosure_of(j["candidate_pool"], recs, "one GPU",
                                            "over every setup's continuation, since the winners are not known in advance")
    shards = max(recs[x]["host"]["train_data"]["shards_read"] for c in cands for x in c["confirm_records"])
    lines = [j["launch"] for j in jobs if not j.get("bind")]
    n = len(chains)
    brief = (f"We're screening {COUNT_WORDS.get(n, n)} optimizer setups for our small-GPT pretraining, then "
             + ("finishing" if not k2 else "running further") + f" the best {COUNT_WORDS.get(top, top)}. screen.txt has each setup's screening run: the first {k} steps of its schedule, one "
             "GPU each, run from the task root; each saves a checkpoint when it stops. When every screen has finished, take the "
             f"{COUNT_WORDS.get(top, top)} setups with the lowest validation loss at step {k} and continue each from its own "
             + (f"checkpoint to the end of its schedule: the same command with --resume checkpoints/seed<its seed>-step{k}.pt in "
                f"place of --stop-at {k}" if not k2 else
                f"checkpoint for {k2 - k} more steps: the same command with --resume checkpoints/seed<its seed>-step{k}.pt "
                f"--stop-at {k2} in place of --stop-at {k}")
             + ", again on one GPU. Report every screen's validation loss and the finalists' final ones. "
             "The machine has four H100s.")
    return {
        "task_id": pid, "rung": 2, "env_kind": "mng3",
        "title": (f"Package {pid}: screen {n} setups for {k} steps, then " + (f"finish the best {top}" if not k2 else
                  f"continue the best {top} to step {k2}") + " from their checkpoints"),
        **({"wave": "Q", "wave_how": "short jobs to find failures fast (the study, 2026-10-21 ~01:12 UTC); the standard package "
                                     "confirms them"} if pid.endswith("-q") else {}),
        "brief": {"text": brief, "provenance": "written by gpu-tasks (the study, 2026-10-20: results drive the next stage); "
                                               "no deadline or time words (the study, 2026-10-20)"},
        "files": files + [{"path": "screen.txt", "content": "\n".join(lines) + "\n"},
                          {"path": "data/fineweb10B/", "hub_files": fineweb_files(shards),
                           "from": f"kjj0/fineweb10B-gpt2 at {FINEWEB['revision'][:12]}: the validation shard and the first {shards} "
                                   "training shards, the ones these runs read"}],
        "jobs": jobs,
        "decisions": [{"decision_id": "pick", "after": [c["screen_job"] for c in cands],
                       "rule": f"the {top} setups whose screen ends with the lowest validation loss at step {k}, ties broken by name",
                       "rule_spec": {"metric": "final_val_loss", "aggregate": "single", "direction": "min", "top": top},
                       "observable": f"the validation loss each screen prints at step {k}, as served", "launches": follow,
                       "slots": top, "shape_independent_of_outcome": True,
                       "shape_how": "every continuation is one GPU; their lengths differ by setup", "candidates": cands}],
        "measuring_run": dict(measuring_run(jobs[:n]), decisions="a scripted (oracle) plan binds each confirm job to its "
                              "recorded_winner: its records and its launch, after every screen"),
        "sources": ["modded-nanogpt track 3, commit 4ea6b937; records on torch 2.11.0; segment records by build/gen_segment.py"],
        "notes": ["Each screen is the first segment of one recorded run (build/gen_segment.py), and each confirm job is the rest of "
                  "that run, resumed from the screen's checkpoint, so the continuation's served values are the same run's.",
                  "Which setups win is the recorded screen values' to decide; the task records the recorded winners "
                  "(recorded_winner on each confirm job) for the scripted references, and a model agent decides for itself.",
                  "The setups use distinct seeds because a segment script names its checkpoint by seed and step.",
                  "Under replay a checkpoint holds the model's initial weights and an empty optimizer state (gpu-mock, 2026-10-20); "
                  "the next run loads it for real, and nothing reads its values."],
    }


def t_pkg_halving(recs, pid, chains):
    """Successive halving from segments (gpu-tasks' proposal, the study's priority 3, 2026-10-21): every setup screens for
    its first segment; the best half continue their own recorded runs for one more segment; the best of those for one more.
    Two decisions in a row, each read from losses the agent has to look at before it can launch. chains: [(variant, seed,
    bounds)] with bounds (0, k1, k2, k3)."""
    jobs, files, seen, cands = [], [], set(), []
    for name, seed, bounds in chains:
        cj, f = segment_chain_jobs(recs, name, seed, bounds)
        if len(cj) != 3:
            raise SystemExit(f"{pid}: {name} seed {seed} is not a three-part chain")
        if f["path"] not in seen:
            seen.add(f["path"])
            files.append(f)
        screen, second, third = cj
        screen["job_id"], screen["label"] = f"screen-{name}", f"screen {name} (seed {seed}): its first steps, saving a checkpoint"
        jobs.append(screen)
        cands.append({"variant": name, "seed": seed, "screen_job": screen["job_id"], "records": screen["records"],
                      "second": second, "third": third})
    k1 = recs[jobs[0]["records"][0]]["steps"]
    k2 = recs[cands[0]["second"]["records"][0]]["identity"]["derived_from"]["steps"][1]
    k3 = recs[cands[0]["third"]["records"][0]]["identity"]["derived_from"]["steps"][1]
    for c in cands:
        if (recs[c["records"][0]]["steps"] != k1
                or recs[c["second"]["records"][0]]["identity"]["derived_from"]["steps"][1] != k2
                or recs[c["third"]["records"][0]]["identity"]["derived_from"]["steps"][1] != k3):
            raise SystemExit(f"{pid}: the chains do not share their bounds")
    n, half = len(cands), len(cands) // 2

    def slot_job(jid, label, stage, decision, slot, after, start, stop):
        return {"job_id": jid, "label": label, "records": None,
                "launch": f"torchrun --standalone --nproc_per_node=1 <setup>/<script> --seed <its seed> --resume "
                          f"checkpoints/seed<its seed>-step{start}.pt --stop-at {stop}",
                "bind": {"decision": decision, "slot": slot}, "draw": "the bound setup's continuation record", "seed": None,
                "recorded_gpus": 1, "recorded_gpu": "NVIDIA H100 80GB HBM3", "batch_policy": "fixed_global_grow_accumulation",
                "gpu_options": [1], "after": after, "consumes": [], "produces": [],
                "candidate_pool": [r for c in cands for r in c[stage]["records"]]}

    second = [slot_job(f"round2-pick{i}", f"continue the #{i} screened setup to step {k2}", "second", "round2", i,
                       [c["screen_job"] for c in cands], k1, k2) for i in range(1, half + 1)]
    final = slot_job("round3-pick1", f"continue the better finalist to step {k3}", "third", "round3", 1,
                     [j["job_id"] for j in second], k2, k3)
    jobs += second + [final]
    for j in jobs:
        if j.get("bind"):
            j["disclosure"] = disclosure_of(j["candidate_pool"], recs, "one GPU",
                                            "over every setup's continuation, since the winners are not known in advance")
    shards = max(recs[x]["host"]["train_data"]["shards_read"] for c in cands for x in c["third"]["records"])
    lines = [j["launch"] for j in jobs if not j.get("bind")]

    def cand(c, stage, stage_records=None):
        out = {"variant": c["variant"], "seed": c["seed"], "records": c["records"], "screen_job": c["screen_job"],
               "confirm_records": c[stage]["records"], "confirm_launch": c[stage]["launch"],
               "replicate_group": c[stage]["replicate_group"]}
        if stage_records:
            out["stage_records"] = stage_records
        return out

    brief = (f"We're screening {COUNT_WORDS.get(n, n)} optimizer setups for our small-GPT pretraining and narrowing them down "
             f"to one. screen.txt has each setup's screening run: the first {k1} steps of its schedule, one GPU each, run from "
             "the task root; each saves a checkpoint when it stops. When every screen has finished, take the "
             f"{COUNT_WORDS.get(half, half)} setups with the lowest validation loss at step {k1} and continue each from its own "
             f"checkpoint for {k2 - k1} more steps: the same command with --resume checkpoints/seed<its seed>-step{k1}.pt "
             f"--stop-at {k2} in place of --stop-at {k1}. When those have finished, take the one with the lower validation loss "
             f"at step {k2} and continue it the same way for {k3 - k2} more, with --resume checkpoints/seed<its seed>-step{k2}.pt "
             f"--stop-at {k3}. Every continuation runs on one GPU. Report every run's validation loss at its last step. "
             "The machine has four H100s.")
    return {
        "task_id": pid, "rung": 2, "env_kind": "mng3",
        "title": f"Package {pid}: screen {n} setups for {k1} steps, continue the best {half} to step {k2}, then the best to {k3}",
        **({"wave": "Q", "wave_how": "short jobs to find failures fast (the study, 2026-10-21 ~01:12 UTC); the standard package "
                                     "confirms them"} if pid.endswith("-q") else {}),
        "brief": {"text": brief, "provenance": "written by gpu-tasks (its proposal, the study's priority 3, 2026-10-21: load "
                                               "monitoring and judgement); no deadline or time words (the study, 2026-10-20)"},
        "files": files + [{"path": "screen.txt", "content": "\n".join(lines) + "\n"},
                          {"path": "data/fineweb10B/", "hub_files": fineweb_files(shards),
                           "from": f"kjj0/fineweb10B-gpt2 at {FINEWEB['revision'][:12]}: the validation shard and the first {shards} "
                                   "training shards, the ones these runs read"}],
        "jobs": jobs,
        "decisions": [
            {"decision_id": "round2", "after": [c["screen_job"] for c in cands],
             "rule": f"the {half} setups whose screen ends with the lowest validation loss at step {k1}, ties broken by name",
             "rule_spec": {"metric": "final_val_loss", "aggregate": "single", "direction": "min", "top": half},
             "observable": f"the validation loss each screen prints at step {k1}, as served",
             "launches": [j["job_id"] for j in second], "slots": half, "shape_independent_of_outcome": True,
             "shape_how": "every continuation is one GPU; their lengths differ by setup",
             "candidates": [cand(c, "second") for c in cands]},
            {"decision_id": "round3", "after": [j["job_id"] for j in second], "among": "round2",
             "rule": f"of round 2's setups, the one whose continuation ends with the lower validation loss at step {k2}",
             "rule_spec": {"metric": "final_val_loss", "aggregate": "single", "direction": "min", "top": 1},
             "observable": f"the validation loss each round-2 run prints at step {k2}, as served",
             "launches": [final["job_id"]], "slots": 1, "shape_independent_of_outcome": True,
             "shape_how": "one continuation on one GPU",
             "candidates": [cand(c, "third", c["second"]["records"]) for c in cands]},
        ],
        "measuring_run": dict(measuring_run(jobs[:n]), decisions="a scripted (oracle) plan binds each round's jobs to their "
                              "recorded_winner: records and launch, after the round before it"),
        "sources": ["modded-nanogpt track 3, commit 4ea6b937; records on torch 2.11.0; segment records by build/gen_segment.py"],
        "notes": ["Each screen is the first segment of one recorded run (build/gen_segment.py); each later round resumes that "
                  "same run from the checkpoint the round before saved, so every served value is that run's.",
                  "Which setups go on is the recorded values' to decide; the task records the recorded winners (recorded_winner "
                  "on each bound job) for the scripted references, and a model agent decides for itself. Round 3 chooses only "
                  "among round 2's setups (decisions[1].among).",
                  "The setups use distinct seeds because a segment script names its checkpoint by seed and step.",
                  "Under replay a checkpoint holds the model's initial weights and an empty optimizer state (gpu-mock, 2026-10-20); "
                  "the next run loads it for real, and nothing reads its values."],
    }


# Goal-level packages (the study, 2026-10-21 ~08:45 MDT, via the analysis; GOAL-PACKAGES-2026-10-21.md):
# one question, and the agent decides what to run. A job is an open pool, one per candidate: any number of launches of that
# candidate's script, each served an unused recorded run of the setup (gpu-mock's task draw, 2026-10-21: the exact seed's
# record while unused, else another in descending seed order, never one record twice in a run).
GOAL_RESULT_NUM = re.compile(r"\b3\.\d{3,}|\bp\s*=|\bz\s*=|sqrt\(n\)|non-cherry-picked|mean validation loss", re.I)


def candidate_notes(setup, rec, code):
    """A candidate's NOTES.md: its submission's own description (README up to its first table, or the script's docstring),
    without the paragraphs that state the full run's measured results, which are not what the question asks about."""
    import gen_prep as P
    text, src = P.intro(setup, rec["source"]["submission"], code)
    paras = [p_ for p_ in re.split(r"\n\s*\n", text) if p_.strip() and not GOAL_RESULT_NUM.search(p_)]
    body = "\n\n".join(paras).strip()
    return (f"# {setup}\n\n{body}\n\n_(From {src}; paragraphs reporting the submission's own measured results are left "
            "out.)_\n")


def open_job(recs, setup, k, path_dir="candidates"):
    """One candidate as an open pool: every replay-ready prefix record of its script at stop step k, or, with k None, every
    replay-ready full recorded run of it on the image's torch."""
    import gen_prefix as GP
    parent, seeded = PREFIX_VARIANTS[setup]
    off_torch = []
    if k is None:
        g = parent
        whole = [r for r in groups(recs)[g] if r["readiness"]["replay"] and drawable(r) and not r["identity"].get("derived_from")]
        rs = sorted((r for r in whole if str((r["identity"]["trajectory"].get("environment") or {}).get("torch") or "").startswith("2.11.0")),
                    key=lambda r: r["identity"]["record_id"])
        off_torch = sorted(r["identity"]["record_id"] for r in whole if r not in rs)
    else:
        g = prefix_group(recs, parent, k)
        rs = sorted((r for r in groups(recs)[g] if r["readiness"]["replay"] and drawable(r)), key=lambda r: r["identity"]["record_id"])
    script = rs[0]["identity"]["argv"][0]
    path = f"{path_dir}/{setup}/{script}"
    r0 = rs[0]
    seeds_of = {r["identity"]["record_id"]: r["identity"]["seed"]["value"] for r in rs}
    vals = [r["result"]["final_val_loss"]["value"] for r in rs]
    # The same runs, whole and bound to the candidate's path (build/gen_goal.py), for a launch of an edited script.
    import gen_goal as GG
    full = [(r["identity"].get("derived_from") or {}).get("record", r["identity"]["record_id"]) + GG.SUFFIX for r in rs]
    missing = [x for x in full if x not in recs]
    if missing:
        raise SystemExit(f"{setup}: {len(missing)} path-bound full records missing (run build/gen_goal.py)")
    if k is None:
        # A whole-run pool: the path-bound record carries the run's own entry hash, so it serves the unedited script by exact
        # match and an edited one by path. Listing the parent too would put two records of one run under one exact key.
        rs = [recs[x] for x in full]
        seeds_of = {r["identity"]["record_id"]: r["identity"]["seed"]["value"] for r in rs}
        g = rs[0]["identity"]["replicate_group"]      # the path-bound records' own group, the one gpu-mock draws from
        full = []
    j = {"job_id": setup, "label": f"{setup}: any number of runs of its script, chosen by the agent",
         "replicate_group": g, "records": [r["identity"]["record_id"] for r in rs] + full, "draw": "open",
         "reference_records": [r["identity"]["record_id"] for r in rs],
         "records_how": ((f"each recorded run twice: its prefix record at step {k} (served to the unedited script by exact "
                          "match) and its whole run bound to " + path + " (served to an edited script by path). gpu-mock keys a "
                          "run's draws by the parent run, so one run is served once") if k else
                         ("each recorded run once, whole, bound to " + path + " with the run's own entry hash: served to the "
                          "unedited script by exact match and to an edited one by path")),
         "instances": {"min": 0, "max": len(rs), "how": "an open pool: every launch of this candidate's script is one instance, "
                       "served an unused recorded run of the setup; the agent decides how many"},
         "script_path": path, "seeded": seeded, "seeds_of": seeds_of,
         "launch": f"torchrun --standalone --nproc_per_node=<1|2|4> {path}" + (" --seed <N>" if seeded else ""),
         "gpu_options": [1, 2, 4], "seed": None, "recorded_gpus": r0["reference"]["gpu_count"],
         "recorded_gpu": r0["reference"]["gpu"], "batch_policy": r0["launch"]["batch_policy"]["value"],
         "after": [], "consumes": [], "produces": []}
    if off_torch and GOAL_EXCLUDED_NOTE:
        # the study, 2026-10-21: say why a replay-ready run of the script is not in the pool
        j["excluded_records"] = {"records": off_torch, "how": "recorded on another torch than the box's 2.11.0; every package draws "
                                 "only same-torch runs, so served runs match what the box's stack would produce and the pool's "
                                 "spread does not mix two stacks. On the task-filtered image they are never served"}
    stats = {"mean": round(statistics.mean(vals), 5), "sd": round(statistics.stdev(vals), 5), "n": len(vals),
             "metric": f"val_loss at step {k}" if k else "final val_loss of the whole schedule",
             "how": "over every recorded run in the pool"}
    return j, {"path": path, "from": r0["agent_view"]["script"]}, stats, rs


def runs_to_separate(a, b, z=2.0):
    """Runs per candidate for the mean difference of a and b to stand z standard errors clear, from their recorded spread."""
    d = abs(a["mean"] - b["mean"])
    return math.ceil((z * math.sqrt(a["sd"] ** 2 + b["sd"] ** 2) / d) ** 2) if d > 0 else None


def t_goal_choose(recs, pid, setups, k, title):
    """G1: which optimizer should we adopt? Open pools over recorded prefix runs at stop step k; the agent decides how many
    runs of which candidates, at what width, and when it knows enough."""
    jobs, files, stats, pools = [], [], {}, {}
    for s_ in setups:
        j, f, st, rs = open_job(recs, s_, k)
        jobs.append(j)
        files.append(f)
        files.append({"path": f"candidates/{s_}/NOTES.md",
                      "content": candidate_notes(s_, rs[0], open(os.path.join(REG_DIR, rs[0]["agent_view"]["script"])).read())})
        stats[s_], pools[s_] = st, rs
    best = min(stats, key=lambda s_: stats[s_]["mean"])
    plan = []
    for s_ in setups:
        need = 1 if s_ == best else runs_to_separate(stats[best], stats[s_])
        stats[s_]["runs_to_separate_from_best"] = need
    need_best = max(stats[s_]["runs_to_separate_from_best"] for s_ in setups if s_ != best)
    stats[best]["runs_to_separate_from_best"] = need_best
    for s_ in setups:
        n = max(1, stats[s_]["runs_to_separate_from_best"])
        if n > len(pools[s_]):
            raise SystemExit(f"{pid}: {s_} needs {n} runs to separate from {best} and has {len(pools[s_])} recorded")
        rs = sorted(pools[s_], key=lambda r: (r["identity"]["seed"]["value"] is None, r["identity"]["seed"]["value"] or 0))
        entry = {"job_id": s_, "count": n, "width": 1}
        if jobs[setups.index(s_)]["seeded"]:
            entry["seeds"] = [r["identity"]["seed"]["value"] for r in rs[:n]]
        plan.append(entry)
    shards = max(r["host"]["train_data"]["shards_read"] for rs in pools.values() for r in rs)
    brief = ("We're choosing the optimizer for our small-GPT pretraining runs and need a recommendation we can defend. "
             f"candidates/ has one folder per candidate: its training script and NOTES.md, the submission's own description. "
             f"We compare them at step {k:,} of their own schedules: each script stops after its validation there and prints the "
             "loss. Runs vary from seed to seed by about as much as the closest candidates differ, so one run each won't settle "
             "it; we need enough runs that our pick is clearly ahead of the runner-up, not just ahead in one draw. The scripts "
             "run with torchrun from the task root, each from its own folder (edit a script there if you need to), and read "
             "the tokenised data in data/fineweb10B/; the seeded ones take --seed. Report every run you made, each "
             "candidate's mean and spread, and your recommendation with how sure you are. The machine has four H100s.")
    return {
        "task_id": pid, "rung": 3, "env_kind": "mng3", "goal_level": True,
        "title": title,
        **({"wave": "Q", "wave_how": "the goal-level question at a screening length, for pilots (GOAL-PACKAGES-2026-10-21.md); "
                                     "its long twin asks it of whole runs"} if pid.endswith("-q") else {}),
        "brief": {"text": brief, "provenance": "written by gpu-tasks for the goal-level arm (the study, 2026-10-21 ~08:45 MDT, via "
                                               "the analysis): the question, not the runs; no "
                                               "procedure, no deadline or time words (the study, 2026-10-20); the GPU count "
                                               "stated, as in every brief, so underuse cannot be put down to a hidden machine "
                                               "(the analysis, 2026-10-21)"},
        "files": files + [{"path": "data/fineweb10B/", "hub_files": fineweb_files(shards),
                           "from": f"kjj0/fineweb10B-gpt2 at {FINEWEB['revision'][:12]}: the validation shard and the first {shards} "
                                   "training shards, the ones these runs read"}],
        "jobs": jobs,
        "candidates_stats": stats,
        "candidates_stats_how": (f"from every recorded run in each pool: the validation loss at step {k}; "
                                 "runs_to_separate_from_best is the runs per candidate for its mean to stand 2 standard errors "
                                 f"from the best's ({best}), from the recorded sds"),
        "reference_plan": {"instances": plan, "how": (
            "the scripted reference: for each candidate the runs that separate it from the best at 2 standard errors "
            "(candidates_stats), the best as many as its closest rival needs, every run on one GPU, launched packed. "
            "Its packed makespan is the oracle the expected file reports; a model run is also read against the hindsight bound "
            "of the instances it actually launched (the runner)")},
        "measuring_run": {"policy": "packed", "speed": 1, "plan": "reference_plan",
                          "how": "a speed-1 scripted packed run of reference_plan measures this task's host terms"},
        "sources": ["modded-nanogpt track 3, commit 4ea6b937; records on torch 2.11.0; prefix records by build/gen_prefix.py"],
        "notes": [f"Every launch is served a recorded run of the candidate's script cut at step {k} (a prefix record), so its "
                  "values are that run's. The pools hold every replay-ready run on the image's torch; gpu-mock serves the "
                  "exact seed's record while it is unused, else an unused one of the setup, never one twice in a run.",
                  "Nothing is synthetic. The candidates' recorded means and spreads are in candidates_stats, for the scorer; "
                  "the agent sees none of them, and NOTES.md leaves out the submissions' measured results.",
                  "The candidates scale differently with GPUs per run (gpu-mock's convert() from each record's recorded GPU "
                  "count), and one runs about three times slower per step than the rest."],
    }


GOAL_HORIZON_CAP_S = 30 * 3600
# A goal task whose agents run far more than its reference plan: the capped rule plans from the reference plan's runs, so it
# can stop an agent that samples until the rule is decided (task id -> (floor in seconds, why)).
GOAL_HORIZON_FLOOR_S = {
    "goal-pass-bar": (28 * 3600, (
        "G-passbar-astra-L0-v2 ran 36 width-4 runs of about 1,285 s back to back and still had soda-fade's ninth running when "
        "the 16-h horizon hit. An agent that samples until the rule is decided can reasonably take 10-12 runs per candidate: 12 "
        "per candidate, capped at each pool, serial at width 4, is 68,754 s of paced work (19.1 h), so 28 h leaves room for "
        "decisions and the report (the analysis, 2026-10-22 21:25Z; the study 21:20Z). The open-pool window "
        "still ends at the agent's report. Runs collected before 2026-10-22 ran under the 16-h horizon (57,600 s), recorded in "
        "each run's own task.json")),
}
# Off until goal-pass-bar's pilots end: turning it on changes the task file,
# which holds every queued line of the task on the machines' task-file check.
GOAL_EXCLUDED_NOTE = False
TRACK3_BAR, TRACK3_MARGIN = 3.28, 0.004      # modded-nanogpt track 3's acceptance rule: (bar - mean) * sqrt(n) >= margin


def passes_bar(vals):
    n = len(vals)
    return n > 0 and (TRACK3_BAR - statistics.mean(vals)) * math.sqrt(n) >= TRACK3_MARGIN


def pass_bar_plans(pools, seeded, gpus):
    """The two references of goal-pass-bar, both from the recorded runs in seed order (the order a seed-counting agent is
    served them; unseeded pools in record order):
    clairvoyant: for each candidate exactly the runs after which the rule first holds, packed on `gpus` one-GPU lanes;
    adaptive: an agent that knows no means. Whenever a GPU is free it starts another run of the unresolved candidate with the
    fewest runs made or running, and it stops a candidate once its finished runs pass. Runs started before the pass is seen
    still count (they run to the end), which is what such an agent does. Ideal: paced one-GPU durations, no host time."""
    order = {}
    for s_, rs in pools.items():
        order[s_] = sorted(rs, key=lambda r: (r["identity"]["seed"]["value"] is None, r["identity"]["seed"]["value"] or 0,
                                              r["identity"]["record_id"]))
    clair = {}
    for s_, rs in order.items():
        vals = []
        for r in rs:
            vals.append(r["result"]["final_val_loss"]["value"])
            if passes_bar(vals):
                break
        clair[s_] = len(vals) if passes_bar(vals) else None
    import heapq
    free = [0.0] * gpus
    heapq.heapify(free)
    done = {s_: [] for s_ in order}          # (finish time, value)
    started = {s_: 0 for s_ in order}
    running = []                             # (finish, setup, value)
    t, launches = 0.0, 0

    def passed(s_, at):
        return passes_bar([v for f, v in done[s_] if f <= at])
    while True:
        t = heapq.heappop(free)
        for f, s_, v in [x for x in running if x[0] <= t]:
            done[s_].append((f, v))
            running.remove((f, s_, v))
        open_ = [s_ for s_ in order if not passed(s_, t) and started[s_] < len(order[s_])]
        if not open_:
            if not running:
                break
            heapq.heappush(free, min(x[0] for x in running))
            continue
        s_ = min(open_, key=lambda x: (started[x], x))
        r = order[s_][started[s_]]
        started[s_] += 1
        launches += 1
        f = t + C.paced_duration(r, 1)
        running.append((f, s_, r["result"]["final_val_loss"]["value"]))
        heapq.heappush(free, f)
    end = max(f for s_ in done for f, _ in done[s_])
    return clair, {"makespan_s": round(end, 1), "instances": launches, "by_candidate": dict(started),
                   "passed": {s_: passed(s_, end) for s_ in order}}


def t_goal_pass_bar(recs, pid, setups, title):
    """goal-pass-bar: which candidates pass the track's own acceptance rule over all their runs, and which should we adopt?
    Whole recorded runs; the agent decides how many of each, at what width, and when a candidate has passed or will not."""
    jobs, files, stats, pools, seeded = [], [], {}, {}, {}
    for s_ in setups:
        j, f, st, rs = open_job(recs, s_, None)
        jobs.append(j)
        files.append(f)
        files.append({"path": f"candidates/{s_}/NOTES.md",
                      "content": candidate_notes(s_, rs[0], open(os.path.join(REG_DIR, rs[0]["agent_view"]["script"])).read())})
        vals = [r["result"]["final_val_loss"]["value"] for r in rs]
        st["bar_value_full_pool"] = round((TRACK3_BAR - statistics.mean(vals)) * math.sqrt(len(vals)), 5)
        stats[s_], pools[s_], seeded[s_] = st, rs, j["seeded"]
    clair, adaptive = pass_bar_plans(pools, seeded, 4)
    plan = []
    for s_ in setups:
        n = clair[s_]
        if n is None:
            raise SystemExit(f"{pid}: {s_} never passes within its {len(pools[s_])} recorded runs")
        stats[s_]["runs_to_pass_in_seed_order"] = n
        rs = sorted(pools[s_], key=lambda r: (r["identity"]["seed"]["value"] is None, r["identity"]["seed"]["value"] or 0,
                                              r["identity"]["record_id"]))
        e = {"job_id": s_, "count": n, "width": 1}
        if seeded[s_]:
            e["seeds"] = [r["identity"]["seed"]["value"] for r in rs[:n]]
        plan.append(e)
    shards = max(r["host"]["train_data"]["shards_read"] for rs in pools.values() for r in rs)
    brief = ("We're choosing the optimizer for our small-GPT pretraining runs. candidates/ has one folder per candidate: its "
             "training script and NOTES.md, the submission's own description. Our track accepts an optimizer only if its mean "
             f"final validation loss is below {TRACK3_BAR} with ({TRACK3_BAR} - mean) x sqrt(n) >= {TRACK3_MARGIN} over all n "
             "runs made, none dropped. Which of these candidates pass? We'll adopt the best of those that do. Each script trains "
             "its whole schedule and prints its final validation loss. They run with torchrun from the task root, each from its "
             "own folder (edit a script there if you need to), and read the tokenised data in data/fineweb10B/; the seeded "
             "ones take --seed. Report every run you made, each candidate's mean and the rule's value, which candidates pass, "
             "and which one you recommend. The machine has four H100s.")
    return {
        "task_id": pid, "rung": 3, "env_kind": "mng3", "goal_level": True, "set": "evaluation",
        "set_how": "the analysis, 2026-10-21: the first reported goal-level package (EVALUATION-PLAN.md)",
        "title": title,
        "brief": {"text": brief, "provenance": "written by gpu-tasks for the goal-level arm (the study, 2026-10-21 ~08:45 MDT, via "
                                               "the analysis): modded-nanogpt track 3's own acceptance rule "
                                               "(records/track_3_optimization/README.md), the question and not the runs; no "
                                               "procedure, no deadline or time words (the study, 2026-10-20); the GPU count stated"},
        "files": files + [{"path": "data/fineweb10B/", "hub_files": fineweb_files(shards),
                           "from": f"kjj0/fineweb10B-gpt2 at {FINEWEB['revision'][:12]}: the validation shard and the first {shards} "
                                   "training shards, the ones these runs read"}],
        "jobs": jobs,
        "candidates_stats": stats,
        "candidates_stats_how": ("from every recorded run in each pool: the whole run's final validation loss; bar_value_full_pool "
                                 f"is ({TRACK3_BAR} - mean) x sqrt(n) over the whole pool; runs_to_pass_in_seed_order is the "
                                 "fewest runs, taken in seed order (record order for an unseeded script), after which the rule "
                                 "holds"),
        "reference_plan": {
            "instances": plan,
            "how": ("clairvoyant: each candidate's runs exactly up to the rule's first pass in seed order, one GPU each, launched "
                    "packed. It knows the outcomes, so it is the experimenter's floor, not a plan an agent could follow; the "
                    "scripted measure runs it. adaptive is an agent that knows no means (pass_bar_plans)"),
            "adaptive": dict(adaptive, how=(
                "an agent that knows no means: on every free GPU it starts another one-GPU run of the unresolved candidate with "
                "the fewest runs made or running, and it drops a candidate once its finished runs pass. Runs already going when "
                "the pass shows still finish. Recorded runs in seed order, paced one-GPU durations, no host time"))},
        "measuring_run": {"policy": "packed", "speed": 1, "plan": "reference_plan",
                          "how": "a speed-1 scripted packed run of reference_plan measures this task's host terms"},
        "sources": ["modded-nanogpt track 3, commit 4ea6b937; whole recorded runs on torch 2.11.0; the acceptance rule from its "
                    "README"],
        "notes": ["Every launch is served a whole recorded run of the candidate's script: the exact seed's while unused, else an "
                  "unused one of the setup (gpu-mock's task draw), never one run twice. An edited script is served a run of the "
                  "setup by path (build/gen_goal.py). Nothing is synthetic.",
                  "newton-muon was recorded on two GPUs and gpu-mock's convert() makes it no faster on four; the others were "
                  "recorded on four or eight and grow their accumulation, so each GPU added shortens a run. Report each run's "
                  "width.",
                  "The candidates' recorded means and the rule's value over each whole pool are in candidates_stats, for the "
                  "scorer; the agent sees none of them, and NOTES.md leaves out the submissions' measured results."],
    }


# Brief variants for the scheduler rungs (the harness, 2026-10-21; design/SOLUTION-LADDER.md S1-S3): the same brief
# without the sentence that tells the agent to launch directly, which contradicts S1's "GPU work runs only inside SLURM jobs".
NO_LAUNCH_LINE_TASKS = ("pkg-p6-par-q", "pkg-stagger-q", "pkg-sc-q", "pkg-halving-q", "pkg-p20-tail-q")
LAUNCH_SENTENCE = re.compile(r"Run, each on one GPU from the task root with torchrun --standalone --nproc_per_node=1 "
                             r"<dir>/<script>: ")


def add_brief_variants(t):
    if t["task_id"] not in NO_LAUNCH_LINE_TASKS:
        return
    text = t["brief"]["text"]
    if LAUNCH_SENTENCE.search(text):
        v = LAUNCH_SENTENCE.sub("The scripts run with torchrun from the task root, one GPU each: ", text, count=1)
        how = ("the brief with its launch instruction (\"Run, each on one GPU from the task root with torchrun ...\") restated as "
               "a fact about the scripts (\"The scripts run with torchrun from the task root, one GPU each: ...\"); every other "
               "word is the same")
    else:
        v = text
        how = ("the same text: this brief names its commands only through screen.txt and says what each run is, not that the "
               "agent should launch it directly, so nothing in it contradicts S1")
    t["brief_variants"] = {"no_launch_line": {
        "text": v, "how": how,
        "for": "the scheduler rungs S1-S3 (the harness, design/SOLUTION-LADDER.md): the campaign appends S1's own text"}}


def t_trl_dpo_4gpu(recs):
    jobs, script = [], None
    for r in sorted((r for r in recs.values() if r["identity"]["record_id"].startswith("trl-dpo-h100-sxmx4-")), key=lambda r: r["identity"]["record_id"]):
        args = r["identity"]["argv"]
        lt = args[args.index("--loss_type") + 1] if "--loss_type" in args else "sigmoid"
        script = r["agent_view"]["script"]
        jobs.append(job_from_group(f"dpo-{lt}", f"DPO with loss {lt}", r["identity"]["replicate_group"], [r], gpu_options=[4],
                                   launch=f"accelerate launch --num_processes 4 examples/scripts/dpo.py {' '.join(args[1:])}"))
    return {
        "task_id": "trl-dpo-loss-variants-4gpu", "rung": 1,
        "title": "Control: three DPO loss variants, each a 4-GPU job",
        "brief": {"text": ("Compare three DPO losses on GPT-2 with examples/scripts/dpo.py: sigmoid (the default), apo_zero and apo_down. "
                           "Use the arguments in runs.sh; each run is a 4-GPU job with accelerate. Report the final reward accuracy and margin "
                           "of each. The machine has four H100s."),
                  "provenance": "written by gpu-tasks from the recorded runs' arguments"},
        "files": [{"path": "examples/scripts/dpo.py", "from": script}, {"path": "runs.sh", "content": "\n".join(j["launch"] for j in jobs) + "\n"},
                  {"path": "trl/", "from": "github.com/huggingface/trl at commit e5295792"}],
        "jobs": jobs,
        "sources": ["huggingface/trl W&B project, runs ig7dh8vc, 2qu7n4h9 and fwjt2dlg (2024-08-23)"],
        "notes": ["A control: each job needs all four GPUs at its recorded configuration, so the best schedule is serial and the gap to it "
                  "measures only the agent's overhead between launches.",
                  "batch_policy fixed_per_device: on fewer GPUs the global batch shrinks, the recorded curve stops applying and gpu-mock serves "
                  "the modelled path."],
    }


def t_mng3_shootout(recs):
    variants = [("baseline-muon", "20260610_tuned_baseline_3250"), ("adamh", "20260430_adamh"), ("muonh", "20260430_muonh"),
                ("normuonh", "20260430_normuonh"), ("psgd", "20260527_psgd")]
    jobs, files = [], []
    for name, sub in variants:
        g, rs = mng3_group(recs, sub)
        if seed_mechanism(rs[0]) != "none":
            raise SystemExit(f"{sub}: expected an unseeded script")
        path = f"optimizers/{name}/{rs[0]['identity']['argv'][0]}"
        files.append({"path": path, "from": rs[0]["agent_view"]["script"]})
        for k in range(5):
            jobs.append(job_from_group(f"{name}-run{k + 1}", f"{name}, run {k + 1}", g, rs, gpu_options=[1, 2, 4],
                                       launch=f"torchrun --standalone --nproc_per_node=<1|2|4> {path}"))
    return {
        "task_id": "mng3-optimizer-shootout", "rung": 1,
        "title": "Five optimizers on the GPT-2-small speedrun setup, five runs each",
        "brief": {"text": (
            "We're picking the optimizer for our small-GPT pretraining runs. optimizers/ has five training scripts, one per optimizer: "
            "baseline-muon, adamh, muonh, normuonh and psgd. They share the model (12 layers, 768 wide), the data (FineWeb, already tokenised "
            "under data/fineweb10B/) and the batch size; each sets its own step count. Each script runs on 1, 2 or 4 GPUs with torchrun and "
            "keeps the same global batch at any GPU count. Run every optimizer five times; the scripts are not seeded, so each run is a fresh "
            "sample. For each optimizer report the mean final validation loss and whether it clears 3.28 reliably: (3.28 - mean) x sqrt(n) >= 0.004. "
            "The machine has four H100s."),
            "provenance": "written by gpu-tasks from modded-nanogpt's records/track_3_optimization/README.md rules; it names no benchmark"},
        "files": files + [{"path": "data/fineweb10B/", "from": "kjj0/fineweb10B-gpt2 on the Hugging Face Hub (data/cached_fineweb10B.py)"}],
        "jobs": jobs,
        "sources": ["modded-nanogpt track 3, commit 4ea6b937"],
        "notes": ["All five groups were recorded on 8x H100 80GB HBM3; every launch on the 4-GPU box needs gpu-mock's convert() with "
                  "batch_policy fixed_global_grow_accumulation.", LINEAR_NOTE,
                  "The scripts carry no seed, so a relaunch draws a different recorded run of the same script."],
    }


def t_mng3_screen_confirm(recs):
    variants = [("baseline-muon", "20260610_tuned_baseline_3250", None), ("muonh-tuned", "20260611_muonh_tuned_aux_3250", None),
                ("muonh", "20260430_muonh", None), ("normuonh", "20260430_normuonh", None), ("skylight", "20260501_skylight001", None),
                ("contra-muon", "20260501_contra_muon", "train_gpt_simple_contra_muon_2.py"),
                ("contra-soft-muon", "20260509_contra_soft_muon", None), ("soap-all-hidden", "20260609_soap_f1_auxb2_clean", None)]
    jobs, files, cands = [], [], []
    for name, sub, script in variants:
        g, rs = mng3_group(recs, sub, script=script)
        path = f"candidates/{name}/{rs[0]['identity']['argv'][0]}"
        files.append({"path": path, "from": rs[0]["agent_view"]["script"]})
        cmd = f"torchrun --standalone --nproc_per_node=<1|2|4> {path}"
        jobs.append(job_from_group(f"screen-{name}", f"screen {name}", g, rs, seed=1, gpu_options=[1, 2, 4],
                                   launch=seeded_launch(cmd, rs[0], 1)))
        cands.append({"variant": name, "replicate_group": g, "records": [r["identity"]["record_id"] for r in rs],
                      "screen_job": f"screen-{name}", "seed_mechanism": seed_mechanism(rs[0])})
    follow = []
    for slot in (1, 2):
        for k in range(2, 6):
            jid = f"confirm-pick{slot}-seed{k}"
            follow.append(jid)
            jobs.append({"job_id": jid, "label": f"confirm the #{slot} screened candidate, seed {k}", "records": None,
                         "launch": "torchrun --standalone --nproc_per_node=<1|2|4> candidates/<winner>/<script>  # with seed " + str(k) + " where the script takes one",
                         "bind": {"decision": "pick-two", "slot": slot, "seed": k},
                         "draw": "from the bound candidate's group: the record of this seed when the script takes one, otherwise without replacement",
                         "seed": k, "recorded_gpus": 8, "recorded_gpu": "NVIDIA H100 80GB HBM3",
                         "batch_policy": "fixed_global_grow_accumulation", "gpu_options": [1, 2, 4],
                         "after": [f"screen-{c['variant']}" for c in cands], "consumes": [], "produces": []})
    pool = [r for c in cands for r in c["records"]]
    for j in jobs:
        if j.get("bind"):
            j["candidate_pool"] = pool
            j["disclosure"] = disclosure_of(pool, recs, "8x NVIDIA H100 80GB HBM3",
                                            "over every candidate's runs, since the winners are not known in advance")
    return {
        "task_id": "mng3-screen-then-confirm", "rung": 2,
        "title": "Screen eight optimizer setups once, then confirm the best two with four more runs each",
        "brief": {"text": (
            "We have eight candidate optimizer setups for our small-GPT pretraining runs in candidates/, one training script each. They share "
            "the model, the FineWeb data under data/fineweb10B/ and the batch size. Each runs on 1, 2 or 4 GPUs with torchrun and keeps its global "
            "batch at any count. Where a script takes a seed (a --seed flag or the SEED environment variable), use 1 for the first run and 2-5 "
            "after. Run each candidate once. Take the two with the lowest final validation loss and run each of those four more times. Report the "
            "five-run mean for the two finalists and whether each clears 3.28 reliably: (3.28 - mean) x sqrt(5) >= 0.004. The machine has four H100s."),
            "provenance": "written by gpu-tasks; the significance rule is modded-nanogpt track 3's"},
        "files": files + [{"path": "data/fineweb10B/", "from": "kjj0/fineweb10B-gpt2 on the Hugging Face Hub"}],
        "jobs": jobs,
        "decisions": [{"decision_id": "pick-two", "after": [f"screen-{c['variant']}" for c in cands],
                       "rule": "the two candidates whose screen run has the lowest final val_loss, ties broken by name",
                       "rule_spec": {"metric": "final_val_loss", "aggregate": "single", "direction": "min", "top": 2},
                       "observable": "the final val_loss each screen job prints, as served", "launches": follow, "slots": 2,
                       "shape_independent_of_outcome": True,
                       "shape_how": "every candidate was recorded on 8x H100 with the same model, batch and grow-accumulation policy, so the eight "
                                    "follow-up jobs have the same GPU options whichever two win; their durations differ by candidate",
                       "candidates": cands}],
        "sources": ["modded-nanogpt track 3, commit 4ea6b937"],
        "notes": ["Which two win depends on the records the daemon serves for the screen, so the online reference reads the served values "
                  "from the occupancy log rather than assuming an outcome.",
                  "Recorded on 8x H100; every launch on the box needs convert().", LINEAR_NOTE],
    }


def t_mls_kv(recs):
    by = collections.defaultdict(dict)
    for r in recs.values():
        k = r["identity"].get("source_key")
        if k and k["task"] == "llm-kv-selection-budgeting":
            by[k["baseline"]][k["label"]] = r
    jobs = []
    for base in sorted(by):
        for label in sorted(by[base]):
            r = by[base][label]
            jobs.append(job_from_group(f"{base}--{label}", f"{base} on {label}", r["identity"]["replicate_group"], [r], gpu_options=[1],
                                       launch=f"bash scripts/{label.replace('-', '_')}.sh  # with the {base} policy"))
    return {
        "task_id": "mls-llm-kv-selection-budgeting", "rung": 1,
        "title": "Four KV-cache retention policies on five long-context and reasoning workloads, one GPU per run",
        "brief": {"text": (
            "We're choosing a KV-cache retention policy for Qwen2.5-3B-Instruct at a 20% cache budget. policies/ has four: full_attention "
            "(no compression, the reference), streamingllm, expected_attention and lagkv. Evaluate each on the five workloads in scripts/: "
            "LongBench hotpotqa, passage_retrieval and repobench, LongBench v2, and GSM8K. Each evaluation is a one-GPU job. Report the score "
            "and retained fraction for every policy and workload. The machine has four H100s."),
            "provenance": "written by gpu-tasks from MLS-Bench tasks/llm-kv-selection-budgeting/task_description.md, without its benchmark framing"},
        "files": [{"path": "transformers-kv-lab/", "from": "MLS-Bench tasks/llm-kv-selection-budgeting scaffold at commit b8739449"}],
        "jobs": jobs,
        "sources": ["MLS-Bench tasks/llm-kv-selection-budgeting/leaderboard.csv, commit b8739449"],
        "notes": ["The selection task's scripts cd to /workspace/transformers-kv-lab unconditionally, while the quantisation task's scripts honour TRANSFORMERS_KV_LAB_DIR; a materialised per-method copy needs that cd line pointed at the copy, or every method would run the same evaluator (checked in the MLS-Bench clone, 2026-10-19)", "Twenty independent 1-GPU jobs of 5 to 124 minutes: a packing problem with a long tail in the GSM8K runs.",
                  "The jobs decode text and score it on the CPU, so they are served in gpu-mock's results-at-end mode, which cannot yet serve "
                  "a CPU-side final metric; each record's readiness says so. Each record's entry file, the evaluator with that "
                  "policy's edit applied and its TRACE_METRICS print removed, is under registry/artifacts/mls/scripts/; laying "
                  "those files out with the task's scripts is job-program work."],
    }


KV_QUICK = ("longbench-hotpotqa", "longbench-passage-retrieval", "longbench-repobench")
KV_METHODS = [("full_attention", "llm-kv-selection-budgeting", "selection"), ("streamingllm", "llm-kv-selection-budgeting", "selection"),
              ("expected_attention", "llm-kv-selection-budgeting", "selection"), ("lagkv", "llm-kv-selection-budgeting", "selection"),
              ("kivi_overlap_4bit", "llm-kv-adaptive-quantization", "quant"), ("kvtuner4_pertoken_qwen25_3b", "llm-kv-adaptive-quantization", "quant"),
              ("kvtuner4_kivi_qwen25_3b", "llm-kv-adaptive-quantization", "quant"), ("squat_subspace_4bit", "llm-kv-adaptive-quantization", "quant")]
KV_HARNESS = {"selection": "custom_selection_eval.py", "quant": "custom_quant_eval.py"}


def t_mls_kv_screen_confirm(recs):
    by = {}
    for r in recs.values():
        k = r["identity"].get("source_key")
        if k and k["task"] in ("llm-kv-selection-budgeting", "llm-kv-adaptive-quantization"):
            by[(k["task"], k["baseline"], k["label"])] = r
    jobs, cands = [], []
    for name, task, fam in KV_METHODS:
        for label in KV_QUICK:
            r = by[(task, name, label)]
            jobs.append(job_from_group(f"screen-{name}--{label}", f"screen {name} on {label}", r["identity"]["replicate_group"], [r],
                                       gpu_options=[1], launch=f"bash scripts/{fam}/{label.replace('-', '_')}.sh  # in methods/{name}/"))
        if name != "full_attention":
            cands.append({"variant": name, "harness": KV_HARNESS[fam],
                          "records": [by[(task, name, l)]["identity"]["record_id"] for l in KV_QUICK + ("gsm8k",)],
                          "screen_jobs": [f"screen-{name}--{l}" for l in KV_QUICK]})
    ref = by[("llm-kv-selection-budgeting", "full_attention", "gsm8k")]
    gpu = ref["reference"]["gpu"]
    jobs.append(job_from_group("reference-gsm8k", "full_attention on gsm8k, the uncompressed reference", ref["identity"]["replicate_group"], [ref],
                               gpu_options=[1], launch="bash scripts/selection/gsm8k.sh  # in methods/full_attention/"))
    screens = [j["job_id"] for j in jobs if j["job_id"].startswith("screen-")]
    pool = [by[(t, n, "gsm8k")]["identity"]["record_id"] for n, t, _ in KV_METHODS if n != "full_attention"]
    follow = []
    for slot in (1, 2):
        jid = f"confirm-pick{slot}-gsm8k"
        follow.append(jid)
        jobs.append({"job_id": jid, "label": f"the #{slot} screened method on gsm8k", "records": None,
                     "bind": {"decision": "pick-two", "slot": slot, "select_by": {"source_key.label": "gsm8k"}},
                     "draw": "the bound method's gsm8k record", "seed": 42, "recorded_gpus": 1, "recorded_gpu": gpu, "batch_policy": None,
                     "gpu_options": [1], "after": screens, "consumes": [], "produces": [], "candidate_pool": pool,
                     "launch": "bash scripts/<family>/gsm8k.sh  # in methods/<winner>/, the family and method the decision binds",
                     "disclosure": disclosure_of(pool, recs, f"1x {gpu}", "over every candidate's gsm8k run, since the winners are not known in advance")})
    return {
        "task_id": "mls-kv-compression-screen-then-confirm", "rung": 2,
        "title": "Screen eight KV-cache compression methods on three quick workloads, then run the best two and the reference on GSM8K; every job one GPU",
        "brief": {"text": (
            "We're choosing a KV-cache compression method for serving Qwen2.5-3B-Instruct. There are seven candidates in two families, both in "
            "transformers-kv-lab/: three token-retention policies at a 20% cache budget (streamingllm, expected_attention, lagkv; evaluated with "
            "custom_selection_eval.py) and four low-bit quantisers at a 4-bit budget (kivi_overlap_4bit, kvtuner4_pertoken_qwen25_3b, "
            "kvtuner4_kivi_qwen25_3b, squat_subspace_4bit; evaluated with custom_quant_eval.py). methods/ holds one copy of the right evaluator "
            "per method with that method applied; full_attention is the uncompressed reference. First screen all eight on the three quick LongBench "
            "workloads (hotpotqa, passage_retrieval, repobench) with scripts/<family>/<workload>.sh. Then take the two compression methods with the "
            "highest mean score over those three workloads and run them on GSM8K; run full_attention on GSM8K too, as the reference. Every "
            "evaluation is a one-GPU job. Report every score, with the retained fraction or effective bits where the harness prints them. The "
            "machine has four H100s."),
            "provenance": "written by gpu-tasks from MLS-Bench tasks/llm-kv-selection-budgeting and tasks/llm-kv-adaptive-quantization, which share "
                          "the model, the scaffold directory and these workloads; the benchmark framing is removed"},
        "files": [{"path": "transformers-kv-lab/", "from": "MLS-Bench's transformers-kv-lab scaffold at commit b8739449, holding both evaluators"},
                  {"path": "methods/<name>/", "from": "the evaluator with each baseline's edit applied, from the two tasks' edits/ directories"}],
        "jobs": jobs,
        "decisions": [{"decision_id": "pick-two", "after": screens,
                       "rule": "the two compression methods, full_attention excluded, with the highest mean final_score over the three screen workloads; ties broken by name",
                       "rule_spec": {"metric": "final_score", "aggregate": "mean", "direction": "max", "top": 2},
                       "observable": "the final_score each screen job prints in its TEST_METRICS line", "launches": follow, "slots": 2,
                       "shape_independent_of_outcome": True,
                       "shape_how": "every candidate's GSM8K evaluation is one 1-GPU job, so the two follow-up jobs have the same shape whichever "
                                    "methods win; only their durations differ, from about 2 h for the retention policies to 8 h for squat",
                       "candidates": cands}],
        "sources": ["MLS-Bench tasks/llm-kv-selection-budgeting/leaderboard.csv and tasks/llm-kv-adaptive-quantization/leaderboard.csv, commit b8739449"],
        "notes": ["The selection task's scripts cd to /workspace/transformers-kv-lab unconditionally, while the quantisation task's scripts honour TRANSFORMERS_KV_LAB_DIR; a materialised per-method copy needs that cd line pointed at the copy, or every method would run the same evaluator (checked in the MLS-Bench clone, 2026-10-19)", "Every job is one GPU: each evaluator loads one model onto one device. The screen is 24 jobs of 5 to 46 minutes, so filling "
                  "the box means running four at once; the reference's GSM8K run depends on nothing and can start at once, which an agent that "
                  "waits for the screen before launching anything long misses.",
                  "Every screen job has one record, so the recorded scores fix the winners (decisions[].recorded_outcome), and the stats use "
                  "their GSM8K durations. The critical path is the longest screen job followed by the longest finalist's GSM8K run; the lower "
                  "bound comes from starting that run the moment its screen jobs end.",
                  "The jobs decode text and score it on the CPU, so they are served in gpu-mock's results-at-end mode, which cannot yet serve "
                  "a CPU-side final metric; each record's readiness says so. Each record's entry file, the evaluator with that "
                  "method's edit applied and its TRACE_METRICS print removed, is under registry/artifacts/mls/scripts/; laying "
                  "those files out with the task's scripts is job-program work."],
    }


def t_mls_pretrain(recs, task, title, words):
    by = collections.defaultdict(dict)
    for r in recs.values():
        k = r["identity"].get("source_key")
        if k and k["task"] == task:
            by[k["baseline"]][k["label"]] = r
    jobs = []
    for base in sorted(by):
        tr, ev = by[base]["gpt-345m"], by[base]["lm-eval-345m"]
        ck = f"out/{base}/ckpt_gpt-345m.pt"
        jobs.append(job_from_group(f"train-{base}", f"pretrain GPT-2 medium with {base}", tr["identity"]["replicate_group"], [tr],
                                   gpu_options=[1, 2, 3, 4], produces=[ck],
                                   launch=f"ENV=gpt-345m OUTPUT_DIR=out/{base} bash scripts/gpt_345m.sh  # in variants/{base}/"))
        jobs.append(job_from_group(f"eval-{base}", f"zero-shot eval of the {base} model", ev["identity"]["replicate_group"], [ev],
                                   gpu_options=[1], after=[f"train-{base}"], consumes=[ck],
                                   launch=f"OUTPUT_DIR=out/{base} bash scripts/lm_eval_345m.sh"))
    return {
        "task_id": f"mls-{task}", "rung": 2, "title": title,
        "brief": {"text": (
            f"We want to know which {words} works best for GPT-2 medium (355M) pretraining on ClimbMix. The repo is nanoGPT with our "
            f"custom_pretrain.py; the three variants to compare are in variants/ ({', '.join(sorted(by))}). For each, train with "
            "scripts/gpt_345m.sh (12,030 steps, about 7B tokens; it uses every visible GPU and keeps the global batch), then run "
            "scripts/lm_eval_345m.sh on its checkpoint for HellaSwag, ARC-Easy, PIQA and WinoGrande. Report validation loss, WikiText-2 and "
            "LAMBADA perplexity and the four accuracies for each variant. The machine has four H100s."),
            "provenance": f"written by gpu-tasks from MLS-Bench tasks/{task}/task_description.md, without its benchmark framing"},
        "files": [{"path": "variants/", "from": f"MLS-Bench tasks/{task} nanoGPT scaffold at commit b8739449, one copy per variant with its edit applied"}],
        "jobs": jobs,
        "sources": [f"MLS-Bench tasks/{task}/leaderboard.csv, commit b8739449"],
        "notes": ["Each evaluation reads its training job's checkpoint: a dependency through a file, fixed in shape (one 1-GPU job per variant).",
                  "Training was recorded on 4 GPUs; GRAD_ACCUM=12 divides by 1, 2, 3 and 4, so a 3-GPU train beside a 1-GPU eval is a valid packing.",
                  "The records hold whole-job durations and final metrics only, no curve. The evaluations are served in results-at-end mode; "
                  "the training jobs print their loss every 10 steps, so they need a modelled curve as well."],
    }


def catalog(reg, recs, tasks):
    src = lambda r: r["identity"]["record_id"].split("-")[0]
    by_src = collections.Counter(src(r) for r in recs.values())
    ready = collections.Counter(src(r) for r in recs.values() if r["readiness"]["replay"])
    status = collections.Counter((src(r), r["outcome"]["status"]) for r in recs.values())
    names = {"mng3": "modded-nanogpt track 3", "trl": "TRL W&B project", "mls": "MLS-Bench baselines"}
    L = ["# GPU task registry: catalog", "",
         "Generated by `build/gen_tasks.py`; regenerate rather than edit. This is an index, not instructions. The contract is "
         "`RECORD-FORMAT.md` and the rules are `AGENTS.md`, both in `experiments/gpu-tasks/`.", "",
         "## Records", "", "| Source | Records | Replay-ready | Outcomes |", "|---|---|---|---|"]
    for s in sorted(by_src):
        oc = ", ".join(f"{k[1]} {v}" for k, v in sorted(status.items()) if k[0] == s)
        L.append(f"| {names.get(s, s)} | {by_src[s]} | {ready[s]} | {oc} |")
    L += ["", "## Tasks", "", "| Task | Rung | Class | Jobs | Replayed GPU-hours | Lower-bound makespan, h | Serial, h | Replay-ready jobs |",
          "|---|---|---|---|---|---|---|---|"]
    for t in tasks:
        st = t["stats"]
        L.append(f"| `{t['task_id']}` | {t['rung']} | {t['task_class'].replace('_', '-')} | {st['jobs']} | {st['recorded_gpu_seconds'] / 3600:.1f} | "
                 f"{st['lower_bound_makespan_s'] / 3600:.2f} | {st['serial_makespan_s'] / 3600:.2f} | "
                 f"{t['readiness']['replay_ready_jobs']} of {t['readiness']['jobs_with_records']} |")
    L += ["", "Hours are at recorded speed on the 4-GPU box, before the speed knob, and are what gpu-mock's replay paces: host time it "
          "does not pace (process start, model loading, checkpoint writes) is left out. Durations off the recorded GPU count come from gpu-mock's "
          "`convert()`, the same function its daemon and the reference-schedule solver use; each task's `stats` says how. The lower bound "
          "ignores the agent's overhead and every constraint the solver sees, so it is a floor under the reference schedule, not the schedule.", "",
          "The class is the study's (`design/IDEA.md`, \"Reference schedules\", 2026-10-19): gaps are reported per class and never pooled "
          "across classes. *Fan-out required*: no job can use the whole box, so filling it means running several jobs at once. *Width-flexible*: "
          "some job can take the whole box, so one job at a time at full width is close to optimal. *Control*: every job needs the whole box, so "
          "the best schedule is serial. Each task's `task_class_how` gives the rule as applied.", ""]
    for t in tasks:
        L += [f"### `{t['task_id']}`", "", t["title"] + f". Class: {t['task_class'].replace('_', '-')}.", ""]
        L += [f"- {n}" for n in t.get("notes", [])]
        ss = t["shared_state"]
        if ss["default_shared"]:
            n_jobs = len(t["jobs"])
            parts = []
            for g in ss["default_shared"]:
                if len(g["jobs"]) == n_jobs:
                    parts.append(f"all {n_jobs} jobs write `{g['path']}`")
                elif len(g["jobs"]) > 6:
                    rest = [j["job_id"] for j in t["jobs"] if j["job_id"] not in g["jobs"]]
                    parts.append(f"{len(g['jobs'])} of {n_jobs} jobs write `{g['path']}`, all but " + ", ".join(f"`{j}`" for j in rest))
                else:
                    parts.append(f"{', '.join(f'`{j}`' for j in g['jobs'])} write `{g['path']}`")
            L.append("- Shared state by default: " + "; ".join(parts)
                     + f". The agent isolates them with `{ss['isolation']['flag']}`"
                     # a builder's own output_isolation (the SF checkpoint package, 2026-11-02) may say how instead of exempt
                     + (f", which is {ss['isolation']['exempt']}." if "exempt" in ss["isolation"]
                        else f": {ss['isolation'].get('how') or 'the builder states no more'}."))
        else:
            L.append("- Shared state by default: none" + (f"; {', '.join(f'`{j}`' for j in ss['by_design'])} read a producer's output by design" if ss["by_design"] else "") + ".")
        if t["readiness"]["blockers"]:
            L.append("- Blockers, by count of records: " + "; ".join(f"{k} ({v})" for k, v in t["readiness"]["blockers"].items()) + ".")
        L.append("")
    L += ["## Recorded failures", "",
          "Every non-completed record carries `failures[]` with a logged `cause` where the source kept one, a cause-matched `collision` "
          "class, and `replay`: `fire` or `suppress`. Under the study's ruling (`design/IDEA.md`, shared-state collisions), a failure "
          "a collision caused in the original project never fires in replay, and neither does a failure in a Hub push, since tasks never push. "
          "A suppressed failure that cut the run short (`continuation: sibling_required`) makes its record undrawable; a sibling of the same "
          "configuration serves instead, and tasks list such records under `excluded_records`.", "",
          "| Source | Failures | Fire | Suppress | Collision classes |", "|---|---|---|---|---|"]
    for sname in sorted(by_src):
        fs = [f for r in recs.values() if src(r) == sname for f in r["failures"]]
        cls = collections.Counter(f"{(f['collision'] or {}).get('class')}{' (matched)' if (f['collision'] or {}).get('matched') else ''}" for f in fs if f.get("collision"))
        L.append(f"| {names.get(sname, sname)} | {len(fs)} | {sum(f['replay']['value'] == 'fire' for f in fs)} | "
                 f"{sum(f['replay']['value'] == 'suppress' for f in fs)} | {', '.join(f'{k} {v}' for k, v in sorted(cls.items())) or 'none'} |")
    L.append("")
    sup = [(r, f) for r in recs.values() for f in r["failures"] if f["replay"]["value"] == "suppress"]
    if sup:
        L += ["Suppressed:", ""]
        for r, f in sorted(sup, key=lambda x: x[0]["identity"]["record_id"]):
            L.append(f"- `{r['identity']['record_id']}`: {f['at'].replace('_', ' ')}, {(f['collision'] or {}).get('class')}"
                     f"{' (matched)' if (f['collision'] or {}).get('matched') else ''}; {f['replay']['continuation']}. {f['replay']['how']}.")
        L.append("")
    C.write_text(os.path.join(reg, "CATALOG.md"), "\n".join(L))


def complete_task(t, recs, reg, sib_pools):
    """One builder's task through every per-task step main applies (build/finish_one_task.py uses it too, so a single
    task finished alone is byte-identical to a full regeneration's)."""
    import siblings as SIB
    t = {"schema": C.SCHEMA_TASK, "box": BOX, **t}
    post = t.pop("_post_finish", None)
    t = finish(t, recs, reg)
    if post:
        # a builder's own step after finish (pkg-sf-zero-lr, 2026-11-02: its horizon counts every initially broken attempt too),
        # named by module so a task file holds no callable
        t = getattr(__import__(post), "post_finish")(t)
    if t.get("environment"):
        image_for_box(t)
    t["task_class"], t["task_class_how"] = C.task_class(t["jobs"], t["box"]["gpus"])
    # Detectability (the study, 2026-10-20, design/IDEA.md): a model run never gets a task that can serve one recorded run,
    # or part of one, to two jobs, because an agent reading its logs sees the repeated losses. Scripted runs read no values.
    problems = C.repeated_records(t, recs)
    if problems and t["task_id"].startswith("pkg-"):
        raise SystemExit(f"{t['task_id']}: a package must serve every job a distinct recorded run: " + "; ".join(problems[:3]))
    t["model_runs"] = {"allowed": not problems, "problems": problems,
                       "how": "allowed when no two jobs can be served the same recorded run or overlapping steps of one "
                              "(common.repeated_records): a seed-substituted job, two prefix jobs cut from one parent, or a "
                              "prefix and its parent's full run all repeat losses an agent can see (the study, 2026-10-20, "
                              "from the p6 SFT pilot). Scripted runs read no values, so they may run any task"}
    add_brief_variants(t)
    # gpu-mock's rerun design (b), agreed 2026-10-21: per exact-seed job, the other recorded runs of the same script and stop
    # step, in serving order, that a rerun is served instead of the same record again (build/siblings.py).
    sib, sib_how = SIB.task_siblings(t, recs, sib_pools)
    for j in t["jobs"]:
        if j["job_id"] in sib:
            j["rerun_siblings"] = sib[j["job_id"]]
            j["rerun_siblings_how"] = sib_how[j["job_id"]]
    if sib:
        t["rerun_siblings_how"] = (
            "gpu-mock serves a job's primary record once per run; a later lookup of it by another job key (a rerun) gets "
            "the first unused id of that job's rerun_siblings, served by id whatever its bound_path, then the primary again "
            "(draw: reused) when the list is spent. Siblings are other recorded runs of the same script and stop step, none "
            "overlapping any record any job of the task lists or another job's siblings (build/siblings.py; validate.py "
            "checks it). Segments have none: a resumed segment follows the run of the checkpoint it loads (gpu-mock's "
            "lineage rule). Evidence that a sibling is a faithful rerun: newton-muon's one script fixes no seed, so its 15 "
            "recorded runs are reruns of identical code, and they vary as much at step 250 as other setups do across "
            "seeds (survey/siblings-2026-10-21.json, rerun_spread)")
    return t


def main():
    reg = sys.argv[1]
    recs = C.load_records(os.path.join(reg, "records"))
    pkg_long = [
        lambda r: t_pkg_long(r, "pkg-p3-par-long", [("contra-soft-muon", 2, ()), ("radial-damp", 2, ()), ("soda-fade", 2, ())],
                             "3 independent jobs",
                             "We're comparing three optimizer setups for our small-GPT pretraining. Each directory has one training script: "
                             "contra-soft-muon/, radial-damp/ and soda-fade/. Run each once with --seed 2, on one GPU, from the task root: "
                             "torchrun --standalone --nproc_per_node=1 <dir>/<script> --seed 2. The three runs are independent of one "
                             "another. Report each run's final validation loss."),
        lambda r: t_pkg_long(r, "pkg-p6-par-long", [("contra-soft-muon", 2, ()), ("contra-soft-muon", 3, ()), ("radial-damp", 2, ()),
                                                    ("radial-damp", 3, ()), ("soda-fade", 2, ()), ("muonh-tuned", 1, ())],
                             "6 independent jobs",
                             "We're comparing four optimizer setups for our small-GPT pretraining, with a second seed for two of them. "
                             "Each directory has one training script: contra-soft-muon/, radial-damp/, soda-fade/ and muonh-tuned/. Run, "
                             "each on one GPU from the task root with torchrun --standalone --nproc_per_node=1 <dir>/<script>: "
                             "contra-soft-muon with --seed 2 and with --seed 3; radial-damp with --seed 2 and with --seed 3; soda-fade "
                             "with --seed 2; and muonh-tuned once (it takes no seed). The six runs are independent of one another. "
                             "Report each run's final validation loss."),
        lambda r: t_pkg_long(r, "pkg-p6-chain-long", [("contra-soft-muon", 1, ()), ("contra-soft-muon", 2, ("contra-soft-muon-run1",)),
                                                      ("contra-soft-muon", 3, ("contra-soft-muon-run2",)), ("contra-soft-muon", 4, ("contra-soft-muon-run3",)),
                                                      ("contra-soft-muon", 5, ("contra-soft-muon-run4",)), ("contra-soft-muon", 6, ("contra-soft-muon-run5",))],
                             "6 jobs in sequence",
                             "Run contra-soft-muon/contra-muon-to-soft-muon.py six times on one GPU, with seeds 1 to 6 (--seed 1, then "
                             "--seed 2, and so on), from the task root with torchrun --standalone --nproc_per_node=1. Run them in order, "
                             "one after another: start each seed only after the previous one has finished. Report each run's final "
                             "validation loss."),
    ]
    PREFIX_RECS[0] = recs
    pkg_prefix = [
        lambda r: t_pkg_prefix(r, "pkg-p3-par", prefix_plan([("contra-soft-muon", 1), ("radial-damp", 1), ("muonh-tuned", 1)]),
                               "3 independent jobs"),
        lambda r: t_pkg_prefix(r, "pkg-p6-par", prefix_plan([("contra-soft-muon", 1), ("soda-fade", 1), ("radial-damp", 1),
                                                              ("aurora-ema-ref", 1), ("muonh-tuned", 1), ("newton-muon", 1)]),
                               "6 independent jobs"),
        lambda r: t_pkg_prefix(r, "pkg-p10-par", prefix_plan([("contra-soft-muon", 2), ("soda-fade", 1), ("radial-damp", 2),
                                                               ("aurora-ema-ref", 2), ("ema-nesterov-aurora", 1), ("muonh-tuned", 1),
                                                               ("newton-muon", 1)]),
                               "10 independent jobs"),
        lambda r: t_pkg_prefix(r, "pkg-p20-par", prefix_plan([("contra-soft-muon", 5), ("soda-fade", 3), ("radial-damp", 3),
                                                               ("aurora-ema-ref", 3), ("ema-nesterov-aurora", 2), ("circuit-muon", 1),
                                                               ("muonh-tuned", 1), ("newton-muon", 1), ("ortho", 1)]),
                               "20 independent jobs"),
        lambda r: t_pkg_prefix(r, "pkg-p20-tail", prefix_plan([("contra-soft-muon", 4), ("soda-fade", 3), ("radial-damp", 3),
                                                                ("aurora-ema-ref", 3), ("ema-nesterov-aurora", 2), ("circuit-muon", 1),
                                                                ("muonh-tuned", 1), ("newton-muon", 1), ("ortho", 1)]),
                               "20 independent jobs, one of them a full run", full=[("contra-soft-muon-full", "contra-soft-muon", 5)]),
    ]
    builders = [t_trl_battery, lambda r: t_trl_battery(r, quick=True), t_trl_dpo_4gpu, *pkg_long, *pkg_prefix,
                lambda r: t_sft_pilot(r, 3), lambda r: t_sft_pilot(r, 6), *segment_builders(),
        lambda r: t_pkg_prep(r, "pkg-prep-p6", [("contra-soft-muon", 0), ("soda-fade", 0), ("radial-damp", 0),
                                                ("aurora-ema-ref", 0), ("muonh-tuned", None), ("newton-muon", None)],
                             "6 independent jobs"),
        lambda r: t_pkg_prep(r, "pkg-prep-p10", [("contra-soft-muon", 0), ("contra-soft-muon", 1), ("soda-fade", 0),
                                                 ("radial-damp", 0), ("radial-damp", 2), ("aurora-ema-ref", 0),
                                                 ("aurora-ema-ref", 1), ("ema-nesterov-aurora", 8), ("muonh-tuned", None),
                                                 ("newton-muon", None)], "10 independent jobs", per_seed=True),
        lambda r: t_pkg_prep(r, "pkg-prep-p20", [("contra-soft-muon", s) for s in range(5)] + [("soda-fade", s) for s in range(3)]
                             + [("radial-damp", s) for s in (0, 2, 3)] + [("aurora-ema-ref", s) for s in range(3)]
                             + [("ema-nesterov-aurora", 8), ("ema-nesterov-aurora", 9), ("circuit-muon", 6), ("muonh-tuned", None),
                                ("newton-muon", None), ("ortho", None)], "20 independent jobs", per_seed=True),
        lambda r: t_pkg_screen(r, "pkg-sc-p5", [("contra-soft-muon", 4), ("soda-fade", 1), ("radial-damp", 2),
                                                ("ema-nesterov-aurora", 14), ("circuit-muon", 6)], 2),
        lambda r: t_pkg_prefix(r, "pkg-p6-par-q", prefix_plan([("contra-soft-muon", 1), ("soda-fade", 1), ("radial-damp", 1),
                                                               ("ema-nesterov-aurora", 1), ("muonh-tuned", 1), ("newton-muon", 1)],
                                                              q=True), "6 independent short jobs"),
        lambda r: t_pkg_prep(r, "pkg-prep-p6-q", [("contra-soft-muon", 0), ("soda-fade", 0), ("radial-damp", 0),
                                                  ("ema-nesterov-aurora", 8), ("muonh-tuned", None), ("newton-muon", None)],
                             "6 independent short jobs", q=True),
        lambda r: t_pkg_segment(r, "pkg-p10-5c5p-q", [("ema-nesterov-aurora", 15, (0, 125, 250, 375, 500, 625))],
                                [e for e in prefix_plan([("contra-soft-muon", 1), ("soda-fade", 1), ("radial-damp", 1),
                                                         ("muonh-tuned", 1), ("newton-muon", 1)], q=True)],
                                "a chain of 5 and 5 independent short jobs", segment_measure(own="its own speed-1 packed run "
                                "(wave Q: short, so it doubles as the check)")),
        lambda r: t_pkg_prefix(r, "pkg-p20-tail-q", prefix_plan([("contra-soft-muon", 5), ("soda-fade", 4), ("radial-damp", 4),
                                                                 ("ema-nesterov-aurora", 2), ("circuit-muon", 1), ("muonh-tuned", 1),
                                                                 ("newton-muon", 1), ("ortho", 1)], q=True),
                               "20 independent jobs, one about six times the others",
                               full=[("contra-soft-muon-long", "contra-soft-muon", 5, 2000)]),
        lambda r: t_pkg_segment(r, "pkg-p3-mixed-q", [("contra-soft-muon", 6, (0, 250, 500))],
                                prefix_plan([("radial-damp", 1)], q=True), "a chain of 2 and 1 independent short job",
                                segment_measure(own="its own speed-1 packed run (wave Q)")),
        lambda r: t_pkg_segment(r, "pkg-p6-chain-q", [("ema-nesterov-aurora", 15, (0, 125, 250, 375, 500, 625, 750))], [],
                                "6 short jobs in one chain", segment_measure(own="its own speed-1 packed run (wave Q)")),
        lambda r: t_pkg_prefix(r, "pkg-p10-par-q", prefix_plan([("contra-soft-muon", 2), ("soda-fade", 2), ("radial-damp", 2),
                                                                ("ema-nesterov-aurora", 2), ("muonh-tuned", 1), ("newton-muon", 1)],
                                                               q=True), "10 independent short jobs"),
        lambda r: t_pkg_prep(r, "pkg-prep-p10-q", [("contra-soft-muon", 0), ("contra-soft-muon", 1), ("soda-fade", 0),
                                                   ("soda-fade", 1), ("radial-damp", 0), ("radial-damp", 2),
                                                   ("ema-nesterov-aurora", 8), ("ema-nesterov-aurora", 9), ("muonh-tuned", None),
                                                   ("newton-muon", None)], "10 independent short jobs", per_seed=True, q=True),
        *([lambda r: t_pkg_dataprep(r, "pkg-dataprep-p6-q", prefix_plan(
            [("contra-soft-muon", 1), ("soda-fade", 1), ("radial-damp", 1), ("ema-nesterov-aurora", 1), ("muonh-tuned", 1),
             ("newton-muon", 1)], q=True), "6 independent short jobs on data the agent prepares", "pkg-p6-par-q")]
          if DATAPREP_OUTPUTS else []),
        lambda r: t_pkg_screen(r, "pkg-sc-q", [("contra-soft-muon", 7, (0, 250, 500)), ("soda-fade", 2, (0, 250, 500)),
                                               ("radial-damp", 3, (0, 250, 500)), ("circuit-muon", 6, (0, 250, 500))], 2),
        # Stagger-q (gpu-tasks' proposal, the study's priority 3, 2026-10-21): independent runs of 6-28 minutes, so GPUs free
        # at uneven times and a blind fixed wait shows as ready-but-idle GPU time. One directory per (setup, length).
        lambda r: t_pkg_prefix(r, "pkg-stagger-q", [
            (v, 1, s_, (), k, f"{v}-{k}") for v, s_, k in [
                ("contra-soft-muon", 0, 250), ("contra-soft-muon", 1, 500), ("contra-soft-muon", 2, 750),
                ("contra-soft-muon", 3, 1000), ("soda-fade", 0, 250), ("soda-fade", 1, 1000), ("radial-damp", 0, 250),
                ("radial-damp", 2, 875), ("ema-nesterov-aurora", 8, 125), ("ema-nesterov-aurora", 9, 375),
                ("circuit-muon", 6, 125), ("newton-muon", None, 250), ("muonh-tuned", None, 250), ("ortho", None, 375)]],
            "14 independent runs of 6 to 28 minutes"),
        lambda r: t_pkg_halving(r, "pkg-halving-q", [("contra-soft-muon", 8, (0, 250, 500, 750)),
                                                     ("soda-fade", 3, (0, 250, 500, 750)),
                                                     ("radial-damp", 4, (0, 250, 500, 750)),
                                                     ("circuit-muon", 6, (0, 250, 500, 750))]),
        # Confirm-phase twins of stagger-q and halving-q: the same mechanisms with long jobs.
        lambda r: t_pkg_prefix(r, "pkg-stagger", [
            (v, 1, s_, (), k, f"{v}-{k}") for v, s_, k in [
                ("contra-soft-muon", 0, 875), ("contra-soft-muon", 1, 1500), ("contra-soft-muon", 2, 2000),
                ("contra-soft-muon", 3, 2750), ("soda-fade", 0, 1000), ("soda-fade", 1, 2000), ("soda-fade", 2, 3000),
                ("radial-damp", 0, 875), ("radial-damp", 2, 1750), ("radial-damp", 3, 2500),
                ("ema-nesterov-aurora", 8, 375), ("ema-nesterov-aurora", 9, 1000), ("muonh-tuned", None, 1125),
                ("ortho", None, 1375)]],
            "14 independent runs of 21 to 74 minutes"),
        # Re-seeded to a clean round 1 and a round-2 leader change (the study, 2026-10-21; gen_segment.CHAINS_EXTRA says why
        # the first round is 625 steps). The near-tie seeding is kept there as a possible judgement variant.
        lambda r: t_pkg_halving(r, "pkg-halving", [("contra-soft-muon", 28, (0, 625, 1500, 2250)),
                                                   ("soda-fade", 1, (0, 625, 1500, 2250)),
                                                   ("radial-damp", 9, (0, 625, 1500, 2250)),
                                                   ("circuit-muon", 6, (0, 625, 1500, 2250))]),
        lambda r: t_goal_pass_bar(r, "goal-pass-bar", ["contra-soft-muon", "newton-muon", "radial-damp", "soda-fade"],
                                  "Goal-level: which candidates pass the track's acceptance rule, from runs the agent chooses"),
        lambda r: t_goal_choose(r, "goal-choose-optimizer-q", ["contra-soft-muon", "soda-fade", "radial-damp", "muonh-tuned",
                                                                "ema-nesterov-aurora"], 1250,
                                "Goal-level: which optimizer should we adopt, from runs the agent chooses (step 1,250)"),
        t_mng3_shootout, t_mls_kv, t_mng3_screen_confirm, t_mls_kv_screen_confirm,
        # the second realistic package (the analysis, 2026-10-22; build/gen_mlskv_task.py)
        lambda r: __import__("gen_mlskv_task").build(r, disclosure_of, REG_DIR),
        # its report variant, the deployment review's full-workload table
        lambda r: __import__("gen_mlskv_task").build(r, disclosure_of, REG_DIR, variant="mls-kv-report"),
        # H5, its hidden-plan pair: the same jobs, the plan in the files
        lambda r: __import__("gen_mlskv_task").build(r, disclosure_of, REG_DIR, variant="mls-kv-report-h"),
        # G4, the goal-only acceptance-rule package (gpu-tasks-kvgate, 2026-10-23; build/gen_mlskv_gate.py)
        lambda r: __import__("gen_mlskv_gate").build(r, disclosure_of, REG_DIR),
        # H7, G4's hidden-plan pair: the rule in methods/ACCEPTANCE.md
        lambda r: __import__("gen_mlskv_gate").build(r, disclosure_of, REG_DIR, hidden=True),
        # G4n, the goal-only adoption-rule twins on nanogpt records (gpu-tasks-kvgate, 2026-10-24; build/gen_g4n.py)
        lambda r: __import__("gen_g4n").build(r, disclosure_of, REG_DIR, twin="a"),
        lambda r: __import__("gen_g4n").build(r, disclosure_of, REG_DIR, twin="b"),
        # H6, twin a's hidden-plan pair: the rule in ADOPTION.md
        lambda r: __import__("gen_g4n").build(r, disclosure_of, REG_DIR, twin="a", hidden=True),
        # twin A re-briefed: the means the decision rests on, not each setup's
        lambda r: __import__("gen_g4n").build(r, disclosure_of, REG_DIR, twin="a", means=True),
        # package-search angle F, 8 GPUs: the twins and G1 on an 8-GPU box
        lambda r: __import__("gen_g4n").build(r, disclosure_of, REG_DIR, twin="a", gpus=8),
        lambda r: __import__("gen_g4n").build(r, disclosure_of, REG_DIR, twin="b", gpus=8),
        lambda r: __import__("gen_mlskv_task").build(r, disclosure_of, REG_DIR, variant="mls-kv-report", gpus=8),
        # package-search angle C, failures and relaunch (gpu-tasks-failures, 2026-10-31; build/gen_failures.py)
        lambda r: __import__("gen_failures").build(r, disclosure_of, REG_DIR),
        # its file-order twin, seed-major (the analysis, 2026-11-01; notes/2026-11-01-late-failure-shapes.md, shape 1)
        lambda r: __import__("gen_failures").build(r, disclosure_of, REG_DIR, order="seed"),
        # independent Astra (2026-11-02): distributed-width reproducibility matrix, 8 GPUs (build/gen_astra_widths.py is Astra's)
        lambda r: __import__("gen_astra_widths").build(r, disclosure_of, REG_DIR),
        # independent Sol (2026-11-02): yield-gated filter adoption on H4's recipes (build/gen_filter_yield.py is Sol's)
        lambda r: __import__("gen_filter_yield").build(r, disclosure_of, REG_DIR),
        # SF lane (2026-11-02): a zero-learning-rate configuration with a real flag repair (build/gen_sf_zero_lr.py is the lane's)
        lambda r: dict(__import__("gen_sf_zero_lr").build(r, disclosure_of, REG_DIR), _post_finish="gen_sf_zero_lr"),
        # the study K2 (2026-11-02): precision-limited replication, a sequential stopping rule over 49 prefixes (build/gen_k2_precision.py is K2's)
        lambda r: __import__("gen_k2_precision").build(r, disclosure_of, REG_DIR),
        # SF lane (2026-11-02): checkpoint-path collision in a 40-model ensemble (build/gen_sf_checkpoint_collision.py is the lane's)
        lambda r: __import__("gen_sf_checkpoint_collision").build(r, disclosure_of, REG_DIR),
        # independent Sol (2026-11-02): TabArena adoption twins, science and operations (build/gen_sol_tabular.py is Sol's; replay pending r8)
        lambda r: __import__("gen_sol_tabular").build(r, disclosure_of, REG_DIR, twin="science"),
        lambda r: __import__("gen_sol_tabular").build(r, disclosure_of, REG_DIR, twin="operations"),
        # and their full-table controls: the same cells, every one owed, no adoption rule (independent Sol, 2026-11-02 13:5x MST)
        lambda r: __import__("gen_sol_tabular").build(r, disclosure_of, REG_DIR, twin="science", adoption=False),
        lambda r: __import__("gen_sol_tabular").build(r, disclosure_of, REG_DIR, twin="operations", adoption=False),
        # angle K (2026-11-02): unneeded work, 9 candidates x 5 runs rejected by a rule from their own early log lines
        # (build/gen_unneeded_k.py is gpu-tasks-k's)
        lambda r: __import__("gen_unneeded_k").build(r, disclosure_of, REG_DIR),
        # package-search angle A, rounds with stragglers (gpu-tasks-rounds, 2026-10-31; build/gen_rounds.py), at 4 and 8 GPUs
        lambda r: __import__("gen_rounds").build(r, disclosure_of, REG_DIR),
        lambda r: __import__("gen_rounds").build(r, disclosure_of, REG_DIR, gpus=8),
        # package-search angle D, a long chain plus side work (gpu-tasks-chain, 2026-10-31; build/gen_chain_side.py)
        lambda r: __import__("gen_chain_side").build(r, disclosure_of, REG_DIR),
        # package-search angles D x C at 8 GPUs: pkg-chain-side's chain with a real checkpoint-save failure (gpu-tasks-dsave,
        # 2026-11-02; build/gen_chain_save.py)
        lambda r: __import__("gen_chain_save").build(r, disclosure_of, REG_DIR),
        # the study K2 (2026-11-02): the late pole, 38 evaluations over MLS-Bench's KV, DLM and sparse-attention tasks
        # on 8 GPUs, two Python environments (build/gen_k2_latepole.py is K2's)
        lambda r: __import__("gen_k2_latepole").build(r, disclosure_of, REG_DIR),
        # lane OBS: the late-pole twin on nanogpt H100 SXM records, 36 jobs
        # on 8 GPUs (build/gen_obs_latepole.py is lane OBS's)
        lambda r: __import__("gen_obs_latepole").build(r, disclosure_of, REG_DIR),
        # H1, the package search's hidden-plan optimizer study (the analysis, 2026-10-23; build/gen_h1.py)
        lambda r: __import__("gen_h1").build(r, disclosure_of, REG_DIR),
        # H3', hidden resume chains of uneven length (gpu-tasks-h3, 2026-10-23; build/gen_h3_chains.py)
        lambda r: __import__("gen_h3_chains").build(r, disclosure_of, REG_DIR),
        # H4, the filter-recipe ablation whose data preparation is real CPU work in the box (cpu-tasks, 2026-10-23; build/gen_h4.py)
        lambda r: __import__("gen_h4").build(r, disclosure_of, REG_DIR),
                lambda r: t_mls_pretrain(r, "llm-pretrain-optimizer", "Three optimizers for GPT-2 medium pretraining, each followed by a zero-shot evaluation", "optimizer"),
                lambda r: t_mls_pretrain(r, "llm-pretrain-lr-schedule", "Three learning-rate schedules for GPT-2 medium pretraining, each followed by a zero-shot evaluation", "learning-rate schedule")]
    import siblings as SIB
    sib_pools = SIB.sibling_pools(recs)
    tasks = []
    for b in builders:
        t = complete_task(b(recs), recs, reg, sib_pools)
        C.write_json(os.path.join(reg, "tasks", t["task_id"] + ".json"), t)
        tasks.append(t)
        st = t["stats"]
        print(f"{t['task_id']}: rung {t['rung']}, {t['task_class']}, {st['jobs']} jobs, {st['recorded_gpu_seconds'] / 3600:.1f} GPU-h, "
              f"lower bound {st['lower_bound_makespan_s'] / 3600:.2f} h, serial {st['serial_makespan_s'] / 3600:.2f} h, "
              f"replay-ready {t['readiness']['replay_ready_jobs']}/{t['readiness']['jobs_with_records']}, blockers {t['readiness']['blockers']}")
    catalog(reg, recs, tasks)


if __name__ == "__main__":
    main()
