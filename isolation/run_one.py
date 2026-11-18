#!/usr/bin/env python3
"""One ComputeWise run: one task, one harness, one model and one seed, in a capped box; one run directory out.

Run on the Docker host that holds the boxes (the GCP VM), as root, because it loop-mounts a disk image
per run:

  sudo python3 -I run_one.py --task trl-dpo-loss-variants-4gpu --harness scripted --plan plan.json \\
       --out /srv/runs/<tag> --cost-tag <tag> [--speed 1] [--disk 40g]   # caps default to the whole machine
  sudo python3 -I run_one.py --task ... --harness claude-code --model claude-haiku-5-5 --out ... --cost-tag ...
  sudo python3 -I run_one.py --stop <run-dir>        end a live run from outside

What it does, in order:
  0. Chooses the box image: --image, or else the one image labelled p006.role=box-image and
     p006.task-env=<the task's image label: environment.image_label.value, else the task id>. An image
     labelled for another environment is refused, naming both, because it holds another Python stack (C4 on
     2026-10-20 ran the battery in the DPO image and was void).
  1. Stages the task's working tree with gpu-tasks' build/stage_task.py, and its Hugging Face cache with
     build/stage_hub.py run under the image's Python. Both are cached under --cache-root, so a later run
     reuses them: the tree keyed by the task file's content, the cache by the task's hub entries and the
     image, since the image's datasets version writes the processed-datasets cache.
  2. Makes the run's disk: a sparse ext4 image of --disk bytes, loop-mounted on the host, holding the
     agent's home (with the task tree at ~/project) and /tmp. Writes past its size fail with ENOSPC.
  3. Starts gpu-mock's daemon and its NVML bridge in a sidecar, from the gpu-daemon image built with the
     box image (no network, the registry read-only, the socket on a fresh named volume, occupancy written
     straight into the run directory). The daemon's code never enters the box.
  4. For claude-code only, starts the forwarder's edge, the box's only route to a model.
  5. Starts the box on an --internal network with the caps, the harness running as uid 1000, and the
     task's environment.env applied over the image's ENV (minus PYTHONPATH and HF_HOME, which run_one
     sets: PYTHONPATH is the image's own, if any, then the project). Samples its cgroup from the host every second. The box's root filesystem is read-only; the
     agent writes under its home and /tmp, which live on the run's disk.
  6. After the harness exits, keeps the box up until the task's horizon, until no process other than
     the box's init is left, or until no task job is connected to gpu-mock's daemon and the box has been
     idle for AFTER_HARNESS_GRACE_S (idle_after_harness), whichever comes first. Then copies the
     transcript out and tears everything down.

The run directory, and what each file means, is specified in notes/single-run-contract-2026-10-19.md.
Exit status: 0 when the run reached an end (status.json says which), 2 for bad arguments, 1 when this
command itself failed.
"""
import argparse
import datetime
import fcntl
import socket
import urllib.request
import hashlib
import math
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REG = os.path.join(EXP, "gpu-tasks", "registry")
STAGE_TASK = os.path.join(EXP, "gpu-tasks", "build", "stage_task.py")
GT_BUILD = os.path.join(EXP, "gpu-tasks", "build")
# The cache filler per image label, where gpu-tasks' stage_hub.py does not fit. It load_dataset()s every dataset repo
# with no config, and datasets 5.x refuses THUDM/LongBench's loader script; both MLS KV measures failed in it on
# 2026-10-22 21:34Z. isolation/image/mlskv/stage_mlskv_cache.py loads by each hub entry's own split, config and files.
HF_STAGERS = {"mlsbench-kv": (os.path.join(HERE, "image", "mlskv"), "stage_mlskv_cache.py")}
# A task with environment.python_environments (the late pole: two interpreters in one image, isolation's image plan
# 2026-11-02) fills one cache per environment with that environment's interpreter (stage_env_cache.py), and mounts an
# environment's cache where the task names its hf_home; env a's stays at ~/.cache/huggingface. Every other task stages
# exactly as before.
ENV_STAGER = (os.path.join(HERE, "image", "latepole"), "stage_env_cache.py")
FORWARDER = os.path.join(HERE, "forwarder", "forwarder.py")
# The runner's rule for when a stream-json-input session should end (closing the harness's stdin), loaded from its
# file so run.json can record exactly which version ran.
LIFECYCLE_PY = os.path.join(EXP, "runner", "harness", "lifecycle.py")
_spec = importlib.util.spec_from_file_location("p006_lifecycle", LIFECYCLE_PY)
LIFECYCLE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LIFECYCLE)
# the Codex driver's Codex session: the app-server conversation and the rule for ending it, loaded from its file so
# run.json can record exactly which version ran (experiments/codex/harness/codex_session.py).
CODEX_PY = os.path.join(EXP, "codex", "harness", "codex_session.py")
_cspec = importlib.util.spec_from_file_location("p006_codex_session", CODEX_PY)
CODEX = importlib.util.module_from_spec(_cspec)
_cspec.loader.exec_module(CODEX)
SAMPLER = os.path.join(HERE, "tools", "cgroup_sampler.py")
# Checkpoint content digests for a task with capture.weights_index (the SF lane's checkpoint-collision package,
# 2026-11-02, fix:capture-weights-digest), loaded from its file so run.json can record exactly which version ran.
WEIGHTS_DIGEST_PY = os.path.join(HERE, "tools", "weights_digest.py")
_wspec = importlib.util.spec_from_file_location("p006_weights_digest", WEIGHTS_DIGEST_PY)
WEIGHTS_DIGEST = importlib.util.module_from_spec(_wspec)
_wspec.loader.exec_module(WEIGHTS_DIGEST)
AGENT_PY = os.path.join(EXP, "runner", "scripted", "agent.py")
# The scheduler's library for a --scheduler on run, mounted read-only over the image's /usr/lib/slurm (the study's
# route (a), 2026-11-04 07:47Z, for the scheduler's controller fix: ctld.py 9e2a3446, a self-pipe wake in place
# of the SIGCHLD handler's threading.Event, which deadlocked on a burst of fast-failing jobs). Every scheduler image then
# runs the deployed tree's library with no rebuild. The four files are copied to a neutral directory beside the box's,
# since a bind mount's source path is in the box's mountinfo; run.json records the hashes of what was mounted.
SCHED_LIB = os.path.join(EXP, "scheduler", "slurm")
SCHED_LIB_FILES = ("common.py", "ctld.py", "cli.py", "slurmstepd.py")

UID = 1000
HOME = "/home/researcher"
# Where a --skill directory is mounted in the box, per harness: Claude Code reads ~/.claude/skills/<name>, and Codex
# 0.163.0 reads <CODEX_HOME>/skills/<name>, CODEX_HOME being ~/.codex here (the analysis 2026-11-11 09:20 MST,
# for the harness's gpu-harness-a5-codex; the roots read from the box's own codex binary, 09:18 MST)
SKILL_ROOT = {"claude-code": ".claude/skills", "codex": ".codex/skills"}
PROJECT = HOME + "/project"
CWD_SLUG = re.sub(r"[^A-Za-z0-9]", "-", PROJECT)
SOCK_DIR = "/var/run/nvidia-persistenced"
EDGE_ALIAS = "api-gw.internal"
EDGE_PORT = 8080

STOP = {"why": None}


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


BASH_TIMEOUT_DEFAULT_MS = 36000000   # 10 h: no Bash call in the box is cut short unless a run asks for it


def BASH_TIMEOUTS_MS(a):
    """(default, max) for the box's BASH_DEFAULT_TIMEOUT_MS / BASH_MAX_TIMEOUT_MS: --bash-timeout-ms, or 10 h both."""
    return a.bash_timeout_ms or (BASH_TIMEOUT_DEFAULT_MS, BASH_TIMEOUT_DEFAULT_MS)


# Grok Build and OpenCode (the study, 2026-10-26; the study' port notes, experiments/harnesses/port/). Both
# take the brief on argv with stdin at /dev/null and end the run by exiting on their own, so neither has a session
# module or a FIFO: the box keeps running the agent's jobs after the exit, as for any harness. Their configs name no
# run tag and no router: the edge tags every request, and the provider is named neutrally, as for Codex.
MODEL_HARNESSES = ("claude-code", "codex", "grok", "opencode")
GROK_TITLE_MODEL = "grok-4.6"   # Grok Build's session-title side calls, 2 per prompt (a product default, kept on)
GROK_CONTEXT_WINDOW = 500000
# --background-wait-timeout, one value for every run: the agent can read its harness's argv, so a value derived from
# the run would show it the horizon, which it is never shown. 100 h is past every task's horizon
# (the longest, 55 h) and Grok's own 10-h cap on a background command, so only run_one ends a run.
GROK_BACKGROUND_WAIT_S = 360000
GROK_BASH_TIMEOUT_S = 36000     # the command-timeout maximum, as for Claude Code (no effect on 1.0.70's shell tool)
OPENCODE_LIMITS = {"gemma-4-31b-bedrock": {"context": 262144, "output": 32768}}
OPENCODE_BASH_TIMEOUT_MS = 36000000   # its only command-timeout setting, an env var with no upper limit to raise


def edge_models(a):
    """The models the run's edge accepts: the run's model, and for Grok Build its title model as well, whose calls
    are the harness's own and would otherwise be refused. Without a "[1m]" suffix: Claude Code reads it from --model to
    set a 1M window and sends the model without it (Astra in Claude Code, `gpt-6-astra:medium[1m]` arrives as
    `gpt-6-astra:medium`, 2026-10-27), and the edge compares the request's model with any "[1m]" removed."""
    m = a.model.removesuffix("[1m]")
    return [m, GROK_TITLE_MODEL] if a.harness == "grok" else [m]


def model_parts(model):
    """{base, effort, context_1m} of a --model string such as `gpt-6-astra:medium[1m]` (the router's effort suffix, Claude
    Code's 1M suffix), recorded so the scorer can label a cell (analysis 2026-10-27: Astra in Claude Code is `astra-cc`)."""
    m = model.removesuffix("[1m]")
    base, _, effort = m.partition(":")
    return {"base": base, "effort": effort or None, "context_1m": model.endswith("[1m]")}


def grok_config_toml(model, base_url):
    """The agent's ~/.grok/config.toml. reasoning_summary auto: some providers' Anthropic-compatible paths refuse Grok's default
    'concise' with HTTP 400; it changes only the summary format. The key's environment variable is named neutrally."""
    key = model.replace(".", "-")
    return ('[cli]\ninstaller = "internal"\n\n'
            f'[models]\ndefault = "{key}"\n\n'
            f'[model.{key}]\nmodel = "{model}"\nbase_url = "{base_url}"\napi_backend = "responses"\n'
            f'env_key = "LLM_API_KEY"\ncontext_window = {GROK_CONTEXT_WINDOW}\nreasoning_summary = "auto"\n\n'
            f'[toolset.bash]\ntimeout_secs = {GROK_BASH_TIMEOUT_S}\nmax_timeout_secs = {GROK_BASH_TIMEOUT_S}\n')


def opencode_config_json(model, base_url, api_key):
    """The agent's ~/.config/opencode/opencode.json: one neutrally named provider at the edge, the Responses API
    (@ai-sdk/openai, bundled in the binary), the model's published limits."""
    m = {"name": model, "tool_call": True}
    if model in OPENCODE_LIMITS:
        m["limit"] = OPENCODE_LIMITS[model]
    return json.dumps({"$schema": "https://opencode.ai/config.json", "model": f"llm/{model}",
                       "provider": {"llm": {"npm": "@ai-sdk/openai", "name": "LLM API",
                                            "options": {"baseURL": base_url, "apiKey": api_key},
                                            "models": {model: m}}}}, indent=2) + "\n"


def tree_sha256(root):
    """One hash over every file under root: sha256 of the sorted lines '<sha256>  <relative path>'."""
    lines = []
    for r, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(r, f)
            lines.append(f"{sha256_file(p)}  {os.path.relpath(p, root)}")
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


def tree_sha(root):
    """One hash over every .py file under root, in path order."""
    return hashlib.sha256("".join(sha256_file(os.path.join(r, f)) for r, _, fs in sorted(os.walk(root))
                                  for f in sorted(fs) if f.endswith(".py")).encode()).hexdigest()


def sh(cmd, check=True, capture=True, **kw):
    r = subprocess.run(cmd, check=False, text=True,
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.PIPE if capture else None, **kw)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}\n{(r.stderr or '')[-2000:]}")
    return r


def docker(*args, check=True):
    return sh(["docker", *args], check=check)


_ONEOFF = [0]


def oneoff_run(*args, cpus="1", memory="1g"):
    """argv for a short-lived helper container, capped and named like every container on this host
    (gcp/AGENTS.md: health.sh fails on any running container without a CPU and a memory cap)."""
    _ONEOFF[0] += 1
    return ["docker", "run", "--rm", "--name", f"p6-oneoff-{os.getpid()}-{_ONEOFF[0]}",
            "--cpus", cpus, "--memory", memory, "--memory-swap", memory, *args]


def log(msg):
    print(f"[run_one {datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def read_float(p):
    try:
        return float(open(p).read().strip())
    except (OSError, ValueError):
        return None


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stop", metavar="RUN_DIR", help="end the live run writing RUN_DIR, and exit")
    ap.add_argument("--task")
    ap.add_argument("--harness", choices=("claude-code", "scripted", "codex", "grok", "opencode"), default="claude-code")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--cost-tag", help="the ledger tag; the runner mints it")
    ap.add_argument("--stop-dir", default="/srv/stage/gate/stop",
                    help="where the budget enforcer creates <tag>; read only, to choose the status")
    ap.add_argument("--budget-usd", type=float, help="recorded only; the runner enforces it")
    ap.add_argument("--plan", help="scripted harness: the plan written by runner/scripted/plan.py")
    ap.add_argument("--claude-arg", action="append", default=[], metavar="TOKEN",
                    help="claude-code harness: one more argv token for claude, appended in order after run_one's own "
                         "flags and recorded as claude_code.extra_flags (e.g. --claude-arg=--disallowedTools "
                         "--claude-arg=Task for the no-subagent arm). Flags run_one sets itself are refused.")
    ap.add_argument("--codex-arg", action="append", default=[], metavar="TOKEN",
                    help="codex harness: one more argv token for `codex app-server`, appended in order and recorded "
                         "as codex.extra_flags (the backlog's harness_flags, e.g. --codex-arg=--disable "
                         "--codex-arg=multi_agent).")
    ap.add_argument("--no-codex-resume", action="store_true",
                    help="codex harness: turn off capacity resume. By default, when the provider ends the root turn on "
                         "capacity (codexErrorInfo serverOverloaded, \"Selected model is at capacity\"), the session sends "
                         "\"continue\" on the same thread after a backoff, as a user at the terminal would (codex_session.py's "
                         "resume rule, at most RESUME_MAX times), for every model (analysis, 2026-10-28 16:36 MDT: an outage, "
                         "not the model's choice). Recorded in run.json codex.resume and each resume in status.json "
                         "codex.resumed_after_capacity. With this flag the session ends at such a turn, as before (known answers)")
    ap.add_argument("--brief-file", help="use this file's contents as the brief instead of the task's brief.text "
                                          "(the task tree, box and route stay the same); recorded in run.json")
    ap.add_argument("--agent-py", default=AGENT_PY)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--max-hours", type=float, default=2.0)
    ap.add_argument("--deadline-s", "--horizon-s", dest="deadline_s", type=float,
                    help="override the task's horizon (recorded seconds); the hard stop, never shown to the agent")
    ap.add_argument("--after-deadline", choices=("stop", "continue"), default="stop",
                    help="a harness still running at the horizon: end the run there (default), or let it go on "
                         "until --max-hours")
    ap.add_argument("--nudge", help="campaign nudge level (L0..L3): append that level's sentence from --campaign to "
                                     "the brief as its final paragraph, verbatim, and record it in run.json (gap G6)")
    ap.add_argument("--campaign", help="campaign.json holding nudges.levels; required with --nudge")
    ap.add_argument("--cpus", default="machine",
                    help="CPU quota for the box. Default machine: every core but the sidecar's (the machine is the "
                         "isolation boundary; the study, 2026-10-20). A number is honoured, for tests")
    ap.add_argument("--cpuset", default="auto",
                    help="cores the box is pinned to: auto (cores 2 up to the quota), an explicit list such as 2-17, "
                         "or none")
    ap.add_argument("--memory", default="machine",
                    help="memory cap for the box. Default machine: the machine's RAM minus an 8 GB host reserve, so a "
                         "runaway job still dies inside the box's cgroup. A size such as 64g is honoured, for tests")
    ap.add_argument("--disk", default="40g")
    ap.add_argument("--shm", default="8g")
    ap.add_argument("--pids", type=int, default=8192)
    ap.add_argument("--oom-protect", choices=("on", "off"), default="on")
    ap.add_argument("--bash-timeout-ms", metavar="DEFAULT,MAX", default=None,
                    help="claude-code harness: BASH_DEFAULT_TIMEOUT_MS and BASH_MAX_TIMEOUT_MS for the box, e.g. 120000,600000 "
                         "(Claude Code's product defaults). Without it both are 36000000, so no tool call is cut short. "
                         "Recorded in run.json claude_code.bash_timeout_ms (analysis' labelled factor, 2026-10-22)")
    ap.add_argument("--agent-teams", action="store_true",
                    help="claude-code harness: set CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 in the box, which gives the Agent tool its "
                         "`name` parameter (teammates addressable by SendMessage); without it Claude Code 2.1.303 offers no "
                         "name. A per-line opt-in for the gpu-team skill (the harness, 2026-10-27), recorded in run.json "
                         "env and claude_code.agent_teams; every other Claude cell keeps today's tool schema")
    ap.add_argument("--auto-compact-window", type=int, default=None, metavar="TOKENS",
                    help="claude-code harness: CLAUDE_CODE_AUTO_COMPACT_WINDOW for the box (Claude Code 2.1.303 compacts the "
                         "conversation at this many tokens); unset leaves the product's own point. Recorded in run.json "
                         "claude_code.auto_compact_window (Astra in Claude Code at Codex's compaction point, analysis 2026-10-27)")
    ap.add_argument("--effort-level", choices=("low", "medium", "high", "xhigh"), default=None,
                    help="claude-code harness: CLAUDE_CODE_EFFORT_LEVEL for the box. Claude Code 2.1.303 sends its own effort "
                         "(output_config.effort, high by default) whatever the model; this sets that field. Unset leaves the "
                         "product's default. Recorded in run.json claude_code.effort_level (Astra in Claude Code at medium, so the "
                         "field and the model's :medium suffix agree, analysis 2026-10-27)")
    ap.add_argument("--skill", action="append", default=[], metavar="DIR",
                    help="claude-code or codex harness: a skill directory (with SKILL.md) mounted read-only at "
                         "~/.claude/skills/<basename> (claude-code) or ~/.codex/skills/<basename> (codex); repeatable; "
                         "recorded in run.json skills (solution's S rungs)")
    ap.add_argument("--scheduler", choices=("on", "off"), default="off",
                    help="on: box-init starts the node's job scheduler (slurmctld) before the harness; needs an image "
                         "labelled p006.scheduler (the scheduler's experiments/scheduler/). Recorded in run.json")
    ap.add_argument("--dressing", choices=("off", "on"), default="off", help="recorded only; not built yet")
    ap.add_argument("--disclosure", choices=("off", "on"), default="off", help="recorded only; not built yet")
    ap.add_argument("--image", help="the box image. Default: the one image labelled p006.role=box-image and "
                                     "p006.task-env=<the task's image label: environment.image_label.value, else the task id>. "
                                     "An image labelled for another environment is refused")
    ap.add_argument("--sidecar-image", help="default: the image the box image names in its p006.sidecar-image label")
    ap.add_argument("--gpu-mock-src", help="development only: run the daemon from this host tree (gpu-mock's src/) "
                                           "instead of the sidecar image's copy; recorded in run.json")
    ap.add_argument("--cache-root", default="/srv/cache")
    ap.add_argument("--disk-root", default="/srv/boxdisk")
    ap.add_argument("--edge-upstream", help="default: http://<docker bridge gateway>:8790, the tunnel's end")
    ap.add_argument("--link-key", default="/srv/stage/gate/link.key")
    ap.add_argument("--sample-interval", type=float, default=1.0)
    a = ap.parse_args(argv)
    if a.stop:
        return a
    missing = [n for n in ("task", "out", "cost_tag") if not getattr(a, n)]
    if missing:
        ap.error("required: " + ", ".join("--" + m.replace("_", "-") for m in missing))
    if a.harness == "scripted" and not a.plan:
        ap.error("--harness scripted needs --plan")
    if a.harness != "claude-code" and a.claude_arg:
        ap.error("--claude-arg is for the claude-code harness only")
    if a.harness != "codex" and a.codex_arg:
        ap.error("--codex-arg is for the codex harness only")
    if a.bash_timeout_ms is not None:
        if a.harness != "claude-code":
            ap.error("--bash-timeout-ms is for the claude-code harness only")
        try:
            d, m = (int(x) for x in a.bash_timeout_ms.split(","))
        except ValueError:
            ap.error("--bash-timeout-ms takes DEFAULT,MAX in milliseconds, e.g. 120000,600000")
        if not 0 < d <= m:
            ap.error("--bash-timeout-ms: need 0 < DEFAULT <= MAX")
        a.bash_timeout_ms = (d, m)
    if a.skill and a.harness not in SKILL_ROOT:
        ap.error("--skill is for the claude-code and codex harnesses only")
    if a.agent_teams and a.harness != "claude-code":
        ap.error("--agent-teams is for the claude-code harness only")
    if a.no_codex_resume and a.harness != "codex":
        ap.error("--no-codex-resume is for the codex harness only")
    a.codex_resume = a.harness == "codex" and not a.no_codex_resume
    if a.codex_resume and not hasattr(CODEX, "RESUME_MAX"):
        # refused rather than run without it: the rule is one for every Codex run, and this run_one ships with db728687
        ap.error(f"{CODEX_PY} has no capacity-resume rule (codex_session.py db728687 or later has it); "
                 "install that module, or pass --no-codex-resume for the old behaviour")
    if a.effort_level and a.harness != "claude-code":
        ap.error("--effort-level is for the claude-code harness only")
    if a.auto_compact_window is not None:
        if a.harness != "claude-code":
            ap.error("--auto-compact-window is for the claude-code harness only")
        if a.auto_compact_window <= 0:
            ap.error("--auto-compact-window takes a positive number of tokens")
    for d in a.skill:
        if not os.path.isfile(os.path.join(d, "SKILL.md")):
            ap.error(f"--skill {d}: no SKILL.md there")
    if len({os.path.basename(os.path.abspath(d).rstrip("/")) for d in a.skill}) != len(a.skill):
        ap.error("--skill: two skills with one directory name would mount at one path")
    reserved = [t for t in a.claude_arg if t.split("=", 1)[0] in RESERVED_CLAUDE_FLAGS]
    if reserved:
        ap.error(f"--claude-arg may not set what run_one sets itself: {', '.join(reserved)}")
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$", a.cost_tag):
        ap.error("--cost-tag must match ^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$ (the forwarder's rule)")
    if a.speed <= 0:
        ap.error("--speed must be positive")
    if a.nudge and not a.campaign:
        ap.error("--nudge needs --campaign <campaign.json>")
    return a


def image_label_of(task, task_id):
    """The p006.task-env label of the image that holds this task's environment: environment.image_label.value
    when the task names one (gpu-tasks, 2026-10-20: a task whose environment is another task's names that task's
    label, so one image serves both, e.g. the quick battery runs in the battery's image), else the task id."""
    il = (task.get("environment") or {}).get("image_label")
    if isinstance(il, dict) and il.get("value"):
        return str(il["value"])
    return task_id


def resolve_image(task_id, image):
    """(image, None) for the box image this task runs in, or (None, why). task_id here is the image label the task
    needs (image_label_of), usually the task id. With no --image: the one image labelled p006.role=box-image and
    p006.task-env=<label>, among tagged images only: rebuild_check.sh builds a tag twice, and the first build's
    image stays behind untagged, still carrying the labels. An explicit --image must carry that label too.
    An image built for another task's stack runs that task's environment, not this one's: C4 on 2026-10-20 ran
    the battery in the DPO image, found trl failing to import, and was void. So a mismatch fails here, before
    anything is staged."""
    if image is None:
        rows = docker("images", "--filter", "dangling=false", "--filter", "label=p006.role=box-image",
                      "--filter", f"label=p006.task-env={task_id}",
                      "--format", "{{.ID}} {{.Repository}}:{{.Tag}}", check=False).stdout.splitlines()
        by_id = {}
        for row in rows:
            if not row.strip():
                continue
            iid, ref = (row.split(" ", 1) + [""])[:2]
            if not ref or "<none>" in ref:
                continue
            by_id.setdefault(iid, ref)
        if not by_id:
            return None, (f"no box image is labelled p006.task-env={task_id}; "
                          f"build one with isolation/image/build.sh --task {task_id} (the task whose environment it holds)")
        if len(by_id) > 1:
            return None, (f"{len(by_id)} box images are labelled p006.task-env={task_id} "
                          f"({', '.join(sorted(by_id.values()))}); pass --image to choose one")
        return next(iter(by_id.values())), None
    r = docker("image", "inspect", "-f", '{{ index .Config.Labels "p006.task-env" }}', image, check=False)
    if r.returncode != 0:
        return None, f"no image {image}; build it with isolation/image/build.sh --task {task_id}"
    label = r.stdout.strip()
    if label in ("", "<no value>"):
        label = None
    if label != task_id:
        return None, (f"image {image} is labelled p006.task-env={label or '(none)'}, but the task needs "
                      f"p006.task-env={task_id}. It holds another task's stack; use the image built for {task_id}, "
                      f"or omit --image")
    return image, None


# Keys whose values run_one sets itself for the box's layout. The task's values for these are placeholders,
# such as "<task root>".
RUN_ONE_ENV_KEYS = ("PYTHONPATH", "HF_HOME")


def task_env(task):
    """The task's environment.env without the keys run_one sets itself: the task's own runtime settings, such
    as the offline flags and the battery's WANDB_MODE=offline. They go into the box's environment, over the
    image's ENV, so run.json records them under env."""
    raw = (task.get("environment") or {}).get("env") or {}
    return {k: str(v) for k, v in raw.items() if k not in RUN_ONE_ENV_KEYS and "<" not in str(v)}


def host_identity():
    """The machine this run is on, from the GCE metadata server (the host may read it; boxes cannot, DOCKER-USER
    drops it). Each field is None when the lookup fails, so a non-GCE host still records a run."""
    out = {}
    for key, field in (("name", "instance"), ("id", "instance_id"), ("machine-type", "machine_type"),
                       ("zone", "zone"), ("cpu-platform", "cpu_platform")):
        try:
            req = urllib.request.Request(f"http://169.254.169.254/computeMetadata/v1/instance/{key}",
                                         headers={"Metadata-Flavor": "Google"})
            v = urllib.request.urlopen(req, timeout=2).read().decode().strip()
            out[field] = v.rsplit("/", 1)[-1] if field in ("machine_type", "zone") else v
        except Exception:
            out[field] = None
    out["hostname"] = socket.gethostname()
    out["mem_total_kb"] = next((int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemTotal:")), None)
    return out


MEM_BASELINE = os.path.join(EXP, "gcp", "host", "mem-baseline.py")


def mem_baseline(seconds=60):
    """gcp's 60 s host-memory baseline (mem-baseline.py: MemAvailable, MemFree, and memory owned by nothing, which
    is where an E2 balloon's pages land), taken before the box starts and recorded as a label. Never blocks."""
    if not os.path.exists(MEM_BASELINE):
        return {"error": "mem-baseline.py not deployed"}
    try:
        r = subprocess.run([sys.executable, "-I", MEM_BASELINE, str(seconds)], capture_output=True, text=True,
                           timeout=seconds + 30)
        return json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else \
            {"error": (r.stderr or "no output")[-300:]}
    except Exception as e:  # a label, never a reason to refuse the run
        return {"error": f"{type(e).__name__}: {e}"[:300]}


# ---- staging ------------------------------------------------------------------------------------

def staged_inputs(task):
    """The registry files stage_task.py copies into the tree besides the task file: every files[] entry taken from
    artifacts/, and the environment's constraints file, each with its sha256 (None if absent). The tree's cache key
    covers them, because gpu-tasks can change an artifact and leave the task file alone, and a cache keyed on the task
    file only went on serving the old bytes (pkg-h1-study's two READMEs, 2026-10-23 20:49 UTC)."""
    paths = {f["from"] for f in task.get("files") or [] if str(f.get("from", "")).startswith("artifacts/")}
    cons = ((task.get("environment") or {}).get("install") or {}).get("constraints")
    if cons:
        paths.add(cons)
    return {p: (sha256_file(os.path.join(REG, p)) if os.path.exists(os.path.join(REG, p)) else None)
            for p in sorted(paths)}


def stage_task_tree(task_id, task_sha, cache_root, inputs=None):
    # A task with no mapped registry files keeps the key it always had, so its cached tree stays valid.
    key = task_sha[:12] + ("-" + hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:8]
                           if inputs else "")
    d = os.path.join(cache_root, "tasks", task_id, key)
    if os.path.exists(os.path.join(d, ".complete")):
        return d
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    log(f"staging the task tree for {task_id}")
    # GPU_TASKS_HUB_CACHE: stage_task.py's `hub_files` entries (the modded-nanogpt packages' FineWeb shards, about
    # 3.5 GB per task) download once into this shared directory and are hard-linked into each task's tree.
    hub_files_cache = os.path.join(cache_root, "hub-files")
    os.makedirs(hub_files_cache, exist_ok=True)
    sh([sys.executable, "-I", STAGE_TASK, task_id, os.path.join(d, "project"), os.path.join(d, "staging")], capture=False,
       env={**os.environ, "GPU_TASKS_HUB_CACHE": hub_files_cache})
    open(os.path.join(d, ".complete"), "w").write(now_utc() + "\n")
    return d


def stage_hf(task, task_id, staged, cache_root, image):
    """The Hugging Face cache for this task's hub entries, filled under this image. Its directory is keyed by the
    image label rather than the task id, so two tasks with one environment (the battery and its quick variant)
    share one staged cache instead of each downloading the same 2 GB."""
    hub = (task.get("environment") or {}).get("hub") or []
    if not hub:
        # Nothing to fetch (the modded-nanogpt packages carry their data as files in the task tree, staged by
        # stage_task.py): an empty cache tree, so the box's ~/.cache/huggingface exists and is empty, and no
        # container starts.
        d = os.path.join(cache_root, "hf", image_label_of(task, task_id), "none")
        os.makedirs(os.path.join(d, "hf"), exist_ok=True)
        if not os.path.exists(os.path.join(d, ".complete")):
            open(os.path.join(d, ".complete"), "w").write(now_utc() + "\n")
        return d
    # The image's own datasets version fills the processed-datasets cache, so the key includes the image, and a
    # cache filled under one image is never reused under another. C4 on 2026-10-20 filled the battery's cache under
    # the DPO image (datasets 2.20.0; the battery pins 2.16.1).
    image_id = docker("image", "inspect", "-f", "{{.Id}}", image).stdout.strip()
    key = hashlib.sha256(json.dumps({"hub": hub, "image_id": image_id}, sort_keys=True).encode()).hexdigest()[:12]
    d = os.path.join(cache_root, "hf", image_label_of(task, task_id), key)
    if os.path.exists(os.path.join(d, ".complete")):
        hub_version_file(d)
        return d
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "hf"))
    label = image_label_of(task, task_id)
    stager_dir, stager = HF_STAGERS.get(label, (GT_BUILD, "stage_hub.py"))
    log(f"filling the Hugging Face cache for {task_id} ({len(hub)} entries) with {stager}")
    # The image's Python, as stage_hub.py requires. Host network, because on 2026-10-19 containers on this
    # host's default bridge could not resolve names. PYTHONPATH is emptied so gpu-mock's patch never loads here.
    sh(oneoff_run("--network", "host", "--user", "0", "--entrypoint", "python3",
        "-e", "HF_HUB_OFFLINE=0", "-e", "HF_DATASETS_OFFLINE=0", "-e", "TRANSFORMERS_OFFLINE=0",
        "-e", "PYTHONPATH=", "-e", "HF_HOME=/hf",
        "-v", f"{os.path.join(d, 'hf')}:/hf", "-v", f"{os.path.join(staged, 'staging')}:/staging:ro",
        "-v", f"{stager_dir}:/b:ro", image, "-I", f"/b/{stager}", "/staging/MANIFEST.json", "/hf", cpus="4", memory="16g"), capture=False)
    open(os.path.join(d, ".complete"), "w").write(now_utc() + "\n")
    hub_version_file(d)
    return d


def stage_env_hf(task, task_id, staged, cache_root, image):
    """For a task with environment.python_environments: {env: cache dir} for each environment that has hub entries
    (an entry with no python_environment is env a's), each filled by stage_env_cache.py under that environment's own
    interpreter and keyed like stage_hf's cache plus the environment and its interpreter, under the task's image label.
    The task tree's own copies of any file under an environment's hf_home are checked against the filled cache, so a
    file the records bind there (the late pole's two LLaDA modules) is the same bytes in both; a difference refuses."""
    e = task.get("environment") or {}
    envs = e.get("python_environments") or {}
    image_id = docker("image", "inspect", "-f", "{{.Id}}", image).stdout.strip()
    label = image_label_of(task, task_id)
    out = {}
    for name in sorted(envs):
        hub = [h for h in e.get("hub") or [] if (h.get("python_environment") or "a") == name]
        if not hub:
            continue
        py = os.path.join(envs[name]["path"], "bin", "python")
        key = hashlib.sha256(json.dumps({"hub": hub, "image_id": image_id, "python": py}, sort_keys=True).encode()).hexdigest()[:12]
        d = os.path.join(cache_root, "hf", label, f"{name}-{key}")
        if not os.path.exists(os.path.join(d, ".complete")):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(os.path.join(d, "hf"))
            json.dump({"hub": hub}, open(os.path.join(d, "hub.json"), "w"), indent=1)
            stager_dir, stager = ENV_STAGER
            log(f"filling environment {name}'s Hugging Face cache for {task_id} ({len(hub)} entries) with {py}")
            # the cache directory itself is mounted, so STAGED.json (written beside hf/, outside what the box mounts) stays
            sh(oneoff_run("--network", "host", "--user", "0", "--entrypoint", py,
                "-e", "HF_HUB_OFFLINE=0", "-e", "HF_DATASETS_OFFLINE=0", "-e", "TRANSFORMERS_OFFLINE=0",
                "-e", "PYTHONPATH=", "-e", "HF_HOME=/c/hf", "-v", f"{d}:/c",
                "-v", f"{stager_dir}:/b:ro", image, "-I", f"/b/{stager}", "/c/hub.json", "/c/hf", cpus="4", memory="16g"),
               capture=False)
            open(os.path.join(d, ".complete"), "w").write(now_utc() + "\n")
        rel = envs[name].get("hf_home")
        if rel:
            sub = rel.split("<workspace>/", 1)[-1].strip("/")
            mine = os.path.join(staged, "project", sub)
            for root, _, fs in os.walk(mine):
                for f in fs:
                    p = os.path.join(root, f)
                    q = os.path.realpath(os.path.join(d, "hf", os.path.relpath(p, mine)))
                    if not os.path.isfile(q) or sha256_file(p) != sha256_file(q):
                        raise RuntimeError(f"the task tree's {os.path.relpath(p, staged)} is not the filled cache's "
                                           f"{os.path.relpath(q, d)}; the records would bind other bytes than the jobs load")
            # every file a record of this task binds under the cache (serving.module_shas, gpu-tasks' sync 15) must be
            # those bytes in the filled cache, or gpu-mock refuses every launch that loads it (identity_module_dep)
            bound = {}
            for j in task.get("jobs") or []:
                for rid in j.get("records") or []:
                    rp = os.path.join(REG, "records", rid + ".json")
                    if os.path.exists(rp):
                        ms = ((json.load(open(rp)).get("materialisation") or {}).get("serving") or {}).get("module_shas") or {}
                        bound.update({k: v for k, v in ms.items() if k.startswith(sub + "/")})
            for k, want in sorted(bound.items()):
                q = os.path.realpath(os.path.join(d, "hf", k[len(sub) + 1:]))
                got = sha256_file(q) if os.path.isfile(q) else None
                if got != want:
                    raise RuntimeError(f"a record binds {k} as {want[:12]}, and the filled cache holds "
                                       f"{(got or 'nothing')[:12]}; gpu-mock would refuse every launch that loads it")
            log(f"environment {name}'s cache holds the {len(bound)} files the records bind under {sub}")
        out[name] = d
    return out


def prepare_tree(task, task_id, staged, image):
    """Run the task's environment.preparation steps once per staged tree, before any box, so they add no host time to
    a run and their outputs are inputs (the cached tree is what every run copies). Only the steps run_one knows by
    name: `inputs` (prepare/source_inputs.py) and `sparse_filler` (prepare_longctx.py, with the endpoint set to the Hub
    and the resolved wikitext revision recorded); `dlm_cache` is the env b cache fill's own three pinned loads, so it
    is not run. Each runs in a one-off container under the named environment's interpreter, as root, with the tree
    mounted at /w; a marker records what ran."""
    e = task.get("environment") or {}
    prep = e.get("preparation") or {}
    marker = os.path.join(staged, "prepared.json")
    if not prep or os.path.exists(marker):
        return
    envs = e.get("python_environments") or {}
    py_b = os.path.join(envs.get("b", envs.get("a", {})).get("path", "/usr/local"), "bin", "python")
    steps = []
    if "inputs" in prep:
        steps.append(("inputs", py_b, ["/w/prepare/source_inputs.py", "--workspace", "/w"], []))
    if "sparse_filler" in prep:
        steps.append(("sparse_filler", py_b, ["/w/source/vendor/data_scripts/sparse-attn-eval/prepare_longctx.py",
                                              "--data-root", "/w/data"], ["-e", "HF_ENDPOINT=https://huggingface.co"]))
    done = {}
    for name, py, argv, extra in steps:
        log(f"preparing {task_id}'s tree: {name}")
        r = sh(oneoff_run("--network", "host", "--user", "0", "--entrypoint", py, "-e", "PYTHONPATH=",
                          "-e", "HF_HUB_OFFLINE=0", "-e", "HF_DATASETS_OFFLINE=0", "-e", "HF_HOME=/tmp/prep-hf", *extra,
                          "-v", f"{os.path.join(staged, 'project')}:/w", "-w", "/w", image, "-I", *argv,
                          cpus="4", memory="16g"))
        done[name] = {"argv": argv, "stdout_tail": (r.stdout or "")[-1500:]}
    # the revision wikitext resolved to, unpinned in the task: datasets keeps it as the 4th level of its cache tree
    # (<namespace>___<name>/<config>/<version>/<revision>)
    lc = os.path.join(staged, "project", "data", "longctx")
    if "sparse_filler" in done and os.path.isdir(lc):
        done["sparse_filler"]["cache_dirs"] = sorted(os.path.relpath(r_, lc) for r_, _, _ in os.walk(lc)
                                                     if os.path.relpath(r_, lc).count(os.sep) == 3)
    write_json(marker, {"utc": now_utc(), "steps": done})


def hub_version_file(d):
    """transformers writes hub/version.txt the first time it sees a cache. The box mounts hub/ read-only,
    so without the file every job prints two warnings: that the cache is from an old version, and that
    the cache folder is not writable. Write it here, once, as transformers would."""
    vt = os.path.join(d, "hf", "hub", "version.txt")
    if os.path.isdir(os.path.dirname(vt)) and not os.path.exists(vt):
        open(vt, "w").write("1")


def chown_tree(p, uid=UID):
    for root, dirs, files in os.walk(p):
        os.lchown(root, uid, uid)
        for n in dirs + files:
            os.lchown(os.path.join(root, n), uid, uid)


def make_disk(name, size, disk_root, image, staged, hf, env_hf=None):
    img = os.path.join(disk_root, name + ".img")
    mnt = os.path.join(disk_root, name)
    os.makedirs(mnt, exist_ok=True)
    sh(["truncate", "-s", size.upper().replace("B", ""), img])
    sh(["mkfs.ext4", "-q", "-F", "-m", "0", "-E", "lazy_itable_init=1,lazy_journal_init=1", img])
    sh(["mount", "-o", "loop", img, mnt])
    home, tmp = os.path.join(mnt, "home"), os.path.join(mnt, "tmp")
    os.makedirs(home)
    os.makedirs(tmp)
    os.chmod(tmp, 0o1777)
    # The image's own home skeleton, so the mounted home looks like the user's.
    p1 = subprocess.Popen(oneoff_run("--network", "none", "--entrypoint", "tar", image, "-C", HOME, "-cf", "-", "."),
                          stdout=subprocess.PIPE)
    sh(["tar", "-x", "-C", home], stdin=p1.stdout)
    p1.wait()
    shutil.copytree(os.path.join(staged, "project"), os.path.join(home, "project"), symlinks=True)
    hfh = os.path.join(home, ".cache", "huggingface")
    os.makedirs(hfh)
    ro = []
    for n in sorted(os.listdir(os.path.join(hf, "hf"))):
        src = os.path.join(hf, "hf", n)
        if n in ("hub", "modules"):
            os.makedirs(os.path.join(hfh, n))
            ro.append((src, f"{HOME}/.cache/huggingface/{n}"))
        elif os.path.isdir(src):
            shutil.copytree(src, os.path.join(hfh, n), symlinks=True)
        else:
            shutil.copy2(src, os.path.join(hfh, n))
    # an environment's own cache (stage_env_hf) where the task names its hf_home under the workspace, in HF's layout
    # (hub/ and modules/ read-only, so snapshots/ -> ../../blobs/ resolves; a writable copy of datasets/). The task tree
    # may already hold files there (the late pole's two LLaDA modules, checked equal to the cache's): the read-only
    # hub/ mount covers them with the same bytes.
    for rel, d in (env_hf or []):
        sub = rel.split("<workspace>/", 1)[-1].strip("/")
        dst = os.path.join(home, "project", sub)
        os.makedirs(dst, exist_ok=True)
        for n in sorted(os.listdir(os.path.join(d, "hf"))):
            src = os.path.join(d, "hf", n)
            if n in ("hub", "modules"):
                os.makedirs(os.path.join(dst, n), exist_ok=True)
                ro.append((src, f"{PROJECT}/{sub}/{n}"))
            elif os.path.isdir(src):
                shutil.copytree(src, os.path.join(dst, n), symlinks=True, dirs_exist_ok=True)
            else:
                shutil.copy2(src, os.path.join(dst, n))
    os.makedirs(os.path.join(home, ".claude", "projects", CWD_SLUG))
    chown_tree(home)
    return img, mnt, ro


# ---- the run ------------------------------------------------------------------------------------

SCHED_ACTIVE = "sched-active"   # /run/box/sched-active: slurmctld's count of jobs not yet ended (S rungs)


def box_processes(cid, boxd=None):
    """Host pids in the box other than docker's init, box-init's closing `sleep infinity`, and a resident slurmctld.
    The controller is the box's own service, not work; while its queue holds a job not yet ended (pending included,
    which no process shows) the controller's pid stays in the list, so the run lasts while the queue does
    (the scheduler's rule, 2026-10-21). Its jobs run under slurmstepd and count as ordinary processes."""
    st = docker("inspect", "-f", "{{.State.Pid}} {{.State.Running}}", cid, check=False)
    if st.returncode != 0:
        return None
    pid_s, running = st.stdout.split()
    if running != "true":
        return None
    init = int(pid_s)
    cg = cgroup_path(cid)
    try:
        pids = [int(p) for p in open(os.path.join(cg, "cgroup.procs")).read().split()]
    except OSError:
        return None
    others = []
    for p in pids:
        if p == init:
            continue
        try:
            ppid = int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1])
            cmd = open(f"/proc/{p}/cmdline").read().replace("\0", " ").strip()
        except (OSError, IndexError, ValueError):
            continue
        if ppid == init and cmd == "sleep infinity":
            continue
        if comm_of(p) == "slurmctld" and not sched_active(boxd):
            continue
        others.append(p)
    return others


def comm_of(pid):
    try:
        return open(f"/proc/{pid}/comm").read().strip()
    except OSError:
        return None


def sched_active(boxd):
    """True while slurmctld's active file says a job has not yet ended; a missing or unreadable file counts as active,
    so a controller that has not written it yet keeps the run (the horizon still ends it)."""
    if boxd is None:
        return True
    try:
        return int(open(os.path.join(boxd, SCHED_ACTIVE)).read().strip() or 1) > 0
    except (OSError, ValueError):
        return True


_CG = {}


def cgroup_path(cid):
    if "driver" not in _CG:
        _CG["driver"] = docker("info", "-f", "{{.CgroupDriver}}").stdout.strip()
    if _CG["driver"] == "systemd":
        return f"/sys/fs/cgroup/system.slice/docker-{cid}.scope"
    return f"/sys/fs/cgroup/docker/{cid}"


RESERVED_CORES = 2  # cores 0-1: the gpu-mock sidecar, the forwarder edge and the host's own processes


def parse_cpuset(spec):
    cores = set()
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cores.update(range(int(lo), int(hi or lo) + 1))
    return cores


THINKING_DISPLAY = "summarized"
# Flags run_one.py sets on claude itself; --claude-arg may not set them again (a later duplicate could override the
# model, the permission mode, the output or input format the monitor depends on, or the session id).
RESERVED_CLAUDE_FLAGS = {"-p", "--print", "--output-format", "--input-format", "--verbose", "--permission-mode",
                         "--model", "--thinking-display", "--session-id", "--dangerously-skip-permissions"}  # what the API returns of the model's thinking; see the claude_code argv below
HOST_RESERVE_BYTES = 8 << 30  # RAM kept back from the box: dockerd, containerd, the sampler, the sidecar's 4 GB, the edge


def machine_caps(a, ncpu=None, mem_total=None):
    """The box's CPU quota and memory cap. The machine is the isolation boundary: one box
    per machine, and the box takes the whole machine but the sidecar's cores and a host reserve. Explicit
    --cpus/--memory are honoured, for tests. Returns (cpus_str, memory_str, source)."""
    ncpu = ncpu or os.cpu_count() or 1
    if mem_total is None:
        mem_total = next((int(l.split()[1]) * 1024 for l in open("/proc/meminfo") if l.startswith("MemTotal:")), 0)
    cpus = str(max(1, ncpu - RESERVED_CORES)) if a.cpus == "machine" else a.cpus
    if a.memory == "machine":
        mem_bytes = max(1 << 30, mem_total - HOST_RESERVE_BYTES)
        memory = f"{mem_bytes >> 20}m"
    else:
        memory = a.memory
    source = "machine" if (a.cpus == "machine" and a.memory == "machine") else \
        "explicit" if (a.cpus != "machine" and a.memory != "machine") else "mixed"
    return cpus, memory, source


def running_boxes():
    """Names of p006 boxes running on this machine. One box per machine: a second is refused, never squeezed in."""
    out = docker("ps", "--filter", "label=p006.role=box", "--format", "{{.Names}}", check=False).stdout.split()
    return [n for n in out if n]


def choose_cpuset(a):
    """The box's cpuset. With only --cpus, torch and every OpenMP library see all the host's cores, start
    that many threads under a smaller quota, and the box is throttled in bursts: gpu-mock measured 134 s of
    throttled thread time in 12 s under --cpus 4, and a 3.7x slowdown of its modelled compute (2026-10-20).
    Pinning makes nproc, os.sched_getaffinity and torch's thread count match the grant. Cores 0 to
    RESERVED_CORES-1 are the sidecar's and the host's; the box gets the next ceil(--cpus) cores."""
    if a.cpuset == "none":
        return None
    n = math.ceil(float(a.cpus))
    total = os.cpu_count() or 1
    if a.cpuset != "auto":
        want = parse_cpuset(a.cpuset)
        if len(want) < n:
            raise RuntimeError(f"--cpuset {a.cpuset} has {len(want)} cores, fewer than --cpus {a.cpus}")
        return a.cpuset
    if RESERVED_CORES + n > total:
        raise RuntimeError(f"--cpus {a.cpus} needs {n} cores above the {RESERVED_CORES} reserved, but the machine "
                           f"has {total}")
    return f"{RESERVED_CORES}-{RESERVED_CORES + n - 1}"


def bridge_gateway():
    r = docker("network", "inspect", "bridge", "-f", "{{(index .IPAM.Config 0).Gateway}}", check=False)
    return r.stdout.strip() or "172.17.0.1"


def main(argv):
    a = parse_args(argv)
    if a.stop:
        open(os.path.join(a.stop, ".stop"), "w").write(now_utc() + "\n")
        print(f"stop requested for {a.stop}")
        return 0
    if os.geteuid() != 0:
        sys.exit("run as root: the run's disk is a loop-mounted image")
    if sys.platform != "linux":
        sys.exit("runs on a Linux Docker host only")

    tpath = os.path.join(REG, "tasks", a.task + ".json")
    if not os.path.exists(tpath):
        print(f"no task {a.task} in {REG}/tasks", file=sys.stderr)
        return 2
    task = json.load(open(tpath))
    out = os.path.abspath(a.out)
    if os.path.exists(out) and os.listdir(out):
        print(f"{out} is not empty", file=sys.stderr)
        return 2
    boxes = running_boxes()
    if boxes:
        print(f"a box is already running on this machine ({', '.join(boxes)}); one box per machine "
              "(the machine is the isolation boundary: the study, 2026-10-20). Wait for it to end, or use another machine",
              file=sys.stderr)
        return 2
    a.cpus, a.memory, a.caps_source = machine_caps(a)
    a.image_chosen_by = "--image" if a.image else "p006.task-env label"
    a.image_label = image_label_of(task, a.task)
    a.image, why = resolve_image(a.image_label, a.image)
    if why:
        print(why, file=sys.stderr)
        return 2
    sidecar = a.sidecar_image or docker("image", "inspect", "-f", '{{ index .Config.Labels "p006.sidecar-image" }}',
                                        a.image).stdout.strip()
    if not sidecar or sidecar == "<no value>" or docker("image", "inspect", sidecar, check=False).returncode != 0:
        print(f"no gpu-daemon image for {a.image} (its p006.sidecar-image label: {sidecar or 'none'}); "
              "rebuild with isolation/image/build.sh, or pass --sidecar-image", file=sys.stderr)
        return 2
    if a.harness in MODEL_HARNESSES and not os.path.exists(a.link_key):
        print(f"no link key at {a.link_key}", file=sys.stderr)
        return 2

    name = re.sub(r"[^A-Za-z0-9_.-]", "-", a.cost_tag)
    # Anything the box mounts shows its host path in the agent's /proc/self/mountinfo, so the box directory and the
    # socket volume get random names that say nothing about the run (2026-10-21: they were <run dir>/box and
    # p6-sock-<cost tag>, which put the runner's line name, its arm and model in front of any agent reading
    # mountinfo). The box directory lives at <srv>/bx/<hex> during the run, <run>/box links to it, and teardown
    # moves it into <run>/box, so every reader of <run>/box/ after the run is unchanged.
    neutral = uuid.uuid4().hex
    names = {"box": f"p6-box-{name}", "gpu": f"p6-gpu-{name}", "edge": f"p6-edge-{name}",
             "net": f"p6-net-{name}", "vol": f"nvpd-{neutral[:12]}"}
    session_id = str(uuid.uuid4())
    os.makedirs(out, exist_ok=True)
    os.chmod(out, 0o755)
    bx_root = os.path.join(os.path.dirname(os.path.abspath(a.disk_root.rstrip("/"))), "bx")
    os.makedirs(bx_root, exist_ok=True)
    os.chmod(bx_root, 0o711)
    boxd = os.path.join(bx_root, neutral[:16])
    os.makedirs(boxd)
    os.chmod(boxd, 0o700)
    os.symlink(boxd, os.path.join(out, "box"))

    task_sha = sha256_file(tpath)
    shutil.copy2(tpath, os.path.join(out, "task.json"))
    # The horizon is the run's hard stop, in recorded seconds from the agent's start. It is never shown to the
    # agent (design/IDEA.md, "No deadline in the brief", the study 2026-10-20): not in the brief, the environment,
    # or any file in the box. A task without horizon_s falls back to its deadline_s (the earlier design), and
    # run.json says which was used.
    if a.deadline_s:
        deadline_s, deadline_how = a.deadline_s, "--horizon-s"
    elif task.get("horizon_s"):
        deadline_s, deadline_how = float(task["horizon_s"]), "task.horizon_s"
    elif task.get("deadline_s"):
        deadline_s, deadline_how = float(task["deadline_s"]), "task.deadline_s (earlier design: the brief stated it)"
    else:
        deadline_s = 1.5 * float(task["stats"]["lower_bound_makespan_s"])
        deadline_how = "1.5 x stats.lower_bound_makespan_s (task has no horizon_s or deadline_s)"
    if a.brief_file:
        brief = open(a.brief_file).read().rstrip("\n")
        brief_source = {"brief_source": "file", "brief_file": os.path.abspath(a.brief_file)}
    else:
        brief = task["brief"]["text"].rstrip("\n")
        brief_source = {"brief_source": "task"}
    brief_source["task_brief_sha256"] = hashlib.sha256(brief.encode()).hexdigest()
    nudge = None
    if a.nudge:
        camp = json.load(open(a.campaign))
        levels = camp["nudges"]["levels"]
        if a.nudge not in levels:
            print(f"--nudge {a.nudge} is not a level in {a.campaign} (levels: {', '.join(levels)})", file=sys.stderr)
            return 2
        text = levels[a.nudge]["text"]
        nudge = {"level": a.nudge, "name": levels[a.nudge].get("name"), "version": camp["nudges"].get("version"),
                 "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "campaign_file": os.path.abspath(a.campaign),
                 "campaign_sha256": sha256_file(a.campaign)}
        if text:  # L0 is empty: nothing appended, still recorded
            brief = brief + "\n\n" + text
    brief_source["brief_sha256"] = hashlib.sha256(brief.encode()).hexdigest()  # the brief actually used
    open(os.path.join(out, "brief.txt"), "w").write(brief + "\n")
    records = sorted({r for j in task["jobs"] for r in (j.get("records") or []) + (j.get("candidate_pool") or [])})
    run = {
        "schema": "box-run/0.1", "cost_tag": a.cost_tag, "session_id": session_id,
        "inputs": {"task": a.task, "harness": a.harness, "model": a.model if a.harness in MODEL_HARNESSES else None,
                   "seed": a.seed, "speed": a.speed, "budget_usd": a.budget_usd, "max_hours": a.max_hours,
                   "stop_dir": a.stop_dir, "after_deadline": a.after_deadline, **brief_source, "nudge": nudge,
                   "dressing": a.dressing, "disclosure": a.disclosure, "scheduler": a.scheduler},
        "skills": [{"name": os.path.basename(os.path.abspath(d).rstrip("/")), "source": os.path.abspath(d),
                    "skill_md_sha256": sha256_file(os.path.join(d, "SKILL.md")),
                    "tree_sha256": tree_sha256(d),
                    "mounted_at": f"{HOME}/{SKILL_ROOT[a.harness]}/{os.path.basename(os.path.abspath(d).rstrip('/'))}"}
                   for d in a.skill],
        "effects_not_built": ["dressing", "disclosure", "seed (the scripted agent and replay draw do not read it)"],
        "caps": {"cpus": a.cpus, "memory": a.memory, "memory_swap": a.memory, "disk": a.disk,
                 "shm": a.shm, "pids": a.pids, "oom_protect": a.oom_protect, "source": a.caps_source,
                 "host_reserve_bytes": HOST_RESERVE_BYTES if a.caps_source != "explicit" else None},
        "horizon_s": deadline_s, "horizon_how": deadline_how, "horizon_wall_s": deadline_s / a.speed,
        "deadline_s": deadline_s, "deadline_how": deadline_how, "deadline_wall_s": deadline_s / a.speed,  # older readers
        "task_sha256": task_sha,
        "staged_inputs_sha256": staged_inputs(task),
        "records_sha256": {r: sha256_file(os.path.join(REG, "records", r + ".json")) for r in records
                           if os.path.exists(os.path.join(REG, "records", r + ".json"))},
        # The sidecar mounts all of records/ and gpu-mock's daemon (r4 on) loads every example-order-*.json there, listed
        # or not, so the tables a run's daemon could read are recorded here
        "example_order_sha256": {n[:-5]: sha256_file(os.path.join(REG, "records", n))
                                 for n in sorted(os.listdir(os.path.join(REG, "records"))
                                                 if os.path.isdir(os.path.join(REG, "records")) else [])
                                 if n.startswith("example-order-") and n.endswith(".json")},
        "code": {"deployed_at": (open("/opt/computewise-deploy/DEPLOYED_AT").read().strip()
                                 if os.path.exists("/opt/computewise-deploy/DEPLOYED_AT") else None),
                 "run_one_sha256": sha256_file(os.path.abspath(__file__)),
                 "cgroup_sampler_sha256": sha256_file(SAMPLER),
                 "weights_digest_sha256": sha256_file(WEIGHTS_DIGEST_PY)},
        "sidecar": {"image": sidecar, "id": docker("image", "inspect", "-f", "{{.Id}}", sidecar).stdout.strip(),
                    "src_sha256": sh(oneoff_run("--network", "none", "--entrypoint", "cat", sidecar,
                                         "/opt/gpu-daemon/SRC_SHA256"), check=False).stdout.strip() or None,
                    "src_override": os.path.abspath(a.gpu_mock_src) if a.gpu_mock_src else None,
                    "src_override_sha256": tree_sha(a.gpu_mock_src) if a.gpu_mock_src else None},
        "containers": names, "host": {"uname": os.uname().release, "nproc": os.cpu_count(),
                                      "cgroup_driver": None, **host_identity(), "mem_baseline": mem_baseline()},
        "image": {"name": a.image, "chosen_by": a.image_chosen_by, "task_env_label": a.image_label, "id": docker("image", "inspect", "-f", "{{.Id}}", a.image).stdout.strip()},
    }
    try:
        run["image"]["manifest"] = json.loads(sh(oneoff_run("--network", "none", "--entrypoint", "cat", a.image,
                                                     "/usr/local/share/image-manifest/build-args.json")).stdout)
        run["image"]["claude_version"] = sh(oneoff_run("--network", "none", "--entrypoint", "cat", a.image,
                                                "/usr/local/share/image-manifest/claude-version.txt")).stdout.strip()
    except RuntimeError:
        run["image"]["manifest"] = None
    if a.scheduler == "on" and docker("image", "inspect", "-f", '{{ index .Config.Labels "p006.scheduler" }}', a.image,
                                      check=False).stdout.strip() in ("", "<no value>"):
        print(f"image {a.image} has no scheduler (label p006.scheduler); build it with the current "
              "isolation/image/build.sh", file=sys.stderr)
        return 2
    if a.scheduler == "on":
        missing = [f for f in SCHED_LIB_FILES if not os.path.isfile(os.path.join(SCHED_LIB, f))]
        if missing:
            print(f"the scheduler library to mount is incomplete: {SCHED_LIB} lacks {missing}", file=sys.stderr)
            return 2
    if a.harness == "codex":
        # Codex runs only in an image that carries it and whose box-init takes the FIFO (codex app-server reads its
        # JSON-RPC requests on stdin, written by this command; the Codex driver's spec, 2026-10-21).
        lbl = lambda k: docker("image", "inspect", "-f", '{{ index .Config.Labels "%s" }}' % k, a.image,
                               check=False).stdout.strip()
        if lbl("p006.box-stdin") != "fifo" or lbl("p006.codex") in ("", "<no value>"):
            print(f"image {a.image} has no Codex or no FIFO box-init (labels p006.codex, p006.box-stdin=fifo); "
                  "build it with the current isolation/image/build.sh", file=sys.stderr)
            return 2
        run["codex"] = {"argv": ["codex", "app-server"] + list(a.codex_arg), "extra_flags": list(a.codex_arg),
                        "version": lbl("p006.codex"),
                        "session_module_sha256": sha256_file(CODEX_PY), "grace_s": CODEX.GRACE_S,
                        "resume": a.codex_resume,
                        "resume_rule": ({"errors": list(CODEX.RESUME_ERRORS), "backoff_s": CODEX.RESUME_BACKOFF_S,
                                         "reset_s": CODEX.RESUME_RESET_S, "max": CODEX.RESUME_MAX}
                                        if a.codex_resume else None),
                        "config_toml": CODEX.config_toml(a.model, f"http://{EDGE_ALIAS}:{EDGE_PORT}/v1"),
                        "codex_home": f"{HOME}/.codex", "stdin": "fifo"}
    if a.harness in ("grok", "opencode"):
        # Each runs only in an image that carries it (labels p006.grok, p006.opencode: the harness's version).
        ver = docker("image", "inspect", "-f", '{{ index .Config.Labels "p006.%s" }}' % a.harness, a.image,
                     check=False).stdout.strip()
        if ver in ("", "<no value>"):
            print(f"image {a.image} has no {a.harness} (label p006.{a.harness}); build it with "
                  "isolation/image/build_harness_layer.sh", file=sys.stderr)
            return 2
        base_url = f"http://{EDGE_ALIAS}:{EDGE_PORT}/v1"
        if a.harness == "grok":
            # --background-wait-timeout: by default a headless run kills its background work 600 s after the agent's
            # latest turn and exits 1, which the interactive product never does (analysis' ruling, 2026-10-26 14:08
            # MDT); GROK_BACKGROUND_WAIT_S says why it is one constant. No --max-turns: the product has none.
            bg_wait = GROK_BACKGROUND_WAIT_S
            argv = ["grok", "-p", brief, "-m", a.model.replace(".", "-"), "--always-approve",
                    "--output-format", "streaming-json", "--background-wait-timeout", str(bg_wait)]
            cfg = grok_config_toml(a.model, base_url)
            run["grok"] = {"argv_flags": argv[3:], "version": ver, "background_wait_timeout_s": bg_wait,
                           "config_toml": cfg, "config_toml_sha256": hashlib.sha256(cfg.encode()).hexdigest(),
                           "title_model": GROK_TITLE_MODEL, "stdin": "/dev/null"}
        else:
            argv = ["opencode", "run", "--format", "json", "--auto", "-m", f"llm/{a.model}", brief]
            cfg = opencode_config_json(a.model, base_url, "sk-" + hashlib.sha256(session_id.encode()).hexdigest()[:40])
            run["opencode"] = {"argv_flags": argv[2:-1], "version": ver,
                               "bash_default_timeout_ms": OPENCODE_BASH_TIMEOUT_MS,
                               "config_json_sha256": hashlib.sha256(cfg.encode()).hexdigest(), "stdin": "/dev/null"}
        state_cfg = cfg
    if a.harness == "scripted":
        shutil.copy2(a.plan, os.path.join(out, "plan.json"))
        plan = json.load(open(a.plan))
        run["scripted"] = {"policy": plan.get("policy"), "plan_sha256": sha256_file(a.plan),
                           "agent_py_sha256": sha256_file(a.agent_py)}
    elif a.harness == "claude-code":
        # --thinking-display summarized: the API returns a summary of the model's thinking, so the transcript carries
        # the reason behind each scheduling choice. The campaign runs every arm
        # with it from the start. It is one of the two display values the gateway accepts, and sends no first-party
        # beta (notes/thinking-display-400-2026-10-20.md, "Gap G7").
        stdin_mode = "fifo" if docker("image", "inspect", "-f", '{{ index .Config.Labels "p006.box-stdin" }}',
                                      a.image, check=False).stdout.strip() == "fifo" else "file"
        argv = ["claude", "-p", "--output-format", "stream-json", "--verbose"]
        if stdin_mode == "fifo":
            argv += ["--input-format", "stream-json"]
        argv += ["--permission-mode", "bypassPermissions", "--model", a.model,
                 "--thinking-display", THINKING_DISPLAY, "--session-id", session_id]
        # The backlog's harness_flags, passed through unchanged (runner, 2026-10-21; first use: --disallowedTools Task,
        # the no-subagent arm, where the agent simply has no subagent tool and is told nothing about it).
        argv += list(a.claude_arg)
        run["claude_code"] = {"argv": argv, "thinking_display": THINKING_DISPLAY, "stdin": stdin_mode,
                              "bash_timeout_ms": {"default": BASH_TIMEOUTS_MS(a)[0], "max": BASH_TIMEOUTS_MS(a)[1],
                                                  "source": "--bash-timeout-ms" if a.bash_timeout_ms else "run_one default"},
                              "extra_flags": list(a.claude_arg), "agent_teams": bool(a.agent_teams),
                              "model": model_parts(a.model), "auto_compact_window": a.auto_compact_window,
                              "effort_level": a.effort_level,
                              "lifecycle_sha256": sha256_file(LIFECYCLE_PY) if stdin_mode == "fifo" else None,
                              "lifecycle_grace_s": LIFECYCLE.GRACE_S if stdin_mode == "fifo" else None}
    write_json(os.path.join(out, "run.json"), run)

    status = {"status": None, "reason": None, "harness_exit_code": None, "start_utc": now_utc(),
              "end_utc": None, "agent_start_wall": None, "harness_end_wall": None,
              "deadline_wall": None, "teardown_wall": None, "teardown_reason": None}
    state = {"img": None, "mnt": None, "sampler": None, "cid": None, "started": False, "staged_project": None,
             "stdin_fd": None, "lifecycle": None, "codex": None, "codex_rpc": None, "session_id": session_id}
    status["stdin_closed_wall"], status["stdin_close_reason"], status["harness_results"] = None, None, None
    run["box_dir_host"] = boxd
    t0 = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: STOP.update(why="signal"))
    signal.signal(signal.SIGINT, lambda *_: STOP.update(why="signal"))
    rc = 0
    try:
        staged = stage_task_tree(a.task, task_sha, a.cache_root, run["staged_inputs_sha256"])
        envs = (task.get("environment") or {}).get("python_environments")
        env_hf = None
        if envs:
            # one cache per environment (stage_env_hf); an environment that names no hf_home is mounted at
            # ~/.cache/huggingface as every task's is, one that names one under the workspace is mounted there
            caches = stage_env_hf(task, a.task, staged, a.cache_root, a.image)
            prepare_tree(task, a.task, staged, a.image)
            home_env = [n for n in sorted(caches) if not envs[n].get("hf_home")]
            hf = caches[home_env[0]] if home_env else stage_hf({**task, "environment": {**task["environment"], "hub": []}},
                                                              a.task, staged, a.cache_root, a.image)
            env_hf = [(envs[n]["hf_home"], caches[n]) for n in sorted(caches) if envs[n].get("hf_home")]
            run["hf_caches"] = {n: {"dir": caches[n], "mounted_at": envs[n].get("hf_home") or "~/.cache/huggingface"}
                                for n in sorted(caches)}
            if os.path.exists(os.path.join(staged, "prepared.json")):
                run["prepared"] = json.load(open(os.path.join(staged, "prepared.json")))
        else:
            hf = stage_hf(task, a.task, staged, a.cache_root, a.image)
        os.makedirs(a.disk_root, exist_ok=True)
        log("making the run's disk")
        # The disk's file name is neutral too: `mount` inside the box prints a loop device's backing file, which was
        # <disk-root>/<cost tag>.img (found by the seat probe's our-names pattern, 2026-10-21).
        state["img"], state["mnt"], ro_mounts = make_disk(f"d-{neutral[:16]}", a.disk, a.disk_root, a.image, staged, hf,
                                                          env_hf)
        run["disk_image"] = state["img"]
        state["staged_project"] = os.path.join(staged, "project")
        run["host"]["cgroup_driver"] = docker("info", "-f", "{{.CgroupDriver}}").stdout.strip()

        docker("network", "create", "--internal", "--label", f"p006.run={a.cost_tag}", names["net"])
        docker("volume", "create", "--label", f"p006.run={a.cost_tag}", names["vol"])
        vol_mp = docker("volume", "inspect", "-f", "{{.Mountpoint}}", names["vol"]).stdout.strip()
        os.chown(vol_mp, UID, UID)
        occ = os.path.join(out, "occupancy.jsonl")
        open(occ, "w").close()
        os.chown(occ, UID, UID)

        log("starting gpu-mock's sidecar")
        sock = f"{SOCK_DIR}/socket"
        src_mount = ["-v", f"{os.path.abspath(a.gpu_mock_src)}:/opt/gpu-daemon/src:ro"] if a.gpu_mock_src else []
        side_cores = f"0-{RESERVED_CORES - 1}" if (os.cpu_count() or 1) > RESERVED_CORES else None
        side_pin = ["--cpuset-cpus", side_cores] if side_cores and a.cpuset != "none" else []
        # The daemon reads the run's task file when it takes --task (gpu-mock, 2026-10-21: jobs[].rerun_siblings, the
        # task's record lists, candidate_pool decision slots, and which surrogate families serve), and synthesises from
        # registry/surrogates when given --surrogates. Asked of this sidecar's own daemon, so an older sidecar runs as
        # before. Its output directory is the run's: occupancy.jsonl and, since 20b43ee3, lr_stream.jsonl beside it.
        daemon_help = sh(oneoff_run("--network", "none", "--entrypoint", "python3", *src_mount, sidecar, "-I",
                                    "/opt/gpu-daemon/src/daemon/gpu_daemon.py", "--help"), check=False).stdout or ""
        task_arg = "--task" in daemon_help
        surr_dir = os.path.join(REG, "surrogates")
        surr_arg = "--surrogates" in daemon_help and os.path.isdir(surr_dir)
        run["gpu_daemon"] = {"task_arg": task_arg, "surrogates": surr_dir if surr_arg else None}
        lr = os.path.join(out, "lr_stream.jsonl")
        open(lr, "w").close()
        os.chown(lr, UID, UID)
        extra_mounts = (["-v", f"{os.path.join(out, 'task.json')}:/out/task.json:ro"] if task_arg else []) + \
                       (["-v", f"{surr_dir}:/surrogates:ro"] if surr_arg else []) + ["-v", f"{lr}:/out/lr_stream.jsonl"]
        extra_args = (" --task /out/task.json" if task_arg else "") + (" --surrogates /surrogates" if surr_arg else "")
        docker("run", "-d", "--name", names["gpu"], "--label", "p006.role=gpu-daemon", *side_pin,
               "--label", f"p006.run={a.cost_tag}", "--network", "none", "--init", "--user", f"{UID}:{UID}",
               "--cpus", "2", "--memory", "4g", "--entrypoint", "sh",
               *src_mount, *extra_mounts, "-v", f"{REG}/records:/registry:ro",
               "-v", f"{names['vol']}:{SOCK_DIR}", "-v", f"{occ}:/out/occupancy.jsonl", sidecar, "-c",
               f"python3 -I /opt/gpu-daemon/src/daemon/gpu_daemon.py --socket {sock} --registry /registry "
               f"--gpus {task['box']['gpus']} --speed {a.speed} --occupancy /out/occupancy.jsonl{extra_args} & D=$!; "
               f"i=0; while [ ! -S {sock} ] && [ $i -lt 300 ]; do sleep 0.1; i=$((i+1)); done; "
               f"python3 -I /opt/gpu-daemon/src/bridge/overrides_bridge.py --socket {sock} --overrides {SOCK_DIR}/overrides.yaml & "
               f"wait $D")
        for _ in range(300):
            if os.path.exists(os.path.join(vol_mp, "socket")):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("the gpu-mock daemon never created its socket: " +
                               docker("logs", names["gpu"], check=False).stderr[-1500:])

        env = task_env(task)  # the task's own runtime settings first; run_one's keys below win
        # PYTHONPATH is the image's own (if it has one) followed by the project, and the speed goes into the box only
        # for an image whose patch reads it from the environment. Images built before the scrub of mock names
        # (2026-10-21: the DPO and battery images, and every mng3 image up to d6ab84ffe8b3) carry
        # PYTHONPATH=/opt/gpu-mock/site and GPU_MOCK_SOCKET in their ENV and need GPU_MOCK_SPEED, exactly as before;
        # images built after it have neither, load their patch from site-packages and take the speed from the
        # daemon's hello, so their box environment names nothing of the mock (isolation/notes/mng3-image-2026-10-20.md).
        image_env = dict(kv.split("=", 1) for kv in json.loads(
            docker("image", "inspect", "-f", "{{json .Config.Env}}", a.image).stdout.strip() or "[]") if "=" in kv)
        env.update({"HOME": HOME, "USER": "researcher", "LOGNAME": "researcher", "SHELL": "/bin/bash",
               "PYTHONPATH": ":".join(p for p in (image_env.get("PYTHONPATH"), PROJECT) if p),
               "HF_HOME": f"{HOME}/.cache/huggingface", "BOX_UID": str(UID),
               "BOX_OOM_PROTECT": "1" if a.oom_protect == "on" else "0",
               "BOX_SCHEDULER": "1" if a.scheduler == "on" else "0"})
        if "GPU_MOCK_SOCKET" in image_env:
            env["GPU_MOCK_SPEED"] = str(a.speed)
        mounts = [f"{state['mnt']}/home:{HOME}", f"{state['mnt']}/tmp:/tmp", f"{boxd}:/run/box",
                  f"{names['vol']}:{SOCK_DIR}"] + [f"{s}:{t}:ro" for s, t in ro_mounts]
        if a.skill:
            # Copied to the box's neutral scratch directory first: a bind mount's source path is in the box's
            # mountinfo, and the skills' own directory is under experiments/ (seat probe's names, 2026-10-21).
            sk = boxd + ".k"
            for d in a.skill:
                n = os.path.basename(os.path.abspath(d).rstrip("/"))
                shutil.copytree(os.path.abspath(d), os.path.join(sk, n))
                mounts.append(f"{os.path.join(sk, n)}:{HOME}/{SKILL_ROOT[a.harness]}/{n}:ro")
            chown_tree(sk)
        if a.scheduler == "on":
            # the deployed library over the image's, read-only (SCHED_LIB above); the commands' shims and slurmctld
            # import it from /usr/lib/slurm, the root fs is read-only anyway, and nothing else in the box changes
            sl = boxd + ".l"
            os.makedirs(sl)
            for f in SCHED_LIB_FILES:
                shutil.copyfile(os.path.join(SCHED_LIB, f), os.path.join(sl, f))
                os.chmod(os.path.join(sl, f), 0o644)
            os.chmod(sl, 0o755)
            mounts.append(f"{sl}:/usr/lib/slurm:ro")
            run["scheduler_lib"] = {"mounted_at": "/usr/lib/slurm", "mode": "ro", "source": SCHED_LIB,
                                    "sha256": {f: sha256_file(os.path.join(sl, f)) for f in SCHED_LIB_FILES},
                                    "why": "route (a), 2026-11-04: the controller fix without an image rebuild"}
            log(f"scheduler library mounted: ctld.py {run['scheduler_lib']['sha256']['ctld.py'][:16]}")
        if a.harness == "scripted":
            scr = boxd + ".s"
            os.makedirs(scr, exist_ok=True)
            shutil.copy2(os.path.abspath(a.agent_py), os.path.join(scr, "agent.py"))
            shutil.copy2(os.path.join(out, "plan.json"), os.path.join(scr, "plan.json"))
            mounts += [f"{os.path.join(scr, 'agent.py')}:/opt/scripted/agent.py:ro",
                       f"{os.path.join(scr, 'plan.json')}:/opt/scripted/plan.json:ro"]
            harness = ["python3", "-I", "/opt/scripted/agent.py", "--plan", "/opt/scripted/plan.json",
                       "--transcript", f"{HOME}/.claude/projects/{CWD_SLUG}/scripted.jsonl",
                       "--session-id", session_id]
        else:
            edged = os.path.join(out, "edge")
            os.makedirs(edged)
            os.chmod(edged, 0o700)
            shutil.copy2(a.link_key, os.path.join(edged, "link.key"))
            upstream = a.edge_upstream or f"http://{bridge_gateway()}:8790"
            write_json(os.path.join(edged, "edge.json"),
                       {"mode": "edge", "upstream": upstream, "cost_tag": a.cost_tag, "session_id": session_id,
                        "allowed_models": edge_models(a), "models": edge_models(a), "log_path": "/state/edge.jsonl",
                        "link_token_file": "/state/link.key", "listen_port": EDGE_PORT})
            run["edge"] = {"upstream": upstream, "alias": EDGE_ALIAS, "port": EDGE_PORT,
                           "forwarder_sha256": sha256_file(FORWARDER)}
            docker("run", "-d", "--name", names["edge"], "--label", f"p006.run={a.cost_tag}",
                   "--label", "p006.role=edge", *side_pin, "--cpus", "1", "--memory", "512m",
                   "--entrypoint", "python3",
                   "-v", f"{FORWARDER}:/app/forwarder.py:ro", "-v", f"{edged}:/state", a.image,
                   "-I", "/app/forwarder.py", "/state/edge.json")
            docker("network", "connect", "--alias", EDGE_ALIAS, names["net"], names["edge"])
        if a.harness == "codex":
            # The agent's ~/.codex/config.toml: a neutrally named provider at the edge, the Responses API, effort
            # medium (the product's default for every gpt-6 model, which a custom provider would not get). Codex
            # sends no API key, so the edge's run tag is the only identity, as for Claude.
            cdir = os.path.join(state["mnt"], "home", ".codex")
            os.makedirs(cdir, exist_ok=True)
            open(os.path.join(cdir, "config.toml"), "w").write(run["codex"]["config_toml"])
            if a.skill:
                # the skill mounts' parent, made here so it is the agent's: Docker would create it root-owned as the
                # mountpoint's parent, and Codex writes its bundled skills under <CODEX_HOME>/skills/.system
                os.makedirs(os.path.join(cdir, "skills"), exist_ok=True)
            chown_tree(cdir)
            run["codex"]["config_toml_sha256"] = hashlib.sha256(run["codex"]["config_toml"].encode()).hexdigest()
            fifo = os.path.join(boxd, "brief")
            os.mkfifo(fifo, 0o600)
            fd = os.open(fifo, os.O_RDWR)
            # resume= only when asked: codex_session.py before db728687 has no such parameter
            cs = CODEX.CodexSession(a.model, PROJECT, brief, **({"resume": True} if a.codex_resume else {}))
            rpc = open(os.path.join(boxd, "codex-rpc.jsonl"), "a", buffering=1)
            state.update(stdin_fd=fd, codex=cs, codex_rpc=rpc, codex_sent=0, stdout_off=0, stdout_buf=b"")
            write_codex(state, cs.start_lines())
            harness = run["codex"]["argv"]
        elif a.harness == "claude-code":
            if run["claude_code"]["stdin"] == "fifo":
                # The brief goes in as one stream-json user message on a pipe whose write end this command holds, so
                # Claude Code keeps the session while background work runs and its completion starts a new turn, as
                # in the interactive product; the monitor closes the pipe on the runner's lifecycle rule
                # (runner/harness/lifecycle.py; Q-prepP6-haiku-L0 lost six subagents and three jobs when `-p` ended
                # the session after its bounded background wait, 2026-10-21).
                fifo = os.path.join(boxd, "brief")
                os.mkfifo(fifo, 0o600)
                fd = os.open(fifo, os.O_RDWR)  # does not wait for the reader; the box's init opens the other end
                msg = LIFECYCLE.brief_message(brief).encode()
                threading.Thread(target=lambda: os.write(fd, msg), daemon=True).start()
                state["stdin_fd"], state["lifecycle"], state["stdout_off"], state["stdout_buf"] = \
                    fd, make_lifecycle(out), 0, b""
            else:
                open(os.path.join(boxd, "brief"), "w").write(brief + "\n")
            # CLAUDE_CODE_THINKING_DISPLAY_UPDATES=0: in -p stream-json mode Claude Code 2.1.303 asks for
            # thinking display "updates", a first-party beta that some providers' Anthropic-compatible paths refuse (HTTP 400).
            # Claude Code then retries without it and never sends it again, so this only removes the two
            # refused requests at session start (isolation/notes/thinking-display-400-2026-10-20.md).
            env.update({"ANTHROPIC_BASE_URL": f"http://{EDGE_ALIAS}:{EDGE_PORT}",
                        "ANTHROPIC_API_KEY": "sk-ant-api03-" + hashlib.sha256(session_id.encode()).hexdigest()[:40],
                        "ANTHROPIC_MODEL": a.model, "ANTHROPIC_SMALL_FAST_MODEL": a.model,
                        "ANTHROPIC_DEFAULT_HAIKU_MODEL": a.model, "ANTHROPIC_DEFAULT_SONNET_MODEL": a.model,
                        "ANTHROPIC_DEFAULT_OPUS_MODEL": a.model, "ANTHROPIC_DEFAULT_FABLE_MODEL": a.model,
                        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1",
                        "BASH_DEFAULT_TIMEOUT_MS": str(BASH_TIMEOUTS_MS(a)[0]), "BASH_MAX_TIMEOUT_MS": str(BASH_TIMEOUTS_MS(a)[1]),
                        "CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"})
            if a.agent_teams:
                env["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] = "1"
            if a.auto_compact_window is not None:
                env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(a.auto_compact_window)
            if a.effort_level:
                env["CLAUDE_CODE_EFFORT_LEVEL"] = a.effort_level
            harness = run["claude_code"]["argv"]
        elif a.harness in ("grok", "opencode"):
            # No /run/box/brief: box-init then gives the harness /dev/null as stdin, which both need (opencode run hangs
            # on an open pipe). The config goes into the agent's home on the run's disk, which both write into.
            home = os.path.join(state["mnt"], "home")
            rel = os.path.join(".grok", "config.toml") if a.harness == "grok" else os.path.join(".config", "opencode", "opencode.json")
            os.makedirs(os.path.dirname(os.path.join(home, rel)), exist_ok=True)
            open(os.path.join(home, rel), "w").write(state_cfg)
            chown_tree(os.path.join(home, rel.split(os.sep)[0]))
            env.update({"TERM": "dumb"})
            if a.harness == "grok":
                # the key is a placeholder: Grok starts no model without one, and the edge drops client credentials
                # GROK_DISABLE_AUTOUPDATER: an updater could swap the harness version mid-campaign; the box blocks
                # egress anyway, so the agent sees no difference
                env.update({"LLM_API_KEY": "sk-" + hashlib.sha256(session_id.encode()).hexdigest()[:40],
                            "GROK_DISABLE_AUTOUPDATER": "1"})
            else:
                env.update({"OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS": str(OPENCODE_BASH_TIMEOUT_MS)})
            harness = argv
        run["env"] = {k: v for k, v in env.items() if k not in ("ANTHROPIC_API_KEY", "LLM_API_KEY")}
        write_json(os.path.join(out, "run.json"), run)

        log(f"starting the box ({a.harness})")
        lock = open(os.path.join(a.disk_root, ".cpuset.lock"), "w")
        fcntl.flock(lock, fcntl.LOCK_EX)  # held until the box is running: the one-box-per-machine check and the start are one step
        cpuset = choose_cpuset(a)
        run["caps"]["cpuset"] = cpuset
        run["caps"]["sidecar_cpuset"] = side_pin[1] if side_pin else None
        write_json(os.path.join(out, "run.json"), run)  # so the pin is on disk even if this command dies mid-run
        log(f"box cores: {cpuset or 'not pinned'}")
        cmd = ["docker", "run", "-d", "--name", names["box"], "--label", "p006.role=box",
               "--label", f"p006.run={a.cost_tag}", "--network", names["net"], "--init", "--read-only",
               "--cpus", a.cpus, "--memory", a.memory, "--memory-swap", a.memory, "--pids-limit", str(a.pids),
               "--shm-size", a.shm, "--cap-drop", "ALL", "--cap-add", "MKNOD", "--cap-add", "SETUID",
               "--cap-add", "SETGID", "--security-opt", "no-new-privileges", "-w", PROJECT]
        if cpuset:
            cmd += ["--cpuset-cpus", cpuset]
        if a.scheduler == "on":
            # the controller's socket and its job scripts: the root is read-only, and the agent must reach the socket,
            # so neither can be under /run/box (root-only); a few KB per job
            # exec on the spool: Docker mounts a tmpfs noexec by default, and slurmstepd execs each job's slurm_script
            # from there (fake-slurm, 2026-10-22: every job of the first in-box run failed with execve(): Permission
            # denied). Root-owned 0755, so the agent cannot write a script there.
            cmd += ["--tmpfs", "/run/slurm:mode=0755", "--tmpfs", "/var/spool/slurmd:mode=0755,exec"]
        if a.oom_protect == "on":
            cmd += ["--oom-score-adj", "-1000"]
        for m in mounts:
            cmd += ["-v", m]
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
        try:
            state["cid"] = sh(cmd + [a.image] + harness).stdout.strip()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        state["started"] = True
        state["sampler"] = subprocess.Popen(
            [sys.executable, "-I", SAMPLER, "--cgroup", cgroup_path(state["cid"]), "--out",
             os.path.join(out, "resources.jsonl"), "--interval", str(a.sample_interval),
             "--disk", state["mnt"], "--container-id", state["cid"]],
            stdout=open(os.path.join(out, "sampler.log"), "w"), stderr=subprocess.STDOUT)

        status["status"], status["reason"], status["teardown_reason"] = monitor(a, out, boxd, state, status,
                                                                                deadline_s, t0)
    except Exception as e:  # the command itself failed: record it, tear down, exit 1
        status["status"] = status["status"] or "harness-error"
        status["reason"] = f"run_one failed: {type(e).__name__}: {e}"[:2000]
        rc = 1
    finally:
        try:
            finish(a, out, boxd, state, status, names)
        finally:
            status["end_utc"] = now_utc()
            write_json(os.path.join(out, "status.json"), status)
            write_json(os.path.join(out, "run.json"), run)
            log(f"status: {status['status']} ({status['reason']})")
    return rc


def monitor(a, out, boxd, state, status, deadline_s, t0):
    max_t = t0 + a.max_hours * 3600
    agent_start = deadline = None
    while True:
        time.sleep(1.0)
        if agent_start is None:
            agent_start = read_float(os.path.join(boxd, "agent_start"))
            if agent_start is not None:
                deadline = agent_start + deadline_s / a.speed
                status["agent_start_wall"], status["deadline_wall"] = agent_start, deadline
        if os.path.exists(os.path.join(a.stop_dir, a.cost_tag)) and os.path.exists(os.path.join(out, ".stop")):
            return "killed-on-budget", "budget crossed; stopped by the runner", "stop"
        if os.path.exists(os.path.join(out, ".stop")) or STOP["why"]:
            return "stopped", f"external stop ({STOP['why'] or '.stop'})", "stop"
        if time.monotonic() > max_t:
            return "killed-on-time", f"--max-hours {a.max_hours} reached", "max_hours"
        if state.get("stdin_fd") is not None:
            if state.get("codex") is not None:
                feed_codex(boxd, state, status)
            else:
                feed_lifecycle(boxd, state, status)
        end_held_harness(state, status, boxd)
        procs = box_processes(state["cid"], boxd)
        if procs is None:
            return "crashed", "the box exited unexpectedly: " + \
                docker("logs", "--tail", "20", state["cid"], check=False).stderr[-1500:], "box_exited"
        hx = os.path.join(boxd, "harness_exit")
        if os.path.exists(hx):
            if status["harness_end_wall"] is None:
                status["harness_end_wall"] = read_float(os.path.join(boxd, "harness_end"))
                status["harness_exit_code"] = int(open(hx).read().strip() or -1)
                log(f"harness exited {status['harness_exit_code']}; the box stays up for running jobs")
            if deadline is not None and time.time() >= deadline:
                return final_status(boxd, status) + ("horizon",)
            if not procs:
                return final_status(boxd, status) + ("jobs_ended",)
            if status.get("harness_end_reason") == "lifecycle: work done, background task held the session":
                # Every task job had completed; what is left is the agent's own background work (a stuck poll held
                # the session that ended it), so the run ends here rather than at the horizon.
                return final_status(boxd, status) + ("work_done",)
            if idle_after_harness(state, status, out, procs):
                return final_status(boxd, status) + ("idle_after_harness",)
        elif deadline is not None and time.time() >= deadline and a.after_deadline == "stop":
            # The horizon is the hard stop: nothing after it is scored, so a harness still working then is ended.
            return "killed-on-time", "the task's horizon passed with the harness still running", "horizon"


AFTER_HARNESS_GRACE_S = 300.0   # no task job rank connected and the box idle this long after the harness exited
AFTER_HARNESS_IDLE_CORES = 0.25  # "idle": the box's processes used under a quarter of one core over the window
AFTER_HARNESS_WINDOW_S = 60.0


def idle_after_harness(state, status, out, procs):
    """True when the box should come down although processes remain: the harness has exited, no task job's rank is
    connected to gpu-mock's daemon (a job_hello in occupancy.jsonl with no job_bye; the daemon writes job_bye when the
    rank's connection closes, so a dead rank is never counted), and the box has used under AFTER_HARNESS_IDLE_CORES
    for AFTER_HARNESS_GRACE_S. What is left then is the agent's own shell work that does nothing: a poll that matches
    itself, a sleep, a tail. A job the agent left running keeps the box up (its rank is connected), and so does CPU
    work that may start one (data preparation chained before a launch, a rank's start-up before its hello), and a
    blind sleep shorter than the grace. (the study, 2026-10-21 23:49Z: c3-11's G-passbar-haiku-L0-v2 held its box
    24 min after a clean harness exit by a self-matching `while pgrep -f "seed 0"` loop; on a goal package that is up
    to the 18 h horizon.) The processes and the time are recorded as held_after_harness_by / held_after_harness_s."""
    now = time.time()
    st = state.setdefault("after_harness", {"off": 0, "open": {}, "cpu": [], "idle_since": None})
    try:
        with open(os.path.join(out, "occupancy.jsonl"), "rb") as fh:
            fh.seek(st["off"])
            chunk = fh.read()
    except OSError:
        chunk = b""
    end = chunk.rfind(b"\n") + 1
    st["off"] += end
    for line in chunk[:end].splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") == "job_hello":
            st["open"][e.get("job")] = e.get("pid")
        elif e.get("kind") == "job_bye":
            st["open"].pop(e.get("job"), None)
    # A job is connected only while its rank's process is alive in the box too: a rank killed between a request and
    # its reply can leave the daemon without a job_bye (gpu-mock, 2026-10-22 00:05Z; fixed in their next sidecar, not in
    # the 43f or a457 pairs). job_hello's pid is the box's own pid namespace; the box's processes are host pids, so each
    # is mapped through the last field of NSpid in /proc/<pid>/status.
    alive = box_ns_pids(procs)
    leaked = sorted(str(j) for j, pid in st["open"].items() if alive is not None and pid not in alive)
    if leaked:
        status["after_harness_leaked_jobs"] = leaked
    connected = [j for j, pid in st["open"].items() if alive is None or pid in alive]
    try:
        usec = int(next(l.split()[1] for l in open(os.path.join(cgroup_path(state["cid"]), "cpu.stat"))
                        if l.startswith("usage_usec ")))
        st["cpu"].append((now, usec))
    except (OSError, StopIteration, ValueError, IndexError):
        pass
    st["cpu"] = [(t, u) for t, u in st["cpu"] if now - t <= AFTER_HARNESS_WINDOW_S]
    span = st["cpu"][-1][0] - st["cpu"][0][0] if len(st["cpu"]) > 1 else 0.0
    cores = (st["cpu"][-1][1] - st["cpu"][0][1]) / 1e6 / span if span >= 10.0 else None
    if connected or cores is None or cores >= AFTER_HARNESS_IDLE_CORES:
        st["idle_since"] = None
        return False
    if st["idle_since"] is None:
        st["idle_since"] = now
        return False
    if now - st["idle_since"] < AFTER_HARNESS_GRACE_S:
        return False
    held = []
    for pid in procs:
        try:
            held.append({"pid": pid, "cmd": open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ")
                         .decode(errors="replace").strip()[:300]})
        except OSError:
            pass
    # The scope: on an open pool, or a fixed job list with no task work remaining,
    # the rule ends the box. On a fixed list with work remaining and a launch-like process alive, the box stays up as
    # before (an agent that answered early but left a detached queue still gets its jobs run and scored, as the scorer
    # has measured all campaign); the hold is labelled. The scorer's work_remaining, read as the runner's lifecycle does.
    kind = task_work_state(out, st)
    # A process naming a launcher or one of the task's job scripts is recorded, not exempted: on c3-11 the process
    # left was the agent's own launch queue (`while pgrep ... ; do sleep 20; done` then torchrun lines) whose wait never
    # ended, so exempting launch-like lines would have kept exactly that box up (the study's rule, 23:49Z: only a
    # task job's connected ranks keep the box). Analysis counts these as launches the agent left queued.
    if "launch_tokens" not in st:
        try:
            jobs = json.load(open(os.path.join(out, "task.json"))).get("jobs") or []
        except (OSError, ValueError):
            jobs = []
        toks = {"torchrun", "accelerate launch", "deepspeed", "srun", "sbatch"}
        for j in jobs:
            for w in str(j.get("launch") or "").split():
                if w.endswith((".py", ".sh")):
                    toks.add(os.path.basename(w))
        st["launch_tokens"] = sorted(toks)
    status["held_after_harness_launch_like"] = [h["pid"] for h in held if any(t in h["cmd"] for t in st["launch_tokens"])]
    status["held_after_harness_by"] = held
    status["held_after_harness_work"] = kind
    if kind == "fixed_list_work_remaining" and status["held_after_harness_launch_like"]:
        status["held_after_harness_s"] = round(now - (status.get("harness_end_wall") or now), 1)
        if not st.get("kept_logged"):
            st["kept_logged"] = True
            log("idle after the harness with task work remaining and a launch-like process alive; the box stays up")
        return False
    status["held_after_harness_s"] = round(now - (status.get("harness_end_wall") or now), 1)
    status["after_harness_rule"] = {"grace_s": AFTER_HARNESS_GRACE_S, "idle_cores": AFTER_HARNESS_IDLE_CORES,
                                    "window_s": AFTER_HARNESS_WINDOW_S}
    log(f"no task job connected and the box idle for {AFTER_HARNESS_GRACE_S:g} s after the harness exited; "
        f"ending the run (left: {', '.join(h['cmd'][:60] for h in held) or 'nothing readable'})")
    return True


def box_ns_pids(procs):
    """The box-namespace pids of the given host pids (NSpid's last field), or None when none can be read."""
    out = set()
    for pid in procs or []:
        try:
            for line in open(f"/proc/{pid}/status"):
                if line.startswith("NSpid:"):
                    out.add(int(line.split()[-1]))
                    break
        except (OSError, ValueError, IndexError):
            continue
    return out if out or not procs else None


def task_work_state(out, st):
    """"open_pool", "fixed_list_done" or "fixed_list_work_remaining", by the scorer's own work_remaining on the run's
    occupancy log (runner/score/idle.py, as the runner's lifecycle.py reads it). Unreadable counts as work remaining,
    so a doubt keeps the box up as before."""
    try:
        if "score" not in st:
            sd = os.path.join(EXP, "runner", "score")
            mods = {}
            for name in ("score_run", "idle"):
                spec = importlib.util.spec_from_file_location(f"p006_run_one_{name}", os.path.join(sd, f"{name}.py"))
                m = importlib.util.module_from_spec(spec)
                sys.path.insert(0, sd)
                try:
                    spec.loader.exec_module(m)
                finally:
                    sys.path.remove(sd)
                mods[name] = m
            st["score"] = mods
        task = json.load(open(os.path.join(out, "task.json")))
        ev = st["score"]["score_run"].load_events(os.path.join(out, "occupancy.jsonl"))
        rem = st["score"]["idle"].work_remaining(ev, task)
        if rem.get("open_pool"):
            return "open_pool"
        if rem.get("remaining") == 0 and rem.get("all_done_wall") is not None:
            return "fixed_list_done"
    except Exception:
        pass
    return "fixed_list_work_remaining"


def make_lifecycle(out):
    """The runner's Lifecycle, given the run directory when it takes one (since 2026-10-21 it closes stdin once every
    task job has completed and the turn has ended, even with background tasks alive: the study's ruling (b')
    on a self-matching pgrep poll that held a session to its horizon). An older lifecycle.py takes no argument."""
    try:
        return LIFECYCLE.Lifecycle(run_dir=out)
    except TypeError:
        return LIFECYCLE.Lifecycle()


def write_codex(state, lines):
    """Lines for codex app-server's stdin: written to the FIFO, and to box/codex-rpc.jsonl with the session's times."""
    cs, rpc = state["codex"], state["codex_rpc"]
    for t, line in cs.sent[state["codex_sent"]:]:
        try:
            rpc.write(json.dumps({"t": t, "dir": "out", "msg": json.loads(line)}) + "\n")
        except ValueError:
            rpc.write(json.dumps({"t": t, "dir": "out-raw", "msg": line}) + "\n")
    state["codex_sent"] = len(cs.sent)
    data = "".join(lines).encode()
    # Written here, in order, by the one thread that also closes the fd: a writer thread per message could reorder
    # them, or write after close into a reused descriptor. The messages are small and Codex reads its stdin
    # continuously; the largest, turn/start with the brief, is sent only once Codex has answered thread/start.
    while data and state.get("stdin_fd") is not None:
        data = data[os.write(state["stdin_fd"], data):]


def feed_codex(boxd, state, status):
    """As feed_lifecycle, for codex app-server: every stdout line goes to the session (and to codex-rpc.jsonl with the
    time it was read), whatever the session answers goes back on stdin, and stdin closes on the session's rule."""
    cs, rpc = state["codex"], state["codex_rpc"]
    try:
        with open(os.path.join(boxd, "harness.stdout"), "rb") as fh:
            fh.seek(state["stdout_off"])
            chunk = fh.read()
    except OSError:
        chunk = b""
    if chunk:
        state["stdout_off"] += len(chunk)
        *lines, state["stdout_buf"] = (state["stdout_buf"] + chunk).split(b"\n")
        now = time.time()
        for raw in lines:
            line = raw.decode("utf-8", "replace")
            try:
                rpc.write(json.dumps({"t": now, "dir": "in", "msg": json.loads(line)}) + "\n")
            except ValueError:
                if line.strip():
                    rpc.write(json.dumps({"t": now, "dir": "in-raw", "msg": line}) + "\n")
            write_codex(state, cs.feed(line, now))
    if getattr(cs, "resume", False):
        # every tick, output or not: a resume is due by time, not by a message, and without this call a
        # capacity-failed run neither resumes nor closes before the horizon
        write_codex(state, cs.poll())
    status["harness_results"] = cs.results
    failed = getattr(cs, "failed", None) or next((f"codex: {m} failed" for (_, m, _) in cs.errors
                                                  if m in ("initialize", "thread/start", "turn/start")), None)
    if failed:
        status["codex_failed"] = failed  # a harness error, not an agent result (final_status reports it)
        close_stdin(state, status, failed)
    elif cs.should_close():
        close_stdin(state, status, f"codex: root turn complete, no other thread active and no command running "
                                   f"for {cs.grace_s:g} s")


def feed_lifecycle(boxd, state, status):
    """Read what Claude Code has written since the last call, feed each complete line to the lifecycle, and close the
    harness's stdin when it says so."""
    lc = state["lifecycle"]
    try:
        with open(os.path.join(boxd, "harness.stdout"), "rb") as fh:
            fh.seek(state["stdout_off"])
            chunk = fh.read()
    except OSError:
        chunk = b""
    if chunk:
        state["stdout_off"] += len(chunk)
        buf = state["stdout_buf"] + chunk
        *lines, state["stdout_buf"] = buf.split(b"\n")
        now = time.time()
        for line in lines:
            lc.feed(line.decode("utf-8", "replace"), now)
    status["harness_results"] = lc.results
    if lc.should_close():
        why = getattr(lc, "close_reason", None) or f"lifecycle: a result with no background task alive for {lc.grace_s:g} s"
        close_stdin(state, status, why)
        if why.startswith(WORK_DONE_PREFIX):
            # Closing stdin does not end a Claude Code session that still holds a background task (the runner's
            # probe P006-HELDOPEN-DONE-PROBE-20261021T151419Z: stdin closed at 912 s, claude waited to the horizon).
            # Every task job is done, so the session is ended: SIGTERM after END_TERM_S, SIGKILL after END_KILL_S.
            state["end_harness_from"] = time.time()


WORK_DONE_PREFIX = "lifecycle: every task job completed"
END_TERM_S, END_KILL_S = 10.0, 30.0


def harness_pids(cid, session_id):
    """Host pids of the harness itself: the claude process whose argv carries this run's session id."""
    out = []
    try:
        pids = [int(x) for x in open(os.path.join(cgroup_path(cid), "cgroup.procs")).read().split()]
    except OSError:
        return out
    for pid in pids:
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if argv and os.path.basename(argv[0].decode(errors="replace")) == "claude" and session_id.encode() in argv:
            out.append(pid)
    return out


def end_held_harness(state, status, boxd):
    """After a work-done close: signal the harness if it is still running past the grace (see feed_lifecycle)."""
    t0 = state.get("end_harness_from")
    if t0 is None or os.path.exists(os.path.join(boxd, "harness_exit")):
        return
    waited = time.time() - t0
    for sig, after, key in ((signal.SIGTERM, END_TERM_S, "TERM"), (signal.SIGKILL, END_KILL_S, "KILL")):
        if waited >= after and key not in status.get("harness_signalled", {}):
            sent = []
            for pid in harness_pids(state["cid"], state["session_id"]):
                try:
                    os.kill(pid, sig)
                    sent.append(pid)
                except OSError:
                    pass
            status.setdefault("harness_signalled", {})[key] = {"t": time.time(), "pids": sent}
            status["harness_end_reason"] = "lifecycle: work done, background task held the session"
            if "work_done_at_end" not in status:
                # the scorer's own check again, at the moment the harness is ended (runner, 2026-10-21: a close that
                # was wrong must not read `completed`; G-passbar-sonnet-L0-v2 lost four jobs to one)
                wd = getattr(state.get("lifecycle"), "work_done", None)
                try:
                    status["work_done_at_end"] = bool(wd()) if wd else None
                except Exception:
                    status["work_done_at_end"] = False
            log(f"sent SIG{key} to the harness (host pids {sent}): every task job is done and a background task "
                "held the session")


def close_stdin(state, status, why):
    if state.get("stdin_fd") is None:
        return
    try:
        os.close(state["stdin_fd"])
    except OSError:
        pass
    state["stdin_fd"] = None
    status["stdin_closed_wall"], status["stdin_close_reason"] = time.time(), why
    log(f"closed the harness's stdin ({why})")


INFERENCE_LOST = ("Can't reach the API server", "EAI_AGAIN", "ECONNREFUSED", "ECONNRESET", "ETIMEDOUT",
                  "fetch failed", "socket hang up")


def harness_result(boxd):
    """The last `result` record of Claude Code's stream-json output, or None (the scripted harness has none).
    Its `subtype` can read "success" while `is_error` is true, as when the model was unreachable."""
    last = None
    try:
        for line in open(os.path.join(boxd, "harness.stdout"), errors="replace"):
            if '"type":"result"' in line or '"type": "result"' in line:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("type") == "result":
                    last = r
    except OSError:
        return None
    if last is None:
        return None
    return {k: last.get(k) for k in ("subtype", "is_error", "num_turns", "api_error_status", "duration_ms",
                                     "total_cost_usd")} | {"result": str(last.get("result") or "")[:500]}


def final_status(boxd, status):
    if status.get("codex_failed"):
        return "harness-error", status["codex_failed"]
    if os.path.exists(os.path.join(boxd, "init_error")):
        return "harness-error", open(os.path.join(boxd, "init_error")).read().strip()
    hr = harness_result(boxd)
    if hr is not None:
        status["harness_result"] = hr
        if hr.get("is_error"):
            text = hr.get("result") or ""
            if any(m in text for m in INFERENCE_LOST):
                return "harness-error", f"inference path lost: {text[:300]}"
            return "harness-error", f"harness reported an error: {text[:300]}"
    if status.get("harness_end_reason"):
        # run_one ended the harness after the lifecycle's work-done close (end_held_harness); its exit code is the
        # signal's or whatever claude returns on SIGTERM, and is kept in harness_exit_code. `completed` only when the
        # scorer's check still said done when the harness was ended.
        if status.get("work_done_at_end") is False:
            return "harness-error", ("ended by the work-done close while task work remained "
                                     f"(harness exit {status['harness_exit_code']})")
        return "completed", f"{status['harness_end_reason']} (harness exit {status['harness_exit_code']})"
    if status["harness_exit_code"] != 0:
        tail = ""
        try:
            tail = open(os.path.join(boxd, "harness.stderr"), errors="replace").read()[-800:]
        except OSError:
            pass
        return "harness-error", f"harness exit {status['harness_exit_code']}: {tail}"
    return "completed", "harness exited 0"


WORKSPACE_FILE_CAP = 50 << 20  # bytes; the campaign's rule for what a run keeps (campaign/AGENTS.md)
# Weights, optimizer states and checkpoints: listed, never copied, whatever their size.
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".msgpack", ".gguf", ".npz", ".onnx", ".pkl")
WEIGHT_NAMES = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin", "pytorch_model.bin")
# Not the agent's work: the staged inputs and the harness's own state, copied or recorded elsewhere.
SKIP_DIRS = {".cache", ".claude", ".config", ".local", ".npm", ".git", "__pycache__", ".ipynb_checkpoints"}


def capture_workspace(home, tmp, out, staged_project):
    """Gap G1: keep what the agent wrote. Every regular file under the agent's home and /tmp up to WORKSPACE_FILE_CAP
    is copied to <out>/workspace/{home,tmp}/..., except weights, optimizer and checkpoint binaries, which are listed
    with their sizes in workspace/manifest.jsonl like every other file (copied: true/false, and why not). Files
    identical to the staged task tree (same relative path, size and mtime) are the inputs, not the agent's work,
    and are listed, not copied. Every regular file under project/data/ also gets its sha256 in the manifest, copied or
    not. Never raises: a failure is recorded in the summary and teardown goes on."""
    summary = {"copied": 0, "copied_bytes": 0, "listed_only": 0, "skipped_dirs": 0, "errors": 0}
    wdir = os.path.join(out, "workspace")
    os.makedirs(wdir, exist_ok=True)
    staged = {}
    if staged_project and os.path.isdir(staged_project):
        for root, dirs, files in os.walk(staged_project):
            for f in files:
                p = os.path.join(root, f)
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                staged[os.path.relpath(p, staged_project)] = (st.st_size, int(st.st_mtime))
    try:
        mf = open(os.path.join(wdir, "manifest.jsonl"), "w")
    except OSError as e:
        return {**summary, "errors": 1, "error": str(e)}
    with mf:
        for label, base in (("home", home), ("tmp", tmp)):
            if not os.path.isdir(base):
                continue
            for root, dirs, files in os.walk(base):
                keep = []
                for d in sorted(dirs):
                    if d in SKIP_DIRS or os.path.islink(os.path.join(root, d)):
                        summary["skipped_dirs"] += 1
                    else:
                        keep.append(d)
                dirs[:] = keep
                for f in sorted(files):
                    p = os.path.join(root, f)
                    rel = os.path.relpath(p, base)
                    rec = {"tree": label, "path": rel}
                    try:
                        st = os.lstat(p)
                    except OSError as e:
                        rec.update(copied=False, why=f"stat: {e}"); summary["errors"] += 1
                        mf.write(json.dumps(rec) + "\n"); continue
                    rec.update(size=st.st_size, mtime=int(st.st_mtime))
                    # Content hash of every regular file under the task's data directory, copied or not: the record of
                    # whether the data a job read matched the recorded runs' shards, which gpu-mock serves whatever is
                    # on disk.
                    if label == "home" and rel.startswith("project/data/") and os.path.isfile(p) and not os.path.islink(p):
                        t_h = time.time()
                        try:
                            h = hashlib.sha256()
                            with open(p, "rb") as fh:
                                for chunk in iter(lambda: fh.read(8 << 20), b""):
                                    h.update(chunk)
                            rec["sha256"] = h.hexdigest()
                        except OSError as e:
                            rec["sha256_error"] = str(e)[:200]; summary["errors"] += 1
                        summary["hashed"] = summary.get("hashed", 0) + 1
                        summary["hash_seconds"] = round(summary.get("hash_seconds", 0) + time.time() - t_h, 2)
                    why = None
                    if not os.path.isfile(p) or os.path.islink(p):
                        why = "not a regular file"
                    elif f in WEIGHT_NAMES or f.lower().endswith(WEIGHT_EXT) or "/checkpoint-" in "/" + rel:
                        why = "weights or checkpoint"
                    elif st.st_size > WORKSPACE_FILE_CAP:
                        why = f"over {WORKSPACE_FILE_CAP >> 20} MB"
                    elif label == "home" and rel.startswith("project/") and \
                            staged.get(rel[len("project/"):]) == (st.st_size, int(st.st_mtime)):
                        why = "staged input, unchanged"
                    if why:
                        rec.update(copied=False, why=why); summary["listed_only"] += 1
                    else:
                        dst = os.path.join(wdir, label, rel)
                        try:
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            shutil.copy2(p, dst)
                            rec["copied"] = True; summary["copied"] += 1; summary["copied_bytes"] += st.st_size
                        except OSError as e:
                            rec.update(copied=False, why=f"copy: {e}"); summary["errors"] += 1
                    mf.write(json.dumps(rec) + "\n")
    return summary


def finish(a, out, boxd, state, status, names):
    close_stdin(state, status, "teardown")
    if state.get("codex") is not None:
        cs = state["codex"]
        others, cmds = cs.busy()
        status["codex"] = {"root_thread": cs.root, "root_turns": cs.results, "errors": [list(e) for e in cs.errors],
                           "other_threads_active": others, "open_commands": cmds}
        if getattr(cs, "resume", False):
            rs = cs.resumes
            status["codex"]["resumed_after_capacity"] = {"count": len(rs), "total_gap_s": round(sum(r["gap_s"] for r in rs), 1),
                                                         "resumes": rs, "pending_at_teardown": cs.resume_due is not None}
        if state.get("codex_rpc"):
            state["codex_rpc"].close()
    if state["started"]:
        status["teardown_wall"] = time.time()
        # The runner enforces the budget: it creates <stop-dir>/<tag>, and the forwarder's gate then
        # answers that run with 429. Whatever happened after that, the run ended because of the budget.
        if os.path.exists(os.path.join(a.stop_dir, a.cost_tag)):
            status["status"], status["reason"] = "killed-on-budget", "budget crossed: the runner's stop file exists"
    for n in ("box", "edge", "gpu"):
        docker("stop", "-t", "10", names[n], check=False)
    if state["sampler"]:
        try:
            state["sampler"].wait(timeout=30)
        except subprocess.TimeoutExpired:
            state["sampler"].terminate()
    for n in ("box", "edge", "gpu"):
        r = docker("logs", names[n], check=False)
        if r.returncode == 0 and (r.stdout or r.stderr):
            open(os.path.join(out, f"container-{n}.log"), "w").write((r.stdout or "") + (r.stderr or ""))
        docker("rm", "-f", names[n], check=False)
    docker("network", "rm", names["net"], check=False)
    docker("volume", "rm", names["vol"], check=False)
    tdir = os.path.join(out, "transcript")
    os.makedirs(tdir, exist_ok=True)
    for src, dst in (("harness.stdout", "stream.jsonl"), ("harness.stderr", "stderr.log")):
        if os.path.exists(os.path.join(boxd, src)):
            shutil.copy2(os.path.join(boxd, src), os.path.join(tdir, dst))
    if os.path.exists(os.path.join(boxd, "sched-events.jsonl")):
        # the scheduler's event log, where the runner's reader looks first
        shutil.copy2(os.path.join(boxd, "sched-events.jsonl"), os.path.join(out, "scheduler.jsonl"))
    if os.path.exists(os.path.join(out, "edge", "link.key")):
        os.remove(os.path.join(out, "edge", "link.key"))
    if os.path.exists(os.path.join(out, "edge", "edge.jsonl")):
        os.makedirs(os.path.join(out, "forwarder"), exist_ok=True)
        shutil.copy2(os.path.join(out, "edge", "edge.jsonl"), os.path.join(out, "forwarder", "edge.jsonl"))
    mnt = state["mnt"]
    staged_project = state.get("staged_project")
    if mnt and os.path.ismount(mnt):
        home = os.path.join(mnt, "home")
        proj = os.path.join(home, ".claude", "projects")
        if os.path.isdir(proj):
            shutil.copytree(proj, os.path.join(tdir, "session"), symlinks=True, dirs_exist_ok=True)
        # Codex's rollout files, the root's and each subagent's: the transcript the runner scores for --harness codex
        #.
        csess = os.path.join(home, ".codex", "sessions")
        if os.path.isdir(csess):
            shutil.copytree(csess, os.path.join(tdir, "codex-home", "sessions"), symlinks=True, dirs_exist_ok=True)
        # Grok Build's sessions (updates.jsonl is its scoring transcript) and process log; OpenCode's session database
        # (the only record that shows a tool call still running) and log (port notes, experiments/harnesses/port/)
        for src, dst in ((os.path.join(home, ".grok", "sessions"), os.path.join(tdir, "grok-home", "sessions")),
                         (os.path.join(home, ".grok", "logs"), os.path.join(tdir, "grok-home", "logs")),
                         (os.path.join(home, ".local", "share", "opencode"), os.path.join(tdir, "opencode-data"))):
            if os.path.isdir(src):
                shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("bin", "*.lock"))
        with open(os.path.join(out, "workspace.lst"), "w") as fh:
            for root, dirs, files in os.walk(os.path.join(home, "project")):
                dirs.sort()
                for f in sorted(files):
                    p = os.path.join(root, f)
                    try:
                        st = os.lstat(p)
                    except OSError:
                        continue
                    fh.write(f"{st.st_size}\t{st.st_blocks * 512}\t{os.path.relpath(p, home)}\n")
        jl = os.path.join(home, "project", "logs")
        if os.path.isdir(jl):  # kept where older readers look (runner's overshoot.py); workspace/ holds the same files
            dst = os.path.join(out, "agent-logs")
            os.makedirs(dst, exist_ok=True)
            for f in sorted(os.listdir(jl)):
                p = os.path.join(jl, f)
                if os.path.isfile(p) and os.path.getsize(p) <= WORKSPACE_FILE_CAP:
                    shutil.copy2(p, os.path.join(dst, f))
        status["workspace_capture"] = capture_workspace(home, os.path.join(mnt, "tmp"), out, staged_project)
        # Which checkpoint survived is read from the files, and the disk image goes below: digest the files the agent's
        # ensemble index names now, for a task that asks (capture.weights_index; nothing else changes without it).
        try:
            wspec = (json.load(open(os.path.join(out, "task.json"))).get("capture") or {}).get("weights_index")
        except (OSError, ValueError):
            wspec = None
        if wspec:
            os.makedirs(os.path.join(out, "workspace"), exist_ok=True)
            status["weights_digest"] = WEIGHTS_DIGEST.capture(home, os.path.join(mnt, "tmp"), wspec,
                                                              os.path.join(out, "workspace", "weights-digests.jsonl"), HOME,
                                                              os.path.relpath(PROJECT, HOME))
        u = shutil.disk_usage(mnt)
        status["disk_used_bytes_at_end"] = u.used
        sh(["umount", mnt], check=False)
    if mnt and not os.path.ismount(mnt):
        try:
            os.rmdir(mnt)
        except OSError:
            pass
    if state["img"] and os.path.exists(state["img"]):
        os.remove(state["img"])
    # The box directory comes home: <run>/box was a link to the neutral directory the box mounted.
    link = os.path.join(out, "box")
    if os.path.islink(link) and os.path.realpath(link) == os.path.realpath(boxd) and os.path.isdir(boxd):
        os.unlink(link)
        shutil.move(boxd, link)
    shutil.rmtree(boxd + ".s", ignore_errors=True)
    shutil.rmtree(boxd + ".k", ignore_errors=True)
    shutil.rmtree(boxd + ".l", ignore_errors=True)


def write_json(p, obj):
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")
    os.replace(tmp, p)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
