# ComputeWise

**gpu-harness** is a plugin for Claude Code and Codex. When your agent runs several GPU jobs to reach an answer, the
harness runs the jobs for it. It starts the jobs that decide the answer first, refills a GPU as soon as a job ends,
and stops a candidate's jobs as soon as the agent rejects it under your rule, so the answer comes sooner. It installs with
plugin commands; there is nothing to clone or `pip install`. **ComputeWise** is the research project behind it: a
benchmark that measures how much GPU time AI agents waste, with the paper's code and data.

| You want to… | Go to |
|---|---|
| **Add the gpu-harness to your own Claude Code or Codex** | [Install](#install) |
| **See it work in five minutes, with or without GPUs** | [The demo](harness/demo/README.md) |
| **Reproduce or extend the paper's benchmark** | [The benchmark](#the-benchmark) |

## Use the gpu-harness in your own agent

When a coding agent runs a sweep, an ablation, an evaluation table, or a "which of these setups passes the bar"
study, it tends to leave GPUs idle between jobs and to keep them busy with runs the answer never needed. With the
gpu-harness, the agent still plans the work and makes every judgement. The harness runs the jobs: the deciding ones
first, a new job on a GPU the moment one ends, and none for a candidate the agent has rejected under your rule.

### How much it helps

The paper measured this on a simulated node that replays recorded H100 jobs. Each agent's time to its answer is
compared with a reference schedule of the same work. That schedule follows the task's rule, knows each job's length,
and is within 10% of the best possible schedule. "Time over the reference" is the extra time: an agent that takes
14.4 hours where the reference takes 10 has 4.4 hours over it.

- **With no help,** the typical agent run took 1.44 times as long as the reference.
- **Claude Haiku 5.5, Sonnet 5.5 and Opus 5.5 with the harness:**
  - 25 of their 26 model-and-task pairs improved: the three models on nine tasks, less one pair with no run without
    help;
  - their summed time over the reference fell by about 90%;
  - pairs taking at least 1.25 times as long as the reference went from 25 of 26 to 3.
- **Cost against saving:** a harness run cost $0.17 to $0.42 in model tokens with Haiku, $2.28 to $7.16 with Sonnet
  and $4.60 to $14.89 with Opus. It saved 1.2 to 57 GPU-hours, $8 to $394 at $6.88 per H100-hour, on 25 of the 26
  pairs.
- **The Codex edition:**
  - GPT-6 Astra improved on 7 of 9 tasks and cut its time over the reference by 63%;
  - GPT-6.1 Sol was run without help on fewer tasks. It improved on 3 of the 4 tasks with such a run, and cut its
    time over the reference by 48%.

The harness starts each job as an ordinary process with `CUDA_VISIBLE_DEVICES` set. In the paper those processes ran
on the simulated node. This release has been tried there and on pretend GPUs, not yet on real GPUs.

All nine of the paper's tasks can be built and run from this release; [The benchmark](#the-benchmark) says how. The
[dataset](https://huggingface.co/datasets/Salesforce/computewise) holds the scored runs on the modded-nanogpt tasks.

### Is it a fit?

- **Good fit:** several jobs on one machine that lead to one answer, each job using one GPU. For example, seeds of a
  few configs, a sweep with an adoption rule, an evaluation table, or a screen-then-confirm study. CPU-only jobs, such
  as data preparation, run beside the GPU jobs.
- **Biggest gain:** jobs of minutes to hours, and a rule that can rule a candidate out before all of its runs finish.
  If you ask for every number, nothing is cut, and the harness only orders and dispatches the jobs.
- **Not supported:** jobs that need several GPUs (`torchrun`, DDP), several machines, and submitting through
  `sbatch`. The agent runs on the machine whose GPUs it uses.
- **Clusters:** inside a SLURM job, start the agent on the allocated compute node, for example in an `srun --pty bash`
  shell, where the GPUs are local. This should work, but has not been tested; `salloc` alone leaves you on the login
  node. If SLURM sets `CUDA_VISIBLE_DEVICES` to GPU UUIDs rather than numbers, the harness cannot read it.
- **Not worth it:** a single run, or a handful of jobs of a few minutes each.
- **Needs:**
  - Linux with NVIDIA GPUs, which the harness finds through `CUDA_VISIBLE_DEVICES` or `nvidia-smi`. To try the
    harness on any machine, macOS included, use pretend GPUs. These are numbered slots with no device behind them,
    which you ask for in the request, as the demo does. They suit the demo and CPU-only jobs, not real training. WSL2 has not been
    tested, and native Windows is not supported.
  - Python 3.9 or later.
  - Claude Code or Codex. The paper's runs used Claude Code 2.1.303 and Codex 0.163.0. Installation has been checked
    on Claude Code 2.1.322 and Codex 0.167.0-alpha.4.1. If `claude plugin` or `codex plugin` is an unknown command,
    update the CLI, or copy the skill by hand (see Install).
- **How it works:** the harness is a command, `gpuh`, that the agent runs. The agent groups the candidates into at
  most four branches (three in Codex). Each branch has a teammate, a sub-agent that judges its candidates, so a sweep
  of any size works.

### Install

**Claude Code:**
1. Add the plugin:

   ```bash
   claude plugin marketplace add SalesforceAIResearch/computewise
   claude plugin install gpu-harness@computewise
   ```

2. Turn on agent teams. The skill gives each teammate a name, and messages it by name when a branch stops listening.
   Without agent teams, Claude Code gives teammates no names; checked on 2.1.322, a message to a named teammate then
   fails. A run can still finish without them, but a stalled branch cannot be woken. This block adds the setting to
   `~/.claude/settings.json` and keeps everything else in the file:

   ```bash
   python3 - <<'EOF'
   import json, os
   p = os.path.expanduser("~/.claude/settings.json")
   d = json.load(open(p)) if os.path.exists(p) else {}
   d.setdefault("env", {})["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] = "1"
   os.makedirs(os.path.dirname(p), exist_ok=True)
   json.dump(d, open(p, "w"), indent=2)
   print("agent teams on in", p)
   EOF
   ```

   To do it by hand instead, add `"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"` to the `env` object in that file,
   or start Claude Code with `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 claude`.
3. Restart any Claude Code session that was already open. Check that `claude plugin list` shows
   `gpu-harness@computewise` as enabled.

**Codex:**
1. Add the plugin:

   ```bash
   codex plugin marketplace add SalesforceAIResearch/computewise
   codex plugin add gpu-harness@computewise
   ```

2. Check that sub-agents are on: `codex features list | grep -w multi_agent` should show `true`. If it doesn't, run
   `codex features enable multi_agent`. `codex plugin list | grep gpu-harness` shows the plugin; the full list can take
   a minute to print.
3. Start Codex from your project with `GPUH_HOME` pointing inside it:

   ```bash
   cd your-project
   export GPUH_HOME="$PWD/.gpuh"
   codex
   ```

   The harness keeps its run record in `~/.gpuh` unless told otherwise. Codex's default sandbox lets the agent write
   only inside the project, so without this, `gpuh start` fails with `PermissionError` on `~/.gpuh`. Keep the `export`
   in the shell you start Codex from, whatever flags you add. You may want `.gpuh/` in your `.gitignore`.

<details>
<summary><b>Without the plugin system</b> (copy the skill yourself)</summary>

```bash
git clone https://github.com/SalesforceAIResearch/computewise
mkdir -p ~/.claude/skills && cp -r computewise/harness/skills/gpu-harness ~/.claude/skills/        # Claude Code
mkdir -p ~/.codex/skills && cp -r computewise/harness/codex-skills/gpu-harness ~/.codex/skills/    # Codex
```

Run only the line for your tool. The copied skill is called `/gpu-harness` in Claude Code, without the
`gpu-harness:` plugin prefix, and `$gpu-harness` in Codex. The rest of Install still applies: agent teams in Claude
Code; in Codex, `multi_agent` and `GPUH_HOME`. Its `gpuh` command is `~/.claude/skills/gpu-harness/bin/gpuh` or `~/.codex/skills/gpu-harness/bin/gpuh`.
</details>

### Before you start a run

- **Environment:** activate the environment your jobs need (a venv, conda) before starting the agent. The jobs inherit
  it.
- **GPUs:** the harness treats the GPUs it can see as its own. On a shared machine, start the agent with only the free
  ones visible, for example `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 claude`. `CUDA_DEVICE_ORDER` makes
  CUDA number the GPUs the way `nvidia-smi` does, which on a machine with mixed GPU models it otherwise may not. The harness then uses physical GPUs 2 and 3, and gives
  each job one of them. In your request, either don't name GPUs, or name them by their `nvidia-smi` numbers (2 and
  3 here, not 0 and 1). `CUDA_VISIBLE_DEVICES` must hold GPU numbers, not UUIDs.
- **Remote machines:** start the agent inside `tmux` or `screen`, so a dropped connection doesn't end the session.
- **One record per project:** `gpuh` commands act on the latest run under `GPUH_HOME`, by default `~/.gpuh`, shared by
  every project on the machine. To keep each project's runs apart, `export GPUH_HOME="$PWD/.gpuh"` before starting
  the agent, in Claude Code as in Codex, and in the terminal you inspect from.
- **Live output:** the harness reads progress and results from each job's log as it is written. Python buffers
  output written to a file, so run your scripts with `python -u`, set `PYTHONUNBUFFERED=1`, or print with
  `flush=True`. Otherwise a rule checked at step 500 sees step 500 late.
- **Codex's sandbox and real jobs:** beyond the run record, Codex's default sandbox turns the network off and blocks
  writes outside the project and `/tmp`, and jobs started by the harness inherit it. Running CUDA jobs inside it has not been tested, and on Linux it may block the GPU
  device files as well as caches under `$HOME` (`~/.cache`, `~/.triton`, `~/.nv`, W&B). For real GPU jobs, keep
  `GPUH_HOME` exported as in Install and start `codex --sandbox danger-full-access` on a machine you trust, as the
  paper's runs did inside an isolated container. Whatever the setting, make your scripts fail when
  `torch.cuda.is_available()` is false, so a job that cannot reach its GPU stops instead of training on the CPU.
- **Claude Code permissions:** the agent and its teammates run `gpuh` and shell commands many times. In the default
  permission mode, each new kind of command asks for approval. Approve `gpuh` with the option not to ask again; for a
  run you will leave unattended, allow its commands in your permission settings, or use a permission mode that allows
  them on a machine you trust. The paper's runs used `--permission-mode bypassPermissions`, inside an isolated
  container.

### Run it

Begin your request with the skill's name, so the agent uses it even on a small task:
- **Claude Code plugin:** `/gpu-harness:gpu-harness`, which is the plugin name, a colon, then the skill name;
- **copied skill in Claude Code:** `/gpu-harness`;
- **Codex:** `$gpu-harness`.

Then say three things:
- how to run a job, and what line it prints;
- what decides the answer, and when a candidate is out;
- which GPUs to use, by their `nvidia-smi` numbers, or none to use all the visible ones. To try the harness on a
  machine without GPUs, say "use pretend GPUs 0 and 1, listed in the plan".

The harness does not judge results: a candidate's jobs stop only when the agent rejects it under your rule, usually
seconds after the deciding number appears. A rejection is final, so a noisy early check can drop a config that would
have won. Make the early rule as cautious as you can afford: a later step, more than one seed, or a margin:

> /gpu-harness:gpu-harness `python -u train.py --config configs/<name>.yaml --seed <n>` trains one config and prints
> `step N/3000 val_loss X`. Run the baseline and the four configs in `configs/` with 3 seeds each on GPUs 0-3.
> Adopt a config if its mean val_loss at step 3000 is at least 0.02 below the baseline's. Drop a config as soon as
> its val_loss at step 1000 is above the baseline's on two seeds.

Your scripts run unchanged, apart from live output. Each job gets its GPU through `CUDA_VISIBLE_DEVICES`, so a script
must use the default CUDA device rather than a hard-coded one. The rule's numbers must be printed to stdout or stderr,
not only logged to W&B or TensorBoard.

**What happens:**
1. The agent writes a `plan.json` in your project: the candidates, their jobs, and the checks that gate them.
2. It starts the harness and one teammate per branch.
3. The harness runs the jobs, one per GPU. A candidate's first job is usually a check, and its other jobs wait until
   the check passes.
4. Each teammate judges its candidates from short events, and rejects one as soon as your rule allows.
5. The agent gives the answer.

Between events the agent mostly waits. In headless `claude -p` you see many "Waiting." turns; read the last one.

### Watch, stop and inspect a run

Open a second terminal, `cd` to the project, and set up the `gpuh` command there. It lives in the installed plugin,
not on your `PATH`.

For the Claude Code plugin (add `export GPUH_HOME="$PWD/.gpuh"` if you started Claude Code with it):

```bash
alias gpuh=~/.claude/plugins/cache/computewise/gpu-harness/0.1.0/skills/gpu-harness/bin/gpuh
```

For the Codex plugin:

```bash
alias gpuh=~/.codex/plugins/cache/computewise/gpu-harness/0.1.0/codex-skills/gpu-harness/bin/gpuh
export GPUH_HOME="$PWD/.gpuh"
```

For a copied skill, use `alias gpuh=~/.claude/skills/gpu-harness/bin/gpuh` or
`alias gpuh=~/.codex/skills/gpu-harness/bin/gpuh` (and, for Codex, the same `export`).

The paths assume the default config directories and version 0.1.0. Then:

```bash
gpuh status -v    # GPUs, each candidate (open, rejected or decided), the queue in start order and why each job waits
gpuh results      # every job's outcome and each candidate's verdict
gpuh export       # writes timeline.html, every job on every GPU, into the run directory
gpuh finish       # stop: cancels every running and queued job and ends the run
```

**Where runs are kept:**
- Each run has a directory, `$GPUH_HOME/runs/<start time>/` (by default `~/.gpuh/runs/`), holding the plan, every
  job's log and the scheduler's record.
- `gpuh` commands act on the latest run under `GPUH_HOME`.
- When the agent starts the harness, it prints the run directory (`harness up ...; run dir ...`). If that directory
  is elsewhere, add `--dir <run dir>` to any `gpuh` command.

The jobs and the scheduler run in their own processes, so closing the session does not stop them, but then no one is
left to judge the results. Use `gpuh finish` to stop them.

**If the agent sits idle after every job has finished,** a branch has probably lost its teammate. In Claude Code,
check that agent teams are on; in Codex, that `multi_agent` is on. Otherwise, tell the agent to read `gpuh status` and
answer.

### Update or remove

Every copy of the plugin says version 0.1.0, so `update` may not see a newer one. To update, refresh and reinstall:
- **Claude Code:** `claude plugin marketplace update computewise`, then uninstall and install again.
- **Codex:** `codex plugin marketplace upgrade computewise`, then remove and add again.

To remove the plugin, run `claude plugin uninstall gpu-harness@computewise` then `claude plugin marketplace remove
computewise`, or `codex plugin remove gpu-harness@computewise` then `codex plugin marketplace remove computewise`.
Past runs stay in `~/.gpuh/` (or each project's `.gpuh/`) until you delete them.

How the scheduler works, and its guarantees, are in
[`harness/skills/gpu-harness/README.md`](harness/skills/gpu-harness/README.md). The plugin's layout is in
[`harness/README.md`](harness/README.md).

## The benchmark

An agent gets a realistic brief: produce a results table, decide which optimizer setups pass an adoption rule, or
decide which data filter to adopt. It works on a simulated multi-GPU node whose GPUs replay the job times, scores and
logs of recorded real H100 runs. The run is scored against reference schedules, not raw utilisation, so a GPU that is
busy with work the answer never needed counts as waste, just like an idle one.

- **Dataset:** [Salesforce/computewise](https://huggingface.co/datasets/Salesforce/computewise) on Hugging Face. It
  holds the benchmark packages, the recorded job runs and the scored agent runs.
- **Paper:** TBA

> Status: research release. Interfaces, paths and documentation may change.

**The paper's nine tasks.** All nine are in this release.
- **Built from modded-nanogpt, in the dataset:** optimizer adoption A and B (`pkg-g4n-a`, `pkg-g4n-b`),
  many-candidates adoption (`pkg-unneeded-k`), precision stopping (`pkg-k2-precision`) and data filtering
  (`pkg-filter-yield`), with every run the paper reports on them.
- **Built from MLS-Bench, on your machine:** the KV-cache results tables, plan given and plan hidden
  (`mls-kv-report`, `mls-kv-report-h`), the KV acceptance gate (`mls-kv-gate`) and the 8-GPU cache-and-decoding table
  (`pkg-k2-latepole`). [MLS-Bench](https://github.com/Imbernoulli/MLS-Bench) declares no licence, so none of its code
  or data is in this repository or the dataset. `gpu-tasks/build/add_mlsbench_tasks.sh` clones it at the commit the
  paper used and builds the four tasks from it and from our own measurements in `gpu-tasks/survey/`.

**The conditions it compares:**
- no help;
- one added sentence of instruction;
- a SLURM-like scheduler (`scheduler/`);
- written team procedures (`skills/`);
- the gpu-harness, in Claude Code and in its Codex edition.

The dataset card gives the condition codes used in the scored runs.

### What is in this repository

| Path | What it is |
|---|---|
| `harness/` | **The gpu-harness plugin** for Claude Code and Codex (see above). The Claude Code skill and its Codex edition are the versions the paper reports. They run the same code; only the agent-facing instructions differ. |
| `.claude-plugin/`, `.agents/plugins/` | The marketplace files that `claude plugin marketplace add` and `codex plugin marketplace add` read. They point to `harness/`. |
| `gpu-mock/` | **The simulated GPU node:** a torch patch, an NVML and `nvidia-smi` bridge, and a replay daemon. The agent's real commands run unchanged, and each job's step times, scores and logs come from a recorded H100 run. `archive/INDEX.tsv` lists every simulator build and what changed in it. |
| `scheduler/` | A small SLURM-compatible scheduler (`sbatch`, `squeue`, `srun`, `scancel`, `sacct`, ...) that runs inside the benchmark's container, for the study condition that gives the agent a scheduler. It is not part of the plugin. |
| `gpu-tasks/` | The task registry tooling. `build/` holds the generators that build each benchmark package from upstream records, and `RECORD-FORMAT.md` is the record contract between the tasks and the simulator. The generated task files and records are in the Hugging Face dataset; [`gpu-tasks/registry/README.md`](gpu-tasks/registry/README.md) says where to put them. `build/add_mlsbench_tasks.sh` builds the four MLS-Bench tasks from a clone of MLS-Bench. |
| `isolation/` | The sandbox: the container image (`image/`); the single-run launcher `run_one.py`, which runs one task, one coding agent, one model and one seed in a capped container with a read-only root and its own disk; the model-API forwarder; and the cgroup sampler. |
| `runner/` | The campaign driver, the scorer (`score/`), scripted reference agents (`scripted/`), timelines (`observe/`) and the code that starts and stops each coding agent (`runner/harness/`, not the plugin). |
| `codex/harness/` | The Codex CLI session driver the launcher uses. It is not the Codex plugin, which is in `harness/codex-skills/`. |
| `skills/` | The study's other conditions, as instructions without code: a team procedure, a decide-first rule, a scheduler playbook and a parallel-attention prompt. They are kept so the study can be reproduced; they are not the gpu-harness. `history/` holds the harness's previous development step. |

### Running the benchmark

Everything runs on one Linux Docker host with no GPUs, because every GPU is simulated. A run uses the whole machine;
the study used 44-core, 352 GB machines (GCP `c3-highmem-44`). The steps below were run from a fresh clone of this
repository and the dataset on 2026-11-17, and the run they produced is compared with the paper's at the end.

1. **The host.** Linux x86_64 with Docker CE (cgroup v2), the `loop` kernel module, Python 3.9 or later, `git-lfs`,
   and about 200 GB free under `/srv` for Docker's data, the task caches and the runs. The commands below run as root.
2. **Code and data.** Clone this repository and put the dataset in place as
   [`gpu-tasks/registry/README.md`](gpu-tasks/registry/README.md) says. Its check should report 0 contract violations.
   For the four MLS-Bench tasks, then run `bash gpu-tasks/build/add_mlsbench_tasks.sh`. It clones MLS-Bench, fetches
   the late-pole task's datasets and model files from their own hosts (each checked by sha256), and builds the four
   tasks into the registry in about 4 minutes. Its check should report 66 tasks and 0 contract violations.
3. **Images.** Build the agent's box image and the simulator's image for the task's environment:

   ```bash
   sudo bash isolation/image/build.sh --task pkg-g4n-a --image ml-node:py311-torch2110 --sidecar-image gpu-daemon:py311-torch2110
   ```

   This took about 7 minutes. It fetches pinned packages from Debian, PyPI, the PyTorch index and npm. The four
   modded-nanogpt tasks (`pkg-g4n-a`, `pkg-g4n-b`, `pkg-unneeded-k`, `pkg-k2-precision`) share this image;
   `pkg-filter-yield` needs its own (`--task pkg-filter-yield --image ml-node:mng3-filter --sidecar-image gpu-daemon:mng3-filter`).
   The three KV tasks add a layer on top of this image with `isolation/image/mlskv/build_mlskv_layer.sh`, and
   `pkg-k2-latepole` has its own 8-GPU pair from `isolation/image/latepole/build_latepole.sh`; each script's header gives
   its arguments. These two builds were used for the paper's runs and were not part of the check at the end.
4. **The model gate.** Every model call leaves the box through a gate on the host. The gate drops whatever credentials
   the agent sends and adds yours:

   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   sudo -E CW_UPSTREAM=https://api.anthropic.com CW_UPSTREAM_KEY_ENV=ANTHROPIC_API_KEY sh isolation/forwarder/start_gate.sh
   ```

   For Codex runs, use `CW_UPSTREAM=https://api.openai.com CW_UPSTREAM_KEY_ENV=OPENAI_API_KEY
   CW_UPSTREAM_KEY_HEADER=authorization`. A gate serves one upstream; to switch, stop it, delete
   `/srv/stage/gate/gate.json` and start it again. A local model router that adds its own credentials also works as
   `CW_UPSTREAM` (an `http://` address, without `CW_UPSTREAM_KEY_ENV`). The study enforced each run's dollar ceiling in
   its own router; here, set a spending limit on your API key, and pass `--ceiling-confirmed` below.
5. **Run and score.** The brief is the task's own, plus one sentence for some conditions (table below). The gpu-harness
   on Haiku:

   ```bash
   python3 -c 'import json; print(json.load(open("gpu-tasks/registry/tasks/pkg-g4n-a.json"))["brief"]["text"], end="")' > brief.txt
   printf '\n\nUse the gpu-harness skill.' >> brief.txt
   sudo python3 -I runner/run_smoke.py --task pkg-g4n-a --harness claude-code --model claude-haiku-5-5 \
     --budget-usd 144 --max-hours 32.5 --brief-file "$PWD/brief.txt" --ceiling-confirmed \
     -- --agent-teams --skill="$PWD/harness/skills/gpu-harness" --disk 40g
   ```

   The run directory is `/srv/runs/<tag>/`. `score.txt` gives the time to the answer and the lower bound, and
   `timeline.html` shows every job on every GPU. Simulated jobs keep their recorded durations, so a run takes hours.

**The paper's conditions.** Each is the command in step 5 with these changes. The brief sentence goes where step 5
puts "Use the gpu-harness skill.":

| Condition | Code in the data | Sentence added to the brief | Flags after `--` |
|---|---|---|---|
| No help | `L0` | none | `--disk 40g` |
| One added sentence | `L3` | "Please make sure no GPU sits idle while there is still work to do: have the next run starting up while the current ones are still training, and keep every GPU busy until all the runs are finished." | `--disk 40g` |
| SLURM-like scheduler | `S1` | "This machine runs SLURM. GPU work runs only inside SLURM jobs: submit with `sbatch` (`--gres=gpu:N` for N GPUs), and see `squeue`, `sacct` and `scancel`." | `--scheduler on --disk 40g` |
| Written team procedure | `T4` | "Use the gpu-team skill." | `--agent-teams --skill=$PWD/skills/gpu-team --disk 40g` |
| gpu-harness, Claude Code | `A5` | "Use the gpu-harness skill." | `--agent-teams --skill=$PWD/harness/skills/gpu-harness --disk 40g` |
| gpu-harness, Codex | `A5-codex` | "Use the gpu-harness skill." | `--skill=$PWD/harness/codex-skills/gpu-harness --disk 40g`, with `--harness codex` and a Codex model |

The models: `claude-haiku-5-5`, `claude-sonnet-5-5` and `claude-opus-5-5` in Claude Code; `gpt-6-astra` and
`gpt-6.1-sol` in Codex. A scripted agent needs no model and no gate: `--harness scripted --policy packed --speed 20`
runs every job, longest first, in about 25 minutes, which checks that the box, the simulator and the scorer work (a
run above speed 1 checks the pipeline only; its numbers are not results).

The simulator here is its final build (r8.9.3 in [`gpu-mock/archive/INDEX.tsv`](gpu-mock/archive/INDEX.tsv)). Each run
in the dataset records the build it ran on (`gpumock_build`); most ran on r7 or r7.1, and the builds after them
changed how the MLS-Bench tasks and the tabular task are served.

**Reading a run against the paper.** The paper's figure for a run is `over_ref`: its time to the answer over the
time of the task's scripted reference run (W0). The dataset ships the W0 runs the paper's readings use, in
`data/runs/references/`. Put them beside your run and read it with the study's screen:

```bash
mkdir -p ~/computewise-runs
ln -s /path/to/computewise-data/data/runs/references/W0-* ~/computewise-runs/
sudo cp -r /srv/runs/<tag> ~/computewise-runs/ && sudo chown -R "$USER" ~/computewise-runs
python3 -I runner/score/screen.py --task pkg-g4n-a --all-validities
```

`--all-validities` is needed because a run through the gate alone reads `validity UNCHECKED`: the served-model check
reads the cost ledger of the study's model router. The arm column shows `?` unless the run directory is named the
study's way (`G-g4na-haiku-A5`).

**The check.** On 2026-11-17, these steps reran the gpu-harness on Haiku on `pkg-g4n-a` from a fresh clone and the
dataset on a new `c3-highmem-44`. The run took 5 h 42 min and adopted the right setups (aurora-pruned, radial-damp and
soda-fade). It read `over_ref` 1.021 and `over_best` 1.020, against 1.010 to 1.021 and 1.008 to 1.020 for the paper's
three runs. Each of its 16 jobs matched its recorded run.

The tests:

```bash
(cd scheduler && python3 -I tests/selftest.py)                       # scheduler self-test
python3 -m unittest discover -s harness/skills/gpu-harness/tests      # harness tests (88, about 6 s)
python3 -I isolation/tests/gate_pin_known_answer.py isolation/forwarder/forwarder.py          # model gate
python3 -I isolation/tests/gate_upstream_key_known_answer.py isolation/forwarder/forwarder.py # gate's API key
```

## Third-party material

See `NOTICE.md` for the third-party sources, what we use from each, and their licences. The container images are built
locally and never distributed, because they contain third-party binaries. Benchmark datasets and model weights are
fetched from their own hosts when a run's inputs are prepared, under their own terms.

## License

Apache License 2.0; see `LICENSE.txt`.

## Contributing and security

This project is published but not supported; see `CONTRIBUTING.md`. Report security issues as described in
`SECURITY.md`.
