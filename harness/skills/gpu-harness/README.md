# gpu-harness

A harness for agents that run GPU jobs to reach a decision: which method to deploy, which optimizer setups to adopt, the full table of a benchmark. The agent writes the job graph and makes every judgement. The harness owns the GPUs and does the mechanics, by an algorithm with stated guarantees.

It ships as a Claude Code skill: `SKILL.md` tells the agent how to use it, and `bin/gpuh` is the one command it calls. It is pure-stdlib Python (3.9+), with nothing to install. Any agent that can run a shell command can use it.

## Why

Agents waste GPUs in two ways. **Waiting badly:** a GPU sits free because nobody noticed a job ended. **Running the wrong work:** a GPU is busy with a job the answer never needed. A team structure written in prose (a planner, a GPU teammate, a teammate per candidate, a subagent per job) removes most of both, but every failure left over was mechanics done in prose:

- a grant written in a ledger and never launched;
- a ledger that drifted from the machine;
- a cancel that killed a wrapper script and left its evaluator running;
- a progress pattern too strict to ever match.

The harness moves the mechanics into code and leaves the judgement with the model.

## Architecture

```
            agent (main session)                       branch teammates (one per candidate)
            writes plan.json, reads REJECTED/DECIDED   judge their candidates from one-line events
                     │  gpuh start / wait                       │  gpuh wait / pass / reject / decide
                     ▼                                          ▼
   ┌────────────────────────────────── gpuh (cli) ──────────────────────────────────┐
   │                 command spool  ─────────────►  daemon (single writer)           │
   │                 event log      ◄─────────────    each tick:                     │
   │                                                   1 apply commands               │
   │   plan ── validate graph                          2 observe the machine          │
   │   estimate ── online duration model               3 emit events                  │
   │   scheduler ── the policy                         4 schedule free GPUs           │
   │   progress ── pace from logs                      5 write the board              │
   │   launcher ── process groups, liveness, kill                                     │
   │   ledger ── board.json, events.jsonl, cursors, timeline                          │
   │   timeline ── status, results, HTML Gantt                                        │
   └─────────────────────────────────────────────────────────────────────────────────┘
                     │ one process group per job, CUDA_VISIBLE_DEVICES = its GPU
                     ▼
                GPU 0   GPU 1   GPU 2   GPU 3
```

| Module | What it does |
|---|---|
| `model.py` | candidates, jobs, roles (check, needed, speculative), states |
| `plan.py` | loads the job graph the agent writes, expands `foreach` templates, refuses cycles and dangling references |
| `estimate.py` | online duration model, log d = a_candidate + b_kind, fitted from priors, live projections, measured runs, and the elapsed time of jobs past their prediction |
| `progress.py` | reads pace from log lines (`step:N/M … train_time`, `step_time`, tqdm, counters), projects each job's length, and catches the values a plan names as milestones |
| `scheduler.py` | **the algorithm**: decision-aware critical-path list scheduling, with its guarantees in the docstring |
| `launcher.py` | starts each job in its own process group, with its one GPU, or none for a CPU-only job; liveness from the machine; TERM then KILL to the whole group |
| `ledger.py` | the run on disk: board, append-only events, per-agent cursors, waiters, command spool |
| `daemon.py` | the engine: commands, observation, events, scheduling, once a second |
| `timeline.py` | `status`, `results`, and an HTML timeline of every job on every GPU |
| `cli.py` | `gpuh`, with `--help` on every subcommand |

## The algorithm

The scheduler gives each free GPU the runnable job of highest priority:

1. **The reference's checks first.** Every other candidate is judged against the reference.
2. **Calibration and the judgement hold: one mechanism that starts the long pole as soon as it can be identified.**
   - Each candidate nobody has measured yet runs its shortest job next, whatever the plan's priors say, because the priors are what the measurement corrects. Ties go to a workload no other unmeasured candidate is running, so one wave measures several workloads.
   - The reference runs its job of each workload it hasn't measured early. Its pace is the yardstick, so this measures how long the workload itself is.
   - When a check that gates the longest chain becomes judgeable, the GPU it freed is held for up to 60 s for the agent's pass. Once the pass comes, that GPU goes to the long job.
   - The duration model carries each measurement to the other jobs, so a slow method's long job starts on the first free GPU, not last.
3. **The decision-aware critical path.** A job's chain is its remaining length plus the longest chain that waits on it. Work gated on a check counts as if the check passes, plus a judgement latency. So a candidate's cheap check inherits the long run it gates, and starts before shorter independent work. With no checks this is Graham's critical-path list scheduling; with independent jobs it is longest processing time first.
4. **The ratio rule among one candidate's checks:** probability of rejection per second of cost, Mitten's rule for testing a series system. The agent may supply each check's prior.
5. **Speculation only on idle GPUs.** Gated work runs early on a GPU nothing else wants, and is preempted by real work.
6. **Rejection cancels the candidate's whole subtree,** process groups and all.

**Guarantees:**
- work-conserving to within one tick, except for judgement holds: each costs at most 60 GPU-seconds, and only a check gating the longest chain earns one;
- speculation delays real work by at most one tick;
- cancellation completes within `kill_grace + 2 × tick`;
- for the work that runs, T ≤ W/m + (1 − 1/m)·L, the list-scheduling bound.

`tests/test_scheduler.py` checks the bound on random graphs.

**Two kinds of job, one graph.**
- **GPU jobs** hold one GPU each.
- **CPU-only jobs** (`"gpu": 0`) hold none and run beside them, at most `policy.cpu_slots` at once. A data preparation never keeps a GPU from training.

**Milestones.** A plan can name a value its rule reads before a job ends, such as the loss at step 1000, as a regex with a `value` group. The harness sends it the moment it appears in the log, with the reference's value beside it, so a branch can reject while the run is still going. The code reads the numbers; it never compares them with the rule.

**Boundaries.** The code never judges a candidate. The model passes checks, rejects and decides. The harness refuses one thing: a rejection whose only evidence is screens (slices, one seed of several). The rule cannot act on a screen.

## Use

```bash
gpuh check plan.json                 # validate
gpuh start plan.json                 # run; returns at once
gpuh wait --for main                 # (background) returns when main has something to decide
gpuh --as quant wait --for quant     # each branch the same
gpuh --as quant pass kivi_hotpotqa   # a check passed: open the gated work
gpuh --as quant reject squat --why 'passage 59.96 vs 62.42 (-2.45 > 2)' --evidence squat_passage,fa_passage
gpuh status -v                       # GPUs, candidates, the queue and why
gpuh results; gpuh export            # results; timeline.html
gpuh finish
```

The plan format and every event are described in `SKILL.md` and in the module docstrings.

## Tests

```bash
python3 -m unittest discover -s tests
```

The tests use real processes: a wrapper script's child dies with its group, a killed job leaves no exit code (DIED), and `CUDA_VISIBLE_DEVICES` is set. Pace is read from each log shape, the estimator carries a measured slowdown across kinds, and plans that should be refused are refused. The scheduler is tested on order, gates, speculation, preemption and the bound.
