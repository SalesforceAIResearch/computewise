# Where the lost time goes: delay, ordering, unneeded work

the scheduler, 2026-10-29 (evening MDT), for the analysis: RV-001, Opus's findings 3, 4 and 9 (`design/review-2026-10-29/revisions/r1/reviews/RV-001-narrative-plan/REPORT-1.md`). Code: `decompose.py` beside this note; method and definitions are in its docstring. Run `python3 -I decompose.py --json out.json` from this directory, and `--selftest` for the known answers. The CP-SAT optimum needs OR-Tools in a venv (`~/.venvs/ortools`; the docstring has the two commands). This is a replay of each run's own jobs on the same 4 GPUs, offline, with no agent re-run. The tables are its output on the valid L0, S1 and A3 runs in `~/computewise-runs` as of 2026-11-02 23:25Z (149 runs), harness code `c429f3b9a095`, after the two fixes below. The 2026-10-30 tables (99 runs) are in git.

## Corrected 2026-11-02, second: the optimum's model

**The CP-SAT floor was too high on the twins.** Lane OBS found it. The model combined two symmetry breaks, machines in load order and the longest job on the first machine. Together they force the longest job onto the most-loaded machine, which an optimum needn't do, so the solver could report OPTIMAL at a makespan above the true optimum. Known answer: {10, 6, 6, 6, 6} on 2 machines is 18 ({10, 6} and {6, 6, 6}); the pair of breaks gave 22.
- **The fix:** only the load-order break is kept. The `--selftest` known answer for {10, 6, 6, 6, 6} gives 18, proved OPTIMAL. 40 of 40 known answers pass.
- **Where it came from:** this tool's first full run (2026-10-30 02:15Z) gave G4n A (b) 19,328 s. The second break was added within the hour to speed up a slow solve, and from the 02:43Z run on the figure read 19,540 s. Every table since, the 10-30 JSON included, carried it.
- **What moved:** 84 of the 218 run floors, all of them twin runs (G4n A and B, both readings), each down by 2–556 s. No G1, H5 or G4 floor moved. The twin (b) package optima moved too, A from 19,540 to 19,328 s and B from 19,487 to 19,361 s, so the twin (b) planning ratios each rise by about 0.011 (A: FCFS 1.040 → 1.051, the harness 1.017 → 1.028). No reading changes.

Pooled L0, before → after (a share that moved by less than 0.005 is shown once):

| package | models | n | loss, h | delay | ordering | unneeded | unruled | harness_gap |
|---|---|---|---|---|---|---|---|---|
| G1 | Claude | 8 | 35.1 → 35.1 | 0.17 | 0.69 | 0.00 | 0.12 | 0.01 |
| G1 | OpenAI | 6 | 25.5 → 25.5 | 0.01 | 0.98 | 0.00 | 0.00 | 0.01 |
| H5 | Claude | 8 | 35.3 → 35.3 | 0.32 | 0.57 | 0.00 | 0.11 | 0.01 |
| H5 | OpenAI | 3 | 7.5 → 7.5 | -0.06 | 1.01 | 0.00 | 0.03 | 0.02 |
| G4 | Claude | 7 | 49.6 → 49.6 | 0.16 | 0.05 | 0.65 | 0.09 | 0.04 |
| G4 | OpenAI | 4 | 3.1 → 3.1 | 0.14 | 0.52 | 0.02 | -0.02 | 0.34 |
| G4n A (b) | Claude | 9 | 19.5 → 19.8 | 0.17 | 0.01 | 0.76 → 0.75 | -0.00 | 0.05 → 0.06 |
| G4n A (b) | OpenAI | 6 | 14.8 → 15.0 | 0.02 | 0.06 | 0.88 → 0.87 | -0.00 | 0.04 → 0.06 |
| G4n A (a) | Claude | 9 | 7.8 → 8.0 | 0.44 → 0.43 | 0.07 | 0.02 | -0.00 | 0.47 → 0.49 |
| G4n A (a) | OpenAI | 6 | 4.8 → 4.8 | 0.07 | 0.33 | 0.00 | 0.00 | 0.60 |
| G4n B (b) | Claude | 8 | 15.6 → 15.8 | 0.13 | 0.03 | 0.79 → 0.77 | 0.01 | 0.05 → 0.06 |
| G4n B (b) | OpenAI | 5 | 12.2 → 12.4 | 0.02 | 0.04 | 0.90 → 0.89 | -0.00 | 0.04 → 0.05 |
| G4n B (a) | Claude | 8 | 5.6 → 5.7 | 0.37 | 0.08 | 0.00 | 0.00 | 0.54 → 0.55 |
| G4n B (a) | OpenAI | 5 | 3.9 → 3.9 | 0.07 | 0.26 | 0.00 | 0.00 | 0.67 |

- **Who else used it:**
  - **This floor code:** OBS's `mine_waste.py` and `sim_designs.py` call `decompose.optimum_many`, so they take the fix when rerun. analysis' `taxonomy/counterfactual.py` reads its floors from `decompose-2026-10-30.json`, which has the defect.
  - **No other copy of the two breaks exists.** Every CP-SAT model under the paper's tree was grepped: OBS's `floors_recheck.py` and `olp_optimum.py`, and `gen_obs_latepole.py`, keep the load-order break only.
  - **The scorer's hindsight bound and gpu-tasks' best schedules don't use CP-SAT** (list schedules and their own builders), so bar 2 and bar 4 are untouched.

## Corrected 2026-11-02: jobs on a GPU another job held

**The table below replaces the one of 2026-10-30.** Lane OBS found the defect: the loader took each job's GPU from gpu-mock's `busy_start` alone, and gpu-mock logs that only when a GPU goes from idle to busy. So a job launched onto a GPU another job held had no GPU and was dropped. Fixing it exposed a second defect, which the first had hidden: a GPU several jobs shared was divided in equal slices, and gpu-mock serves a shared GPU's units first in, first out. So a long job stacked with short ones was given far too little time, and the short ones far too much (−58% to +393% against their recorded solo lengths).
- **The fix** (`shared_pool.job_gpu_map` and `shared_pool.job_work`, used by both tools):
  - a job's GPU comes from the first `busy_start`, `hold` or `enqueue` that names one. Over the 91 runs checked, `hold` and `busy_start` agree on all 1,768 jobs that have both, and no job names two GPUs;
  - a stretch of a GPU that several jobs shared is divided in proportion to each job's own work: the daemon's unit costs, `modelled_s + host_gap_s`, plus its start-up. That work is a full job's recorded solo length to a median 0.1%, and sums to the stretch's busy time within 1.5%. A job alone keeps its wall time, as before.
- **What it reaches:** 81 jobs were dropped in 8 runs, and 12 runs shared a GPU (64 shared stretches, 367 jobs). The new per-job lengths of the stacked runs sit within 1% of the recorded solo lengths in the median (worst job +13%), against medians of +7% to +16% before.
- **Known answers (`--selftest`, both tools):**
  - a job stacked onto a busy GPU is read from its hold, and `busy_start` alone drops it;
  - a shared stretch of 140 s with 30 s and 90 s of work splits 35 and 105;
  - a job alone keeps its wall time;
  - a stretch with a job of unknown work falls back to equal slices.
- **What it doesn't touch:** the scorer's own figures (over_best, over_ref, over_feasible). They read total work per GPU, which this defect doesn't affect.
- **Still affected:** `codex/tick_decomposition.py`'s `jobs_of` (the Codex driver) has the same `busy_start`-only read. So do the two callers that import it, `gpu-tasks/survey/rv001-2026-10-29/g4_rule_sensitivity.py` and lane OBS's `mine_waste.py`. They are not fixed here; their owners have been told.

**What moved.** Eleven run readings, all in runs that shared a GPU. Everything else is unchanged to the second (the 2026-10-30 run with `--gpu-from busy_start --share equal` reproduces that day's table on all 99 shared runs). The rows below are the defect run against the fixed one, on the same 149 runs (`compare_decompose.py`):

| run | reading | floor, s | loss, h | delay | ordering | unneeded | unruled | harness_gap |
|---|---|---|---|---|---|---|---|---|
| G-kvgate-fable-L0 | - | 21,427 → 13,287 | 6.15 → 8.41 | 0.05 → 0.34 | -0.01 → -0.08 | 0.00 → 0.55 | 0.95 → 0.14 | 0.02 → 0.05 |
| G-kvgate-haiku-L0-r3 | - | 13,162 → 13,447 | 7.66 → 7.58 | 0.49 → 0.52 | -0.03 → -0.09 | 0.50 → 0.52 | 0.01 | 0.04 |
| G-kvgate-opus-L0-r2 | - | 18,159 → 13,282 | 6.16 → 7.52 | 0.11 | 0.05 → -0.15 | 0.64 → 0.70 | 0.10 → 0.30 | 0.10 → 0.04 |
| G-kvrep-opus-L0-r2 | - | 32,553 → 32,513 | 0.34 → 0.35 | 0.60 | 0.29 → -0.24 | 0.00 | 0.00 → 0.54 | 0.11 |
| G-kvreph-astra-L0 | - | 32,498 → 32,495 | 1.32 → 1.32 | -0.01 → -0.06 | 0.99 → 1.01 | 0.00 | 0.00 | 0.04 |
| G-kvreph-astra-L0-m2 | - | 33,262 → 32,449 | 3.36 → 3.59 | 1.00 → -0.09 | 0.00 → 1.08 | 0.00 | 0.00 | 0.01 |
| G-kvreph-astra-L0-m3 | - | 33,314 → 32,501 | 2.42 → 2.64 | 0.01 → -0.03 | 0.83 → 0.93 | 0.00 | 0.00 → 0.08 | 0.16 → 0.02 |
| G-kvreph-fable-L0-r2 | - | 36,257 → 32,451 | 0.13 → 1.19 | 0.91 → 0.81 | -2.75 → -0.01 | 0.00 | -1.08 → 0.14 | 3.91 → 0.05 |
| G-kvreph-opus-L0-r2 | - | 41,565 → 32,459 | 5.78 → 8.31 | 0.55 → 0.39 | 0.30 → 0.19 | 0.00 | 0.00 → 0.42 | 0.15 → 0.00 |
| G-kvreph-opus-L0-r3 | - | 32,034 → 32,448 | 3.51 → 3.39 | 0.51 → 0.20 | 0.34 → 0.79 | 0.00 | 0.00 | 0.15 → 0.01 |
| G-kvreph-opus-L0-r4 | - | 36,142 → 32,452 | 3.65 → 4.67 | 0.33 → -0.05 | 0.13 → 1.05 | 0.00 | 0.00 | 0.55 → 0.00 |

Pooled L0, before → after (a share that moved by less than 0.005 is shown once):

| package | models | n | loss, h | delay | ordering | unneeded | unruled | harness_gap |
|---|---|---|---|---|---|---|---|---|
| G1 | Claude | 8 | 35.1 → 35.1 | 0.17 | 0.70 → 0.69 | 0.00 | 0.12 → 0.12 | 0.01 |
| G1 | OpenAI | 6 | 25.5 → 25.5 | 0.01 | 0.98 | 0.00 | 0.00 | 0.01 |
| H5 | Claude | 8 | 31.9 → 35.3 | 0.43 → 0.32 | 0.45 → 0.57 | 0.00 | 0.01 → 0.11 | 0.11 → 0.01 |
| H5 | OpenAI | 3 | 7.1 → 7.5 | 0.47 → -0.06 | 0.47 → 1.01 | 0.00 | 0.00 → 0.03 | 0.06 → 0.02 |
| G4 | Claude | 7 | 48.3 → 49.6 | 0.16 | 0.10 → 0.05 | 0.64 → 0.65 | 0.06 → 0.09 | 0.05 → 0.04 |
| G4 | OpenAI | 4 | 3.1 → 3.1 | 0.14 | 0.52 | 0.02 | -0.02 | 0.34 |
| G4n A (b) | Claude | 9 | 19.5 → 19.5 | 0.18 | 0.01 | 0.76 | -0.00 | 0.05 |
| G4n A (b) | OpenAI | 6 | 14.8 → 14.8 | 0.02 | 0.06 | 0.88 | -0.00 | 0.04 |
| G4n A (a) | Claude | 9 | 7.8 → 7.8 | 0.44 | 0.07 | 0.02 | -0.00 | 0.47 |
| G4n A (a) | OpenAI | 6 | 4.8 → 4.8 | 0.07 | 0.33 | 0.00 | 0.00 | 0.60 |
| G4n B (b) | Claude | 8 | 15.6 → 15.6 | 0.14 | 0.03 | 0.79 | 0.01 | 0.05 |
| G4n B (b) | OpenAI | 5 | 12.2 → 12.2 | 0.02 | 0.04 | 0.90 | -0.00 | 0.04 |
| G4n B (a) | Claude | 8 | 5.6 → 5.6 | 0.38 | 0.08 | 0.00 | 0.00 | 0.54 |
| G4n B (a) | OpenAI | 5 | 3.9 → 3.9 | 0.07 | 0.26 | 0.00 | 0.00 | 0.67 |


**What this changes in the reading:**
- **H5's split moves.** The 10-30 note said H5's delay "is close to half (0.43–0.47), and ordering the rest". That was the defect.
  - **Corrected:** Claude 0.32 delay and 0.57 ordering, plus 0.11 unneeded work never ruled out (Opus's partial GSM8K reruns). The OpenAI models are 1.01 ordering.
  - **So H5 is ordering, like G1.** Delay is no longer the largest part of H5 for either family. The two-family split stands, and it is cleaner: G1 and H5 are ordering, G4 for Claude is unneeded work, and the twins depend on the brief's reading.
- **The OpenAI H5 row's delay is now slightly negative (−0.06).** Astra's runs pinned jobs to GPUs and stacked them, so replaying its jobs in launch order on any free GPU (V1f) is *slower* than its own placement. FCFS lets a long job land late.
- **To show placement, delay is now split in two:** idle, the agent's GPUs empty between its jobs (V1 − V1g), and placement, its own choice of GPU against FCFS (V1g − V1f). V1g replays each job on the agent's own GPU, back to back in launch order.

| package | models | n | delay | idle | placement |
|---|---|---|---|---|---|
| G1 | Claude | 8 | 0.17 | 0.15 | 0.03 |
| G1 | OpenAI | 6 | 0.01 | 0.01 | 0.00 |
| H5 | Claude | 8 | 0.32 | 0.17 | 0.14 |
| H5 | OpenAI | 3 | -0.06 | 0.06 | -0.12 |
| G4 | Claude | 7 | 0.16 | 0.12 | 0.04 |
| G4 | OpenAI | 4 | 0.14 | 0.11 | 0.03 |
| G4n A (a) | Claude | 9 | 0.43 | 0.17 | 0.26 |
| G4n A (a) | OpenAI | 6 | 0.07 | 0.04 | 0.03 |
| G4n A (b) | Claude | 9 | 0.17 | 0.07 | 0.11 |
| G4n A (b) | OpenAI | 6 | 0.02 | 0.01 | 0.01 |
| G4n B (a) | Claude | 8 | 0.37 | 0.14 | 0.23 |
| G4n B (a) | OpenAI | 5 | 0.07 | 0.04 | 0.04 |
| G4n B (b) | Claude | 8 | 0.13 | 0.05 | 0.08 |
| G4n B (b) | OpenAI | 5 | 0.02 | 0.01 | 0.01 |

  - **Idle is never negative** (checked on all 149 runs), because a GPU can't finish its jobs sooner than their summed length.
  - **Placement is the GPU-pinning cost** OBS calls stacked GPUs (`static_per_gpu_queue`, M4). It's largest on the twins under reading (a) and on H5 for Claude. On H5 the OpenAI models' placement packed better than FCFS (−0.12).
- **G4 barely moves:** Claude 0.65 unneeded, ruled out in the run. Fable's G4 run moves most, because 17 of its 44 jobs were dropped before: unneeded ruled out goes from 0.00 to 0.55. The planning row and the G4 LPT separation do not move: they replay the task's own job matrix, not a run's.
- **The shared-pool note's numbers move only slightly** (`shared_pool.md`, same fix). Over the 64 tenants now in it: lost 0.38 of box capacity, a pool recovers 0.24, and 0.13 is unneeded work (0.36 of the loss), against 0.39, 0.25, 0.14 and 0.35 on 10-29's 53. The one cell that moves visibly is H5 for Astra and Sol: lost 0.34 → 0.21 of the box, all of it idle either way.

## The answer

**Two kinds of waste, in a mix set by the package, plus the agent's own delay as a third component.** Each run's loss is its time to the deliverable over the clairvoyant optimum of the needed work it ran. Pooled L0 runs, as shares of the summed loss (149 runs as of 2026-11-02, harness code `c429f3b9a095`):

| package | models | n | loss (h) | delay | ordering | unneeded, ruled out in the run | unneeded, never ruled out | harness gap |
|---|---|---|---|---|---|---|---|---|
| G1 | Claude | 8 | 35.1 | 0.17 | 0.69 | 0.00 | 0.12 | 0.01 |
| G1 | OpenAI | 6 | 25.5 | 0.01 | 0.98 | 0.00 | 0.00 | 0.01 |
| H5 | Claude | 8 | 35.3 | 0.32 | 0.57 | 0.00 | 0.11 | 0.01 |
| H5 | OpenAI | 3 | 7.5 | -0.06 | 1.01 | 0.00 | 0.03 | 0.02 |
| G4 | Claude | 7 | 49.6 | 0.16 | 0.05 | 0.65 | 0.09 | 0.04 |
| G4 | OpenAI | 4 | 3.1 | 0.14 | 0.52 | 0.02 | -0.02 | 0.34 |
| G4n A (a) | Claude | 9 | 8.0 | 0.43 | 0.07 | 0.02 | -0.00 | 0.49 |
| G4n A (a) | OpenAI | 6 | 4.8 | 0.07 | 0.33 | 0.00 | 0.00 | 0.60 |
| G4n A (b) | Claude | 9 | 19.8 | 0.17 | 0.01 | 0.75 | -0.00 | 0.06 |
| G4n A (b) | OpenAI | 6 | 15.0 | 0.02 | 0.06 | 0.87 | -0.00 | 0.06 |
| G4n B (a) | Claude | 8 | 5.7 | 0.37 | 0.08 | 0.00 | 0.00 | 0.55 |
| G4n B (a) | OpenAI | 5 | 3.9 | 0.07 | 0.26 | 0.00 | 0.00 | 0.67 |
| G4n B (b) | Claude | 8 | 15.8 | 0.13 | 0.03 | 0.77 | 0.01 | 0.06 |
| G4n B (b) | OpenAI | 5 | 12.4 | 0.02 | 0.04 | 0.89 | -0.00 | 0.05 |

- **G1 and H5 are ordering.** No unneeded work the run ruled out. On G1, 0.69 of Claude's loss and 0.98 of the OpenAI models' is the order. squat's 8.1-h GSM8K is 36th of 40 in the task file, and 11 of the 13 G1 L0 runs of 10-30 launched it 30th or later, at 3.9–6.3 h. On H5, ordering is 0.57 for Claude and 1.01 for the OpenAI models, with Claude's delay at 0.32.
- **G4 is unneeded work for Claude, and ordering for the OpenAI models.** Claude: 0.65 of the loss is work the run's own results had already ruled out, 0.09 more is unneeded work they never ruled out, and 0.16 is delay. The OpenAI models lose 3.1 h in all, against Claude's 49.6 h.
- **G4n turns entirely on the reading of the brief** (Opus finding 1). Under (a) SOAP's mean needed, there's no unneeded work. Under (b) it's 0.76–0.90 of every family's loss, and the OpenAI models' share is the higher.
- **The scheduler arm (S1) leaves ordering in place.** See the per-cell table.

## Classical schedulers on the same instances

This is the planning paper's row. Each package's whole job matrix is replayed with true lengths, and a judge applies the rule as each result lands (G4: a method is rejected once its failing workload and the reference's are both in; G4n (b): SOAP is rejected at its first step-time print). Each figure is over the optimum of the reference plan's jobs.

| package | FCFS, file order | harness (calibration, critical path) | LPT, true lengths | optimum (CP-SAT) | harness, prior 1 / 5 s per example | A3 runs as run, over each run's floor (n) |
|---|---|---|---|---|---|---|
| G1 | 1.466 | 1.007 | 1.004 | 1.000 (32,400 s) | 1.006 / 1.007 | 1.009, 1.009, 1.010, 1.010, 1.011, 1.012, 1.012, 1.017, 1.017 (9) |
| H5 | 1.466 | 1.007 | 1.004 | 1.000 (32,400 s) | 1.006 / 1.007 | 1.007, 1.008, 1.008, 1.008, 1.008, 1.011, 1.011, 1.014, 1.053 (9) |
| G4 | 1.129 | 1.083 | 2.229 | 1.000 (13,455 s) | 1.149 / 1.088 | 1.088, 1.092, 1.154, 1.193, 1.203, 1.219, 1.219, 1.617, 1.627 (9) |
| G4n A (a) | 1.101 | 1.097 | 1.078 | 1.000 (25,965 s) | 1.097 / 1.097 | 1.012, 1.014, 1.023, 1.025, 1.026, 1.030, 1.031, 1.034, 1.034 (9) |
| G4n A (b) | 1.051 | 1.028 | 1.039 | 1.000 (19,328 s) | 1.028 / 1.028 | 1.023, 1.025, 1.026, 1.030, 1.031, 1.032, 1.034, 1.034, 1.034 (9) |
| G4n B (a) | 1.101 | 1.095 | 1.078 | 1.000 (25,965 s) | 1.095 / 1.095 | 1.018, 1.022, 1.022, 1.023, 1.025, 1.028, 1.028, 1.029, 1.040 (9) |
| G4n B (b) | 1.049 | 1.027 | 1.039 | 1.000 (19,361 s) | 1.027 / 1.027 | 1.022, 1.022, 1.025, 1.028, 1.028, 1.037, 1.039, 1.040, 1.042 (9) |

- **On G1 and H5, the harness's scheduler is the whole fix, and it needs no clairvoyance.** It learns each method's pace from that method's shortest full job, then schedules the longest predicted first: 1.007, against LPT-with-true-lengths 1.004 and first-come-first-served in file order 1.466. The A3 runs, nine by 2026-11-02, read 1.009–1.017 of their own floors on G1, and 1.007–1.014 on H5 with one at 1.053.
  - **Cross-check with gpu-tasks.** Its duration-blind reference (`gpu-tasks/survey/rv001-2026-10-29/blind_reference-2026-10-30T0125Z.json`) was computed independently and agrees with this replay to 0.1 s. First-come-first-served in file order is 47,487.9 s here and in its `file_order_s`; LPT is 32,542.0 s here and in its `informed_s`. Ordering by workload size alone is 36,593.0 s (its `blind_s`), 1.129 over this optimum.
  - **What each step buys on G1:** 1.466 for file order, 1.129 for ordering by workload size without probing, and 1.007 with the harness's probing.
- **On G4, clairvoyant LPT is the worst schedule (2.229), and that is with the rule.** Every scheduler in this table gets the rule: a judge cancels a method's remaining jobs the moment its failing workload is in. So 2.229 is perfect lengths *plus* cancel-on-evidence. Separating the two:

  | G4, over the optimum (13,455 s) | |
  |---|---|
  | LPT, true lengths, no rule: all 40 jobs run | 2.419 (32,542 s) |
  | LPT, true lengths, plus the rule | **2.229** (29,988 s) |
  | LPT, true lengths, told the needed set (19 jobs) | 1.000 (13,461 s) |
  | the harness, no rule | 2.426 |
  | the harness, plus the rule | 1.083 |

  - **The rule buys LPT almost nothing:** 2.419 → 2.229. LPT starts the eight GSM8Ks (2.0–8.1 h each) and the long LongBench v2 runs first, so its first cheap decisive evaluation starts at 20,099 s, 5.6 h in. By then the work the rule would have cancelled is mostly done. Of its 14 cancellations, 13 drop jobs that had not started, and the one running job it stops saves 65 GPU-s.
  - **What fixes G4 is the order the evidence arrives in, not lengths.** With the rule, the harness goes from 2.426 to 1.083, because it runs each method's shortest job first, and on G4 those are the decisive evaluations. Lengths alone are worth nothing here: LPT given the needed set is optimal (1.000), so LPT's order is wrong only because it doesn't know which jobs are needed, and the rule can't tell it in time.
  - **The harness's advantage is partly by design and partly this package's structure.** By design, because a cheap check first is Mitten's rule (the harness's own note). By structure, because on G4 the cheap evaluations happen to be the decisive ones. A package where the decisive evaluation is the long one would not reward probing cheapest-first. Say both.
- **On G4n, the order barely matters (1.017–1.101), and the brief's reading is worth more than any scheduler.** Reading (a) has more needed work, so its optimum is 25,965 s against (b)'s 19,328 s (twin B: 19,361 s).
- **The harness's priors matter only on G4:** 1.149 at 1 s per example, 1.088 at 5. G1 and H5 move by 0.001.

## Per cell

Each run is measured against its own floor: each needed record once, at its longest instance in the run. Columns V1–V5 are medians of variant / floor. The split is of the summed loss.
- **V1** as run;
- **V1f** the agent's own order with no delay;
- **V2** LPT, true lengths;
- **V3** the harness's scheduler, no agent;
- **V4** V3 plus cancelling what the run's own results had ruled out;
- **V5** V3 on the needed work alone.

"Short of the reading" counts runs that did not finish all of that reading's work: under G4n (a), runs that left SOAP unfinished. Their floor is lower, so their loss reads lower. n is 1–3 a cell.

| package | arm | model | n | V1 as run | V1f agent order, no delay | V2 LPT, true lengths | V3 harness | V4 harness + cancel | V5 harness, needed only | loss (h, summed) | delay | ordering | unneeded, ruled out | unneeded, never ruled out | harness gap | short of the reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 | A3 | haiku | 3 | 1.017 | 1.012 | 1.003 | 1.005 | 1.005 | 1.005 | 0.4 | 0.35 | 0.33 | 0.00 | 0.00 | 0.32 |  |
| G1 | A3 | opus | 3 | 1.012 | 1.004 | 1.003 | 1.004 | 1.004 | 1.004 | 0.3 | 0.65 | 0.02 | 0.00 | 0.00 | 0.33 |  |
| G1 | A3 | sonnet | 3 | 1.009 | 1.004 | 1.003 | 1.002 | 1.002 | 1.002 | 0.3 | 0.57 | 0.13 | 0.00 | 0.00 | 0.29 |  |
| G1 | L0 | astra | 3 | 1.473 | 1.464 | 1.003 | 1.004 | 1.004 | 1.004 | 12.4 | 0.01 | 0.97 | 0.00 | 0.00 | 0.01 |  |
| G1 | L0 | astra-cc | 2 | 1.211 | 1.203 | 1.004 | 1.003 | 1.003 | 1.003 | 3.5 | 0.04 | 0.95 | 0.00 | 0.00 | 0.01 |  |
| G1 | L0 | fable | 1 | 1.107 | 1.019 | 1.014 | 1.192 | 1.192 | 1.003 | 1.0 | 0.82 | -1.62 | 0.00 | 1.77 | 0.02 |  |
| G1 | L0 | haiku | 3 | 1.394 | 1.329 | 1.003 | 1.005 | 1.005 | 1.004 | 10.6 | 0.16 | 0.79 | 0.00 | 0.04 | 0.01 |  |
| G1 | L0 | opus | 2 | 1.593 | 1.416 | 1.103 | 1.222 | 1.222 | 1.004 | 10.7 | 0.30 | 0.33 | 0.00 | 0.37 | 0.01 |  |
| G1 | L0 | sol | 3 | 1.473 | 1.465 | 1.003 | 1.004 | 1.004 | 1.004 | 13.1 | 0.01 | 0.98 | 0.00 | 0.00 | 0.01 |  |
| G1 | L0 | sonnet | 3 | 1.597 | 1.515 | 1.003 | 1.004 | 1.004 | 1.004 | 13.9 | 0.09 | 0.90 | 0.00 | -0.00 | 0.01 |  |
| G1 | S1 | haiku | 3 | 1.504 | 1.498 | 1.003 | 1.004 | 1.004 | 1.004 | 13.6 | 0.04 | 0.95 | 0.00 | 0.00 | 0.01 |  |
| G1 | S1 | sonnet | 3 | 1.502 | 1.498 | 1.003 | 1.005 | 1.005 | 1.005 | 13.6 | 0.01 | 0.98 | 0.00 | 0.00 | 0.01 |  |
| H5 | A3 | haiku | 3 | 1.008 | 1.004 | 1.003 | 1.005 | 1.005 | 1.005 | 0.2 | 0.54 | -0.05 | 0.00 | 0.00 | 0.51 |  |
| H5 | A3 | opus | 3 | 1.008 | 1.003 | 1.003 | 1.005 | 1.005 | 1.005 | 0.3 | 0.55 | -0.21 | 0.00 | 0.18 | 0.47 |  |
| H5 | A3 | sonnet | 3 | 1.011 | 1.006 | 1.003 | 1.005 | 1.005 | 1.005 | 0.6 | 0.21 | 0.04 | 0.00 | 0.00 | 0.75 |  |
| H5 | L0 | astra | 3 | 1.293 | 1.300 | 1.005 | 1.006 | 1.006 | 1.006 | 7.5 | -0.06 | 1.01 | 0.00 | 0.03 | 0.02 |  |
| H5 | L0 | astra-cc | 1 | 1.078 | 1.066 | 1.003 | 1.005 | 1.005 | 1.005 | 0.7 | 0.15 | 0.78 | 0.00 | -0.00 | 0.07 |  |
| H5 | L0 | fable | 1 | 1.132 | 1.025 | 1.003 | 1.026 | 1.026 | 1.007 | 1.2 | 0.81 | -0.01 | 0.00 | 0.14 | 0.05 |  |
| H5 | L0 | haiku | 3 | 1.365 | 1.264 | 1.021 | 1.021 | 1.009 | 1.005 | 10.6 | 0.33 | 0.62 | 0.00 | 0.03 | 0.01 |  |
| H5 | L0 | opus | 3 | 1.518 | 1.544 | 1.005 | 1.002 | 1.002 | 1.002 | 16.4 | 0.23 | 0.56 | 0.00 | 0.21 | 0.00 |  |
| H5 | L0 | sonnet | 2 | 1.463 | 1.245 | 1.003 | 1.005 | 1.005 | 1.005 | 8.4 | 0.47 | 0.52 | 0.00 | 0.00 | 0.01 |  |
| H5 | S1 | haiku | 3 | 1.482 | 1.464 | 1.003 | 1.005 | 1.005 | 1.005 | 13.8 | 0.19 | 0.80 | 0.00 | -0.00 | 0.01 |  |
| H5 | S1 | sonnet | 2 | 1.501 | 1.498 | 1.003 | 1.006 | 1.006 | 1.006 | 9.1 | 0.01 | 0.98 | 0.00 | 0.00 | 0.01 |  |
| G4 | A3 | haiku | 3 | 1.154 | 1.143 | 1.143 | 1.179 | 1.124 | 1.048 | 1.6 | 0.08 | 0.12 | 0.12 | 0.40 | 0.28 |  |
| G4 | A3 | opus | 3 | 1.617 | 1.599 | 1.518 | 1.573 | 1.124 | 1.062 | 5.4 | 0.03 | 0.43 | 0.31 | 0.11 | 0.12 |  |
| G4 | A3 | sonnet | 3 | 1.219 | 1.210 | 1.000 | 1.080 | 1.080 | 1.048 | 2.0 | 0.05 | 0.56 | 0.00 | 0.16 | 0.23 |  |
| G4 | L0 | astra | 2 | 1.224 | 1.205 | 1.000 | 1.068 | 1.068 | 1.075 | 1.7 | 0.09 | 0.61 | 0.00 | -0.03 | 0.33 |  |
| G4 | L0 | astra-cc | 1 | 1.366 | 1.353 | 1.000 | 1.088 | 1.088 | 1.080 | 1.4 | 0.04 | 0.58 | 0.14 | 0.02 | 0.22 |  |
| G4 | L0 | fable | 1 | 3.279 | 2.502 | 2.446 | 3.552 | 1.421 | 1.109 | 8.4 | 0.34 | -0.08 | 0.55 | 0.14 | 0.05 |  |
| G4 | L0 | haiku | 3 | 3.031 | 2.088 | 2.399 | 2.400 | 1.084 | 1.080 | 23.0 | 0.30 | 0.04 | 0.61 | 0.01 | 0.04 |  |
| G4 | L0 | opus | 2 | 3.116 | 2.978 | 2.389 | 3.000 | 1.418 | 1.087 | 15.7 | 0.06 | 0.05 | 0.69 | 0.16 | 0.04 |  |
| G4 | L0 | sol | 2 | 1.247 | 1.206 | 1.080 | 1.118 | 1.093 | 1.092 | 1.4 | 0.20 | 0.41 | 0.04 | 0.00 | 0.35 |  |
| G4 | L0 | sonnet | 2 | 2.452 | 2.440 | 2.092 | 2.275 | 1.346 | 1.071 | 10.9 | 0.01 | 0.09 | 0.66 | 0.19 | 0.05 |  |
| G4 | S1 | haiku | 3 | 2.589 | 2.088 | 2.399 | 2.401 | 1.083 | 1.080 | 19.0 | 0.35 | -0.07 | 0.66 | 0.00 | 0.05 |  |
| G4 | S1 | sonnet | 3 | 2.546 | 2.540 | 2.362 | 2.361 | 1.083 | 1.080 | 19.6 | 0.03 | 0.13 | 0.80 | 0.01 | 0.05 |  |
| G4n A (a) | A3 | haiku | 3 | 1.023 | 1.019 | 1.015 | 1.025 | 1.025 | 1.025 | 0.4 | 0.22 | -0.09 | 0.00 | 0.00 | 0.86 | 3 |
| G4n A (b) | A3 | haiku | 3 | 1.030 | 1.025 | 1.015 | 1.025 | 1.025 | 1.025 | 0.5 | 0.17 | -0.05 | 0.01 | 0.00 | 0.87 |  |
| G4n A (a) | A3 | opus | 3 | 1.025 | 1.020 | 1.015 | 1.024 | 1.024 | 1.024 | 0.4 | 0.17 | 0.01 | 0.00 | 0.05 | 0.77 | 3 |
| G4n A (b) | A3 | opus | 3 | 1.031 | 1.028 | 1.015 | 1.026 | 1.024 | 1.024 | 0.5 | 0.13 | 0.01 | 0.03 | 0.04 | 0.79 |  |
| G4n A (a) | A3 | sonnet | 3 | 1.034 | 1.031 | 1.015 | 1.024 | 1.024 | 1.024 | 0.5 | 0.10 | -0.43 | 0.00 | 0.00 | 1.34 | 3 |
| G4n A (b) | A3 | sonnet | 3 | 1.034 | 1.031 | 1.015 | 1.024 | 1.024 | 1.024 | 0.5 | 0.10 | -0.44 | 0.00 | 0.00 | 1.34 |  |
| G4n A (a) | L0 | astra | 3 | 1.110 | 1.107 | 1.091 | 1.073 | 1.073 | 1.073 | 2.5 | 0.08 | 0.41 | 0.00 | 0.00 | 0.51 |  |
| G4n A (b) | L0 | astra | 3 | 1.455 | 1.451 | 1.430 | 1.406 | 1.025 | 1.025 | 7.7 | 0.03 | 0.08 | 0.84 | -0.00 | 0.05 |  |
| G4n A (a) | L0 | astra-cc | 1 | 1.103 | 1.098 | 1.091 | 1.073 | 1.073 | 1.073 | 0.7 | 0.06 | 0.24 | 0.00 | 0.00 | 0.70 |  |
| G4n A (b) | L0 | astra-cc | 1 | 1.447 | 1.439 | 1.431 | 1.407 | 1.025 | 1.025 | 2.5 | 0.02 | 0.03 | 0.89 | 0.00 | 0.06 |  |
| G4n A (a) | L0 | fable | 1 | 1.107 | 1.097 | 1.091 | 1.073 | 1.073 | 1.073 | 0.8 | 0.09 | 0.23 | 0.00 | 0.00 | 0.68 |  |
| G4n A (b) | L0 | fable | 1 | 1.452 | 1.438 | 1.431 | 1.406 | 1.025 | 1.025 | 2.5 | 0.03 | 0.03 | 0.88 | -0.00 | 0.06 |  |
| G4n A (a) | L0 | grok | 2 | 1.135 | 1.118 | 1.091 | 1.073 | 1.073 | 1.073 | 1.9 | 0.12 | 0.34 | 0.00 | 0.00 | 0.54 |  |
| G4n A (b) | L0 | grok | 2 | 1.488 | 1.466 | 1.431 | 1.406 | 1.025 | 1.025 | 5.4 | 0.04 | 0.06 | 0.84 | 0.00 | 0.05 |  |
| G4n A (a) | L0 | haiku | 3 | 1.146 | 1.055 | 1.060 | 1.057 | 1.057 | 1.057 | 3.5 | 0.70 | -0.05 | 0.00 | 0.00 | 0.36 | 3 |
| G4n A (b) | L0 | haiku | 3 | 1.299 | 1.202 | 1.191 | 1.233 | 1.025 | 1.025 | 5.9 | 0.41 | -0.01 | 0.52 | 0.00 | 0.07 |  |
| G4n A (a) | L0 | opus | 3 | 1.113 | 1.097 | 1.091 | 1.073 | 1.073 | 1.073 | 2.6 | 0.17 | 0.23 | 0.00 | 0.00 | 0.60 |  |
| G4n A (b) | L0 | opus | 3 | 1.458 | 1.439 | 1.430 | 1.406 | 1.025 | 1.025 | 7.8 | 0.06 | 0.04 | 0.85 | 0.00 | 0.05 |  |
| G4n A (a) | L0 | sol | 3 | 1.103 | 1.097 | 1.091 | 1.073 | 1.073 | 1.073 | 2.2 | 0.05 | 0.24 | 0.00 | 0.00 | 0.71 |  |
| G4n A (b) | L0 | sol | 3 | 1.446 | 1.439 | 1.431 | 1.406 | 1.025 | 1.025 | 7.4 | 0.02 | 0.03 | 0.89 | -0.00 | 0.06 |  |
| G4n A (a) | L0 | sonnet | 3 | 1.053 | 1.042 | 1.034 | 1.040 | 1.040 | 1.040 | 1.9 | 0.30 | 0.06 | 0.08 | -0.01 | 0.57 | 2 |
| G4n A (b) | L0 | sonnet | 3 | 1.302 | 1.288 | 1.278 | 1.286 | 1.024 | 1.025 | 6.2 | 0.09 | 0.01 | 0.84 | -0.00 | 0.07 |  |
| G4n A (a) | S1 | haiku | 2 | 1.117 | 1.098 | 1.091 | 1.073 | 1.073 | 1.073 | 1.7 | 0.17 | 0.21 | 0.00 | 0.00 | 0.62 |  |
| G4n A (b) | S1 | haiku | 2 | 1.464 | 1.439 | 1.431 | 1.406 | 1.025 | 1.025 | 5.1 | 0.05 | 0.03 | 0.86 | -0.00 | 0.05 |  |
| G4n A (a) | S1 | sonnet | 2 | 1.109 | 1.106 | 1.091 | 1.054 | 1.054 | 1.054 | 1.6 | 0.03 | 0.47 | 0.00 | 0.00 | 0.50 |  |
| G4n A (b) | S1 | sonnet | 2 | 1.454 | 1.450 | 1.431 | 1.382 | 1.025 | 1.025 | 5.0 | 0.01 | 0.08 | 0.85 | 0.00 | 0.05 |  |
| G4n B (a) | A3 | haiku | 3 | 1.028 | 1.023 | 1.015 | 1.023 | 1.023 | 1.023 | 0.4 | 0.17 | -0.07 | 0.00 | 0.00 | 0.90 | 3 |
| G4n B (b) | A3 | haiku | 3 | 1.028 | 1.023 | 1.015 | 1.023 | 1.023 | 1.023 | 0.5 | 0.15 | -0.06 | 0.21 | 0.00 | 0.69 |  |
| G4n B (a) | A3 | opus | 3 | 1.023 | 1.014 | 1.002 | 1.008 | 1.008 | 1.008 | 0.4 | 0.28 | 0.22 | 0.00 | -0.07 | 0.57 | 3 |
| G4n B (b) | A3 | opus | 3 | 1.037 | 1.031 | 1.020 | 1.027 | 1.019 | 1.019 | 0.6 | 0.19 | 0.18 | 0.12 | -0.05 | 0.56 |  |
| G4n B (a) | A3 | sonnet | 3 | 1.022 | 1.019 | 1.015 | 1.023 | 1.023 | 1.023 | 0.5 | 0.14 | 0.09 | 0.00 | 0.00 | 0.78 | 3 |
| G4n B (b) | A3 | sonnet | 3 | 1.022 | 1.019 | 1.015 | 1.023 | 1.023 | 1.023 | 0.5 | 0.14 | 0.09 | 0.00 | 0.00 | 0.78 |  |
| G4n B (a) | L0 | astra | 3 | 1.100 | 1.097 | 1.092 | 1.073 | 1.073 | 1.073 | 2.2 | 0.03 | 0.27 | 0.00 | 0.00 | 0.71 |  |
| G4n B (b) | L0 | astra | 3 | 1.438 | 1.434 | 1.428 | 1.402 | 1.023 | 1.023 | 7.3 | 0.01 | 0.04 | 0.90 | -0.00 | 0.05 |  |
| G4n B (a) | L0 | astra-cc | 1 | 1.101 | 1.096 | 1.092 | 1.072 | 1.072 | 1.072 | 0.7 | 0.04 | 0.24 | 0.00 | 0.00 | 0.72 |  |
| G4n B (b) | L0 | astra-cc | 1 | 1.439 | 1.434 | 1.429 | 1.402 | 1.023 | 1.023 | 2.4 | 0.01 | 0.03 | 0.90 | -0.00 | 0.05 |  |
| G4n B (a) | L0 | fable | 1 | 1.112 | 1.095 | 1.092 | 1.090 | 1.090 | 1.090 | 0.8 | 0.15 | 0.04 | 0.00 | 0.00 | 0.80 |  |
| G4n B (b) | L0 | fable | 1 | 1.455 | 1.432 | 1.429 | 1.426 | 1.023 | 1.018 | 2.5 | 0.05 | 0.03 | 0.87 | 0.01 | 0.04 |  |
| G4n B (a) | L0 | grok | 2 | 1.135 | 1.106 | 1.092 | 1.074 | 1.074 | 1.072 | 1.9 | 0.21 | 0.24 | 0.00 | 0.01 | 0.53 |  |
| G4n B (b) | L0 | grok | 2 | 1.485 | 1.448 | 1.429 | 1.405 | 1.025 | 1.023 | 5.3 | 0.08 | 0.04 | 0.84 | 0.00 | 0.05 |  |
| G4n B (a) | L0 | haiku | 3 | 1.122 | 1.095 | 1.092 | 1.071 | 1.071 | 1.071 | 2.2 | 0.37 | 0.04 | 0.00 | 0.00 | 0.59 |  |
| G4n B (b) | L0 | haiku | 3 | 1.467 | 1.432 | 1.429 | 1.401 | 1.022 | 1.022 | 7.3 | 0.11 | 0.01 | 0.82 | 0.00 | 0.05 |  |
| G4n B (a) | L0 | opus | 2 | 1.061 | 1.048 | 1.021 | 1.035 | 1.035 | 1.035 | 0.7 | 0.20 | 0.22 | 0.00 | 0.00 | 0.58 | 2 |
| G4n B (b) | L0 | opus | 2 | 1.137 | 1.124 | 1.094 | 1.109 | 1.025 | 1.021 | 1.5 | 0.10 | 0.10 | 0.63 | 0.03 | 0.15 |  |
| G4n B (a) | L0 | sol | 2 | 1.116 | 1.101 | 1.092 | 1.072 | 1.072 | 1.072 | 1.7 | 0.13 | 0.25 | 0.00 | 0.00 | 0.62 |  |
| G4n B (b) | L0 | sol | 2 | 1.459 | 1.440 | 1.429 | 1.401 | 1.023 | 1.023 | 5.1 | 0.04 | 0.04 | 0.87 | -0.00 | 0.05 |  |
| G4n B (a) | L0 | sonnet | 3 | 1.134 | 1.077 | 1.092 | 1.071 | 1.071 | 1.071 | 2.8 | 0.42 | 0.07 | 0.00 | 0.00 | 0.51 | 1 |
| G4n B (b) | L0 | sonnet | 3 | 1.483 | 1.408 | 1.429 | 1.400 | 1.023 | 1.021 | 7.1 | 0.17 | 0.02 | 0.76 | 0.01 | 0.05 |  |
| G4n B (a) | S1 | haiku | 2 | 1.116 | 1.105 | 1.092 | 1.054 | 1.054 | 1.054 | 1.7 | 0.09 | 0.44 | 0.00 | 0.00 | 0.47 |  |
| G4n B (b) | S1 | haiku | 2 | 1.459 | 1.445 | 1.428 | 1.378 | 1.023 | 1.021 | 5.1 | 0.03 | 0.08 | 0.84 | 0.00 | 0.05 |  |
| G4n B (a) | S1 | sonnet | 2 | 1.094 | 1.086 | 1.065 | 1.037 | 1.037 | 1.037 | 1.3 | 0.09 | 0.53 | 0.00 | 0.00 | 0.39 | 1 |
| G4n B (b) | S1 | sonnet | 2 | 1.347 | 1.337 | 1.311 | 1.275 | 1.024 | 1.018 | 3.8 | 0.03 | 0.11 | 0.79 | 0.02 | 0.05 |  |

## How to read it, and what it isn't

- **What the components are.**
  - **Delay** is V1 − V1f: the agent's own latency, including any wait to read a result before launching.
  - **Ordering and unneeded** split V1f − V4. They interact, because cancelling a job saves more under one order than another, so each is the mean over the two orders it can be removed in (a Shapley split). The brief's sequential path (V1f − V3, then V3 − V4) is in the JSON; it differs from the Shapley split by at most 0.05 of a cell's loss, so the reading doesn't rest on the choice.
  - **"Never ruled out"** is unneeded work the run's own results gave no reason to stop, for example a method's long evaluation run without the cheap one that would have rejected it, or G1 Opus's cut GSM8K that was later rerun in full.
  - **The harness gap** is the online scheduler's distance from the optimum on the needed work.
- **What "ruled out" means.** It is the earliest moment the run's own evidence existed, never the reference plan's set. On G4 that means a method's failing workload and full_attention's of the same workload have both finished in full. On G4n (b) it means a SOAP launch's first step-time print. Every run is also checked for repeats: a record already finished in full.
- **What it is not.** It is a replay, not a re-run:
  - an agent paced by a better scheduler would decide at different moments;
  - cancellation here is instant, where the harness's own stop takes a tick;
  - every job is ready at 0, which holds because these packages have no data dependencies.

  The duration model is the harness's own, which transfers a method's measured pace across workloads. That holds on the simulator, which paces examples uniformly, and is an assumption on real hardware.
- **The selection effect (Opus finding 2) stands.** These five packages were kept because baselines failed on them, so the shares describe these packages, not agents' work in general.

## Checks

- **Known answers (`--selftest`): 40 of 40 pass in decompose.py** (shared_pool.py's own: 27 of 27).
  - Synthetic runs where one component is set by hand come out exact: delay alone 7 s, ordering alone 10 s, unneeded alone 10 s.
  - In a run where all three interact, the four parts sum to the loss, and both paths are checked by hand.
  - Also checked: a running job stopped mid-run; G4's rule-out, including the reference's own jobs; G4n (b)'s rejection at the step-time print; a repeat; the harness reducing to LPT on true priors; and CP-SAT's optimum of 6 where LPT gives 7.
  - Added 2026-11-02:
    - a job stacked onto a busy GPU is read from its hold or enqueue, and `busy_start` alone drops it;
    - a gpu −1 record names no GPU;
    - the agent's own placement (V1g) splits delay into idle 10 s and placement 10 s, by hand.
    - CP-SAT's optimum of {10, 6, 6, 6, 6} on 2 GPUs is 18, proved OPTIMAL: the longest job need not go on the most loaded machine.
- **Variant 1's idle matches shared_pool.py's on the same runs.** The two read the same loader. As of 2026-11-02, over 64 L0 runs, both give idle 629.2 GPU-h of 2,579.2. On 10-29 it was 538.5 of 2,137.2 over 53.
- **The floors are proved optimal, with the load-order break only.** 218 instances: 29 settled where LPT meets the lower bound, 189 proved OPTIMAL by CP-SAT. The package optima (G1 32,400 s, G4 13,455 s, G4n A (b) 19,328 s) sit at or below the expected files' packed makespans (32,476, 16,456 and 19,746 s), as an optimum should.
- **Each fixed job length is checked against the record.** In the 12 runs that shared a GPU, each full job's length is within 1% of its recorded solo length in the median, the worst job off by 13%. In runs that never shared, every job keeps its wall time exactly, as before.
- **The harness's own scheduler tests pass on the code replayed** (`tests/test_scheduler.py`, 21 of 21).
