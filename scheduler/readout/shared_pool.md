# A shared pool recovers the idle, not the unneeded work

the scheduler, 2026-10-29, for the analysis (the study-approved). Code: `shared_pool.py` beside this
note. Run: `python3 -I shared_pool.py --json out.json` from this directory; `--selftest` for the known answers. The
tables below are its output on the valid L0 runs in `~/computewise-runs` as of 2026-10-29 (screen.py's selection).

**Corrected 2026-11-02, the conclusion unchanged.** The tables below are 10-29's 53 runs, read with two loader defects:
- jobs stacked onto a busy GPU were dropped;
- a shared GPU was split in equal slices.

Both are fixed, and `decompose.md`'s correction section has the details. Rerun on the 64 L0 tenants valid on 11-02, with the fix, at 4 GPUs per tenant:
- **Split:** needed 0.616, unneeded 0.134, idle handed to a backlog 0.251.
- **Useful:** 0.866, rising to 1.000 with unneeded jobs removed.
- **Lost:** 0.38 of the box. A pool recovers 0.24 of it, and unneeded work is 0.13 (0.36 of the loss).

The one cell that moves visibly is H5 for Astra and Sol: lost 0.34 → 0.21 of the box, all idle either way. The hypothesis, and every reading below, hold.

## The question, and the answer

Would a shared cluster with a good scheduler recover the GPU time our agents waste? **It recovers the part the agents
left idle, and none of the part they spent on work the answer didn't need.** A scheduler packs a queue; it cannot know
that a job's result will be thrown away, so an unneeded job occupies a GPU exactly as a needed one does. The only thing
that reclaims unneeded GPU-time is judgement — not launching the job — and the counterfactual row measures that
separately. The two are disjoint: idle is the scheduler's to recover, unneeded is judgement's, and together they are the
whole loss.

**How much of the loss that is depends on the package.** Each tenant's own box (4 GPUs over its window) loses capacity
two ways: idle, which a pool can hand to other work, and unneeded work, which it can't. As shares of box capacity:

| tenants | n | lost (idle + unneeded) | idle: a pool recovers it | unneeded: a pool can't | unneeded share of the loss |
|---|---|---|---|---|---|
| G1, Claude | 8 | 0.31 | 0.31 | 0.00 | 0.00 |
| G1, Astra and Sol | 3 | 0.31 | 0.31 | 0.00 | 0.00 |
| H5, Claude | 6 | 0.33 | 0.33 | 0.00 | 0.00 |
| H5, Astra and Sol | 3 | 0.34 | 0.34 | 0.00 | 0.00 |
| G4, Claude | 6 | 0.80 | 0.28 | 0.53 | 0.66 |
| G4, Astra and Sol | 3 | 0.53 | 0.51 | 0.01 | 0.02 |
| G4n A, Claude | 8 | 0.31 | 0.15 | 0.16 | 0.52 |
| G4n A, Astra and Sol | 5 | 0.34 | 0.12 | 0.22 | 0.66 |
| G4n B, Claude | 7 | 0.32 | 0.12 | 0.19 | 0.61 |
| G4n B, Astra and Sol | 4 | 0.34 | 0.11 | 0.23 | 0.67 |
| all, Claude | 35 | 0.41 | 0.25 | 0.15 | 0.38 |
| all, Astra and Sol | 18 | 0.35 | 0.25 | 0.10 | 0.29 |
| all | 53 | 0.39 | 0.25 | 0.14 | 0.35 |

Three readings, none of which the pooled row alone shows:

- **G1 and H5 have no unneeded work** (their reference plan is the full 40-evaluation matrix, so only a repeat counts,
  and almost none occur). All of their loss is idle, and a pool recovers all of it. On these two the hypothesis is true
  but empty: there is nothing a pool fails to recover.
- **G4 is where the models split.** Claude loses 0.80 of its box, two-thirds of it unneeded evaluations the gate rule
  didn't need; a pool recovers only the idle third. Astra and Sol lose 0.53, almost all of it idle, so a pool recovers
  nearly all of theirs. The pool narrows the Claude-vs-Astra gap on idle and leaves the unneeded part of it intact.
- **On G4n every model wastes alike.** Astra and Sol are not the low-waste contrast here: their unneeded share of the
  loss (0.66, 0.67) is as high as Claude's or higher. Every model ran the rejected setup's jobs, and the reference plan
  counts those in full. So on G4n a pool recovers well under half the loss for every model.

Pooled, a pool recovers 0.25 of the box's capacity out of a 0.39 loss, about two-thirds; the third it can't is the
unneeded work.

**By model, on the packages with unneeded work**. This is the same split
as above, with Haiku, Sonnet and Opus separately, plus Astra in Claude Code (`astra-cc`) on G4 only, as a harness
contrast. Run with `--models haiku,sonnet,opus,astra,sol,astra-cc`. The split is per tenant, on its own box, so adding
astra-cc moves no other row.

| package | model | n | lost (idle + unneeded) | idle: a pool recovers it | unneeded: a pool can't | unneeded share of the loss |
|---|---|---|---|---|---|---|
| G4 | haiku | 3 | 0.83 | 0.34 | 0.49 | 0.59 |
| G4 | sonnet | 2 | 0.76 | 0.16 | 0.60 | 0.79 |
| G4 | opus | 1 | 0.80 | 0.23 | 0.57 | 0.71 |
| G4 | astra | 2 | 0.52 | 0.51 | 0.02 | 0.03 |
| G4 | sol | 1 | 0.53 | 0.53 | 0.01 | 0.01 |
| G4 | astra-cc | 1 | 0.56 | 0.52 | 0.04 | 0.07 |
| G4n A | haiku | 3 | 0.29 | 0.19 | 0.10 | 0.34 |
| G4n A | sonnet | 3 | 0.29 | 0.11 | 0.18 | 0.62 |
| G4n A | opus | 2 | 0.35 | 0.13 | 0.22 | 0.63 |
| G4n A | astra | 3 | 0.35 | 0.12 | 0.22 | 0.65 |
| G4n A | sol | 2 | 0.34 | 0.11 | 0.23 | 0.67 |
| G4n B | haiku | 3 | 0.33 | 0.10 | 0.23 | 0.68 |
| G4n B | sonnet | 3 | 0.33 | 0.14 | 0.19 | 0.57 |
| G4n B | opus | 1 | 0.22 | 0.13 | 0.10 | 0.43 |
| G4n B | astra | 3 | 0.34 | 0.11 | 0.23 | 0.67 |
| G4n B | sol | 1 | 0.34 | 0.12 | 0.22 | 0.66 |

- **On G4, all three Claude models waste alike.** Unneeded work is 0.49–0.60 of the box for each. Their unneeded
  share of the loss differs (0.59 Haiku, 0.71 Opus, 0.79 Sonnet), but that comes from idle, not from waste: Haiku also
  idles more (0.34 against Sonnet's 0.16), which dilutes its share.
- **The harness contrast on G4 says the low waste is the model's.** Astra in Claude Code wastes 0.04 of its box, like
  Astra in Codex (0.02) and Sol (0.01), and nothing like the Claude models' 0.49–0.60. On the same harness, Astra stays
  low and Claude stays high. This rests on one astra-cc run.
- **On G4n, no model ordering holds within Claude.** Unneeded work runs 0.10–0.23 of the box. Haiku is lowest on A and
  highest on B; Opus is the reverse, on one run for B. Every Claude model falls in the band Astra and Sol share
  (0.22–0.23). With n = 1–3 a cell, the G4n differences within Claude are not a finding. On a pool of 4 GPUs per tenant (each tenant a box's worth, the no-contention case), the tenants'
capacity over their own windows splits:

| | share of capacity |
|---|---|
| needed work | 0.608 |
| unneeded work | 0.138 |
| idle, handed to a backlog tenant | 0.254 |

"Useful" — needed plus what a backlog tenant reclaims from idle — is **0.862**; the missing 0.138 is exactly the unneeded
work, untouched by the pool. Remove each tenant's unneeded jobs (the counterfactual) and useful reaches **1.000**: the
backlog share rises from 0.254 to 0.383, a gain of 295.5 GPU-h, which is precisely the unneeded GPU-h. That identity —
the pool's useful ceiling moves only by what judgement removes — is the finding.

A bigger pool does not shorten any single tenant's time to its deliverable. At 4 GPUs per tenant every tenant's
deliverable ratio is 1.000 (median; max 1.021 from within-box GPU sharing): the deliverable is gated by the agent's own
think time between launches, not by GPU availability, so free GPUs it isn't waiting on don't help it. A *smaller* pool
only hurts: at 2 GPUs per tenant the median deliverable stretches to 1.58×.

## What is modelled, and what is not

A **replay**, not a multi-tenant experiment. Recorded single-tenant runs are placed as tenants of one pool; the agents
are not re-run, so this does not capture how an agent would behave on a shared cluster — it might queue differently, or
see other tenants' load. Specifically not modelled, and each makes the pool look better or contention worse than reality:

- **Arrivals are simultaneous.** Every tenant starts at 0, the worst alignment for contention; staggered arrivals would
  queue less.
- **Delays carry in full.** When the pool delays a job, the delay is added to the jobs the agent launched after seeing
  it. An agent that was busy thinking when a job ended would have absorbed part of the delay, so the contention
  deliverables (c < 4) are an upper estimate.
- **A shared GPU is divided by each job's own work** (corrected 2026-11-02). Where a run's own box ran two or more jobs
  on one GPU, gpu-mock served them first in, first out, so wall time says little about each job's length. Each job's
  length here is its share of the stretch, in proportion to its own work: the daemon's unit costs, which match a full
  job's recorded solo length to a median 0.1%. Until 2026-11-02 the split was equal slices, and jobs stacked onto a busy
  GPU were dropped outright (lane OBS's finding). Fixing both moved only the runs that shared a GPU. No job ever spanned
  two GPUs, so every job is 1 GPU wide.

What is faithful: each job's length and its needed/unneeded label; each job submitted at its recorded launch time
relative to the run's start, so the agent's think time and sleeps are unchanged; the scheduler (EASY backfill, FCFS as
the floor — identical here, since nothing blocks a 1-GPU job while a GPU is free); and a preemptible backlog tenant that
takes every idle GPU-second and yields at once to a tenant, the most any pool can hand out.

**Needed vs unneeded** is tick_decomposition's split: a job is needed when its record is in the package's brief-only
reference plan (`reference_records(task)`), unneeded otherwise — a repeated evaluation, a rejected setup's run, an
off-plan probe. **Deliverable** is screen.py's time to the deliverable (`over_ref` × the package's W0 reference), the
later of the final report and the work's end.

## Per cell (and pooled), 4 / 3 / 2 GPUs per tenant

n is the run count. "box GPU-h" is 4·W summed over the cell's tenants (W = max of the deliverable time and the last job's
end). Idle's share is of that box capacity; "tail" is idle after the last job ended. Deliverable columns are the pool
time over the own-box time, median [max] across the cell's tenants. The last column removes unneeded jobs at 2/tenant.

| tenants | n | box GPU-h (4·W) | needed | unneeded (share of busy) | idle (share of box; of it, tail) | deliverable at 4/tenant: median [max] | deliverable at 3/tenant: median [max] | deliverable at 2/tenant: median [max] | at 2/tenant, unneeded removed |
|---|---|---|---|---|---|---|---|---|---|
| G1 haiku | 3 | 150.7 | 108.4 | 0.0 (0.00) | 42.3 (0.28; 2.3) | 1.007 [1.008] | 1.266 [1.273] | 1.605 [1.628] | 1.391 [1.447] |
| G1 sonnet | 3 | 163.7 | 108.3 | 0.0 (0.00) | 55.4 (0.34; 4.8) | 1.006 [1.006] | 1.245 [1.261] | 1.596 [1.635] | 1.369 [1.415] |
| G1 opus | 2 | 115.0 | 79.5 | 0.0 (0.00) | 35.5 (0.31; 3.7) | 1.000 [1.000] | 1.079 [1.100] | 1.357 [1.452] | 1.209 [1.252] |
| G1 astra | 3 | 158.0 | 108.3 | 0.0 (0.00) | 49.6 (0.31; 0.4) | 1.006 [1.008] | 1.310 [1.345] | 1.690 [1.753] | 1.475 [1.519] |
| H5 haiku | 3 | 150.8 | 110.3 | 0.0 (0.00) | 40.6 (0.27; 0.3) | 1.005 [1.006] | 1.207 [1.262] | 1.577 [1.607] | 1.357 [1.378] |
| H5 sonnet | 2 | 105.6 | 72.2 | 0.0 (0.00) | 33.4 (0.32; 3.0) | 1.005 [1.006] | 1.193 [1.205] | 1.500 [1.501] | 1.338 [1.359] |
| H5 opus | 1 | 69.3 | 36.1 | 0.0 (0.00) | 33.1 (0.48; 11.1) | 1.003 [1.003] | 1.143 [1.143] | 1.436 [1.436] | 1.267 [1.267] |
| H5 astra | 3 | 138.5 | 91.3 | 0.0 (0.00) | 47.2 (0.34; 13.4) | 1.000 [1.001] | 1.073 [1.167] | 1.314 [1.576] | 1.135 [1.420] |
| G4 haiku | 3 | 162.0 | 28.2 | 78.7 (0.74) | 55.1 (0.34; 0.1) | 1.009 [1.009] | 1.267 [1.345] | 1.656 [1.840] | 1.351 [1.458] |
| G4 sonnet | 2 | 73.4 | 17.4 | 44.0 (0.72) | 12.0 (0.16; 0.0) | 1.003 [1.006] | 1.285 [1.435] | 1.761 [1.930] | 1.476 [1.604] |
| G4 opus | 1 | 44.8 | 9.1 | 25.5 (0.74) | 10.3 (0.23; 0.1) | 1.000 [1.000] | 1.095 [1.095] | 1.385 [1.385] | 1.135 [1.135] |
| G4 astra | 2 | 36.6 | 17.4 | 0.6 (0.03) | 18.6 (0.51; 0.2) | 1.018 [1.021] | 1.493 [1.497] | 2.282 [2.449] | 1.985 [2.050] |
| G4 sol | 1 | 18.5 | 8.7 | 0.1 (0.01) | 9.7 (0.53; 0.1) | 1.012 [1.012] | 1.524 [1.524] | 2.386 [2.386] | 2.026 [2.026] |
| G4n A haiku | 3 | 89.5 | 63.2 | 9.0 (0.12) | 17.3 (0.19; 1.4) | 1.000 [1.000] | 1.214 [1.242] | 1.640 [1.767] | 1.392 [1.408] |
| G4n A sonnet | 3 | 90.8 | 64.6 | 16.2 (0.20) | 10.0 (0.11; 0.6) | 1.000 [1.000] | 1.127 [1.128] | 1.519 [1.527] | 1.350 [1.353] |
| G4n A opus | 2 | 65.2 | 42.2 | 14.4 (0.25) | 8.6 (0.13; 1.5) | 1.000 [1.000] | 1.150 [1.158] | 1.568 [1.600] | 1.319 [1.327] |
| G4n A astra | 3 | 96.7 | 63.3 | 21.6 (0.25) | 11.8 (0.12; 0.1) | 1.000 [1.000] | 1.154 [1.161] | 1.534 [1.562] | 1.299 [1.327] |
| G4n A sol | 2 | 63.6 | 42.2 | 14.4 (0.25) | 7.0 (0.11; 0.2) | 1.000 [1.000] | 1.164 [1.172] | 1.646 [1.652] | 1.342 [1.344] |
| G4n B haiku | 3 | 95.2 | 63.6 | 21.6 (0.25) | 10.0 (0.10; 0.7) | 1.000 [1.000] | 1.116 [1.195] | 1.460 [1.618] | 1.317 [1.369] |
| G4n B sonnet | 3 | 94.4 | 63.3 | 17.9 (0.22) | 13.2 (0.14; 1.4) | 1.000 [1.000] | 1.109 [1.147] | 1.428 [1.621] | 1.223 [1.295] |
| G4n B opus | 1 | 27.2 | 21.1 | 2.6 (0.11) | 3.5 (0.13; 0.1) | 1.000 [1.000] | 1.150 [1.150] | 1.548 [1.548] | 1.323 [1.323] |
| G4n B astra | 3 | 95.4 | 63.3 | 21.6 (0.25) | 10.4 (0.11; 0.1) | 1.000 [1.000] | 1.155 [1.164] | 1.560 [1.562] | 1.322 [1.328] |
| G4n B sol | 1 | 32.0 | 21.1 | 7.2 (0.25) | 3.7 (0.12; 0.1) | 1.000 [1.000] | 1.169 [1.169] | 1.625 [1.625] | 1.355 [1.355] |
| pooled, Claude | 35 | 1,497.8 | 887.6 | 229.9 (0.21) | 380.4 (0.25; 31.3) | 1.000 [1.009] | 1.181 [1.435] | 1.562 [1.930] | 1.347 [1.604] |
| pooled, Astra and Sol | 18 | 639.4 | 415.7 | 65.6 (0.14) | 158.1 (0.25; 14.6) | 1.000 [1.021] | 1.166 [1.524] | 1.600 [2.449] | 1.342 [2.050] |
| pooled, all | 53 | 2,137.2 | 1,303.3 | 295.5 (0.18) | 538.5 (0.25; 45.9) | 1.000 [1.021] | 1.167 [1.524] | 1.576 [2.449] | 1.344 [2.050] |

Astra-cc, Fable and Grok are left out (`--models` adds them; one to three runs a cell). The "pooled, Astra and Sol" row's
low unneeded share, 0.14, comes from G4 alone. On G4n their share is 0.25, the same as Claude's (see the loss table above).

## The pool, over the tenants' own windows

Capacity here is c GPUs per tenant from 0 to that tenant's end in the pool (not to the common horizon H). The pool also
holds the GPUs of tenants that have already left — simultaneous arrivals, none later — which is no agent's idle and is
shown in the last column, apart.

| pool | GPUs | tenants' capacity, c × their windows (GPU-h) | needed | unneeded | backlog in those windows | used by tenants | useful (needed + backlog) | tenant jobs' queue wait (GPU-h) | free after tenants left (GPU-h) |
|---|---|---|---|---|---|---|---|---|---|
| 4/tenant | 212 | 2,143.1 | 1,303.3 (0.608) | 295.5 (0.138) | 544.4 (0.254) | 0.746 | 0.862 | 3.6 | 1,965.9 |
| 4/tenant, unneeded removed | 212 | 2,113.4 | 1,303.3 (0.617) | 0.0 (0.000) | 810.1 (0.383) | 0.617 | 1.000 | 0.7 | 1,995.6 |
| 3/tenant | 159 | 1,911.1 | 1,303.3 (0.682) | 295.5 (0.155) | 312.3 (0.163) | 0.837 | 0.845 | 337.3 | 1,347.5 |
| 3/tenant, unneeded removed | 159 | 1,692.7 | 1,303.3 (0.770) | 0.0 (0.000) | 389.4 (0.230) | 0.770 | 1.000 | 112.2 | 1,501.8 |
| 2/tenant | 106 | 1,671.0 | 1,303.3 (0.780) | 295.5 (0.177) | 72.2 (0.043) | 0.957 | 0.823 | 1,082.0 | 965.9 |
| 2/tenant, unneeded removed | 106 | 1,425.2 | 1,303.3 (0.914) | 0.0 (0.000) | 121.9 (0.086) | 0.914 | 1.000 | 600.5 | 972.4 |

Read the "useful" column down: at every pool size, with unneeded work present, useful is capped below 1 by the unneeded
share; remove it and useful is 1.000. Tenant jobs never queue at 4/tenant (3.6 GPU-h, from within-box GPU sharing); the
queue wait grows as the pool shrinks, which is the contention that stretches the deliverables above.

## Checks

- **Needed/unneeded reproduces the analysis' 11:55 MDT split.** G4 unneeded share of busy GPU-time: Claude
  (haiku 0.751, 0.749, 0.695; sonnet 0.712, 0.721; opus 0.757) in 0.69–0.75; Astra 0.011, 0.053 and Sol 0.011 in
  0.01–0.05. Printed by the run.
  - **Corrected 2026-11-02:** with stacked jobs no longer dropped, Haiku r3 reads 0.719 (it was 0.695) and Opus r2 0.718 (it was 0.757). Every Claude run now sits in 0.71–0.75, and codex's fixed `tick_decomposition.py` gives the same 0.719.
- **The c = 4 accounting identity holds to 0.000 GPU-s:** backlog = tenants' own idle + the capacity of tenants that left
  before H. Printed each run.
- **EASY and FCFS give identical deliverables for all 53 tenants** (every job 1 GPU wide), as expected; the selftest
  separates them on a constructed wide job.
- **Known answers (`--selftest`, needs no runs):** GPU-share attribution; two synthetic tenants with hand-computed idle
  gaps whose recovered GPU-h (3,900, then 4,600 with unneeded removed) come out exact; a contention case where a
  dependent job's delay carries (deliverable 260, backlog 230); and the EASY-vs-FCFS backfill of a 1-GPU job under a wide
  job's reservation (starts at 2 s under EASY, 110 s under FCFS). All pass.
