---
name: gpu-harness
description: Use whenever a task runs several GPU (or other scarce-resource) jobs to reach a decision or a deliverable — evaluations, sweeps, candidate comparisons, acceptance rules, benchmark tables, training runs. You write the job graph and make every judgement; the `gpuh` harness owns the GPUs, schedules the jobs, launches each in its own process group, watches the machine, reads each job's pace from its log, and cancels a rejected candidate's whole subtree. Branch teammates judge their candidates from one-line events. No one sleeps or polls.
---

# GPU harness

The aim is the answer as early as possible, with every GPU running a job the answer needs. **You decide; the harness does the mechanics.** It is `bin/gpuh` in this skill's directory (`<skill dir>/bin/gpuh`; `gpuh --help` lists everything). It is plain Python, already here; nothing to install.

What the harness does, so you don't:
- **Owns the GPUs.** One job per GPU, assigned by its scheduler; you never pick a GPU or run a job in your shell.
- **Schedules by the decision's critical path.** The reference's checks go first, then every runnable job by the longest chain of work that waits on it, so checks that gate a long run start before shorter independent work. Needed work waits for its candidate's checks to pass; nothing runs for a rejected candidate.
- **Launches each job in its own process group,** with `CUDA_VISIBLE_DEVICES` set, logs under the run dir, and the exit code recorded by the job's own shell.
- **Watches the machine, not notes.** A job is running while its process group lives. One that vanishes without an exit code is reported `DIED`.
- **Reads pace from the log.** It recognises `step:N/M … train_time`, `step N/M … step_time`, tqdm bars and plain `N/M` counters, projects each job's whole length, and compares a candidate's job with the reference's job of the same kind.
- **Cancels whole subtrees.** `gpuh reject` stops every job of the candidate, its child processes included, and frees each GPU only once the whole group is gone.
- **Wakes the right agent.** Each event goes to the branch that owns the job; a `gpuh wait` returns only when there's something to decide.

**Decide how to run the work yourself, and start it, through the harness.** Questions about how to run the jobs are yours: which seeds when a script has none, how to read a README's run list, what counts as the same kind of work. Make the sensible choice, write it in the plan with the reason, and go on. Stop to ask only if the brief contradicts itself or the work is impossible.

## 1. You: write the plan

Read the brief and the files, then write `plan.json` (absolute paths):

```json
{
  "rule": "the rule that decides the answer, in the brief's own words",
  "evidence": "what the rule accepts: e.g. whole workloads (--max-examples 0); three seeds; a run's measured step time",
  "workdir": "/abs/path/to/project",
  "candidates": [
    {"id": "baseline", "branch": "ref", "reference": true},
    {"id": "methodA", "branch": "a"},
    {"id": "methodB", "branch": "b"}
  ],
  "jobs": [
    {"foreach": {"m": ["baseline", "methodA", "methodB"], "w": ["qa", "code"]},
     "id": "{m}_{w}", "candidate": "{m}", "kind": "{w}",
     "command": "bash methods/{m}/scripts/{w}.sh",
     "estimate_s_by": {"qa": 600, "code": 3600},
     "role_by": {"qa": "check"}}
  ]
}
```

- **Candidates:** one per option the decision is about. Mark the comparison baseline `"reference": true`. Give each candidate a `branch`: the teammate that judges it (one per candidate, at most four; group similar ones). The reference may have its own branch or share main's.
- **Jobs:** every run the rule could need, each one command on one GPU. `foreach` expands a template over lists.
  - `role`: `check` for the cheapest observation of a kind the rule accepts that could settle the candidate (its shortest whole workload; its first seed, whose pace is read minutes in). `needed` for the rest of what the rule needs. `speculative` for nice-to-have work that should only use idle GPUs. **A needed job of a candidate waits until each of that candidate's checks is passed**; a reference's jobs never wait.
  - `kind`: the workload, so the harness can compare a candidate with the reference on the same work and carry a measured slowdown to its other jobs.
  - `estimate_s`: your best prior length, from the job's own settings (examples × time per example, steps × time per step). Read every job's settings. The harness corrects it from measurements.
  - `after`: jobs that must finish first (a checkpoint, a data file). `needs_pass`: to choose exactly which checks gate a job (`[]` for none).
  - `screen: true` on anything less than the rule's evidence (a slice of a workload, one seed of three). A screen may order work; **the harness refuses a rejection that rests only on screens**.
  - `p_reject` (optional, 0–1) on a check: your belief it rules the candidate out. Among one candidate's checks, the likeliest to fail per second runs first.
  - `milestones` (optional): a value the rule reads before a job ends, e.g. the loss at step 1000 of a longer run: `"milestones": [{"name": "step1000", "pattern": "step:1000/\\d+ val_loss:(?P<value>[\\d.]+)"}]`. When the running job's log first matches, you get a `MILESTONE` line with the value and the reference's value beside it, while the job still runs. **If the rule reads a value partway through a run, declare it as a milestone**: otherwise you see it only when the job ends. If the rule compares a mean with the reference's, add which way is better and the margin it needs, `"better": "lower", "margin": 0.005`: the line then says what the candidate's other runs would have to average to pass, and how many run-to-run sds away that is.
  - `"gpu": 0` for a job that needs no GPU: data preparation, tokenising, a CPU-only metric. It runs beside the GPU jobs (at most `policy.cpu_slots` at once, default 4) and never holds a GPU, so training can use every GPU meanwhile. Give each CPU job its share of the machine's cores, `nproc` divided by the CPU jobs that will run at once (all of them with `cpu_slots: 1`), through the tool's own options: as many workers as that share, and, if the tool splits its input into tasks, shards or chunks, at least as many of those as workers, since it runs no more at once than it has. `gpuh check` prints the share. Jobs that use its output still say `after`. When several preparations each gate their own GPU runs, a preparation given all the cores and run one at a time (`policy.cpu_slots: 1`) usually finishes first, so its runs start sooner than if all the preparations share the cores.
- Mark the checks inside a template with `role_by` (above: every `qa` job is a check, every other job is needed). `except` leaves combinations out: `[{"m": "baseline", "w": "needle"}]`. Jobs can also be listed one by one.
- Check it: `<skill dir>/bin/gpuh check plan.json`. Fix what it refuses, and act on its notes.

## 2. Start the harness and the team

**Start every branch as its own teammate, however small the task looks.** You never judge a branch's candidate and never run `gpuh --as <branch> …` yourself: a branch's judgements, waits and context are its own, which is what keeps yours small enough to last the whole task. These are your next tool calls, in this order:
1. Bash: `<skill dir>/bin/gpuh start /abs/path/plan.json` (it detects the GPUs; or add `--gpus 0,1,2,3`). It returns at once; the first jobs are already running.
2. For each branch, Agent tool, `name: "<branch>"`, `run_in_background: true`, prompt:
   > You are branch `<branch>` of a gpu-harness team. Read `<skill dir>/SKILL.md`, section "3. Branch teammates", and follow it exactly. The rule: `<rule>`. The evidence it accepts: `<evidence>`; nothing less rejects a candidate. You judge: `<its candidates>`. The harness is `<skill dir>/bin/gpuh`. Your first tool call: `<skill dir>/bin/gpuh wait --for <branch>` as a background Bash task.
3. Bash, `run_in_background: true`: `<skill dir>/bin/gpuh wait --for main`. Then end your turn.

**You never start, stop or inspect jobs yourself.** When your wait returns, read its lines, act, and **start the wait again before you end your turn** (only `ALL_DECIDED` or `FINISH` ends it). With no wait running, nothing can wake you:
- `REJECTED` / `DECIDED`: note it in `plan.md`. Start your wait again.
- `NOWAITER <branch>`: that branch has events and isn't waiting. If the teammate exists, SendMessage it one line: "run `gpuh --as <branch> wait --for <branch>` in the background"; if it never started or has ended, start it again as above.
- `ALL_DECIDED`: run `gpuh results`, write the answer, run `gpuh finish`, and report.
- A teammate's turn-end notice is not an event: end your turn at once with one word unless your own wait returned. **If your context is ever compacted, run `gpuh status` and `gpuh results`; everything is there.**

## 3. Branch teammates

You judge your candidates; the harness runs their jobs. **When your wait returns, read its output first (the Read tool, or `cat` in the foreground), act on every line, and only then start the next wait by itself;** never fold the reading into the new background command, or you will never see what it says. Lines marked `STILL OPEN` are decisions you have not made yet; make them, and don't message main about them. Nothing asks you for a decision the evidence can't support yet: if you are waiting for the reference, the harness waits too. Every turn ends the same way: start `<skill dir>/bin/gpuh --as <branch> wait --for <branch>` as a background Bash task, then end your turn. When it returns, act on its lines, then start it again. Never sleep or poll; the wait wakes you.

- **`PROGRESS <job> … projected <length> … reference … -> x<ratio>`:** a running job's measured pace (sent for a check, and for each candidate's first job once the reference has a pace). **For a time, speed or budget rule, this line is the rule's own measurement, read from the real run:** a step time is a step time whether the run has finished or not. If it puts the candidate clearly over the limit, reject now, citing the job; waiting for the run to end only burns its GPU. Otherwise do nothing.
- **`DONE <job> … | reference <job>: …`:** a job of your candidate finished, with the reference's result beside it when there is one, and both lengths compared on one clock each (`timer … -> x…; wall … -> x…`). For a time rule, use those ratios: the line's lead is wall time and a result line's `train_time` is the script's own timer, so never divide one by the other. **Every result can settle a candidate, not only its checks.** Compare it with the rule at once:
  - it fails the rule: `gpuh --as <branch> reject <candidate> --why '<the observation, in the rule's terms>' --evidence <job>,<reference job>`. This stops every other job of the candidate now, which is the point: the line says how many are still queued or running.
  - it passes the rule: if it's a check with gated jobs waiting, `gpuh --as <branch> pass <job>` at once; otherwise nothing.
- **`MILESTONE <job> <name> = <value> … | reference <name>: …`:** a value the rule reads has appeared in a running job's log. Judge it as you would a `DONE`, now: if it fails the rule, reject, citing the job, which stops it and every other job of the candidate. For a mean rule the line also gives the average the candidate's other runs would need, and the run-to-run sd: the gap in sds is the number the rule turns on (below). Until three runs of one method are in it says the sd is not measured yet, and the line comes again when it is. Never use the differences between candidates as the spread: they are differences between methods, often ten times larger.
- **`JUDGE`:** the reference's number arrived for a result of yours that finished before it. Judge it as above.
- **A silent job is not a hung one.** Many evaluations print nothing between loading the model and their final line, for hours. Before you call a job stuck, run `gpuh status` (or `gpuh log <job>`): for each running job whose log has been quiet for 10 min, it reads the job's processes and says `working` (they are using CPU) or `STUCK?` (they used none). `gpuh cancel` refuses a silent job that is working, unless you pass `--force --why '...'`. A job the harness reports `DIED` or `FAILED` has really stopped; one that is silent and working has not.
- **`FAILED` / `DIED`:** read `gpuh log <job>`. If the failure is the job's fault (a bug, a missing file) and you can fix it, `gpuh retry <job> [--command '...']`; if it's the candidate's fault under the rule, reject it with that reason.
- **`IDLE <candidate>`:** every job has ended and every reference result it's compared with is in, or (`nothing can start`) its jobs wait on something that will never come, a cancelled dependency or gate, which the line names; then fix that (`gpuh retry`, `gpuh cancel`, `gpuh add`) or decide. To decide, now, without waiting for anything else: `gpuh --as <branch> decide <candidate> --verdict pass|fail --why '<the results it rests on>' --evidence <jobs>`. If the rule needs more runs, `gpuh add fragment.json` (same format as the plan's `jobs`), then wait.
- **`RETIRE`:** everything you judge is decided. End your turn with one line, and don't start another wait.
- **A whole run can settle a mean rule.** If the rule averages several runs and you hold one run that is the rule's own evidence (not a screen), compare it with the margin the rule needs, against the run-to-run spread: the reference's runs, or the candidate's own. If the other runs would each have to be more than 3 sd better than this one for the mean to pass, reject now, citing it; running the rest only confirms it. **If it is close (within a few spreads of the margin), run the rest:** that is what the other runs are for. If no spread is measured yet, do not estimate one from other candidates: wait for the line that carries it (with milestones declared, the harness runs the reference's first three runs early).
- **Reject only on evidence the rule accepts.** A screen never rejects; the harness refuses it. Whether a job is a screen is the plan's call, and each `DONE` line says it (`[screen]` or `[the rule's evidence, per the plan]`): take it from there, never from a job's log or settings. A result that is the rule's evidence and fails the rule is a rejection, not a pass.
- Keep your turn's last text to one line; it is sent to main.

## 4. Rules everyone keeps

- **Only the harness runs and stops jobs.** Nobody runs a job command in a shell, nobody kills a process by pid, `ps` or name. To stop a job: `gpuh cancel <job>`.
- **Nobody waits by sleeping or polling.** `gpuh wait` is the only wait, as a background task.
- **Never ask a teammate to look at something later;** it acts only when its wait returns. Send it the line itself, by SendMessage.
- **Cheap before expensive, by the plan's roles:** a candidate's needed runs start only after its checks pass, unless the plan says otherwise. The scheduler orders everything else.

## 5. Finish

When `ALL_DECIDED` arrives (or every candidate is decided and you have every result the answer rests on), run `gpuh results`, then `gpuh finish`, and report at once: the answer, the results it rests on with each workload's size, and each rejection with the evidence that settled it. `gpuh export` writes a timeline of the run if you want to show it.

<!-- build: d3917f1d3a0a30cd -->
