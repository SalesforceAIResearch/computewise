---
name: gpu-team
description: Use whenever a task runs several GPU (or other scarce-resource) jobs to reach a decision or a deliverable — evaluations, sweeps, candidate comparisons, acceptance rules, benchmark tables, training runs. Organises the work as a team: you plan the graph, a GPU teammate owns the GPUs, a branch teammate owns each candidate, and every job runs in its own subagent that reports when it ends. No one sleeps or polls; cheap decisive checks run first; useless work is cancelled at once.
---

# GPU team

The aim is the answer as early as possible, with every GPU running a job the answer needs. Two things defeat that:
- **waiting badly:** a GPU sits free because nobody noticed a job ended;
- **running the wrong work:** a GPU is busy with a job the answer never needed.

The team below removes both. No one waits by sleeping, because every job reports its own end. No GPU goes to work that a cheaper check could rule out.

## The team

```
you (main session): plan the graph and the rule; keep plan.md; collect the answer
 ├─ gpus: the GPU teammate. Owns the ledger, grants GPUs in priority order, starts each granted job's subagent
 ├─ branch teammates: one per candidate (or group of similar candidates), plus one for shared reference work
 │     prepare their jobs, read every result, apply the rule, request and cancel their own jobs
 └─ job subagents: one per running job. Run it, wait in the background, report DONE or FAILED
```

**Decide how to run the work yourself, and start it, through the team below; that part is not a choice.** Questions about how to run the jobs are yours: which seeds when a script has none, how to time a job that prints no progress, how to read a README's run list. If you can name a sensible choice, make it, write it in `plan.md` with the reason, and go on. Report your choices with the answer. Stop to ask only if the brief contradicts itself or the work is impossible, never before the team is running.

**Build the team every time you use this skill, however small the task looks, and never run a job command in your own shell, not even the first one: no `nohup`, no `&`, no loop that waits on jobs.** The team is the method, not an option for big tasks. Jobs launched from the main session put every wait and every decision in one context, and that is where a free GPU goes unnoticed. The team costs a few seconds per job.

Create every teammate with the Agent tool, `run_in_background: true`, and a `name`. Agents talk by SendMessage, one line per message, and a message wakes an idle teammate. **A subagent's completion is announced to you, the main session, not to the teammate that started it.** So a job's subagent reports its own result by SendMessage, to `gpus` and to its branch. You can ignore those announcements; the branches report decisions.

## 1. You: plan

Before any launch, write `plan.md`:
- **The rule** that decides the answer, in the brief's own words.
- **The graph:** every job the rule could need, what it depends on, and a first estimate of its length. Mark which jobs are **needed whatever happens** (a baseline, a table the brief asks for in full) and which are needed **only if their candidate survives**.
- **What the rule accepts as evidence:** the exact observation it names, such as whole workloads, three seeds, a measured step time, or the full table. Only that evidence passes or rejects a candidate. For a time or budget rule, a real run's own pace, read from its log, is a measurement: when it puts the run clearly over the limit, it settles the rule without waiting for the run to end. Anything less is a screen: a subset of a workload, one seed of three, a guess from names or sizes, a shortened copy of a job. A screen may order which runs go first. It never rejects, because the rule can't act on it.
- **Each candidate's cheapest decisive check:** the fastest observation of a kind the rule accepts that could rule it out. Examples:
  - a run's step time in its first minutes, for a time or budget rule. If the job prints no progress line, its first validation or logging line usually carries the step and the time so far; watch for that. Time the real run, not a shortened copy, which is only an estimate;
  - the candidate's shortest whole workload, for a rule on whole workloads (a 20-example slice of a workload is not that workload);
  - the baseline, which every comparison needs. **Start it in the very first wave**: put its first run in `gpus`'s opening queue as `check`, ahead of the candidates' checks, and its other runs as `needed`. It runs alongside the candidates' first checks: no candidate can be judged until its number is in, so it never waits behind them. Equally, the candidates' first checks don't wait for the baseline to finish; they run beside it and are judged the moment its number arrives.
- **The branches:** one per candidate, at most four. Group similar candidates if there are more. Shared reference work gets its own branch. If the task has no candidates, only a set of jobs to run, one branch owns them all.

Write each job into `plan.md` with its exact command, its estimated length, its **chain**, and whether it's needed whatever happens. A job's chain is its own length plus the longest run of work that can only start after it ends. When a decision waits for a whole group (pick the best two of four screens), every member of the group carries the follow-up: a 120-s screen whose result picks a 90-s confirm has a chain of 210 s, and it goes ahead of an independent 150-s job. **Take first estimates from each job's own settings**, where the files give them: steps × time per step, examples × time per example. Read every job's settings, not one and assume the rest; jobs that look alike often differ.

**Then start the team.** These are your next tool calls, in this order. `<skill dir>` is the base directory this skill was loaded from.
1. Agent tool, `name: "gpus"`, `run_in_background: true`, prompt:
   > You are `gpus`, the GPU teammate of a gpu-team. Read `<skill dir>/SKILL.md`, section "2. gpus: the GPU teammate", and follow it exactly. The GPUs: `<ids>`. Jobs run in `<working directory>`; logs go to `<log directory>`. The plan is `<path>/plan.md`. The opening queue, already requested: `<every check, and every job needed whatever happens that can start now; each as job, priority, chain seconds (not its length), command, branch>`. Grant from it at once, in priority order.
2. For each branch, Agent tool, `name: "<branch>"`, `run_in_background: true`, prompt:
   > You are branch `<branch>` of a gpu-team. Read `<skill dir>/SKILL.md`, section "3. Branch teammates", and follow it exactly. The rule: `<rule>`. The evidence it accepts: `<from plan.md>`; nothing less rejects a candidate. You own: `<its candidates and jobs, each with command, estimate, chain, and needed-whatever-happens or only-if-it-survives>`. Already in `gpus`'s opening queue: `<which of them>`. The plan is `<path>/plan.md`. Report to the main session by SendMessage.
3. End your turn.

You're woken by the branches' decisions and by `gpus`, and also each time any teammate or job subagent ends a turn. **You never start or stop jobs yourself; that's `gpus`'s.** **Those turn-end notices aren't messages to act on.** Unless one is a branch's `REJECTED` or `DECIDED`, or a `gpus` problem report, end your turn at once with one word, without thinking it through. Your context has to last the whole task. Record each decision in `plan.md`. When every candidate is decided and every result the answer rests on is in, report. Don't report while any candidate is undecided, and don't stop to ask whether to run work the rule needs: run it.

## 2. gpus: the GPU teammate

`gpus` owns the GPUs. Nobody else starts GPU work.
- **The ledger:** each GPU's state in `gpus.md`, one job per GPU, always: `free`; `granted <job> <log> <agent id> <time>` once its subagent is started; `running <job> <log> pid <pid>` once the subagent confirms it. **The ledger is your notes, not the machine.** Every GPU the ledger calls busy must have a live process behind it.
- **The opening queue** comes from main with your instructions: grant from it at once.
- **Requests:** a branch sends `NEED <job> <priority> <chain seconds> <command> [watch=<pattern>]`. Priority is `check`, `needed` or `speculative`. **Chain seconds** is the job's own estimated length plus the longest chain of jobs that can only start after it ends. A screen whose result picks a 90-s follow-up has a chain of its own length plus 90 s.
- **Re-estimates:** a branch sends `UPDATE <job> <chain seconds>` when a measurement changes a waiting job's chain.
- **Grants**, whenever a GPU is free, in this order:
  1. `check` jobs: cheap decisive observations;
  2. `needed` jobs, longest chain first by the latest estimate. The job that gates the most remaining work goes first, because everything after it waits;
  3. `speculative` jobs, only when nothing else is waiting.
- **To grant, start the job's subagent yourself** (Agent tool, `run_in_background: true`) with the JOB prompt below. Only once that call returns its agent id, write `granted` in the ledger with that id. No agent id, no grant. Every grant is an Agent call: never run a job command in your own shell, not even as a placeholder.
- **On `STARTED <job> GPU <g> pid <p>`** from a job's subagent: mark the GPU `running` with that pid.
- **Check the machine every time you're woken, before you answer.** Run one command over every GPU the ledger calls busy: `for f in <each busy GPU's {log}.pid>; do printf '%s ' $f; P=$(cat $f 2>/dev/null); { kill -0 -- -$P || kill -0 $P; } 2>/dev/null && echo alive || echo GONE; done`. (The pid file names the job's process group; the leading `-` checks the whole group, children included.) Then:
  - **`granted` with no STARTED for over 60 s, or no pid file at all:** the grant was lost. Re-grant it at once, with a new Agent call.
  - **`running` but GONE, with no DONE or FAILED:** read the log's last lines yourself, send the branch `DONE` or `FAILED` from them, and free the GPU.
  Never believe "no GPU is free" without this check. A GPU your notes call busy while nothing runs on it is the costliest mistake you can make.
- **When a job's subagent reports DONE or FAILED:** mark its GPU free and grant the next job on that GPU at once. (The subagent tells the branch itself.)
- **On `CANCEL <job>`** from a branch, stop the job's whole process group: `P=$(cat <its log>.pid); kill -TERM -- -$P 2>/dev/null || { pkill -TERM -P $P; kill -TERM $P; }`. A job is often a script that starts the real program as a child, and killing only the script leaves that child running on the GPU. The GPU is free only once the whole group is gone: `{ kill -0 -- -$P || kill -0 $P; } 2>/dev/null` fails. If it doesn't within a few seconds, `kill -KILL -- -$P` and check again. Then mark it free and grant the next job at once. A waiting job is just dropped from the queue. The killed job's subagent then reports FAILED; that report frees nothing a second time.
- Answer every message in one line and end your turn with one short line of text. Never sleep or poll.

**JOB prompt** (fill in the braces):

> Run this as one background Bash task (`run_in_background: true`), exactly as written, with no `&`, `nohup` or `setsid`: `cd {dir} && echo $$ > {log}.pid && exec env CUDA_VISIBLE_DEVICES={gpu} {command} > {log} 2>&1`. Then confirm it started: `kill -0 $(cat {log}.pid) && cat {log}.pid`, and send `gpus` one line by SendMessage: `STARTED {job} GPU {gpu} pid <that pid>`. (`{log}.pid` holds the id of the job's shell, which leads the job's process group, so the job and every child it starts are cancelled together by `kill -- -<id>`. If you find you ran it without the `cd`, the job is fine as long as `pwd` is `{dir}`; check once.) **Start the job once.** Never kill, restart or relaunch it yourself; if something looks wrong, report it in your one line and let `gpus` decide. {only if the request had watch=<pattern>:} Then start a second background Bash task: `command grep -m1 -E '<pattern>' <(tail -n +1 -F --pid=$(cat {log}.pid) {log})`. Write `command grep` exactly: in Claude Code's shell a plain `grep` is a wrapper that waits for the job to end, so the line would arrive too late. When it ends with a match, send `{branch}` one line by SendMessage: `PROGRESS {job} <the matched line>`. {end} Then end your turn without sleeping or polling. When the job's task finishes, read the last lines of `{log}`. Send one line, by SendMessage, to `gpus` and to `{branch}` (the branch teammate's exact name): `DONE {job} GPU {gpu} <result line>` if it exited 0, else `FAILED {job} GPU {gpu} <last error line>`. Then end with the single word `sent`.

`{command}` must be one command, because of the `exec`; put anything longer in a script first.

With a scheduler (`sbatch` present), the JOB prompt runs `sbatch --wait …` as its one background task instead, and the scheduler assigns the GPU. `gpus` still holds the priorities and the order of submission.

## 3. Branch teammates

Each branch owns one candidate, or a small group, and thinks only about it.
- **Prepare** its jobs: scripts, configs, data, commands. Do it before they're needed, while other jobs run.
- **Its cheapest decisive check, and its jobs needed whatever happens, are already in `gpus`'s opening queue** (main put them there, so `gpus` sees the whole set at once and orders it). Request any such job main missed at once, as `needed`; never hold one back for checks, because the longest must not start late.
- **Jobs needed only if the candidate survives:** request them once it has passed two checks (or its only one).
- **Estimate lengths from measurements, and keep them current.** A job's pace belongs to its method and its input together, so a finished job of another method or input isn't its measurement. Measure the method's slowdown against the reference on a short input, and scale the reference's length on the long input by it. When jobs print no progress, time one short job of each kind first. Send `UPDATE` whenever a measurement moves a waiting job's estimate.
- **On every result, apply the rule at once:**
  - **Rejected, on evidence the rule accepts:** send `gpus` a `CANCEL` for every job of the candidate, running or requested. Then tell main: `REJECTED <candidate> <the observation>`. No job you know is useless keeps running.
  - **A screen looks bad:** it doesn't reject. Use it to order the candidate's next runs (request its likeliest failure first), never to skip them.
  - **Passed a check:** request the next job the rule needs.
  - **Decided:** tell main `DECIDED <candidate> <verdict> <the results it rests on>`.
- **For a time or budget rule, read speed early, from the running job itself.** Add `watch=<pattern>` to the NEED, a regex matching the job's first progress or validation line (one with its step and its time so far). Read the script's own print statement first and match it loosely, with `.*` between fields: `step:[0-9]+/[0-9]+.*train_time` matches `step:125/2900 val_loss:4.45 train_time:344s`, while `step:[0-9]+/2900 train_time` never does. The job's subagent sends you `PROGRESS <job> <line>` the moment that line appears, which is minutes into a run of hours. Apply the rule to it at once: keep the job, or cancel it and every sibling the same reading condemns.
- Answer in one line, end your turn with one short line of text, and never sleep or poll. You're woken by messages. **Your last text in every turn is sent to main**, so keep it to one line; put detail in `plan.md`, not in your turn's last message.

## 4. Rules everyone keeps

- **Only `gpus` stops a job, and only through its pid file, as a whole process group.** Nobody else kills anything, main included. Nobody kills a process found by `ps`, `pgrep` or a name, even one that looks like a stray copy of a job: on a shared machine it may belong to someone else. If you see something wrong, tell `gpus` in one line.
- **Never ask a teammate to look at something later.** A teammate acts only when a message arrives, so "read the log when it gets to step 250" is never done. Send it the line itself, or set up the thing that will send it. If you find a `watch` can't fire (its pattern doesn't match the job's log), `gpus` re-arms it at once: a second background task, `command grep -m1 -E '<fixed pattern>' <(tail -n +1 -F --pid=$(cat <log>.pid) <log>)`, and sends the branch `PROGRESS <job> <line>` when it ends with a match.
- **No one waits by sleeping, polling, or looping on `pgrep`/`ps`/`squeue`.** A job's subagent waits on the job itself, as a background task, and its report wakes the team. If you're waiting, end your turn.
- **One run per job,** and one job per GPU. No worker scripts that loop over runs, and no fixed per-GPU queues.
- **Cheap before expensive:** a long run that a check could make useless starts only after that check. Jobs needed whatever happens never wait for checks.
- **Useless work stops:** the moment a candidate is rejected, all its jobs are cancelled.
- **Longest chain first, by measurement:** among needed jobs, the one with the most work waiting on it and in it starts first, estimated from measured pace, never from names or sizes alone.
- **Absolute paths;** create directories in a separate step before launching.

## 5. Finish

When the rule has its answer for every candidate and every result it rests on is in, have `gpus` cancel anything still running that the answer doesn't need. Then report at once: the answer, the results it rests on, and each rejection with the evidence that settled it, which is evidence the rule accepts.
