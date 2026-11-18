---
name: parallel-attention
description: Use whenever a task runs several GPU (or other scarce-resource) jobs with any structure — independent runs, chains, screen-then-confirm, sweeps — and you want every GPU busy until the work is done. Each job runs in its own background subagent so you are woken the moment a GPU frees, and prepared work is started at once, longest first.
---

# Parallel attention: keep every GPU busy

**Why:** you don't know when jobs will end, and while you think or sleep, a freed GPU stays idle. So never plan GPU assignments up front, and never wait by sleeping. Run each job inside its own background subagent: Claude Code wakes you the moment that subagent finishes, and you start the next job on the freed GPU immediately.

## Hard rules

1. **A job's background task must BE the job.** The command runs in the foreground of that task: no `&`, no `nohup`, no `setsid`. If you background it inside the task, the task ends at once, and nobody is ever told when the job finishes.
2. **Never assign jobs to GPUs up front.** No lane scripts, no per-GPU loops, no `job & job & wait`. A job gets a GPU only when the GPU is free.
3. **Nobody sleeps or polls to wait.** Start background work, then end your turn. You'll be woken.
4. **Critical path first.** Rank each ready job by its **chain length**: its own estimated time plus the longest chain of work that must wait for it. Estimate times from steps × step time, the config, or the name. Give each free GPU to the ready job with the longest chain.
   - Example (illustrative): two sweeps X1 and X2 (40 min each) each feed a 60 min follow-up, and a standalone eval E takes 50 min. The chains are X1 = X2 = 100 and E = 50, so X1 and X2 start before E. A job that feeds a later decision or stage comes before a standalone job of similar length.
5. **Read results as soon as they arrive**, and decide the next stage (pick the best, launch the confirm runs) in that same turn.
6. **One line per report.**
7. **One job per GPU, always.** Never put two jobs on one GPU to look busy. They slow each other down, and the work finishes later. A GPU is busy when it runs exactly one job.
8. **Launch hygiene.** Create directories in a separate foreground command before launching; never background a `mkdir … &&` chain. Use absolute paths. Wait only on a background task or a PID, never on a `pgrep -f` pattern, which matches its own shell.

## Simple mode: most tasks

You are the dispatcher. Keep `gpus.md`, which records each GPU's job, or free.

1. **Read the job scripts once** to estimate each job's length and see its dependencies. Write the ready queue, longest first, into `gpus.md`.
2. **For each free GPU,** take the head of the queue and start a job subagent: Agent tool, `run_in_background: true`, `model: "haiku"`, with the JOB prompt below. Start all of them in **one** turn, then end your turn.
3. **Each time you're woken by `DONE <job> GPU <id> …`:**
   - mark the GPU free;
   - record the result;
   - if this completes a stage, decide the next stage now and add its jobs to the queue;
   - start the next queued job on that GPU;
   - end your turn.
4. **On `FAILED`:** read the log, fix the cause, requeue only that job, and never redo a finished job.
5. When the queue is empty and every GPU is free, report the results.

**JOB prompt:**

> Run this as a single background Bash task (`run_in_background: true`), exactly as written, with no `&` or `nohup`: `cd {abs_dir} && CUDA_VISIBLE_DEVICES={id} {command} > {abs_log} 2>&1`. Then end your turn without sleeping or polling. When the task finishes, read the last lines of `{abs_log}` and reply with exactly one line: `DONE {job} GPU {id} <final metric line>` if it exited 0, otherwise `FAILED {job} GPU {id} <last error line>`.

## Teammate mode: only when jobs need real preparation or reading between them

Use this mode when writing a job's script or config takes you minutes, or when several branches each have decisions to make. Thinking in the dispatcher would leave GPUs idle.

- **You stay the dispatcher,** exactly as in Simple mode.
- **Start one named teammate per branch** (Agent tool, `name: "<branch>"`, `run_in_background: true`). Its job is to prepare: write and check each job of its branch before it's needed, and send you `READY <job> <est_seconds> <command>` (SendMessage to `main`).
- **You add READY jobs to the queue** and dispatch them as GPUs free up.
- **After each DONE,** tell the job's branch in one line (`DONE <job> <result>`), so it can prepare what comes next.
- **Teammates never launch jobs or hold GPUs.**

## With a scheduler (sbatch present)

The scheduler is the dispatcher's queue.
- Submit every ready job at once with `sbatch --parsable`, using `--dependency=afterok:<id>` for chain steps.
- Each JOB subagent runs `sbatch --wait <script>` as its single background task, and reports from `sacct -j <id> -P -n -X --format=State,ExitCode`. The batch script runs the job in the foreground: anything a script leaves running in the background is killed when it exits.
- Decisions are the same: on the DONE that completes a stage, submit the next stage at once.
