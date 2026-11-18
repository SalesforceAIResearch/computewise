---
name: scheduler-playbook
description: Use whenever a task involves running several GPU (or other scarce-resource) jobs on a machine with a SLURM scheduler — training runs, sweeps, screens, evaluations, chains of runs. How to plan the jobs as a graph, submit everything that can run, and react to completions so no GPU sits idle while work remains.
---

# Running many jobs through the scheduler

The goal is to finish all the work as early as possible with no GPU idle while runs remain. Every idle GPU-second is waste.

## 1. Plan the graph before running anything

Write down, in a file (e.g. `plan.md`), every job the task needs:
- its command;
- its GPU count;
- what it depends on;
- roughly how long it takes, if you can tell.

Mark the **critical path**, the longest chain of dependent work. Its first job starts first.

## 2. Submit everything that can run, at once

**Order matters: the scheduler starts queued jobs in submit order.** Among jobs that can run now, submit **the longest first**. Estimate each job's time from its steps × step time, its config or its name. Jobs that feed a later stage or decision count their whole downstream chain.

**Use this exact template, one Bash call per job, each as a background task (`run_in_background: true`), longest first, all in one turn:**

```
sbatch --wait --parsable -J <name> --gres=gpu:1 -o /abs/logs/<name>.out --wrap "cd /abs/workdir && <the job's command>"
```

`--wait` makes that background task the job's own completion signal. It exits when the job ends, with the job's exit code, and you're woken. Don't poll `squeue` in a loop instead: polling adds its interval to every decision.

- Submit every job whose inputs exist **now**. The scheduler queues what doesn't fit and starts each queued job the moment a GPU frees. You don't refill GPUs; the scheduler does.
- Submit a dependent job immediately too, with `sbatch --dependency=afterok:<id>[:<id>...] ...`. Don't wait for its parent to finish.
- **One run per job.** Never put several runs in one job: no worker script that loops over a list of runs, and no array task that runs several trainings in turn. The scheduler can't repack a job that holds its own queue, so some GPUs finish early and sit idle while one job works through its list.
- Use absolute paths. Put `cd /abs/path` inside the job script, never relying on your shell's current directory.
- Each job's output goes to `slurm-<id>.out`. Name jobs (`-J`) so `squeue` and `sacct` are readable.
- **Never launch GPU work outside the scheduler,** and never choose a GPU yourself. Request exactly the GPUs a job needs (`--gres=gpu:1` for a one-GPU job). The scheduler gives each job its own GPUs, and two jobs on one GPU slow both and finish later.
- **Create directories in a separate foreground command** before submitting; never background a `mkdir … &&` chain.
- **A batch script runs its work in the foreground.** When the script exits, anything it left running is killed, as in real SLURM, so `sbatch --wrap 'python train.py &'` ends the training at once. No `&`, `nohup` or `setsid` inside a job.

## 3. Never wait blind

- **Don't `sleep` for a fixed time, loop on `squeue`, or set a timer to wait for jobs.** A job that ends early leaves a GPU idle until you wake up. A foreground `while squeue …; do sleep N; done` also blocks your turn: you can't react to any job until the loop exits, and every decision waits up to N seconds.
- To be told when a job ends, submit it with `sbatch --wait ...` **as a background task** (or run `srun ...` as one). The task exits with the job's exit code when the job ends, and your session is notified. Never watch a `pgrep -f` pattern, which can match itself and never end.
- After starting background waits, **end your turn.** You'll be woken when something finishes.
- A background task isn't cut off by the Bash time limit, so a background `sbatch --wait` can wait for hours.

## 4. On every completion: act at once

1. `sacct -j <id> --format=JobID,JobName,State,ExitCode,Start,End`.
2. **Failed:** read `slurm-<id>.out` immediately. Fix the cause, and resubmit only that job. Before re-running anything, check `sacct` for a completed run of it; never redo finished work.
3. **Succeeded:** if a decision depends on its result (pick the best runs, choose the next stage), read the result now and submit the next stage at once.
4. Submit any job that became ready, so the queue never runs dry while work remains.

## 5. Prepare ahead

If later jobs need code, configs or data you have to write, prepare them **while** the current jobs run, not after. A job should be ready to submit before a GPU frees for it.

## 6. Finish

When `squeue` is empty and every planned job has a completed `sacct` record, report the results at once, in the same turn as the last completion.
