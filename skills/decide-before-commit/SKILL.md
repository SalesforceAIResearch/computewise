---
name: decide-before-commit
description: Use whenever a task runs several GPU (or other scarce-resource) jobs to reach a decision or a deliverable — evaluations, sweeps, candidate comparisons, acceptance rules, benchmark tables. How to find the cheap observations that settle the decision before committing GPUs to long runs, cancel work the moment it becomes useless, and order the rest so no long job starts late.
---

# Decide before you commit

The goal is the deliverable, as early as possible, with no GPU-hour spent on work the answer doesn't need. Two kinds of waste hide behind busy GPUs: **long runs the decision never needed**, and **a long job started late**. A scheduler packs both perfectly, and both still cost hours.

## 1. Write down what the answer needs

Before launching anything, write `plan.md`:
- **The rule:** what decides the answer, in the brief's own terms (a threshold, a comparison to a baseline, a budget).
- **Every candidate and every job** the rule could need, each with a rough length if the files tell you.
- **For each candidate, its cheapest decisive check:** the fastest observation that can rule it out. Examples:
  - a run's step time shows in its first minutes, which settles a time or budget rule;
  - a short whole workload can fail a quality rule long before the longest workload does;
  - a baseline's numbers are needed by every comparison, so they come first.
- **What's truly required** whatever happens (the baseline, a table the brief asks for in full) and what's needed **only if the candidate survives**.

## 2. Run the decisive checks first

- Launch the cheap checks first: the baseline's, then every candidate's cheapest decisive check. Fill every free GPU with them.
- **Don't start a candidate's long run until it has passed at least two of its cheap checks** (or its only one). A long run started before the check that could rule it out is the costliest mistake in this kind of task.
- A check is a real observation, not a guess from a name or from what you believe about a method. If a rule is about time, measure the time: start the run and read its step time in its log. If it's about quality on whole workloads, run whole workloads; partial ones can mislead.

## 3. Act on every result the moment it arrives

- Wait on jobs in the background (a background task per job, or `sbatch --wait` as a background task) and end your turn. You're woken when a job ends.
- On each result, apply the rule at once.
  - **If it rules a candidate out, cancel every job of that candidate now** (`kill <pid>` or `scancel <id>`), including long runs already going. A job you know is useless is never left running "just in case". Write the rejection and the reason in `plan.md`.
  - If it passes, launch that candidate's next needed job.
- **To read a running job's speed,** look at its log after its first progress or validation line, then decide at once: keep it, or cancel it and every sibling.

## 4. Order what's left: longest first, by measured length

- When everything left is needed, start the longest first. Estimate lengths from measurements, not names: a few minutes of a job's own log gives its pace, and pace × size gives its length.
- When several jobs look alike but may differ in length, **probe the largest ones briefly first** (a short run, or the first minutes of the real one), then order by what you measured. A long job of unknown length started last sets the finish time for everything.
- Fill free GPUs with short, certainly-needed jobs while probes run.

## 5. Mechanics

- **One run per job.** No worker scripts that loop over a list of runs, and no fixed per-GPU queues. Give each free GPU the next job.
- **One job per GPU.** Never double up to look busy.
- Never sleep for a fixed time, and never wait on a `pgrep -f` pattern, which can match its own shell. Wait on a background task or a job id.
- Absolute paths. Create directories in a separate foreground command.

## 6. Finish

When the rule has an answer and every job it rests on has finished, cancel anything still running that the answer doesn't need. Then report at once: the answer, the results it rests on, and each rejection with the observation that settled it.
