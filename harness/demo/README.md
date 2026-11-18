# Try the gpu-harness in five minutes

A toy sweep that shows the harness at work on any machine, with or without GPUs. `train.py` stands in for a training
script. It does no GPU work and needs no packages. It prints progress lines like a real run and takes about a minute.

The sweep compares three configs with a baseline, two seeds each, under an adoption rule:
- `cosine` really is better.
- `warmup` is within noise of the baseline.
- `high-lr` looks best at step 100 and goes wrong after that, so a rule that checks step 200 rules it out before its
  runs finish. A one-seed cut like this is safe here only because the toy is built to be clear-cut; for real work,
  see the README's advice on early rules.

## Run it

1. Install the plugin as the repository README's [Install](../../README.md#install) section says: for Claude Code,
   including agent teams; for Codex, with `multi_agent` on.
2. Copy this directory to a working directory, `~/gpuh-demo`. From a clone of the repository, run this from the
   folder that holds the clone:

   ```bash
   cp -r computewise/harness/demo ~/gpuh-demo
   ```

   From the installed Claude Code plugin:

   ```bash
   cp -r ~/.claude/plugins/cache/computewise/gpu-harness/0.1.0/demo ~/gpuh-demo
   ```

   From the installed Codex plugin:

   ```bash
   cp -r ~/.codex/plugins/cache/computewise/gpu-harness/0.1.0/demo ~/gpuh-demo
   ```

3. Start your agent in that directory. To make each job take 20 seconds instead of 60, put `DEMO_SECONDS=20` in
   front of the command (`DEMO_SECONDS=20 claude`). For Claude Code:

   ```bash
   cd ~/gpuh-demo && claude
   ```

   For Codex, keep the harness's record inside the directory, where Codex's sandbox lets it write:

   ```bash
   cd ~/gpuh-demo && export GPUH_HOME="$PWD/.gpuh" && codex
   ```

4. Paste this request. In Codex, begin it with `$gpu-harness` instead of `/gpu-harness:gpu-harness`; with a copied
   skill in Claude Code, with `/gpu-harness`.

   > /gpu-harness:gpu-harness In this directory, `python3 train.py --config configs/<name>.json --seed <n>` trains one
   > config and prints `step:N/1000 val_loss:...` lines, then `final val_loss <x>`. Run `baseline`, `cosine`, `warmup`
   > and `high-lr` with seeds 0 and 1 on pretend GPUs 0 and 1, listed in the plan (these jobs do no GPU work).
   > Adopt a config if its mean final val_loss is at least 0.05 below the baseline's mean. A config whose val_loss at step 200 is above the baseline's at step 200, on the same seed, is
   > rejected.

## What you should see

The whole run takes about four to six minutes.
1. **Planning.** The agent writes a `plan.json` and starts the harness. In Claude Code it also starts one teammate per
   branch (they appear as named agents); in Codex, sub-agents.
2. **Running.** The harness runs the baseline's and each config's jobs on the two GPUs, refilling a GPU as soon as a
   job ends.
3. **Ruling out `high-lr`.** When `high-lr` reaches step 200, its teammate compares it with the baseline and rejects
   it, and the harness stops its remaining jobs.
4. **The answer.** The agent adopts `cosine` and not `warmup` or `high-lr`.

In Claude Code's default permission mode, approve the agent's `gpuh` commands as they come (choose not to be asked
again).

To see the run's record, set up `gpuh` as the repository README's
[Watch, stop and inspect a run](../../README.md#watch-stop-and-inspect-a-run) says (for Codex, from `~/gpuh-demo`
with the same `GPUH_HOME`):
- `gpuh status -v` lists every candidate as `decided`, `rejected` or `open`;
- `gpuh results` lists every job's final line;
- `gpuh export` writes `timeline.html`, a picture of every job on every GPU, into the run directory.

