# The study's other conditions (not the gpu-harness)

**Looking for the gpu-harness skill?** It is in [`harness/`](../harness/README.md): the Claude Code skill in
`harness/skills/gpu-harness/`, the Codex edition in `harness/codex-skills/gpu-harness/`. Install it as the
repository [README](../README.md#install) describes.

This directory holds the other conditions the paper compares the harness with. Each is a set of instructions for the
agent, with no code of its own:

| Directory | What it is |
|---|---|
| `gpu-team/` | a written team procedure: the agent plans, a GPU teammate owns the GPUs, and a teammate per candidate judges it |
| `decide-before-commit/` | how to find the cheap observations that settle the decision before committing GPUs to long runs |
| `scheduler-playbook/` | how to plan the jobs as a graph and keep a SLURM scheduler's GPUs busy |
| `parallel-attention/` | each job in its own background sub-agent, so the agent is woken the moment a GPU frees |
| `history/gpu-harness-a4.3/` | the harness's previous development step, kept for reference |

They are kept so the study can be reproduced. They are not recommended as tools.
