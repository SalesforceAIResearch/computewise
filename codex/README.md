# Codex session driver (benchmark plumbing, not the Codex plugin)

**Looking for the gpu-harness for Codex?** It is the Codex edition in
[`harness/codex-skills/gpu-harness/`](../harness/codex-skills/gpu-harness/), installed with
`codex plugin marketplace add SalesforceAIResearch/computewise` and `codex plugin add gpu-harness@computewise` (see
the repository [README](../README.md#install)).

`harness/codex_session.py` here is the driver the benchmark's launcher (`isolation/run_one.py`) uses to run a Codex CLI
session inside the benchmark's container.
