# gpu-harness plugin

A harness for coding agents that run GPU jobs to reach a decision: which method to deploy, which optimizer setups to
adopt, the full table of a benchmark. The agent plans the work and makes every judgement. The harness runs the jobs on
the GPUs, with an algorithm whose guarantees are stated in
[`skills/gpu-harness/README.md`](skills/gpu-harness/README.md). This directory is one plugin for both Claude Code and
Codex.

The repository's [README](../README.md#install) is the user guide. It covers what the plugin is for, when it
fits, the requirements, install, use, and how to watch and stop a run. The short version for each tool follows.

**Claude Code.** Install, then turn on agent teams, so the skill can name its teammates and message them:

```bash
claude plugin marketplace add SalesforceAIResearch/computewise
claude plugin install gpu-harness@computewise
python3 - <<'EOF'
import json, os
p = os.path.expanduser("~/.claude/settings.json")
d = json.load(open(p)) if os.path.exists(p) else {}
d.setdefault("env", {})["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] = "1"
os.makedirs(os.path.dirname(p), exist_ok=True)
json.dump(d, open(p, "w"), indent=2)
print("agent teams on in", p)
EOF
```

Restart Claude Code, then begin your request with `/gpu-harness:gpu-harness`. Before a real run, read the README's
"Before you start a run" (environment, which GPUs, live output, permissions).

**Codex:**

```bash
codex plugin marketplace add SalesforceAIResearch/computewise
codex plugin add gpu-harness@computewise
```

Check that `codex features list | grep -w multi_agent` shows `true` (if not, `codex features enable multi_agent`).
From your project, `export GPUH_HOME="$PWD/.gpuh"` and start `codex`, so the harness can write its run record inside
Codex's sandbox. Then begin your request with `$gpu-harness`. Before a real run, read the README's "Before you start a
run": Codex's sandbox needs more for real GPU jobs.

[`demo/`](demo/README.md) is a five-minute toy sweep that needs no GPUs.

## What is here

| Path | What it is |
|---|---|
| `skills/gpu-harness/` | The skill Claude Code loads: `SKILL.md` (the instructions the agent follows), the `bin/gpuh` command and the `gpuharness` Python package (standard library only, Python 3.9+). Its `README.md` describes the design, the scheduling algorithm and its guarantees. |
| `codex-skills/gpu-harness/` | The Codex edition: the same `bin/gpuh` and `gpuharness` package, with a `SKILL.md` written for Codex. Codex starts sub-agents with `spawn_agent` and waits on `gpuh wait` in the foreground, since nothing wakes a Codex agent once its turn ends. |
| `demo/` | A toy sweep for trying the plugin on any machine (see its `README.md`). |
| `.claude-plugin/plugin.json`, `.codex-plugin/plugin.json` | The two manifests. Claude Code loads `skills/`; the Codex manifest points Codex at `codex-skills/`. |
| `sync_skill.sh` | A maintainer script; see below. |

`bin/gpuh` finds its package from its own location, so it works from wherever the tool installs the plugin. It is not
put on your `PATH`; the repository README gives an alias for each install method.

## Update or remove

The repository README's [Update or remove](../README.md#update-or-remove) section has the commands. In short: every
copy says version 0.1.0, so refresh the marketplace and reinstall to update; to remove, uninstall the plugin and remove
the `computewise` marketplace.

## Tests

```bash
(cd skills/gpu-harness && python3 -m unittest discover -s tests)
(cd codex-skills/gpu-harness && python3 -m unittest discover -s tests)   # the same tests, on the Codex edition's copy
```

## For maintainers: which version this is

Both editions run the same harness code, build `74a43c895282f9ab`; only `SKILL.md` differs. These are the versions
the paper reports: the Claude Code skill (`SKILL.md` sha256 `2a0e6d1d…`) and its Codex edition (`af41dfa5…`).
`stamp.py --check` in either skill directory confirms that its `SKILL.md` carries the build stamp of the code beside
it.

`sync_skill.sh` replaces an edition with another version of the skill:

```bash
./sync_skill.sh /path/to/another/gpu-harness             # the Claude Code edition
./sync_skill.sh /path/to/another/gpu-harness-codex codex # the Codex edition
```

It copies the tree in, keeping each file's mode and leaving out `__pycache__`. It then runs `stamp.py --check` and
reports whether the two editions' code is still identical. Reinstall the plugin afterwards, as above.
