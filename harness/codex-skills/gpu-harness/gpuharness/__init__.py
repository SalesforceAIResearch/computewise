"""gpuharness: a GPU job harness for agents.

The agent writes the job graph and makes every judgement (what the rule is, what counts as evidence, which
candidate passes). The harness does the mechanics: it owns the GPUs, launches each job in its own process group,
watches the machine rather than its own notes, measures pace from the jobs' logs, schedules by the policy in
`scheduler`, cancels a rejected candidate's whole subtree, and wakes the right agent with one-line events.

Modules:
    model      the plan's data types: candidates, jobs, their states
    plan       load and validate the job graph the agent writes
    estimate   online duration model: priors from the plan, corrected by measured runs
    progress   read pace from a running job's log lines
    scheduler  the policy (decision-aware critical-path list scheduling) and its guarantees
    launcher   start a job in its own process group, check it, stop the whole group
    ledger     the harness's state on disk, and the event log agents wait on
    daemon     the loop that ties them together
    client     the command spool between the CLI and the daemon
    timeline   status views and timeline export
    cli        the `gpuh` command
"""

__version__ = "0.1.0"
