"""Run one ComputeWise smoke run end to end: isolation's single-run command, the budget stop, the
ledger copy, and the score.

usage:
  python3 -I run_smoke.py --task trl-dpo-loss-variants-4gpu [--model claude-haiku-5-5] [--seed 0]
                          [--budget-usd 3] [--max-hours 2] [--speed 1] [--runs-root /srv/runs]
                          [--harness claude-code | --harness scripted --policy packed]
                          [--run-one <path to run_one.py>] [--stop-dir /srv/stage/gate/stop] [--dry-run]
                          [-- <extra flags passed to run_one.py, e.g. --cpus 8 --memory 12g --disk 30g>]

What it does, in order:
  1. Mints the run's cost tag (P006-<task>-<model>-s<seed>-<UTC stamp>) and its run directory
     <runs-root>/<tag>/. It runs on the VM, as root, because run_one.py does.
     With --harness scripted it first writes the plan with scripted/plan.py, and no budget watcher is
     started, since a scripted agent makes no model calls.
  2. Refuses a model run unless --ceiling-confirmed: the budget is enforced on the controller by the the router
     session's budget_watch.py (experiments/isolation/forwarder/), which reads the ledger there and
     touches ~/.p006-gate/stop/<tag>, which the controller gate answers with 429. One enforcer;
     run_one.py records --budget-usd but does not enforce it. RUNBOOK.md, step 3.
  3. Runs isolation's run_one.py with the tag, the stop dir and the inputs, and waits.
  4. Prints how to copy the run's ledger turns on the controller (raw inputs for charged latency).
  5. Scores with score/score_all.py: score_run.py, idle.py, overshoot.py and served_model.py into
     <run-dir>/score.txt and score.json (with the scorers' provenance), and the timeline.

Defaults follow the study's first-run objective (STATE.md, "Current objective"): Claude Code on
Haiku 5.5, speed 1, a $3 cap. No machine size is assumed; caps go through to run_one.py.
"""
import argparse, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
DEFAULT_RUN_ONE = os.path.join(EXP, "isolation", "run_one.py")

def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--harness", default="claude-code")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-usd", type=float, default=3.0)
    ap.add_argument("--max-hours", type=float, default=2.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--runs-root", default="/srv/runs")
    ap.add_argument("--run-one", default=DEFAULT_RUN_ONE)
    ap.add_argument("--stop-dir", default="/srv/stage/gate/stop",
                    help="must match run_one.py's --stop-dir and stop_relay.py's --vm-stop-dir, "
                         "so run_one.py sees the stop and labels the run killed-on-budget")
    ap.add_argument("--policy", default=None, help="scripted harness only: plan.py policy")
    ap.add_argument("--plan-file", default=None, help="scripted harness only: a plan written elsewhere (a probe's, e.g. "
                    "the scheduler's scheduler probe), used as it is instead of plan.py's; its sha256 is printed")
    ap.add_argument("--host-terms", default=None,
                    help="scripted packed_overlap: host_terms.py output from a measuring run, on the VM")
    ap.add_argument("--dry-run", action="store_true", help="print the run_one.py command and exit")
    ap.add_argument("--tag", default=None, help="reuse a tag minted earlier, so its budget ceiling matches")
    ap.add_argument("--brief-file", default=None,
                    help="replace the task's brief for this run (C2's one-line brief); passed to run_one.py")
    ap.add_argument("--ceiling-confirmed", action="store_true",
                    help="the controller's budget_watch.py already has a ceiling for this tag")
    ap.add_argument("--nudge", default=None, help="campaign nudge level L0..L3; passed to run_one.py, which appends the "
                                                   "level's sentence from --campaign to the brief and records it (gap G6)")
    ap.add_argument("--campaign", default=None, help="campaign.json, for --nudge")
    ap.add_argument("--no-score", action="store_true", help="skip score_all after run_one: the campaign driver scores the run "
                                                             "on the controller after collect, so the unit ends at teardown and the "
                                                             "machine is free (the study, 2026-10-22: a goal run's machine score "
                                                             "held c3-11 for 30 min after teardown)")
    a = ap.parse_args(argv)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    short = re.sub(r"[^A-Za-z0-9]+", "", a.model.replace("claude-", ""))[:12]
    tag = a.tag or f"P006-{a.task}-{short}-s{a.seed}-{stamp}"[:128]
    run_dir = os.path.join(a.runs_root, tag)
    scripted = a.harness == "scripted"
    if scripted and not a.policy and not a.plan_file:
        sys.exit("--harness scripted needs --policy or --plan-file")
    if scripted and not a.tag:
        tag = f"P006-{a.task}-scripted-{a.policy}-s{a.seed}-{stamp}"[:128]
        run_dir = os.path.join(a.runs_root, tag)
    plan_path = os.path.join(a.runs_root, ".plans", f"{tag}.json")
    cmd = [sys.executable, "-I", a.run_one, "--task", a.task, "--harness", a.harness, "--seed", str(a.seed),
           "--out", run_dir, "--speed", str(a.speed), "--max-hours", str(a.max_hours), "--cost-tag", tag,
           "--stop-dir", a.stop_dir, *extra]
    cmd += ["--plan", plan_path] if scripted else ["--model", a.model, "--budget-usd", str(a.budget_usd)]
    if a.brief_file:
        cmd += ["--brief-file", a.brief_file]
    if a.nudge:
        cmd += ["--nudge", a.nudge] + (["--campaign", a.campaign] if a.campaign else [])
    if a.dry_run:
        print(" ".join(cmd)); return
    if not os.path.exists(a.run_one):
        sys.exit(f"isolation's single-run command is not at {a.run_one} yet; nothing was started")
    if scripted:
        os.makedirs(os.path.dirname(plan_path), exist_ok=True)
        task_json = os.path.join(EXP, "gpu-tasks", "registry", "tasks", f"{a.task}.json")
        if a.policy == "packed_overlap" and not a.host_terms:
            sys.exit("packed_overlap needs --host-terms from a measuring run (scripted/host_terms.py); nothing was started")
        if a.plan_file:
            import hashlib, shutil
            shutil.copyfile(a.plan_file, plan_path)
            print(f"  plan file {a.plan_file} sha256 {hashlib.sha256(open(plan_path, 'rb').read()).hexdigest()[:16]}", flush=True)
        else:
            plan_cmd = [sys.executable, "-I", os.path.join(HERE, "scripted", "plan.py"), task_json, a.policy, "--out", plan_path]
            if a.host_terms:
                plan_cmd += ["--host-terms", a.host_terms]
            if subprocess.call(plan_cmd, stdout=subprocess.DEVNULL) != 0:
                sys.exit("plan.py failed; nothing was started")
        print(f"run {tag}\n  dir  {run_dir}\n  plan {plan_path}", flush=True)
        rc = subprocess.call(cmd)
        print(f"run_one.py exited {rc}", flush=True)
        if os.path.exists(os.path.join(run_dir, "occupancy.jsonl")):
            subprocess.call([sys.executable, "-I", os.path.join(HERE, "scripted", "check.py"), run_dir])
            subprocess.call([sys.executable, "-I", os.path.join(HERE, "score", "score_all.py"), run_dir], stdout=subprocess.DEVNULL)
        return
    os.makedirs(run_dir, exist_ok=True)

    # The budget is enforced on the controller, not here: the ledger lives there, and so does the gate
    # whose stop directory it reads. One budget_watch.py (experiments/isolation/forwarder/) covers
    # every run; the ceiling for this tag must be in ~/.p006-gate/budget.json before the run starts.
    # This command refuses to start unless --ceiling-confirmed says that was done (see RUNBOOK.md).
    if not a.ceiling_confirmed:
        sys.exit(f"set a ceiling for {tag} in the controller's ~/.p006-gate/budget.json first (RUNBOOK.md, step 3),\n"
                 f"then re-run with --ceiling-confirmed --tag {tag}")
    print(f"run {tag}\n  dir  {run_dir}\n  cap  ${a.budget_usd:.2f}, enforced by the controller's budget_watch.py;\n"
          f"  on the controller, run: python3 -I experiments/runner/score/stop_relay.py {tag}", flush=True)
    rc = subprocess.call(cmd)
    print(f"run_one.py exited {rc}", flush=True)
    print("ledger: copy this run's turns from the controller with\n"
          f"  python3 -I experiments/runner/score/ledger.py dump {tag} <local file>", flush=True)
    if not os.path.exists(os.path.join(run_dir, "occupancy.jsonl")):
        print("no occupancy.jsonl in the run directory: nothing to score"); return
    if a.no_score:
        print("scoring skipped (--no-score): the campaign driver scores this run on the controller after collect"); return
    # score.txt, score.json (with the served-model check once ledger.jsonl is in the run directory, and the
    # scorers' provenance) and timeline.html; the campaign driver re-runs it after copying the ledger in (G2, G4, G8)
    subprocess.call([sys.executable, "-I", os.path.join(HERE, "score", "score_all.py"), run_dir])

if __name__ == "__main__":
    main()
