"""Every attempt on the paper's five packages, by package x model x arm, with its outcome class (the analysis 2026-10-29
18:20 MDT, from adversarial review RV-001: Astra, Sol and Opus found that dropping no-reading runs makes each cell its
survivors). One row per collected, scored run (W0 references excluded; runs still in flight have no score and are not counted).

Outcome classes, in this order of precedence (the first that applies):
  stopped        superseded or stopped by us: stopped_to_protect_subscription, stopped_for_r6, stopped_by_watcher_false_positive
  not_counted    a package's brief defect, not the agent's (brief_scope_ambiguous, handed_back_brief_ambiguous; the analysis
                 2026-11-03 22:16 MST): outside the attempt count and the failure rate. A void label wins over it, so a void run keeps
                 its void class (G-k2lp-sonnet-L0 stays simulator)
  infrastructure infrastructure void: testbed_outage, codex_turn_failed (provider capacity), a served-model / route void
                 (subscription_limit, served_model_fallback), a harness API 502
  simulator      simulator-caused: a detection refusal (detected_mock_refused, detected_mock_no_answer), a mock defect (a void
                 testbed defect from served_paths_disagree or synthesised_subset; dropped_workload_on_subsets, where the agent
                 dropped a workload on synthesised subset zeros; killed_on_served_miss on a run with no reading, a record that
                 lacked the launch's argument)
  agent_failure  valid, and no answer by the horizon, an unsupported answer, or handed back mid-run (no_bar_reading, or an
                 answer the scorer reads unsupported); a horizon-capped run counts here with its capped value
  completed      valid with a reading
Each row names its rule. The simulator rows marked "judgement" are the runner's reading, for analysis to confirm.

Per cell: the attempt count and the agent-failure rate (agent failures over completed + agent failures), and two pairs of
medians, tokens and bar over the same runs:
  completed only                      the figure the bars use today
  completed + horizon-capped          adds each run cut at the horizon at its capped value (a lower bound on how slow it was):
                                      the survivor-free figure. A run that handed back early or stopped with no answer is counted
                                      in the failure rate but kept out of this median, because its figure measures the little it
                                      did before stopping (G-kvreph-sol-L0 handed back at coverage 0 and reads 0.008), which
                                      would make the cell look faster, the opposite of correcting survivor bias
Tokens are the router's ledger rows for the run's cost tag (runner/reports/harness_tokens.py's reading). The simulator build is
gpu-mock's release (screen.py's gpumock_build, from experiments/gpu-mock/archive/INDEX.tsv).

usage: python3 -I attempts.py [--out-dir DIR]     writes attempts.md and attempts.json beside this script. One process, read-only."""
import argparse, json, os, sqlite3, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "score")); sys.path.insert(0, HERE)
import screen as S                       # noqa: E402
import harness_tokens as HT              # noqa: E402

PACKAGES = {"mls-kv-gate": "G4", "mls-kv-report": "G1", "mls-kv-report-h": "H5", "pkg-g4n-a": "twin A", "pkg-g4n-b": "twin B"}
# the screened candidates (the analysis 2026-11-03 12:50 MST, from EVALUATION-PLAN.md "The package search stops taking new
# packages"): a second block below the paper's five, so the paper's table keeps its five-package scope while every run of every model
# appears once with its outcome (the limits section and the taxonomy: detection and hang counts on new packages)
SCREENED = {"pkg-filter-yield": "filter-yield", "pkg-unneeded-k": "K", "pkg-k2-precision": "K2 precision", "pkg-k2-latepole": "K2 late pole",
            "pkg-obs-latepole": "OBS late pole", "pkg-sf-zero-lr": "SF zero-LR", "pkg-sf-checkpoint-collision": "SF checkpoint",
            "pkg-oom-relaunch-seedmajor": "C seed-major", "pkg-astra-widths": "widths", "pkg-sol-tabular-science": "TTC science",
            "pkg-sol-tabular-operations": "TTC operations", "pkg-sol-tabular-science-adoption": "TTC science adoption",
            "pkg-sol-tabular-operations-adoption": "TTC operations adoption", "pkg-chain-save-8gpu": "D8 chain save"}
ALL_PACKAGES = dict(PACKAGES, **SCREENED)
CLASSES = ("completed", "agent_failure", "simulator", "infrastructure", "stopped", "not_counted")
NOT_COUNTED = ("brief_scope_ambiguous", "handed_back_brief_ambiguous")
STOPPED = ("stopped_to_protect_subscription", "stopped_for_r6", "stopped_by_watcher_false_positive")
INFRA = ("testbed_outage", "codex_turn_failed", "served_model_fallback", "route_502_in_lapse",
         "scheduler_deadlock",      # fake-slurm's scancel deadlock (kept below)
         # our lifecycle closed the session while the agent's own GPU work or wait was live
         "closed_by_work_done_grace_with_gpu_work_live", "closed_by_work_done_grace_before_wait_bound",
         "closed_by_work_done_grace_before_wait_returned")  # fake-slurm's scancel deadlock (G-k-haiku-S1-r2, the analysis 2026-11-04 00:14 MST; fake-slurm's name, 00:24 MST)
SIM_REFUSAL = ("detected_mock_refused", "detected_mock_no_answer")
SIM_DEFECT = ("served_paths_disagree", "synthesised_subset", "disengaged_smoke_ran_real", "unmatched_launch_ran_real",   # the last: the analysis 2026-11-03 12:43 MST
              "zero_mib_on_busy_gpu", "killed_because_of_zero_mib")                      # 0 MiB on a busy GPU, acted on (12:51 MST)

def J(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return {}

def classify(row, d):
    """(class, rule) for one attempt: its validity, its run_stop analysis labels, and its scorer fields"""
    s = J(os.path.join(d, "score.json")); h = s.get("headline") or {}
    al = (J(os.path.join(d, "run_stop.json")).get("analysis_labels") or {})
    reasons = " ".join((s.get("validity") or {}).get("reasons") or [])
    st = J(os.path.join(d, "status.json"))
    v = row["validity"]
    for k in STOPPED:
        if al.get(k):
            return "stopped", k
    if v != "valid":
        for k in INFRA:
            if al.get(k):
                return "infrastructure", k
        if "subscription_limit" in reasons or reasons.startswith("served model"):
            return "infrastructure", "served model / route"
        if "API Error: 502" in reasons or "502 Bad Gateway" in reasons:
            return "infrastructure", "harness API 502"
        for k in SIM_DEFECT:
            if al.get(k):
                return "simulator", f"mock defect ({k})"
        if v == "review" and al.get("killed_on_served_miss"):
            return "simulator", "judgement: served miss, the record lacked the launch's argument (review)"
        return "infrastructure", "void, reason: " + reasons[:80]
    for k in NOT_COUNTED:
        if al.get(k):
            return "not_counted", k
    for k in SIM_REFUSAL:
        if al.get(k) and row["no_bar_reading"]:
            return "simulator", f"detection refusal ({k})"
    if al.get("dropped_workload_on_subsets"):
        return "simulator", "judgement: dropped a workload on synthesised subset zeros (r4; r6 serves measured heads)"
    capped = bool(al.get("horizon_capped_lower_bound")) or bool(h.get("no_gpu_job_by_horizon")) or (
        row["no_bar_reading"] and st.get("status") == "killed-on-time")
    if row["no_bar_reading"]:
        why = "horizon-capped" if capped else ("handed back mid-run" if al.get("handed_back_mid_run") else "no answer")
        return "agent_failure", why
    if row.get("pick_supported") is False:
        return "agent_failure", "unsupported answer"
    return "completed", "valid with a reading"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=HERE)
    a = ap.parse_args()
    con = sqlite3.connect(f"file:{HT.LEDGER}?mode=ro", uri=True)
    out = []
    for r in S.rows(tasks=set(ALL_PACKAGES), all_validities=True):
        rid = r["run_id"]; d = os.path.join(S.RUNS, rid)
        cls, rule = classify(r, d)
        col = "over_ref" if r["task"] in S.GOAL_ONLY else "over_feasible" if S.feasible_best(r["task"]) is not None else "over_best"
        tag = J(os.path.join(d, "run.json")).get("cost_tag")
        tok = HT.tokens(con, tag) if tag else None
        h = HT.run_hours(rid)
        out.append({"run_id": rid, "package": ALL_PACKAGES[r["task"]], "block": "paper" if r["task"] in PACKAGES else "screened",
                    "task": r["task"], "model": r["model"], "arm": r["arm"],
                    "class": cls, "rule": rule, "validity": r["validity"], "no_bar_reading": r["no_bar_reading"],
                    "bar_column": col, "bar": r.get(col), "coverage": r.get("coverage"),
                    "build": r.get("gpumock_build"), "gpumock_patch": r.get("gpumock_patch"), "gpumock_sidecar": r.get("gpumock_sidecar"),
                    "tokens_total": tok[0]["total"] if tok else None, "tokens_uncached": tok[0]["uncached"] if tok else None,
                    "requests": tok[0]["requests"] if tok else None, "run_h": round(h, 3) if h else None})
    order = list(ALL_PACKAGES.values())
    out.sort(key=lambda x: (order.index(x["package"]), x["arm"], x["model"] or "", x["run_id"]))
    json.dump(out, open(os.path.join(a.out_dir, "attempts.json"), "w"), indent=1)
    md = lambda v: statistics.median(v) if v else None
    F = lambda v, n=3: "" if v is None else f"{v:.{n}f}"
    M = lambda v: "" if v is None else f"{v / 1e6:.2f}"
    L = ["# Attempts on the five packages: every run, its outcome, and the cells without survivor bias", "",
         "Written by `attempts.py` beside this file (the runner, for the analysis's 2026-10-29 18:20 MDT request from "
         "RV-001). Regenerate with `python3 -I attempts.py`; `attempts.json` beside it carries every run. Read-only over the controller "
         "copies and the router's cost ledger.", "",
         "**Classes,** first that applies: **stopped** (by us: subscription protection, r6 preemption, a watcher false positive), "
         "**not counted** (a brief defect in our package, not the agent's: brief_scope_ambiguous, handed_back_brief_ambiguous; outside the "
         "attempt count and the failure rate; a void run keeps its void class), "
         "**infrastructure** (outage, provider capacity, route/served-model, harness API 502), **simulator** (a detection refusal or a "
         "mock defect), **agent failure** (valid, but no answer by the horizon, an unsupported answer, or handed back mid-run; a "
         "horizon-capped run counts here at its capped value), **completed** (valid with a reading). Rows whose rule starts "
         "\"judgement:\" are the runner's reading, for analysis to confirm.", "",
         "**Medians:** *completed* is the figure the bars use today. *Completed + horizon-capped* adds every run cut at the horizon at "
         "its capped value, a lower bound on how slow it was: the survivor-free figure. A run that handed back early or stopped with no "
         "answer counts in the **failure rate** (agent failures over completed + agent failures) but stays out of that median, because "
         "its figure measures the little it did before stopping and would make the cell look faster. Tokens in millions (total = "
         "input + cache_read + cache_write + output), over the same runs as the bar beside them. **Build** is gpu-mock's release.", ""]
    paper = [x for x in out if x["block"] == "paper"]; scr = [x for x in out if x["block"] == "screened"]
    tot = {c: sum(1 for x in paper if x["class"] == c) for c in CLASSES}
    stot = {c: sum(1 for x in scr if x["class"] == c) for c in CLASSES}
    L += [f"**All five packages: {len(paper) - tot['not_counted']} attempts** — " + ", ".join(f"{c.replace('_', ' ')} {tot[c]}" for c in CLASSES if c != "not_counted")
          + f"; not counted {tot['not_counted']}.", ""]
    L += [f"**Screened candidates (a second block, below the five): {len(scr) - stot['not_counted']} attempts** — "
          + ", ".join(f"{c.replace('_', ' ')} {stot[c]}" for c in CLASSES if c != "not_counted") + f"; not counted {stot['not_counted']}.", ""]
    for pkg in order:
        if pkg == list(SCREENED.values())[0]:
            L += ["# Screened candidates", "", "Every package screened under the 11-03 stop on new packages; not the paper's five.", ""]
        rs = [x for x in out if x["package"] == pkg]
        if not rs:
            continue
        col = rs[0]["bar_column"]
        L += [f"## {pkg} ({rs[0]['task']}, bar = {col})", "",
              "| model | arm | attempts | completed | agent failure | failure rate | simulator | infra | stopped | not counted | bar median, completed | tokens median, completed | bar median, completed + horizon-capped | tokens median, same runs | builds |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        cells = {}
        for x in rs:
            cells.setdefault((x["model"], x["arm"]), []).append(x)
        for (mod, arm), xs in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0] or "")):
            n = {c: sum(1 for x in xs if x["class"] == c) for c in CLASSES}
            comp = [x for x in xs if x["class"] == "completed" and x["bar"] is not None]
            both = comp + [x for x in xs if x["class"] == "agent_failure" and x["rule"] == "horizon-capped" and x["bar"] is not None]
            tk = lambda ys: [y["tokens_total"] for y in ys if y["tokens_total"] is not None]
            den = n["completed"] + n["agent_failure"]
            rate = f"{n['agent_failure'] / den:.0%}" if den else ""
            L.append(f"| {mod} | {arm} | {len(xs) - n['not_counted']} | {n['completed']} | {n['agent_failure']} | {rate} | {n['simulator']} | {n['infrastructure']} | "
                     f"{n['stopped']} | {n['not_counted']} | {F(md([x['bar'] for x in comp]))} | {M(md(tk(comp)))} | {F(md([x['bar'] for x in both]))} | "
                     f"{M(md(tk(both)))} | {', '.join(sorted({x['build'] or '?' for x in xs}))} |")
        L.append("")
    L += ["## Every attempt", "", "| run | package | model | arm | class | rule | bar | build | tokens (M) |", "|---|---|---|---|---|---|---:|---|---:|"]
    for x in out:
        L.append(f"| {x['run_id']} | {x['package']} | {x['model']} | {x['arm']} | {x['class']} | {x['rule']} | {F(x['bar'])} | "
                 f"{x['build'] or ''} | {M(x['tokens_total'])} |")
    open(os.path.join(a.out_dir, "attempts.md"), "w").write("\n".join(L) + "\n")
    print(f"{len(paper)} paper + {len(scr)} screened attempts -> {a.out_dir}/attempts.md, .json;", ", ".join(f"{c} {tot[c]}" for c in CLASSES))

if __name__ == "__main__":
    main()
