"""mls-kv-gate: the goal-only MLS-KV package, an acceptance rule over MLS-Bench's KV-cache evaluations (gpu-tasks-kvgate,
2026-10-23; package G4 in design/EVALUATION-PLAN.md, "Package search"; ordered by the study 15:20Z, rule fixed by
the analysis 10:25 MDT).

usage: from gen_tasks.main, as one more builder: lambda r: __import__("gen_mlskv_gate").build(r, disclosure_of, REG_DIR)

The same 40 records, per_example serving, files, environment and image as mls-kv-deploy and mls-kv-report (build/gen_mlskv_task.py,
imported read-only). Only the brief and the reference plan differ. The brief states the team's acceptance rule and asks for a
decision: deploy a method only if, on the whole workloads, it scores within 2 points of full attention on every workload both run.
It names no procedure. Which evaluations to run in full is the agent's choice, which is what makes the package goal-only.

What the rule does on the records (checked against them on 2026-10-23, survey/mls-kv-gate-answer-2026-10-23.json):
- it passes kivi_overlap_4bit alone. Every other method fails on a 5-22-minute LongBench-E workload: squat_subspace_4bit and
  both kvtuner4 methods on passage retrieval (-2.45, -5.31, -5.78), the retention policies on passage retrieval or hotpotqa;
- so the rule needs about half of the 40 evaluations, and none of the long GSM8K runs but kivi's and full attention's. squat's
  8.1-h GSM8K, the kvtuner4 methods' 3.5-h ones and the retention policies' 2-h ones are evidence the rule never asks for;
- a --max-examples screen cannot satisfy it (the rule reads whole workloads), and on the served vectors a k=50 screen reads
  kivi's passage retrieval about 12 points low, so an agent that stops at screens rejects the right answer.

The reference plan is the rule run brief-only and decisive-first (after the harness's decisive_first.py, 2026-10-23):
on each free GPU the shortest full evaluation still needed by a method not yet rejected; a method rejected on its first full
workload more than 2 below full attention's; a method's long GSM8K started once it has passed SPEC_PASSES of its other shared
workloads. It sees only finished full scores. The trace it produces is written as the plan's instances, with `after` on the
GSM8K launches, so plan.py and expected_scores.py run it as launches (bar 2). Its figures are computed in gate_plan and stored
in reference_plan; none is written here, so they cannot go stale.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import gen_mlskv_task as K  # noqa: E402

TASK_ID = "mls-kv-gate"
# H7, mls-kv-gate-h: the same task with the acceptance rule in
# methods/ACCEPTANCE.md and the methods paragraph moved into methods/task_description.md, so the brief names neither. Jobs,
# records, reference_plan, decisions, box, answer key and horizon are mls-kv-gate's, so screen.py's plan_key is the same and
# mls-kv-gate's W0s are its reference. The files state the rule, never which check to make first or which workloads are short.
ACCEPTANCE = """# Acceptance rule

This is the team's rule for deploying a KV-cache compression method, with full_attention as the reference: deploy a method only
if, on the whole workloads, it scores within 2 points of full_attention on every workload both run.
"""
RULE_POINTS = 2.0
SPEC_PASSES = 2        # a live method's GSM8K starts once it has passed this many of its other shared workloads
REF = "full_attention"


def scores(jobs, stats):
    """(method, workload label) -> (recorded full score, paced one-GPU seconds), from the deploy build's jobs and stats."""
    out = {}
    for j in jobs:
        m, label = j["job_id"].split("--")
        out[(m, label)] = (stats[j["job_id"]]["final_score"], stats[j["job_id"]]["paced_s"])
    return out


def rule_trace(S):
    """The brief's rule, brief-only and decisive-first on 4 one-GPU lanes. Returns the launches in start order, each
    (method, label, start_s, end_s), the accepted methods and, per rejected method, the workload that rejected it."""
    import heapq
    cands = sorted({m for m, _ in S} - {REF})
    shared = {m: [w for (mm, w) in S if mm == m and (REF, w) in S] for m in cands}
    done, running, started = {}, [], {}

    def failed(m):
        return next((w for w in shared[m] if (m, w) in done and (REF, w) in done
                     and S[(m, w)][0] < S[(REF, w)][0] - RULE_POINTS), None)

    def status(m):
        if failed(m):
            return "rejected"
        return "accepted" if all((m, w) in done and (REF, w) in done for w in shared[m]) else "live"

    def n_passed(m):
        return sum((m, w) in done and (REF, w) in done and S[(m, w)][0] >= S[(REF, w)][0] - RULE_POINTS
                   for w in shared[m] if w != "gsm8k")

    released = {}

    def others_passed(m):
        need = min(SPEC_PASSES, len([w for w in shared[m] if w != "gsm8k"]))
        if n_passed(m) >= need and m not in released:
            released[m] = [(q, w) for w in shared[m] if w != "gsm8k" for q in (m, REF)
                           if (m, w) in done and (REF, w) in done and S[(m, w)][0] >= S[(REF, w)][0] - RULE_POINTS]
        return n_passed(m) >= need

    def pick():
        live = [m for m in cands if status(m) == "live"]
        taken = {x for _, x in running} | set(done)
        need = ({(m, w) for m in live for w in shared[m]} | {(REF, w) for m in live for w in shared[m]}) - taken
        # a method's long GSM8K as soon as the rule releases it, then the shortest decisive evaluation
        gsm = sorted((x for x in need if x[1] == "gsm8k" and x[0] != REF and others_passed(x[0])), key=lambda x: (-S[x][1], x))
        if gsm:
            return gsm[0]
        short = sorted((x for x in need if x[1] != "gsm8k"), key=lambda x: (S[x][1], x))
        if short:
            return short[0]
        if (REF, "gsm8k") in need and any((m, "gsm8k") in taken for m in live):
            return (REF, "gsm8k")
        return None

    free = [0.0] * 4
    while True:
        t = heapq.heappop(free)
        for e, x in sorted(r for r in running if r[0] <= t + 1e-9):
            done[x] = e
            running.remove((e, x))
        x = pick()
        if x is None:
            if not running:
                break
            heapq.heappush(free, min(e for e, _ in running))
            continue
        started[x] = t
        running.append((t + S[x][1], x))
        heapq.heappush(free, t + S[x][1])
    launches = [(x[0], x[1], started[x], done[x]) for x in sorted(done, key=lambda x: (started[x], x))]
    acc = [m for m in cands if status(m) == "accepted"]
    killed = [x for x in started if x not in done]
    if killed:
        raise SystemExit(f"{TASK_ID}: the trace started {killed} and a later result rejected the method; the plan cannot carry a kill")
    return launches, acc, {m: failed(m) for m in cands if status(m) == "rejected"}, released


def _wrap(text, width=128):
    """The methods paragraph wrapped as task_description.md's other lines are (about 128 columns)."""
    import textwrap
    return "\n".join(textwrap.wrap(text, width=width))


def gate_dependency(S, launches, gate):
    """The evidence launch the GSM8K confirms wait on: the one of `gate` that ends last when the plan's launches run packed
    on 4 lanes, longest paced duration first (plan.py's and expected_scores.py's packed order), the GSM8K launches held back.
    Waiting on it means the rule's evidence is all in before a GSM8K starts, whatever order the shorter launches take."""
    short = sorted(((m, w) for m, w, _, _ in launches if w != "gsm8k"), key=lambda x: (-S[x][1], x))
    lanes, end = [0.0] * 4, {}
    for x in short:
        i = lanes.index(min(lanes))
        end[x] = lanes[i] + S[x][1]
        lanes[i] = end[x]
    return max(gate, key=lambda x: (end[x], x))


def gate_plan(jobs, stats):
    S = scores(jobs, stats)
    launches, acc, rejected, released = rule_trace(S)
    if acc != ["kivi_overlap_4bit"]:
        raise SystemExit(f"{TASK_ID}: the rule's answer on the records moved to {acc}; recheck the brief and the answer key")
    gate = released[acc[0]]               # the evaluations whose passing released the accepted method's GSM8K
    dep = gate_dependency(S, launches, gate)
    inst = []
    for m, w, _, _ in launches:
        e = {"job_id": f"{m}--{w}", "count": 1, "width": 1, "phase": "confirm" if w == "gsm8k" else "decide"}
        if w == "gsm8k":
            # released when the accepted method's other shared workloads and full attention's have passed the rule.
            # expand_reference_plan maps one "<job_id> screen" entry to that pool's first instance, so the release is the
            # one evidence launch that ends last when the plan runs packed (longest disclosed first): gate_dependency
            e["after"] = f"{dep[0]}--{dep[1]} screen"
            # every evaluation the release rests on: plan.py, expect.py and expected_scores.py honour a list of `after`, and
            # common.expand_reference_plan reads one entry per plan item, so the whole set is carried here for it to map
            #; until it does, `after` alone is exact for packed (gate_dependency)
            e["after_all"] = [f"{a}--{b} screen" for a, b in sorted(gate)]
            e["after_how"] = (f"released when {acc[0]} has passed {SPEC_PASSES} of its shared non-GSM8K workloads against full "
                              f"attention's ({', '.join(f'{a} {b}' for a, b in gate)}): packed runs {dep[0]} on {dep[1]} last "
                              "of those, so this waits on it (gen_mlskv_gate.gate_dependency)")
        inst.append(e)
    makespan = max(e0 for _, _, _, e0 in launches)
    gpu_s = sum(e0 - s0 for _, _, s0, e0 in launches)
    return {
        "instances": inst,
        "rule": {"text": "a method passes when, on every workload it shares with full_attention, its whole-workload score is no "
                         f"more than {RULE_POINTS:g} points below full_attention's whole-workload score",
                 "points": RULE_POINTS, "reference_method": REF, "reads": "whole workloads only (--max-examples 0)",
                 "accepted": acc, "rejected_on": rejected},
        "trace_makespan_s": round(makespan, 1), "trace_gpu_s": round(gpu_s, 1), "evaluations": len(launches),
        "how": ("the brief's rule run brief-only and decisive-first: on each free GPU the shortest full evaluation still needed "
                f"by a method not yet rejected; a method rejected on its first full workload more than {RULE_POINTS:g} below full "
                f"attention's; a method's long GSM8K started once it has passed {SPEC_PASSES} of its other shared workloads (a "
                "promising method's long run started before its last short one, which a later failure would have to kill; on "
                "these records none is killed), and full attention's GSM8K with it. The verdict still rests on every shared "
                "whole workload. It sees only finished full scores, never a recorded outcome before its launch ends, so it is a "
                f"plan an agent could follow. On the records: {len(launches)} of 40 evaluations, {gpu_s / 3600:.2f} GPU-h, "
                f"{makespan / 3600:.2f} h on 4 GPUs (gpu-tasks-kvgate's trace). Starting GSM8K only after all three short "
                "workloads pass, as the harness's decisive_first.py does, gives 4.82-4.83 h. The instances are the "
                "trace in launch order, the two GSM8K launches released by `after`. No --max-examples screens: they cannot "
                "satisfy the rule, and on the served vectors they misread kivi's passage retrieval"),
    }


def build(recs, disclosure_of, reg=None, hidden=False):
    """hidden (H7): see ACCEPTANCE above; the task id is mls-kv-gate-h and its level hidden_plan."""
    t = K.build(recs, disclosure_of, reg, variant="mls-kv-deploy")
    stats = t["candidates_stats"]
    plan = gate_plan(t["jobs"], stats)
    methods_text = t["brief"]["text"].split("cutting the cache. ", 1)[1].split(" Which method should we deploy?")[0]
    if hidden:
        # the methods paragraph moves into the existing task_description.md as its own section; ACCEPTANCE.md holds the rule
        for f in t["files"]:
            if f["path"] == "methods/task_description.md":
                f["content"] = f["content"].rstrip("\n") + "\n\n## The methods\n\n" + _wrap(methods_text) + "\n"
        t["files"].insert(1, {"path": "methods/ACCEPTANCE.md", "content": ACCEPTANCE})
    brief = ("We need a KV-cache compression method for serving Qwen2.5-3B-Instruct. The candidates are in methods/, and "
             "ACCEPTANCE.md there is the team's acceptance rule. Which method, if any, passes it? Report the whole-workload scores "
             "the decision rests on, with each workload's number of examples, and your recommendation. The machine has four "
             "H100s.") if hidden else (
             "We need a KV-cache compression method for serving Qwen2.5-3B-Instruct, and the team's acceptance rule is fixed: "
             "deploy a method only if, on the whole workloads, it scores within 2 points of full_attention on every workload "
             "both run. " + methods_text + " Which method, if any, passes the rule? Report the whole-workload scores the "
             "decision rests on, with each workload's number of examples, and your recommendation. The machine has four H100s.")
    t.update({
        "task_id": TASK_ID + ("-h" if hidden else ""),
        **({"level": "hidden_plan"} if hidden else {}),
        "set_how": ("H7 (the analysis, 2026-10-25 16:39 MDT): mls-kv-gate with its acceptance rule in "
                    "methods/ACCEPTANCE.md and the methods paragraph in methods/task_description.md, at the hidden-plan level; "
                    "the same jobs, plan and answer, so mls-kv-gate's W0s are its reference" if hidden else
                    "the analysis, 2026-10-23: package G4 of the package search, the goal-only MLS-KV "
                    "package: mls-kv-deploy's records with an acceptance rule the agent must satisfy on whole workloads"),
        "title": ("H7, hidden plan: which KV-cache compression method passes the team's acceptance rule in methods/ACCEPTANCE.md, "
                  "from evaluations the agent chooses; every job one GPU" if hidden else
                  "Goal-level: which KV-cache compression method passes the team's acceptance rule, from evaluations the agent "
                  "chooses; every job one GPU"),
        "brief": {"text": brief, "provenance": (
            "written by gpu-tasks-kvgate from mls-kv-deploy's brief (MLS-Bench tasks/llm-kv-selection-budgeting and "
            "tasks/llm-kv-adaptive-quantization at b8739449): the same question and methods paragraph, with the acceptance "
            "rule the analysis fixed (2 points absolute against full_attention, on every workload both "
            "run, whole workloads; 2026-10-23 10:25 MDT). A rule, not a procedure: it says what an answer must rest on, not "
            "which evaluations to run or in what order. No deadline or time words (the study, 2026-10-20); the GPU count "
            "stated" + ("; H7 (2026-10-25): the rule moved to methods/ACCEPTANCE.md and the methods paragraph to "
                        "methods/task_description.md, neither saying which check to make first" if hidden else ""))},
        "reference_plan": plan,
        "measuring_run": {"policy": "packed", "speed": 1, "plan": "reference_plan",
                          "how": ("brief-only: plan.py runs reference_plan.instances, the rule's trace, never a recorded "
                                  "winner (bar 2). The scripts, image and records are mls-kv-deploy's, so its measured host "
                                  "terms (W0-mlskv-measure-c) transfer by build/transfer_host_terms.py; a speed-10 check10 of "
                                  "this plan confirms the transfer")},
        "answer": {"accepted": plan["rule"]["accepted"], "rule_points": RULE_POINTS,
                   "rejected_on": plan["rule"]["rejected_on"],
                   "margins": {"kivi_overlap_4bit": ("passes by 0.67 points on its tightest workload (passage retrieval, 61.08 "
                                                     "against 62.42)"),
                               "squat_subspace_4bit": "fails by 0.45 points (passage retrieval, 59.96 against 62.42, -2.45)"},
                   "how": ("the brief's rule on the recorded whole-workload scores, which a full launch prints exactly (gpu-mock's "
                           "per_example vector averages to the record's final_score), so the verdict carries no sampling noise "
                           "in the served world. The margins are thin: a rule of 3 points passes squat too and makes its 8.1-h "
                           "GSM8K necessary, which erases the package's waste (the harness, 2026-10-23)"),
                   "grading": ("bar 3 on a goal-only package is the answer (the analysis, 09:33 MDT): a pick of "
                               "kivi_overlap_4bit supported by whole-workload scores for its four shared workloads and full "
                               "attention's, under the rule; anything else, or an unsupported pick, is an early exit")},
    })
    t["notes"] = [n for n in t["notes"] if not n.startswith("At the 20% budget the three retention policies")] + [
        ("Why this is goal-only: the rule says what an answer must rest on, and the agent decides which of the 40 evaluations "
         f"to run in full. On the records the rule needs {plan['evaluations']} of them ({plan['trace_gpu_s'] / 3600:.1f} GPU-h, "
         f"{plan['trace_makespan_s'] / 3600:.2f} h on 4 GPUs, decisive-first; reference_plan). Running all 40 in full, or any "
         "plan that includes squat's 8.1-h GSM8K, cannot finish under 8.1 h. That waste is evaluations the rule never needed, so "
         "a scheduler packs it perfectly and does not remove it (survey/mls-kv-gate-answer-2026-10-23.json has the ratios)."),
        ("Shared with mls-kv-deploy and mls-kv-report: the same 40 records, so no fresh twin can be built for this package from "
         "different records (design/EVALUATION-PLAN.md, selection guard). It is development and reported as such."),
        ("A --max-examples screen cannot satisfy the rule, which reads whole workloads; it can order the work. On the served "
         "per-example vectors the first 50 examples of kivi's passage retrieval read about 12 points low (solution's "
         "screen_check.py), so stopping at a screen rejects the right answer."),
    ]
    return t
