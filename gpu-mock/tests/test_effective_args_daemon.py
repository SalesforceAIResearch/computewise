#!/usr/bin/env python3
"""Effective-argument matching:
a launch whose literal argv misses but whose parsed arguments (defaults filled
in) equal a record's explicit flags is that record, `match: exact_effective`; a
different value still misses, with `near_miss` and `fallback_from: near_miss`
on the lookup; an explicit launch flag the record never named misses, unless
its value is the script's own default (the patch sends argv_defaults): then it is
the record, with `defaulted_flags` on the lookup event (2026-10-25)."""
import json, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "daemon"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "torch_patch"))
sys.path.insert(0, os.path.dirname(__file__))
from test_daemon_e2e import start_daemon, read_occ  # noqa: E402
from daemon_client import DaemonClient  # noqa: E402

fails = 0; n = 0
def chk(name, cond, detail=""):
    global fails, n
    n += 1; fails += 0 if cond else 1
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

H = "f" * 64
REC_ARGV = ["custom_selection_eval.py", "--workload", "gsm8k", "--model-id", "Qwen/Qwen2.5-3B-Instruct", "--compression-ratio", "0.8", "--max-examples", "0", "--seed", "42"]
def rec(rid, argv, extra=None):
    r = {"identity": {"record_id": rid, "entry_hash": H, "argv": argv, "argv_exempt": ["--max-examples", "--model-id"], "replicate_group": rid, "match_policy": "exact_only"},
         "mode": "replay", "steps": None, "timing": {"pacing": "total", "wall_total_s": {"value": 7440.4, "provenance": "recorded"}},
         "memory": {"peak_allocated_bytes": 8 * 1024 ** 3, "peak": 8 * 1024 ** 3}, "series": {},
         "result": {"final_score": {"value": 31.842305, "provenance": "recorded"}},
         "materialisation": {"serving": {"mode": "per_example", "generate_seam": "generate_with_selection", "score_seam": "score_prediction",
                                         "n_examples_full": {"value": 1319, "provenance": "derived"}, "score_kind": "binary",
                                         "subset_source": {"flag": "--max-examples", "env": "SELECTION_KV_MAX_EXAMPLES"}, "load_s": {"value": 6.0, "provenance": "recorded"}, "scale": [0, 100]}},
         "readiness": {"replay": True}, "failures": []}
    if extra: r["identity"]["argv"] = argv + extra
    return r
# what the real script's parser yields for `--workload gsm8k --max-examples K` (its defaults: seed 7, ratio 0.8, sdpa, cpu False)
def eff(k, ratio="0.8", **over):
    d = {"workload": "gsm8k", "model_id": "Qwen/Qwen2.5-3B-Instruct", "seed": "7", "compression_ratio": ratio, "attn_implementation": "sdpa", "cpu": False, "max_examples": str(k)}
    d.update(over); return d
with tempfile.TemporaryDirectory() as td:
    R = Path(td) / "records"; R.mkdir()
    (R / "fa.json").write_text(json.dumps(rec("mls-fa-gsm8k", REC_ARGV)))
    (R / "fa05.json").write_text(json.dumps(rec("mls-fa-gsm8k-r05", ["custom_selection_eval.py", "--workload", "gsm8k", "--compression-ratio", "0.5", "--seed", "42"])))
    (R / "cmp.json").write_text(json.dumps(rec("mls-fa-gsm8k-compile", ["custom_selection_eval.py", "--workload", "gsm8k", "--compression-ratio", "0.8", "--compile", "--seed", "42"])))
    (Path(td) / "d").mkdir()
    d, sock, occ = start_daemon(Path(td) / "d", speed=1000.0, registry=R)
    def look(key, argv, e):
        c = DaemonClient(str(sock)); c.hello(argv=argv)
        r = c.lookup_v2(job_key=key, rank=0, world_size=1, entry_hash=H, argv=argv, env={}, subset_env={}, argv_effective=e, script_path="/w/methods/full_attention/custom_selection_eval.py", fingerprint=None, fingerprint_inputs=None, stage="exact")
        c.bye(); c.close(); return r
    r1 = look("A", REC_ARGV[:-2] + ["--max-examples", "5"], eff(5, seed="42"))
    chk("the literal argv (the job script's full line) is still an exact match", r1.get("match") == "exact" and r1.get("record_id") == "mls-fa-gsm8k", str({k: r1.get(k) for k in ("match", "record_id")}))
    r2 = look("B", ["custom_selection_eval.py", "--workload", "gsm8k", "--max-examples", "5"], eff(5))
    chk("haiku's launch (--workload gsm8k, ratio omitted at its default 0.8): exact_effective on the 0.8 record, subset k=5", r2.get("match") == "exact_effective" and r2.get("record_id") == "mls-fa-gsm8k" and r2.get("subset") == {"k": 5, "n_full": 1319}, str({k: r2.get(k) for k in ("match", "record_id", "subset")}))
    r3 = look("C", ["custom_selection_eval.py", "--workload", "gsm8k", "--compression-ratio", "0.5", "--max-examples", "5"], eff(5, ratio="0.5"))
    chk("the same launch with --compression-ratio 0.5 is the 0.5 record (literal exact)", r3.get("match") == "exact" and r3.get("record_id") == "mls-fa-gsm8k-r05", str({k: r3.get(k) for k in ("match", "record_id")}))
    r4 = look("D", ["custom_selection_eval.py", "--workload", "gsm8k", "--compression-ratio", "0.3", "--max-examples", "5"], eff(5, ratio="0.3"))
    chk("a ratio no record has (0.3) misses: mode model, match none", r4.get("mode") != "replay" and r4.get("match") == "none", str({k: r4.get(k) for k in ("mode", "match")}))
    r5 = look("E", ["custom_selection_eval.py", "--workload", "gsm8k", "--attn-implementation", "eager", "--max-examples", "5"], eff(5, attn_implementation="eager"))
    chk("an explicit launch flag no record named (--attn-implementation eager) misses: the record ran the default", r5.get("match") == "none", str(r5.get("match")))
    r6 = look("F", ["custom_selection_eval.py", "--workload", "gsm8k", "--compile", "--max-examples", "5"], eff(5, compile=True))
    chk("a bare flag matches the record that carries it (compile True)", r6.get("match") in ("exact", "exact_effective") and r6.get("record_id") == "mls-fa-gsm8k-compile", str({k: r6.get(k) for k in ("match", "record_id")}))
    r7 = look("G", ["custom_selection_eval.py", "--workload", "longbench_v2", "--max-examples", "5"], eff(5, workload="longbench_v2"))
    chk("a workload no record of this script has misses", r7.get("match") == "none", str(r7.get("match")))
    r8 = look("H2", ["custom_selection_eval.py", "--workload", "gsm8k", "--max-examples", "5"], None)
    chk("no parsed arguments sent (not an argparse script): the literal key stands, miss", r8.get("match") == "none", str(r8.get("match")))
    # --- a launch-explicit flag at the script's OWN default ---
    # The quant script's record names only the job script's flags; the agent's launch spells out
    # --model-id at its default. With the parser's defaults sent (argv_defaults), that is the record;
    # a different value still misses; without defaults (an older patch) it misses as before.
    HQ = "e" * 64
    (R / "kivi.json").write_text(json.dumps(dict(rec("mls-kivi-gsm8k-s42", ["custom_quant_eval.py", "--workload", "gsm8k", "--budget-bits", "4", "--seed", "42"]),
                                                identity={"record_id": "mls-kivi-gsm8k-s42", "entry_hash": HQ, "argv": ["custom_quant_eval.py", "--workload", "gsm8k", "--budget-bits", "4", "--seed", "42"],
                                                          "argv_exempt": ["--max-examples"], "replicate_group": "kivi", "match_policy": "exact_only"})))
    try:
        d.registry.__init__(R)
    except Exception:
        pass
    R4 = ["/home/researcher/project/methods/kivi_overlap_4bit/custom_quant_eval.py", "--workload", "gsm8k", "--max-examples", "0", "--seed", "42", "--model-id", "Qwen/Qwen2.5-3B-Instruct", "--budget-bits", "4"]
    def effq(model="Qwen/Qwen2.5-3B-Instruct", k="0"):
        return {"workload": "gsm8k", "budget_bits": "4", "model_id": model, "max_examples": k, "seed": "42"}
    DEFQ = {"workload": None, "budget_bits": None, "model_id": "Qwen/Qwen2.5-3B-Instruct", "max_examples": "0", "seed": "0"}
    def lookq(key, argv, e, dflt):
        c = DaemonClient(str(sock)); c.hello(argv=argv)
        r = c.lookup_v2(job_key=key, rank=0, world_size=1, entry_hash=HQ, argv=argv, env={}, subset_env={}, argv_effective=e, argv_defaults=dflt,
                        script_path="/home/researcher/project/methods/kivi_overlap_4bit/custom_quant_eval.py", fingerprint=None, fingerprint_inputs=None, stage="exact")
        c.bye(); c.close(); return r
    q1 = lookq("Q1", R4, effq(), DEFQ)
    chk("r4's argv (--model-id at the script's default, defaults sent): exact_effective on the kivi record", q1.get("match") == "exact_effective" and q1.get("record_id") == "mls-kivi-gsm8k-s42", str({k: q1.get(k) for k in ("match", "record_id", "mode")}))
    q2 = lookq("Q2", [a if a != "Qwen/Qwen2.5-3B-Instruct" else "Other/Model-7B" for a in R4], effq(model="Other/Model-7B"), DEFQ)
    chk("the same launch with --model-id Other/Model-7B (not the default) misses", q2.get("match") == "none" and q2.get("mode") != "replay", str({k: q2.get(k) for k in ("match", "mode")}))
    q3 = lookq("Q3", R4, effq(), None)
    chk("no defaults sent (an older patch): the explicit flag still misses, as before", q3.get("match") == "none", str(q3.get("match")))
    q4 = lookq("Q4", ["custom_quant_eval.py", "--workload", "gsm8k", "--budget-bits", "4", "--seed", "42"], effq(), DEFQ)
    chk("the job script's own line stays a literal exact match", q4.get("match") == "exact" and q4.get("record_id") == "mls-kivi-gsm8k-s42", str(q4.get("match")))
    evq = [e for e in read_occ(occ) if e["kind"] == "lookup" and e.get("job_key") in ("Q1", "Q2", "Q3")]
    byq = {e["job_key"]: e for e in evq}
    chk("the Q1 lookup event names the defaulted flag", byq.get("Q1", {}).get("defaulted_flags") == ["--model-id"], str(byq.get("Q1", {}).get("defaulted_flags")))
    nmq = [e for e in read_occ(occ) if e["kind"] == "near_miss" and e.get("job") == byq.get("Q2", {}).get("job")]
    chk("Q2's near_miss names --model-id Other/Model-7B as only_launch", bool(nmq) and "Other/Model-7B" in (nmq[0].get("only_launch") or []), str([e.get("only_launch") for e in nmq]))
    ev = read_occ(occ)
    looks = {e["job_key"] if "job_key" in e else e["job"]: e for e in ev if e["kind"] == "lookup"}
    byjob = [e for e in ev if e["kind"] == "lookup"]
    chk("the lookup events carry match exact_effective for B and fallback_from near_miss for the misses D, E, G", [e.get("match") for e in byjob][1] == "exact_effective" and all(e.get("fallback_from") == "near_miss" for e in byjob if e.get("match") == "none"), str([(e.get("match"), e.get("fallback_from")) for e in byjob]))
    nm = [e for e in ev if e["kind"] == "near_miss"]
    chk("near_miss events name the differing tokens (D: ratio 0.3 vs 0.8/0.5)", any(e.get("only_launch") == ["0.3"] for e in nm), str([(e.get("only_launch"), e.get("only_record")) for e in nm][:4]))
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
