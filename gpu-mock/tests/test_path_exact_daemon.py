#!/usr/bin/env python3
"""path_exact: an exact_only record that
names its path (identity.bound_path) serves an EDITED copy of its script at that
path with its argv, literal or effective; a new config at a known path still
misses (modelled, near_miss); a flattened copy matches by basename only when one
family has such records; path_bound records keep their own stage and label."""
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

H_FA, H_KIVI, H_STREAM, H_H4 = "a" * 64, "b" * 64, "c" * 64, "d" * 64
H_EDIT = "e" * 64          # the edited copy's bytes: no record has them
SEL = "custom_selection_eval.py"; QNT = "custom_quant_eval.py"
def rec(rid, h, bp, argv, policy="exact_only", pe=True):
    r = {"identity": {"record_id": rid, "entry_hash": h, "argv": argv, "argv_exempt": ["--max-examples", "--model-id"], "replicate_group": rid,
                      "match_policy": policy, "bound_path": bp},
         "mode": "replay", "steps": None if pe else 10, "timing": ({"pacing": "total", "wall_total_s": {"value": 700.0, "provenance": "recorded"}} if pe
                                                                   else {"pacing": "per_step", "per_step_gpu_s": {"value": 0.01, "provenance": "recorded"}, "steady_step_s": 0.01}),
         "memory": {"peak_allocated_bytes": 8 << 30, "peak": 8 << 30}, "series": ({} if pe else {"train_loss": {"steps": list(range(1, 11)), "values": [1.0] * 10}}),
         "result": {"final_score": {"value": 40.0, "provenance": "recorded"}}, "readiness": {"replay": True}, "failures": []}
    if pe:
        r["materialisation"] = {"serving": {"mode": "per_example", "generate_seam": "generate_with_selection", "score_seam": "score_prediction",
                                            "n_examples_full": {"value": 1319, "provenance": "derived"}, "score_kind": "binary",
                                            "subset_source": {"flag": "--max-examples", "env": "SELECTION_KV_MAX_EXAMPLES"}, "load_s": {"value": 6.0, "provenance": "recorded"}, "scale": [0, 100]}}
    return r
def eff(workload="gsm8k", ratio="0.8", k="5", **o):
    d = {"workload": workload, "model_id": "Qwen/Qwen2.5-3B-Instruct", "seed": "7", "compression_ratio": ratio, "attn_implementation": "sdpa", "cpu": False, "max_examples": k}
    d.update(o); return d
with tempfile.TemporaryDirectory() as td:
    R = Path(td) / "records"; R.mkdir()
    (R / "fa.json").write_text(json.dumps(rec("fa-gsm8k", H_FA, "methods/full_attention/" + SEL, [SEL, "--workload", "gsm8k", "--compression-ratio", "0.8", "--seed", "42"])))
    (R / "fa-lb.json").write_text(json.dumps(rec("fa-longbench", H_FA, "methods/full_attention/" + SEL, [SEL, "--workload", "longbench_v2", "--compression-ratio", "0.8", "--seed", "42"])))
    (R / "stream.json").write_text(json.dumps(rec("stream-gsm8k", H_STREAM, "methods/streamingllm/" + SEL, [SEL, "--workload", "gsm8k", "--compression-ratio", "0.8", "--seed", "42"])))
    (R / "kivi.json").write_text(json.dumps(rec("kivi-gsm8k", H_KIVI, "methods/kivi/" + QNT, [QNT, "--workload", "gsm8k", "--seed", "42"])))
    (R / "h4.json").write_text(json.dumps(rec("h4-run", H_H4, "tasks/h4/run.py", ["run.py", "--epochs", "3"], policy="path_bound", pe=False)))
    (R / "h1.json").write_text(json.dumps(rec("h1-muonh", "9" * 64, "setups/muonh-tuned/", ["train_gpt.py"], policy="path_bound", pe=False)))
    (Path(td) / "d").mkdir()
    d, sock, occ = start_daemon(Path(td) / "d", speed=1000.0, registry=R)
    jobs = 0
    def look(h, argv, e, path):
        global jobs; jobs += 1
        c = DaemonClient(str(sock)); c.hello(argv=argv)
        r = c.lookup_v2(job_key=f"J{jobs}", rank=0, world_size=1, entry_hash=h, argv=argv, env={}, subset_env={}, argv_effective=e,
                        script_path=path, fingerprint=None, fingerprint_inputs=None, stage="exact")
        c.bye(); c.close(); return r
    A = [SEL, "--workload", "gsm8k", "--compression-ratio", "0.8", "--max-examples", "5", "--seed", "42"]
    r = look(H_FA, A, eff(), "/home/researcher/scratch/" + SEL)
    chk("1: identical bytes anywhere (a copy) is exact, as before", r.get("match") == "exact" and r.get("record_id") == "fa-gsm8k", str({k: r.get(k) for k in ("match", "record_id")}))
    r = look(H_EDIT, A, eff(), "/home/researcher/scratch/methods/full_attention/" + SEL)
    chk("2: an EDITED copy under .../methods/full_attention/ with the recorded argv: path_exact, the fa record, subset k=5", r.get("match") == "path_exact" and r.get("record_id") == "fa-gsm8k" and r.get("subset") == {"k": 5, "n_full": 1319}, str({k: r.get(k) for k in ("match", "record_id", "subset")}))
    r = look(H_EDIT, [SEL, "--workload", "longbench_v2", "--compression-ratio", "0.8", "--max-examples", "5", "--seed", "42"], eff("longbench_v2"), "/w/methods/full_attention/" + SEL)
    chk("3: the argv picks the record within the path (longbench_v2)", r.get("match") == "path_exact" and r.get("record_id") == "fa-longbench", str({k: r.get(k) for k in ("match", "record_id")}))
    r = look(H_EDIT, [SEL, "--workload", "gsm8k", "--max-examples", "5"], eff(), "/w/methods/full_attention/" + SEL)
    chk("4: effective arguments (ratio omitted at its default) serve through path_exact too", r.get("match") == "path_exact" and r.get("record_id") == "fa-gsm8k", str({k: r.get(k) for k in ("match", "record_id")}))
    r = look(H_EDIT, [SEL, "--workload", "gsm8k", "--compression-ratio", "0.3", "--max-examples", "5"], eff(ratio="0.3"), "/w/methods/full_attention/" + SEL)
    chk("5: a known path with a config no record has (ratio 0.3) misses: mode model, match none (never a guess)", r.get("match") == "none" and r.get("mode") != "replay", str({k: r.get(k) for k in ("match", "mode")}))
    r = look(H_EDIT, A, eff(), "/home/researcher/scratch/" + SEL)
    chk("6: a flattened edited copy whose basename two families share (full_attention, streamingllm): ambiguous, miss", r.get("match") == "none", str(r.get("match")))
    r = look(H_EDIT, [QNT, "--workload", "gsm8k", "--max-examples", "5", "--seed", "42"], None, "/home/researcher/scratch/" + QNT)
    chk("7: a flattened edited copy of the one family with that basename (kivi): path_exact_basename", r.get("match") == "path_exact_basename" and r.get("record_id") == "kivi-gsm8k", str({k: r.get(k) for k in ("match", "record_id")}))
    r = look(H_EDIT, ["run.py", "--epochs", "7"], None, "/w/tasks/h4/run.py")
    chk("8: a path_bound record (cpu-tasks) keeps its own stage and label: match path, argv not checked", r.get("match") == "path" and r.get("record_id") == "h4-run", str({k: r.get(k) for k in ("match", "record_id")}))
    r = look(H_EDIT, ["train_gpt_simple_muonh.py", "--seed", "2", "--stop-at", "4"], None, "/home/researcher/project/setups/muonh-tuned/train_gpt_simple_muonh.py")
    chk("12: a DIRECTORY-bound record (setups/muonh-tuned/) serves whatever the agent named its script there, seed and stop-at ignored: match path", r.get("match") == "path" and r.get("record_id") == "h1-muonh", str({k: r.get(k) for k in ("match", "record_id")}))
    r = look(H_EDIT, ["train_gpt.py", "--seed", "99"], None, "/home/researcher/project/setups/muonh-tuned/run1/train_gpt.py")
    chk("13: nested under the bound directory also matches", r.get("match") == "path" and r.get("record_id") == "h1-muonh", str(r.get("match")))
    r = look(H_EDIT, ["train_gpt.py"], None, "/home/researcher/project/setups/ema-nesterov/train_gpt.py")
    chk("14: another setup's directory does not", r.get("match") == "none", str(r.get("match")))
    ev = read_occ(occ)
    lk = [e for e in ev if e["kind"] == "lookup"]
    pe = [e for e in lk if str(e.get("match") or "").startswith("path_exact")]
    chk("9: every path_exact lookup logs both entry hashes (live edited, record) and the bound path", pe and all(e.get("entry_hash_live") == H_EDIT and e.get("entry_hash_record") and e.get("bound_path") for e in pe), str([(e.get("match"), (e.get("entry_hash_live") or "")[:4], (e.get("entry_hash_record") or "")[:4], e.get("bound_path")) for e in pe]))
    chk("10: the misses say why: suffix_argv_miss for the new config, ambiguous for the flattened shared basename (a miss outside any bound path says nothing)", [e.get("path_exact") for e in lk if e.get("match") == "none"] == ["suffix_argv_miss", "ambiguous", None], str([e.get("path_exact") for e in lk if e.get("match") == "none"]))
    nm = [e for e in ev if e["kind"] == "near_miss"]
    chk("11: the new-config miss still gets a near_miss (the modelled path is disclosed)", any(e.get("only_launch") == ["0.3"] for e in nm), str([(e.get("only_launch"), e.get("only_record")) for e in nm]))
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
