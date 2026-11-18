#!/usr/bin/env python3
"""Write the TTC probe's whole_job record to <out>. If <realrec> exists, use it
(Sol's overlay record for single_cell.py), with readiness.replay forced True FOR
THE PROBE ONLY (labelled; the real record keeps its blockers and stays readiness
0). Else write a stand-in keyed to single_cell.py's sha256 <sc_sha> with a chosen
metric value (labelled STANDIN; not TabArena's numbers). Used by
probe_ttc_torch291.sh so the probe needs no nested heredoc.
    python3 probe_mkrec.py <out.json> <single_cell.py sha256> <realrec path>
"""
import json
import os
import sys

out, sc_sha, realrec = sys.argv[1], sys.argv[2], sys.argv[3]

if os.path.isfile(realrec):
    r = json.load(open(realrec))
    rd = r.setdefault("readiness", {})
    rd["replay"] = True
    rd["probe_forced"] = ("blockers cleared for the torch-2.9.1 compatibility probe, 2026-11-02; "
                          "NOT a replay-ready record")
    json.dump(r, open(out, "w"))
    ser = (r.get("materialisation") or {}).get("serving") or {}
    print("REAL record", r["identity"]["record_id"], "entry", str(r["identity"].get("entry_hash"))[:16],
          "units", ser.get("n_units"), "seam", ser.get("seam"))
else:
    rec = {"identity": {"record_id": "ttc-standin", "entry_hash": sc_sha,
                        "argv": ["single_cell.py", "--dataset", "Bioresponse", "--split", "0",
                                 "--arm", "tabpfn3_aggregation_96"],
                        "match_policy": "exact_only", "replicate_group": "ttc-standin"},
           "mode": "replay", "steps": None,
           "timing": {"pacing": "total", "wall_total_s": {"value": 300.0, "provenance": "recorded", "how": "standin"}},
           "memory": {"peak_allocated_bytes": 4 << 30, "peak": 4 << 30}, "series": {},
           "result": {"metrics": {"value": {"metric_error": 0.1234, "time_train_s": 200.0,
                                            "time_infer_s": 100.0, "metric": "log_loss"},
                                  "provenance": "recorded", "how": "standin: a chosen value, NOT TabArena's"}},
           "materialisation": {"serving": {"mode": "whole_job",
                               "seam": "tabarena.benchmark.experiment.experiment_runner:ExperimentRunner._run",
                               "unit": "phase", "n_units": 5, "unit_shares": [0.13, 0.13, 0.14, 0.3, 0.3],
                               "result_key": "metrics", "result_kind": "metric dict", "result_shape": "value"}},
           "readiness": {"replay": True}, "failures": []}
    json.dump(rec, open(out, "w"))
    print("STANDIN record for single_cell.py", sc_sha[:16])
