"""Known-answer test for host-term derivation and transfer.

usage: python3 -I build/test_host_terms.py [<registry dir>] [--keep DIR]

--keep DIR copies the emitted run directory (with its run.json) and the terms that went in to DIR, so another derivation
(the runner's scripted/host_terms.py) can be run on the same run and compared.

1. Simulates pkg-p20-par under `packed` with chosen host terms (build/expected_scores.py, emit), which writes a run directory in
   run_one's shape: occupancy events, a transcript, status.json and task.json; a run.json is added here.
2. Derives the terms back from that directory (build/derive_host_terms.py) and checks every term against what went in:
   hello, post, cold and warm start-up, and each per_event gap class.
3. Transfers them to every other mng3 package (build/transfer_host_terms.py) and checks every script of every package is
   found.
Exit status 1 on any mismatch. Writes only under a fresh temporary directory. Run it after changing any of those three files,
or common.gap_classes / common.unit_kinds / common.shard_load_steps.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = sys.argv[1:]
KEEP = ARGS[ARGS.index("--keep") + 1] if "--keep" in ARGS else None
POS = [x for i, x in enumerate(ARGS) if x != "--keep" and (i == 0 or ARGS[i - 1] != "--keep")]
REG = POS[0] if POS else os.path.join(HERE, "..", "registry")
PE = {"once_s": 1.5, "val_after_s": 0.5, "val_before_s": 0.3, "shard_s": 0.4, "step_s": 0.01}
TERMS = {"hello_s": 2.0, "pre_s": 12.0, "pre_s_first": 30.0, "pre_s_rest": 12.0, "post_s": 1.5, "gaps": [], "per_event": PE}
KEYS = ["contra-soft-muon/contra-muon-to-soft-muon.py", "soda-fade/contra-muon-to-soft-muon.py",
        "radial-damp/radial_scale_then_radius_correct.py", "ema-nesterov-aurora/radial_scale_then_radius_correct.py",
        "train_gpt_simple_aurora_ema_ref.py", "train_gpt_circuit_muon.py", "train_gpt_simple_muonh.py",
        "train_gpt_simple_newton_muon.py", "train_gpt_simple.py"]
OTHERS = ["pkg-p3-par", "pkg-p6-par", "pkg-p10-par", "pkg-p20-tail", "pkg-p3-par-long", "pkg-p6-par-long", "pkg-p6-chain-long"]


def run(*args):
    return subprocess.run([sys.executable, "-I", *args], check=True, capture_output=True, text=True).stdout


def main():
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        tf = os.path.join(tmp, "terms.json")
        json.dump({"terms": {k: json.loads(json.dumps(TERMS)) for k in KEYS}}, open(tf, "w"))
        run(os.path.join(HERE, "expected_scores.py"), REG, "pkg-p20-par", "packed", "--host", tf, "--emit", os.path.join(tmp, "emit"),
            "--json")
        d = os.path.join(tmp, "emit", "packed")
        json.dump({"inputs": {"speed": 1.0}, "image": {"name": "test", "id": "sha256:test", "manifest": {}},
                   "host": {"machine_type": "test-type"}, "caps": {}, "cost_tag": "known-answer", "scripted": {"policy": "packed"}},
                  open(os.path.join(d, "run.json"), "w"))
        got = json.loads(run(os.path.join(HERE, "derive_host_terms.py"), d, "--registry", REG))
        if KEEP:
            if os.path.exists(KEEP):
                raise SystemExit(f"{KEEP} exists; --keep wants a new directory")
            shutil.copytree(d, os.path.join(KEEP, "run"))
            shutil.copy(tf, os.path.join(KEEP, "terms-in.json"))
            json.dump(got, open(os.path.join(KEEP, "terms-derived.json"), "w"), indent=1)
        for k in KEYS:
            g = got["terms"].get(k)
            if g is None:
                print(f"FAIL {k}: not derived")
                bad += 1
                continue
            checks = [("hello_s", g["hello_s"], 2.0), ("post_s", g["post_s"], 1.5)]
            if "pre_s_first" in g:
                checks.append(("pre_s_first", g["pre_s_first"], 30.0))
            if g.get("pre_s_rest") is not None:
                checks.append(("pre_s_rest", g["pre_s_rest"], 12.0))
            if "pre_s_first" not in g:
                checks.append(("pre_s (one cold job)", g["pre_s"], 30.0))
            checks += [(f"per_event.{c}", (g.get("per_event") or {}).get(c), v) for c, v in PE.items()]
            for name, x, want in checks:
                if x is None or abs(x - want) > 1e-3:
                    print(f"FAIL {k} {name}: {x}, want {want}")
                    bad += 1
        src = os.path.join(tmp, "pkg-p20-par.test-type.json")
        got["task_id"] = "pkg-p20-par"
        json.dump(got, open(src, "w"))
        try:
            out = run(os.path.join(HERE, "transfer_host_terms.py"), src, *OTHERS, "--out-dir", os.path.join(tmp, "xfer"),
                      "--registry", REG)
            print(out.strip())
        except subprocess.CalledProcessError as e:
            print("FAIL transfer:", e.stdout, e.stderr)
            bad += 1
    print(f"{'ok' if not bad else 'FAILED'}: {len(KEYS)} scripts derived back, transfer to {len(OTHERS)} packages; {bad} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
