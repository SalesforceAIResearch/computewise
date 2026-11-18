"""Expected scores for a task whose jobs mix widths, so packed_overlap refuses it.

usage: python3 -I build/merge_expected.py <registry> <task_id> [--host <host terms>]   (writes the merged JSON to stdout)

Runs build/expected_scores.py once per policy except packed_overlap and merges the outputs into the shape `all` writes, with
packed_overlap named under `policies_not_run` and why."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
POLICIES = ["serial", "packed", "packed_cp", "delayed", "quit_after_first", "quit_while_running", "quit_at_first_completion"]


def main():
    reg, tid, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    out = None
    for p in POLICIES:
        d = json.loads(subprocess.run([sys.executable, "-I", os.path.join(HERE, "expected_scores.py"), reg, tid, p, "--json", *extra],
                                      check=True, capture_output=True, text=True).stdout)
        if out is None:
            out = {k: v for k, v in d.items() if k not in ("policies", "known_divergences")}
            out["policies"], out["known_divergences"] = {}, []
        for k in ("schema", "task_id", "speed", "lower_bound_makespan_s", "horizon_s", "host_terms", "gpu_mock_daemon_sha256"):
            if d[k] != out[k]:
                raise SystemExit(f"{p}: {k} differs from the first policy's")
        out["policies"][p] = d["policies"][p]
        out["known_divergences"] += d["known_divergences"]
    out["policies_not_run"] = {"packed_overlap": (
        "refused by build/expected_scores.py run_overlap: it needs every job at one width that divides the box, and this task's "
        "jobs mix widths. Each other policy was run alone and merged here (build/merge_expected.py)")}
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
