"""C1 check: compare a scripted run's scores with the scores predicted for it in advance.

usage: python3 -I check.py <run_dir> [--host-s 2 | --host-s-file calib.json] [--json]

Expected values come from expect.py, which simulates the run's plan.json over the run's task.json
and never touches the scorer. Scored values come from score/idle.py and score/score_run.py, which
read the run directory. A pass means two independent paths agree.

Two tolerance classes (design: the C1 note, 2026-10-20):
  exact      work remaining, think time: must match exactly
  predicted  window, idle shares, first launch, idle after the agent's end: within the tolerance,
             which is the scheduling jitter of one launch (default 0.25 s per launch, scaled by the
             number of launches) plus the given host-time spread
Exits 0 on pass, 1 on any failure, and prints every comparison either way.
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.dirname(HERE)

def run_json(cmd):
    p = subprocess.run([sys.executable, "-I", *cmd, "--json"], capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed:\n{p.stderr[-2000:]}")
    return json.loads(p.stdout)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--host-s", type=float, default=None)
    ap.add_argument("--host-s-file", default=None)
    ap.add_argument("--jitter-s", type=float, default=0.25, help="per-launch scheduling jitter, wall seconds")
    ap.add_argument("--host-spread-s", type=float, default=0.0, help="per-launch host-time spread, wall seconds")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    d = a.run_dir
    sys.path.insert(0, os.path.join(RUNNER, "score"))
    import idle
    speed = float(idle.run_inputs(d).get("speed") or 1.0)
    cmd = [os.path.join(HERE, "expect.py"), os.path.join(d, "task.json"), os.path.join(d, "plan.json"), "--speed", str(speed)]
    if a.host_s_file:
        cmd += ["--host-s-file", a.host_s_file]
    elif a.host_s is not None:
        cmd += ["--host-s", str(a.host_s)]
    exp = run_json(cmd)
    got = run_json([os.path.join(RUNNER, "score", "idle.py"), d])
    pe, ee = exp["predicted"], exp["exact"]
    launches = sum(1 for s in json.load(open(os.path.join(d, "plan.json")))["steps"] if s["op"] in ("run", "launch", "launch_at")) + \
        sum(len(s["jobs"]) for s in json.load(open(os.path.join(d, "plan.json")))["steps"] if s["op"] == "pool")
    tol_s = (a.jitter_s + a.host_spread_s) * max(1, launches)
    win = got["window"]["wall_s"] or 1.0
    rows = [
        ("work remaining", "exact", ee["work_remaining_at_deadline"], got["work_remaining_at_window_end"]["remaining"], 0),
        ("window, wall s", "predicted", pe["window_wall_s"], got["window"]["wall_s"], tol_s),
        ("whole-box idle share", "predicted", pe["whole_box_idle_share_of_wall"], got["whole_box_idle_share_of_wall"], tol_s / win),
        ("partial idle share", "predicted", pe["partial_idle_share_of_gpu_seconds"], got["partial_idle_share_of_gpu_seconds"], tol_s / win),
        ("first launch, s", "predicted", pe["time_to_first_launch_wall_s"], (got["time_to_first_launch_recorded_s"] or 0) / speed
         if got["time_to_first_launch_recorded_s"] is not None else None, tol_s),
        ("whole-box idle after agent end, s", "predicted", pe["whole_box_idle_after_agent_end_wall_s"],
         (got["whole_box_idle_after_agent_end_recorded_s"] or 0) / speed, tol_s),
    ]
    out, fails = [], 0
    for name, kind, e, g, tol in rows:
        ok = (e is None and g is None) or (e is not None and g is not None and abs(g - e) <= tol + 1e-9)
        fails += not ok
        out.append({"check": name, "class": kind, "expected": e, "scored": g, "tolerance": tol, "ok": ok})
    res = {"run_dir": d, "pass": fails == 0, "launches": launches, "tolerance_s": tol_s, "checks": out}
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print(f"C1 check {os.path.basename(os.path.abspath(d))}: {'PASS' if not fails else f'FAIL ({fails})'}   "
              f"tolerance {tol_s:.2f} s over {launches} launches")
        for r in out:
            e = "none" if r["expected"] is None else f"{r['expected']:.4f}"
            g = "none" if r["scored"] is None else f"{r['scored']:.4f}"
            print(f"  {'ok  ' if r['ok'] else 'FAIL'} {r['check']:<36} expected {e:>10}  scored {g:>10}  (tol {r['tolerance']:.4f}, {r['class']})")
    sys.exit(0 if not fails else 1)

if __name__ == "__main__":
    main()
