"""Score gpu-tasks' emitted synthetic runs with the runner's idle.py and compare with its expected values.

usage, on the VM: python3 -I compare_gpu_tasks.py <emit_root> <gpu-tasks/registry/expected> <runner dir>
where <emit_root> holds <task>/<policy>/ run directories from gpu-tasks/build/expected_scores.py --emit."""
import json, os, subprocess, sys
root, expdir, runner = sys.argv[1:4]
worst = {}
for task in sorted(os.listdir(root)):
    exp = json.load(open(os.path.join(expdir, task + ".json")))
    for pol in sorted(os.listdir(os.path.join(root, task))):
        d = os.path.join(root, task, pol)
        p = subprocess.run([sys.executable, "-I", os.path.join(runner, "score", "idle.py"), d, "--json"], capture_output=True, text=True)
        if p.returncode:
            print(f"{task} {pol}: idle.py FAILED {p.stderr[-300:]}"); continue
        got = json.loads(p.stdout); want = exp["policies"][pol]["idle"]
        diffs = []
        for k in ("whole_box_idle_share_of_wall", "partial_idle_share_of_gpu_seconds", "time_to_first_launch_recorded_s",
                  "idle_after_agent_stopped_recorded_s"):
            a, b = got.get(k), want.get(k)
            if a is None or b is None:
                if a != b: diffs.append(f"{k} {a}/{b}")
                continue
            if abs(a - b) > 1e-3: diffs.append(f"{k} {a:.4f}/{b:.4f}")
            worst[k] = max(worst.get(k, 0), abs(a - b))
        gw, ww = got["window"]["wall_s"], (want["window"].get("wall_s") if isinstance(want["window"], dict) else want["window"])
        if abs(gw - ww) > 1e-3: diffs.append(f"window {gw:.3f}/{ww:.3f}")
        gr = got["work_remaining_at_window_end"]["remaining"]
        wr = want["work_remaining_at_window_end"]["remaining"] if isinstance(want["work_remaining_at_window_end"], dict) else want["work_remaining_at_window_end"]
        if gr != wr: diffs.append(f"remaining {gr}/{wr}")
        gc, wc = got["whole_box_idle_by_cause_recorded_s"], want["whole_box_idle_by_cause_recorded_s"]
        for c in sorted(set(gc) | set(wc)):
            if abs(gc.get(c, 0) - wc.get(c, 0)) > 0.01: diffs.append(f"cause {c} {gc.get(c,0):.3f}/{wc.get(c,0):.3f}")
        print(f"{'ok  ' if not diffs else 'DIFF'} {task[:12]:<12} {pol:<26} " + ("; ".join(diffs) if diffs else ""))
print("largest abs diff per field:", {k: round(v, 6) for k, v in worst.items()})
