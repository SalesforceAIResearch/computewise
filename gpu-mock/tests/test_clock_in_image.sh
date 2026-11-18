#!/usr/bin/env bash
# time.sleep and waits under the patch, inside an image with gpu-mock installed,
# in both clock modes (modelled, and replay's recorded mode): sleep(1) takes 1 s
# of wall time and moves the job's clock by at least 1 s, and a 30 s wait loop
# (sleep + time.monotonic, as wandb waits for its service) sees a 2 s process the
# patch does not reach finish, rather than timing out. In recorded mode, a
# real-time window (begin/end_real_window) makes the job's clock follow real time.
# Usage (on the VM):  sudo IMG=gm-battery:x86 [PATCH_SRC=1] bash tests/test_clock_in_image.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:-gm-battery:x86}"
M=(); [ "${PATCH_SRC:-0}" = 1 ] && M=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 2 --memory 4g "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh; unset LD_PRELOAD MOCK_PCI_ROOT; export NVRT_NO_REPLAY=1; cd /tmp
python - <<PY 2>&1 | grep -v "datetime.datetime size"
import os, subprocess, sys, time, torch, virtual_clock as vc
R = lambda: time.clock_gettime(time.CLOCK_MONOTONIC)       # not patched
clk = vc.get_clock(); res = []
res.append(("the virtual clock is installed", vc._time_patched, ""))
for mode in ("modelled", "replay"):
    if mode == "replay":
        clk.set_recorded_mode(True)
    r0, m0 = R(), time.monotonic(); time.sleep(1.0); r, m = R() - r0, time.monotonic() - m0
    res.append((f"{mode}: sleep(1) takes 1 s of wall time", 0.95 <= r <= 1.5, f"{r:.3f} s"))
    res.append((f"{mode}: the job clock advances at least 1 s", m >= 0.999, f"{m:.3f} s"))
    flag = f"/tmp/ready-{mode}"
    p = subprocess.Popen([sys.executable, "-I", "-c", f"import time; time.sleep(2); open({flag!r}, \"w\").close()"])
    r0, deadline, seen = R(), time.monotonic() + 30, False
    while time.monotonic() < deadline:
        if os.path.exists(flag):
            seen = True
            break
        time.sleep(0.2)
    p.wait()
    res.append((f"{mode}: a 30 s wait sees a 2 s external process finish", seen, f"after {R() - r0:.1f} s"))
# a real-time window in recorded mode (an evaluation the record holds no time for)
clk.set_recorded_mode(True)
clk.advance_recorded(100.0)
a = time.monotonic(); clk.begin_real_window(); vc._REAL["sleep"](0.3); mid = time.monotonic()
clk.end_real_window(); b = time.monotonic(); clk.advance_recorded(5.0); c = time.monotonic()
res.append(("replay: the job clock follows real time inside a window", 0.25 < mid - a < 0.6, f"{mid - a:.3f} s"))
res.append(("replay: closing the window keeps the time", 0 <= b - mid < 0.05, f"{b - mid:.3f} s"))
res.append(("replay: a step after the window adds exactly its time", abs(c - b - 5.0) < 1e-6, f"{c - b:.6f} s"))
for name, ok, d in res:
    print(("[ok  ] " if ok else "[FAIL] ") + name + (f"  -- {d}" if d else ""))
print(f"{sum(bool(o) for _, o, _ in res)} passed, {sum(not o for _, o, _ in res)} failed")
PY'
