"""Unit check of replay._step_time: step 1 carries the record's startup.

Run inside a box image (torch present), with this tree's patch on the path:
  python /gm/tests/test_step_time.py
"""
import sys
sys.path.insert(0, "/usr/local/nvidia/python")
import replay  # noqa: E402

fails = 0
def chk(name, got, want):
    global fails
    ok = abs(got - want) < 1e-6
    fails += 0 if ok else 1
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {got} (want {want})")

# mng3-style record (641 in the registry): no first_step_s, startup inside the loop
replay._s["serving"] = {"steady_step_s": 1.19582, "startup_in_train_s": 184.258}
chk("mng3 step 1 = steady + startup", replay._step_time(1), 185.45382)
chk("mng3 step 2 = steady", replay._step_time(2), 1.19582)
chk("mng3 step 3250 = steady", replay._step_time(3250), 1.19582)
# track-3 record: first_step_s already includes the startup
replay._s["serving"] = {"steady_step_s": 0.15, "first_step_s": 5.4, "startup_in_train_s": 5.25}
chk("track-3 step 1 = first_step_s", replay._step_time(1), 5.4)
chk("track-3 step 2 = steady", replay._step_time(2), 0.15)
# no startup at all
replay._s["serving"] = {"steady_step_s": 0.2}
chk("no startup: step 1 = steady", replay._step_time(1), 0.2)
# per_step_gpu_s fallback
replay._s["serving"] = {"per_step_gpu_s": 0.3, "startup_in_train_s": 10.0}
chk("per_step_gpu_s fallback + startup", replay._step_time(1), 10.3)
print(f"{7 - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
