"""A stand-in GPU job for testing the runner's tooling without the box, torch or a record.

usage: python3 -I standin_job.py --output_dir=out-<job id> --gpu-s 30 --host-s 2 [--steps 30]

It talks to gpu-mock's real daemon with gpu-mock's own client (experiments/gpu-mock/src/
torch_patch/daemon_client.py), so the occupancy log it produces is the daemon's, not a
hand-written one. It says hello, spends --host-s on the CPU (the job's host phase: start-up and
loading), then paces --gpu-s seconds of GPU work in --steps units on each GPU it was given, and
says bye. The GPUs are CUDA_VISIBLE_DEVICES; with none set it uses GPU 0. Each step blocks until
the previous step's GPU work drains, as gpu-mock's replay pacing does (one step of lookahead).

This is a test fixture for the scorer and the timeline, not a model of a real job: a real job's
start-up is real library work in the box, and only the box measures that.
"""
import argparse, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "gpu-mock", "src", "torch_patch")))
from daemon_client import DaemonClient

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gpu-s", type=float, required=True)
    ap.add_argument("--host-s", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpus = [int(x) for x in cvd.split(",") if x.strip()]
    c = DaemonClient()
    c.hello(argv=[os.path.basename(sys.argv[0])] + sys.argv[1:])
    end = time.monotonic() + a.host_s
    x = 0
    while time.monotonic() < end:                     # real CPU work, as a job's start-up is
        x = (x * 1103515245 + 12345) % 2**31
    per = a.gpu_s / a.steps
    for _ in range(a.steps):
        # pace as gpu-mock's replay does (replay.py _pace): enqueue this step, then block only
        # until the PREVIOUS work drains (cursor - this unit's wall cost), so the GPU queue never
        # empties between steps; drain fully at the end
        wait = 0.0
        for g in gpus:
            r = c.enqueue(gpu=g, op="replay", modelled_s=per)
            wait = max(wait, (r["cursor"] - float(r.get("wall_cost") or 0.0)) - time.monotonic())
        if wait > 0:
            time.sleep(wait)
    for g in gpus:
        c.sync(g)
    os.makedirs(a.output_dir, exist_ok=True)
    open(os.path.join(a.output_dir, "done"), "w").write("ok\n")
    c.bye()

if __name__ == "__main__":
    main()
