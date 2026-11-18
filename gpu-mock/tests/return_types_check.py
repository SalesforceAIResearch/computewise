#!/usr/bin/env python3
"""The checks of tests/test_return_types_in_image.sh, run inside the box."""
import os, subprocess, sys, traceback
import torch
fails = 0; n = 0
def chk(name, cond, detail=""):
    global fails, n
    n += 1; fails += 0 if cond else 1
    mark = "ok" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f": {detail}" if detail else ""))
def ty(x): return type(x).__module__ + "." + type(x).__name__
cpu = torch.arange(6.0).reshape(2, 3); cuda = cpu.to("cuda")
eye_c, eye_g = torch.eye(2), torch.eye(2, device="cuda")
ops = {
    "topk": lambda t: t.topk(2), "sort": lambda t: t.sort(), "max(dim)": lambda t: t.max(dim=1), "min(dim)": lambda t: torch.min(t, 1),
    "median(dim)": lambda t: t.median(dim=1), "kthvalue": lambda t: t.kthvalue(1), "mode": lambda t: t.mode(), "cummax": lambda t: t.cummax(1),
    "cummin": lambda t: t.cummin(1), "aminmax": lambda t: torch.aminmax(t), "frexp": lambda t: torch.frexp(t),
    "linalg.svd": lambda t: torch.linalg.svd(t), "linalg.qr": lambda t: torch.linalg.qr(t), "unique(return_counts)": lambda t: torch.unique(t, return_counts=True),
    "nn.functional.max_pool1d(return_indices)": lambda t: torch.nn.functional.max_pool1d(t.unsqueeze(0), 2, return_indices=True),
}
for name, f in ops.items():
    a, b = f(cpu), f(cuda)
    chk(f"{name}: same result type on cuda as on cpu ({ty(a)})", ty(a) == ty(b), f"cuda gave {ty(b)}")
for name, f in {"linalg.slogdet": lambda e: torch.linalg.slogdet(e), "linalg.eigh": lambda e: torch.linalg.eigh(e), "lu_factor": lambda e: torch.linalg.lu_factor(e)}.items():
    a, b = f(eye_c), f(eye_g)
    chk(f"{name}: same result type on cuda as on cpu ({ty(a)})", ty(a) == ty(b), f"cuda gave {ty(b)}")
r = cuda.topk(2)
try:
    ok_attr = torch.equal(r.values.cpu(), cpu.topk(2).values) and torch.equal(r.indices.cpu(), cpu.topk(2).indices); dev = str(r.values.device)
except AttributeError as e:
    ok_attr = False; dev = repr(e)
chk("topk on cuda: .values and .indices work and equal the cpu result", ok_attr, dev if not ok_attr else "")
v, i = cuda.max(dim=1)
chk("tuple unpacking of a structseq still works", torch.equal(v.cpu(), cpu.max(dim=1).values) and torch.equal(i.cpu(), cpu.max(dim=1).indices))
chk("the structseq elements are cuda tensors (wrapped)", ok_attr and dev.startswith("cuda"), dev)
tb = ""
try:
    torch.ones(3, device="cuda").topk(5)
except Exception:
    tb = traceback.format_exc()
hidden = ("/usr/local/nvidia/python" not in tb) and ("nvcompat" not in tb)
chk("traceback.format_exc() through a fake-cuda op shows no patch or vendored frame", hidden, tb[-400:] if not hidden else "")
chk("...but still shows the error and torch frames", "selected index k out of range" in tb and "site-packages/torch" in tb)
cmd = [sys.executable, "-c", 'import torch; torch.ones(3, device="cuda").topk(5)']
p = subprocess.run(cmd, capture_output=True, text=True)
chk("an uncaught exception (default excepthook) shows no patch or vendored frame", p.returncode != 0 and "/usr/local/nvidia/python" not in p.stderr and "nvcompat" not in p.stderr and "out of range" in p.stderr, p.stderr[-300:])
p2 = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "NVRT_DEBUG": "1"})
chk("with NVRT_DEBUG=1 the frames are visible again", "/usr/local/nvidia/python" in p2.stderr, p2.stderr[-200:])
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
