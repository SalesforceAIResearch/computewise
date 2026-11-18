#!/usr/bin/env bash
# The box reports the accelerator build it emulates, not the CPU wheel beneath
# (G-goalq-astra-L0-v3 identified the mock via torch.__version__ +cpu,
# torch._C._has_cuda False, and torch.cuda.set_device.__module__
# nvcompat.torch_patch). After the patch: +cu<ver>, _has_cuda True, and no
# torch.cuda callable reporting a patch/vendored module.
# Usage (on the VM):  sudo IMG=sha256:<box id> [PATCH_SRC=1] bash tests/test_identity_surface_in_image.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:-ml-node:py311-torch2110}"
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
M=(); [ "${PATCH_SRC:-0}" = 1 ] && M=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 2 --memory 4g "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh
python - <<PY 2>&1 | grep -v Warning
import torch, types, sys
fails=0; n=0
def tag(cond): return "[ok]" if cond else "[FAIL]"
def chk(name,cond,detail=""):
    global fails,n; n+=1; fails+=0 if cond else 1
    # no backslash inside an f-string expression: Python 3.11 rejects it, and this
    # heredoc sits inside a single-quoted bash -c body (the 32c7 gate printed the
    # source line instead of a result, 2026-10-23)
    print(tag(cond)+" "+name+(": "+str(detail) if detail else ""))
v=str(torch.__version__)
chk("torch.__version__ is not a +cpu wheel", not v.endswith("+cpu"), v)
chk("torch.__version__ carries the emulated CUDA build tag", "+cu" in v, v)
chk("torch._C._has_cuda is True", getattr(torch._C,"_has_cuda",None) is True)
chk("torch.cuda.is_available() is True and count is 4", torch.cuda.is_available() and torch.cuda.device_count()==4)
sm=torch.cuda.set_device.__module__
chk("torch.cuda.set_device.__module__ reads torch, not a patch/vendored module", "torch" in sm and "nvcompat" not in sm and "cuda_hooks" not in sm and "cuda_runtime" not in sm, sm)
leak=[(x,getattr(getattr(torch.cuda,x),"__module__","")) for x in dir(torch.cuda) if isinstance(getattr(torch.cuda,x,None),types.FunctionType)]
bad=[x for x in leak if any(k in x[1] for k in ("cuda_hooks","cuda_runtime","nvcompat","fakegpu"))]
chk("no torch.cuda callable reports a patch or vendored module", not bad, str(bad[:5]))
chk("torch.version.cuda is set", getattr(torch.version,"cuda",None) is not None, str(getattr(torch.version,"cuda",None)))
print(f"{n-fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
PY'
