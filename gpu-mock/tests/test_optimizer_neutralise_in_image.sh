#!/usr/bin/env bash
# The replay optimizer stub under an LR scheduler, inside an image with gpu-mock
# installed: a scheduler can wrap it (torch reads step.__func__), a replayed step
# changes no weight and counts one replay step, and after replay is withdrawn the
# optimizer's own step runs again with no scheduler warning.
# Usage (on the VM):  sudo IMG=gm-dpo:x86 [PATCH_SRC=1] bash tests/test_optimizer_neutralise_in_image.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:-gm-dpo:x86}"
M=(); [ "${PATCH_SRC:-0}" = 1 ] && M=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 2 --memory 4g "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh; unset LD_PRELOAD MOCK_PCI_ROOT; export NVRT_NO_REPLAY=1; cd /tmp
python - <<PY 2>&1 | grep -v "datetime.datetime size"
import warnings, torch, replay, fingerprint_runtime as fr
replay._s["serving"] = {"steady_step_s": 0.01}
p = torch.nn.Parameter(torch.ones(3, device="cuda"))
opt = torch.optim.AdamW([p], lr=0.1)
replay._neutralise(opt)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
ok = []
replay._s["active"] = True
p.grad = torch.ones(3, device="cuda")
fr._state["backward_epoch"] = fr._state.get("backward_epoch", 0) + 1
r = opt.step(); sched.step()
ok.append(("scheduler wraps the stub; replayed step returns None", r is None))
ok.append(("replayed step leaves weights unchanged", bool((p.detach().cpu() == 1).all())))
ok.append(("replayed step counted once", replay._s["step"] == 1))
replay._s["active"] = False
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    opt.step(); sched.step()
ok.append(("after withdrawal the real step updates weights", bool((p.detach().cpu() != 1).all())))
ok.append(("no scheduler/optimizer-order warning", not [x for x in w if "step" in str(x.message)]))
for name, v in ok:
    print(("[ok  ] " if v else "[FAIL] ") + name)
print(f"{sum(v for _, v in ok)} passed, {sum(not v for _, v in ok)} failed")
PY'
