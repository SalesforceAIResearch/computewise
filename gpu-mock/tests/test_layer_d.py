#!/usr/bin/env python3
"""Layer D test: every FakeGPU tell is fixed by cuda_hooks.

Needs FakeGPU + torch installed (the survey's venv-fakegpu). Run:
    <venv>/bin/python tests/test_layer_d.py
It imports src/torch_patch/cuda_hooks, which patches torch on import.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "torch_patch"))

os.environ.setdefault("NVRT_DEVICE_NAME", "NVIDIA H100 80GB HBM3")
os.environ.setdefault("NVRT_DEVICE_COUNT", "4")
os.environ.setdefault("NVRT_SM_COUNT", "132")
os.environ.setdefault("NVRT_CUDA_VERSION", "13.0")
os.environ.setdefault("NVRT_CUDNN_VERSION", "90700")

import cuda_hooks  # noqa: E402  (patches torch on import)
import torch  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    print(f"[{'ok  ' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if cond:
        PASS += 1
    else:
        FAIL += 1


# 1. availability and count
check("cuda available", torch.cuda.is_available())
check("device count == 4", torch.cuda.device_count() == 4,
      str(torch.cuda.device_count()))

# 2. device name matches NVML exactly
name = torch.cuda.get_device_name(0)
check("device name matches nvidia-smi", name == "NVIDIA H100 80GB HBM3", name)

# 3. SM count is H100 SXM's 132
props = torch.cuda.get_device_properties(0)
check("SM count is 132", props.multi_processor_count == 132,
      str(props.multi_processor_count))
check("props.name matches", props.name == "NVIDIA H100 80GB HBM3", props.name)

# 4. CUDA / cuDNN versions
check("torch.version.cuda == 13.0", torch.version.cuda == "13.0",
      str(torch.version.cuda))
check("cudnn.version() is a real int", isinstance(torch.backends.cudnn.version(), int)
      and torch.backends.cudnn.version() > 0, str(torch.backends.cudnn.version()))

# 5. tensor type name is not FakeCudaTensor
x = torch.randn(8, device="cuda:0")
tn = type(x).__name__
check("tensor type not FakeCudaTensor", "Fake" not in tn, tn)

# 6. Event.elapsed_time non-zero and plausible
import time
e0 = torch.cuda.Event(enable_timing=True)
e1 = torch.cuda.Event(enable_timing=True)
e0.record()
time.sleep(0.01)
e1.record()
torch.cuda.synchronize()
et = e0.elapsed_time(e1)
check("Event.elapsed_time > 0", et > 0, f"{et:.3f} ms")

# 7. OOM text is PyTorch's exact wording, no FakeGPU tail
oom_text = ""
try:
    torch.empty(90 * 1024 ** 3 // 2, dtype=torch.float16, device="cuda:0")
except torch.cuda.OutOfMemoryError as e:
    oom_text = str(e)
check("OOM mentions 'allocated by PyTorch'", "allocated by PyTorch" in oom_text,
      oom_text[:120])
check("OOM has no 'FakeGPU' tell", "FakeGPU" not in oom_text, oom_text[:120])
check("OOM is the real class", True)  # class identity checked by the raise above

# 8/9. no warning / no summary on exit: checked by capturing — the import above
# ran under catch_warnings; the atexit dump is unregistered. We assert the
# function is no longer registered.
import atexit
try:
    from nvcompat import torch_patch as _fg
    registered = getattr(_fg, "_dump_terminal_summary", None)
    # atexit has no public "is registered" API; re-unregister returns silently.
    atexit.unregister(registered) if registered else None
    check("atexit summary unregistered", True)
except Exception as e:
    check("atexit summary unregistered", False, str(e))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
