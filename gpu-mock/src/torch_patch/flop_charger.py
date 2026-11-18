"""Automatic per-op timing: charge the virtual clock for every tensor op.

An agent will not call FlopCounterMode for us — the timing illusion has to be
transparent. This installs a persistent TorchDispatchMode that runs under every
op, computes the op's FLOPs (via torch's own flop_registry for matmul/conv/
attention, which dominate training time) and its memory traffic (input + output
bytes, for the bandwidth-bound ops that otherwise count as zero FLOPs), and
charges the virtual clock the modelled GPU time for whichever bound is larger.

So a bare `x @ x` advances the job's perf_counter by the H100 time for that
matmul, with no cooperation from the agent's code.

Composition with FakeGPU: FakeGPU intercepts at __torch_function__ and runs ops
on CPU. TorchDispatchMode runs *below* that, at the aten level, so it sees the
real decomposed ops with real shapes and dtypes. We charge there and let the op
proceed unchanged.

Cost control: the mode must be cheap, because it runs on every op. FLOP lookup
is a dict hit; the byte estimate is a sum over tensor arguments. No tensor is
materialised or copied.
"""

from __future__ import annotations

import threading

import torch
from torch.utils._python_dispatch import TorchDispatchMode
import logging as _logging
# torch.utils.flop_counter warns at import, "triton not found; flop counting
# will not work for triton kernels", whenever torch.version.cuda is set and
# triton is absent. The box reports the emulated cu130 build (identity surface,
# 2026-10-23), so from 32c7 every process printed that line to stderr at
# `import torch`; a real cu130 install ships triton and prints nothing. Quiet
# for this import only; the logger's level is restored after.
_fc_log = _logging.getLogger("torch.utils.flop_counter")
_fc_level = _fc_log.level
_fc_log.setLevel(_logging.ERROR)
try:
    try:
        from torch.utils.flop_counter import flop_registry   # torch >= 2.2: formulas take tensors
        _FORMULAS_TAKE_SHAPES = False
    except ImportError:
        # torch 2.1 (the TRL battery's pin) has flop_mapping, whose formulas take
        # shapes. The import of flop_registry failed there, which silently took the
        # whole virtual clock down with it (2026-10-20).
        from torch.utils.flop_counter import flop_mapping as flop_registry
        from torch.utils.flop_counter import get_shape as _get_shape
        _FORMULAS_TAKE_SHAPES = True
finally:
    _fc_log.setLevel(_fc_level)

import virtual_clock as vc

# FakeGPU unwraps cuda tensors to CPU tensors BEFORE __torch_dispatch__ runs, so
# at dispatch time we cannot tell a fake-cuda op from a genuine CPU op the agent
# ran on purpose. Charging a genuine CPU op at GPU rates is a measurement-
# validity defect (reviewer finding #11). We fix it with a thread-local flag:
# FakeCudaTensor.__torch_function__ sets "we are inside a cuda op" for the
# duration of the unwrapped call, and the dispatch mode charges only when the
# flag is set. A genuine CPU op never enters FakeCudaTensor.__torch_function__,
# so it is never charged. The flag is thread-local so a CPU worker thread is
# unaffected.
_cuda_region = threading.local()


def _in_cuda_region() -> bool:
    return getattr(_cuda_region, "depth", 0) > 0


def _current_device() -> int:
    return getattr(_cuda_region, "device", 0)


def _enter_cuda_region(device: int = 0) -> None:
    _cuda_region.depth = getattr(_cuda_region, "depth", 0) + 1
    _cuda_region.device = device


def _exit_cuda_region() -> None:
    _cuda_region.depth = max(0, getattr(_cuda_region, "depth", 0) - 1)


def _infer_device(args, kwargs) -> int:
    """Read the FakeCudaTensor device_index from the ORIGINAL (pre-unwrap) args at
    the __torch_function__ layer, where device info is still present."""
    stack = list(args) + list((kwargs or {}).values())
    while stack:
        o = stack.pop()
        idx = getattr(o, "device_index", None)
        if idx is not None:
            return int(idx)
        if isinstance(o, (list, tuple)):
            stack.extend(o)
    return 0


def install_cuda_region_marker() -> None:
    """Wrap FakeCudaTensor.__torch_function__ so it marks the thread as being
    inside a cuda op while the unwrapped CPU call runs."""
    try:
        from nvcompat import _upstream as up
    except Exception:
        return
    cls = getattr(up, "FakeCudaTensor", None)
    if cls is None or getattr(cls, "_nvrt_region_wrapped", False):
        return
    orig = cls.__torch_function__.__func__

    def wrapped(klass, func, types, args=(), kwargs=None):
        if vc.REPLAY_ACTIVE:
            return orig(klass, func, types, args, kwargs)
        # Ensure the FLOP-charging dispatch mode is active IN THIS THREAD. The
        # mode is thread-local, so a DataLoader worker (or any new thread) would
        # otherwise run cuda ops at CPU speed. We enter a per-thread mode instance
        # lazily on the thread's first cuda op; all instances charge the one
        # process-global clock, so timing stays coherent across threads.
        _ensure_thread_mode()
        _enter_cuda_region(_infer_device(args, kwargs))
        try:
            return orig(klass, func, types, args, kwargs)
        finally:
            _exit_cuda_region()

    cls.__torch_function__ = classmethod(wrapped)
    cls._nvrt_region_wrapped = True


_thread_mode = threading.local()


def _ensure_thread_mode() -> None:
    if getattr(_thread_mode, "entered", False):
        return
    if not _installed:          # charger not installed; nothing to enter
        return
    m = FlopChargingMode()
    m.__enter__()
    _thread_mode.entered = True
    _thread_mode.mode = m       # keep a ref so it is not GC'd

# Map an op packet to the registry key the flop formulas use.
_aten = torch.ops.aten

# Metadata-only ops: return an alias/view with no data movement. A real GPU does
# no HBM traffic for these, so they cost ~0 (reviewer #14).
_METADATA_OPS = frozenset({
    "view", "_unsafe_view", "reshape", "reshape_alias", "transpose", "t",
    "permute", "squeeze", "unsqueeze", "expand", "expand_as", "as_strided",
    "detach", "alias", "slice", "select", "narrow", "flatten", "ravel",
    "movedim", "swapaxes", "swapdims", "broadcast_to", "contiguous",
    "view_as", "diagonal", "unfold",
})


def _op_base_name(func) -> str:
    packet = getattr(func, "_overloadpacket", None)
    name = str(packet if packet is not None else func)
    # "aten.svd" -> "svd"; "aten.linalg_svd.default" -> "linalg_svd"
    name = name.split(".")[-1] if "." in name else name
    return name


def _storage_ptr(t):
    try:
        return t.untyped_storage().data_ptr()
    except Exception:
        try:
            return t.storage().data_ptr()
        except Exception:
            return None


def _is_alias(args, result) -> bool:
    """True if result shares storage with an input tensor (a view): no transfer."""
    if not isinstance(result, torch.Tensor):
        return False
    rp = _storage_ptr(result)
    if rp is None:
        return False
    for a in args:
        if isinstance(a, torch.Tensor) and _storage_ptr(a) == rp:
            return True
    return False


def _extra_complexity_flops(opname: str, args, result) -> float:
    """Algorithmic FLOP estimates for compute-heavy ops torch does not register.
    Coefficients are order-of-magnitude (the point is to not undercount by 1000x,
    not textbook exactness). n is the larger matrix dimension where relevant."""
    def _mat_dims():
        for a in args:
            if isinstance(a, torch.Tensor) and a.dim() >= 2:
                return a.shape[-2], a.shape[-1], a.numel() // (a.shape[-1] * a.shape[-2] or 1)
        return 0, 0, 1

    m, n, batch = _mat_dims()
    big = max(m, n)
    small = min(m, n) if m and n else 0
    # dense linear-algebra decompositions: O(small^2 * big) with a constant.
    if opname in ("linalg_svd", "svd", "_linalg_svd"):
        return 6.0 * small * small * big * batch
    if opname in ("linalg_eig", "linalg_eigh", "eig", "symeig", "_linalg_eigh"):
        return 9.0 * big ** 3 * batch
    if opname in ("linalg_qr", "qr", "geqrf"):
        return 2.0 * m * n * n * batch
    if opname in ("linalg_cholesky", "cholesky", "_cholesky"):
        return (big ** 3 / 3.0) * batch
    if opname in ("linalg_solve", "solve", "linalg_lu", "lu", "_lu_with_info",
                  "triangular_solve", "linalg_inv", "inverse"):
        return (2.0 / 3.0) * big ** 3 * batch
    if opname in ("linalg_lstsq", "lstsq"):
        return 2.0 * m * n * n * batch
    # sort / topk / unique: O(N log N) comparisons.
    if opname in ("sort", "argsort", "topk", "msort", "unique", "unique2",
                  "_unique2", "kthvalue", "median"):
        import math
        N = result.numel() if isinstance(result, torch.Tensor) else (
            args[0].numel() if args and isinstance(args[0], torch.Tensor) else 0)
        return float(N) * max(1.0, math.log2(N + 1)) * 2.0
    # FFTs: O(N log N) complex ops.
    if "fft" in opname:
        import math
        N = result.numel() if isinstance(result, torch.Tensor) else 0
        return 5.0 * float(N) * max(1.0, math.log2(N + 1))
    return 0.0


def _bytes_of(x) -> int:
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    return 0


def _tree_bytes(args, kwargs) -> int:
    total = 0
    stack = list(args) + list((kwargs or {}).values())
    while stack:
        o = stack.pop()
        if isinstance(o, torch.Tensor):
            total += _bytes_of(o)
        elif isinstance(o, (list, tuple)):
            stack.extend(o)
    return total


def _result_bytes(result) -> int:
    if isinstance(result, torch.Tensor):
        return _bytes_of(result)
    if isinstance(result, (list, tuple)):
        return sum(_bytes_of(o) for o in result if isinstance(o, torch.Tensor))
    return 0


def _dtype_of(args, result) -> torch.dtype:
    # Prefer the result dtype; fall back to the first tensor arg.
    if isinstance(result, torch.Tensor):
        return result.dtype
    for a in args:
        if isinstance(a, torch.Tensor):
            return a.dtype
    return torch.float32


class FlopChargingMode(TorchDispatchMode):
    """Charges the virtual clock the modelled GPU time of each op."""

    def __init__(self, clock_charge=vc.charge_op):
        super().__init__()
        self._charge = clock_charge
        self.total_flops = 0
        self.total_charged_s = 0.0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if vc.REPLAY_ACTIVE:
            # replay: time comes from the record, not from ops
            return func(*args, **kwargs)
        # Only charge ops that touch a (fake) cuda tensor. A genuine CPU-tensor op
        # the agent runs on purpose must stay at real CPU speed and must NOT be
        # re-timed as GPU work — otherwise the mock would reward an agent for
        # "accelerating" CPU work it deliberately kept off the GPU, poisoning the
        # paper's own measurement (reviewer finding #11). FakeGPU unwraps cuda
        # tensors to CPU before dispatch, so we check the ORIGINAL args' device.
        if not _in_cuda_region():
            # a genuine CPU op (or an op outside any fake-cuda call): run it, do
            # not charge GPU time. CPU work stays at CPU speed.
            return func(*args, **kwargs)
        # Time the real op so the virtual clock can replace that real cost with
        # the modelled GPU cost (Phantora accounting). Use the genuine perf
        # counter, not the patched one.
        _t0 = vc.real_perf_counter()
        result = func(*args, **kwargs)
        real_s = vc.real_perf_counter() - _t0

        opname = _op_base_name(func)

        # Metadata-only ops (view/reshape/transpose/as_strided/...) move no data
        # on a real GPU — they return an alias. Charging their logical bytes as an
        # HBM transfer invented a ~0.2 ms transfer per view (reviewer #14). Charge
        # nothing. Detect by name only: a storage-pointer alias check is too
        # expensive to run on every op in the hot path (it dispatches more ops).
        if opname in _METADATA_OPS:
            # instant on a real GPU, and its CPU cost is framework overhead the
            # GPU queue hides. Charge a zero-duration op on the current device so
            # the clock skips the op's own real time (resets _last_perf) instead
            # of letting it leak into the next op's gap and inflate the step.
            self._charge(0.0, 0.0, real_s, _dtype_of(args, result),
                         op_start_perf=_t0, device=_current_device())
            return result

        # FLOPs for the compute-bound ops torch knows how to count.
        flops = 0
        packet = getattr(func, "_overloadpacket", None)
        formula = flop_registry.get(packet) if packet is not None else None
        if formula is not None and _FORMULAS_TAKE_SHAPES:
            try:
                from torch.utils._pytree import tree_map
                a_s, k_s, out_shape = tree_map(_get_shape, (args, kwargs or {}, result))
                flops = int(formula(*a_s, **k_s, out_shape=out_shape))
            except Exception:
                flops = 0
        elif formula is not None:
            try:
                flops = formula(*args, out=result, **kwargs)
            except TypeError:
                try:
                    flops = formula(*args, **kwargs)
                except Exception:
                    flops = 0
            except Exception:
                flops = 0

        # Compute-heavy ops torch does NOT register (SVD, eig, qr, cholesky,
        # solve, sort, fft...) would otherwise be costed as mere byte traffic
        # (reviewer #13). Estimate their FLOPs from algorithmic complexity.
        if not flops:
            flops = _extra_complexity_flops(opname, args, result)

        # Bytes for the memory-bound ops (everything else). Read inputs + write
        # outputs; a crude but standard roofline input.
        bytes_moved = _tree_bytes(args, kwargs) + _result_bytes(result)
        dtype = _dtype_of(args, result)

        if flops or bytes_moved:
            s = self._charge(float(flops), float(bytes_moved), real_s, dtype,
                             op_start_perf=_t0, device=_current_device())
            self.total_flops += int(flops)
            self.total_charged_s += s
        else:
            # An op we don't model (no flops, no tensors): its real CPU cost is
            # charged by the clock as ordinary elapsed time, so nothing to do.
            pass
        return result


_installed = False
_mode: FlopChargingMode | None = None


def install() -> FlopChargingMode:
    """Enter a global FlopChargingMode that stays active for the process."""
    global _installed, _mode
    if _installed:
        return _mode
    install_cuda_region_marker()
    _mode = FlopChargingMode()
    _mode.__enter__()
    _installed = True
    # the installing (main) thread already has the mode on its dispatch stack;
    # mark it so _ensure_thread_mode does not enter a second one here.
    _thread_mode.entered = True
    _thread_mode.mode = _mode
    return _mode


def uninstall() -> None:
    global _installed, _mode
    if _installed and _mode is not None:
        _mode.__exit__(None, None, None)
        _installed = False
        _mode = None
