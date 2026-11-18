"""Fake distributed: a torchrun job of N processes behaves like N GPUs on NCCL,
while the collectives really run on CPU over gloo.

gpu-tasks' records come from torchrun --nproc_per_node=N jobs using NCCL
all_reduce / all_gather / broadcast. FakeGPU models N GPUs inside ONE process and
makes every collective a no-op, which is wrong for N separate processes: a no-op
all_reduce leaves each rank with its own value, so a job whose control flow
depends on the reduced value (a reduced val_loss, a gathered parameter) would
desync. gpu-tasks confirmed (2026-10-19) that no record needs real cross-rank
data movement — every rank must simply get the SAME value — but a job we are NOT
serving from a record (an agent's own distributed experiment, or any step before
a record's value is available) still needs the collective to be correct.

So we back the collectives with a real gloo process group over TCP. gloo is
CPU-only and needs no GPU, so an N-process torchrun really reduces across the N
processes and every rank gets the identical, correct result — which is also
exactly the "same value on every rank" invariant gpu-tasks needs. We keep NCCL's
reported identity: get_backend() says "nccl", the tensors stay cuda-labelled.

When the daemon serves a value for a collective's output (replay), that served
value is written into the tensor after the gloo op, so all ranks still agree and
the recorded number wins. Without a daemon, the gloo result stands.

Design: capture torch.distributed's genuine functions before FakeGPU patches
them, init a hidden gloo group alongside, and route each collective through it on
the unwrapped CPU storage of the (meta/cpu-backed) cuda tensors.
"""

from __future__ import annotations

import os

# Captured at import, before FakeGPU's patch replaces them. cuda_hooks imports
# this module BEFORE calling fakegpu patch() so these are genuine.
import torch
import torch.distributed as _dist

_GENUINE = {
    "init_process_group": _dist.init_process_group,
    "all_reduce": _dist.all_reduce,
    "broadcast": _dist.broadcast,
    "all_gather": _dist.all_gather,
    "all_gather_into_tensor": getattr(_dist, "all_gather_into_tensor", None),
    "reduce_scatter_tensor": getattr(_dist, "reduce_scatter_tensor", None),
    "barrier": _dist.barrier,
    "is_initialized": _dist.is_initialized,
    "destroy_process_group": _dist.destroy_process_group,
    "reduce": getattr(_dist, "reduce", None),
    "gather": getattr(_dist, "gather", None),
    "scatter": getattr(_dist, "scatter", None),
    "_reduce_scatter_base": getattr(_dist, "_reduce_scatter_base", None),
    # subgroup surface (next/ r8.2): captured genuine at dist_hooks IMPORT, which
    # cuda_hooks runs BEFORE FakeGPU patch(), so these are torch's own, not
    # FakeGPU's group-ignoring stand-ins. NeMo's model-parallel code makes a
    # singleton PP group with new_group and then asserts get_rank(group=pp)==0;
    # FakeGPU's _upstream getters ignore the group and return the GLOBAL rank, so
    # the assert fails on every rank but 0 (confirmed, Astra subgroup-probe-4,
    # 2026-11-02). Wrapped in install() to be group-aware on the hidden gloo.
    "new_group": getattr(_dist, "new_group", None),
    "get_rank": getattr(_dist, "get_rank", None),
    "get_world_size": getattr(_dist, "get_world_size", None),
}

# Under replay a collective over more elements than this is skipped (gradients,
# parameter broadcasts); smaller ones run for real. 1 M elements is 4 MB in fp32.
_REPLAY_SKIP_NUMEL = int(os.environ.get("NVRT_REPLAY_SKIP_NUMEL", str(1 << 20)))


def _numel(objs) -> int:
    n, stack = 0, list(objs)
    while stack:
        o = stack.pop()
        if isinstance(o, torch.Tensor):
            n += o.numel()
        elif isinstance(o, (list, tuple)):
            stack.extend(o)
    return n


_gloo_group = None          # the hidden real gloo group
_reported_backend = "nccl"
_initialized = False


def _unwrap(t):
    """Get the real CPU storage behind a (possibly FakeCuda) tensor."""
    raw = getattr(t, "raw_data", None)
    if raw is not None:
        return raw
    if t.is_cuda if hasattr(t, "is_cuda") else False:
        try:
            return t.cpu()
        except Exception:
            return t
    return t



def _loopback_ifname() -> str:
    """gloo's TCP transport binds to a named interface; pick one that exists on
    THIS host. The old code hardcoded "lo0" (macOS), which does not exist in a
    Linux container (loopback is "lo"), so gloo raised
    'Unable to find address for: lo0' and every rank died at init. Honour an
    explicit GLOO_SOCKET_IFNAME; otherwise choose the real loopback by probing
    /sys/class/net (Linux) and falling back to the BSD/macOS name."""
    env = os.environ.get("GLOO_SOCKET_IFNAME")
    if env:
        return env
    try:
        names = set(os.listdir("/sys/class/net"))  # Linux
        for cand in ("lo", "eth0", "en0"):
            if cand in names:
                return cand
        if names:
            return sorted(names)[0]
    except OSError:
        pass
    return "lo0"  # BSD / macOS loopback


def install() -> None:
    """Patch torch.distributed so collectives run on a hidden gloo group but the
    job sees NCCL. Call AFTER fakegpu patch(), so we win over its no-ops."""
    global _initialized
    if _initialized:
        return

    def init_process_group(backend=None, init_method=None, timeout=None,
                           world_size=-1, rank=-1, store=None, group_name="",
                           pg_options=None, device_id=None):
        global _gloo_group, _reported_backend
        _reported_backend = "nccl" if backend in (None, "nccl") else str(backend)
        # torchrun jobs call init_process_group() with no rank/world_size and
        # rely on the env:// defaults (RANK, WORLD_SIZE set by the launcher).
        # Defaulting to a 1-rank group here would silently turn an N-rank job
        # into N separate 1-rank jobs. Read the launcher's env like torch does.
        if world_size in (-1, None):
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if rank in (-1, None):
            rank = int(os.environ.get("RANK", "0"))
        ws = int(world_size)
        rk = int(rank)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        # gloo tries to resolve the hostname and warns on failure; pin the
        # loopback interface that actually exists on this host (lo on Linux).
        os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_ifname())
        # a real gloo default group so collectives actually move data CPU-side
        _GENUINE["init_process_group"](backend="gloo", init_method=init_method,
                                       world_size=ws, rank=rk, store=store)
        # keep FakeGPU's own distributed globals consistent so its is_initialized
        # / get_rank / get_world_size (which some code calls) report correctly.
        try:
            import nvcompat._upstream as up
            up._DIST_INITIALIZED = True
            up._DIST_BACKEND = _reported_backend
            up._DIST_WORLD_SIZE = ws
            up._DIST_RANK = rk
        except Exception:
            pass
        return None

    def _collective(name):
        genuine = _GENUINE[name]

        def wrapper(*args, **kwargs):
            if genuine is None:
                return None
            # Under replay, gradient-sized collectives are skipped. Weights
            # never change and gradients are placeholders, so moving them
            # observes nothing, while on gloo an N-rank all-reduce of a 124 M
            # model costs real seconds per step. Tags on placeholder grads do
            # not survive FakeGPU's sync_parameter_grads, which rewraps .grad
            # after every backward, so size is the test. Small collectives run
            # for real. The Trainer gathers its loss and metrics with all_gather
            # into an empty buffer, so skipping that left garbage in the logged
            # loss, and an all_reduce SUM of equal values is world_size times
            # the value, not the value (2026-10-20). Barriers are real.
            try:
                import replay as _replay
                if _replay.is_active() and _numel(list(args) + list(kwargs.values())) > _REPLAY_SKIP_NUMEL:
                    return _DoneWork() if kwargs.get("async_op") else None
            except Exception:
                pass
            # ReduceOp.AVG exists on NCCL but not on gloo ("Cannot use
            # ReduceOp.AVG with Gloo"). Track-3 scripts average val_loss with it.
            # Emulate: SUM over the group, then divide the tensor by world size.
            avg = False
            op = kwargs.get("op", args[1] if len(args) > 1 else None)
            try:
                if name in ("all_reduce", "reduce") and op is not None \
                        and op == _dist.ReduceOp.AVG:
                    avg = True
                    if "op" in kwargs:
                        kwargs = dict(kwargs, op=_dist.ReduceOp.SUM)
                    else:
                        args = (args[0], _dist.ReduceOp.SUM) + tuple(args[2:])
            except Exception:
                avg = False
            # Operate on unwrapped CPU tensors so gloo accepts them. The results
            # are copied back into the original (cuda-labelled) tensors.
            work = _run_on_cpu(genuine, args, kwargs)
            if avg and args and isinstance(args[0], torch.Tensor):
                # Divide by the REQUESTED GROUP's size, not the global world size
                # (next/ r8.2): an AVG all_reduce over a model-parallel subgroup
                # of G ranks divides by G, not by the whole world (Astra, 2026-11-02).
                _grp = kwargs.get("group", args[3] if len(args) > 3 else None)
                ws = max(1, _dist.get_world_size(group=_grp))
                t = args[0]
                raw = getattr(t, "raw_data", None)
                tgt = raw if raw is not None else t
                # A replay dummy (a zero grad) averages to itself, and a slim
                # one (one element expanded to the shape) rejects in-place
                # writes outright; one rank divides by one. Skip all three
                # (psgd's `all_reduce(p.grad, op=AVG)`, probe 2026-10-21).
                slim = tgt.dim() > 0 and tgt.numel() > 1 and any(st == 0 for st in tgt.stride())
                if ws > 1 and not slim and not _all_dummy(args):
                    tgt.div_(ws)
            return work
        return wrapper

    _dist.init_process_group = init_process_group
    # get_backend reports NCCL once we have initialised; use the genuine
    # is_initialized (gloo really did init) rather than FakeGPU's flag.
    _genuine_is_init = _GENUINE.get("is_initialized") or _dist.is_initialized

    def get_backend(group=None):
        try:
            inited = _dist.is_initialized()
        except Exception:
            inited = False
        return _reported_backend if inited else None

    _dist.get_backend = get_backend

    # Group-aware rank/size on the hidden gloo (next/ r8.2). FakeGPU's _upstream
    # getters ignore the group argument and return the global rank/world size, so
    # a model-parallel singleton subgroup reports the wrong rank and NeMo's
    # init_model_parallel assertion fails. Delegate to GENUINE torch, which does
    # know the real gloo group's membership: a member gets its rank-within-group
    # (0 for a singleton), a non-member gets -1, world_size is the group's size.
    # With no group (the default group) report the values init_process_group set.
    _gen_get_rank = _GENUINE.get("get_rank")
    _gen_get_world_size = _GENUINE.get("get_world_size")
    _gen_new_group = _GENUINE.get("new_group")

    def _default_rank():
        return int(os.environ.get("RANK", "0"))

    def _default_world_size():
        return int(os.environ.get("WORLD_SIZE", "1"))

    def get_rank(group=None):
        if group is None:
            return _default_rank()
        if _gen_get_rank is not None:
            return int(_gen_get_rank(group=group))
        return _default_rank()

    def get_world_size(group=None):
        if group is None:
            return _default_world_size()
        if _gen_get_world_size is not None:
            return int(_gen_get_world_size(group=group))
        return _default_world_size()

    def new_group(ranks=None, timeout=None, backend=None, pg_options=None,
                  use_local_synchronization=False, group_desc=None):
        # Translate any requested backend to gloo: an explicit backend="nccl"
        # otherwise builds a native ProcessGroupNCCL, which raises on a no-GPU
        # host ("only supported with GPUs, no GPUs found", Astra subgroup-probe-4).
        # The genuine new_group over the hidden gloo makes a real subgroup whose
        # membership the group-aware getters above then read correctly.
        if _gen_new_group is None:
            return None
        kw = {}
        if ranks is not None:
            kw["ranks"] = ranks
        if timeout is not None:
            kw["timeout"] = timeout
        return _gen_new_group(backend="gloo", **kw)

    _dist.get_rank = get_rank
    _dist.get_world_size = get_world_size
    _dist.new_group = new_group

    # Barriers are real, on the hidden gloo group. FakeGPU's patch makes
    # barrier a no-op, and this module captured the genuine one but never put
    # it back, so accelerate's main_process_first / local_main_process_first
    # let every rank through at once. In the DPO replay all four ranks then ran
    # the 32-process dataset map together, and the job's cgroup OOM-killed
    # them. A real node maps on rank 0 and the others read its cache
    # (2026-10-20). device_ids is NCCL-only, so it is dropped.
    def barrier(group=None, async_op=False, device_ids=None):
        try:
            inited = _GENUINE["is_initialized"]()
        except Exception:
            inited = False
        if not inited:
            # same error a real torch gives before init_process_group
            raise ValueError("Default process group has not been initialized, "
                             "please make sure to call init_process_group.")
        return _GENUINE["barrier"](group=group, async_op=async_op)

    _dist.barrier = barrier
    for name in ("all_reduce", "broadcast", "all_gather",
                 "all_gather_into_tensor", "reduce_scatter_tensor",
                 "reduce", "gather", "scatter", "_reduce_scatter_base"):
        if _GENUINE.get(name) is not None:
            setattr(_dist, name, _collective(name))
    _initialized = True


class _DoneWork:
    """A completed async work handle (for async_op=True on a skipped collective)."""
    def wait(self, timeout=None):
        return True

    def is_completed(self):
        return True

    def is_success(self):
        return True


def _all_dummy(args) -> bool:
    """True when every tensor argument is a replay dummy (a zero grad carrying
    no information): a collective over them moves nothing meaningful, and on an
    N-rank job of a 124 M model it would cost real seconds per step on gloo."""
    saw = False
    stack = list(args)
    while stack:
        o = stack.pop()
        if isinstance(o, torch.Tensor):
            saw = True
            if not getattr(o, "_gm_dummy", False):
                return False
        elif isinstance(o, (list, tuple)):
            stack.extend(o)
    return saw


def _run_on_cpu(genuine, args, kwargs):
    """Run a genuine gloo collective on the CPU storage of its tensor arguments,
    then copy results back into the original tensors so cuda-labelled tensors see
    the reduced values."""
    if _all_dummy(args):
        # replay: every rank's zero grads reduce to zero grads; skip the data
        # movement, keep the "every rank gets the same value" invariant.
        return _DoneWork() if kwargs.get("async_op") else None
    # Map each tensor arg to its CPU storage, remember the pairing to copy back.
    pairs = []  # (original_tensor, cpu_tensor)

    def to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            cpu = _unwrap(obj).detach().clone()
            pairs.append((obj, cpu))
            return cpu
        if isinstance(obj, (list, tuple)):
            return type(obj)(to_cpu(o) for o in obj)
        return obj

    cpu_args = tuple(to_cpu(a) for a in args)
    cpu_kwargs = {k: to_cpu(v) for k, v in kwargs.items()}
    work = genuine(*cpu_args, **cpu_kwargs)
    # copy results back
    for orig, cpu in pairs:
        try:
            raw = getattr(orig, "raw_data", None)
            (raw if raw is not None else orig).copy_(cpu)
        except Exception:
            pass
    return work
