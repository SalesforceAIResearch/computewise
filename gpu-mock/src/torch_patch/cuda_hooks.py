"""gpu-mock layer D: patch torch.cuda to present a 4×H100 box that agrees with

the NVML layer, with every FakeGPU tell removed.

This wraps FanBB2333/FakeGPU's `torch_patch.patch()` (which backs cuda tensors
with CPU tensors and runs ops transparently) and then corrects every tell an
adversarial agent could use (tests/adversarial_probes.py), reaching into
FakeGPU's own classes by name so the fixes are at the source of each tell rather
than bolted on from outside.

Imported for its side effects, normally from `sitecustomize.py`, so every
interpreter the agent starts is patched before its first `import torch`.

The identity it presents is read from NVRT_* env vars so it stays identical
to the NVML config the bridge serves — the two layers must tell one story.

Tells handled, each verified by tests/adversarial_probes.py:
  identity:  device name, SM count, cc, arch list, CUDA/cuDNN versions, hip
  type:      tensor type name, tensor repr (device='cuda:N'), properties repr
  nvml:      torch.cuda.utilization/temperature/power_draw/clock_rate
  autocast:  bf16 autocast actually yields bf16
  memory:    mem_get_info / memory_allocated track allocations
  noise:     patch warning and atexit summary suppressed
  timing:    compute wall-time pacing is wired to the daemon when present
             (TimingPacer); without a daemon the standalone fallback is honest
             about being unpaced.
"""

from __future__ import annotations

import os
import sys
import warnings

# ---- target identity: keep in lockstep with the NVML config ----------------
DEVICE_NAME = os.environ.get("NVRT_DEVICE_NAME", "NVIDIA H100 80GB HBM3")
# The box's GPU count: install_into_image.sh writes the device count of the NVML
# profile it installs in place of the placeholder, so torch.cuda.device_count(),
# nvidia-smi and the daemon's --gpus agree on any box (4 for every box before the
# 8-GPU image, 2026-10-31). A dev tree keeps the placeholder and reads 4.
_INSTALLED_DEVICE_COUNT = "__NVRT_DEVICE_COUNT__"
DEVICE_COUNT = int(os.environ.get("NVRT_DEVICE_COUNT")
                   or (_INSTALLED_DEVICE_COUNT if _INSTALLED_DEVICE_COUNT.isdigit() else "4"))
SM_COUNT = int(os.environ.get("NVRT_SM_COUNT", "132"))      # H100 SXM
CUDA_VERSION = os.environ.get("NVRT_CUDA_VERSION", "13.0")
CUDNN_VERSION = int(os.environ.get("NVRT_CUDNN_VERSION", "90700"))
PROFILE = os.environ.get("NVRT_PROFILE", "h100")
ARCH_LIST = os.environ.get("NVRT_ARCH_LIST", "sm_90").split(",")

_applied = False
_RelabelDtypeMode = None  # set by _fix_autocast (version-robust fp16 relabel)


def _vendor_alias() -> None:
    """The patch imports FakeGPU as `nvcompat` (vendored under that name by
    install_into_image.sh). Where FakeGPU is still installed under its own
    name (the TRL images until their rebuild, a dev venv), alias it: the
    package and the three submodules the patch uses, so no module is imported
    twice under two names."""
    import importlib
    try:
        importlib.import_module("nvcompat")
        return
    except ImportError:
        pass
    try:
        fg = importlib.import_module("fakegpu")
    except ImportError:
        return
    sys.modules["nvcompat"] = fg
    for sub in ("_upstream", "torch_patch", "_allocator"):
        sys.modules[f"nvcompat.{sub}"] = importlib.import_module(f"fakegpu.{sub}")


def _map_driver_libraries() -> None:
    """Map the box's driver library into the job process, as a CUDA build of
    torch does at its first CUDA call (cuInit loads libcuda.so.1). FakeGPU runs
    on the CPU build, so a job's /proc/<pid>/maps held libtorch_cpu and no
    libcuda: the Grok canary read that beside its job's CPU time (2026-10-26).
    The bundle's stub libcuda.so.1 on LD_LIBRARY_PATH (which also answers cuInit
    and the device queries) is loaded with RTLD_GLOBAL and called by no one; a
    box without it is left as it is. The bundle's libcudart.so.1x are symlinks
    to the same file, so the maps name only libcuda.so.1, and a separate copy
    would load a second Go runtime into the process, which Go does not support.
    Checked on the packages image: DataLoader fork workers, signal handlers and
    cuda ops are unaffected. NVRT_NO_DRIVER_MAP=1 skips it (the tests' control)."""
    if os.environ.get("NVRT_NO_DRIVER_MAP") == "1":
        return
    import ctypes
    try:
        ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass


def apply() -> None:
    _vendor_alias()
    global _applied
    if _applied:
        return

    os.environ.setdefault("FAKEGPU_DEVICE_COUNT", str(DEVICE_COUNT))
    os.environ.setdefault("FAKEGPU_PROFILE", PROFILE)
    os.environ.setdefault("FAKEGPU_CUDA_VERSION", CUDA_VERSION)
    os.environ.setdefault("FAKEGPU_PRINT_SUMMARY", "0")

    # Capture torch.distributed's genuine functions BEFORE FakeGPU replaces them
    # with no-ops, so the fake process group can back collectives with real gloo.
    try:
        import dist_hooks  # noqa: F401  (captures genuine dist funcs at import)
        _have_dist_hooks = True
    except Exception:
        _have_dist_hooks = False

    import torch

    # Fully initialise torch._dynamo while torch is still PRISTINE, before
    # FakeGPU's patch runs (why: see _preimport_dynamo). The Event base must be
    # a proper _EventBase subclass before this, or _dynamo's device_interface
    # assert fires here instead; and a failed first import must not poison
    # every later one (why: see _tolerate_stale_artifact_registration), which
    # has to be in place before _fix_event_base makes the first attempt.
    _tolerate_stale_artifact_registration(torch)
    _fix_event_base(torch)
    _preimport_dynamo(torch)

    # Prime torch's set of device constructors while the factory functions are
    # still the genuine ones. `with torch.device(...)` works through a
    # TorchFunctionMode that injects `device=` only for functions in that
    # lru-cached set; FakeGPU replaces torch.empty/zeros/... with wrappers, and
    # a set built after that holds the wrappers while the mode sees the
    # originals, so the context did nothing: `with torch.device("meta")` gave
    # CPU tensors, and transformers 5.x then built a 3B model on the CPU and
    # initialised every weight in bf16 for real (100 s per model load, 0.6 s
    # without the patch; found 2026-10-22 on the MLS KV box). Built now, the
    # set holds the originals the mode is handed, and meta construction works.
    try:
        from torch.utils import _device as _tdev
        _tdev._device_constructors()
    except Exception:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nvcompat import torch_patch as _fg
        _fg.patch(num_devices=DEVICE_COUNT, device_name=DEVICE_NAME)

    # patch() may reinstall Event/Stream; re-apply the base fix after it.
    _fix_event_base(torch)
    _fix_device_context(torch)
    _fix_tree_map(torch)
    _fix_identity_surface(torch)

    _map_driver_libraries()

    if _have_dist_hooks:
        import dist_hooks
        dist_hooks.install()

    _fix_summary_atexit(_fg)
    _fix_torch_save(torch)
    _fix_visible_devices(torch)
    _fix_device_identity(torch)
    _fix_stream_repr()
    _fix_nvtx(torch)
    _fix_arch_list(torch)
    _fix_versions(torch)
    _fix_tensor_class(torch)
    # A no-dtype .type() query on a cuda-labelled fake tensor must report
    # torch.cuda.<DType>Tensor (Megatron's distributed optimizer string-compares it);
    # next/ r8.4, from Astra's NeMo diagnosis. CPU tensors and conversion calls untouched.
    try:
        import tensor_type_compat
        tensor_type_compat.install()
    except Exception as e:
        if os.environ.get("NVRT_DEBUG"):
            sys.stderr.write(f"nvrt: tensor_type_compat skipped: {e!r}\n")
    _fix_generator(torch)
    _fix_missing_factories(torch)
    _fix_properties_class()
    _fix_nvml_surface(torch)
    _fix_autocast(torch)
    _fix_oom_text(torch)
    _fix_pin_memory(torch)
    _fix_pin_memory_factories(torch)
    _install_timing(torch)
    _wire_event_timing(torch)
    _install_fingerprint_runtime()
    _install_first_cuda_touch()
    _hide_patch_module_names(torch)
    # whole_job (next/ r8): a job whose seamed function does the first CUDA use
    # (TabArena ExperimentRunner._run) must have its seam installed before that
    # call, which the first-CUDA-touch lookup is too late for. Probe once here,
    # at import torch (before the script's own code), and for a whole_job record
    # arm engagement at the seam's module import; every other record is untouched
    # and engages on its normal path. Off with NVRT_NO_REPLAY / NVRT_FINGERPRINT_ONLY.
    try:
        import replay
        # next/ r8.6: a direct C++ engine backward (Megatron's custom_backward) on
        # a fake cuda tensor reached an empty graph. The wrapper routes it to the
        # stub under replay and to a real backward on the inner CPU tensors on the
        # modelled path, so it is installed at apply, whether or not replay engages.
        replay._hook_engine_backward()
        replay.try_whole_job_early()
    except Exception as e:
        if os.environ.get("NVRT_DEBUG"):
            sys.stderr.write(f"nvrt: whole_job early probe skipped: {e!r}\n")

    _applied = True


def _install_first_cuda_touch() -> None:
    """Run once, at the process's first CUDA use: the first tensor placed on a
    fake GPU (`.cuda()`, `.to("cuda")`, a factory with device="cuda"), which
    is where FakeGPU wraps it. Two things happen there:
      - the host allocator is set up for fake-GPU tensors (_fix_host_allocator),
        in every process that uses the fake GPU, replay or not;
      - the exact-key registry lookup (REPLAY-INTERFACE §2): the entry script's
        hash and argv are known by then and nothing is on the GPU yet, so an
        exact_only record engages replay before the model occupies the GPU.
    A process that never touches CUDA (a launcher such as accelerate or
    torchrun, the agent's `python -c`, a CPU-only script) never says hello
    and never appears in the occupancy log. That replaces a skip list of
    launcher names, which could not know every non-job. The job's real start
    time travels with the hello (daemon_client.process_start_time). A structural
    match still waits for the fingerprint at the first step."""
    try:
        from nvcompat import _upstream as up
    except Exception:
        return
    orig = up.wrap_tensor      # FakeGPU's own memory-tracking hook, if installed
    done = {"v": False}
    lookup = not (os.environ.get("NVRT_NO_REPLAY") == "1" or
                  os.environ.get("NVRT_FINGERPRINT_ONLY") == "1")

    def wrap_tensor(t, device_index=None):
        if not done["v"]:
            done["v"] = True
            up.wrap_tensor = orig
            _fix_host_allocator()
            if lookup:
                try:
                    import replay
                    replay.try_lookup("exact")
                except Exception as e:
                    if os.environ.get("NVRT_DEBUG"):
                        sys.stderr.write(f"nvrt: exact lookup at first CUDA touch skipped: {e!r}\n")
        return orig(t, device_index=device_index)

    up.wrap_tensor = wrap_tensor


def _fix_pin_memory_factories(torch) -> None:
    """`torch.empty(..., pin_memory=True)` (a factory keyword, not the
    `.pin_memory()` method) raises on a CPU-only torch because there is no
    accelerator to pin for; a real GPU host accepts it. Track 3's data loader
    does exactly this. Strip the keyword from the factories and mark the result
    pinned, so `is_pinned()` reports True as it would on a GPU host."""
    _flag = "_nvrt_pinned"

    def _wrap(orig):
        def f(*a, **k):
            pin = k.pop("pin_memory", None)
            out = orig(*a, **k)
            if pin:
                try:
                    object.__setattr__(out, _flag, True)
                except Exception:
                    pass
            return out
        try:
            f.__name__ = orig.__name__
            f.__qualname__ = getattr(orig, "__qualname__", orig.__name__)
            f.__doc__ = orig.__doc__
        except Exception:
            pass
        return f

    for name in ("empty", "zeros", "ones", "full", "tensor", "randn", "rand",
                 "randint", "arange", "linspace", "empty_like", "zeros_like",
                 "ones_like", "full_like", "randn_like", "rand_like"):
        orig = getattr(torch, name, None)
        if orig is not None and callable(orig):
            setattr(torch, name, _wrap(orig))


def _install_fingerprint_runtime() -> None:
    """Hooks that compute the live structural fingerprint (registry lookup key)
    at the first optimizer.step(); NVRT_FINGERPRINT_ONLY=1 makes it a
    dry run that prints the fingerprint and exits. Installed last so it wraps
    the autocast class _fix_autocast bound."""
    try:
        import fingerprint_runtime
        fingerprint_runtime.install()
    except Exception as e:  # never let the extractor break a job
        if os.environ.get("NVRT_DEBUG"):
            sys.stderr.write(f"nvrt: fingerprint runtime not installed: {e}\n")


def _fix_pin_memory(torch) -> None:
    """On a real CUDA host, tensor.pin_memory() returns a tensor whose
    is_pinned() is True. FakeGPU leaves it a no-op reporting unpinned. We make
    is_pinned report True for tensors that went through pin_memory(). Pinned
    memory is a host-side page-locking; there is no real GPU involved, so we only
    need the reported flag to be consistent."""
    _orig_pin = torch.Tensor.pin_memory
    _flag = "_nvrt_pinned"

    def pin_memory(self, device=None):
        try:
            out = _orig_pin(self)
        except (RuntimeError, TypeError):
            out = self.clone()
        try:
            object.__setattr__(out, _flag, True)
        except (AttributeError, TypeError):
            pass
        return out

    def is_pinned(self, device=None):
        return bool(getattr(self, _flag, False))

    try:
        torch.Tensor.pin_memory = pin_memory
        torch.Tensor.is_pinned = is_pinned
    except (AttributeError, TypeError):
        pass


# ---------------------------------------------------------------------------
# Timing: the central tell. CPU runs GPU ops at the wrong speed, so the job must
# observe a virtual clock advanced by modelled GPU time. We install a FLOP-
# charging dispatch mode (counts each op's modelled GPU seconds) and patch the
# process clock to read virtual time. Disabled by NVRT_NO_TIMING=1 (for the
# tell tests that want to measure the raw CPU gap).
# ---------------------------------------------------------------------------

def _debug(msg: str) -> None:
    """A diagnostic the agent never sees unless NVRT_DEBUG is set."""
    if os.environ.get("NVRT_DEBUG"):
        sys.stderr.write(f"nvrt: {msg}\n")


def _install_timing(torch) -> None:
    if os.environ.get("NVRT_NO_TIMING") == "1":
        return
    # The clock patch and the FLOP charger install independently. Both used to
    # sit in one try block, so when flop_charger failed to import on torch 2.1
    # the clock was never patched either. Nothing said so, and every job ran on
    # real CPU time (2026-10-20).
    try:
        import virtual_clock as vc
    except Exception as e:
        _debug(f"virtual clock unavailable: {e!r}")
        return
    try:
        import flop_charger
        flop_charger.install()
    except Exception as e:
        _debug(f"FLOP charger unavailable, GPU ops cost real CPU time: {e!r}")
    vc.patch_time()


# ---------------------------------------------------------------------------
def _fix_summary_atexit(fg) -> None:
    import atexit
    fn = getattr(fg, "_dump_terminal_summary", None)
    if fn is not None:
        try:
            atexit.unregister(fn)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Device identity: name + SM count on both the function and the properties.
# ---------------------------------------------------------------------------

def _fix_device_identity(torch) -> None:
    want_name, want_sm = DEVICE_NAME, SM_COUNT
    _orig_props = torch.cuda.get_device_properties

    def _validate(device):
        if device is None:
            return 0
        if isinstance(device, torch.device):
            if device.type != "cuda":
                raise RuntimeError("Invalid device, must be cuda device")
            idx = device.index if device.index is not None else 0
        else:
            idx = int(device)
        n = torch.cuda.device_count()
        if idx < 0 or idx >= n:
            raise RuntimeError(f"Invalid device id: {idx}. Device count is {n}.")
        return idx

    def get_device_name(device=None):
        _validate(device)
        return want_name

    def get_device_properties(device=None):
        _validate(device)
        props = _orig_props(device)
        for attr, val in (("name", want_name),
                          ("multi_processor_count", want_sm)):
            try:
                setattr(props, attr, val)
            except (AttributeError, TypeError):
                pass
        return props

    torch.cuda.get_device_name = get_device_name
    torch.cuda.get_device_properties = get_device_properties


def _tolerate_stale_artifact_registration(torch) -> None:
    """Let torch.compiler._cache.CacheArtifactFactory.register replace an
    entry left behind by a FAILED import, instead of asserting.

    torch 2.11's torch/_dynamo/package.py registers the 'precompile' cache
    artifact at import (line 459) and then builds the inductor cache dir at
    module level (line 1173), which calls getpass.getuser(). Under a uid with
    no passwd entry — a bare container started as the host's uid, which is how
    cpu-tasks' H4 staging check ran the filter image on 2026-10-24 — that
    raises KeyError, so the import fails after the registration and Python
    drops the half-run module. Every later import of torch._dynamo — ours in
    _fix_event_base and _preimport_dynamo, FakeGPU's, torch's own lazy one at
    the first op under a dispatch mode, the script's — then dies in register()
    with "Artifact of type=precompile already registered in mega-cache
    artifact factory", which hides the cause and is not what the script sees
    on a host that never pre-imports dynamo: there the first attempt is the
    script's, and it sees the KeyError. Stock torch under a TorchDispatchMode
    fails at the same first op with that KeyError, so with this in place the
    mock matches it. A stale entry is one whose class is no longer the object
    its module exports (the module was dropped, or is re-executing and has
    not reached the class yet); a live duplicate still asserts."""
    try:
        from torch.compiler import _cache as _cc
    except Exception:
        return
    factory = _cc.CacheArtifactFactory
    if getattr(factory, "_nvrt_tolerant", False):
        return
    orig = factory.__dict__.get("register")
    if orig is None or not hasattr(orig, "__func__"):
        return
    orig_fn = orig.__func__

    def register(cls, artifact_cls):
        try:
            key = artifact_cls.type()
            prev = cls._artifact_types.get(key)
            if prev is not None:
                mod = sys.modules.get(getattr(prev, "__module__", "") or "")
                if mod is None or getattr(mod, getattr(prev, "__name__", ""), None) is not prev:
                    del cls._artifact_types[key]
        except Exception:
            pass
        return orig_fn(cls, artifact_cls)

    factory.register = classmethod(register)
    factory._nvrt_tolerant = True


def _preimport_dynamo(torch) -> None:
    """Import torch._dynamo once, pristine, before FakeGPU's patch runs.

    On torch 2.4 FakeGPU's patch leaves _dynamo in a half-initialised state,
    so a later first import of it (triggered by our TorchDispatchMode, or by
    the job) re-enters mid-init and fails with "partially initialized module
    torch._dynamo has no attribute external_utils". Importing it first,
    pristine, fully loads it. Harmless where _dynamo is absent (very new
    torch) or already imported. If the import fails (a uid with no passwd
    entry: see _tolerate_stale_artifact_registration) it is swallowed here,
    and the script's own first touch of dynamo raises the same error, as on
    a real host; NVRT_DEBUG=1 names it on stderr."""
    try:
        import torch._dynamo  # noqa: F401
    except Exception as e:
        if os.environ.get("NVRT_DEBUG"):
            sys.stderr.write(f"nvrt: dynamo pre-import failed ({type(e).__name__}: {e}); "
                             "the script's first use of dynamo raises it\n")


def _dynamo_enforces_stream_base(torch) -> bool:
    """True iff this torch's Dynamo device_interface asserts that
    torch.cuda.Event/Stream subclass the stream base (torch 2.4-era behaviour).
    Read the source of DeviceInterfaceMeta without importing it (importing would
    bind the current, possibly-plain, class and could trip the very assert).
    2.14 removed the assert, so this returns False there and we leave FakeGPU's
    plain classes — whose __init__ sets attributes the 2.14 readonly-slot base
    would forbid — untouched."""
    try:
        import inspect
        from torch._dynamo import device_interface as di
        src = inspect.getsource(di)
        return "should be inherit from _EventBase" in src or \
               "_EventBase" in src and "assert" in src
    except Exception:
        # device_interface not importable (very new torch) → no enforcement
        return False


def _fix_event_base(torch) -> None:
    """On torch >= 2.4 (dropped by ~2.14), Dynamo's device_interface binds
    `CudaInterface.Event = torch.cuda.Event` at import and asserts it subclasses
    `torch._streambase._EventBase` (Stream vs _StreamBase likewise). FakeGPU's
    `_FakeEvent`/`_FakeStream` are plain classes, so the first import of
    torch._dynamo (which FakeGPU's own patch triggers) crashes. We rebuild the
    current `torch.cuda.Event`/`Stream` with the required base, keeping their
    attributes, and run this BOTH before and after FakeGPU's patch so the class
    Dynamo sees always has the base. No-op where the base module is absent."""
    # Only torch versions whose Dynamo device_interface ENFORCES the base need
    # this; 2.14 has the base classes but does not assert, and its base uses
    # readonly slots that break FakeGPU's __init__ (it sets device_index). So
    # apply the fix only when dynamo's device_interface would reject the current
    # class — detected by whether that module enforces it. We check cheaply: the
    # assert lives in DeviceInterfaceMeta; it runs iff device_interface is not
    # yet imported AND the torch version pins it. Rather than parse versions, we
    # probe: rebuild only if we can CONSTRUCT the rebuilt class (writable slots);
    # on 2.14 construction fails the readonly-slot way, so we skip and leave the
    # plain FakeGPU class (which 2.14 accepts).
    try:
        import torch._streambase as sb
    except Exception:
        return
    if not _dynamo_enforces_stream_base(torch):
        return
    for attr, base in (("Event", getattr(sb, "_EventBase", None)),
                       ("Stream", getattr(sb, "_StreamBase", None))):
        cls = getattr(torch.cuda, attr, None)
        if cls is None or base is None:
            continue
        if isinstance(cls, type) and issubclass(cls, base):
            continue
        if not isinstance(cls, type):
            continue
        try:
            ns = {k: v for k, v in cls.__dict__.items()
                  if k not in ("__dict__", "__weakref__")}

            def __new__(subcls, *a, __b=base, **k):
                return __b.__new__(subcls)

            ns.setdefault("__new__", __new__)
            # The base is an ABC. FakeGPU's class leaves some of its abstract
            # methods undefined (_StreamBase: __eq__; others vary by torch
            # version), which made the rebuilt class abstract, so on torch 2.4
            # `torch.cuda.Stream()` raised "Can't instantiate abstract class"
            # (adversarial probe, 2026-10-20). Fill each missing one with what a
            # real stream does when it has nothing queued: equality is identity
            # (hash kept), query() reports done, the rest return None.
            for name in getattr(base, "__abstractmethods__", ()):
                if name in ns:
                    continue
                if name == "__eq__":
                    ns["__eq__"] = lambda self, other: self is other
                    ns.setdefault("__hash__", object.__hash__)
                elif name == "query":
                    ns["query"] = lambda self: True
                else:
                    ns[name] = lambda self, *a, **k: None
            rebuilt = type(cls.__name__, (base,), ns)
            rebuilt.__abstractmethods__ = frozenset()
            setattr(torch.cuda, attr, rebuilt)
            # keep FakeGPU's module reference in sync, if it has one
            try:
                from nvcompat import _upstream as up
                for alias_name in (f"_Fake{attr}",):
                    if getattr(up, alias_name, None) is cls:
                        setattr(up, alias_name, rebuilt)
            except Exception:
                pass
        except Exception:
            pass


def _fix_host_allocator() -> None:
    """Keep freed host memory for reuse, as CUDA's caching allocator keeps
    freed device memory. A fake-GPU tensor is backed by host memory. glibc
    serves each block over its mmap threshold (32 MB at most) with a fresh
    mmap, and returns it with munmap the moment the tensor is freed. That
    munmap runs BETWEEN ops, where the virtual clock counts real time. It took
    about 4 ms per 36 MB result on the GCP host, so a loop of 3072² matmuls was
    paced by munmap and not by the modelled GPU time. Spreading the loop over 4
    GPUs then gave 0.9× instead of about 4× (adversarial probe, 2026-10-20). With
    mmap off for new blocks and the heap never trimmed, the free takes 0.06 ms
    and the probe gives 3.3×. Applied at the process's first CUDA touch, not at
    import. The first version ran at import, so the 32-process dataset map that
    every DPO rank forks before it touches CUDA inherited it. Every tokenizer
    worker then kept its heap, and the job's container went from 28 GiB to
    54 GiB under a 64 GiB cap (2026-10-20). Workers forked after the first touch
    (DataLoader workers) inherit it. Trimming is kept, above _ALLOC_TRIM_BYTES of
    free heap top, so RSS stays within that of live memory, and a loop of
    results is still reused without munmap. Large blocks from threads other
    than the main one can still be mmapped, as glibc falls back to mmap
    there. NVRT_NO_MALLOPT=1 disables this."""
    if os.environ.get("NVRT_NO_MALLOPT") == "1":
        return
    try:
        import ctypes
        mallopt = ctypes.CDLL(None).mallopt       # glibc; absent on musl
    except (OSError, AttributeError):
        return
    M_TRIM_THRESHOLD, M_MMAP_MAX = -1, -4
    try:
        mallopt(M_MMAP_MAX, 0)
        mallopt(M_TRIM_THRESHOLD, _ALLOC_TRIM_BYTES)
    except Exception:
        pass


# Free heap kept at the top before glibc trims it: room for a working set of
# fake-GPU results to be reused without unmapping, bounded so a process's RSS
# exceeds its live memory by at most this much.
_ALLOC_TRIM_BYTES = int(os.environ.get("NVRT_ALLOC_TRIM_BYTES", str(1 << 30)))


def _fix_torch_save(torch) -> None:
    """Save fake-GPU tensors as plain tensors. A FakeCudaTensor is a Tensor
    subclass with no pickling of its own, so torch.save wrote the subclass's
    name into the file. torch.load's default safe mode (weights_only, the
    default since torch 2.6) refuses unknown classes, so a checkpoint a replayed
    job saved could not be loaded by the next segment of a chain (gpu-tasks'
    segment scripts, 2026-10-20). Each fake-GPU tensor is replaced by its CPU
    storage before pickling; FakeGPU's patched torch.load then maps the plain
    tensors back onto the fake GPU under map_location, as a real checkpoint
    saved from cuda would come back. Containers keep their types."""
    try:
        from nvcompat import _upstream as up
        base_cls = up.FakeCudaTensor
    except Exception:
        return
    orig = torch.save
    if getattr(orig, "_nvrt_wrapped", False):
        return

    def plain(o):
        if isinstance(o, base_cls):
            raw = getattr(o, "raw_data", None)
            t = raw if isinstance(raw, torch.Tensor) else o.cpu()
            return t.detach()
        if isinstance(o, dict):
            return type(o)((k, plain(v)) for k, v in o.items())
        if isinstance(o, (list, tuple)):
            vals = [plain(v) for v in o]
            return type(o)(vals) if not hasattr(o, "_fields") else type(o)(*vals)
        return o

    def save(obj, f, *a, **k):
        return orig(plain(obj), f, *a, **k)

    save._nvrt_wrapped = True
    try:
        import functools
        functools.update_wrapper(save, orig)
    except Exception:
        pass
    torch.save = save
    try:
        torch.serialization.save = save
    except Exception:
        pass


def _fix_generator(torch) -> None:
    """`torch.Generator(device="cuda")` asks ATen for a CUDA generator, which
    the CPU build has not got ("Cannot get CUDA generator without ATen_cuda
    library"). On the fake device a generator asked for cuda is a CPU
    generator that remembers the device it was asked for; sampling with it on
    a cuda-labelled tensor runs on the CPU as every fake op does. 5 of the 86
    registry scripts build one (all-scripts probe, 2026-10-21)."""
    base = torch.Generator
    if getattr(base, "_gm_generator", False):
        return

    class Generator(base):
        def __new__(cls, device="cpu", *a, **k):
            d = str(getattr(device, "type", device))
            if d.startswith("cuda"):
                self = base.__new__(cls, "cpu")
                try:
                    idx = getattr(device, "index", None)
                    if idx is None and ":" in str(device):
                        idx = int(str(device).split(":")[1])
                    self._gm_device = torch.device("cuda", idx if idx is not None else 0)
                except Exception:
                    self._gm_device = torch.device("cuda", 0)
                return self
            return base.__new__(cls, device, *a, **k)

        @property
        def device(self):
            d = getattr(self, "_gm_device", None)
            return d if d is not None else base.device.__get__(self)

    Generator.__name__ = Generator.__qualname__ = "Generator"
    Generator.__module__ = "torch"
    Generator._gm_generator = True
    torch.Generator = Generator


# Factories FakeGPU does not route to the CPU when asked for a cuda device
# (they reach ATen's CUDA dispatch key and raise NotImplementedError). Found by
# enumerating torch's factories inside the mng3 image, 2026-10-21; 10 registry
# scripts call torch.randperm(..., device=cuda, generator=gen).
_MISSING_FACTORIES = ("randperm", "normal", "empty_strided", "tril_indices", "triu_indices",
                      "hann_window", "hamming_window", "bartlett_window", "blackman_window",
                      "kaiser_window")


def _fix_missing_factories(torch) -> None:
    """Run the factory on the CPU and move the result to the fake device,
    which is what FakeGPU does for the factories it covers."""
    def _is_cuda(dev) -> bool:
        return dev is not None and str(getattr(dev, "type", dev)).startswith("cuda")

    def _wrap(orig):
        def f(*a, **k):
            dev = k.get("device")
            if not _is_cuda(dev):
                return orig(*a, **k)
            k = dict(k)
            k["device"] = "cpu"
            out = orig(*a, **k)
            return out.to(dev)
        try:
            f.__name__ = orig.__name__
            f.__qualname__ = getattr(orig, "__qualname__", orig.__name__)
            f.__doc__ = orig.__doc__
        except Exception:
            pass
        f._gm_factory = True
        return f

    for name in _MISSING_FACTORIES:
        orig = getattr(torch, name, None)
        if orig is not None and callable(orig) and not getattr(orig, "_gm_factory", False):
            setattr(torch, name, _wrap(orig))


def _hide_patch_module_names(torch) -> None:
    """Run last: every callable in torch and torch.cuda that reports this
    module or the vendored library as its __module__ is rebound to torch's
    own namespace, as a native build's C functions read. Runs after all
    installs so closures bound late (device_count, is_available, ...) are
    covered. Cosmetic: __module__ only, no behaviour."""
    import types as _t
    try:
        for ns, modname in ((getattr(torch, "cuda", None), "torch.cuda"), (torch, "torch")):
            if ns is None or not hasattr(ns, "__dict__"):
                continue
            for attr in list(vars(ns)):
                fn = getattr(ns, attr, None)
                mod = getattr(fn, "__module__", "") or ""
                if isinstance(fn, _t.FunctionType) and (mod == __name__ or mod.startswith("nvcompat") or mod.startswith("fakegpu")):
                    try:
                        fn.__module__ = modname
                    except Exception:
                        pass
    except Exception:
        pass


def _fix_identity_surface(torch) -> None:
    """Make the box report the accelerator build it emulates, not the CPU
    wheel underneath. The emulated node is a 4xH100 CUDA 13.0 host; the
    installed torch is a CPU wheel with the patch providing the CUDA surface,
    so three queries disagreed with a real node: torch.__version__ ended
    +cpu, torch._C._has_cuda was False, and patched callables in torch.cuda
    reported this module as their __module__. A real build answers +cu130,
    True, and torch.cuda. This aligns those three to the build the box
    already claims through torch.version.cuda and nvidia-smi; it changes no
    computation, only what the box says it is.
    """
    try:
        cu = getattr(getattr(torch, "version", None), "cuda", None)
        tag = "cu" + str(cu).replace(".", "") if cu else None
        v = str(getattr(torch, "__version__", ""))
        if tag and v.endswith("+cpu"):
            newv = v[:-4] + "+" + tag
            try:
                torch.__version__ = newv
            except Exception:
                pass
            ver = getattr(torch, "version", None)
            if ver is not None:
                try:
                    ver.__version__ = newv
                except Exception:
                    pass
        C = getattr(torch, "_C", None)
        if C is not None:
            for name in ("_has_cuda", "_has_cudnn"):
                try:
                    if getattr(C, name, True) is False:
                        setattr(C, name, True)
                except Exception:
                    pass
    except Exception:
        pass


def _fix_tree_map(torch) -> None:
    """FakeGPU wraps the results of every op on a fake-cuda tensor through its
    `_tree_map`, which rebuilds any tuple as a plain tuple, so `topk`, `sort`,
    `max(dim)`, `median`, `kthvalue`, `mode`, `cummax`, `aminmax`, `svd`,
    `slogdet`, `frexp` … came back without their `torch.return_types` type and
    `.indices` / `.values` raised AttributeError (G-mlskv-haiku-L3's debug
    script, analysis 2026-10-23 03:50Z); CPU tensors were fine. Replace it with
    one that rebuilds namedtuples and structseqs as their own type. The
    wrappers look the name up at call time, so the module attribute is enough."""
    try:
        from nvcompat import _upstream as _up
    except Exception:
        return
    orig = getattr(_up, "_tree_map", None)
    if orig is None or getattr(orig, "_gm_structseq", False):
        return
    from collections import OrderedDict

    def _tree_map(fn, obj):
        if isinstance(obj, torch.Size):
            return obj
        if isinstance(obj, tuple):
            vals = [_tree_map(fn, x) for x in obj]
            t = type(obj)
            if t is tuple:
                return tuple(vals)
            if hasattr(t, "n_fields"):          # a structseq (torch.return_types.*): one sequence argument
                try:
                    return t(vals)
                except TypeError:
                    return tuple(vals)
            if hasattr(obj, "_fields"):         # a namedtuple
                try:
                    return t(*vals)
                except TypeError:
                    return tuple(vals)
            try:
                return t(vals)
            except TypeError:
                return tuple(vals)
        if isinstance(obj, list):
            return [_tree_map(fn, x) for x in obj]
        if isinstance(obj, OrderedDict):
            return OrderedDict((k, _tree_map(fn, v)) for k, v in obj.items())
        if isinstance(obj, dict):
            return obj.__class__((k, _tree_map(fn, v)) for k, v in obj.items())
        return fn(obj)

    _tree_map._gm_structseq = True
    _up._tree_map = _tree_map


def _fix_device_context(torch) -> None:
    """Make `with torch.device("cuda")` build fake-cuda tensors. The context is
    a TorchFunctionMode that injects `device=` into the genuine factory
    functions (its set was primed above, so the meta context now works); for a
    cuda device that injection would reach torch's CUDA backend, which does not
    exist here, so route it the way an explicit device="cuda" goes: build on
    the CPU and wrap as FakeGPU's tensor. Before this fix the context silently
    produced plain CPU tensors (the set held FakeGPU's wrappers, the mode saw
    the originals, nothing was injected)."""
    try:
        from torch.utils import _device as _tdev
        from nvcompat import _upstream as _up
    except Exception:
        return
    cls = getattr(_tdev, "DeviceContext", None)
    if cls is None or getattr(cls, "_gm_device_fixed", False):
        return
    orig = cls.__torch_function__
    wrap = getattr(_up, "_wrap_function_result", None)
    if wrap is None:
        return

    def _torch_function(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        dev = getattr(self, "device", None)
        if (dev is not None and getattr(dev, "type", None) == "cuda" and kwargs.get("device") is None
                and func in _tdev._device_constructors()):
            idx = dev.index if dev.index is not None else int(getattr(torch.cuda, "current_device", lambda: 0)())
            cpu_kwargs = dict(kwargs); cpu_kwargs["device"] = torch.device("cpu")
            res = func(*args, **cpu_kwargs)
            return wrap(res, idx)
        return orig(self, func, types, args, kwargs)

    cls.__torch_function__ = _torch_function
    cls._gm_device_fixed = True


def _fix_visible_devices(torch) -> None:
    """Honour CUDA_VISIBLE_DEVICES. Real CUDA restricts the visible set to the
    listed indices; an empty value means no CUDA; an invalid entry truncates the
    list at that point. Training launchers (torchrun) set this per rank, so this
    is a correctness requirement as well as a tell. We compute the effective
    count once and override device_count / is_available."""
    from daemon_client import visible_physical
    effective = len(visible_physical(os.environ.get("CUDA_VISIBLE_DEVICES"), DEVICE_COUNT))

    def device_count():
        return effective

    def is_available():
        return effective > 0

    torch.cuda.device_count = device_count
    torch.cuda.is_available = is_available
    try:
        torch.cuda._device_count = effective
        torch.cuda._cached_device_count = effective
    except Exception:
        pass


def _fix_stream_repr() -> None:
    """Stream/Event repr must not name _FakeStream."""
    try:
        from nvcompat import _upstream as up
    except Exception:
        return
    for cls_name, real_name, mod in (("_FakeStream", "Stream", "torch.cuda"),
                                     ("_FakeEvent", "Event", "torch.cuda")):
        cls = getattr(up, cls_name, None)
        if cls is None:
            continue
        try:
            cls.__name__ = real_name
            cls.__qualname__ = real_name
            cls.__module__ = mod
        except (AttributeError, TypeError):
            pass


def _fix_nvtx(torch) -> None:
    """NVTX range functions exist on a real CUDA build. FakeGPU leaves torch's
    stubs that raise 'NVTX functions not installed'. Replace with no-ops."""
    try:
        nvtx = torch.cuda.nvtx
    except AttributeError:
        return
    nvtx.range_push = lambda msg: 0
    nvtx.range_pop = lambda: 0
    nvtx.mark = lambda msg: None
    try:
        nvtx.range_start = lambda msg: 0
        nvtx.range_end = lambda rid: None
    except Exception:
        pass


def _fix_arch_list(torch) -> None:
    arches = list(ARCH_LIST)
    torch.cuda.get_arch_list = lambda: arches
    # gencode flags should match the arch list
    flags = " ".join(f"-gencode compute=compute_{a.split('_')[1]},"
                     f"code={a}" for a in arches)
    torch.cuda.get_gencode_flags = lambda: flags


def _fix_versions(torch) -> None:
    torch.version.cuda = CUDA_VERSION
    torch.version.hip = None
    cudnn = torch.backends.cudnn
    cudnn.version = lambda: CUDNN_VERSION
    try:
        cudnn.is_available = lambda: True
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tensor class: fix the type name AND the repr. A real cuda tensor's repr ends
# with `device='cuda:N'`; FakeCudaTensor inherits torch.Tensor.__repr__, which
# reads the CPU storage and omits the device. We give the class a __repr__ that
# formats the underlying data and appends the device, matching PyTorch.
# ---------------------------------------------------------------------------

def _fix_tensor_class(torch) -> None:
    try:
        from nvcompat import _upstream as up
    except Exception:
        up = None
    cls = None
    if up is not None:
        cls = getattr(up, "FakeCudaTensor", None)
    if cls is None:
        x = torch.empty(0, device="cuda:0")
        cls = type(x)
    if cls is torch.Tensor:
        return

    # name
    try:
        cls.__name__ = "Tensor"
        cls.__qualname__ = "Tensor"
        cls.__module__ = "torch"
    except (AttributeError, TypeError):
        pass

    # repr: render the real data via the CPU tensor, then append the device the
    # way PyTorch does. PyTorch prints `tensor([...], device='cuda:0')` and, for
    # non-default dtypes, a dtype= field; torch's own formatter already adds the
    # dtype, so we only need to inject device= before the closing paren.
    def _nvrt_repr(self):
        raw = getattr(self, "raw_data", None)
        base = torch.Tensor.__repr__(raw if raw is not None else self)
        dev = f"cuda:{getattr(self, 'device_index', 0)}"
        if "device=" in base:
            return base
        # base ends with ")"; insert device= before it, after any existing field
        inner = base.rstrip()
        assert inner.endswith(")")
        return inner[:-1].rstrip() + f", device='{dev}')"

    try:
        cls.__repr__ = _nvrt_repr
        cls.__str__ = _nvrt_repr
    except (AttributeError, TypeError):
        pass


# ---------------------------------------------------------------------------
# Properties repr: _FakeDeviceProperties.__repr__ names the fake class. Rename
# it and make its repr look like torch._C._CudaDeviceProperties.
# ---------------------------------------------------------------------------

def _fix_properties_class() -> None:
    try:
        from nvcompat import torch_patch as fg
    except Exception:
        return
    cls = getattr(fg, "_FakeDeviceProperties", None)
    if cls is None:
        return
    try:
        cls.__name__ = "_CudaDeviceProperties"
        cls.__qualname__ = "_CudaDeviceProperties"
        cls.__module__ = "torch._C"
    except (AttributeError, TypeError):
        pass

    def _props_repr(self):
        return (f"_CudaDeviceProperties(name='{self.name}', "
                f"major={self.major}, minor={self.minor}, "
                f"total_memory={self.total_memory // (1024**2)}MB, "
                f"multi_processor_count={self.multi_processor_count})")

    try:
        cls.__repr__ = _props_repr
    except (AttributeError, TypeError):
        pass


# ---------------------------------------------------------------------------
# NVML surface: torch.cuda.utilization/temperature/power_draw/clock_rate call
# pynvml and raise when it is absent. On a real node they return numbers. We
# serve plausible numbers; inside the container a later revision can read the
# mock NVML via the daemon snapshot so these match nvidia-smi exactly.
# ---------------------------------------------------------------------------

def _fix_nvml_surface(torch) -> None:
    # idle-ish defaults; the daemon overrides when a job is pacing (see
    # TimingPacer.publish_nvml). Values are read from env so they can be aligned
    # with the bridge's current snapshot.
    def _dev_index(device):
        # validate like real torch: reject out-of-range / non-cuda devices.
        if device is None:
            return 0
        if isinstance(device, torch.device):
            if device.type != "cuda":
                raise RuntimeError(
                    f"Invalid device, must be cuda device")
            idx = device.index if device.index is not None else 0
        else:
            idx = int(device)
        n = torch.cuda.device_count()
        if idx < 0 or idx >= n:
            raise RuntimeError(
                f"Invalid device id: {idx}. Device count is {n}.")
        return idx

    def _util(device=None):
        _dev_index(device)
        return int(os.environ.get("NVRT_UTIL", "0"))

    def _temp(device=None):
        _dev_index(device)
        return int(os.environ.get("NVRT_TEMP", "34"))

    def _power(device=None):
        _dev_index(device)
        # torch.cuda.power_draw returns milliwatts (matches NVML), not watts.
        return int(float(os.environ.get("NVRT_POWER_W", "95")) * 1000)

    def _clock(device=None):
        _dev_index(device)
        return int(os.environ.get("NVRT_CLOCK_MHZ", "1980"))  # MHz

    torch.cuda.utilization = _util
    torch.cuda.temperature = _temp
    torch.cuda.power_draw = _power
    torch.cuda.clock_rate = _clock


# ---------------------------------------------------------------------------
# Autocast: FakeGPU redirects cuda autocast to CPU, and CPU autocast does not
# cast the same ops, so a bf16 cuda-autocast matmul comes back float32. We wrap
# it so that under a requested cuda autocast with bf16/fp16, the eligible ops
# produce the reduced dtype, matching a real GPU. The simplest correct behaviour
# is to run CPU autocast with the SAME dtype (CPU autocast supports bf16), which
# does cast matmul/linear outputs.
# ---------------------------------------------------------------------------

def _make_relabel_mode(torch):
    """A TorchDispatchMode that recasts an op's bf16 result to a target dtype
    (fp16). Used only on torch too old for CPU fp16 autocast: the compute runs
    in bf16 and the eligible outputs are relabelled to fp16, so the dtype a
    script observes matches a real GPU. Narrow: only mm/addmm/bmm/conv/linear,
    the ops autocast would have produced the reduced dtype for."""
    from torch.utils._python_dispatch import TorchDispatchMode
    _TARGETS = {torch.ops.aten.mm, torch.ops.aten.addmm, torch.ops.aten.bmm,
                torch.ops.aten.baddbmm, torch.ops.aten.matmul,
                torch.ops.aten.convolution, torch.ops.aten.linear}

    class _RelabelDtypeMode(TorchDispatchMode):
        def __init__(self, target_dtype):
            super().__init__()
            self.target = target_dtype

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            packet = getattr(func, "_overloadpacket", None)
            if packet in _TARGETS and isinstance(out, torch.Tensor) \
                    and out.dtype == torch.bfloat16:
                return out.to(self.target)
            return out

    return _RelabelDtypeMode


def _fix_autocast(torch) -> None:
    global _RelabelDtypeMode
    _RelabelDtypeMode = _make_relabel_mode(torch)
    # FakeGPU replaced torch.amp.autocast with a class that enables *cuda*
    # autocast and redirects tensor storage to CPU. The CPU ops run inside
    # __torch_function__ then do not observe cuda autocast, so eligible ops are
    # not cast and a bf16 cuda-autocast matmul comes back float32 — a tell.
    #
    # We replace it with a context manager that enables real *CPU* autocast with
    # the requested dtype where CPU autocast supports it. Newer torch (2.4+) casts
    # matmul/linear to exactly bf16 OR fp16 under CPU autocast, so explicit fp16
    # yields fp16 with real overflow. Older torch (e.g. 2.1.2, the TRL battery's
    # pin) only supports bf16 for CPU autocast and RAISES on fp16, so there we
    # run bf16 internally and relabel the eligible outputs to the requested dtype
    # (a dispatch wrapper), which reproduces the dtype a real GPU would report.
    # The CUDA default (no dtype given) is fp16, as on real hardware.
    _real_autocast = torch.amp.autocast_mode.autocast
    _CUDA_DEFAULT = torch.float16   # real torch.autocast('cuda') default

    def _cpu_supports(dtype) -> bool:
        """Whether CPU autocast runs in `dtype`. torch 2.1 does not raise for an
        unsupported dtype: it warns ("In CPU autocast, but the target dtype is not
        supported. Disabling autocast.") and turns autocast off. The first version
        took the absence of an exception as support, and the warning reached the
        stderr of every process that imported torch (2026-10-20). Any warning
        here counts as unsupported, and none escapes."""
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with _real_autocast("cpu", dtype=dtype):
                    pass
            return not caught
        except (RuntimeError, ValueError):
            return False

    _CPU_FP16_OK = _cpu_supports(torch.float16)

    class _Autocast:
        def __init__(self, device_type="cuda", dtype=None, enabled=True,
                     cache_enabled=None):
            self.req_device = device_type
            if dtype is None:
                dtype = _CUDA_DEFAULT if device_type == "cuda" else torch.bfloat16
            self.req_dtype = dtype
            self.enabled = enabled
            # fp16 requested but CPU autocast can't do it: run bf16, relabel after
            self._relabel = (dtype == torch.float16 and not _CPU_FP16_OK
                             and device_type == "cuda")
            cpu_dtype = torch.bfloat16 if self._relabel else dtype
            self._cpu_ctx = _real_autocast(
                "cpu", dtype=cpu_dtype, enabled=enabled,
                cache_enabled=cache_enabled)
            self._relabel_mode = None

        def __enter__(self):
            self._cpu_ctx.__enter__()
            if self._relabel:
                self._relabel_mode = _RelabelDtypeMode(torch.float16)
                self._relabel_mode.__enter__()
            return self

        def __exit__(self, *exc):
            if self._relabel_mode is not None:
                self._relabel_mode.__exit__(*exc)
                self._relabel_mode = None
            return self._cpu_ctx.__exit__(*exc)

        def __call__(self, func):
            import functools

            @functools.wraps(func)
            def wrapper(*a, **k):
                with self.__class__(self.req_device, self.req_dtype,
                                    self.enabled):
                    return func(*a, **k)
            return wrapper

    torch.amp.autocast = _Autocast
    torch.autocast = _Autocast                      # the binding most code uses
    torch.amp.autocast_mode.autocast = _Autocast
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        torch.cuda.amp.autocast = _Autocast


# ---------------------------------------------------------------------------
# OOM text: PyTorch's exact wording.
# ---------------------------------------------------------------------------

def _fix_oom_text(torch) -> None:
    """Rewrite the OOM message to PyTorch's exact wording WITHOUT replacing the
    exception class (which would break `torch.cuda.OutOfMemoryError is
    torch.OutOfMemoryError` and any `except torch.cuda.OutOfMemoryError`). We
    patch FakeGPU's allocator so it raises the genuine class with our message."""
    try:
        from nvcompat import _allocator as alloc
    except Exception:
        return
    # The allocator builds the message inline at its raise site. Wrap the method
    # that raises so we rewrite the string but keep torch.cuda.OutOfMemoryError.
    # FakeGPU's allocator raises in its register/allocate path; we wrap the whole
    # module's reference to OutOfMemoryError so its `raise` uses a factory that
    # rewrites the text while returning a genuine OutOfMemoryError instance.
    _real = torch.cuda.OutOfMemoryError

    def _factory(*args):
        if args and isinstance(args[0], str) and "FakeGPU has" in args[0]:
            return _real(_rewrite_oom(args[0]))
        return _real(*args)

    # alloc references torch.cuda.OutOfMemoryError via `import torch`; redirect
    # the attribute it reads to our factory. The class identity on torch.cuda is
    # left untouched, so the alias and isinstance checks still hold.
    try:
        alloc._nvrt_oom_factory = _factory
        _patch_allocator_raise(alloc, _factory)
    except Exception:
        pass


def _patch_allocator_raise(alloc, factory) -> None:
    """Wrap the allocator's allocate() so an OOM it raises carries PyTorch's exact
    message. The class identity on torch.cuda is left untouched, so the alias and
    isinstance checks still hold; only the string is rewritten."""
    import functools
    tracker = getattr(alloc, "_DeviceMemoryTracker", None)
    if tracker is None:
        return
    for name in ("allocate", "allocate_saved_tensor"):
        fn = getattr(tracker, name, None)
        if fn is None:
            continue

        @functools.wraps(fn)
        def wrapper(self, *a, __fn=fn, **k):
            try:
                return __fn(self, *a, **k)
            except Exception as e:
                msg = str(e)
                if "FakeGPU has" in msg:
                    raise factory(msg) from None
                raise
        try:
            setattr(tracker, name, wrapper)
        except Exception:
            pass


def _rewrite_oom(text: str) -> str:
    import re
    m = re.search(r"allocate ([\d.]+) MiB.*?GPU (\d+).*?capacity of "
                  r"([\d.]+) GiB of which ([\d.]+) GiB is free.*?FakeGPU has "
                  r"([\d.]+) GiB allocated and ([\d.]+) GiB reserved",
                  text, re.S)
    if not m:
        return text
    tried, gpu, cap, free, alloc_gib, resv_gib = m.groups()
    resv_unalloc_mib = max(0.0, (float(resv_gib) - float(alloc_gib))) * 1024
    return (f"CUDA out of memory. Tried to allocate {tried} MiB. "
            f"GPU {gpu} has a total capacity of {cap} GiB of which "
            f"{free} GiB is free. Of the allocated memory {alloc_gib} GiB is "
            f"allocated by PyTorch, and {resv_unalloc_mib:.2f} MiB is reserved "
            f"by PyTorch but unallocated. If reserved but unallocated memory is "
            f"large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"to avoid fragmentation. See documentation for Memory Management "
            f"(https://pytorch.org/docs/stable/notes/cuda.html"
            f"#environment-variables)")


# ---------------------------------------------------------------------------
# Event timing: real monotonic ms standalone; daemon-paced when a client is set.
# ---------------------------------------------------------------------------

def _wire_event_timing(torch) -> None:
    import time
    Event = torch.cuda.Event
    _attr = "_nvrt_stamp"
    _orig_record = Event.record

    def record(self, *a, **k):
        setattr(self, _attr, time.monotonic())
        return _orig_record(self, *a, **k)

    def elapsed_time(self, end_event):
        t0 = getattr(self, _attr, None)
        t1 = getattr(end_event, _attr, None)
        if t0 is None or t1 is None:
            return 0.0
        return max(0.0, (t1 - t0) * 1000.0)

    try:
        Event.record = record
        Event.elapsed_time = elapsed_time
    except (AttributeError, TypeError):
        pass


apply()
