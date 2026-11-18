"""Runtime fingerprint extractor: compute a live job's structural fingerprint
from what the patched torch observes, using the same canonicalisation the
registry used (fingerprint.py), so a launched script hashes to its record.

What is observed, and from where (REPLAY-INTERFACE §2; conventions pinned in
docs/FINGERPRINT-CANONICALISATION.md):

  model           first depth-0 nn.Module call: root.config.model_type if the
                  root has an HF config, else type(root).__name__.lower()
  num_layers /    from root.config (n_layer|num_hidden_layers, n_embd|hidden_size,
  model_dim /     vocab_size) when present; else derived from the first
  vocab_size      nn.Embedding (num_embeddings, embedding_dim) and the longest
                  nn.ModuleList (len)
  micro_batch     the first positional tensor's shape at that first root call
  optimizers      every torch.optim.Optimizer subclass constructed (class names)
  accumulation    Tensor.backward() calls between optimizer steps (count at the
                  first optimizer.step())
  world_size      torch.distributed.get_world_size() if initialised, else 1
  global batch    micro_batch[0] × accumulation × world_size (sequences), and
                  × micro_batch[1] when the input is a 2-D integer token tensor
                  (tokens) — both emitted; canonicalisation picks per its version
  precision       the autocast dtype entered before the first optimizer.step()
                  ("bf16"/"fp16"), else "fp32" — the mixed-precision mode, never
                  the parameter dtype
  torch_compile   whether torch.compile() was called on an nn.Module

The fingerprint is final at the FIRST optimizer.step(), which is also the
earliest moment the global batch is known (accumulation needs the backward
count). That is the registry lookup point (REPLAY-INTERFACE §2 amendment).

Dry-run mode (NVRT_FINGERPRINT_ONLY=1): every depth-0 forward is short-
circuited to a dummy 0-d loss (no compute), backward() only counts, and at the
first optimizer.step() the inputs + hash are printed as JSON and the process
exits. A script that materialises a value before its first forward cannot be
dry-run and is reported as such.

Caveat (recorded): micro_batch is read from the FIRST root forward. If a script
runs an eval forward before training (track 3 does, at step 0) the eval
micro-batch is used; for track 3 it equals the training mbs. A script whose
eval batch differs from its training batch needs the training-batch rule; log it.
"""

from __future__ import annotations

import json
import os
import sys
import threading

import torch
import torch.nn as nn

import fingerprint as _fp

DRY_RUN = os.environ.get("NVRT_FINGERPRINT_ONLY") == "1"

_state = {
    "root_seen": False,
    "root_grad": False,     # the root was chosen at a grad-enabled forward
    "model": None, "num_layers": None, "model_dim": None, "vocab_size": None,
    "micro_batch_shape": None, "input_is_tokens": False,
    "optimizers": set(),
    "backward_count": 0,
    "autocast_dtype": None,
    "torch_compile": False,
    "dataloader_batch_size": None,   # first DataLoader's batch_size, if any
    "root": None,           # the root nn.Module seen at the first forward
    "final": None,          # dict once finalised
    "materialised_before_forward": False,
}
_depth = threading.local()
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _first_tensor(args, kwargs):
    for a in list(args) + list((kwargs or {}).values()):
        if isinstance(a, torch.Tensor):
            return a
    return None


def _unwrap(m: nn.Module) -> nn.Module:
    """The model under its wrappers: DistributedDataParallel / DataParallel /
    FSDP (.module), torch.compile's OptimizedModule (._orig_mod) and a
    PeftModel (get_base_model()). The fingerprint describes the architecture
    the record was built from, so the parameter count (field 11) is the base
    model's, adapters excluded, and the model family is not "distributeddataparallel"."""
    for _ in range(8):
        if hasattr(m, "get_base_model") and callable(m.get_base_model):   # peft
            try:
                m = m.get_base_model()
                continue
            except Exception:
                pass
        # DDP, DataParallel, FSDP, and FakeGPU's _FakeDistributedDataParallel,
        # which accelerate builds under the patch: matching only the exact torch
        # names left the family "_fakedistributeddataparallel" and no config,
        # so no multi-rank job could ever match a record structurally
        if "DataParallel" in type(m).__name__ and isinstance(getattr(m, "module", None), nn.Module):
            m = m.module
            continue
        if type(m).__name__ == "OptimizedModule" and isinstance(getattr(m, "_orig_mod", None), nn.Module):
            m = m._orig_mod
            continue
        break
    return m


def _adapter_markers(m: nn.Module) -> list:
    """Name fragments of the parameters peft injects INTO the base model (LoRA
    puts lora_A/lora_B inside the wrapped layers, so the base model's
    parameters() include them) and of modules_to_save copies. Read from the
    tuner's own `prefix` (LoraModel.prefix == "lora_"), searched through any
    DDP / compile wrapper around the PeftModel."""
    marks = []
    for _ in range(8):
        if hasattr(m, "get_base_model") and callable(getattr(m, "get_base_model")):
            pre = getattr(getattr(m, "base_model", None), "prefix", None)
            if isinstance(pre, str) and pre:
                marks.append(pre)
            marks.append("modules_to_save.")
            break
        nxt = None
        for attr in ("module", "_orig_mod"):
            cand = m.__dict__.get("_modules", {}).get(attr) or m.__dict__.get(attr)
            if isinstance(cand, nn.Module):
                nxt = cand
                break
        if nxt is None:
            break
        m = nxt
    return marks


def _describe_root(root: nn.Module) -> None:
    marks = _adapter_markers(root)
    root = _unwrap(root)
    cfg = getattr(root, "config", None)
    model = None
    if cfg is not None:
        model = getattr(cfg, "model_type", None)
        _state["num_layers"] = getattr(cfg, "n_layer", None) or getattr(cfg, "num_hidden_layers", None)
        _state["model_dim"] = getattr(cfg, "n_embd", None) or getattr(cfg, "hidden_size", None)
        _state["vocab_size"] = getattr(cfg, "vocab_size", None)
    if not model:
        model = type(root).__name__.lower()
    _state["model"] = model
    try:
        _state["num_parameters"] = sum(int(p.numel()) for n, p in root.named_parameters()
                                       if not any(mk in n for mk in marks))
    except Exception:
        _state["num_parameters"] = None
    # derive what the config did not give
    if _state["vocab_size"] is None or _state["model_dim"] is None:
        for m in root.modules():
            if isinstance(m, nn.Embedding):
                _state["vocab_size"] = _state["vocab_size"] or m.num_embeddings
                _state["model_dim"] = _state["model_dim"] or m.embedding_dim
                break
    if _state["num_layers"] is None:
        longest = 0
        for m in root.modules():
            if isinstance(m, nn.ModuleList) and len(m) > longest:
                longest = len(m)
        _state["num_layers"] = longest or None


def _precision_str() -> str:
    d = _state["autocast_dtype"]
    if d == torch.bfloat16:
        return "bf16"
    if d == torch.float16:
        return "fp16"
    return "fp32"


def _world_size() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
    except Exception:
        pass
    return 1


def _rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


def inputs() -> dict:
    """The fingerprint_inputs dict, in the registry's field names."""
    mb = _state["micro_batch_shape"]
    accum = max(1, _state["backward_count"])
    ws = _world_size()
    # field 6 (fp256/0.2): per-device micro-batch in SEQUENCES. The DataLoader's
    # batch_size is the authority when the script built one (HF pads T per
    # batch and DPO's forward sees 2×B, so a forward shape is not a stable key);
    # else the first forward's leading dimension (track 3 has no DataLoader).
    micro = _state["dataloader_batch_size"]
    if micro is None and mb:
        micro = int(mb[0])
    d = {
        "model": _state["model"],
        "num_layers": _state["num_layers"],
        "model_dim": _state["model_dim"],
        "vocab_size": _state["vocab_size"],
        "optimizers": sorted(_state["optimizers"]),
        "micro_batch": micro,
        "micro_batch_shape": list(mb) if mb else None,     # informational
        "micro_batch_source": ("dataloader" if _state["dataloader_batch_size"]
                               is not None else "forward"),
        "world_size": ws,
        "accumulation": accum,
        "precision": _precision_str(),
        "torch_compile": bool(_state["torch_compile"]),
        "num_parameters": _state.get("num_parameters"),
    }
    if micro is not None:
        d["global_batch_sequences"] = int(micro) * accum * ws
    return d


def get() -> dict | None:
    return _state["final"]


# ---------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------

def _finalise() -> dict:
    inp = inputs()
    fp = _fp.fingerprint(inp)
    final = {"fingerprint": fp, "version": _fp.VERSION, "inputs": inp,
             "rank": _rank(), "materialised_before_forward":
             _state["materialised_before_forward"]}
    _state["final"] = final
    return final


def _replay_active() -> bool:
    try:
        import replay
        return replay.is_active()
    except Exception:
        return False


def _try_replay(stage: str) -> None:
    """Hand the identity to the replay client; never let a failure here break
    the job (it then runs unreplayed, which is logged)."""
    if os.environ.get("NVRT_NO_REPLAY") == "1":
        return
    try:
        import replay
        if stage == "exact":
            replay.try_lookup("exact")
        else:
            final = _state["final"] or {}
            replay.try_lookup("structural", fingerprint=final.get("fingerprint"),
                              fingerprint_inputs=final.get("inputs"))
    except Exception as e:
        if os.environ.get("NVRT_DEBUG"):
            sys.stderr.write(f"nvrt: replay lookup[{stage}] failed: {e!r}\n")


_optimizer_instances: list = []


def _on_first_step() -> None:
    with _lock:
        first = _state["final"] is None
        if first:
            final = _finalise()
    if DRY_RUN:
        if first:
            line = json.dumps(final, separators=(",", ":"))
            # every rank prints (tagged by rank) so a multi-rank dry run is checkable
            sys.stdout.write("NVRT_FINGERPRINT " + line + "\n")
            sys.stdout.flush()
            os._exit(0)
        return
    if first:
        # STRUCTURAL lookup at the first step: the fingerprint is complete now.
        _try_replay("structural")


def _install_module_hook() -> None:
    orig_call = nn.Module.__call__

    def __call__(self, *args, **kwargs):
        depth = getattr(_depth, "d", 0)
        _depth.d = depth + 1
        try:
            if depth == 0:
                # The root is the first depth-0 module called with grad
                # enabled: the one being trained. A no-grad forward that comes
                # first (DPO's reference model when it precomputes log-probs, an
                # eval pass, a shape probe) sets a provisional root, replaced by
                # the first grad-enabled one, so the fingerprint and micro-batch
                # describe the trained model.
                grad = torch.is_grad_enabled()
                if not _state["root_seen"] or (grad and not _state["root_grad"]):
                    _state["root_seen"] = True
                    _state["root_grad"] = grad
                    _state["root"] = self
                    t = _first_tensor(args, kwargs)
                    if t is not None:
                        _state["micro_batch_shape"] = tuple(int(s) for s in t.shape)
                        _state["input_is_tokens"] = (not t.is_floating_point()
                                                     and t.dim() == 2)
                    _describe_root(self)
                    # EXACT lookup at the first forward: the entry hash + argv
                    # are known now, so an exact_only record engages replay
                    # before any forward computes (REPLAY-INTERFACE §2 fast path).
                    if not DRY_RUN and not _state.get("exact_tried"):
                        _state["exact_tried"] = True
                        _try_replay("exact")
                if DRY_RUN:
                    # short-circuit: no compute, a dummy differentiable loss on
                    # the input's device so val loops and backward() still flow.
                    t = _first_tensor(args, kwargs)
                    dev = t.device if t is not None else "cpu"
                    return torch.zeros((), device=dev, requires_grad=True)
                if _replay_active():
                    import replay
                    return replay.forward_stub(self, args, kwargs)
            return orig_call(self, *args, **kwargs)
        finally:
            _depth.d = depth

    nn.Module.__call__ = __call__


def _install_optimizer_hooks() -> None:
    """Record every Optimizer subclass constructed, and finalise the fingerprint
    at the first optimizer step.

    Wrapping the base class's Optimizer.step would NOT fire for real optimizers:
    torch's AdamW, SGD and a script's custom class (track 3's AdamH) each define
    their own step(), which shadows the base method. So we use torch's official
    per-instance hooks, registered in __init__: the pre-hook finalises at the
    first step on any optimizer, the post-hook resets the backward count so the
    next step's accumulation is counted afresh."""
    from torch.optim import Optimizer
    orig_init = Optimizer.__init__

    def _pre(opt, args, kwargs):
        _on_first_step()           # finalise at the FIRST step (dry run exits)

    def _post(opt, args, kwargs):
        _state["backward_count"] = 0

    def __init__(self, *a, **k):
        _state["optimizers"].add(type(self).__name__)
        orig_init(self, *a, **k)
        _optimizer_instances.append(self)
        # an optimizer constructed after replay engaged is neutralised too,
        # and may mark a trial boundary (replay.on_new_optimizer)
        if _replay_active():
            try:
                import replay
                replay._neutralise(self)
                replay.on_new_optimizer(self)
            except Exception:
                pass
        try:
            self.register_step_pre_hook(_pre)
            self.register_step_post_hook(_post)
        except Exception:
            # very old torch without step hooks: fall back to instance wrap
            _orig = self.step

            def step(*sa, **sk):
                _on_first_step()
                try:
                    return _orig(*sa, **sk)
                finally:
                    _state["backward_count"] = 0
            self.step = step

    Optimizer.__init__ = __init__


def _install_backward_hook() -> None:
    orig_backward = torch.Tensor.backward

    def backward(self, *a, **k):
        _state["backward_count"] += 1
        if _replay_active() and not DRY_RUN:
            import replay
            replay.backward_stub(_state.get("root"))
            return None
        if DRY_RUN:
            # No compute, but a backward must still leave every trainable
            # parameter's .grad PRESENT with the right shape: scripts assert it
            # (track 3: `assert p.grad is not None, name`) and all-reduce it
            # before optimizer.step(). This is the "short-circuited backward
            # leaves .grad present" requirement from REPLAY-INTERFACE §3,
            # verified here for the first time. Zero grads of the param's shape
            # and dtype; the first optimizer.step() then finalises and exits.
            root = _state.get("root")
            if root is not None:
                for p in root.parameters():
                    if p.requires_grad and p.grad is None:
                        try:
                            p.grad = torch.zeros_like(p)
                        except Exception:
                            pass
            return None
        return orig_backward(self, *a, **k)

    torch.Tensor.backward = backward


def _install_autocast_spy() -> None:
    """Record the autocast dtype a script enters. cuda_hooks rebinds the
    autocast names to its _Autocast class (which stores req_dtype); we wrap
    whatever class is bound now."""
    names = [("torch", "autocast"), ("torch.amp", "autocast"),
             ("torch.amp.autocast_mode", "autocast"), ("torch.cuda.amp", "autocast")]
    import importlib
    seen = {}
    for modname, attr in names:
        try:
            mod = importlib.import_module(modname)
            cls = getattr(mod, attr, None)
        except Exception:
            continue
        if cls is None or not isinstance(cls, type):
            continue
        if cls in seen:
            setattr(mod, attr, seen[cls])
            continue

        class _Spy(cls):  # type: ignore[misc,valid-type]
            def __init__(s, *a, **k):
                super().__init__(*a, **k)
                d = getattr(s, "req_dtype", None) or getattr(s, "fast_dtype", None)
                if d is not None and _state["final"] is None:
                    _state["autocast_dtype"] = d

        _Spy.__name__ = cls.__name__
        _Spy.__qualname__ = cls.__qualname__
        seen[cls] = _Spy
        setattr(mod, attr, _Spy)


def _install_compile_spy() -> None:
    orig = getattr(torch, "compile", None)
    if orig is None:
        return

    def compile(model=None, *a, **k):
        target = model if model is not None else k.get("model")
        if isinstance(target, nn.Module):
            _state["torch_compile"] = True
        return orig(model, *a, **k) if model is not None else orig(*a, **k)

    torch.compile = compile


def _install_dataloader_hook() -> None:
    """Capture the first DataLoader's batch_size: the authority for field 6
    (per-device micro-batch in sequences) when a script builds a loader."""
    try:
        from torch.utils.data import DataLoader
    except Exception:
        return
    orig_init = DataLoader.__init__

    def __init__(self, dataset, batch_size=1, *a, **k):
        bs = k.get("batch_size", batch_size)
        if _state["dataloader_batch_size"] is None and bs is not None \
                and _state["final"] is None:
            try:
                _state["dataloader_batch_size"] = int(bs)
            except (TypeError, ValueError):
                pass
        return orig_init(self, dataset, batch_size, *a, **k)

    DataLoader.__init__ = __init__


_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    _install_module_hook()
    _install_optimizer_hooks()
    _install_backward_hook()
    _install_autocast_spy()
    _install_compile_spy()
    _install_dataloader_hook()
    _installed = True
