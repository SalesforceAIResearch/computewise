"""Replay: a job matched to a recorded run is served the record's values and
paced by the record's timing, with no real compute (REPLAY-INTERFACE §3–§5).

How a replayed job runs, end to end:

  lookup      At the first root forward the client sends the EXACT key
              (sha256 of the entry script + argv) to the daemon; for an
              `exact_only` record that is enough and replay engages before the
              first forward ever computes anything. A structural match (by
              fingerprint, `structural_verified` records) is tried at the first
              optimizer.step(), when the fingerprint is complete.
  forward     Every depth-0 forward is short-circuited to a 0-d tensor tagged
              with a ROLE by provenance: `train_loss` in training mode,
              `val_loss` in eval/no_grad mode. The tag propagates through the
              script's own arithmetic (+=, /=, all_reduce) because it rides on
              the tensor and FakeCudaTensor.__torch_function__ copies it to
              results.
  read        When the script MATERIALISES a tagged tensor — f-string format,
              .item(), float(), bool(), .tolist(), repr — the value is served
              from the record by (role, step), not computed. Every rank asks
              the daemon and gets the same answer.
  backward    Leaves cached zero `.grad`s present on every trainable parameter
              (scripts assert and all-reduce them), tagged as dummies so the
              collectives skip the data movement.
  step        optimizer.step() does no arithmetic. It advances the job's clock
              by the record's step time (first_step_s for step 1, steady after),
              and paces WALL time through the daemon's per-GPU cursor
              (enqueue + sync, divided by the speed knob). Validation phases
              are paced by the record's per-validation eval time.
  failures    A recorded failure fires at its step (daemon `fail?`).

What stays real: data loading (the loader really reads files), the script's
own Python, logging, checkpoints. What is never run: the model's math.

Memory: parameters still live as real CPU tensors (fine at 124 M; meta backing
for multi-B models is a later unit). Zero grads are cached once per parameter
and reused every step, so there is no per-step allocation churn.
"""

from __future__ import annotations

import functools
import hashlib
import os
import types
import sys
import threading

import torch
import torch.nn as nn

import lr_verify
import virtual_clock as vc
from daemon_client import DaemonClient, physical_gpu_index
from daemon_client import job_key as _job_key

# NVRT_PROBE_STEPS=N: end a replayed job after its N-th optimizer step with
# a marker line on stderr (the all-scripts probe in gate 1). 0 = off.
_PROBE_STEPS = int(os.environ.get("NVRT_PROBE_STEPS", "0") or 0)

_ROLE = "_gm_role"       # attribute on tensors: train_loss | val_loss
_DUMMY = "_gm_dummy"     # attribute on tensors: a zero grad with no information

_REAL_SLEEP = vc._REAL["sleep"]


def _gpu_wait(seconds: float) -> None:
    """Wait for paced GPU work the way a CUDA host thread does: busy on a
    core, not asleep. CUDA's default schedule (cudaDeviceScheduleAuto) spins
    while there are fewer active contexts than cores, so a real job waiting
    on its GPU shows about one core and state R; a plain sleep showed 0% CPU
    and hrtimer_nanosleep in wchan, which agents read as a hung job
    (G-kvreph-opus-L0 and G-kvrep-haiku-L3; the Grok canary killed two
    evaluations for it, 2026-10-26). The spin yields every pass, which lets
    the job's other threads run at their full rate (a bare spin halved a
    worker thread's progress), and it reads the genuine monotonic clock, the
    one the daemon's cursor runs on. NVRT_WAIT=sleep restores the old
    sleeping wait."""
    if seconds <= 0:
        return
    if os.environ.get("NVRT_WAIT") == "sleep":
        _REAL_SLEEP(seconds)
        return
    mono = vc._REAL_MONO
    end = mono() + seconds
    while mono() < end:
        os.sched_yield()

_s = {
    "client": None,          # DaemonClient
    "active": False,         # replay engaged
    "serving": None,         # serving info from the daemon
    "record_id": None,
    "mode": None,
    "match": None,
    "step": 0,               # optimizer steps completed
    "in_eval_phase": False,
    "grad_cache": {},        # id(param) -> zero grad tensor
    "lookup_done_exact": False,
    "lookup_done_structural": False,
    "entry_hash": None,
    "entry_path": None,   # resolved once by _entry_path; "" = not found
    "log": [],
    "neutralised": [],       # optimizers whose step() replay replaced
    "disengaged": None,      # reason, once replay has been withdrawn
    "trial": 1,              # in-process trial (see on_new_optimizer)
    "stepped_param_ids": set(),   # ids of the params of every optimizer that stepped this trial
    "param_digest": {},      # id(param) -> digest of its contents at the trial's first steps
    "digested_opts": set(),  # optimizers whose params the digest already covers this trial
    "skip_startup_once": False,   # the next step 1 pays no start-up (in-place re-init, no recompile)
    "whole_job_probed": False,    # the early whole_job probe ran (once, at import torch)
    "pe_gen_index": 0,       # per_example serving (MLS-Bench KV): examples generated so far
    "pe_score_index": 0,     # ... and scored so far
    "pe_t_engaged": None,    # real monotonic time at engagement (the model load starts there)
    "pe_load_s": None,       # the launch's start-up on the box's clock (what the script's timer saw), measured at the first generate-seam call
    "pe_load_real_s": None,  # the same start-up in real seconds
    "pe_last_drain": None,   # real monotonic time the previous example unit drained (host gap measurement)
}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

def _launch_cwd() -> str:
    """The directory the process was launched in: read by sitecustomize before
    any script code ran, else the current one."""
    try:
        import sitecustomize as _sc
        if getattr(_sc, "LAUNCH_CWD", None):
            return _sc.LAUNCH_CWD
    except Exception:
        pass
    return os.getcwd()


def _entry_path() -> str | None:
    """The entry script's absolute path, resolved ONCE and independently of
    the current directory. sys.argv[0] is usually relative to the launch
    directory and a script may chdir before its first CUDA touch (H4's
    recipes chdir to their own directory before their data check); reading
    it against os.getcwd() then fails, and until 2026-10-24 that returned no
    entry hash and no script path, so a path_bound record whose script was
    byte-identical missed on every stage (W0-h4-measure: 21 of 21 lookups
    `match none`, each job then OOM'd on the model path). Prefer
    __main__.__file__, which Python 3.9+ makes absolute; else argv[0] joined
    to the launch directory; else argv[0] as given."""
    if _s.get("entry_path") is not None:
        return _s["entry_path"] or None
    cands: list[str] = []
    try:
        mf = getattr(sys.modules.get("__main__"), "__file__", None)
        if mf:
            cands.append(mf if os.path.isabs(mf) else os.path.join(_launch_cwd(), mf))
    except Exception:
        pass
    try:
        a0 = sys.argv[0]
        if a0 and a0 not in ("-c", "-m", ""):
            cands.append(a0 if os.path.isabs(a0) else os.path.join(_launch_cwd(), a0))
            cands.append(a0)
    except Exception:
        pass
    found = ""
    for c in cands:
        try:
            if os.path.isfile(c):
                found = os.path.realpath(c)
                break
        except Exception:
            continue
    _s["entry_path"] = found
    return found or None


def entry_hash() -> str | None:
    if _s["entry_hash"] is not None:
        return _s["entry_hash"]
    try:
        ep = _entry_path()
        if ep is None:
            raise FileNotFoundError(sys.argv[0] if sys.argv else "<no argv>")
        with open(ep, "rb") as fh:
            _s["entry_hash"] = hashlib.sha256(fh.read()).hexdigest()
    except Exception:
        _s["entry_hash"] = None
    return _s["entry_hash"]


def _script_path() -> str | None:
    """The entry script's resolved path, for path_bound records (the brief
    named the path; the agent wrote the bytes). See _entry_path."""
    return _entry_path()


def job_key() -> str:
    """One key per job: a launched multi-rank job shares its launcher's; a
    launcher-less process has its own (daemon_client.job_key). Computed once
    per process: the launcher-less key carries the process start time, whose
    derivation moved by half a second between two calls in one process
    (tests/test_trials_in_image.sh, 2026-10-22), and a key that moves between
    the exact and the structural lookup would let one job draw twice."""
    k = _s.get("job_key")
    if not k:
        k = _job_key()
        _s["job_key"] = k
    return k


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _daemon_gpu() -> int:
    """The daemon's index of the GPU this process's work runs on. The local
    device is LOCAL_RANK when a launcher set it (accelerate, torchrun), else the
    current device; CUDA_VISIBLE_DEVICES then maps it to the physical GPU.
    Pacing on LOCAL_RANK alone put every pinned 1-GPU job
    (CUDA_VISIBLE_DEVICES=2 python sft.py) on daemon GPU 0, so four jobs pinned
    to GPUs 0-3 serialised on one cursor."""
    single = str(((_s.get("serving") or {}).get("process_model") or "")).lower() == "single"
    lr = (os.environ.get("LOCAL_RANK") or "").strip()
    if lr.isdigit() and not single:
        # a single-process script hard-codes cuda:0 and set_device(0); every
        # torchrun rank is a whole copy on its current device, not LOCAL_RANK
        local = int(lr)
    else:
        try:
            local = int(torch.cuda.current_device())
        except Exception:
            local = 0
    gp = sys.modules.get("cuda_hooks")
    count = int(getattr(gp, "DEVICE_COUNT", 0) or os.environ.get("NVRT_DEVICE_COUNT", "4"))
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    phys = physical_gpu_index(local, cvd, count)
    if not _s.get("gpu_noted"):
        _s["gpu_noted"] = True
        _note(f"pacing on daemon GPU {phys} (local {local}, CUDA_VISIBLE_DEVICES={cvd!r})")
    return phys


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _client() -> DaemonClient | None:
    if _s["client"] is not None:
        return _s["client"]
    try:
        c = DaemonClient()
        # the launch directory, not the current one: the daemon resolves
        # path-valued arguments against it, and a script may have chdir'd
        c.hello(cwd=_launch_cwd())
        _s["client"] = c
    except Exception as e:
        _note(f"no daemon ({e!r}); replay unavailable, running unpaced")
        _s["client"] = None
    return _s["client"]


def _note(msg: str) -> None:
    _s["log"].append(msg)
    if os.environ.get("NVRT_DEBUG"):
        sys.stderr.write(f"nvrt replay[r{_rank()}]: {msg}\n")


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------

def try_lookup(stage: str, fingerprint: str | None = None,
               fingerprint_inputs: dict | None = None) -> bool:
    """stage 'exact' (first forward) or 'structural' (first optimizer step).
    Returns True if replay is engaged after this call."""
    if _s["active"] and stage == "structural" and _s.get("match") in ("surrogate_pending", "surrogate_forced_pending") \
            and not _s.get("surrogate_final"):
        # a synthesised job (no record covers it): the live fingerprint now
        # fixes its configuration, timing and curve; one more lookup, and the
        # serving it returns replaces the provisional one, replay stays engaged
        _s["surrogate_final"] = True
        c = _client()
        if c is not None:
            try:
                resp = c.lookup_v2(job_key=job_key(), rank=_rank(), world_size=_world_size(),
                                   entry_hash=entry_hash(), argv=list(sys.argv), env={},
                                   argv_effective=_effective_args(), argv_defaults=_arg_defaults(),
                                   script_path=_script_path(), fingerprint=fingerprint,
                                   fingerprint_inputs=fingerprint_inputs, stage="structural")
                if resp.get("serving"):
                    _s["serving"] = resp["serving"]
                    _s["match"] = resp.get("match")
                    _s["record_id"] = resp.get("record_id")
                    _note(f"surrogate final: record={_s['record_id']} steady_step_s={_s['serving'].get('steady_step_s')}")
            except Exception as e:
                _note(f"surrogate final lookup failed: {e!r}")
        return True
    if _s["active"]:
        if stage == "structural" and _s.get("match") == "path" and not _s.get("fp_checked"):
            # matched by the brief's path, the agent's own script: log how its
            # live fingerprint compares with the record's (never a gate)
            _s["fp_checked"] = True
            c = _s["client"]
            if c is not None:
                try:
                    r = c._rpc({"cmd": "fingerprint_check", "fingerprint": fingerprint,
                                "inputs": fingerprint_inputs, "match": "path"})
                    _note(f"fingerprint check vs record: agree={r.get('agree')} differs={r.get('differs')}")
                except Exception as e:
                    _note(f"fingerprint check failed: {e!r}")
        return True
    if str(_s.get("disengaged") or "").startswith(("identity_", "whole_job_identity_")):
        return False              # refused as not the recorded configuration (r8.9): stays refused
    if stage == "exact" and _s["lookup_done_exact"]:
        return False
    if stage == "structural" and _s["lookup_done_structural"]:
        return False
    c = _client()
    if c is None:
        return False
    resp = c.lookup_v2(
        job_key=job_key(), rank=_rank(), world_size=_world_size(),
        entry_hash=entry_hash(), argv=list(sys.argv), env={},
        # the subset knobs a per_example launch may carry in its environment
        # (SELECTION_KV_MAX_EXAMPLES, ADAPTIVE_KV_MAX_EXAMPLES, ...): the daemon
        # reads the one the record's serving block names (docs/MLS-KV-SERVING.md §2.1)
        subset_env={k: v for k, v in os.environ.items() if k.endswith("_MAX_EXAMPLES")},
        # the script's parsed arguments, for the effective match (defaults filled in)
        argv_effective=_effective_args(), argv_defaults=_arg_defaults(),
        script_path=_script_path(),
        fingerprint=fingerprint, fingerprint_inputs=fingerprint_inputs,
        # the project's own modules this launch has loaded: one of them may be
        # the recorded evaluator, imported and run by the agent's own runner
        modules=_loaded_modules(),
        stage=stage)
    _s[f"lookup_done_{stage}"] = True
    _s["entry_module"] = resp.get("entry_module")
    if not _s["entry_module"]:
        vc.stop_module_timers()          # kept for the stand-in's own timer until its first example (gen_seam)
    _s["mode"] = resp.get("mode")
    _s["record_id"] = resp.get("record_id")
    _s["match"] = resp.get("match")
    _s["daemon_speed"] = resp.get("speed")
    _note(f"lookup[{stage}] -> mode={_s['mode']} record={_s['record_id']} "
          f"match={_s['match']} draw={resp.get('draw')}")
    if resp.get("mode") == "replay" and resp.get("serving") and not _cuda_visible():
        # a launch that sees no CUDA device (CUDA_VISIBLE_DEVICES empty or naming
        # no GPU the box has) runs on the CPU, as on hardware: the script chose its
        # device from torch.cuda.is_available() and never touches a GPU, so a
        # served record would hand GPU-shaped stand-ins to CPU code. Replay is not
        # engaged; the daemon logs replay_disengaged with this reason and models
        # the job. G-kvrep-haiku-L2's 32 empty-token launches crashed on r6 at
        # the per-example stand-in for want of this.
        _s["mode"] = "model"
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        _note(f"replay not engaged: no visible CUDA device (CUDA_VISIBLE_DEVICES={cvd!r})")
        c = _s["client"]
        if c is not None:
            try:
                c._rpc({"cmd": "disengage", "reason": "no_visible_cuda_device", "step": _s["step"],
                        "cuda_visible_devices": cvd})
            except Exception as e:
                _note(f"disengage rpc failed: {e!r}")
        return False
    if resp.get("mode") == "replay" and resp.get("serving"):
        _s["serving"] = resp["serving"]
        _engage()
        return True
    return False


def try_whole_job_early() -> bool:
    """A whole_job record's seam is the function that does the job's GPU work;
    replay's normal engagement is the first CUDA use, which for a lazy job (the
    model placed on the GPU inside the seamed function) is already inside the
    seam, too late to replace it (gpu-mock, 2026-11-02, TabArena ExperimentRunner
    ._run). This runs once, at `import torch` (cuda_hooks.apply, before the
    script's own code), does ONE exact lookup, and engages ONLY if the record is
    whole_job, installing the seam before the script ever calls it. For every
    other record it does nothing and leaves the first-CUDA-touch lookup to engage
    as before: the daemon caches the draw by job key, so the later lookup under
    the same key returns the same record and the draw is not consumed twice.
    Returns True when it engaged a whole_job record here."""
    if _s["active"] or _s.get("whole_job_probed"):
        return False
    if os.environ.get("NVRT_NO_REPLAY") == "1" or os.environ.get("NVRT_FINGERPRINT_ONLY") == "1":
        _s["whole_job_probed"] = True
        return False
    # A stateless peek first, on its own connection with no hello (next/ r8.8).
    # From r8 this probe said hello at `import torch`, so every process that
    # imports torch registered a job, CPU-only helpers included (Astra's 4-rank
    # TP4 fixture on r8.7: 9 job_hello), and a job's job_hello carried the argv
    # it had at import, not the one it runs under: a shard runner that sets
    # sys.argv to the evaluator's before main() was logged under its own name
    # (tests/test_shard_serving_in_image.sh 4/9 on r8-r8.7, 9/9 on r7.1). Only a
    # whole_job record needs the job registered here; every other launch says
    # hello at its first CUDA use, as before r8.
    try:
        pc = DaemonClient()
    except Exception:
        return False
    try:
        peek = pc.lookup_v2(job_key=None, rank=_rank(), world_size=_world_size(),
                            entry_hash=entry_hash(), argv=list(sys.argv), env={}, subset_env={},
                            argv_effective=_effective_args(), argv_defaults=_arg_defaults(),
                            script_path=_script_path(), cwd=_launch_cwd(), fingerprint=None,
                            fingerprint_inputs=None, modules=None, stage="whole_job_probe")
    except Exception as e:
        _note(f"whole_job early peek failed: {e!r}")
        return False
    finally:
        try:
            pc.close()
        except Exception:
            pass
    if peek.get("mode") != "replay" or not (peek.get("serving") or {}).get("whole_job"):
        _s["whole_job_probed"] = True
        return False
    pmod, _, pqual = str(((peek.get("serving") or {}).get("whole_job") or {}).get("seam") or "").partition(":")
    if _module_initialising(pmod) and _resolve_qualname(sys.modules[pmod], pqual) is None:
        # next/ r8.9.2: the seam module is still executing its own body and its
        # seamed function is not defined yet: it imports torch at its top, which
        # runs this probe (the late-pole DLM and sparse modules; gpu-mock's proof
        # on the installed box, 2026-11-03 08:09Z: whole_job_seam_not_found). Not
        # probed yet: the pre-import arm calls again right after the module's body
        # has run (importlib still marks it initialising then, so the test is
        # whether the seam is defined, not whether the module is loading), and the
        # first-CUDA-touch lookup engages it otherwise.
        _note(f"whole_job early probe: seam module {pmod!r} is still loading; deferred")
        return False
    _s["whole_job_probed"] = True
    c = _client()
    if c is None:
        return False
    try:
        resp = c.lookup_v2(job_key=job_key(), rank=_rank(), world_size=_world_size(),
                           entry_hash=entry_hash(), argv=list(sys.argv), env={},
                           subset_env={k: v for k, v in os.environ.items() if k.endswith("_MAX_EXAMPLES")},
                           argv_effective=_effective_args(), argv_defaults=_arg_defaults(),
                           script_path=_script_path(), fingerprint=None, fingerprint_inputs=None,
                           modules=_loaded_modules(), stage="whole_job_probe")
    except Exception as e:
        _note(f"whole_job early probe failed: {e!r}")
        return False
    serving = resp.get("serving") or {}
    if resp.get("mode") != "replay" or not serving.get("whole_job"):
        # not a whole_job job (or no record): leave the normal path alone, so the
        # first-CUDA-touch exact lookup still runs for step/per_example records
        return False
    if not _cuda_visible():
        _s["mode"] = "model"
        _note("whole_job early probe: no visible CUDA device; not engaged")
        return False
    # a whole_job job: arm engagement at the seam's module import, BEFORE the
    # seamed function (and its first CUDA use) runs. A __main__ seam, or a module
    # already loaded, engages now; a library module not yet imported engages when
    # it loads. Engaging means the normal first-CUDA-touch lookup is a no-op.
    seam = str((serving.get("whole_job") or {}).get("seam") or "")
    mod = seam.partition(":")[0]

    def _engage_whole_job():
        if _s["active"] or _s.get("mode") == "model":
            return
        _s["serving"] = serving
        _s["mode"] = resp.get("mode")
        _s["record_id"] = resp.get("record_id")
        _s["match"] = resp.get("match")
        _s["daemon_speed"] = resp.get("speed")
        _s["lookup_done_exact"] = True
        _note(f"whole_job early probe -> record={_s['record_id']} match={_s['match']}; engaging before the seam runs")
        _engage()

    if not mod or mod == "__main__" or mod in sys.modules:
        _engage_whole_job()
    else:
        _note(f"whole_job early probe: record {_s.get('record_id')} seam in {mod!r}; engaging at its import")
        _when_imported(mod, lambda _m: _engage_whole_job())
    return True


def _module_initialising(name: str) -> bool:
    """Whether the module `name` is in sys.modules but still executing its body
    (importlib sets spec._initializing for the length of exec_module)."""
    m = sys.modules.get(name) if name else None
    if m is None:
        return False
    return bool(getattr(getattr(m, "__spec__", None), "_initializing", False))


def _cuda_visible() -> bool:
    """Whether this process sees a CUDA device: torch.cuda.is_available() as the
    script saw it (cuda_hooks counts CUDA_VISIBLE_DEVICES as a driver does). An
    error answers True, so a broken probe never changes what a launch is served."""
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return True


def _engage() -> None:
    bad = _identity_any_path()
    if bad is not None:
        # next/ r8.9: the launch is not the recorded configuration (an edited bound
        # file, a project module shadowing a library one): nothing is served, the
        # job runs its own code on the modelled path and the draw goes back
        reason, detail = bad
        _s["mode"] = "model"
        _note(f"replay not engaged: {detail}; the job runs its own code")
        _disengage(reason, detail=detail)
        return
    _s["active"] = True
    vc.REPLAY_ACTIVE = True
    vc.get_clock().set_recorded_mode(True)
    _disable_device_memory_tracking()
    _hook_device_backward()
    # Optional rank-0 profile of the replayed loop, to locate host overhead:
    # NVRT_CPROFILE=<path>.prof (pstats format, written at exit).
    # NOT NVRT_PROFILE: that name is the FakeGPU device profile ("h100")
    # read by cuda_hooks, and a path there would break the device identity.
    prof_path = os.environ.get("NVRT_CPROFILE")
    if prof_path and _rank() == 0:
        import atexit
        import cProfile
        _prof = cProfile.Profile()
        _prof.enable()

        def _dump():
            _prof.disable()
            _prof.dump_stats(prof_path)
        atexit.register(_dump)
    # Poll the daemon for failures only when the record has one that fires:
    # an RPC per step costs real wall time on every step of every job.
    _s["has_failures"] = _live_failures(_s["serving"])
    # The job's wall time ends after its last queued GPU work, as on hardware.
    import atexit
    atexit.register(_drain)
    _install_materialisation_hooks()
    _install_role_propagation()
    _install_device_move_roles()
    _neutralise_optimizer_steps()
    _hold_weights()                       # the model's memory appears at engagement, as on hardware
    if (_s["serving"] or {}).get("per_example"):
        _s["pe_t_engaged"] = _REAL_MONO()
        _install_per_example_seams()
        if not _s["active"]:
            return                        # withdrawn: the seams were not found (see there)
    if (_s["serving"] or {}).get("whole_job"):
        _s["pe_t_engaged"] = _REAL_MONO()
        _install_whole_job_seam()
        if not _s["active"]:
            return                        # withdrawn: the seam was not found (see there)
    _install_grad_norm_stub()
    _install_excepthook()
    _when_imported("transformers.trainer", _patch_trainer_module)
    _when_imported("megatron.core.optimizer.optimizer", _patch_megatron_optimizer_module)
    # The record's startup_in_train_s is paced as part of step 1 (see
    # _step_time), not here: engagement for an exact match is the first CUDA
    # touch, before the script starts its own training timer, and pacing the
    # startup here left the script's train_time 184 s short on every mng3
    # record (gate 2 on the mng3 image, 2026-10-21).
    _note(f"replay engaged: steady_step_s={_s['serving'].get('steady_step_s')} "
          f"steps={_s['serving'].get('steps')}")


# ---------------------------------------------------------------------------
# pacing
# ---------------------------------------------------------------------------

_REAL_MONO = vc._REAL_MONO
_last_cursor = {"t": None}


def _disable_device_memory_tracking() -> None:
    """FakeGPU's per-tensor memory tracking (mark_category on every grad wrap,
    sync_parameter_grads after every backward) is the dominant host overhead of
    a replayed loop — a profile showed ~44 s of its own time in mark_category
    over 19 M calls. In replay the memory model comes from the record, so none
    of it is needed. Turn it off at the module globals FakeGPU gates on, which
    short-circuits _register_tensor_for_memory_tracking, mark_category and the
    dispatch tracking mode. This lifts the high-knob host-bound ceiling."""
    try:
        from nvcompat import torch_patch as fg
        fg._MEMORY_TRACKING = False
        fg._DISPATCH_MEMORY_TRACKING = False
        _note("device memory tracking disabled for replay")
    except Exception as e:
        _note(f"could not disable device memory tracking: {e!r}")


# ---------------------------------------------------------------------------
# meta backing, cheap form: slim parameters as they reach the fake GPU
# ---------------------------------------------------------------------------

def _slim_tensor(p) -> bool:
    """Replace a parameter's storage with a one-element CPU tensor expanded to
    its shape: shape, numel, dtype and device all stay as they were, grads can be
    attached, read-only ops work, and the storage is ~0 bytes. A real GPU holds
    the weights in HBM, not host RAM; in replay nothing ever reads them. Same-
    kind swap (CPU dense → CPU dense), so set_data accepts it where a meta
    tensor would not. In-place writes to a slim tensor fail (expanded storage),
    so zero_grad(set_to_none=False) would; HF and track 3 use set_to_none=True."""
    try:
        if getattr(p, "_gm_slim", False) or p.numel() <= 1:
            return False
        slim = torch.zeros((), dtype=p.dtype).expand(p.shape)
        object.__setattr__(p, "_gm_shape", tuple(p.shape))
        p.data = slim
        if hasattr(p, "raw_data"):
            p.raw_data = slim
        object.__setattr__(p, "_gm_slim", True)
        return True
    except Exception:
        return False


def slim_module(module: nn.Module) -> int:
    n = 0
    for p in module.parameters():
        if getattr(p, "is_cuda", False) and _slim_tensor(p):
            n += 1
    for b in module.buffers():
        if getattr(b, "is_cuda", False) and _slim_tensor(b):
            n += 1
    return n


def slim_root_once() -> None:
    """Slim the model's parameters ONCE, at the first optimizer step. Not
    earlier: a script writes to parameters during weight init (track 3 does
    `p.data.mul_(1.25)`, `.zero_()` right after building the model), and a slim
    tensor's expanded storage cannot be written in place. By the first step all
    init is done, the recorded curve starts, and a replayed job never reads or
    writes parameter values again (steps are neutralised, grads are dummy). So
    this is the safe moment to free the storage. The optimizer may hold its own
    references to the full parameters; FakeGPU's registries are the live holder
    the earlier probe found, so we also drop those."""
    if _s.get("slimmed"):
        return
    _s["slimmed"] = True
    try:
        import fingerprint_runtime as fr
        root = fr._state.get("root")
    except Exception:
        root = None
    if root is None:
        return
    n = slim_module(root)
    # drop FakeGPU's registry references to the originals so the storage frees
    try:
        from nvcompat import _upstream as up
        up._REGISTERED_PARAMETERS.clear()
    except Exception:
        pass
    import gc
    gc.collect()
    _note(f"slimmed {n} params/buffers at first step")


def _recorded_mem_bytes() -> int:
    """Device memory this rank held in the recorded run, in bytes: its own GPU's
    peak where the record has per-GPU figures (rank r on recorded GPU r), else
    the peak over GPUs; scaled by convert()'s memory_factor when the GPU count
    differs. A replayed job allocates almost nothing on the fake GPU, so without
    this nvidia-smi showed 0 MiB on a GPU the record had at 40-68 GB
    (2026-10-20)."""
    sv = _s["serving"] or {}
    m = sv.get("memory") or {}
    peak = m.get("peak_bytes")
    if not peak:
        return 0
    b = float(peak)
    pct = m.get("per_gpu_peak_pct") or {}
    lr = (os.environ.get("LOCAL_RANK") or "").strip() or "0"
    if pct and lr in pct:
        top = max(float(v) for v in pct.values()) or 0.0
        if top > 0:
            b = float(peak) * float(pct[lr]) / top
    f = float(((sv.get("converted") or {}).get("memory_factor")) or 1.0)
    return int(b * f)


def _recorded_weights_bytes() -> int:
    """The model as loaded on this rank (RECORD memory.weights_bytes: weights
    plus context and allocator slack), scaled like the peak. 0 when the record
    has no such figure: the peak then arrives whole at the first paced unit."""
    sv = _s["serving"] or {}
    m = sv.get("memory") or {}
    w = m.get("weights_bytes")
    if not w:
        return 0
    conv = sv.get("converted") or {}
    # sharded records: the weights' own factor (their sharded part scales with
    # width, the rest does not); else the peak's factor, as before
    f = float(conv.get("weights_factor") or conv.get("memory_factor") or 1.0)
    return int(float(w) * f)


def _hold_weights() -> None:
    """Bring the record's weights onto the GPU at engagement and step up to the
    peak at the first paced unit: what nvidia-smi shows on hardware while the
    model loads (gpu-tasks 2026-10-23 17:52Z, the KV records' weights_bytes).
    The hold obeys the oversubscription rule: weights that do not fit beside
    the other holders OOM at load, as they would."""
    c = _s["client"]
    if c is None or _s.get("weights_held") or _s.get("mem_reported"):
        return
    w = _recorded_weights_bytes()
    if w <= 0:
        return
    _s["weights_held"] = True
    try:
        resp = c.hold(gpu=_daemon_gpu(), bytes=w, oom_aware=True)
    except Exception as e:
        _note(f"weights hold rpc failed: {e!r}")
        return
    _s["mem_bytes"] = int(_s.get("mem_bytes") or 0) + w
    _note(f"holding {w / 2**30:.2f} GiB on daemon GPU {_daemon_gpu()} from engagement (the record's weights)")
    if resp.get("oom") and resp.get("oom_detail"):
        _raise_oversubscribed(resp["oom_detail"])


def _pace(recorded_s: float, unit: str = "step") -> None:
    """Advance the job clock by `recorded_s` and pace WALL time through the
    daemon's per-GPU cursor, with one unit of lookahead.

    A real GPU executes asynchronously: the host launches step k's kernels and
    goes on to prepare step k+1 while they run, so per-step wall time is
    max(host overhead, GPU time) — not their sum. Pacing by "enqueue, then
    block until the cursor" serialises them, which the first end-to-end replay
    showed: ~12 ms of host overhead per step was ADDED to the recorded 150 ms.
    So we block only until the PREVIOUS work drains (cursor − this unit's wall
    cost). Real sync points (a materialised value, process exit) drain fully
    via _drain(). The daemon's cursor is CLOCK_MONOTONIC on the same host, so
    we compare it with the genuine monotonic clock, not the patched one."""
    if recorded_s <= 0:
        return
    vc.get_clock().advance_recorded(recorded_s)
    c = _s["client"]
    if c is None:
        return
    try:
        mem = 0
        if not _s.get("mem_reported"):
            # the first paced unit brings the rank's recorded memory onto its GPU;
            # the daemon frees it when the job ends
            _s["mem_reported"] = True
            # the peak less what the weights hold already brought
            mem = max(0, _recorded_mem_bytes() - int(_s.get("mem_bytes") or 0))
            _s["mem_bytes"] = int(_s.get("mem_bytes") or 0) + mem
            if mem:
                _note(f"holding {_s['mem_bytes'] / 2**30:.1f} GiB on daemon GPU {_daemon_gpu()} (the record's peak for this rank)")
        resp = c.enqueue(gpu=_daemon_gpu(), op="replay", modelled_s=recorded_s, mem_delta=mem,
                         step=_s["step"], unit=unit,
                         lr=(_s.get("lr_now") or None) if unit == "step" else None,
                         oom_aware=bool(mem))
    except Exception as e:
        _note(f"pacing rpc failed: {e!r}")
        return
    if mem and resp.get("oom") and resp.get("oom_detail"):
        _raise_oversubscribed(resp["oom_detail"])
    cursor = resp.get("cursor")
    if cursor is None:
        return
    wall_cost = float(resp.get("wall_cost") or 0.0)
    wait = (cursor - wall_cost) - _REAL_MONO()
    if wait > 0:
        _gpu_wait(wait)
    _last_cursor["t"] = cursor


def _drain() -> None:
    """A real synchronisation point: wait until all queued work has finished."""
    t = _last_cursor["t"]
    if t is None:
        return
    wait = t - _REAL_MONO()
    if wait > 0:
        _gpu_wait(wait)


def _step_time(step_index: int) -> float:
    """Recorded time of optimizer step `step_index` (1-based), converted.

    Step 1 carries the record's one-off startup. Where the record has a
    first_step_s (track 3: 5.4 s, mostly compile warm-up) that figure already
    includes it. Otherwise startup_in_train_s is added: RECORD-FORMAT defines
    it as the script's own train_time at the first validation after step 0
    minus steady steps, i.e. time inside the training loop at step 1 (mostly
    torch.compile), so it must pass on the script's own timer, which only
    pacing it inside step 1 achieves. 641 registry records are of this kind;
    pacing it at engagement instead left their train_time short by exactly
    the startup (gate 2, 2026-10-21)."""
    sv = _s["serving"]
    steady = float(sv.get("steady_step_s") or sv.get("per_step_gpu_s") or 0.0)
    if step_index == 1:
        if _s.get("skip_startup_once"):
            _s["skip_startup_once"] = False
            return steady
        if sv.get("first_step_s"):
            return float(sv["first_step_s"])
        return steady + float(sv.get("startup_in_train_s") or 0.0)
    return steady


def on_optimizer_step() -> None:
    """Called once per optimizer.step() of a replayed job (any optimizer)."""
    _s["step"] += 1
    if _s["step"] == 1 and os.environ.get("NVRT_SLIM_PARAMS") == "1":
        # Parameter slimming (freeing weight storage for multi-B models) is OFF
        # by default: FakeGPU's grad setter writes raw_data.grad, and an
        # expanded (slim) raw_data silently drops the grad, so a job that
        # re-populates .grad each step (track 3) loses it after step 1. The
        # largest replay-ready model today is 355M (~3 GB/rank), which fits, so
        # correctness wins over the memory saving. Multi-B replay needs the
        # from_pretrained-on-meta path (never materialises weights), tracked
        # separately; this storage-swap approach is kept behind the flag only
        # for experiments. See docs/BUILD-STATUS "parameter slimming".
        slim_root_once()
    _pace(_step_time(_s["step"]))
    _check_failure()
    if _PROBE_STEPS and _s["step"] >= _PROBE_STEPS:
        # tests/probe_all_scripts.sh: every registry script to its first
        # optimizer step on an image, four at a time. A probe, not a job: no
        # drain, no job_bye, the daemon frees the job when the socket closes.
        _note(f"probe: reached optimizer step {_s['step']}, exiting (NVRT_PROBE_STEPS={_PROBE_STEPS})")
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(0)


def on_eval_phase_begin() -> None:
    ev = _s["serving"].get("eval_s_per_validation")
    if ev:
        _pace(float(ev), unit="eval")


def _install_excepthook() -> None:
    """Report the job's uncaught exception to the daemon before the process
    dies, chaining to the hook already installed (torch's rank prefixing)."""
    prev = sys.excepthook
    if getattr(prev, "_gm_wrapped", False):
        return

    def hook(tp, val, tb):
        try:
            c = _s["client"]
            if c is not None and _s["active"]:
                # the innermost frame: where the exception was raised, so a
                # reader can tell the agent's own code from torch, the patch
                # or anything else (runner's scorer, the study 2026-10-21)
                frame = {}
                t = tb
                while t is not None and t.tb_next is not None:
                    t = t.tb_next
                if t is not None:
                    code = t.tb_frame.f_code
                    frame = {"file": code.co_filename, "line": t.tb_lineno, "func": code.co_name}
                c.exception(getattr(tp, "__name__", str(tp)), val, _s["step"], frame=frame)
        except Exception:
            pass
        if getattr(val, "_gm_served", False) and _s.get("served_stderr"):
            _write_served_stderr()     # the recorded run's stderr, not the live traceback
            return
        prev(tp, val, tb)
    hook._gm_wrapped = True
    sys.excepthook = hook


_OOM_TEXT = ("CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total "
             "capacity of 79.19 GiB of which 1.12 GiB is free. Of the allocated "
             "memory 76.51 GiB is allocated by PyTorch, and 1.03 GiB is reserved "
             "by PyTorch but unallocated. If reserved but unallocated memory is "
             "large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
             "to avoid fragmentation. See documentation for Memory Management "
             "(https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)")
_XID_TEXT = ("CUDA error: an illegal memory access was encountered\n"
             "CUDA kernel errors might be asynchronously reported at some "
             "other API call, so the stacktrace below might be incorrect.")


def _live_failure_wheres(serving) -> set:
    """The seams at which a failure can fire for THIS launch: {"forward",
    "backward", "step"} over the record's failures that are not suppressed and
    are armed (the daemon marks a triggered failure `armed: False` for a launch
    under its limit, docs/FAILURE-SERVING.md). Empty: nothing to poll — an RPC
    per step costs real wall time on every step of every job."""
    return {str((f or {}).get("where") or "step")
            for f in ((serving or {}).get("failures") or [])
            if (f or {}).get("replay") != "suppress" and (f or {}).get("armed") is not False}


def _live_failures(serving) -> bool:
    w = _live_failure_wheres(serving)
    _s["failure_wheres"] = w
    return bool(w)


def _write_served_stderr() -> None:
    """Print the recorded run's stderr verbatim, once, in place of a live
    traceback: the agent reads the real run's error text."""
    text = _s.get("served_stderr")
    if not text:
        return
    _s["served_stderr"] = None
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.write(text if text.endswith("\n") else text + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _oversubscribed_message(detail: dict) -> str:
    """PyTorch's OOM wording for a GPU whose held memory the job's arrival
    pushed past the card (docs/FAILURE-SERVING.md, oversubscription). The
    numbers are the daemon's: the card, the other holders by pid, this job's
    recorded peak. The job is modelled as having got what was free and failing
    on the remainder, the one allocation story the first step supports."""
    cap = int(detail.get("capacity_bytes") or 0)
    others = [o for o in (detail.get("others") or []) if int(o.get("bytes") or 0) > 0]
    others_b = sum(int(o["bytes"]) for o in others)
    mine = int(detail.get("mine_bytes") or 0)
    fit = max(0, min(mine, cap - others_b))
    tried = max(mine - fit, 1 << 20)
    free_now = max(0, cap - others_b - fit)
    try:
        gpu = int(torch.cuda.current_device())
    except Exception:
        gpu = int(detail.get("gpu") or 0)

    def size(b: int) -> str:
        b = int(b)
        if b >= 2 ** 30:
            return f"{b / 2 ** 30:.2f} GiB"
        if b >= 2 ** 20:
            return f"{b / 2 ** 20:.2f} MiB"
        if b >= 1024:
            return f"{b / 1024:.2f} KiB"
        return f"{b} bytes"
    procs = "".join(f" Process {o['pid']} has {size(o['bytes'])} memory in use."
                    for o in others if o.get("pid"))
    return (f"CUDA out of memory. Tried to allocate {tried / 2 ** 20:.2f} MiB. GPU {gpu} has a total "
            f"capacity of {size(cap)} of which {size(free_now)} is free. Including non-PyTorch memory, "
            f"this process has {size(fit)} memory in use.{procs} Of the allocated memory {size(fit)} is "
            f"allocated by PyTorch, and 0 bytes is reserved by PyTorch but unallocated. If reserved but "
            f"unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"to avoid fragmentation.  See documentation for Memory Management  "
            f"(https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)")


def _raise_oversubscribed(detail: dict) -> None:
    _note(f"oversubscribed GPU: this job's recorded peak {int(detail.get('mine_bytes') or 0) / 2 ** 30:.1f} GiB "
          f"does not fit beside {len(detail.get('others') or [])} other holder(s) on the card; raising OOM")
    _s["served_stderr"] = None            # the live traceback: this is the job's own allocation failing
    raise _served_exc(torch.cuda.OutOfMemoryError, _oversubscribed_message(detail))


def _served_exc(cls, msg):
    e = cls(msg)
    try:
        e._gm_served = True      # the excepthook prints the recorded stderr for this one
    except Exception:
        pass
    return e


def _exc_class(name):
    """The exception class a recorded type name denotes: OutOfMemoryError in
    any spelling is torch's; builtins and torch's own by their last component."""
    if not name:
        return None
    last = str(name).rsplit(".", 1)[-1]
    if last == "OutOfMemoryError":
        return torch.cuda.OutOfMemoryError
    import builtins
    for ns in (builtins, torch):
        c = getattr(ns, last, None)
        if isinstance(c, type) and issubclass(c, BaseException):
            return c
    return None


def _check_failure(phase: str = "training", where: str = "step",
                   step: Optional[int] = None) -> None:
    """Serve a recorded failure (docs/FAILURE-SERVING.md). oom / xid / nan raise
    the recorded exception type with the recorded message in the script, so a
    script's own `except torch.OutOfMemoryError` works as on hardware; an
    uncaught one prints the run's recorded stderr verbatim (when the record has
    it) instead of the live traceback. crash / killed end the process with the
    recorded exit code after printing that stderr."""
    c = _s["client"]
    if c is None or not _s.get("has_failures"):
        return
    if phase != "after_training" and where not in (_s.get("failure_wheres") or ()):
        return
    try:
        f = c.pending_failure(step=(_s["step"] if step is None else int(step)), phase=phase, where=where)
    except Exception:
        return
    if not f:
        return
    mode = f.get("mode")
    exc = f.get("exception") or {}
    text = f.get("stderr")
    _note(f"recorded failure fires at step {_s['step']}: mode={mode} exception={exc.get('type')} "
          f"stderr={'recorded' if text else 'none'} trigger={f.get('trigger')} label={f.get('label')}")
    if text:
        _s["served_stderr"] = text
    if mode in ("crash", "killed"):
        _write_served_stderr()
        code = f.get("exit_code")
        os._exit(int(code) if code is not None else (137 if mode == "killed" else 1))
    cls = _exc_class(exc.get("type"))
    msg = exc.get("message") or ""
    if mode == "oom":
        raise _served_exc(cls or torch.cuda.OutOfMemoryError, msg or _OOM_TEXT)
    if mode == "xid":
        raise _served_exc(cls or RuntimeError, msg or _XID_TEXT)
    if cls is not None:
        raise _served_exc(cls, msg)
    if text:
        _write_served_stderr()
        code = f.get("exit_code")
        os._exit(int(code) if code is not None else 1)
    _note(f"recorded failure (mode={mode}) carries no exception and no stderr: nothing raised "
          f"(a nan failure lives in the served series)")


# ---------------------------------------------------------------------------
# forward / backward short-circuits (called from fingerprint_runtime's hooks)
# ---------------------------------------------------------------------------

def forward_stub(module: nn.Module, args, kwargs):
    """Return the role-tagged dummy for a depth-0 forward in replay."""
    t = None
    for a in list(args) + list((kwargs or {}).values()):
        if isinstance(a, torch.Tensor):
            t = a
            break
    dev = t.device if t is not None else "cpu"
    eval_mode = (not module.training) or (not torch.is_grad_enabled())
    role = "val_loss" if eval_mode else "train_loss"
    if eval_mode and not _s["in_eval_phase"]:
        _s["in_eval_phase"] = True
        on_eval_phase_begin()
    elif not eval_mode:
        _s["in_eval_phase"] = False
        # a recorded failure placed at the forward (where a real OOM mostly is)
        # surfaces here, in the step this forward belongs to
        _check_failure(where="forward", step=_s["step"] + 1)
    out = torch.zeros((), device=dev, requires_grad=not eval_mode)
    try:
        object.__setattr__(out, _ROLE, role)
    except Exception:
        pass
    return out


def _hook_device_backward() -> None:
    """FakeCudaTensor overrides .backward() (it calls raw_data.backward() +
    sync_parameter_grads), which SHADOWS the Tensor.backward hook that
    fingerprint_runtime installs. A replayed job's loss is a FakeCudaTensor, so
    `loss.backward()` would run real autograd and never reach backward_stub,
    leaving .grad None and tripping the script's `assert p.grad is not None`.
    Replace the method on the class with the stub while replay is active."""
    try:
        from nvcompat import _upstream as up
        cls = up.FakeCudaTensor
    except Exception:
        return
    if getattr(cls, "_gm_backward_hooked", False):
        return

    def backward(self, *a, **k):
        if _s["active"]:
            import fingerprint_runtime as fr
            backward_stub(fr._state.get("root"))   # bumps the epoch itself
            return None
        return up.FakeCudaTensor.__mro__[1].backward(self, *a, **k)

    cls._gm_orig_backward = cls.backward
    cls.backward = backward
    cls._gm_backward_hooked = True
    _hook_engine_backward()


class _ReplayEngine:
    """Stands in for torch.autograd.Variable._execution_engine while replay is
    active (next/ r8.6). Megatron's custom_backward (schedules.py, taken when
    config.deallocate_pipeline_outputs) calls _execution_engine.run_backward
    directly, bypassing Tensor.backward and torch.autograd.backward. Under the
    patch a cuda tensor's autograd graph lives on its inner CPU tensor, so the
    engine called on the wrapper sees an empty graph: nothing runs, no grad is
    set, no accumulate-grad hook fires, and Megatron's grad buffer waits
    forever (finish_grad_sync: '0/8 params have grad available', Astra
    step-probe-1..4, 2026-11-02). While replay is active run_backward goes to
    the same stub loss.backward() already reaches; otherwise, and for every
    other attribute, the real engine is used unchanged."""

    def __init__(self, inner):
        self._gm_inner = inner

    def run_backward(self, *a, **k):
        if _s["active"] and self._is_replayed_backward(a, k):
            import fingerprint_runtime as fr
            backward_stub(fr._state.get("root"), fire_hooks=True)
            return ()
        if self._is_replayed_backward(a, k):
            # The modelled path (no record, or replay withdrawn): a REAL backward.
            # FakeCudaTensor.backward() runs it on the inner CPU tensor and syncs
            # the grads; a direct engine call on the wrapper reached an empty graph
            # and computed nothing (FakeGPU, before r8.6). Do the same here.
            try:
                from nvcompat import _upstream as up
            except Exception:
                return self._gm_inner.run_backward(*a, **k)
            names = ("tensors", "grad_tensors", "keep_graph", "create_graph", "inputs",
                     "allow_unreachable", "accumulate_grad")
            kw = dict(zip(names, a))
            kw.update(k)
            for key in ("tensors", "grad_tensors"):
                v = kw.get(key)
                if isinstance(v, torch.Tensor):
                    v = (v,)
                if v is not None:
                    kw[key] = tuple(up.unwrap_tensor(t) for t in v)
            out = self._gm_inner.run_backward(**kw)
            up.sync_parameter_grads()
            return out
        return self._gm_inner.run_backward(*a, **k)

    @staticmethod
    def _is_replayed_backward(a, k) -> bool:
        """A backward (no `inputs`: torch.autograd.grad passes some and gets the
        real engine, as before) of a tensor on the fake GPU — the replayed
        loss. A real CPU backward a script runs for its own reasons is left to
        the real engine."""
        inputs = k.get("inputs", a[4] if len(a) > 4 else ())
        if inputs:
            return False
        tensors = k.get("tensors", a[0] if a else ())
        if isinstance(tensors, torch.Tensor):
            tensors = (tensors,)
        try:
            return any(getattr(t, "is_cuda", False) for t in tensors)
        except Exception:
            return False

    def __getattr__(self, name):
        return getattr(self._gm_inner, name)


def _hook_engine_backward() -> None:
    try:
        from torch.autograd import Variable
    except Exception:
        return
    eng = getattr(Variable, "_execution_engine", None)
    if eng is None or isinstance(eng, _ReplayEngine):
        return
    try:
        Variable._execution_engine = _ReplayEngine(eng)
    except Exception as e:
        _note(f"engine backward not wrapped: {e!r}")


# Real (writable) zero grads up to this many parameter bytes per process;
# slim expanded ones beyond. 4 GiB covers every registry model today (<= 355 M
# params) with room; a multi-B model falls back to slim grads.
_REAL_GRADS_MAX_BYTES = int(os.environ.get("NVRT_REAL_GRADS_MAX_BYTES", str(4 << 30)))


def _real_grads(root: nn.Module) -> bool:
    key = id(root)
    cached = _s.setdefault("real_grads", {})
    if key not in cached:
        total = 0
        for p in root.parameters():
            if p.requires_grad:
                total += p.numel() * p.element_size()
        cached[key] = total <= _REAL_GRADS_MAX_BYTES
        _note(f"grads under replay: {'real zeros' if cached[key] else 'slim (expanded)'} "
              f"for {total / 2**30:.2f} GiB of parameters (budget {_REAL_GRADS_MAX_BYTES / 2**30:.0f} GiB)")
    return cached[key]


def backward_stub(root: nn.Module | None, fire_hooks: bool = False) -> None:
    _check_failure(where="backward", step=_s["step"] + 1)   # a failure placed at the backward
    # Each backward marks a new training iteration epoch; the first optimizer
    # stepped after it paces the iteration, later optimizers in the same
    # iteration see the same epoch and do not pace again (track 3 steps two).
    try:
        import fingerprint_runtime as fr
        fr._state["backward_epoch"] = fr._state.get("backward_epoch", 0) + 1
    except Exception:
        pass
    if root is None:
        return
    cache = _s["grad_cache"]
    for p in root.parameters():
        if not p.requires_grad or p.grad is not None:
            continue
        g = cache.get(id(p))
        if g is None:
            if _real_grads(root):
                # a real zero grad: scripts write into grads in place (psgd's
                # all_reduce AVG divides, two scripts copy_ a preconditioned
                # grad over it), which the slim form below rejects. Costs the
                # model's parameter bytes once per process; written values
                # persist across steps (the cache is not re-zeroed), which no
                # served number reads.
                g = torch.zeros_like(p)
            else:
                # a slim zero grad: one element expanded to the shape, for a
                # model whose grads would not fit the budget. Read-only: norms,
                # asserts, skipped collectives; an in-place write raises.
                g = torch.zeros((), dtype=p.dtype).expand(p.shape)
            try:
                object.__setattr__(g, _DUMMY, True)
            except Exception:
                pass
            cache[id(p)] = g
        try:
            p.grad = g
        except Exception:
            pass
    if fire_hooks:
        _fire_grad_acc_hooks(root)


def _grad_acc_node(p):
    """The AccumulateGrad node of parameter p, the one a job's grad-ready hook is
    registered on (Megatron DDP: p.expand_as(p).grad_fn.next_functions[0][0]
    .register_hook). Under the patch the graph lives on p.raw_data when p is a
    fake cuda tensor."""
    t = getattr(p, "raw_data", None)
    t = p if t is None else t
    try:
        return t.expand_as(t).grad_fn.next_functions[0][0]
    except Exception:
        return None


def _fire_grad_acc_hooks(root) -> None:
    """A real backward runs every hook registered on a parameter's
    accumulate-grad node: Megatron's DDP uses that to fold .grad into main_grad
    and mark the parameter ready in its bucket (register_grad_ready), and
    finish_grad_sync asserts every parameter of the bucket was marked. The stub
    sets .grad but runs no autograd, so those hooks never fired (next/ r8.6,
    Astra's TP4 NeMo fixture). Called only for a backward that came through the
    C++ engine directly (_ReplayEngine.run_backward: Megatron's custom_backward);
    a loss.backward() job takes the plain stub, unchanged. It runs a near-free
    real backward through `p.reshape(-1)[:1] * 0` for every trainable
    parameter, so the engine accumulates a zero gradient into p and runs the
    job's own hooks exactly as on hardware. No readiness flag is set by hand and
    no assertion is bypassed."""
    if root is None:
        return
    # AccumulateGrad is an immutable C type whose hook dict is not readable, so
    # which parameters carry a hook cannot be seen; fire them all. The cost is a
    # one-element graph per parameter (0.03 s for a 1M-element tensor, probe
    # 2026-11-02), and the zero it accumulates leaves .grad's values unchanged.
    hooked = [p for p in root.parameters() if p.requires_grad]
    if not hooked:
        return
    raw = [getattr(p, "raw_data", None) if getattr(p, "raw_data", None) is not None else p for p in hooked]
    try:
        s = None
        for t in raw:
            term = (t.reshape(-1)[:1] * 0).sum()
            s = term if s is None else s + term
        if s is not None:
            # the real engine, not the replay stand-in (which would re-enter the stub)
            from torch.autograd import Variable
            eng = Variable._execution_engine
            inner = getattr(eng, "_gm_inner", eng)
            inner.run_backward(tensors=(s,), grad_tensors=(torch.ones_like(s),), keep_graph=False,
                               create_graph=False, inputs=(), allow_unreachable=True,
                               accumulate_grad=True)
        try:
            from nvcompat import _upstream as up
            up.sync_parameter_grads()
        except Exception:
            pass
        if not _s.get("grad_hooks_noted"):
            _s["grad_hooks_noted"] = True
            _note(f"backward under replay: fired the accumulate-grad hooks of {len(hooked)} parameter(s)")
    except Exception as e:
        _note(f"backward under replay: accumulate-grad hooks not fired: {e!r}")


def _neutralise_optimizer_steps() -> None:
    """Replace every constructed optimizer's step with pacing only: no
    arithmetic (values are served; a real AdamW over 124 M params per step
    would cost real CPU seconds × thousands of steps). fingerprint_runtime
    records optimizer instances; also patch future ones."""
    try:
        import fingerprint_runtime as fr
        insts = list(getattr(fr, "_optimizer_instances", []))
    except Exception:
        insts = []
    for opt in insts:
        _neutralise(opt)
    _s["_neutralise_new"] = True


def _neutralise(opt) -> None:
    """Replace this optimizer's step with pacing only while replay is active,
    as a BOUND method. torch's LRScheduler wraps `opt.step` and reads
    `step_fn.__func__`. The first version stored a plain function, so
    `get_linear_schedule_with_warmup` raised "'function' object has no
    attribute '__func__'" and the DPO smoke job died in create_scheduler
    (2026-10-20). When replay is not active (withdrawn by _disengage), the
    stub calls the optimizer's own step, so nothing has to be unwound: deleting
    the attribute would also remove a scheduler's wrapper around it."""
    if getattr(opt, "_gm_neutralised", False):
        return
    orig = opt.step

    def step(self, *a, **k):
        # several optimizers may step per loop iteration (track 3 steps two);
        # on_optimizer_step_if_due paces once per iteration
        if _s["active"]:
            loss = _run_closure(a, k)
            _structural_check_if_due()
            on_optimizer_step_if_due(self)
            return loss
        return orig(*a, **k)

    opt.step = types.MethodType(step, opt)
    opt._gm_neutralised = True
    _s["neutralised"].append(opt)


def _run_closure(a, k):
    """torch's step(closure): the closure re-evaluates the model and its loss is
    the step's return value. Lightning's automatic optimization runs the training
    step and its backward INSIDE the closure it passes to optimizer.step (NeMo,
    Astra's TP4 fixture, 2026-11-03), so a replayed step that skipped it skipped
    the iteration's forward and backward: no backward epoch, one paced step for
    the whole run (next/ r8.7). Under replay the closure's forward and backward
    are the stubs, so running it costs nothing and bumps the epoch the pacing
    reads. A step called without one returns None, as before."""
    closure = k.get("closure", a[0] if a else None)
    if not callable(closure):
        return None
    with torch.enable_grad():
        return closure()


def _patch_megatron_optimizer_module(mod) -> None:
    """Replay Megatron's optimizer step (next/ r8.7). NeMo's
    McoreDistributedOptimizer is a torch Optimizer that never calls
    Optimizer.__init__, so it is never neutralised, and its step(closure) runs
    the closure and then the Megatron optimizer's step: clip_grad_norm runs
    TransformerEngine's multi-tensor kernel on the fake tensors ('expected input
    to be on cuda', Astra's TP4 fixture on r8.6, all four ranks, 2026-11-03).
    A Megatron optimizer is not a torch Optimizer either, so per-instance
    neutralising never reaches it. Under replay its step does what a
    neutralised step does: one paced step and no arithmetic (the inner torch
    optimizer, the grad-norm kernel, the main-to-model param copy all skipped),
    and returns Megatron's (update_successful, grad_norm, num_zeros_in_grad):
    True, the record's grad_norm at this step or 0.0 (the norm of the replay's
    zero grads, as the clip_grad_norm_ stub), None. The wrapper's own closure
    still runs, so the iteration's forward and backward happen as on hardware.
    Every subclass that defines its own step is wrapped: those defined now, and
    later ones through MegatronOptimizer.__init_subclass__. Installed at
    engagement, like the Trainer seam; withdrawn replay runs the original."""
    M = getattr(mod, "MegatronOptimizer", None)
    if M is None or M.__dict__.get("_gm_seam"):
        return
    M._gm_seam = True

    def wrap(cls):
        f = cls.__dict__.get("step")
        if not isinstance(f, types.FunctionType) or getattr(f, "_gm_wrapped", False) \
                or getattr(f, "__isabstractmethod__", False):
            return

        @functools.wraps(f)
        def step(self, *a, **k):
            if not _s["active"]:
                return f(self, *a, **k)
            _structural_check_if_due()
            on_optimizer_step_if_due(self)
            roles = (_s["serving"] or {}).get("series_roles") or []
            v = _values_at(_s["step"]).get("grad_norm") if "grad_norm" in roles else None
            if not _s.get("megatron_step_noted"):
                _s["megatron_step_noted"] = True
                _note(f"Megatron optimizer step under replay: {cls.__module__}.{cls.__qualname__} paced, not run")
            return True, (float(v) if v is not None else 0.0), None
        step._gm_wrapped = True
        cls.step = step

    todo, seen = [M], set()
    while todo:
        c = todo.pop()
        if c in seen:
            continue
        seen.add(c)
        wrap(c)
        todo.extend(c.__subclasses__())
    prev = M.__dict__.get("__init_subclass__")

    def __init_subclass__(cls, **kw):
        if prev is not None:
            prev.__func__(cls, **kw)
        else:
            super(M, cls).__init_subclass__(**kw)
        wrap(cls)
    M.__init_subclass__ = classmethod(__init_subclass__)
    _note("Megatron optimizer seam installed")


_last_step_token = {"token": None}


def _structural_check_if_due() -> None:
    """Finalise the live fingerprint and send the daemon its check, once, for
    a job matched by the brief's path. The structural stage normally runs in
    an optimizer step pre-hook, but a path match engages replay at the first
    CUDA touch and the neutralised step above never calls the hooks, so for
    every path-matched job the fingerprint was never finalised and the
    daemon's `fingerprint_check` never fired (Q-prepP6-haiku-L0-nosub,
    2026-10-21: twelve path matches, no checks). Same point in the job as the
    hook: the first optimizer step, root module and optimizers known."""
    if _s.get("match") not in ("path", "surrogate_pending", "surrogate_forced_pending") or _s.get("fp_checked") or _s.get("fp_check_tried"):
        return
    _s["fp_check_tried"] = True
    try:
        import fingerprint_runtime as fr
        if fr._state.get("final") is None:
            fr._on_first_step()          # finalises, then try_lookup("structural") sends the check
        else:
            final = fr._state["final"]
            try_lookup("structural", fingerprint=final.get("fingerprint"),
                       fingerprint_inputs=final.get("inputs"))
    except Exception as e:
        _note(f"fingerprint check for the path match failed: {e!r}")


def on_optimizer_step_if_due(opt) -> None:
    """Pace once per training iteration even if the script steps several
    optimizers. The iteration epoch is advanced by backward_stub (each backward
    bumps it); the first optimizer stepped after a new epoch paces, and later
    optimizers in the same epoch return without pacing."""
    import fingerprint_runtime as fr
    epoch = fr._state.get("backward_epoch", 0)
    if _last_step_token["token"] != epoch:
        _last_step_token["token"] = epoch
        if not _verify_lr(opt, _s["step"] + 1):
            return          # replay withdrawn: this step runs for real from here
        _s["lr_now"] = _observed_lrs(opt)     # rides on this step's enqueue (the schedule the code chose)
        try:
            _s["stepped_param_ids"].update(id(p) for g in getattr(opt, "param_groups", []) or [] for p in g.get("params", []))
            if id(opt) not in _s["digested_opts"]:
                # the contents of this optimizer's parameters as the trial trains
                # them (under replay nothing but the script's own code writes them)
                _s["digested_opts"].add(id(opt))
                _s["param_digest"].update(_digest_params(_opt_params(opt)))
        except Exception:
            pass
        on_optimizer_step()


def _observed_lrs(opt) -> list:
    out = []
    for g in getattr(opt, "param_groups", []) or []:
        lr = g.get("lr")
        try:
            out.append(float(lr.item() if hasattr(lr, "item") else lr))
        except Exception:
            return []
    return out


def _verify_lr(opt, k: int) -> bool:
    """RECORD-FORMAT §5: a structural draw replays only while the rate every
    param group carries at each optimizer.step() agrees with the record's
    trace. Checked before step k paces. A mismatch on a structural match
    withdraws replay (the job continues on the modelled path, logged on both
    sides); on an exact match it is logged only, since the bytes and argv are
    the record's and the trace, not the job, is then in question. Returns
    False when replay was withdrawn by this call."""
    if not _s["active"]:
        return False
    trace = (_s["serving"] or {}).get("lr_trace")
    bad = lr_verify.mismatch(trace, k, _observed_lrs(opt))
    if bad is None:
        return True
    if _s.get("match") != "structural":
        _note(f"lr trace mismatch on an exact match, kept: {bad}")
        return True
    _disengage(f"lr mismatch at step {bad['step']} group {bad['group']}: "
               f"expected {bad['expected']:.6g}, observed {bad['observed']:.6g}")
    return False


def _opt_params(opt) -> list:
    return [p for g in getattr(opt, "param_groups", []) or [] for p in g.get("params", [])]


def _digest_params(params) -> dict:
    """A cheap content digest per parameter: numel and the sum and sum of
    squares of its first 4096 elements, read through a CPU copy of the sample
    so no replay hook touches it. A re-initialisation rewrites every element,
    so the sample is enough; a neutralised optimizer step rewrites none."""
    out = {}
    for p in params:
        try:
            flat = p.detach().reshape(-1)
            sample = flat[: min(4096, flat.numel())].to("cpu", copy=True).double()
            out[id(p)] = (int(p.numel()), float(sample.sum()), float((sample * sample).sum()))
        except Exception:
            continue
    return out


def on_new_optimizer(opt) -> None:
    """A trial boundary inside one process. Scripts of the modded-nanogpt
    kind run `num_trials = int(sys.argv[-1])` trials in a loop, each with a
    new model and a new optimizer and steps counted from 0; the replay had no
    notion of it, ran off the end of its record and served the last value
    for every later read (newton-muon on c3-12, 2026-10-22, a dead replay an
    agent can tell). The boundary is visible without reading the script: an
    optimizer constructed after at least one optimizer step, over parameters
    none of the stepped optimizers own. (A second optimizer created at start,
    Muon beside AdamW, is constructed before any step, so it never triggers;
    a late optimizer over the same parameters, a phase change, shares them.)
    Each trial is then served as a rerun of the same configuration: a new
    lookup under the job key suffixed `#t<n>`, the daemon's rerun rule (next
    unused record, siblings, the task filter, `recycled` when spent), the
    step counter reset so the new record is paced from its own step 1 with
    its start-up (a real trial recompiles its new model), events `trial: n`."""
    if not _s["active"] or _s["step"] < 1:
        return
    try:
        params = _opt_params(opt)
        new_ids = {id(p) for p in params}
    except Exception:
        return
    stepped = _s.get("stepped_param_ids") or set()
    if not new_ids or not stepped:
        return
    shared = new_ids & stepped
    if not shared:
        # a new model: the trial rebuilt it (and recompiles it), so its first
        # step pays the start-up again
        _begin_trial(rebuilt=True)
        return
    # the same parameter tensors: a trial boundary only if the script
    # re-initialised them in place (newton-muon, the modded-nanogpt family:
    # the model is built and compiled once outside the loop, `w.zero_()` /
    # `w.normal_()` per trial); an optimizer swapped in for a phase change
    # leaves the contents as they were and is no boundary
    baseline = _s.get("param_digest") or {}
    now = _digest_params([p for p in params if id(p) in shared and id(p) in baseline])
    changed = [pid for pid, d in now.items() if baseline.get(pid) != d]
    if not changed:
        return
    _note(f"trial boundary: {len(changed)} of {len(now)} shared parameters re-initialised in place")
    _begin_trial(rebuilt=False)


def _begin_trial(rebuilt: bool = True) -> None:
    n = int(_s.get("trial", 1)) + 1
    c = _client()
    if c is None:
        return
    try:
        import fingerprint_runtime as fr
        final = fr._state.get("final") or {}
    except Exception:
        final = {}
    try:
        resp = c.lookup_v2(job_key=f"{job_key()}#t{n}", rank=_rank(), world_size=_world_size(),
                           entry_hash=entry_hash(), argv=list(sys.argv), env={},
                           argv_effective=_effective_args(), argv_defaults=_arg_defaults(),
                           script_path=_script_path(),
                           fingerprint=final.get("fingerprint"), fingerprint_inputs=final.get("inputs"),
                           stage="structural" if final.get("fingerprint") else "exact", trial=n,
                           trial_rebuilt=bool(rebuilt))
    except Exception as e:
        _note(f"trial {n}: lookup failed: {e!r}")
        return
    _s["trial"] = n
    _s["stepped_param_ids"] = set()
    _s["param_digest"] = {}
    _s["digested_opts"] = set()
    # a compiled model re-initialised in place is not recompiled: its first
    # step is a steady step, not the record's compile-laden first step
    _s["skip_startup_once"] = not rebuilt
    if resp.get("mode") == "replay" and resp.get("serving"):
        _s["serving"] = resp["serving"]
        _s["record_id"] = resp.get("record_id")
        _s["match"] = resp.get("match")
        _s["step"] = 0
        _s["fp_checked"] = False
        _s["fp_check_tried"] = False
        _s["surrogate_final"] = False
        _s["has_failures"] = _live_failures(_s["serving"])
        _step_values["step"], _step_values["values"] = None, {}
        _note(f"trial {n}: record={_s['record_id']} match={_s['match']} draw={resp.get('draw')} "
              f"steady_step_s={_s['serving'].get('steady_step_s')} steps={_s['serving'].get('steps')}")
        return
    _disengage(f"trial {n}: no record to serve (mode={resp.get('mode')}, match={resp.get('match')})")


# ---------------------------------------------------------------------------
# per_example serving: MLS-Bench KV evaluations (docs/MLS-KV-SERVING.md)
# ---------------------------------------------------------------------------

def _effective_args() -> Optional[dict]:
    """The script's own parsed arguments, as {dest: value}: every argparse
    Namespace reachable from __main__'s globals or from a live frame's locals
    (a script that parses inside main() keeps it local; at engagement main()
    is on the stack). Sent with the lookup so the daemon can match a record
    whose explicit flags equal the launch's effective values — a launch that
    omits a flag at its default (`--workload gsm8k` without the job script's
    `--compression-ratio 0.8`) missed and ran the modelled path at 10× the
    served rate. None when
    the script has no Namespace (not argparse): the literal key stands."""
    try:
        import argparse
    except Exception:
        return None
    out: dict = {}

    def take(ns) -> None:
        for k, x in vars(ns).items():
            if k in out:
                continue
            if isinstance(x, bool) or x is None:
                out[k] = x
            elif isinstance(x, (str, int, float)):
                out[k] = str(x)
    try:
        main = sys.modules.get("__main__")
        for v in list(vars(main).values()) if main is not None else []:
            if isinstance(v, argparse.Namespace):
                take(v)
        f = sys._getframe(1)
        depth = 0
        while f is not None and depth < 60:
            for v in list(f.f_locals.values()):
                if isinstance(v, argparse.Namespace):
                    take(v)
            f = f.f_back; depth += 1
    except Exception:
        pass
    return out or None


def _arg_defaults() -> Optional[dict]:
    """The script's own argparse DEFAULTS, as {dest: value}, from every
    ArgumentParser reachable the way _effective_args reaches the Namespace
    (main's globals, then live frames' locals; a parser built inside main()
    is local there, and main() is on the stack at engagement). Sent with the
    lookup beside the parsed values so the daemon can accept a flag the
    launch spells out at the script's own default although the record's
    argv never named it: G-kvrep-astra-L0-r4
    launched `custom_quant_eval.py … --model-id Qwen/Qwen2.5-3B-Instruct`,
    the parser's default (`os.environ.get("MODEL_ID", …)`, resolved by the
    time the parser exists), against a record whose argv omits the flag, and
    missed; H5's agents pass it on every quant launch. A string default is
    passed through the action's type as argparse does, so it compares like
    the parsed value. None when no parser is reachable."""
    try:
        import argparse
    except Exception:
        return None
    out: dict = {}

    def norm(x):
        if isinstance(x, bool) or x is None:
            return x
        if isinstance(x, (str, int, float)):
            return str(x)
        return None

    def take(parser) -> None:
        for act in getattr(parser, "_actions", []) or []:
            dest = getattr(act, "dest", None)
            if not dest or dest in out or dest == "help":
                continue
            d = getattr(act, "default", None)
            if d is argparse.SUPPRESS:
                continue
            if isinstance(d, str) and callable(getattr(act, "type", None)):
                try:
                    d = act.type(d)
                except Exception:
                    pass
            v = norm(d)
            if v is not None or d is None:
                out[dest] = v
    try:
        main = sys.modules.get("__main__")
        for v in list(vars(main).values()) if main is not None else []:
            if isinstance(v, argparse.ArgumentParser):
                take(v)
        f = sys._getframe(1)
        depth = 0
        while f is not None and depth < 60:
            for v in list(f.f_locals.values()):
                if isinstance(v, argparse.ArgumentParser):
                    take(v)
            f = f.f_back; depth += 1
    except Exception:
        pass
    return out or None


def _process_elapsed_s() -> Optional[float]:
    """Seconds since this process started, from /proc (no patched clock): the
    script's own clock starts at its first line, so the real start-up the
    box pays before the first example (imports, the dataset, the model) is
    the load the recorded wall already contains."""
    try:
        with open("/proc/self/stat") as fh:
            fields = fh.read().rsplit(")", 1)[1].split()
        start_ticks = int(fields[19])
        with open("/proc/uptime") as fh:
            uptime = float(fh.read().split()[0])
        return max(0.0, uptime - start_ticks / os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, IndexError):
        return None


def _library_dirs() -> tuple:
    """The interpreter's own library, its site-packages and this layer's
    directories, as real-path prefixes: modules under them are not the
    project's."""
    dirs = set()
    try:
        import sysconfig
        for k in ("stdlib", "platstdlib", "purelib", "platlib"):
            p = sysconfig.get_paths().get(k)
            if p:
                dirs.add(p)
    except Exception:
        pass
    try:
        import site
        dirs.update(site.getsitepackages())
        dirs.add(site.getusersitepackages())
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    dirs.update({here, os.path.join(os.path.dirname(here), "pylib")})
    dirs = {d for d in dirs if d}
    return tuple({os.path.join(d, "") for d in dirs} | {os.path.join(os.path.realpath(d), "") for d in dirs})


def _project_namespaces() -> list:
    """(globals dict, real path) of each project module whose code is
    running now: the modules of the frames on this thread's stack whose
    __file__ is a .py outside the interpreter's library, site-packages and
    this layer, the entry script excepted. At the first CUDA touch of an
    agent's runner that imported the evaluator and called its main(), the
    evaluator's frames are there; a script that only imported the evaluator
    for a helper (its loader, its scorer) and does its own CUDA work has none
    of the evaluator's frames on the stack, and is not offered as it."""
    lib = _s.get("lib_dirs")
    if lib is None:
        lib = _s["lib_dirs"] = _library_dirs()
    entry = _entry_path()
    out, seen = [], set()
    try:
        f = sys._getframe(1)
    except Exception:
        return out
    depth = 0
    while f is not None and depth < 200 and len(out) < 64:
        g = f.f_globals
        f = f.f_back
        depth += 1
        if id(g) in seen:
            continue
        seen.add(id(g))
        fn = g.get("__file__")
        if not isinstance(fn, str) or not fn.endswith(".py") or fn.startswith(lib):
            continue
        try:
            rp = os.path.realpath(fn if os.path.isabs(fn) else os.path.join(_launch_cwd(), fn))
        except Exception:
            continue
        if rp.startswith(lib) or (entry and rp == entry) or not os.path.isfile(rp):
            continue
        out.append((g, rp))
    return out


def _file_sha256(path: str) -> str | None:
    cache = _s.setdefault("mod_hash", {})
    try:
        st = os.stat(path)
        if st.st_size > 4 * 1024 * 1024:
            return None
        k = (path, st.st_mtime_ns, st.st_size)
        if k not in cache:
            with open(path, "rb") as fh:
                cache[k] = hashlib.sha256(fh.read()).hexdigest()
        return cache[k]
    except OSError:
        return None


def _loaded_modules() -> list:
    """The running project modules for the lookup: name, real path, sha256."""
    out = []
    try:
        for g, rp in _project_namespaces():
            h = _file_sha256(rp)
            if h:
                out.append({"name": str(g.get("__name__")), "path": rp, "sha256": h})
    except Exception as e:
        _note(f"loaded modules not listed: {e!r}")
    return out


def _module_namespace(path: str):
    """The globals dict of the loaded module whose file is `path`."""
    for g, rp in _project_namespaces():
        if rp == path:
            return g
    return None


def _original_functions(g: dict, names: list, path: str) -> dict:
    """{name: the function object `name` that the module with globals `g`
    defined}: the module's own attribute while it is still that function,
    else the one the agent's runner captured before replacing it (a wrapper
    that calls the original), found among the functions whose globals are
    `g`."""
    base = os.path.basename(path)

    def ours(f, name) -> bool:
        try:
            return isinstance(f, types.FunctionType) and f.__globals__ is g and f.__name__ == name \
                and os.path.basename(f.__code__.co_filename) == base
        except Exception:
            return False
    out = {n: g.get(n) for n in names if ours(g.get(n), n)}
    if len(out) < len(names):
        import gc
        for r in gc.get_referrers(g):
            for n in names:
                if n not in out and ours(r, n):
                    out[n] = r
    return out


_REDIRECTS: dict = {}
_REDIRECT_ORIG: dict = {}      # id(f) -> (f, its code, defaults, kwdefaults) before _redirect


def _unredirect(f) -> None:
    """Undo _redirect: f runs its own code again (a refusal at the first served unit)."""
    t = _REDIRECT_ORIG.pop(id(f), None)
    if t is None:
        return
    _, code, d, kd = t
    f.__code__ = code
    f.__defaults__ = d
    f.__kwdefaults__ = kd
    try:
        del f.__signature__
    except AttributeError:
        pass


_BYTES_MATCHES = ("exact", "exact_effective", "exact_prefix", "exact_module", "exact_effective_module")


def _inproc_check(extra: list, own_ids: set = frozenset(), own_objs: set = frozenset()) -> str | None:
    """The in-process patch check (next/ r8.9) over `extra` [(globals, path)] and
    every bound file loaded into this process. Only for a record that carries
    serving.module_shas (gpu-tasks' all-paths field; {} included)."""
    sv = _s["serving"] or {}
    ms = sv.get("module_shas")
    if not isinstance(ms, dict):
        return None
    spaces = list(extra)
    seen = {os.path.realpath(p) for _, p in spaces if p}
    for path in _bound_files(ms, _ancestors(_s.get("entry_module"))):
        if path in seen:
            continue
        g = _loaded_namespace_of(path)
        if g is not None:
            spaces.append((g, path))
            seen.add(path)
    own = set(own_objs) | set(_s.get("own_objs") or ())
    return _inproc_mismatch(spaces, own_ids=own_ids, own_objs=own)


def _redirect(f, to) -> bool:
    """Make every reference to the function object `f` call `to`: `f` keeps
    its identity, name, signature and source, and its code becomes a call
    through. A runner that captured `f` before it engaged replay, and wraps
    it, then calls `to`."""
    if not isinstance(f, types.FunctionType) or f.__closure__:
        return False
    try:
        import inspect
        sig = inspect.signature(f)
        k = len(_REDIRECTS)
        _REDIRECTS[k] = to
        ns: dict = {}
        exec(compile(f"def {f.__name__}(*a, **k):\n    return __import__('sys').modules[{__name__!r}]._REDIRECTS[{k}](*a, **k)\n",
                     f.__code__.co_filename, "exec"), ns)
        code = ns[f.__name__].__code__.replace(co_firstlineno=f.__code__.co_firstlineno, co_qualname=f.__code__.co_qualname)
        _REDIRECT_ORIG[id(f)] = (f, f.__code__, f.__defaults__, f.__kwdefaults__)
        f.__code__ = code
        f.__defaults__ = None
        f.__kwdefaults__ = None
        f.__signature__ = sig
        return True
    except Exception as e:
        _note(f"per_example: {getattr(f, '__name__', f)!r} not redirected: {e!r}")
        return False


def _example_key(example) -> str | None:
    """An example's content key, as the example-order tables hold it:
    sha256 of its example_id, a NUL and its prompt."""
    try:
        return hashlib.sha256((str(example["example_id"]) + "\0" + example["prompt"]).encode("utf-8")).hexdigest()
    except Exception:
        return None


class _DictNamespace:
    """getattr/setattr over a plain globals dict, so a seam installed in an
    exec'd script's dict is seen by its own global-name calls."""
    def __init__(self, g: dict):
        object.__setattr__(self, "_g", g)

    def __getattr__(self, name):
        try:
            return self._g[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self._g[name] = value


def _seam_namespace(gen_name, score_name):
    """Where the evaluator's seam functions live: the `__main__` module when
    the script is run as a script (the usual case, and the only one in 3ec0),
    else the globals dict of a calling frame that defines both names and whose
    `__file__` is the entry script, as when an agent execs the script's source
    into its own dict (`exec(compile(src, path, "exec"), g)`)."""
    main = sys.modules.get("__main__")
    if gen_name and score_name and hasattr(main, gen_name) and hasattr(main, score_name):
        return main
    entry = _entry_path() or ""
    try:
        f = sys._getframe(1)
    except Exception:
        f = None
    depth = 0
    while f is not None and depth < 80:
        g = f.f_globals
        if gen_name in g and score_name in g and g is not getattr(main, "__dict__", None):
            fn = g.get("__file__") or ""
            try:
                same = bool(entry) and os.path.realpath(fn) == os.path.realpath(entry)
            except Exception:
                same = False
            if same or not entry:
                return _DictNamespace(g)
        f = f.f_back
        depth += 1
    return None


def _install_per_example_seams() -> None:
    """Replace, by name in the script's __main__, the generate function and the
    per-example scorer the record's serving block names. The script's loop
    calls both through their global names, so the replacements run: the
    generate seam returns a placeholder prediction and the trace the script
    aggregates (selection: the retained fraction the compressor would have
    computed from the real prompt length; quant: a positive efficiency the
    script discards), and paces one example; the score seam returns the i-th
    entry of the record's per-example score vector from the daemon. The
    decode loop never runs. Missing seams are noted and nothing is replaced."""
    import inspect
    pe = (_s["serving"] or {}).get("per_example") or {}
    gen_name, score_name = pe.get("generate_seam"), pe.get("score_seam")
    em = _s.get("entry_module")
    if em:
        # the recorded evaluator is a module the agent's runner imported and
        # whose main() it called (next/ r4): its seams are redirected in place,
        # so the runner's own wrappers around them call the served seams
        g = _module_namespace(em)
        main = _DictNamespace(g) if g is not None else None
        fns = _original_functions(g, [n for n in (gen_name, score_name) if n], em) if g is not None else {}
        gen, score = fns.get(gen_name), fns.get(score_name)
        _s["pe_timer_module"] = g.get("__name__") if g is not None else None
    else:
        main = _seam_namespace(gen_name, score_name)
        gen = getattr(main, gen_name, None) if (main is not None and gen_name) else None
        score = getattr(main, score_name, None) if (main is not None and score_name) else None
    if gen is None or score is None:
        # an engaged per_example job whose seams cannot be replaced would run
        # the evaluator's real decode against the replay forward stub and crash
        # (G-kvreph-haiku-L0-m3, 2026-10-27: the agent exec'd the script's source
        # into its own dict; AttributeError at outputs.past_key_values): give the
        # record back and let the job run on the modelled path instead
        _note(f"per_example: seams not found (generate={gen_name!r} score={score_name!r}); replay withdrawn")
        _disengage("per_example_seams_not_found")
        return
    try:
        gen_sig = inspect.signature(gen)
    except (TypeError, ValueError):
        gen_sig = None
    try:
        score_sig = inspect.signature(score)
    except (TypeError, ValueError):
        score_sig = None
    quant = "quant" in (gen_name or "").lower()
    # next/ r8.9: the evaluator's own functions, unpatched (analysis ruling (A)):
    # the namespace the seams live in and the file it was compiled from
    if em:
        pe_space = (g, em)
    elif isinstance(main, _DictNamespace):
        pe_space = (main._g, (main._g.get("__file__") or _entry_path()))
    else:
        pe_space = (getattr(main, "__dict__", None), _entry_path())
    own_ids = {id(gen), id(score)}
    bad = _inproc_check([pe_space], own_ids=own_ids)
    if bad is not None:
        _note(f"per_example: not served: {bad}; replay withdrawn, the job runs its own code")
        _disengage("identity_inprocess_patch", detail=bad)
        return

    def gen_seam(*a, **k):
        bound = {}
        if gen_sig is not None:
            try:
                bound = dict(gen_sig.bind_partial(*a, **k).arguments)
            except TypeError:
                bound = {}
        if _s["pe_gen_index"] == 0 and not _s.get("pe_rechecked"):
            # again at the first served example: a runner may patch after main() starts
            _s["pe_rechecked"] = True
            late2 = _identity_any_path()
            reason = late2[0] if late2 else "identity_inprocess_patch"
            late = late2[1] if late2 else _inproc_check([pe_space], own_ids=own_ids, own_objs=set(_s.get("pe_seam_objs") or ()))
            if late is not None:
                _note(f"per_example: not served: {late}; replay withdrawn before the first example, the job runs its own code")
                if em:
                    _unredirect(gen); _unredirect(score)
                else:
                    try:
                        setattr(main, gen_name, gen); setattr(main, score_name, score)
                    except Exception:
                        pass
                _disengage(reason, detail=late)
                return gen(*a, **k)
        i = _s["pe_gen_index"]
        _s["pe_gen_index"] += 1
        if _s.get("pe_load_s") is None:
            # the launch's start-up up to the first example, measured from the
            # script's own timer (its first perf_counter() in __main__, the
            # evaluator's eval_start) ON THE BOX'S CLOCK, the clock the script's
            # runtime_seconds runs on, so that runtime = startup + k × pace holds
            # exactly and a full launch prints the recorded wall exactly (the
            # real-clock figure differs: the fake-GPU part of the load is charged
            # at its modelled time; gpu-tasks found the 0.9 s gap, 2026-10-22
            # 20:00Z). Fallbacks (no __main__ timer): process start, engagement,
            # both real. It stands in for the recorded machine's start-up inside
            # the recorded wall.
            t_main = vc.first_main_perf_counter_real(); v_main = vc.first_main_perf_counter_virtual()
            fm = vc.first_perf_counter_of(_s["pe_timer_module"]) if _s.get("pe_timer_module") else None
            vc.stop_module_timers()
            if fm:
                t_main, v_main = fm             # the evaluator's own timer, in its module
            real = None
            if t_main is not None:
                real = max(0.0, vc.real_perf_counter() - t_main)
                elapsed = max(0.0, vc.get_clock().now_monotonic() - v_main) if v_main is not None else real
                how = "since the script's own timer started"
            else:
                elapsed = _process_elapsed_s(); how = "since process start"
                if elapsed is None and _s.get("pe_t_engaged") is not None:
                    elapsed = max(0.0, _REAL_MONO() - _s["pe_t_engaged"]); how = "since engagement"
                real = elapsed
            _s["pe_load_s"] = float(elapsed or 0.0)
            _s["pe_load_real_s"] = float(real if real is not None else (elapsed or 0.0))
            _note(f"per_example: start-up took {_s['pe_load_s']:.1f} s on the box's clock ({_s['pe_load_real_s']:.1f} s real) {how}; pacing from here")
        _pace_example(i, _example_key(bound.get("example")))
        return "", (_quant_trace(bound) if quant else _selection_trace(bound, main))

    def score_seam(*a, **k):
        i = _s["pe_score_index"]
        _s["pe_score_index"] += 1
        key = None
        if score_sig is not None:
            try:
                key = _example_key(score_sig.bind_partial(*a, **k).arguments.get("example"))
            except TypeError:
                key = None
        c = _s["client"]
        v = None
        if c is not None:
            try:
                v = c.materialise("example_score", i, key=key)
            except Exception as e:
                _note(f"per_example: score {i} not served: {e!r}")
        return float(v) if v is not None else 0.0

    if em:
        if not (_redirect(gen, gen_seam) and _redirect(score, score_seam)):
            # a runner may hold the originals, and would run the real decode
            # against the replay forward stub: give the record back instead
            _disengage("per_example_seams_not_redirected")
            return
    else:
        setattr(main, gen_name, gen_seam)
        setattr(main, score_name, score_seam)
    _s["pe_seam_objs"] = {id(gen_seam), id(score_seam)}
    _note(f"per_example: seams installed ({gen_name}, {score_name}){' in ' + em if em else ''}; "
          f"n_full={pe.get('n_examples_full')} kind={pe.get('score_kind')}")


# ---------------------------------------------------------------------------
# whole_job serving (next/ r8)
# ---------------------------------------------------------------------------
# A job that runs a recorded script whose source records only its total duration
# and its final result (an MTEB task, an MLS-Bench non-KV evaluation, a cleanrl
# run's return): no per-step or per-example series. The record's serving block
# names a seam, `<module>:<qualname>` (`__main__` for the script itself), and the
# unit plan: n_units units, each costing its share of the recorded wall
# (unit_shares, or equal shares). The seam's replacement paces the units on the
# job's GPU, then fetches the recorded result through `materialise` (the daemon
# serves it only after every unit's GPU time has passed) and returns it in the
# shape the original returned (result_shape). The work inside the original never
# runs. A job killed or cut short before its last unit gets no result, as on
# hardware; its draw goes back (gpu_daemon._end_job).

def _resolve_qualname(root, qualname: str):
    """(owner, attribute name, the function) for `A.B.c` under `root`, or None."""
    parts = [x for x in qualname.split(".") if x]
    if not parts:
        return None
    owner = root
    for part in parts[:-1]:
        owner = getattr(owner, part, None)
        if owner is None:
            return None
    fn = owner.__dict__.get(parts[-1]) if hasattr(owner, "__dict__") else None
    if fn is None:
        fn = getattr(owner, parts[-1], None)
    return (owner, parts[-1], fn) if fn is not None else None


# live_fields (next/ r8.3): allowlisted LIVE instrument getters a whole_job record
# may merge into its served dict for a field the CSV never recorded. Each returns
# what the PATCHED torch call returns in the seamed context (so a live field inside
# the seam == the same call outside it). The recorded result always wins: a live
# field only fills a key the recorded dict does not already carry.
def _live_field_value(spec, seam_elapsed_s):
    """Resolve one live_fields spec to its current instrument value. `spec` is a
    string naming an allowlisted getter; `seam_elapsed_s` is the real wall the
    seam's pacing took (for a timer field)."""
    s = str(spec)
    try:
        if s in ("torch.cuda.max_memory_allocated", "max_memory_allocated"):
            import torch
            return int(torch.cuda.max_memory_allocated())
        if s in ("torch.cuda.max_memory_reserved", "max_memory_reserved"):
            import torch
            return int(torch.cuda.max_memory_reserved())
        if s in ("torch.cuda.memory_allocated", "memory_allocated"):
            import torch
            return int(torch.cuda.memory_allocated())
        if s in ("seam_elapsed_s", "elapsed_s", "seam_wall_s"):
            return float(seam_elapsed_s)
        if s in ("recorded_peak_bytes", "modeled_peak_bytes", "held_peak_bytes"):
            # The modeled peak the record holds on the GPU at engagement (weights +
            # cache), IN BYTES, for a whole_job job where replay disabled per-tensor
            # tracking so torch.cuda.max_memory_allocated reads 0. This is the
            # instrument's modeled figure (NOT a per-example recorded vector); the
            # same _recorded_mem_bytes nvidia-smi 'used' shows for the job (analysis
            # ruling 2026-11-02). Use the _mb variant to match a field printed in MB.
            return int(_recorded_mem_bytes())
        if s in ("recorded_peak_mb", "modeled_peak_mb", "held_peak_mb"):
            # Same modeled peak, in MiB (bytes / 1024**2), to match a source field
            # like DLM's peak_memory_mb = max_memory_allocated / (1024**2). Binding
            # the _bytes getter to an MB field inflates it 1,048,576x (K2, 2026-11-02).
            return float(_recorded_mem_bytes()) / (1024.0 ** 2)
    except Exception as e:
        _note(f"whole_job live_field {s!r}: not read: {e!r}")
    return None


def _merge_live_fields(wj: dict, result, seam_elapsed_s):
    """Merge each live_fields entry into `result` (a dict) for keys the recorded
    result does not already carry. No recorded field is ever overridden."""
    live = (wj or {}).get("live_fields")
    if not isinstance(live, dict) or not isinstance(result, dict):
        return result
    out = dict(result)
    for key, spec in live.items():
        if key in out and out[key] is not None:
            # a recorded field wins; never overridden by a live one
            continue
        v = _live_field_value(spec, seam_elapsed_s)
        if v is not None:
            out[key] = v
    return out


def _whole_job_result_value(wj: dict, value, seam_elapsed_s=0.0):
    """The recorded result in the shape the seam's original returns:
    `value` (as served), `dict` (a dict, None as {}), `float`, or `none` (the
    original returns nothing; the result reaches the output another way).
    live_fields are merged into a dict result (recorded keys win)."""
    shape = (wj or {}).get("result_shape") or "value"
    if shape == "none":
        return None
    if shape == "dict":
        d = dict(value) if isinstance(value, dict) else {}
        return _merge_live_fields(wj, d, seam_elapsed_s)
    if shape == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    # "value": merge live fields when the served value is itself a dict
    if isinstance(value, dict):
        return _merge_live_fields(wj, value, seam_elapsed_s)
    return value


def _pace_whole_job(wj: dict) -> None:
    """Pace the record's units, each its share of the recorded wall: the job's
    GPU is busy for the recorded time inside the seam. Nothing is subtracted for
    start-up: a whole_job record's wall_total_s is the time the seamed function
    took (MTEB's evaluation_time excludes the loads), and everything outside the
    seam (imports, data and model loads) runs for real on the box."""
    sv = _s["serving"] or {}
    try:
        wall = float(sv.get("wall_total_s"))
    except (TypeError, ValueError):
        wall = 0.0
    try:
        n = max(1, int(wj.get("n_units") or 1))
    except (TypeError, ValueError):
        n = 1
    shares = wj.get("unit_shares")
    if not (isinstance(shares, list) and len(shares) == n):
        shares = [1.0 / n] * n
    c = _s["client"]
    for i in range(n):
        per = max(0.0, wall * float(shares[i]))
        if i == 0:
            vc.get_clock().anchor_wall_floor()
        if per > 0:
            vc.get_clock().advance_recorded(per)
        if c is None:
            continue
        mem = 0
        if not _s.get("mem_reported"):
            _s["mem_reported"] = True
            mem = max(0, _recorded_mem_bytes() - int(_s.get("mem_bytes") or 0))
            _s["mem_bytes"] = int(_s.get("mem_bytes") or 0) + mem
        try:
            resp = c.enqueue(gpu=_daemon_gpu(), op="replay", modelled_s=per, mem_delta=mem, step=i + 1,
                             unit="whole_job", oom_aware=bool(mem))
        except Exception as e:
            _note(f"whole_job: pacing rpc failed: {e!r}")
            return
        if mem and resp.get("oom") and resp.get("oom_detail"):
            _raise_oversubscribed(resp["oom_detail"])
        cursor = resp.get("cursor")
        if cursor is not None:
            wait = cursor - _REAL_MONO()
            if wait > 0:
                _gpu_wait(wait)
            _last_cursor["t"] = cursor
        _check_failure(step=i + 1)


# Keys of identity.trajectory.environment that record provenance, not an installed
# package: a source or task commit, the package name, the OS, the interpreter.
_WJ_PROVENANCE_KEYS = ("package", "os", "python")


def _whole_job_identity_mismatch(wj: dict, module) -> tuple[str, str] | None:
    """Whether this launch is the configuration a whole_job record was recorded
    under (next/ r8.5, the study 2026-11-02). Checked HERE, in the job's own
    interpreter, because the daemon is a sidecar that cannot see the job's env, its
    installed packages or the module file it loaded. Returns (reason, detail) for
    the first mismatch, None when it is the recorded configuration:
      env      each identity.env key not in env_exempt: an unset var is its
               declared value (the record states what the source does with it
               unset); a set var must equal the declared string exactly (an empty
               string is set, and refused: refusing is never a wrong result).
      runtime  each identity.trajectory.environment key that names a package is
               compared with importlib.metadata.version() of the INSTALLED
               distribution — never an env marker, which an inherited environment
               could carry into another interpreter. A package that is not
               installed is a mismatch. Commit keys (*_commit) and package/os/
               python are provenance and are not checked here.
      module   serving.seam_sha256, when the record carries it, must equal the
               sha256 of the seam module's file as loaded.
      deps     serving.module_shas {task-root-relative path: sha256}: each file is
               hashed ON DISK, imported yet or not, so a lazily imported file is
               bound too (r8.5.1).
    A record without these fields is not checked on them (r8 records serve as
    before)."""
    idt = (wj or {}).get("identity") or {}
    env = idt.get("env") or {}
    exempt = set(idt.get("env_exempt") or [])
    for k in sorted(env):
        if k in exempt:
            continue
        if k in os.environ and os.environ[k] != str(env[k]):
            return ("whole_job_identity_env",
                    f"env {k}={os.environ[k]!r}, the record was made with {k}={str(env[k])!r}")
    pins = idt.get("environment") or {}
    if pins:
        import importlib.metadata as _md
        for k in sorted(pins):
            if k.endswith("_commit") or k in _WJ_PROVENANCE_KEYS:
                continue
            want = str(pins[k])
            try:
                have = _md.version(k)
            except Exception:
                have = None
            if have != want:
                return ("whole_job_identity_runtime",
                        f"{k} {have if have is not None else 'not installed'}, the record was made with {k} {want}")
    want_sha = idt.get("seam_sha256")
    if want_sha:
        path = getattr(module, "__file__", None)
        have_sha = _file_sha256(path) if path else None
        if have_sha != str(want_sha):
            return ("whole_job_identity_module",
                    f"seam module {path or getattr(module, '__name__', '?')} sha256 "
                    f"{(have_sha or 'unreadable')[:16]}, the record was made with {str(want_sha)[:16]}")
    # next/ r8.5.1: the project files the job's code runs, beyond the seam module.
    # The file an agent edits need not be the seam module (sparse: the method lives
    # in custom_sparse_attn, imported by a seam module byte-identical across
    # methods), and some are imported LAZILY, after seam install (DLM: dllm_cache
    # hooks and the LLaDA modeling files load inside the generation path). So the
    # record binds FILES, not loaded modules: serving.module_shas maps a path
    # relative to the task root to its sha256, and each file is hashed on disk
    # here, imported or not. Missing or different ->
    # the job runs its own code. A path is resolved against the task root: the
    # directory the launch was started in (the launcher cd's there), else the
    # nearest ancestor of the entry script that holds the declared file.
    deps = idt.get("module_shas") or {}
    if deps:
        bad = _module_shas_mismatch(deps)
        if bad is not None:
            return ("whole_job_identity_module_dep", bad)
    return None


def _module_shas_mismatch(deps: dict, extra_roots: tuple = ()) -> str | None:
    """The first file of a record's module_shas {task-root-relative path: sha256}
    that is missing or differs on disk, as a detail line; None when all match.
    Each file is hashed on disk, imported yet or not (r8.5.1). Roots: the launch
    cwd, the entry script's ancestors, then `extra_roots` (r8.8: a module-path
    launch's matched evaluator's ancestors, for a runner outside the task root)."""
    roots = _task_roots()
    for r in extra_roots:
        if r and r not in roots:
            roots.append(r)
    for rel in sorted(deps):
        want = str(deps[rel])
        found = None
        if os.path.isabs(rel):
            found = rel if os.path.isfile(rel) else None
        else:
            for r in roots:
                p = os.path.join(r, rel)
                if os.path.isfile(p):
                    found = p
                    break
        have = None
        if found:
            # no size cap here (_file_sha256 skips files over 4 MB, which would
            # refuse a large bound file on every launch)
            try:
                hh = hashlib.sha256()
                with open(found, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        hh.update(chunk)
                have = hh.hexdigest()
            except OSError:
                have = None
        if have != want:
            where = found or "missing under " + (", ".join(roots) or "no task root")
            return (f"file {rel} ({where}) sha256 {(have or 'unreadable')[:16]}, "
                    f"the record was made with {want[:16]}")
    return None


def _identity_any_path() -> tuple[str, str] | None:
    """next/ r8.9, the helper-dependency bind on every serving path (analysis GO
    2026-11-02; gpu-tasks' records, sync 15). Only for a record that carries
    serving.module_shas (a dict; {} = the entry imports no staged project file);
    a record without the field is not checked (r8 behaviour).
      dep     each declared file hashed on disk, imported yet or not; a whole_job
              record's files are checked at its seam install instead, with its own
              reason (r8.5.1)
      shadow  a project module loaded under the name of a stdlib module, or of an
              installed package that the entry or a bound file imports, whose file
              the record does not bind (a `sched.py` or `datasets.py` beside the
              runner): the evaluator's computation could be running it
    Returns (reason, detail) for the first mismatch, else None."""
    sv = _s["serving"] or {}
    ms = sv.get("module_shas")
    if not isinstance(ms, dict):
        return None
    extra = _ancestors(_s.get("entry_module"))
    if ms and not sv.get("whole_job"):
        bad = _module_shas_mismatch(ms, extra)
        if bad is not None:
            return ("identity_module_dep", bad)
    bad = _shadow_mismatch(set(_bound_files(ms, extra)), extra)
    if bad is not None:
        return ("identity_module_shadow", bad)
    return None


def _imported_top_names(paths) -> set:
    """Top-level module names the files at `paths` import (static)."""
    import ast
    cache = _s.setdefault("imports_cache", {})
    out: set = set()
    for path in paths:
        if not path:
            continue
        if path not in cache:
            names = set()
            try:
                with open(path, "rb") as fh:
                    tree = ast.parse(fh.read())
                for n in ast.walk(tree):
                    if isinstance(n, ast.Import):
                        names.update(a.name.split(".")[0] for a in n.names)
                    elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                        names.add(n.module.split(".")[0])
            except Exception:
                pass
            cache[path] = names
        out |= cache[path]
    return out


def _editable_roots() -> list:
    """(top-level names, source root) of every editable distribution whose metadata
    is installed in a SYSTEM site-packages directory (next/ r8.9.1). A `pip install
    -e` package is installed in site-packages as a .pth or finder entry, while its
    files live in its source checkout, outside the library dirs; the shadow check read
    the TTC image's editable TabArena as a project file shadowing the `tabarena`
    distribution, which it IS (Sol's installed-box probe, 2026-11-03 05:37Z, a false
    refusal). The root is the distribution's own direct_url.json (dir_info.editable).
    A distribution in the user site (an agent's `pip install --user -e .`) does not
    count. Cached per process."""
    got = _s.get("editable_roots")
    if got is not None:
        return got
    out = []
    try:
        import importlib.metadata as _md
        import json as _json
        import site as _site
        import sysconfig as _sc
        from urllib.parse import unquote, urlparse
        system = set()
        for d in list(_site.getsitepackages()) + [_sc.get_paths().get(k) for k in ("purelib", "platlib")]:
            if d:
                system.add(os.path.join(os.path.realpath(d), ""))
        for d in _md.distributions():
            try:
                du = d.read_text("direct_url.json")
                if not du:
                    continue
                info = _json.loads(du)
                if not (info.get("dir_info") or {}).get("editable"):
                    continue
                u = urlparse(str(info.get("url") or ""))
                if u.scheme != "file":
                    continue
                meta = getattr(d, "_path", None)
                meta = os.path.realpath(str(meta)) if meta is not None else ""
                if not any(meta.startswith(sd) for sd in system):
                    continue          # user-site metadata: an install the job itself made
                names = set((d.read_text("top_level.txt") or "").split())
                out.append((names, os.path.join(os.path.realpath(unquote(u.path)), "")))
            except Exception:
                continue
    except Exception as e:
        _note(f"editable distributions not read: {e!r}")
    _s["editable_roots"] = out
    return out


def _editable_installed(name: str, rp: str) -> bool:
    """Whether the module `name` loaded from `rp` is an installed editable
    distribution's own file: rp lies under the source root of an editable
    distribution installed in system site-packages, which names `name` as a
    top-level package when it names any; rp is not under the entry script's
    directory; and, for a job that is not root, neither rp nor its directory is
    owned by the job's user (the image's checkouts are root-owned; a file the agent
    made is its own, whatever its mode)."""
    ed = os.path.dirname(_entry_path() or "")
    if ed and ed != "/" and rp.startswith(os.path.join(os.path.realpath(ed), "")):
        return False
    try:
        me = os.geteuid()
        if me != 0 and (os.stat(rp).st_uid == me or os.stat(os.path.dirname(rp)).st_uid == me):
            return False
    except Exception:
        return False
    for names, root in _editable_roots():
        if rp.startswith(root) and (not names or name in names):
            return True
    return False


def _shadow_mismatch(bound: set, extra_roots: tuple = ()) -> str | None:
    lib = _s.get("lib_dirs")
    if lib is None:
        lib = _s["lib_dirs"] = _library_dirs()
    stdlib = set(getattr(sys, "stdlib_module_names", ()) or ())
    entry = _entry_path()
    em = _s.get("entry_module")
    installed = wanted = None
    for name, m in list(sys.modules.items()):
        if "." in name or m is None:
            continue
        fn = getattr(m, "__file__", None)
        if not isinstance(fn, str):
            continue
        try:
            rp = os.path.realpath(fn)
        except Exception:
            continue
        if rp.startswith(lib) or rp in bound or (entry and rp == entry) or (em and rp == os.path.realpath(em)):
            continue
        if _editable_installed(name, rp):
            continue              # the installed (editable) distribution itself, not a shadow of it
        if name not in stdlib:
            if installed is None:
                try:
                    import importlib.metadata as _md
                    installed = set(_md.packages_distributions())
                except Exception:
                    installed = set()
                wanted = _imported_top_names([entry, em] + sorted(bound))
            if not (name in installed and name in wanted):
                continue
        what = "the standard library's" if name in stdlib else "the installed package's"
        return f"project file {rp} is loaded as {name!r}, shadowing {what} {name!r}; the record does not bind it"
    return None


def _ancestors(path: str | None) -> tuple:
    out = []
    if path:
        d = os.path.dirname(os.path.realpath(path))
        while True:
            out.append(d)
            nd = os.path.dirname(d)
            if nd == d:
                break
            d = nd
    return tuple(out)


def _bound_files(module_shas: dict, extra_roots: tuple = ()) -> list:
    """Resolved on-disk paths of a record's module_shas files that exist (for the
    in-process check of what they define)."""
    roots = _task_roots() + [r for r in extra_roots if r]
    out = []
    for rel in sorted(module_shas or {}):
        if os.path.isabs(rel):
            if os.path.isfile(rel):
                out.append(os.path.realpath(rel))
            continue
        for r in roots:
            p = os.path.join(r, rel)
            if os.path.isfile(p):
                out.append(os.path.realpath(p))
                break
    return out


def _inproc_mismatch(namespaces: list, own_ids: set = frozenset(), own_objs: set = frozenset()) -> str | None:
    """next/ r8.8, analysis ruling (A) 2026-11-02 17:42: every function and method a
    bound file defines must still be, in the namespace it was loaded into, the
    original or a wrapper whose code can call it (inproc_bind). `namespaces`:
    [(globals dict, real path of the file)]. The first mismatch as a detail line."""
    try:
        import inproc_bind
    except Exception as e:
        _note(f"in-process check unavailable: {e!r}")
        return None
    for g, path in namespaces:
        if g is None or not path:
            continue
        try:
            bad = inproc_bind.mismatch(g, path, own_ids=own_ids, own_objs=own_objs)
        except Exception as e:
            _note(f"in-process check of {path} failed: {e!r}")
            bad = None
        if bad:
            return bad
    return None


def _loaded_namespace_of(path: str):
    """The globals dict of a loaded module whose file is `path` (any module in
    sys.modules, not only those with frames on the stack)."""
    rp = os.path.realpath(path)
    for m in list(sys.modules.values()):
        fn = getattr(m, "__file__", None)
        if not isinstance(fn, str):
            continue
        try:
            if os.path.realpath(fn) == rp:
                return vars(m)
        except Exception:
            continue
    return None


def _task_roots() -> list:
    """Directories a record's task-root-relative path may be resolved against, in
    order: the directory the launch was started in, then each ancestor of the
    entry script (a launch from elsewhere still finds its own tree)."""
    out = []
    lc = _launch_cwd()
    if lc:
        out.append(os.path.realpath(lc))
    ep = _entry_path()
    if ep:
        d = os.path.dirname(os.path.realpath(ep))
        while True:
            if d not in out:
                out.append(d)
            nd = os.path.dirname(d)
            if nd == d:
                break
            d = nd
    return out


def _install_whole_job_seam() -> None:
    """Replace the function the record's whole_job block names with one that
    paces the record's units and returns its recorded result. `__main__:name`
    is the script's own function; any other module is replaced when it is
    imported (or now, if it already is). A seam that cannot be found withdraws
    replay, so the job runs on the modelled path instead of half-served."""
    wj = (_s["serving"] or {}).get("whole_job") or {}
    seam = str(wj.get("seam") or "")
    mod, _, qual = seam.partition(":")
    if not mod or not qual:
        _note(f"whole_job: seam {seam!r} is not <module>:<qualname>; replay withdrawn")
        _disengage("whole_job_seam_malformed")
        return

    wj_state: dict = {}

    def replacement(*a, **k):
        if not wj_state.get("rechecked"):
            # again at the first unit: a runner may patch after the seam module loaded
            wj_state["rechecked"] = True
            late2 = _identity_any_path()
            reason = late2[0] if late2 else "identity_inprocess_patch"
            late = late2[1] if late2 else (_inproc_check([wj_state["space"]], own_ids=wj_state.get("own_ids") or set(),
                                                         own_objs=wj_state.get("own_objs") or set()) if wj_state.get("space") else None)
            if late is not None:
                _note(f"whole_job: {seam} not served: {late}; replay withdrawn before the first unit, the job runs its own code")
                try:
                    wj_state["undo"]()
                except Exception:
                    pass
                _disengage(reason, detail=late)
                return wj_state["orig"](*a, **k)
        _t_seam0 = _REAL_MONO()
        _pace_whole_job(wj)
        seam_elapsed_s = _REAL_MONO() - _t_seam0   # the real wall the paced seam took (for a live elapsed field)
        c = _s["client"]
        v = None
        if c is not None:
            try:
                v = c.materialise("whole_job_result", _s["step"])
            except Exception as e:
                _note(f"whole_job: result not served: {e!r}")
        _note(f"whole_job: {seam} served its recorded result after {wj.get('n_units')} unit(s)")
        return _whole_job_result_value(wj, v, seam_elapsed_s)

    def running(code) -> bool:
        """Whether a call of the seam's function is on this thread's stack now.
        Replay engages at the job's first CUDA use; if that use is inside the
        seamed function, the call already running keeps its own code (a frame
        is not re-pointed by _redirect) while replay stubs its forwards, so it
        would run half-served. That case is withdrawn instead."""
        try:
            f = sys._getframe(1)
        except Exception:
            return False
        while f is not None:
            if f.f_code is code:
                return True
            f = f.f_back
        return False

    def install(module) -> bool:
        found = _resolve_qualname(module, qual)
        if found is None:
            return False
        bad = _whole_job_identity_mismatch(wj, module)
        if bad is not None:
            # next/ r8.5: this launch is not the recorded configuration, so the
            # recorded aggregate is not its result. Leave the seam alone: the job
            # runs its own code on the modelled path, as a launch with no record.
            reason, detail = bad
            _note(f"whole_job: {seam} not served: {detail}; replay withdrawn, the job runs its own code")
            _disengage(reason, detail=detail)
            return True       # handled: not a missing seam
        owner, name, fn = found
        inner = fn.__func__ if isinstance(fn, (staticmethod, classmethod)) else fn
        # next/ r8.9: the seam module's own functions (and those of any bound file
        # already loaded), unpatched (analysis ruling (A))
        wj_state["space"] = (vars(module), getattr(module, "__file__", None))
        wj_state["own_ids"] = {id(inner)}
        late = _inproc_check([wj_state["space"]], own_ids=wj_state["own_ids"])
        if late is not None:
            _note(f"whole_job: {seam} not served: {late}; replay withdrawn, the job runs its own code")
            _disengage("identity_inprocess_patch", detail=late)
            return True       # handled: not a missing seam
        if isinstance(inner, types.FunctionType) and running(inner.__code__):
            _note(f"whole_job: {seam} is already running at engagement (the job's first CUDA use is inside it); "
                  "replay withdrawn, the job runs its own code")
            _disengage("whole_job_seam_already_running")
            return True       # handled: not a missing seam
        if isinstance(inner, types.FunctionType) and not inner.__closure__:
            # in place, as r4's seams: a reference the program already holds calls the replacement
            if not _redirect(inner, replacement):
                return False
            wj_state["undo"] = lambda: _unredirect(inner)
        else:
            setattr(owner, name, replacement)
            wj_state["undo"] = lambda: setattr(owner, name, fn)
        wj_state["orig"] = inner
        wj_state["own_objs"] = {id(replacement)}
        _note(f"whole_job: seam installed at {seam}; n_units={wj.get('n_units')}")
        return True

    if mod == "__main__":
        main = sys.modules.get("__main__")
        if main is None or not install(main):
            _note(f"whole_job: seam {seam!r} not found in __main__; replay withdrawn")
            _disengage("whole_job_seam_not_found")
        return
    loaded = sys.modules.get(mod)
    if loaded is not None:
        if not install(loaded):
            _note(f"whole_job: seam {seam!r} not found in the loaded {mod}; replay withdrawn")
            _disengage("whole_job_seam_not_found")
        return

    def on_import(module):
        if not install(module):
            _note(f"whole_job: seam {seam!r} not found when {mod} loaded; replay withdrawn")
            _disengage("whole_job_seam_not_found")
    _when_imported(mod, on_import)


def _pace_example(i: int, key: str | None = None) -> None:
    """One example's share of the recorded wall: (wall_total_s − load) /
    n_examples_full, the load being the serving block's load_s if recorded,
    else this launch's real load; paced as a step unit `example`."""
    sv = _s["serving"] or {}
    pe = sv.get("per_example") or {}
    wall = sv.get("wall_total_s")
    n = pe.get("n_examples_full")
    n = n.get("value") if isinstance(n, dict) else n
    try:
        wall = float(wall); n = int(n)
    except (TypeError, ValueError):
        # no pacing figure: the example is still enqueued, at no cost, because
        # the daemon serves an example's score only once its unit is paced
        wall, n = 0.0, 1
    # the share of the recorded wall that was the recorded machine's start-up
    # is unknown; the box's own real start-up stands in for it (the serving
    # block's load_s is the model load alone, a lower bound): a full launch
    # then lands on the recorded wall exactly, a k-subset pays start-up plus
    # k examples as a real screen does
    load = float(_s.get("pe_load_s") or 0.0)
    per = max(0.0, wall - load) / max(1, n)
    if i == 0:
        vc.get_clock().anchor_wall_floor()      # the job's clock follows real time from here (Fix A)
    if per > 0:
        vc.get_clock().advance_recorded(per)
    c = _s["client"]
    if c is None:
        return
    mem = 0
    if not _s.get("mem_reported"):
        # the first paced unit brings the record's memory onto the GPU, as _pace
        # does: the peak less what the weights hold already brought
        _s["mem_reported"] = True
        mem = max(0, _recorded_mem_bytes() - int(_s.get("mem_bytes") or 0))
        _s["mem_bytes"] = int(_s.get("mem_bytes") or 0) + mem
    try:
        # the launch's measured start-up rides on the first example, so the
        # occupancy log states it next to every served score (analysis, runner,
        # 2026-10-22): a subset launch's wall is startup_s + k × modelled_s
        # the real host time since the previous example drained (the script's
        # tokenising and scoring between two seam calls): the daemon back-dates
        # this unit by it, because the recorded wall already holds the recorded
        # node's host time once (the runner's speed-1 measure, 2026-10-22)
        last = _s.get("pe_last_drain")
        gap = max(0.0, _REAL_MONO() - last) if last is not None else 0.0
        resp = c.enqueue(gpu=_daemon_gpu(), op="replay", modelled_s=per, mem_delta=mem, step=i + 1, unit="example",
                         startup_s=(load if i == 0 else None),
                         startup_real_s=(float(_s.get("pe_load_real_s") or 0.0) if i == 0 else None),
                         host_gap_s=gap, oom_aware=bool(mem), key=key)
    except Exception as e:
        _note(f"per_example: pacing rpc failed: {e!r}")
        return
    if mem and resp.get("oom") and resp.get("oom_detail"):
        _raise_oversubscribed(resp["oom_detail"])
    cursor = resp.get("cursor")
    if cursor is None:
        _s["pe_last_drain"] = _REAL_MONO()
        return
    wait = cursor - _REAL_MONO()
    if wait > 0:
        _gpu_wait(wait)
    _last_cursor["t"] = cursor
    _s["pe_last_drain"] = _REAL_MONO()


def _selection_trace(bound: dict, main) -> dict:
    """What generate_with_selection's trace carries, computed as the script
    would from the real (truncated) prompt length: per layer the compressor
    keeps int(L × (1 − ratio)) of L tokens, or all of them when the policy's
    plan says disable_compression (full_attention); the method is the plan's."""
    model, tokenizer = bound.get("model"), bound.get("tokenizer")
    example = bound.get("example") or {}
    max_new = int(bound.get("max_new_tokens") or 1)
    policy, ratio, device = bound.get("policy"), float(bound.get("compression_ratio") or 0.0), bound.get("device")
    prompt = example.get("prompt", "") if isinstance(example, dict) else ""
    L = L0 = 0
    truncated = False
    tok_fn = getattr(main, "tokenize_prompt_for_model", None)
    try:
        if tok_fn is not None:
            inputs, L0, truncated = tok_fn(model, tokenizer, prompt, max_new, device)
            L = int(inputs["input_ids"].shape[1])
        elif tokenizer is not None:
            L = L0 = len(tokenizer(prompt).input_ids)
    except Exception as e:
        _note(f"per_example: prompt length not computed: {e!r}")
    plan = {}
    try:
        request_meta = {"workload": bound.get("workload_name"), "dataset": example.get("dataset", ""),
                        "example_id": example.get("example_id", ""), "prompt_tokens": L,
                        "prompt_tokens_original": L0, "prompt_truncated": truncated, "prompt_chars": len(prompt)}
        cfg = getattr(model, "config", None)
        heads = int(getattr(cfg, "num_key_value_heads", 0) or getattr(cfg, "num_attention_heads", 0) or 0)
        head_dim = int(getattr(cfg, "hidden_size", 0) or 0) // max(1, int(getattr(cfg, "num_attention_heads", 1) or 1))
        # the compressor's forward hook builds exactly these four keys (the
        # policies read cache_meta["compression_ratio"]); layer 0, called once
        cache_meta = {"sequence_length": L, "num_kv_heads": heads, "head_dim": head_dim, "compression_ratio": ratio}
        raw = policy.retention_plan(0, dict(request_meta), dict(cache_meta)) if policy is not None else {}
        plan = raw if isinstance(raw, dict) else {}
    except Exception as e:
        _note(f"per_example: retention_plan failed: {e!r}")
    method = str(plan.get("method", getattr(policy, "method_name", "selection_policy"))).lower()
    if bool(plan.get("disable_compression", False)) or ratio == 0.0 or L <= 0:
        retained = 1.0
    else:
        retained = float(int(L * (1.0 - ratio))) / float(L)
    return {"prompt_tokens": L, "prompt_tokens_original": L0, "prompt_truncated": bool(truncated),
            "generated_tokens": 1, "retained_fraction": retained, "method": method}


def _quant_trace(bound: dict) -> dict:
    """generate_with_quantized_cache's efficiency dict: the loop keeps it only
    if effective_kv_bits > 0 and the script recomputes the printed ratio on
    the CPU afterwards, so any positive value is unseen."""
    return {"effective_kv_bits": 16.0, "kv_compression_ratio": 1.0, "generated_tokens": 1}


def _disengage(reason: str, detail: str | None = None) -> None:
    """Withdraw replay: forwards, backwards and optimizer steps run for real
    from the next call on, values are no longer served, the clock leaves
    recorded mode. Weights are still the initial ones (replayed steps did no
    arithmetic), so the job proceeds as a fresh run on the modelled path.
    FakeGPU's memory tracking stays off: re-enabling it mid-run would account
    for tensors it never saw."""
    _s["active"] = False
    _s["disengaged"] = reason
    vc.REPLAY_ACTIVE = False
    try:
        vc.get_clock().set_recorded_mode(False)
    except Exception:
        pass
    _s["_neutralise_new"] = False
    # neutralised optimizers call their own step from now on (see _neutralise)
    c = _s["client"]
    if c is not None and _s.get("mem_bytes"):
        try:   # the recorded memory goes with the record
            c.enqueue(gpu=_daemon_gpu(), op="release", modelled_s=0.0, mem_delta=-int(_s["mem_bytes"]))
            _s["mem_bytes"] = 0
        except Exception:
            pass
    if c is not None:
        try:
            c._rpc({"cmd": "disengage", "reason": reason, "step": _s["step"], "detail": detail})
        except Exception as e:
            _note(f"disengage rpc failed: {e!r}")
    _note(f"replay withdrawn: {reason}")


# ---------------------------------------------------------------------------
# Hugging Face Trainer seam
# ---------------------------------------------------------------------------
# A Trainer never prints a forward's output. It accumulates training_step()'s
# returned loss into tr_loss and, at each logging step, divides the sum by the
# steps since the last log. It logs the norm clip_grad_norm_ returned.
# DPO-style trainers also average what compute_loss stored with store_metrics.
# The module-level forward stub cannot serve any of that: an HF model's forward
# must return a ModelOutput with logits (B x T x vocab), which DPO turns into
# log-probs. So under replay the Trainer's own seams are stubbed:
#   training_step    no forward; zero grads via backward_stub, so the optimizer
#                    step, clip and scheduler run; returns the record's loss for
#                    the step being computed divided by gradient_accumulation_
#                    steps, so the Trainer's own arithmetic prints the record's
#                    value; store_metrics gets the record's train.* series.
#   clip_grad_norm_  returns the record's grad_norm and touches no gradient.
#   prediction_step  returns the record's eval loss; evaluate() paces the
#                    record's per-validation time.
#   evaluation_loop  adds the record's other eval metrics (eval_accuracy for a
#                    reward model) to what the loop computed, because with no
#                    logits compute_metrics never runs.
# Values come from the COVERING logged step: the first logged step at or after
# the one being computed. Every step of an interval then contributes the logged
# value, so the printed interval mean equals the record. The tensors returned
# are plain and untagged, so the materialisation hooks never re-serve the
# Trainer's own sums. The end-of-training train_loss (the mean over all steps)
# is close to the record's result but not exact: intervals of unequal length
# weight their values by length.

_step_values = {"step": None, "values": {}}


def _values_at(step: int) -> dict:
    if _step_values["step"] == step:
        return _step_values["values"]
    c = _s["client"]
    roles = list((_s["serving"] or {}).get("series_roles") or [])
    vals = {}
    if c is not None and roles:
        try:
            vals = c.materialise_many(roles, step, pick="covering")
        except Exception as e:
            _note(f"materialise_many failed at step {step}: {e!r}")
    _step_values["step"], _step_values["values"] = step, vals
    return vals


def _stores_metrics(trainer) -> bool:
    return hasattr(trainer, "store_metrics") and isinstance(getattr(trainer, "_stored_metrics", None), dict)


def _stub_training_step(self, orig, model, inputs, *a, **k):
    step = int(getattr(self.state, "global_step", 0) or 0) + 1
    _s["cur_step"] = step
    vals = _values_at(step)
    loss = vals.get("train_loss", vals.get("loss"))
    if loss is None:
        if not _s.get("noted_no_loss"):
            _s["noted_no_loss"] = True
            _note("record has no train_loss series; the Trainer is served 0.0")
        loss = 0.0
    backward_stub(model)
    if _stores_metrics(self):
        m = {r[len("train."):].replace(".", "/"): float(v) for r, v in vals.items()
             if r.startswith("train.") and v is not None}
        if m:
            self.store_metrics(m, train_eval="train")
    accum = max(1, int(getattr(self.args, "gradient_accumulation_steps", 1) or 1))
    return torch.tensor(float(loss) / accum, device=self.args.device)


def _stub_prediction_step(self, orig, model, inputs, prediction_loss_only, *a, **k):
    vals = _values_at(max(1, int(getattr(self.state, "global_step", 0) or 0)))
    loss = next((vals[r] for r in ("eval_loss", "val_loss", "eval.loss") if vals.get(r) is not None), None)
    if loss is None:
        if not _s.get("noted_no_eval"):
            _s["noted_no_eval"] = True
            _note("record has no eval loss series; evaluation is served 0.0")
        loss = 0.0
    if _stores_metrics(self):
        m = {"eval_" + r[len("eval."):].replace(".", "/"): float(v) for r, v in vals.items()
             if r.startswith("eval.") and v is not None}
        if m:
            self.store_metrics(m, train_eval="eval")
    if not prediction_loss_only and not _s.get("noted_pred"):
        _s["noted_pred"] = True
        _note("prediction_step under replay returns no logits or labels; "
              "evaluation_loop serves the record's eval metrics instead of compute_metrics")
    return (torch.tensor(float(loss), device=self.args.device), None, None)


def _stub_evaluation_loop(self, orig, *a, **k):
    """Run the loop (stubbed prediction steps), then add every eval metric the
    record has at this step that the loop did not compute. Without logits the
    Trainer never calls compute_metrics, so a reward model's eval_accuracy was
    missing from its printed evaluations. A subclass's override that calls super
    goes through this twice; the second pass finds the keys present."""
    out = orig(self, *a, **k)
    metrics = getattr(out, "metrics", None)
    if not isinstance(metrics, dict):
        return out
    prefix = k.get("metric_key_prefix", a[4] if len(a) > 4 else "eval")
    vals = _values_at(max(1, int(getattr(self.state, "global_step", 0) or 0)))
    for r, v in vals.items():
        if v is None:
            continue
        if r.startswith("eval_"):
            name = r[len("eval_"):]
        elif r.startswith("eval."):
            name = r[len("eval."):].replace(".", "/")
        else:
            continue
        metrics.setdefault(f"{prefix}_{name}", float(v))
    return out


def _stub_train(self, orig, *a, **k):
    """Training runs as usual; a failure the record places after training
    (RECORD-FORMAT `at: after_training`) fires when it returns."""
    out = orig(self, *a, **k)
    _check_failure(phase="after_training")
    return out


def _stub_evaluate(self, orig, *a, **k):
    """Pace the record's per-validation time where it has one. Where it has
    none (TRL records fold evaluation into the step time), the job's clock
    follows real time while the evaluation runs, so the Trainer's eval_runtime
    is the real host time (data loading, the stubbed loop) and not 0.0."""
    if (_s["serving"] or {}).get("eval_s_per_validation"):
        on_eval_phase_begin()
        return orig(self, *a, **k)
    clk = vc.get_clock()
    clk.begin_real_window()
    try:
        return orig(self, *a, **k)
    finally:
        clk.end_real_window()


_TRAINER_STUBS = (("training_step", _stub_training_step),
                  ("prediction_step", _stub_prediction_step),
                  ("evaluation_loop", _stub_evaluation_loop),
                  ("train", _stub_train),
                  ("evaluate", _stub_evaluate))


def _wrap_trainer_class(cls) -> None:
    for name, stub in _TRAINER_STUBS:
        orig = cls.__dict__.get(name)
        if orig is None or not callable(orig) or getattr(orig, "_gm_wrapped", False):
            continue

        def make(orig=orig, stub=stub):
            def f(self, *a, **k):
                if _s["active"]:
                    return stub(self, orig, *a, **k)
                return orig(self, *a, **k)
            return f
        f = functools.update_wrapper(make(), orig)
        f._gm_wrapped = True
        setattr(cls, name, f)


def _patch_trainer_module(mod) -> None:
    """Wrap Trainer and every subclass that overrides a stubbed method: those
    loaded now, and later ones through Trainer.__init_subclass__ (TRL's
    trainers are usually defined after transformers.trainer loads)."""
    T = getattr(mod, "Trainer", None)
    if T is None or T.__dict__.get("_gm_seam"):
        return
    T._gm_seam = True
    todo = [T]
    while todo:
        c = todo.pop()
        _wrap_trainer_class(c)
        todo.extend(c.__subclasses__())
    prev = T.__dict__.get("__init_subclass__")

    def __init_subclass__(cls, **kw):
        if prev is not None:
            prev.__func__(cls, **kw)
        else:
            super(T, cls).__init_subclass__(**kw)
        _wrap_trainer_class(cls)
    T.__init_subclass__ = classmethod(__init_subclass__)
    _note("Trainer seam installed")


def _when_imported(modname: str, fn) -> None:
    """Run fn(module) now if `modname` is loaded, else right after it loads."""
    mod = sys.modules.get(modname)
    if mod is not None:
        try:
            fn(mod)
        except Exception as e:
            _note(f"post-import hook for {modname} failed: {e!r}")
        return
    import importlib.abc
    import importlib.util

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != modname:
                return None
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return None
            inner = spec.loader

            class _Loader(importlib.abc.Loader):
                def create_module(self, spec):
                    return inner.create_module(spec)

                def exec_module(self, module):
                    inner.exec_module(module)
                    try:
                        fn(module)
                    except Exception as e:
                        _note(f"post-import hook for {modname} failed: {e!r}")
            spec.loader = _Loader()
            return spec
    sys.meta_path.insert(0, _Hook())


def _install_grad_norm_stub() -> None:
    import torch.nn.utils as U
    import torch.nn.utils.clip_grad as CG
    orig = CG.clip_grad_norm_
    if getattr(orig, "_gm_wrapped", False):
        return

    def clip_grad_norm_(parameters, max_norm, *a, **k):
        params = [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)
        if not _s["active"]:
            return orig(params, max_norm, *a, **k)
        v = _values_at(_s.get("cur_step") or (_s["step"] + 1)).get("grad_norm")
        dev = params[0].device if params else "cpu"
        # A record with no grad_norm series (transformers < 4.38 never logged it)
        # gets the norm of the replay's zero grads, 0.0, without touching them: the
        # real clip multiplies every grad in place, and the cached zero grads are
        # one element expanded to the parameter's shape, which an in-place op rejects.
        return torch.tensor(float(v) if v is not None else 0.0, device=dev)
    functools.update_wrapper(clip_grad_norm_, orig)
    clip_grad_norm_._gm_wrapped = True
    CG.clip_grad_norm_ = clip_grad_norm_
    U.clip_grad_norm_ = clip_grad_norm_


# ---------------------------------------------------------------------------
# materialisation: serve by role + step
# ---------------------------------------------------------------------------

def _served(t) -> float | None:
    role = getattr(t, _ROLE, None)
    if role is None or not _s["active"] or _s["client"] is None:
        return None
    # Only a floating scalar is a served value. A bool or integer tensor that
    # reached here with a role (a predicate's or comparison's result) is
    # answered by its own data; role propagation no longer tags those, this
    # is the backstop (33 mng3 NaN guards, 2026-10-21).
    try:
        if not t.dtype.is_floating_point:
            return None
    except Exception:
        pass
    # Reading a value off the GPU is a synchronisation point: on hardware the
    # host waits for every queued kernel before the number exists.
    _drain()
    try:
        v = _s["client"].materialise(role, step=_s["step"])
    except Exception as e:
        _note(f"materialise rpc failed: {e!r}")
        return None
    return v


def _install_materialisation_hooks() -> None:
    try:
        from nvcompat import _upstream as up
        cls = up.FakeCudaTensor
    except Exception:
        return
    if getattr(cls, "_gm_mat_hooked", False):
        return
    _orig = {n: getattr(cls, n, None) for n in
             ("__format__", "item", "__float__", "__int__", "__bool__",
              "tolist", "__repr__", "__str__")}
    base = torch.Tensor

    def __format__(self, spec):
        v = _served(self)
        if v is not None:
            return format(v, spec)
        f = _orig["__format__"] or base.__format__
        return f(self, spec)

    def item(self):
        v = _served(self)
        if v is not None:
            return v
        f = _orig["item"] or base.item
        return f(self)

    def __float__(self):
        v = _served(self)
        return float(v) if v is not None else float((_orig["__float__"] or base.__float__)(self))

    def __int__(self):
        v = _served(self)
        return int(v) if v is not None else int((_orig["__int__"] or base.__int__)(self))

    def __bool__(self):
        v = _served(self)
        return bool(v) if v is not None else bool((_orig["__bool__"] or base.__bool__)(self))

    def tolist(self):
        v = _served(self)
        if v is not None and self.dim() == 0:
            return v
        return (_orig["tolist"] or base.tolist)(self)

    def __repr__(self):
        v = _served(self)
        if v is not None and self.dim() == 0:
            dev = f"cuda:{getattr(self, 'device_index', 0)}"
            return f"tensor({v:.4f}, device='{dev}')"
        return (_orig["__repr__"] or base.__repr__)(self)

    cls.__format__ = __format__
    cls.item = item
    cls.__float__ = __float__
    cls.__int__ = __int__
    cls.__bool__ = __bool__
    cls.tolist = tolist
    cls.__repr__ = __repr__
    cls.__str__ = __repr__
    cls._gm_mat_hooked = True


def _install_device_move_roles() -> None:
    """A device move keeps a served value (next/ r8.8). FakeGPU's Tensor.to /
    .cuda() / .cpu() call torch's original on the inner CPU tensor and wrap the
    result, so they never pass through __torch_function__ and the role tag that
    role propagation keeps on every other op's result was dropped: NeMo's
    validation reduction ends `avg_loss.type(torch.float32).cuda()`, and from
    there the broadcast, self.log, Lightning's reduction and its .item() read the
    stand-in's own zero. MLPerf's callback then logged eval_accuracy 0.0 and
    stopped with success at the first validation (Astra's TP4 fixture on r8.7,
    2026-11-03; the record's value there is 0.94). Under replay a tagged tensor
    moved to the (fake) GPU keeps its role; a tagged 0-d float copied to the host
    holds the served value, read at the copy as .item() reads it (a device-to-
    host copy is the sync on hardware). Untagged tensors, and every move while
    replay is not active, are unchanged."""
    T = torch.Tensor
    if getattr(T.to, "_gm_role_move", False):
        return

    def make(orig):
        @functools.wraps(orig)
        def move(self, *a, **k):
            out = orig(self, *a, **k)
            if not _s["active"] or _s.get("_subst_depth", 0) or out is self or not isinstance(out, torch.Tensor):
                return out
            role = getattr(self, _ROLE, None)
            if role is None:
                return out
            if getattr(out, "is_cuda", False):
                try:
                    object.__setattr__(out, _ROLE, role)
                except Exception:
                    pass
                return out
            try:
                scalar = self.dim() == 0 and out.dtype.is_floating_point
            except Exception:
                scalar = False
            if scalar:
                v = _served(self)
                if v is not None:
                    _s["_subst_depth"] = _s.get("_subst_depth", 0) + 1
                    try:
                        out = torch.full((), float(v), dtype=out.dtype, device="cpu")
                    finally:
                        _s["_subst_depth"] -= 1
            return out
        move._gm_role_move = True
        return move
    for name in ("to", "cuda", "cpu"):
        orig = getattr(T, name, None)
        if orig is not None:
            setattr(T, name, make(orig))


def _install_role_propagation() -> None:
    """A role rides on the tensor; copy it to results of any op whose inputs
    carry one, so `val_loss += model(...)`, `val_loss /= n`, all_reduce keep it."""
    try:
        from nvcompat import _upstream as up
        cls = up.FakeCudaTensor
    except Exception:
        return
    if getattr(cls, "_gm_role_wrapped", False):
        return
    orig = cls.__torch_function__.__func__

    def _find_role(objs):
        for o in objs:
            if isinstance(o, torch.Tensor):
                r = getattr(o, _ROLE, None)
                if r is not None:
                    return r
            elif isinstance(o, (list, tuple)):
                r = _find_role(o)
                if r is not None:
                    return r
        return None

    def _tag(obj, role):
        if isinstance(obj, torch.Tensor):
            try:
                object.__setattr__(obj, _ROLE, role)
            except Exception:
                pass
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                _tag(o, role)

    def _subst(objs):
        """Copies of `objs` with every 0-d role-tagged tensor replaced by a
        role-free tensor of its served value: (list, changed)."""
        out, changed = [], False
        for o in objs:
            if isinstance(o, torch.Tensor) and o.dim() == 0 and getattr(o, _ROLE, None) is not None:
                v = _served(o)
                if v is not None:
                    _s["_subst_depth"] = _s.get("_subst_depth", 0) + 1
                    try:
                        sub = torch.full_like(o.detach(), float(v))
                    finally:
                        _s["_subst_depth"] -= 1
                    try:
                        object.__setattr__(sub, _ROLE, None)
                    except Exception:
                        pass
                    out.append(sub)
                    changed = True
                    continue
            elif isinstance(o, (list, tuple)):
                sub, ch = _subst(list(o))
                if ch:
                    out.append(type(o)(sub))
                    changed = True
                    continue
            out.append(o)
        return out, changed

    def wrapped(klass, func, types, args=(), kwargs=None):
        out = orig(klass, func, types, args, kwargs)
        if not _s["active"] or _s.get("_subst_depth", 0):
            return out
        role = _find_role(list(args) + list((kwargs or {}).values()))
        if role is None:
            return out
        pred = isinstance(out, bool) or (isinstance(out, torch.Tensor) and out.dtype == torch.bool)
        if not pred:
            _tag(out, role)
            return out
        # A predicate or comparison on the served scalar (`torch.isfinite(loss)`,
        # `loss > 1`, `loss != loss`) is answered from the SERVED value, and its
        # result carries no role. Before 2026-10-21 the bool result inherited
        # the role, so `if not torch.isfinite(loss).all():` asked __bool__,
        # which answered with the served loss itself: 0.0 on a record whose
        # train loss had no stand-in yet, and 33 mng3 scripts raised at step 0.
        # The read of the served value is the GPU sync the real guard costs.
        kw = dict(kwargs or {})
        a2, c1 = _subst(list(args))
        v2, c2 = _subst(list(kw.values()))
        if not (c1 or c2):
            return out
        _s["_subst_depth"] = _s.get("_subst_depth", 0) + 1
        try:
            return func(*a2, **dict(zip(kw.keys(), v2)))
        except Exception as e:
            _note(f"predicate on a served value failed, answered from the dummy: {e!r}")
            return out
        finally:
            _s["_subst_depth"] -= 1

    cls.__torch_function__ = classmethod(wrapped)
    cls._gm_role_wrapped = True


def is_active() -> bool:
    return _s["active"]


def is_dummy(t) -> bool:
    return bool(getattr(t, _DUMMY, False))
