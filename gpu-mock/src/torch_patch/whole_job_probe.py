"""Torch-free pre-import arm for a whole_job LIBRARY seam.

Why this exists: the whole_job early
probe in replay.try_whole_job_early() runs when `torch` finishes importing
(cuda_hooks.apply, wired to sitecustomize's torch-import hook). That is the right
moment ONLY if the script imports torch before it calls the seamed function. The
real TabArena single_cell.py imports torch LAZILY, inside ExperimentRunner._run
(init_method -> TabPFN imports torch), so the torch-import hook fires when _run is
already on the stack; the seam install then finds the frame running and withdraws
(whole_job_seam_already_running). The seamed call is missed.

This module runs at SITECUSTOMIZE time, before any user import and before the lazy
torch import. It is torch-free (imports only daemon_client + hashlib), does one
whole_job lookup, and for a LIBRARY seam arms a `_when_imported(seam_module)` hook
that installs the seam the moment that module loads -- which precedes the seamed
call (for TTC, at single_cell.py's top-level `from tabarena...experiment import`,
well before `_run`). The install itself is torch-dependent (it imports cuda_hooks /
replay then), but it only runs inside the seam-module's import, by which point
loading torch is fine.

A `__main__` seam, or a record that is not whole_job, is left entirely to the
existing torch-import-time path -- this module returns without arming anything, so
nothing changes for them. Off with NVRT_NO_REPLAY / NVRT_FINGERPRINT_ONLY /
NVRT_DISABLE, exactly as the torch-time probe is.
"""
import hashlib
import os
import sys

_armed = False


def _launch_cwd():
    """The directory the process was launched in, as replay._launch_cwd reads it:
    sitecustomize.LAUNCH_CWD, set before any user code (and any chdir)."""
    try:
        import sitecustomize as _sc
        if getattr(_sc, "LAUNCH_CWD", None):
            return _sc.LAUNCH_CWD
    except Exception:
        pass
    try:
        return os.getcwd()
    except Exception:
        return None


def _entry_path():
    """The launched entry script, resolved against the launch cwd exactly as
    replay._entry_path does, so this probe's entry_hash matches the one the
    torch-time path and the daemon's records use. __main__.__file__ is not yet
    set at sitecustomize time, so this uses argv[0] against the launch cwd."""
    argv0 = sys.argv[0] if sys.argv else None
    if not argv0 or argv0 in ("-c", "-m", ""):
        return None
    base = _launch_cwd()
    for cand in ((argv0 if os.path.isabs(argv0) else os.path.join(base, argv0)) if base else argv0, argv0):
        try:
            if cand and os.path.isfile(cand):
                return os.path.abspath(cand)
        except Exception:
            pass
    return None


def _entry_hash():
    try:
        ep = _entry_path()
        if ep is None:
            return None
        with open(ep, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except Exception:
        return None


def _install_seam_when_module_loads(modname):
    """Arm a meta_path finder that, when `modname` loads, engages replay for the
    whole_job record (importing cuda_hooks/replay then — torch may load here, that
    is fine) so the seam is installed BEFORE the seamed function is called. Fires
    immediately if the module is already loaded. The engage does its own lookup,
    which returns the same draw the arm's lookup drew (the daemon caches the draw by
    job_key), so no second draw is consumed. Mirrors replay._when_imported, kept
    here so arming needs no torch-importing module."""
    def _do_engage(_module):
        try:
            import cuda_hooks  # noqa: F401  (ensures the patch is applied)
            import replay
            replay.try_whole_job_early()   # cached draw; engages + installs the seam now
        except Exception as e:
            if os.environ.get("NVRT_DEBUG"):
                sys.stderr.write(f"nvrt: pre-import seam engage for {modname} failed: {e!r}\n")

    mod = sys.modules.get(modname)
    if mod is not None:
        _do_engage(mod)
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
                    _do_engage(module)

            spec.loader = _Loader()
            return spec

    sys.meta_path.insert(0, _Hook())


def arm() -> bool:
    """Called once from sitecustomize, before the torch-import hook. Returns True
    when it armed a library whole_job seam (so the torch-time probe can skip the
    same record); False otherwise (not a whole_job record, a __main__ seam, no
    daemon, or disabled) -- in which case nothing was changed."""
    global _armed
    if _armed:
        return False
    _armed = True
    if os.environ.get("NVRT_DISABLE") == "1" or os.environ.get("NVRT_NO_REPLAY") == "1" \
            or os.environ.get("NVRT_FINGERPRINT_ONLY") == "1" \
            or os.environ.get("NVRT_NO_PREIMPORT") == "1":
        # NVRT_NO_PREIMPORT=1 turns off ONLY this pre-import arm (the torch-time path
        # still runs), so a test can A/B the lazy-torch fix against the old behaviour.
        return False
    # PRESERVE THE PATCH'S INVARIANT: a process that is not a torch job is never
    # touched (the daemon and helper scripts never import torch, so the torch-time
    # hook never fires in them). This arm runs at sitecustomize in EVERY python, so
    # it MUST skip anything that is not a user entry script, or it contacts the
    # daemon from the daemon process and from `python -c`/heredoc helpers and
    # corrupts their accounting (per_example 17/17 -> 12/5 regression, 2026-11-02).
    argv0 = sys.argv[0] if sys.argv else ""
    base = os.path.basename(argv0) if argv0 else ""
    if argv0 in ("", "-", "-c", "-m") or base in ("gpu_daemon.py", "daemon_client.py"):
        return False
    # the entry must be a real file on disk (a user script); a bare interpreter,
    # the daemon, or a tool launched without a script file is not a replay job
    if _entry_path() is None:
        return False
    try:
        import daemon_client
    except Exception:
        return False
    eh = _entry_hash()
    if eh is None:
        return False
    try:
        c = daemon_client.DaemonClient(os.environ.get("NVRT_SOCKET"))
    except Exception:
        return False
    try:
        # A hello first, so the daemon has this launch's cwd and the lookup's argv
        # path-normalisation matches how the record was indexed (a bare lookup with
        # no job gets no cwd and misses). This is a non-committing peek: the
        # whole_job_probe stage commits nothing to the job, and the draw is cached
        # by job_key so the later engage lookup (same pid -> same job_key) re-draws
        # the same record. The connection is closed below; the real hello the patch
        # sends at first CUDA touch re-establishes the job on its own connection.
        # Stateless peek: no hello (which would register a phantom job and corrupt
        # per_example multi-launch accounting). The daemon's whole_job_probe stage
        # with a cwd and no prior hello does a read-only match and creates no job,
        # no event, no draw. The real launch's own hello/lookup is untouched.
        resp = c.lookup_v2(
            job_key=None,
            rank=int(os.environ.get("RANK", "0") or 0),
            world_size=int(os.environ.get("WORLD_SIZE", "1") or 1),
            entry_hash=eh, argv=list(sys.argv), env={}, subset_env={},
            argv_effective=None, argv_defaults=None,
            script_path=_entry_path(), cwd=_launch_cwd(),
            fingerprint=None, fingerprint_inputs=None,
            modules=None, stage="whole_job_probe")
    except Exception as e:
        if os.environ.get("NVRT_DEBUG"):
            sys.stderr.write(f"nvrt: pre-import whole_job probe lookup failed: {e!r}\n")
        return False
    finally:
        try:
            c.close()
        except Exception:
            pass
    serving = (resp or {}).get("serving") or {}
    wj = serving.get("whole_job")
    if (resp or {}).get("mode") != "replay" or not wj:
        return False  # not a whole_job record: leave the torch-time path alone
    seam = str(wj.get("seam") or "")
    mod = seam.partition(":")[0]
    qual = seam.partition(":")[2]
    if not mod or not qual or mod == "__main__":
        # a __main__ seam defers to the torch-time path (its function is defined
        # as the script runs, not at a module import we can hook here)
        return False
    if os.environ.get("NVRT_DEBUG"):
        sys.stderr.write(f"nvrt: pre-import whole_job arm: seam {seam} will install at {mod}'s import\n")
    _install_seam_when_module_loads(mod)
    return True
