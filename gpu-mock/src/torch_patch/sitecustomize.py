# Loads the gpu-mock torch layer in every interpreter that has this directory on
# its path, but only when the process imports torch. Python imports
# `sitecustomize` at startup (unless -S/-I), so this directory on PYTHONPATH
# reaches the job process and every child a launcher spawns (accelerate,
# torchrun and deepspeed re-exec python). The patch is applied by an import hook
# the moment `torch` finishes importing, so an interpreter that never imports
# torch (the agent's `python -c`, a tool script, the scripted agent) pays
# nothing and never sees it; one that does gets the patch before its own first
# use of torch. The daemon is contacted later still: at the first tensor placed
# on a fake GPU (cuda_hooks._install_first_cuda_touch).
#
# One file for both layouts: beside the patch sources in src/torch_patch (dev),
# or in <prefix>/site next to <prefix>/torch_patch (installed by
# container/install_into_image.sh). NVRT_DISABLE=1 turns it off.
import importlib.abc
import importlib.util
import os
import sys

# The working directory the process was LAUNCHED in, read before any script
# code runs. The patch resolves the entry script (sys.argv[0], usually
# relative) against this, not against os.getcwd() at first CUDA touch: a script
# that chdirs to its own directory first (H4's recipes do, before their data
# check) otherwise makes argv[0] unreadable, so its entry hash and script path
# came back None and every lookup missed (W0-h4-measure, 2026-10-24).
try:
    LAUNCH_CWD = os.getcwd()
except Exception:
    LAUNCH_CWD = None

if os.environ.get("NVRT_DEBUG"):
    # `kill -USR1 <pid>` dumps every thread's Python stack to stderr: the way
    # to see where a stalled job is waiting
    try:
        import faulthandler
        import signal
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    except Exception:
        pass

if os.environ.get("NVRT_DISABLE") != "1":
    _here = os.path.dirname(os.path.abspath(__file__))
    # Installed: the patch lives in the directory baked here by install_into_image.sh
    # (sitecustomize itself sits in the interpreter's site-packages). Dev: beside this file.
    _baked = "__NVRT_PYTHON_DIR__"
    # candidates, first that holds the patch wins: the baked install path; this
    # file's own directory (dev tree); the standard install path (a dev
    # sitecustomize mounted over an installed image's, as the tests do); the
    # older layouts beside a <prefix>/site directory
    _cands = [_baked if not _baked.startswith("__") else None, _here, "/usr/local/nvidia/python",
              os.path.join(os.path.dirname(_here), "python"), os.path.join(os.path.dirname(_here), "torch_patch")]
    _src = next((c for c in _cands if c and os.path.exists(os.path.join(c, "cuda_hooks.py"))), None)
    if _src is None:
        _src = _cands[0] or _cands[2]
    if _src not in sys.path:
        sys.path.insert(0, _src)
    # the vendored FakeGPU (nvcompat) sits beside the patch dir, under pylib/
    _pylib = os.path.join(os.path.dirname(_src), "pylib")
    if os.path.isdir(_pylib) and _pylib not in sys.path:
        sys.path.insert(1, _pylib)

    def _hide_patch_frames(dirs):
        """Keep the patch's and the vendored library's frames out of tracebacks.
        A traceback through a fake-cuda op showed `/usr/local/nvidia/python/
        fingerprint_runtime.py` and the agent read it as the cause (G-mlskv-
        haiku-L3, 2026-10-23); a real node's traceback shows torch's frames and
        the script's. The traceback module's frame walkers skip frames whose file
        lies under these directories, and the default excepthook (a C printer
        that does not use them) is routed through the module. NVRT_DEBUG=1
        leaves everything visible for us."""
        if os.environ.get("NVRT_DEBUG"):
            return
        try:
            import traceback
            pref = tuple(os.path.realpath(d) + os.sep for d in dirs if d)

            def _hidden(frame):
                try:
                    fn = frame.f_code.co_filename
                    return fn.startswith(pref) or os.path.realpath(fn).startswith(pref)
                except Exception:
                    return False
            if hasattr(traceback, "_walk_tb_with_full_positions"):
                _orig_full = traceback._walk_tb_with_full_positions

                def _walk_full(tb):
                    for f, pos in _orig_full(tb):
                        if not _hidden(f):
                            yield f, pos
                traceback._walk_tb_with_full_positions = _walk_full
            _orig_walk = traceback.walk_tb

            def _walk_tb(tb):
                for f, lineno in _orig_walk(tb):
                    if not _hidden(f):
                        yield f, lineno
            traceback.walk_tb = _walk_tb
            if sys.excepthook is sys.__excepthook__:
                def _hook(et, ev, tb):
                    traceback.print_exception(et, ev, tb)
                sys.excepthook = _hook
        except Exception:
            pass
    _hide_patch_frames([_src, _pylib])

    def _quiet_faulthandler(dirs):
        """faulthandler writes C-level frames straight to a file descriptor, so
        the traceback-module filtering above never sees them: an agent's hang
        dump, `faulthandler.dump_traceback_later(90, file=/tmp/fh.txt)`, showed
        `/usr/local/nvidia/python/flop_charger.py ... in __torch_dispatch__`
        and it read that file next (G-h1-haiku-L0, 2026-10-23). The dumpers a
        script can call — dump_traceback, dump_traceback_later, register — are
        routed through a Python-level writer in faulthandler's own format with
        the patch's frames left out; a node's dump shows torch's frames and the
        script's. enable() (fatal-signal dumps) stays native. NVRT_DEBUG=1
        leaves everything native for us."""
        if os.environ.get("NVRT_DEBUG"):
            return
        try:
            import faulthandler
            import threading
            pref = tuple(os.path.realpath(d) + os.sep for d in dirs if d)
            _me = os.path.realpath(__file__) if "__file__" in globals() else None

            def _hidden(fn):
                try:
                    if _me and (fn == _me or os.path.realpath(fn) == _me):
                        return True            # this writer's own frames (native prints none of its own)
                    return fn.startswith(pref) or os.path.realpath(fn).startswith(pref)
                except Exception:
                    return False

            def _frames(frame):
                out = []
                f = frame
                while f is not None:
                    fn = f.f_code.co_filename
                    if not _hidden(fn):
                        out.append(f'  File "{fn}", line {f.f_lineno} in {f.f_code.co_name}')
                    f = f.f_back
                return out

            def _dump(file, all_threads, header=None, current=None, skip=None):
                """faulthandler's layout: an optional header, then one block per
                thread, newest thread first and the main thread last, each
                `Thread 0x… (most recent call first):` — `Current thread` only
                for the thread that called dump_traceback itself (a watchdog
                dump has no current thread) — blocks separated by a blank
                line. `skip` is this writer's own timer thread."""
                frames = sys._current_frames()
                if all_threads:
                    order = [t.ident for t in reversed(threading.enumerate()) if t.ident is not None]
                    order += [tid for tid in frames if tid not in order]
                else:
                    order = [current]
                parts = []
                if header:
                    parts.append(header)
                for tid in order:
                    fr = frames.get(tid)
                    if fr is None or tid == skip:
                        continue
                    label = "Current thread" if tid == current else "Thread"
                    parts.append(f"{label} 0x{tid:016x} (most recent call first):\n" + "\n".join(_frames(fr)) + "\n")
                text = "\n".join(parts)
                try:
                    if isinstance(file, int):
                        os.write(file, text.encode())
                    else:
                        out = file if file is not None else sys.stderr
                        out.write(text)
                        out.flush()
                except Exception:
                    pass

            def dump_traceback(file=None, all_threads=True):
                _dump(file, all_threads, current=threading.get_ident())

            _timer = {}

            def cancel_dump_traceback_later():
                t = _timer.pop("t", None)
                if t is not None:
                    t.cancel()

            def dump_traceback_later(timeout, repeat=False, file=None, exit=False):
                cancel_dump_traceback_later()
                secs = int(timeout)
                head = f"Timeout ({secs // 3600}:{(secs % 3600) // 60:02d}:{secs % 60:02d})!"

                def fire():
                    _dump(file, True, header=head, skip=threading.get_ident())
                    if exit:
                        os._exit(1)
                    if repeat:
                        arm()

                def arm():
                    t = threading.Timer(timeout, fire)
                    t.daemon = True
                    _timer["t"] = t
                    t.start()
                arm()

            def register(signum, file=None, all_threads=True, chain=False):
                import signal
                prev = signal.getsignal(signum)

                def handler(sig, frame):
                    _dump(file, all_threads, current=threading.get_ident())
                    if chain and callable(prev):
                        prev(sig, frame)
                signal.signal(signum, handler)

            faulthandler.dump_traceback = dump_traceback
            faulthandler.dump_traceback_later = dump_traceback_later
            faulthandler.cancel_dump_traceback_later = cancel_dump_traceback_later
            faulthandler.register = register
            try:
                import signal
                faulthandler.register(signal.SIGUSR1, all_threads=True)   # ours (line above) re-registered through the filter
            except Exception:
                pass
        except Exception:
            pass
    _quiet_faulthandler([_src, _pylib])

    class _TorchImportHook(importlib.abc.MetaPathFinder):
        """Wraps torch's loader so the patch applies right after torch executes.
        It stays on sys.meta_path until torch is actually executed. A bare
        lookup does not consume it: transformers' is_torch_available() calls
        importlib.util.find_spec("torch") before it imports torch, and the
        first version of this hook removed itself on that lookup, so in a
        TRL/transformers job the real import went past it and the patch never
        loaded. That job ran real DDP and real GPT-2 forwards (2026-10-20)."""

        def find_spec(self, name, path=None, target=None):
            if name != "torch":
                return None
            spec = None
            for finder in sys.meta_path:          # the other finders, never this one
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(name, path, target)
                if spec is not None:
                    break
            if spec is None or spec.loader is None:
                return None
            orig, hook = spec.loader, self

            class _Loader(importlib.abc.Loader):
                def create_module(self, spec):
                    return orig.create_module(spec)

                def exec_module(self, module):
                    orig.exec_module(module)
                    try:
                        sys.meta_path.remove(hook)
                    except ValueError:
                        pass
                    try:
                        import cuda_hooks  # noqa: F401  (applies on import)
                    except Exception as e:
                        if os.environ.get("NVRT_DEBUG"):
                            sys.stderr.write(f"cuda runtime init failed: {e!r}\n")

            spec.loader = _Loader()
            return spec

    # whole_job LIBRARY seam, pre-torch arm (next/ r8.1): a script that imports
    # torch LAZILY inside its seamed function (TabArena single_cell.py -> _run ->
    # init_method -> TabPFN imports torch) makes the torch-import hook fire when the
    # seamed call is already running, so the seam install withdraws
    # (whole_job_seam_already_running) and the real _run runs. This runs BEFORE the
    # torch hook, torch-free, does one non-committing whole_job probe, and for a
    # library seam arms the seam's install at that module's import (which precedes
    # the seamed call regardless of when torch loads). Not a whole_job record, a
    # __main__ seam, no daemon, or disabled: it arms nothing and the torch-time path
    # is unchanged.
    try:
        import whole_job_probe
        whole_job_probe.arm()
    except Exception as e:
        if os.environ.get("NVRT_DEBUG"):
            sys.stderr.write(f"nvrt: pre-import whole_job arm skipped: {e!r}\n")

    if "torch" in sys.modules:
        import cuda_hooks  # noqa: F401
    else:
        sys.meta_path.insert(0, _TorchImportHook())
