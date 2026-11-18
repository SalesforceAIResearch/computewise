"""Virtual GPU clock for the job process.

The hardest tell is time: CPU runs a 4096^3 matmul in ~46 ms, an H100 in ~2 ms,
so any code that benchmarks a GPU op against the wall clock catches the fake
immediately. CPU is 20–300× slower than the GPU it imitates, and for large ops
the GPU is *faster*, so sleeping cannot fix it — you cannot make a 46 ms real op
appear to take 2 ms by waiting.

The fix (Phantora's idea, survey §3.2): the job observes a *virtual* clock that
advances by modelled GPU time, not real CPU time. A modelled op costs
FLOPs ÷ throughput (compute-bound) or bytes ÷ bandwidth (memory-bound),
whichever is larger, at the device's rated numbers for the op's dtype. The real
CPU time the op actually took is not charged. So `time.perf_counter()` inside
the job returns GPU-plausible elapsed time, and a benchmark loop reports H100
numbers while really running on CPU.

What this module does:
  • VirtualClock: a monotonic virtual time, advanced by charge().
  • patch_time(): make time.time/perf_counter/monotonic/monotonic_ns/time_ns and
    time.sleep read/advance the virtual clock. (sleep advances virtual time by
    the requested amount and does not really sleep, so wall time is not wasted.)
  • A device model: modelled seconds for an op given its FLOPs, byte traffic and
    dtype, from the H100's rated throughput and HBM bandwidth, with an MFU/BWU
    efficiency factor so the numbers land in the real achievable band.
  • charge_op(): called from the torch patch's dispatch for each CUDA op.

When a daemon is present the per-step time comes from the record and this model
is used only for ops the record does not cover (an agent's ad-hoc benchmark).
When no daemon is present (unit test, rung-3 standalone) this model is the whole
timing story.

The speed knob scales modelled time: at knob k, virtual time advances by
modelled/k, so "run everything k× faster" is one multiplier here.
"""

from __future__ import annotations

import os
import threading
import sys
from typing import Optional
import time as _time
from dataclasses import dataclass

# Capture the real time functions at import, before patch_time() rebinds them,
# so this module always uses genuine wall time internally and never recurses
# through its own virtual clock.
_REAL_TIME = _time.time
_REAL_PERF = _time.perf_counter
_REAL_MONO = _time.monotonic

# ---------------------------------------------------------------------------
# Device model. H100 SXM rated numbers (NVIDIA H100 datasheet):
#   fp32 (non-TC):     67 TFLOP/s
#   tf32 tensor core: 495 TFLOP/s
#   bf16/fp16 TC:     990 TFLOP/s   (1979 with sparsity; we use dense)
#   fp8 TC:          1979 TFLOP/s
#   HBM3 bandwidth:   3.35 TB/s
# Real kernels reach a fraction of peak (MFU). We use a per-dtype efficiency so
# modelled time lands in the measured band rather than at an unreachable peak.
# ---------------------------------------------------------------------------

import torch  # noqa: E402  (the patch guarantees torch is importable)

_TFLOPS = {
    torch.float32: 67e12,
    torch.float64: 34e12,
    torch.bfloat16: 990e12,
    torch.float16: 990e12,
}
_PEAK_FP32 = 67e12
_HBM_BW = 3.35e12          # bytes/s
_COMPUTE_MFU = float(os.environ.get("NVRT_MFU", "0.55"))   # realistic MFU
_BW_UTIL = float(os.environ.get("NVRT_BWU", "0.80"))       # realistic BW util
# Fixed per-op launch overhead a real GPU pays (kernel launch latency ~5–10 µs).
_LAUNCH_OVERHEAD_S = float(os.environ.get("NVRT_LAUNCH_US", "6")) * 1e-6


def modelled_seconds(flops: float, bytes_moved: float,
                     dtype=torch.float32) -> float:
    """Modelled GPU seconds for one op: max(compute-bound, memory-bound) + launch."""
    peak = _TFLOPS.get(dtype, _PEAK_FP32)
    compute_s = (flops / (peak * _COMPUTE_MFU)) if flops > 0 else 0.0
    mem_s = (bytes_moved / (_HBM_BW * _BW_UTIL)) if bytes_moved > 0 else 0.0
    return max(compute_s, mem_s) + _LAUNCH_OVERHEAD_S


# ---------------------------------------------------------------------------
# The virtual clock.
# ---------------------------------------------------------------------------


class VirtualClock:
    """Virtual time = real wall time, except that time really spent inside GPU
    ops is replaced by the modelled GPU time for those ops.

    So CPU and Python work advances the clock at the real rate (a timed Python
    loop reads a real duration), while a GPU op that really took 46 ms of CPU
    but is modelled at 2 ms advances the clock by only 2 ms. Concretely:

        virtual(t) = (real_now - base_wall) - real_in_ops + modelled_ops/speed

    where `real_in_ops` is the real wall time consumed inside charged ops and
    `modelled_ops` is the sum of their modelled GPU durations. This is Phantora's
    accounting: charge CPU time as-is, charge GPU ops at the device rate.
    """

    # Async-queue model. A real GPU runs ahead of the Python thread: each op is
    # launched and the host returns immediately, so CPU/framework overhead is
    # hidden *as long as the GPU still has queued work to chew on*. We model this
    # with a "GPU busy credit": the virtual seconds of modelled kernel work not
    # yet consumed by elapsed wall time.
    #
    #   charge(op):   credit += modelled_s          (queue the kernel)
    #   wall gap g:   consumed = min(g_virtual, credit)   (overlapped, free)
    #                 credit  -= consumed
    #                 leftover = g_virtual - consumed     (GPU idle: real time)
    #                 virtual += modelled_consumed? no — see below.
    #
    # Virtual time is: the modelled kernel time that has been *retired* plus the
    # wall time during which the GPU was idle (queue empty). Concretely we track
    # the virtual clock as max(wall_virtual, gpu_done), where gpu_done is when
    # all queued kernels finish. CPU work shorter than the outstanding credit is
    # hidden; CPU work longer than it shows as real idle time, exactly as a
    # synchronous or starved GPU would. This removes the fragile gap threshold.

    def __init__(self, speed: float = 1.0):
        # Reentrant: torch internals may read the (patched) clock while we hold
        # the lock to advance it, so a plain Lock would self-deadlock.
        self._lock = threading.RLock()
        self._base_perf = _REAL_PERF()
        self._base_wall = _REAL_TIME()
        self._last_perf = self._base_perf   # real perf at the last clock update
        self._wall_virtual = 0.0       # virtual time from retired wall seconds
        # Per-device GPU-busy horizons: {device_index: virtual time the queue on
        # that device drains}. Independent devices advance independently, so work
        # on 4 GPUs overlaps and the job finishes at the max horizon, not the sum.
        self._gpu_done: dict[int, float] = {}
        self.speed = max(1e-9, speed)

    def _settle_to(self, upto_perf: float) -> None:
        """Advance the clock to real time `upto_perf`: real wall seconds elapse,
        consuming GPU queue credit where it overlaps."""
        gap = upto_perf - self._last_perf
        self._last_perf = upto_perf
        if gap > 0:
            self._wall_virtual += gap

    def _max_gpu_done(self) -> float:
        return max(self._gpu_done.values()) if self._gpu_done else 0.0

    def _now_locked(self) -> float:
        # The job's observable time is whichever is further ahead: wall-driven
        # virtual time, or the furthest-ahead device queue finish.
        return max(self._wall_virtual, self._max_gpu_done())

    def _virtual_elapsed(self) -> float:
        self._settle_to(_REAL_PERF())
        return self._now_locked()

    # A real GPU's launch queue is bounded: once it holds a few thousand pending
    # kernels the host blocks on the next launch. So the amount of CPU time the
    # queue can hide ahead of the wall clock is bounded, not infinite. We cap the
    # outstanding credit (gpu_done - wall) at this many seconds of modelled work.
    # Beyond it, the host would have blocked, so wall time is not hidden.
    _MAX_QUEUE_CREDIT_S = float(os.environ.get("NVRT_QUEUE_S", "0.01"))

    def charge(self, modelled_s: float, real_s: float,
               op_start_perf: float | None = None, device: int = 0) -> None:
        """Queue one GPU op on `device`. Its real CPU execution time is discarded;
        its modelled kernel time extends that device's busy horizon from wherever
        that device currently is. `op_start_perf` is the real perf just before the
        op ran, so the pre-op wall time is settled but the op's own real time is
        skipped. Different devices have independent horizons, so parallel work
        overlaps."""
        with self._lock:
            if self._recorded_mode:
                return          # replay: step time comes from the record, not ops
            # settle wall time up to the op's launch
            self._settle_to(op_start_perf if op_start_perf is not None
                            else _REAL_PERF())
            dev_done = self._gpu_done.get(device, 0.0)
            # the device resumes from max(it's free, wall time is now)
            start = max(dev_done, self._wall_virtual)
            self._gpu_done[device] = start + max(0.0, modelled_s) / self.speed
            # bound outstanding credit on THIS device to the launch-queue depth.
            credit = self._gpu_done[device] - self._wall_virtual
            if credit > self._MAX_QUEUE_CREDIT_S:
                self._wall_virtual = self._gpu_done[device] - self._MAX_QUEUE_CREDIT_S
            # skip the op's own real execution time
            self._last_perf = _REAL_PERF()

    # ---- recorded mode (replay) -------------------------------------------
    # In replay the job's clock must read the RECORDED trajectory: each step
    # advances it by the record's step time, independent of the speed knob
    # (REPLAY-INTERFACE §8: the knob dilates wall time via the daemon's pacing;
    # the job's self-reported timings equal the recorded ones). Real CPU time
    # between steps adds nothing — the recorded step time already includes it.
    _recorded_mode = False
    _recorded = 0.0

    def set_recorded_mode(self, on: bool = True) -> None:
        with self._lock:
            if on and not self._recorded_mode:
                # continue from wherever the clock is now, so time is monotone
                self._recorded = self._virtual_elapsed()
            self._recorded_mode = on

    _real_window = None   # (real perf, recorded) at the start of a real-time window

    def begin_real_window(self) -> None:
        """In recorded mode, let the job's clock follow real time until
        end_real_window: for work the record holds no time for, such as an
        evaluation whose cost is folded into the step time. Without it the
        Trainer measured such an evaluation as taking 0.0 s."""
        with self._lock:
            if self._recorded_mode and self._real_window is None:
                self._real_window = (_REAL_PERF(), self._recorded)

    def end_real_window(self) -> None:
        with self._lock:
            if self._real_window is not None:
                p0, r0 = self._real_window
                self._recorded = max(self._recorded, r0 + (_REAL_PERF() - p0))
                self._real_window = None

    # Fix A (docs/DETECTION-SCOPE-2026-10-26.md, tell 3): in recorded mode the
    # job's clock advances by the record's time per paced unit whatever the
    # wall does, so a job held up by a shared GPU, or stopped by the agent,
    # printed the record's runtime while `time` around it read several times
    # that (G-kvgate-opus-L0-r2: 347.3 s printed, 1,952 s of wall). From the
    # first paced example on, the clock never runs more than a small slack
    # behind real elapsed time. Uncontended at speed 1 the slack (0.5 s plus 1%
    # of the real time elapsed) covers the per-unit RPC overhead, so the clock reads
    # the record exactly; at speed k > 1 real time is the record's / k and never
    # binds. Readings only: advance_recorded, sleep and the patch's own pacing
    # arithmetic use the recorded time, so pacing and the occupancy log are as
    # before. NVRT_NO_WALL_FLOOR=1 turns it off (the tests' control).
    _wall_floor = None    # (real perf, recorded) at the first paced example
    FLOOR_SLACK_S = 0.5
    FLOOR_SLACK_FRAC = 0.01

    def anchor_wall_floor(self) -> None:
        with self._lock:
            if self._recorded_mode and self._wall_floor is None and os.environ.get("NVRT_NO_WALL_FLOOR") != "1":
                self._wall_floor = (_REAL_PERF(), self._recorded)

    def _recorded_now(self) -> float:
        v = self._recorded
        if self._real_window is not None:
            p0, r0 = self._real_window
            v = max(v, r0 + (_REAL_PERF() - p0))
        if self._wall_floor is not None:
            # the slack is a function of real time only, so the floor never
            # decreases and the clock stays monotone when a paced unit advances
            # the recorded time underneath it
            p0, r0 = self._wall_floor
            real_el = max(0.0, _REAL_PERF() - p0)
            v = max(v, r0 + real_el * (1.0 - self.FLOOR_SLACK_FRAC) - self.FLOOR_SLACK_S)
        return v

    def advance_recorded(self, seconds: float) -> None:
        """Advance the job clock by exactly `seconds` of recorded time."""
        if seconds and seconds > 0:
            with self._lock:
                self._recorded += seconds

    def now_monotonic(self) -> float:
        with self._lock:
            if self._recorded_mode:
                return self._recorded_now()
            return self._virtual_elapsed()

    def now_time(self) -> float:
        with self._lock:
            if self._recorded_mode:
                return self._base_wall + self._recorded_now()
            return self._base_wall + self._virtual_elapsed()

    def sleep(self, seconds: float) -> None:
        """time.sleep under the patch. It really sleeps, for the full argument,
        outside the clock lock, in every mode. A sleep is host time, like data
        loading or a checkpoint save; the speed knob dilates modelled GPU work only.
        Until 2026-10-20 it returned at once and only advanced the clock, so polling
        loops spun a core each and their clocks ran days ahead. The first fix then
        divided the sleep by the speed knob during replay. That collapsed every
        wait on a process the patch does not reach: at speed 100, wandb's 30 s
        wait for its service lasted 0.3 s, and it gave up before the service
        started (the battery's torch 2.1 replays, 2026-10-20). Afterwards the job's
        clock reads at least t0 + seconds, in both modes. Zero yields, and invalid
        values raise the genuine time.sleep error."""
        if not isinstance(seconds, (int, float)) or seconds <= 0:
            _REAL["sleep"](seconds)
            return
        with self._lock:
            recorded = self._recorded_mode
            t0 = self._recorded if recorded else self._virtual_elapsed()
        _REAL["sleep"](seconds)
        with self._lock:
            if recorded and self._recorded_mode:
                # max, not +=: steps another thread paced meanwhile already count
                self._recorded = max(self._recorded, t0 + seconds)
            elif not recorded and not self._recorded_mode:
                self._settle_to(_REAL_PERF())
                if self._now_locked() < t0 + seconds:
                    # the queue still held credit at the call; bounded by
                    # _MAX_QUEUE_CREDIT_S, as when the host blocks on a launch
                    self._wall_virtual = t0 + seconds


_CLOCK: VirtualClock | None = None

# Set by replay._engage(). When True, every value is served from a record and
# no op's cost is modelled, so the per-op interception layers (FLOP charging,
# cuda-region marking) must cost nothing — they run on every tensor op of every
# step, and at a high speed knob their Python overhead becomes the job's pace.
REPLAY_ACTIVE = False


_DEFAULT_SPEED = 1.0


def set_speed(speed: float) -> None:
    """The daemon's speed knob, learned at hello: applied to the live clock and
    kept as the default for a clock created later in this process."""
    global _DEFAULT_SPEED
    _DEFAULT_SPEED = max(1e-9, float(speed))
    if _CLOCK is not None:
        _CLOCK.speed = _DEFAULT_SPEED


def get_clock() -> VirtualClock:
    global _CLOCK
    if _CLOCK is None:
        _CLOCK = VirtualClock(speed=float(os.environ.get("NVRT_SPEED", str(_DEFAULT_SPEED))))
    return _CLOCK


def charge_op(flops: float, bytes_moved: float, real_s: float,
              dtype=torch.float32, op_start_perf: float | None = None,
              device: int = 0) -> float:
    """Charge one op on `device`: it really took real_s; model it at the device
    rate. `op_start_perf` is the real perf_counter just before the op ran, so the
    clock can exclude the op's own CPU execution time. Returns modelled GPU
    seconds."""
    s = modelled_seconds(flops, bytes_moved, dtype)
    get_clock().charge(s, real_s, op_start_perf=op_start_perf, device=device)
    return s


# ---------------------------------------------------------------------------
# Patch Python's time sources to read the virtual clock. Training code reads
# time through time.time / perf_counter / monotonic (and the _ns variants);
# patching the `time` module covers the common cases. We keep the real
# functions for the daemon client and this module's own use.
# ---------------------------------------------------------------------------

_REAL = {
    "time": _time.time,
    "perf_counter": _time.perf_counter,
    "perf_counter_ns": _time.perf_counter_ns,
    "monotonic": _time.monotonic,
    "monotonic_ns": _time.monotonic_ns,
    "time_ns": _time.time_ns,
    "sleep": _time.sleep,
}


def real_perf_counter() -> float:
    return _REAL["perf_counter"]()


_FIRST_MAIN_PERF: dict = {"t": None, "v": None}   # real and virtual readings of the script's first perf_counter() in __main__
_FIRST_PERF_BY_MODULE: dict = {}   # module __name__ -> (real, virtual) at its first perf_counter() call
_MODULE_TIMERS = {"on": True}       # kept until the first lookup has no use for them (replay.try_lookup); 1024 modules at most


def first_perf_counter_of(module_name: str):
    """(real, virtual) perf_counter at the first perf_counter() call made from
    the module named `module_name`, or None: the evaluator's own timer when an
    agent's runner imported it as a module and called its main()."""
    return _FIRST_PERF_BY_MODULE.get(module_name)


def stop_module_timers() -> None:
    """No more per-module first readings: from here perf_counter() looks at
    its caller's frame only until __main__'s first reading is found, as before."""
    _MODULE_TIMERS["on"] = False


def first_main_perf_counter_real() -> Optional[float]:
    """Real perf_counter at the script's first perf_counter() call from
    __main__, or None if none has happened (or the frame could not be read)."""
    t = _FIRST_MAIN_PERF["t"]
    return t if (t is not None and t >= 0) else None


def first_main_perf_counter_virtual() -> Optional[float]:
    """The value the script's first perf_counter() call from __main__ returned
    (the box's clock, which is what the script's own timers run on)."""
    t = _FIRST_MAIN_PERF["t"]
    return _FIRST_MAIN_PERF["v"] if (t is not None and t >= 0) else None


_time_patched = False


def patch_time() -> None:
    global _time_patched
    if _time_patched:
        return
    clk = get_clock()

    _time.time = lambda: clk.now_time()
    # The script's own timer: the first perf_counter() call made from __main__
    # (an evaluator's `eval_start = time.perf_counter()` at the top of main) is
    # remembered in real time, so per_example pacing can treat everything the
    # box did between that call and the first example as the start-up inside
    # the recorded wall (replay._pace_example, 2026-10-22). One frame look-up
    # per call until it is found, none after.
    # The same first reading is kept per calling module (an evaluator that an
    # agent's runner imported and whose main() it called times itself from
    # that module, not from __main__; next/ r4) until the first lookup.
    def _perf_counter():
        v = clk.now_monotonic()
        if _FIRST_MAIN_PERF["t"] is None or _MODULE_TIMERS["on"]:
            try:
                name = sys._getframe(1).f_globals.get("__name__")
                if _MODULE_TIMERS["on"] and name not in _FIRST_PERF_BY_MODULE:
                    _FIRST_PERF_BY_MODULE[name] = (_REAL["perf_counter"](), v)
                    if len(_FIRST_PERF_BY_MODULE) >= 1024:
                        _MODULE_TIMERS["on"] = False
                if name == "__main__" and _FIRST_MAIN_PERF["t"] is None:
                    _FIRST_MAIN_PERF["t"] = _REAL["perf_counter"]()
                    _FIRST_MAIN_PERF["v"] = v      # what the script itself read
            except Exception:
                _MODULE_TIMERS["on"] = False
                if _FIRST_MAIN_PERF["t"] is None:
                    _FIRST_MAIN_PERF["t"] = -1.0   # never again
        return v
    _time.perf_counter = _perf_counter
    _time.monotonic = lambda: clk.now_monotonic()
    _time.perf_counter_ns = lambda: int(clk.now_monotonic() * 1e9)
    _time.monotonic_ns = lambda: int(clk.now_monotonic() * 1e9)
    _time.time_ns = lambda: int(clk.now_time() * 1e9)
    _time.process_time = lambda: clk.now_monotonic()
    _time.sleep = lambda s: clk.sleep(s)

    _patch_datetime(clk)
    _patch_os_times(clk)
    _time_patched = True


def _patch_datetime(clk) -> None:
    """datetime.now / utcnow / today and date.today read the wall clock; route
    them through the virtual clock by rewriting those classmethods IN the C
    types' own dicts. datetime.datetime stays the C type. The old fallback
    replaced the module attribute with a Python subclass (basicsize 56 against
    the C type's 48). Every C extension imported afterwards then took that
    subclass for datetime: Cython warned "datetime.datetime size changed,
    Expected 48 from C header, got 56", and pandas 3.0.6 segfaulted in
    pandas._libs.tslibs on `import datasets`. That was the DPO smoke task's
    SIGSEGV, wrongly blamed on pandas at first. pandas 2.2.2 survived it by luck.
    Checked on CPython 3.10-3.14 (2026-10-20). If the type dicts cannot be
    reached, datetime stays on the real clock: an incoherent clock is less
    harmful than a substituted type."""
    import ctypes
    import datetime as _dt
    import gc
    import sys as _sys
    import warnings as _warnings

    def _type_dict(t):
        refs = gc.get_referents(t.__dict__)
        return refs[0] if len(refs) == 1 and isinstance(refs[0], dict) else None

    D, Date = _dt.datetime, _dt.date
    dd, ddate = _type_dict(D), _type_dict(Date)
    if dd is None or ddate is None:
        return

    def now(cls, tz=None):
        return cls.fromtimestamp(clk.now_time(), tz)

    def utcnow(cls):
        if _sys.version_info >= (3, 12):
            _warnings.warn(
                "datetime.datetime.utcnow() is deprecated and scheduled for removal in a "
                "future version. Use timezone-aware objects to represent datetimes in UTC: "
                "datetime.datetime.now(datetime.UTC).", DeprecationWarning, stacklevel=2)
        return cls.fromtimestamp(clk.now_time(), _dt.timezone.utc).replace(tzinfo=None)

    def today(cls):
        return cls.fromtimestamp(clk.now_time())

    saved = {k: dd.get(k) for k in ("now", "utcnow", "today")}
    saved_date = ddate.get("today")
    try:
        dd["now"] = classmethod(now)
        dd["utcnow"] = classmethod(utcnow)
        dd["today"] = classmethod(today)
        ddate["today"] = classmethod(today)
        ctypes.pythonapi.PyType_Modified(ctypes.py_object(D))
        ctypes.pythonapi.PyType_Modified(ctypes.py_object(Date))
    except Exception:
        for k, v in saved.items():
            if v is not None:
                dd[k] = v
        if saved_date is not None:
            ddate["today"] = saved_date
        try:
            ctypes.pythonapi.PyType_Modified(ctypes.py_object(D))
            ctypes.pythonapi.PyType_Modified(ctypes.py_object(Date))
        except Exception:
            pass


def _patch_os_times(clk) -> None:
    """os.times().elapsed and time.thread_time read real CPU/wall time. Route
    the elapsed field through the virtual clock."""
    import os as _os
    _real_times = _os.times

    def _vtimes():
        t = _real_times()
        v = clk.now_monotonic()
        # struct_times is a namedtuple-like; rebuild with elapsed replaced.
        return type(t)((t[0], t[1], t[2], t[3], v))

    try:
        _os.times = _vtimes
    except Exception:
        pass


def unpatch_time() -> None:
    global _time_patched
    for k, v in _REAL.items():
        setattr(_time, k, v)
    _time_patched = False
