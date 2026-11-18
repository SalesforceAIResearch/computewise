"""convert(record, gpu_count, launch) — one converter, two callers.

A record's times are on its recorded hardware and GPU count, unconverted
(RECORD-FORMAT §6). Almost every LLM-era record used 8 GPUs; the mock box has 4,
and the agent may give a job 1, 2 or 4. This function turns a record's timing
into the timing for the launched GPU count.

It NEVER converts across GPU *types*. A record from a V100 or A100 run is served
as its recorded seconds under its recorded identity; no card-speed factor is
applied, ever (analysis, 2026-11-02: "serve recorded seconds as they are, under
the recorded identity, never with a card factor" — the rule for any non-H100
source). A package that wants a non-H100 source to present that card gives the
box that card's NVML identity (an isolation image change), not a time factor
here. GPU-COUNT conversion below is the only conversion this function does. The daemon calls it at lookup to pace
the job, and the reference-schedule solver calls it to build each job's 1/2/4-GPU
modes — the same function, so the mock's durations and the best-possible
schedule's durations are the same numbers by construction
(GPU-COUNT-CONVERSION-NOTE §5).

What changes with the GPU count depends on the script's batch policy
(GPU-COUNT-CONVERSION-NOTE §7, RECORD-FORMAT §7):

  fixed_global_grow_accumulation   global batch kept; each GPU runs more
      accumulation micro-steps of the SAME micro-batch. Batching gains nothing,
      so step time scales with recorded_gpus / gpu_count: ~2× for 8→4 on NVLink.
      The recorded curve still applies. (track 3, MLS-Bench)
  fixed_global_grow_microbatch     global batch kept; each GPU's micro-batch
      grows. One GPU's step time grows sub-linearly with its batch until it
      saturates: ElasticFlow A100 tables give 1.07–1.37× for 8→4. We use a
      per-doubling factor (default 1.25, the midpoint) until a per-family
      one-GPU curve is calibrated; the provenance says so. Curve applies.
  fixed_per_device                 per-device batch kept; the global batch and
      the step count change, so the recorded curve does NOT apply: another record
      or the modelled path (curve_applies=False).

Components that do not scale with GPU count — setup, compile warm-up, the
first step's one-off cost beyond its steady share, checkpoint writes — are left
as recorded. Validation (eval) scales like a step: each rank runs more eval
micro-batches.

Per-GPU memory: under grow_accumulation it is unchanged (same micro-batch);
under grow_microbatch activations grow with the micro-batch; sharded state
(ZeRO-3/FSDP) grows as recorded_gpus / gpu_count — a job that fit on 8×80 GB may
not fit on 4. Reported as a factor; the daemon's memory accounting applies it.

Scaling to MORE GPUs than recorded is not extrapolated: the factor is clamped at
1.0 (never faster than recorded), with provenance "clamped".

Sharded state (memory.sharded_bytes, from 2026-10-31): a record may say how many
of its per-GPU peak bytes are state sharded across its GPUs (ZeRO-3 / FSDP:
parameters, gradients, optimizer state split by world size). On a launch of a
different width that part scales by recorded_gpus / gpu_count and the rest of
the peak (activations, replicated buffers, context) stays as recorded, so a
model that fits on 4 cards may not fit on 1. The result is a memory factor on
the peak, `memory_factor`, as before; time is untouched. The per-card weights
held at engagement scale with the sharded share as well (`weights_factor`).
Absent from a record, nothing changes. Unlike time, this is not clamped: more
GPUs than recorded hold less sharded state per card, as on hardware.
"""

from __future__ import annotations

import math

_GROW_MICROBATCH_PER_DOUBLING = 1.25     # ElasticFlow 8→4 band 1.07–1.37; calibrate


def _val(v, default=None):
    if isinstance(v, dict) and "value" in v:
        return v["value"] if v["value"] is not None else default
    return v if v is not None else default


def convert(serving: dict, gpu_count: int, launch: dict | None = None) -> dict:
    """serving: the daemon's serving-info dict for a record (timings on the
    recorded hardware, plus world_size_recorded and batch_policy).
    gpu_count: GPUs the agent gave the job. launch: optional launch config
    (unused today; reserved for an explicit accumulation override).

    Returns a dict with the converted timing fields, curve_applies, a per-GPU
    memory factor, and provenance."""
    rec_gpus = int(_val(serving.get("world_size_recorded"), gpu_count) or gpu_count)
    policy = _val(serving.get("batch_policy")) or "fixed_global_grow_accumulation"
    gpu_count = max(1, int(gpu_count))
    ratio = rec_gpus / gpu_count                      # >1 means fewer GPUs now

    out = {
        "from_gpus": rec_gpus, "to_gpus": gpu_count, "batch_policy": policy,
        "curve_applies": True, "time_factor": 1.0, "memory_factor": 1.0,
        "provenance": "identity",
    }
    if rec_gpus == gpu_count:
        return _apply(serving, out)

    if ratio < 1.0:
        out.update(time_factor=1.0, memory_factor=1.0,
                   provenance="clamped: more GPUs than recorded is not extrapolated")
        return _apply(serving, out)

    if policy == "fixed_per_device":
        out.update(curve_applies=False, time_factor=None, memory_factor=1.0,
                   provenance="fixed_per_device: global batch changes, recorded "
                              "curve does not apply; use another record or the "
                              "modelled path")
        return _apply(serving, out)

    if policy == "fixed_global_grow_microbatch":
        doublings = math.log2(ratio)
        f = _GROW_MICROBATCH_PER_DOUBLING ** doublings
        out.update(time_factor=f, memory_factor=ratio,
                   provenance=f"grow_microbatch: {_GROW_MICROBATCH_PER_DOUBLING}× "
                              f"per doubling of per-GPU batch (ElasticFlow 8→4 "
                              f"band 1.07–1.37; uncalibrated midpoint)")
        return _apply(serving, out)

    # default: fixed_global_grow_accumulation
    out.update(time_factor=ratio, memory_factor=1.0,
               provenance="grow_accumulation: same micro-batch run ratio× more "
                          "times per step; ~2× for 8→4 (track-3 logs 1.8×)")
    return _apply(serving, out)


def _sharded(serving: dict, out: dict) -> dict:
    """Fold a record's sharded state into the memory factor (docstring above):
    peak' = (peak - sharded) * f + sharded * recorded / launch, where f is the
    batch policy's own factor on the unsharded part. Records without the field
    are returned unchanged."""
    mem = serving.get("memory") or {}
    sh = _val(mem.get("sharded_bytes"))
    peak = _val(mem.get("peak_bytes"))
    if not sh or not peak:
        return out
    sh, peak = float(sh), float(peak)
    sh = min(sh, peak)
    rec_gpus, gpu_count = out["from_gpus"], out["to_gpus"]
    f_rest = float(out.get("memory_factor") or 1.0)
    new_peak = (peak - sh) * f_rest + sh * rec_gpus / gpu_count
    w = _val(mem.get("weights_bytes"))
    wf = 1.0
    if w:
        w = float(w)
        wsh = min(w, sh)                  # the weights' sharded part is at most the state's
        wf = ((w - wsh) + wsh * rec_gpus / gpu_count) / w
    out.update(memory_factor=new_peak / peak, weights_factor=wf, sharded_bytes=sh,
               memory_provenance=f"sharded state {sh / 2 ** 30:.2f} GiB of a {peak / 2 ** 30:.2f} GiB "
                                 f"per-GPU peak, scaled {rec_gpus}->{gpu_count} GPUs; the rest x{f_rest:g}")
    return out


def _apply(serving: dict, out: dict) -> dict:
    """Produce the converted timing fields from the factor."""
    out = _sharded(serving, out)
    f = out["time_factor"]
    steady = serving.get("steady_step_s")
    first = serving.get("first_step_s")
    eval_pv = serving.get("eval_s_per_validation")
    if f is None or steady is None:
        out.update(steady_step_s=steady, first_step_s=first,
                   eval_s_per_validation=eval_pv)
        return out
    out["steady_step_s"] = float(steady) * f
    # the first step = steady share (scales) + one-off startup (does not)
    if first is not None:
        one_off = max(0.0, float(first) - float(steady))
        out["first_step_s"] = one_off + float(steady) * f
    else:
        out["first_step_s"] = None
    out["eval_s_per_validation"] = (float(eval_pv) * f) if eval_pv else eval_pv
    out["startup_in_train_s"] = serving.get("startup_in_train_s")   # unscaled
    out["setup_s"] = serving.get("setup_s")                         # unscaled
    return out
