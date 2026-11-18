"""Structural fingerprint canonicalisation — the shared definition both gpu-mock
and gpu-tasks compute, so a launched process hashes to the same value the
registry stored.

The fingerprint is the second identity key (after the exact entry-file hash): it
matches an agent's launch to a sweep-grid record when the bytes differ but the
*structure* is the same (REPLAY-INTERFACE §2). It must be computed identically on
both sides, so the canonicalisation is pinned here and gpu-tasks' generators load
this file from its path and run it against each record's
`identity.fingerprint_inputs`; `build/validate.py` recomputes every stored hash.

## Version fp256/0.2 (2026-10-19) — what changed from 0.1 and why

Two fields of 0.1 were computed differently per source, so a live runtime could
not reproduce them before it had matched a record (which is the point):

- field 6 was a shape (`[64, 1024]` for track 3, `[16]` for TRL). Under the HF
  Trainer the sequence dimension changes every step (padding to the batch's
  longest), and DPO's forward sees 2×B (chosen+rejected concatenated), so a
  forward shape is not a stable key. It is now ONE INTEGER: the per-device
  micro-batch in sequences — the DataLoader's batch_size when the script builds
  one, else the first forward's leading dimension.
- field 8 was tokens for track 3 and derived sequences for TRL. It is now ALWAYS
  sequences = micro_batch × accumulation × world_size; a source that gives
  tokens is converted via T = micro_batch_shape[1]. The unit field is gone.

Every 0.1 hash changes under 0.2; the registry rebuilds from this file.

## Version fp256/0.3 (2026-10-20) — the model's parameter count, field 11

Fields 1-10 describe a trunk, and two heads on one trunk shared them: TRL's
sft.py (OPTForCausalLM, 331,196,416 parameters) and reward_modeling.py
(OPTForSequenceClassification, 331,196,928) on facebook/opt-350m had one 0.2
fingerprint across 19 structural_verified records with the same learning rate,
so a structural draw could hand an SFT job a reward model's record, pacing it at
half its step time with the wrong losses.
Field 11 is the total parameter count of the root module, every parameter
whether trainable or not, before any wrapper (DDP, FSDP, peft). It is exact,
known at the first forward, and carries no class name. Every 0.2 hash changes
under 0.3; the registry rebuilds from this file.

## The ordered field list (11 fields)

Each normalised as stated. An absent field is JSON null and still occupies its
slot, so the hash is stable across sources that carry different detail.

  1. model_family    str/null   lowercased; aliases folded; unknown names stripped
                                to alphanumerics (gpt_neox → gptneox)
  2. num_layers      int/null
  3. model_dim       int/null
  4. vocab_size      int/null
  5. optimizers      list[str]  class names lowercased and SORTED
  6. micro_batch     int/null   per-device batch in sequences (see above)
  7. world_size      int/null
  8. global_batch    int/null   sequences = micro_batch × accumulation × world_size
  9. precision       str/null   "bf16"|"fp16"|"fp32", lowercased
 10. torch_compile   bool
 11. num_parameters  int/null   sum of numel() over root.parameters() (0.3)

## How it is hashed

  obj = [field1, …, field11]                       # a LIST, order explicit
  s = json.dumps(obj, sort_keys=False, separators=(",", ":"), ensure_ascii=True)
  "fp256:" + sha256(s.encode("utf-8")).hexdigest()

## Input keys accepted (either source's names)

  model | num_layers | model_dim | vocab_size | optimizers | precision |
  torch_compile | world_size | num_parameters |
  micro_batch (int)  OR  per_device_batch (int)  OR  micro_batch_shape ([B, T])
  accumulation | accumulation_steps  (default 1)
  global_batch_sequences  OR  global_batch_tokens (converted with T)  OR derived

## Conventions for the raw strings (pinned with the registry)

- model: HF config.model_type when the root module has one, else the root
  class name lowercased (`gpt` for track 3 / MLS-Bench).
- precision: the mixed-precision mode — bf16/fp16 when a Trainer flag or an
  autocast context sets one, "fp32" when neither does; never the parameter dtype
  and never null.
- optimizers: the Optimizer subclass names constructed.
- torch_compile: only a torch.compile() on the model itself counts.
- num_parameters: sum of numel() over root.parameters(), trainable or not,
  before any wrapper (DDP, FSDP, peft) is applied; a record derives it from
  the architecture or the checkpoint (artefacts.model_parameters).

This file is the authority. If gpu-tasks' generator and this disagree, they must
be reconciled here first.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

VERSION = "fp256/0.3"

_MODEL_ALIASES = {
    "gpt-2": "gpt2", "gpt2": "gpt2",
    "nanogpt": "nanogpt", "modded-nanogpt": "nanogpt", "modded_nanogpt": "nanogpt",
    "gpt_nano": "nanogpt",
}


def _norm_model(name: Any) -> str | None:
    if name is None:
        return None
    s = str(name).strip().lower()
    if s in _MODEL_ALIASES:
        return _MODEL_ALIASES[s]
    return "".join(c for c in s if c.isalnum())


def _as_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _norm_optimizers(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    return sorted(str(o).strip().lower() for o in v)


def _norm_precision(v: Any) -> str | None:
    if v is None:
        return None
    return str(v).strip().lower()


def canonical_list(inputs: dict) -> list:
    """Build the ordered, normalised 11-field list from a fingerprint_inputs
    dict. Accepts the field names either source writes; missing fields are null.

    Field 11, num_parameters (fp256/0.3): the model's total parameter count,
    every parameter of the root module whether trainable or not. Fields 1-10
    describe the trunk, and two different heads on one trunk share them: TRL's
    sft.py (OPTForCausalLM, 331,196,416 parameters) and reward_modeling.py
    (OPTForSequenceClassification, 331,196,928) on opt-350m had one fp256/0.2
    fingerprint across 19 records, so a structural draw could serve a reward
    model's record to an SFT job at half its step time.
    The count is exact, known at the first forward, and free of class names."""
    shape = inputs.get("micro_batch_shape")
    shape = [_as_int(x) for x in shape] if shape else None

    # field 6: per-device micro-batch in sequences (one integer)
    micro = _as_int(inputs.get("micro_batch"))
    if micro is None:
        micro = _as_int(inputs.get("per_device_batch"))
    if micro is None and shape:
        micro = shape[0]

    # sequence length, only needed to convert a tokens global batch
    seq_len = shape[1] if shape and len(shape) >= 2 else _as_int(inputs.get("sequence_length"))

    world_size = _as_int(inputs.get("world_size"))
    accum = _as_int(inputs.get("accumulation", inputs.get("accumulation_steps")))
    if accum is None:
        accum = 1

    # field 8: global batch in sequences, always
    gb = _as_int(inputs.get("global_batch_sequences"))
    if gb is None:
        toks = _as_int(inputs.get("global_batch_tokens"))
        if toks is not None and seq_len:
            gb = toks // seq_len
    if gb is None and micro is not None and world_size is not None:
        gb = micro * accum * world_size

    return [
        _norm_model(inputs.get("model")),
        _as_int(inputs.get("num_layers")),
        _as_int(inputs.get("model_dim")),
        _as_int(inputs.get("vocab_size")),
        _norm_optimizers(inputs.get("optimizers")),
        micro,
        world_size,
        gb,
        _norm_precision(inputs.get("precision")),
        bool(inputs.get("torch_compile", False)),
        _as_int(inputs.get("num_parameters")),
    ]


def fingerprint(inputs: dict) -> str:
    obj = canonical_list(inputs)
    s = json.dumps(obj, sort_keys=False, separators=(",", ":"), ensure_ascii=True)
    return "fp256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        with open(path) as fh:
            rec = json.load(fh)
        fi = rec.get("identity", {}).get("fingerprint_inputs", {})
        print(f"{fingerprint(fi)}  {path}")
