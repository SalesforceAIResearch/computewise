"""Surrogate timing and loss for configurations the records do not cover
(docs/SURROGATE-SERVING.md). gpu-tasks fits a family's forms into
registry/surrogates/<family>.json; this module loads them, resolves a launch's
configuration from its argv and its live fingerprint, and evaluates the loss
curve, the step timing and the memory for it, deterministically per seed.

Supported forms (anything else is refused at load, with the family named):
  loss:    power_law   L(s) = L_inf + A * (s + t0) ** -alpha, times the effects
  effects: log_ratio       * (x / ref) ** coef
           log_quadratic   * exp(curv * ln(x / opt) ** 2)
  noise:   ar1 over the validation grid, sd interpolated from sd_at_step
  timing:  steady_step_s   flops: 6 * params * tokens / (achieved_tflops * 1e12 * ref.world_size) + overhead_s
                           const
           startup_in_train_s / setup_s      const
           eval_s_per_validation             ratio_of_step | const
  memory:  peak_bytes      linear: base + bytes_per_param * params + bytes_per_batch_token * tokens
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Optional


# loss forms and the parameters each requires; `power_law` is the placeholder
# shape, `cumulative_lr` the schedule-driven law gpu-tasks fits (2026-10-21)
LOSS_FORMS = {"power_law": ("L_inf", "A", "t0", "alpha"),
              "cumulative_lr": ("L0", "A", "alpha", "C")}


class SurrogateError(ValueError):
    pass


def _num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


class Family:
    def __init__(self, spec: dict, path: Optional[Path] = None):
        self.spec = spec
        self.path = path
        self.name = str(spec.get("family") or (path.stem if path else "surrogate"))
        self.version = spec.get("version", 0)
        self.scripts = [str(s) for s in (spec.get("scripts") or [])]
        self.knobs = spec.get("knobs") or {}
        self._check()

    # -- validation ---------------------------------------------------------
    def _check(self) -> None:
        loss = self.spec.get("loss") or {}
        form = loss.get("form")
        if form not in LOSS_FORMS:
            raise SurrogateError(f"{self.name}: loss form {form!r} not supported")
        for k in LOSS_FORMS[form]:
            if _num((loss.get("params") or {}).get(k)) is None:
                raise SurrogateError(f"{self.name}: loss.params.{k} missing")
        for e in loss.get("effects") or []:
            if e.get("kind") not in ("log_ratio", "log_quadratic"):
                raise SurrogateError(f"{self.name}: effect kind {e.get('kind')!r} not supported")
            if e.get("knob") not in self.knobs:
                raise SurrogateError(f"{self.name}: effect on unknown knob {e.get('knob')!r}")
        noise = loss.get("noise") or {}
        if noise and noise.get("kind") not in ("ar1",):
            raise SurrogateError(f"{self.name}: noise kind {noise.get('kind')!r} not supported")
        t = self.spec.get("timing") or {}
        ss = t.get("steady_step_s") or {}
        if ss.get("form") not in ("flops", "const"):
            raise SurrogateError(f"{self.name}: timing.steady_step_s form {ss.get('form')!r} not supported")
        for k in ("startup_in_train_s", "setup_s"):
            f = (t.get(k) or {"form": "const", "value": 0.0}).get("form")
            if f != "const":
                raise SurrogateError(f"{self.name}: timing.{k} form {f!r} not supported")
        ev = t.get("eval_s_per_validation") or {"form": "const", "value": 0.0}
        if ev.get("form") not in ("ratio_of_step", "const"):
            raise SurrogateError(f"{self.name}: timing.eval_s_per_validation form {ev.get('form')!r} not supported")
        m = (self.spec.get("memory") or {}).get("peak_bytes") or {}
        if m.get("form") not in ("linear", "const"):
            raise SurrogateError(f"{self.name}: memory.peak_bytes form {m.get('form')!r} not supported")

    # -- matching -------------------------------------------------------------
    def serves(self, script_path: Optional[str]) -> bool:
        if not script_path:
            return False
        sp = str(script_path).replace("\\", "/")
        return any(sp == s or sp.endswith("/" + s) for s in self.scripts)

    # -- configuration ----------------------------------------------------------
    def resolve(self, argv: Optional[list], fingerprint_inputs: Optional[dict]) -> dict:
        """Knob values: fingerprint over argv over default; also every
        fingerprint field a form refers to (num_parameters, batch tokens)."""
        argv = [str(a) for a in (argv or [])]
        fp = fingerprint_inputs or {}
        cfg: dict[str, Any] = {}
        for name, k in self.knobs.items():
            val = None
            flag = k.get("flag")
            if flag:
                for i, a in enumerate(argv):
                    if a == flag and i + 1 < len(argv):
                        val = argv[i + 1]
                    elif a.startswith(flag + "="):
                        val = a.split("=", 1)[1]
            fpk = k.get("fingerprint")
            if fpk and fp.get(fpk) is not None:
                val = fp.get(fpk)
            if val is None:
                val = k.get("default")
            cfg[name] = _num(val, val)
        for extra in ("num_parameters", "global_batch_tokens", "micro_batch", "per_device_batch", "world_size"):
            if fp.get(extra) is not None and extra not in cfg:
                cfg[extra] = _num(fp.get(extra), fp.get(extra))
        return cfg

    def config_hash(self, cfg: dict) -> str:
        return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def seed(self, cfg: dict, launch_ordinal: int) -> int:
        h = hashlib.sha256(f"{self.name}:{self.version}:{self.config_hash(cfg)}:{launch_ordinal}".encode()).hexdigest()
        return int(h[:16], 16)

    # -- loss ------------------------------------------------------------------
    def _params(self, cfg: dict) -> float:
        p = _num(cfg.get("num_parameters"))
        if p:
            return p
        # a GPT-2-shaped estimate from the knobs when the fingerprint is not in yet
        d = _num(cfg.get("model_dim"), 768.0); n = _num(cfg.get("num_layers"), 12.0); v = _num(cfg.get("vocab_size"), 50304.0)
        return 12.0 * n * d * d + v * d

    def _tokens(self, cfg: dict) -> float:
        return _num(cfg.get("batch_tokens")) or _num(cfg.get("global_batch_tokens")) or 524288.0

    def _eta_series(self, step: float, cfg: dict, lr_history: Optional[list]) -> list:
        """The learning rate at every optimizer step 1..step of the group the
        form names (`lr_group`, default 0), from the run's observed stream
        ([[step, [per group]], ...] as the daemon keeps it), forward-filled
        over gaps and past the last observation; before any observation the
        knob `lr` (or the form's `lr_default`) held constant."""
        n = max(0, int(step))
        p = self.spec["loss"]["params"]
        gi = int(_num(p.get("lr_group"), 0) or 0)
        by_step: dict[int, float] = {}
        for item in lr_history or []:
            try:
                st, groups = int(item[0]), item[1]
                by_step[st] = float(groups[gi] if gi < len(groups) else groups[0])
            except (TypeError, ValueError, IndexError):
                continue
        fallback = _num(cfg.get("lr"), _num(p.get("lr_default"), 0.0)) or 0.0
        later = sorted(by_step)
        out, last = [], None
        for k in range(1, n + 1):
            if k in by_step:
                last = by_step[k]
            if last is not None:
                out.append(last)
            else:
                nxt = next((by_step[j] for j in later if j > k), None)
                out.append(nxt if nxt is not None else fallback)
        return out

    def base_loss(self, step: float, cfg: dict, lr_history: Optional[list] = None) -> float:
        loss = self.spec["loss"]; p = loss["params"]
        if loss.get("form") == "cumulative_lr":
            # Tissue et al. 2024, "Scaling Law with Learning Rate Annealing":
            # L(s) = L0 + A·S1(s)^-alpha − C·S2(s), S1 the summed LR, S2 the
            # annealing area with momentum decay lam; the curve answers to
            # whatever schedule, length or LR scale the agent's code chose
            etas = self._eta_series(step, cfg, lr_history)
            S1 = sum(etas)
            lam = float(_num(p.get("lam"), 0.999))
            m, S2, prev = 0.0, 0.0, (etas[0] if etas else 0.0)
            for eta in etas[1:]:
                m = lam * m + (prev - eta)
                S2 += m
                prev = eta
            floor = float(p["L0"])
            reducible = float(p["A"]) * max(S1, 1e-12) ** (-float(p["alpha"])) - float(p["C"]) * S2
        else:
            floor = float(p["L_inf"])
            A, t0, alpha = float(p["A"]), float(p["t0"]), float(p["alpha"])
            reducible = A * (max(0.0, float(step)) + t0) ** (-alpha)
        for e in loss.get("effects") or []:
            x = _num(cfg.get(e["knob"]))
            if x is None or x <= 0:
                continue
            if e["kind"] == "log_ratio":
                reducible *= (x / float(e["ref"])) ** float(e["coef"])
            elif e["kind"] == "log_quadratic":
                reducible *= math.exp(float(e["curv"]) * math.log(x / float(e["opt"])) ** 2)
        return floor + reducible

    def _sd_at(self, step: float) -> float:
        table = sorted((float(s), float(sd)) for s, sd in ((self.spec["loss"].get("noise") or {}).get("sd_at_step") or [[0, 0.0]]))
        if step <= table[0][0]:
            return table[0][1]
        for (s0, v0), (s1, v1) in zip(table, table[1:]):
            if s0 <= step <= s1:
                return v0 + (v1 - v0) * (step - s0) / max(1e-9, s1 - s0)
        return table[-1][1]

    def noise(self, step: float, seed: int) -> float:
        """AR(1) over the validation grid, deterministic per seed."""
        every = max(1, int((self.spec.get("validation_steps") or {}).get("every") or 125))
        rho = float((self.spec["loss"].get("noise") or {}).get("rho") or 0.0)
        k = int(step // every)
        e = 0.0
        for i in range(k + 1):
            z = random.Random(seed * 1000003 + i).gauss(0.0, 1.0)
            s_i = self._sd_at(i * every)
            e = rho * e + math.sqrt(max(0.0, 1.0 - rho * rho)) * s_i * z if i else s_i * z
        return e

    def diverged(self, step: float, cfg: dict) -> bool:
        d = self.spec["loss"].get("divergence") or {}
        when = d.get("when") or {}
        hit = False
        for key, thr in when.items():
            if key.endswith("_above"):
                x = _num(cfg.get(key[: -len("_above")]))
                hit = hit or (x is not None and x > float(thr))
            elif key.endswith("_below"):
                x = _num(cfg.get(key[: -len("_below")]))
                hit = hit or (x is not None and x < float(thr))
        return hit and step >= float(d.get("after_steps") or 0)

    def value(self, role: str, step: float, cfg: dict, seed: int, lr_history: Optional[list] = None) -> Optional[float]:
        """val_loss / train_loss (and loss) at a step; NaN after divergence;
        None for a role the surrogate does not model."""
        if role not in ("val_loss", "train_loss", "loss", "eval_loss"):
            return None
        if self.diverged(step, cfg):
            return float("nan")
        if step <= 0 and self.spec.get("init_loss"):
            il = self.spec["init_loss"]
            v = float(il.get("value", 10.9)) + random.Random(seed).gauss(0.0, float(il.get("sd", 0.0)))
            return v
        v = self.base_loss(step, cfg, lr_history) + self.noise(step, seed)
        if role in ("train_loss", "loss"):
            tl = self.spec.get("train_loss") or {}
            v = v - float(tl.get("gap_below_val", 0.0)) + random.Random(seed ^ int(step)).gauss(0.0, float(tl.get("noise_sd", 0.0)))
        return v

    # -- timing and memory ------------------------------------------------------
    def timing(self, cfg: dict) -> dict:
        t = self.spec["timing"]; ss = t["steady_step_s"]
        if ss["form"] == "flops":
            # achieved_tflops is per GPU; the reference step is on ref.world_size GPUs
            # (convert() rescales it for the launched count, as for a record)
            flops = float(ss.get("flops_per_token_per_param", 6)) * self._params(cfg) * self._tokens(cfg)
            ws = max(1.0, float((ss.get("ref") or {}).get("world_size") or 1))
            steady = flops / (float(ss["achieved_tflops"]) * 1e12 * ws) + float(ss.get("overhead_s", 0.0))
        else:
            steady = float(ss["value"])
        ev = t.get("eval_s_per_validation") or {"form": "const", "value": 0.0}
        eval_pv = steady * float(ev["factor"]) if ev["form"] == "ratio_of_step" else float(ev["value"])
        return {"steady_step_s": steady,
                "startup_in_train_s": float((t.get("startup_in_train_s") or {}).get("value", 0.0)),
                "setup_s": float((t.get("setup_s") or {}).get("value", 0.0)),
                "eval_s_per_validation": eval_pv,
                "ref": (ss.get("ref") or {}), "batch_policy": ss.get("batch_policy", "fixed_global_grow_accumulation")}

    def memory_bytes(self, cfg: dict) -> int:
        m = self.spec["memory"]["peak_bytes"]
        if m["form"] == "const":
            return int(m["value"])
        return int(float(m.get("base_bytes", 0)) + float(m.get("bytes_per_param", 0)) * self._params(cfg)
                   + float(m.get("bytes_per_batch_token", 0)) * self._tokens(cfg))

    def steps(self, cfg: dict) -> int:
        return int(_num(cfg.get("train_steps"), 0) or _num((self.knobs.get("train_steps") or {}).get("default"), 0) or 0)

    # -- the synthesised record the daemon serves -------------------------------
    def record(self, cfg: dict, seed: int, launch_ordinal: int, pending: bool) -> dict:
        """A record-shaped dict: _serving_info() and convert() treat it like a
        registry record; `surrogate` carries what the curve evaluation needs."""
        t = self.timing(cfg); steps = self.steps(cfg)
        n_val = steps // max(1, int((self.spec.get("validation_steps") or {}).get("every") or 125)) + 1
        return {
            "identity": {"record_id": f"surrogate:{self.name}:v{self.version}:{self.config_hash(cfg)}:{launch_ordinal}",
                         "match_policy": "surrogate", "replicate_group": f"surrogate:{self.name}:{self.config_hash(cfg)}"},
            "mode": "replay", "steps": steps, "planned_steps": steps,
            "per_step_gpu_s": t["steady_step_s"],
            "timing": {"pacing": "per_step", "steady_step_s": t["steady_step_s"], "first_step_s": None,
                       "startup_in_train_s": t["startup_in_train_s"], "setup_s": t["setup_s"],
                       "eval_s": t["eval_s_per_validation"] * n_val, "train_time_s": t["steady_step_s"] * steps + t["startup_in_train_s"]},
            "memory": {"peak_allocated_bytes": self.memory_bytes(cfg), "peak": self.memory_bytes(cfg)},
            "launch": {"world_size": {"value": int((t["ref"] or {}).get("world_size") or 1)}, "batch_policy": t["batch_policy"]},
            "series": {}, "materialisation": {"serving": {"mode": "per_op"}},
            "readiness": {"replay": True}, "failures": [],
            "surrogate": {"family": self.name, "version": self.version, "config": cfg,
                          "config_hash": self.config_hash(cfg), "seed": seed, "launch_ordinal": launch_ordinal,
                          "pending": pending, "n_validations": n_val,
                          # the family's resolved step count (knob, argv or fingerprint);
                          # the agent's code decides when it stops, so the authoritative
                          # count is the last enqueue step at job_bye
                          "steps": steps},
        }


def load_families(root: Optional[Path]) -> dict[str, "Family"]:
    """Every registry/surrogates/*.json; a bad file is refused loudly."""
    out: dict[str, Family] = {}
    if not root:
        return out
    d = Path(root)
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        if p.name.startswith("._"):
            continue
        fam = Family(json.loads(p.read_text()), p)
        out[fam.name] = fam
    return out
