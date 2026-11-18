#!/usr/bin/env python3
"""Compare what a replayed Hugging Face Trainer job printed with the record it
replayed. Works for any Trainer script (TRL's DPO, SFT and reward modelling).

    python3 -I tests/check_trainer_replay.py <job.log> <record.json>

The Trainer prints one dict per logging step, e.g. {'loss': 0.69,
'learning_rate': ..., 'rewards/accuracies': ..., 'epoch': 0.02}, one per
evaluation, e.g. {'eval_loss': ..., 'eval_accuracy': ..., 'eval_runtime': ...},
and a final {'train_runtime': ..., 'train_loss': ...}. The n-th training dict
is matched to the n-th step of the record's train_loss series, and the n-th
evaluation dict to the n-th step of its eval_loss series. Both follow the run's
logging and evaluation steps. Keys map to record roles: loss -> train_loss,
grad_norm, learning_rate, a/b -> train.a.b, and eval_x -> eval_x (or eval.x).
Tolerance: the Trainer rounds the training loss to 4 decimals, and prints
everything else in full, compared at relative 1e-6. learning_rate is not
served: it comes from the job's own scheduler, so agreement shows the job is
configured like the recorded run. The final train_loss is the Trainer's mean
over all steps and is compared with result.trainer_train_loss (a TRL record's
result.train_loss is the last logged value), within 2e-4. The final
train_runtime, read on the job's own clock, must lie between the record's paced
step time (steps x steady_step_s) and its train_time_s plus 10 %.
"""
from __future__ import annotations
import ast, json, re, sys


def series_at(series, step):
    if isinstance(series, dict):
        for s, v in zip(series.get("steps") or [], series.get("values") or []):
            if s == step:
                return v
        return None
    if isinstance(series, list) and 0 <= step < len(series):
        return series[step]
    return None


def steps_of(series):
    if isinstance(series, dict):
        return list(series.get("steps") or [])
    return list(range(len(series or [])))


def val(x):
    return x.get("value") if isinstance(x, dict) else x


def train_role(k, series):
    r = {"loss": "train_loss"}.get(k, k)
    if r in series:
        return r
    if "/" in k and "train." + k.replace("/", ".") in series:
        return "train." + k.replace("/", ".")
    return None


def eval_role(k, series):
    if k in series:
        return k
    alt = "eval." + k[len("eval_"):].replace("/", ".")
    return alt if alt in series else None


def compare(logs, steps, series, role_of, label, bad, worst):
    n = 0
    for i, d in enumerate(logs[: len(steps)]):
        st = steps[i]
        for k, v in d.items():
            role = role_of(k, series)
            if role is None or not isinstance(v, (int, float)):
                continue
            want = series_at(series[role], st)
            if want is None:
                continue
            tol = 5.1e-5 if k == "loss" else max(1e-12, 1e-6 * abs(float(want)))
            err = abs(float(v) - float(want))
            n += 1
            worst[k] = max(worst.get(k, 0.0), err)
            if err > tol:
                bad.append((f"{label} step {st}", k, v, want))
    return n


def main(log_path, rec_path):
    text = open(log_path, errors="replace").read().replace("\r", "\n")
    rec = json.load(open(rec_path))
    series = rec.get("series") or {}
    dicts = []
    for m in re.finditer(r"\{'[a-z_/]+': [^{}]*\}", text):
        try:
            dicts.append(ast.literal_eval(m.group(0)))
        except Exception:
            pass
    train_logs = [d for d in dicts if "loss" in d and "train_runtime" not in d]
    eval_logs = [d for d in dicts if "eval_loss" in d]
    final = next((d for d in dicts if "train_runtime" in d), None)
    t_steps = steps_of(series.get("train_loss"))
    e_key = next((k for k in ("eval_loss", "eval.loss") if k in series), None)
    e_steps = steps_of(series.get(e_key)) if e_key else []
    print(f"training dicts printed: {len(train_logs)}; record train_loss steps: {len(t_steps)}")
    if e_key:
        print(f"evaluation dicts printed: {len(eval_logs)}; record {e_key} steps: {len(e_steps)}")
    bad, worst = [], {}
    n = compare(train_logs, t_steps, series, train_role, "train", bad, worst)
    n += compare(eval_logs, e_steps, series, eval_role, "eval", bad, worst)
    for k in sorted(worst):
        print(f"  {k:24s} max |printed - record| = {worst[k]:.3g}")
    print(f"compared {n} values; mismatches: {len(bad)}")
    if final:
        rt = val((rec.get("timing") or {}).get("train_time_s"))
        res = val((rec.get("result") or {}).get("trainer_train_loss"))
        last = val((rec.get("result") or {}).get("train_loss"))
        print(f"final: train_runtime {final.get('train_runtime')} s on the job's clock (record train_time_s {rt}); "
              f"train_loss {final.get('train_loss')} (record trainer_train_loss {res}; last logged {last})")
        if res is not None and final.get("train_loss") is not None and abs(float(final["train_loss"]) - float(res)) > 2e-4:
            bad.append(("final", "train_loss", final["train_loss"], res))
        # The job's own clock reads recorded time under replay, whatever the speed
        # knob: it advances by the record's step time at each optimizer step. So
        # train_runtime should be at least steps x steady_step_s (the paced part),
        # and at most the record's train_time_s, which also holds work replay does
        # not pace (checkpoint saves), with 10 % to spare. On torch 2.1 the clock
        # patch once failed silently, and jobs reported their real wall time
        # instead: 86 s against a paced 59.5 s (2026-10-20).
        rtv = final.get("train_runtime")
        st = val((rec.get("timing") or {}).get("steady_step_s"))
        nsteps = rec.get("steps")
        if rtv is not None and st and nsteps:
            lo = 0.98 * float(st) * int(nsteps)
            hi = 1.10 * max(float(rt or 0.0), float(st) * int(nsteps))
            print(f"job clock: train_runtime {float(rtv):.1f} s; paced steps {float(st) * int(nsteps):.1f} s; "
                  f"accepted {lo:.1f}-{hi:.1f} s")
            if not (lo <= float(rtv) <= hi):
                bad.append(("final", "train_runtime (job clock)", rtv, f"{lo:.1f}-{hi:.1f}"))
    else:
        print("final: no train_runtime dict printed (the job did not finish training)")
    zero = [d for d in eval_logs if "eval_runtime" in d and not float(d["eval_runtime"]) > 0]
    if eval_logs:
        print(f"evaluations with eval_runtime <= 0 on the job's clock: {len(zero)} of {len(eval_logs)}")
    if zero:
        bad.append(("eval", "eval_runtime", zero[0].get("eval_runtime"), "> 0"))
    for b in bad[:10]:
        print(f"  MISMATCH {b[0]} {b[1]}: printed {b[2]} record {b[3]}")
    ok = (bool(train_logs) and len(train_logs) >= len(t_steps) and len(eval_logs) >= len(e_steps)
          and final is not None and not bad)
    print("RECORD CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
