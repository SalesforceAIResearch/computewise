"""Derived prefix records: a job that is the first k steps of a recorded modded-nanogpt run.

usage: python3 -I build/gen_prefix.py <registry dir>

Run after build/gen_track3.py, before build/gen_tasks.py (build/rebuild.sh does). Writes one record per parent record of each
variant in PREFIX_SPEC into <registry>/records/, and the prefix script into <registry>/artifacts/mng3/scripts/<hash12>/.

The design is PACKAGES-2026-10-20.md §1.2 (the study's packages brief, approved 2026-10-20 ~17:10 UTC).

The script change. A prefix script is the parent's script with one change, made by stop_after():
- a constant `stop_after_step = k` on the line after the loop bound (`train_steps = ...`, or `run_stop_step = ...` where the
  script loops to that);
- the loop's end check `if step == train_steps:` becomes `if step == train_steps or step == stop_after_step:`.
Nothing else changes: the learning-rate schedule still reads `train_steps` or `FINAL_SCHEDULE_STEPS`, and the data order, the
seed and every hyperparameter are the parent's. So the prefix job's first k steps are the parent's first k steps, at every k.
k must be one of the parent's validation steps (every 125 in these scripts), so the run ends on a real validation loss, the
last line it prints.

The derived record, under gpu-mock's three conditions (2026-10-20, PACKAGES §1.2):
- steps and every series end at k;
- timing.eval_s is scaled to the validations kept, so each validation is paced as in the parent;
- failures past k are dropped (no parent here has one);
- the record is exact_only, in its own replicate group (the prefix script's config hash), because it shares the parent's
  fingerprint and its first k steps of learning rate.
first_step_s, startup_in_train_s, steady_step_s and setup_s are the parent's. identity.derived_from names the parent and the
change; gpu-mock logs it in the lookup event.

Only parents that replay as briefed are used: readiness true, recorded on torch 2.11.0 (the mng3 image's), a script that reads
no setting from the environment, and a seed taken from --seed or none at all.
"""
import copy
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

# variant -> (parent replicate group, k). k is chosen per variant so that a one-GPU job is about 22-28 minutes at speed 1
#.
PREFIX_SPEC = {
    "contra-soft-muon": ("mng3-cfg-d649f96e97a1", 1000),       # 20260509_contra_soft_muon, 30 seeds
    "soda-fade": ("mng3-cfg-aa973e631555", 1000),              # 20260518_soda_fade_3000_n9, 9 seeds
    "radial-damp": ("mng3-cfg-f04ccdc1544e", 875),             # 20260511_dampen_radial_gradient_component, 9 seeds on 2.11
    "muonh-tuned": ("mng3-cfg-d7542e12eaf3", 1125),            # 20260611_muonh_tuned_aux_3250, 10 runs, unseeded
    "newton-muon": ("mng3-cfg-1a80484e7aec", 875),             # 20260505_newton_muon, 15 runs, unseeded
    "aurora-ema-ref": ("mng3-cfg-8461a69fc755", 250),          # 20260525_aurora_ema_ref, 16 seeds
    "ema-nesterov-aurora": ("mng3-cfg-47e82714e8fe", 375),     # 20260522_ema_nesterov_aurora_proj, 8 seeds on 2.11
    "circuit-muon": ("mng3-cfg-c736a13dc03f", 625),            # 20260523_circuit_muon, 1 run (seed 6)
    "ortho": ("mng3-cfg-0b8aa31c1e43", 1375),                  # 20260517_ortho, 1 run, unseeded
}
# Wave Q (the study, 2026-10-21 ~01:12 UTC, relayed by the study: "run shorter jobs just to make progress faster ... then
# we can like foolproof it with the longer jobs"): a second prefix per variant, at the validation step nearest 8 minutes on one
# GPU at speed 1 and not under 5. aurora-ema-ref is left out: its shortest validation step is 13 minutes.
PREFIX_Q = {
    "contra-soft-muon": 250,      # 6.6 min
    "soda-fade": 250,             # 6.6 min
    "radial-damp": 250,           # 6.8 min
    "muonh-tuned": 250,           # 8.5 min
    "newton-muon": 250,           # 7.0 min
    "ema-nesterov-aurora": 125,   # 9.2 min
    "circuit-muon": 125,          # 5.7 min
    "ortho": 375,                 # 7.2 min
}
# Further prefixes a package needs: P20-tail-q's tail job, about six times its other jobs.
PREFIX_EXTRA = [("contra-soft-muon", 2000),    # 50 min
                ("contra-soft-muon", 500),     # 13 min: stagger-q's spread of lengths
                ("contra-soft-muon", 750),     # 20 min
                # pkg-stagger, stagger-q's confirm-phase twin: runs of about 20-75 minutes
                ("contra-soft-muon", 875), ("contra-soft-muon", 1500), ("contra-soft-muon", 2750),
                ("soda-fade", 2000), ("soda-fade", 3000),
                ("radial-damp", 1750), ("radial-damp", 2500),
                ("ema-nesterov-aurora", 1000),
                # G1-Q, the goal-level screening question (GOAL-PACKAGES-2026-10-21.md): every candidate at step 1000
                ("radial-damp", 1000), ("newton-muon", 1000), ("muonh-tuned", 1000),
                # G1-Q's question step: 1,250 steps, where the two leaders sit 0.002 apart with sd 0.0016
                ("contra-soft-muon", 1250), ("soda-fade", 1250), ("radial-damp", 1250), ("muonh-tuned", 1250),
                ("ema-nesterov-aurora", 1250)]
TORCH = "2.11.0"
LAUNCHER_ENV = {"LOCAL_RANK", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
ENV_READ = re.compile(r"""os\.environ(?:\.get)?\s*[\(\[]\s*["']([A-Za-z_0-9]+)["']|os\.getenv\(\s*["']([A-Za-z_0-9]+)["']""")
BREAK = re.compile(r"^([ \t]*)if step == (train_steps|run_stop_step):[ \t]*\n(\1[ \t]+)break[ \t]*$", re.M)
STOP_COMMENT = "# short run: stop after this step's validation; the learning-rate schedule is unchanged"


def stop_after(code, k):
    """The parent script with the documented stop after step k. Refuses a script where the change is not unambiguous."""
    if "stop_after_step" in code:
        raise ValueError("the script already names stop_after_step")
    m = BREAK.findall(code)
    if len(m) != 1:
        raise ValueError(f"{len(m)} loop-end checks `if step == train_steps:` / `run_stop_step:` followed by break")
    var = m[0][1]
    defs = list(re.finditer(rf"^([ \t]*){var} = [^\n]+$", code, re.M))
    if len(defs) != 1:
        raise ValueError(f"{len(defs)} assignments to {var}")
    d = defs[0]
    out = code[:d.end()] + f"\n{d.group(1)}stop_after_step = {k}  {STOP_COMMENT}" + code[d.end():]
    out, n = BREAK.subn(lambda b: f"{b.group(1)}if step == {var} or step == stop_after_step:\n{b.group(3)}break", out, count=1)
    assert n == 1
    return out


def config_hash(code, base):
    """gen_track3's replicate-group key: the script bytes with any `SEED = N` line blanked, plus the script name."""
    norm = re.sub(r"^SEED\s*=\s*\d+\s*$", "SEED = <seed>", code, flags=re.M)
    return C.sha256_text(norm + "\0" + base)


def env_settings(code):
    return sorted({a or b for a, b in ENV_READ.findall(code)} - LAUNCHER_ENV)


def seed_how(r):
    ident = r["identity"]
    if "--seed" in ident["argv"]:
        return "argv"
    if ident.get("env"):
        return "env"
    if ident["seed"]["value"] is not None and "constant" in (ident["seed"].get("how") or ""):
        return "constant"
    return "none"


def cut(series, k, how):
    st, vs = series["steps"], series["values"]
    keep = [i for i, s in enumerate(st) if s <= k]
    out = {kk: v for kk, v in series.items() if kk not in ("steps", "values", "how")}
    out.update({"steps": [st[i] for i in keep], "values": [vs[i] for i in keep],
                "how": (series.get("how") + "; " if series.get("how") else "") + how})
    return out


def derive(parent, code, k, reg):
    """The prefix record of `parent` at k, and the prefix script's text."""
    pid = parent["identity"]["record_id"]
    vl = parent["series"]["val_loss"]
    if k not in vl["steps"]:
        raise ValueError(f"{pid}: k={k} is not one of the parent's validation steps")
    if k >= parent["steps"]:
        raise ValueError(f"{pid}: k={k} is not before the parent's last step {parent['steps']}")
    pcode = stop_after(code, k)
    base = parent["identity"]["argv"][0]
    sha = C.sha256_text(pcode)
    script_rel = f"artifacts/mng3/scripts/{sha[:12]}/{base}"
    r = copy.deepcopy(parent)
    ident = r["identity"]
    ch = config_hash(pcode, base)
    ident["record_id"] = f"{pid}-stop{k}"
    ident["entry_hash"] = sha
    ident["replicate_group"] = "mng3-cfg-" + ch[:12]
    ident["match_policy"] = "exact_only"
    ident["match_policy_how"] = ("a prefix shares its parent's fingerprint and its first k steps of learning rate, so a structural "
                                 "draw could not tell them apart (gpu-mock, 2026-10-20); only the prefix script's exact bytes match it")
    ident["trajectory"]["config_hash"] = ch
    ident["trajectory"]["config_hash_how"] = ("sha256 of the prefix script bytes with any `SEED = N` line blanked, plus the script "
                                              "name: its own replicate group, apart from the parent's")
    batch = ident["trajectory"].get("global_batch_tokens") or (r["launch"]["global_batch_tokens"] or {}).get("value")
    ident["trajectory"]["data"]["train_tokens_consumed"] = k * batch
    n_par = len(vl["steps"])
    ident["derived_from"] = {
        "kind": "prefix", "record": pid, "steps": [1, k], "stop_after_step": k, "parent_steps": parent["steps"],
        "parent_entry_hash": parent["identity"]["entry_hash"], "parent_replicate_group": parent["identity"]["replicate_group"],
        "change": (f"stop_after_step = {k} after the `{BREAK.search(code).group(2)} = ...` line, and the loop's end check also "
                   "breaks at it (build/gen_prefix.py stop_after); nothing else in the script differs"),
        "how": ("the first k optimizer steps of the parent run: the same schedule, data order and seed, so every series value at a "
                "step <= k is the parent's, recorded"),
    }
    r["source"] = dict(r["source"], derived="a prefix of the run this log records (identity.derived_from); its values at steps <= "
                                            f"{k} are this log's")
    r["steps"] = k
    r["planned_steps"] = k
    r["series"] = {name: cut(s, k, f"cut at step {k}, the prefix's stop; the parent's values") for name, s in r["series"].items()}
    kept = len(r["series"]["val_loss"]["steps"])
    t = r["timing"]
    tt = r["series"].get("train_time_s")
    if tt and tt["steps"] and tt["steps"][-1] == k:
        t["train_time_s"] = C.tv(tt["values"][-1], "recorded", f"the parent run's own clock at its validation at step {k}")
    else:
        t["train_time_s"] = C.absent(f"the parent's log has no train_time at step {k}")
    # The parent's wall_total_s is the whole run's (`date` around it). A prefix stops at step k, so it has none of its own;
    # inherited, it disclosed the full run's length to the agent and the packed policy ordered by it (soda-fade's prefixes
    # read 1157 s, the 3040-step run's wall, for a 250-step job; found 2026-10-22 from the runner's packed_cp launch order).
    if (t.get("wall_total_s") or {}).get("value") is not None:
        t["wall_total_s"] = C.absent(f"the parent's wall time is the whole run's; the prefix stops at step {k}")
    ev = (t.get("eval_s") or {}).get("value")
    if ev:
        t["eval_s"] = C.tv(round(ev * kept / n_par, 3), "modelled",
                           f"the parent's modelled eval_s x {kept}/{n_par}, the validations kept: each is paced as in the parent "
                           "(gpu-mock divides eval_s by the length of the val_loss series)")
    for key in ("steady_step_s", "startup_in_train_s", "first_step_s", "setup_s"):
        if isinstance(t.get(key), dict) and t[key].get("value") is not None:
            t[key] = dict(t[key], how=(t[key].get("how") or "") + "; the parent's, carried over")
    r["failures"] = [f for f in parent["failures"]
                     if f.get("step") is not None and f["step"] <= k and f.get("phase") != "after_training"]
    v = r["series"]["val_loss"]["values"][-1]
    res = r["result"]
    res["final_val_loss"] = C.tv(v, "recorded", f"the parent's validation loss at step {k}, the last the prefix script prints")
    if "reached_target" in res:
        res["reached_target"] = C.tv(v < (res.get("target_val_loss") or 3.28), "derived",
                                     f"the loss at step {k} below the track's target; a prefix is not expected to reach it")
    res["stopped_at_step"] = k
    r["outcome"] = {"status": "completed", "provenance": "derived",
                    "how": f"the prefix script stops after its validation at step {k}, as its one change makes it"}
    a = r["artefacts"]
    if isinstance(a.get("log_bytes"), dict) and a["log_bytes"].get("value") is not None:
        a["log_bytes"] = C.tv(a["log_bytes"]["value"], "inferred",
                              "at most the parent's log: the same script text and fewer step lines")
    td = r["host"].get("train_data")
    if td:
        td["shards_read"] = math.ceil(k * batch / 1e8)
        td["how"] = "every rank reads each shard it reaches in full: ceil(k x batch tokens / 100M)"
    r["agent_view"] = dict(r["agent_view"], script=script_rel,
                           launch_command=r["agent_view"]["launch_command"])
    r["readiness"] = C.readiness(r)
    return r, pcode, script_rel


def main():
    reg = sys.argv[1]
    recs = C.load_records(os.path.join(reg, "records"))
    by_group = {}
    for r in recs.values():
        by_group.setdefault(r["identity"]["replicate_group"], []).append(r)
    out, written_scripts = [], {}
    specs = [(name, g, k) for name, (g, k) in PREFIX_SPEC.items()] + [(name, PREFIX_SPEC[name][0], k) for name, k in PREFIX_Q.items()]
    specs += [(name, PREFIX_SPEC[name][0], k) for name, k in PREFIX_EXTRA]
    for name, g, k in specs:
        parents = sorted(by_group.get(g) or [], key=lambda r: r["identity"]["record_id"])
        if not parents:
            raise SystemExit(f"{name}: no records in {g}")
        code = open(os.path.join(reg, parents[0]["agent_view"]["script"])).read()
        reads = env_settings(code)
        if reads:
            raise SystemExit(f"{name}: the script reads {', '.join(reads[:4])} from the environment")
        n = 0
        for p in parents:
            torch = str((p["identity"]["trajectory"].get("environment") or {}).get("torch") or "")
            if not p["readiness"]["replay"] or not torch.startswith(TORCH) or seed_how(p) not in ("argv", "none"):
                continue
            if open(os.path.join(reg, p["agent_view"]["script"])).read() != code:
                raise SystemExit(f"{name}: {p['identity']['record_id']} binds a different script from its group")
            r, pcode, rel = derive(p, code, k, reg)
            written_scripts[rel] = pcode
            out.append(r)
            n += 1
        if not n:
            raise SystemExit(f"{name}: no parent on torch {TORCH} replays as briefed")
        print(f"{name}: {n} prefix records at k={k}, group {out[-1]['identity']['replicate_group']}")
    for rel, pcode in written_scripts.items():
        p = os.path.join(reg, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        C.write_text(p, pcode)
    C.add_disclosure(out)
    for r in out:
        r["readiness"] = C.readiness(r)
        C.write_json(os.path.join(reg, "records", r["identity"]["record_id"] + ".json"), r)
    print(f"prefix records: {len(out)}, scripts: {len(written_scripts)}")


if __name__ == "__main__":
    main()
