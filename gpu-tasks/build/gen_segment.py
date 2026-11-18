"""Segment scripts: a recorded modded-nanogpt script that can stop at a step with a checkpoint and resume from one (gpu-tasks,
2026-10-20; PACKAGES-2026-10-20.md §8).

usage: python3 -I build/gen_segment.py <registry dir>

Writes, for each variant in SEGMENT_SPEC, the segment script under <registry>/artifacts/mng3/scripts/<hash12>/ and prints its
path. No record is written: gpu-mock runs its save-then-resume round trip on the first segment script before any segment
record exists. The records come later, from the same parents, re-based to each segment's own steps.

The change, the same in every script, made by segment_script():
- two arguments beside --seed: --stop-at N (validate at step N as the script already does, save a checkpoint, exit) and
  --resume PATH (load a checkpoint, continue from its step);
- before the training loop: on --resume, load the model's and each optimizer's state dicts and each optimizer's step_count
  (the Muon class keeps its own step counter, which its update ramps read), set the start step, and advance the data generator
  by that many batches, so the loader stands where the previous segment left it;
- the loop runs from the start step; at --stop-at, rank 0 saves {model, optimizers, step_counts, step} to
  checkpoints/seed<SEED>-step<N>.pt, all ranks wait, and the loop ends.
The schedule still reads train_steps and FINAL_SCHEDULE_STEPS, so a segment trains exactly as the parent did over its steps. A
resumed segment validates at its first step, as the loop is written (the parent's value at that step, same weights).

Under replay the saved weights are the initial ones and the optimizer state is empty, because no real
step runs; the checkpoint is a real file of the right shape, and the next segment loads it for real. The task notes say so.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

# variant -> parent replicate group. contra-soft-muon first: gpu-mock's round trip runs on its segment script.
SEGMENT_SPEC = {
    "contra-soft-muon": "mng3-cfg-d649f96e97a1",      # 80.8 min at one GPU, 3040 steps, 30 seeds on torch 2.11
    "ema-nesterov-aurora": "mng3-cfg-47e82714e8fe",   # 209 min at one GPU, 2900 steps, seeds 8-15 on torch 2.11
    "soda-fade": "mng3-cfg-aa973e631555",             # 74.7 min, 3040 steps, seeds 0-8 (screen-then-confirm)
    "radial-damp": "mng3-cfg-f04ccdc1544e",           # 81.9 min, 3020 steps, 9 seeds on torch 2.11 (screen-then-confirm)
    "circuit-muon": "mng3-cfg-c736a13dc03f",          # 119.4 min, 2900 steps, one run, seed 6 (screen-then-confirm)
}

ARGS = '''parser.add_argument("--stop-at", type=int, default=None)  # segment end: validate at this step, save a checkpoint, exit
parser.add_argument("--resume", type=str, default=None)  # a checkpoint from the previous segment
'''
RESUME = '''# Segments: resume from the previous segment's checkpoint, and stop at --stop-at with one of our own.
start_step = 0
if args.resume:
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint["model"])
    for opt, opt_state, step_count in zip(optimizers, checkpoint["optimizers"], checkpoint["step_counts"]):
        opt.load_state_dict(opt_state)
        if step_count is not None:
            opt.step_count = step_count
    start_step = checkpoint["step"]
    for _ in range(start_step):  # the batches the previous segments trained on
        next(train_loader)
    print0(f"Resumed from {args.resume} at step {start_step}", console=True)
stop_at = args.stop_at if args.stop_at is not None else train_steps
assert start_step < stop_at <= train_steps

'''
SAVE = '''{i}if step == stop_at and stop_at < train_steps:
{i}    if dist.get_rank() == 0:
{i}        os.makedirs("checkpoints", exist_ok=True)
{i}        torch.save({{"model": model.state_dict(), "optimizers": [opt.state_dict() for opt in optimizers],
{i}                    "step_counts": [getattr(opt, "step_count", None) for opt in optimizers], "step": step}},
{i}                   f"checkpoints/seed{{SEED}}-step{{step}}.pt")
{i}        print0(f"Saved checkpoints/seed{{SEED}}-step{{step}}.pt", console=True)
{i}    dist.barrier()
{i}    break
'''


# Scripts whose optimizers wrap others (ema-nesterov-aurora's EMA_Nesterov holds AdamW and Muon as inner_optimizer, and counts
# its own steps in `it`) save every optimizer, wrapped ones included, and every step counter. Scripts without a wrapper keep the
# simpler text above, which gpu-mock's round trip ran on.
RESUME_WRAPPED = '''# Segments: resume from the previous segment's checkpoint, and stop at --stop-at with one of our own.
def all_optimizers(opts):  # the optimizers, and any optimizer they wrap
    out = []
    for opt in opts:
        out.append(opt)
        inner = getattr(opt, "inner_optimizer", None)
        if inner is not None:
            out += all_optimizers(inner if isinstance(inner, list) else [inner])
    return out
STEP_COUNTERS = ("step_count", "it")
start_step = 0
if args.resume:
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint["model"])
    for opt, opt_state, counters in zip(all_optimizers(optimizers), checkpoint["optimizers"], checkpoint["counters"]):
        opt.load_state_dict(opt_state)
        for name, value in counters.items():
            setattr(opt, name, value)
    start_step = checkpoint["step"]
    for _ in range(start_step):  # the batches the previous segments trained on
        next(train_loader)
    print0(f"Resumed from {args.resume} at step {start_step}", console=True)
stop_at = args.stop_at if args.stop_at is not None else train_steps
assert start_step < stop_at <= train_steps

'''
SAVE_WRAPPED = '''{i}if step == stop_at and stop_at < train_steps:
{i}    if dist.get_rank() == 0:
{i}        os.makedirs("checkpoints", exist_ok=True)
{i}        torch.save({{"model": model.state_dict(), "optimizers": [opt.state_dict() for opt in all_optimizers(optimizers)],
{i}                    "counters": [{{name: getattr(opt, name) for name in STEP_COUNTERS if hasattr(opt, name)}}
{i}                                 for opt in all_optimizers(optimizers)], "step": step}},
{i}                   f"checkpoints/seed{{SEED}}-step{{step}}.pt")
{i}        print0(f"Saved checkpoints/seed{{SEED}}-step{{step}}.pt", console=True)
{i}    dist.barrier()
{i}    break
'''


def segment_script(code):
    """The recorded script with the documented save-and-resume change. Refuses a script where any anchor is not unique."""
    def one(pattern, what, flags=re.M):
        m = list(re.finditer(pattern, code, flags))
        if len(m) != 1:
            raise ValueError(f"{len(m)} matches for {what}")
        return m[0]
    if "stop_at" in code or "--resume" in code:
        raise ValueError("the script already names stop_at or --resume")
    seed = one(r'^parser\.add_argument\("--seed", type=int, default=0\)\n', "the --seed argument")
    loop = one(r"^for step in range\(train_steps \+ 1\):\n", "the module-level training loop")
    bcast = one(r"^for p in model\.parameters\(\):\n    dist\.broadcast\(p\.detach\(\), 0\)\n", "the parameter broadcast")
    end = one(r"^(    )if step == train_steps:\n    \s*break\n", "the loop's end check")
    if not (seed.end() < bcast.start() < loop.start() < end.start()):
        raise ValueError("the anchors are not in the expected order")
    for name in ("optimizers", "train_loader", "model", "device", "print0", "SEED"):
        if not re.search(rf"^{name}\b|^def {name}\b", code, re.M):
            raise ValueError(f"no module-level {name}")
    wrapped = "inner_optimizer" in code
    resume, save = (RESUME_WRAPPED, SAVE_WRAPPED) if wrapped else (RESUME, SAVE)
    out = (code[:seed.end()] + ARGS + code[seed.end():bcast.start()] + resume + code[bcast.start():loop.start()]
           + "for step in range(start_step, train_steps + 1):\n" + code[loop.end():end.start()]
           + save.format(i=end.group(1)) + code[end.start():])
    compile(out, "segment", "exec")
    return out


def config_hash(code, base, argv_rest):
    """The replicate-group key: the script bytes with any `SEED = N` line blanked, the script name, and the segment's arguments
    (seed removed), so each segment of a run is its own group."""
    norm = re.sub(r"^SEED\s*=\s*\d+\s*$", "SEED = <seed>", code, flags=re.M)
    return C.sha256_text(norm + "\0" + base + "\0" + " ".join(argv_rest))


def segment_argv(base, seed, start, stop, steps):
    out = [base] + (["--seed", str(seed)] if seed is not None else [])
    if start:
        out += ["--resume", f"checkpoints/seed{seed}-step{start}.pt"]
    if stop < steps:
        out += ["--stop-at", str(stop)]
    return out


def derive_segment(parent, code, start, stop):
    """The record of the segment (start, stop] of `parent`, run by the segment script `code`. Series are re-based to the
    segment's own steps 1..stop-start, as gpu-mock counts a process's optimizer steps from 0; a resumed
    segment's first validation, at its own step 0, is the parent's value at `start`."""
    import copy
    import math
    pid = parent["identity"]["record_id"]
    vl = parent["series"]["val_loss"]
    for k in (start, stop):
        if k and k not in vl["steps"] and k != parent["steps"]:
            raise ValueError(f"{pid}: segment bound {k} is not a validation step of the parent")
    if not 0 <= start < stop <= parent["steps"]:
        raise ValueError(f"{pid}: bad segment ({start}, {stop}]")
    base = parent["identity"]["argv"][0]
    seed = parent["identity"]["seed"]["value"]
    argv = segment_argv(base, seed, start, stop, parent["steps"])
    r = copy.deepcopy(parent)
    ident = r["identity"]
    ch = config_hash(code, base, [a for i, a in enumerate(argv[1:]) if a != "--seed" and (i == 0 or argv[i] != "--seed")])
    ident.update({
        "record_id": f"{pid}-seg{start}-{stop}", "entry_hash": C.sha256_text(code), "argv": argv,
        "argv_how": "the segment script's launch: --seed as the parent's, --resume the previous segment's checkpoint, --stop-at "
                    "the segment's last step (build/gen_segment.py)",
        "replicate_group": "mng3-cfg-" + ch[:12], "match_policy": "exact_only",
        "match_policy_how": "a segment shares its parent's fingerprint; only the segment script with this segment's arguments "
                            "matches it"})
    ident["trajectory"]["config_hash"] = ch
    ident["trajectory"]["config_hash_how"] = "sha256 of the segment script with `SEED = N` blanked, its name, and its arguments"
    batch = ident["trajectory"]["global_batch_tokens"]
    ident["trajectory"]["data"]["train_tokens_consumed"] = (stop - start) * batch
    if start:
        ident["consumes"] = [{"path": f"checkpoints/seed{seed}-step{start}.pt", "from": "the previous segment's --stop-at save"}]
    ident["derived_from"] = {"kind": "segment", "record": pid, "steps": [start + 1, stop], "parent_steps": parent["steps"],
                             "parent_entry_hash": parent["identity"]["entry_hash"],
                             "parent_replicate_group": parent["identity"]["replicate_group"],
                             "change": "save-and-resume (build/gen_segment.py segment_script); steps re-based to the segment"}
    r["source"] = dict(r["source"], derived=f"steps {start + 1}-{stop} of the run this log records (identity.derived_from)")
    n = stop - start
    r["steps"] = r["planned_steps"] = n
    out = {}
    for name, sr in r["series"].items():
        keep = [(st - start, v) for st, v in zip(sr["steps"], sr["values"]) if (start < st <= stop) or (start and st == start)]
        if name == "train_time_s" and start:
            t0 = dict(zip(sr["steps"], sr["values"])).get(start)
            keep = [(st, round(v - t0, 3)) for st, v in keep] if t0 is not None else []
        if keep:
            out[name] = {**{k: v for k, v in sr.items() if k not in ("steps", "values", "how")},
                         "steps": [x for x, _ in keep], "values": [y for _, y in keep],
                         "how": (sr.get("how") or "") + f"; the parent's steps {start}-{stop}, re-based to this segment's own count"
                                + ("; train_time is the parent's minus its value at the segment's start" if name == "train_time_s" and start else "")}
    r["series"] = out
    kept = len(out["val_loss"]["steps"])
    t = r["timing"]
    ev = (t.get("eval_s") or {}).get("value")
    if ev:
        t["eval_s"] = C.tv(round(ev * kept / len(vl["steps"]), 3), "modelled",
                           f"the parent's modelled eval_s x {kept}/{len(vl['steps'])}, the validations this segment runs")
    tt = out.get("train_time_s")
    t["train_time_s"] = (C.tv(tt["values"][-1], "derived", "the parent's clock over the segment's steps")
                         if tt and tt["steps"][-1] == n else C.absent("no train_time at the segment's last step"))
    # the parent's wall_total_s is the whole run's; a segment covers only its steps (as for a prefix, build/gen_prefix.py)
    if (t.get("wall_total_s") or {}).get("value") is not None:
        t["wall_total_s"] = C.absent(f"the parent's wall time is the whole run's; this segment runs steps {start + 1}-{stop}")
    for key in ("first_step_s", "startup_in_train_s"):
        if isinstance(t.get(key), dict) and t[key].get("value") is not None:
            t[key] = dict(t[key], how=(t[key].get("how") or "") + "; paid again by a resumed process, which compiles again "
                                                                   "(gpu-mock, 2026-10-20)")
    r["failures"] = [f for f in parent["failures"] if f.get("step") is not None and start < f["step"] <= stop]
    for f in r["failures"]:
        f["step"] -= start
    v = out["val_loss"]["values"][-1]
    r["result"]["final_val_loss"] = C.tv(v, "recorded", f"the parent's validation loss at step {stop}")
    if "reached_target" in r["result"]:
        r["result"]["reached_target"] = C.tv(v < (r["result"].get("target_val_loss") or 3.28), "derived", "at the segment's end")
    r["result"]["segment"] = [start, stop]
    r["outcome"] = {"status": "completed", "provenance": "derived",
                    "how": "the segment script stops at --stop-at after saving its checkpoint" if stop < parent["steps"]
                           else "the last segment runs to the parent's end"}
    a = r["artefacts"]
    saves = 1 if stop < parent["steps"] else 0
    a["checkpoint_saves"] = C.tv(saves, "derived", "one save at --stop-at; the last segment saves none")
    if saves:
        a["checkpoint_bytes"] = {"value": None, "provenance": "derived", "how": "build/gen_segment.py",
                                 "null_means": "not known: under replay it holds the model's initial weights and an empty "
                                               "optimizer state"}
    td = r["host"].get("train_data")
    if td:
        td["start_batch"] = start
        td["shards_read"] = math.ceil(stop * batch / 1e8)
        td["how"] = "the loader reads every shard up to the segment's end: a resumed one skips the earlier batches by reading them"
    r["agent_view"] = dict(r["agent_view"], script=f"artifacts/mng3/scripts/{C.sha256_text(code)[:12]}/{base}",
                           launch_command="torchrun --standalone --nproc_per_node=<N> " + " ".join(argv))
    r["readiness"] = C.readiness(r)
    return r


# Candidate segment records for gpu-mock's save-then-resume round trip: written under registry/candidates/<name>/records/, which
# no task reads and the validator does not load, so nothing uses them before the round trip passes. A gpu-mock daemon pointed
# at registry/candidates/<name> (its Registry reads <root>/records/*.json) replays them.
CANDIDATES = {"segment-contra-soft-muon": ("contra-soft-muon", (0, 1), (0, 1000, 2000, 3040))}
# The chains the segment packages run (PACKAGES-2026-10-20.md §6 and §8; the study, 2026-10-20). (variant, seed): the segment
# bounds, each a validation step of the parent, every segment at least 20 minutes on one GPU at speed 1. Each (variant, seed) is
# one recorded run, used by one chain only, so no package serves a record twice.
CHAINS = {
    ("contra-soft-muon", 0): (0, 1000, 2000, 3040),                         # P6-2x3, first chain
    ("contra-soft-muon", 1): (0, 1000, 2000, 3040),                         # P6-2x3, second chain
    ("contra-soft-muon", 2): (0, 1000, 2000, 3040),                         # P3-chain
    ("contra-soft-muon", 3): (0, 1500, 3040),                               # P3-mixed, the chain of two
    ("ema-nesterov-aurora", 8): (0, 500, 1000, 1500, 2000, 2500, 2900),     # P6-chain
    ("ema-nesterov-aurora", 9): (0, 625, 1250, 1875, 2500, 2900),           # P10-5c5p, the chain of five
    ("ema-nesterov-aurora", 10): (0, 625, 1250, 1875, 2500, 2900),          # P20-4x5
    ("ema-nesterov-aurora", 11): (0, 625, 1250, 1875, 2500, 2900),          # P20-4x5
    ("ema-nesterov-aurora", 12): (0, 625, 1250, 1875, 2500, 2900),          # P20-4x5
    ("ema-nesterov-aurora", 13): (0, 625, 1250, 1875, 2500, 2900),          # P20-4x5
    # screen-then-confirm: each setup's first 1000 steps (the screen), and the rest of its run (the confirm a winner gets). The
    # seeds are distinct across setups, because a segment script names its checkpoint by seed and step only.
    ("contra-soft-muon", 4): (0, 1000, 3040),
    ("soda-fade", 1): (0, 1000, 3040),
    ("radial-damp", 2): (0, 1000, 3020),
    ("ema-nesterov-aurora", 14): (0, 1000, 2900),
    ("circuit-muon", 6): (0, 1000, 2900),
}
# Wave Q chains: segments of about 8 minutes,
# floor 5. A run may carry several chains with different bounds; their records differ by bounds (…-seg<a>-<b>).
CHAINS_Q = [
    ("ema-nesterov-aurora", 15, (0, 125, 250, 375, 500, 625)),        # P10-5c5p-q, the chain of five
    ("ema-nesterov-aurora", 15, (0, 125, 250, 375, 500, 625, 750)),   # P6-chain-q
    ("contra-soft-muon", 6, (0, 250, 500)),                           # P3-mixed-q, the chain of two
    # SC-q: screen four setups for 250 steps, continue the best two for 250 more. ema-nesterov-aurora
    # is left out: 250 of its steps take about 18 minutes, against 6-10 for these.
    ("contra-soft-muon", 7, (0, 250, 500)),
    ("soda-fade", 2, (0, 250, 500)),
    ("radial-damp", 3, (0, 250, 500)),
    ("circuit-muon", 6, (0, 250, 500)),
    # Halving-q (gpu-tasks' proposal 2, 2026-10-21): four screens of 250 steps, the best two continue 250 more, the best of
    # those 250 more again.
    ("contra-soft-muon", 8, (0, 250, 500, 750)),
    ("soda-fade", 3, (0, 250, 500, 750)),
    ("radial-damp", 4, (0, 250, 500, 750)),
    ("circuit-muon", 6, (0, 250, 500, 750)),
]
Q_FLOOR_MIN = 5.0
# pkg-halving, halving-q's confirm-phase twin. Explicit bounds, since circuit-muon seed 6 already has
# sc-p5's chain in CHAINS. the study asked for a clean round 1 (second and third at least 0.005 apart) and, if the records
# allow, a leader change in round 2. Over every seed combination of these four setups, no rounds of 750-1000 steps give a
# 0.005 gap: at those steps soda, radial and circuit sit within 0.005 of each other. A first round of 625 steps does, so the
# rounds are 625, 875 and 750 steps (15-28, 23-40 and 20-34 minutes; circuit-muon is the slow one). With these seeds the
# step-625 order is radial 1st, circuit 2nd, then soda 0.0054 behind and contra last; at step 1500 circuit overtakes radial
# by 0.0051.
CHAINS_EXTRA = [
    ("contra-soft-muon", 28, (0, 625, 1500, 2250)),
    ("soda-fade", 1, (0, 625, 1500, 2250)),
    ("radial-damp", 9, (0, 625, 1500, 2250)),
    ("circuit-muon", 6, (0, 625, 1500, 2250)),
    # The first seeding, kept on file as a possible judgement variant: rounds of 875 steps, with a
    # three-way near-tie in round 1 (soda 3.6818, circuit 3.6822, radial 3.6824).
    ("contra-soft-muon", 9, (0, 875, 1750, 2625)),
    ("soda-fade", 4, (0, 875, 1750, 2625)),
    ("radial-damp", 5, (0, 875, 1750, 2625)),
    ("circuit-muon", 6, (0, 875, 1750, 2625)),
]
EXTRA_FLOOR_MIN = 15.0
# Where the chains' records go: records/ since gpu-mock's save-then-resume round trip passed (2026-10-20 ~00:10 UTC, 15 of 15
# checks on the mng3 image's patch baf1ea46d6ea; the study: the chains are built once it passes). Until then they were
# candidates under candidates/segment-chains/records.
CHAIN_RECORDS = "records"


def main():
    reg = sys.argv[1]
    recs = C.load_records(os.path.join(reg, "records"))
    scripts = {}
    for name, g in SEGMENT_SPEC.items():
        rs = sorted((r for r in recs.values() if r["identity"]["replicate_group"] == g), key=lambda r: r["identity"]["record_id"])
        if not rs:
            raise SystemExit(f"{name}: no records in {g}")
        base = rs[0]["identity"]["argv"][0]
        code = open(os.path.join(reg, rs[0]["agent_view"]["script"])).read()
        out = segment_script(code)
        rel = f"artifacts/mng3/scripts/{C.sha256_text(out)[:12]}/{base}"
        path = os.path.join(reg, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        C.write_text(path, out)
        scripts[name] = (out, rs)
        print(f"{name}: segment script {rel} (parent {rs[0]['agent_view']['script']}, group {g})")
    for cand, (name, seeds, bounds) in CANDIDATES.items():
        out, rs = scripts[name]
        d = os.path.join(reg, "candidates", cand, "records")
        stale = set(os.listdir(d)) if os.path.isdir(d) else set()
        n = 0
        for p in rs:
            if p["identity"]["seed"]["value"] not in seeds or not p["readiness"]["replay"]:
                continue
            for a, b in zip(bounds, bounds[1:]):
                r = derive_segment(p, out, a, b)
                C.write_json(os.path.join(d, r["identity"]["record_id"] + ".json"), r)
                stale.discard(r["identity"]["record_id"] + ".json")
                n += 1
        for f in stale:                          # written by an earlier spec, not this one
            os.remove(os.path.join(d, f))
        print(f"{cand}: {n} candidate records in {os.path.relpath(d, reg)}")
    d = os.path.join(reg, CHAIN_RECORDS)
    if CHAIN_RECORDS != "records" and os.path.isdir(d):
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
    n = 0
    chains = [(name, seed, bounds, 20.0) for (name, seed), bounds in CHAINS.items()]
    chains += [(name, seed, bounds, Q_FLOOR_MIN) for name, seed, bounds in CHAINS_Q]
    chains += [(name, seed, bounds, EXTRA_FLOOR_MIN) for name, seed, bounds in CHAINS_EXTRA]
    written = set()
    for name, seed, bounds, floor in chains:
        out, rs = scripts[name]
        p = [r for r in rs if r["identity"]["seed"]["value"] == seed and r["readiness"]["replay"]
             and str((r["identity"]["trajectory"].get("environment") or {}).get("torch") or "").startswith("2.11.0")]
        if len(p) != 1:
            raise SystemExit(f"{name} seed {seed}: {len(p)} replayable torch-2.11 records")
        for a, b in zip(bounds, bounds[1:]):
            r = derive_segment(p[0], out, a, b)
            mins = C.paced_duration(r, 1) / 60
            if mins < floor:
                raise SystemExit(f"{r['identity']['record_id']}: {mins:.1f} min on one GPU, under the {floor:g}-minute floor")
            if r["identity"]["record_id"] in written:
                continue
            written.add(r["identity"]["record_id"])
            C.write_json(os.path.join(d, r["identity"]["record_id"] + ".json"), r)
            n += 1
    print(f"chains: {n} segment records in {CHAIN_RECORDS}")


if __name__ == "__main__":
    main()
