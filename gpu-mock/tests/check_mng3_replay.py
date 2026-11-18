#!/usr/bin/env python3
"""Compare a replayed modded-nanogpt job's printed validations with its record.
    python3 -I tests/check_mng3_replay.py <job.log> <record.json>
The script prints `step:N/T val_loss:X train_time:Ys` at each validation. Every
printed val_loss must equal the record's val_loss at that step (within 5e-5),
and the job must print at least as many validations as the record holds. The
final train_time is on the job's own clock and is reported against the
record's train_time_s, not asserted.
"""
import json, re, sys
text = open(sys.argv[1], errors="replace").read()
rec = json.load(open(sys.argv[2])); vl = rec["series"]["val_loss"]
got = [(int(m.group(1)), float(m.group(2)), float(m.group(3)))
       for m in re.finditer(r"^step:(\d+)/\d+ val_loss:([0-9.]+) train_time:([0-9.]+)s", text, re.M)]
want = dict(zip(vl["steps"], vl["values"]))
# A segment record is re-based to its own steps (0..n for a resumed segment,
# gpu-tasks 2026-10-21) while the resumed script keeps counting from where it
# resumed: when the printed steps do not match the record's directly, compare
# at printed step minus the offset between the first printed and the first
# recorded step, and say so.
offset = 0
if got and vl["steps"] and got[0][0] not in want:
    offset = got[0][0] - vl["steps"][0]
bad = [(s, v, want.get(s - offset)) for s, v, _ in got if want.get(s - offset) is None or abs(v - want[s - offset]) > 5e-5]
print(f"validations printed: {len(got)}; record val_loss steps: {len(vl['steps'])}; mismatches: {len(bad)}"
      + (f" (record re-based: printed step - {offset})" if offset else ""))
for b in bad[:5]:
    print("  MISMATCH step", *b)
if got:
    last = got[-1]; tt = rec["timing"].get("train_time_s"); tt = tt.get("value") if isinstance(tt, dict) else tt
    print(f"final: step {last[0]} val_loss {last[1]} (record {want.get(last[0] - offset)}); "
          f"train_time {last[2]} s on the job's clock (record train_time_s {tt})")
ok = bool(got) and len(got) >= len(vl["steps"]) and not bad
print("RECORD CHECK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
