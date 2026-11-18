"""Path-bound full records for the goal-level packages (gpu-tasks, 2026-10-21; GOAL-PACKAGES-2026-10-21.md).

usage: python3 -I build/gen_goal.py <registry>

In a goal-level task the agent may edit a candidate's script: stop it earlier, add logging, change its log directory. An edited
script misses the prefix record's exact key. For each candidate setup this writes one record per recorded run of its script,
bound by path to `candidates/<setup>/<script>`, serving that run's whole recorded schedule. An edited launch is then served a
recorded run of the setup whatever its bytes, and any stop up to the full schedule shows that run's own values (gpu-mock,
2026-10-21: exact match first, so an unedited launch still gets its prefix record).

Each is `derived_from.kind: prepared` with `via` and `record` set to its full parent run, so `common.served_unit` names the
parent for both this record and the run's prefix records. gpu-mock keys a task run's draws by that unit, so one run is never
served twice, as a prefix and as a full record.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import gen_prefix as GP  # noqa: E402

# The candidate setups of the goal-level packages and the folder their scripts sit in.
GOAL_SETUPS = ("contra-soft-muon", "soda-fade", "radial-damp", "muonh-tuned", "ema-nesterov-aurora", "newton-muon")
GOAL_DIR = "candidates"
SUFFIX = "-goalpath"


def goal_record(parent, setup):
    r = copy.deepcopy(parent)
    ident = r["identity"]
    bound = f"{GOAL_DIR}/{setup}/{ident['argv'][0]}"
    key = C.sha256_text(f"goal\0{setup}\0{parent['identity']['replicate_group']}")
    ident.update({
        "record_id": parent["identity"]["record_id"] + SUFFIX,
        "bound_path": bound,
        "match_policy": "path_bound",
        "match_policy_how": ("served to a script whose resolved path ends with /" + bound + ", whatever its bytes: a goal-level "
                             "task's candidate script as the agent may edit it. An exact match (the unedited prefix script) comes "
                             "first"),
        "replicate_group": "mng3-goal-" + key[:12],
        "derived_from": {"kind": "prepared", "record": parent["identity"]["record_id"], "via": parent["identity"]["record_id"],
                         "steps": [1, parent["steps"]], "bound_path": bound, "parent_replicate_group": parent["identity"]["replicate_group"],
                         "change": "the recorded full run, served by path to a candidate script the agent may have edited"},
    })
    ident["trajectory"]["config_hash"] = key
    ident["trajectory"]["config_hash_how"] = "sha256 of 'goal', the setup and the parent group: one group per setup"
    r["readiness"] = C.readiness(r)
    return r


def main():
    reg = sys.argv[1]
    recs = C.load_records(os.path.join(reg, "records"))
    n = 0
    for setup in GOAL_SETUPS:
        g = GP.PREFIX_SPEC[setup][0]
        parents = sorted((r for r in recs.values() if r["identity"]["replicate_group"] == g and not r["identity"].get("derived_from")
                          and r["readiness"]["replay"]
                          and str((r["identity"]["trajectory"].get("environment") or {}).get("torch") or "").startswith("2.11.0")),
                         key=lambda r: r["identity"]["record_id"])
        for p in parents:
            r = goal_record(p, setup)
            C.write_json(os.path.join(reg, "records", r["identity"]["record_id"] + ".json"), r)
            n += 1
        print(f"{setup}: {len(parents)} path-bound full records at {GOAL_DIR}/{setup}/")
    print(f"goal path-bound records: {n}")


if __name__ == "__main__":
    main()
