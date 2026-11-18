"""Derived records for the data-preparation arm: a wave-Q prefix script that checks its data files at start-up (gpu-tasks,
2026-10-21; the study, from cpu-tasks' experiments/cpu-tasks/survey/PREP-STEPS-2026-10-21.md).

usage: python3 -I build/gen_dataprep.py <registry>

In a data-preparation package the tokenised shards are absent from the agent's tree. data/raw/ holds the documents they were
made from, and data/fineweb.py is the script that made them. Tokenising that text with fineweb.py's rule reproduces the
shards byte for byte, so a job that trains on the agent's shards is the recorded run.

Unchanged, a script finds a missing validation shard at start-up and a missing first training shard at its first batch, but
finds a missing or half-written later shard only when its loader reaches it, mid-run. the study asked for every such
failure at start-up. So each script here is its wave-Q prefix script with one change, inserted after the line that reads the
script's own code: a check that every file the run reads (the validation shard and training shards 1..shards_read) exists and
is complete, its header's token count matching its size, as the script's own loader asserts. A run whose data is not ready
stops with the file's name before any training.

Each record is its prefix record's run (identity.derived_from.via), exact_only in its own replicate group and keyed by the
checked script's bytes. derived_from keeps the prefix's record and steps, so common.served_unit traces it to the recorded run.
"""
import copy
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import gen_prefix as GP  # noqa: E402

SUFFIX = "-dcheck"
ANCHOR = re.compile(r"^with open\(sys\.argv\[0\]\) as f:\n[ \t]+code = f\.read\(\)[^\n]*\n", re.M)
CHECK = '''# Data check (gpu-tasks, 2026-10-21): stop at start-up unless every data file this run reads is present and complete.
for _path in {files!r}:
    _ok = os.path.isfile(_path) and os.path.getsize(_path) >= 1024
    if _ok:
        with open(_path, "rb") as _fh:
            _magic, _version, _ntok = (int.from_bytes(_fh.read(4), "little") for _ in range(3))
        _ok = _magic == 20240520 and _version == 1 and os.path.getsize(_path) == 1024 + 2 * _ntok
    if not _ok:
        raise SystemExit(f"data file {{_path}} is missing or incomplete")
'''


def data_files(rec):
    """The files a record's run reads: the validation shard and training shards 1..shards_read."""
    n = int(rec["host"]["train_data"]["shards_read"])
    return ("data/fineweb10B/fineweb_val_000000.bin",) + tuple(f"data/fineweb10B/fineweb_train_{i:06d}.bin" for i in range(1, n + 1))


def data_check(code, files):
    """The script with the documented data check after its `code = f.read()` line. Refuses a script without exactly one."""
    if "Data check (gpu-tasks" in code:
        raise ValueError("the script already has the data check")
    m = ANCHOR.findall(code)
    if len(m) != 1:
        raise ValueError(f"{len(m)} `with open(sys.argv[0]) as f: code = f.read()` blocks")
    if not re.search(r"^import os$", code[:ANCHOR.search(code).start()], re.M):
        raise ValueError("os is not imported before the code-reading block")
    a = ANCHOR.search(code)
    return code[:a.end()] + CHECK.format(files=tuple(files)) + code[a.end():]


def derive(via, code):
    """The data-checked record of prefix record `via`, and its script's text."""
    files = data_files(via)
    dcode = data_check(code, files)
    base = via["identity"]["argv"][0]
    sha = C.sha256_text(dcode)
    r = copy.deepcopy(via)
    ident = r["identity"]
    ch = GP.config_hash(dcode, base)
    ident["record_id"] = via["identity"]["record_id"] + SUFFIX
    ident["entry_hash"] = sha
    ident["replicate_group"] = "mng3-cfg-" + ch[:12]
    ident["match_policy"] = "exact_only"
    ident["match_policy_how"] = ("only the data-checked script's exact bytes match it: it shares its prefix's fingerprint and "
                                 "schedule, as a prefix shares its parent's (gpu-mock, 2026-10-20)")
    ident["trajectory"]["config_hash"] = ch
    ident["trajectory"]["config_hash_how"] = ("sha256 of the data-checked script bytes with any `SEED = N` line blanked, plus the "
                                              "script name: its own replicate group")
    ident["derived_from"] = dict(via["identity"]["derived_from"], kind="data_checked", via=via["identity"]["record_id"],
                                 files=list(files),
                                 change=("a start-up check that each of `files` exists with a complete header and body, inserted "
                                         "after the script's `code = f.read()` line (build/gen_dataprep.py data_check); nothing "
                                         "else differs from the prefix script"),
                                 how=("the prefix record's run: the check runs before any training and reads nothing the run "
                                      "uses, so every series value is the prefix's, recorded"))
    r["agent_view"] = dict(r["agent_view"], script=f"artifacts/mng3/scripts/{sha[:12]}/{base}")
    r["readiness"] = C.readiness(r)
    return r, dcode


def main():
    reg = sys.argv[1]
    recs = C.load_records(os.path.join(reg, "records"))
    n = 0
    for name, k in GP.PREFIX_Q.items():
        parent_group = GP.PREFIX_SPEC[name][0]
        vias = sorted((r for r in recs.values() if (r["identity"].get("derived_from") or {}).get("kind") == "prefix"
                       and r["identity"]["derived_from"]["parent_replicate_group"] == parent_group
                       and r["identity"]["derived_from"]["stop_after_step"] == k), key=lambda r: r["identity"]["record_id"])
        for v in vias:
            code = open(os.path.join(reg, v["agent_view"]["script"])).read()
            r, dcode = derive(v, code)
            out = os.path.join(reg, r["agent_view"]["script"])
            os.makedirs(os.path.dirname(out), exist_ok=True)
            C.write_text(out, dcode)
            C.write_json(os.path.join(reg, "records", r["identity"]["record_id"] + ".json"), r)
            n += 1
        print(f"{name}: {len(vias)} data-checked records at step {k}")
    print(f"data-checked records: {n}")


if __name__ == "__main__":
    main()
