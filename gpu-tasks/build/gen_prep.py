"""The preparation arm's content and records (gpu-tasks, 2026-10-20; the study's top priority of 2026-10-20 ~23:50 UTC, from the study's reading at ~23:40 UTC).

usage: python3 -I build/gen_prep.py <registry dir>

Run after build/gen_prefix.py. Writes:
- <registry>/artifacts/mng3/prep/<setup>/{optimizer.py,README.md,hparams.md}: what defines each setup, as a repo would hold it,
  generated from the setup's reference script (its prefix script) and the track's base script (train_gpt_simple.py at commit
  4ea6b937, registry artifact d39470e5b053). The agent writes runs/<setup>/train_gpt.py from them; the references are never in
  its tree.
- <registry>/records/<prefix record>-prep.json: one record per prefix record of each setup, served by gpu-mock to a script at
  runs/<setup>/train_gpt.py whatever its bytes (match_policy path_bound, identity.bound_path; gpu-mock 2026-10-20: the
  daemon matches when the script's resolved path ends with "/" + bound_path). Its series, timing and result are the prefix
  record's; entry_hash and argv are the reference's at that path, so a scripted preparer that copies the reference into place
  matches exactly.

What is real and what is not under a path match, for the task notes: the agent's script runs for real
until its first optimizer step (imports, arguments, data files, model build), so a bug there crashes for real; from then on the
forward, backward and optimizer arithmetic never run, so a bug inside them is invisible and the job prints the record's losses;
steps are paced at the record's step time however many the script runs, past the record's end holding its last values; a
script that departs from the setup is paced and scored as the recorded setup, and gpu-mock's fingerprint_check event names
what differed.
"""
import ast
import copy
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
BASE_SCRIPT = "artifacts/mng3/scripts/d39470e5b053/train_gpt_simple.py"   # the track's train_gpt_simple.py at 4ea6b937
BASE_HOW = ("records/track_3_optimization/train_gpt_simple.py at modded-nanogpt commit 4ea6b937, the track's baseline that "
            "every submission diffs against")
# setup -> (parent replicate group, the prefix records' stop step k); the six of pkg-p6-par first, the rest for P10 and P20.
PREP_SETUPS = {
    "contra-soft-muon": "mng3-cfg-d649f96e97a1",
    "soda-fade": "mng3-cfg-aa973e631555",
    "radial-damp": "mng3-cfg-f04ccdc1544e",
    "aurora-ema-ref": "mng3-cfg-8461a69fc755",
    "muonh-tuned": "mng3-cfg-d7542e12eaf3",
    "newton-muon": "mng3-cfg-1a80484e7aec",
    "ema-nesterov-aurora": "mng3-cfg-47e82714e8fe",    # P10 and P20 add these three
    "circuit-muon": "mng3-cfg-c736a13dc03f",
    "ortho": "mng3-cfg-0b8aa31c1e43",
}
README_DIR = os.path.join(ROOT, "survey", "prep-readmes")
BOUND = "runs/{setup}/train_gpt.py"
# Packages that run a setup with more than one seed (prep-P10, prep-P20) give each seed its own script and directory, so each
# job's preparation has its own path (the runner's scorer ties tool calls to jobs by path).
BOUND_SEED = "runs/{setup}/seed{seed}/train_gpt.py"
# Wave Q: the same preparation content for the short prefixes (gen_prefix.PREFIX_Q), at a path
# of its own, so a path match can never serve a standard record to a Q script or the reverse.
BOUND_Q = "quick/{setup}/train_gpt.py"
BOUND_Q_SEED = "quick/{setup}/seed{seed}/train_gpt.py"     # prep-P10-Q: a setup with several seeds, as BOUND_SEED
PREP_Q = ("contra-soft-muon", "soda-fade", "radial-damp", "ema-nesterov-aurora", "muonh-tuned", "newton-muon")


def defs(code):
    """Top-level function and class definitions, in order: name -> source."""
    out = {}
    for n in ast.parse(code).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out[n.name] = ast.get_source_segment(code, n)
    return out


def constants(code):
    """Module-level UPPER_CASE assignments, in order: name -> (value source, line)."""
    out = {}
    for n in ast.parse(code).body:
        if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id.isupper()):
            out[n.targets[0].id] = ast.get_source_segment(code, n.value)
    return out


def imports(code):
    return [ast.get_source_segment(code, n) for n in ast.parse(code).body if isinstance(n, (ast.Import, ast.ImportFrom))]


def procedure(code):
    """The reference's setup and loop: from the model's construction to the end of the file."""
    m = re.search(r"^[ \t]*model = GPT\(", code, re.M)
    return code[m.start():] if m else ""


def intro(setup, submission, code):
    """The setup's description: its submission README up to the first second-level heading, else the script's docstring."""
    p = os.path.join(README_DIR, submission + ".md")
    if os.path.exists(p):
        text = open(p).read()
        body = text.split("\n## ", 1)[0].strip()
        lines = body.splitlines()
        if lines and lines[0].startswith("# "):
            lines = lines[1:]
        # the description only: up to the first table, without images (the README's results table lists every recorded
        # run's final losses, which the setup's description does not need)
        keep = []
        for l in lines:
            if l.lstrip().startswith("|"):
                break
            if not l.lstrip().startswith("!["):
                keep.append(l)
        return re.sub(r"\n{3,}", "\n\n", "\n".join(keep)).strip(), f"the submission's README ({submission}/README.md)"
    doc = ast.get_docstring(ast.parse(code)) or ""
    return doc.strip(), "the reference script's docstring (the submission has no README)"


def setup_files(setup, rec, base, code):
    """optimizer.py, README.md and hparams.md for one setup."""
    bd, vd = defs(base), defs(code)
    added = [k for k in vd if k not in bd]
    changed = [k for k in vd if k in bd and vd[k] != bd[k]]
    vc = constants(code)
    bi = set(imports(base))
    new_imports = [i for i in imports(code) if i not in bi]
    sub = rec["source"]["submission"]
    k = rec["identity"]["derived_from"]["stop_after_step"]
    parent_steps = rec["identity"]["derived_from"]["parent_steps"]
    head = (f'"""Code for the {setup} setup that train_gpt.py does not have: the definitions it adds, and those that replace\n'
            f"train_gpt.py's definitions of the same name (each marked). From modded-nanogpt track 3, submission {sub},\n"
            f'at commit 4ea6b937. The constants are the ones its training run used."""\n\n')
    parts = [head, "\n".join(["import torch", "from torch import Tensor, nn", "import torch.nn.functional as F",
                              "import torch.distributed as dist"] + new_imports) + "\n\n"]
    if vc:
        parts.append("".join(f"{n} = {v}\n" for n, v in vc.items()) + "\n\n")
    for name in vd:
        if name in added or name in changed:
            mark = f"# replaces train_gpt.py's {name}\n" if name in changed else ""
            parts.append(mark + vd[name] + "\n\n\n")
    optimizer_py = "".join(parts).rstrip() + "\n"
    text, src = intro(setup, sub, code)
    args = [l for l in code.splitlines() if "add_argument(" in l]
    readme = (f"# {setup}\n\n{text}\n\n_(From {src}.)_\n\n"
              "## How it differs from train_gpt.py\n\n"
              + (f"- Replaces: {', '.join(f'`{n}`' for n in changed)}.\n" if changed else "")
              + (f"- Adds: {', '.join(f'`{n}`' for n in added)}.\n" if added else "")
              + "- All of them are in `optimizer.py` in this folder, with the constants its run used (`hparams.md` lists them).\n"
              + (f"- Its reference run takes these arguments: {'; '.join(f'`{a.strip()}`' for a in args)}.\n" if args else
                 "- Its reference run takes no command-line arguments.\n")
              + "\nThe reference run's training setup and loop, from the model's construction to the end:\n\n"
              "```python\n" + procedure(code).rstrip() + "\n```\n\n"
              "## Our run of it\n\n"
              f"A short screening run: the first {k} optimizer steps of this schedule, stopping after the validation at step "
              f"{k}. Keep the schedule's own length ({parent_steps} steps and its decay constants) so the first {k} steps are "
              "the same as a full run's; only the stop is earlier. One GPU.\n")
    rows = "".join(f"| `{n}` | `{v}` |\n" for n, v in vc.items())
    m = re.search(r"^\s*train_steps\s*=\s*(.+)$", code, re.M)
    hparams = (f"# {setup}: hyperparameters\n\nFrom the reference run's script ({sub}).\n\n"
               "| setting | value |\n|---|---|\n"
               + (f"| training steps (`train_steps`) | `{m.group(1).strip()}` |\n" if m else "")
               + f"| our screening run stops after step | `{k}` |\n" + rows)
    return {"optimizer.py": optimizer_py, "README.md": readme, "hparams.md": hparams}


def prepared_record(prefix, setup, per_seed=False, q=False):
    """The record a script at runs/<setup>/train_gpt.py (or, per_seed, runs/<setup>/seed<N>/train_gpt.py) is served: the prefix
    record's run, bound to that path."""
    r = copy.deepcopy(prefix)
    ident = r["identity"]
    seed = ident["seed"]["value"]
    bound = ((BOUND_Q_SEED if per_seed else BOUND_Q) if q else BOUND_SEED if per_seed else BOUND).format(setup=setup, seed=seed)
    key = C.sha256_text(f"prepared\0{setup}\0{prefix['identity']['replicate_group']}" + (f"\0seed{seed}" if per_seed else "")
                        + ("\0q" if q else ""))
    d = prefix["identity"]["derived_from"]
    ident.update({
        "record_id": prefix["identity"]["record_id"] + ("-prepq" if q else "-prep") + ("seed" if per_seed else ""),
        "argv": ["train_gpt.py"] + (["--seed", str(seed)] if "--seed" in prefix["identity"]["argv"] else []),
        "argv_how": "the reference script's launch at the bound path: what a scripted preparer runs after copying it there",
        "bound_path": bound,
        "match_policy": "path_bound",
        "match_policy_how": ("served to a script whose resolved path ends with /" + bound + ", whatever its bytes (gpu-mock, "
                             "2026-10-20): the agent writes that script itself. An exact match (the reference's bytes and argv) "
                             "comes first, then the path"),
        "replicate_group": "mng3-prep-" + key[:12],
        "derived_from": {**d, "kind": "prepared", "via": prefix["identity"]["record_id"], "bound_path": bound,
                         "change": "the prefix record's run, served by path to the script the agent writes there"},
    })
    ident["trajectory"]["config_hash"] = key
    ident["trajectory"]["config_hash_how"] = "sha256 of 'prepared', the setup and the prefix group: one group per setup"
    r["agent_view"] = dict(r["agent_view"], launch_command="torchrun --standalone --nproc_per_node=<N> " + bound
                           + (f" --seed {seed}" if "--seed" in prefix["identity"]["argv"] else ""),
                           reference_note="agent_view.script is the reference a scripted preparer copies into place; a model "
                                          "agent never sees it")
    r["readiness"] = C.readiness(r)
    return r


def main():
    reg = sys.argv[1]
    recs = C.load_records(os.path.join(reg, "records"))
    base = open(os.path.join(reg, BASE_SCRIPT)).read()
    import gen_prefix as GP
    n = 0
    plans = [(setup, g, GP.PREFIX_SPEC[setup][1], False) for setup, g in PREP_SETUPS.items()]
    plans += [(setup, PREP_SETUPS[setup], GP.PREFIX_Q[setup], True) for setup in PREP_Q]
    for setup, parent_group, k, q in plans:
        prefixes = sorted((r for r in recs.values() if (r["identity"].get("derived_from") or {}).get("kind") == "prefix"
                           and r["identity"]["derived_from"]["parent_replicate_group"] == parent_group
                           and r["identity"]["derived_from"]["stop_after_step"] == k),
                          key=lambda r: r["identity"]["record_id"])
        if not prefixes:
            raise SystemExit(f"{setup}: no prefix records at step {k} (run build/gen_prefix.py first)")
        code = open(os.path.join(reg, prefixes[0]["agent_view"]["script"])).read()
        out = os.path.join(reg, "artifacts", "mng3", "prep-q" if q else "prep", setup)
        os.makedirs(out, exist_ok=True)
        for name, text in setup_files(setup, prefixes[0], base, code).items():
            C.write_text(os.path.join(out, name), text)
        seeded = "--seed" in prefixes[0]["identity"]["argv"]
        for p in prefixes:
            for per_seed in ((False, True) if seeded else (False,)):
                r = prepared_record(p, setup, per_seed, q)
                C.write_json(os.path.join(reg, "records", r["identity"]["record_id"] + ".json"), r)
                n += 1
        print(f"{setup}{' (Q)' if q else ''}: {len(prefixes)} prefix runs at step {k}, files in "
              f"artifacts/mng3/{'prep-q' if q else 'prep'}/{setup}/")
    print(f"prepared records: {n}")


if __name__ == "__main__":
    main()
