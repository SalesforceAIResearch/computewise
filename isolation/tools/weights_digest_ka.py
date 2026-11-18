"""Known answer for weights_digest.py, in two halves. `make <dir>` runs inside a box image (it has torch) and writes the
test checkpoints under <dir>/home and <dir>/tmp, plus the ensemble index; `check <dir>` runs on the host with no torch
and asserts each case's digest or label. Run make in a capped one-off container, network none:
  docker run --rm --network none --cpus 2 --memory 4g -v <dir>:/ka --entrypoint python3 <box image> -I /ka/ka.py make /ka
  python3 -I weights_digest_ka.py check <dir>
"""
import json
import os
import sys


def make(d):
    import pickle
    import torch
    home, tmp = os.path.join(d, "home"), os.path.join(d, "tmp")
    os.makedirs(os.path.join(home, "project", "out", "a"), exist_ok=True)
    os.makedirs(os.path.join(home, "project", "out", "b"), exist_ok=True)
    os.makedirs(tmp, exist_ok=True)
    g = torch.Generator().manual_seed(1)
    sd = {"w": torch.randn(64, 32, generator=g), "b": torch.zeros(32), "step": 7}
    other = {"w": torch.randn(64, 32, generator=g), "b": torch.zeros(32), "step": 7}
    out = os.path.join(home, "project", "out")
    torch.save(sd, os.path.join(out, "a", "ckpt.pt"))                    # 0: member A
    torch.save(sd, os.path.join(out, "b", "member_renamed.pt"))          # 1: same weights, another name and dir
    torch.save(other, os.path.join(out, "ckpt_other.pt"))                # 2: different weights
    opt = {"model": sd, "optimizer": torch.optim.SGD([torch.nn.Parameter(sd["w"].clone())], lr=0.1).state_dict()}
    torch.save(opt, os.path.join(tmp, "with_opt.pt"))                     # 3: model + optimizer state, in /tmp
    raw = open(os.path.join(out, "a", "ckpt.pt"), "rb").read()
    open(os.path.join(out, "truncated.pt"), "wb").write(raw[: len(raw) // 2])   # 4: cut in half
    open(os.path.join(out, "notweights.pt"), "wb").write(b"hello, not a checkpoint\n")   # 5
    torch.save(sd, os.path.join(out, "legacy.pt"), _use_new_zipfile_serialization=False)  # 6: legacy format

    class Evil:                                                            # 7: a zip whose data.pkl calls os.system
        def __reduce__(self):
            return (os.system, ("touch /ka/PWNED",))
    torch.save(sd, os.path.join(out, "evil.pt"))
    import zipfile
    src = os.path.join(out, "evil.pt"); dst = os.path.join(out, "evil2.pt")
    with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as zo:
        for it in zi.infolist():
            data = zi.read(it.filename)
            if it.filename.endswith("/data.pkl"):
                data = pickle.dumps({"w": Evil()}, protocol=2)
            zo.writestr(it, data)
    os.replace(dst, src)
    os.symlink("/etc/passwd", os.path.join(out, "escape.pt"))              # 8: a symlink out of the workspace
    torch.save({"_orig_mod." + k: v for k, v in sd.items() if k != "step"}, os.path.join(out, "compiled.pt"))  # 14
    torch.save({"w": sd["w"].t().contiguous().t(), "b": sd["b"]}, os.path.join(out, "colmajor.pt"))          # 15
    # 9: duplicate of 0 through a different spelling; 10: missing; 11: absolute in-box path to member A;
    # 12: "../" escape; 13: absolute path outside home and /tmp
    index = {"members": [{"recipe": "r", "seed": i, "checkpoint": c, "validation_loss": 1.0} for i, c in enumerate([
        "out/a/ckpt.pt", "out/b/member_renamed.pt", "out/ckpt_other.pt", "/tmp/with_opt.pt",
        "out/truncated.pt", "out/notweights.pt", "out/legacy.pt", "out/evil.pt",
        "out/escape.pt", "out/./a/ckpt.pt", "out/missing.pt",
        "/home/researcher/project/out/a/ckpt.pt", "../../outside.pt", "/etc/passwd",
        "out/compiled.pt", "out/colmajor.pt", "./out/ckpt_other.pt", "../notes/home_member.pt"])]}
    os.makedirs(os.path.join(home, "notes"), exist_ok=True)
    torch.save(other, os.path.join(home, "notes", "home_member.pt"))      # 17: under ~ but outside ~/project
    json.dump(index, open(os.path.join(home, "project", "ensemble.json"), "w"))
    print("made", len(index["members"]), "members under", d)


def check(d):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import weights_digest as wd
    home, tmp = os.path.join(d, "home"), os.path.join(d, "tmp")
    outp = os.path.join(d, "weights-digests.jsonl")
    s = wd.capture(home, tmp, {"path": "ensemble.json", "max_files": 40, "max_bytes_per_file": 4 << 30, "max_seconds": 300}, outp)
    r = [json.loads(l) for l in open(outp)]
    by = {x["i"]: x for x in r}
    fails = []

    def want(cond, what):
        print(("PASS " if cond else "FAIL ") + what)
        if not cond:
            fails.append(what)
    want(len(r) == 18, f"18 lines written ({len(r)})")
    want("full_digest" in by[0] and by[0].get("full_digest") == by[1].get("full_digest"),
         "same state dict under another file name and directory: same full digest")
    want("model_digest" in by[0] and by[0].get("model_digest") == by[1].get("model_digest"), "... and the same model digest")
    want(by[2].get("full_digest") not in (None, by[0].get("full_digest")) and by[2].get("model_digest") not in (None, by[0].get("model_digest")),
         "different weights: different full and model digests")
    want(by[3].get("model_digest") == by[0].get("model_digest") and by[3].get("full_digest") != by[0].get("full_digest"),
         f"model + optimizer state in /tmp: same model digest as model alone, different full digest (from {by[3].get('model_from')})")
    want(by[14].get("model_digest") == by[0].get("model_digest"), f"compiled names (_orig_mod.): same model digest ({by[14].get('model_error')})")
    want(by[15].get("model_digest") == by[0].get("model_digest") and by[15].get("full_digest") != by[0].get("full_digest"),
         f"same values in column-major layout: same model digest, different full digest ({by[15].get('model_error')})")
    want(by[4].get("error") == "truncated_or_crc", f"truncated file: truncated_or_crc ({by[4].get('error')})")
    want(by[5].get("error") == "not_zip", f"not a checkpoint: not_zip ({by[5].get('error')})")
    want(by[6].get("error") == "legacy_format", f"legacy save: legacy_format ({by[6].get('error')})")
    want(any(g.endswith(".system") for g in by[7].get("foreign_globals", [])) and by[7].get("model_error") == "unsupported_or_repacked",
         f"data.pkl calling os.system: named in foreign_globals, no model digest ({by[7].get('foreign_globals')} {by[7].get('model_error')})")
    want(not os.path.exists(os.path.join(d, "PWNED")), "nothing executed: no PWNED file")
    want(by[8].get("error") == "outside_workspace", f"symlink to /etc/passwd: outside_workspace ({by[8].get('error')})")
    want(by[9].get("error") == "duplicate_of" and by[9].get("of") == 0, f"same file by another spelling: duplicate_of 0 ({by[9]})")
    want(by[10].get("error") == "not_found", f"missing file: not_found ({by[10].get('error')})")
    want(by[11].get("error") == "duplicate_of" and by[11].get("of") == 0, f"absolute in-box path to member A: duplicate_of 0 ({by[11].get('error')})")
    want(by[12].get("error") == "outside_workspace", f"../../ escape: outside_workspace ({by[12].get('error')})")
    want(by[16].get("error") == "duplicate_of" and by[16].get("of") == 2, f"./out/... relative to the project: duplicate_of 2 ({by[16]})")
    want(by[17].get("model_digest") == by[2].get("model_digest") and by[17].get("resolved") == "~/notes/home_member.pt",
         f"../notes/... from the project, still under ~: digested ({by[17].get('resolved')})")
    want(by[13].get("error") == "outside_workspace", f"/etc/passwd: outside_workspace ({by[13].get('error')})")
    want(s["listed"] == 18 and s["model_digested"] == 7, f"summary listed 18, 7 model digests ({s})")
    # bounds
    s2 = wd.capture(home, tmp, {"path": "ensemble.json", "max_files": 2}, os.path.join(d, "wd2.jsonl"))
    want(s2["listed"] == 2 and any("first 2" in n for n in s2["notes"]), f"max_files 2: two read, noted ({s2['notes']})")
    s3 = wd.capture(home, tmp, {"path": "ensemble.json", "max_bytes_per_file": 100}, os.path.join(d, "wd3.jsonl"))
    r3 = [json.loads(l) for l in open(os.path.join(d, "wd3.jsonl"))]
    want(r3[0].get("error") == "too_large", f"max_bytes_per_file 100: too_large ({r3[0].get('error')})")
    s4 = wd.capture(home, tmp, {"path": "../x.json"}, os.path.join(d, "wd4.jsonl"))
    want(s4.get("index_error") == "index_path_invalid", f"index path ../x.json: index_path_invalid ({s4.get('index_error')})")
    s6 = wd.capture(home, tmp, {"path": "ensemble.json"}, os.path.join(d, "wd6.jsonl"), cwd=".")
    want(s6.get("index_error") == "index_not_found", f"index taken from home, not the project: index_not_found ({s6.get('index_error')})")
    s5 = wd.capture(home, tmp, {"path": "nope.json"}, os.path.join(d, "wd5.jsonl"))
    want(s5.get("index_error") == "index_not_found", f"no index: index_not_found ({s5.get('index_error')})")
    print(f"{'ALL PASS' if not fails else str(len(fails)) + ' FAILED'}: {len(fails)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(make(sys.argv[2]) if sys.argv[1] == "make" else check(sys.argv[2]))
