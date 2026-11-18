#!/usr/bin/env python3
"""Compare a freshly filled mlsbench-kv HF cache tree with the verified one:
    python3 -I tests/mls_cache_fill_check.py <filled .../hf> <reference .../hf>
Reads both STAGED.json (per-file sizes and, where present, sha256) and walks
hub/ on disk (relative path + size), ignoring xet/logs, .lock and .complete,
which differ between fills by nature. Exit 1 on any difference."""
import json, os, sys
from pathlib import Path

filled, ref = Path(sys.argv[1]), Path(sys.argv[2])
fails = 0


def chk(name, cond, detail=""):
    global fails
    fails += 0 if cond else 1
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def staged_files(root):
    """{(repo, filename): (size, sha256 or None)} from STAGED.json."""
    out = {}
    try:
        st = json.load(open(root / "STAGED.json"))
    except Exception as e:
        print(f"no STAGED.json under {root}: {e!r}"); return None
    for ent in st.get("hub", []):
        repo = ent.get("repo"); files = ent.get("files") or {}
        shas = ent.get("sha256") or ent.get("hashes") or {}
        for name, v in files.items():
            size = v if isinstance(v, int) else (v.get("bytes") or v.get("size") if isinstance(v, dict) else None)
            sha = (v.get("sha256") if isinstance(v, dict) else None) or shas.get(name)
            out[(repo, name)] = (size, sha)
    return out


def staged_hashes(root):
    """{path: (bytes, sha256)} from STAGED.json's top-level files list, without the per-fill noise."""
    try:
        st = json.load(open(root / "STAGED.json"))
    except Exception:
        return None
    out = {}
    for f in st.get("files", []):
        pth = f.get("path", "")
        if "xet/logs" in pth or pth.endswith(".lock") or pth.endswith(".complete"):
            continue
        out[pth] = (f.get("bytes"), f.get("sha256"))
    return out


ha, hb = staged_hashes(filled), staged_hashes(ref)
if ha is not None and hb is not None:
    chk(f"STAGED.json files: same paths ({len(hb)} in the reference)", set(ha) == set(hb), f"only filled: {sorted(set(ha) - set(hb))[:5]} only ref: {sorted(set(hb) - set(ha))[:5]}")
    diff = [k for k in set(ha) & set(hb) if ha[k] != hb[k]]
    chk("STAGED.json files: same bytes and sha256 for every path", not diff, str([(k, ha[k], hb[k]) for k in diff][:5]))
    blobs_sha = [k for k in hb if "/blobs/" in k]
    chk(f"the {len(blobs_sha)} hub blobs match by sha256", all(ha.get(k) == hb[k] for k in blobs_sha), str([k for k in blobs_sha if ha.get(k) != hb[k]][:5]))
    chk("offline_ok recorded true in the filled tree", bool(json.load(open(filled / "STAGED.json")).get("offline_ok")))
a, b = staged_files(filled), staged_files(ref)
chk("both STAGED.json readable", a is not None and b is not None)
if a is not None and b is not None:
    chk("same set of staged (repo, file) entries", set(a) == set(b), f"only filled: {sorted(set(a) - set(b))[:5]} only ref: {sorted(set(b) - set(a))[:5]}")
    common = sorted(set(a) & set(b))
    chk("every staged file has the same size", all(a[k][0] == b[k][0] for k in common), str([(k, a[k][0], b[k][0]) for k in common if a[k][0] != b[k][0]][:5]))
    with_sha = [k for k in common if a[k][1] and b[k][1]]
    chk(f"every staged file with a sha256 on both sides matches ({len(with_sha)} of {len(common)})", all(a[k][1] == b[k][1] for k in with_sha), str([k for k in with_sha if a[k][1] != b[k][1]][:5]))


def walk(root):
    out = {}
    for p in (root / "hub").rglob("*"):
        rel = p.relative_to(root)
        s = str(rel)
        if "xet/logs" in s or s.endswith(".lock") or s.endswith(".complete") or p.is_dir():
            continue
        if p.is_symlink():
            out[s] = ("link", os.readlink(p))
        else:
            out[s] = ("file", p.stat().st_size)
    return out


fa, fb = walk(filled), walk(ref)
chk("hub/: same file list on disk", set(fa) == set(fb), f"only filled: {sorted(set(fa) - set(fb))[:5]} only ref: {sorted(set(fb) - set(fa))[:5]}")
chk("hub/: same sizes and link targets", all(fa[k] == fb[k] for k in set(fa) & set(fb)), str([(k, fa[k], fb[k]) for k in set(fa) & set(fb) if fa[k] != fb[k]][:5]))
blobs = [k for k in fa if "/blobs/" in k and fa[k][0] == "file"]
total = sum(fa[k][1] for k in blobs)
print(f"filled hub blobs: {len(blobs)}, {total / 1e9:.2f} GB")
print(f"{'PASS' if not fails else 'FAIL'}: {fails} differences")
sys.exit(1 if fails else 0)
