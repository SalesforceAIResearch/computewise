# Known answer for run_one.py's staging, two halves (isolation, 2026-11-03, the study's check for the late pole's
# per-environment staging). Run on the E2, as root, from a copy outside the deployed tree:
#   python3 -I stage_ka.py same <old run_one.py> <new run_one.py> <task>=<image>...
#       for tasks without environment.python_environments: each run_one's stage_task_tree and stage_hf return the same
#       cache directories, and make_disk lays out the same home (file list, sizes, hashes) and the same read-only mounts.
#       No container runs beyond make_disk's own home-skeleton copy; the disks are made under a scratch root and removed.
#   python3 -I stage_ka.py latepole <new run_one.py> <image>
#       pkg-k2-latepole: stage_task_tree, stage_env_hf (fills env b, links nothing), prepare_tree, then make_disk; checks
#       env b's cache under ~/project/data/huggingface_cache (hub/ and modules/ mounted read-only, datasets/ copied) and the
#       records' bound files there, then runs stage_env_cache.py --check as uid 1000 over the mounts, network none.
# Prints one JSON line prefixed STAGE-KA.
import hashlib, importlib.util, json, os, shutil, subprocess, sys, tempfile

CACHE = "/srv/cache"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(m)
    return m


def tree(d):
    out = {}
    for root, dirs, files in os.walk(d):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, d)
            if os.path.islink(p):
                out[rel] = "link:" + os.readlink(p)
            elif os.path.isfile(p):
                h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
                out[rel] = f"{os.path.getsize(p)}:{h}"
    return out


def disk(r, task_id, image, staged, hf, env_hf=None):
    os.makedirs("/srv/boxdisk", exist_ok=True)
    root = tempfile.mkdtemp(prefix="stage-ka-", dir="/srv/boxdisk")
    try:
        img, mnt, ro = r.make_disk("d-ka", "40g", root, image, staged, hf, *([env_hf] if env_hf is not None else []))
        try:
            home = tree(os.path.join(mnt, "home"))
        finally:
            subprocess.run(["umount", mnt], check=False)
        return home, [(s, t) for s, t in ro]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def same(old_p, new_p, tasks):
    old, new = load(old_p, "run_one_old"), load(new_p, "run_one_new")
    res = {"old": old.sha256_file(old_p)[:16], "new": new.sha256_file(new_p)[:16], "tasks": {}}
    ok = True
    for ti in tasks:
        t, image = ti.split("=", 1)
        tpath = os.path.join(new.REG, "tasks", t + ".json")
        task = json.load(open(tpath))
        sha = new.sha256_file(tpath)
        rec = {"image": image}
        so = old.stage_task_tree(t, sha, CACHE, old.staged_inputs(task))
        sn = new.stage_task_tree(t, sha, CACHE, new.staged_inputs(task))
        ho = old.stage_hf(task, t, so, CACHE, image)
        hn = new.stage_hf(task, t, sn, CACHE, image)
        rec.update(staged_same=so == sn, hf_same=ho == hn, staged=sn, hf=hn)
        a, ra = disk(old, t, image, so, ho)
        b, rb = disk(new, t, image, sn, hn)
        rec.update(home_files=len(b), home_same=a == b, ro_same=ra == rb, ro=rb)
        ok &= rec["staged_same"] and rec["hf_same"] and rec["home_same"] and rec["ro_same"]
        res["tasks"][t] = rec
    res["all_same"] = ok
    print("STAGE-KA " + json.dumps(res))
    return 0 if ok else 1


def latepole(new_p, image):
    r = load(new_p, "run_one_new")
    t = "pkg-k2-latepole"
    tpath = os.path.join(r.REG, "tasks", t + ".json")
    task = json.load(open(tpath))
    envs = task["environment"]["python_environments"]
    res = {"run_one": r.sha256_file(new_p)[:16], "image": image}
    staged = r.stage_task_tree(t, r.sha256_file(tpath), CACHE, r.staged_inputs(task))
    # env a's five entries are mls-kv-deploy's hub exactly, under the same package set (env-a.txt is the KV image's
    # freeze), so its key links to the verified KV tree, as the KV keys do (stage-one.sh), instead of a second 6.8 GB fill
    hub_a = [h for h in task["environment"]["hub"] if (h.get("python_environment") or "a") == "a"]
    image_id = r.docker("image", "inspect", "-f", "{{.Id}}", image).stdout.strip()
    key_a = hashlib.sha256(json.dumps({"hub": hub_a, "image_id": image_id, "python": "/opt/envs/a/bin/python"},
                                      sort_keys=True).encode()).hexdigest()[:12]
    la = os.path.join(CACHE, "hf", r.image_label_of(task, t), "a-" + key_a)
    os.makedirs(os.path.dirname(la), exist_ok=True)
    if not os.path.lexists(la):
        os.symlink("/srv/cache/hf/mlsbench-kv/aa8e7253", la)
    res["env_a_link"] = {la: os.readlink(la)}
    caches = r.stage_env_hf(task, t, staged, CACHE, image)
    r.prepare_tree(task, t, staged, image)
    res.update(staged=staged, caches=caches, prepared=json.load(open(os.path.join(staged, "prepared.json"))))
    home_env = [n for n in sorted(caches) if not envs[n].get("hf_home")]
    hf = caches[home_env[0]] if home_env else r.stage_hf({**task, "environment": {**task["environment"], "hub": []}},
                                                          t, staged, CACHE, image)
    env_hf = [(envs[n]["hf_home"], caches[n]) for n in sorted(caches) if envs[n].get("hf_home")]
    os.makedirs("/srv/boxdisk", exist_ok=True)
    root = tempfile.mkdtemp(prefix="stage-ka-", dir="/srv/boxdisk")
    ok = True
    try:
        img, mnt, ro = r.make_disk("d-ka", "40g", root, image, staged, hf, env_hf)
        try:
            res["ro"] = ro
            sub = "data/huggingface_cache"
            want = {f"{r.PROJECT}/{sub}/hub", f"{r.PROJECT}/{sub}/modules"} & {t_ for _, t_ in ro}
            res["env_b_ro_mounts"] = sorted(want)
            ok &= f"{r.PROJECT}/{sub}/hub" in want
            ok &= os.path.isdir(os.path.join(mnt, "home", "project", sub, "datasets"))
            ok &= os.path.isfile(os.path.join(mnt, "home", "project", "data", "niah", "cases.jsonl"))
            ok &= any(f.startswith("wikitext") and f.endswith("_test.txt")
                      for f in os.listdir(os.path.join(mnt, "home", "project", "data", "longctx")))
            mounts = []
            for s, t_ in ro:
                mounts += ["-v", f"{s}:{t_}:ro"]
            b = caches.get("b")
            hub_b = os.path.join(b, "hub.json")
            chk = subprocess.run(["docker", "run", "--rm", "--network", "none", "--cpus", "4", "--memory", "32g",
                                  "--user", "1000:1000", "-v", f"{mnt}/home:{r.HOME}", *mounts,
                                  "-v", f"{hub_b}:/tmp/hub.json:ro", "-v", f"{r.ENV_STAGER[0]}:/b:ro",
                                  "-e", f"HF_HOME={r.PROJECT}/{sub}", "-e", f"HF_DATASETS_CACHE={r.PROJECT}/{sub}/datasets",
                                  "-e", f"HF_MODULES_CACHE={r.PROJECT}/{sub}/modules", "-e", "HF_HUB_OFFLINE=1",
                                  "-e", "HF_DATASETS_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1", "-e", "PYTHONPATH=",
                                  "-w", r.PROJECT, "--entrypoint", "/opt/envs/b/bin/python", image,
                                  "-I", "/b/stage_env_cache.py", "--check", "/tmp/hub.json"],
                                 capture_output=True, text=True)
            res["offline_check_b"] = {"rc": chk.returncode, "out": chk.stdout[-2000:], "err": chk.stderr[-800:]}
            ok &= chk.returncode == 0
        finally:
            subprocess.run(["umount", mnt], check=False)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    res["ok"] = bool(ok)
    print("STAGE-KA " + json.dumps(res))
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1] == "same":
        sys.exit(same(sys.argv[2], sys.argv[3], sys.argv[4:]))
    sys.exit(latepole(sys.argv[2], sys.argv[3]))
