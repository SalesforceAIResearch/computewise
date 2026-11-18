# Dry staging: run run_one.py's own staging functions for the modded-nanogpt package tasks, with no box, on an
# idle machine. It exercises the empty-hub path (stage_hf: no container, an empty cache tree) and the shared shard
# cache (stage_task_tree: GPU_TASKS_HUB_CACHE, shards downloaded once and hard-linked into each task's tree), and
# prints one JSON line prefixed DRY-STAGE-RESULT. First run on the E2 on 2026-10-20 19:47 UTC: 37.5 s for the first
# task (3.2 GB, 17 shards), 0.1 s for the second (all shards shared), no container. Run on the VM, as root, from a
# copy outside the deployed tree, e.g.:
#   sudo systemd-run --unit=p006-iso-drystage --collect bash -c \
#       "python3 -I /srv/work/isolation/dry_stage.py > /srv/work/isolation/dry-stage.log 2>&1"
# It fills /srv/cache/tasks/<task>/<sha> and /srv/cache/hub-files, which the real runs then reuse.
import sys, os, json, time, subprocess
sys.path.insert(0, "/opt/computewise/isolation")
import run_one as r
CACHE = "/srv/cache"
IMAGE = "ml-node:py311-torch2110"
out = {"run_one_sha256": r.sha256_file(r.__file__)[:16], "tasks": {}}
for task_id in ("pkg-p6-chain-long", "pkg-p3-par-long"):
    tpath = os.path.join(r.EXP, "gpu-tasks", "registry", "tasks", task_id + ".json")
    task = json.load(open(tpath))
    rec = {"image_label": r.image_label_of(task, task_id)}
    t0 = time.time(); staged = r.stage_task_tree(task_id, r.sha256_file(tpath), CACHE); rec["stage_task_s"] = round(time.time() - t0, 1)
    t0 = time.time(); staged2 = r.stage_task_tree(task_id, r.sha256_file(tpath), CACHE); rec["stage_task_again_s"] = round(time.time() - t0, 3)
    rec["staged"] = staged; rec["staged_again_same"] = staged == staged2
    t0 = time.time(); hf = r.stage_hf(task, task_id, staged, CACHE, IMAGE); rec["stage_hf_s"] = round(time.time() - t0, 3)
    rec["hf"] = hf; rec["hf_complete"] = os.path.exists(os.path.join(hf, ".complete"))
    rec["hf_tree"] = sorted(os.path.join(dp, f)[len(hf):] for dp, dn, fn in os.walk(hf) for f in fn)
    proj = os.path.join(staged, "project")
    n = 0; size = 0; linked = 0
    for dp, dn, fn in os.walk(proj):
        for f in fn:
            p = os.path.join(dp, f); st = os.lstat(p); n += 1; size += st.st_size
            if st.st_nlink > 1: linked += 1
    rec["project_files"] = n; rec["project_bytes"] = size; rec["hardlinked_files"] = linked
    out["tasks"][task_id] = rec
hub = os.path.join(CACHE, "hub-files")
out["hub_files"] = subprocess.run(["du", "-sh", hub], capture_output=True, text=True).stdout.strip()
out["hub_files_count"] = sum(len(fn) for _, _, fn in os.walk(hub))
out["containers_started_by_stage_hf"] = subprocess.run(["docker", "ps", "-a", "--filter", "name=p6-stage", "-q"], capture_output=True, text=True).stdout.split()
print("DRY-STAGE-RESULT " + json.dumps(out, indent=1))
