#!/usr/bin/env bash
# Checks the box's out-of-memory protection with the built image, on the Docker host, as root:
#   - the harness keeps oom_score_adj -1000 (docker --oom-score-adj, inherited through box-init);
#   - every bash the harness starts, however it is started, resets itself to 0, and so do its children;
#   - a job that overruns the box's memory cap is the process the kernel kills, and the harness survives;
#   - cgroup_sampler.py records the kill as kind=oom_kill, with the dead process's name and score.
# The container takes run_one.py's flags for the box, with a 512 MB cap and no network.
#
#   sudo bash test_oom_protect.sh [image]        exit 0 when every check passes
set -uo pipefail
IMAGE=${1:-ml-node:py311-torch240}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p /srv/build
W=$(mktemp -d /srv/build/oomtest.XXXXXX)
NAME=p6-oomtest-$$
fail=0
check() { if [ "$2" = "$3" ]; then echo "PASS  $1 ($2)"; else echo "FAIL  $1: got '$2', want '$3'"; fail=1; fi; }

mkdir -p "$W/box" && chmod 0700 "$W/box"
cat > "$W/harness.py" <<'PY'
import os, subprocess, time
def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
out = {"harness": open("/proc/self/oom_score_adj").read().strip(),
       "bash": run(["bash", "-c", "cat /proc/self/oom_score_adj"]),
       "bin_bash": run(["/bin/bash", "-c", "cat /proc/self/oom_score_adj"])}
open("/tmp/s.sh", "w").write("#!/bin/bash\ncat /proc/self/oom_score_adj\n")
os.chmod("/tmp/s.sh", 0o755)
out["shebang"] = run(["/tmp/s.sh"])
out["job_child"] = run(["bash", "-c", "python3 -I -c 'print(open(\"/proc/self/oom_score_adj\").read().strip())'"])
time.sleep(5)  # let the host-side sampler attach before the kill
r = subprocess.run(["bash", "-c", "exec python3 -I -c 'a = b\"x\" * (900 << 20); print(len(a))'"])
out["hog_rc"] = str(r.returncode)
out["harness_alive_after"] = "yes"
print(" ".join(f"{k}={v}" for k, v in out.items()), flush=True)
PY

CID=$(docker run -d --name "$NAME" --network none --init --read-only --tmpfs /tmp:exec \
  --cpus 1 --memory 512m --memory-swap 512m --oom-score-adj -1000 \
  --cap-drop ALL --cap-add MKNOD --cap-add SETUID --cap-add SETGID --security-opt no-new-privileges \
  -v "$W/box:/run/box" -v "$W/harness.py:/opt/test/harness.py:ro" "$IMAGE" python3 -I /opt/test/harness.py)
if [ "$(docker info -f '{{.CgroupDriver}}')" = systemd ]; then CG=/sys/fs/cgroup/system.slice/docker-$CID.scope
else CG=/sys/fs/cgroup/docker/$CID; fi
python3 -I "$HERE/cgroup_sampler.py" --cgroup "$CG" --out "$W/resources.jsonl" --interval 1 --container-id "$CID" &
SP=$!
for _ in $(seq 1 90); do [ -f "$W/box/harness_exit" ] && break; sleep 1; done
docker rm -f "$NAME" >/dev/null
wait "$SP" 2>/dev/null

line=$(cat "$W/box/harness.stdout" 2>/dev/null)
get() { printf '%s\n' "$line" | tr ' ' '\n' | sed -n "s/^$1=//p"; }
check "harness keeps its protection"           "$(get harness)"     "-1000"
check "bash from PATH resets to 0"            "$(get bash)"        "0"
check "/bin/bash resets to 0"                 "$(get bin_bash)"    "0"
check "a #!/bin/bash script resets to 0"      "$(get shebang)"     "0"
check "a job started from bash inherits 0"    "$(get job_child)"   "0"
check "the job over the cap is killed"        "$(get hog_rc)"      "-9"
check "the harness survives the kill"         "$(get harness_alive_after)" "yes"
check "the harness exits normally"            "$(cat "$W/box/harness_exit" 2>/dev/null)" "0"
k=$(python3 -I -c 'import json, sys
ks = [json.loads(l) for l in open(sys.argv[1])]
ks = [k for k in ks if k.get("kind") == "oom_kill"]
print("%s:%s:%s" % (ks[0]["comm"], ks[0]["oom_score_adj"], ks[0].get("scope")) if ks else "none")' "$W/resources.jsonl" 2>/dev/null)
check "the sampler names the killed process and its scope" "$k" "python3:0:box"
ov=$(python3 -I -c 'import json, sys
print([json.loads(l) for l in open(sys.argv[1])][-1].get("kmsg_overruns"))' "$W/resources.jsonl" 2>/dev/null)
check "no kernel-log records were lost"       "$ov" "0"
echo "work dir: $W"
[ "$fail" = 0 ] && rm -rf "$W"
exit "$fail"
