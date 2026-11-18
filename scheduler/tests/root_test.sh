#!/bin/sh
# The scheduler under the box's privileges, on Linux, as root: the controller runs as root with only the capabilities
# the box keeps (MKNOD, SETUID, SETGID; no CAP_KILL, CAP_CHOWN or CAP_DAC_OVERRIDE) and no-new-privileges, and jobs are
# submitted by an ordinary user. Checks what the controller's selftest.py cannot: jobs run as their submitter, their output
# files are the submitter's, a root controller without CAP_KILL can still cancel, time out and clean up a user's job,
# the caller's uid comes from the socket rather than the request, and an srun that dies frees its allocation.
# No container and no image: it uses setpriv from util-linux, and leaves nothing behind but its temp directory.
#   usage: sudo sh tests/root_test.sh [python3]     (JOB_UID, default 65534, is the submitting user)
set -u
[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 2; }
SRC=$(cd "$(dirname "$0")/.." && pwd)
PY=${1:-python3}
JU=${JOB_UID:-65534}
T=$(mktemp -d /tmp/sched-root.XXXXXX)
chmod 755 "$T"
mkdir -p "$T/run" "$T/spool" "$T/work" "$T/bin"
chown "$JU:$JU" "$T/work"
# the commands, with this python, readable by the job user wherever SRC is
cp -r "$SRC/slurm" "$T/lib"
for f in "$SRC"/bin/*; do sed "1s|.*|#!$(command -v "$PY") -IS|" "$f" > "$T/bin/$(basename "$f")"; done
chmod -R a+rX "$T/lib" "$T/bin"; chmod 755 "$T"/bin/*
export SLURM_LIBDIR="$T/lib" SLURMSTEPD="$T/bin/slurmstepd" SLURMCTLD_SOCKET="$T/run/ctld.sock" PATH="$T/bin:$PATH"
FAILS=0
ok() { if [ "$2" = 0 ]; then echo "PASS $1"; else echo "FAIL $1 :: $3"; FAILS=$((FAILS + 1)); fi; }
as_user() { (cd "$T/work" && setpriv --reuid="$JU" --regid="$JU" --clear-groups -- env HOME="$T/work" sh -c "$1"); }
st() { as_user "sacct -j $1 -X -n -P --format=State" 2>/dev/null; }
wait_st() { i=0; while [ $i -lt 60 ]; do s=$(st "$1"); case " $2 " in *" ${s%% *} "*) echo "$s"; return 0;; esac; sleep 0.25; i=$((i + 1)); done; st "$1"; }

setpriv --bounding-set=-all,+mknod,+setuid,+setgid --inh-caps=-all --no-new-privs -- \
  "$T/bin/slurmctld" --gpus 4 --cpus 4 --mem-mb 8000 --spool "$T/spool" --socket "$SLURMCTLD_SOCKET" \
  --active-file "$T/active" --event-log "$T/events.jsonl" --first-job-id 500
ok "daemonized controller returned ready" $? "slurmctld exit $?"
CPID=$("$PY" -c "import json,sys; print(json.loads(open(sys.argv[1]).readline())['pid'])" "$T/events.jsonl")
CAPS=$(awk '/^CapEff/ {print $2}' "/proc/$CPID/status")
ok "controller runs with the box's capabilities only (CapEff 00000000080000c0)" "$([ "$CAPS" = 00000000080000c0 ]; echo $?)" "$CAPS"
ok "controller's process name is slurmctld" "$([ "$(cat /proc/$CPID/comm)" = slurmctld ]; echo $?)" "$(cat /proc/$CPID/comm)"

# 1. a job runs as its submitter, and its output file is the submitter's
j=$(as_user "sbatch --parsable --gres=gpu:1 --wrap 'id -u; echo CVD=\$CUDA_VISIBLE_DEVICES'")
s=$(wait_st "$j" "COMPLETED FAILED")
out=$(cat "$T/work/slurm-$j.out" 2>/dev/null)
owner=$(stat -c %u "$T/work/slurm-$j.out" 2>/dev/null)
ok "job COMPLETED as uid $JU with a GPU" "$([ "$s" = COMPLETED ] && [ "$(echo "$out" | head -1)" = "$JU" ] && echo "$out" | grep -q '^CVD=[0-3]$'; echo $?)" "$s / $out"
ok "output file owned by the submitter" "$([ "$owner" = "$JU" ]; echo $?)" "owner $owner"

# 2. a job's leftover background process goes with it when the batch script ends
j=$(as_user "sbatch --parsable --wrap 'sleep 300 & echo \$! > bg.pid; sleep 1'")
wait_st "$j" "COMPLETED" >/dev/null; sleep 1
bg=$(cat "$T/work/bg.pid" 2>/dev/null)
ok "leftover process killed at the job's end" "$([ -n "$bg" ] && ! kill -0 "$bg" 2>/dev/null; echo $?)" "pid $bg still alive"

# 3. the user cancels a running job; a root controller without CAP_KILL still stops it
j=$(as_user "sbatch --parsable --wrap 'echo \$\$ > cx.pid; sleep 300'")
wait_st "$j" "RUNNING" >/dev/null; sleep 0.5
as_user "scancel $j"
s=$(wait_st "$j" "CANCELLED")
p=$(cat "$T/work/cx.pid" 2>/dev/null)
ok "scancel stops a running job (CANCELLED, process gone)" "$(case "$s" in CANCELLED*) ! kill -0 "$p" 2>/dev/null;; *) false;; esac; echo $?)" "$s pid $p"

# 4. the time limit: SIGTERM at the limit, TIMEOUT
j=$(as_user "sbatch --parsable -t 0:02 --wrap 'sleep 300'")
s=$(wait_st "$j" "TIMEOUT")
ok "--time 2 s gives TIMEOUT" "$([ "$s" = TIMEOUT ]; echo $?)" "$s"

# 5. the caller's uid comes from the socket: a request claiming uid 0 cannot shut the controller down
as_user "$PY -I -c \"import sys; sys.path.insert(0, '$T/lib'); import common; print(common.request('shutdown', uid=0))\"" >"$T/spoof.txt" 2>&1
sleep 0.5
ok "a spoofed uid 0 shutdown is denied" "$(grep -q 'permission denied' "$T/spoof.txt" && kill -0 "$CPID" 2>/dev/null; echo $?)" "$(cat "$T/spoof.txt")"
as_user "scontrol shutdown" >/dev/null 2>&1
ok "scontrol shutdown by the user is refused" "$(kill -0 "$CPID" 2>/dev/null; echo $?)" "controller stopped"

# 6. srun as the user; an srun killed outright frees its allocation
out=$(as_user "srun --gres=gpu:2 sh -c 'echo \$(id -u) \$CUDA_VISIBLE_DEVICES'" 2>&1)
ok "srun runs as the user with 2 GPUs" "$(echo "$out" | grep -Eq "^$JU [0-3],[0-3]$"; echo $?)" "$out"
(cd "$T/work" && exec setpriv --reuid="$JU" --regid="$JU" --clear-groups -- srun -J held sleep 300) >/dev/null 2>&1 &
SP=$!
i=0; while [ $i -lt 40 ]; do hj=$(as_user "squeue -h -n held -t R -o %i"); [ -n "$hj" ] && break; sleep 0.25; i=$((i + 1)); done
kill -9 "$SP"; wait "$SP" 2>/dev/null
s=$(wait_st "$hj" "CANCELLED FAILED COMPLETED")
ok "a killed srun's job ends (job $hj)" "$(case "$s" in CANCELLED*) true;; *) false;; esac; echo $?)" "$s"

# 7. the event log is root's, and the active file reads 0 with the queue empty
ok "event log owned by root" "$([ "$(stat -c %u "$T/events.jsonl")" = 0 ]; echo $?)" "$(stat -c %u "$T/events.jsonl")"
ok "active file reads 0" "$([ "$(cat "$T/active")" = 0 ]; echo $?)" "$(cat "$T/active")"
pkill -f "sleep 300" -u "$JU" 2>/dev/null
"$T/bin/scontrol" shutdown >/dev/null 2>&1; sleep 0.5
ok "root's scontrol shutdown stops the controller" "$(! kill -0 "$CPID" 2>/dev/null; echo $?)" "still running"
kill "$CPID" 2>/dev/null
rm -rf "$T"
echo; [ "$FAILS" = 0 ] && echo "all passed" || echo "$FAILS failed"
exit $([ "$FAILS" = 0 ]; echo $?)
