#!/bin/sh
# Installs the scheduler into an image the way a distribution package lays out SLURM: the commands in /usr/bin,
# the controller and step daemon in /usr/sbin, the library in /usr/lib/slurm, a minimal /etc/slurm/slurm.conf.
# usage (in a Dockerfile, as root): sh install.sh <source dir> [<python3 path>]
#   DESTDIR=<dir> sh install.sh ...  stages the same tree under <dir> instead (for a run that mounts it into the box)
#   BOX_GPUS=<n> sh install.sh ...   a box with other than 4 GPUs: slurm.conf's gres says n (unset: 4)
#   BOX_GPU_TYPE=<t> sh install.sh ... a box of another GPU type (a100): slurm.conf's gres type (unset: h100), which the
#                                    controller reads as its node's type
# <python3 path> is the interpreter inside the box, which need not exist where this runs. Commands run it with -IS:
# no user site, no PYTHONPATH, no sitecustomize, so nothing installed for the task's Python reaches the scheduler.
set -eu
SRC=${1:?source dir}
PY=${2:-$(command -v python3)}
D=${DESTDIR:-}
install -d "$D/usr/bin" "$D/usr/sbin" "$D/usr/lib/slurm" "$D/etc/slurm"
install -d -m 0755 "$D/var/spool/slurmd" "$D/run/slurm"
install -m 0644 "$SRC"/slurm/common.py "$SRC"/slurm/ctld.py "$SRC"/slurm/cli.py "$SRC"/slurm/slurmstepd.py "$D/usr/lib/slurm/"
for c in sbatch squeue sacct scancel sinfo srun scontrol; do
  sed "1s|.*|#!$PY -IS|" "$SRC/bin/$c" > "$D/usr/bin/$c"; chmod 0755 "$D/usr/bin/$c"
done
for c in slurmctld slurmstepd; do
  sed "1s|.*|#!$PY -IS|" "$SRC/bin/$c" > "$D/usr/sbin/$c"; chmod 0755 "$D/usr/sbin/$c"
done
# The node's GPU count for gres: BOX_GPUS, the build argument the image's Dockerfile declares for a box with other
# than 4 GPUs (unset or empty: 4, so a 4-GPU image's slurm.conf is unchanged). The controller sizes itself from the
# box's /dev/nvidia<N>, so this keeps the config file saying what sinfo says.
G=${BOX_GPUS:-4}
case $G in ''|*[!0-9]*|0) echo "install.sh: BOX_GPUS must be a positive integer, not '$G'" >&2; exit 1;; esac
# The GPU type for gres, likewise: BOX_GPU_TYPE (unset or empty: h100). Lower-case letters, digits, '_' and '-' only,
# since it sits inside gpu:<type>:<n>
T=${BOX_GPU_TYPE:-h100}
case $T in ''|*[!abcdefghijklmnopqrstuvwxyz0123456789_-]*) echo "install.sh: BOX_GPU_TYPE must be lower-case letters, digits, _ or -, not '$T'" >&2; exit 1;; esac
cat > "$D/etc/slurm/slurm.conf" <<CONF
# slurm.conf for this node
ClusterName=cluster
SlurmctldHost=localhost
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
SchedulerType=sched/backfill
GresTypes=gpu
DependencyParameters=kill_invalid_depend
ProctrackType=proctrack/linuxproc
TaskPlugin=task/none
JobAcctGatherType=jobacct_gather/linux
NodeName=DEFAULT Gres=gpu:$T:$G State=UNKNOWN
PartitionName=gpu Nodes=ALL Default=YES MaxTime=INFINITE State=UP
CONF
echo "slurm $(sed -n 's/^VERSION = "\(.*\)"/\1/p' "$D/usr/lib/slurm/common.py") installed${D:+ under $D}"
