#!/usr/bin/env bash
# Build the paper's four MLS-Bench tasks into gpu-tasks/registry/: mls-kv-gate, mls-kv-report, mls-kv-report-h and
# pkg-k2-latepole, with their records and expected scores, so they run and read like the other five.
#
# usage (from the repository root, after the Hugging Face dataset's registry is in place, gpu-tasks/registry/README.md):
#   bash gpu-tasks/build/add_mlsbench_tasks.sh [scratch dir]          default scratch dir: ~/.cache/computewise-mlsbench
#
# Nothing from MLS-Bench ships with this repository. The script clones MLS-Bench (github.com/Imbernoulli/MLS-Bench) at the
# commit the paper used, and the generators read its leaderboards, task configurations and evaluators from that clone. The
# datasets and model files the late-pole task stages are fetched from their own hosts at pinned revisions
# (build/mlsbench-inputs.json; every file is checked by sha256). What ships is ours: the generators in build/, and our own
# measurements under survey/: per-example scores of each KV evaluation's first 50 examples, the example-order tables, prompt
# lengths, the measured host terms, mls-kv-report-h's reference schedule, and the late pole's input layout and reader.
#
# Needs git, Python 3.9 or later, network to GitHub and Hugging Face, and about 2 GB in the scratch dir. Standard library
# only; it starts no container and touches no GPU. To run the tasks, build their box images as the README says
# (isolation/image/mlskv/, isolation/image/latepole/).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
G=$ROOT/gpu-tasks
S=${1:-$HOME/.cache/computewise-mlsbench}
MLS=b873944969c53b01e4ee6e843c77767b8c702485
TASKS="mls-kv-gate mls-kv-report mls-kv-report-h pkg-k2-latepole"
export GPU_TASKS_MACHINE_TYPE=${GPU_TASKS_MACHINE_TYPE:-c3-highmem-44}   # the machine type the paper's runs and host terms used
[ -f "$G/registry/tasks/pkg-g4n-a.json" ] || {
  echo "put the Hugging Face dataset's registry in place first (gpu-tasks/registry/README.md)" >&2; exit 2; }
quiet() { grep -v '^\[gpu_daemon\] task file' || true; }

mkdir -p "$S"
if [ ! -d "$S/mls-bench/.git" ]; then
  echo "== cloning MLS-Bench into $S/mls-bench"
  git clone -q --filter=blob:none --no-checkout https://github.com/Imbernoulli/MLS-Bench "$S/mls-bench"
fi
git -C "$S/mls-bench" checkout -q "$MLS"
echo "== MLS-Bench at $(git -C "$S/mls-bench" rev-parse --short HEAD)"

cd "$G"
echo "== the late pole's inputs: from the clone and their own hosts"
python3 -I build/fetch_mlsbench_inputs.py "$S/mls-bench"

echo "== records: the 40 KV evaluations and the late pole's 20"
python3 -I build/gen_mlsbench.py "$S/mls-bench" registry llm-kv-selection-budgeting llm-kv-adaptive-quantization | tail -1
python3 -I build/gen_mlskv_serving.py registry
cp survey/example-order-2026-10-28/example-order-*.json registry/records/
python3 -I build/gen_mlskv_first50.py registry survey/mlskv-first50-2026-10-29
python3 -I build/gen_k2_latepole.py --records registry | grep '"new_records"'

echo "== task files"
python3 -I build/finish_one_task.py registry gen_mlskv_task --kw '{"variant": "mls-kv-report"}' 2>&1 | quiet
python3 -I build/finish_one_task.py registry gen_mlskv_task --kw '{"variant": "mls-kv-report-h"}' 2>&1 | quiet
python3 -I build/finish_one_task.py registry gen_mlskv_gate 2>&1 | quiet
python3 -I build/finish_one_task.py registry gen_k2_latepole 2>&1 | quiet

echo "== expected scores"
for t in $TASKS; do
  python3 -I build/expected_scores.py registry "$t" all --json > "registry/expected/$t.json"
  H="survey/host-terms/$t.$GPU_TASKS_MACHINE_TYPE.json"
  if [ -f "$H" ]; then
    python3 -I build/expected_scores.py registry "$t" all --host "$H" --json > "registry/expected/$t.$GPU_TASKS_MACHINE_TYPE.host.json"
  fi
done
python3 -I build/add_feasible_best.py registry mls-kv-report-h survey/mls-kv-report-h-reference-2026-10-25/reference.json \
  "registry/expected/mls-kv-report-h.$GPU_TASKS_MACHINE_TYPE.host.json, policies.packed (written by build/expected_scores.py)"
python3 -I build/module_shas_tasks.py registry $TASKS

echo "== check"
python3 -I build/validate.py registry | head -1
echo "done: $TASKS are in gpu-tasks/registry/"
