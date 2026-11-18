#!/bin/bash
# Rebuild the registry from its sources: the chain in AGENTS.md, "Rebuilding the registry", as one script.
#
# usage: bash build/rebuild.sh <gpu-tasks root> <scratch dir> [gpu-mock src dir]
#
# Downloads land under <scratch dir> (untrusted: nothing there is executed). The registry is rebuilt in place under
# <gpu-tasks root>/registry, then validated; the expected scores follow. The optional third argument is a snapshot of gpu-mock's
# src/ for the fingerprint, convert() and Registry imports (default: the sibling experiments/gpu-mock/src). Standard library,
# git and network to GitHub, Hugging Face and W&B. Heavy enough for the VM, not the controller.
set -euo pipefail
D=${1:?gpu-tasks root}; S=${2:?scratch dir}; export GPU_TASKS_GPU_MOCK_SRC=${3:-}
[ -z "$GPU_TASKS_GPU_MOCK_SRC" ] && unset GPU_TASKS_GPU_MOCK_SRC
B=$D/build; R=$D/registry
MNG3=4ea6b937337a4889b8cfe3f38a93d120048d8f71; MLS=b873944969c53b01e4ee6e843c77767b8c702485
log(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }
mkdir -p "$S"
if [ ! -d "$S/modded-nanogpt/.git" ]; then git clone -q https://github.com/KellerJordan/modded-nanogpt "$S/modded-nanogpt"; fi
git -C "$S/modded-nanogpt" checkout -q $MNG3; log "modded-nanogpt at $(git -C "$S/modded-nanogpt" rev-parse --short HEAD)"
if [ ! -d "$S/mls-bench/.git" ]; then git clone -q https://github.com/Imbernoulli/MLS-Bench "$S/mls-bench"; fi
git -C "$S/mls-bench" checkout -q $MLS; log "MLS-Bench at $(git -C "$S/mls-bench" rev-parse --short HEAD)"
if [ ! -f "$S/trl-wandb/.complete" ]; then
  rm -rf "$S/trl-wandb" "$S/trl-scripts"; python3 -I "$B/fetch_trl.py" "$S/trl-wandb" "$S/trl-scripts"; touch "$S/trl-wandb/.complete"
fi
log "W&B snapshot: $(ls "$S/trl-wandb"/*.json | wc -l) run files, $(ls "$S/trl-scripts" | wc -l) scripts"
rm -rf "$R/records" "$R/tasks" "$R/artifacts" "$R/expected"
python3 -I "$B/gen_track3.py" "$S/modded-nanogpt" "$R" | tail -3
python3 -I "$B/gen_trl.py" "$S/trl-wandb" "$S/trl-scripts" "$R" | tail -3
python3 -I "$B/gen_mlsbench.py" "$S/mls-bench" "$R" | tail -3
python3 -I "$B/gen_mlskv_serving.py" "$R"           # gpu-mock's per_example serving block on the 40 KV records
# gpu-mock r4's KV example-order tables: built by gen_example_order.py in the box image on a staged mls-kv tree, so a rebuild
# carries the built copies rather than rebuilding them (their README says how and from what)
cp "$D/survey/example-order-2026-10-28/"example-order-*.json "$R/records/"
# gpu-mock r6's measured first-50 scores on the KV records, from gpu-mock's decode harness files (after the tables: it checks each
# file's keys against them). Records without a file keep their synthesised vectors
python3 -I "$B/gen_mlskv_first50.py" "$R" "$D/survey/mlskv-first50-2026-10-29" | tail -1
python3 -I "$B/gen_prefix.py" "$R" | tail -3                                                  # derived prefix records, after track 3
python3 -I "$B/gen_segment.py" "$R" | tail -3                                                 # segment scripts; candidate records
python3 -I "$B/gen_h3_chains.py" "$R" | tail -1                                              # H3': single-process resume chains
python3 -I "$B/gen_rounds.py" "$R" | tail -1                                                  # angle A: rounds (after gen_h3_chains)
python3 -I "$B/gen_chain_side.py" "$R" | tail -1                                              # angle D: chain + side work (after gen_h3_chains)
python3 -I "$B/gen_chain_save.py" "$R" | tail -1                                              # D8: chain-side at 8 with a real save failure (after gen_chain_side)
python3 -I "$B/gen_g4n.py" "$R" | tail -1                                                     # G4n: adoption-rule twins, SOAP probe
python3 -I "$B/gen_unneeded_k.py" "$R" | tail -1                                              # angle K: adoption/ full runs, reference probes (reads gen_g4n's twin draws)
python3 -I "$B/gen_prep.py" "$R" | tail -3                                                    # preparation arm: setup folders, records
python3 -I "$B/gen_dataprep.py" "$R" | tail -1                                                # data-preparation arm: data-checked records
python3 -I "$B/gen_goal.py" "$R" | tail -1                                                    # goal-level arm: path-bound full runs
python3 -I "$B/gen_h1.py" "$R" | tail -1                                                      # H1: single-process screens, continuations, prep
python3 -I "$B/gen_h4.py" "$R" | tail -1                                                      # H4: filter-recipe records, scripts, prepare tool
python3 -I "$B/gen_failures.py" "$R" | tail -1                                                # angle C: OOM-relaunch records (needs gen_prefix's)
python3 -I "$B/gen_filter_yield.py" "$R" | tail -1                                            # independent Sol: yield-gated filter records (needs gen_h4's)
python3 -I "$B/gen_sf_zero_lr.py" "$R" | tail -1                                              # SF lane: zero-LR and clean records (needs gen_prefix's)
python3 -I "$B/gen_k2_precision.py" "$R" | tail -1                                           # K2: 1500-step prefixes and their single-process copies (needs gen_prefix's)
python3 -I "$B/gen_obs_latepole.py" "$R" | tail -1                                           # lane OBS: the late-pole twin's prefixes and single-process scripts (needs gen_prefix's)
python3 -I "$B/gen_sf_checkpoint_collision.py" "$R" | tail -1                                # SF lane: checkpoint-output records (needs gen_prefix's)
python3 -I "$B/gen_sol_tabular.py" "$R" | tail -1                                              # independent Sol: TabArena cell records from survey/sol-tabular-ttc/
python3 -I "$B/gen_k2_latepole.py" --records "$R" | tail -3                                     # K2: the late pole's 20 DLM and sparse-attention records (its KV jobs reuse the 40 frozen)
python3 -I "$B/gen_tasks.py" "$R" | tail -9
python3 -I "$B/validate.py" "$R" | head -1
mkdir -p "$R/expected"
HT="${GPU_TASKS_HOST_TERMS_DIR:-$D/survey/host-terms}"
for t in trl-dpo-loss-variants-4gpu trl-regression-battery trl-regression-battery-quick pkg-p3-par-long pkg-p6-par-long pkg-p6-chain-long \
         pkg-p3-par pkg-p6-par pkg-p10-par pkg-p20-par pkg-p20-tail pilot-sft-p3-par pilot-sft-p6-par pkg-prep-p6 \
         pkg-prep-p10 pkg-prep-p20 pkg-p3-chain pkg-p3-mixed pkg-p6-chain pkg-p6-2x3 pkg-p10-5c5p pkg-p20-4x5 pkg-sc-p5 \
         pkg-p6-par-q pkg-prep-p6-q pkg-p10-5c5p-q pkg-p20-tail-q pkg-p3-mixed-q pkg-prep-p10-q pkg-p6-chain-q pkg-p10-par-q pkg-dataprep-p6-q pkg-sc-q pkg-stagger-q pkg-halving-q pkg-stagger pkg-halving goal-choose-optimizer-q \
         goal-pass-bar mls-kv-deploy mls-kv-report mls-kv-report-h mls-kv-gate mls-kv-gate-h pkg-h1-study pkg-h3-chains pkg-h4-filter-ablation pkg-g4n-a pkg-g4n-b pkg-g4n-a-h pkg-g4n-a-means pkg-g4n-a-8gpu pkg-g4n-b-8gpu mls-kv-report-8gpu pkg-oom-relaunch pkg-oom-relaunch-seedmajor pkg-astra-widths pkg-filter-yield pkg-sf-zero-lr pkg-k2-precision pkg-sf-checkpoint-collision pkg-unneeded-k pkg-rounds pkg-rounds-8gpu pkg-chain-side pkg-chain-save-8gpu pkg-sol-tabular-science-adoption pkg-sol-tabular-operations-adoption pkg-sol-tabular-science pkg-sol-tabular-operations pkg-k2-latepole pkg-obs-latepole; do
  if [ "$t" = pkg-astra-widths ]; then
    # mixed widths (1, 2, 4): packed_overlap refuses them, so `all` stops; the other policies run one by one and merge (gpu-tasks,
    # 2026-11-02; the merged file names packed_overlap under policies_not_run)
    python3 -I "$B/merge_expected.py" "$R" "$t" > "$R/expected/$t.json"; continue
  fi
  python3 -I "$B/expected_scores.py" "$R" "$t" all --json > "$R/expected/$t.json"          # ideal: host time zero
  for H in "$HT/$t".[a-z]*-*[0-9].json; do                                                                 # measured, per machine type
    [ -f "$H" ] || continue
    MT=$(basename "$H" .json); MT=${MT#"$t".}
    # Single-process packages (H1, H3'): the host terms that set the horizon are transferred from multi-process torchrun runs
    # (0.047-0.049 s host time a step), but these scripts measure 0.039-0.040 s (W0-h3-check10, 2026-10-23). Their expected file
    # is built from the measured single-process terms, kept outside host-terms/ because gen_tasks takes speed-1 terms only
    # there; host seconds a step do not scale with the speed knob
    SP="$D/survey/$t-singleprocess-terms.json"
    if [ -f "$SP" ]; then H="$SP"; fi
    python3 -I "$B/expected_scores.py" "$R" "$t" all --host "$H" --json > "$R/expected/$t.$MT.host.json"
  done
done
# H4's measured feasible-best reference: written back after the loop above rewrites the file
python3 -I "$B/add_feasible_best.py" "$R" pkg-h4-filter-ablation "$D/survey/h4-feasible-best-2026-10-24/W0-h4-measure-r2.json" \
        "W0-h4-measure-r2 (worker-21, task 46f76e43, speed 1, 2026-10-24)"
# H5's fixed reference: its packed policy's makespan, kept beside the policies block
python3 -I "$B/add_feasible_best.py" "$R" mls-kv-report-h "$D/survey/mls-kv-report-h-reference-2026-10-25/reference.json" \
        "registry/expected/mls-kv-report-h.c3-highmem-44.host.json, policies.packed (written by build/expected_scores.py)"
log "expected scores written: $(ls "$R/expected")"
