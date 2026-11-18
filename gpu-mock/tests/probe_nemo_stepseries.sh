#!/usr/bin/env bash
# PROBE (not a gate), NeMo LoRA step-series on Astra's source stack. Rewritten
# 2026-11-02 to fix six defects Astra found in the placeholder (their review):
#   (1) a TIMED validation BILLS its recorded eval duration (~75 s per 4-GPU eval,
#       the record's eval_s_per_validation); ONLY the pre-run_start / warmup eval is
#       inert. Zero validation cost would make the sufficiency bound false. So the
#       check is: warmup/pre-start eval reads None AND is unbilled; the first TIMED
#       eval reads the recorded val_loss AND bills the recorded eval time.
#   (2) real probe is torchrun with 4 ranks and the daemon at --gpus 4 (TP4), full
#       Hydra+env, a REDUCED-MODEL fixture (LOAD_CKPT=False, synthetic data) -- this
#       is CODE-PATH evidence only, never the recorded 70B task.
#   (3) every docker run is CPU/mem-capped.
#   (4) all logs preserved (no rm -rf); no `| head` on a live probe (SIGPIPE).
#   (5) occupancy is JSON with spaces -- PARSE it (python -c json), do not grep
#       '"unit":"..."'; measure enqueue unit=eval costs and materialise t_wall order;
#       busy events do not carry 'valid', so eval billing is read from the eval-unit
#       enqueue/pace entries, not a 'valid' substring.
#   (6) the env-only twin is a DIAGNOSTIC, not a prerequisite; do NOT claim a missing
#       env matcher is already supported -- a twin that collides is a REAL finding to
#       report (env must be mirrored into Hydra args for a unique full-Hydra key).
#
# NeMo LoRA val_loss is read off a tensor (.item()), so this is the STEP-SERIES path:
# NO whole_job seam. It therefore runs on the FROZEN r8 patch (90c2f7b1); it does NOT
# need r8.1's lazy-import whole_job fix. gpu-mock owns the patch assertions; Astra owns
# the base image, source tree, the reduced-model fixture, and the record.
#
# Fill before running (Astra provides): IMG (compat image digest), ENTRY (train.py in
# the image), REC (the step-series record keyed to train.py hash + full Hydra argv +
# mirrored env), RANKS (TP size, 4), GBS/MBS (8/1), and the reduced-model Hydra overrides.
#
# Usage (on c3-23, Astra's namespace):
#   sudo IMG=<digest> ENTRY=<train.py> REC=<record.json> G=<frozen-r8 tree> \
#        RANKS=4 GBS=8 MBS=1 HYDRA_OVERRIDES="model.restore_from_path=null ...synthetic..." \
#        bash probe_nemo_stepseries.sh
set -uo pipefail
IMG=${IMG:?set IMG to the compat NeMo image digest}
ENTRY=${ENTRY:?set ENTRY to train.py path inside the image}
REC=${REC:?set REC to the step-series record json path on the host}
G=${G:?set G to the FROZEN r8 tree (patch 90c2f7b1); this probe needs no whole_job seam}
FAKEGPU=${FAKEGPU:-/srv/work/gpu-mock/FakeGPU}
RANKS=${RANKS:-4}
GBS=${GBS:-8}
MBS=${MBS:-1}
HYDRA_OVERRIDES=${HYDRA_OVERRIDES:-}
CPUS=${CPUS:-16}
MEM=${MEM:-64g}
PYSITE=$(sudo docker run --rm --cpus "$CPUS" --memory "$MEM" --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)
W=${W:-/srv/work/astra-independent/mlperf-probe/nemo-probe-keep}
sudo rm -rf "$W"; mkdir -p "$W/reg"
cp "$REC" "$W/reg/nemo.json"
echo "=== NeMo step-series probe $(date -u +%FT%TZ)  image $IMG  TP=$RANKS GBS=$GBS MBS=$MBS"
echo "patch $(cd "$G/src/torch_patch" && ls *.py | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)  (expect frozen r8 90c2f7b1)"

# nvcompat vendored offline (pure-Python tree copy; no pip/cmake), written to a file.
cat > "$W/prep.sh" <<'PREP'
set -e
SRC=/gm/FakeGPU/fakegpu
mkdir -p /gm/pylib; rm -rf /gm/pylib/nvcompat; cp -r "$SRC" /gm/pylib/nvcompat
find /gm/pylib/nvcompat -name __pycache__ -type d -prune -exec rm -rf {} +
sed -i -E 's/^(\s*)import fakegpu$/\1import nvcompat as fakegpu/; s/^(\s*)from fakegpu import/\1from nvcompat import/' /gm/pylib/nvcompat/*.py
PREP

# 1: patch applies on the NeMo compat image (torch imports, NVML surface is the profile)
cat > "$W/check1.sh" <<'C1'
source /usr/local/nvidia/env.sh 2>/dev/null || true
python3 -c '
import torch; print("TORCH_VERSION", torch.__version__)
import cuda_hooks
print("DEVICE_COUNT", torch.cuda.device_count(), "AVAIL", torch.cuda.is_available())
print("NAME", torch.cuda.get_device_name(0) if torch.cuda.device_count() else "none")
'
C1
echo "--- 1. patch applies"
sudo docker run --entrypoint '' --rm --cpus "$CPUS" --memory "$MEM" --network none \
  -v "$G/src:/gm/src:ro" -v "$FAKEGPU:/gm/FakeGPU:ro" \
  -v "$G/src/torch_patch:/usr/local/nvidia/python:ro" \
  -v "$G/src/torch_patch/sitecustomize.py:$PYSITE/sitecustomize.py:ro" \
  -v "$W:/w" -e NVRT_SOCKET=/w/gm.sock -e FAKEGPU_PROFILE_DIR=/gm/FakeGPU/profiles \
  -e NVRT_PROFILE=h100 -e PYTHONPATH=/gm/pylib "$IMG" \
  bash -c 'bash /w/prep.sh; bash /w/check1.sh' 2>&1 | grep -E "TORCH_VERSION|DEVICE_COUNT|NAME|Error|Traceback" || echo "(check1 produced no recognised line)"

# 2-5: torchrun TP4, the daemon at --gpus 4; read occupancy by PARSING json.
cat > "$W/run.sh" <<RUN
set -uo pipefail
source /usr/local/nvidia/env.sh 2>/dev/null || true
export NVRT_SOCKET=/w/gm.sock NVRT_DEBUG=1
python3 -I /gm/src/daemon/gpu_daemon.py --socket /w/gm.sock --registry /w/reg --gpus $RANKS --speed 50 --occupancy /w/occ.jsonl > /w/daemon.log 2>&1 &
for i in \$(seq 200); do [ -S /w/gm.sock ] && break; sleep 0.1; done
# reduced-model fixture: LOAD_CKPT off, synthetic data, TP=$RANKS, GBS=$GBS MBS=$MBS -- CODE-PATH evidence only.
export SEED=\${SEED:-1} MINIBS=$MBS VAL_CHECK_INTERVAL=\${VAL_CHECK_INTERVAL:-384} SKIP_EVALS=\${SKIP_EVALS:-3} MAX_STEPS=\${MAX_STEPS:-800} LOAD_CKPT=False
torchrun --nproc_per_node=$RANKS --nnodes=1 $ENTRY $HYDRA_OVERRIDES > /w/train.log 2>&1
echo "TRAIN rc=\$?"
RUN
echo "--- 2-5. TP$RANKS warmup inert, eval None+billing, first timed eval bills recorded eval time"
sudo docker run --entrypoint '' --rm --cpus "$CPUS" --memory "$MEM" --network none \
  -v "$G/src:/gm/src:ro" -v "$FAKEGPU:/gm/FakeGPU:ro" \
  -v "$G/src/torch_patch:/usr/local/nvidia/python:ro" \
  -v "$G/src/torch_patch/sitecustomize.py:$PYSITE/sitecustomize.py:ro" \
  -v "$W:/w" -e NVRT_SOCKET=/w/gm.sock -e FAKEGPU_PROFILE_DIR=/gm/FakeGPU/profiles \
  -e NVRT_PROFILE=h100 -e PYTHONPATH=/gm/pylib "$IMG" \
  bash -c 'bash /w/prep.sh; bash /w/run.sh'

# analysis parses the occupancy JSON (it has spaces; grep on "unit":"x" misses it).
python3 - "$W/occ.jsonl" <<'PY'
import json, sys
steps=evals=0; eval_billed=0.0; step_billed=0.0
unreached=0; eval_series_reads=0; first_timed_eval=None; mats=[]
for line in open(sys.argv[1], encoding="utf-8"):
    try: e=json.loads(line)
    except Exception: continue
    k=e.get("kind")
    if k=="enqueue":
        u=e.get("unit")
        if u=="step": steps+=1; step_billed+=float(e.get("recorded_s") or e.get("cost_s") or 0) or 0.0
        elif u=="eval": evals+=1; eval_billed+=float(e.get("recorded_s") or e.get("cost_s") or 0) or 0.0
    if k=="materialise":
        mats.append((e.get("t_wall"), e.get("role"), e.get("tier"), e.get("value")))
        if e.get("tier")=="eval_unreached": unreached+=1
        if e.get("role") in ("val_loss","eval_loss") and e.get("tier") in ("series","extrapolated"):
            eval_series_reads+=1
            if first_timed_eval is None: first_timed_eval=e.get("value")
print(f"STEP units paced: {steps} (billed {step_billed:.1f}s)")
print(f"EVAL units paced: {evals} (billed {eval_billed:.1f}s)  <-- TIMED evals MUST bill recorded eval time, not 0")
print(f"eval_unreached (warmup/pre-start val -> None, UNBILLED): {unreached}")
print(f"val_loss series reads (timed): {eval_series_reads}; first timed val_loss = {first_timed_eval}")
# materialise t_wall ordering: evals must be non-decreasing in t_wall and interleaved after steps
tw=[m[0] for m in mats if m[0] is not None]
print("materialise t_wall monotonic non-decreasing:", all(tw[i]<=tw[i+1] for i in range(len(tw)-1)) if len(tw)>1 else "n/a")
PY
echo "--- consumer on None (a Lightning reducer may reject None at the pre-start eval -- a REAL finding if so):"
grep -iE 'TypeError|NoneType|cannot .*None|reduce.*None' "$W/train.log" 2>/dev/null | head -3 || echo "no None-consumer crash"
echo "--- train tail:"; tail -6 "$W/train.log" | cut -c1-160

# 6: env-only twin DIAGNOSTIC (not a prerequisite). Run twice with one env changed,
# confirm the two lookups select DIFFERENT record_ids; if the SAME, the env is not in
# the key -- a REAL finding (mirror env into Hydra args), NOT a pass.
echo "--- 6. env-only twin DIAGNOSTIC (SEED=<a> vs SEED=<b>, same argv): record_ids must differ."
echo "    Astra drives the two-run pair with the real record set; if they collide, report it;"
echo "    do NOT label a missing env matcher as supported."
echo "=== logs preserved at $W ; done $(date -u +%FT%TZ) ==="
