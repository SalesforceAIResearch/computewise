#!/usr/bin/env bash
# MLS-Bench KV risk check: can a 3B bf16
# checkpoint load through the mock, and does transformers behave on a device
# that reports CUDA? Runs in the TRL box image (it carries transformers) with
# this tree's patch mounted, no daemon (no replay: the forward computes for
# real on the CPU through FakeGPU, which is the worst case for time).
# Reports: transformers/torch versions, load wall, peak host RSS, memory the
# box reports, one-token forward wall, DynamicCache import, empty_cache/synchronize.
# Usage (on the VM):  sudo [IMG=gm-dpo:x86] [HF=/srv/hf] [MODEL=Qwen/Qwen2.5-3B-Instruct] bash tests/smoke_qwen_load_in_image.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:-gm-dpo:x86}"; HF="${HF:-/srv/hf}"; MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
M=(-v "$G/src:/gm/src:ro" -v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:$PYSITE/sitecustomize.py:ro" -v "$HF:/hf:ro")
docker run --entrypoint '' --rm --cpus 8 --memory 48g --network none "${M[@]}" -e HF_HOME=/hf -e MODEL="$MODEL" -e NVRT_DEBUG=1 "$IMG" bash -c '
source /usr/local/nvidia/env.sh 2>/dev/null || true
python - <<PY
import os, time, resource, sys
t0 = time.perf_counter()
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
print(f"transformers {transformers.__version__} torch {torch.__version__} python {sys.version.split()[0]}", flush=True)
print(f"cuda available {torch.cuda.is_available()} device_count {torch.cuda.device_count()} name {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}", flush=True)
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024   # GiB on Linux (KiB)
model_id = os.environ["MODEL"]
t1 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, local_files_only=True)
t2 = time.perf_counter()
print(f"from_pretrained (cpu, bf16): {t2 - t1:.1f} s, peak rss {rss():.2f} GiB, params {sum(p.numel() for p in model.parameters())/1e9:.2f} B", flush=True)
model = model.to("cuda"); model.eval()
t3 = time.perf_counter()
print(f".to(cuda) through the mock: {t3 - t2:.1f} s, peak rss {rss():.2f} GiB, device of a param {next(model.parameters()).device}, dtype {next(model.parameters()).dtype}", flush=True)
try:
    print(f"torch.cuda.memory_allocated {torch.cuda.memory_allocated()/2**30:.2f} GiB, max {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
except Exception as e:
    print(f"memory_allocated failed: {e!r}", flush=True)
ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")
t4 = time.perf_counter()
with torch.inference_mode():
    kw = dict(input_ids=ids, use_cache=True, past_key_values=DynamicCache(), cache_position=torch.arange(ids.shape[1], device="cuda"))
    import inspect as _i
    if "logits_to_keep" in _i.signature(model.forward).parameters:
        kw["logits_to_keep"] = 1          # the MLS-Bench script passes it; transformers < 4.45 has no such argument
    else:
        print("note: this transformers has no logits_to_keep; the MLS-Bench script needs a newer one", flush=True)
    out = model(**kw)
torch.cuda.synchronize()
t5 = time.perf_counter()
nxt = out.logits[:, -1:].argmax(dim=-1)
pkv = out.past_key_values
nlayers = len(pkv) if hasattr(pkv, "__len__") else -1
ntok = int(ids.shape[1]); fwd = t5 - t4; shape = tuple(out.logits.shape); nexttok = tok.decode(nxt[0])
print(f"one forward (real CPU compute through FakeGPU, {ntok} tokens): {fwd:.1f} s; logits {shape} next token {nexttok!r}; cache layers {nlayers}", flush=True)
torch.cuda.empty_cache()
print(f"empty_cache ok; peak rss {rss():.2f} GiB; total {time.perf_counter() - t0:.1f} s", flush=True)
print("SMOKE OK", flush=True)
PY'
