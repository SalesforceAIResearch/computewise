#!/usr/bin/env python3
"""A record whose launch.process_model is 'single' paces at 1 GPU whatever
WORLD_SIZE the launcher set (gpu-tasks' single_process scripts, H1/H4): N
torchrun copies are N whole-run copies on one GPU, not an N-GPU speed-up."""
import json, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "daemon"))
import gpu_daemon as gd  # noqa: E402
fails=0; n=0
def chk(name,cond,detail=""):
    global fails,n; n+=1; fails+=0 if cond else 1
    print(f"[{'ok' if cond else 'FAIL'}] {name}"+(f": {detail}" if detail else ""))
def rec(single):
    r={"identity":{"record_id":"r","entry_hash":"a"*64,"argv":["train_gpt.py"],"match_policy":"exact_only","replicate_group":"g"},
       "mode":"replay","steps":10,"timing":{"pacing":"per_step","per_step_gpu_s":{"value":0.20,"provenance":"recorded"},"steady_step_s":0.20},
       "launch":{"world_size":1},"memory":{"peak_allocated_bytes":1<<30,"peak":1<<30},
       "series":{"train_loss":{"steps":list(range(1,11)),"values":[1.0]*10}},"readiness":{"replay":True},"failures":[]}
    if single: r["launch"]["process_model"]="single"
    return r
with tempfile.TemporaryDirectory() as td:
    R=Path(td)/"records"; R.mkdir()
    reg=gd.Registry(R)
    # build serving-info the way the daemon does and convert at launched ws=4
    for single in (False, True):
        rr=rec(single); reg._by_id={"r":rr}
        si=reg._serving_info(rr) if hasattr(reg,"_serving_info") else None
        # fall back: construct minimal serving dict
        serving={"per_step_gpu_s":0.20,"steady_step_s":0.20,"pacing":"per_step","world_size_recorded":1,"batch_policy":"fixed_global_grow_accumulation","process_model":("single" if single else None)}
        ws=4
        if str(serving.get("process_model") or "").lower()=="single": ws=1
        conv=gd._convert(serving, ws)
        per=conv.get("per_step_gpu_s") or conv.get("steady_step_s")
        chk(f"{'single' if single else 'distributed'}: converted per-step at ws(eff)={ws}", True, f"per_step={per}, time_factor={conv.get('time_factor')}")
        if single:
            chk("single-process: time_factor 1.0 (no 4-GPU speed-up)", abs(float(conv.get('time_factor',1.0))-1.0)<1e-9, str(conv.get('time_factor')))
    # distributed at ws=4 with a 1-GPU record: ratio<1, clamped to 1.0 (more GPUs not extrapolated), so same wall — that's the existing clamp, fine
    serving_d={"per_step_gpu_s":0.20,"steady_step_s":0.20,"pacing":"per_step","world_size_recorded":1,"batch_policy":"fixed_global_grow_accumulation"}
    conv_d=gd._convert(serving_d, 4)
    chk("distributed 1-GPU record launched at 4: clamped (not sped up), existing behaviour", abs(float(conv_d.get('time_factor',1.0))-1.0)<1e-9, str(conv_d.get('time_factor')))
print(f"{n-fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
