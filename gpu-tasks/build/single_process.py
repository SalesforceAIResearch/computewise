"""Make a modded-nanogpt track-3 script single-process: one GPU as a fact of the code.

usage: python3 -I build/single_process.py <script> [<out>]       (prints the result, or writes it to <out>)

the analysis, 2026-10-23 16:08Z: the track-3 scripts are DDP-capable and fixed_global_grow_accumulation, so
gpu-mock's width rule serves a run on four GPUs at a quarter of its one-GPU time, and an agent that runs every job wide, one at
a time, reaches the best schedule whatever order it takes. A hidden-plan package then measures nothing. Real research code is
often single-process; this makes these scripts so, the way a researcher would strip DDP out of them:

- no torch.distributed: the process group, barriers, broadcast and destroy go; an all_reduce of a gradient or a loss over one
  process is the identity, so it goes; Muon's all_gather over its world-size-strided parameter blocks becomes a loop over every
  parameter on this one process (world_size 1, rank 0), which is what the original computes with one rank;
- the device is cuda:0 rather than LOCAL_RANK's, set by `torch.device("cuda", 0)`;
- rank and world size are 0 and 1 wherever the code reads them, so the data loader takes the whole global batch, and the
  training loop's own accumulation over micro-batches (mbs = 64, `for i in range(len(inputs) // mbs)`) does the rest. The
  global batch is the recorded one: nothing about the optimisation changes.

The run is launched `python <script>`; torchrun is not needed and, without the process group, a torchrun --nproc_per_node=4
launch starts four independent copies of the same one-GPU run, each on cuda:0, not one wide run. The records these scripts are
served are the recorded runs converted to one GPU by gpu-mock's convert() (fixed_global_grow_accumulation), the timing every
package here already uses for a one-GPU job.

The transform refuses a script it would leave with any `dist.` reference, so a variant that uses torch.distributed in a way not
listed here is caught rather than half-converted.
"""
import re
import sys

# (pattern, replacement) on whole lines, in order; each pattern is anchored to what the track-3 scripts contain (read from every
# PREFIX_SPEC variant's script, 2026-10-23)
LINE_RULES = [
    (r'^(\s*)device = torch\.device\("cuda", int\(os\.environ\["LOCAL_RANK"\]\)\)\s*$', r'\1device = torch.device("cuda", 0)'),
    (r'^(\s*)dist\.init_process_group\(.*\)\s*$', None),
    (r'^(\s*)dist\.barrier\(\)\s*$', None),
    (r'^(\s*)dist\.destroy_process_group\(\)\s*$', None),
    (r'^(\s*)dist\.broadcast\(p\.detach\(\), 0\)\s*$', r'\1pass  # one process: the parameters are already this process\'s'),
    (r'^(\s*)dist\.all_reduce\((p\.grad|val_loss|K|self\._precond_count), op=dist\.ReduceOp\.SUM\)\s*$',
     r'\1pass  # one process: a sum over one rank is the value itself'),
    # psgd (20260527_psgd) averages instead of summing: over one process a mean is the value itself too
    (r'^(\s*)dist\.all_reduce\((p\.grad|val_loss), op=dist\.ReduceOp\.AVG\)\s*$',
     r'\1pass  # one process: a mean over one rank is the value itself'),
    (r'^(\s*)assert 8 % dist\.get_world_size\(\) == 0\s*$', None),
    # Muon's step: with one rank, every block of world_size parameters is this process's, so the all_gather is a no-op
    (r'^(\s*)dist\.all_gather\(params_pad\[base_i:base_i \+ world_size\], params_pad\[base_i \+ rank\]\)\s*$',
     r'\1pass  # one process: every parameter is updated in place on this rank'),
]
# expression rewrites inside lines
EXPR_RULES = [
    (r'dist\.get_world_size\(\)', '1'),
    (r'dist\.get_rank\(\)', '0'),
]
HEADER = ("# Single-process: one GPU (cuda:0) and no torch.distributed; run it with `python <this script>`. The global batch is\n"
          "# accumulated over micro-batches in the training loop, as the multi-GPU original does on fewer than eight GPUs.\n")


def single_process(code):
    out = []
    for line in code.splitlines():
        for pat, rep in LINE_RULES:
            if re.match(pat, line):
                line = None if rep is None else re.sub(pat, rep, line)
                break
        if line is None:
            continue
        for pat, rep in EXPR_RULES:
            line = re.sub(pat, rep, line)
        out.append(line)
    text = "\n".join(out) + ("\n" if code.endswith("\n") else "")
    left = [l.strip() for l in text.splitlines() if re.search(r"\bdist\.", l)]
    if left:
        raise ValueError(f"torch.distributed still referenced after the transform: {left[:3]}")
    if "LOCAL_RANK" in text:
        raise ValueError("LOCAL_RANK still read after the transform")
    # keep the import (a reader may still import torch.distributed as dist) but nothing uses it; put the header after any
    # shebang/encoding line and before the module docstring's code
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and (lines[i].startswith("#!") or lines[i].startswith("# -*-")):
        i += 1
    return "".join(lines[:i]) + HEADER + "".join(lines[i:])


def main():
    src = open(sys.argv[1]).read()
    out = single_process(src)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(out)
    else:
        sys.stdout.write(out)


if __name__ == "__main__":
    main()
