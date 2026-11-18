#!/usr/bin/env python3
"""Summarise an nvidia-smi sample log: lines '<epoch> idx, util, mib;idx, util, mib;...'.
    python3 -I tests/smi_summary.py <nvidia-smi.log>"""
import sys
n, busy, peak_mem, peak_util = 0, {}, {}, {}
for line in open(sys.argv[1]):
    parts = line.strip().split(None, 1)
    if len(parts) != 2:
        continue
    gpus = [g.strip() for g in parts[1].split(";") if g.strip()]
    if not gpus:
        continue
    n += 1
    for g in gpus:
        try:
            i, u, m = (int(x.strip()) for x in g.split(","))
        except ValueError:
            continue
        busy[i] = busy.get(i, 0) + (u > 0)
        peak_util[i] = max(peak_util.get(i, 0), u)
        peak_mem[i] = max(peak_mem.get(i, 0), m)
print(f"samples {n}; per GPU: samples with util>0 {dict(sorted(busy.items()))}, "
      f"peak util % {dict(sorted(peak_util.items()))}, peak MiB used {dict(sorted(peak_mem.items()))}")
