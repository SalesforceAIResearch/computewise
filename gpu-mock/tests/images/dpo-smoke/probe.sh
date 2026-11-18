#!/usr/bin/env bash
# Runs inside a DPO test image: layer A, then the import sequence that crashed on
# the controller (torch, then datasets -> pandas), with and without the torch patch.
# The crash was the patch's datetime.datetime substitution (fixed 2026-10-20). A pass
# now means rc 0, zero 'datetime.datetime size changed' warnings, and datetime's
# module still 'datetime' with basicsize 48, both patched and unpatched.
source /usr/local/nvidia/env.sh; unset LD_PRELOAD MOCK_PCI_ROOT
nvidia-smi -L
for mode in patched unpatched; do
  [ "$mode" = unpatched ] && unset PYTHONPATH        # drops /opt/gpu-mock/site, so no sitecustomize
  for i in 1 2 3; do
    python -X faulthandler -c "import torch, datasets, pandas, datetime; print('$mode import ok: pandas', pandas.__version__, '| cuda devices', torch.cuda.device_count(), '| datetime.datetime', datetime.datetime.__module__, datetime.datetime.__basicsize__)" > /tmp/o 2>&1
    rc=$?
    echo "$mode run $i: rc=$rc, datetime-size warnings=$(grep -c 'datetime.datetime size changed' /tmp/o)"
    grep -E "import ok|Fatal Python" /tmp/o | head -2
  done
done
