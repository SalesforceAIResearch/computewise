# Registry data

This directory is the benchmark's registry: the task files (`tasks/`), the recorded job runs the simulator replays
(`records/`), the files staged into each task's project (`artifacts/`), the expected scores (`expected/`) and the record
contract (`RECORD-FORMAT.md`). It is a byte-for-byte copy of `data/registry/` in the
[Salesforce/computewise](https://huggingface.co/datasets/Salesforce/computewise) dataset on Hugging Face, which holds the
same files plus the scored agent runs (`data/runs/`). The tabular study's results CSVs, which the tabular generators
check their records against, are at `gpu-tasks/survey/sol-tabular-ttc/source/hf/`.

Licence: CC-BY-4.0 (`LICENSE`), as the dataset. The upstream material keeps its own terms: modded-nanogpt (MIT), the
tabular test-time-compute study (CC-BY-4.0) and TabPFN (`third_party/`).

Check it from the repository root:

```bash
(cd gpu-tasks/build && python3 -I validate.py ../registry)    # expect: tasks 62 | ... | contract violations 0
```

The paper's four MLS-Bench tasks (`mls-kv-gate`, `mls-kv-report`, `mls-kv-report-h`, `pkg-k2-latepole`) are built here
from a clone of MLS-Bench rather than shipped, because MLS-Bench declares no licence:

```bash
bash gpu-tasks/build/add_mlsbench_tasks.sh                     # expect: tasks 66 | ... | contract violations 0
```
