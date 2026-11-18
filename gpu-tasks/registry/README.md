# Registry data

The task files, recorded job runs, staged artifacts and expected-score tables are distributed as the
[Salesforce/computewise](https://huggingface.co/datasets/Salesforce/computewise) dataset. To use them with this
code, copy the dataset into place from the repository root:

```bash
git clone https://huggingface.co/datasets/Salesforce/computewise /tmp/computewise-data
rsync -a /tmp/computewise-data/data/registry/ gpu-tasks/registry/
rsync -a /tmp/computewise-data/data/sources/sol-tabular-ttc/source/hf gpu-tasks/survey/sol-tabular-ttc/source/
(cd gpu-tasks/build && python3 -I validate.py ../registry)    # expect: contract violations 0
```

The paper's four MLS-Bench tasks (`mls-kv-gate`, `mls-kv-report`, `mls-kv-report-h`, `pkg-k2-latepole`) are built here
from a clone of MLS-Bench rather than shipped, because MLS-Bench declares no licence. After the copy above:

```bash
bash gpu-tasks/build/add_mlsbench_tasks.sh                     # expect: tasks 66 | ... | contract violations 0
```
