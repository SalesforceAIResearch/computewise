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
