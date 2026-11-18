---
license: cc-by-4.0
pretty_name: Test-Time Compute for Tabular Foundation Models - TabArena Results
task_categories:
- tabular-classification
- tabular-regression
tags:
- tabular
- benchmark
- tabarena
- test-time-compute
- tabular-foundation-models
- tabpfn
- tabicl
size_categories:
- 1K<n<10K
configs:
- config_name: results
  data_files: data/results.parquet
  default: true
- config_name: arms
  data_files: data/arms.parquet
---

# Test-Time Compute for Tabular Foundation Models: TabArena Results

Per-cell TabArena test errors for the 12 methods ("arms") in the main comparison of
*Test-Time Compute for Tabular Foundation Models: Mechanisms, Gains, and Limits*.
Every arm covers all 51 TabArena datasets (816 cells), so these files are enough to recompute
the paper's Elo ratings and paired comparisons.

**Code:** [GitHub](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models) · **Paper:** [arXiv:2610.12005](https://arxiv.org/abs/2610.12005)

## Files

| file | content |
|---|---|
| `data/results.parquet` | config `results`: one row per arm × cell (9,792 rows) |
| `data/arms.parquet`, `data/arms.csv` | config `arms`: one row per arm (config, Elo, ΔElo, GPU hours, file checksum) |
| `raw/<arm>/results_per_split.csv` | each arm in TabArena's format, the input to `ttc.eval.elo` |

## Arms

ΔElo is measured against TabArena's default entry for TabPFN-3 and TabICL v2, and against our
frozen run for TabFM. GPU hours are fit + predict time over 816 cells on one H100.

| backbone | arm | LR | Elo | ΔElo | GPU h |
|---|---|---:|---:|---:|---:|
| TabPFN-3 | [`tabpfn3_frozen`](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models/blob/main/configs/tabpfn3/frozen.yaml) |  | 1658 | +1 | 0.5 |
|  | [`tabpfn3_native_views_32`](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models/blob/main/configs/tabpfn3/native_views_32.yaml) |  | 1675 | +21 | 1.7 |
|  | [`tabpfn3_full_ft`](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models/blob/main/configs/tabpfn3/full_ft.yaml) | 1e-6 | 1681 | +25 | 7.7 |
|  | [`tabpfn3_diagscale`](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models/blob/main/configs/tabpfn3/diagscale.yaml) | 3e-2 | 1682 | +26 | 6.6 |
|  | [`tabpfn3_aggregation_96`](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models/blob/main/configs/tabpfn3/aggregation_96.yaml) |  | 1721 | +67 | 247.5 |
|  | [`tabpfn3_diagscale_aggregation_96`](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models/blob/main/configs/tabpfn3/diagscale_aggregation_96.yaml) | 1e-2 | 1748 | +93 | 253.1 |
| TabICL v2 | [`tabiclv2_frozen`](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models/blob/main/configs/tabiclv2/frozen.yaml) |  | 1592 | +9 | 1.2 |
|  | [`tabiclv2_full_ft`](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models/blob/main/configs/tabiclv2/full_ft.yaml) | 1e-5 | 1659 | +77 | 10.9 |
|  | [`tabiclv2_diagscale`](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models/blob/main/configs/tabiclv2/diagscale.yaml) | 3e-2 | 1664 | +82 | 10.8 |
| TabFM | [`tabfm_frozen`](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models/blob/main/configs/tabfm/frozen.yaml) |  | 1792 | 0 | 19.3 |
|  | [`tabfm_full_ft`](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models/blob/main/configs/tabfm/full_ft.yaml) | 1e-4 | 1820 | +28 | 61.9 |
|  | [`tabfm_diagscale`](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models/blob/main/configs/tabfm/diagscale.yaml) | 1e-2 | 1812 | +20 | 59.4 |

## Columns of `results`

| column | meaning |
|---|---|
| `arm`, `backbone`, `method`, `learning_rate` | arm identity; `learning_rate` is null without fine-tuning |
| `dataset`, `fold` | TabArena dataset and split (`fold` enumerates repeat × fold) |
| `metric`, `metric_error` | `roc_auc` (binary, error = 1 − AUC), `log_loss` (multiclass) or `rmse` (regression); lower is better |
| `time_train_s`, `time_infer_s` | wall-clock seconds per cell |
| `problem_type` | `binary`, `multiclass` or `regression` |
| other columns | TabArena bookkeeping (`tabarena_method`, `imputed`, ...); `metric_error_val` is not recorded (null) |

## Usage

```python
from datasets import load_dataset

repo = "nkh/test-time-compute-for-tabular-foundation-models-results"
results = load_dataset(repo, "results", split="train")
arms = load_dataset(repo, "arms", split="train")
```

Recompute an Elo rating with the code repository:

```bash
hf download nkh/test-time-compute-for-tabular-foundation-models-results \
    --repo-type dataset --local-dir hf_results
python -m ttc.eval.elo \
    --candidate hf_results/raw/tabpfn3_diagscale/results_per_split.csv \
    --reference-field hf_results/raw/tabfm_frozen/results_per_split.csv \
    --baseline "TA-TABPFN-3 (default)"            # ΔElo +26
```

## Notes

- **Elo.** Fixed field of 68 entries: the 67 official TabArena entries (fetched by TabArena, not
  included here) plus our frozen TabFM run. Each arm is added alone and rated with TabArena's
  Bradley–Terry solver, `RF (default)` = 1000.
- GPU fine-tuning is not bitwise deterministic; re-running an adaptation config gives slightly
  different cell errors.

## License

The results are released under CC-BY-4.0. TabArena data and reference results, and the TabPFN-3,
TabICL v2 and TabFM weights (none included), are under their own licenses.

## Citation

```bibtex
@article{ning2026testtime,
  title   = {Test-Time Compute for Tabular Foundation Models: Mechanisms, Gains, and Limits},
  author  = {Ning, Kanghui and Bilo{\v{s}}, Marin and Wilson, James T. and Zhang, Yilang and Rasul, Kashif
             and Song, Dongjin and Schneider, Anderson and Nevmyvaka, Yuriy},
  journal = {arXiv preprint arXiv:2610.12005},
  year    = {2026}
}
```
