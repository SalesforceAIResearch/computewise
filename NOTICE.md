# Third-party notices

ComputeWise's own code is licensed under the Apache License 2.0 (`LICENSE.txt`). This file lists the
third-party material ComputeWise uses, the form it takes here, and its licence.

## Sources the benchmark packages are built from

| Source | Licence | How ComputeWise uses it |
|---|---|---|
| [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) @ `4ea6b937` | MIT, Copyright (c) 2024 Keller Jordan | The optimizer packages are built from its track-3 optimization records (timings and validation-loss series) and from the training scripts embedded in those logs. The data is in the Hugging Face dataset, not in this repository. |
| [test-time-compute-for-tabular-foundation-models](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models) @ `45fb141b` and its [results dataset](https://huggingface.co/datasets/nkh/test-time-compute-for-tabular-foundation-models-results) | Apache-2.0 (code), CC-BY-4.0 (results) | Timings and metrics for the tabular packages. |
| [huggingface/trl](https://github.com/huggingface/trl) | Apache-2.0 | Example training scripts for the early smoke tasks, fetched by `gpu-tasks/build/fetch_trl.py`. |
| [maomaocun/dLLM-Cache](https://github.com/maomaocun/dLLM-Cache), [Kamichanw/d2Cache](https://github.com/Kamichanw/d2Cache), [VILA-Lab/Elastic-Cache](https://github.com/VILA-Lab/Elastic-Cache) | Apache-2.0 | For `pkg-k2-latepole`: dLLM-Cache is fetched at a pinned commit by `gpu-tasks/build/fetch_mlsbench_inputs.py`; the d2Cache and Elastic-Cache files come from the user's MLS-Bench clone. None is in this repository. |
| [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) @ `08b83a6f` | MIT | For `pkg-k2-latepole`: its `configuration_llada.py` and `modeling_llada.py`, fetched by `gpu-tasks/build/fetch_mlsbench_inputs.py`. Not in this repository. |
| [Imbernoulli/MLS-Bench](https://github.com/Imbernoulli/MLS-Bench) @ `b8739449` | No licence declared upstream | The paper's four MLS-Bench tasks (`mls-kv-gate`, `mls-kv-report`, `mls-kv-report-h`, `pkg-k2-latepole`) are built by `gpu-tasks/build/add_mlsbench_tasks.sh`, which clones MLS-Bench at this commit; the generators read its leaderboards, task configurations and evaluators from that clone. This repository and the dataset include no MLS-Bench code or data. |

## Fetched at staging time, never shipped

The datasets (GSM8K, HumanEval, MATH-500, ARC-Challenge, LongBench, LongBench-v2, NeedleBench, FineWeb) and
the model weights (for example Qwen2.5-3B-Instruct, under the Qwen Research Licence, non-commercial) are
referenced by pinned revision and sha256. The staging tools download them from their own hosts. They stay
under their own licences and terms.

## Fetched by the simulator's image build

When a box image is built, `gpu-mock/container/stage_bundle.sh` and `runtime.Dockerfile` fetch the following at
pinned revisions. None of their code is in this repository.

| Source | Licence | Use |
|---|---|---|
| [FanBB2333/FakeGPU](https://github.com/FanBB2333/FakeGPU) @ `3b5b6b47` | MIT | Base of the fake CUDA runtime. Our patches are applied at build time. |
| [NVIDIA/k8s-test-infra](https://github.com/NVIDIA/k8s-test-infra) @ `f7bbbf02` and `a82e8b9c` | Apache-2.0 | The mock NVML (`libnvidia-ml`), the stub CUDA driver (`libcuda`), `libmockfs`, and the H100 profile. |
| NVIDIA `nvidia-utils-580` = `580.65.06-0ubuntu1`, sha256-checked | NVIDIA driver licence | `nvidia-smi`, downloaded from NVIDIA's repository and run against the mock NVML. It ends up inside the locally built image, which is one reason the images are not redistributed. |

The study's own images were built with [brantje/fake-nvidia](https://github.com/brantje/fake-nvidia)'s runtime
script, which has no declared licence. This release builds the same NVIDIA userland from the sources above
instead. The one difference is the stub `libcuda`, which the box loads but never calls, so served timings and
scores are unchanged.

## Box images

`isolation/image/` contains only Dockerfiles and build scripts. The images they build install third-party
tools, including proprietary agent CLIs, under those tools' own terms. Build the images locally and do not
redistribute them.
