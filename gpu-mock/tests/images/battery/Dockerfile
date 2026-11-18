# Test image for gpu-mock on the trl-regression-battery task's stack, built on the
# GCP host: Python 3.10, torch 2.1.2 (CPU wheel), transformers 4.36.2, accelerate
# 0.25.0, datasets 2.16.1, peft 0.7.1, tyro 0.5.18, rich 11.2.0, wandb 0.13.11,
# under the recorded runs' constraints (gpu-tasks' environment.install,
# artifacts/trl/env/0022af66577e/constraints.txt). gpu-mock installs through
# install_into_image.sh, as in the agent's box. isolation builds the box image for
# this task; this image exists so the patch is proven on this stack first.
# Build context: constraints.txt and gpu-mock/ (a staged bundle); see build.sh.
# The task's recorded interpreter; isolation's battery box uses the same tag (pinned there by digest).
FROM python:3.10.13-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates procps build-essential cmake \
 && rm -rf /var/lib/apt/lists/*

COPY constraints.txt /tmp/constraints.txt
# torch's CPU wheel alone first (--no-deps): resolving its dependencies on the CPU index
# failed, since pip discarded the index's typing_extensions wheel and could not build the sdist
# there (no flit_core). Its dependencies then come from PyPI under the constraints.
RUN pip install --index-url https://download.pytorch.org/whl/cpu --no-deps torch==2.1.2 \
 && pip install -c /tmp/constraints.txt torch==2.1.2 transformers==4.36.2 accelerate==0.25.0 \
      datasets==2.16.1 peft==0.7.1 tyro==0.5.18 rich==11.2.0 wandb==0.13.11 \
 && python -c "import sys, torch, transformers, accelerate, datasets, peft, pandas, numpy, pyarrow; print('stack: python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__, 'accelerate', accelerate.__version__, 'datasets', datasets.__version__, 'peft', peft.__version__, 'pandas', pandas.__version__, 'numpy', numpy.__version__, 'pyarrow', pyarrow.__version__)"

COPY gpu-mock/ /tmp/gm/
RUN PIP_CONSTRAINT=/tmp/constraints.txt bash /tmp/gm/install_into_image.sh \
      --prefix /opt/gpu-mock --bundle /tmp/gm/runtime --config /tmp/gm/h100-4gpu.yaml \
      --mock-pci /tmp/gm/mock-pci --src /tmp/gm/torch_patch --fakegpu /tmp/gm/FakeGPU \
      --python /usr/local/bin/python3 \
 && install -m 0755 /tmp/gm/stage_dev_nodes.sh /usr/local/sbin/stage-dev-nodes \
 && rm -rf /tmp/gm
ENV HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
