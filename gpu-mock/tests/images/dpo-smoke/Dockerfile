# Test image for gpu-mock's DPO smoke-task replay (tests/replay_dpo.sh), built on
# the GCP host. Its Python layer repeats experiments/isolation/image/Dockerfile's
# lines (same pins, same order), and it installs gpu-mock through the same
# install_into_image.sh call, so a result here should transfer to the agent's box.
# Differences from the box: no Claude Code binary and no user; trl stays installed
# (the box installs the agent's trl tree at start); pandas is a build argument.
# Correction, 2026-10-20: the unpinned pandas (3.0.6) segfaulted only because
# the torch patch replaced datetime.datetime with a subclass
# (virtual_clock._patch_datetime, fixed). pandas was never the cause, and the pin is not a fix.
#
# Build context: the parent of a staged bundle (container/stage_bundle.sh OUT=<ctx>/gpu-mock).
#   docker build -f tests/images/dpo-smoke/Dockerfile -t gm-dpo:x86 <ctx>
#   docker build ... --build-arg PANDAS_SPEC=pandas -t gm-dpo:x86-unpinned <ctx>
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates procps less ripgrep \
      build-essential cmake util-linux \
 && rm -rf /var/lib/apt/lists/*

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
# isolation's constraints (2026-10-20) pin pandas==2.2.2 and pyarrow==17.0.0 as well;
# PANDAS_SPEC=pandas builds the unpinned variant, which crashed under the old patch
# and is kept as the regression test for the datetime fix.
ARG PANDAS_SPEC=pandas==2.2.2
RUN printf '%s\n' torch==2.4.0 transformers==4.44.2 accelerate==0.33.0 datasets==2.20.0 \
      peft==0.12.0 tokenizers==0.19.1 safetensors==0.4.4 huggingface-hub==0.24.5 numpy==1.26.4 \
      pyarrow==17.0.0 "$PANDAS_SPEC" > /tmp/constraints.txt \
 && pip install --index-url "$TORCH_INDEX" torch==2.4.0 \
 && pip install -c /tmp/constraints.txt transformers accelerate datasets peft tokenizers \
      safetensors huggingface-hub numpy pyarrow pandas \
 && pip install -c /tmp/constraints.txt "trl @ git+https://github.com/huggingface/trl@e5295792" \
 && python -c "import importlib.util as u, pandas, numpy, pyarrow; print('stack: pandas', pandas.__version__, 'numpy', numpy.__version__, 'pyarrow', pyarrow.__version__, 'rich', bool(u.find_spec('rich')))"

COPY gpu-mock/ /tmp/gm/
RUN PIP_CONSTRAINT=/tmp/constraints.txt bash /tmp/gm/install_into_image.sh \
      --prefix /opt/gpu-mock --bundle /tmp/gm/runtime --config /tmp/gm/h100-4gpu.yaml \
      --mock-pci /tmp/gm/mock-pci --src /tmp/gm/torch_patch --fakegpu /tmp/gm/FakeGPU \
      --python /usr/local/bin/python3 \
 && install -m 0755 /tmp/gm/stage_dev_nodes.sh /usr/local/sbin/stage-dev-nodes \
 && rm -rf /tmp/gm /tmp/constraints.txt
# trl@e5295792's dpo_trainer imports rich unconditionally (the controller image needed
# the same layer); a separate layer so the gpu-mock layer above stays cached.
ARG RICH_SPEC=rich
RUN pip install "$RICH_SPEC" && python -c "import rich, trl.trainer.dpo_trainer; print('rich', rich.__version__ if hasattr(rich, '__version__') else 'ok', '| dpo_trainer imports')"
ENV HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
