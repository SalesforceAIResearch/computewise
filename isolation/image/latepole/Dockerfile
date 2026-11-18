# The agent's box for K2's late pole (p006.task-env=k2-latepole-8gpu; analysis accepted the layout 2026-11-02 12:09 MST,
# isolation/notes/k2-latepole-image-plan-2026-11-02.md). Built by build_latepole.sh beside this file. Two Python
# environments in one 8-GPU image, because the source's two halves need incompatible transformers majors:
#   /opt/envs/a  the mlsbench-kv stack exactly as ml-node:py311-torch2110-mlskv carries it (env-a.txt is that image's
#                pip freeze less torch, pip and transformers; transformers from the MLS-Bench commit), 18 KV jobs
#   /opt/envs/b  transformers 4.46.3 and K2's 46-package freeze (env-b.txt, gpu-tasks' environment-b.freeze.txt 71f7e260
#                less torch), 11 DLM and 9 sparse jobs
# Both take torch 2.11.0+cpu from PyTorch's CPU index first, then every other package with --no-deps from its list, so
# each resolves to exactly the recorded set; pip check runs on each. gpu-mock's installer runs once per interpreter with
# the same --prefix, --bundle, --config, --src and --fakegpu: each run writes that venv's
# sitecustomize, and rewrites the shared PREFIX/python and PREFIX/pylib identically. The venvs are plain (no
# --system-site-packages), so neither sees the base interpreter's site-packages. /usr/local/bin/python3, the base
# interpreter, carries neither stack: the scheduler and box-init run on it.
# The rest is isolation/image/Dockerfile's, in the same order and with the same names. No PIP_CONSTRAINT: one file
# cannot fit both environments, and the box is offline.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG DEBIAN_SNAPSHOT
ARG TORCH_CPU_INDEX
ARG TORCH_VERSION
ARG TRANSFORMERS_COMMIT
ARG BUILD_JSON
ARG BOX_GPUS=8
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN printf 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/%s bookworm main\ndeb [check-valid-until=no] https://snapshot.debian.org/archive/debian/%s bookworm-updates main\ndeb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/%s bookworm-security main\n' \
      "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list \
 && rm -f /etc/apt/sources.list.d/debian.sources \
 && apt-get -o Acquire::Retries=5 update \
 && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
      git procps less ripgrep curl util-linux build-essential cmake \
 && rm -rf /var/lib/apt/lists/*

COPY env-a.txt env-b.txt /tmp/envs/
RUN python3 -m venv /opt/envs/a \
 && /opt/envs/a/bin/pip install --index-url "$TORCH_CPU_INDEX" --no-deps "torch==${TORCH_VERSION}" \
 && /opt/envs/a/bin/pip install --no-deps -r /tmp/envs/env-a.txt \
 && /opt/envs/a/bin/pip install --no-deps "git+https://github.com/huggingface/transformers@${TRANSFORMERS_COMMIT}" \
 && /opt/envs/a/bin/python -c "import transformers; assert transformers.__version__ == '5.3.0', transformers.__version__; from transformers import DynamicCache" \
 && /opt/envs/a/bin/pip check
RUN python3 -m venv /opt/envs/b \
 && /opt/envs/b/bin/pip install --index-url "$TORCH_CPU_INDEX" --no-deps "torch==${TORCH_VERSION}" \
 && /opt/envs/b/bin/pip install --no-deps -r /tmp/envs/env-b.txt \
 && /opt/envs/b/bin/python -c "import importlib.metadata as m; v = m.version('transformers'); assert v == '4.46.3', v" \
 && /opt/envs/b/bin/pip check

# gpu-mock, once per interpreter, each under its own environment's list (FakeGPU adds nothing on 3.11)
COPY gpu-mock/ /tmp/gm/
RUN for e in a b; do \
      PIP_CONSTRAINT=/tmp/envs/env-$e.txt bash /tmp/gm/install_into_image.sh \
        --prefix /usr/local/nvidia --bundle /tmp/gm/runtime --config /tmp/gm/h100-${BOX_GPUS}gpu.yaml \
        --src /tmp/gm/torch_patch --fakegpu /tmp/gm/FakeGPU --python /opt/envs/$e/bin/python || exit 1; \
    done \
 && install -m 0755 /tmp/gm/stage_dev_nodes.sh /usr/local/sbin/stage-dev-nodes \
 && rm -rf /tmp/gm
RUN echo "GPUS=${BOX_GPUS}" > /etc/box-gpus

COPY claude /usr/local/bin/claude
COPY codex/ /usr/local/lib/codex/
RUN ln -s /usr/local/lib/codex/bin/codex /usr/local/bin/codex
COPY scheduler/ /tmp/sched/
RUN BOX_GPUS=${BOX_GPUS} sh /tmp/sched/install.sh /tmp/sched /usr/local/bin/python3 && rm -rf /tmp/sched
COPY box-init /usr/local/sbin/box-init
COPY bash-oom-reset /tmp/bash-oom-reset
RUN chmod 0755 /usr/local/bin/claude /usr/local/sbin/box-init \
 && useradd -m -u 1000 -s /bin/bash researcher \
 && mv /usr/bin/bash /usr/bin/bash.real \
 && install -m 0755 /tmp/bash-oom-reset /usr/bin/bash && rm /tmp/bash-oom-reset

RUN mkdir -p /usr/local/share/image-manifest && cd /usr/local/share/image-manifest \
 && printf '%s\n' "$BUILD_JSON" > build-args.json \
 && pip freeze --all > pip-freeze.txt \
 && /opt/envs/a/bin/pip freeze --all > pip-freeze-a.txt \
 && /opt/envs/b/bin/pip freeze --all > pip-freeze-b.txt \
 && cp /tmp/envs/env-a.txt /tmp/envs/env-b.txt . && rm -rf /tmp/envs \
 && dpkg-query -W -f '${Package}=${Version}\n' > dpkg.txt \
 && HOME=/tmp claude --version > claude-version.txt \
 && HOME=/tmp codex --version > codex-version.txt \
 && python3 --version > python-version.txt

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
LABEL p006.box-stdin=fifo
ENTRYPOINT ["/usr/local/sbin/box-init"]
