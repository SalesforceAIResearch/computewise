# The agent's box for ComputeWise: the harness, the task's Python stack and gpu-mock's layers in one image.
# Built only by build.sh beside this file, on the Docker host that runs the boxes. build.sh pins every input,
# passes it in as a build argument, assembles the context, and writes the build's manifest.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG DEBIAN_SNAPSHOT
ARG TORCH_CPU_INDEX
ARG TORCH_VERSION
ARG TASK_PACKAGES
ARG BUILD_JSON

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# Debian packages from a dated snapshot, so a rebuild installs the same versions.
RUN printf 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/%s bookworm main\ndeb [check-valid-until=no] https://snapshot.debian.org/archive/debian/%s bookworm-updates main\ndeb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/%s bookworm-security main\n' \
      "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list \
 && rm -f /etc/apt/sources.list.d/debian.sources \
 && apt-get -o Acquire::Retries=5 update \
 && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
      git procps less ripgrep curl util-linux build-essential cmake \
 && rm -rf /var/lib/apt/lists/*

# The task's Python stack. torch comes from PyTorch's CPU index, because PyPI's x86_64 torch is the
# CUDA build; its dependencies and every other package come from PyPI under the recorded run's
# constraints, so each version is the one the recorded run used. torch's runtime deps are named here
# because --no-deps skipped them. A torch that needs more (2.11 wants setuptools<82) gets it through the
# task's install.packages under its constraints, not here: adding a package here would change the
# versions in every image built from this file, and the DPO and battery images are proven as they are.
# triton is a dependency of the CUDA build only and is never installed.
COPY constraints.txt /tmp/constraints.txt
RUN pip install --index-url "$TORCH_CPU_INDEX" --no-deps "torch==${TORCH_VERSION}" \
 && pip install -c /tmp/constraints.txt $TASK_PACKAGES \
      sympy networkx jinja2 filelock typing-extensions fsspec \
 && (pip check || true)
# Caches a task's packages fetch on first use, filled here because the box is offline. tiktoken (the mng3 data-prep
# package, gpu-tasks 2026-10-21) downloads GPT-2's vocab.bpe and encoder.json on first get_encoding('gpt2'); the task's
# environment.env points TIKTOKEN_CACHE_DIR at this directory. A no-op for images without tiktoken.
RUN if python3 -c "import tiktoken" 2>/dev/null; then \
      mkdir -p /usr/local/share/tiktoken \
      && TIKTOKEN_CACHE_DIR=/usr/local/share/tiktoken python3 -c "import tiktoken; tiktoken.get_encoding('gpt2')" \
      && chmod -R a+rX /usr/local/share/tiktoken; fi

# gpu-mock's layers, installed by gpu-mock's own script. FakeGPU builds from a writable copy. Its install runs
# under the recorded run's constraints, as gpu-mock's own test images do. FakeGPU's only runtime requirement is
# tomli>=1.1.0 on Python before 3.11, so on the battery's 3.10 tomli gets the recorded version (2.0.1) instead of
# the newest. On 3.11 the step installs nothing but FakeGPU itself. A constraint FakeGPU could not meet fails the
# build here instead of moving a pinned package.
# Everything goes under /usr/local/nvidia, where NVIDIA's own CUDA images keep driver libraries, with neutral names,
# so nothing the agent can read without root names the mock (the study, 2026-10-21; isolation's seat probe in
# tools/seat-probe/ is the check). The install script vendors FakeGPU without its pip metadata and puts sitecustomize
# in this interpreter's site-packages, so no PYTHONPATH entry is needed.
COPY gpu-mock/ /tmp/gm/
RUN PIP_CONSTRAINT=/tmp/constraints.txt bash /tmp/gm/install_into_image.sh \
      --prefix /usr/local/nvidia --bundle /tmp/gm/runtime --config /tmp/gm/h100-4gpu.yaml \
      --src /tmp/gm/torch_patch --fakegpu /tmp/gm/FakeGPU \
      --python /usr/local/bin/python3 \
 && install -m 0755 /tmp/gm/stage_dev_nodes.sh /usr/local/sbin/stage-dev-nodes \
 && rm -rf /tmp/gm
# The GPU count the box presents (build.sh --gpus; package-search angle F, 2026-10-31). install_into_image.sh writes the
# profile's device count into cuda_hooks; stage-dev-nodes reads GPUS when box-init runs it, so a count other than 4 goes
# into the image's environment. A 4-GPU build passes no BOX_GPUS, and this layer then sets nothing.
ARG BOX_GPUS
RUN if [ -n "${BOX_GPUS:-}" ]; then echo "GPUS=${BOX_GPUS}" > /etc/box-gpus; fi

# The harness: Claude Code's native Linux binary, checksum-verified by build.sh.
COPY claude /usr/local/bin/claude
# Codex CLI, its native musl tree as the npm platform package ships it (bin/, codex-path/, codex-resources/); the
# binary finds its resources beside itself, so the link on PATH is all it needs.
COPY codex/ /usr/local/lib/codex/
RUN ln -s /usr/local/lib/codex/bin/codex /usr/local/bin/codex
# The node's job scheduler, laid out as a distribution package lays out SLURM (the scheduler's
# experiments/scheduler/install.sh; only its bin/, slurm/ and install.sh are in the build context). It starts only when
# a run asks for it (box-init, BOX_SCHEDULER=1); otherwise its commands answer as on a node whose controller is down.
COPY scheduler/ /tmp/sched/
RUN sh /tmp/sched/install.sh /tmp/sched /usr/local/bin/python3 && rm -rf /tmp/sched
# The entrypoint, and the shell wrapper that undoes the harness's OOM protection for the agent's
# commands. The wrapper replaces bash itself (the real binary moves to bash.real), so every way of
# starting bash goes through it: PATH lookups, absolute /bin/bash, and #!/bin/bash scripts.
COPY box-init /usr/local/sbin/box-init
COPY bash-oom-reset /tmp/bash-oom-reset
RUN chmod 0755 /usr/local/bin/claude /usr/local/sbin/box-init \
 && useradd -m -u 1000 -s /bin/bash researcher \
 && mv /usr/bin/bash /usr/bin/bash.real \
 && install -m 0755 /tmp/bash-oom-reset /usr/bin/bash && rm /tmp/bash-oom-reset

# What this build contains, for build.sh's manifest and for every run's run.json.
RUN mkdir -p /usr/local/share/image-manifest && cd /usr/local/share/image-manifest \
 && printf '%s\n' "$BUILD_JSON" > build-args.json \
 && pip freeze --all > pip-freeze.txt \
 && dpkg-query -W -f '${Package}=${Version}\n' > dpkg.txt \
 && HOME=/tmp claude --version > claude-version.txt \
 && HOME=/tmp codex --version > codex-version.txt \
 && python3 --version > python-version.txt

# Pip installs go through the recorded run's constraints, as the build's did.
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt
RUN cp /tmp/constraints.txt /etc/pip-constraints.txt && rm /tmp/constraints.txt

# gpu-mock's runtime environment (its env.sh): only the library path NVIDIA's CUDA images set. The patch finds the
# daemon's socket and the device count by default and takes the speed from the daemon; nvidia-smi is a wrapper that
# sets its library's two variables for its own process only. run_one.py sets PYTHONPATH to the project and reads this
# image's ENV to know it must not put GPU_MOCK_SPEED in the box. Images built before 2026-10-21 03:00 UTC carried
# /opt/gpu-mock paths and MOCK_*/GPU_MOCK_* variables here.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
# p006.box-stdin=fifo: this image's box-init takes /run/box/brief as a FIFO, so run_one.py may hold the harness's
# stdin open (stream-json input) instead of handing it a file; images without the label get a regular file.
LABEL p006.box-stdin=fifo
ENTRYPOINT ["/usr/local/sbin/box-init"]
