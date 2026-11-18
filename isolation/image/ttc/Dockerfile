# The agent's box for independent Sol's TabArena adoption twins (p006.task-env=sol-tabular-ttc-r8; the study 2026-11-02
# 16:55Z, orders 1023-1038). Built by build_ttc.sh beside this file, over Sol's source-only image (the pinned TTC project,
# torch 2.9.1 CPU, AutoGluon 1.5.0; Sol's package-search/sol-independent/tabular-ttc/Dockerfile.probe), which this file
# never rebuilds. It adds what isolation/image/Dockerfile adds to a box, in the same order and with the same names:
# gpu-mock's r8 install (FakeGPU native, network on at build time), the harnesses, the scheduler, box-init and the shell
# wrapper, the researcher user; plus Sol's OpenML snapshot baked read-only at /opt/cache/openml (Sol, 2026-11-02 17:15Z)
# and Sol's 805f single_cell.py, which copies that snapshot to a private writable directory per job (openml 0.15.1 rewrites
# a pickle on read; isolation's read-only loader check, 17:39Z). No torch install here: Sol's 2.9.1 stays.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG BUILD_JSON
ARG DEBIAN_SNAPSHOT
USER root
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# the box's tools, as the H100 boxes carry them (git and build-essential are in the base), from the same dated snapshot
# as isolation/image/Dockerfile, so a rebuild installs the same versions
RUN printf 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/%s bookworm main\ndeb [check-valid-until=no] https://snapshot.debian.org/archive/debian/%s bookworm-updates main\ndeb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/%s bookworm-security main\n' \
      "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list \
 && rm -f /etc/apt/sources.list.d/debian.sources \
 && apt-get -o Acquire::Retries=5 update \
 && apt-get -o Acquire::Retries=5 install -y --no-install-recommends procps less ripgrep curl util-linux cmake \
 && rm -rf /var/lib/apt/lists/*
COPY gpu-mock/ /tmp/gm/
RUN bash /tmp/gm/install_into_image.sh \
      --prefix /usr/local/nvidia --bundle /tmp/gm/runtime --config /tmp/gm/h100-4gpu.yaml \
      --src /tmp/gm/torch_patch --fakegpu /tmp/gm/FakeGPU --python "$(command -v python3)" \
 && install -m 0755 /tmp/gm/stage_dev_nodes.sh /usr/local/sbin/stage-dev-nodes \
 && rm -rf /tmp/gm
COPY claude /usr/local/bin/claude
COPY codex/ /usr/local/lib/codex/
RUN ln -s /usr/local/lib/codex/bin/codex /usr/local/bin/codex
COPY scheduler/ /tmp/sched/
RUN sh /tmp/sched/install.sh /tmp/sched "$(command -v python3)" && rm -rf /tmp/sched
COPY box-init /usr/local/sbin/box-init
COPY bash-oom-reset /tmp/bash-oom-reset
RUN chmod 0755 /usr/local/bin/claude /usr/local/sbin/box-init \
 && useradd -m -u 1000 -s /bin/bash researcher \
 && mv /usr/bin/bash /usr/bin/bash.real \
 && install -m 0755 /tmp/bash-oom-reset /usr/bin/bash && rm /tmp/bash-oom-reset
# Sol's OpenML snapshot: org/ only (sol-ttc-inputs.json is build metadata and goes to the manifest), root-owned and
# read-only to the agent. Sol's adapter copies it per job; nothing writes here.
COPY openml/ /opt/cache/openml/
RUN chown -R root:root /opt/cache/openml && find /opt/cache/openml -type d -exec chmod 0555 {} + \
 && find /opt/cache/openml -type f -exec chmod 0444 {} +
# Historical, off the run path: every task job runs tools/single_cell.py from its task tree (gen_sol_tabular embeds the
# adapter as task content, matched to the records' entry_hash). This baked copy stays at 805f54de in the installed pair
# (670ed7123389 + 8a02c0f13bfc) after Sol's docstring-only fix to 143f69ed (2026-11-04, Sol: keep the pair, no rebuild);
# no task job reads it. A rebuild would carry the then-current adapter here, by build_ttc.sh's SC= and its hash check.
COPY single_cell.py /opt/sol-ttc/single_cell.py
RUN chmod 0644 /opt/sol-ttc/single_cell.py
RUN mkdir -p /usr/local/share/image-manifest && cd /usr/local/share/image-manifest \
 && printf '%s\n' "$BUILD_JSON" > build-args.json \
 && pip freeze --all > pip-freeze.txt \
 && dpkg-query -W -f '${Package}=${Version}\n' > dpkg.txt \
 && HOME=/tmp claude --version > claude-version.txt \
 && HOME=/tmp codex --version > codex-version.txt \
 && python3 --version > python-version.txt \
 && (cd /opt/cache/openml && find . -type f | LC_ALL=C sort | xargs sha256sum) > openml-cache.sha256
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 OPENML_CACHE_DIR=/opt/cache/openml \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
WORKDIR /
LABEL p006.box-stdin=fifo
ENTRYPOINT ["/usr/local/sbin/box-init"]
