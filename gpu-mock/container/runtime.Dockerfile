# syntax=docker/dockerfile:1.7
# The box's NVIDIA userland, built from NVIDIA's own sources and packages at pinned revisions:
#   lib/libnvidia-ml.so*   mock NVML, NVIDIA/k8s-test-infra pkg/gpu/mocknvml at NVML_REVISION (Apache-2.0)
#   lib/libcuda.so*        stub CUDA driver, NVIDIA/k8s-test-infra shims/libcuda at LIBCUDA_REVISION (Apache-2.0);
#                          libcudart.so.12/.13 and libcudart.so are symlinks to it, so a process maps one file
#   bin/nvidia-smi.real    the genuine nvidia-smi from NVIDIA's nvidia-utils-580 package, sha256-checked, with its
#                          RPATH set to ../lib so it reads the mock NVML
#   licenses/              k8s-test-infra's LICENSE
# The nvidia-smi recipe follows k8s-test-infra's deployments/nvml-mock/Dockerfile. Build with build_bundle.sh, which
# passes the pins below and exports the result as a directory. Nothing here is redistributed: the NVIDIA binary
# stays in the locally built image (see NOTICE.md).
# The stub libcuda reports 8 devices unless MOCK_NVML_NUM_DEVICES / MOCK_CUDA_NUM_DEVICES say otherwise. In the box
# it is loaded but never called (the torch layer answers every CUDA query), so that default is never read there.

ARG NVML_GO_IMAGE=golang:1.26.6-bookworm@sha256:116d58cbd88c1297624acc6e967a060012422bacf9930927e23fb719189c6f36
ARG LIBCUDA_GO_IMAGE=golang:1.27.2-bookworm@sha256:5cf287a799e6b94384bad13d16b14904c531f51ba65792237e122ce42b392f61
ARG DEBIAN_IMAGE=debian:bookworm-20260824-slim@sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171

FROM ${NVML_GO_IMAGE} AS mocknvml
ARG NVML_REVISION
WORKDIR /src
RUN test -n "${NVML_REVISION}" \
    && git init -q . \
    && git remote add origin https://github.com/NVIDIA/k8s-test-infra.git \
    && git fetch -q --depth=1 origin "${NVML_REVISION}" \
    && git checkout -q --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${NVML_REVISION}"
RUN make -C pkg/gpu/mocknvml clean && make -C pkg/gpu/mocknvml
RUN mkdir -p /out/lib /out/licenses \
    && cp -a pkg/gpu/mocknvml/libnvidia-ml.so* /out/lib/ \
    && rm -f /out/lib/*.h \
    && cp LICENSE /out/licenses/k8s-test-infra-APACHE-2.0.txt

FROM ${LIBCUDA_GO_IMAGE} AS libcuda
ARG LIBCUDA_REVISION
ENV GOTOOLCHAIN=local
WORKDIR /src
RUN test -n "${LIBCUDA_REVISION}" \
    && git init -q . \
    && git remote add origin https://github.com/NVIDIA/k8s-test-infra.git \
    && git fetch -q --depth=1 origin "${LIBCUDA_REVISION}" \
    && git checkout -q --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${LIBCUDA_REVISION}"
# this revision vendors nothing: fetch the module graph first, as k8s-test-infra's own nvml-mock Dockerfile does
RUN go mod download && make -C shims/libcuda clean && make -C shims/libcuda
RUN mkdir -p /out/lib \
    && cp -L shims/libcuda/libcuda.so.1 /out/lib/libcuda.so.1 \
    && ln -s libcuda.so.1 /out/lib/libcuda.so \
    && ln -s libcuda.so.1 /out/lib/libcudart.so.12 \
    && ln -s libcuda.so.1 /out/lib/libcudart.so.13 \
    && ln -s libcudart.so.13 /out/lib/libcudart.so

FROM ${DEBIAN_IMAGE} AS nvidia_smi
ARG TARGETARCH
ARG NVIDIA_KEY_FINGERPRINT
ARG NVIDIA_UTILS_VERSION
ARG NVIDIA_UTILS_SHA256_AMD64
ARG NVIDIA_UTILS_SHA256_ARM64
# the build tools from a dated Debian snapshot at pinned versions, so the stage builds the same way later
ARG DEBIAN_SNAPSHOT=20260828T000000Z
RUN set -eux; \
    rm -f /etc/apt/sources.list.d/debian.sources; \
    printf '%s\n' \
      "deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}/ bookworm main" \
      "deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}/ bookworm-security main" \
      > /etc/apt/sources.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends "ca-certificates=20250419~deb12u1" "curl=7.88.1-10+deb12u15" \
      "gnupg2=2.2.40-1.1+deb12u2" "patchelf=0.14.3-1+b1"; \
    rm -rf /var/lib/apt/lists/*; \
    case "${TARGETARCH}" in \
      amd64) repo_arch=x86_64; want_sha="${NVIDIA_UTILS_SHA256_AMD64}" ;; \
      arm64) repo_arch=sbsa;   want_sha="${NVIDIA_UTILS_SHA256_ARM64}" ;; \
      *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    repo="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${repo_arch}"; \
    curl -fsSL "${repo}/3bf863cc.pub" -o /tmp/nvidia.pub; \
    fpr=$(gpg --batch --show-keys --with-colons /tmp/nvidia.pub | awk -F: '$1 == "fpr" { print $10; exit }'); \
    test "${fpr}" = "${NVIDIA_KEY_FINGERPRINT}"; \
    gpg --batch --dearmor -o /usr/share/keyrings/nvidia.gpg /tmp/nvidia.pub; \
    echo "deb [signed-by=/usr/share/keyrings/nvidia.gpg] ${repo} /" > /etc/apt/sources.list.d/nvidia.list; \
    apt-get update; \
    cd /tmp; \
    apt-get download "nvidia-utils-580=${NVIDIA_UTILS_VERSION}"; \
    deb=$(ls /tmp/nvidia-utils-580_*.deb); \
    echo "${want_sha}  ${deb}" | sha256sum -c -; \
    dpkg-deb -x "${deb}" /tmp/nvidia-utils; \
    mkdir -p /out/bin; \
    cp /tmp/nvidia-utils/usr/bin/nvidia-smi /out/bin/nvidia-smi.real; \
    chmod 0755 /out/bin/nvidia-smi.real; \
    patchelf --set-rpath '$ORIGIN/../lib' /out/bin/nvidia-smi.real

FROM scratch AS export
COPY --from=mocknvml /out/ /
COPY --from=libcuda /out/ /
COPY --from=nvidia_smi /out/ /
