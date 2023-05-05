FROM ubuntu:focal
ARG ROCM_VERSION=5.1.3
ARG AMDGPU_VERSION=22.10.3
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl libnuma-dev gnupg \
    && curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
    && printf "deb [arch=amd64] https://repo.radeon.com/rocm/apt/$ROCM_VERSION/ ubuntu main" | tee /etc/apt/sources.list.d/rocm.list \
    && printf "deb [arch=amd64] https://repo.radeon.com/amdgpu/$AMDGPU_VERSION/ubuntu focal main" | tee /etc/apt/sources.list.d/amdgpu.list \
    && curl -sL https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && printf "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-15 main" | tee /etc/apt/sources.list.d/rocm.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        rocm-dev libssl-dev build-essential ninja-build cmake git llvm-15 \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh /dev/stdin -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV PATH=/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
