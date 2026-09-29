# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# fold-cp-ops — CuTe-DSL + NVSHMEM dev image for the A2A-fused TriMul workflow (SM90). The repo is
# bind-mounted at job runtime, not baked in. Versions live ONLY in pyproject.toml; the TAG restates
# them: cutedsl<dsl>-nvshmem<nvshmem>-ngc<base>, lowercase because docker repo names must be.

# T=cutedsl4.4.2-nvshmem3.7-ngc26.07  H=<registry-host>:<port>  P=<registry-namespace>/fold-cp-ops
# Build: docker build -t fold-cp-ops:$T -f docker/Dockerfile .        # context = REPO ROOT
# Push:  docker tag fold-cp-ops:$T $H/$P/$T && docker push $H/$P/$T

# Enroot, on an allocated node. enroot's URI splits registry from path on '#', not '/'. A cluster may
# pin an https_proxy in /etc/enroot/enroot.conf that 403s CONNECT here while direct TCP is open:
#   export no_proxy=<your-registry-domain>; enroot import -o $T.sqsh "docker://$H#$P/$T:latest"
FROM nvcr.io/nvidia/pytorch:26.07-py3

# Take the container's torch (26.07 = 2.13.0a0+…nv26.07, CUDA 13.3), then override ITS dependencies
# with our pins -- never the reverse. torch declares nvidia-nvshmem-cu13==3.4.5 against our ==3.7.0,
# so a plain resolve is free to move torch to 2.9.1/cu12 instead; the gate refuses that.
COPY docker/image_gate.py /opt/image_gate.py
COPY pyproject.toml LICENSE /opt/fold-cp-ops-deps/
COPY fold_cp_ops/_version.py /opt/fold-cp-ops-deps/fold_cp_ops/_version.py

# The base image's torch version, for the gate's preservation check to compare against.
RUN python -c "import importlib.metadata as m, pathlib; \
    pathlib.Path('/opt/torch_before.txt').write_text(m.version('torch'))"

# `pip install '.[dev]'` is the README command. The empty fold-cp-ops shell it creates is then
# uninstalled so nothing shadows the bind-mounted tree, and a second `--no-deps` pass re-forces the
# pins over torch's own -- explicit, rather than relying on this base declaring no nvidia-* deps.
RUN set -eux; \
    cd /opt/fold-cp-ops-deps; \
    pip install --no-cache-dir '.[dev]'; \
    pip uninstall -y fold-cp-ops; \
    python /opt/image_gate.py --print-pins > /opt/pins.txt; cat /opt/pins.txt; \
    pip install --no-cache-dir --no-deps -r /opt/pins.txt; \
    python -c "import cutlass, cutlass.cute as cute; print('cutlass-dsl', cutlass.__version__)"; \
    python -c "import nvshmem.core; print('nvshmem4py OK')"; \
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# The base resolves libnvshmem_host.so.3 from /usr/local/cuda via ldconfig; prepend the wheel's dir
# so the host lib matches the device bitcode the kernels link. The path is a literal because
# `nvidia.nvshmem` is a namespace package (`__file__` is None); the assertion fails a wrong one.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH}
RUN python -c "import ctypes; ctypes.CDLL('libnvshmem_host.so.3'); \
    m=open('/proc/self/maps').read(); \
    assert 'dist-packages/nvidia/nvshmem/lib/libnvshmem_host' in m, \
        'nvshmem host lib is NOT the pip wheel — LD_LIBRARY_PATH did not take'; \
    print('nvshmem host lib -> wheel OK')"

# cuEquivariance: benchmark/distributed/trimul_dtensor_reference.py imports cuequivariance_torch lazily as
# the reference CP baseline. Unversioned on purpose -- not pinned in pyproject.toml, and pinning it here
# would restore the second source of truth this file exists to remove.
RUN set -eux; \
    CUDA_MAJOR="$(echo "${CUDA_VERSION:-13}" | cut -d. -f1)"; \
    pip install --no-cache-dir \
        "cuequivariance_ops_cu${CUDA_MAJOR}" \
        "cuequivariance_ops_torch_cu${CUDA_MAJOR}" \
        cuequivariance_torch

# `cuequivariance_ops_torch._ext` dlopens libcue_ops.so from inside the wheel's own lib/ dir, which
# is not on the default loader path; without this the import dies while pip reports success.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/cuequivariance_ops/lib:${LD_LIBRARY_PATH}

# Final gate: pins hold, torch unchanged, every dependency imports. Runs LAST, after cuEquivariance,
# which is free to have moved an earlier package behind pip's back.
RUN python /opt/image_gate.py

WORKDIR /workspace/fold-cp-ops
