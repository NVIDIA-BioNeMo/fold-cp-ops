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

"""Execute the README's `Use` tutorial verbatim, under a torchrun/srun launch of its own.

Purpose
    `docs/refactor_dtensor_api.md` §2 makes the README tutorial the acceptance artifact for the
    whole DTensor API, and §6 L7 specifies that the fenced block is EXTRACTED FROM `README.md` AND
    EXECUTED UNDER TORCHRUN. This is that runner. It is a launched artifact rather than a pytest
    test for a measured reason, recorded below.

Why this is not a pytest test
    The block owns a full process lifecycle: it calls `DistributedManager.initialize()` and
    `init_nvshmem()`, and ends with `cleanup()`, which destroys the process group. Inside a pytest
    session that would take every subsequent test down with it, so the obvious workaround is to run
    it in a CHILD process, one per rank, on a shifted rendezvous port.

    **That was tried and it HANGS**, at cp=2 locally and at cp=16 on two nodes over IB. The child
    reaches nvshmem and blocks there; its captured stdout carries `cudaHostRegister with IoMemory
    failed with error=800` and `ibgda_alloc_and_map_qp_uar with GPU as handler failed`. A second
    nvshmem runtime cannot be brought up inside a process tree whose parent already holds one --
    the runtime is per-process but the device and IPC resources it registers are per-GPU. So the
    nesting is not a test-harness detail to be tuned; it is a thing that does not work, and the
    correct shape is a launch of its own. The same block runs clean that way: measured RC=0 at cp=2.

Functionality & semantics
    Reads `README.md` relative to this file (four parents up), extracts the single ```python fenced
    block, and `exec`s it in a fresh module namespace **VERBATIM** -- no substitution of any kind.

    It used to rewrite the shape line to `B, N, D = 1, 256, 128` "so the cell fits". That made the
    runner a poor witness for the only thing it exists to prove: a reader runs the shape that is
    PUBLISHED, and 256x128 is small enough to miss anything the published 4096x256 would hit --
    a different tile path, a different store variant, a symmetric allocation an order of magnitude
    larger. A tutorial checked at a shape nobody will run is checked in name only. The substitution
    is gone; what executes is character-for-character what is on the page.

    Reads `WORLD_SIZE` / `MASTER_ADDR` / `MASTER_PORT` from the launcher, exactly as the block does
    -- nothing is set here that a reader would not have.

Input requirements
    Must be launched with >= 2 ranks under `torchrun` or `srun` with an `env://` rendezvous. A
    single-rank launch would exercise the cp=1 fallback and prove nothing about the distributed
    path. `CPO_CACHE_ENABLED=0` is recommended so the compile is cold, like a reader's first run.

Returns / Raises
    Exit 0 on success. Any exception from the block propagates with its own traceback -- the block
    IS the subject, so wrapping it would hide the thing under test.

Run it
    CPO_CACHE_ENABLED=0 PYTHONPATH=$PWD python -m torch.distributed.run --nproc_per_node=2 \
        tests/distributed/workflows/run_readme_tutorial.py
"""

from __future__ import annotations

import pathlib
import re
import sys

#: The shape the tutorial publishes. NOT substituted -- this runner executes the block verbatim.
#: Kept only so the runner can PRINT what it ran, and so a README edit that changes the shape shows
#: up here as a mismatch to be looked at rather than as a silent change of subject.
_PUBLISHED_SHAPE = "B, N, D = 1, 4096, 256"


def readme_python_block(readme: pathlib.Path | None = None) -> str:
    """Return the README's single ```python fenced block, with the shape line substituted.

    Args:
        readme: Path to `README.md`. ``None`` resolves it relative to THIS file
            (`tests/distributed/workflows/` -> four parents up), so the result does not depend on
            the working directory a launcher happened to choose.

    Returns:
        The block's source text, fences stripped. VERBATIM -- nothing is rewritten.

    Raises:
        AssertionError: if the README is missing, or does not hold EXACTLY ONE ```python block
            ("the first of several" is a silent way to execute the wrong one).
    """
    readme = readme or pathlib.Path(__file__).resolve().parents[3] / "README.md"
    assert readme.is_file(), f"README.md not found at {readme}"
    blocks = re.findall(r"```python\n(.*?)\n```", readme.read_text(), re.S)
    assert len(blocks) == 1, (
        f"README.md has {len(blocks)} ```python blocks; this runner executes THE tutorial and "
        f"cannot tell which one that is. Keep one, or teach this function to name it."
    )
    return blocks[0]


def main() -> None:
    """Execute the block. Prints one line per rank so a launch log shows which ranks finished."""
    import os

    src = readme_python_block()
    # Print the shape line that is about to run. A log saying "OK" without saying at WHAT shape is
    # the failure mode this runner just came out of.
    shape = next((ln.strip() for ln in src.splitlines() if ln.strip().startswith("B, N, D")), "?")
    if _PUBLISHED_SHAPE not in src:
        print(f"[warn] README shape is {shape!r}, expected {_PUBLISHED_SHAPE!r}", flush=True)
    print(
        f"[rank {os.environ.get('RANK', '?')}] executing README block VERBATIM: {shape}", flush=True
    )
    ns: dict = {"__name__": "__readme_tutorial__", "__file__": "README.md"}
    exec(compile(src, "README.md#Use", "exec"), ns)  # noqa: S102 -- the block IS the subject
    print(f"[rank {os.environ.get('RANK', '?')}] README tutorial OK", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
