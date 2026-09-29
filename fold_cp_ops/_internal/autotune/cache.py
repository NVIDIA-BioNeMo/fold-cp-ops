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

"""The on-disk record of what autotuning decided, so a restart does not re-sweep.

Distinct from `_internal/cache_utils.py`, which caches COMPILED ARTIFACTS. This caches a
*measurement result*: for one (kernel, shape) key, what each candidate cost and therefore which one
won. The two have different invalidation rules -- a compiled artifact is invalid when the source
changes, a timing is invalid when the source OR the hardware OR the measurement geometry changes --
which is why they are separate files rather than one cache with two kinds of entry.

Three debts retired relative to upstream:

* **No Triton.** Upstream subclassed ``triton.runtime.cache.FileCacheManager`` to write one JSON
  file, pulling a large compiler stack into a CuTe-DSL package for a directory and an open().
* **Keys are canonical, not stringly.** Upstream hashed ``str(config)`` and asserted at runtime that
  the strings happened to be unique. Here the key is built from `AutotuneConfig.key`, so renaming a
  knob invalidates cleanly instead of colliding, and ``1`` and ``True`` are different entries.
* **The device is part of the key.** A timing measured on an H100 says nothing about an H200, and
  upstream's key did not include the device -- so a shared home directory silently handed one
  machine another's answer. The device name and capability are in the key.

**Under a collective, only rank 0 writes.** Every rank holds the same reconciled result (see
`consensus`), so N ranks writing the same file is at best redundant and at worst a torn read on a
shared filesystem. Reads happen on every rank, which is safe and is what lets all of them skip the
sweep together.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from fold_cp_ops._internal.autotune.config import AutotuneConfig
from fold_cp_ops._internal.cache_security import (
    CacheRoot,
    ensure_private_file,
    validate_cache_root,
)
from fold_cp_ops._version import __version__

#: Bumped when the MEASUREMENT changes meaning -- the timer, its geometry, the reduction. Source
#: changes are covered by the package version; this covers "same version, different question".
_SCHEMA = 1


def cache_root_status() -> CacheRoot:
    """Where autotuning results are stored, and whether this process may use that directory.

    Purpose
        Resolve the location and the permission in one step, so a caller cannot obtain a path
        without also being handed the verdict on it.

    Functionality & semantics
        Honors ``CPO_AUTOTUNE_CACHE_DIR``, else ``CPO_HOME``, else ``~/.fold_cp_ops/autotune``. All
        three spellings are ``CPO_*`` per the de-branding rule; the upstream harness read
        ``FOLD_CP_OPS_*``, which no launcher in this repo sets, so its disk cache was silently
        inert. All three go through `cache_security.validate_cache_root`, which creates the
        directory ``0700`` and refuses it when somebody else owns it or others can write it.

        This cache stores timings, not code -- so a hostile entry here mis-STEERS a config choice
        rather than executing anything. That is a smaller hazard than the artifact caches and is
        still worth closing: the winning config decides a kernel's tile shape, and a run silently
        tuned by somebody else's numbers is a wrong answer about performance with nothing pointing
        at its cause.

    Returns:
        A `cache_security.CacheRoot`. ``usable`` False makes :meth:`ResultCache.load` a miss and
        :meth:`ResultCache.store` a no-op; a sweep then re-measures, which is the same cost as a
        cold cache.
    """
    explicit = os.environ.get("CPO_AUTOTUNE_CACHE_DIR", "").strip()
    if explicit:
        root = Path(explicit)
    else:
        home = os.environ.get("CPO_HOME", "").strip() or str(Path.home())
        root = Path(home) / ".fold_cp_ops" / "autotune"
    return validate_cache_root(root)


def cache_root() -> Path:
    """Where autotuning results are stored.

    Returns:
        The directory, created ``0700`` if absent. Path only: a caller that is about to read or
        write must ask :func:`cache_root_status` whether the directory is safe, because a path
        cannot carry that answer.
    """
    return cache_root_status().path


def _device_tag() -> str:
    """A stable label for the part the timings were measured on.

    Timings are not portable across parts, so the device belongs in the key. Falls back to
    ``"nogpu"`` rather than raising, so the cache is usable in a CPU-only unit test.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "nogpu"
        name = torch.cuda.get_device_name(0).replace(" ", "_")
        major, minor = torch.cuda.get_device_capability(0)
        return f"{name}_sm{major}{minor}"
    except Exception:  # pragma: no cover - defensive
        return "nogpu"


class ResultCache:
    """Reads and writes one kernel's autotuning results, keyed by shape and candidate set.

    Args:
        kernel: The tuned function's name, used only to make the file readable on disk.
        enabled: Whether the cache is consulted at all. Defaults ON, and to the ``CPO_AUTOTUNE_CACHE``
            environment variable when it is set (``"0"`` disables). Pass an explicit bool to override
            both.

    Why ON is the default, having been OFF
        `main` persists its autotuning results: its ``autotune`` decorator defaults
        ``cache_results=True`` (which overrides the ``Autotuner.__init__`` default of ``False``, and
        contradicts that parameter's own docstring), so `main` sweeps a given shape ONCE EVER and
        every later process loads the same winner -- its tuner reports "finished after 0.00s" and
        there are 14 ``gemm_tuned.autotune.json`` files on disk to show for it.

        Defaulting OFF here made this package **re-sweep in every process**, so the elected config
        varied run to run. That is not merely noise: a candidate can be genuinely BIMODAL, and a
        sweep that happens to catch its fast sample elects it for the whole process. Measured on
        op 7 at the A2A cell ``N_token=2048, D=256, outgoing`` -- tile ``(128, 160)`` timed
        5.955 / 11.249 / 6.411 ms across three measurements in ONE process, was elected whenever the
        sweep saw the low sample, and then cost 8.2 ms in-chain against 5.8 ms for the stable
        winners. End to end that is an intermittent ~16% regression against `main`, which never
        shows it because it is not re-deciding.

        The original reason for OFF -- "a stale timing is a silent perf regression" -- applies to
        `main` equally, and `main` accepted it. It is also already mitigated the same way in both
        trees: `_path` keys on the CANDIDATE SET as well as the shape, so a changed pool cannot
        return an old winner. What remains is that ONE sweep's choice is frozen; that too is `main`'s
        behaviour, and a wrong pick is now at least reproducible rather than intermittent.
    """

    def __init__(self, kernel: str, *, enabled: Optional[bool] = None):
        self.kernel = kernel
        self.enabled = (
            os.environ.get("CPO_AUTOTUNE_CACHE", "1") != "0" if enabled is None else enabled
        )

    def _path(self, tuning_key: Tuple, configs: Sequence[AutotuneConfig]) -> Optional[Path]:
        """The file backing one (shape, candidate-set) entry.

        The CANDIDATE SET is in the key, not just the shape. If it were not, adding a config to the
        pool would return the old winner from cache and the new config would never be measured --
        which reads exactly like "the new config is not faster".

        Args:
            tuning_key: The shape/dtype facts identifying this call.
            configs: The candidates that were (or would be) benchmarked.

        Returns:
            The path, whose parent exists and is ``0700``; or ``None`` when the cache root is not
            safe to use. ``None`` PROPAGATES the verdict rather than restating it -- every caller
            already has to handle "no entry", so an unusable root becomes the case they were
            written for instead of a second one they might forget.
        """
        material = json.dumps(
            {
                "schema": _SCHEMA,
                "version": __version__,
                "device": _device_tag(),
                "kernel": self.kernel,
                "key": repr(tuning_key),
                "configs": sorted(repr(c.key()) for c in configs),
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(material.encode()).hexdigest()[:32]
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in self.kernel)[:80]
        root = cache_root_status()
        if not root.usable:
            return None
        return root.path / f"{safe}.{digest}.json"

    def load(
        self, tuning_key: Tuple, configs: Sequence[AutotuneConfig]
    ) -> Optional[Dict[AutotuneConfig, float]]:
        """Read a previous sweep's timings, or None.

        Args:
            tuning_key: The shape/dtype facts identifying this call.
            configs: The candidates for this call, used both to key the file and to map the stored
                keys back to config objects.

        Returns:
            Config to milliseconds, or None when disabled, absent, unreadable, or when the stored
            entry does not cover exactly this candidate set. A partial hit is treated as a MISS
            rather than merged: merging would compare a fresh timing against a stale one measured
            under different thermal conditions.
        """
        if not self.enabled or os.environ.get("CPO_AUTOTUNE_REFRESH", "0") == "1":
            return None
        path = self._path(tuning_key, configs)
        if path is None:
            return None
        # `ensure_private_file` replaces `path.exists()` and subsumes it: a missing file, a
        # symlink, a non-regular entry and a file owned by somebody else all read as a miss, which
        # is what a caller of a cache wants for every one of them.
        if not ensure_private_file(path):
            return None
        try:
            stored = json.loads(path.read_text())["timings"]
        except (OSError, ValueError, KeyError):
            return None
        by_key = {repr(c.key()): c for c in configs}
        if set(stored) != set(by_key):
            return None
        return {by_key[k]: float(v) for k, v in stored.items()}

    def store(
        self,
        tuning_key: Tuple,
        timings: Dict[AutotuneConfig, float],
        *,
        is_writer: bool = True,
    ) -> None:
        """Record a sweep's timings, atomically.

        Args:
            tuning_key: The shape/dtype facts identifying this call.
            timings: Config to milliseconds, as reconciled across ranks.
            is_writer: Whether THIS process should write. Pass ``consensus.rank == 0`` under a
                collective; every rank holds the same reconciled result, so having all of them
                write is redundant and risks a torn read on a shared filesystem.

        Returns:
            None. Write failures are swallowed: a cache that cannot be written must not break a run
            that is otherwise fine.
        """
        if not self.enabled or not is_writer or not timings:
            return
        path = self._path(tuning_key, list(timings))
        if path is None:
            return
        payload = {
            "schema": _SCHEMA,
            "kernel": self.kernel,
            "key": repr(tuning_key),
            "device": _device_tag(),
            "timings": {repr(c.key()): t for c, t in timings.items()},
            "readable": {str(c): t for c, t in timings.items()},
        }
        try:
            # `mkstemp` already creates 0600 -- it is documented to, and this relies on it rather
            # than re-chmod'ing, so the file is never briefly wider. The mode therefore survives
            # the rename onto the published name.
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
            os.replace(tmp, path)  # atomic: a concurrent reader sees old or new, never partial
        except OSError:
            pass

    @staticmethod
    def describe_env() -> List[str]:
        """The environment variables that steer this cache, for a diagnostic banner.

        Returns:
            Human-readable lines. Kept here so the names live beside the code that reads them --
            a half-renamed env var is how the upstream cache became silently inert.
        """
        return [
            "CPO_AUTOTUNE_CACHE=0        DISABLE the on-disk result cache (default ON, as main)",
            "CPO_AUTOTUNE_CACHE_DIR=...  where to store it (default $CPO_HOME/.fold_cp_ops/autotune)",
            "CPO_AUTOTUNE_REFRESH=1      ignore stored results and re-measure",
            "CPO_AUTOTUNE_VERBOSE=1      print each config's timing",
        ]
