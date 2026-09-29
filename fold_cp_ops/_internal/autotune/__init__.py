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

"""Measurement-driven config selection, safe for kernels that contain a collective.

Six modules, each owning one decision, so that the distributed case is a property of the
composition rather than a special case bolted onto a local tuner:

* `axes` -- DECLARED tunable axes, which generate the pool (preferred over a flat list).
* `config` -- one candidate, and the canonical identity everything else keys on.
* `space` -- the candidate pool, with VALIDITY (may reject; must be pure) kept apart from
  PREFERENCE (may only reorder).
* `timing` -- how a candidate is measured; refuses the adaptive timer that desynchronizes ranks.
* `consensus` -- how ranks agree on one winner. A no-op in a single process.
* `cache` -- the on-disk record of a sweep, keyed on the device as well as the shape.
* `precompile` -- parallel compilation, parent and worker in one file.
* `tuner` -- the sequencing, and the `@autotune` decorator.

Import the decorator from here; the modules are an implementation detail::

    from fold_cp_ops._internal.autotune import autotune, AutotuneConfig

Environment (all ``CPO_*``; the upstream harness read ``FOLD_CP_OPS_*``, which nothing in this repo
sets, so its disk cache never engaged):

* ``CPO_AUTOTUNE=0`` -- do not measure; run each kernel's first admissible config.
* ``CPO_AUTOTUNE_VERBOSE=1`` -- print each candidate's timing and each failure.
* ``CPO_AUTOTUNE_CACHE=0`` -- do NOT persist sweep results across processes. Persisting is
  the default, matching `main`; without it every process re-sweeps and may elect a
  different config, which is how an intermittent perf regression gets in.
* ``CPO_AUTOTUNE_CACHE_DIR`` -- where; defaults under ``$CPO_HOME``.
* ``CPO_AUTOTUNE_REFRESH=1`` -- ignore stored results and re-measure.
* ``CPO_AUTOTUNE_WORKERS`` -- pre-compile worker count (default 8).
"""

from fold_cp_ops._internal.autotune.axes import AxisSpace, TuneAxis
from fold_cp_ops._internal.autotune.cache import ResultCache
from fold_cp_ops._internal.autotune.config import AutotuneConfig
from fold_cp_ops._internal.autotune.consensus import Consensus
from fold_cp_ops._internal.autotune.space import ConfigSpace
from fold_cp_ops._internal.autotune.timing import TimingPolicy
from fold_cp_ops._internal.autotune.tuner import Autotuner, autotune, enabled, freeze

__all__ = [
    "AxisSpace",
    "Autotuner",
    "AutotuneConfig",
    "TuneAxis",
    "ConfigSpace",
    "Consensus",
    "ResultCache",
    "TimingPolicy",
    "autotune",
    "enabled",
    "freeze",
]
