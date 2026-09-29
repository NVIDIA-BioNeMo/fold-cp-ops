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

import sqlite3
import sys

db = sys.argv[1]
c = sqlite3.connect(db)
tabs = [r[0] for r in c.execute("select name from sqlite_master where type='table'").fetchall()]
gpus = []
if "CUPTI_ACTIVITY_KIND_KERNEL" in tabs:
    gpus = [r[0] for r in c.execute("select distinct deviceId from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()]
nvtx = []
if "NVTX_EVENTS" in tabs:
    nvtx = [r[0] for r in c.execute("select distinct text from NVTX_EVENTS where text is not null").fetchall()]
ntags = sorted({t.split("[")[-1].rstrip("]") for t in nvtx if t and ("N1024" in t or "N2048" in t or "N4096" in t)})
gm = c.execute("select count(*) from GPU_METRICS").fetchone()[0] if "GPU_METRICS" in tabs else 0
print(f"DISTINCT_GPU_DEVICE_IDS={sorted(gpus)} (count={len(gpus)})")
print(f"N_DISTINCT_NVTX={len(nvtx)}  N_TOKEN_TAGS={ntags}")
print(f"NVTX_SAMPLE={[t for t in nvtx if t and 'fused' in t][:6]}")
print(f"COMPILE_NVTX={[t for t in nvtx if t and 'construct_compile' in t][:4]}")
print(f"GPU_METRICS_ROWS={gm}")
