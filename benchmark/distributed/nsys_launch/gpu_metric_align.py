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
import bisect

c = sqlite3.connect(sys.argv[1])
# Build metricId -> name from TARGET_INFO_GPU_METRICS (positional: small-int col = id, str col = name).
rows = c.execute("SELECT * FROM TARGET_INFO_GPU_METRICS").fetchall()
id2name = {}
for r in rows:
    ints = [(i, v) for i, v in enumerate(r) if isinstance(v, int) and 0 <= v <= 40]
    strs = [v for v in r if isinstance(v, str) and v]
    if ints and strs:
        id2name.setdefault(ints[-1][1], strs[-1])
# confirm the NVLink/PCIe ids
print("relevant metric ids:")
for mid, nm in sorted(id2name.items()):
    if "NVLink" in nm and "User" in nm or "PCIe" in nm and "Throughput" in nm:
        print(f"  {mid:2d}  {nm}")
WANT = {"NVLtx": 25, "NVLrx": 21, "PCIetx": 29, "PCIerx": 28}


def windows(pat):
    return sorted(c.execute("SELECT e.start,e.end FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId=s.id "
                            "WHERE COALESCE(e.text,s.value) LIKE ?", (pat,)).fetchall())


# GPU_METRICS: (rawTimestamp, timestamp, typeId, metricId, value); use the PROJECTED 'timestamp'.
data = {}
for k, mid in WANT.items():
    r = c.execute("SELECT timestamp,value FROM GPU_METRICS WHERE metricId=?", (mid,)).fetchall()
    r.sort()
    data[k] = r
print("sample counts: " + ", ".join(f"{k}={len(v)}" for k, v in data.items()))


def peak(rows, wins):
    starts = [r[0] for r in rows]
    m = 0
    for a, b in wins:
        i = bisect.bisect_left(starts, a)
        while i < len(starts) and starts[i] <= b:
            if rows[i][1] > m:
                m = rows[i][1]
            i += 1
    return m


print(f"\n{'phase':40}{'NVLtx':>10}{'NVLrx':>10}{'PCIetx':>10}{'PCIerx':>10}  (raw max)")
for lab in ['fused[N4000_D256_cp8_outgoing]', 'dtensor_baseline_ring_reducescatter[N4000_D256_cp8_outgoing]',
            'fused[N4000_D256_cp8_incoming]', 'dtensor_baseline_ring_reducescatter[N4000_D256_cp8_incoming]',
            'fused[N2000_D256_cp8_outgoing]', 'dtensor_baseline_ring_reducescatter[N2000_D256_cp8_outgoing]']:
    w = windows(lab)
    if not w:
        print(f"{lab:40} NO WINDOW")
        continue
    print(f"{lab:40}{peak(data['NVLtx'], w):>10.0f}{peak(data['NVLrx'], w):>10.0f}"
          f"{peak(data['PCIetx'], w):>10.0f}{peak(data['PCIerx'], w):>10.0f}")
