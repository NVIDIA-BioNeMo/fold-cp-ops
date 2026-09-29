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

c = sqlite3.connect(sys.argv[1])


def sec(t):
    return t / 1e9


def wins(pat):
    return sorted(c.execute(
        "SELECT e.start,e.end FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId=s.id "
        "WHERE COALESCE(e.text,s.value) LIKE ?", (pat,)).fetchall())


for pat in ['fused[N4096_D256_cp16_incoming]', 'dtensor_baseline_ring_reducescatter[N4096_D256_cp16_incoming]',
            'fused[N4096_D256_cp16_outgoing]', 'dtensor_baseline_ring_reducescatter[N4096_D256_cp16_outgoing]']:
    w = wins(pat)
    if w:
        print(f"{pat:34} windows {sec(w[0][0]):7.2f}s .. {sec(w[-1][1]):7.2f}s")
tx = c.execute("SELECT metricsListId,metricsIdx FROM TARGET_INFO_NETWORK_METRICS WHERE name='IB: Bytes sent'").fetchone()
act = c.execute("SELECT min(start),max(start) FROM NET_NIC_METRIC WHERE metricsListId=? AND metricsIdx=? AND value>1000000", tx).fetchone()
if act[0]:
    print(f"{'>>> IB-tx ACTIVE (>1MB/ms) span':34} {sec(act[0]):7.2f}s .. {sec(act[1]):7.2f}s")
top = c.execute("SELECT start,value FROM NET_NIC_METRIC WHERE metricsListId=? AND metricsIdx=? ORDER BY value DESC LIMIT 3", tx).fetchall()
print("    top IB-tx samples (t_sec, B/ms):", [(round(sec(s), 2), int(v)) for s, v in top])
