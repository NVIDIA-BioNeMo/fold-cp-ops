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

# --- CLOCK SANITY: do NIC metrics and NVTX/kernel share the same timeline? ---
nv = c.execute("SELECT min(start),max(end) FROM NVTX_EVENTS").fetchone()
ni = c.execute("SELECT min(start),max(end) FROM NET_NIC_METRIC").fetchone()
kn = c.execute("SELECT min(start),max(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
print(f"NVTX   span ns: {nv[0]:>18,} .. {nv[1]:>18,}")
print(f"KERNEL span ns: {kn[0]:>18,} .. {kn[1]:>18,}")
print(f"NIC    span ns: {ni[0]:>18,} .. {ni[1]:>18,}")
lo, hi = max(nv[0], ni[0], kn[0]), min(nv[1], ni[1], kn[1])
print(f"3-way overlap : {lo:>18,} .. {hi:>18,}  ({'SAME clock' if hi > lo else 'DISJOINT -> different clock!'})")


def windows(pat):
    return c.execute(
        "SELECT e.start,e.end FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId=s.id "
        "WHERE COALESCE(e.text,s.value) LIKE ?", (pat,)).fetchall()


def mid(name):
    return c.execute("SELECT metricsListId,metricsIdx FROM TARGET_INFO_NETWORK_METRICS WHERE name=?",
                     (name,)).fetchone()


txrows = c.execute("SELECT start,value FROM NET_NIC_METRIC WHERE metricsListId=? AND metricsIdx=?", mid('IB: Bytes sent')).fetchall()
rxrows = c.execute("SELECT start,value FROM NET_NIC_METRIC WHERE metricsListId=? AND metricsIdx=?", mid('IB: Bytes received')).fetchall()
txrows.sort()
rxrows.sort()


def peak(rows, wins):
    import bisect
    starts = [r[0] for r in rows]
    m = 0
    for a, b in wins:
        i = bisect.bisect_left(starts, a)
        while i < len(starts) and starts[i] <= b:
            if rows[i][1] > m:
                m = rows[i][1]
            i += 1
    return m


print(f"\n{'NVTX phase':42} {'#win':>5} {'span_ms':>8} {'IBtx_peak':>14} {'IBrx_peak':>14}  (B/ms)")
for lab in ['fused[N4096_D256_cp16_incoming]', 'fused[N4096_D256_cp16_outgoing]',
            'dtensor_baseline_ring_reducescatter[N4096_D256_cp16_incoming]', 'dtensor_baseline_ring_reducescatter[N4096_D256_cp16_outgoing]',
            'fused[N2000_D256_cp16_incoming]', 'fused[N2000_D256_cp16_outgoing]',
            'dtensor_baseline_ring_reducescatter[N2000_D256_cp16_incoming]', 'dtensor_baseline_ring_reducescatter[N2000_D256_cp16_outgoing]']:
    w = windows(lab)
    if not w:
        print(f"{lab:42} {'--':>5}  NO WINDOW")
        continue
    span = sum(b - a for a, b in w) / 1e6
    print(f"{lab:42} {len(w):>5} {span:>8.1f} {peak(txrows, w):>14,.0f} {peak(rxrows, w):>14,.0f}")
