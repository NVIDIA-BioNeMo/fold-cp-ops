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
# per-NIC: how many metric samples + peak IB throughput. Distinguishes:
#   NIC absent from result  -> NO rows collected (collection failure)
#   NIC present, maxv == 0   -> collected but ZERO traffic (routing/rail-mapping)
#   NIC present, maxv  > 0   -> collected + carried traffic
q = """
SELECT ni.name AS nic,
       SUM(CASE WHEN nm.name='IB: Bytes sent'     THEN 1 ELSE 0 END) AS ib_tx_rows,
       MAX(CASE WHEN nm.name='IB: Bytes sent'     THEN m.value ELSE 0 END) AS ib_tx_max,
       MAX(CASE WHEN nm.name='IB: Bytes received' THEN m.value ELSE 0 END) AS ib_rx_max,
       MAX(CASE WHEN nm.name LIKE 'IPoIB: Bytes%'  THEN m.value ELSE 0 END) AS ipoib_max,
       COUNT(*) AS total_rows
FROM NET_NIC_METRIC m
JOIN NIC_ID_MAP idm ON m.globalId = idm.globalId
JOIN TARGET_INFO_NIC_INFO ni ON idm.nicId = ni.nicId
JOIN TARGET_INFO_NETWORK_METRICS nm ON m.metricsListId=nm.metricsListId AND m.metricsIdx=nm.metricsIdx
GROUP BY ni.name ORDER BY ni.name
"""
print(f"{'NIC':10} {'IB_tx_rows':>10} {'IB_tx_max[B/ms]':>16} {'IB_rx_max[B/ms]':>16} {'IPoIB_max':>12} {'rows':>9}")
for nic, txr, txm, rxm, ipm, tot in c.execute(q).fetchall():
    print(f"{nic:10} {txr:10d} {txm:16.0f} {rxm:16.0f} {ipm:12.0f} {tot:9d}")
# also: total distinct NICs that appear at all
print("distinct NICs with ANY NET_NIC_METRIC rows:",
      c.execute("SELECT COUNT(DISTINCT globalId) FROM NET_NIC_METRIC").fetchone()[0])
