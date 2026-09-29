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


"""Assemble profiling/distributed/trimul/summary.csv from the nsys_trimul_dtensor.py sweep.

Globs the per-cell driver consoles (``console_D*_cp*.log`` — the ``[LAT]``/``[NA]`` lines) + the matching
``trimul_<gpu>_<nvlink>_D<D>_cp<cp>.sqlite`` (nsys-export of the .nsys-rep), extracts BOTH the fused[...]
and dtensor[...] NVTX-range GPU metrics (Tensor Active % = HMMA, NVLink TX/RX User-Data Throughput %), and
writes one CSV row per (N, D, cp, direction). OVERWRITES the summary. GPU-metric %s are Throughput-% of the
nsys internal peak (GB/s = % * H200-NVLink-peak); see docs/trimul_nvshmem_design.md §5.5 for the caveats
(short-kernel under-sampling; DTensor dispatch-bound).

Run (after the sweep):  python benchmark/distributed/assemble_all.py \
    --console-dir <dir with console_*.log> --outdir profiling/distributed/trimul --gpu H200 --nvlink NV18
"""

import argparse
import bisect
import glob
import os
import re
import sqlite3
from collections import defaultdict

LAT_RE = re.compile(
    r"\[LAT\] D=(\d+) cp=(\S+) N=(\d+) dir=(\w+) fused_ms=([\d.]+) \w+_ms=([\d.]+) speedup=([\d.]+)"
)
NA_RE = re.compile(r"\[NA\] D=(\d+) cp=(\S+) N=(\d+) dir=(\w+) reason=(.+)")
TAG_RE = re.compile(r"(fused|dtensor)\[N(\d+)_D(\d+)_cp(\S+?)_(\w+)\]")
WANT = {
    "Tensor Active [Throughput %]": "hmma",
    "NVLink TX Requests User Data [Throughput %]": "tx",
    "NVLink TX Responses User Data [Throughput %]": "tx",
    "NVLink RX Requests User Data [Throughput %]": "rx",
    "NVLink RX Responses User Data [Throughput %]": "rx",
}


def _parse_console(path):
    lat, na = {}, {}
    with open(path) as f:
        for line in f:
            m = LAT_RE.search(line)
            if m:
                D, cp, N, d, fm, dm, sp = m.groups()
                lat[(int(N), int(D), cp, d)] = (float(fm), float(dm), float(sp))
            m = NA_RE.search(line)
            if m:
                D, cp, N, d, reason = m.groups()
                na[(int(N), int(D), cp, d)] = reason.strip()
    return lat, na


def _extract_metrics(sqlite_path):
    """{(path,N,D,cp,dir): (hmma,tx,rx)} for fused AND dtensor NVTX ranges (avg over instances+GPUs)."""
    out = {}
    if not os.path.exists(sqlite_path):
        return out
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    name_by_mid = {
        mid: nm
        for mid, nm in cur.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS")
    }
    mid2grp = {mid: WANT[nm] for mid, nm in name_by_mid.items() if nm in WANT}
    grp_mids = defaultdict(list)
    for mid, grp in mid2grp.items():
        grp_mids[grp].append(mid)
    samples = defaultdict(list)
    if mid2grp:
        for ts, mid, val in cur.execute(
            "SELECT timestamp, metricId, value FROM GPU_METRICS WHERE metricId IN (%s)"
            % ",".join(str(m) for m in mid2grp)
        ):
            samples[mid].append((ts, val))
        for mid in samples:
            samples[mid].sort()
    ranges = defaultdict(list)
    for text, start, end in cur.execute(
        "SELECT text, start, end FROM NVTX_EVENTS WHERE (text LIKE 'fused[%' OR text LIKE 'dtensor[%')"
    ):
        if end is not None:
            ranges[text].append((start, end))

    def avg_in(mid, windows):
        ts_list = [t for t, _ in samples.get(mid, [])]
        vals = [v for _, v in samples.get(mid, [])]
        acc, n = 0.0, 0
        for s, e in windows:
            lo, hi = bisect.bisect_left(ts_list, s), bisect.bisect_right(ts_list, e)
            for k in range(lo, hi):
                acc += vals[k]
                n += 1
        return acc / n if n else 0.0

    for tag, windows in ranges.items():
        m = TAG_RE.match(tag)
        if not m:
            continue
        path, N, D, cp, d = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), m.group(5)
        out[(path, N, D, cp, d)] = (
            sum(avg_in(mid, windows) for mid in grp_mids["hmma"]),
            sum(avg_in(mid, windows) for mid in grp_mids["tx"]),
            sum(avg_in(mid, windows) for mid in grp_mids["rx"]),
        )
    con.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--console-dir", default="profiling/distributed/trimul")
    ap.add_argument("--outdir", default="profiling/distributed/trimul")
    ap.add_argument("--gpu", default="H200")
    ap.add_argument("--nvlink", default="NV18")
    args = ap.parse_args()
    summary = os.path.join(args.outdir, "summary.csv")

    rows, metrics = {}, {}
    for console in sorted(glob.glob(os.path.join(args.console_dir, "console_D*_cp*.log"))):
        base = os.path.basename(console)[len("console_") : -len(".log")]  # D<D>_cp<label>
        lat, na = _parse_console(console)
        sqlite_path = os.path.join(args.outdir, f"trimul_{args.gpu}_{args.nvlink}_{base}.sqlite")
        metrics.update(_extract_metrics(sqlite_path))
        for k, v in lat.items():
            rows[k] = ("OK", v)
        for k, r in na.items():
            rows[k] = ("NA(" + r + ")", None)

    def cp_key(cp):
        return (len(cp), cp)

    with open(summary, "w") as f:
        f.write(
            "gpu,nvlink,N,D,cp,direction,status,fused_ms,dtensor_ms,speedup,"
            "fused_hmma_pct,fused_nvlink_tx_pct,fused_nvlink_rx_pct,"
            "dtensor_hmma_pct,dtensor_nvlink_tx_pct,dtensor_nvlink_rx_pct\n"
        )
        for N, D, cp, d in sorted(rows, key=lambda k: (cp_key(k[2]), k[1], k[0], k[3])):
            status, v = rows[(N, D, cp, d)]
            if v is None:
                f.write(f"{args.gpu},{args.nvlink},{N},{D},{cp},{d},{status},,,,,,,,,\n")
                continue
            fm, dm, sp = v
            fh, ftx, frx = metrics.get(("fused", N, D, cp, d), (0.0, 0.0, 0.0))
            dh, dtx, drx = metrics.get(("dtensor_baseline_a2a", N, D, cp, d), (0.0, 0.0, 0.0))
            f.write(
                f"{args.gpu},{args.nvlink},{N},{D},{cp},{d},{status},{fm:.4f},{dm:.4f},{sp:.3f},"
                f"{fh:.2f},{ftx:.2f},{frx:.2f},{dh:.2f},{dtx:.2f},{drx:.2f}\n"
            )
    n_ok = sum(1 for s, v in rows.values() if v is not None)
    print(
        f"[assemble_all] {len(rows)} rows ({n_ok} OK, {len(rows) - n_ok} NA) -> {summary}",
        flush=True,
    )


if __name__ == "__main__":
    main()
