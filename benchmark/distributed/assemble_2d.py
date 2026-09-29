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


"""B12 2-D incoming no-copy summary assembler — emits rows in the COMMITTED summary.csv schema.

Parses the 2-D incoming ``console_D*_cp*x*.log`` ([LAT]/[NA]) + the matching 2-D .sqlite (fused[...] GPU
metrics), pivots outgoing+incoming per (cp,N,D), and emits one row per cell in the SAME distilled schema
as the committed 1-D rows:

  gpu,nvlink,cp,N,D,status,outgoing_ms,incoming_ms,ratio_in_over_out,dtensor_ms,outgoing_speedup,
  incoming_speedup,fused_in_hmma_pct,fused_in_nvlink_pct

cp label is "2x2"/"2x4"/"4x2". ``ratio_in_over_out`` = incoming_ms/outgoing_ms (the copy-tax; the no-copy
store collapses it to ~1.0). ``dtensor_ms``/speedups are filled from the WITH-dtensor cells (D256) and NA
where the run was ``--no-dtensor`` (nan dtensor_ms in [LAT]) — precedented (committed 1-D D512/cp2 rows are
NA-dtensor too). ``fused_in_*`` are the INCOMING fused NVTX range's Tensor-Active% and (TX+RX)/2 NVLink%.
The ``_plain`` (route2_ni-OFF) cells are SKIPPED here (they feed the separate in/out OFF baseline +
kernel-breakdown copies-gone proof).

Run:  <py> benchmark/distributed/assemble_2d.py --dir profiling/distributed/trimul --gpu H200 --nvlink NV18
"""

import argparse
import csv
import glob
import math
import os
import re

from assemble_all import _extract_metrics  # reuse the §5.5 sqlite GPU-metrics extractor

# [LAT] tolerating dtensor_ms=nan (the --no-dtensor cells). fused_ms is always numeric.
_LAT = re.compile(r"\[LAT\] D=(\d+) cp=(\S+) N=(\d+) dir=(\w+) fused_ms=([\d.]+) \w+_ms=(\S+)")
_NA = re.compile(r"\[NA\] D=(\d+) cp=(\S+) N=(\d+) dir=(\w+) reason=(.+)")


def _f(x):
    try:
        v = float(x)
        return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


def _parse_console(path):
    lat, na = {}, {}
    with open(path) as f:
        for line in f:
            m = _LAT.search(line)
            if m:
                D, cp, N, d, fm, dm = m.groups()
                lat[(int(N), int(D), cp, d)] = (float(fm), _f(dm))
            m = _NA.search(line)
            if m:
                D, cp, N, d, reason = m.groups()
                na[(int(N), int(D), cp, d)] = reason.strip()
    return lat, na


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="profiling/distributed/trimul")
    ap.add_argument("--gpu", default="H200")
    ap.add_argument("--nvlink", default="NV18")
    ap.add_argument("--out", default=None, help="output CSV (default <dir>/summary_2d.csv)")
    args = ap.parse_args()
    out_csv = args.out or os.path.join(args.dir, "summary_2d.csv")

    lat, na, metrics = {}, {}, {}
    # 2-D cells only: cp label contains 'x' (console_D<D>_cp<cp0>x<cp1>.log). Skip _plain (OFF) cells.
    for console in sorted(glob.glob(os.path.join(args.dir, "console_D*_cp*x*.log"))):
        base = os.path.basename(console)[len("console_") : -len(".log")]  # D<D>_cp<cp0>x<cp1>[tag]
        if base.endswith("_plain"):
            continue
        l, n = _parse_console(console)
        lat.update(l)
        na.update(n)
        sq = os.path.join(args.dir, f"trimul_{args.gpu}_{args.nvlink}_{base}.sqlite")
        metrics.update(_extract_metrics(sq))

    cells = sorted(
        {(N, D, cp) for (N, D, cp, d) in lat} | {(N, D, cp) for (N, D, cp, d) in na},
        key=lambda k: (k[2], k[1], k[0]),
    )
    hdr = [
        "gpu",
        "nvlink",
        "cp",
        "N",
        "D",
        "status",
        "outgoing_ms",
        "incoming_ms",
        "ratio_in_over_out",
        "dtensor_ms",
        "outgoing_speedup",
        "incoming_speedup",
        "fused_in_hmma_pct",
        "fused_in_nvlink_pct",
    ]
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for N, D, cp in cells:
            if (N, D, cp, "incoming") in na:
                w.writerow(
                    [args.gpu, args.nvlink, cp, N, D, "NA(" + na[(N, D, cp, "incoming")] + ")"]
                    + [""] * 8
                )
                continue
            fo = lat.get((N, D, cp, "outgoing"))
            fi = lat.get((N, D, cp, "incoming"))
            if fo is None or fi is None:
                continue
            out_ms, dt_out = fo
            in_ms, dt_in = fi
            ratio = in_ms / out_ms if out_ms else float("nan")
            # single dtensor_ms column (mean of the two directions' dtensor; NA if --no-dtensor);
            # each speedup divides its OWN direction's dtensor for honesty.
            dts = [x for x in (dt_out, dt_in) if x is not None]
            dt_ms = (sum(dts) / len(dts)) if dts else None
            out_sp = (dt_out / out_ms) if (dt_out and out_ms) else None
            in_sp = (dt_in / in_ms) if (dt_in and in_ms) else None
            hmma, tx, rx = metrics.get(("fused", N, D, cp, "incoming"), (0.0, 0.0, 0.0))
            g = lambda x: f"{x:.4f}" if x is not None else "NA"
            gp = lambda x: f"{x:.2f}" if x is not None else "NA"
            w.writerow(
                [
                    args.gpu,
                    args.nvlink,
                    cp,
                    N,
                    D,
                    "OK",
                    f"{out_ms:.4f}",
                    f"{in_ms:.4f}",
                    f"{ratio:.3f}",
                    g(dt_ms),
                    gp(out_sp),
                    gp(in_sp),
                    f"{hmma:.2f}",
                    f"{(tx + rx) / 2:.2f}",
                ]
            )
    print(f"[assemble_2d] {len(cells)} cells -> {out_csv}", flush=True)
    with open(out_csv) as f:
        print(f.read(), flush=True)


if __name__ == "__main__":
    main()
