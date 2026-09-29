# Multi-rank / multi-node nsys profiling (all GPUs, multi-N, dynamic-compile capture)

Profile the distributed TriMulAutotuned e2e so **every rank's GPU timeline** shows up in the report(s),
with **multiple N_token in one report** and the **dynamic compilation phase** captured. Pipecleaned on
A pyxis container site (validated single-node 8 GPU + 2-node 16 GPU).

## The recipe (why it works)

One `nsys` session **per node** wrapping that node's `torchrun --nproc_per_node=8`. nsys follows all 8
child ranks under a **single CUPTI subscriber** → all 8 GPU timelines in ONE report per node, no clash.

> The old rank-0-only reports came from `srun --ntasks-per-node=8` (8 separate tasks) + an
> `[ $SLURM_PROCID = 0 ]` nsys guard. Running **separate** nsys sessions per rank on one node clashes on
> the CUPTI kernel-activity subscriber (all but one report comes back EMPTY) — that is why it was rank-0
> only. `--ntasks-per-node=1` + one nsys wrapping torchrun avoids it entirely.

- **Single-node** (`NNODES=1`): one report, 8 GPUs.
- **Multi-node** (`NNODES=2`, NVLink-intra + IB-inter): one report **per node** (`_n0`, `_n1`) = all 16
  GPUs across 2 files. There is **NO single merged cross-node `.nsys-rep`** (confirmed by NVIDIA:
  `nsys export` only converts format). View the per-node reports together via the nsys GUI
  **multi-report view** (File → New multi-report view; auto-time-aligned) or the CLI `nsys recipe`.

## Usage (container site)

```bash
# single-node, 8 GPU, multi-N (1024 + 2048 in one report), compile captured:
bash submit_nsys_prof.sh mytag 1 "1024,2048" TM_CP=8 TM_D=256 TM_BASELINE=none \
     TM_DIR=outgoing TM_DYNAMIC=1 CAPTURE=full TM_WARMUP=6 TM_ITERS=3
# 2-node, 16 GPU (cp16):
bash submit_nsys_prof.sh mytag 2 "1024,2048" TM_CP=16 TM_D=256 TM_BASELINE=none \
     TM_DIR=outgoing TM_DYNAMIC=1 CAPTURE=full TM_WARMUP=6 TM_ITERS=3
```

Knobs (`nsys_prof_multinode.sbatch`): `NNODES NPROC | TM_D TM_NS(comma-list) TM_CP TM_DIR TM_BASELINE
TM_TAG | GPU_METRICS TM_ROUTE2NI TM_DYNAMIC CAPTURE TM_WARMUP TM_ITERS`.

- **Multi-N in one report**: `TM_NS="1024,2048,4096"` — the merged driver loops per-N (build/free, peak =
  one N) with per-N NVTX `N{N}_D{D}_cp{cp}_{dir}`. **Requires `TM_DYNAMIC=1`** (static bakes ONE shape).
- **`CAPTURE`**: `iters` = `--capture-range=cudaProfilerApi` (profiled iters only, small report). `full`
  = whole process, so the `ftm_construct_compile[dynamic]` NVTX (the cute.compile phase) IS captured.
- `verify_report.py <report.sqlite>` prints distinct GPU device IDs + N-token tags + the compile NVTX +
  GPU-metrics rows (the sbatch self-verifies each node's report at the end).

## GPU metrics — host-admin gated (currently BLOCKED on the container site)

`GPU_METRICS=1` adds `--gpu-metrics-devices=all`, but that site's host driver has
`RmProfilingAdminOnly=1` and the enroot container is namespaced-root **without host `CAP_SYS_ADMIN`** →
`ERR_NVGPUCTRPERM` ("Insufficient privilege", zero GPUs qualify). To enable, a **cluster admin** must set
`NVreg_RestrictProfilingToAdminUsers=0` on the host driver (modprobe param + reload/reboot) **or** the job
must run with real `CAP_SYS_ADMIN` / a privileged container. The `GPU_METRICS=1` toggle is ready for when
that is enabled. (CPU sampling is likewise gated by `perf_event_paranoid=4` — we use `--sample=none`.)

## Bare-metal variant — where GPU metrics DO work

A bare-metal DGX-H100 site has `RmProfilingAdminOnly=0` and runs **bare-metal conda** (direct ssh, no enroot
container), so `--gpu-metrics-devices=all` works there (verified: GPU_METRICS 97–137M rows — Tensor
Active %, SM Active, NVLink, clocks). Use `dt_nsys_{node,drive}.sh` (same recipe, conda + direct-ssh
2-node instead of srun/pyxis):

```bash
# 2-node cp16 — all 16 GPU + metrics + compile, multi-N in one report/node:
bash dt_nsys_drive.sh mytag 2 "2000,4096" TM_CP=16 GPU_METRICS=1 CAPTURE=full
# single-node cp8:
bash dt_nsys_drive.sh mytag 1 "1000,2000,4000" TM_CP=8 GPU_METRICS=1 CAPTURE=full
# 2-D mesh4x4 (square mesh):
bash dt_nsys_drive.sh mytag 2 "2048,4096" TM_MESH2D=4,4 GPU_METRICS=1 CAPTURE=full
```

Reports land in `$MD/nsys_out/<tag>_n<node>.nsys-rep` (a shared filesystem, so both per-node
reports are visible from either node). Its nsys is **2025.5.2** — it has no `nccl` trace target, so
`--trace=cuda,nvtx` (NCCL still shows via NVTX + cuda). Timing: cp16/mesh4x4 match the container site within ~5%; cp8
is clean on a dedicated idle node (throttle only under contention — use the container site for cp8 absolute ms). The
`profiling/distributed/trimul/hybrid_nvlink-IB/nsys_devtools_*_allgpu_metrics_*.nsys-rep` were collected
this way (replacing the old rank-0-only `*_r0` form).
