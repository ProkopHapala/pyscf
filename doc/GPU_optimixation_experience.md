---
type: ExperienceReport
title: GPU XC Optimization — Lessons Learned
description: Strategies, caveats, disproved assumptions, and generalizations from OpenCL small-molecule DFT XC optimization (benzene benchmark arc)
tags: [opencl, dft, xc, gpu, optimization, profiling, lessons]
timestamp: 2026-07-08
---

Distilled from the benzene XC optimization arc (OTF → hybrid radial vmat → split-K → tile sweep). Numbers and tables: [`doc/GPU_benchmark.md`](/home/prokop/git/pyscf/doc/GPU_benchmark.md). Kernel analysis: [`doc/GPU_smallDFT_optimization.chat.md`](/home/prokop/git/pyscf/doc/GPU_smallDFT_optimization.chat.md). Workflow: [`.cursor/skills/gpu-optimize/SKILL.md`](/home/prokop/git/pyscf/.cursor/skills/gpu-optimize/SKILL.md).

**Problem class:** grid integration over AO basis — per SCF cycle compute ρ, XC potential, and vmat on ~10⁵ grid points × ~10² AOs × few atoms. Same structural pattern as many PDE/stencil codes: **outer parallelism choice** dominates GPU saturation on small systems.

---

## Outcome arc (benzene, ccpvdz, grid 3, RTX 3090)

| Stage of work | Per-cycle gpu CL | Main lever |
|---------------|----------------:|------------|
| CPU libxc reference | — (~450 ms host) | baseline |
| OTF cubic (ρ + vmat tiled) | ~28 ms | GPU port |
| Hybrid OTF ρ + radial vmat | ~21 ms | **precompute in setup, gather in hot loop** |
| Split-K radial vmat + tile tune | ~14 ms | **shard serial dimension + 1-neighborhood sweep** |
| Screened radial + split-K + zeroing removal | **~10 ms** | **screening (asymptotic work) + split-K on CSR gTile list + dead-write elimination** |

vmat went from ~23 ms → ~16 ms → ~7 ms → **~6 ms**. ρ went from ~5 ms → ~3 ms with screening. Session details: `doc/GPU_screened_splitk_2026-07-17.md`.

### PTCDA (6-31g, grid 2, RTX 3090)

| Stage of work | Per-cycle gpu CL | Main lever |
|---------------|----------------:|------------|
| OTF cubic | ~314 ms | baseline GPU |
| OTF split-K | ~307 ms | split-K on dense grid |
| Screened radial (no split-K) | ~94 ms | **screening: ~5.8× fewer active pairs** |
| Screened split-K + zeroing removal | **~73 ms** | **split-K on CSR gTile list + dead-write elimination** |

PTCDA vmat: 194 ms → 188 ms (OTF splitK) → 67 ms (screened) → **45 ms** (screened splitK + zeroing removal).

---

## Strategies that worked

### 1. Profile before optimizing — and profile correctly

- Use **`queue.finish()` + wall clock** for honest end-to-end stage time.
- Use **`clGetEventProfilingInfo`** on kernel events for device time (`gpu_*_cl`).
- When wall ≈ CL (±0.1–0.4 ms), stage breakdown is trustworthy; optimize the largest stage first.
- One timed call per path is enough for ranking; sweeps need **warmup + min of N runs** (variance ~1 ms on benzene).

### 2. Structural changes beat micro-optimizations

Ordered by impact on this codebase:

| Tier | Change | Why it worked |
|------|--------|---------------|
| **Parallelism remap** | Split-K vmat (grid shards + reduce) | Fixed **too few long-running WGs** on small molecules |
| **Reuse / precompute** | Radial `R,dR` at setup; gather in vmat | Removed **Hermite spline arithmetic** from hot loop (~7 ms) |
| **Hybrid pipeline** | OTF ρ + radial vmat (different best kernel per stage) | Stages are independent after ρ/wv on device |
| **Tile / launch tuning** | `WGS=128`, `splits=64` for split-K only | Fixed **idle lanes** in pair fill; not a algorithm change |
| **Device residency** | GPU PBE + GPU nelec/exc reduction | Eliminated small but annoying PCIe sync (stage was already tiny) |

Micro-opts (vectorize inner dot, `native_sqrt`, branch removal) were **deferred** until structure was right — correct prioritization per `gpu-optimize` decision tree.

### 3. Match outer parallelism to problem size

ρ used **grid-outer** tiling → thousands of WGs. vmat tiled used **atom-outer** → 36 WGs on benzene, each serial over ~2000 grid tiles. That asymmetry was the root cause of vmat dominance, not “slow arithmetic.”

**General pattern:** for small systems, ask *which loop is outer in the launch grid?* before tuning inner loops.

### 4. Setup-time work when geometry is fixed

Grid coords and basis don't change during SCF → **precompute once**:

- Hermite tables (OTF, small)
- Radial values `R,dR[ir,g]` (~62 MB GGA, benzene)
- Kernel compile with `TileConfig`

Per-cycle cost dropped; setup cost (200–600 ms) amortizes over SCF iterations.

### 5. Split-K when one dimension serializes

When a kernel must accumulate a global object (here `vmat[ncart,ncart]`) and the natural outer loop is too coarse:

1. Launch **partial accumulators** per shard (grid splits).
2. **Reduce** on device (cheap here: ~0.01 ms).

Same idiom as split-K GEMM, histogram privatization, and map-reduce on GPUs.

### 6. Tile sweep: 1-neighborhood coordinate descent

Tile knobs `{NPTILE, NATILE, WGS_VMAT, splits}` are **power-of-2 coupled**:

- Constraint: `WGS ≥ NPTILE × NATILE`
- Larger tiles ↔ more `__local` / lane pressure

**Do not** brute-force the full Cartesian product. From center config `c`:

- **Axis neighbors:** one parameter ± one step (×2 / ÷2)
- **Diagonal neighbors:** two parameters ± one step, often opposite (e.g. `NPTILE×2, WGS÷2`)

Pick best parity-OK neighbor → recenter → repeat. Implementation: `expamples_prokop/sweep_splitk_tiles.py --neighbor`.

### 7. Profile-specific compile flags

`WGS_VMAT=128` won for **split-K pair vmat** but **regressed OTF tiled vmat ~2×** when set globally. Solution: `_ensure_splitk_tile_config()` recompiles only for split-K profiles.

**Rule:** compile-time flags tied to **kernel family**, not one global “best.”

### 8. Add variants; don't replace

New kernels (`splitk`, radial pair) and profiles were **additive**. Old paths remain for parity and A/B. Enables bisection when something breaks.

### 9. Parity gate on every benchmark point

`|vxc − vxc_CPU| < 1e-4` (benzene ~3e-5). Faster wrong is worthless. Run parity before trusting timing sweeps.

---

## Caveats and pitfalls

| Pitfall | What happened | Mitigation |
|---------|---------------|------------|
| **Async queue without drain** | Wall time under-reported GPU work | `queue.finish()` before/after timed region |
| **Python GC + OpenCL buffers** | `INVALID_MEM_OBJECT` when only kernel args held refs | Retain buffers in `plan.otf` / plan object |
| **Global tile default** | One path's win broke another | Profile-scoped `init_device(tile_config=...)` |
| **Single timing sample** | ±1 ms swing on ~12 ms total | Warmup + min of 3 in sweeps |
| **Invalid lattice points** | `WGS=64` from `(64,2,128,*)` fails constraint | Skip or use coupled multi-hop moves |
| **GEMM chains** | `*_cl` = wall for dm→cart (no per-GEMM events) | Wall still valid; don't over-interpret CL sub-split |
| **Benzene-only tuning** | `(64,2,128)` may not transfer | Re-run `--neighbor` on H₂O, dimers before locking |
| **Compiler / first-run noise** | Cold compile, outlier runs (e.g. gpuCL 27 ms) | Discard obvious outliers; report min-of-N |

---

## Rules of thumb

### Profiling

1. If you haven't verified **wall ≈ CL**, you don't know the bottleneck yet.
2. Optimize stages in **descending gpu CL order**; ignore stages < ~5% until the leader is addressed.
3. Report **both** outer wall and CL sum — host overhead mattered little here but can on other paths.

### Parallelism (small molecules)

4. **Count work-groups** before tuning inner loops. If WGs ≪ SM count, remap outer loop or batch systems.
5. **Grid-outer** for point-wise integrals; **atom/pair-outer** only if you can keep grid parallel inside or accept low occupancy.
6. When occupancy is low, prefer **more smaller kernels + reduction** over one giant serial kernel.

### Memory / compute

7. **Gather from precomputed** beats **recompute in inner loop** when precompute amortizes (fixed grid).
8. Don't upload full χ (~262 MB) for small molecules unless measurably faster — radial `R,dR` (~62 MB) was the sweet spot.
9. Keep ρ, wv, vmat on device across PBE — PCIe only for dm in and vmat out (small).

### Tuning

10. Tile parameters: **power-of-2 lattice**, **1-neighborhood descent**, parity at every point.
11. **One structural hypothesis per experiment** — don't change kernel + tiles + splits simultaneously.
12. Document locked defaults in **named profiles** (`gpu_profiles.py`), not scattered env vars.

### Process

13. **KISS / surgical edits** — comment out, don't delete; fail fast, no silent fallbacks.
14. Tests in `expamples_prokop/` + stage driver reproduce the report.

---

## False premises disproved by benchmarks

| Assumption | Expected | Measured | Lesson |
|------------|----------|----------|--------|
| “Half the radial nodes (quintic) → faster cycle” | ρ/vmat speedup | Same table bytes (memory-equivalent `du`); cycle ≈ cubic; setup 2× faster | **Distinguish setup vs cycle**; read design docs before chasing node count |
| “ρ is the bottleneck” | tune Hermite in ρ first | ρ ~5 ms, vmat **15–23 ms** | **Profile stages**; dominant stage was mis-identified without events |
| “Fuse PBE into ρ kernel” | fewer launches | PBE **~0.3 ms** | Don't fuse stages **<5%** of total |
| “Larger WGS always better” | 256 ≥ 128 | WGS=256 **worse** for pair fill (idle lanes) | WGS must match **fill pattern**, not maximize |
| “NATILE=4 improves throughput” | more atoms per tile | **Slower** on benzene (14–15 ms vs ~11–12 ms) | More `__local` / complexity ≠ faster without measurement |
| “Smaller WGS (64, 32) helps split-K” | more WGs | **Not better** than 128 at same tile shape | Coupled to `NPTILE×NATILE`; probe via diagonal moves |
| “Brute-force all tile combos needed” | find optimum | **11 probes** from center found `splits=64` | Coordinate descent on coupled lattice |
| “Coalesced χ path wins on benzene” | better memory | High setup, not production winner | **Problem size** changes optimal layout |
| “Mega-kernel (ρ+PBE+vmat)” | fewer launches | High register/ICache pressure; guidelines deprioritize | Fusion is not free on GPUs |
| “Radial precomp always beats hybrid” | full precomp fastest | Hybrid **marginally faster** (ρ path similar) | **Per-stage** choice beats monolithic path |
| “Global WGS=128 is fine” | one default | OTF tiled vmat **~2× slower** | Compile flags are **path-specific** |

---

## Generalization: similar GPU optimization problems

These patterns apply beyond DFT XC to any **integral / stencil / batched reduction** where system size can be small:

### A. Diagnose saturation before bandwidth

```
If (num_work_groups << num_SMs) AND (work_per_WG is large):
    → Type B: scheduling / parallelism (fix outer loop or split-K)
Else if (memory throughput near roofline):
    → Type A: coalescing, prefetch, compress representation
```

Our vmat was **Type B** on benzene; radial gather addressed **Type A** inside each WG but didn't fix outer geometry until split-K.

### B. Pipeline decomposition with heterogeneous stages

Stages with different optimal data layouts (ρ grid-outer, vmat pair + gather) → **hybrid profiles** beat forcing one representation. Generalize as: *don't require one kernel strategy for the whole pipeline*.

### C. Setup vs steady-state

| When inputs fixed across iterations | Move to setup |
|-------------------------------------|---------------|
| Grid coordinates | radial tables, screening lists |
| Basis / molecule | AO metadata, shell lists |
| Tile geometry | kernel compile |

Optimize **steady-state per-iteration** cost for SCF; report setup separately.

### D. Coupled discrete parameters

When knobs trade occupancy ↔ shared memory ↔ tile size (OpenCL `__local`, CUDA shared mem, WG size):

- Search on a **sparse lattice** (powers of 2)
- Include **diagonal moves** (trade one axis for another)
- Avoid full factorial sweeps

Analogous to autotuning GEMM (MC, KC, NC) but with validity constraints.

### E. Small-system GPU strategy menu

| Strategy | When to use |
|----------|-------------|
| **Remap outer parallelism** | Serial loop inside few WGs |
| **Split-K / privatize + reduce** | Global accumulation from parallel shards |
| **Precompute + gather** | Fixed geometry, expensive eval |
| **Batch replicas** (dimers, conformers, MD) | Can't reshape kernel; many independent systems |
| **GEMM fallback** | When `ncart` small and BLAS-like structure exists |

### F. Optimization order (transferable checklist)

1. Correct profiling (drain queue, stage events, parity)
2. Fix transfers / residency (keep hot data on device)
3. Fix launch count only if launches are measurable %
4. **Remap parallelism / split reductions**
5. Reduce bytes / reuse (precompute, hoist invariants)
6. Access pattern / coalescing
7. Register / shared memory / occupancy
8. Arithmetic micro-opts

We stopped at step 4–5 for major wins; step 8 deliberately not exhausted.

---

## What we still have not solved

- **ρ ~4–5 ms** is ~40% of split-K total — radial-precomp ρ (reuse `R,dR`) is next structural target.
- **Symmetric GGA vmat** (upper-triangle atom pairs) — potential ~2× pair work reduction; needs math care.
- **Cross-molecule validation** of tile defaults.
- **OTF tiled vmat** (~22 ms) if split-K / radial path not used — still grid-serial inside atom WGs.

---

## Quick reference

| Artifact | Role |
|----------|------|
| `expamples_prokop/profile_xc_stages_benzene.py` | Stage timing table |
| `expamples_prokop/sweep_splitk_tiles.py --neighbor` | Tile coordinate descent |
| `pyscf/OpenCL/gpu_profiles.py` | Named production configs |
| `pyscf/OpenCL/gpu_timing.py` | Wall + CL event profiling |
| `pyscf/OpenCL/tile_config.py` | Compile-time tile SSOT |
| `doc/GPU_benchmark.md` | Numbers and sweep tables |
| `doc/opencl_gpu_paths_cookbook.md` | Path / knob compatibility |

**Locked split-K defaults (benzene):** `NPTILE=64`, `NATILE=2`, `WGS_VMAT=128` (profile-only), `vmat_grid_splits=64`.
