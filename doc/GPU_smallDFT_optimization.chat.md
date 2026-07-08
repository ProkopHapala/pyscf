# USER

OK, now lets focus on what could be further improved on our kernels for vmat

do it according to guidelines in 
@.cursor/skills/gpu-optimize/SKILL.md 
@doc/Optimization_guidelines.md

do not write any code yet, just notes, ideas what are currnely performance bottlenecks, and how we can rearange things to improve the performance (espacially for small molecule)

(btw. are all these kernels for GGA or only for LDA) 

---

# Cursor Composer 2.5

# vmat kernel optimization notes

Analysis follows the workflow in `gpu-optimize/SKILL.md` and `Optimization_guidelines.md`: profile first, classify bottleneck, one structural hypothesis at a time, no code yet.

---

## GGA vs LDA — what exists?

**Both.** The OpenCL XC stack supports LDA and GGA (`XCGridPlan` raises for meta-GGA). Kernel inventory:

| Family | LDA | GGA |
|--------|-----|-----|
| OTF tiled | `vmat_lda_tiled` | `vmat_gga_tiled` |
| OTF pair (`NATILE=1`) | `vmat_lda_pair` | `vmat_gga_pair` |
| Precomp pair | `vmat_lda_precomp_pair` | `vmat_gga_precomp_pair` |
| Precomp coalesced | — | `vmat_gga_precomp_coalesced_pair` |
| Radial precomp | — | `vmat_gga_radial_precomp_pair` |
| Blocked fallback | `scale_aow_lda` + GEMM | `scale_aow_gga_split` + GEMM |

**Production PBE work is GGA.** Radial-precomp and coalesced vmat are **GGA-only** in the host wiring (`vmat_mode='radial_precomp'` requires GGA). LDA paths exist for other functionals / parity but are not the hot PBE path.

**GGA vs LDA inside kernels:**
- LDA: `fill_atom_ao_lda` + `wv[g] * aoI * aoJ`; tiled LDA skips upper triangle (`jTile < iTile`) and writes symmetric `vmat` in-kernel.
- GGA: `fill_atom_aow_gga` (weighted φ + ∇φ) × plain φ; **one-sided** matrix; host does `vmat + vmat.T`.

---

## Current bottleneck classification (benzene, RTX 3090)

From `doc/GPU_benchmark.md` (event profiling validated):

| Stage | OTF cubic | Hybrid (OTF ρ + rad vmat) |
|-------|----------:|----------------------------:|
| ρ | 5.3 ms | 5.0 ms |
| vmat | **22.6 ms** | **15.5 ms** |
| PBE+xc | 0.3 ms | 0.3 ms |

**vmat is the limiter** (~75–80% of GPU time). PBE, host transfers, dm→cart are negligible.

### Bottleneck type (decision tree)

| Symptom | Classification | Evidence |
|---------|----------------|----------|
| vmat ≫ ρ despite same grid | **Compute + memory reuse** inside vmat, not PCIe | wall ≈ CL events |
| Radial vmat −7 ms vs OTF | **Hermite eval in fill phase** is a large fraction | same geometry, no spline in radial fill |
| ρ fast, vmat slow | **Parallel mapping asymmetry** | see below |
| 36 tiled vmat WGs on 82 SMs | **Scheduling / insufficient parallelism** (Type B) | small-molecule specific |

---

## Root cause 1: parallelism geometry (critical for small molecules)

**ρ and vmat use opposite outer loops.**

```
rho_gga_tiled:   outer = grid tiles  → ~2244 × NATILE ≈ 4488 WGs  (benzene)
vmat_gga_tiled:  outer = atom tiles  → 6 × 6 = 36 WGs            (benzene, NATILE=2)
```

Each vmat WG then runs a **serial loop over all grid tiles** (~2244 × `NPTILE=64`):

```2175:2224:pyscf/OpenCL/kernels.cl
    for (int gTile = 0; gTile < ngrids; gTile += NPTILE) {
        // fill aoI (Hermite + unfold) + fill aoJ
        barrier(...);
        // dot over NPTILE grid points per (iao, jao)
        barrier(...);
    }
```

For benzene this is the dominant structural issue: the GPU is fed **36 long-running serial jobs** instead of **thousands of short grid-parallel jobs** like ρ.

Hybrid path already improves this partially: `vmat_gga_radial_precomp_pair` uses **pair geometry** → 12×12 = **144 WGs** (vs 36 tiled), but still grid-serial inside each WG → 15.5 ms remains.

**Guideline alignment:** gpu-optimize §GPU Saturation — *"If one molecule is too small, batch replicas/systems"* OR remap parallelism so grid is outer.

---

## Root cause 2: duplicate expensive work per grid point (OTF path)

Inside each `gTile`, for each atom in the tile:

1. `hermite_map_point(d)` — log-map + knot index
2. `eval_radials_slice(_deriv)` — spline eval per shell (reads `rad_node`)
3. `unfold_shell(_deriv)` — angular factors

`fill_atom_aow_gga` and `fill_atom_ao_lda` each call `hermite_map_point` **once per atom per grid point per shell loop**, but map `(t, ik)` is identical for all shells on the same atom at the same `g`.

**Cost stack per grid point (GGA OTF vmat fill):**
- i-side: `fill_atom_aow_gga` — spline + deriv + 4-component weighted unfold
- j-side: `fill_atom_ao_lda` — spline + unfold
- Then `NPTILE` multiply-accumulates

ρ already paid similar Hermite cost (~5 ms). vmat **repeats** it and adds the grid loop × atom-pair outer structure.

**Guideline:** Tier 2 in optimization order — fix reuse before tuning arithmetic.

---

## Root cause 3: register / `__local` pressure

Per vmat tiled WG (`NATILE=2`, `NPTILE=64`, `MAX_AO_ATOM=16`):

- `__local aoI[64][32]`, `aoJ[64][32]` → ~16 KB each
- Per-thread private: `acc[QPT]` with `QPT=4`, plus `fill_atom_*` private arrays (`R[MAX_SHELL]`, `dR`, `f0..f3[6]`)

Likely **moderate occupancy** — not the primary issue on benzene (too few WGs already), but matters when atom count grows and WG count rises.

**Guideline:** Do not add more `__local` without measured reuse; consider splitting fill vs dot phases if spills appear in compiler report.

---

## Root cause 4: GGA does not skip redundant atom-pair tiles

LDA tiled: `if (jTile < iTile) return` + symmetric write.

GGA tiled: **all** `iTile × jTile` tiles computed; symmetry via host `vmat + vmat.T`.

For 12 atoms: 36 GGA tiles vs 21 LDA upper-triangle tiles (~71% extra work). GGA one-sided formulation may require this unless you skip `ja > ia` and accept incomplete coverage before symmetrization (needs careful math check).

---

## What radial precomp already solved (~7 ms)

`vmat_gga_radial_precomp_pair` replaces Hermite with:

```3397:3399:pyscf/OpenCL/kernels.cl
        int ir = ir_list[s];
        int n = unfold_shell_deriv(l_list[s], rad_val[ir * ngrids + g], rad_dr[ir * ngrids + g], ...);
```

- **Coalesced gather** `rad_val[ir*ngrids+g]` across `g` in a tile
- No spline arithmetic in the hot loop
- `R,dR` built once at setup (`build_radial_on_grid_tiled`)

Remaining 15.5 ms is mostly: **grid-serial loop × barriers × unfold**, not Hermite.

---

## Rearrangement ideas (prioritized, small-molecule focus)

Following gpu-optimize order: transfers/launches → access pattern → bytes/output → reuse → parallelism → registers → arithmetic.

### Tier A — Structural remapping (largest leverage for small molecules)

| # | Idea | Hypothesis | Small-mol impact |
|---|------|------------|------------------|
| **A1** | **Grid-outer vmat** (invert ρ geometry) | One WG per `gTile`; accumulate into `vmat` via atom-pair local reduction or scratch + assembly | Fixes 36-WG starvation; ~2244 WGs like ρ |
| **A2** | **Split: grid-parallel `aow` build + GEMM** | Match CPU/`blocked` path: `scale_aow_gga` over grid → `matmul_gpu_buf_accum(ao, aow, vmat)` | GEMM uses mature tiling; grid pass is parallel |
| **A3** | **`NATILE=1` pair kernels for vmat** | 144 WGs (benzene) vs 36; already used for hybrid radial vmat | Easy experiment via tile sweep; partial fix |
| **A4** | **Batch molecules / replicas** | Same kernel, many systems → saturate GPU | Best for MD, conformer scans, dimer z-sweeps without kernel rewrite |

**A2 detail (most aligned with existing code):** Precomp non-fused path already does `scale_aow_gga_split` + `matmul_gpu_buf_accum`. For OTF, intermediate `buf_aow[ngrids, ncart]` (~69 MB benzene) trades memory for parallelism. Grid-parallel OTF fill into `aow` → one GEMM. Hypothesis: on small `ncart`, GEMM may beat 2244 serial iterations inside 36 WGs.

**A1 vs A2:** A2 reuses existing matmul infrastructure; A1 needs new reduction/ownership design for `vmat[i,j]`.

### Tier B — Reduce work per grid point (OTF and radial)

| # | Idea | Hypothesis |
|---|------|------------|
| **B1** | **Hoist `hermite_map_point` per (g, atom)** | Same `(t, ik)` for all shells; eval radials in a tight loop | Cuts map + branch overhead in OTF fill |
| **B2** | **Share `R,dR` prepass kernel** | One launch fills `rad_val/dr` tile; ρ and vmat consume (extend hybrid) | Removes duplicate Hermite in ρ too; 3 launches but less total work |
| **B3** | **Atom-pair grid screening** | Skip `gTile` when both atoms beyond `Rcut` (from `grid_screen.py`) | Bigger win for large systems; modest for compact benzene |
| **B4** | **Skip redundant GGA atom-pair tiles** | Upper-triangle atom pairs + host symmetrize | ~45% fewer pair WGs if mathematically valid |

### Tier C — Inner-loop / micro (only after A/B)

| # | Idea | Notes |
|---|------|-------|
| **C1** | Vectorize `ip` dot (`float4`/`float8` over grid in tile) | Inner loop is `aoI[ip][a]*aoJ[ip][b]` — memory-bound |
| **C2** | Compile-time `SPLINE_ORDER` / separate cubic/quintic kernels | Remove `spline_order` branch in `hermite_eval_ir` |
| **C3** | `native_sqrt`, `native_rsqrt` in unfold | Under parity budget |
| **C4** | Tune `NPTILE`, `WGS_VMAT`, `NATILE` | `sweep_opencl_tiles.py`; pair vs tiled is not obvious without events |
| **C5** | Fuse fill+dot within tile only | Keeps atom-pair outer; saves barriers per gTile, not grid parallelism |

### Tier D — Do not pursue first

- Mega-kernel fusing ρ + PBE + vmat (register/ICache pressure; guidelines §7)
- More `__local` without measured reuse
- Quintic-specific speed opts under memory-equivalent `du`
- Coalesced χ path for vmat on small molecules (high setup, mixed ρ cost)

---

## Suggested experiment sequence (one hypothesis each)

Per gpu-optimize workflow — measure with `clGetEventProfilingInfo`, parity vs CPU `NumInt`:

1. **Profile sub-phases inside vmat** — time `fill` vs `dot` vs `barrier` (custom counters or split kernels temporarily). Confirms whether grid loop or fill dominates in radial (15.5 ms) vs OTF (22.6 ms).

2. **`NATILE=1` sweep** — same benzene, compare `vmat_gga_tiled` vs `vmat_gga_pair` vs `vmat_gga_radial_precomp_pair` with event times. Tests A3.

3. **Blocked GEMM vmat baseline** — precomp path `fused=False`: `scale_aow_gga_split` + `matmul_gpu_buf_accum` on benzene. If faster than 15.5 ms radial pair, validates A2 for OTF.

4. **Grid-outer prototype** — even a throwaway kernel: grid-parallel `aow` write + existing GEMM. Compare total vmat time vs pair kernel.

5. **B1 hoist `hermite_map_point`** — only if OTF path remains after A2/A3; isolated parity check.

---

## Small molecule vs large molecule

| Regime | Dominant issue | Best direction |
|--------|----------------|----------------|
| **Small** (benzene, H₂O, formic, `natoms<20`) | Too few vmat WGs; grid loop serial inside WG | A2/A3/A4: grid-outer or GEMM or batch |
| **Medium** (`natoms` 20–100) | WG count grows; Hermite fill still costly | Hybrid radial vmat (done) + B1 + screening |
| **Large** (`nao` 300+) | Memory for χ; `MAX_ITILE` caps | OTF or radial; screening essential |

For your stated focus (small molecules), **parallelism remapping (Tier A)** matters more than spline micro-opts (Tier C). Hybrid radial vmat already captured the big Hermite win; the next ~5–10 ms likely needs **changing who owns the outer loop** — grid, not atom-pair.

---

## Summary answer to your question

**Kernels are not GGA-only** — full LDA + GGA families exist. **Production PBE profiling is GGA.** The vmat bottleneck is structural: atom-pair-outer × grid-inner serial loop, duplicate Hermite in OTF fill, and severe GPU under-utilization on small molecules. Radial precomp removed spline cost (−7 ms); what remains needs **rearranging parallelism** (grid-outer / GEMM split / pair geometry / batching), not arithmetic tuning first.