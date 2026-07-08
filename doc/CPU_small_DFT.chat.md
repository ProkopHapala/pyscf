# USER

yes, I want you to make detailed plan how we can write optimized paralle path for small molecule focusing on grid opertions (projectiong denity on grid, projection AO on grid, assembling VMAT form grid), if we hould do it using python array slicing or in C/C++ with OpenMP, but plese think carefully about data oriented desing since this will be perhaps memory-bound operation, and we must avoid thread synchornizaton (atomic write shared variables) 

do not program anythin yet, just analyse the problem and write notes

---

# Cursore Composer 2.5

# Plan: Optimized CPU Parallel Path for Small-Molecule Grid Operations

Analysis notes only — no implementation. Goal: a CPU path competitive with (or complementary to) your OpenCL work for **nao ≲ 200**, **ngrids ≈ 30k–150k**, focusing on AO projection, ρ on grid, and vmat assembly.

---

## 1. Problem statement

Per SCF cycle, GGA XC needs:

| Step | Math | Current CPU hot spot |
|------|------|----------------------|
| **AO on grid** | χ_μ(g), ∂χ_μ/∂x,y,z | `eval_gto` → `GTOeval_loop` in `grid_ao_drv.c` |
| **ρ on grid** | ρ(g) = Σ_μν DM_μν χ_μ(g)χ_ν(g); GGA adds ∇ρ | `eval_rho1` → `_dot_ao_dm_sparse` + `_contract_rho_sparse` |
| **vmat** | V_μν = Σ_g w_g · v_xc(g) · χ_μ(g) · (weighted χ)_ν(g) | `_scale_ao_sparse` + `_dot_ao_ao_sparse` |
| **XC** (out of scope here but coupled) | ρ → vrho, vsigma via libxc | `LIBXC_eval_xc` — already OpenMP over grid |

Your OpenCL kernels (`rho_*_pair`, `vmat_*_pair`) already solve the **parallel decomposition** problem correctly: tile over grid points, reuse DM in fast memory, avoid scatter. The CPU path should mirror that logic, not the current PySCF `block_loop` + tiny GEMM approach.

---

## 2. Why the existing CPU path is the wrong shape

Three structural issues (from prior analysis):

1. **Serial Python orchestration** — `block_loop` runs 1–3 iterations for small molecules; no cross-block parallelism.
2. **`nao < SWITCH_SIZE (800)` gate** — disables sparse C kernels; falls back to `lib.dot`/`lib.ddot` on matrices too small for threaded BLAS.
3. **Wrong memory layout for grid-parallel access** — PySCF stores χ as **F-contiguous `[iG, iAO]`** (stride `nao` along grid). Grid-parallel threads want **contiguous `iG`** at fixed `iAO` → your OpenCL doc calls this the "stride-n gather" bug.

Existing sparse C code (`nr_numint_sparse.c`) is designed for **large nao** (shell-box decomposition, `BOXSIZE1_N` etc.). It is not tuned for tiny nao + huge ngrids.

---

## 3. Roofline / memory-bound analysis

For precomputed AO (GGA, 4 components):

```
Memory (read per SCF):  χ ≈ 4 × nao × ngrids × 8 bytes
                        DM ≈ nao² × 8 bytes  (fits in L2/L3)
                        ρ, wv ≈ O(ngrids)

Flops (ρ):              ≈ 2 × nao² × ngrids  (symmetric DM)
Flops (vmat):           ≈ 2 × nao² × ngrids  (same structure)
```

Example H2O cc-pVDZ: nao=24, ngrids=33k

| Quantity | Value |
|----------|-------|
| χ read (GGA) | 4 × 24 × 33k × 8 ≈ **25 MB** |
| DM | 24² × 8 ≈ **4.6 KB** |
| ρ flops | 2 × 576 × 33k ≈ **38 MFLOP** |
| Arithmetic intensity | ~1.5 FLOP/byte → **memory-bound** |

Example benzene: nao=114, ngrids=144k

| Quantity | Value |
|----------|-------|
| χ read (GGA) | 4 × 114 × 144k × 8 ≈ **525 MB** |
| DM | 114² × 8 ≈ **100 KB** |
| ρ flops | 2 × 13k × 144k ≈ **3.7 GFLOP** |
| Arithmetic intensity | ~7 FLOP/byte → still **memory-bound** on typical CPUs (~10–20 FLOP/byte peak) |

**Conclusion:** This is a **bandwidth + cache reuse** problem, not a raw FLOP problem. Design must:

- Stream χ with **unit stride in `iG`**
- **Reuse DM** maximally while touching each χ slice once
- **Fuse** ρ and (where possible) vmat to avoid multiple full χ passes
- Avoid Python/NumPy temporaries that double memory traffic

---

## 4. Python slicing vs C/C++ OpenMP

| Approach | Verdict |
|----------|---------|
| **Python `block_loop` + NumPy slices** | Too slow: GIL released per call but each step allocates/strides; 4–5 separate passes over χ; no fusion; BLAS on tiny nao doesn't thread well. Good for prototyping parity only. |
| **Numba/Cython** | Possible for ρ/vmat if precomputed χ; still awkward for `eval_gto` (libcint integration). Not PySCF convention. |
| **Extend existing `libdft` C + OpenMP** | **Recommended.** Matches PySCF architecture, reuses `non0tab`/screening, callable from `numint.py` like `VXCdot_ao_ao_sparse`. |
| **Separate small-mol module** | Also fine: `lib/dft/small_grid.c` with clean API, opt-in from `numint.py` when `nao < N_SMALL`. |

**Rule:** Python owns setup, grid build, libxc call, and result packaging. **All per-grid hot loops in C.**

---

## 5. Data-oriented design (SSOT layouts)

Adopt the same layouts your OpenCL precomp doc recommends — CPU and GPU should share indexing logic.

### 5.1 Primary arrays (Structure of Arrays, grid-major)

```
coords[g, 3]          // unchanged from Grids
weights[g]            // unchanged

χ0[iAO, iG]           // AO value;     C-contiguous in iG  (transpose of PySCF default)
χ1[iAO, iG]           // dφ/dx
χ2[iAO, iG]           // dφ/dy  
χ3[iAO, iG]           // dφ/dz

ρ[4, iG]              // or ρ0[iG], ρx[iG], ... separate (better for LDA)

wv[4, iG]             // weighted XC potential factors after libxc

DM[iAO, jAO]          // symmetric; store full or upper-tri in registers

V[iAO, jAO]           // vmat output; symmetric for LDA, full for GGA then hermi
```

**Index:** `χc[iAO * ngrids + iG]` — threads with consecutive `iG` read consecutive addresses.

### 5.2 Atom-blocked variant (optional, mirrors OpenCL `NATILE`)

For molecules with ≤ ~20 atoms:

```
χ[c, ia, a, iG]   where a ∈ [0, nao_atom[ia]), pad to MAX_AO_ATOM
DM_blk[ia, ja, a, b]  // pre-extracted per atom pair, setup once per SCF
```

Benefits:
- Inner loops are dense `a × b` without `ao_loc` indirection
- Natural parallel task = `(gTile, ia, ja)` atom pair
- Screening: skip pair if `pair_mask[ia,ja]` or atom not in grid shell

### 5.3 Tile constants (start values, tune per CPU)

| Symbol | Suggested | Role |
|--------|-----------|------|
| `NPTILE` | 32 or 64 | Grid points per tile (fits L1: 32 × nao × 8 × 4 comp) |
| `NATILE` | 1–2 atoms | Atom batch in inner loop |
| `NAO_TILE` | 16–32 | AO block when not atom-grouped |

For H2O (nao=24): entire AO fits one tile → `NPTILE=64` gives 64×24×4×8 ≈ 49 KB χ tile in L1.

### 5.4 What not to store

- Avoid materializing `aow = χ × wv` as full `[nao, ngrids]` for GGA if fusing vmat — that's an extra nao × ngrids write+read.
- Avoid `[4, ngrids, nao]` NumPy layout from current precomp staging.

---

## 6. Parallel decomposition without atomics

**Principle:** every parallel task must own **disjoint output memory** for the duration of the parallel region. Merge with **deterministic reduction** after the parallel loop (or use disjoint writes that cover the output once).

### 6.1 ρ on grid — parallel over `gTile` (recommended primary)

```
Task t owns grid indices [t*NPTILE, (t+1)*NPTILE)
Writes: ρ[c, g] for g in that range only
Reads:  DM (shared, read-only), χ[c, :, g] (private streaming)
```

- **No atomics.** Each `g` written by exactly one thread.
- Inner loop: for each `g` in tile, accumulate `Σ_μν DM_μν χ_μ(g) χ_ν(g)` with μ,ν loops in registers.
- Symmetric DM: iterate `μ ≤ ν`, factor 2 off-diagonal.

**Alternative (not preferred for small nao):** parallel over AO shells, private `ρ_t[ngrids]` buffer per thread → `NPomp_dsum_reduce` at end (existing pattern in `VXCdcontract_rho_sparse` when `nao*2 >= ngrids`). Works but multiplies ρ buffer by n_threads and needs reduction.

### 6.2 vmat assembly — parallel over `(iAO_block, jAO_block)` or fused with ρ

**Option A — disjoint AO blocks (mirrors existing `VXCdot_ao_ao_sparse`):**

```
Task (ib, jb) owns V[i0:i1, j0:j1]
Accumulates over all g:  V_μν += Σ_g wv(g) χ_μ(g) χ_ν(g)   [GGA: more terms]
```

- Each matrix block written by one thread → **no atomics**
- After parallel loop: symmetrize if hermi (`V += V.T` for GGA, or only compute upper triangle)
- Existing code already does this with `outbuf` per task; problem is task granularity is shell-box sized for large systems, not tuned for nao=24

**Option B — parallel over `gTile` with private `V_t[nao, nao]`:**

```
Per thread t:  V_t = 0
For g in tile:  V_t[μ,ν] += wv(g) χ_μ(g) χ_ν(g)
After parallel: V = Σ_t V_t   (nao² reduction, nao small → cheap)
```

- **No atomics** during accumulation
- Memory: `n_threads × nao² × 8` — for nao=114, 8 threads → ~800 KB, acceptable
- Advantage: same gTile traversal as ρ → **fuse ρ + vmat in one χ pass**

**Recommendation:** Option B (gTile + private vmat) for **nao < 150** because it enables fusion and matches OpenCL `rho_gga_pair` structure. Option A for larger nao if memory for per-thread vmat becomes costly.

### 6.3 AO on grid — two sub-strategies

**Path 1 — Precomputed χ (per geometry, like GPU `precomputed`):**

- Setup once: evaluate χ on full grid (can use existing `eval_gto`), transpose to `[iAO, iG]`
- Per SCF: only ρ + vmat (χ reused) — **this is the highest ROI for repeated SCF**
- AO eval parallelization only matters at setup / geometry change

**Path 2 — On-the-fly Hermite (like GPU `onthefly`):**

- Parallel over `(gTile, ia, ja)` atom pair — same task graph as `rho_lda_tiled`
- Each task: evaluate radial × angular for atoms ia, ja at grid tile, contract with DM block immediately
- **No χ buffer at all** — trades compute for memory
- For CPU: worth it only if χ memory exceeds L3 and hurts; for H2O χ is 25 MB → precompute is fine

**AO eval parallelization (when needed):**

```
Parallel over gTile:
  Each thread evaluates all AOs at coords[g0:g1]  → writes χ[:, g0:g1]
  Disjoint output columns → no atomics
```

This is better than current `GTOeval_loop` (parallel over shell×grid_block) for small molecules because it keeps **one grid tile hot in cache** while sweeping shells, rather than one shell hot while jumping grid blocks.

However, rewriting `eval_gto` is a large project (libcint). **Pragmatic phase 1:** use existing `eval_gto` at setup, transpose once. **Phase 2:** Hermite CPU port from your `radial_hermite.py` / `ao_hermite.py` logic.

---

## 7. Operation-by-operation design

### 7.1 AO projection (setup or per-geometry)

```
Input:  coords[g], mol, basis
Output: χ[c, iAO, iG]  for c=0..3 (GGA)

Steps:
  1. eval_gto (existing) → buf_F[c, iG, iAO]   # PySCF layout
  2. transpose per component (one-time, can OpenMP):
       χ[c, iAO, iG] = buf_F[c, iG, iAO].T      # C-contiguous in iG
  3. optional: atom-block reorder χ[c, ia, a, iG]
  4. optional: build pair_mask[ia,ja], atom_ao_offsets
```

**Screening:** keep `non0tab[g_blk, shell]` from grid build; at ρ/vmat time skip atoms whose shells are zero on that grid tile. Don't skip the transpose — it's cheap vs SCF.

### 7.2 ρ projection (per SCF)

**Kernel: `small_rho_gga_gtile`**

```
#pragma omp parallel for schedule(static)  // gTile loop, no dynamic overhead
for (gTile = 0; gTile < ngrids; gTile += NPTILE):
    g0 = gTile; g1 = min(g0+NPTILE, ngrids)
    // stack arrays: chi0_tile[NAO_TILE][NPTILE] loaded with coalesced reads
    
    for g in g0..g1:
        ρ[g] = 0; ρx[g] = 0; ...
        for ia in atoms:
          if !atom_active_on_tile(ia, gTile): continue
          for ja in ia..natoms:   // symmetric
            if !pair_mask[ia,ja]: continue
            load DM_blk[ia,ja] into registers (small)
            for a,b in atom AOs:
              accumulate ρ, ρx, ρy, ρz from χ values at g
```

**Micro-optimizations:**
- `schedule(static)` — equal work per gTile
- `#pragma omp simd` on inner `g` or `b` loop where compiler supports it
- `fma` for ρ accumulation
- For H2O: unroll entire nao=24 double loop (specialized `small_rho_gga_nao24` dispatch)

### 7.3 vmat assembly (per SCF, after libxc)

**Fused kernel: `small_vmat_gga_gtile_fused`** (if ρ already computed, still can fuse χ read with vmat):

```
#pragma omp parallel for schedule(static)
for gTile ...:
    V_priv[nao, nao] = 0   // thread-local, stack or malloc once per thread
    
    for g in tile:
        w0 = weights[g] * vrho[g]
        ws = weights[g] * vsigma[g] * ...   // GGA chain rule factors
        for μ,ν:
            V_priv[μ,ν] += w0*χ0[μ,g]*χ0[ν,g] + ws * (∂χ terms)...
    
    // reduction: only upper triangle if hermi
    merge V_priv into global V  // see below
```

**Merge without atomics:**

```c
// After parallel region — serial reduction over threads (nao² small)
for (t = 0; t < nthreads; t++)
    for (ij = 0; ij < nao*nao; ij++)
        V[ij] += V_thread[t][ij];
```

Or: give each thread a **disjoint slice of gTile** and accumulate into **one shared V** using **private buffers + single parallel reduction** (`NPomp_dsum_reduce` pattern already in PySCF).

**Do not** use `omp atomic` on `V[μ,ν]` inside the `g` loop.

### 7.4 libxc (keep as-is for now)

`LIBXC_eval_xc` already parallelizes over grid points with thread-local buffers. After ρ:

```
Input:  ρ[4, ngrids]  (C-contiguous in g)
Output: vrho[g], vsigma[g], exc[g]
```

Then compute `wv[g]` (weighted combination for vmat) — can fuse into vmat kernel to avoid storing `wv` if desired.

---

## 8. Avoiding synchronization pitfalls

| Anti-pattern | Why bad | Fix |
|--------------|---------|-----|
| `omp critical` on `V[μ,ν]` per grid point | Serializes hot loop | Private `V_t`, reduce after |
| `omp atomic` on ρ or V | Memory-bound + contention | Disjoint `gTile` ownership |
| Nested `omp parallel` (eval_ao + BLAS + libxc + vmat) | Oversubscription | **One parallel region** per phase; use `lib.num_threads(n)` globally; consider `OMP_MAX_ACTIVE_LEVELS=1` |
| Dynamic scheduling on uniform gTile | Overhead > benefit for small mol | `schedule(static)` |
| Multiple χ passes (ρ, then scale_ao, then dot) | 3× memory traffic | Fuse ρ+vmat or ρ+wv+vmat |
| Per-block `malloc` in parallel region | Allocator contention | Thread-local buffers allocated once in `#pragma omp parallel` before `for` |

Existing `VXCdot_ao_ao` (dense) uses `omp critical` for reduction — **do not copy this pattern** for the small-mol path.

---

## 9. Integration with PySCF (when implementing)

Suggested hook (later):

```
numint.py:block_loop / nr_rks
  if nao <= N_SMALL and xctype in ('LDA','GGA'):
      if plan.has_precomputed_chi:
          libdft.SMALL_nr_rks_precomp(...)
      else:
          libdft.SMALL_nr_rks_hermite(...)
  else:
      existing path
```

Or separate `backend=4` / `xc_path='small_cpu'` mirroring your GPU hook in `rks.py:get_veff()`.

**API sketch:**

```c
// setup (once per geometry)
void SMALL_setup_chi(double *chi[4], int nao, int ngrids, ...);

// per SCF
void SMALL_rho_gga(const double *dm, const double *chi[4],
                   double *rho, int nao, int ngrids, int nthreads);

void SMALL_vmat_gga(const double *chi[4], const double *wv,
                    double *vmat, int nao, int ngrids, int hermi);

// fused
void SMALL_rho_vmat_gga(...);  // one χ pass if libxc done in between — or two-phase
```

Keep `float64` for CPU parity with reference; optional `float32` dispatch later.

---

## 10. Phased implementation plan

### Phase 0 — Baseline & gates (1–2 days)
- [ ] Parity tests: H2O, benzene — ρ, vmat vs `ni.nr_rks` (backend=1)
- [ ] Micro-benchmark: time per component vs threads (1,2,4,8)
- [ ] Document crossover: `nao` where current sparse path wins

### Phase 1 — Layout + precomputed ρ/vmat (highest ROI)
- [ ] Transpose χ to `[iAO, iG]` after `eval_ao` (OpenMP parallel transpose)
- [ ] C kernel: `SMALL_rho_gga_gtile` — gTile parallel, no atomics
- [ ] C kernel: `SMALL_vmat_gga_gtile` — private V_t + reduction
- [ ] Wire thin Python wrapper; call libxc between ρ and vmat
- [ ] Target: beat dense `lib.ddot` path for nao < 200

### Phase 2 — Fusion & specialization
- [ ] Fused ρ+vmat where libxc allows (or fused wv application)
- [ ] Atom-blocked χ layout + pair screening
- [ ] Static dispatch: `nao ≤ 24`, `≤ 64`, `≤ 128` unrolled kernels
- [ ] SIMD intrinsics on inner `μ,ν` for fixed small nao

### Phase 3 — AO evaluation (optional, bigger scope)
- [ ] gTile-parallel AO eval wrapper around libcint (or)
- [ ] CPU Hermite on-the-fly port from OpenCL logic (`rho_*_pair` structure)
- [ ] Decide precomp vs OTF per molecule size / memory budget

### Phase 4 — SCF integration
- [ ] Hook in `get_veff()` or `numint.nr_rks` with auto-dispatch
- [ ] `setup_small_cpu()` mirroring `setup_gpu()` — cache χ across SCF cycles
- [ ] Profile full SCF: compare 1 vs N threads, vs OpenCL backend

---

## 11. Verification strategy

| Check | Tolerance |
|-------|-----------|
| ρ vs `ni.nr_rks` | max abs ~1e-10, rel ~1e-8 |
| vmat vs reference | max abs ~1e-9 (GGA symmetrization) |
| nelec, exc | match libxc path |
| Thread safety | 1-thread vs N-thread bitwise identical (reduction order fixed) |

Run with `backend=3` (CPU new + CPU old) during development, same pattern as GPU parity.

---

## 12. Key design decisions (summary)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | **C + OpenMP** | PySCF convention; no GIL; fusion; SIMD |
| Primary parallel axis | **`gTile` (grid tiles)** | Disjoint ρ[g]; enables χ streaming with stride-1 |
| vmat accumulation | **Private V_t per thread + reduction** | nao² small; avoids atomics; enables fusion |
| Memory layout | **`χ[iAO, iG]` C-contiguous in iG** | Memory-bound; matches OpenCL A1 layout |
| AO strategy (phase 1) | **Precompute + transpose at setup** | Avoids libcint rewrite; χ reused every SCF |
| AO strategy (phase 3) | **Hermite OTF atom-pair** | Port OpenCL kernel structure to CPU |
| Python role | **Setup + libxc + dispatch only** | No hot-loop slicing |
| Atomics | **Never in hot loop** | Reduction after parallel region only |

---

## 13. Relation to OpenCL work

Your OpenCL `rho_gga_pair` / `vmat_gga_pair` kernels are the **algorithmic spec** for the CPU port:

| OpenCL concept | CPU equivalent |
|----------------|----------------|
| `gTile`, `NPTILE` | `gTile` with `NPTILE=32..64` |
| `__local dm_blk` | L1/stack `DM_blk[nao_i][nao_j]` |
| `__local wfRj[NPTILE][NATILE][MAX_SHELL]` | stack arrays per thread |
| Coalesced `χ[iAO, iG]` | same layout, prefetch/stream |
| One workgroup per gTile | one OpenMP task per gTile |
| `psum` tree reduction in local mem | thread-private ρ, no cross-thread reduction needed |

The CPU path won't match GPU throughput but should achieve **near-linear scaling to 4–8 cores** on small molecules — the thing current PySCF cannot do.

---

I can turn this into a `doc/small_mol_cpu_grid_plan.md` in the repo, or go deeper on one piece next (e.g. exact fused kernel pseudocode for GGA vmat, or the atom-blocked index formulas matching your OpenCL headers).

---

