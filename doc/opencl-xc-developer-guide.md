# OpenCL XC / DF Developer Guide

> **Purpose**: Living document for the PySCF OpenCL GPU offloading effort.
> **Rule of thumb**: Almost every performance problem is a **memory-layout problem**.
> If you find yourself fighting the compiler, you probably laid out the data wrong.
>
> Last updated: 2026-06-29

---

## 1. SCF Loop — Where does GPU work sit?

```text
for cycle = 1 .. max_cycle:
    # -------- PER-CYCLE (hot path) --------
    # These run every SCF iteration.  Must be zero-alloc, minimal H<->D transfer.

    build_fock = get_jk() + get_veff()
        get_jk()  -> DF J/K contraction (GPU, float32)
        get_veff():
            1. DM[nao,nao] -> rho[ngrids]      (GPU)
            2. rho[ngrids] -> vxc[ngrids]      (CPU libxc, or GPU PBE)
            3. vxc[ngrids] -> VMAT[nao,nao]    (GPU)

    diagonalize -> new DM
    # ---------------------------------------

# -------- PER-LOOP (once per SCF) --------
# These should be hoisted OUTSIDE the SCF loop.
# If anything below is called inside the loop, it is a bug.

    grids.build()                # grid points, weights, screen_index
    setup_xc_grid_gpu()          # compile CL, build Hermite tables, alloc buffers
    setup_precomputed_gto()      # evaluate GTO on ALL grid points, upload AO to GPU
    dfobj.build()                # build 3-center ERIs (cderi), upload to GPU
    unpack cderi on GPU          # (triangular -> full, done once and cached)
```

**Critical invariant**: Anything in the `PER-LOOP` block must be called **exactly once**.
The `PER-CYCLE` block must not allocate GPU buffers, compile kernels, or rebuild tables.

### Per-cycle vs per-system costs (benzene cc-pVDZ, 143k grids)

| Operation | Frequency | Cost |
|-----------|-----------|------|
| Hermite table build | Once | ~0.1s |
| Grid build | Once | ~0.1s |
| OpenCL compile | Once | ~0.5s |
| Static buffer upload | Once | ~0.01s |
| DM → Cartesian + upload | Per-cycle | ~0.001s (114² matmul) |
| `rho` kernel | Per-cycle | **0.14s ← BOTTLENECK** |
| `rho` D2H copy | Per-cycle | ~0.002s |
| `eval_xc_eff` (CPU libxc) | Per-cycle | ~0.01s |
| `wv` H2D upload | Per-cycle | ~0.001s |
| `vmat` kernel | Per-cycle | **0.089s** |
| `vmat` D2H copy | Per-cycle | ~0.001s |
| `c2s` transform | Per-cycle | ~0.001s |

**Note**: AO evaluation is **fused inside** the rho and vmat kernels in the on-the-fly path.
It is NOT a separate step. The Hermite radial tables depend only on geometry + basis,
not on DM, so they are built and uploaded once.

---

## 2. Central Data Structures & Memory Footprints

### 2.1 Naming convention

| Symbol | Meaning | Typical shape |
|--------|---------|---------------|
| `iG` | grid point index | `0 .. ngrids-1` |
| `iAO` / `mu, nu` | atomic-orbital (spherical) index | `0 .. nao-1` |
| `iCart` | Cartesian basis function index | `0 .. ncart-1` |
| `iAtom` / `iat` | atom index | `0 .. natoms-1` |
| `P` | auxiliary basis index (DF) | `0 .. naux-1` |

### 2.2 Core arrays

| Array | Layout | Size (float32) | Sparsity |
|-------|--------|----------------|----------|
| `DM[nao, nao]` | row-major | `nao² × 4 B` | dense |
| `VMAT[nao, nao]` | row-major | `nao² × 4 B` | dense |
| `AO_GTO[iG, iAO]` | **row-major** `[ngrids, nao]` | `ngrids·nao × 4 B` | dense (precomp) |
| `AO_GGA[c, iG, iAO]` | `[4, ngrids, nao]` | `4·ngrids·nao × 4 B` | dense (precomp) |
| `cderi[P, nao_pair]` | row-major | `naux·nao_pair × 4 B` | dense |
| `cderi_full[P, i, j]` | `[naux, nao, nao]` | `naux·nao² × 4 B` | dense |

**Example — benzene cc-pVDZ (nao=114, ngrids~46k):**

```
AO_LDA  = 46e3 * 114 * 4 B  ≈  21 MB
AO_GGA  = 4 * 21 MB         ≈  84 MB
DM/VMAT = 114² * 4 B        ≈  52 KB   (negligible)
cderi   = 612 * 6555 * 4 B  ≈  16 MB
cderi_full                 ≈  32 MB
```

**Example — PTCDA cc-pVDZ (nao~300, ngrids~200k):**

```
AO_LDA  ≈ 240 MB
AO_GGA  ≈ 960 MB   (may exceed GPU memory)
```

This is why the **on-the-fly Hermite path** exists: it trades compute for memory by
reconstructing AOs from radial tables inside the kernel instead of storing AO globally.

### 2.3 Hermite radial tables (on-the-fly path)

These are tiny, atom-blocked, and uploaded once:

```
rad_val        [nradial, nrad]     float32  ~ (natom * n_shell_per_atom * nctr) * nrad
rad_du         same                float32
rad_dy         same                float32
radial_l       [nradial]           int32
radial_cart0   [nradial]           int32
atom_radial_offset [natoms+1]      int32   (CSR-style)
atom_radial_list   [nradial]       int32
c2s            [ncart, nao]        float32  (Cartesian -> spherical transform)
```

For water cc-pVDZ: total Hermite table size ≈ **~1–5 MB**, vs AO precompute ≈ **~20 MB**.

---

## 3. The Three Central Operations

### 3.1 Operation 1: DM → Real-space density `rho[iG]`

**Name in PySCF**: `make_rho` (CPU) / `contract_rho_*` (GPU)

**Math**:

```
LDA :  rho[iG] = sum_mu sum_nu  phi[iG,mu] * DM[mu,nu] * phi[iG,nu]
GGA :  rho[0,iG] = sum_mu sum_nu  phi[0,iG,mu] * DM[mu,nu] * phi[0,iG,nu]
       rho[1,iG] = 2 * sum_mu sum_nu  phi[0,iG,mu] * DM[mu,nu] * phi[1,iG,nu]
       (and similarly for y, z components)
```

**Is it a bottleneck?** **YES.**  For precomputed AO, this is a dense GEMM
`phi @ DM` (O(ngrids·nao²)) followed by a pointwise contraction (O(ngrids·nao)).
For on-the-fly Hermite, the AO reconstruction is fused into the contraction.

#### Implemented variants

| Variant | Where | Kernel(s) | Status |
|---------|-------|-----------|--------|
| **CPU reference** | `pyscf/dft/numint.py::block_loop` | `make_rho` | correct, slow |
| **GPU block-loop (CPU AO)** | `xc_grid.py::nr_rks` | `contract_rho_lda_from_aodm` / `contract_rho_gga_from_aodm` | **working** |
| **GPU full-grid (Hermite AO)** | `xc_grid.py::nr_rks_hermite_ao` | same kernels, one call for all grids | **working** |
| **GPU on-the-fly (legacy)** | `kernels.cl` | `rho_lda_onthefly` / `rho_gga_onthefly` | works, limited to ncart/nao ≤ 128 |
| **GPU on-the-fly (pair)** | `kernels.cl` | `rho_lda_pair` / `rho_gga_pair` | **working, preferred** |
| **GPU on-the-fly (tiled)** | `kernels.cl` | `rho_lda_tiled` / `rho_gga_tiled` | **working, preferred** |
| **GPU precomp (pair)** | `kernels.cl` | `rho_lda_precomp_pair` / `rho_gga_precomp_pair` | **working, preferred** |
| **GPU precomp (tiled)** | `kernels.cl` | `rho_lda_precomp_tiled` / `rho_gga_precomp_tiled` | **working** |
| **GPU precomp (fused GEMM)** | `kernels.cl` | `rho_lda_precomp_fused` / `rho_gga_precomp_fused` | working, fallback |

**Not finished**: None — all major variants are implemented. Tuning (register tiling,
memory coalescing) remains open.

---

### 3.2 Operation 2: `rho[iG]` → XC potential `vxc[iG]`

**Name in PySCF**: `eval_xc_eff` (CPU, libxc) / `pbe_xc_*` (GPU)

**Math**:

```
LDA :  vxc[iG] = dE/drho  at rho[iG]
GGA :  exc[iG], vrho[iG], vsigma[iG] = f(rho, |grad rho|²)
       then wv[0] = weight * vrho
            wv[1..3] = weight * 2 * vsigma * grad_rho[1..3]
```

**Is it a bottleneck?** **YES for GGA.**  The libxc call is CPU-only and requires
`rho` to be on the host as `float64`.  Every time we download `rho` from GPU → CPU
and upload `wv` back, we pay a **PCIe round-trip**.

#### Implemented variants

| Variant | Where | Notes | Status |
|---------|-------|-------|--------|
| **CPU libxc (float64)** | `pyscf.dft.libxc.eval_xc` | called from Python, requires `rho` on host | correct, **transfer bottleneck** |
| **GPU PBE float32** | `pbe.cl::pbe_xc_f32` + `compute_wv_gga_f32` | pure GPU, no D2H for rho | **working**, accuracy ~1e-5 |
| **GPU PBE float64** | `pbe.cl::pbe_xc_f64` + `compute_wv_gga_f64` | pure GPU, no D2H for rho | **working**, slower |

**Not finished**: Meta-GGA, hybrid functionals, range-separated hybrids.

---

### 3.3 Operation 3: `vxc[iG]` → VMAT[nao, nao]

**Name in PySCF**: `_dot_ao_ao` / `_dot_ao_ao_sparse` (CPU) / `vmat_*` kernels (GPU)

**Math**:

```
LDA :  VMAT[mu,nu] += sum_iG  phi[iG,mu] * wv[iG] * phi[iG,nu]
GGA :  VMAT[mu,nu] += sum_iG  aow[iG,mu] * phi[0,iG,nu]
       where aow = sum_c  wv[c,iG] * phi[c,iG,:]
       then hermi_sum: VMAT += VMAT.T
```

**Is it a bottleneck?** **YES.**  This is another O(ngrids·nao²) contraction.
For GGA it also needs the 3 gradient components of the AO basis.

#### Implemented variants

| Variant | Where | Kernel(s) | Status |
|---------|-------|-----------|--------|
| **CPU sparse** | `pyscf/dft/numint.py` | `_dot_ao_ao_sparse` | correct, uses pair-screening |
| **GPU block-loop (CPU AO)** | `xc_grid.py::nr_rks` | `scale_aow_lda` / `scale_aow_gga_split` + `matmul_tiled_transpose_A_accum` | **working** |
| **GPU full-grid (Hermite AO)** | `xc_grid.py::nr_rks_hermite_ao` | same kernels, one call | **working** |
| **GPU on-the-fly (legacy)** | `kernels.cl` | `vmat_lda_onthefly` / `vmat_gga_onthefly` | works, limited to ≤128 |
| **GPU on-the-fly (pair)** | `kernels.cl` | `vmat_lda_pair` / `vmat_gga_pair` | **working, preferred** |
| **GPU on-the-fly (tiled)** | `kernels.cl` | `vmat_lda_tiled` / `vmat_gga_tiled` | **working, preferred** |
| **GPU precomp (pair)** | `kernels.cl` | `vmat_lda_precomp_pair` / `vmat_gga_precomp_pair` | **working, preferred** |
| **GPU precomp (tiled)** | `kernels.cl` | `vmat_lda_precomp_tiled` / `vmat_gga_precomp_tiled` | **working** |

**Not finished**: CPU-style sparse pair-screening on GPU (atom-blocked pair lists).

---

## 3b. Three XC Paths (Implementation Variants)

### Path A: Precomputed GTO (CPU AO eval → GPU GEMM)

```
setup:  CPU eval_ao() → ao[ngrids, nao] → upload to GPU (62 MB for benzene)
cycle:  upload DM → GEMM(ao, DM) → aodm → contract → rho → D2H → libxc → H2D wv
        → GEMM(aow^T, ao) → vmat → D2H
```

- **Pros**: Simple, uses cuBLAS-style GEMM, AO values computed exactly by PySCF CPU
- **Cons**: AO array is huge (62 MB LDA, 249 MB GGA for benzene), upload cost, device memory limit
- **When to use**: Small systems where AO fits in device memory
- **Files**: `xc_grid.py::setup_precomputed_gto()`, `nr_rks_precomputed_gto()`

### Path B: Materialized Hermite AO (GPU Hermite eval → GPU GEMM)

```
setup:  build Hermite tables → upload
cycle:  eval_ao_hermite_gpu() → buf_ao on device (no host transfer)
        GEMM(buf_ao, DM) → aodm → contract → rho → D2H → libxc → H2D wv
        → GEMM(aow^T, buf_ao) → vmat → D2H
```

- **Pros**: AO evaluation on GPU, no CPU eval_ao, no host AO transfer
- **Cons**: Still materializes full AO array on device (62 MB), GEMM is general-purpose (not AO-aware)
- **When to use**: Medium systems where AO fits in device memory
- **Files**: `xc_grid.py::nr_rks_hermite_ao()`, `ao_hermite.py`

### Path C: On-the-fly Hermite (fused kernels, no AO materialization) ← CURRENT FOCUS

```
setup:  build Hermite tables → upload (256 KB, static)
cycle:  rho_kernel(DM, hermite_tables, coords) → buf_rho  [AO eval fused inside]
        D2H rho → libxc → H2D wv
        vmat_kernel(wv, hermite_tables, coords) → buf_vmat  [AO eval fused inside]
        D2H vmat
```

- **Pros**: No AO materialization (saves 62–249 MB device memory), AO values computed
  on-the-fly from tiny Hermite tables (256 KB), naturally exploits atom-tile locality
- **Cons**: More complex kernels, redundant AO eval between rho and vmat kernels
  (they re-evaluate the same radial functions). Hermite eval is cheap; the grid loop dominates.
- **When to use**: Large systems, or when device memory is limited
- **Files**: `xc_grid.py::setup_onthefly()`, `nr_rks_hermite_onthefly()`,
  `kernels.cl: rho_*_tiled/pair, vmat_*_tiled/pair`

### Path selection logic

```python
if gpu_mem_sufficient_for_full_grid_AO_float32:
    plan = setup_precomputed_gto(mol, grids, xc)   # Path A
else:
    plan = setup_xc_grid_gpu(mol, grids, xc)       # Path C

# In get_veff():
if backend & 2:
    if use_precomp:
        n, exc, vxc = plan.nr_rks_precomputed_gto(dm)   # Path A
    else:
        n, exc, vxc = plan.nr_rks_hermite_onthefly(dm)  # Path C
```

---

## 4. AO Evaluation — Where does `phi[iG, iAO]` come from?

This is the only operation that **could** be done once per SCF loop (precomputed),
but we also support on-the-fly reconstruction for memory savings.

### 4.1 CPU GTO evaluation (`eval_ao`)

- **Path**: `pyscf.gto.eval_gto` → libcint C code
- **Output layout**: `[ncomp, nblk, nao]` with **Fortran-ordered** inner `[nblk, nao]` slabs.
- **Cost**: expensive, but only once if precomputed.
- **Problem for GPU**: the Fortran-order slabs are **not** what our row-major GPU
  kernels expect.  We currently copy/astype them into C-contiguous `float32` buffers.

### 4.2 GPU Hermite interpolation (atom-block)

- **Path**: `ao_hermite.py::OpenCLAOHermiteEvaluator`
- **Kernel**: `eval_ao_mapped_hermite_cart_atom` / `eval_ao_mapped_hermite_cart_deriv1_atom`
- **Thread layout**: `(grid_point, atom)` 2D dispatch
- **Method**: each thread evaluates ALL radial channels for ONE atom at ONE grid point.
  Radial part via cubic Hermite spline on mapped log grid `u = log1p(r/r0)`.
  Angular part is explicit s/p/d/f Cartesian unroll.
- **Output**: Cartesian AOs on GPU → multiplied by `c2s` (Cartesian→spherical) via GEMM.
- **Cost**: moderate compute, but no large `AO` buffer stored.

### 4.3 On-the-fly (no AO materialization)

- **Path**: `xc_grid.py::nr_rks_hermite_onthefly`
- **Kernel**: `rho_lda_pair` / `rho_gga_pair` / `vmat_lda_pair` / `vmat_gga_pair`
- **Method**: each thread evaluates AOs on-the-fly, contracts immediately with DM
  (for rho) or accumulates to VMAT (for vmat).  No `AO[ngrids, nao]` array exists.
- **Trade-off**: saves memory (no AO buffer), but each thread rebuilds AOs → high
  register pressure and redundant radial interpolation across threads.

---

## 5. The CPU AO Memory Layout Problem

### What PySCF `eval_ao` returns

```
LDA:  ao = mol.eval_gto('GTOval_sph', coords)   # shape [nblk, nao], order='F'
GGA:  ao = mol.eval_gto('GTOval_sph_deriv1', coords)  # shape [4, nblk, nao], order='F'
```

The inner `[nblk, nao]` matrix is **Fortran-contiguous** (column-major).
This is because libcint writes data in column-major order for compatibility with
BLAS/LAPACK routines used elsewhere in PySCF.

### Why this hurts GPU performance

Our GPU kernels are row-major:

```c
// Row-major: phi[iG, iAO] = ao[iG * nao + iAO]
// Consecutive threads (varying iG) stride by nao -> non-coalesced
```

If we naively copy the Fortran-ordered CPU buffer to GPU, **consecutive threads
read non-consecutive memory** (stride `nao * sizeof(float)`), destroying GPU
memory throughput.

### What we currently do (workaround)

```python
ao_staging[c][ip0:ip1] = ao[c].astype(np.float32)   # copy + cast -> C-contiguous
```

`astype(np.float32)` creates a **new C-contiguous array**, which is then uploaded.
This is correct but adds:
1. A temporary `float64` → `float32` allocation per block
2. A host-side memcpy

### The 4 specific problems with PySCF's libcint layout

PySCF stores basis set information in the `mol` object using the **libcint convention**:
- `mol._bas[nbas, BAS_SLOTS=8]`: per-shell info (atom, angular momentum, nprim, nctr, ptr_exp, ptr_coeff)
- `mol._env[]`: flat array of exponents and contraction coefficients
- `mol._atm[natm, ATM_SLOTS=6]`: per-atom info (nuclear charge, coordinates pointer)

This layout is **optimized for serial CPU evaluation** where you loop over shells,
then primitives, then contracted functions. It is **not suitable for GPU** because:

1. **Indirect indexing**: To get AO value for atom `ia`, shell `ib`, you must:
   `bas[ib*BAS_SLOTS+ATOM_OF]` → `atm[atom*ATM_SLOTS+PTR_COORD]` → `env[ptr]`.
   Three levels of indirection, no coalescing.

2. **Variable-length shells**: Different shells have different `nprim` and `nctr`.
   No fixed stride. GPU kernels need `if` guards or worst-case loops.

3. **No atom-grouped layout**: Shells are ordered by shell index, not by atom.
   To get all AOs for atom `ia`, you must search all shells for `bas[ib].atom == ia`.

4. **Spherical/Cartesian mismatch**: PySCF works in spherical harmonics (`nao`),
   but the Hermite interpolation naturally produces Cartesian AOs (`ncart`).
   The `c2s` matrix converts between them.

### The solution (already implemented): atom-grouped CSR format

The `MappedHermiteRadialBasis` class in `radial_hermite.py` reorganizes the basis
into a **GPU-friendly atom-grouped format**:

```
radial_values[nradial, nrad]       — Hermite table values per radial channel
radial_du_values[nradial, nrad]    — Hermite slopes per radial channel
radial_dy_values[nradial, nrad]    — Precomputed y[i+1]-y[i] for float32 stability
radial_l[nradial]                  — Angular momentum per radial channel
radial_cart0[nradial]              — Starting Cartesian AO index per radial channel
atom_radial_offset[natoms+1]       — CSR offset: which radial channels belong to atom ia
atom_radial_list[nradial]          — CSR list: radial channel indices per atom
```

This gives:
- **O(1) access** to all radial channels for a given atom (CSR lookup)
- **Fixed-size per-atom tiles** (`MAX_AO_ATOM=16`) with zero-padding
- **Coalesced reads** from `rad_node` (interleaved value+slope, contiguous per channel)
- **No shell search** — direct index `iao_l = il*MAX_AO_ATOM + a`

### What remains problematic

- The `c2s` (Cartesian→spherical) transform is still done on the host as a matrix
  multiply (`c2s.T @ vmat_cart @ c2s`). For large `ncart`, this is non-trivial.
  Could be done on GPU using `matmul_gpu_buf` (`buf_c2s` is already uploaded).

- DM is uploaded in spherical basis and transformed to Cartesian on host
  (`c2s @ dm @ c2s.T`) before the rho kernel. This could also be done on GPU.

- The `MAX_AO_ATOM` constant (16) limits the basis set size per atom.
  Atoms with more than 16 Cartesian AOs (e.g. heavy atoms with f-functions)
  require increasing this constant and recompiling.

### What should be better

**Option A**: Keep AO on GPU in **transposed layout** `[nao, ngrids]`.
Then the contraction kernels read with stride-1 across threads.
All our `contract_rho` and `scale_aow` kernels would need to be rewritten for
`phi[iAO, iG]` layout.

**Option B**: Evaluate AOs **directly on GPU** in row-major order.
The Hermite path already does this (`eval_ao_mapped_hermite_cart_atom`).
The precomputed-GTO path still downloads from CPU because libcint is the
reference implementation.

**Option C**: libcint custom output format.  Unlikely to be accepted upstream.

---

## 6. DF J/K Contraction

### 6.1 J matrix (Coulomb)

```python
# CPU math:  vj = unpack_tril( dmtril @ cderi.T @ cderi )
# GPU path:
matmul_gpu_buf(bufDmtril, bufCderi, bufTmp,      nset, naux, nao_pair, transpose_B=True)
matmul_gpu_buf(bufTmp,     bufCderi, bufVjPacked, nset, nao_pair, naux)
_unpack_tril_batched_to_buf_gpu(bufVjPacked, bufVjFull, nset, nao, nao_pair)
```

- `cderi` uploaded **once** and cached.
- `bufCderiFull` (unpacked `[naux, nao, nao]`) is built on first K call and cached.

### 6.2 K matrix (exchange)

```python
# CPU math:  vk[i,j] = sum_P sum_k cderi[P,i,k] * DM[k,j] * cderi[P,j,i]
# GPU path (rearranged for GEMM):
dm_all = dms.transpose(1,0,2).reshape(nao, nset*nao)   # [nao, nset*nao]
matmul_gpu_buf(bufCderiFull, bufDmAll, bufBuf1All, naux*nao, nset*nao, nao)
transpose_k_buf1_batched(bufBuf1All, bufBuf1RAll, naux, nao, nset)
matmul_gpu_buf(bufBuf1RAll, bufCderiFull, bufVkAll, nset*nao, nao, naux*nao)
```

- **Bottleneck**: `transpose_k_buf1_batched` is a naive per-element transpose
  with no local-memory tiling → non-coalesced both read and write.
- **Fix needed**: implement tiled local-memory transpose (see Section 8).

---

## 7. Timing / Profiling Methodology

### 7.1 GPU timing (queue.finish)

**Never** time GPU kernels with `time.perf_counter()` alone.
OpenCL commands are **asynchronous**; they return immediately after enqueue.

```python
# CORRECT — per-stage timing used in our code
t0 = time.perf_counter()
cl.enqueue_nd_range_kernel(queue, knl, global_size, local_size)
queue.finish()          # <-- BLOCK until GPU is done
stage_time = time.perf_counter() - t0
```

Our `xc_grid.py` already does this via `_gpu_sync(queue)` inside `_precomp_rho_fused`,
`_precomp_vmat_fused`, and `_nr_rks_precomputed_gpu`.

### 7.2 Host-side timing stages

```python
TIMING_STAGE_ORDER = (
    'host_h2d_dm', 'host_dm_cart',
    'gpu_rho', 'host_rho_d2h',
    'gpu_xc_pbe', 'host_xc_libxc', 'host_xc_reduce',
    'gpu_vmat', 'host_vmat_d2h',
    'host_pair_mask', 'host_cpu_projection',
    'gpu_total', 'host_total', 'wall_profiled', 'n_blocks',
)
```

When profiling, print these stages to see where time goes.

### 7.3 Key metric: `gpu_total` vs `host_total`

- `gpu_total >> host_total` → good, GPU is doing the work.
- `host_total >> gpu_total` → bad, Python / PCIe / libxc dominates.
  Likely causes: per-block D2H of rho, too many kernel launches, small matrices.

---

## 8. Known Bottlenecks (ranked by impact)

### vmat kernel design (current optimized — tiled path)

```
workgroup = one (iTile, jTile) atom-pair tile  [NATILE atoms each]
thread    = owns QPT = ceil(AO_TILE² / WGS_VMAT) AO-pair matrix elements
local     = aoI[NPTILE][AO_TILE], aoJ[NPTILE][AO_TILE]  [unfolded AO cache]
private   = acc[QPT]  [accumulators]

for each grid tile gTile:
    cooperatively fill aoI (iTile atoms, NPTILE grid points)
    cooperatively fill aoJ (jTile atoms, NPTILE grid points)
    barrier
    each thread: for each of QPT AO-pairs:
        acc[t] += sum_{ip=0..NPTILE} wv[gTile+ip] * aoI[ip][iao_l] * aoJ[ip][jao_l]
    barrier

write acc[t] → vmat[iao, jao]  (and symmetric vmat[jao, iao] if iTile≠jTile)
```

**Key optimization**: AO values are unfolded once into local memory.
No redundant radial or angular computation. Each thread reads from local cache,
not re-evaluating. This is a **58× speedup** over the earlier abTile hybrid design
(5.17s → 0.089s on benzene).

### Performance results (on-the-fly path, NVIDIA GTX 1650)

| System | natoms | ncart | ngrids | rho | vmat | Total OTF | CPU ref |
|--------|--------|-------|--------|-----|------|-----------|---------|
| water | 3 | 114 | 34,576 | 0.006s | 0.027s | 0.18s | 0.03s |
| benzene | 12 | 114 | 143,560 | 0.140s | 0.089s | 0.58s | 0.49s |
| pentacene | 36 | 226 | 321,784 | 1.182s | 0.570s | 2.98s | 4.53s |
| PTCDA | 38 | 286 | 379,216 | 1.896s | 0.859s | 4.33s | 8.84s |

### rho kernel detailed analysis

`rho` is the bottleneck (0.14s vs 0.089s for vmat on benzene). The rho kernel
uses `(gTile, iTile)` 2D workgroups with WGS=64. Each workgroup:
- Iterates over ALL j-atoms (inner loop `for jTile = 0..natoms`)
- Evaluates j-atom radial functions cooperatively into `wfRj[NPTILE][NATILE][MAX_SHELL]`
- Loads DM block `dm_blk[NATILE][NATILE][MAX_AO_ATOM][MAX_AO_ATOM]` from global memory
- Each thread (one grid point × one i-atom) contracts over all j-atoms

For benzene: `ceil(143560/64) * ceil(12/2) = 2243 * 6 = 13458` workgroups,
each iterating over 12 atoms in 6 jTile steps. The DM block load is
`2*2*16*16 = 1024 floats = 4 KB` per jTile per iTile iteration, loaded from
global memory every time.

**Optimization opportunities**:
- **Grid screening**: Skip atom pairs where the grid tile is outside the atom's
  cutoff radius. `grid_screen.py` is implemented but **not wired in**.
  Could reduce work by 5–10× for sparse systems.
- **Cache DM blocks**: The DM block is reloaded for every grid tile.
  Could be loaded once per workgroup if it fits in local memory.
- **Symmetry**: Only compute upper triangle (DM is symmetric).
  Currently all pairs are computed.

### GPU PBE wiring for on-the-fly path

Currently the on-the-fly path **always** downloads `rho` to host for libxc.
To eliminate the round-trip:

```
rho_kernel → buf_rho (stay on device)
pbe_xc_f32 → buf_exc, buf_vrho, buf_vsigma (on device)
compute_wv_gga_f32 → buf_wv (on device)
vmat_kernel → buf_vmat
```

Only `exc` (for energy) needs reduction — can be done with a small kernel
or downloaded as a scalar sum. Expected savings: ~0.013s (benzene), ~0.05s (PTCDA).

---

## 8. Known Bottlenecks (ranked by impact)

| # | Bottleneck | Where | Impact | Fix |
|---|------------|-------|--------|-----|
| 1 | **PCIe round-trip for rho + wv** | `nr_rks`, `nr_rks_hermite_ao` | **~50–80% of wall time** on small systems | Use GPU PBE (`gpu_xc='pbe_f32'`) or fuse rho+xc+vmat in single kernel |
| 2 | **GEMM: no register tiling** | `matmul_tiled` | 2–10× slower than cuBLAS | Add 4×4 or 8×8 register tiles per thread; shrink workgroup to 16×16 |
| 3 | **Strided memory in `contract_rho` / `scale_aow`** | `contract_rho_*_from_aodm`, `scale_aow_*` | Non-coalesced reads (stride `nao`) | Transpose AO to `[nao, ngrids]` or swap kernel launch dims |
| 4 | **Naive transpose in K-build** | `transpose_k_buf1_batched` | Non-coalesced read+write | Local-memory tiled transpose (32×32 tile) |
| 5 | **Per-call buffer allocation** | `nr_rks_hermite_onthefly` (legacy) | Driver overhead | Preallocate all buffers in `setup_onthefly()` |
| 6 | **`.wait()` prevents CPU/GPU overlap** | Every `cl.enqueue_copy(...).wait()` | Serializes pipeline | Use event dependencies, out-of-order queue |
| 7 | **On-the-fly register pressure** | `rho_*_onthefly` (legacy) | Private arrays >128 floats/thread → spill | Use pair/tiled kernels instead |
| 8 | **CPU-side numpy ops in DF J/K** | `df_jk.py::get_jk` | `dm_sym`, `dmtril`, `dm_all` built on CPU | Move tril packing to GPU kernel |
| 9 | **Fortran-order AO from CPU** | `setup_precomputed_gto` | Requires copy+transpose per block | Evaluate AOs directly on GPU (Hermite) |
| 10 | **In-order queue** | `__init__.py` | No kernel overlap | `cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE` |

---

## 9. File Index

| File | Role |
|------|------|
| `pyscf/OpenCL/__init__.py` | Context/queue init, `to_device` / `to_host` helpers |
| `pyscf/OpenCL/kernels.cl` | **All OpenCL kernels**: GEMM, rho/vmat contractions, AO eval, unpack, transpose |
| `pyscf/OpenCL/pbe.cl` | GPU PBE functional (`pbe_xc_f32`, `pbe_xc_f64`, `compute_wv_gga_*`) |
| `pyscf/OpenCL/tile_config.py` | Compile-time tile sizes (`NPTILE`, `NATILE`, `WGS_VMAT`, `MAX_ITILE`) |
| `pyscf/OpenCL/xc_grid.py` | **Main harness**: `XCGridPlan`, all `nr_rks_*` variants, buffer management |
| `pyscf/OpenCL/df_jk.py` | **DF J/K harness**: `DFJKPlan`, `get_jk`, triangular unpack |
| `pyscf/OpenCL/ao_hermite.py` | GPU Hermite AO evaluator (atom-block kernels) |
| `pyscf/OpenCL/radial_hermite.py` | CPU build of Hermite radial tables |
| `pyscf/OpenCL/buffers.py` | `CLBuffer` wrapper (preallocated upload/download) |
| `pyscf/OpenCL/grid_screen.py` | Grid-point tile atom screening (sphere-AABB, **not yet wired in**) |
| `pyscf/dft/rks.py` | `RKS.get_veff` — SCF integration, calls GPU plan |
| `pyscf/dft/numint.py` | CPU reference `nr_rks`, `eval_xc_eff`, `block_loop` |
| `pyscf/dft/libxc.py` | libxc interface (CPU XC functional evaluation) |
| `doc/vmat_optimization_report.md` | vmat kernel optimization history (abTile → local cache) |
| `expamples_prokop/test_opencl.py` | Quick parity test (XC + DF J/K) |
| `expamples_prokop/test_opencl_hermite_ao.py` | AO Hermite interpolation accuracy test |
| `expamples_prokop/test_opencl_xc_hermite_ao.py` | XC with Hermite AO (full-grid GEMM path) |
| `expamples_prokop/test_opencl_xc_onthefly.py` | On-the-fly Hermite XC benchmark |
| `expamples_prokop/test_opencl_xc_scf.py` | **Rigorous benchmark**: all GPU paths vs CPU libxc |
| `expamples_prokop/profile_dft.py` | Monkey-patch PySCF timers for 1-cycle profiling |
| `expamples_prokop/sweep_opencl_tiles.py` | Tile-size sweep for auto-tuning |

---

## 10. Test Results (float32 accuracy)

Run: `PYTHONPATH=/home/prokophapala/git/pyscf OMP_NUM_THREADS=1 python3 expamples_prokop/test_opencl.py`

| Quantity | Max abs error | Relative error |
|----------|---------------|----------------|
| Vxc (XC potential matrix) | ~1.3e-5 | ~1e-6 |
| J (Coulomb) | ~4.3e-6 | ~1e-6 |
| K (Exchange) | ~8.4e-6 | ~1e-6 |
| Hermite AO cart vs CPU | ~1.7e-6 | — |
| Hermite AO sph vs PySCF | ~1.4e-6 | — |
| Hermite AO deriv1 vs PySCF | ~2.2e-4 | — |

**Acceptable threshold**: `~1e-5` max abs for Vxc, `~1e-6` for J/K.
Errors above `~5e-3` indicate a bug (wrong kernel, wrong layout, or wrong sign).

---

## 11. Decision Tree: Which Path to Use?

```
Is GPU memory sufficient for full-grid AO float32?
  ├── YES  ->  setup_precomputed_gto() + nr_rks_precomputed_gto()
  │            Choose fused='tiled' (default) or 'pair' if natoms small
  │            Choose gpu_xc='pbe_f32' if xc_code == 'PBE' (avoids D2H)
  │            Choose gpu_xc='cpu' otherwise (falls back to libxc)
  │
  └── NO   ->  setup_onthefly() + nr_rks_hermite_onthefly()
               (reconstructs AOs on-the-fly, no large AO buffer)
               Always uses libxc on CPU for xc (currently)
```

---

## 12. Quick Checklist Before Commit

- [ ] Did you add `queue.finish()` before every GPU stage timing measurement?
- [ ] Did you verify no new buffer allocations happen inside the SCF loop?
- [ ] Did you run `test_opencl.py` and confirm errors < 5e-3?
- [ ] Did you run `test_opencl_xc_scf.py` to see wall-time breakdown?
- [ ] Did you check that `gpu_total` is the dominant stage in the timing?
- [ ] Did you document any new kernel in this file?
