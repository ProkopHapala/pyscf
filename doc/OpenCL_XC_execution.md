# OpenCL XC Grid Integration — Execution Model and Array Layout

This document specifies what happens inside one SCF cycle for exchange–correlation (XC)
grid integration in PySCF, how the OpenCL path differs from CPU, array dimensions,
and what is computed once vs every `get_veff` call.

**Notation** (matches your convention):

| Symbol | Meaning |
|--------|---------|
| `iAO` | index of atomic orbital basis function, `0 … nao-1` |
| `iR`  | index of grid point, `0 … nR-1` |
| `nao` | number of spatial AO functions |
| `nR`  | number of grid points (`grids.coords.shape[0]`) |
| `DM[iAO,jAO]` | density matrix |

---

## 1. Three distinct operations (do not conflate)

XC grid work splits into **three** steps. Earlier notes that said "AO evaluation"
without context were ambiguous. Here is the precise split:

### Step A — Build basis values on the grid (`eval_ao`)

Evaluate atomic orbitals at grid coordinates. This produces **Chi**, the AO-to-grid
projection matrix.

```
Chi[iAO, iR] = φ_iAO(r_iR)
```

**Does not use DM.** Depends only on `mol` and grid coordinates `r_iR`.
If the grid is fixed for the SCF loop, this step could be done **once per SCF**
(and reused across iterations). **Currently neither CPU nor OpenCL path caches it.**

### Step B — ρ projection (AO basis → grid)

Contract DM with Chi to get density (and gradient for GGA) at each grid point:

```
ρ(iR) = Σ_{μ,ν} Chi[μ,iR] · DM[μ,ν] · Chi[ν,iR]
```

Implementation uses an intermediate **ao_dm** (called `aodm` in OpenCL):

```
ao_dm[c, iAO, iR] = Σ_j DM[iAO,j] · Chi[j, iR]     # for each Cartesian component c
ρ[c, iR]            = Σ_μ ao_dm[c,μ,iR] · Chi[μ,iR]  # dotted on GPU
```

For GGA, `c = 0..3` is `(φ, ∂φ/∂x, ∂φ/∂y, ∂φ/∂z)`.

### Step C — Vmat assembly (grid → AO basis)

After libxc returns `vxc` on the grid, build the XC potential matrix:

```
V[iAO,jAO] = Σ_iR  w(iR) · vxc(iR) · Chi[iAO,iR] · Chi[jAO,iR]   (LDA)

GGA: form weighted AO combinations aow[iAO,iR] first, then
V[iAO,jAO] = Σ_iR Chi[iAO,iR] · aow[jAO,iR]
```

(`hermi_sum` at the end doubles the symmetric part for GGA.)

**Steps B and C must run every SCF iteration** because DM and `vxc` change.

---

## 2. PySCF storage convention vs Chi[iAO,iR]

PySCF stores the grid AO array as **`ao[iR, iAO]`** (grid index first), **F-contiguous**
in the `(iR, iAO)` plane:

```
ao.shape     = (nR_blk, nao)           # LDA, one block
ao.shape     = (4, nR_blk, nao)         # GGA with deriv=1
ao[iR, iAO]  = Chi[iAO, iR]
```

So your `Chi[iAO, iR]` is the **transpose** of PySCF's `ao`:

```
Chi[iAO, iR] = ao[iR, iAO]
Chi.shape    = (nao, nR)
```

### Is Chi tiled or sparse?

**CPU path (production `nr_rks`):** yes, sparse block screening.

- `grids.non0tab[iblk, ibas]` — uint8 mask: which AO **shells** are non-negligible
  at each grid block (`BLKSIZE = 56` grid points per screening block).
- `pair_mask[iAO_shell, jAO_shell]` — which shell pairs can be skipped in DM contraction.
- `_dot_ao_dm_sparse`, `_dot_ao_ao_sparse` in `pyscf/lib/dft/nr_numint_sparse.c` skip
  work using these masks.
- `ao` itself is still a **dense** `(nR_blk, nao)` array for active points; sparsity is
  in the **contraction**, not in storing a sparse Chi matrix.

**OpenCL path (`nr_rks_gpu`):** **dense**, no screening.

- Full `(nR_blk, nao)` blocks uploaded to GPU.
- ρ and vmat via dense GEMM + pointwise kernels.
- Simpler but no shell-pair skipping. For small molecules (benzene `nao=66`) this is fine.

**OpenCL GPU layout** for uploaded Chi:

```
bufAo[c]  flat index:  iR * nao + iAO     (row-major [nR_blk, nao], C-contiguous float32)
```

---

## 3. What runs when — SCF loop context

```
SCF kernel loop (each iteration):
│
├─ [once per SCF, if grids not rebuilt]
│    grids.build()          → coords[iR,3], weights[iR], non0tab
│    (OpenCL Hermite path)  → radial spline tables from mol (could be once)
│
├─ get_veff(dm)  ── called every SCF iteration
│    │
│    ├─ get_j, get_k        (DF or 4-center — separate topic)
│    │
│    └─ nr_rks / nr_rks_gpu(dm)   ← THIS DOCUMENT
│         Step A: build Chi       (depends on coords only)
│         Step B: ρ projection    (depends on DM)
│         libxc: ρ → exc, vxc     (pointwise, CPU)
│         Step C: vmat assembly   (depends on vxc, weights, Chi)
│
└─ Fock build, diagonalization, new DM
```

| Work | Depends on | Could cache per SCF? | Currently cached? |
|------|------------|----------------------|-------------------|
| Grid coords/weights | geometry, grid level | yes | yes (`grids.build`) |
| Hermite radial tables | basis, grid extent | yes | yes (`OpenCLAOHermiteEvaluator.__init__`) |
| **Chi on full grid** | coords, basis | **yes** | **no** — rebuilt every `nr_rks` call |
| ρ projection | DM | no | — |
| libxc | ρ | no | — |
| vmat assembly | vxc, Chi | no | — |

Within one `nr_rks` call, Chi is built **block by block** over `nR` points (not stored
for the full grid at once — only one block in memory).

---

## 4. CPU path pseudocode (`numint.nr_rks`, GGA)

Block size `blksize` chosen from `max_memory` (typically several × 56, not 8192).

```
INPUT:  DM[nao,nao], grids.coords[nR,3], grids.weights[nR]
OUTPUT: nelec, excsum, V[nao,nao]

V = zeros(nao, nao)
for each block [iR0, iR1) of grid:
    coords  = grids.coords[iR0:iR1]          # (nR_blk, 3)
    weight  = grids.weights[iR0:iR1]          # (nR_blk,)
    mask    = non0tab[iR0//56 :]              # screening

    # --- Step A: build Chi ---
    ao = eval_ao(mol, coords, deriv=1, non0tab=mask)
    # ao.shape = (4, nR_blk, nao),  ao[c, iR, iAO] = Chi[iAO,iR] or its derivative
    # ao[c] is F-contiguous (nR_blk, nao)

    # --- Step B: ρ projection ---
    rho = eval_rho1(ao, DM, mask, xctype='GGA')   # sparse C kernel
    # rho.shape = (4, nR_blk)
    # rho[0,iR] = ρ(r); rho[1:4,iR] = ∇ρ

    # --- libxc (CPU) ---
    exc, vxc = eval_xc_eff(xc_code, rho, deriv=1, xctype='GGA')
    # exc: (nR_blk,),  vxc: (4, nR_blk)

    nelec  += sum(rho[0] * weight)
    excsum += dot(rho[0]*weight, exc)
    wv     = weight * vxc                         # (4, nR_blk)

    # --- Step C: vmat assembly ---
    wv[0] *= 0.5                                  # hermi trick
    aow = scale_ao_sparse(ao[0:4], wv[0:4], mask) # (nR_blk, nao) weighted sum of components
    V  += dot_ao_ao_sparse(ao[0], aow, mask)      # (nao, nao), sparse

V = V + V.T                                       # hermi_sum
return nelec, excsum, V
```

**LDA:** `deriv=0`, `ao.shape = (nR_blk, nao)`, single `wv` and one `_dot_ao_ao_sparse`.

---

## 5. OpenCL path pseudocode (`nr_rks_gpu`, GGA)

Block size fixed `BLK = 8192`. **Both LDA and GGA are implemented.**

```
INPUT:  DM[nao,nao] float64 → uploaded once as DM_f32[nao,nao]
OUTPUT: nelec, excsum, V[nao,nao] float64

upload DM_f32 to bufDm                         # once per nr_rks_gpu call
allocate bufAo[4], bufAoDm[4], bufRho, bufWv, bufAow, bufVmat

V = zeros(nao, nao)
for each block [iR0, iR1):
    nR_blk = iR1 - iR0
    coords = grids.coords[iR0:iR1]
    weight = grids.weights[iR0:iR1]

    # --- Step A: build Chi (CPU libcint today) ---
    ao = eval_ao(mol, coords, deriv=1)           # float64, (4, nR_blk, nao)
    ao32[c] = convert_f32(ao[c])                 # per-component; see §7
    upload ao32[c] → bufAo[c]   for c=0..3

    # --- Step B: ρ projection (GPU, dense) ---
    for c in 0..3:
        bufAoDm[c] = bufAo[c] @ DM_f32           # GEMM: (nR_blk,nao) @ (nao,nao)
                                                 #      → ao_dm[c] stored as [iR,iAO]
    contract_rho_gga_from_aodm(bufAo, bufAoDm → bufRho)
    # bufRho layout: [ρ(iR), ∂ρ/∂x(iR), ∂ρ/∂y(iR), ∂ρ/∂z(iR)] each length nR_blk

    download bufRho → rho_f64[4, nR_blk]

    # --- libxc (CPU) ---
    exc, vxc = eval_xc_eff(xc_code, rho_f64, ...)
    wv[c,iR] = weight[iR] * vxc[c,iR];  wv[0] *= 0.5
    upload wv → bufWv

    # --- Step C: vmat assembly (GPU) ---
    scale_aow_gga_split(bufAo[0:4], bufWv → bufAow)
    # bufAow[iR, iAO] = Σ_c ao[c,iR,iAO] * wv[c,iR]
    bufVmat_blk = bufAow.T @ bufAo[0]            # GEMM transpose_A: (nao,nR_blk)@(nR_blk,nao)
    download bufVmat_blk → V_blk
    V += V_blk

V = V + V.T
return nelec, excsum, V
```

**LDA kernels:** `contract_rho_lda_from_aodm`, `scale_aow_lda` — one AO component only.

**GGA kernels:** `contract_rho_gga_from_aodm`, `scale_aow_gga_split` — four components
(φ and ∂φ/∂x, ∂φ/∂y, ∂φ/∂z). Derivatives come from `eval_ao(deriv=1)` in Step A.

**Alternate Step A (implemented, not wired into `nr_rks_gpu`):**

`OpenCLAOHermiteEvaluator.eval_sph` — GPU Hermite spline for **φ only** (no ∂φ yet).
Kernel: `eval_ao_mapped_hermite_cart` in `kernels.cl`.

---

## 6. Array dimensions reference (benzene 6-31g, grid level 3)

| Array | Shape | Dtype | Where | Layout notes |
|-------|-------|-------|-------|--------------|
| `DM` | `(nao, nao)` = `(66,66)` | f64 CPU / f32 GPU | host / `bufDm` | C-contiguous |
| `coords` | `(nR, 3)` = `(143560, 3)` | f64 | host | fixed per SCF |
| `weights` | `(nR,)` | f64 | host | fixed per SCF |
| `ao` / Chi (CPU) | `(4, nR_blk, nao)` | f64 | host per block | `ao[c]` F-contiguous `(nR_blk,nao)` |
| `Chi[iAO,iR]` | `(nao, nR_blk)` | — | conceptual | `= ao[c].T` |
| `bufAo[c]` | `nR_blk × nao` flat | f32 | GPU | C `[iR, iAO]` row-major |
| `bufAoDm[c]` | `nR_blk × nao` flat | f32 | GPU | `= Chi[c] @ DM` |
| `bufRho` | `4 × nR_blk` flat | f32 | GPU | `[comp*nR_blk + iR]` |
| `rho` (host) | `(4, nR_blk)` | f64 | host | downloaded for libxc |
| `vxc` | `(4, nR_blk)` | f64 | host | libxc output |
| `bufWv` | `4 × nR_blk` flat | f32 | GPU | weighted vxc |
| `bufAow` | `nR_blk × nao` flat | f32 | GPU | `[iR, iAO]` |
| `bufVmat` | `nao × nao` flat | f32 | GPU | one block contribution |
| `V` / vmat | `(nao, nao)` | f64 | host | accumulated |

**Per block** (`nR_blk ≤ 8192`): Chi size `4 × 8192 × 66 × 4 B` ≈ **8.4 MB** f32 on GPU.

**Hermite tables** (once per mol): `rad_val[nshell, nctr_max, nrad]` ≈ 0.06 MB for benzene.

---

## 7. Timing notes (benzene, RTX 3090, OMP_NUM_THREADS=1)

All GPU times use `queue.finish()` **before and after** each timed region.

### Per block (nR_blk = 8192, nao = 66)

| Stage | Operation | Time |
|-------|-----------|------|
| **A** | CPU `eval_ao(deriv=1)` — build Chi | **16.3 ms** |
| — | CPU `ao32[:,:,:] = ao` f64→f32 (bad layout) | **78 ms** ← harness bug |
| — | CPU per-component `ao[c].astype(f32)` | **5.5 ms** |
| — | GPU upload Chi ×4 | 0.5 ms |
| **B** | GPU `Chi @ DM` GEMM ×4 | 1.0 ms |
| **B** | GPU `contract_rho_gga` kernel | 0.1 ms |
| — | GPU download ρ | 0.02 ms |
| — | CPU libxc | 0.8 ms |
| — | GPU upload wv | 0.02 ms |
| **C** | GPU `scale_aow_gga_split` | 0.06 ms |
| **C** | GPU vmat GEMM `Chiᵀ @ aow` | 1.9 ms |
| — | GPU download vmat block | 0.01 ms |
| | **GPU kernels + PCIe (excl. Chi build)** | **~4 ms** |

### Full grid (nR = 143560, 18 blocks)

| Path | Time |
|------|------|
| CPU `nr_rks` | **284 ms** |
| GPU `nr_rks_gpu` (current harness) | **1489 ms** |
| Estimated if copy fixed (16+5.5+4 ms/blk × 18) | **~470 ms** |

**Conclusion:** GPU ρ projection and vmat GEMMs are **not** the bottleneck.
The current slowdown is (1) the strided `ao32[:] = ao` copy in `xc_grid.py:176` and
(2) rebuilding Chi on CPU every block every iteration instead of caching.

The 78 ms copy is a **real CPU cost** (verified with `queue.finish()` and timing copy
alone without any GPU calls). It is not a missing-`finish()` artifact. It arises because
`ao` is `(4, nR, nAO)` with non-contiguous 3D layout; `ao32[:] = ao` forces a slow
strided copy. Per-component `ao[c].astype(f32)` is 14× faster.

---

## 8. Kernel map (GGA)

| Step | Math | CPU | OpenCL kernel |
|------|------|-----|---------------|
| A | `Chi[iAO,iR] = φ_iAO(r_iR)` | `libcint eval_gto` | `eval_ao_mapped_hermite_cart` (alt., value only) |
| B | `ao_dm = Chi @ DM` | `_dot_ao_dm_sparse` | `matmul_tiled` ×4 |
| B | `ρ = diag(ao_dm · Chi)` | `eval_rho1` | `contract_rho_gga_from_aodm` |
| — | `exc, vxc = f(ρ)` | libxc | CPU only |
| C | `aow = Σ_c wv_c · Chi_c` | `_scale_ao_sparse` | `scale_aow_gga_split` |
| C | `V = Chi₀ᵀ @ aow` | `_dot_ao_ao_sparse` | `matmul_tiled_transpose_A` |

LDA uses `contract_rho_lda_from_aodm` and `scale_aow_lda` (one component).

Unused legacy kernels in `kernels.cl` (`eval_gto_sph`, `contract_rho`, `pbe_xc`, …) are
from an earlier all-GPU attempt; the active path uses CPU Chi + GPU GEMM/ρ/vmat.

---

## 9. Open issues

1. **Fix f32 staging** in `xc_grid.py` — per-component cast, or evaluate directly into
   C-contiguous `(nR_blk, nao)` f32 buffers.
2. **Cache Chi** across SCF iterations when grid coords are unchanged.
3. **Wire Hermite GPU** for Step A (φ); add derivative tables for GGA Step A on GPU.
4. **Persistent `GridPlan`** — avoid per-call `cl.Buffer` allocation (minor for benzene).
5. **Restore sparse screening** on GPU for large molecules (optional, for nao ≫ 100).
