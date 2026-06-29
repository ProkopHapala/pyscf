# OpenCL XC / DF Architecture & Performance Report

> **Rule of thumb**: Almost every performance problem is a **memory-layout problem**.
> If you find yourself fighting the compiler, you probably laid out the data wrong.
>
> Last updated: 2026-06-29

---

## 1. Naming Convention

| Symbol | Meaning | Typical shape |
|--------|---------|---------------|
| `iG` | grid point index | `0 .. ngrids-1` |
| `iAO` / `mu, nu` | atomic-orbital (spherical) index | `0 .. nao-1` |
| `iCart` | Cartesian basis function index | `0 .. ncart-1` |
| `iAtom` / `iat` | atom index | `0 .. natoms-1` |
| `P` | auxiliary basis index (DF) | `0 .. naux-1` |

---

## 2. Central Quantities and Memory Layout

### 2.1 Core arrays

| Symbol | Shape | dtype | Description |
|--------|-------|-------|-------------|
| `DM[iAO, iAO]` | `(nao, nao)` | f64 host / f32 device | Density matrix in spherical LCAO basis |
| `DM_cart[iCart, iCart]` | `(ncart, ncart)` | f32 device | Density matrix in Cartesian basis (device working format) |
| `VMAT[iAO, iAO]` | `(nao, nao)` | f64 host | XC potential matrix (returned to SCF) |
| `VMAT_cart[iCart, iCart]` | `(ncart, ncart)` | f32 device | XC potential in Cartesian basis (device working format) |
| `AO[iAO, iG]` | `(ngrids, nao)` | f32 device | AO basis values on grid (materialized path) |
| `AO_cart[iCart, iG]` | `(ngrids, ncart)` | f32 device | Cartesian AO values on grid (Hermite path) |
| `AO_GGA[c, iG, iAO]` | `[4, ngrids, nao]` | f32 device | AO values + gradients (GGA precomputed) |
| `rho[iG]` (LDA) | `(ngrids,)` | f32 device → f64 host | Electron density per grid point |
| `rho[4, iG]` (GGA) | `(4, ngrids)` | f32 device → f64 host | `[rho, grad_x, grad_y, grad_z]` |
| `wv[iG]` (LDA) | `(ngrids,)` | f32 device | `weight * vxc` — weighted XC potential |
| `wv[4, iG]` (GGA) | `(4, ngrids)` | f32 device | `weight * [vrho, 2*vsigma*gx, 2*vsigma*gy, 2*vsigma*gz]` |
| `rad_node[iRad, iRad_tab, 2]` | `(nradial, nrad, 2)` | f32 device | Hermite radial tables: `[value, slope]` per radial channel |
| `atom_radial_offset/list` | CSR `(natoms+1,) / (nradial,)` | int32 device | Atom-to-radial-channel CSR mapping |
| `c2s` | `(ncart, nao)` | f32 device | Cartesian → spherical transform matrix |
| `cderi[P, nao_pair]` | `(naux, nao*(nao+1)/2)` | f32 device | 3-center ERIs (DF), uploaded once |
| `cderi_full[P, i, j]` | `(naux, nao, nao)` | f32 device | Unpacked cderi (triangular → full), cached on device |

### 2.2 Key dimensions

- `nao` = number of spherical AOs (PySCF `mol.nao_nr()`)
- `ncart` = number of Cartesian AOs (`mol.cart2sph_coeff` maps ncart → nao)
- `ngrids` = number of DFT grid points (e.g. 143560 for benzene/cc-pVDZ/grid=3)
- `natoms` = number of atoms
- `nradial` = total radial channels (sum of shell × contracted primitives across all atoms)
- `naux` = auxiliary basis size for density fitting

### 2.3 Data structure sizes

**Benzene cc-pVDZ (nao=114, ngrids~143k):**

| Array | Size | Bytes | Notes |
|-------|------|-------|-------|
| `DM_cart` | 114² | 52 KB | Uploaded once per SCF cycle |
| `AO_cart` (materialized) | 143560 × 114 | 62 MB | **Huge — this is why on-the-fly wins** |
| `AO_cart` deriv1 (4 components) | 4 × 143560 × 114 | 249 MB | Even bigger for GGA |
| `rho` (GGA) | 4 × 143560 | 2.2 MB | Small, downloaded to host every cycle |
| `wv` (GGA) | 4 × 143560 | 2.2 MB | Small, uploaded from host every cycle |
| `VMAT_cart` | 114² | 52 KB | Small, downloaded to host every cycle |
| `rad_node` | ~80 × 400 × 2 | 256 KB | **Static — uploaded once, never changes** |
| `coords` | 143560 × 4 | 2.2 MB | Static per SCF (grid fixed) |
| `cderi` | 612 × 6555 | 16 MB | Static (DF), uploaded once |
| `cderi_full` | 612 × 114² | 32 MB | Unpacked on device, cached |

**PTCDA cc-pVDZ (nao~300, ngrids~200k):**

```
AO_LDA  ≈ 240 MB
AO_GGA  ≈ 960 MB   (may exceed GPU memory)
cderi   ≈ much larger
```

This is why the **on-the-fly Hermite path** exists: it trades compute for memory by reconstructing AOs from radial tables inside the kernel instead of storing AO globally.

### 2.4 Hermite radial tables (on-the-fly path)

These are tiny, atom-blocked, and uploaded once:

```
rad_val        [nradial, nrad]     float32  ~ (natom * n_shell_per_atom * nctr) * nrad
rad_du         same                float32
rad_dy         same                float32  (precomputed y[i+1]-y[i] for float32 stability)
radial_l       [nradial]           int32
radial_cart0   [nradial]           int32
atom_radial_offset [natoms+1]      int32   (CSR-style)
atom_radial_list   [nradial]       int32
c2s            [ncart, nao]        float32  (Cartesian -> spherical transform)
```

For water cc-pVDZ: total Hermite table size ≈ **~1–5 MB**, vs AO precompute ≈ **~20 MB**.

### 2.5 Sparsity

AO values are **block-sparse**: each grid point only has non-zero AO values for atoms within a cutoff radius. PySCF uses `non0tab` screening (grid blocks × shell pairs). The Hermite on-the-fly kernels exploit this implicitly: each workgroup only evaluates radial functions for atoms in its tile, and zero-pads inactive atoms. The `grid_screen.py` module provides explicit sphere-AABB screening to build per-gTile active atom lists, but this is **not yet integrated** into the on-the-fly kernels.

---

## 3. SCF Loop Pseudocode

```text
# === ONE-TIME SETUP (before SCF loop) ===
build grids (coords, weights)                      # per-system, not per-cycle
build Hermite radial tables (rad_node, atom_radial_*)  # per-system
compile OpenCL program                             # per-system
upload static buffers: coords, rad_node, atom_coords, atom_radial_*  # per-system
allocate device buffers: buf_dm_cart, buf_rho, buf_wv, buf_vmat      # per-system
dfobj.build()                # build 3-center ERIs (cderi), upload to GPU
unpack cderi on GPU          # (triangular -> full, done once and cached)

# === SCF LOOP (per cycle) ===
for cycle in range(max_cycle):
    # 1. Build Fock: F = h1e + V_eff
    #    V_eff = V_HF(J,K) + V_xc
    #    V_HF from density fitting (separate OpenCL path in df_jk.py)
    #    V_xc from XC grid integration (this report)

    # 2. Diagonalize F → mo_coeff, mo_occ
    # 3. Build new DM: dm = make_rdm1(mo_coeff, mo_occ)
    # 4. Compute V_xc(dm) → vmat  [THIS IS THE HOT PATH]

    # --- V_xc computation (3 operations) ---

    # Op 1: DM → rho (density on grid)
    #   rho[g] = sum_{ij} DM[i,j] * AO[i,g] * AO[j,g]
    rho_kernel(dm_cart, hermite_tables, coords) → buf_rho

    # Op 2: rho → vxc (XC functional evaluation)
    #   PBE: exc[g], vxc[g] = f(rho[g], |∇rho[g]|)
    #   Currently on CPU via libxc (eval_xc_eff)
    #   GPU PBE kernel exists (pbe.cl) but only for precomputed path
    rho_d2h() → host
    exc, vxc = ni.eval_xc_eff(xc_code, rho)    # CPU libxc
    wv = weight * vxc
    wv_h2d() → buf_wv

    # Op 3: vxc → vmat (potential matrix assembly)
    #   vmat[i,j] = sum_g wv[g] * AO[i,g] * AO[j,g]
    vmat_kernel(wv, hermite_tables, coords) → buf_vmat
    vmat_d2h() → host
    vmat = c2s.T @ vmat_cart @ c2s             # Cartesian → spherical

    # 5. Check convergence
```

**Critical invariant**: Anything in the setup block must be called **exactly once**.
The per-cycle block must not allocate GPU buffers, compile kernels, or rebuild tables.

### Per-cycle vs per-system operations

| Operation | Frequency | Cost |
|-----------|-----------|------|
| Hermite table build | Once | ~0.1s |
| Grid build | Once | ~0.1s |
| OpenCL compile | Once | ~0.5s |
| Static buffer upload (coords, rad_node) | Once | ~0.01s |
| DF cderi build + upload | Once | system-dependent |
| DM → Cartesian transform + upload | Per-cycle | ~0.001s (114² matmul) |
| `rho` kernel | Per-cycle | **0.14s (benzene) — BOTTLENECK** |
| `rho` D2H copy | Per-cycle | ~0.002s |
| `eval_xc_eff` (CPU libxc) | Per-cycle | ~0.01s |
| `wv` H2D upload | Per-cycle | ~0.001s |
| `vmat` kernel | Per-cycle | **0.089s (benzene)** |
| `vmat` D2H copy | Per-cycle | ~0.001s |
| `c2s` transform | Per-cycle | ~0.001s |

**Critical note**: AO evaluation (projecting basis functions onto grid) is done **inside** the rho and vmat kernels in the on-the-fly path. It is NOT a separate per-cycle step. The Hermite radial tables are built once and uploaded once. This is correct — AO values depend only on geometry + basis, not on DM.

---

## 4. The Three Central XC Operations

### 4.1 Operation 1: DM → Real-space density `rho[iG]`

**Name in PySCF**: `make_rho` (CPU) / `contract_rho_*` (GPU)

**Formula**: `rho[g] = sum_{ij} DM[i,j] * phi_i(g) * phi_j(g)`

For GGA also: `grad_rho[g] = sum_{ij} DM[i,j] * (nabla phi_i(g) * phi_j(g) + phi_i(g) * nabla phi_j(g))`

**Is it a bottleneck?** **YES.** For precomputed AO, this is a dense GEMM `phi @ DM` (O(ngrids·nao²)) followed by a pointwise contraction (O(ngrids·nao)). For on-the-fly Hermite, the AO reconstruction is fused into the contraction.

#### Variants implemented

| Kernel | File | Layout | Status | Notes |
|--------|------|--------|--------|-------|
| **CPU reference** | `numint.py::block_loop` | `make_rho` | correct, slow | |
| `rho_lda_tiled` | `kernels.cl` | `(gTile, iTile)` 2D, WGS=64 | **Active** (tiled path) | Caches j-atom radials in local `wfRj`, loops i-atoms in private registers |
| `rho_gga_tiled` | `kernels.cl` | `(gTile, iTile)` 2D, WGS=64 | **Active** (tiled path) | Same + `dwfRj` for gradients |
| `rho_lda_pair` | `kernels.cl` | `(gTile,)` 1D, WGS=NPTILE | **Active** (pair path, NATILE=1) | One atom pair at a time, simpler |
| `rho_gga_pair` | `kernels.cl` | `(gTile,)` 1D, WGS=NPTILE | **Active** (pair path) | |
| `rho_lda_onthefly` / `rho_gga_onthefly` | `kernels.cl` | legacy | works, limited to ncart/nao ≤ 128 | High register pressure |
| `rho_lda_precomp_pair` | `kernels.cl` | Precomputed AO | **Active** (precomputed path) | Reads AO from global memory |
| `rho_gga_precomp_pair` | `kernels.cl` | Precomputed AO | **Active** (precomputed path) | |
| `rho_lda_precomp_tiled` / `rho_gga_precomp_tiled` | `kernels.cl` | Precomputed AO | **Active** | |
| `rho_lda_precomp_fused` / `rho_gga_precomp_fused` | `kernels.cl` | Precomputed AO | working, fallback | |
| `contract_rho_lda_from_aodm` | `kernels.cl` | GEMM path | **Active** (materialized Hermite AO) | Uses `matmul_gpu_buf` + contraction |

#### What is NOT done

- **No grid screening in on-the-fly kernels**: All atoms are evaluated for every grid tile, even when the atom is far away. `grid_screen.py` exists but is not wired in. This is the biggest opportunity for rho speedup.
- **No fused rho+vmat kernel**: rho and vmat are separate kernel launches. A fused kernel could avoid re-evaluating AO values, but the two operations have different workgroup geometries.

#### Current bottleneck analysis

`rho` is the bottleneck (0.14s vs 0.089s for vmat on benzene). The rho kernel uses `(gTile, iTile)` 2D workgroups with WGS=64. Each workgroup:
- Iterates over ALL j-atoms (inner loop `for jTile = 0..natoms`)
- Evaluates j-atom radial functions cooperatively into `wfRj[NPTILE][NATILE][MAX_SHELL]`
- Loads DM block `dm_blk[NATILE][NATILE][MAX_AO_ATOM][MAX_AO_ATOM]` from global memory
- Each thread (one grid point × one i-atom) contracts over all j-atoms

For benzene: `ceil(143560/64) * ceil(12/2) = 2243 * 6 = 13458` workgroups, each iterating over 12 atoms in 6 jTile steps. The DM block load is `2*2*16*16 = 1024 floats = 4 KB` per jTile per iTile iteration, loaded from global memory every time.

### 4.2 Operation 2: `rho[iG]` → XC potential `vxc[iG]`

**Name in PySCF**: `eval_xc_eff` (CPU, libxc) / `pbe_xc_*` (GPU)

**Formula**:

```
LDA :  vxc[iG] = dE/drho  at rho[iG]
GGA :  exc[iG], vrho[iG], vsigma[iG] = f(rho, |grad rho|²)
       then wv[0] = weight * vrho
            wv[1..3] = weight * 2 * vsigma * grad_rho[1..3]
```

**Is it a bottleneck?** **YES for GGA.** The libxc call is CPU-only and requires `rho` to be on the host as `float64`. Every time we download `rho` from GPU → CPU and upload `wv` back, we pay a **PCIe round-trip**.

#### Variants

| Implementation | File | Hardware | Status | Notes |
|----------------|------|----------|--------|-------|
| `ni.eval_xc_eff()` (libxc) | `pyscf/dft/libxc.py` | **CPU** | **Active** (on-the-fly path) | General-purpose, any XC functional. **Transfer bottleneck** |
| `pbe_xc_f32` | `pbe.cl:287` | **GPU** | Active (precomputed path only) | PBE-only, float32, auto-generated from libxc maple2c |
| `pbe_xc_f64` | `pbe.cl:602` | **GPU** | Active (precomputed path only) | PBE-only, float64, slower |
| `compute_wv_gga_f32` | `pbe.cl:315` | **GPU** | Active (precomputed path only) | Computes `wv = weight * [vrho, 2*vsigma*grad]` on GPU |
| `compute_wv_gga_f64` | `pbe.cl:630` | **GPU** | Active (precomputed path only) | Same, float64 |

**Not finished**: Meta-GGA, hybrid functionals, range-separated hybrids.

#### What is NOT done

- **GPU PBE is NOT wired into the on-the-fly path**. The on-the-fly path always downloads rho to host, evaluates XC on CPU via libxc, then uploads wv back. This adds:
  - D2H copy: ~0.002s
  - CPU libxc: ~0.01s
  - H2D copy: ~0.001s
  - Total overhead: ~0.013s per cycle
- For large systems, the CPU libxc call grows: PTCDA (379216 grids) takes ~0.05s on CPU.
- **GPU PBE could eliminate this** by keeping rho on device, evaluating PBE on GPU, and computing wv on GPU — no host round-trip. This requires:
  1. After rho kernel, keep `buf_rho` on device
  2. Launch `pbe_xc_f32` on `buf_rho` → `buf_exc`, `buf_vrho`, `buf_vsigma`
  3. Launch `compute_wv_gga_f32` with `buf_vrho`, `buf_vsigma`, `buf_rho` → `buf_wv`
  4. Only download `exc` (for energy) — 1 scalar sum, not full array
  5. Feed `buf_wv` directly to vmat kernel

### 4.3 Operation 3: `vxc[iG]` → VMAT[nao, nao]

**Name in PySCF**: `_dot_ao_ao` / `_dot_ao_ao_sparse` (CPU) / `vmat_*` kernels (GPU)

**Formula**:

```
LDA :  VMAT[mu,nu] += sum_iG  phi[iG,mu] * wv[iG] * phi[iG,nu]
GGA :  VMAT[mu,nu] += sum_iG  aow[iG,mu] * phi[0,iG,nu]
       where aow = sum_c  wv[c,iG] * phi[c,iG,:]
       then hermi_sum: VMAT += VMAT.T
```

**Is it a bottleneck?** **YES.** This is another O(ngrids·nao²) contraction. For GGA it also needs the 3 gradient components of the AO basis.

#### Variants

| Kernel | File | Layout | Status | Notes |
|--------|------|--------|--------|-------|
| **CPU sparse** | `numint.py` | `_dot_ao_ao_sparse` | correct | Uses pair-screening |
| `vmat_lda_tiled` | `kernels.cl` | `(iTile, jTile*WGS_V)` 2D, WGS=256 | **Active** (tiled path) | Local AO cache, private acc[QPT] |
| `vmat_gga_tiled` | `kernels.cl` | `(iTile, jTile*WGS_V)` 2D, WGS=256 | **Active** (tiled path) | aoI = weighted AO (aow), aoJ = plain AO |
| `vmat_lda_pair` | `kernels.cl` | `(ia, ja*WGS_V)` 2D, WGS=256 | **Active** (pair path) | One atom pair per workgroup |
| `vmat_gga_pair` | `kernels.cl` | `(ia, ja*WGS_V)` 2D, WGS=256 | **Active** (pair path) | |
| `vmat_lda_onthefly` / `vmat_gga_onthefly` | `kernels.cl` | legacy | works, limited to ≤128 | High register pressure |
| `vmat_lda_precomp_pair` | `kernels.cl` | Precomputed AO | **Active** (precomputed path) | |
| `vmat_gga_precomp_pair` | `kernels.cl` | Precomputed AO | **Active** (precomputed path) | |
| `vmat_lda_precomp_tiled` / `vmat_gga_precomp_tiled` | `kernels.cl` | Precomputed AO | **Active** | |
| GEMM path (`matmul_gpu_buf`) | `xc_grid.py` | Materialized AO | **Active** (materialized Hermite AO) | `aow^T @ ao` via tiled GEMM |

**Not finished**: CPU-style sparse pair-screening on GPU (atom-blocked pair lists).

#### vmat kernel design (current optimized — tiled path)

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

**Key optimization**: AO values are unfolded once into local memory. No redundant radial or angular computation. Each thread reads from local cache, not re-evaluating.

---

## 5. Three XC Paths (Implementation Variants)

### Path A: Precomputed GTO (CPU AO eval → GPU GEMM)

```
setup:  CPU eval_ao() → ao[ngrids, nao] → upload to GPU (62 MB for benzene)
cycle:  upload DM → GEMM(ao, DM) → aodm → contract → rho → D2H → libxc → H2D wv
        → GEMM(aow^T, ao) → vmat → D2H
```

- **Pros**: Simple, uses cuBLAS-style GEMM, AO values computed exactly by PySCF CPU
- **Cons**: AO array is huge (62 MB LDA, 249 MB GGA for benzene), upload cost, device memory limit
- **When to use**: Small systems where AO fits in device memory
- **Files**: `xc_grid.py: setup_precomputed_gto()`, `nr_rks_precomputed_gto()`

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
- **Files**: `xc_grid.py: nr_rks_hermite_ao()`, `ao_hermite.py: OpenCLAOHermiteEvaluator`

### Path C: On-the-fly Hermite (fused kernels, no AO materialization) ← CURRENT FOCUS

```
setup:  build Hermite tables → upload (256 KB, static)
cycle:  rho_kernel(DM, hermite_tables, coords) → buf_rho  [AO eval fused inside]
        D2H rho → libxc → H2D wv
        vmat_kernel(wv, hermite_tables, coords) → buf_vmat  [AO eval fused inside]
        D2H vmat
```

- **Pros**: No AO materialization (saves 62-249 MB device memory), AO values computed on-the-fly from tiny Hermite tables (256 KB), naturally exploits atom-tile locality
- **Cons**: More complex kernels, redundant AO eval between rho and vmat kernels (they re-evaluate the same radial functions)
- **When to use**: Large systems, or when device memory is limited
- **Files**: `xc_grid.py: setup_onthefly()`, `nr_rks_hermite_onthefly()`, `kernels.cl: rho_*_tiled, vmat_*_tiled`

### Path selection logic

```python
# In rks.py setup_gpu():
if xc_path == 'precomputed':
    plan = setup_precomputed_gto(mol, grids, xc)  # Path A
elif xc_path == 'onthefly':
    plan = setup_xc_grid_gpu(mol, grids, xc)       # Path C

# In rks.py get_veff():
if backend & 2:
    if xc_path == 'precomputed':
        n, exc, vxc = plan.nr_rks_precomputed_gto(dm)  # Path A
    else:
        n, exc, vxc = plan.nr_rks_hermite_onthefly(dm) # Path C
```

### Decision tree

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

## 6. AO Evaluation — Where does `phi[iG, iAO]` come from?

This is the only operation that **could** be done once per SCF loop (precomputed),
but we also support on-the-fly reconstruction for memory savings.

### 6.1 CPU GTO evaluation (`eval_ao`)

- **Path**: `pyscf.gto.eval_gto` → libcint C code
- **Output layout**: `[ncomp, nblk, nao]` with **Fortran-ordered** inner `[nblk, nao]` slabs.
- **Cost**: expensive, but only once if precomputed.
- **Problem for GPU**: the Fortran-order slabs are **not** what our row-major GPU kernels expect. We currently copy/astype them into C-contiguous `float32` buffers.

### 6.2 GPU Hermite interpolation (atom-block)

- **Path**: `ao_hermite.py::OpenCLAOHermiteEvaluator`
- **Kernel**: `eval_ao_mapped_hermite_cart_atom` / `eval_ao_mapped_hermite_cart_deriv1_atom`
- **Thread layout**: `(grid_point, atom)` 2D dispatch
- **Method**: each thread evaluates ALL radial channels for ONE atom at ONE grid point.
  Radial part via cubic Hermite spline on mapped log grid `u = log1p(r/r0)`.
  Angular part is explicit s/p/d/f Cartesian unroll.
- **Output**: Cartesian AOs on GPU → multiplied by `c2s` (Cartesian→spherical) via GEMM.
- **Cost**: moderate compute, but no large `AO` buffer stored.

### 6.3 On-the-fly (no AO materialization)

- **Path**: `xc_grid.py::nr_rks_hermite_onthefly`
- **Kernel**: `rho_lda_pair` / `rho_gga_pair` / `vmat_lda_pair` / `vmat_gga_pair`
- **Method**: each thread evaluates AOs on-the-fly, contracts immediately with DM
  (for rho) or accumulates to VMAT (for vmat). No `AO[ngrids, nao]` array exists.
- **Trade-off**: saves memory (no AO buffer), but each thread rebuilds AOs → high
  register pressure and redundant radial interpolation across threads.

---

## 7. CPU AO Memory Layout Problem

### 7.1 The libcint convention

PySCF stores basis set information in the `mol` object using the **libcint convention**:
- `mol._bas[nbas, BAS_SLOTS=8]`: per-shell info (atom, angular momentum, nprim, nctr, ptr_exp, ptr_coeff)
- `mol._env[]`: flat array of exponents and contraction coefficients
- `mol._atm[natm, ATM_SLOTS=6]`: per-atom info (nuclear charge, coordinates pointer)

This layout is **optimized for serial CPU evaluation** where you loop over shells, then primitives, then contracted functions. It is **not suitable for GPU** because:

1. **Indirect indexing**: To get AO value for atom `ia`, shell `ib`, you must: `bas[ib*BAS_SLOTS+ATOM_OF]` → `atm[atom*ATM_SLOTS+PTR_COORD]` → `env[ptr]`. Three levels of indirection, no coalescing.

2. **Variable-length shells**: Different shells have different `nprim` and `nctr`. No fixed stride. GPU kernels need `if` guards or worst-case loops.

3. **No atom-grouped layout**: Shells are ordered by shell index, not by atom. To get all AOs for atom `ia`, you must search all shells for `bas[ib].atom == ia`.

4. **Spherical/Cartesian mismatch**: PySCF works in spherical harmonics (`nao`), but the Hermite interpolation naturally produces Cartesian AOs (`ncart`). The `c2s` matrix converts between them.

### 7.2 The Fortran-order problem

PySCF `eval_ao` returns:

```python
LDA:  ao = mol.eval_gto('GTOval_sph', coords)   # shape [nblk, nao], order='F'
GGA:  ao = mol.eval_gto('GTOval_sph_deriv1', coords)  # shape [4, nblk, nao], order='F'
```

The inner `[nblk, nao]` matrix is **Fortran-contiguous** (column-major).
This is because libcint writes data in column-major order for compatibility with
BLAS/LAPACK routines used elsewhere in PySCF.

Our GPU kernels are row-major:

```c
// Row-major: phi[iG, iAO] = ao[iG * nao + iAO]
// Consecutive threads (varying iG) stride by nao -> non-coalesced
```

If we naively copy the Fortran-ordered CPU buffer to GPU, **consecutive threads
read non-consecutive memory** (stride `nao * sizeof(float)`), destroying GPU
memory throughput.

### 7.3 Current workaround

```python
ao_staging[c][ip0:ip1] = ao[c].astype(np.float32)   # copy + cast -> C-contiguous
```

`astype(np.float32)` creates a **new C-contiguous array**, which is then uploaded.
This is correct but adds:
1. A temporary `float64` → `float32` allocation per block
2. A host-side memcpy

### 7.4 The solution: atom-grouped Hermite format (already implemented)

The `MappedHermiteRadialBasis` class in `radial_hermite.py` reorganizes the basis into a **GPU-friendly atom-grouped format**:

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
- **Fixed-size per-atom tiles** (`MAX_AO_ATOM=16` or 15) with zero-padding
- **Coalesced reads** from `rad_node` (interleaved value+slope, contiguous per channel)
- **No shell search** — direct index `iao_l = il*MAX_AO_ATOM + a`

### 7.5 What remains problematic

- The `c2s` (Cartesian→spherical) transform is still done on the host as a matrix multiply (`c2s.T @ vmat_cart @ c2s`). For large `ncart`, this is a non-trivial host operation.
- DM is uploaded in spherical basis and transformed to Cartesian on host (`c2s @ dm @ c2s.T`) before the rho kernel. This could be done on GPU.
- The `MAX_AO_ATOM` constant (16) limits the basis set size per atom. Atoms with more than 16 Cartesian AOs (e.g. heavy atoms with f-functions) require increasing this constant and recompiling.

### 7.6 Future improvement options

**Option A**: Keep AO on GPU in **transposed layout** `[nao, ngrids]`.
Then the contraction kernels read with stride-1 across threads.
All our `contract_rho` and `scale_aow` kernels would need to be rewritten for
`phi[iAO, iG]` layout.

**Option B**: Evaluate AOs **directly on GPU** in row-major order.
The Hermite path already does this (`eval_ao_mapped_hermite_cart_atom`).
The precomputed-GTO path still downloads from CPU because libcint is the
reference implementation.

**Option C**: libcint custom output format. Unlikely to be accepted upstream.

---

## 8. DF J/K Contraction

### 8.1 J matrix (Coulomb)

```python
# CPU math:  vj = unpack_tril( dmtril @ cderi.T @ cderi )
# GPU path:
matmul_gpu_buf(bufDmtril, bufCderi, bufTmp,      nset, naux, nao_pair, transpose_B=True)
matmul_gpu_buf(bufTmp,     bufCderi, bufVjPacked, nset, nao_pair, naux)
_unpack_tril_batched_to_buf_gpu(bufVjPacked, bufVjFull, nset, nao, nao_pair)
```

- `cderi` uploaded **once** and cached.
- `bufCderiFull` (unpacked `[naux, nao, nao]`) is built on first K call and cached.

### 8.2 K matrix (exchange)

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
- **Fix needed**: implement tiled local-memory transpose (32×32 tile).

---

## 9. Timing / Profiling Methodology

### 9.1 GPU timing (queue.finish)

**Never** time GPU kernels with `time.perf_counter()` alone.
OpenCL commands are **asynchronous**; they return immediately after enqueue.

```python
# CORRECT — per-stage timing used in our code
t0 = time.perf_counter()
cl.enqueue_nd_range_kernel(queue, knl, global_size, local_size)
queue.finish()          # <-- BLOCK until GPU is done
stage_time = time.perf_counter() - t0
```

Current code uses two patterns:

1. **`_gpu_sync(queue)` = `queue.finish()`** — used in `nr_rks_hermite_onthefly()` profile path
2. **`cl.enqueue_copy(...).wait()`** — used for D2H copies, implicitly waits

The timing instrumentation in `nr_rks_hermite_onthefly()` uses `_time.perf_counter()` with `_gpu_sync()` before each `_timing_record()`. This is correct.

**Warning**: The older inline timing in the non-setup path uses `_tick`/`_tock` with `self.queue.finish()`. This is also correct but should be unified with the profile-based approach.

### 9.2 Host-side timing stages

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

### 9.3 Key metric: `gpu_total` vs `host_total`

- `gpu_total >> host_total` → good, GPU is doing the work.
- `host_total >> gpu_total` → bad, Python / PCIe / libxc dominates.
  Likely causes: per-block D2H of rho, too many kernel launches, small matrices.

---

## 10. Performance Results

### 10.1 Current timings (on-the-fly path, NVIDIA GTX 1650)

| System | natoms | ncart | ngrids | rho | vmat | Total OTF | CPU ref |
|--------|--------|-------|--------|-----|------|-----------|---------|
| water | 3 | 114 | 34576 | 0.006s | 0.027s | 0.18s | 0.03s |
| benzene | 12 | 114 | 143560 | 0.140s | 0.089s | 0.58s | 0.49s |
| pentacene | 36 | 226 | 321784 | 1.182s | 0.570s | 2.98s | 4.53s |
| PTCDA | 38 | 286 | 379216 | 1.896s | 0.859s | 4.33s | 8.84s |

### 10.2 Breakdown for benzene (on-the-fly path)

```
setup0:     0.002s   (buffer allocation, DM transform)
rho:        0.140s   ← BOTTLENECK
rho_copy:   0.002s   (D2H)
libxc:      ~0.01s   (CPU PBE evaluation, not timed separately)
wv_upload:  0.001s   (H2D)
vmat:       0.089s
vmat_copy:  0.001s   (D2H)
c2s:        ~0.001s  (Cartesian → spherical)
```

### 10.3 vmat optimization history

| Design | vmat time (benzene) | Issue |
|--------|---------------------|-------|
| abTile hybrid (3D) | 5.17s | 57x redundant radial eval, 6x wasted angular unfolding |
| Local AO cache + private acc | 0.089s | **58x faster** — current design |

### 10.4 Test results (float32 accuracy)

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

## 11. Known Bottlenecks (ranked by impact)

| # | Bottleneck | Where | Impact | Fix |
|---|------------|-------|--------|-----|
| 1 | **rho kernel: no grid screening** | `rho_*_tiled` on-the-fly | All atoms evaluated for every grid tile | Wire in `grid_screen.py` — 5-10x speedup for sparse systems |
| 2 | **PCIe round-trip for rho + wv** | `nr_rks_hermite_onthefly` | ~0.013s benzene, ~0.05s PTCDA | Use GPU PBE (`gpu_xc='pbe_f32'`) or fuse rho+xc+vmat |
| 3 | **GEMM: no register tiling** | `matmul_tiled` | 2–10× slower than cuBLAS | Add 4×4 or 8×8 register tiles per thread; shrink workgroup to 16×16 |
| 4 | **Strided memory in `contract_rho` / `scale_aow`** | `contract_rho_*_from_aodm`, `scale_aow_*` | Non-coalesced reads (stride `nao`) | Transpose AO to `[nao, ngrids]` or swap kernel launch dims |
| 5 | **Naive transpose in K-build** | `transpose_k_buf1_batched` | Non-coalesced read+write | Local-memory tiled transpose (32×32 tile) |
| 6 | **Redundant AO eval between rho and vmat** | on-the-fly kernels | Same radial functions computed twice | Accept (Hermite eval is cheap) or fuse kernels |
| 7 | **c2s transform on host** | `nr_rks_hermite_onthefly` | Host matmul for Cartesian→spherical | Do on GPU via `matmul_gpu_buf` |
| 8 | **DM transform on host** | `nr_rks_hermite_onthefly` | `dm_cart = c2s @ dm @ c2s.T` on CPU | Move to GPU |
| 9 | **Per-call buffer allocation** | legacy `nr_rks_hermite_onthefly` | Driver overhead | Preallocate all buffers in `setup_onthefly()` |
| 10 | **`.wait()` prevents CPU/GPU overlap** | Every `cl.enqueue_copy(...).wait()` | Serializes pipeline | Use event dependencies, out-of-order queue |
| 11 | **On-the-fly register pressure** | `rho_*_onthefly` (legacy) | Private arrays >128 floats/thread → spill | Use pair/tiled kernels instead |
| 12 | **CPU-side numpy ops in DF J/K** | `df_jk.py::get_jk` | `dm_sym`, `dmtril`, `dm_all` built on CPU | Move tril packing to GPU kernel |
| 13 | **Fortran-order AO from CPU** | `setup_precomputed_gto` | Requires copy+transpose per block | Evaluate AOs directly on GPU (Hermite) |
| 14 | **In-order queue** | `__init__.py` | No kernel overlap | `cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE` |

---

## 12. File Index

| File | Purpose |
|------|---------|
| `pyscf/OpenCL/__init__.py` | Context/queue init, `to_device` / `to_host` helpers |
| `pyscf/OpenCL/kernels.cl` | **All OpenCL kernels**: GEMM, rho/vmat contractions, AO eval, unpack, transpose |
| `pyscf/OpenCL/pbe.cl` | GPU PBE functional (`pbe_xc_f32`, `pbe_xc_f64`, `compute_wv_gga_*`) |
| `pyscf/OpenCL/tile_config.py` | Compile-time tile sizes (`NPTILE`, `NATILE`, `WGS_VMAT`, `MAX_ITILE`) |
| `pyscf/OpenCL/xc_grid.py` | **Main harness**: `XCGridPlan`, all `nr_rks_*` variants, buffer management |
| `pyscf/OpenCL/df_jk.py` | **DF J/K harness**: `DFJKPlan`, `get_jk`, triangular unpack |
| `pyscf/OpenCL/ao_hermite.py` | GPU Hermite AO evaluator (atom-block kernels) |
| `pyscf/OpenCL/radial_hermite.py` | CPU build of Hermite radial tables, `MappedHermiteRadialBasis` |
| `pyscf/OpenCL/grid_screen.py` | Grid-point tile atom screening (sphere-AABB, not yet wired in) |
| `pyscf/OpenCL/buffers.py` | `CLBuffer` wrapper (preallocated upload/download) |
| `pyscf/dft/rks.py` | RKS.get_veff — SCF integration, calls GPU plan |
| `pyscf/dft/numint.py` | CPU reference nr_rks, eval_xc_eff, block_loop |
| `pyscf/dft/libxc.py` | libxc interface (CPU XC functional evaluation) |
| `pyscf/dft/xc_deriv.py` | XC derivative transformation (CPU) |
| `expamples_prokop/test_opencl.py` | Quick parity test (XC + DF J/K) |
| `expamples_prokop/test_opencl_hermite_ao.py` | AO Hermite interpolation accuracy test |
| `expamples_prokop/test_opencl_xc_hermite_ao.py` | XC with Hermite AO (full-grid GEMM path) |
| `expamples_prokop/test_opencl_xc_onthefly.py` | On-the-fly Hermite XC benchmark |
| `expamples_prokop/test_opencl_xc_onthefly_scaling.py` | Scaling benchmark (benzene, pentacene, PTCDA) |
| `expamples_prokop/test_opencl_xc_scf.py` | **Rigorous benchmark**: all GPU paths vs CPU libxc |
| `expamples_prokop/profile_dft.py` | Monkey-patch PySCF timers for 1-cycle profiling |
| `expamples_prokop/sweep_opencl_tiles.py` | Tile-size sweep for auto-tuning |
| `doc/vmat_optimization_report.md` | vmat kernel optimization report (abTile → local cache) |

---

## 13. Quick Checklist Before Commit

- [ ] Did you add `queue.finish()` before every GPU stage timing measurement?
- [ ] Did you verify no new buffer allocations happen inside the SCF loop?
- [ ] Did you run `test_opencl.py` and confirm errors < 5e-3?
- [ ] Did you run `test_opencl_xc_scf.py` to see wall-time breakdown?
- [ ] Did you check that `gpu_total` is the dominant stage in the timing?
- [ ] Did you document any new kernel in this file?
