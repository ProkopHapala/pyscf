
CODEMAP:
PySCF DFT GPU Offloading: XC Grid Integration & DF J/K Contraction Execution Paths
https://windsurf.com/codemaps/a799ba46-9fbe-4274-9c90-61cdb0558d65-fe86ab10a43f3d18

PySCF DFT Calculation & Profiling Execution Paths
https://windsurf.com/codemaps/cc0ccfbd-11ed-426a-b442-a935950fcaeb-fe86ab10a43f3d18

---

# USER

perfect! now analyse which function it would make most sense to move to GPU in pyOpenCL and how dificult it is

I guess either 4-center terms get_jk or rhe DF related function or the grid integration, right? 

what is the main chalange/obstacle? We need to know the basis-set and have way to evaluate it. Do we have access to C++/fotran backend? Or it is libcint or libxc which are in the background (binary instaled by pip) ?

---

# GLM 5.2

# GPU Port Analysis: What to Move to OpenCL and How Difficult

## Backend Architecture

PySCF uses **three C libraries** (all pre-compiled `.so` from pip install at `~/.local/lib/python3.10/site-packages/pyscf/lib/`):

| Library | Size | Role |
|---|---|---|
| `libcgto.so` | 1.3 MB | **libcint** — GTO integral evaluation (1e, 2e, 3c, 2c, AO-on-grid). All `int2e`, `int3c2e`, `int2c2e`, `GTOval_*` functions live here |
| `libcvhf.so` | 1.6 MB | **Coulomb/exchange contraction** — `CVHFnr_direct_drv` etc. Takes ERIs + density matrix → J/K matrices |
| `libxc_itrf.so` | 92 KB | **libxc wrapper** — `LIBXC_eval_xc` evaluates XC functionals on grid density |

The C source is in the repo at `pyscf/lib/gto/` (`grid_ao_drv.c`, `fill_int2e.c`, etc.) and includes `cint.h` from libcint. The key data structures are flat arrays: `atm` (int32), `bas` (int32), `env` (float64) — the "libcint convention."

There's already a **GPU path** via `gpu4pyscf` (CUDA/cupy), accessed through [method.to_gpu()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/lib/misc.py:1601:0-1671:14). But no OpenCL backend exists.

## Profile Breakdown by Scenario

### Without DF (4-center ERIs) — pentacene, 25s total

| Function | Time (s) | % | Backend |
|---|---|---|---|
| `getints4c` (libcint 4c ERI) | 15.9 | 63% | `libcgto.so` (C) |
| `_dgemm` (matrix multiply) | 3.2 | 13% | numpy/BLAS |
| [_dot_ao_dm_sparse](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:875:0-899:14) (AO·DM) | 1.6 | 6% | Python+numpy |
| `eval_gto` (AO on grid) | 1.0 | 4% | `libcgto.so` (C) |
| [_dot_ao_ao_sparse](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:835:0-873:14) | 0.9 | 4% | Python+numpy |
| [gen_grid_partition](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/gen_grid.py:386:8-401:25) | 0.8 | 3% | `libcgto.so` (C) |

### With DF — PTCDA, 23s total

| Function | Time (s) | % | Backend |
|---|---|---|---|
| [nr_direct_drv](cci:1://file:///home/prokophapala/git/pyscf/pyscf/scf/_vhf.py:504:0-603:14) (DF J/K contraction) | 8.7 | 37% | `libcvhf.so` (C) |
| [_dot_ao_dm_sparse](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:875:0-899:14) (AO·DM) | 5.7 | 24% | Python+numpy |
| [_dot_ao_ao_sparse](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:835:0-873:14) | 3.5 | 15% | Python+numpy |
| `eval_gto` (AO on grid) | 1.3 | 5% | `libcgto.so` (C) |
| `_dgemm` | 1.2 | 5% | numpy/BLAS |
| [gen_grid_partition](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/gen_grid.py:386:8-401:25) | 1.0 | 4% | `libcgto.so` (C) |
| `_eval_xc` (libxc) | 0.1 | <1% | `libxc_itrf.so` (C) |

## Three Candidates for GPU Port, Ranked

### 1. XC Grid Integration ([nr_rks](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:1073:0-1189:30) / [block_loop](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:1246:4-1265:37)) — **Easiest, moderate impact**

**What**: Evaluate AO basis on grid points → contract with density → get ρ(r) → call libxc → get Vxc → contract back with AO → Vxc matrix.

**Why easiest**:
- The inner loop is **embarrassingly parallel over grid points** — perfect for GPU
- AO evaluation on grid (`eval_gto`) is just computing Gaussian-type orbitals at 3D coordinates — a well-understood mathematical formula: $\phi_i(\mathbf{r}) = \sum_k c_k x^a y^b z^c e^{-\alpha_k |\mathbf{r}-\mathbf{R}|^2}$
- The C source is available: `pyscf/lib/gto/grid_ao_drv.c` — it's ~200 lines of straightforward C
- After AO eval, the rest is dense matrix multiply (AO·DM, AO·Vxc) — trivially GPU-parallel
- libxc evaluation is **pointwise** (each grid point independent) — but it's a C library call; you'd need to either reimplement PBE in OpenCL (simple for GGA) or batch-transfer to CPU for libxc call

**Main challenge**: The [_dot_ao_dm_sparse](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:875:0-899:14) and [_dot_ao_ao_sparse](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:835:0-873:14) functions use a **screening scheme** (`non0tab` — a uint8 mask of which AO shells are non-zero at which grid blocks). This is the sparse-blocked structure in `numint.py:836` and `numint.py:876`. You'd need to replicate this screening logic or use dense matrices (wasteful but simpler).

**Impact**: ~55% of DF time (13s of 23s for PTCDA-DF). This is the **dominant cost with DF**.

### 2. 4-center ERI / [get_jk](cci:1://file:///home/prokophapala/git/pyscf/pyscf/scf/hf.py:954:0-1028:17) (without DF) — **Hardest, highest impact without DF**

**What**: Compute $(ij|kl)$ 4-center electron repulsion integrals and contract with density to get J and K matrices.

**Why hardest**:
- The actual integral evaluation is in **libcint** (a separate C library, compiled into `libcgto.so`). The source for libcint itself is **not in this repo** — it's a dependency. You'd need to either:
  - Reimplement GTO integral evaluation in OpenCL from scratch (very hard — McMurchie-Davidson or Obara-Saika recurrence, angular momentum up to g-functions)
  - Or call libcint on CPU shell-by-shell and only do the contraction on GPU (limited speedup — the integral eval IS the bottleneck, 16s of 25s)
- The `CVHFnr_direct_drv` in `libcvhf.so` does the J/K contraction using the Schwarz-prescreened shell quartet loop — this is complex algorithmic logic with dynamic load balancing, not just a simple kernel
- O(N^4) scaling means the shell quartet loop has ~N^4/8 iterations with heavy branching (screening)

**Impact**: 63% of non-DF time. But DF already eliminates this bottleneck (3x speedup), so porting this is **less urgent** if DF is used.

### 3. DF J/K contraction ([nr_direct_drv](cci:1://file:///home/prokophapala/git/pyscf/pyscf/scf/_vhf.py:504:0-603:14) with 3c integrals) — **Medium difficulty, medium impact**

**What**: After DF decomposition, J/K are computed as:
- $J_{ij} = \sum_P L_{ij}^P \sum_Q L_{kl}^Q D_{kl} \rightarrow$ matrix multiply chain
- $K_{ij} = \sum_P L_{ik}^P D_{kl} L_{jl}^P \rightarrow$ matrix multiply chain

Where $L_{ij}^P$ is the Cholesky-decomposed 3-center integral tensor.

**Why medium**:
- The 3-center integral evaluation (`getints3c` → `int3c2e`) still goes through libcint, but it's only O(N^2·N_aux) and relatively cheap
- The contraction itself is **dense matrix multiply** (`lib.dot`, `_dgemm`) — perfect for GPU
- But the [nr_direct_drv](cci:1://file:///home/prokophapala/git/pyscf/pyscf/scf/_vhf.py:504:0-603:14) function in `libcvhf.so` is a C driver that loops over auxiliary basis blocks and calls `CVHFdot_nr*` contraction functions — you'd need to replace this Python→C call chain

**Impact**: 37% of DF time (8.7s of 23s). The remaining time is the grid integration.

## The Main Obstacle

**You need GTO basis evaluation on GPU.** This is the fundamental building block for both:
- AO-on-grid evaluation (for XC integration) — simpler, just $\phi(\mathbf{r})$ at grid points
- Shell-pair integral evaluation (for ERIs) — much harder, involves recurrence relations

**Good news**: The C source for AO-on-grid evaluation is **in the repo** (`pyscf/lib/gto/grid_ao_drv.c`). It's ~200 lines of straightforward C that computes contracted GTOs at grid points. This is portable to OpenCL.

**Bad news**: The 4-center integral engine (libcint) is **not in the repo** — it's a compiled dependency. The `cint.h` header is included but the libcint source is external. For DF, you only need 3-center integrals, which are simpler but still require libcint.

**For libxc**: The XC evaluation is a simple pointwise function call. For PBE (GGA), the formula is simple enough to reimplement in ~50 lines of OpenCL. For arbitrary functionals, you'd need to keep calling libxc on CPU (transfer density → CPU, eval, transfer Vxc back → GPU).

## Recommended Strategy

1. **Start with XC grid integration** — port `eval_gto` (AO on grid) + the AO·DM and AO·Vxc contractions to OpenCL. Reimplement PBE in OpenCL (it's just a few formulas). This addresses the **dominant cost when using DF** (55% of time).

2. **Then port DF contraction** — the matrix multiply chains for J/K are trivially GPU-parallel. Keep 3-center integral eval on CPU (libcint), transfer L tensor to GPU.

3. **Skip 4-center ERI port** — DF already eliminates this bottleneck. Not worth the enormous effort of reimplementing libcint in OpenCL.

The basis set data (`atm`, `bas`, `env` arrays) is already available as flat numpy arrays — you can pass them directly to OpenCL kernels as buffers. The GTO evaluation formula is well-documented in `grid_ao_drv.c`.

---

# USER

OK, examine deeply what exacly we need to implement for these 2-3 jobs assuming we use DF, so we focus on XC grid integration  DF contraction

1) do we have the reference CPU code (in C++ or fotran?)
2) where exacly we can plug-it in - we should do it in a way which allow us to easily switch between them by single if/flag
3) how we can test it. Ideally we should run both simultanously (flag passing both-e.g. binary mask 1=CPU,2=GPU,3=both) and then we compare and compute error (discrapency) between the to matrixes
4) we should consider what is the format of the arrays. is it just dense matrix [nAO,nAO] or does it have some sparse/block structure? That is important for layour of workgroups and distribution of work among threads on GPU.

---

# GLM 5.2

# Deep Analysis: What Exactly to Implement for GPU OpenCL (DF Scenario)

## Job 1: XC Grid Integration

### 1a) Reference CPU Code — YES, we have it in C

The entire XC grid integration pipeline has **C source in the repo**. Three layers:

**Layer 1: AO evaluation on grid** — [pyscf/lib/gto/deriv1.c](cci:7://file:///home/prokophapala/git/pyscf/pyscf/lib/gto/deriv1.c:0:0-0:0) + [grid_ao_drv.c](cci:7://file:///home/prokophapala/git/pyscf/pyscf/lib/gto/grid_ao_drv.c:0:0-0:0)

The core computational kernel is [GTOcontract_exp0](cci:1://file:///home/prokophapala/git/pyscf/pyscf/lib/gto/deriv1.c:30:0-57:1) at `@/home/prokophapala/git/pyscf/pyscf/lib/gto/deriv1.c:31-58`:
```c
// For each grid point i, for each primitive j:
arr = alpha[j] * rr[i];           // rr = |r - R_atom|^2
eprim = exp(-arr) * fac;
for (k = 0; k < nctr; k++)
    ectr[k*BLKSIZE+i] += eprim * coeff[k*nprim+j];
```
Then [GTOshell_eval_grid_cart](cci:1://file:///home/prokophapala/git/pyscf/pyscf/lib/gto/deriv1.c:126:0-222:1) (`deriv1.c:129-223`) multiplies by Cartesian powers `x^a * y^b * z^c` via a switch on angular momentum `l=0,1,2,3,...`. For `l>3`, it precomputes power arrays and loops over `(lx,ly,lz)` combinations.

For GGA (deriv=1), [GTOcontract_exp1](cci:1://file:///home/prokophapala/git/pyscf/pyscf/lib/gto/deriv1.c:224:0-261:1) and `GTOshell_eval_grid_cart_deriv1` compute AO + gradients (4 components: value + ∂x, ∂y, ∂z).

The driver [GTOeval_sph_drv](cci:1://file:///home/prokophapala/git/pyscf/pyscf/lib/gto/grid_ao_drv.c:450:0-458:1) (`grid_ao_drv.c:222-284`) loops over shells, calls [GTOcontract_exp0](cci:1://file:///home/prokophapala/git/pyscf/pyscf/lib/gto/deriv1.c:30:0-57:1) then [GTOshell_eval_grid_cart](cci:1://file:///home/prokophapala/git/pyscf/pyscf/lib/gto/deriv1.c:126:0-222:1), then converts Cartesian→spherical via `CINTc2s_ket_sph1` (from libcint).

**Layer 2: Sparse dot products** — [pyscf/lib/dft/nr_numint_sparse.c](cci:7://file:///home/prokophapala/git/pyscf/pyscf/lib/dft/nr_numint_sparse.c:0:0-0:0)

Four C functions called from Python via ctypes:
- [VXCdot_ao_dm_sparse](cci:1://file:///home/prokophapala/git/pyscf/pyscf/lib/dft/nr_numint_sparse.c:225:0-303:1) (`:226`) — computes `ao · dm` with shell-pair screening
- [VXCdot_ao_ao_sparse](cci:1://file:///home/prokophapala/git/pyscf/pyscf/lib/dft/nr_numint_sparse.c:888:0-971:1) (`:890`) — computes `ao.T · ao` with screening → Vxc matrix
- [VXCdcontract_rho_sparse](cci:1://file:///home/prokophapala/git/pyscf/pyscf/lib/dft/nr_numint_sparse.c:973:0-1099:1) (`:975`) — computes `ao · ao → rho` (density at grid points)
- [VXCdscale_ao_sparse](cci:1://file:///home/prokophapala/git/pyscf/pyscf/lib/dft/nr_numint_sparse.c:1101:0-1134:1) (`:1103`) — scales AO by weights for GGA

These use a two-level blocking scheme: `BOXSIZE1_M` grid blocks × `BOXSIZE1_N` shell blocks, with `mask_l1` (uint8) abstracting the `screen_index` to decide which block pairs are non-zero. OpenMP parallelizes over block pairs.

**Layer 3: XC functional evaluation** — `libxc_itrf.so` (C, **source NOT in repo**)

Called via [ni.eval_xc_eff()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:2750:4-2799:19) → [libxc.eval_xc1()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:2746:4-2748:68) → ctypes → `LIBXC_eval_xc`. Input: `rho[4,Ngrids]` (density + gradients). Output: `exc[Ngrids]`, `vxc[4,Ngrids]` (potential terms). This is pointwise — each grid point independent.

### 1b) Where to Plug In

The cleanest insertion point is **[nr_rks](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:1073:0-1189:30)** at `@/home/prokophapala/git/pyscf/pyscf/dft/numint.py:1074-1190`. This function:

1. Calls [block_loop](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:2886:4-2927:42) (line 1127-1129) which yields `ao, mask, weight, coords` per grid block
2. Calls [make_rho](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:2974:8-2992:47) → [eval_rho1](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:226:0-325:14) which does `dot_ao_dm_sparse` + `contract_rho_sparse` → `rho`
3. Calls [eval_xc_eff](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:2750:4-2799:19) → libxc → `exc, vxc`
4. Calls [_dot_ao_ao_sparse](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:835:0-873:14) or [_scale_ao_sparse](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:901:0-934:14) + [_dot_ao_ao_sparse](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:835:0-873:14) → `vmat`

**Proposed approach**: Create a new function `nr_rks_gpu` with the same signature, and dispatch in [get_veff](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/rks.py:36:0-140:14):

```python
# In rks.py:get_veff, line 81:
if ks._numint.backend == 'gpu':
    n, exc, vxc = ni.nr_rks_gpu(mol, ks.grids, ks.xc, dm, max_memory=max_memory)
else:
    n, exc, vxc = ni.nr_rks(mol, ks.grids, ks.xc, dm, max_memory=max_memory)
```

Or even simpler — add a `backend` flag to [NumInt](cci:2://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:2834:0-3000:54) and branch inside [nr_rks](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:1073:0-1189:30) itself. The [NumInt](cci:2://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:2834:0-3000:54) class is at `@/home/prokophapala/git/pyscf/pyscf/dft/numint.py:2723` ([LibXCMixin](cci:2://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:2722:0-2827:32)), and `ks._numint` is the instance.

For the "both" mode (flag=3), [nr_rks](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/numint.py:1073:0-1189:30) would call both paths and compare:
```python
if backend & 1: n_cpu, exc_cpu, vmat_cpu = nr_rks_cpu(...)
if backend & 2: n_gpu, exc_gpu, vmat_gpu = nr_rks_gpu(...)
if backend == 3:
    err = numpy.abs(vmat_cpu - vmat_gpu).max()
    print(f"Vxc max discrepancy: {err}")
```

### 1c) How to Test

**Existing tests**: `@/home/prokophapala/git/pyscf/pyscf/lib/dft/test/test_sparse_dot.py` tests the sparse dot functions against numpy reference. We can write analogous tests for the OpenCL versions.

**Best approach**: The "both" mode (flag=3) is ideal. Run CPU and GPU side by side on the same `dm`, `grids`, `mol`, compare:
- `vmat` (shape `(nao, nao)`) — max abs error should be < 1e-12 (float64 round-off)
- `nelec` (scalar) — relative error < 1e-10
- `excsum` (scalar) — relative error < 1e-10

Test script: use small molecules (H2O, NH3) with PBE, grid level 3, compare Vxc matrix element-by-element.

### 1d) Array Formats

| Array | Shape | Layout | Notes |
|---|---|---|---|
| `ao` (deriv=0) | `(ngrids, nao)` | Fortran-order (column-major) | Each column = one AO function across all grid points |
| `ao` (deriv=1, GGA) | `(4, ngrids, nao)` | Fortran-order | `[value, ∂x, ∂y, ∂z]` × grids × AO |
| `dm` | `(nao, nao)` | C-order (row-major) | Dense, symmetric (hermi=1) |
| `vmat` | `(nao, nao)` | C-order | Dense, symmetric output |
| `rho` (LDA) | `(ngrids,)` | C-order | 1D density per grid point |
| `rho` (GGA) | `(4, ngrids)` | C-order | `[ρ, ∂xρ, ∂yρ, ∂zρ]` |
| `screen_index` | `(ngrids//BLKSIZE, nbas)` | uint8 | 0 = screened out, else ~log-magnitude |
| `pair_mask` | `(nbas, nbas)` | uint8 | 1 = shell pair non-negligible |
| `ao_loc` | `(nbas+1,)` | int32 | Shell offsets into AO array |
| `atm` | `(natm, 6)` | int32 | libcint convention: atom coords etc. |
| `bas` | `(nbas, 8)` | int32 | libcint convention: angular momentum, nprim, nctr, pointers |
| `env` | `(N,)` | float64 | libcint convention: exponents, coefficients, coords |

**Key insight for GPU workgroup layout**: The `ao` array is Fortran-order — `ao[igrid, iao]` has `igrid` as the fast index. The C code processes grid points in blocks of `BLKSIZE=56`. For OpenCL:
- **AO evaluation kernel**: workgroup = 1 shell, threads = grid points in block. Each thread computes `exp(-α·r²) · x^a·y^b·z^c` for one grid point. Natural parallelism.
- **dot_ao_dm kernel**: This is a matrix multiply `ao(ngrids,nao) · dm(nao,nao) → (nao,ngrids)`. On GPU, use BLAS-like tiled GEMM. The screening just means some tiles are zero — can skip them or let them run (wasted work but simpler).
- **dot_ao_ao kernel**: `ao.T · ao → (nao,nao)`. This is a reduction over grid points. Each output element `(i,j)` = `Σ_grid ao[grid,i] * wv[grid] * ao[grid,j]`. Natural for GPU: one workgroup per `(i,j)` shell-pair block, threads reduce over grid points.

**The screening structure** (`screen_index`, `pair_mask`) is a 2-level blocking: shell blocks × grid blocks. On GPU, you can either:
1. **Replicate the sparse logic** in OpenCL (complex, but avoids wasted FLOPs)
2. **Ignore screening, use dense GEMM** (simpler, but ~2-5x wasted work for small molecules). For 10-100 atoms, the grid is ~10k-100k points and nao ~200-1000, so dense GEMM is still fast on GPU.

**Recommendation**: Start with dense (no screening) on GPU. The screening saves ~50% on CPU but GPU has 10x more FLOPs. Dense GEMM on GPU will likely still be faster.

---

## Job 2: DF J/K Contraction

### 2a) Reference CPU Code — YES, in Python + C

The DF J/K is implemented in **pure Python + numpy** at `@/home/prokophapala/git/pyscf/pyscf/df/df_jk.py:280-413`. The main loop ([get_jk](cci:1://file:///home/prokophapala/git/pyscf/pyscf/df/df_jk.py:279:0-412:17)):

1. [dfobj.loop(blksize)](cci:1://file:///home/prokophapala/git/pyscf/pyscf/df/df.py:213:4-241:30) yields blocks of `eri1` (shape `(naux_block, nao_pair)`) from the pre-computed [_cderi](cci:1://file:///home/prokophapala/git/pyscf/pyscf/df/df_jk.py:181:4-184:52) tensor
2. `lib.unpack_tril(eri1)` unpacks triangular → full `(naux_block, nao, nao)`
3. For J: `vj += dmtril.dot(eri1.T).dot(eri1)` — two GEMMs
4. For K: `buf1 = einsum('pij,jk->ipk', eri1, dm)` then `vk += dot(buf1.T, buf2)` — two GEMMs per block

The [_cderi](cci:1://file:///home/prokophapala/git/pyscf/pyscf/df/df_jk.py:181:4-184:52) tensor is built once by [DF.build()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/df/df.py:146:4-198:19) (`@/home/prokophapala/git/pyscf/pyscf/df/df.py:147-199`) which calls `cholesky_eri` (`@/home/prokophapala/git/pyscf/pyscf/df/incore.py:129`). This calls `getints3c` (libcint) for 3-center integrals and `scipy.linalg.cholesky` for decomposition. **This build step stays on CPU** — it's done once and reused across SCF cycles.

The [loop()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/df/df.py:213:4-241:30) method (`df.py:214-242`) just reads blocks from memory or HDF5 file.

### 2b) Where to Plug In

The insertion point is [get_jk](cci:1://file:///home/prokophapala/git/pyscf/pyscf/df/df_jk.py:279:0-412:17) in [df_jk.py](cci:7://file:///home/prokophapala/git/pyscf/pyscf/df/df_jk.py:0:0-0:0). The DF object's [get_jk](cci:1://file:///home/prokophapala/git/pyscf/pyscf/df/df_jk.py:279:0-412:17) is called from `rks.py:get_veff` line 111: [vj, vk = ks.get_jk(mol, _dm, hermi)](cci:1://file:///home/prokophapala/git/pyscf/pyscf/df/df_jk.py:279:0-412:17).

The `ks` object (RKS with DF) has `with_df` attribute. The [get_jk](cci:1://file:///home/prokophapala/git/pyscf/pyscf/df/df_jk.py:279:0-412:17) dispatch is at `@/home/prokophapala/git/pyscf/pyscf/df/df.py:259-263`:
```python
def get_jk(self, dm, hermi=1, ...):
    return df_jk.get_jk(self, dm, hermi, with_j, with_k, direct_scf_tol)
```

**Proposed approach**: Add a `backend` flag to the [DF](cci:2://file:///home/prokophapala/git/pyscf/pyscf/df/df.py:39:0-334:23) class. In [df_jk.get_jk](cci:1://file:///home/prokophapala/git/pyscf/pyscf/df/df_jk.py:279:0-412:17), branch:
```python
def get_jk(dfobj, dm, hermi=0, with_j=True, with_k=True, direct_scf_tol=1e-13):
    backend = getattr(dfobj, 'backend', 'cpu')
    if backend & 2:
        vj_gpu, vk_gpu = get_jk_gpu(dfobj, dm, hermi, with_j, with_k)
    if backend & 1:
        vj_cpu, vk_cpu = get_jk_cpu(dfobj, dm, hermi, with_j, with_k)  # current code
    if backend == 3:
        print(f"J max discrepancy: {abs(vj_cpu - vj_gpu).max()}")
        print(f"K max discrepancy: {abs(vk_cpu - vk_gpu).max()}")
        return vj_cpu, vk_cpu  # or vj_gpu
    if backend == 2:
        return vj_gpu, vk_gpu
    return vj_cpu, vk_cpu
```

### 2c) How to Test

Same "both" mode. Compare `vj` and `vk` matrices (both `(nao, nao)` dense). Expected accuracy: < 1e-12 (it's just GEMM, no approximation).

Test with: PTCDA or pentacene, PBE (no exact exchange, so `with_k=False` for pure DFT). For hybrid functionals, test with_k=True.

### 2d) Array Formats

| Array | Shape | Layout | Notes |
|---|---|---|---|
| [_cderi](cci:1://file:///home/prokophapala/git/pyscf/pyscf/df/df_jk.py:181:4-184:52) (full) | `(naux, nao_pair)` | C-order | `nao_pair = nao*(nao+1)/2` (packed triangular) |
| `eri1` (block) | `(naux_blk, nao_pair)` | C-order | Packed triangular, from [loop()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/df/df.py:213:4-241:30) |
| `eri1` (unpacked) | `(naux_blk, nao, nao)` | Fortran-order | After `unpack_tril` |
| `dm` | `(nao, nao)` | C-order | Dense, symmetric |
| `dmtril` | `(nset, nao_pair)` | C-order | Packed triangular of DM |
| `vj` | `(nao, nao)` | C-order | Dense output (packed during accumulation) |
| `vk` | `(nset, nao, nao)` | C-order | Dense output |

**Key insight**: The DF contraction is **pure dense matrix multiply**. No screening, no sparsity. The loop over `eri1` blocks is just to manage memory — each block does:
- J: `dmtril · eri1.T · eri1` → two GEMMs, accumulate into `vj` (triangular packed)
- K: `unpack(eri1)`, then `einsum('pij,jk->ipk', eri1, dm)` → `dot(buf.T, buf)` → two GEMMs, accumulate into `vk`

**For GPU**: This is trivially parallel. Load entire [_cderi](cci:1://file:///home/prokophapala/git/pyscf/pyscf/df/df_jk.py:181:4-184:52) to GPU memory (for small molecules, `naux × nao_pair × 8` bytes — e.g. PTCDA: ~500 × 5000 × 8 ≈ 20MB, fits easily). Then:
- J: single `dmtril · cderi.T · cderi` → two GEMMs (or one if you precompute)
- K: `unpack(cderi)`, then `einsum` → GEMM chain

No workgroup complexity — just call clBLAS GEMM. The only subtlety is the triangular packing (`unpack_tril`), which is a simple data rearrangement kernel.

---

## Summary: Implementation Roadmap

| Job | Reference C code? | Plug-in point | Array format | GPU difficulty |
|---|---|---|---|---|
| **AO eval on grid** | YES: `deriv1.c:31-223`, `grid_ao_drv.c:222-284` | `numint.py:eval_ao` or new `nr_rks_gpu` | Dense `(ngrids, nao)` F-order + `atm/bas/env` flat arrays | **Medium** — write OpenCL kernel for `exp(-α·r²)·x^a·y^b·z^c`, ~100 lines |
| **Sparse dot products** | YES: `nr_numint_sparse.c:226-1100` | `numint.py:_dot_ao_dm_sparse` etc. | Dense matrices + uint8 screening masks | **Easy** if dense (ignore screening), **Hard** if replicate screening |
| **XC functional eval** | NO (libxc binary only) | `numint.py:eval_xc_eff` | `rho[4,N]` → `vxc[4,N]` pointwise | **Easy** for PBE (~50 lines OpenCL), **Hard** for arbitrary functionals |
| **DF J/K contraction** | YES: `df_jk.py:280-413` (Python!) | `df_jk.py:get_jk` | Dense GEMM, triangular packing | **Easy** — just clBLAS GEMM calls |

**Simplest path to first speedup**: Port DF J/K first (it's just GEMM, use clBLAS). Then port AO eval + dense dot products for XC integration. Reimplement PBE in OpenCL. Skip libxc for now.

---

# USER

Fine, so implement it. use pyOpenCL
and put it to this folder
/home/prokophapala/git/pyscf/pyscf/OpenCL
kernels should be in separate .cl file (perhaps all in one? for simplicity)
NOTE: pyOpenCL code should use single-precision float32 not double (double is slow), this may lower accuracy to 1e-6 but that is fine for us.
also we should do gather operation no scatter, do not use atomic write
also be sure to use local memory with prefered workgroupsize 32 (optimal for nVidia)
make sure you use local memory efficienly (tile-based desing for matrix multiplication, that is classic hellow world tutorial example)
by default initialize nVida OpnCL device if more is available

---

# GLM 5.2

## Results

All three tests pass with **float32-level accuracy (~1e-6)**:

| Test | Metric | Error |
|------|--------|-------|
| XC Grid (PBE/GGA) | nelec relative | 3e-7 |
| XC Grid (PBE/GGA) | vxc max abs | 4.17e-6 |
| DF J/K | J max abs | 4.30e-6 |
| DF J/K | K max abs | 8.42e-6 |
| Full DFT (backend=3) | nelec_err | 4e-9 |
| Full DFT (backend=3) | vxc_max_err | 1.6e-5 |

## Architecture

**Strategy**: AO evaluation stays on CPU (PySCF's `eval_gto`), expensive matrix multiplications go to GPU via tiled GEMM kernels with local memory (TILE=32). XC functional eval stays on CPU (libxc). All GPU computation in float32.

**Key design decisions**:
- [matmul_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:8:0-62:12) in `@/home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py` uses `cl.Buffer` directly with `cl.enqueue_copy` for explicit upload/download control
- [CLBuffer](cci:2://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:3:0-34:30) class in `@/home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py` provides preallocated buffer infrastructure for future optimization
- Kernel objects cached via [_knl()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:12:0-15:25) to avoid repeated `cl.Kernel` retrieval
- `backend` flag: 1=CPU, 2=GPU, 3=both (compare and log discrepancies)

## Files created/modified

- **Created**: `pyscf/OpenCL/__init__.py`, [kernels.cl](cci:7://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/kernels.cl:0:0-0:0), [xc_grid.py](cci:7://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:0:0-0:0), [df_jk.py](cci:7://file:///home/prokophapala/git/pyscf/pyscf/df/df_jk.py:0:0-0:0), [buffers.py](cci:7://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:0:0-0:0)
- **Modified**: [pyscf/dft/rks.py](cci:7://file:///home/prokophapala/git/pyscf/pyscf/dft/rks.py:0:0-0:0) (backend dispatch in [get_veff](cci:1://file:///home/prokophapala/git/pyscf/pyscf/dft/rks.py:36:0-153:14)), [pyscf/df/df_jk.py](cci:7://file:///home/prokophapala/git/pyscf/pyscf/df/df_jk.py:0:0-0:0) (backend dispatch in [get_jk](cci:1://file:///home/prokophapala/git/pyscf/pyscf/df/df_jk.py:279:0-295:25)), [pyscf/df/df.py](cci:7://file:///home/prokophapala/git/pyscf/pyscf/df/df.py:0:0-0:0) (backend attr on [DF](cci:2://file:///home/prokophapala/git/pyscf/pyscf/df/df.py:39:0-335:23) class)
- **Test**: [expamples_prokop/test_opencl.py](cci:7://file:///home/prokophapala/git/pyscf/expamples_prokop/test_opencl.py:0:0-0:0)

## Next steps for performance

The current implementation creates `cl.Buffer` per call. To eliminate this overhead:
1. Preallocate persistent buffers for `dm`, `ao`, `aow`, `vmat` sized to max block
2. Reuse buffers across SCF iterations (upload only when `dm` changes)
3. Pipeline AO evaluation (CPU) with matmul (GPU) using events
4. Move XC functional eval to GPU (PBE kernel already in [kernels.cl](cci:7://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/kernels.cl:0:0-0:0))