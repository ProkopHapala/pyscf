
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

---

# USER

OK, now analyse deeply possible performance problems, divide it to two categories
1) kernels - care especially aboug local memory managment, branching etc. 

I see immediately horrible things

WHAT THE HELL IS THSI CRAP ?!?!?!? 
``
        float fac = 1.0f;
        if (l == 0)      fac = 1.0f;
        else if (l == 1) fac = 1.0f;
        else if (l == 2) fac = 3.0f;  // CINTcommon_fac_sp(2) = 3
        else if (l == 3) fac = 15.0f;
        else if (l == 4) fac = 105.0f;
        else if (l == 5) fac = 945.0f;
        else if (l == 6) fac = 10395.0f;
        else if (l == 7) fac = 135135.0f;
`
this can be just local array!

also thin about how to use group local memory efficiently, you are not using local memroy and tiled desing at all now?

---

2) python harness overhead

- on the fly allocation/deallocation is pretty bad. Once we initialize the system the size of all matrixes is fixed, so within SCF loop we do not need to reallocate matrixes, we do not need to create or destroy buffers

- GPU/CPU transfer is qute costly, we should try to minimize it and do as much work on GPU before we need to download or upload something

- python has huge overhead, loops in python are bad. Can we do instead thing in one numpy array operation 

I see there netest loops

`
        for k in range(nset):
            dm_sym = dms[k] + dms[k].conj().T
            dmtril[k] = _pack_tril_cpu(dm_sym.astype(np.float32))
            dmtril[k, idx*(idx+1)//2+idx] *= 0.5

        # tmp = dmtril * cderi^T  -> [nset, naux]
        # vj_packed = tmp * cderi -> [nset, nao_pair]
        vj_packed = np.zeros((nset, nao_pair), dtype=np.float32)
        for k in range(nset):
            tmp = matmul_gpu(dmtril[k:k+1], cderi, transpose_B=True)  # [1, naux]
            vj_packed[k] = matmul_gpu(tmp, cderi)[0]  # [1, nao_pair] -> [nao_pair]

        # Unpack triangular to full
        vj = np.zeros((nset, nao, nao), dtype=np.float64)
        for k in range(nset):
            vj_full = _unpack_tril_gpu(prg, queue, ctx, vj_packed[k], nao)
            vj[k] = vj_full.astype(np.float64)

``
@df_jk.py 


``
 for ip0 in range(0, ngrids, BLK):
        ip1 = min(ip0 + BLK, ngrids)
        nblk = ip1 - ip0
        coords_blk = grids.coords[ip0:ip1]
        weight_blk = np.ascontiguousarray(grids.weights[ip0:ip1], dtype=np.float64)

        if xctype == 'LDA':
            ao = ni.eval_ao(mol, coords_blk, deriv=0)  # [nblk, nao] CPU

            ao32 = np.ascontiguousarray(ao, dtype=np.float32)
            ao_dm = matmul_gpu(ao32, dm32)  # [nblk, nao]
            rho = np.sum(ao_dm * ao32, axis=1).astype(np.float64)

            exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype='LDA', spin=0)[:2]

            den = rho * weight_blk
            nelec += float(den.sum())
            excsum += float(np.dot(den, exc))

            wv = np.ascontiguousarray(weight_blk * vxc, dtype=np.float32)
            aow = ao32 * wv[:, np.newaxis]  # [nblk, nao]
            vmat_blk = matmul_gpu(aow, ao32, transpose_A=True)
            vmat += vmat_blk.astype(np.float64)

        elif xctype == 'GGA':
            ao = ni.eval_ao(mol, coords_blk, deriv=1)  # [4, nblk, nao] CPU

            ao0_32 = np.ascontiguousarray(ao[0], dtype=np.float32)  # [nblk, nao]
            ao_dm0 = matmul_gpu(ao0_32, dm32)  # [nblk, nao]

            rho = np.zeros((4, nblk), dtype=np.float64)
            rho[0] = np.sum(ao_dm0 * ao0_32, axis=1).astype(np.float64)

            for c in range(1, 4):
                ao_c_32 = np.ascontiguousarray(ao[c], dtype=np.float32)
                ao_dm_c = matmul_gpu(ao_c_32, dm32)  # [nblk, nao]
                rho[c] = (np.sum(ao_dm0 * ao_c_32, axis=1) +
                          np.sum(ao_dm_c * ao0_32, axis=1)).astype(np.float64)
``
@xc_grid.py 

I hate this! it is really necessary? Canno we move this also to GPU?

what I hate the most is crating new arrays inside nested loops!!!! THIS IS REAL CRAP!!!!

``
            for c in range(4):
                wv[c] = w32 * np.ascontiguousarray(vxc[c], dtype=np.float32)
            wv[0] *= 0.5
``

this is not data oriented desing for high-performance numerical code this is some cral like from some stupid web developer without collage degree!!!

do not program now. Spot these crap, and suggest how to correct it and improve it one by one

---

# GLM 5.2

Let me read all the relevant files carefully before giving the analysis.











Here's the full analysis. No code changes, just identification and suggested fixes.

---

# Category 1: Kernels ([kernels.cl](cci:7://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/kernels.cl:0:0-0:0))

## 1.1 — `fac` if-else chain (you spotted this)

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/kernels.cl:73-81` and `:254-262` (duplicated in both `eval_gto_sph` and `eval_gto_sph_deriv1`)

**Problem**: 8-branch if-else chain for angular momentum normalization factor. Causes warp divergence. Duplicated in two kernels.

**Fix**: `__constant` array lookup:
```c
__constant float CINT_FAC[MAX_L+1] = {1.0f, 1.0f, 3.0f, 15.0f, 105.0f, 945.0f, 10395.0f, 135135.0f};
float fac = CINT_FAC[l];
```
Zero branching, one global→constant read.

## 1.2 — Massive branching for Cartesian-to-spherical conversion

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/kernels.cl:94-183`

**Problem**: Completely different code paths for l=0, l=1, l=2, l=3, l>=4. Each work-item in a workgroup may take a different branch (different `ibas` shells have different `l`). This causes:
- **Warp divergence**: threads in same warp execute different branches serially
- **Register pressure**: compiler allocates registers for ALL branches simultaneously, even though only one executes. The l=3 branch alone uses ~10 local variables (`c_xxx` through `c_zzz`, `sqrt3`, `sqrt5`, `sqrt15`). This kills occupancy.
- **l=3 formulas are wrong** (lines 171-175 use division by `c_yzz`, `c_zzz` which can be zero → NaN)

**Fix**: Precompute a Cartesian-to-spherical transformation matrix `c2s[l]` on host, pass as `__constant` array. Then the kernel just does a small matrix-vector multiply for each shell — uniform control flow, no branching on `l`.

## 1.3 — `eval_gto_sph_deriv1` writes zeros for l>=2

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/kernels.cl:293-304`

**Problem**: For l>=2, all 4 components (val + 3 gradients) are set to zero. This is both a correctness bug and a waste — the kernel still runs through the full primitive contraction loop (lines 239-252) computing `ectr`, `dectr_dx/dy/dz`, then throws them away.

**Note**: These kernels are currently **unused** (AO eval moved to CPU), but they still get compiled, wasting build time and potentially affecting other kernels through shared `#define` constants.

## 1.4 — No local memory in `eval_gto` kernels

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/kernels.cl:39-184`

**Problem**: Each work-item (grid point) independently reads `bas[ibas]`, `atm[atm_id]`, and `env[...]` from global memory for every shell. All 32 threads in a workgroup read the **same** shell data. That's 32x redundant global memory reads.

**Fix**: Load shell metadata into `__local` (or `__constant`) memory cooperatively:
```c
__local int s_l, s_nprim, s_nctr, s_ao_off;
__local float s_rx, s_ry, s_rz;
if (get_local_id(0) == 0) {
    s_l = bas[ibas * BAS_SLOTS + ANG_OF];
    // ... etc
}
barrier(CLK_LOCAL_MEM_FENCE);
```
Or better: precompute a compact shell info array on host and pass as `__constant`.

## 1.5 — `contract_rho`: O(nao²) per work-item, `dm` in global memory

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/kernels.cl:468-487`

**Problem**: Each grid point reads the **entire** `dm` matrix (nao×nao floats) from global memory. For 8192 grid points, that's 8192 × 576 × 4 = ~18MB of global reads for a 2.3KB matrix. No local memory, no tiling.

**Fix**: Load `dm` into `__local` memory cooperatively (nao=24 → 576 floats → 2.3KB, fits easily in local memory). Or use `__constant` memory (cached, broadcast to all work-items). Then each work-item reads `dm` from fast local/constant memory.

## 1.6 — `contract_rho_grad`: Recomputes `aodm0` twice

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/kernels.cl:516-547`

**Problem**: The kernel has two separate loops over `j`. In the first loop (lines 516-530), it computes `aodm0_j` and `aodm1_j`. In the second loop (lines 533-547), it recomputes `aodm0_j` again (plus `aodm2_j`, `aodm3_j`). This doubles the global memory reads of `ao[0]` and `dm`.

**Fix**: Single loop computing all 4 `aodm` components simultaneously. Also load `dm` into local memory (see 1.5).

## 1.7 — `matmul_tiled`: No register tiling

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/kernels.cl:317-361`

**Problem**: Each thread computes one output element: `sum += Asub[tx][i] * Bsub[i][ty]` for i=0..31. That's 32 FMAs per tile iteration, with 2 local memory reads per FMA. The compute-to-local-memory ratio is 1:2.

**Fix**: Register tiling — each thread computes a small block (e.g., 4×4 or 8×8) of output. This reduces local memory reads by 4-8x while increasing register usage. Classic optimization for tiled GEMM. For TILE=32 with 4×4 register tiles, each thread does 16 FMAs per tile iteration with only 2 local memory reads (one Asub row, one Bsub column).

## 1.8 — `matmul_tiled`: No double buffering

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/kernels.cl:335-355`

**Problem**: Load and compute are strictly sequential:
```
load tile t → barrier → compute tile t → barrier → load tile t+1 → ...
```
The GPU is idle during the load phase and the load is idle during compute.

**Fix**: Double buffer with 2 local memory tiles:
```
load tile 0 into bufA → barrier → 
  load tile 1 into bufB || compute from bufA → barrier →
  load tile 2 into bufA || compute from bufB → ...
```

## 1.9 — `pbe_xc`: Expensive `pow()` calls

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/kernels.cl:578-579, 586-587, 615`

**Problem**: `pow(rho, 1.0f/3.0f)` is called multiple times per grid point. `pow` is a slow transcendental on GPU (~10-20 cycles).

**Fix**: Use `native_powr(rho, 1.0f/3.0f)` (NVIDIA OpenCL extension, ~4 cycles, less accurate). Or precompute `rho13 = native_powr(rho, 0.333333f)` once and reuse. Also `pbe_c` has `pow(rho, 4.0f/3.0f)` which is just `rho13 * rho`.

## 1.10 — `pbe_xc`: Early return causes warp divergence

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/kernels.cl:571-676, 584, 613`

**Problem**: `if (rho < 1e-20f) { ...; return; }` — if some threads in a warp have rho≈0 and others don't, the entire warp executes both paths.

**Fix**: Compute everything unconditionally, then mask: `exc[igrid] = (rho > 1e-20f) ? exc_val : 0.0f;`

## 1.11 — [unpack_tril](cci:1://file:///home/prokophapala/git/pyscf/pyscf/lib/numpy_helper.py:368:0-452:18): Branching for triangular index

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/kernels.cl:819-825`

**Problem**: `if (i >= j) ... else ...` causes warp divergence.

**Fix**: Branchless:
```c
int maxi = max(i, j), mini = min(i, j);
int idx = maxi * (maxi + 1) / 2 + mini;
full[i * nao + j] = tril[idx];
```

## 1.12 — Dead/unused kernels still compiled

**Where**: All of [kernels.cl](cci:7://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/kernels.cl:0:0-0:0)

**Problem**: Since AO eval moved to CPU, these kernels are never called: `eval_gto_sph`, `eval_gto_sph_deriv1`, `contract_rho`, `contract_rho_grad`, `pbe_xc`, `scale_ao_gga`, `vxc_mat_gga`, `compute_wv_gga`, `compute_nelec_exc`. They waste compilation time and create maintenance burden.

**Fix**: Either remove them (and keep only `matmul_tiled*`, [unpack_tril](cci:1://file:///home/prokophapala/git/pyscf/pyscf/lib/numpy_helper.py:368:0-452:18), [pack_tril](cci:1://file:///home/prokophapala/git/pyscf/pyscf/lib/numpy_helper.py:327:0-365:18), `reduce_sum`) or move them to a separate file that's only compiled when needed. The `pbe_xc` kernel could be useful later if we move XC eval to GPU.

---

# Category 2: Python harness overhead

## 2.1 — Buffer alloc/dealloc per [matmul_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:17:0-72:39) call

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:54-60`

**Problem**: Every [matmul_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:17:0-72:39) call creates 2-3 `cl.Buffer` objects. For GGA path: 7 matmuls per block × ~5 blocks = ~35 buffer allocations per SCF iteration. `cl.Buffer` creation involves OpenCL runtime calls — expensive.

**Fix**: Preallocate persistent buffers at function entry (or in a class), sized to max block dimensions. Reuse across all blocks and all SCF iterations. The [CLBuffer](cci:2://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:3:0-34:30) class already exists but isn't used.

## 2.2 — `dm32` uploaded to GPU every matmul call

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:55-57` (via `COPY_HOST_PTR`)

**Problem**: `dm32` is identical for every block and every matmul. But [matmul_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:17:0-72:39) creates a new buffer with `COPY_HOST_PTR` each time. For GGA: 7 matmuls/block × 5 blocks = 35 uploads of the same 24×24 matrix.

**Fix**: Upload `dm32` to a persistent `cl.Buffer` once at function entry. Pass as `bufA` or `bufB` to all subsequent matmul calls.

## 2.3 — `np.ascontiguousarray` called repeatedly on same data

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:125, 132, 143, 150, 155`

**Problem**: Inside the block loop, `np.ascontiguousarray(ao[c], dtype=np.float32)` is called for c=0..3, creating a new array each time. `ao[c]` is a float64 slice — this always copies. Then `ao[0]` is converted again at line 125 and again at line 155.

**Fix**: Preallocate a single `ao32` array of shape `(4, BLK, nao)` dtype float32. Fill it once per block: `ao32[:] = ao.astype(np.float32)`. Then use `ao32[c]` views — already contiguous, no copy.

## 2.4 — Python for-loops with array creation inside

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:149-156`

**Problem**:
```python
for c in range(4):
    wv[c] = w32 * np.ascontiguousarray(vxc[c], dtype=np.float32)
# ...
for c in range(4):
    ao_c_32 = np.ascontiguousarray(ao[c], dtype=np.float32)
    aow += wv[c:c+1].T * ao_c_32
```
Two Python loops, each creating 4 temporary arrays. The `aow` accumulation does 4 separate `+=` operations with broadcasting.

**Fix**: Vectorize with a single numpy operation:
```python
wv = w32[None, :] * vxc.astype(np.float32)  # (4, nblk) one op
wv[0] *= 0.5
aow = (ao32 * wv[:, :, None]).sum(axis=0)  # (nblk, nao) one op
```
No Python loop, no temporary arrays per iteration.

## 2.5 — [queue.finish()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:29:4-30:27) after every matmul

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:72`

**Problem**: [matmul_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:17:0-72:39) calls [queue.finish()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:29:4-30:27) after every kernel launch. This forces a full CPU-GPU synchronization barrier — CPU blocks until GPU is completely done. No overlap between CPU AO evaluation and GPU computation is possible.

**Fix**: Remove [queue.finish()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:29:4-30:27) from [matmul_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:17:0-72:39). Only synchronize when actually downloading results that the CPU needs. Use `cl.enqueue_copy` (which is blocking by default) for downloads, or use events for fine-grained sync. The CPU can start computing the next block's AO while the GPU computes the current block's matmul.

## 2.6 — Every matmul result downloaded to CPU

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:70-71`

**Problem**: Every [matmul_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:17:0-72:39) call downloads the result to a numpy array. But `aow` is computed on CPU, then immediately uploaded back to GPU for the vmat matmul. That's a pointless download+upload cycle.

**Fix**: Keep `aow` on GPU. The vmat matmul can read directly from the `aow` buffer. Only download `vmat_blk` (the final result per block).

## 2.7 — [df_jk.py](cci:7://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:0:0-0:0): Python loop over `naux` for unpacking cderi

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:70-72`

**Problem**:
```python
for p in range(naux):
    cderi_full[p] = _unpack_tril_gpu(prg, queue, ctx, cderi[p], nao)
```
This is the **worst** performance issue in df_jk. For naux=100, this launches 100 tiny kernels, each with buffer alloc + upload + download + finish + buffer release. That's ~600 OpenCL runtime calls for something that should be one kernel launch.

**Fix**: Write a batched `unpack_tril_batched` kernel that takes `cderi[naux, nao_pair]` and outputs `cderi_full[naux, nao, nao]` in a single launch. Or just unpack on CPU with numpy — `np.tril_indices` + broadcasting would be faster than 100 kernel launches.

## 2.8 — [df_jk.py](cci:7://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:0:0-0:0): cderi uploaded to GPU multiple times

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:59-60, 80-81, 87-89`

**Problem**: In the J path, `cderi` is uploaded twice per nset (once as bufB for `dmtril @ cderi^T`, once as bufB for `tmp @ cderi`). In the K path, `cderi_full` is uploaded twice per nset (as `cderi_2d` and as `cderi_r` — which are the **same data** reshaped). For nao=24, naux=100, that's 4 × 100 × 24 × 24 × 4 = ~2.3MB of redundant uploads per SCF iteration.

**Fix**: Upload cderi/cderi_full once. Reuse the buffer for all matmuls.

## 2.9 — [df_jk.py](cci:7://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:0:0-0:0): K contraction creates huge reshaped contiguous arrays

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:80, 87-88`

**Problem**:
```python
cderi_2d = np.ascontiguousarray(cderi_full.reshape(naux * nao, nao))
buf1_r = np.ascontiguousarray(buf1.transpose(1, 0, 2).reshape(nao, naux * nao))
cderi_r = np.ascontiguousarray(cderi_full.reshape(naux * nao, nao))
```
Three `ascontiguousarray` calls, each copying the full cderi_full tensor. The transpose+reshape creates a new contiguous copy. `cderi_r` is literally the same as `cderi_2d`.

**Fix**: 
- `cderi_2d` and `cderi_r` are the same — compute once, reuse.
- Avoid the transpose by using [matmul_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:17:0-72:39) with `transpose_A`/`transpose_B` flags instead of reshaping.
- Keep `cderi_full` on GPU and use the same buffer for both matmuls.

## 2.10 — [df_jk.py](cci:7://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:0:0-0:0): Python loops over `nset`

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:50-53, 58-60, 64-66, 75-89`

**Problem**: Multiple Python for-loops over `nset`. For nset=1 (typical SCF), this is just overhead. For nset>1 (e.g., TDDFT), it's sequential when it could be batched.

**Fix**: Batch all nset together. [matmul_gpu(dmtril, cderi, transpose_B=True)](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:17:0-72:39) → `[nset, naux]` in one call. [matmul_gpu(tmp, cderi)](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:17:0-72:39) → `[nset, nao_pair]` in one call. The [unpack_tril](cci:1://file:///home/prokophapala/git/pyscf/pyscf/lib/numpy_helper.py:368:0-452:18) can also be batched.

## 2.11 — No CPU/GPU pipelining

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:98-163`

**Problem**: The block loop is strictly sequential:
```
CPU: eval_ao → CPU: astype → GPU: matmul → CPU: download → CPU: eval_xc → CPU: wv → GPU: matmul → CPU: download
```
No overlap between CPU and GPU. While CPU evaluates AO for block i+1, GPU could be computing matmul for block i.

**Fix**: Double-buffered pipeline:
1. Start AO eval for block 0 on CPU
2. Upload block 0 AO to GPU, start matmul
3. While GPU computes block 0, CPU evaluates AO for block 1
4. Download block 0 result, upload block 1, start matmul
5. etc.

This requires removing [queue.finish()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:29:4-30:27) (see 2.5) and using events for synchronization.

## 2.12 — Array allocations inside the block loop

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:102, 107, 117, 118, 128, 132, 147, 148, 153, 159`

**Problem**: ~12+ numpy array allocations per block iteration:
- `weight_blk`, `ao32`, `ao_dm`, `rho`, `wv`, `w32`, `aow`, `vmat_blk`, plus 3× `ao_c_32` and 3× `ao_dm_c`

For 5 blocks: ~60 allocations. Python's memory allocator is not designed for this.

**Fix**: Preallocate all arrays before the loop, sized to `BLK × nao`. Overwrite contents each iteration. Only the last block may be smaller — handle with slicing or zero-padding.

## 2.13 — `weight_blk` double conversion

**Where**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:102, 148`

**Problem**: `weight_blk = np.ascontiguousarray(grids.weights[ip0:ip1], dtype=np.float64)` then later `w32 = weight_blk.astype(np.float32)`. Two copies of the same data.

**Fix**: `w32 = grids.weights[ip0:ip1].astype(np.float32)` directly. Keep `weight_blk` as a view (already float64).

---

# Priority ranking

**Critical** (order-of-magnitude slowdowns):
1. **2.7** — naux Python loop for cderi unpacking (100+ kernel launches)
2. **2.1 + 2.2** — buffer alloc/dealloc + dm re-upload per matmul (35× per SCF iter)
3. **2.5** — [queue.finish()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:29:4-30:27) blocking all CPU/GPU overlap
4. **2.11** — no pipelining (CPU idle while GPU works, GPU idle while CPU works)

**High** (2-5x slowdowns):
5. **2.3 + 2.4 + 2.12** — array allocations and Python loops inside block loop
6. **2.8 + 2.9** — cderi re-uploaded and reshaped multiple times in df_jk
7. **2.6** — aow downloaded then re-uploaded pointlessly

**Medium** (kernel-level, matters for larger systems):
8. **1.7** — matmul no register tiling (2-8x kernel speedup possible)
9. **1.5 + 1.6** — contract_rho dm in global memory (if these kernels get used)
10. **1.1** — fac if-else chain (easy fix, small gain)
11. **1.8** — matmul no double buffering (1.5-2x kernel speedup)

**Low** (unused or minor):
12. **1.2-1.4, 1.9-1.12** — eval_gto and pbe_xc kernels (currently unused)
13. **1.12** — dead kernels (cleanup)

---

# USER

yes, in general I agree that we should explore approach which exploit gPU paralleism. Instread of lunching 100 small kernels, we pack the data in one big array and run one large kernel which each small progrem is processed by one workgroup, or something like that. This is most important. I think curently we have biggest inefficincy on the side of hraness (together with the reallocation of CPU/memory buffers and unnecessayr sync/finish upload/downlod, that cost a lot!)

plese go over it once more focusin on these issule and making more polished plan what to change and how

---

# GLM 5.2

Let me re-read the current state of all files to make sure the plan is accurate.

Now let me check how these are called from the PySCF side to understand the full call chain.

Here's the polished plan, focused on the harness/launch overhead and GPU parallelism exploitation.

# Polished Plan: Eliminate Harness Overhead and Exploit GPU Parallelism

## Guiding principles

1. **One allocation per SCF run, not per call** — all buffer sizes are fixed once `mol` and `grids` are known
2. **One upload per changed input, not per kernel launch** — `dm` changes once per SCF iteration, `ao` changes once per block
3. **One download per final result, not per intermediate** — keep intermediates on GPU
4. **One kernel launch per logical operation, not per element** — batch over `naux`, `nset`, `nblk` using workgroup-per-problem design
5. **No [queue.finish()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:29:4-30:27) except at true data dependencies** — let the queue run asynchronously

## Phase 1: Persistent buffer manager (`CLContext`)

**Problem**: [matmul_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:62:0-116:12) creates+destroys 2-3 `cl.Buffer` per call. [nr_rks_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:64:0-157:30) calls it 7× per block × 5 blocks = 35× per SCF iteration. [df_jk_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:9:0-95:17) calls it 4× per nset + 100× for unpacking. Total: ~200+ buffer alloc/dealloc per SCF iteration.

**Solution**: A `CLContext` class (not to be confused with `cl.Context`) that preallocates all buffers once and reuses them:

```
class CLWorkspace:
    '''Persistent GPU workspace. Allocated once, reused across SCF iterations.'''
    
    def __init__(self, nao, ngrids_max, naux_max, nset_max, xctype):
        # Upload-once buffers (change rarely)
        self.buf_dm      # [nset_max, nao, nao]     — uploaded once per SCF iter
        self.buf_cderi   # [naux_max, nao_pair]    — uploaded once per SCF run  
        self.buf_weights # [ngrids_max]            — uploaded once per SCF run
        
        # Per-block buffers (overwritten each block)
        self.buf_ao      # [4, BLK, nao]           — uploaded each block
        self.buf_ao_dm   # [4, BLK, nao]           — GPU output, stays on GPU
        self.buf_rho     # [4, BLK]                — GPU output, downloaded for XC
        self.buf_aow     # [BLK, nao]              — GPU output, stays on GPU
        self.buf_vmat_blk# [nao, nao]              — GPU output, downloaded per block
        
        # DF J/K buffers
        self.buf_dmtril  # [nset_max, nao_pair]    — CPU compute, upload once
        self.buf_tmp     # [nset_max, naux_max]    — GPU intermediate
        self.buf_vj_packed # [nset_max, nao_pair]  — GPU output
        self.buf_cderi_full # [naux_max, nao, nao] — GPU intermediate (if needed)
```

**Key**: All buffers are `cl.Buffer` with `READ_WRITE` flags. Upload via `cl.enqueue_copy(queue, buf, host_arr)` (non-blocking). Download only when CPU needs the data.

**File**: [pyscf/OpenCL/buffers.py](cci:7://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:0:0-0:0) — extend [CLBuffer](cci:2://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:3:0-34:30) or replace with `CLWorkspace`

## Phase 2: Remove [queue.finish()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:29:4-30:27) and minimize sync points

**Problem**: [matmul_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:62:0-116:12) calls [queue.finish()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:29:4-30:27) at `@/home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:72`. This blocks the CPU until the GPU is completely idle. Called 35+ times per SCF iteration. Each [finish()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:29:4-30:27) is a full pipeline stall.

**Solution**: Remove [queue.finish()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:29:4-30:27) from [matmul_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:62:0-116:12) entirely. The only synchronization needed is:
- **Before download**: `cl.enqueue_copy(queue, host, buf)` is already blocking — it waits for all prior queued commands to complete. No explicit [finish()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:29:4-30:27) needed.
- **Before upload to same buffer**: `cl.enqueue_copy(queue, buf, host)` is also blocking from the host perspective — it enqueues and the queue ordering guarantees correctness.

So the only [finish()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:29:4-30:27) should be at the very end of [nr_rks_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:64:0-157:30) / [df_jk_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:9:0-95:17) before returning to PySCF, and even that is only needed if PySCF code after the call reads the result arrays (which it does, so keep one final [finish()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:29:4-30:27)).

**Change in [matmul_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:62:0-116:12)**: Delete line 72 ([queue.finish()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:29:4-30:27)). When caller provides `bufC`, no download happens — kernel just launches and returns. When caller doesn't provide `bufC`, `cl.enqueue_copy` (line 71) is blocking, so the result is ready.

## Phase 3: Keep intermediates on GPU in [nr_rks_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:64:0-157:30)

**Problem**: Current flow per block (GGA):
```
CPU: eval_ao → CPU: astype → GPU: matmul(ao, dm) → CPU: download ao_dm
CPU: compute rho → CPU: eval_xc → CPU: compute wv → CPU: compute aow
GPU: matmul(aow, ao, transpose_A) → CPU: download vmat_blk
```
That's 2 downloads + 2 uploads per block for intermediates that could stay on GPU.

**Solution**: Two approaches depending on whether we move XC eval to GPU:

### Option A (minimal, keep XC on CPU):
```
CPU: eval_ao → upload ao to GPU (once per block)
GPU: matmul(ao[c], dm) → ao_dm stays on GPU for all c
GPU: element-wise: rho[c] = sum(ao_dm[c] * ao[c])  ← new kernel, pointwise
CPU: download rho → CPU: eval_xc → upload wv to GPU
GPU: element-wise: aow = sum_c ao[c] * wv[c]  ← existing scale_ao_gga kernel
GPU: matmul(aow, ao[0], transpose_A) → download vmat_blk
```
Downloads: 1 (rho) + 1 (vmat_blk) = 2 per block (was 4+).
Uploads: 1 (ao) + 1 (wv) = 2 per block (was 4+).

### Option B (move XC to GPU, eliminate CPU roundtrip):
```
CPU: eval_ao → upload ao to GPU (once per block)
GPU: matmul(ao[c], dm) → ao_dm on GPU
GPU: contract_rho_grad kernel → rho on GPU
GPU: pbe_xc kernel → exc, vxc on GPU
GPU: compute_wv_gga kernel → wv on GPU
GPU: scale_ao_gga kernel → aow on GPU
GPU: matmul(aow, ao[0], transpose_A) → vmat_blk on GPU
GPU: reduce_sum → nelec, excsum on GPU
Download: vmat_blk + nelec + excsum (once per block)
```
Downloads: 1 per block (vmat_blk) + scalar reduction.
Uploads: 1 per block (ao).
**This is the target** — but requires fixing the PBE kernel first.

**For now, implement Option A.** It already eliminates most transfers.

## Phase 4: Batched kernels — one launch instead of N

### 4a: Batched [unpack_tril](cci:1://file:///home/prokophapala/git/pyscf/pyscf/lib/numpy_helper.py:368:0-452:18) (critical for [df_jk_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:9:0-95:17))

**Problem**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:70-72` — Python loop over `naux`, each iteration: buffer alloc + upload + kernel launch + download + finish + buffer release. For naux=100: **600 OpenCL runtime calls**.

**Solution**: New kernel `unpack_tril_batched`:
```c
// One workgroup unpacks one (p, i, j) block
// Global: (round_up(nao, TILE), round_up(nao, TILE), naux)
// Local: (TILE, TILE, 1)
__kernel void unpack_tril_batched(
    __global const float *tril,   // [naux, nao_pair]
    __global float       *full,   // [naux, nao, nao]
    int nao, int nao_pair)
{
    int p = get_global_id(2);     // which auxiliary index
    int i = get_global_id(0);     // row
    int j = get_global_id(1);     // col
    if (p >= naux || i >= nao || j >= nao) return;
    
    int maxi = max(i, j), mini = min(i, j);
    int idx = p * nao_pair + maxi * (maxi + 1) / 2 + mini;
    full[p * nao * nao + i * nao + j] = tril[idx];
}
```
**One kernel launch** for all naux matrices. Upload `tril[naux, nao_pair]` once, download `full[naux, nao, nao]` once.

### 4b: Batched [unpack_tril](cci:1://file:///home/prokophapala/git/pyscf/pyscf/lib/numpy_helper.py:368:0-452:18) for J result

Same kernel, different data. `vj_packed[nset, nao_pair]` → `vj[nset, nao, nao]` in one launch. Replace the Python loop at `df_jk.py:64-66`.

### 4c: Batched matmul over nset in [df_jk_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:9:0-95:17)

**Problem**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:58-60` — Python loop over `nset`, each iteration calls [matmul_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:62:0-116:12) twice (alloc + upload + launch + download + finish × 2).

**Solution**: Batch the matmul. [matmul_gpu(dmtril, cderi, transpose_B=True)](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:62:0-116:12) with `dmtril` shape `[nset, nao_pair]` and `cderi` shape `[naux, nao_pair]` → result `[nset, naux]` in one launch. Then [matmul_gpu(tmp, cderi)](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:62:0-116:12) → `[nset, nao_pair]` in one launch. The existing tiled matmul kernel already handles arbitrary M — just pass `M=nset` (typically 1, but the kernel doesn't care).

Actually this already works with the current [matmul_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:62:0-116:12) — just remove the for-loop and pass the full batched arrays. The kernel treats it as a single `[nset, naux]` matmul.

### 4d: Eliminate cderi_full unpacking for K contraction

**Problem**: The K path at `df_jk.py:68-89` unpacks all `naux` triangular matrices to full `[naux, nao, nao]`, then does two huge matmuls with reshaped copies. This is O(naux × nao²) memory and 3 full copies.

**Alternative**: Write a specialized kernel `df_k_contraction` that does the K contraction directly from packed triangular format:
```
vk[i,j] = sum_P sum_k cderi[P, pack(i,k)] * dm[k,j] * cderi[P, pack(j,k)]
```
One workgroup computes one `(i,j)` pair. Each workgroup iterates over P and k, reading from packed `cderi` directly. No unpacking needed at all.

This is a bigger kernel to write but eliminates:
- 100 unpack kernel launches
- `cderi_full` allocation (naux × nao × nao × 4 bytes)
- 3 `ascontiguousarray` copies
- 2 huge matmul launches with reshaped arrays

**If this is too complex for now**, the fallback is: unpack on CPU with numpy (faster than 100 GPU kernel launches for small nao), then do the two matmuls on GPU with preallocated buffers.

## Phase 5: Preallocate all CPU arrays in [nr_rks_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:64:0-157:30)

**Problem**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:98-163` — ~12 numpy array allocations per block iteration inside the loop.

**Solution**: Preallocate before the loop:
```python
# Before loop:
ao32_buf   = np.zeros((4, BLK, nao), dtype=np.float32)  # if GGA, else (1, BLK, nao)
ao_dm_buf  = np.zeros((4, BLK, nao), dtype=np.float32)
rho_buf    = np.zeros((4, BLK), dtype=np.float64)
wv_buf     = np.zeros((4, BLK), dtype=np.float32)
aow_buf    = np.zeros((BLK, nao), dtype=np.float32)
vmat_blk   = np.zeros((nao, nao), dtype=np.float32)
w32_buf    = np.zeros(BLK, dtype=np.float32)

# Inside loop: overwrite, never allocate
ao32_buf[:, :nblk, :] = ao.astype(np.float32)  # one cast, one copy
# ... use slices [:nblk] everywhere
```

## Phase 6: Vectorize Python loops in [nr_rks_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:64:0-157:30)

**Problem**: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:149-156` — two Python for-loops, each creating 4 temporary arrays.

**Solution** (already identified in 2.4, here's the concrete form):
```python
# Replace lines 147-156 with:
wv_buf[:, :nblk] = w32_buf[:nblk] * vxc.astype(np.float32)  # (4, nblk) broadcast
wv_buf[0, :nblk] *= 0.5
aow_buf[:nblk] = (ao32_buf[:, :nblk] * wv_buf[:, :nblk, None]).sum(axis=0)  # (nblk, nao)
```
Two numpy operations instead of 8+ with Python loop overhead.

Similarly for the GGA rho computation (lines 131-135):
```python
# Replace for c in range(1,4) loop with:
ao_dm_buf[1:4] = [matmul_gpu(ao32_buf[c, :nblk], dm_buf) for c in range(1,4)]  # still 3 matmuls but no ascontiguousarray
rho_buf[0, :nblk] = np.sum(ao_dm_buf[0, :nblk] * ao32_buf[0, :nblk], axis=1)
rho_buf[1:4, :nblk] = (np.sum(ao_dm_buf[0:1, :nblk] * ao32_buf[1:4, :nblk], axis=(2)) +
                       np.sum(ao_dm_buf[1:4, :nblk] * ao32_buf[0:1, :nblk], axis=(2))).astype(np.float64)
```
Actually the 3 matmuls for ao_dm[1..3] are unavoidable unless we write a custom kernel. But we can at least eliminate the `ascontiguousarray` calls by using preallocated `ao32_buf`.

## Phase 7: Upload `dm` once, reuse across all blocks and matmuls

**Problem**: [matmul_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:62:0-116:12) creates a new buffer with `COPY_HOST_PTR` for `dm32` every call. For GGA: 7 matmuls/block × 5 blocks = 35 uploads of the same 24×24 matrix.

**Solution**: In [nr_rks_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:64:0-157:30), upload `dm32` to a persistent `cl.Buffer` once before the block loop:
```python
buf_dm = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, dm32.nbytes, dm32)
# Pass buf_dm as bufB to every matmul_gpu call
```
Same for [df_jk_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:9:0-95:17): upload `cderi` once, pass to all matmul calls.

## Phase 8: Remove `ascontiguousarray` calls

**Problem**: `np.ascontiguousarray(ao[c], dtype=np.float32)` is called 7+ times per block in the GGA path. Each call: check contiguity → allocate new array → copy + cast.

**Solution**: Preallocate `ao32_buf` as float32 C-contiguous. Fill once per block:
```python
ao32_buf[:, :nblk, :] = ao  # numpy handles dtype conversion + contiguity in one shot
```
Then `ao32_buf[c, :nblk]` is already a contiguous float32 view (since the first axis is indexed with a scalar, the remaining slice is contiguous in the last two dims... actually no, it's a slice of a 3D array so it's not contiguous). 

Better: use `np.ascontiguousarray(ao32_buf[c, :nblk])` once and cache, or restructure `ao32_buf` as 4 separate 2D arrays:
```python
ao0_32 = np.zeros((BLK, nao), dtype=np.float32)
ao1_32 = np.zeros((BLK, nao), dtype=np.float32)
ao2_32 = np.zeros((BLK, nao), dtype=np.float32)
ao3_32 = np.zeros((BLK, nao), dtype=np.float32)
# Fill: ao0_32[:nblk] = ao[0]; ao1_32[:nblk] = ao[1]; etc.
```
Each is independently contiguous. No `ascontiguousarray` needed when passing to [matmul_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:62:0-116:12).

## Implementation order (by impact)

| Step | What | Effort | Expected speedup |
|------|------|--------|-----------------|
| 1 | Remove [queue.finish()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:29:4-30:27) from [matmul_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:62:0-116:12) | 1 line | 2-3× (enables async) |
| 2 | Upload `dm` once, pass buffer to all matmuls | ~10 lines | 1.5× (eliminates 35 re-uploads) |
| 3 | Batched `unpack_tril_batched` kernel | ~20 lines CL + ~10 lines Python | 10-50× for df_jk K path |
| 4 | Preallocate all CPU arrays before block loop | ~30 lines | 1.5-2× (eliminates 60 allocs/iter) |
| 5 | Vectorize wv/aow computation | ~5 lines | 1.2× (eliminates Python loops) |
| 6 | Keep `ao_dm` on GPU, download only `rho` | ~20 lines + new kernel | 1.5× (eliminates intermediate downloads) |
| 7 | Batch nset in df_jk (remove for-loops) | ~10 lines | 2× for nset>1 |
| 8 | Upload cderi once in df_jk | ~10 lines | 1.5× (eliminates re-uploads) |
| 9 | Persistent `CLWorkspace` across SCF iters | ~50 lines | 1.3× (eliminates all per-call allocs) |
| 10 | Custom `df_k_contraction` kernel | ~50 lines CL | 5-10× for K path (eliminates unpack+reshape+2 matmuls) |

Steps 1-5 are quick wins with minimal code changes. Steps 6-10 are structural improvements. Step 3 is the single biggest win for [df_jk_gpu](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:9:0-95:17).

Shall I start implementing? I'd suggest starting with steps 1-5 (the quick wins), then running the test to verify correctness before moving to the structural changes.

---

# GPT 5.5 High

# Completed

I implemented the OpenCL performance cleanup and followed the read-only backup memory for `@/home/prokophapala/git/pyscf/pyscf/OpenCL-bak:1`.

## What changed

- **Buffer-only GEMM path**: Added [matmul_gpu_buf()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:17:0-32:15) so callers can launch GEMM directly on existing `cl.Buffer` objects without creating/downloading host arrays `@/home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:18-33`.

- **Removed global queue stalls**: Replaced per-GEMM [queue.finish()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/buffers.py:29:4-30:27) with explicit copy wait only when a host result is needed `@/home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:81-89`.

- **XC harness optimized**:
  - Uploads `dm` once per [nr_rks_gpu()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:66:0-160:30).
  - Preallocates AO, AO-DM, rho, wv, aow, vmat buffers once per call.
  - Computes `rho` and `aow` on GPU.
  - Downloads only `rho` for CPU libxc and final `vmat_blk`.
  - Main implementation: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:114-228`.

- **DF J/K optimized**:
  - Uploads `cderi` once.
  - Batches J GEMMs over `nset`.
  - Uses batched triangular unpack instead of many tiny unpack kernels.
  - Keeps K-path `cderi_full`, `buf1`, transpose, and final GEMM on GPU.
  - Main implementation: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:39-115`.

- **New/updated kernels**:
  - Branchless `unpack_tril`.
  - `unpack_tril_batched`.
  - `transpose_k_buf1`.
  - `contract_rho_lda_from_aodm`.
  - `contract_rho_gga_from_aodm`.
  - `scale_aow_lda`.
  - `scale_aow_gga_split`.
  - Kernel definitions: `@/home/prokophapala/git/pyscf/pyscf/OpenCL/kernels.cl:810-955`.

## Tests run after major steps

Ran after:
- **Quick-win harness changes**
- **Batched triangular unpack**
- **DF buffer reuse**
- **XC preallocation**
- **XC rho/aow GPU kernels**

Final command passed:

```bash
PYTHONPATH=/home/prokophapala/git/pyscf OMP_NUM_THREADS=1 python3 expamples_prokop/test_opencl.py
```

Final summary:
- **Vxc max abs error**: `4.14e-06`
- **J max abs error**: `4.30e-06`
- **K max abs error**: `8.42e-06`

These are still within expected float32 precision.

## Sanity checks

- **OpenCL code whitespace check** passed:
  - `@/home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:1`
  - `@/home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:1`
  - `@/home/prokophapala/git/pyscf/pyscf/OpenCL/kernels.cl:1`

Status: **performance improvements implemented and validated by tests**.

---

# USER

OK, I'm still not very satified how the code looks and especially the python harness. 

One problem is that we use independnet functions, instead of class which can keep persistent arrays and buffers in memory (they should be pre-initialized, bked when the program start, and not rebuild every SCF cycle)

perhaps it would be best to make singke pySCFocl.py with single class which manage the function centrally and build @kernels.cl with all kernels, this is majhor refactro but I think worth it

still you are doing a lot in tight loops
this looks pretty horrible when ngrid is large !
``
for ip0 in range(0, ngrids, BLK):
        ip1 = min(ip0 + BLK, ngrids)
        nblk = ip1 - ip0
        coords_blk = grids.coords[ip0:ip1]
        weight_blk = grids.weights[ip0:ip1]
        if xctype == 'LDA':
``
@xc_grid.py 
this should be batched! all grid point must run in parallel on GPU (each workgroup can process one grid point or something like that.

here again we do some loop which should be pposible to batch possibly @df_jk.py 
``
        for k in range(nset):
            dm32 = np.ascontiguousarray(dms[k], dtype=np.float32)
            cl.enqueue_copy(queue, bufDm, dm32).wait()

            # buf1[p, i, k] = sum_j cderi_full[p, i, j] * dm[j, k]
            # Reshape cderi_full [naux, nao, nao] -> [naux*nao, nao], matmul with dm
            matmul_gpu_buf(bufCderiFull, bufDm, bufBuf1, naux * nao, nao, nao)

            # vk = einsum('ipk,pkj->ij', buf1, cderi_full)
            # buf1_reshaped[i, p*nao+k] = buf1[i, p, k]
            # cderi_reshaped[p*nao+k, j] = cderi_full[p, k, j]
            _knl(prg, 'transpose_k_buf1')(
                queue, (round_up(nao, TILE), round_up(naux * nao, TILE)), (TILE, TILE),
                bufBuf1, bufBuf1R,
                np.int32(naux), np.int32(nao)
            )
            matmul_gpu_buf(bufBuf1R, bufCderiFull, bufVk, nao, nao, naux * nao)
            cl.enqueue_copy(queue, vk_tmp, bufVk).wait()
            vk[k] = vk_tmp.astype(np.float64)
``

note - the main desing question is how to do the batching (packing data to GPU array and unpacking the results) with minimal python overhead. That means with minimal python loops. I think in general it can be done with pre-initialized index arrays and mask allowing us to do simple numpy array operations in runtime instead of costly python loops. These index arrays and masks should be baked at startup of the problem. Figure out how to do it, think deepluy about it is it chalange. only once you think it through concpetulaly providning some pseudocode, we start to implement it

---

# GPT 5.5 High

I’m not going to implement yet. Here is the design I think is correct before touching the code again.# Design Direction

You are right: the current code improves individual hot spots, but the **architecture is still wrong** for repeated SCF use.

The right direction is a persistent OpenCL manager, probably `PySCFOCL`, owning:

- **Context/program/queues**
- **Compiled kernels**
- **Reusable buffers**
- **Problem-specific plans**
- **Baked index/mask arrays**
- **Batched execution paths**

This should replace scattered standalone helpers in `@/home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:18-228` and `@/home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:39-152`.

# Key Correction

The goal is not literally “no loop over grid blocks” in all cases.

For large grids, a block loop may still be necessary because:

- **AO values** are currently evaluated by CPU PySCF.
- Full `ao[4, ngrids, nao]` may be too large for memory.
- `libxc` is still CPU-side, so `rho` must come back unless we port XC.
- GPU memory limits require chunking for big molecules.

But the loop must become a **persistent scheduled streaming loop**, not a Python loop that reallocates and rebuilds objects.

So the design target is:

- **No per-SCF allocation**
- **No per-block allocation**
- **Minimal Python scalar loops**
- **All index/mask logic baked once**
- **All runtime work = upload DM, launch batched kernels, download final results**

# Proposed File Layout

I would make one central manager:

- `@/home/prokophapala/git/pyscf/pyscf/OpenCL/pyscfocl.py:1`

Then keep thin compatibility wrappers:

- `@/home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:1`
- `@/home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:1`

The wrappers just call the singleton/session object.

Example structure:

```python
class PySCFOCL:
    def __init__(self, device=None, dtype=np.float32):
        self.ctx = ...
        self.queue = ...
        self.prg = build_all_kernels()
        self.pool = CLBufferPool(self.ctx)
        self.grid_plans = {}
        self.df_plans = {}

    def prepare_grid(self, mol, grids, xc_code, nao, max_memory):
        key = GridKey(...)
        if key not in self.grid_plans:
            self.grid_plans[key] = GridPlan(self, mol, grids, xc_code, nao, max_memory)
        return self.grid_plans[key]

    def prepare_df(self, dfobj, nao, nset_max):
        key = DFKey(...)
        if key not in self.df_plans:
            self.df_plans[key] = DFPlan(self, dfobj, nao, nset_max)
        return self.df_plans[key]

    def nr_rks(self, mol, grids, xc_code, dm):
        plan = self.prepare_grid(...)
        return plan.run(dm)

    def df_jk(self, dfobj, dms, with_j=True, with_k=True):
        plan = self.prepare_df(...)
        return plan.run(dms, with_j, with_k)
```

# Persistent Buffer Pool

The core primitive should be this:

```python
class CLBufferPool:
    def __init__(self, ctx):
        self.ctx = ctx
        self.buf = {}

    def ensure(self, name, nbytes, flags):
        old = self.buf.get(name)
        if old is None or old.nbytes < nbytes:
            if old is not None:
                old.release()
            self.buf[name] = cl.Buffer(self.ctx, flags, nbytes)
        return self.buf[name]
```

Then every plan does:

```python
buf_dm = pool.ensure("grid.dm", nao*nao*fbytes, READ_ONLY)
buf_ao = pool.ensure("grid.ao", ncomp*blk*nao*fbytes, READ_WRITE)
buf_rho = pool.ensure("grid.rho", ncomp*blk*fbytes, READ_WRITE)
buf_vmat = pool.ensure("grid.vmat", nao*nao*fbytes, WRITE_ONLY)
```

Important: allocation only occurs when shape grows.

# Grid Plan Design

## Current Problem

In `@/home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:133-213`, the loop does:

- Slice coordinates
- Slice weights
- Evaluate AO on CPU
- Upload AO
- GEMM
- Launch rho kernel
- Download rho
- Call libxc
- Upload weights/XC potential
- Launch aow kernel
- GEMM
- Download vmat block

This is better than before but still too much orchestration in Python.

## Better Design

Create a `GridPlan` once per molecule/grid/xc type.

It stores:

- `ngrids`
- `nao`
- `xctype`
- `ncomp = 1 or 4`
- `BLK`
- `nblk_total`
- `block_starts`
- `block_sizes`
- `padded_blk`
- Persistent host arrays:
  - `ao_h`
  - `rho_h`
  - `wv_h`
  - `vmat_h`
- Persistent GPU arrays:
  - `dm_g`
  - `ao_g`
  - `aodm_g`
  - `rho_g`
  - `wv_g`
  - `aow_g`
  - `vmat_g`
  - `vmat_accum_g`
- Optional pinned/mapped host arrays if PyOpenCL platform supports them.

Pseudocode:

```python
class GridPlan:
    def __init__(self, ocl, mol, grids, xc_code, nao, max_memory):
        self.ocl = ocl
        self.mol = mol
        self.grids = grids
        self.xc_code = xc_code
        self.xctype = detect_xctype(xc_code)
        self.nao = nao
        self.ngrids = len(grids.coords)

        self.BLK = choose_blk(nao, ngrids, gpu_mem, max_memory)
        self.starts = np.arange(0, ngrids, self.BLK, dtype=np.int32)
        self.sizes = np.minimum(self.BLK, ngrids - self.starts).astype(np.int32)

        self.prepare_host_workspaces()
        self.prepare_device_workspaces()
        self.prepare_static_device_arrays()

    def prepare_static_device_arrays(self):
        self.weight_g = upload_once(grids.weights.astype(float32))
        self.coords_h = grids.coords
```

Runtime:

```python
def run(self, dm):
    upload dm once
    zero vmat_accum_g

    for block_id in baked_block_ids:
        nblk = self.sizes[block_id]

        ao_h = eval_ao_cpu_into_reused_array(block_id)

        upload ao_h into ao_g
        launch all rho kernels for this block
        download rho_h only
        call libxc on rho_h
        upload wv_h
        launch aow kernel
        launch GEMM contribution
        launch vmat_accum kernel OR download block and accumulate CPU

    download final vmat
    symmetrize if GGA
    return nelec, excsum, vmat
```

This still has a block loop, but it is now a scheduler over preallocated buffers. No allocations. No dynamic indexing. No temporary arrays.

# Can We Process All Grid Points in One GPU Launch?

Only if we move **AO evaluation** and **XC functional evaluation** to GPU.

Otherwise the true dependency is:

```text
coords -> CPU AO eval -> GPU rho -> CPU libxc -> GPU vmat
```

So with current PySCF AO/libxc, full GPU grid parallelism is blocked by CPU stages.

## Option A: Near-Term Practical

Keep CPU AO and CPU libxc, but reduce overhead:

```text
CPU AO block -> GPU rho/aow/GEMM block
```

This is what we should polish now.

## Option B: Full GPU Grid

Port or call GPU kernels for:

- AO basis evaluation
- Becke/grid weights if needed
- XC functional evaluation

Then runtime becomes:

```text
coords_g, weights_g, dm_g -> rho_g -> exc_g/wv_g -> vmat_g
```

Then all grids can run in one or a few global kernels.

This is the real “all grid points in parallel” design, but it is much larger.

# Better Grid Batching Without Python Scalar Logic

Use baked arrays:

```python
self.block_starts = np.arange(0, ngrids, BLK, dtype=np.int32)
self.block_sizes = np.minimum(BLK, ngrids - self.block_starts).astype(np.int32)
self.valid_mask = np.arange(BLK)[None, :] < self.block_sizes[:, None]
```

Then runtime avoids repeated Python shape/index decisions.

For GPU kernels, use padded block size:

```python
padded_ngrid = nblocks * BLK
```

Store AO as:

```text
ao_g[ncomp, nblocks, BLK, nao]
rho_g[ncomp, nblocks, BLK]
wv_g[ncomp, nblocks, BLK]
```

Then kernels use:

```c
block = get_global_id(0)
g_local = get_global_id(1)
i = get_global_id(2)

valid = g_local < block_sizes[block]
global_grid = block_starts[block] + g_local
```

This lets one kernel cover all blocks:

```c
__kernel void contract_rho_gga_batched(
    ao_g, aodm_g, rho_g,
    block_starts, block_sizes,
    int nao, int BLK)
{
    int b = get_global_id(0);
    int g = get_global_id(1);

    if (g >= block_sizes[b]) return;

    ...
}
```

But this requires all AO blocks to be resident on GPU. If AO is CPU-generated, we can still fill `ao_g` block-by-block, but then launch one batched rho/aow/GEMM after uploading many blocks.

# Recommended Grid Refactor Stages

## Stage 1: Persistent `GridPlan`

No algorithm change yet.

- Move all buffers from [nr_rks_gpu()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:90:0-227:30) into a persistent class.
- [nr_rks_gpu()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:90:0-227:30) becomes wrapper.
- No per-call allocation except if `nao/ngrids/BLK` grows.

## Stage 2: Pinned host buffers

Use pinned host arrays for:

- AO
- rho
- wv
- vmat block

Pseudocode:

```python
self.ao_h = cl.array.empty(queue, shape, dtype, allocator=pinned_allocator)
```

This reduces transfer overhead and avoids NumPy allocation churn.

## Stage 3: Batched multiple AO blocks

Instead of:

```text
eval AO block
upload AO block
compute block
```

Use a ring buffer:

```text
CPU fills AO slot 0
GPU computes slot 0 while CPU fills slot 1
GPU computes slot 1 while CPU fills slot 0
```

Pseudo:

```python
for b in blocks:
    slot = b % NSTREAM
    wait_until_slot_free(slot)

    fill_ao_host(slot, b)
    ev_upload = enqueue_upload(slot)
    ev_compute = enqueue_compute(slot, wait_for=[ev_upload])
    mark_slot_busy(slot, ev_compute)
```

This is probably the best near-term win because AO/libxc are CPU bottlenecks.

## Stage 4: GPU accumulation

Instead of downloading `vmat_blk` every block, accumulate on GPU:

```c
atomic_add(vmat_accum[i,j], vmat_blk[i,j])
```

or better:

```text
one vmat per block -> reduction kernel
```

Avoid atomics if possible:

```text
vmat_blocks[nblocks, nao, nao]
reduce_vmat_blocks -> vmat
```

Memory permitting.

# DF Plan Design

## Current Problem

In `@/home/prokophapala/git/pyscf/pyscf/OpenCL/df_jk.py:83-101`, K still loops over `nset`:

```text
for each density matrix:
    upload dm
    cderi_full @ dm
    transpose
    buf1_r @ cderi_full
    download vk
```

This should be batched over `nset`.

## Correct Batched K Math

For each density matrix `s`:

```text
buf1[s, p, i, k] = sum_j cderi[p, i, j] * dm[s, j, k]
vk[s, i, j] = sum_{p,k} buf1[s, p, i, k] * cderi[p, k, j]
```

Pack all density matrices:

```text
dms_g[nset, nao, nao]
```

Then treat first GEMM as a batched operation.

Current GEMM helper is 2D only. Need either:

## Option A: Use strided batched GEMM kernel

Add kernel:

```c
batched_matmul_tiled(
    A, B, C,
    strideA, strideB, strideC,
    M, N, K,
    nbatch)
```

Global dimensions:

```text
global = (ceil(M/TILE)*TILE, ceil(N/TILE)*TILE, nbatch)
local  = (TILE, TILE, 1)
```

Then:

```python
batched_matmul(
    A = cderi_full,        # same for all batches
    B = dms,               # batch-varying
    C = buf1,              # batch-varying
    M = naux*nao,
    N = nao,
    K = nao,
    batch = nset,
    strideA = 0,
    strideB = nao*nao,
    strideC = naux*nao*nao,
)
```

Use `strideA=0` to broadcast `cderi_full` over all density matrices.

Then transpose batched:

```c
buf1_r[s, i, p, k] = buf1[s, p, i, k]
```

Then second batched GEMM:

```python
batched_matmul(
    A = buf1_r,            # batch-varying
    B = cderi_full,        # same for all batches
    C = vk,                # batch-varying
    M = nao,
    N = nao,
    K = naux*nao,
    batch = nset,
    strideA = nao*naux*nao,
    strideB = 0,
    strideC = nao*nao,
)
```

This removes the Python loop over `nset`.

## Option B: Fuse K contraction

A custom kernel computes:

```text
vk[s, i, j] = sum_p sum_k cderi[p,i,k] * dm[s,k,l?] ...
```

But this is basically a large tensor contraction. GEMM-based is simpler and likely faster.

So I recommend **batched GEMM first**, not a hand-fused K kernel.

# DF Baked Index Arrays

For J, packing triangular density matrices still has Python loop:

```python
for k in range(nset):
    dm_sym = dms[k] + dms[k].T
    dmtril[k] = pack_tril(dm_sym)
```

This can be vectorized with baked indices:

At plan init:

```python
tril_i, tril_j = np.tril_indices(nao)
diag_mask = tril_i == tril_j
tril_i_g = upload(tril_i)
tril_j_g = upload(tril_j)
diag_mask_g = upload(diag_mask)
```

Runtime CPU vectorized version:

```python
dmtril = dms[:, tril_i, tril_j] + dms[:, tril_j, tril_i]
dmtril[:, diag_mask] *= 0.5
```

Better GPU version:

```c
__kernel void pack_dmtril_batched(
    dms, dmtril, tril_i, tril_j, diag_mask,
    nset, nao, nao_pair)
{
    int s = get_global_id(0);
    int q = get_global_id(1);
    int i = tril_i[q];
    int j = tril_j[q];

    float v = dms[s*nao*nao + i*nao + j] + dms[s*nao*nao + j*nao + i];
    if (i == j) v *= 0.5f;
    dmtril[s*nao_pair + q] = v;
}
```

Then J becomes fully batched:

```text
upload dms once
pack_dmtril_batched
batched or normal GEMM: dmtril @ cderi.T
batched or normal GEMM: tmp @ cderi
unpack_tril_batched
download vj
```

Actually J can use normal 2D GEMM because `dmtril` is already `[nset, nao_pair]`.

# Proposed `DFPlan`

```python
class DFPlan:
    def __init__(self, ocl, dfobj, nao, nset_max):
        self.ocl = ocl
        self.nao = nao
        self.nao_pair = nao * (nao + 1) // 2
        self.naux = ...
        self.nset_max = nset_max

        self.cderi_h = load_cderi_once(dfobj)
        self.cderi_g = upload_once(cderi_h)

        self.tril_i, self.tril_j = np.tril_indices(nao)
        self.tril_i_g = upload_once(self.tril_i.astype(int32))
        self.tril_j_g = upload_once(self.tril_j.astype(int32))

        self.cderi_full_g = alloc(naux * nao * nao)
        launch unpack_tril_batched(cderi_g -> cderi_full_g)

        self.dms_g = alloc(nset_max * nao * nao)
        self.dmtril_g = alloc(nset_max * nao_pair)
        self.tmp_j_g = alloc(nset_max * naux)
        self.vj_pack_g = alloc(nset_max * nao_pair)
        self.vj_g = alloc(nset_max * nao * nao)

        self.buf1_g = alloc(nset_max * naux * nao * nao)
        self.buf1_r_g = alloc(nset_max * nao * naux * nao)
        self.vk_g = alloc(nset_max * nao * nao)
```

Runtime:

```python
def run(self, dms, with_j, with_k):
    nset = dms.shape[0]
    upload dms once

    if with_j:
        pack_dmtril_batched(dms_g -> dmtril_g)
        matmul(dmtril_g, cderi_g.T -> tmp_j_g)
        matmul(tmp_j_g, cderi_g -> vj_pack_g)
        unpack_tril_batched(vj_pack_g -> vj_g)

    if with_k:
        batched_matmul(cderi_full_g, dms_g -> buf1_g, broadcast_A=True)
        transpose_k_buf1_batched(buf1_g -> buf1_r_g)
        batched_matmul(buf1_r_g, cderi_full_g -> vk_g, broadcast_B=True)

    download requested vj/vk
```

This removes the `for k in range(nset)` loop entirely.

# Required New Kernels

## `batched_matmul_tiled`

Generic and reusable.

```c
__kernel void batched_matmul_tiled(
    __global const float *A,
    __global const float *B,
    __global float *C,
    int M, int N, int K,
    int strideA, int strideB, int strideC)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    int batch = get_global_id(2);

    A += batch * strideA;
    B += batch * strideB;
    C += batch * strideC;

    tiled GEMM body...
}
```

Broadcast is handled by `strideA=0` or `strideB=0`.

## `pack_dmtril_batched`

Uses baked triangular indices.

```c
__kernel void pack_dmtril_batched(
    __global const float *dms,
    __global float *dmtril,
    __global const int *tril_i,
    __global const int *tril_j,
    int nset, int nao, int nao_pair)
{
    int s = get_global_id(0);
    int q = get_global_id(1);
    if (s >= nset || q >= nao_pair) return;

    int i = tril_i[q];
    int j = tril_j[q];

    float v = dms[s*nao*nao + i*nao + j] + dms[s*nao*nao + j*nao + i];
    if (i == j) v *= 0.5f;

    dmtril[s*nao_pair + q] = v;
}
```

## `transpose_k_buf1_batched`

```c
__kernel void transpose_k_buf1_batched(
    __global const float *buf1,
    __global float *buf1_r,
    int naux, int nao)
{
    int s = get_global_id(0);
    int i = get_global_id(1);
    int pk = get_global_id(2);

    int p = pk / nao;
    int k = pk - p * nao;

    buf1_r[(s*nao + i)*naux*nao + pk] =
        buf1[((s*naux + p)*nao + i)*nao + k];
}
```

# Runtime Python Should Look Like This

For DF:

```python
vj, vk = ocl.df_plan(dfobj, nao, nset_max).run(dms, with_j=True, with_k=True)
```

Internally:

```python
upload dms
if with_j:
    launch pack_dmtril_batched
    launch GEMM
    launch GEMM
    launch unpack_tril_batched
if with_k:
    launch batched GEMM
    launch batched transpose
    launch batched GEMM
download vj/vk
```

No Python `for k in range(nset)`.

For XC:

```python
nelec, excsum, vmat = ocl.grid_plan(mol, grids, xc).run(dm)
```

Internally near-term:

```python
upload dm
for slot in pipeline:
    CPU AO eval into pinned host buffer
    async upload AO
    GPU rho
    async download rho
    CPU libxc
    async upload wv
    GPU aow/GEMM
GPU reduce vmat
download vmat
```

Long-term full GPU:

```python
upload dm
launch eval_ao_grid
launch rho
launch xc
launch aow
launch reduce_vmat
download scalar nelec/excsum/vmat
```

# Implementation Order I Recommend

## Step 1: Introduce `PySCFOCL` and plan classes

No numerical algorithm change.

- `PySCFOCL`
- `CLBufferPool`
- `GridPlan`
- `DFPlan`
- Keep wrappers compatible.

This gives us persistent memory and centralized kernel compilation.

## Step 2: Convert DF to fully batched `nset`

Highest value and cleanest.

- Add `batched_matmul_tiled`.
- Add `pack_dmtril_batched`.
- Add `transpose_k_buf1_batched`.
- Remove `for k in range(nset)` in K.

## Step 3: Make DFPlan cache `cderi_full_g`

Avoid rebuilding/unpacking `cderi` every SCF cycle.

## Step 4: Convert XC into persistent `GridPlan`

No AO/libxc port yet.

- Move buffers out of [nr_rks_gpu()](cci:1://file:///home/prokophapala/git/pyscf/pyscf/OpenCL/xc_grid.py:90:0-227:30).
- Add pinned/ring buffers.
- Accumulate `vmat` on GPU.

## Step 5: Optional: batch AO blocks

If memory allows, evaluate multiple AO blocks on CPU, upload them into `[batch, BLK, nao]`, then launch batched kernels once per several blocks.

## Step 6: Full GPU AO/XC

Major project, separate validation.

# Important Constraint

For grids, “all grid points parallel on GPU” is only truly possible after moving AO evaluation and XC evaluation to GPU. Until then, the best possible design is a **persistent streaming pipeline** that hides CPU/GPU latency and avoids allocation/loop overhead.

# My Recommendation

Start with the refactor skeleton and DF batching first.

Reason:

- DF batching is mathematically straightforward.
- It removes the obviously bad `for k in range(nset)` loop.
- It benefits repeated SCF cycles immediately.
- It requires only generic kernels we will reuse elsewhere.
- It is easier to test against the existing DF J/K reference.

Status: **conceptual design ready; I would implement Step 1 + Step 2 first, guarded by tests after each step.**

