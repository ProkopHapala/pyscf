# ρ Projection Kernels — Memory Layout & Gather Design

> **Scope**: Operation 1 only — `DM[iAO,iAO] → ρ[iG]` (LDA + GGA).
> Vmat assembly is out of scope until ρ path is validated.
>
> **Goal**: Two new precomputed variants with coalesced global reads and
> `__local` tiling of DM + AO. Reflect CPU sparse screening where profitable.

---

## 1. Math (unchanged)

```
LDA:  ρ(g) = Σ_{μν} DM_{μν} · χ_μ(g) · χ_ν(g)

GGA:  ρ(g)   = Σ_{μν} DM_{μν} · χ_μ(g) · χ_ν(g)
      ∂ρ/∂x  = Σ_{μν} DM_{μν} · (χ_μ ∂χ_ν/∂x + ∂χ_μ/∂x · χ_ν)    (symmetric DM → factor 2 on cross terms)
      (∂ρ/∂y, ∂ρ/∂z analogous)
```

`χ_μ(g)` = AO μ evaluated at grid point `g`. In PySCF spherical basis (precomp GTO path)
or Cartesian (Hermite path).

**Symmetric DM** (`hermi=1`): only atom-pair tiles with `ia ≤ ja` need full blocks;
mirror for `ia > ja`, or store upper triangle in local DM tile.

---

## 2. What we store today (baseline)

### 2.1 Precomputed GTO (`setup_precomputed_gto`)

| Buffer | Layout | Shape (GGA) | Bytes (benzene cc-pVDZ) |
|--------|--------|-------------|-------------------------|
| `ao0..ao3` | **`[iG, iAO]` row-major** per component | `[4, ngrids, nao]` | 4 × 143560 × 114 × 4 ≈ **262 MB** |

- Built once per SCF (or once per geometry) via CPU `eval_ao(deriv=1)`.
- Components are **spherical** φ, ∂φ/∂x, ∂φ/∂y, ∂φ/∂z from libcint — **not** Cartesian.
- Staging: `ao_staging[c][iG] = ao[c].astype(f32)` → C-contiguous `[ngrids, nao]`.

**Yes — for GGA precomp you currently need 4 planes** `[4, nAO, nGrid]` (stored as 4 separate
`[ngrids, nao]` buffers). There is no separate `dχ/dr`; you store Cartesian derivatives
in the spherical basis (what libcint returns).

### 2.2 Why the current layout is bad for cooperative loads

Existing `rho_*_precomp_{tiled,pair}` load a **tile of grid points** into `aoJ[NPTILE][…]`:

```c
int gbase = gj * nao;           // gj = gTile * NPTILE + pp
aoJ[pp][b] = ao0[gbase + j0 + b];
```

Threads with consecutive `pp` read addresses separated by **`nao` floats** →
**stride-n gather, not coalesced**. This is the main bandwidth bug for precomp ρ.

Single-thread-per-`g` kernel `contract_rho_*_precomp` does:

```c
for (i) for (j) … ao0[g*nao + j]   // fixed g, varying j → coalesced for one thread
```

but repeats the **O(nao²) DM inner loop per grid point** with DM in global memory —
no `__local` DM tile, no workgroup reuse.

### 2.3 On-the-fly Hermite (reference, not precomp)

No `χ` buffer; radial tables `rad_node[nchan, nrad, 2]` (~256 KB) + fused eval in
`rho_*_tiled`. Fast on RTX 3090 (ρ ≈ 5 ms benzene) but recomputes Hermite every pair tile.

---

## 3. Two new variants to implement

| Variant | Name | Stored on GPU | Build cost |
|---------|------|---------------|------------|
| **A** | Full χ precomp, layout-fixed | `χ[c, iAO, iG]` or atom-blocked `χ[c, ia, a, iG]` | CPU `eval_ao` + transpose |
| **B** | Radial-only precomp | `R[c, iG]`, `dR/dr[c, iG]` + static angular metadata | CPU Hermite or GTO radial on grid |

---

## 4. Variant A — Full χ, coalesced layout

### 4.1 Layout options (ranked)

#### **A1 — Transpose: `χ[c, iAO, iG]` (recommended first)**

```
index:  ao_c[ c * (nao*ngrids) + iAO * ngrids + iG ]
```

| Access pattern | Coalescing |
|----------------|------------|
| Workgroup threads `ip = 0..NPTILE-1`, fixed `(c, iAO)` | `iG` consecutive → **fully coalesced** |
| Fixed `iG`, loop `iAO` (single thread) | stride-1 in inner loop over `iAO` at fixed `iG` — use **`χ[iAO, iG]` with `iG` inner** in registers |

**CPU reshuffle once after `eval_ao`** (before upload):

```python
# ao_cpu[c] : (ngrids, nao) F-contiguous from libcint
ao_gpu[c] = np.ascontiguousarray(ao_cpu[c].T)   # (nao, ngrids) C-contiguous
# upload ao_gpu[c] as flat [nao * ngrids]
```

Memory: **same 262 MB** as today for GGA — only index order changes.

#### **A2 — Atom-blocked: `χ[c, ia, a, iG]`**

```
index:  ao[c, ia, a, iG]  →  flat[ ((c*natoms + ia) * MAX_AO_ATOM + a) * ngrids + iG ]
```

- `a ∈ [0, atom_nao[ia])`, pad to `MAX_AO_ATOM` (16).
- Drops `atom_ao0[]` gather offset at runtime — each atom’s AOs are dense in `a`.
- Cooperative load for grid tile: threads `ip` read `χ[c, ja, b, gTile*NPTILE+ip]` → coalesced on `iG`.
- **CPU reorder**: after eval_ao, scatter global `iAO` → `(ia, a)` using `mol.aoslice_by_atom()`.

Extra memory: padding `(MAX_AO_ATOM - atom_nao[ia])` per atom per grid — small for benzene.

#### **A3 — gTile-compressed (sparse, later)**

For each grid tile `t`, store only active atoms’ χ:

```
gtile_atom_off[t+1] - gtile_atom_off[t] = n_active_atoms[t]
chi_packed[c, t, ia_local, a, ip]   ip ∈ [0, NPTILE)
```

Built using `grid_screen.py::build_gtile_atom_lists`. Saves memory on large sparse systems;
variable size per tile; more complex indexing. **Phase 2** after A1 works.

### 4.2 CPU setup pipeline (Variant A1)

```text
# ONCE per geometry (before SCF loop)
grids.build(with_non0tab=True)          # keep for CPU reference / optional mask

for c in 0..ncomp-1:
    ao_cpu[c] = eval_ao block or full grid   # (ngrids, nao) F-order per slab
    ao_gpu[c] = transpose + astype f32       # (nao, ngrids) C-order

upload ao_gpu[c] → buf_chi[c]
upload atom_ao0, atom_nao (if using global iAO indexing)
upload DM once per SCF cycle
```

Optional: fuse transpose + `astype` in one OpenMP parallel loop over `(iAO, iG)` to avoid extra pass.

---

## 5. Variant B — Radial-only precomp

### 5.1 Factorization (Cartesian / Hermite)

For each cartesian AO indexed by `iao`:

```
χ_iao(r) = R_chan(iao)(|r - R_atom|) · A_iao(dx, dy, dz)
```

- `R` depends only on **radial channel** (shell + contraction index) — shared by all angular
  powers in that shell.
- `A_iao` = `x^ix y^iy z^iz` (integer powers, cheap in registers).

**Storage reduction** (per grid point, channels not cart AOs):

| Shell | n_cart | n_radial (per ctr) | χ storage ratio | radial-only planes |
|-------|--------|--------------------|-----------------|---------------------|
| s | 1 | 1 | 1× | 1 |
| p | 3 | 1 | **3×** | 1 |
| d | 6 | 1 | **6×** | 1 |
| f | 10 | 1 | **10×** | 1 |

User’s “1/3 for p, 1/9 for d” is the **cart count ratio** (3 and 6 angular functions share one radial).

For **GGA**, storing `∂χ/∂x, ∂χ/∂y, ∂χ/∂z` triples memory. Instead store:

```
R[c, iG]      — value
dRdr[c, iG]   — dR/dr (Bohr⁻¹)
```

At grid point `g` with `dxyz = r - R_atom`, `r = |dxyz|`, `r̂ = dxyz/r`:

```
χ     = R · A
∂χ/∂x = (dR/dr)(r̂_x) · A + R · ∂A/∂x
```

`∂A/∂x` for monomials is trivial (`ix * x^(ix-1) y^iy z^iz`). **No `ao1..ao3` buffers.**

### 5.2 Layout for radial planes

```
R    [nchan, ngrids]   float32   index: R[c * ngrids + iG]
dRdr [nchan, ngrids]   float32
```

Coalescing: identical to A1 — threads with consecutive `iG` read consecutive memory.

Static (constant) buffers:

```
chan_atom[c], chan_l[c], chan_ang_ixyz[c, 3]   # or cart_shell mapping
cart_chan[iao], cart_ixyz[iao, 3]
```

### 5.3 Build radial on grid (CPU, once per geometry)

```python
for iG in range(ngrids):
    for c in range(nchan):
        r = |coords[iG] - atom_coords[chan_atom[c]]|
        u = log1p(r / r0)
        R[c, iG], dRdr[c, iG] = hermite_eval(u)   # same tables as MappedHermiteRadialBasis
```

Memory (benzene, estimate `nchan ≈ 50–80`):

```
2 × nchan × ngrids × 4  ≈  2 × 70 × 143560 × 4  ≈  80 MB
```

vs GGA full χ: **262 MB** (~3.3× smaller). vs LDA full χ: **69 MB** (similar order).

**Trade-off**: Variant B still pays **angular unfold in registers** each ρ accumulation;
Variant A reads χ directly. B wins when memory bandwidth dominates (large grids, GGA).

### 5.4 Spherical vs Cartesian

Precomp GTO path uses **spherical** `nao`. Variant B is naturally **Cartesian** (Hermite tables).
Match OTF path: work in **Cartesian** `ncart` for ρ, then `c2s` on DM before / `c2s.T @ …` after
(same as `nr_rks_hermite_onthefly`).

---

## 6. Sparse screening — reflect CPU blocks in kernel design

### 6.1 PySCF CPU screening

- `non0tab[grid_blk, shell]` — shell active if any AO in shell is above cutoff on that grid block (`BLKSIZE=56`).
- `pair_mask[shell_i, shell_j]` — skip DM shell pairs.
- Atom-blocked GPU tiles are **coarser** than shell blocks but same idea.

### 6.2 GPU screening (integrate `grid_screen.py`)

Per **gTile** (`NPTILE` consecutive grid points):

1. Bbox grid points → sphere-AABB test each atom → `active_atoms[t]`.
2. Optional pair list: `|R_ia - R_ja| ≤ rcut_ia + rcut_ja`.
3. Kernel loops **`ja ∈ active_atoms[t]`** only, not `0..natoms-1`.

```text
gtile_atom_off[t] : gtile_atom_list[off:off+na] = active atom indices
```

For Variant A3, χ upload can also be **per-gTile packed** (only active atoms) — largest win on big systems.

### 6.3 Dense benzene

Screening removes few atoms (all 12 active in most tiles). Keep dense path as default;
sparse gTile lists as compile-time / runtime flag.

---

## 7. Kernel design — `rho_precomp_gather` (Variant A1)

### 7.1 Ownership model

```
Workgroup:  gTile (1D or 2D)
Thread:     ip ∈ [0, NPTILE)     owns grid point  g = gTile * NPTILE + ip
Optional:   il ∈ [0, NATILE)      owns i-atom slot in iTile (existing tiled pattern)
```

**One grid point per thread** for ρ output; cooperation is for **loading χ and DM**, not for splitting ρ across threads.

### 7.2 Local memory budget (NPTILE=64, NATILE=2, MAX_AO_ATOM=16)

| Buffer | Size (floats) | ~KB |
|--------|---------------|-----|
| `aoJ[NPTILE][NATILE][MAX_AO_ATOM]` | 64×2×16 | 8 |
| `dm_blk[NATILE][NATILE][16][16]` | 2×2×256 | 4 |
| `psum[NPTILE]` reduction | 64 | 0.25 |

Fits easily in 48 KB `__local`.

### 7.3 Pseudocode (LDA, layout `χ[iAO, iG]`)

```opencl
// global: chi [nao, ngrids], dm [nao, nao], rho [ngrids]
// atom_ao0[ia], atom_nao[ia]

__kernel void rho_lda_precomp_gather(
    __global const float *chi,      // iAO * ngrids + iG
    __global const float *dm,
    __global const int *atom_ao0,
    __global const int *atom_nao,
    __global float *rho,
    int nao, int ngrids, int natoms)
{
    const int ip = get_local_id(0);
    const int g  = get_group_id(0) * NPTILE + ip;
    if (g >= ngrids) return;

    __local float aoJ[NPTILE][NATILE][MAX_AO_ATOM];
    __local float dm_blk[NATILE][NATILE][MAX_AO_ATOM][MAX_AO_ATOM];

    float rho_priv = 0.0f;

    // χ at owned grid point g, per i-atom tile (private or small local cache)
    float chiI[NATILE][MAX_AO_ATOM];
    for (int il = 0; il < NATILE; il++) {
        int ia = get_group_id(1) * NATILE + il;   // or loop iTiles inside 1D WG
        int ni = (ia < natoms) ? atom_nao[ia] : 0;
        int i0 = atom_ao0[ia];
        for (int a = 0; a < ni; a++)
            chiI[il][a] = chi[(i0 + a) * ngrids + g];
    }

    for (int jTile = 0; jTile < natoms; jTile += NATILE) {
        // --- Coalesced gather: all threads ip load χ for grid points in tile ---
        for (int jj = 0; jj < NATILE; jj++) {
            int ja = jTile + jj;
            int nj = (ja < natoms) ? atom_nao[ja] : 0;
            int j0 = atom_ao0[ja];
            for (int b = get_local_id(0); b < MAX_AO_ATOM; b += NPTILE) {
                float v = 0.0f;
                if (b < nj)
                    v = chi[(j0 + b) * ngrids + (get_group_id(0) * NPTILE + get_local_id(0))];
                // ^ each thread ip loads own g; for aoJ need all pp in tile:
            }
        }
        // Correct cooperative pattern:
        for (int k = get_local_id(0); k < NPTILE * NATILE * MAX_AO_ATOM; k += NPTILE) {
            int pp = k / (NATILE * MAX_AO_ATOM);
            int jj = (k / MAX_AO_ATOM) % NATILE;
            int b  = k % MAX_AO_ATOM;
            int gj = get_group_id(0) * NPTILE + pp;
            int ja = jTile + jj;
            aoJ[pp][jj][b] = 0.0f;
            if (gj < ngrids && ja < natoms && b < atom_nao[ja]) {
                int j0 = atom_ao0[ja];
                aoJ[pp][jj][b] = chi[(j0 + b) * ngrids + gj];  // COALESCED in pp
            }
        }
        barrier();

        for (int il = 0; il < NATILE; il++) {
            int ia = iTile_base + il;
            // load dm_blk[il][jj][a][b] cooperatively from global dm
            load_dm_tile(dm, dm_blk, ia, jTile, atom_ao0, atom_nao, natoms);
            barrier();

            if (ia < natoms) {
                int ni = atom_nao[ia];
                for (int jl = 0; jl < NATILE; jl++) {
                    int ja = jTile + jl;
                    if (ja >= natoms) continue;
                    int nj = atom_nao[ja];
                    for (int a = 0; a < ni; a++)
                        for (int b = 0; b < nj; b++)
                            rho_priv += chiI[il][a] * dm_blk[il][jl][a][b] * aoJ[ip][jl][b];
                }
            }
            barrier();
        }
    }
    rho[g] = rho_priv;
}
```

**Key fix vs current `rho_lda_precomp_tiled`**: `chi[(j0+b)*ngrids + gj]` with consecutive `gj` across `pp` — not `chi[gj*nao + j0+b]`.

### 7.4 GGA from full χ (A1)

Same gather for `chi0..chi3` with layout `chi_c[c][iAO, iG]`.

Or single buffer:

```
chi_all[c, iAO, iG]  with c=0..3
```

Accumulate `rho, gx, gy, gz` using same `dm_blk` and cross terms:

```c
rho_priv += ai0 * dm_ab * bj0;
gx_priv  += (ai0 * dm_ab * bj1 + ai1 * dm_ab * bj0);
// ...
```

### 7.5 GGA from radial (B)

Replace `aoJ[pp][jj][b]` with on-the-fly:

```c
float R, dRdr, A, dAdx, ...;
eval_cartesian_ao(gj, ja, b, &R, &dRdr, &A, &dAdx, ...);
float bj0 = R * A;
float bj1 = dRdr * rhat_x * A + R * dAdx;
```

Use `R[c(gj,ja,b), gj]` gather (coalesced). **No χ0..χ3 arrays.**

---

## 8. DM tiling in `__local` (gather)

DM is symmetric; atom-blocked tile:

```c
dm_blk[il][jl][a][b] = dm[(i0+a)*nao + (j0+b)];
```

Cooperative load: threads stride over `MAX_AO_ATOM²` entries per atom pair `(ia, ja)`.

**DM is tiny** (`nao² ≈ 52 KB` for benzene) — entire DM can live in `__constant` or
`__local` once per workgroup if `nao ≤ 256`. For large molecules, tile as now.

Optional: store `DM` in **atom-blocked upper triangle** at upload to match tile loops.

---

## 9. Implementation plan (ρ only)

| Step | Task | Verify |
|------|------|--------|
| **1** | CPU `transpose_ao_for_gpu()` in `xc_grid.py`; upload `[nao, ngrids]` | χ coalescing unit test |
| **2** | Kernel `rho_lda_precomp_gather` (A1 layout) | vs CPU `nr_rks` LDA |
| **3** | Extend to `rho_gga_precomp_gather` (4 χ planes or 1 strided) | vs CPU GGA ρ only |
| **4** | Wire into `XCGridPlan` as `fused='gather'`; benchmark vs `tiled` | `test_opencl_xc_scf.py` |
| **5** | CPU build `R`, `dRdr` on grid; kernel `rho_gga_radial_gather` (B) | parity + memory |
| **6** | Wire `grid_screen.py` gTile atom lists (optional) | pentacene / PTCDA |

**Do not** touch vmat kernels until step 3–4 pass parity at `vxc_tol ~ 1e-5` on ρ-derived quantities.

---

## 10. Expected impact (hypothesis)

| Path | Benzene ρ (RTX 3090) | Notes |
|------|----------------------|-------|
| Current `rho_precomp_tiled` | ~60 ms | stride-n AO gather |
| A1 `rho_precomp_gather` | **target 15–30 ms** | coalesced χ + same DM tile |
| B radial gather | **target 10–25 ms** | less DRAM; more ALU |
| OTF tiled (reference) | ~5 ms | no χ DRAM |

Precomp wins when χ reuse amortizes setup (multiple SCF cycles, same geometry) or when
OTF register pressure hurts (large `ncart`).

---

## 11. Quick answers

**Q: GGA precomp needs `[4, nAO, nGrid]`?**  
A: Today **yes** — four spherical derivative planes from `eval_ao(deriv=1)`.
Variant B needs only **`[2, nchan, nGrid]`** (`R`, `dR/dr`) plus static angular metadata.

**Q: Optimal layout for one-thread-one-grid-point + cooperative tile load?**  
A: **`χ[iAO, iG]`** (iG innermost / fastest). Atom-blocked **`χ[ia, a, iG]`** is equivalent
with simpler indexing.

**Q: Gather?**  
A: Yes — χ is **gathered** from global memory into `aoJ[NPTILE][…]` per jTile;
DM is **gathered** into `dm_blk`. Not a dense GEMM.

**Q: Reflect CPU sparse blocks?**  
A: Use **`gtile_atom_list`** (already in `grid_screen.py`) to skip inactive atoms;
optional packed χ per gTile for large systems.

---

## 12. Related files

| File | Role |
|------|------|
| `pyscf/OpenCL/xc_grid.py` | `setup_precomputed_gto`, transpose upload |
| `pyscf/OpenCL/kernels.cl` | New `rho_*_precomp_gather`, `rho_*_radial_gather` |
| `pyscf/OpenCL/grid_screen.py` | gTile atom/pair lists |
| `pyscf/OpenCL/radial_hermite.py` | Channel tables for Variant B |
| `expamples_prokop/test_opencl_xc_scf.py` | Benchmark + parity |
