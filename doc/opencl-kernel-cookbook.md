# OpenCL Kernel Optimization Cookbook (PySCF DFT / XC)

Dense guidelines for writing and reviewing GPU kernels in `pyscf/OpenCL/kernels.cl`.  
**Goal:** maximize useful FLOPs, minimize PCIe, Python, and idle threads.  
**Non-goal:** identical structure across unrelated operations.

---

## 1. Map the data movement first

Before tiling, answer:

| Question | rho | vmat |
|----------|-----|------|
| **Output** | vector on grid `ρ[g]` (GGA: 4 components) | matrix `V[i,j]` |
| **Reduction axis** | sum atom-pair contractions → one value per grid point | sum grid points → one value per matrix element |
| **Natural gather** | grid points (threads own `g`, accumulate in private/`__local`) | AO pairs (threads own `(i,j)`, accumulate over `g`) |
| **Natural loop inside WG** | all `(iTile, jTile)` atom tiles | all `gTile` grid tiles |

**Do not force rho and vmat into the same launch geometry.** They are dual operations:

- **rho:** scatter-free **grid gather** — many atom pairs → few grid outputs.
- **vmat:** scatter-free **matrix gather** — many grid points → few matrix outputs.

---

## 2. Gather good, scatter bad, atomics last resort

### Gather (preferred)

- Each thread (or WG) **owns output slot(s)** and **reads** contributing inputs.
- Writes are **disjoint** → no races, no `atomic_*`.
- Reductions use **`__local` + `barrier()`** within the workgroup.

```c
// grid-point owner reduces NATILE i-atom contributions in local memory
psum[lid] = rho_priv;
barrier(CLK_LOCAL_MEM_FENCE);
if (il == 0) {
    float s = 0;
    for (int k = 0; k < NATILE; k++) s += psum[k*NPTILE + ip];
    rho[g] = s;   // one writer per g
}
```

### Scatter (avoid)

- Many threads writing the same global index (even without atomics if serialized) → contention, cache line bouncing, hard to reason about.

### Atomics (almost never)

- Use only when output indices are genuinely unpredictable and cannot be bucketed.
- In grid XC: **always avoid.** If you think you need atomics, you put the **parallel axis on the wrong dimension** (classic bug: `iTile` on `group_id` when reduction needs all i-atoms in one WG).

### Cross-workgroup reduction

Workgroups **cannot** `barrier()` together. Options:

1. **Loop the outer tile inside the kernel** (what rho does for `iTile`) — best when tile count is modest (`n_iTiles < ~32`).
2. **Second kernel** that gathers partial buffers — acceptable if the inner kernel cannot absorb the loop without blowing register/local limits.
3. **CPU sum of partials** — debug only; never in the hot SCF path.

---

## 3. Put the parallel axis on the output owner

**Rule:** the workgroup should contain all threads that must collaborate to finish **one output tile** without cross-WG communication.

### rho (`rho_*_tiled`) — current design

```
global: (ceil(ngrids/NPTILE), NATILE)
local:  (NPTILE, NATILE)          // 64 threads
group_id(0) = gTile               // one WG per grid-point tile
```

Inside WG:

```c
for (iTile = 0; iTile < n_iTiles; iTile++)
    for (jTile = 0; jTile < natoms; jTile += NATILE)
        rho_priv += contract_pair(...);
// __local reduce over il → rho[g]
```

- Thread `(ip, il)` handles grid point `g` and i-atom slot `il` within each `iTile`.
- **Wrong (old):** `iTile = group_id(1)` → partial `rho[iTile, g]` → host sum.

### vmat (`vmat_*_tiled`) — current design

```
global: (n_iTiles, n_jTiles * WGS_VMAT)
local:  (1, WGS_VMAT)             // 256 threads
group_id = (iTile, jTile)         // one WG per atom-pair tile
```

Inside WG:

```c
for (gTile = 0; gTile < ngrids; gTile += NPTILE) {
    // cooperative fill aoI[NPTILE][AO_TILE], aoJ[NPTILE][AO_TILE]
  acc[q] += wv[g] * aoI[ip][iao] * aoJ[ip][jao];
}
// write vmat[iao, jao] from private acc[q]
```

- Thread owns **QPT matrix elements** `(iao, jao)`; loops grid tiles.
- **Wrong (old abTile design):** `abTile` on grid → 57× redundant radial/AO eval per `(iTile,jTile,gTile)`.

---

## 4. `__local` memory cookbook

| Pattern | Use when | Size discipline |
|---------|----------|-----------------|
| **Tile cache** | reuse data across threads in WG (`wfRj`, `dm_blk`, `aoI`) | pad to fixed `MAX_*`; zero unused slots |
| **Staging for cooperative load** | `for (k = lid; k < SIZE; k += WGS)` | one barrier after load |
| **Reduction buffer** | `psum[WGS]` | one barrier before tree/serial reduce |
| **Private accumulator** | per-thread output (`acc[QPT]`, `rho_priv`) | keep `QPT` small; spill → revisit tile sizes |

**Occupancy trade:** local mem per WG limits active WGs. Budget:

- rho: `wfRj` 16×4×6 + `dm_blk` 4×4×15×15 ≈ **4 KB** (+ GGA `dwfRj`)
- vmat: `aoI`+`aoJ` 2×16×60×4 ≈ **7.5 KB**

If occupancy collapses, **reduce `NPTILE` or split kernels** — do not silently shrink WGS without remeasuring.

---

## 5. Cooperative loads: one writer per `__local` cell

```c
for (int k = lid; k < WFJ_SIZE; k += WGS_TILED) {
    decode k → (pp, jj, shell s);
    wfRj[pp][jj][s] = hermite_eval(...);   // gather from global tables
}
barrier(CLK_LOCAL_MEM_FENCE);
// consume wfRj[ip][jl] — each thread reads only its row
```

- **Load phase:** strided loop, disjoint `__local` indices → no races.
- **Compute phase:** read local, accumulate private.
- Avoid reloading the same `(g, ja)` radial in every `iTile` iteration if you can cache across `iTile` at constant `g` (future rho tweak).

---

## 6. Branches and divergence

| OK | Bad |
|----|-----|
| `if (g >= ngrids) return;` at boundaries | `if (l==0) ... else if (l==1) ...` inside hot loop over all grid points |
| `if (ia >= natoms) continue;` per tile | per-thread shell search in inner `g` loop |
| compile-time `MAX_SHELL`, `MAX_AO_ATOM` bounds | data-dependent loop bounds without uniform WG early-exit |

**Prefer:** table-driven dispatch, `unfold_shell(l, ...)` with small fixed `l ≤ 3`, hoist `natoms`/`ngrids` checks to tile level.

**Padding:** launch `ceil(n/NPTILE)*NPTILE` threads; guard with `g < ngrids`. Cheaper than special-case last tile on host.

---

## 7. Load balancing

- **rho:** WG count = `n_gTiles` ≈ `ngrids/16`. Large molecules: enough WGs to fill GPU if each WG does `n_iTiles × n_jTiles` inner work.
- **vmat:** WG count = `n_iTiles × n_jTiles` (benzene: 9). Inner loop length = `n_gTiles` (~9000). Fine for medium grids; huge grids → consider multiple `gTile` WGs per `(i,j)` tile only if register pressure allows splitting.

**Measure:** if kernel time scales with `ngrids × natoms²` but WG count is tiny, increase parallelism on the **cheap** axis (more WGs, less work per WG) until PCIe or launch overhead dominates.

---

## 8. Host / Python: keep out of the hot path

SCF per-cycle should be:

1. upload `DM` (or `DM_cart`)
2. `enqueue` rho kernel → `finish`
3. libxc on CPU (`eval_xc_eff`) — unavoidable for now
4. upload `wv`
5. `enqueue` vmat kernel → `finish`
6. download `vmat` + small `c2s` transform

**Pre-SCF (`setup_xc_grid_gpu` / `mf.setup_gpu()`):**

- compile OpenCL program
- build Hermite tables
- allocate **all** buffers
- bake kernel `set_args` for static pointers
- upload coords, atom metadata

**Time with `profile=True`:**

- `kernel_*` — from immediately before `enqueue` to `queue.finish()` after that kernel.
- `harness_*` — everything else (NumPy, libxc, PCIe).

Never use `queue.finish()` between back-to-back GPU kernels unless you need the result on host.

---

## 9. Tile constants (current production)

```c
#define NPTILE      16    // LOG_NPTILE = 4
#define NATILE      4     // LOG_NATILE = 2
#define MAX_SHELL   6
#define MAX_AO_ATOM 16    // padded pow2; LOG_MAX_AO_ATOM = 4 (l≤3 max cart is 10–15)
#define WGS_TILED   (NPTILE * NATILE)   // 64
#define WGS_VMAT    256
#define AO_TILE     64    // LOG_AO_TILE = 6
#define VBLK_SIZE   4096  // AO_TILE² (was 3600 before pow2 padding)
#define QPT         16
```

Index decode: `>>` / `&` with `LOG_*` constants; vmat uses `decode_q_vmat()`. Spherical fac: `__constant CINT_FAC_SP[l]` — not `if (l==k)` chains.

Tuning order: **`NPTILE`** (vmat inner iterations, local AO size) → **`NATILE`** (VBLK_SIZE, WG count) → **`WGS_VMAT`**.

---

## 10. Review checklist (before merging a kernel)

- [ ] Output indices: disjoint writers or explicit `__local` reduce?
- [ ] No `atomic_*` in grid XC?
- [ ] No cross-WG reduction without second kernel or host sum?
- [ ] Parallel axis matches **output owner** (grid for rho, matrix elem for vmat)?
- [ ] Hot loops: minimal branches, no redundant eval (radial/AO) per thread?
- [ ] `__local` loads cooperative, one barrier between load and compute?
- [ ] `global_size % local_size == 0` in each dimension?
- [ ] Buffers allocated in `setup_*`, not per SCF call?
- [ ] Benzene parity test: `vxc` max rel err `< 1e-5`?
- [ ] Timings: `kernel_total` dominates `harness_total` on target GPU?

---

## 11. Anti-patterns seen in this codebase

| Anti-pattern | Symptom | Fix |
|--------------|---------|-----|
| Tile on grid when reduction needs full tile in WG | CPU partial sum, huge partial buffers | loop tile inside WG |
| abTile on grid for vmat | 50–200× slowdown | local AO cache + private `acc[QPT]` |
| `eval_gto_sph`: one thread per `g`, loop all basis | terrible on large basis | Hermite + atom blocking |
| Per-call `cl.Buffer` / `cl.Kernel` | Python overhead dominates small systems | `setup_onthefly` |
| Materialized AO + GEMM as default GPU path | slower than on-the-fly for benzene+ | `nr_rks_hermite_onthefly` |
| `queue.finish()` everywhere | GPU idle during libxc | events / defer sync |

---

## 12. References in repo

- Kernels: `pyscf/OpenCL/kernels.cl` (`rho_*_tiled`, `vmat_*_tiled`)
- Host: `pyscf/OpenCL/xc_grid.py` (`setup_onthefly`, `nr_rks_hermite_onthefly`)
- Test: `expamples_prokop/test_opencl_xc_onthefly.py` (benzene, harness vs kernel timing)
- Design notes: `doc/ToOpenCL.chat.md`
- Optimization reports: `doc/opencl-xc-reports/README.md` (append-only; do not rewrite older reports)
