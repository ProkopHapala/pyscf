---
type: BenchmarkReport
title: CPU smallDFT benchmarks (benzene PBE)
description: XC kernel scaling, full SCF-cycle Amdahl profile, ref vs smallDFT C/OpenMP
tags: [cpu, benchmark, smallDFT, benzene, PBE, scf, amdahl]
timestamp: 2026-07-08
---

## Setup

| parameter | value |
|-----------|-------|
| molecule | benzene (C₆H₆) |
| basis | 6-31g |
| XC | PBE |
| grid level | 3 |
| `nao` | 66 |
| `ngrids` | 143560 |
| `OPENBLAS_NUM_THREADS` | 1 |
| OMP threads | `lib.num_threads(N)` — authoritative for C kernels + libcint |
| repo | `PYTHONPATH=/home/prokop/git/pyscf` |
| C lib | `pyscf/lib/libsmalldft.so` via `pyscf/lib/smalldft/build.sh` |

**Policy:** linear scaling is measured on **optimized sub-tasks** (ρ, vmat, `nr_rks`). Python `ThreadPoolExecutor` grid tiles are deprecated. Production path is **C/OpenMP only**.

Implementation guide: [/home/prokop/git/pyscf/doc/smallDFT_cpu_path.md](/home/prokop/git/pyscf/doc/smallDFT_cpu_path.md)

---

## Part A — XC kernel micro-benchmarks

Isolated sub-task timings (min of 5 runs, 2 warmup). Used to validate C kernel scaling before looking at full SCF.

### `rho_gga` — AO cached (isolates ρ kernel)

| method | 1 CPU | 2 CPU | 4 CPU | 8 CPU | scale 1→8 |
|--------|------:|------:|------:|------:|----------:|
| original PySCF `eval_rho` (OMP) | 158.6 | 101.6 | 77.0 | 80.0 | **1.98×** |
| **smallDFT C/OpenMP (stride-1 accum.)** | 42.1 | 21.5 | 12.4 | 10.1 | **4.17×** |

### `vmat_gga` — AO + wv cached (isolates vmat kernel)

| method | 1 CPU | 2 CPU | 4 CPU | 8 CPU | scale 1→8 |
|--------|------:|------:|------:|------:|----------:|
| **smallDFT C/OpenMP (F-order `aow`)** | 38.7 | 19.3 | 11.1 | 11.1 | **3.49×** |

### `rho + libxc + vmat` — AO cached

| method | 1 CPU | 2 CPU | 4 CPU | 8 CPU | scale 1→8 |
|--------|------:|------:|------:|------:|----------:|
| original PySCF (OMP) | 229.1 | 140.2 | 112.0 | 110.4 | **2.07×** |
| **smallDFT C/OpenMP** | 99.1 | 52.2 | 30.4 | 27.4 | **3.62×** |

### Full `nr_rks` (XC integral)

| method | 1 CPU | 2 CPU | 4 CPU | 8 CPU | scale 1→8 | notes |
|--------|------:|------:|------:|------:|----------:|-------|
| original PySCF (OMP) | 306.1 | 182.0 | 128.7 | 126.2 | **2.43×** | includes `eval_ao` each call |
| **smallDFT C (AO cached)** | 99.1 | 52.2 | 30.4 | 27.4 | **3.62×** | `ws.eval_ao()` once per geometry |

### `eval_ao` — libcint (not yet grid-parallel)

| 1 CPU | 2 CPU | 4 CPU | 8 CPU | scale 1→8 |
|------:|------:|------:|------:|----------:|
| 55.0 | 54.8 | 58.3 | 58.4 | **0.94×** |

Driver: `expamples_prokop/test_small_dft.py --rho` and `pyscf.smallDFT.profile_xc_bottleneck`.

### XC bottleneck waterfall @ 8 CPU (AO cached, sub-task view)

After ρ and vmat are both in C:

| step | time @8 CPU | share of XC | scales 1→8? |
|------|------------:|------------:|:-----------:|
| `rho_gga` C | 10.2 ms | 42% | **yes** (4.2×) |
| `vmat_gga` C | 11.1 ms | 46% | partial (3.5×; plateaus after 4 threads) |
| `libxc` | 3.0 ms | 12% | OMP inside libxc |
| **XC subtotal** | **24.3 ms** | 100% | **3.6×** |
| `eval_ao` (per geometry) | ~58 ms | — | **no** |

**Amortized XC** (AO cached): ~27 ms XC + ~58 ms AO ≈ **85 ms** per `get_veff` call vs original **126 ms** full `nr_rks` @8 CPU (AO each call).

---

## Part B — Full SCF cycle profile (Amdahl)

### Why a separate profile?

Micro-benchmarks (Part A) time `nr_rks` in isolation. A real SCF step calls `mf.kernel()` which runs:

```
pre-SCF (once/geometry)  →  scf_init (guess DM)  →  scf_cycle × N
```

The **proper profiler** runs real `mf.kernel(max_cycle=1)` with monkey-patch timers on the actual PySCF call graph (`rks.get_veff` → `NumInt.nr_rks` + `RHF.get_jk`, etc.). Do **not** use the legacy `--manual` mode for this — it bypasses `RHF.get_jk` and `rks.get_veff`.

Driver: `expamples_prokop/profile_scf_cycle.py` (default mode)

### Phases

| phase | when | what is timed | amortization |
|-------|------|---------------|--------------|
| **pre_scf** | before `kernel()` | `Grids.build`, optional `GridWorkspace.eval_ao` | once per geometry |
| **scf_init** | 1st call inside `kernel()` | `get_veff` on guess DM + `energy_tot` | once per SCF job |
| **scf_cycle** | 2nd `get_veff` call (= 1st loop iteration) | `get_fock` → `eig` → `make_rdm1` → `get_veff` → `energy_tot` → `get_grad` | **every SCF iteration** |

Inside `get_veff` (PBE, no hybrid):

```
rks.get_veff
  ├── NumInt.nr_rks          # XC: eval_ao (in block_loop) + ρ + libxc + vmat
  └── ks.get_j → RHF.get_jk  # Coulomb J
```

With `direct_scf=True` (default): cycle 2+ uses **incremental J** on `Δdm = dm_new − dm_old` (~2 ms for benzene 6-31g). Init J on full guess DM is expensive (~68–520 ms depending on threads).

With `--df`: J goes through **`df.get_jk`** (RI density fitting) — appears as separate timer line.

### Hooked functions (monkey-patch)

| timer label | PySCF entry point |
|-------------|-------------------|
| `rks.get_veff` | `pyscf.dft.rks.get_veff` |
| `NumInt.nr_rks` | grid XC integral (or `smallDFT.nr_rks` when `path=smallDFT_ws`) |
| `NumInt.block_loop` | AO block iterator inside reference `nr_rks` |
| `scf.get_jk` | `RHF.get_jk` — **must patch RHF**, not just `SCF.get_jk` |
| `df.get_jk` | `pyscf.df.df_jk.get_jk` (only with `--df`) |
| `scf.eig` | Fock diagonalization |
| `scf.get_fock` / `scf.get_grad` / `scf.energy_tot` | bookkeeping |
| `Grids.build` | quadrature grid (also timed in pre_scf block) |

Timer table columns: **init** = first call (guess DM), **cycle** = second call (one SCF iteration).

---

## Part C — SCF cycle results (benzene, PBE, 6-31g, grid 3)

### Per-iteration cost (`cycle` column) — ref vs smallDFT_ws

| threads | path | `get_veff` | `nr_rks` (XC) | `get_jk` (J) | `eig` | kernel wall |
|--------:|------|----------:|--------------:|-------------:|------:|------------:|
| 1 | ref | 260 | 258 | 2.3 | 0.6 | 1080 |
| 4 | ref | 123 | 120 | 2.1 | 0.6 | 397 |
| 8 | ref | **122** | **120** | **2.1** | 0.6 | 341 |
| 1 | smallDFT_ws | 105 | 103 | 2.3 | 0.6 | 735 |
| 4 | smallDFT_ws | 37 | 35 | 2.1 | 0.6 | 218 |
| 8 | smallDFT_ws | **28** | **26** | **2.0** | 0.6 | 161 |

**Per-cycle speedup (ref → smallDFT_ws @8 CPU):** 122 → 28 ms = **4.3×** on `get_veff`.

### Pre-SCF setup (once per geometry)

| path | `Grids.build` | `eval_ao` (ws) | pre_scf total |
|------|-------------:|---------------:|--------------:|
| ref | ~107 ms | — | ~107 ms |
| smallDFT_ws | ~101 ms | ~66 ms | ~166 ms |

`eval_ao` is paid once; amortized as **66/N ms** per cycle over N SCF iterations.

### scf_init (first `get_veff` on guess DM)

| threads | ref `get_veff` init | smallDFT_ws init | dominated by |
|--------:|--------------------:|-----------------:|--------------|
| 1 | 797 ms | 607 ms | full `get_jk` (~502 ms) + `nr_rks` (~105 ms) |
| 8 | 202 ms | 116 ms | full `get_jk` (~82 ms) + `nr_rks` (~33 ms) |

Init is a one-time cost per SCF job; not representative of converged-iteration cost.

### Density fitting (`--df`, 8 CPU, ref)

| step | init | cycle (1 iter) |
|------|-----:|---------------:|
| `rks.get_veff` | 171 ms | **152 ms** |
| `NumInt.nr_rks` | 134 ms | 123 ms |
| `df.get_jk` | 36 ms | **30 ms** |

With DF, Coulomb is no longer ~2 ms incremental — **`df.get_jk` is ~30 ms/cycle** (~20% of `get_veff`). This becomes a real post-XC bottleneck for larger systems / heavier bases.

---

## Part D — Amdahl analysis

### What dominates one SCF iteration (benzene 6-31g, 8 CPU)?

```
get_veff  122 ms  (100% of cycle)     ref
  nr_rks  120 ms   (98%)              XC grid integral
  get_jk    2 ms    (2%)              incremental Coulomb (direct J)
  eig       <1 ms
```

After smallDFT_ws:

```
get_veff   28 ms  (100%)
  nr_rks   26 ms   (93%)              still XC-limited
  get_jk    2 ms    (7%)
```

**Diagonalization, density update, gradient:** all sub-ms for benzene 6-31g — not worth optimizing at this system size.

### Theoretical cycle floor

If XC → 0 ms: floor ≈ **J_incr + diag ≈ 3 ms** per cycle (benzene 6-31g, direct J). Not reachable until `nr_rks` is fully optimized.

### What was hiding inside XC (`nr_rks`)

| layer | ref @8 (cycle) | smallDFT_ws @8 (cycle) | notes |
|-------|---------------:|-----------------------:|-------|
| `eval_ao` in block_loop | ~included in 120 ms | **0** (AO cached in ws) | largest hidden win from ws |
| ρ C/OpenMP | — | ~10 ms (sub-task) | stride-1 grid accumulation |
| vmat C/OpenMP | — | ~11 ms (sub-task) | F-order `aow`, `dgemm("T","N")` |
| libxc | ~3 ms | ~3 ms | minor |

Reference `nr_rks` @8 CPU plateaus at ~120 ms because libcint `eval_ao` + OMP block loop do not scale on grid axis.

### End-to-end SCF job (rough, 8 CPU, N cycles)

| component | ref | smallDFT_ws |
|-----------|----:|------------:|
| pre_scf | 107 ms | 166 ms |
| scf_init | 202 ms | 116 ms |
| N × cycle | N × 122 ms | N × 28 ms |
| **N=10 total** | **~1530 ms** | **~560 ms** (~2.7×) |
| **N=20 total** | **~2750 ms** | **~840 ms** (~3.3×) |

Pre_scf + init matter for few-cycle jobs; per-cycle dominates for production SCF (10–50 cycles).

### H2O (nao=13)

Cycle already **<3 ms** @8 CPU. Grid build (~35 ms) dominates. smallDFT optimizations are in the noise for tiny systems.

---

## Parity (machine precision)

Verified after `SMALL_vmat_gga` (Jul 2026):

| check | max diff |
|-------|----------|
| H2O PBE `nr_rks` vmat vs ref | 1.4e-15 |
| benzene PBE `nr_rks` vmat vs ref | 4.2e-15 |
| benzene C vmat vs Python `vmat_gga` | 1.8e-14 |
| benzene C ρ_gga vs Python | 9.1e-13 |

---

## Next optimizations (priority)

| P | item | expected impact | notes |
|---|------|-----------------|-------|
| done | Stride-1 ρ accumulation | 20 ms → 10 ms @8 CPU | loop order changed to `mu` outer, grid `t` inner |
| done | F-order `aow` / `chi_w` tile buffers | 16 ms → 11 ms @8 CPU | `dgemm("T","N")`; `TILE=512` still best |
| done | DF `storage='incore'` + prepare order | J stays ~60 ms on PTCDA | hygiene doc; was ~480 ms outcore |
| done | C tile scratch prealloc | malloc once/thread | `small_grid.c` |
| done | Stream ρ/vmat (`ao_mode='stream'`) | no 3.5 GB χ; GGA OK | `stream_grid.c`; slower multi-cycle |
| 2 | GPU / cheaper AO for scans | few-cycle / geometry scans | cache invalidates each geometry |
| 3 | `patch.enable()` ergonomics | less boilerplate | `prepare_smalldft_for_scf` exists |
| low | Fuse ρ+PBE+vmat single χ tile | secondary | deprioritized |

**Amdahl PTCDA one-cycle (2026-07-17, 4 OMP, DF incore, separate processes):** cycle veff **cpu_ref 3053 → cpu_stream 2785 → cpu_small(cache) 886 → gpu_otf 1997**. E match cpu paths. Stream saves ~3.5 GB RAM, pays AO every cycle. Driver: `profile_amdahl_budget.py --modes cpu_stream`. Do not allocate two χ buffers on 16 GiB / 0 swap.

---

## Reproduce

```bash
# build C kernels
pyscf/lib/smalldft/build.sh

# --- Part A: XC sub-task breakdown (AO cached) ---
PYTHONPATH=/home/prokop/git/pyscf python3 -c \
  "from pyscf.smallDFT import profile_xc_bottleneck; profile_xc_bottleneck('benzene', 8)"

OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/prokop/git/pyscf \
  python3 expamples_prokop/test_small_dft.py --rho

# --- Part B/C: full SCF cycle via real mf.kernel() ---
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/prokop/git/pyscf \
  python3 expamples_prokop/profile_scf_cycle.py --mol benzene --path ref smallDFT_ws --threads 1 4 8

# density fitting (shows df.get_jk)
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/prokop/git/pyscf \
  python3 expamples_prokop/profile_scf_cycle.py --mol benzene --df --threads 8

# cProfile on top of timers
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/prokop/git/pyscf \
  python3 expamples_prokop/profile_scf_cycle.py --mol benzene --threads 8 --profile

# parity only
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/prokop/git/pyscf \
  python3 expamples_prokop/test_small_dft.py
```

Related: `expamples_prokop/profile_dft.py` (older timer harness, supports `--df` and cProfile on full kernel).
