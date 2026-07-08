---
type: BenchmarkReport
title: CPU smallDFT benchmarks (benzene PBE)
description: Scaling tables for original PySCF vs smallDFT C/OpenMP; bottleneck waterfall
tags: [cpu, benchmark, smallDFT, benzene, PBE]
timestamp: 2026-07-08
---

## Setup

| parameter | value |
|-----------|-------|
| molecule | benzene |
| basis | 6-31g |
| XC | PBE |
| grid level | 3 |
| `nao` | 66 |
| `ngrids` | 143560 |
| `OPENBLAS_NUM_THREADS` | 1 |
| timing | min of 5 runs, 2 warmup |
| repo | `PYTHONPATH=/home/prokop/git/pyscf` |

**Policy:** linear scaling is measured on **optimized sub-tasks** (ρ, vmat). Python `ThreadPoolExecutor` path is deprecated; report focuses on **original PySCF** vs **smallDFT C/OpenMP**.

Implementation doc: [/home/prokop/git/pyscf/doc/smallDFT_cpu_path.md](/home/prokop/git/pyscf/doc/smallDFT_cpu_path.md)

---

## Scaling tables (ms)

### `rho_gga` — AO cached (isolates ρ kernel)

| method | 1 CPU | 2 CPU | 4 CPU | 8 CPU | scale 1→8 |
|--------|------:|------:|------:|------:|----------:|
| original PySCF `eval_rho` (OMP) | 158.6 | 101.6 | 77.0 | 80.0 | **1.98×** |
| **smallDFT C/OpenMP** | 118.9 | 63.8 | 35.3 | 20.0 | **5.94×** |

### `vmat_gga` — AO + wv cached (isolates vmat kernel)

| method | 1 CPU | 2 CPU | 4 CPU | 8 CPU | scale 1→8 |
|--------|------:|------:|------:|------:|----------:|
| original PySCF (block `eval_rho`+vmat path) | — | — | — | — | — |
| **smallDFT C/OpenMP** | 62.4 | 33.4 | 19.2 | 16.0 | **3.90×** |

### `rho + libxc + vmat` — AO cached

| method | 1 CPU | 2 CPU | 4 CPU | 8 CPU | scale 1→8 |
|--------|------:|------:|------:|------:|----------:|
| original PySCF (OMP) | 229.1 | 140.2 | 112.0 | 110.4 | **2.07×** |
| **smallDFT C/OpenMP** | 199.5 | 107.7 | 61.4 | 40.4 | **4.93×** |

### Full `nr_rks` (XC integral)

| method | 1 CPU | 2 CPU | 4 CPU | 8 CPU | scale 1→8 | notes |
|--------|------:|------:|------:|------:|----------:|-------|
| original PySCF (OMP) | 306.1 | 182.0 | 128.7 | 126.2 | **2.43×** | includes `eval_ao` each call |
| **smallDFT C (AO cached)** | 200.0 | 108.0 | 61.7 | 41.3 | **4.84×** | `ws.eval_ao()` once per geometry |

### `eval_ao` — libcint (not yet grid-parallel)

| 1 CPU | 2 CPU | 4 CPU | 8 CPU | scale 1→8 |
|------:|------:|------:|------:|----------:|
| 55.0 | 54.8 | 58.3 | 58.4 | **0.94×** |

---

## Parity (machine precision)

Verified after `SMALL_vmat_gga` landing (Jul 2026):

| check | max diff |
|-------|----------|
| H2O PBE `nr_rks` vmat vs ref | 1.4e-15 |
| benzene PBE `nr_rks` vmat vs ref | 4.2e-15 |
| benzene C vmat vs Python `vmat_gga` | 1.8e-14 |
| benzene C ρ_gga vs Python | 9.1e-13 |

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/prokop/git/pyscf \
  python3 expamples_prokop/test_small_dft.py
```

---

## Bottleneck waterfall @ 8 CPU (AO cached)

After ρ and vmat are both in C, **new limits appear**:

| step | time @8 CPU | share of XC | scales 1→8? |
|------|------------:|------------:|:-----------:|
| `rho_gga` C | 23.8 ms | 50% | **yes** (5.9×) |
| `vmat_gga` C | 19.5 ms | 41% | partial (3.9×) |
| `libxc` | 4.1 ms | 9% | OMP inside |
| **XC subtotal** | **47.4 ms** | 100% | **4.9×** |
| `eval_ao` (per geometry) | ~53 ms | — | **no** |

**End-to-end** (AO cached, 8 CPU): ~41 ms XC + ~53 ms AO ≈ **94 ms** amortized vs original **126 ms** full `nr_rks` @8 CPU (includes AO each call).

### What was hiding before

1. **Slow ρ** — dominated XC; now ~20 ms @8 CPU with near-linear scaling
2. **Python vmat** — capped ρ+vmat at ~2.5×; C vmat unlocks **~4.9×** on ρ+libxc+vmat
3. **`eval_ao`** — flat ~53 ms; becomes dominant once XC is fast (cache AO across SCF)
4. **vmat bandwidth** — heavier GEMM per grid point than ρ → 3.9× not 6×

---

## Next optimizations (C/OpenMP only)

| P | item | expected impact |
|---|------|-----------------|
| 1 | Fuse ρ+vmat single χ tile pass | ~2× less χ traffic |
| 2 | Grid-parallel `eval_gto` | unlock full-path scaling |
| 3 | vmat TILE / cache tuning | 3.9× → ~5× on vmat |
| 4 | `GridWorkspace` on `mf` via `patch.enable()` | less boilerplate |

---

## Reproduce

```bash
# build
pyscf/lib/smalldft/build.sh

# bottleneck breakdown
PYTHONPATH=/home/prokop/git/pyscf python3 -c \
  "from pyscf.smallDFT import profile_xc_bottleneck; profile_xc_bottleneck('benzene', 8)"

# full parity + scaling
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/prokop/git/pyscf \
  python3 expamples_prokop/test_small_dft.py --rho
```

