---
type: BenchmarkReport
title: GPU XC Benchmarks — GTX 1650 laptop
description: Benzene tile sweeps (OTF + split-K), stage ρ/vmat CL, PTCDA spot stage + SCF Amdahl vs CPU smallDFT on i5-12500H / 16 GiB
tags: [opencl, dft, xc, gpu, benchmark, gtx1650, tile-sweep]
timestamp: 2026-07-17
last_updated: 2026-07-17
---

Laptop host (not the RTX 3090 machine in `doc/GPU_benchmark.md`). Goal: see whether tile / profile choice can make GPU XC competitive with CPU `smallDFT` on this hardware.

All tables below are **self-contained** (no external CSV required).

**Drivers (to reproduce)**
- Stage timing: `expamples_prokop/profile_xc_stages_benzene.py`
- Tile sweep: `expamples_prokop/sweep_splitk_tiles.py` (`--path otf|splitk`, `--quick`)
- SCF Amdahl: `expamples_prokop/profile_amdahl_budget.py`

**Scope:** isolated XC stages use `OMP_NUM_THREADS=1`. PTCDA SCF cycle uses 4 OMP + DF `storage=incore`. Do not mix with 3090 numbers. Absolute ms vary with GPU boost; rankings within one warm session are stable.

---

## Test machine

Recorded 2026-07-17.

### CPU

| Item | Value |
|------|-------|
| Model | **Intel Core i5-12500H** (Alder Lake) |
| Logical CPUs | 16 (threads) |
| Max clock | ~4.5 GHz |
| RAM | **15 GiB** (≈8.5 GiB available under desktop load) |
| Swap | **0** |

### OpenCL GPU

| Item | Value |
|------|-------|
| Platform | NVIDIA CUDA — OpenCL 3.0 (CUDA 12.1.98) |
| Device | **NVIDIA GeForce GTX 1650** |
| Driver | **530.41.03** |
| Compute units | **14** |
| Max clock (reported) | 1155 MHz |
| Global memory | **~4.1 GB** |
| Local memory | **48 KiB** (`49152`) |
| Max work-group | 1024 |

Production XC kernels: **f32**.

---

## Benzene — XC stage baselines (default tiles)

`benzene` · `6-31g` · grid level **2** · `nao=66` · `ngrids=99480` · OMP=1.

Default compile tiles: `NPTILE=64 NATILE=2 WGS_VMAT=256` (split-K profile may recompile `WGS_VMAT=128`).

| method | outer | gpuCL | ρ_cl | vmat_cl | note |
|--------|------:|------:|-----:|--------:|------|
| OTF cubic | 262 | 241 | 67 | 161 | `production_otf` |
| OTF quintic | 226 | 214 | 55 | 149 | |
| Radial precomp | 301 | 291 | 77 | 206 | slowest |
| **OTF ρ + rad vmat** | **212** | **200** | **45** | **145** | best default profile |
| OTF ρ + rad splitK | 241 | 230 | 56 | 164 | 3090 production winner — **not** best here |

**Bottleneck:** vmat ≈ 2–3× ρ on every path. PBE/reduce ≪ 10 ms.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/prokop/git/pyscf \
  python3 -u expamples_prokop/profile_xc_stages_benzene.py \
  --mol benzene --basis 6-31g --grid-level 2 --skip-cpu
```

---

## Benzene — OTF tile quick grid (complete)

`--path otf --quick` · unique `(NPTILE,NATILE,WGS)` · one warm session · times in **ms** (CL = OpenCL event).

| NPTILE | NATILE | WGS | ρ_cl | vmat_cl | gpuCL | outer | max\|Δvxc\| |
|-------:|-------:|----:|-----:|--------:|------:|------:|----------:|
| 32 | 1 | 64 | 16.6 | 50.9 | 67.9 | 68.2 | 3.62e-05 |
| 32 | 1 | 128 | 16.5 | 48.1 | 65.1 | 65.4 | 3.62e-05 |
| 32 | 1 | 256 | 16.2 | 54.8 | 71.5 | 71.6 | 3.62e-05 |
| 32 | 2 | 64 | 15.6 | 39.5 | 55.5 | 55.8 | 3.62e-05 |
| 32 | 2 | 128 | 15.9 | 37.2 | 53.5 | 53.8 | 3.62e-05 |
| 32 | 2 | 256 | 15.6 | 38.7 | 54.7 | 55.0 | 3.62e-05 |
| 64 | 1 | 64 | 16.3 | 49.8 | 66.6 | 66.8 | 3.68e-05 |
| 64 | 1 | 128 | 16.1 | 49.4 | 65.9 | 66.2 | 3.68e-05 |
| 64 | 1 | 256 | 16.1 | 49.0 | 65.5 | 64.2 | 3.68e-05 |
| 64 | 2 | 128 | 13.5 | 37.7 | 51.6 | 51.9 | 3.68e-05 |
| **64** | **2** | **256** | **13.5** | **34.3** | **48.2** | **47.3** | 3.68e-05 |

**Hard fails** (ptxas shared mem > 48 KiB, not in table): `NATILE=4` (any), `NPTILE=128`+`NATILE=2`.

**Readout**
- Span **48–72 ms** (~1.5×) — tile choice is not a multi× lever on benzene/1650.
- `NATILE=1` hurts **vmat** (+10–15 ms vs `NATILE=2`); ρ almost unchanged.
- Small tiles (`32,2,*`) are fine (~54 ms).
- Best: default **`NPTILE=64 NATILE=2 WGS_VMAT=256`**.

---

## Benzene — split-K tile quick grid (complete)

`--path splitk --quick` · all 33 points · one warm session · ms.

| NPTILE | NATILE | WGS | splits | ρ_cl | vmat_cl | gpuCL | outer | max\|Δvxc\| |
|-------:|-------:|----:|-------:|-----:|--------:|------:|------:|----------:|
| 32 | 1 | 64 | 16 | 17.5 | 40.5 | 58.5 | 58.7 | 3.70e-05 |
| 32 | 1 | 64 | 32 | 17.4 | 40.0 | 57.8 | 58.1 | 3.71e-05 |
| 32 | 1 | 64 | 64 | 17.1 | 40.1 | 57.7 | 57.6 | 3.70e-05 |
| 32 | 1 | 128 | 16 | 17.7 | 41.0 | 59.1 | 59.2 | 3.70e-05 |
| 32 | 1 | 128 | 32 | 17.2 | 41.2 | 58.8 | 58.0 | 3.71e-05 |
| 32 | 1 | 128 | 64 | 16.9 | 39.9 | 57.2 | 57.1 | 3.70e-05 |
| 32 | 1 | 256 | 16 | 17.3 | 44.6 | 62.4 | 62.7 | 3.70e-05 |
| 32 | 1 | 256 | 32 | 17.1 | 42.6 | 60.2 | 60.1 | 3.71e-05 |
| 32 | 1 | 256 | 64 | 16.9 | 42.0 | 59.3 | 58.9 | 3.70e-05 |
| 32 | 2 | 64 | 16 | 16.3 | 39.4 | 56.2 | 56.5 | 3.70e-05 |
| 32 | 2 | 64 | 32 | 16.1 | 38.7 | 55.3 | 55.5 | 3.71e-05 |
| 32 | 2 | 64 | 64 | 15.9 | 39.6 | 56.0 | 52.0 | 3.70e-05 |
| 32 | 2 | 128 | 16 | 16.0 | 40.9 | 57.3 | 56.6 | 3.70e-05 |
| 32 | 2 | 128 | 32 | 15.7 | 39.9 | 56.1 | 55.6 | 3.71e-05 |
| 32 | 2 | 128 | 64 | 15.7 | 39.2 | 55.4 | 54.6 | 3.70e-05 |
| 32 | 2 | 256 | 16 | 15.9 | 44.5 | 60.8 | 60.8 | 3.70e-05 |
| 32 | 2 | 256 | 32 | 15.8 | 42.1 | 58.4 | 58.6 | 3.71e-05 |
| 32 | 2 | 256 | 64 | 15.3 | 41.1 | 56.9 | 56.5 | 3.70e-05 |
| 64 | 1 | 64 | 16 | 16.9 | 40.6 | 58.0 | 58.0 | 3.71e-05 |
| 64 | 1 | 64 | 32 | 16.4 | 40.9 | 57.8 | 57.1 | 3.71e-05 |
| 64 | 1 | 64 | 64 | 16.6 | 38.8 | 55.9 | 56.1 | 3.71e-05 |
| 64 | 1 | 128 | 16 | 16.8 | 41.1 | 58.4 | 58.6 | 3.71e-05 |
| 64 | 1 | 128 | 32 | 16.6 | 39.6 | 56.7 | 57.0 | 3.71e-05 |
| 64 | 1 | 128 | 64 | 16.7 | 39.9 | 57.1 | 56.9 | 3.71e-05 |
| 64 | 1 | 256 | 16 | 16.7 | 41.1 | 58.3 | 58.3 | 3.71e-05 |
| 64 | 1 | 256 | 32 | 16.3 | 41.0 | 57.8 | 57.6 | 3.71e-05 |
| 64 | 1 | 256 | 64 | 16.6 | 40.0 | 57.1 | 57.2 | 3.71e-05 |
| 64 | 2 | 128 | 16 | 13.7 | 39.9 | 54.1 | 54.4 | 3.71e-05 |
| 64 | 2 | 128 | 32 | 13.7 | 38.6 | 52.7 | 53.0 | 3.71e-05 |
| **64** | **2** | **128** | **64** | **13.3** | **38.5** | **52.3** | **52.3** | 3.71e-05 |
| 64 | 2 | 256 | 16 | 13.7 | 40.2 | 54.4 | 54.6 | 3.71e-05 |
| 64 | 2 | 256 | 32 | 13.3 | 39.9 | 53.6 | 53.3 | 3.71e-05 |
| 64 | 2 | 256 | 64 | 13.4 | 39.4 | 53.3 | 52.7 | 3.71e-05 |

**Readout**
- Band **~52–62 ms**; splits 16/32/64 barely matter when warm.
- Best split-K **52.3 ms** still **slower than best OTF 48.2 ms**.
- Small configs (`32,1,64`, `32,2,64`) sit only ~6–10 ms behind the winner.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/prokop/git/pyscf \
  python3 -u expamples_prokop/sweep_splitk_tiles.py --path otf --quick \
  --xyz data/xyz/benzene.xyz --basis 6-31g --grid-level 2

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/prokop/git/pyscf \
  python3 -u expamples_prokop/sweep_splitk_tiles.py --path splitk --quick \
  --xyz data/xyz/benzene.xyz --basis 6-31g --grid-level 2
```

---

## PTCDA — XC stage spot (default tiles, no tile loops)

`PTCDA` · `6-31g` · grid **2** · `nao=286` · `natm=38` · `ngrids=379216` · OMP=1.

| method | outer | ρ_cl | vmat_cl | gpuCL |
|--------|------:|-----:|--------:|------:|
| OTF cubic | 2711 | 1113 | 1592 | 2709 |
| OTF quintic | 2730 | 1099 | 1624 | 2727 |
| Radial precomp | 3808 | 1422 | 2380 | 3804 |
| OTF ρ + rad vmat | 3548 | 1149 | 2394 | 3546 |
| OTF ρ + rad splitK | 3418 | 1155 | 2257 | 3415 |
| **radial screened** | **1077** | **285** | **786** | **1074** | `production_radial_screened` · `na≈21/38` · pair×≈5.8 · `|vxc|` parity vs OTF |

Re-bench (same session, post rho-spill fix + screened kernels, 2026-07-17): OTF gpuCL≈2826 ms; screened≈1074 ms (**~2.6×**). Benzene screened ≈ OTF (little screening: `na≈11.5/12`).

On PTCDA/1650, **screened radial** beats plain OTF; older radial/split-K hybrids without screening are still worse. Tile brute-force on PTCDA is too slow on this laptop — use benzene tables above.

---

## PTCDA — one SCF cycle vs CPU (Amdahl)

DF `storage=incore` · 4 OMP · `profile_amdahl_budget.py --n-cycles 1` · separate processes (two full GGA χ buffers ≈ 7 GB → OOM on 16 GiB / 0 swap).

| mode | cycle veff | notes |
|------|----------:|-------|
| cpu_ref | 3053 ms | NumInt `block_loop` |
| cpu_stream | 2785 ms | no full χ; AO every cycle (`ao_mode=stream`) |
| **cpu_small (cache)** | **886 ms** | full GGA χ ~3.5 GB once (`ao_mode=cache`) |
| gpu_otf | 1997 ms | default GPU profile; slower than CPU cache |
| gpu_screened | ~1074 ms XC only | `production_radial_screened` stage; still > CPU cache XC |

**Verdict on this laptop:** PTCDA production → CPU `prepare_smalldft_for_scf(..., ao_mode='cache')` when RAM fits; `'stream'` if not. Best GPU stage path is **`production_radial_screened`** (~2.6× vs OTF); still slightly behind cached CPU XC.

---

## PTCDA — detailed SCF bottleneck (best settings)

Recorded 2026-07-17 on this laptop. **Best production path:** CPU `smallDFT` + `ao_mode='cache'` + DF `storage='incore'` · 4 OMP · `OPENBLAS_NUM_THREADS=1` · PBE / 6-31g / grid2 · `nao=286` · `ngrids=379216`.

GPU comparison uses `production_otf` (plain OTF — best PTCDA stage path on 1650) · `overlap_j_xc=True` · OMP=1 for GPU XC.

### A) One-time setup (outside SCF loop)

| step | wall | notes |
|------|-----:|-------|
| `Grids.build` | ~530–1100 ms | varies with load |
| DF `_cderi` build (incore) | **~1960 ms** | 750 MB `float64` |
| AO χ cache fill GGA `(4,ngrids,nao)` | **~190–480 ms** | ~**3.5 GB**; once per geometry |
| GPU `apply_gpu_profile(production_otf)` | ~7.5 s | compile + Hermite/radial setup (GPU path only) |

Peak RSS with χ + DF ≈ **4.2 GB** (fits in 16 GiB if desktop is quiet; **no swap**).

### B) Steady SCF cycle — CPU smallDFT cache (winner)

`mf.kernel(max_cycle=2)`; steady = **2nd** iteration.

| sub-task | wall | % of cycle veff | notes |
|----------|-----:|----------------:|-------|
| **`get_veff`** | **~989 ms** | 100% | almost entire cycle |
| └─ `nr_rks` (XC) | **~915 ms** | 92% | AO already cached |
| └─ DF `get_j` | **~74 ms** | 7% | incore `_cderi` |
| └─ veff glue | ~1 ms | | |
| `scf.eig` | ~15 ms | | Fock diagonalize |
| `get_fock` / DIIS | ~0–4 ms | | |
| `get_grad` | ~1 ms | | |
| `energy_tot` | ≪1 ms | | |

**Cycle ≈ get_veff (~1.0 s).** Diag/DIIS are noise.

### C) XC micro-breakdown (CPU, χ cached, C kernels, 4 threads)

Isolated min-of-4 after warmup (same DM). Shares applied to observed `nr_rks≈915 ms`:

| XC sub-task | micro ms | % of XC | ≈ ms in cycle | % of veff |
|-------------|---------:|--------:|--------------:|----------:|
| **`rho_gga` (C/OpenMP)** | **420** | **50.7%** | **~464** | **~47%** |
| **`vmat_gga` (C/OpenMP)** | **389** | **47.0%** | **~430** | **~43%** |
| libxc PBE `eval_xc_eff` | 17 | 2.1% | ~19 | ~2% |
| `wv = w·vxc` (+½) | 2 | 0.2% | ~2 | ~0% |
| **XC total** | **828** | 100% | **~915** | **~92%** |

**Bottleneck (CPU path):** almost entirely **ρ + vmat** over the full GGA χ (~3.5 GB working set). libxc is irrelevant. J is small but not free (~7%).

```
steady cycle ~1000 ms
├─ XC  ~915 ms  (92%)
│  ├─ rho_gga   ~464 ms  (47% of veff)
│  ├─ vmat_gga  ~430 ms  (43% of veff)
│  └─ libxc+wv   ~21 ms
├─ DF J  ~74 ms  (7%)
└─ eig+… ~17 ms
```

### D) Steady SCF cycle — GPU `production_otf` (for contrast)

One XC call with `profile=True` (CL events):

| GPU stage | CL ms | % of XC |
|-----------|------:|--------:|
| `gpu_rho` | **797** | 40% |
| `gpu_vmat` | **1181** | 60% |
| `gpu_xc_pbe` + reduce | ~1 | ~0% |
| host H2D/D2H | ~2 | ~0% |
| **gpu_total_cl** | **~1981** | 100% |

SCF steady cycle: `get_veff≈2000 ms` (XC dominates; J~54 ms overlapped under `overlap_j_xc=True` so wall ≈ XC). eig~11 ms.

**GPU vs CPU on same molecule:** XC **~2.0 s vs ~0.9 s**. Same structure: **vmat > ρ ≫ PBE**. Tile tweaks on benzene only move XC ~1.5× — not enough to beat CPU cache here.

### E) What to optimize next (priority from this breakdown)

| P | target | why |
|---|--------|-----|
| 1 | **CPU `rho_gga` / `vmat_gga` bandwidth** | 90%+ of best-path cycle |
| 2 | Avoid full χ when possible (`ao_mode=stream`) | RAM; slower multi-cycle |
| 3 | DF J (~74 ms) | only after XC ≪ 200 ms |
| 4 | GPU vmat on 1650 | still 2× CPU XC; structural, not tile |
| — | eig / DIIS / libxc | already negligible |

Reproduce CPU breakdown (same hygiene):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=/home/prokop/git/pyscf \
  python3 -u expamples_prokop/profile_amdahl_budget.py \
  --mol PTCDA --grid-level 2 --n-cycles 2 --threads 4 \
  --modes cpu_small --df-storage incore
```

---

## Conclusions

1. **Tile tuning (benzene/1650):** full grids above — only ~1.5× span; default OTF `64,2,256` wins; small tiles OK; `NATILE=1` costs vmat.
2. **Local-mem wall:** 48 KiB blocks `NATILE=4` and large `NPTILE×NATILE`.
3. **Profile choice ≠ 3090:** prefer **OTF**; split-K is not automatically fastest.
4. **Amdahl:** PTCDA GPU OTF XC ~2–2.8 s vs CPU smallDFT cache ~0.9 s; screened radial XC ~1.07 s (closer, still slightly behind CPU cache).
5. **Screened radial:** new kernels `rho_gga_radial_screened` + `vmat_gga_radial_screened_pair` via profile `production_radial_screened` — best GPU path on PTCDA/1650 so far.
6. **Machine split:** `doc/GPU_benchmark.md` = RTX 3090 / 32 GiB; **this file** = GTX 1650 / 16 GiB.

---

## Related

- **Session lessons / outlook (2026-07-17):** `doc/GPU_1650_lessons_2026-07-17.md`
- 3090 report: `doc/GPU_benchmark.md`
- 3090 experience arc: `doc/GPU_optimixation_experience.md`
- CPU Amdahl / DF hygiene: `doc/CPU_benchmark.md`, `doc/df_storage_and_benchmark_hygiene.md`
- Profiles / knobs: `doc/opencl_gpu_paths_cookbook.md`
