---
type: BenchmarkReport
title: CPU/OpenCL small-molecule acceleration acceptance test
description: Bounded numerical-rigor and performance review of smallDFT and OpenCL XC paths
tags: [benchmark, parity, cpu, opencl, smallDFT, dft, acceptance]
timestamp: 2026-07-11
---

## Scope and verdict

This report is an acceptance test of the acceleration work in `pyscf/smallDFT`,
`pyscf/OpenCL`, `doc`, and `expamples_prokop`. CPU work was run with
`OPENBLAS_NUM_THREADS=1` and PySCF/OpenMP settings no higher than 14. The
container exposes only 4 CPUs to the process, although the host reports an
AMD Ryzen 7 5800X with 8 physical and 16 logical CPUs.

CPU smallDFT passes the available numerical parity checks and is integrated into
the real RKS SCF call path. The C/OpenMP XC kernels provide a clear speedup;
the remaining CPU costs shift to DF-J and setup for larger molecules. The
corrected host-side run below uses the RTX 3090 through NVIDIA CUDA OpenCL.

## Environment

| Item | Result |
|---|---|
| Repository | `/home/prokop/git/pyscf` |
| Latest relevant commit | `736ff2b2f`, 2026-07-08 |
| Python import | repository checkout, not pip Python code |
| Host CPU | AMD Ryzen 7 5800X, 8C/16T |
| Process CPU visibility | 4 CPUs (`nproc`) |
| Requested CPU ceiling | 14 threads; never exceeded in commands |
| OpenCL devices | NVIDIA RTX 3090 GPU plus PoCL CPU |
| smallDFT C library | available (`has_c_lib() == True`) |
| BLAS setting | `OPENBLAS_NUM_THREADS=1` |

## Commands executed

```bash
PYTHONPATH=/home/prokop/git/pyscf \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 expamples_prokop/test_small_dft.py

PYTHONPATH=/home/prokop/git/pyscf \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 expamples_prokop/profile_scf_cycle.py \
  --mol benzene --path ref smallDFT_ws --threads 1 4 14 --repeat 1

PYTHONPATH=/home/prokop/git/pyscf \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -c "from pyscf.smallDFT import profile_xc_bottleneck; profile_xc_bottleneck('benzene', nthreads=14)"

PYTHONPATH=/home/prokop/git/pyscf \
python3 expamples_prokop/test_opencl_xc_full_gpu_parity.py
```

The smallDFT stdout was also saved to
`debug/acceptance_2026-07-11/test_small_dft.out`.

## Numerical rigor

The following checks passed:

| Test | Result |
|---|---:|
| H₂O reference vs smallDFT PBE vmat | max difference `1.39e-15` |
| Benzene reference vs smallDFT PBE vmat | max difference `4.27e-15` |
| H₂O LDA C/OpenMP vs Python | max difference `1.14e-13` |
| Benzene LDA C/OpenMP vs Python | max difference `9.95e-14` |
| H₂O PBE `rho_gga` C/OpenMP vs Python | max difference `9.09e-13` |
| Benzene PBE `rho_gga` C/OpenMP vs Python | max difference `1.82e-12` |

The tests also performed normal PySCF SCF setup and produced finite converged
energies for H₂O and benzene. These are component and single-path checks; they
do not replace a multi-geometry CPU-vs-GPU acceptance run on a real GPU.

## CPU performance

Real `mf.kernel(max_cycle=1)` instrumentation was used. The reported `cycle`
column is the second `get_veff` call and therefore represents one SCF
iteration after initialization.

### Benzene, 6-31g, PBE, grid level 3

| Path | Threads | `nr_rks` cycle | `get_veff` cycle | Kernel wall |
|---|---:|---:|---:|---:|
| Reference | 1 | 259.7 ms | 262.1 ms | 1087.8 ms |
| Reference | 4 | 114.6 ms | 116.8 ms | 393.9 ms |
| Reference | 14 | 160.7 ms | 163.8 ms | 440.2 ms |
| `smallDFT_ws` | 1 | 105.0 ms | 107.5 ms | 740.2 ms |
| `smallDFT_ws` | 4 | 33.0 ms | 35.1 ms | 216.9 ms |
| `smallDFT_ws` | 14 | 37.1 ms | 40.4 ms | 159.4 ms |

At the measured 14-thread setting, `smallDFT_ws` reduces converged-cycle
`nr_rks` from 160.7 ms to 37.1 ms, or 4.3×, versus the same-setting reference.
Compared with the one-thread reference, the XC kernel reduction is 7.0×.
The 14-thread result is not a scaling claim:
the process is limited to 4 CPUs, and oversubscription makes the reference
14-thread result slower than its 4-thread result.

### CPU XC waterfall at the 14-thread setting

| Stage | Time | Share |
|---|---:|---:|
| C/OpenMP `rho_gga` | 13.5 ms | 40.7% |
| libxc | 3.5 ms | 10.4% |
| C/OpenMP `vmat_gga` | 16.2 ms | 48.9% |
| XC subtotal | 33.2 ms | 100% |
| `eval_ao` per geometry | 109.5 ms | outside XC |

The main CPU gap is therefore grid-parallel or otherwise cheaper `eval_ao`.
Within the cached XC work, vmat is slightly more expensive than rho and is
not yet fused with it.

### Test-harness caveat

The general timing section of `expamples_prokop/test_small_dft.py` calls
`nr_rks(..., n_workers=...)` without a `GridWorkspace` and therefore follows
the Python worker path. Its C/OpenMP parity sections explicitly call
`use_c=True` and are valid correctness checks. Production timing should use
`profile_scf_cycle.py --path smallDFT_ws` or direct C-kernel profiling, not the
`small nw=` rows from the general section.

## GPU/OpenCL result

The first attempt used the restricted runner and saw only a PoCL CPU device.
After switching to the host NVIDIA environment, PyOpenCL reported an NVIDIA
CUDA OpenCL platform and an RTX 3090 with 24 GB device memory.

### Pentacene/PTCDA GPU run

The requested large-molecule runner used:

```bash
PYTHONPATH=/home/prokop/git/pyscf OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -u expamples_prokop/test_opencl_xc_e2e_mols.py \
  --mols pentacene PTCDA --grid-level 2 --n-timed 1 \
  --no-step-audit --skip-gto-ao
```

CPU references and all six GPU XC paths completed on the RTX 3090.

### RTX 3090 GPU XC results

Rows are stages; GPU wall entries include speedup over the CPU `nr_rks`
reference for the same molecule.

| Sub-task | Pent CPU | Pent OTF PBE | Pent radial | Pent coalesced | Pent tiled | PTCDA CPU | PTCDA OTF PBE | PTCDA radial | PTCDA coalesced | PTCDA tiled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Setup | — | 1051.0 ms | 1089.0 ms | 1371.1 ms | 1395.8 ms | — | 1188.7 ms | 1231.9 ms | 1711.6 ms | 1726.3 ms |
| GPU XC wall / speedup | 2999.4 ms | 201.8 ms / **14.86×** | 257.8 ms / 11.63× | 320.8 ms / 9.35× | 1903.3 ms / 1.58× | 4978.8 ms | 285.5 ms / **17.44×** | 390.0 ms / 12.77× | 515.6 ms / 9.66× | 2973.8 ms / 1.67× |
| GPU rho | — | 72.9 ms | 86.7 ms | 168.7 ms | 1110.0 ms | — | 116.7 ms | 149.4 ms | 305.0 ms | 1855.2 ms |
| GPU PBE | — | 0.9 ms | 0.4 ms | 0.5 ms | 0.2 ms | — | 0.6 ms | 0.4 ms | 0.7 ms | 0.2 ms |
| GPU vmat | — | 126.1 ms | 168.8 ms | 150.5 ms | 791.9 ms | — | 166.0 ms | 239.2 ms | 209.3 ms | 1117.6 ms |
| max |vxc_gpu − vxc_cpu| | — | 2.454e-6 | 2.216e-6 | 2.454e-6 | 2.454e-6 | — | 2.605e-6 | 2.605e-6 | 2.843e-6 | 2.843e-6 |

The best path is `gpu_hermite_otf` with GPU PBE: 201.8 ms for pentacene and
285.5 ms for PTCDA. The radial and coalesced paths are slower for these larger
molecules on this RTX 3090; the legacy tiled path is not competitive.

[GPU XC comparison plot](</home/prokop/git/pyscf/debug/acceptance_2026-07-11/pentacene_ptcda_gpu_xc.png>) · [GPU XC data](</home/prokop/git/pyscf/debug/acceptance_2026-07-11/pentacene_ptcda_gpu_xc.csv>)

These are XC-pipeline results; the complete one-iteration SCF result is
reported below, including CPU and GPU DF-J timing.

Historical results already documented in
`doc/GPU_benchmark.md` and `doc/dimer_scan_benchmarks.md` include the
`production_otf_radial_vmat_splitk` profile as the best reported benzene XC
path and successful H₂O/formic dimer scan comparisons for most GPU paths.
Those results also identify the unresolved `gpu_gto` H₂O deviation and the
larger error of the relaxed-tolerance `gpu_full` path.

## Acceptance gaps and priorities

1. Run converged GPU SCF, memory-use, and H₂O/formic dimer scans on the RTX
   3090; the one-iteration full-SCF path is now measured, but not yet accepted
   as a production converged workflow.
2. Improve RI-J / DF J, which becomes the converged-cycle bottleneck when
   cached XC is accelerated.
3. Evaluate a tiled rho → libxc → vmat pipeline; a direct GGA fusion cannot
   construct vmat before rho and libxc have produced vxc.
4. Wire `grid_screen.py` active-atom lists into OTF OpenCL kernels.
5. Fix the benchmark harness so its default performance section uses the C
   path and does not present deprecated Python-worker timings as production
   results.
6. Reconcile older OpenCL documents that still describe rho as the dominant
   stage; current measurements identify vmat and setup/transfer costs instead.
7. Investigate the documented H₂O `gpu_gto` dimer deviation before calling the
   exact-GTO path production-ready.

## Conclusion

The CPU smallDFT implementation is numerically sound and provides a meaningful
SCF-cycle speedup when AO values are cached. The GPU XC path is validated on
the RTX 3090 for pentacene and PTCDA with approximately 15–17× XC speedups and
`2–3e-6` vxc parity errors. Full one-iteration GPU SCF is also measured below;
converged SCF and dimer-scan validation remain.

## 2026-07-11 workspace AO reuse follow-up

`GridWorkspace` now owns the raw C-contiguous buffer required by libcint and
passes it directly through `eval_ao_native(buf=...)`. On the same benzene case,
repeated minimum AO timings changed from 120.5 to 97.3 ms at one thread and
from 54.5 to 35.2 ms at the effective four-thread setting. AO and full
workspace `nr_rks` parity are exact to the displayed numerical precision.

The subsequent full SCF profile with density fitting identifies the next
steady-state bottleneck: at four threads, `smallDFT_ws` takes 38.3 ms in XC
and 46.3 ms in CPU DF J per converged cycle. The next CPU optimization should
therefore target DF J or reduce its use; it should not duplicate libcint's
existing OpenMP grid-AO implementation.

## Pentacene and PTCDA one-iteration comparison

The existing SCF profiler was reused with the XYZ geometries in
`data/xyz/`. These are `mf.kernel(max_cycle=1)` runs at PBE/6-31g, grid level
2, four requested CPU threads, and one initialization plus one measured SCF
cycle. The process exposes four CPUs, so this is the useful non-oversubscribed
setting on this host.

### Density-fitted comparison

Rows are sub-tasks. Columns compare the two methods and molecules; speedup is
vanilla divided by `smallDFT_ws`.

| Sub-task | Pentacene vanilla | Pentacene `smallDFT_ws` | Pentacene speedup | PTCDA vanilla | PTCDA `smallDFT_ws` | PTCDA speedup |
|---|---:|---:|---:|---:|---:|---:|
| AO setup, once/geometry | — | 660.0 ms | — | — | 618.1 ms | — |
| Cycle XC | 1328.6 ms | 454.4 ms | **2.92×** | 1928.0 ms | 841.8 ms | **2.29×** |
| Cycle DF J | 760.8 ms | 756.8 ms | 1.01× | 1328.8 ms | 1454.2 ms | 0.91× |
| Cycle `get_veff` | 2089.7 ms | 1211.5 ms | **1.73×** | 3257.1 ms | 2296.3 ms | **1.42×** |
| `mf.kernel(max_cycle=1)` wall | 4596.1 ms | 2929.2 ms | **1.57×** | 6875.7 ms | 4692.2 ms | **1.47×** |

The steady-state XC speedups are 2.9× for pentacene and 2.3× for PTCDA.
Total `get_veff` speedups are 1.7× and 1.4× because DF J is already the
dominant remaining cost. Total one-iteration kernel wall speedups are 1.57×
and 1.46×, respectively (`4596.1 / 2929.2` and `6875.7 / 4692.2`).

### CPU scaling of the steady-state cycle

The plot uses fresh-process measurements at 1, 2, and 4 CPUs; four is the
maximum usable CPU count in this environment. The CSV contains the plotted
values.

[Scaling plot](</home/prokop/git/pyscf/debug/acceptance_2026-07-11/pentacene_ptcda_scaling.png>) · [Scaling data](</home/prokop/git/pyscf/debug/acceptance_2026-07-11/pentacene_ptcda_scaling.csv>)

The plotted `get_veff` times are:

| Sub-task / method | Pentacene 1→2→4 CPUs (ms) | PTCDA 1→2→4 CPUs (ms) |
|---|---:|---:|
| Cycle XC, vanilla | 2636.6 → 1612.3 → 1199.4 | 4480.0 → 2508.2 → 1928.0 |
| Cycle XC, `smallDFT_ws` | 1442.4 → 767.1 → 429.3 | 2635.6 → 1319.8 → 841.8 |
| Cycle DF J, vanilla | 2820.6 → 1401.8 → 717.7 | 4963.5 → 2499.1 → 1328.8 |
| Cycle DF J, `smallDFT_ws` | 2797.4 → 1430.8 → 723.5 | 4911.4 → 2458.9 → 1454.2 |
| Cycle `get_veff`, vanilla | 5457.5 → 3014.3 → 1917.4 | 9443.7 → 5007.5 → 3257.1 |
| Cycle `get_veff`, `smallDFT_ws` | 4240.0 → 2198.2 → 1153.0 | 7547.2 → 3779.0 → 2296.3 |

The XC path scales well up to four CPUs, especially after C/OpenMP
smallDFT. DF J also scales in the full SCF profile, but the isolated in-memory
contraction does not; its large SCF cost includes setup/loading and should not
be attacked as a simple GEMM first.

### DF-J optimization experiment

Explicitly building the DF tensor before entering SCF removes much of the
one-iteration setup/loading cost:

| Sub-task | Pentacene default | Pentacene prebuilt | Speedup | PTCDA default | PTCDA prebuilt | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Grid build | 569.6 ms | 445.4 ms | — | 699.8 ms | 539.0 ms | — |
| DF tensor build | included in first SCF call | 1012.2 ms, before SCF | — | included in first SCF call | 2009.7 ms, before SCF | — |
| `mf.kernel(max_cycle=1)` wall | 4596.1 ms | 2564.3 ms | **1.79×** | 6875.7 ms | 4026.5 ms | **1.71×** |

Once the DF tensor is resident in memory, isolated J contraction timings are
approximately 23 ms for pentacene and 48 ms for PTCDA, with little benefit from
1→4 CPUs or block sizes 240→1920. The next implementation should therefore
make DF build/cache lifetime explicit and avoid repeated HDF5/out-of-core
loading. GPU DF J remains the strongest future option when GPU hardware is
available.

### Vanilla direct-J comparison

Pentacene without density fitting was also run in fresh processes. The vanilla
cycle was 1369.2 ms, consisting of 1271.4 ms XC and 97.6 ms direct J. The
smallDFT cycle had 442.4 ms XC, but its direct-J behavior became anomalously
slow and the total cycle did not improve. The combined profiler is not safe to
reuse across paths because its monkey-patches accumulate; therefore only the
fresh-process vanilla result and the XC reduction are treated as reliable.

PTCDA no-DF was not completed within the bounded run: the direct four-center J
cost is too large for this environment. This confirms that DF is required for
meaningful large-molecule CPU comparisons.

The direct-J anomaly is a separate performance gap to investigate before
making `smallDFT` the default for non-DF RKS. The next diagnostic should time
the same `get_j` call before and after one smallDFT XC call while checking
`lib.num_threads()`, OpenMP runtime state, and direct-SCF screening reuse.

That isolated diagnostic was run for pentacene without profiler patches: a
full-density `get_j` took 4386.6 ms before smallDFT XC and 4367.7 ms after it,
with `lib.num_threads() == 4` both times. Thus smallDFT did not change the
global thread count; the remaining discrepancy is in the direct-SCF
incremental-J/cache state used by the one-cycle driver and needs a narrower
trace of `dm`, `dm_last`, and `vhf_last.vj`.

## GPU full-SCF one-iteration Amdahl profile

The full SCF driver was run in fresh sequential cases on the host NVIDIA RTX
3090, with PBE/6-31g, grid level 2, density fitting, `max_cycle=1`, and four
CPU threads (`OPENBLAS_NUM_THREADS=1`). `gpu_otf` uses GPU XC with CPU DF-J;
`gpu_full` uses GPU XC and GPU DF-J. The kernel wall includes initialization,
grid construction, first Fock build, and the measured iteration. The cycle
columns isolate the second `get_veff`/DF-J call, so they expose the steady-state
Amdahl limit. `get_veff` includes its nested J call; the rows are therefore
diagnostic timings, not additive components.

| Sub-task | Pentacene CPU | Pentacene `gpu_otf` | Pentacene `gpu_full` | PTCDA CPU | PTCDA `gpu_otf` | PTCDA `gpu_full` |
|---|---:|---:|---:|---:|---:|---:|
| Full `mf.kernel(max_cycle=1)` wall | 4403.0 ms | 1908.3 ms / **2.31×** | 1642.3 ms / **2.68×** | 7096.8 ms | 3177.5 ms / **2.23×** | 2967.4 ms / **2.39×** |
| Initial `get_veff` | 2443.6 ms | 957.2 ms | 1398.9 ms | 3874.6 ms | 1592.1 ms | 2600.2 ms |
| Steady-state cycle `get_veff` | 1909.0 ms | 907.4 ms | 199.5 ms | 3150.4 ms | 1519.7 ms | 300.9 ms |
| Initial DF-J | 751.1 ms | 753.3 ms | 1192.3 ms | 1302.9 ms | 1307.2 ms | 2318.9 ms |
| Steady-state cycle DF-J | 710.6 ms | 720.9 ms | **11.1 ms** | 1278.9 ms | 1249.1 ms | **20.5 ms** |
| Diagonalization | 4.3 ms | 4.1 ms | 4.2 ms | 7.4 ms | 7.7 ms | 7.5 ms |
| GPU XC measured stage sum | — | 386.4 ms | 392.0 ms | — | 551.1 ms | 557.2 ms |

The full-kernel speedups are much smaller than the isolated GPU-XC speedups
because initialization and CPU-side work remain on the critical path. The
largest GPU-specific gain is cached DF-J: after the first build, GPU DF-J is
about 11 ms for pentacene and 21 ms for PTCDA. Its initial cost is still large
(1.19 s and 2.32 s respectively), so persistent DF tensor lifetime and setup
amortization are the next GPU optimization targets. Diagonalization is
negligible here; optimizing it cannot materially improve total wall time.

[GPU full-SCF Amdahl plot](</home/prokop/git/pyscf/debug/acceptance_2026-07-11/pentacene_ptcda_gpu_scf_amdahl.png>) · [GPU full-SCF data](</home/prokop/git/pyscf/debug/acceptance_2026-07-11/pentacene_ptcda_gpu_scf_amdahl.csv>)

The reproducible driver is
`expamples_prokop/profile_gpu_scf.py`; its timer output now reports separate
`init_ms` and `cycle_ms` columns. These runs were one-iteration performance
measurements, not convergence or final-energy acceptance tests: both GPU
profiles intentionally have different documented SCF tolerance/accuracy
settings, and all cases reported `converged=False` because only one cycle was
requested.

## Detailed three-cycle bottleneck decomposition — lazy baseline

The one-cycle result is useful for measuring latency, but it over-weights
one-time setup. A second run used three SCF iterations and retained the setup
timers. The end-to-end value includes GPU backend setup in addition to the
`mf.kernel()` wall; CPU grid construction is already inside its kernel wall.

This section is the pre-fix baseline: DF build/plan preparation was still
lazy for the CPU reference and GPU DF path. The corrected static-preparation
benchmark follows below.

| Sub-task | Pent CPU | Pent `gpu_otf` | Pent `gpu_full` | PTCDA CPU | PTCDA `gpu_otf` | PTCDA `gpu_full` |
|---|---:|---:|---:|---:|---:|---:|
| End-to-end 3-cycle wall | 8.81 s | 5.03 s / 1.75× | 3.16 s / **2.79×** | 13.37 s | 7.47 s / 1.79× | 4.92 s / **2.72×** |
| External GPU/XC setup | — | 1.13 s | 1.14 s | — | 1.26 s | 1.26 s |
| `mf.kernel()` wall | 8.81 s | 3.89 s | 2.02 s | 13.36 s | 6.20 s | 3.59 s |
| Grid build | 0.44 s | included in setup | included in setup | 0.52 s | included in setup | included in setup |
| Initial DF-J / DF plan | 0.79 s CPU DF-J | 0.80 s CPU DF-J | 1.17 s GPU DF-J + plan | 1.31 s CPU DF-J | 1.30 s CPU DF-J | 2.37 s GPU DF-J + plan |
| Repeated cycle `get_veff` | 2.01–2.10 s | 0.94–0.95 s | 0.18–0.19 s | 3.10–3.14 s | 1.51–1.52 s | 0.26–0.28 s |
| Repeated DF-J | 0.74–0.79 s | 0.74–0.75 s | **10–12 ms** | 1.24–1.28 s | 1.25–1.26 s | **20 ms** |
| Repeated eigensolver | 4–5 ms | 4 ms | 4 ms | 7–8 ms | 7–8 ms | 7 ms |
| Repeated GPU-XC outer wall | — | 191–198 ms | 184–187 ms | — | 260–263 ms | 259–261 ms |

This resolves the apparent contradiction. Isolated GPU XC is approximately
10–14× faster, but for a single calculation the GPU pays 1.1–1.3 s of XC
setup and a first DF plan cost of 1.2–2.4 s. Once those are amortized, the
repeated GPU-full cycle is only about 185 ms for pentacene and 260 ms for
PTCDA. CPU/GPU Python orchestration, eigensolver, and density assembly are all
below a few milliseconds and are not worth optimizing yet.

The largest actionable opportunities are therefore:

1. Cache/reuse the GPU XC plan and DF-J plan across geometries and SCF jobs
   when the basis/grid topology is unchanged; avoid rebuilding/uploading DF
   tensors for every single point.
2. Make DF tensor construction explicit and asynchronous with GPU plan setup,
   or persist a device-resident `cderi`; this attacks the 1.2–2.4 s first-call
   cost directly.
3. Optimize the repeated GPU XC wall, especially the Hermite `rho` and vmat
   path. The OpenCL event sum is ~250–355 ms per cycle, while the measured
   outer host wall is ~185–261 ms; event sums overlap and must not be added to
   the SCF wall.
4. For `gpu_otf`, GPU XC is already hidden behind a 0.74–1.26 s CPU DF-J
   call per cycle. Moving only XC further will have little total effect until
   CPU DF-J is replaced by `gpu_full` or a faster cached CPU implementation.

[Detailed GPU Amdahl plot](</home/prokop/git/pyscf/debug/acceptance_2026-07-11/pentacene_ptcda_gpu_scf_amdahl.png>) · [3-cycle GPU profile JSON](</home/prokop/git/pyscf/debug/acceptance_2026-07-11/gpu_full_setup_detailed_3cycle.json>)

## Static preparation fix: no invariant DF work inside SCF cycles

The lifecycle audit found that `gpu_full` created the DF tensor, uploaded the
GPU DF-J plan, and allocated the first density buffers lazily from the first
`get_jk`. CPU DF also entered an out-of-core path when the tensor was first
created from inside the SCF call. This made the first cycle contain work that
is invariant for a fixed molecule, basis, auxiliary basis, and grid.

`prepare_df_for_scf(mf)` now runs from the GPU profile/setup path and performs:

| Lifetime | Work | Expected call count |
|---|---|---:|
| Molecule/geometry | AO/grid construction, DF 3-center tensor build | once per geometry |
| SCF-loop setup | GPU XC plan/table/buffer setup; GPU DF-J plan and buffers | once per `mf.kernel()` setup |
| SCF cycle | density-dependent XC and J/K contraction, Fock, eigensolve | once per iteration |

The prepared two-cycle benchmark confirms that DF setup no longer appears in
the first `get_veff`:

| Sub-task | Pent CPU | Pent `gpu_otf` | Pent `gpu_full` | PTCDA CPU | PTCDA `gpu_otf` | PTCDA `gpu_full` |
|---|---:|---:|---:|---:|---:|---:|
| DF build, before kernel | 1.06 s | 1.07 s | 1.08 s | 2.08 s | 2.13 s | 2.12 s |
| GPU XC setup, before kernel | — | 0.61 s | 0.64 s | — | 0.74 s | 0.72 s |
| GPU DF plan, before kernel | — | — | 0.12 s | — | — | 0.24 s |
| CPU DF-J, repeated cycle | 25 ms | 25 ms | — | 50 ms | 50 ms | — |
| GPU DF-J, repeated cycle | — | — | 11–13 ms | — | — | 20–22 ms |
| First `get_veff` after preparation | 1.72 s | 0.25 s | 0.26 s | 2.81 s | 0.34 s | 0.37 s |
| Later `get_veff` | 1.23–1.30 s | 0.23 s | 0.20–0.22 s | 1.98 s | 0.33 s | 0.31 s |

The old apparent CPU DF-J bottleneck was therefore mostly avoidable data
preparation, not the in-memory contraction. This is the central result of the
deep profile. The prepared path gives full two-cycle kernel speedups of about
5.8× for pentacene and 6.4× for PTCDA; end-to-end speedup is lower because the
one-time DF/grid/GPU setup is now explicitly visible.

[Prepared two-cycle profile JSON](</home/prokop/git/pyscf/debug/acceptance_2026-07-11/gpu_scf_prepared_2cycle.json>)

## Definitive same-cycle Amdahl profile

The earlier tables mixed isolated XC measurements, different CPU thread counts,
and setup-inclusive timings. The strict profile uses the same PBE/6-31g, grid
level 2, density matrix, four CPU threads, and prebuilt static data for both
paths. It manually reproduces one RKS cycle using non-overlapping stages, then
checks the assembled matrix against a real `mf.get_veff` call. The matrix error
is zero at displayed precision and the manual/real wall times agree within
measurement noise.

| Repeated SCF-cycle stage | Pent CPU | Pent GPU full | Speedup | PTCDA CPU | PTCDA GPU full | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| AO evaluation + density contraction (`rho`) | 719.8 ms | 68.5 ms | **10.5×** | 1058.7 ms | 117.7 ms | **9.0×** |
| XC functional (libxc/PBE) | 14.0 ms | 1.0 ms | 14.0× | 15.1 ms | 0.2 ms | 75× |
| XC potential matrix (`vmat`) | 482.3 ms | 114.4 ms | **4.2×** | 800.0 ms | 161.2 ms | **5.0×** |
| Other XC bookkeeping | 1.0 ms | &lt;1 ms | — | 1.0 ms | &lt;1 ms | — |
| XC total | 1220.2 ms | 185.7 ms | **6.6×** | 1878.3 ms | 281.1 ms | **6.7×** |
| DF-J | 23.8 ms | 11.2 ms | 2.1× | 48.3 ms | 20.8 ms | 2.3× |
| Eig + occupancy + DM | 4.7 ms | 4.5 ms | 1.0× | 7.8 ms | 8.5 ms | 0.9× |
| Other SCF operations | 1.0 ms | 0.9 ms | — | 1.1 ms | 1.3 ms | — |
| **Cycle total** | **1249.4 ms** | **202.1 ms** | **6.18×** | **1935.3 ms** | **311.7 ms** | **6.21×** |
| Real `get_veff` wall | 1246.7 ms | 199.2 ms | 6.3× | 1921.7 ms | 303.8 ms | 6.3× |

This resolves the apparent 14× versus 2× contradiction. On the exact
four-thread comparison, `vmat` is not 14× faster: it is 4–5× faster, and it
was only 40–43% of the original CPU XC wall. The 10–14× number came from an
isolated benchmark with a different CPU baseline (not the four-thread full
cycle). `rho` plus AO evaluation is the other 57–59% of CPU XC. Accelerating
both gives a real repeated-cycle speedup of about 6.2×.

The observed ~2× whole-calculation speedup is therefore setup amortization,
not an unexplained per-cycle bottleneck. One-time CPU/GPU setup is respectively
1.11/2.31 s for pentacene and 2.10/3.86 s for PTCDA. At a small number of SCF
cycles this fixed cost dominates; as the number of cycles grows, total speedup
approaches the 6.2× repeated-cycle limit.

The next repeated-cycle targets are unambiguous: `vmat` first (57–62% of GPU
XC wall), then AO/rho (37–42%). GPU PBE, DF-J, eigensolver, DIIS, and transfers
are not material bottlenecks.

The reproducible runner is `expamples_prokop/profile_gpu_amdahl_strict.py`.
