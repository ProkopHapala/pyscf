# OpenCL XC optimization reports (chronological)

Each report documents **one** optimization pass. Older files are not rewritten.

General guidelines: [`../opencl-kernel-cookbook.md`](../opencl-kernel-cookbook.md).

| # | Date | Report | Summary |
|---|------|--------|---------|
| — | (baseline) | [`../vmat_optimization_report.md`](../vmat_optimization_report.md) | vmat abTile → local AO cache + private `acc[QPT]` |
| 2 | 2026-03 | [2026-03-rho-itile-host-setup.md](2026-03-rho-itile-host-setup.md) | rho: `iTile` inner loop, final `rho[g]` on GPU; pre-SCF `setup_*`; harness/kernel timing |
| 3 | 2026-07 | [2026-07-dimer-scan-xc-paths.md](2026-07-dimer-scan-xc-paths.md) | Dimer E(z) scans (H₂O, formic): all 6 XC paths vs CPU; 39-pt DFTB grid; `profile_dimer_scan.py`; gpu_gto H₂O outlier |

Add the next report as `YYYY-MM-short-topic.md` and a row in this table.
