import time
import numpy as np
from pyscf import gto, dft
from pyscf.OpenCL.xc_grid import nr_rks_gpu_hermite_onthefly, nr_rks_gpu_hermite_ao


def main():
    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz', verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build()
    nao = mol.nao_nr()
    np.random.seed(42)
    dm = np.random.rand(nao, nao)
    dm = 0.5 * (dm + dm.T)
    ni = dft.numint.NumInt()

    # CPU reference
    t0 = time.perf_counter()
    n_cpu, exc_cpu, vxc_cpu = ni.nr_rks(mol, grids, 'pbe', dm, max_memory=2000)
    print(f'CPU PySCF time      {time.perf_counter() - t0:.3f}s')

    # GPU on-the-fly
    t0 = time.perf_counter()
    n_otf, exc_otf, vxc_otf = nr_rks_gpu_hermite_onthefly(mol, grids, 'pbe', dm, max_memory=2000)
    print(f'GPU on-the-fly time {time.perf_counter() - t0:.3f}s')

    # GPU Hermite AO (materialized)
    t0 = time.perf_counter()
    n_ao, exc_ao, vxc_ao = nr_rks_gpu_hermite_ao(mol, grids, 'pbe', dm, max_memory=2000)
    print(f'GPU Hermite AO time {time.perf_counter() - t0:.3f}s')

    print()
    print('=== On-the-fly vs CPU ===')
    err_n = abs(n_cpu - n_otf) / max(abs(n_cpu), 1e-10)
    err_exc = abs(exc_cpu - exc_otf) / max(abs(exc_cpu), 1e-10)
    err_vxc = np.abs(vxc_cpu - vxc_otf).max()
    err_vxc_rel = err_vxc / max(np.abs(vxc_cpu).max(), 1e-10)
    print(f'nelec CPU/OTF {n_cpu:.12f} {n_otf:.12f} rel_err={err_n:.3e}')
    print(f'exc   CPU/OTF {exc_cpu:.12f} {exc_otf:.12f} rel_err={err_exc:.3e}')
    print(f'vxc  max_abs_err={err_vxc:.3e} max_rel_err={err_vxc_rel:.3e}')

    print()
    print('=== On-the-fly vs Hermite AO (materialized) ===')
    err_n2 = abs(n_ao - n_otf) / max(abs(n_ao), 1e-10)
    err_exc2 = abs(exc_ao - exc_otf) / max(abs(exc_ao), 1e-10)
    err_vxc2 = np.abs(vxc_ao - vxc_otf).max()
    err_vxc2_rel = err_vxc2 / max(np.abs(vxc_ao).max(), 1e-10)
    print(f'nelec AO/OTF {n_ao:.12f} {n_otf:.12f} rel_err={err_n2:.3e}')
    print(f'exc   AO/OTF {exc_ao:.12f} {exc_otf:.12f} rel_err={err_exc2:.3e}')
    print(f'vxc  max_abs_err={err_vxc2:.3e} max_rel_err={err_vxc2_rel:.3e}')

    if err_vxc > 5e-3:
        raise SystemExit(f'On-the-fly XC parity failed: vxc max_abs_err={err_vxc}')
    print('\nAll checks passed.')


if __name__ == '__main__':
    main()
