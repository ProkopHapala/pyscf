import time
import numpy as np
from pyscf import gto, dft
from pyscf.OpenCL.xc_grid import nr_rks_gpu_hermite_ao


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
    t0 = time.perf_counter()
    n_cpu, exc_cpu, vxc_cpu = ni.nr_rks(mol, grids, 'pbe', dm, max_memory=2000)
    print(f'CPU PySCF time {time.perf_counter() - t0:.3f}s')
    t0 = time.perf_counter()
    n_gpu, exc_gpu, vxc_gpu = nr_rks_gpu_hermite_ao(mol, grids, 'pbe', dm, max_memory=2000)
    print(f'GPU Hermite AO time {time.perf_counter() - t0:.3f}s')
    err_n = abs(n_cpu - n_gpu) / max(abs(n_cpu), 1e-10)
    err_exc = abs(exc_cpu - exc_gpu) / max(abs(exc_cpu), 1e-10)
    err_vxc = np.abs(vxc_cpu - vxc_gpu).max()
    err_vxc_rel = err_vxc / max(np.abs(vxc_cpu).max(), 1e-10)
    print(f'nelec CPU/GPU {n_cpu:.12f} {n_gpu:.12f} rel_err={err_n:.3e}')
    print(f'exc   CPU/GPU {exc_cpu:.12f} {exc_gpu:.12f} rel_err={err_exc:.3e}')
    print(f'vxc max_abs_err={err_vxc:.3e} max_rel_err={err_vxc_rel:.3e}')
    if err_vxc > 5e-3:
        raise SystemExit(f'Hermite AO XC parity failed: vxc max_abs_err={err_vxc}')


if __name__ == '__main__':
    main()
