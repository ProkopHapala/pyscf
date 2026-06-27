import time
import numpy as np
from pyscf import gto, dft
from pyscf.OpenCL.xc_grid import nr_rks_gpu_hermite_onthefly, nr_rks_gpu_hermite_ao

XYZ_DIR = '/home/prokophapala/git/pyscf/data/xyz'

systems = [
    ('benzene',   'ccpvdz', 3),
    ('pentacene', '631g',   2),
    ('PTCDA',     '631g',   2),
]

for name, basis, grid_level in systems:
    xyz_path = f'{XYZ_DIR}/{name}.xyz'
    mol = gto.M(atom=xyz_path, basis=basis, verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.level = grid_level
    grids.build()
    nao = mol.nao_nr()
    ngrids = grids.coords.shape[0]
    natoms = mol.natm
    print(f'\n=== {name} === atoms={natoms} nao={nao} ngrids={ngrids} basis={basis}')

    np.random.seed(42)
    dm = np.random.rand(nao, nao)
    dm = 0.5 * (dm + dm.T)
    ni = dft.numint.NumInt()

    # CPU reference
    t0 = time.perf_counter()
    n_cpu, exc_cpu, vxc_cpu = ni.nr_rks(mol, grids, 'pbe', dm, max_memory=2000)
    t_cpu = time.perf_counter() - t0

    # GPU on-the-fly
    t0 = time.perf_counter()
    n_otf, exc_otf, vxc_otf = nr_rks_gpu_hermite_onthefly(mol, grids, 'pbe', dm, max_memory=2000)
    t_otf = time.perf_counter() - t0

    # GPU Hermite AO (materialized) - skip for large systems (OOM)
    t_ao = 0.0
    n_ao = n_otf
    vxc_ao = vxc_otf
    if ngrids * nao < 50_000_000:
        t0 = time.perf_counter()
        n_ao, exc_ao, vxc_ao = nr_rks_gpu_hermite_ao(mol, grids, 'pbe', dm, max_memory=2000)
        t_ao = time.perf_counter() - t0

    err_n = abs(n_cpu - n_otf) / max(abs(n_cpu), 1e-10)
    err_exc = abs(exc_cpu - exc_otf) / max(abs(exc_cpu), 1e-10)
    err_vxc = np.abs(vxc_cpu - vxc_otf).max()
    err_vxc_rel = err_vxc / max(np.abs(vxc_cpu).max(), 1e-10)

    print(f'  CPU      {t_cpu:.3f}s')
    print(f'  GPU OTF  {t_otf:.3f}s')
    print(f'  GPU AO   {t_ao:.3f}s')
    print(f'  nelec rel_err={err_n:.3e}  exc rel_err={err_exc:.3e}')
    print(f'  vxc max_abs_err={err_vxc:.3e}  max_rel_err={err_vxc_rel:.3e}')

    if err_vxc > 5e-3:
        print(f'  WARNING: vxc error too large for {name}')
