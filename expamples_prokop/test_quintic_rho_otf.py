#!/usr/bin/env python3
"""Compare cubic vs quintic Hermite OTF rho projection vs CPU reference."""
import os

import numpy as np
from pyscf import dft, gto, lib

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_XYZ = os.path.join(_REPO, 'data', 'xyz', 'formic_dimer.xyz')


def read_xyz(path):
    with open(path) as f:
        lines = f.readlines()
    natom = int(lines[0].strip())
    return '; '.join(' '.join(line.split()[:4]) for line in lines[2:2 + natom] if line.strip())


def cpu_rho(ni, mol, grids, dm, xctype):
    dm = np.asarray(dm, order='C', dtype=np.float64)
    make_rho, _, _ = ni._gen_rho_evaluator(mol, dm, hermi=1, with_lapl=False, grids=grids)
    ngrids = grids.coords.shape[0]
    rho = np.zeros((4 if xctype == 'GGA' else 1, ngrids), dtype=np.float64)
    for ip0 in range(0, ngrids, 8192):
        ip1 = min(ip0 + 8192, ngrids)
        ao = ni.eval_ao(mol, grids.coords[ip0:ip1], deriv=1 if xctype == 'GGA' else 0)
        blk = make_rho(0, ao, None, xctype)
        rho[:, ip0:ip1] = blk if xctype == 'GGA' else blk[np.newaxis, :]
    return rho


def gpu_rho(mol, grids, dm, spline_order):
    from pyscf.OpenCL import init_device, reset_opencl
    from pyscf.OpenCL.xc_grid import get_xc_grid_plan, clear_xc_plan_cache
    reset_opencl()
    init_device(quiet=True)
    clear_xc_plan_cache()
    plan = get_xc_grid_plan(mol, grids, 'PBE')
    plan.setup_onthefly(spline_order=spline_order)
    rho = plan.nr_rks_hermite_rho_only(dm)
    meta = plan.ao_hermite.plan
    table_mb = meta.radial_nodes.nbytes / 1e6
    return rho, meta.nrad, meta.du, meta.du_cubic_ref, table_mb


def report(label, rho, ref, rho_peak):
    err = rho - ref
    max_abs = float(np.max(np.abs(err)))
    max_rel = max_abs / rho_peak
    rms = float(np.sqrt(np.mean(err * err)))
    print(f'{label:44s}  max|Δρ|={max_abs:.3e}  max_rel={max_rel:.3e}  RMS={rms:.3e}', flush=True)
    return rho


def main():
    lib.num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    atom = read_xyz(_XYZ)
    mol = gto.M(atom=atom, basis='6-31g', verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build(with_non0tab=True)
    ni = dft.numint.NumInt()
    mf = dft.RKS(mol, xc='PBE').density_fit()
    dm = mf.get_init_guess()
    rho_ref = cpu_rho(ni, mol, grids, dm, 'GGA')
    rho_peak = max(float(np.abs(rho_ref).max()), 1e-10)
    print(f'formic dimer  nao={mol.nao_nr()}  ngrids={grids.coords.shape[0]}', flush=True)
    rhos = {}
    for order in ('cubic', 'quintic'):
        rho, nrad, du, du_in, table_mb = gpu_rho(mol, grids, dm, order)
        label = f'{order} (nrad={nrad}, du={du:.4f}, table={table_mb:.3f} MB)'
        rhos[order] = report(label, rho, rho_ref, rho_peak)
    d = rhos['quintic'] - rhos['cubic']
    max_abs = float(np.max(np.abs(d)))
    print(f'{"quintic − cubic":44s}  max|Δρ|={max_abs:.3e}  max_rel={max_abs/rho_peak:.3e}  RMS={float(np.sqrt(np.mean(d*d))):.3e}', flush=True)


if __name__ == '__main__':
    main()
