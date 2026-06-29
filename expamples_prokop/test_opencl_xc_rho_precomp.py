#!/usr/bin/env python3
"""Rho projection parity: coalesced chi vs radial precomp vs tiled reference."""
import os
import sys
import time

import numpy as np
from pyscf import gto, dft, lib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_XYZ = os.path.join(_REPO, 'data', 'xyz', 'benzene.xyz')


def read_xyz(path):
    with open(path) as f:
        lines = f.readlines()
    natom = int(lines[0].strip())
    return '; '.join(' '.join(line.split()[:4]) for line in lines[2:2 + natom] if line.strip())


def cpu_rho(ni, mol, grids, dm, xctype):
    dm = np.asarray(dm, order='C', dtype=np.float64)
    make_rho, nset, _ = ni._gen_rho_evaluator(mol, dm, hermi=1, with_lapl=False, grids=grids)
    ngrids = grids.coords.shape[0]
    blksize = 8192
    rho = np.zeros((4 if xctype == 'GGA' else 1, ngrids), dtype=np.float64)
    for ip0 in range(0, ngrids, blksize):
        ip1 = min(ip0 + blksize, ngrids)
        ao = ni.eval_ao(mol, grids.coords[ip0:ip1], deriv=1 if xctype == 'GGA' else 0)
        blk = make_rho(0, ao, None, xctype)
        rho[:, ip0:ip1] = blk if xctype == 'GGA' else blk[np.newaxis, :]
    return rho


def main():
    from pyscf.OpenCL.xc_grid import get_xc_grid_plan, clear_xc_plan_cache
    from pyscf.OpenCL import init_device, reset_opencl

    mol = gto.M(atom=read_xyz(_XYZ), basis='ccpvdz', verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build(with_non0tab=True)
    ni = dft.numint.NumInt()
    mf = dft.RKS(mol, xc='PBE').density_fit()
    dm = mf.get_init_guess()
    rho_ref = cpu_rho(ni, mol, grids, dm, 'GGA')
    log = lambda m: print(m, flush=True)
    log(f'benzene nao={mol.nao_nr()} ngrids={grids.coords.shape[0]} OMP={lib.num_threads()}')

    paths = (
        ('tiled', 'tiled'),
        ('coalesced', 'coalesced'),
        ('radial_precomp', 'radial_precomp'),
    )
    for label, fused in paths:
        clear_xc_plan_cache()
        reset_opencl()
        init_device(quiet=True)
        plan = get_xc_grid_plan(mol, grids, 'PBE')
        t0 = time.perf_counter()
        plan.setup_precomputed_gto(gpu_only=True, xc_eval='cpu', fused=fused)
        setup_ms = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        rho_gpu = plan.nr_rks_precomputed_rho_only(dm, profile=True)
        wall_ms = (time.perf_counter() - t0) * 1e3
        err = float(np.abs(rho_gpu - rho_ref).max())
        rel = err / max(float(np.abs(rho_ref).max()), 1e-10)
        log(f'{label:16s} setup={setup_ms:7.1f} ms  rho={wall_ms:7.1f} ms  max_err={err:.3e}  rel={rel:.3e}')
        if plan.precalc_timing.get('radial_gpu'):
            log(f'  radial_gpu={plan.precalc_timing["radial_gpu"]*1e3:.1f} ms')
    log('Done.')


if __name__ == '__main__':
    main()
