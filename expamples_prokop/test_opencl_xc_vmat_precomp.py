#!/usr/bin/env python3
"""Vmat parity: tiled vs coalesced chi vs radial precomp (optimized vmat kernels)."""
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


def cpu_wv(ni, mol, grids, rho, xctype):
    weight = grids.weights
    if xctype == 'GGA':
        evfk = ni.eval_xc_eff('PBE', rho, deriv=1, xctype='GGA', spin=0)
        vxc = evfk[1]
        wv = weight[np.newaxis, :].astype(np.float32) * np.ascontiguousarray(vxc, dtype=np.float32)
        wv[0] *= 0.5
        return wv
    exc, vxc = ni.eval_xc_eff('PBE', rho[0], deriv=1, xctype='LDA', spin=0)[:2]
    return (weight * vxc).astype(np.float32)


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
    log = lambda m: print(m, flush=True)
    log(f'benzene nao={mol.nao_nr()} ngrids={grids.coords.shape[0]} OMP={lib.num_threads()}')

    n_ref, exc_ref, vxc_ref = ni.nr_rks(mol, grids, 'PBE', dm, max_memory=2000)
    log(f'cpu_libxc  nelec={n_ref:.6f}  exc={exc_ref:.8f}  vxc_max={np.abs(vxc_ref).max():.6f}')

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
        rho_gpu = plan.nr_rks_precomputed_rho_only(dm, profile=False)
        wv = cpu_wv(ni, mol, grids, rho_gpu, 'GGA')
        t0 = time.perf_counter()
        vmat_gpu = plan.nr_rks_precomputed_vmat_only(wv, profile=True)
        vmat_ms = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        n_full, exc_full, vxc_full = plan.nr_rks_precomputed_gto(dm, projection='gpu', profile=True)
        full_ms = (time.perf_counter() - t0) * 1e3
        tim = plan.last_timing or {}
        err_vmat = float(np.abs(vmat_gpu - vxc_ref).max())
        err_full = float(np.abs(vxc_full - vxc_ref).max())
        rel_vmat = err_vmat / max(float(np.abs(vxc_ref).max()), 1e-10)
        rel_full = err_full / max(float(np.abs(vxc_ref).max()), 1e-10)
        log(f'{label:16s} setup={setup_ms:7.1f} ms  vmat_only={vmat_ms:6.1f} ms (gpu_vmat={tim.get("gpu_vmat", 0)*1e3:.1f})  '
            f'full={full_ms:6.1f} ms  vmat_err={err_vmat:.3e}  full_err={err_full:.3e}')
        if plan.precalc_timing.get('eval_ao_cpu'):
            log(f'  eval_ao_cpu={plan.precalc_timing["eval_ao_cpu"]*1e3:.1f} ms')
        if plan.precalc_timing.get('eval_ao_hermite_gpu'):
            log(f'  eval_ao_hermite_gpu={plan.precalc_timing["eval_ao_hermite_gpu"]*1e3:.1f} ms  ao_proj={plan.precalc_timing.get("ao_proj")}')
        if plan.precalc_timing.get('radial_gpu'):
            log(f'  radial_gpu={plan.precalc_timing["radial_gpu"]*1e3:.1f} ms')
        if rel_vmat > 5e-3 or rel_full > 5e-3:
            raise SystemExit(f'{label}: vmat parity failed rel_vmat={rel_vmat:.3e} rel_full={rel_full:.3e}')
    log('Done.')


if __name__ == '__main__':
    main()
