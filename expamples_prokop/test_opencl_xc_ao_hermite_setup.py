#!/usr/bin/env python3
"""Compare CPU eval_ao vs GPU Hermite tiled AO setup for precomp paths."""
import os
import sys
import time

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


def main():
    from pyscf.OpenCL.xc_grid import get_xc_grid_plan, clear_xc_plan_cache
    from pyscf.OpenCL import init_device, reset_opencl

    mol = gto.M(atom=read_xyz(_XYZ), basis='ccpvdz', verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build(with_non0tab=True)
    dm = dft.RKS(mol, xc='PBE').density_fit().get_init_guess()
    log = lambda m: print(m, flush=True)
    log(f'benzene nao={mol.nao_nr()} ngrids={grids.coords.shape[0]} OMP={lib.num_threads()}')

    for ao_proj, fused in (('cpu', 'coalesced'), ('hermite_gpu', 'coalesced'), ('hermite_gpu', 'tiled')):
        clear_xc_plan_cache()
        reset_opencl()
        init_device(quiet=True)
        plan = get_xc_grid_plan(mol, grids, 'PBE')
        t0 = time.perf_counter()
        plan.setup_precomputed_gto(gpu_only=True, xc_eval='cpu', fused=fused, ao_proj=ao_proj)
        setup_ms = (time.perf_counter() - t0) * 1e3
        pt = plan.precalc_timing
        label = f'{fused}+{ao_proj}'
        log(f'{label:28s} setup={setup_ms:7.1f} ms  eval_ao_cpu={pt.get("eval_ao_cpu",0)*1e3:7.1f}  '
            f'eval_ao_gpu={pt.get("eval_ao_hermite_gpu",0)*1e3:7.1f}  total_setup={pt.get("setup_total",0)*1e3:.1f}')
        t0 = time.perf_counter()
        plan.nr_rks_precomputed_gto(dm, projection='gpu', profile=True)
        xc_ms = (time.perf_counter() - t0) * 1e3
        log(f'  per-SCF XC={xc_ms:.1f} ms  gpu_rho={plan.last_timing.get("gpu_rho",0)*1e3:.1f}  gpu_vmat={plan.last_timing.get("gpu_vmat",0)*1e3:.1f}')
    log('Done.')


if __name__ == '__main__':
    main()
