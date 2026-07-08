#!/usr/bin/env python3
"""Per-stage XC timing for benzene — wall+finish vs OpenCL event profiling.

Usage:
  PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 python3 -u \\
    expamples_prokop/profile_xc_stages_benzene.py
"""
import os
import re
import time

import numpy as np
from pyscf import dft, gto, lib

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_XYZ = os.path.join(_REPO, 'data', 'xyz', 'benzene.xyz')

MODES = (
    ('CPU libxc', None),
    ('OTF cubic', 'production_otf'),
    ('OTF quintic', 'production_otf_quintic'),
    ('Radial precomp', 'production_radial'),
    ('OTF ρ + rad vmat', 'production_otf_radial_vmat'),
)


def read_xyz(path):
    with open(path) as f:
        lines = f.readlines()
    natom = int(lines[0].strip())
    atoms = []
    for line in lines[2:2 + natom]:
        parts = line.split()
        if re.match(r'^[A-Z][a-z]?$', parts[0]):
            atoms.append(f'{parts[0]} {parts[1]} {parts[2]} {parts[3]}')
    return '; '.join(atoms)


def ms(tim, key, cl=False):
    if not tim:
        return 0.0
    k = key + '_cl' if cl else key
    return tim.get(k, 0.0) * 1e3


def run_cpu(mol, grids, dm, vxc_ref):
    ni = dft.numint.NumInt()
    ni.nr_rks(mol, grids, 'pbe', dm, max_memory=2000)
    t0 = time.perf_counter()
    _, _, vxc = ni.nr_rks(mol, grids, 'pbe', dm, max_memory=2000)
    wall = (time.perf_counter() - t0) * 1e3
    err = float(np.abs(vxc - vxc_ref).max()) if vxc_ref is not None else 0.0
    return dict(setup=0.0, wall_outer=wall, wall_sum=wall, gpu_cl=0.0, rho_w=0.0, rho_cl=0.0,
                vmat_w=0.0, vmat_cl=0.0, xc_w=0.0, host_w=wall, vxc_err=err, timing={})


def run_gpu(profile, mol, grids, dm, vxc_ref):
    from pyscf.OpenCL import init_device, reset_opencl
    from pyscf.OpenCL.gpu_profiles import apply_gpu_profile
    from pyscf.OpenCL.xc_grid import clear_xc_plan_cache

    reset_opencl()
    clear_xc_plan_cache()
    init_device(quiet=True)
    mf = dft.RKS(mol, xc='PBE').density_fit()
    t0 = time.perf_counter()
    apply_gpu_profile(mf, profile, setup=True, dm=dm)
    setup_ms = (time.perf_counter() - t0) * 1e3
    plan = mf._xc_gpu_plan
    path = getattr(mf, '_gpu_xc_path', 'onthefly')
    proj = plan.nr_rks_hermite_onthefly if path == 'onthefly' else plan.nr_rks_precomputed_gto
    proj(dm)
    t0 = time.perf_counter()
    _, _, vxc = proj(dm, profile=True)
    wall_outer = (time.perf_counter() - t0) * 1e3
    tim = plan.last_timing
    rho_w = ms(tim, 'gpu_rho')
    rho_cl = ms(tim, 'gpu_rho', cl=True)
    vmat_w = ms(tim, 'gpu_vmat')
    vmat_cl = ms(tim, 'gpu_vmat', cl=True)
    xc_w = ms(tim, 'gpu_xc_pbe') + ms(tim, 'gpu_xc_reduce')
    host_w = ms(tim, 'host_total') + ms(tim, 'host_h2d_dm') + ms(tim, 'gpu_dm_cart')
    wall_sum = ms(tim, 'wall_profiled') + ms(tim, 'host_h2d_dm')
    gpu_cl = ms(tim, 'gpu_total_cl') or (ms(tim, 'gpu_dm_cart', cl=True) + rho_cl + ms(tim, 'gpu_xc_pbe', cl=True) + ms(tim, 'gpu_xc_reduce', cl=True) + vmat_cl)
    rad_setup = getattr(plan, 'otf', {}) or {}
    if isinstance(rad_setup, dict):
        rad_setup = rad_setup.get('setup_radial_gpu', 0.0)
    else:
        rad_setup = 0.0
    if hasattr(plan, 'precalc_timing') and plan.precalc_timing.get('radial_gpu'):
        rad_setup = max(rad_setup, plan.precalc_timing['radial_gpu'] * 1e3)
    err = float(np.abs(vxc - vxc_ref).max()) if vxc_ref is not None else 0.0
    return dict(setup=setup_ms, wall_outer=wall_outer, wall_sum=wall_sum, gpu_cl=gpu_cl,
                rho_w=rho_w, rho_cl=rho_cl, vmat_w=vmat_w, vmat_cl=vmat_cl, xc_w=xc_w, host_w=host_w,
                vxc_err=err, timing=tim, radial_setup_ms=rad_setup * 1e3 if rad_setup < 1 else rad_setup)


def print_table(rows):
    hdr = f"{'method':<20} {'setup':>7} {'outer':>7} {'sum':>7} {'gpuCL':>7} {'ρ_w':>6} {'ρ_cl':>6} {'vm_w':>6} {'vm_cl':>6} {'xc':>5} {'host':>6} {'|vxc|':>9}"
    print(hdr)
    print('-' * len(hdr))
    for label, r in rows:
        print(f"{label:<20} {r['setup']:7.1f} {r['wall_outer']:7.1f} {r['wall_sum']:7.1f} {r['gpu_cl']:7.1f} "
              f"{r['rho_w']:6.1f} {r['rho_cl']:6.1f} {r['vmat_w']:6.1f} {r['vmat_cl']:6.1f} {r['xc_w']:5.1f} {r['host_w']:6.1f} "
              f"{r['vxc_err']:.2e}")


def print_detail(label, tim):
    if not tim:
        return
    from pyscf.OpenCL.xc_grid import TIMING_STAGE_ORDER
    print(f'  detail [{label}]:')
    for k in TIMING_STAGE_ORDER:
        if k not in tim:
            continue
        cl = tim.get(k + '_cl')
        if cl is not None:
            print(f'    {k:22s} wall={tim[k]*1e3:6.2f} ms  CL={cl*1e3:6.2f} ms')
        elif tim[k] > 0:
            print(f'    {k:22s} {tim[k]*1e3:6.2f} ms')


def main():
    lib.num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    mol = gto.M(atom=read_xyz(_XYZ), basis='ccpvdz', verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build(with_non0tab=True)
    nao = mol.nao_nr()
    ngrids = grids.coords.shape[0]
    np.random.seed(42)
    dm = np.random.rand(nao, nao)
    dm = 0.5 * (dm + dm.T)
    print(f'benzene  nao={nao}  ngrids={ngrids}  basis=ccpvdz  OMP=1')
    print('Profiling: wall = perf_counter after queue.finish(); CL = clGetEventProfilingInfo on kernels\n', flush=True)

    ni = dft.numint.NumInt()
    _, _, vxc_ref = ni.nr_rks(mol, grids, 'pbe', dm, max_memory=2000)

    rows = []
    for label, profile in MODES:
        print(f'--- {label} ---', flush=True)
        try:
            if profile is None:
                row = run_cpu(mol, grids, dm, vxc_ref)
            else:
                row = run_gpu(profile, mol, grids, dm, vxc_ref)
            rows.append((label, row))
            print(f"  outer={row['wall_outer']:.1f} ms  sum={row['wall_sum']:.1f} ms  gpu_CL={row['gpu_cl']:.1f} ms  |vxc|={row['vxc_err']:.2e}", flush=True)
            if row.get('timing'):
                print_detail(label, row['timing'])
        except Exception as e:
            import traceback
            print(f'  FAILED: {e}', flush=True)
            traceback.print_exc()
            rows.append((label, {k: float('nan') for k in ('setup', 'wall_outer', 'wall_sum', 'gpu_cl', 'rho_w', 'rho_cl', 'vmat_w', 'vmat_cl', 'xc_w', 'host_w', 'vxc_err')}))

    print('\n=== timing comparison (ms per veff XC call) ===', flush=True)
    print('outer = whole function wall; sum = staged wall_profiled; CL = OpenCL event sum for kernels', flush=True)
    print_table(rows)


if __name__ == '__main__':
    main()
