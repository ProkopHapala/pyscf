#!/usr/bin/env python3
"""CPU vs GPU on-the-fly Hermite XC: parity + harness/kernel timing breakdown.

Usage:
  PYTHONPATH=/home/prokophapala/git/pyscf OMP_NUM_THREADS=1 \\
    python3 expamples_prokop/test_opencl_xc_onthefly.py --xyz data/xyz/benzene.xyz

  OPENCL_NPTILE=32 PYTHONPATH=... python3 expamples_prokop/test_opencl_xc_onthefly.py \\
    --xyz data/xyz/PTCDA.xyz --ncycles 3
"""
import argparse
import os
import re
import time
import numpy as np
from pyscf import gto, dft
from pyscf.OpenCL.xc_grid import setup_xc_grid_gpu
from pyscf.OpenCL.tile_config import get_active_tile_config

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_DEFAULT_XYZ = os.path.join(_REPO_ROOT, 'data', 'xyz', 'benzene.xyz')


def read_xyz(path):
    with open(path) as f:
        lines = f.readlines()
    natom = int(lines[0].strip())
    atoms = []
    for line in lines[2:2 + natom]:
        parts = line.split()
        el = parts[0]
        if not re.match(r'^[A-Z][a-z]?$', el):
            continue
        atoms.append(f'{el} {parts[1]} {parts[2]} {parts[3]}')
    if not atoms:
        raise ValueError(f'no atoms parsed from {path}')
    return '; '.join(atoms)


def print_timing(tim):
    from pyscf.OpenCL.xc_grid import TIMING_STAGE_ORDER
    for k in TIMING_STAGE_ORDER:
        if k not in tim:
            continue
        if k == 'n_blocks':
            print(f'    {k:22s} {int(tim[k]):8d}')
        elif tim[k] > 0:
            print(f'    {k:22s} {tim[k]*1e3:7.1f} ms')


def parse_args():
    ap = argparse.ArgumentParser(description='CPU vs GPU on-the-fly XC benchmark')
    ap.add_argument('--xyz', default=_DEFAULT_XYZ, help=f'XYZ file (default: {_DEFAULT_XYZ})')
    ap.add_argument('--basis', default='ccpvdz')
    ap.add_argument('--xc', default='pbe')
    ap.add_argument('--grid-level', type=int, default=3, dest='grid_level')
    ap.add_argument('--ncycles', type=int, default=5, help='timed GPU calls')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--vxc-tol', type=float, default=5e-3, help='max abs vxc error (fail if exceeded)')
    return ap.parse_args()


def main():
    args = parse_args()
    xyz_path = os.path.abspath(args.xyz)
    mol_name = os.path.splitext(os.path.basename(xyz_path))[0]

    mol = gto.M(atom=read_xyz(xyz_path), basis=args.basis, verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.level = args.grid_level
    grids.build()
    nao = mol.nao_nr()
    ngrids = grids.coords.shape[0]
    print(f'{mol_name}: natoms={mol.natm} nao={nao} ngrids={ngrids}  xc={args.xc}  basis={args.basis}')

    np.random.seed(args.seed)
    dm = np.random.rand(nao, nao)
    dm = 0.5 * (dm + dm.T)
    ni = dft.numint.NumInt()

    print('\n=== setup (pre-SCF, one-time) ===')
    t0 = time.perf_counter()
    plan = setup_xc_grid_gpu(mol, grids, args.xc)
    t_setup = time.perf_counter() - t0
    print(f'  setup_xc_grid_gpu: {t_setup:.3f}s  ncart={plan.otf["ncart"]}')
    tc = get_active_tile_config()
    pair = plan.otf.get('use_pair_kernels', False)
    print(f'  tile config: NPTILE={tc.NPTILE} NATILE={tc.NATILE} WGS_VMAT={tc.WGS_VMAT} MAX_ITILE={tc.MAX_ITILE}  pair_kernels={pair}')

    print('\n=== CPU reference (1 warm + 1 timed) ===')
    ni.nr_rks(mol, grids, args.xc, dm, max_memory=2000)
    t0 = time.perf_counter()
    n_cpu, exc_cpu, vxc_cpu = ni.nr_rks(mol, grids, args.xc, dm, max_memory=2000)
    t_cpu = time.perf_counter() - t0
    print(f'  CPU nr_rks: {t_cpu:.3f}s')

    plan.nr_rks_hermite_onthefly(dm)

    print(f'\n=== GPU on-the-fly ({args.ncycles} timed SCF-equivalent calls) ===')
    gpu_times = []
    for i in range(args.ncycles):
        t0 = time.perf_counter()
        n_gpu, exc_gpu, vxc_gpu = plan.nr_rks_hermite_onthefly(dm, profile=True)
        dt = time.perf_counter() - t0
        gpu_times.append(dt)
        tim = plan.last_timing
        print(f'  call {i+1}: total={dt*1e3:7.1f} ms  gpu={tim.get("gpu_total", 0)*1e3:7.1f} ms  host={tim.get("host_total", 0)*1e3:7.1f} ms')
    print(f'  GPU min/mean: {min(gpu_times):.3f}s / {sum(gpu_times)/len(gpu_times):.3f}s')
    print('  last call breakdown:')
    print_timing(plan.last_timing)

    print('\n=== parity (CPU vs GPU) ===')
    err_n = abs(n_cpu - n_gpu) / max(abs(n_cpu), 1e-10)
    err_exc = abs(exc_cpu - exc_gpu) / max(abs(exc_cpu), 1e-10)
    err_vxc = np.abs(vxc_cpu - vxc_gpu).max()
    err_vxc_rel = err_vxc / max(np.abs(vxc_cpu).max(), 1e-10)
    print(f'  nelec rel_err={err_n:.3e}  exc rel_err={err_exc:.3e}')
    print(f'  vxc max_abs_err={err_vxc:.3e}  max_rel_err={err_vxc_rel:.3e}')
    print(f'  speedup (CPU / GPU min): {t_cpu/min(gpu_times):.2f}x')

    if err_vxc > args.vxc_tol:
        raise SystemExit(f'On-the-fly XC parity failed: vxc max_abs_err={err_vxc}')
    print('\nAll checks passed.')


if __name__ == '__main__':
    main()
