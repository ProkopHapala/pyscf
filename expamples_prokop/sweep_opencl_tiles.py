#!/usr/bin/env python3
"""Sweep compile-time OpenCL tile sizes on benzene PBE (parity + kernel timing).

Each configuration recompiles kernels.cl with -D flags (see pyscf/OpenCL/tile_config.py).

Usage:
  PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 python3 expamples_prokop/sweep_opencl_tiles.py
  PYTHONPATH=... python3 expamples_prokop/sweep_opencl_tiles.py --quick   # fewer combos
"""
import argparse
import re
import time
import numpy as np
from pyscf import gto, dft
from pyscf.OpenCL import reset_opencl, init_device
from pyscf.OpenCL.tile_config import TileConfig
from pyscf.OpenCL.xc_grid import setup_xc_grid_gpu, clear_xc_plan_cache


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
    return '; '.join(atoms)


def bench_config(cfg, mol, grids, dm, ni, n_warm=1, n_timed=3):
    reset_opencl()
    clear_xc_plan_cache()
    init_device(tile_config=cfg, force_rebuild=True, quiet=True)
    t0 = time.perf_counter()
    plan = setup_xc_grid_gpu(mol, grids, 'pbe')
    t_setup = time.perf_counter() - t0
    for _ in range(n_warm):
        plan.nr_rks_hermite_onthefly(dm)
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        n_gpu, exc_gpu, vxc_gpu = plan.nr_rks_hermite_onthefly(dm, profile=True)
        times.append(time.perf_counter() - t0)
    n_cpu, exc_cpu, vxc_cpu = ni.nr_rks(mol, grids, 'pbe', dm, max_memory=2000)
    err_vxc = np.abs(vxc_cpu - vxc_gpu).max()
    err_vxc_rel = err_vxc / max(np.abs(vxc_cpu).max(), 1e-10)
    tim = plan.last_timing
    return {
        'setup_s': t_setup,
        'total_ms': min(times) * 1e3,
        'kernel_rho_ms': tim.get('kernel_rho', 0) * 1e3,
        'kernel_vmat_ms': tim.get('kernel_vmat', 0) * 1e3,
        'kernel_total_ms': tim.get('kernel_total', 0) * 1e3,
        'harness_ms': tim.get('harness_total', 0) * 1e3,
        'vxc_rel_err': err_vxc_rel,
        'ok': err_vxc_rel < 5e-3,
    }


def main():
    ap = argparse.ArgumentParser(description='Sweep OpenCL XC tile compile flags')
    ap.add_argument('--quick', action='store_true', help='smaller search grid')
    args = ap.parse_args()

    _REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    mol = gto.M(atom=read_xyz(os.path.join(_REPO, 'data', 'xyz', 'benzene.xyz')), basis='ccpvdz', verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build()
    nao = mol.nao_nr()
    np.random.seed(42)
    dm = np.random.rand(nao, nao)
    dm = 0.5 * (dm + dm.T)
    ni = dft.numint.NumInt()
    ni.nr_rks(mol, grids, 'pbe', dm, max_memory=2000)

    if args.quick:
        nptiles = [16, 32]
        natiles = [4]
        wgs_vmat = [256]
    else:
        nptiles = [8, 16, 32]
        natiles = [2, 4, 8]
        wgs_vmat = [128, 256, 512]

    rows = []
    for nptile in nptiles:
        for natile in natiles:
            for wgs in wgs_vmat:
                if wgs < nptile * natile:
                    continue
                cfg = TileConfig(NPTILE=nptile, NATILE=natile, WGS_VMAT=wgs)
                label = f'NPTILE={nptile} NATILE={natile} WGS_VMAT={wgs}'
                print(f'\n=== {label} ===')
                try:
                    r = bench_config(cfg, mol, grids, dm, ni)
                except Exception as e:
                    print(f'  FAILED: {e}')
                    rows.append((label, None, str(e)))
                    continue
                status = 'OK' if r['ok'] else 'PARITY'
                print(f'  {status}  total={r["total_ms"]:.1f} ms  kernel={r["kernel_total_ms"]:.1f} ms  '
                      f'rho={r["kernel_rho_ms"]:.1f}  vmat={r["kernel_vmat_ms"]:.1f}  '
                      f'harness={r["harness_ms"]:.1f}  vxc_rel={r["vxc_rel_err"]:.2e}  setup={r["setup_s"]:.2f}s')
                rows.append((label, r, None))

    ok_rows = [(lbl, r) for lbl, r, err in rows if r is not None and r['ok']]
    if ok_rows:
        best = min(ok_rows, key=lambda x: x[1]['kernel_total_ms'])
        print(f'\n=== best kernel_total (parity OK): {best[0]} ===')
        print(f'  kernel_total={best[1]["kernel_total_ms"]:.1f} ms  total={best[1]["total_ms"]:.1f} ms')
    else:
        print('\nNo configuration passed parity.')


if __name__ == '__main__':
    main()
