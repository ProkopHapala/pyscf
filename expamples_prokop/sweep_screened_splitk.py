#!/usr/bin/env python3
"""Sweep NPTILE / WGS_VMAT / vmat_grid_splits for screened split-K XC path.

Screens rho_gga_radial_screened and vmat_gga_radial_screened_pair_splitk
independently — reports per-kernel CL event timing so optimal parameters
for each kernel are visible separately.

Only times single veff XC calls (not full SCF loop), so each point takes
~1–2 s (setup) + ~0.1 s (timed calls).

Usage:
  PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 python3 -u expamples_prokop/sweep_screened_splitk.py
  PYTHONPATH=... python3 -u expamples_prokop/sweep_screened_splitk.py --mol PTCDA --basis 6-31g --grid-level 2
  PYTHONPATH=... python3 -u expamples_prokop/sweep_screened_splitk.py --out debug/sweep_screened/benzene.csv
  PYTHONPATH=... python3 -u expamples_prokop/sweep_screened_splitk.py --nptile 64,128 --wgs 64,128,256 --splits 1,4,8
"""
import argparse
import csv
import os
import re
import time

import numpy as np
from pyscf import dft, gto, lib

from pyscf.OpenCL import init_device, reset_opencl
from pyscf.OpenCL.tile_config import TileConfig
from pyscf.OpenCL.xc_grid import clear_xc_plan_cache, setup_xc_grid_gpu


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


def _ms(tim, key, cl=False):
    k = key + '_cl' if cl else key
    return tim.get(k, 0.0) * 1e3


def bench_point(cfg, vmat_grid_splits, mol, grids, dm, vxc_ref, n_warm=1, n_timed=3):
    reset_opencl()
    clear_xc_plan_cache()
    init_device(tile_config=cfg, force_rebuild=True, quiet=True)
    t0 = time.perf_counter()
    kw = dict(xc_eval='gpu', gpu_xc='auto', rho_mode='radial_screened', vmat_mode='radial_screened')
    if vmat_grid_splits > 1:
        kw['vmat_grid_splits'] = vmat_grid_splits
    plan = setup_xc_grid_gpu(mol, grids, 'pbe', **kw)
    setup_s = time.perf_counter() - t0
    for _ in range(n_warm):
        plan.nr_rks_hermite_onthefly(dm)
    outers = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        plan.nr_rks_hermite_onthefly(dm, profile=True)
        outers.append((time.perf_counter() - t0) * 1e3)
    tim = plan.last_timing
    _, _, vxc = plan.nr_rks_hermite_onthefly(dm)
    err = float(np.abs(vxc - vxc_ref).max())
    return {
        'NPTILE': cfg.NPTILE, 'NATILE': cfg.NATILE, 'WGS_VMAT': cfg.WGS_VMAT,
        'MAX_AO_ATOM': cfg.MAX_AO_ATOM,
        'vmat_grid_splits': vmat_grid_splits,
        'setup_s': setup_s,
        'outer_ms': min(outers),
        'rho_cl_ms': _ms(tim, 'gpu_rho', cl=True),
        'vmat_cl_ms': _ms(tim, 'gpu_vmat', cl=True),
        'vmat_split_cl_ms': _ms(tim, 'gpu_vmat_split', cl=True),
        'vmat_reduce_ms': _ms(tim, 'gpu_vmat_reduce'),
        'gpu_total_cl_ms': _ms(tim, 'gpu_total_cl') or _ms(tim, 'gpu_total', cl=True),
        'vxc_err': err,
        'ok': err < 5e-3,
    }


def _valid_cfg(nptile, wgs, max_ao=16, natile=1, max_itile=32):
    if nptile < 32 or wgs < nptile * natile:
        return None
    for v in (nptile, wgs, natile, max_ao):
        if v <= 0 or (v & (v - 1)) != 0:
            return None
    try:
        return TileConfig(NPTILE=nptile, NATILE=natile, WGS_VMAT=wgs, MAX_AO_ATOM=max_ao, MAX_ITILE=max_itile)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description='Sweep tile/WGS/splits for screened split-K XC')
    ap.add_argument('--mol', default='benzene', help='molecule name or xyz path')
    ap.add_argument('--basis', default='ccpvdz')
    ap.add_argument('--grid-level', type=int, default=3)
    ap.add_argument('--n-timed', type=int, default=3)
    ap.add_argument('--nptile', default='32,64,128', help='comma-separated NPTILE values')
    ap.add_argument('--wgs', default='32,64,128,256', help='comma-separated WGS_VMAT values')
    ap.add_argument('--splits', default='1,2,4,8,16,32', help='comma-separated vmat_grid_splits')
    ap.add_argument('--max-ao', type=int, default=16, help='MAX_AO_ATOM (must be power of 2 >= max atom nao)')
    ap.add_argument('--out', default=None, help='CSV output path')
    args = ap.parse_args()

    lib.num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    xyz = args.mol
    if not xyz.endswith('.xyz'):
        xyz = os.path.join(repo, 'data', 'xyz', f'{xyz}.xyz')
    mol = gto.M(atom=read_xyz(xyz), basis=args.basis, verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.level = args.grid_level
    grids.build(with_non0tab=True)
    nao = mol.nao_nr()
    natm = mol.natm
    ngrids = grids.coords.shape[0]
    max_itile = 32
    while max_itile < natm:
        max_itile *= 2
    np.random.seed(42)
    dm = np.random.rand(nao, nao)
    dm = 0.5 * (dm + dm.T)
    _, _, vxc_ref = dft.numint.NumInt().nr_rks(mol, grids, 'pbe', dm, max_memory=2000)

    nptiles = [int(x) for x in args.nptile.split(',')]
    wgs_list = [int(x) for x in args.wgs.split(',')]
    splits_list = [int(x) for x in args.splits.split(',')]

    print(f'screened split-K sweep  mol={os.path.basename(xyz)}  basis={args.basis}  '
          f'grid={args.grid_level}  nao={nao}  natm={natm}  ngrids={ngrids}  '
          f'MAX_AO_ATOM={args.max_ao}  MAX_ITILE={max_itile}', flush=True)
    print(f'NPTILE={nptiles}  WGS={wgs_list}  splits={splits_list}', flush=True)

    rows = []
    for nptile in nptiles:
        for wgs in wgs_list:
            cfg = _valid_cfg(nptile, wgs, max_ao=args.max_ao, max_itile=max_itile)
            if cfg is None:
                continue
            for nsplit in splits_list:
                label = f'NPTILE={nptile} WGS={wgs} splits={nsplit}'
                print(f'\n=== {label} ===', flush=True)
                try:
                    r = bench_point(cfg, nsplit, mol, grids, dm, vxc_ref, n_timed=args.n_timed)
                except Exception as e:
                    print(f'  FAILED: {e}', flush=True)
                    rows.append({'NPTILE': nptile, 'WGS_VMAT': wgs, 'vmat_grid_splits': nsplit, 'ok': False, 'error': str(e)})
                    continue
                status = 'OK' if r['ok'] else 'PARITY'
                print(f"  {status}  gpuCL={r['gpu_total_cl_ms']:.1f}  rho_cl={r['rho_cl_ms']:.1f}  "
                      f"vmat_cl={r['vmat_cl_ms']:.1f} (split={r['vmat_split_cl_ms']:.1f} red={r['vmat_reduce_ms']:.3f})  "
                      f"|vxc|={r['vxc_err']:.2e}  setup={r['setup_s']:.1f}s", flush=True)
                rows.append(r)

    ok_rows = [r for r in rows if r.get('ok')]
    if ok_rows:
        best_rho = min(ok_rows, key=lambda r: r['rho_cl_ms'])
        best_vmat = min(ok_rows, key=lambda r: r['vmat_cl_ms'])
        best_total = min(ok_rows, key=lambda r: r['gpu_total_cl_ms'])
        print('\n=== best (parity OK) ===', flush=True)
        for tag, r in [('rho_cl', best_rho), ('vmat_cl', best_vmat), ('gpu_total_cl', best_total)]:
            print(f"  {tag:15s}: NPTILE={r['NPTILE']} WGS={r['WGS_VMAT']} splits={r['vmat_grid_splits']}  "
                  f"rho_cl={r['rho_cl_ms']:.1f}  vmat_cl={r['vmat_cl_ms']:.1f}  gpuCL={r['gpu_total_cl_ms']:.1f}", flush=True)
    else:
        print('\nNo configuration passed parity.', flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        fields = sorted({k for r in rows for k in r})
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        print(f'\nWrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
