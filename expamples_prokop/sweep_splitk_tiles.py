#!/usr/bin/env python3
"""Sweep NPTILE / NATILE / WGS_VMAT / vmat_grid_splits for split-K hybrid XC path.

Uses production_otf_radial_vmat_splitk geometry (OTF rho + radial split-K vmat).
Each point recompiles kernels.cl via TileConfig; records plan.last_timing CL events.

Sweep modes:
  --quick / default full grid: brute-force Cartesian product (legacy exploration).
  --neighbor: 1-step lattice walk from --seed (recommended). See neighbor_configs().

Usage:
  PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 python3 -u expamples_prokop/sweep_splitk_tiles.py
  PYTHONPATH=... python3 -u expamples_prokop/sweep_splitk_tiles.py --quick
  PYTHONPATH=... python3 -u expamples_prokop/sweep_splitk_tiles.py --neighbor
  PYTHONPATH=... python3 -u expamples_prokop/sweep_splitk_tiles.py --neighbor --seed 64,2,128,32
  PYTHONPATH=... python3 -u expamples_prokop/sweep_splitk_tiles.py --out debug/sweep_splitk_tiles/benzene.csv
"""
import argparse
import csv
import os
import re
import time

import numpy as np
from pyscf import dft, gto

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
    plan = setup_xc_grid_gpu(
        mol, grids, 'pbe', xc_eval='gpu', vmat_mode='radial_precomp', vmat_grid_splits=vmat_grid_splits,
    )
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
        'vmat_grid_splits': vmat_grid_splits,
        'setup_s': setup_s,
        'outer_ms': min(outers),
        'rho_ms': _ms(tim, 'gpu_rho'),
        'rho_cl_ms': _ms(tim, 'gpu_rho', cl=True),
        'vmat_ms': _ms(tim, 'gpu_vmat'),
        'vmat_cl_ms': _ms(tim, 'gpu_vmat', cl=True),
        'vmat_split_ms': _ms(tim, 'gpu_vmat_split'),
        'vmat_split_cl_ms': _ms(tim, 'gpu_vmat_split', cl=True),
        'vmat_reduce_ms': _ms(tim, 'gpu_vmat_reduce'),
        'gpu_total_cl_ms': _ms(tim, 'gpu_total_cl') or _ms(tim, 'gpu_total', cl=True),
        'host_ms': tim.get('host_total', 0.0) * 1e3,
        'wall_profiled_ms': tim.get('wall_profiled', 0.0) * 1e3,
        'vxc_err': err,
        'ok': err < 1e-4,
    }


def _pow2_step(val, direction):
    '''One power-of-2 lattice step: direction +1 => ×2, -1 => ÷2.'''
    return val * 2 if direction > 0 else val // 2


def _valid_tile_config(nptile, natile, wgs):
    if nptile < 32 or natile < 1 or wgs < 32:
        return None
    for v in (nptile, natile, wgs):
        if v <= 0 or (v & (v - 1)) != 0:
            return None
    if wgs < nptile * natile:
        return None
    try:
        return TileConfig(NPTILE=nptile, NATILE=natile, WGS_VMAT=wgs)
    except ValueError:
        return None


def neighbor_configs(center_nptile, center_natile, center_wgs, center_splits,
                     nptile_bounds=(32, 128), natile_bounds=(1, 4), wgs_bounds=(32, 256),
                     splits_bounds=(8, 128)):
    '''1-neighborhood on a power-of-2 lattice + coupled diagonal moves.

    *Axis neighbors*: change exactly one parameter by exactly one ×2/÷2 step.
    *Diagonal neighbors*: change two parameters by one step each — typically
    opposite directions on the occupancy/local-memory trade-off (e.g. NPTILE×2
    with WGS÷2, or NATILE×2 with WGS÷2). Never more than one step per axis.

    Yields unique (TileConfig, splits) including the center point first.
    '''
    seen = set()

    def add(nptile, natile, wgs, splits):
        if not (nptile_bounds[0] <= nptile <= nptile_bounds[1]
                and natile_bounds[0] <= natile <= natile_bounds[1]
                and wgs_bounds[0] <= wgs <= wgs_bounds[1]
                and splits_bounds[0] <= splits <= splits_bounds[1]):
            return
        cfg = _valid_tile_config(nptile, natile, wgs)
        if cfg is None:
            return
        key = (cfg.NPTILE, cfg.NATILE, cfg.WGS_VMAT, splits)
        if key in seen:
            return
        seen.add(key)
        return cfg, splits

    c = center_nptile, center_natile, center_wgs, center_splits
    out = []
    center = add(*c)
    if center:
        out.append(center)

    axes = (
        ('nptile', c[0], nptile_bounds),
        ('natile', c[1], natile_bounds),
        ('wgs', c[2], wgs_bounds),
        ('splits', c[3], splits_bounds),
    )
    for _name, val, bounds in axes:
        for d in (-1, 1):
            nv = _pow2_step(val, d)
            if not bounds[0] <= nv <= bounds[1]:
                continue
            kw = dict(nptile=c[0], natile=c[1], wgs=c[2], splits=c[3])
            kw[_name] = nv
            pt = add(**kw)
            if pt:
                out.append(pt)

    # Coupled diagonals: two axes, one step each (±1), at most one step per axis.
    coupled = (
        ('nptile', 'wgs', (-1, 1)),   # NPTILE÷2 WGS×2
        ('nptile', 'wgs', (1, -1)),   # NPTILE×2 WGS÷2
        ('natile', 'wgs', (-1, 1)),
        ('natile', 'wgs', (1, -1)),
        ('nptile', 'natile', (-1, 1)),  # keep NPTILE*NATILE ~ constant
        ('nptile', 'natile', (1, -1)),
        ('splits', 'wgs', (-1, 1)),     # more splits, smaller WG
        ('splits', 'wgs', (1, -1)),
    )
    base = dict(nptile=c[0], natile=c[1], wgs=c[2], splits=c[3])
    for ax1, ax2, (d1, d2) in coupled:
        kw = dict(base)
        kw[ax1] = _pow2_step(kw[ax1], d1)
        kw[ax2] = _pow2_step(kw[ax2], d2)
        pt = add(**kw)
        if pt:
            out.append(pt)
    return out


def iter_grid(quick):
    if quick:
        nptiles = [32, 64]
        natiles = [1, 2]
        wgs_list = [64, 128, 256]
        splits = [16, 32, 64]
    else:
        nptiles = [32, 64, 128]
        natiles = [1, 2]
        wgs_list = [64, 128, 256]
        splits = [8, 16, 32, 64, 128]
    for nptile in nptiles:
        for natile in natiles:
            min_wgs = nptile * natile
            for wgs in wgs_list:
                if wgs < min_wgs:
                    continue
                cfg = TileConfig(NPTILE=nptile, NATILE=natile, WGS_VMAT=wgs)
                for nsplit in splits:
                    yield cfg, nsplit


def main():
    ap = argparse.ArgumentParser(description='Sweep tile/WGS/split-K for hybrid XC')
    ap.add_argument('--quick', action='store_true', help='smaller brute-force grid')
    ap.add_argument('--neighbor', action='store_true',
                    help='1-step lattice neighborhood from --seed (recommended)')
    ap.add_argument('--seed', default='64,2,128,32',
                    help='center for --neighbor: NPTILE,NATILE,WGS_VMAT,splits')
    ap.add_argument('--n-timed', type=int, default=3)
    ap.add_argument('--xyz', default=None, help='molecule xyz (default benzene)')
    ap.add_argument('--out', default=None, help='CSV output path')
    args = ap.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    xyz = args.xyz or os.path.join(repo, 'data', 'xyz', 'benzene.xyz')
    mol = gto.M(atom=read_xyz(xyz), basis='ccpvdz', verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build(with_non0tab=True)
    nao = mol.nao_nr()
    np.random.seed(42)
    dm = np.random.rand(nao, nao)
    dm = 0.5 * (dm + dm.T)
    _, _, vxc_ref = dft.numint.NumInt().nr_rks(mol, grids, 'pbe', dm, max_memory=2000)
    print(f'sweep split-K tiles  nao={nao}  ngrids={grids.coords.shape[0]}  xyz={xyz}', flush=True)

    if args.neighbor:
        snpt, snat, swgs, sspl = (int(x) for x in args.seed.split(','))
        grid_iter = neighbor_configs(snpt, snat, swgs, sspl)
        print(f'neighbor mode  seed=NPTILE={snpt} NATILE={snat} WGS={swgs} splits={sspl}  '
              f'points={len(grid_iter)}', flush=True)
    else:
        grid_iter = list(iter_grid(args.quick))

    rows = []
    for cfg, nsplit in grid_iter:
        label = f'NPTILE={cfg.NPTILE} NATILE={cfg.NATILE} WGS={cfg.WGS_VMAT} splits={nsplit}'
        print(f'\n=== {label} ===', flush=True)
        try:
            r = bench_point(cfg, nsplit, mol, grids, dm, vxc_ref, n_timed=args.n_timed)
        except Exception as e:
            print(f'  FAILED: {e}', flush=True)
            rows.append({'NPTILE': cfg.NPTILE, 'NATILE': cfg.NATILE, 'WGS_VMAT': cfg.WGS_VMAT,
                         'vmat_grid_splits': nsplit, 'ok': False, 'error': str(e)})
            continue
        status = 'OK' if r['ok'] else 'PARITY'
        print(f"  {status}  outer={r['outer_ms']:.1f} ms  gpuCL={r['gpu_total_cl_ms']:.1f} ms  "
              f"rho={r['rho_cl_ms']:.1f}  vmat={r['vmat_cl_ms']:.1f} "
              f"(split={r['vmat_split_cl_ms']:.1f} red={r['vmat_reduce_ms']:.3f})  "
              f"|vxc|={r['vxc_err']:.2e}  setup={r['setup_s']:.2f}s", flush=True)
        rows.append(r)

    ok_rows = [r for r in rows if r.get('ok')]
    if ok_rows:
        best_gpu = min(ok_rows, key=lambda r: r['gpu_total_cl_ms'])
        best_vmat = min(ok_rows, key=lambda r: r['vmat_cl_ms'])
        best_outer = min(ok_rows, key=lambda r: r['outer_ms'])
        print('\n=== best (parity OK) ===', flush=True)
        for tag, r in [('gpu_total_cl', best_gpu), ('vmat_cl', best_vmat), ('outer', best_outer)]:
            print(f"  {tag}: NPTILE={r['NPTILE']} NATILE={r['NATILE']} WGS={r['WGS_VMAT']} "
                  f"splits={r['vmat_grid_splits']}  gpuCL={r['gpu_total_cl_ms']:.1f} ms  "
                  f"vmat_cl={r['vmat_cl_ms']:.1f} ms  outer={r['outer_ms']:.1f} ms", flush=True)
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
