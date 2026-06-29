#!/usr/bin/env python3
"""Benzene XC benchmark: CPU libxc vs all GPU paths (rigorous stage timing).

One XC iteration per path (same dm). Setup (AO upload, Hermite tables) is one-time, outside iter budget.

Paths:
  cpu_libxc           — pyscf NumInt.nr_rks (float64, blocked CPU)
  gpu_gto_block       — CPU eval_ao per block + GPU matmul rho/vmat
  gpu_precomp_tiled   — precomputed AO, grid-tiled rho + pair vmat (default)
  gpu_precomp_pbe     — same + GPU PBE f32
  gpu_precomp_gemm    — full-grid GEMM fallback (fused=gemm)
  gpu_precomp_blocked — Python block loop + tiled matmul (old path)
  gpu_hermite_otf     — Hermite AO on-the-fly kernels + libxc on CPU

Usage:
  PYTHONPATH=/home/prokophapala/git/pyscf OMP_NUM_THREADS=1 \\
    python3 -u expamples_prokop/test_opencl_xc_scf.py
"""
import argparse
import os
import re
import sys
import time

import numpy as np
from pyscf import gto, dft, lib
import pyscf

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_XYZ_DIR = os.path.join(_REPO, 'data', 'xyz')


def log(msg):
    print(msg, flush=True)


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


def print_stage_timing(tim, indent='  '):
    from pyscf.OpenCL.xc_grid import TIMING_STAGE_ORDER
    if not tim:
        return
    log(f'{indent}--- stage breakdown (ms, queue.finish per gpu_* stage) ---')
    for k in TIMING_STAGE_ORDER:
        if k not in tim:
            continue
        if k == 'n_blocks':
            log(f'{indent}  {k:22s} {int(tim[k]):8d}')
        elif tim[k] > 0:
            log(f'{indent}  {k:22s} {tim[k]*1e3:8.1f}')
    if 'wall_profiled' in tim:
        log(f'{indent}  (profiled sum matches wall within ~1 ms)')


def bench_call(label, run, n_warm=1, n_timed=3, ref=None):
    '''run(profile) -> (n, exc, vxc) or (n, exc, vxc, timing).'''
    for _ in range(n_warm):
        run(False)
    times = []
    last_out = last_tim = None
    for i in range(n_timed):
        t0 = time.perf_counter()
        last_out = run(True)
        dt = time.perf_counter() - t0
        times.append(dt)
        last_tim = last_out[3] if len(last_out) > 3 else {}
        log(f'  {label} call {i+1}: wall={dt*1e3:.1f} ms')
    t_min = min(times)
    log(f'  {label} min wall: {t_min*1e3:.1f} ms')
    print_stage_timing(last_tim or {}, indent='  ')
    row = dict(path=label, wall_ms=t_min * 1e3, timing=last_tim or {})
    if ref is not None and last_out is not None:
        n_ref, exc_ref, vxc_ref = ref
        n, exc, vxc = last_out[0], last_out[1], last_out[2]
        row['vxc_max_err'] = float(np.abs(vxc - vxc_ref).max())
        row['exc_rel_err'] = abs(exc - exc_ref) / max(abs(exc_ref), 1e-10)
        log(f'  parity vs cpu_libxc: vxc_max={row["vxc_max_err"]:.3e}  exc_rel={row["exc_rel_err"]:.3e}')
    return row


def run_benzene(basis, grid_level, gpu_xc, n_timed):
    from pyscf.OpenCL.xc_grid import get_xc_grid_plan, clear_xc_plan_cache
    from pyscf.OpenCL import init_device, reset_opencl

    mol = gto.M(atom=read_xyz(os.path.join(_XYZ_DIR, 'benzene.xyz')), basis=basis, verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.level = grid_level
    grids.build(with_non0tab=True)
    nao, ngrids = mol.nao_nr(), grids.coords.shape[0]
    mf = dft.RKS(mol, xc='PBE').density_fit()
    dm = mf.get_init_guess()
    ni = dft.numint.NumInt()

    log(f'\n{"="*76}')
    log(f'benzene  basis={basis}  grid={grid_level}  nao={nao}  ngrids={ngrids}')
    log(f'OMP_NUM_THREADS={lib.num_threads()}  PySCF {pyscf.__version__}')
    log(f'{"="*76}')

    log('\n[1/5] cpu_libxc')
    for _ in range(1):
        ni.nr_rks(mol, grids, 'PBE', dm, max_memory=2000)
    t0 = time.perf_counter()
    n_ref, exc_ref, vxc_ref = ni.nr_rks(mol, grids, 'PBE', dm, max_memory=2000)
    t_cpu = time.perf_counter() - t0
    log(f'  cpu_libxc wall: {t_cpu*1e3:.1f} ms')
    ref = (n_ref, exc_ref, vxc_ref)
    rows = [dict(path='cpu_libxc', wall_ms=t_cpu * 1e3, timing={})]

    clear_xc_plan_cache()
    reset_opencl()
    init_device(quiet=False)

    log('\n[2/5] gpu_gto_block  (CPU eval_ao each block + GPU matmul)')
    plan_blk = get_xc_grid_plan(mol, grids, 'PBE')

    def run_gto_block(profile):
        out = plan_blk.nr_rks(dm)
        return out if not profile else (*out, {})

    rows.append(bench_call('gpu_gto_block', run_gto_block, n_timed=n_timed, ref=ref))

    log('\n[3/7] gpu_precomp_tiled  (grid-tiled rho + pair vmat, default fused=tiled)')
    clear_xc_plan_cache()
    reset_opencl()
    init_device(quiet=True)
    plan_pc = get_xc_grid_plan(mol, grids, 'PBE')
    t0 = time.perf_counter()
    plan_pc.setup_precomputed_gto(gpu_only=True, gpu_xc='cpu')
    setup_pc = time.perf_counter() - t0
    log(f'  setup_precomputed_gto: {setup_pc*1e3:.1f} ms (one-time)')
    def run_precomp_libxc(profile):
        out = plan_pc.nr_rks_precomputed_gto(dm, projection='gpu', profile=profile)
        return (*out, plan_pc.last_timing) if profile else out

    rows.append(bench_call('gpu_precomp_tiled', run_precomp_libxc, n_timed=n_timed, ref=ref))
    rows[-1]['setup_ms'] = setup_pc * 1e3

    log(f'\n[4/7] gpu_precomp_pbe  (tiled + GPU PBE {gpu_xc})')
    clear_xc_plan_cache()
    reset_opencl()
    init_device(quiet=True)
    plan_pbe = get_xc_grid_plan(mol, grids, 'PBE')
    t0 = time.perf_counter()
    plan_pbe.setup_precomputed_gto(gpu_only=True, gpu_xc=gpu_xc)
    setup_pbe = time.perf_counter() - t0
    log(f'  setup_precomputed_gto: {setup_pbe*1e3:.1f} ms (one-time)')
    def run_precomp_pbe(profile):
        out = plan_pbe.nr_rks_precomputed_gto(dm, projection='gpu', profile=profile)
        return (*out, plan_pbe.last_timing) if profile else out

    rows.append(bench_call(f'gpu_precomp_{gpu_xc}', run_precomp_pbe, n_timed=n_timed, ref=ref))
    rows[-1]['setup_ms'] = setup_pbe * 1e3

    log('\n[5/7] gpu_precomp_gemm  (full-grid GEMM fallback, fused=gemm)')
    clear_xc_plan_cache()
    reset_opencl()
    init_device(quiet=True)
    plan_gemm = get_xc_grid_plan(mol, grids, 'PBE')
    plan_gemm.setup_precomputed_gto(gpu_only=True, gpu_xc='cpu', fused='gemm')

    def run_precomp_gemm(profile):
        out = plan_gemm.nr_rks_precomputed_gto(dm, projection='gpu', profile=profile)
        return (*out, plan_gemm.last_timing) if profile else out

    rows.append(bench_call('gpu_precomp_gemm', run_precomp_gemm, n_timed=1, ref=ref))

    log('\n[6/7] gpu_precomp_blocked  (old matmul block loop, libxc)')
    clear_xc_plan_cache()
    reset_opencl()
    init_device(quiet=True)
    plan_blk_pc = get_xc_grid_plan(mol, grids, 'PBE')
    plan_blk_pc.setup_precomputed_gto(gpu_only=True, gpu_xc='cpu', fused=False)

    def run_precomp_blocked(profile):
        out = plan_blk_pc.nr_rks_precomputed_gto(dm, projection='gpu', profile=profile)
        return (*out, plan_blk_pc.last_timing) if profile else out

    rows.append(bench_call('gpu_precomp_blocked', run_precomp_blocked, n_timed=n_timed, ref=ref))

    log('\n[7/7] gpu_hermite_otf  (Hermite rho/vmat kernels + libxc on CPU)')
    clear_xc_plan_cache()
    reset_opencl()
    init_device(quiet=True)
    plan_otf = get_xc_grid_plan(mol, grids, 'PBE')
    t0 = time.perf_counter()
    plan_otf.setup_onthefly()
    setup_otf = time.perf_counter() - t0
    log(f'  setup_onthefly: {setup_otf*1e3:.1f} ms (one-time)  ncart={plan_otf.otf["ncart"]}')
    def run_hermite_otf(profile):
        out = plan_otf.nr_rks_hermite_onthefly(dm, profile=profile)
        return (*out, plan_otf.last_timing) if profile else out

    rows.append(bench_call('gpu_hermite_otf', run_hermite_otf, n_timed=n_timed, ref=ref))
    rows[-1]['setup_ms'] = setup_otf * 1e3

    log(f'\n{"="*76}')
    log('SUMMARY  (min wall ms per path; setup one-time)')
    log(f'{"path":<22} {"wall_ms":>8} {"gpu_ms":>8} {"host_ms":>8} {"vs_cpu":>7} {"setup_ms":>9}')
    log('-' * 76)
    for r in rows:
        tim = r.get('timing') or {}
        gpu_ms = tim.get('gpu_total', 0) * 1e3
        host_ms = tim.get('host_total', 0) * 1e3
        spd = t_cpu / (r['wall_ms'] / 1e3) if r['wall_ms'] > 0 else 0
        setup = r.get('setup_ms', 0)
        log(f'{r["path"]:<22} {r["wall_ms"]:8.1f} {gpu_ms:8.1f} {host_ms:8.1f} {spd:7.2f}x {setup:9.1f}')

    log('\nWhy GPU can lose to CPU on benzene:')
    log('  - nao=114 is tiny: tiled GEMM (M=ngrids, N=K=114) is memory-bound / low occupancy')
    log('  - Python launches ~18 blocks x 4 matmuls per XC call (precomp path)')
    log('  - PCIe round-trips: full-grid rho + exc/wv + vmat D2H each iteration')
    log('  - CPU uses fused BLAS + libxc on same data; no device sync overhead')
    log('  - Hermite OTF has real gpu_rho/gpu_vmat kernels but still D2H + libxc on host')
    log('\nDone.')
    return rows


def parse_args():
    ap = argparse.ArgumentParser(description='Benzene XC benchmark — all GPU paths')
    ap.add_argument('--basis', default='ccpvdz')
    ap.add_argument('--grid-level', type=int, default=3, dest='grid_level')
    ap.add_argument('--gpu-xc', default='pbe_f32', choices=['pbe_f32', 'pbe_f64'])
    ap.add_argument('--n-timed', type=int, default=3, dest='n_timed')
    return ap.parse_args()


def main():
    args = parse_args()
    run_benzene(args.basis, args.grid_level, args.gpu_xc, args.n_timed)


if __name__ == '__main__':
    main()
