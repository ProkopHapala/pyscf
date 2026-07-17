#!/usr/bin/env python3
'''Clear setup / cycle / full-job Amdahl budget (this machine).

Separates:
  A) static setup once per geometry (grids, DF build, AO cache, optional GPU plan)
  B) steady-state SCF cycle (2nd get_veff onward)
  C) full job wall ≈ A + N×B (+ init extras)

Usage:
  OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/prokophapala/git/pyscf \\
    python3 -u expamples_prokop/profile_amdahl_budget.py --mol benzene --n-cycles 8
'''
import argparse
import os
import re
import time

import numpy as np
from pyscf import dft, gto, lib

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_XYZ = os.path.join(_REPO, 'data', 'xyz')


def log(msg):
    print(msg, flush=True)


def read_xyz(name):
    path = os.path.join(_XYZ, f'{name}.xyz')
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0])
    return '; '.join(' '.join(x.split()[:4]) for x in lines[2:2 + n] if re.match(r'^[A-Z][a-z]?\s', x))


def make_mol(name, basis):
    return gto.M(atom=read_xyz(name), basis=basis, verbose=0)


def time_call(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, (time.perf_counter() - t0) * 1e3


def classify_cderi(dfobj):
    '''Deprecated helper; prefer dfobj.describe_cderi().'''
    kind, detail, nbytes = dfobj.describe_cderi()
    return f'{kind} {detail}', nbytes / 1e6


def run_case(name, mode, basis, grid_level, n_cycles, threads, prepare,
             storage=None, require_incore=False):
    lib.num_threads(threads)
    mol = make_mol(name, basis)
    # Give headroom for large DF tensors when forcing incore (PTCDA ~750 MB).
    mol.max_memory = max(mol.max_memory, 8000)
    mf = dft.RKS(mol, xc='PBE').density_fit()
    mf.grids.level = grid_level
    mf.max_cycle = n_cycles
    mf.conv_tol = 1e-7
    mf.conv_tol_grad = 1e-5
    mf.direct_scf = True
    mf.max_memory = mol.max_memory
    mf.with_df.max_memory = mol.max_memory
    if storage:
        mf.with_df.storage = storage

    budget = {'mode': mode, 'prepare': prepare, 'mol': name, 'nao': mol.nao_nr(),
              'df_storage_policy': getattr(mf.with_df, 'storage', 'auto')}

    # --- A) static setup ---
    t_setup0 = time.perf_counter()
    _, budget['grids_ms'] = time_call(mf.grids.build)

    if mode == 'cpu_ref':
        mf.backend = 1
        mf.with_df.backend = 1
    elif mode == 'cpu_small':
        from pyscf.smallDFT import prepare_smalldft_for_scf
        mf.backend = 1
        mf.with_df.backend = 1
        # prepare_smalldft: DF (incore) then AO cache then patch nr_rks
        t0 = time.perf_counter()
        prepare_smalldft_for_scf(
            mf, storage=storage or 'incore', require_incore=require_incore,
            max_memory_mb=mol.max_memory, n_workers=threads, ao_mode='cache')
        budget['prepare_smalldft_ms'] = (time.perf_counter() - t0) * 1e3
        budget['_smalldft_cleanup'] = True
        prepare = True  # DF already prepared
    elif mode == 'cpu_stream':
        from pyscf.smallDFT import prepare_smalldft_for_scf
        mf.backend = 1
        mf.with_df.backend = 1
        t0 = time.perf_counter()
        prepare_smalldft_for_scf(
            mf, storage=storage or 'incore', require_incore=require_incore,
            max_memory_mb=mol.max_memory, n_workers=threads, ao_mode='stream')
        budget['prepare_smalldft_ms'] = (time.perf_counter() - t0) * 1e3
        budget['_smalldft_cleanup'] = True
        prepare = True
    elif mode == 'gpu_otf':
        from pyscf.OpenCL import init_device
        from pyscf.OpenCL.gpu_profiles import apply_gpu_profile
        init_device(quiet=True)
        t0 = time.perf_counter()
        apply_gpu_profile(mf, 'production_otf', setup=True,
                          df_storage=storage, require_df_incore=require_incore)
        budget['gpu_profile_setup_ms'] = (time.perf_counter() - t0) * 1e3
        prepare = True
    elif mode == 'gpu_full':
        from pyscf.OpenCL import init_device
        from pyscf.OpenCL.gpu_profiles import apply_gpu_profile
        init_device(quiet=True)
        t0 = time.perf_counter()
        apply_gpu_profile(mf, 'fast_full_gpu', setup=True,
                          df_storage=storage, require_df_incore=require_incore)
        budget['gpu_profile_setup_ms'] = (time.perf_counter() - t0) * 1e3
        prepare = True
    else:
        raise ValueError(mode)

    if prepare and mode == 'cpu_ref':
        from pyscf.OpenCL.gpu_profiles import prepare_df_for_scf
        t0 = time.perf_counter()
        prepare_df_for_scf(mf, storage=storage, require_incore=require_incore)
        budget['df_build_ms'] = (time.perf_counter() - t0) * 1e3
    elif mode == 'cpu_small':
        # DF+AO already in prepare_smalldft_for_scf
        budget['df_build_ms'] = 0.0
    elif mode == 'cpu_stream':
        budget['df_build_ms'] = 0.0
    else:
        budget['df_build_ms'] = 0.0

    kind, detail, nbytes = mf.with_df.describe_cderi()
    budget['cderi_before_kernel'] = f'{kind} {detail}'
    budget['cderi_mb'] = nbytes / 1e6
    budget['df_storage_policy'] = getattr(mf.with_df, 'storage', 'auto')
    budget['setup_explicit_ms'] = (time.perf_counter() - t_setup0) * 1e3

    # Hook get_veff / nr_rks / get_j for per-call times
    from pyscf.dft import rks as rks_mod
    from pyscf.dft import numint as ni_mod
    veff_ms, jk_ms, xc_ms = [], [], []
    _veff = rks_mod.get_veff
    _getj = mf.get_j
    _nr_current = ni_mod.NumInt.nr_rks

    def veff_w(ks, mol=None, dm=None, *a, **kw):
        t0 = time.perf_counter()
        out = _veff(ks, mol, dm, *a, **kw)
        veff_ms.append((time.perf_counter() - t0) * 1e3)
        return out

    def getj_w(*a, **kw):
        t0 = time.perf_counter()
        out = _getj(*a, **kw)
        jk_ms.append((time.perf_counter() - t0) * 1e3)
        return out

    def nr_timed(self, *a, **kw):
        t0 = time.perf_counter()
        out = _nr_current(self, *a, **kw)
        xc_ms.append((time.perf_counter() - t0) * 1e3)
        return out

    rks_mod.get_veff = veff_w
    rks_mod.RKS.get_veff = veff_w
    mf.get_veff = lambda *a, **kw: veff_w(mf, *a, **kw)
    mf.get_j = getj_w
    ni_mod.NumInt.nr_rks = nr_timed

    # Also time DF.build if it happens lazily
    from pyscf.df import df as df_mod
    df_builds = []
    _df_build = df_mod.DF.build
    def df_build_w(self, *a, **kw):
        t0 = time.perf_counter()
        out = _df_build(self, *a, **kw)
        df_builds.append((time.perf_counter() - t0) * 1e3)
        return out
    df_mod.DF.build = df_build_w

    t0 = time.perf_counter()
    e = mf.kernel()
    kernel_ms = (time.perf_counter() - t0) * 1e3

    # restore
    rks_mod.get_veff = _veff
    rks_mod.RKS.get_veff = _veff
    df_mod.DF.build = _df_build
    if mode == 'cpu_small' or mode == 'cpu_stream':
        from pyscf.smallDFT import disable as small_disable
        small_disable()
    else:
        ni_mod.NumInt.nr_rks = _nr_current

    kind2, detail2, nbytes2 = mf.with_df.describe_cderi()
    budget.update({
        'energy': float(e),
        'converged': bool(mf.converged),
        'n_cycles_done': max(0, len(veff_ms) - 1),
        'kernel_ms': kernel_ms,
        'veff_calls_ms': veff_ms,
        'jk_calls_ms': jk_ms,
        'xc_calls_ms': xc_ms,
        'lazy_df_build_ms': df_builds,
        'cderi_after_kernel': f'{kind2} {detail2}',
        'cderi_mb_after': nbytes2 / 1e6,
        'total_wall_ms': budget['setup_explicit_ms'] + kernel_ms,
    })

    if veff_ms:
        budget['veff_init_ms'] = veff_ms[0]
        budget['veff_cycle_mean_ms'] = float(np.mean(veff_ms[1:])) if len(veff_ms) > 1 else float('nan')
        budget['veff_cycle_sum_ms'] = float(np.sum(veff_ms[1:])) if len(veff_ms) > 1 else 0.0
    if jk_ms:
        budget['jk_init_ms'] = jk_ms[0]
        budget['jk_cycle_mean_ms'] = float(np.mean(jk_ms[1:])) if len(jk_ms) > 1 else float('nan')
    if xc_ms:
        budget['xc_init_ms'] = xc_ms[0]
        budget['xc_cycle_mean_ms'] = float(np.mean(xc_ms[1:])) if len(xc_ms) > 1 else float('nan')
    return budget


def print_budget(b):
    log('=' * 72)
    log(f"{b['mol']}  mode={b['mode']}  prepare={b['prepare']}  "
        f"df.storage={b.get('df_storage_policy')}  nao={b['nao']}")
    log(f"  cderi before kernel: {b['cderi_before_kernel']}  ({b['cderi_mb']:.1f} MB)")
    log(f"  cderi after kernel:  {b['cderi_after_kernel']}  ({b.get('cderi_mb_after', 0):.1f} MB)")
    log(f"  explicit setup: grids={b.get('grids_ms', 0):.1f}  "
        f"prepare_smalldft={b.get('prepare_smalldft_ms', 0):.1f}  "
        f"df_build={b.get('df_build_ms', 0):.1f}  gpu_setup={b.get('gpu_profile_setup_ms', 0):.1f}  "
        f"TOTAL_setup={b['setup_explicit_ms']:.1f} ms")
    if b['lazy_df_build_ms']:
        log(f"  *** LAZY df.build INSIDE kernel: {b['lazy_df_build_ms']} ms")
    else:
        log(f"  lazy df.build inside kernel: none")
    log(f"  veff calls (ms): {[round(x, 1) for x in b['veff_calls_ms']]}")
    log(f"  xc   calls (ms): {[round(x, 1) for x in b.get('xc_calls_ms', [])]}")
    log(f"  jk   calls (ms): {[round(x, 1) for x in b['jk_calls_ms']]}")
    n = max(1, len(b['veff_calls_ms']) - 1)
    log(f"  veff_init={b.get('veff_init_ms', float('nan')):.1f}  "
        f"veff_cycle_mean={b.get('veff_cycle_mean_ms', float('nan')):.1f}  "
        f"xc_cycle_mean={b.get('xc_cycle_mean_ms', float('nan')):.1f}  "
        f"jk_cycle_mean={b.get('jk_cycle_mean_ms', float('nan')):.1f}  "
        f"n_cycles≈{n}  converged={b['converged']}  E={b['energy']:.6f}")
    log(f"  kernel_wall={b['kernel_ms']:.1f} ms   total_wall(setup+kernel)={b['total_wall_ms']:.1f} ms")
    # Budget equation
    setup = b['setup_explicit_ms'] + sum(b['lazy_df_build_ms'])
    cyc = b.get('veff_cycle_sum_ms', 0.0)
    init_extra = b.get('veff_init_ms', 0.0)  # includes first XC+J; if lazy DF, may include DF
    # Better: kernel = init_veff + sum(cycle_veff) + other
    other = b['kernel_ms'] - sum(b['veff_calls_ms'])
    log(f"  BUDGET: setup={setup:.0f} + veff_init={init_extra:.0f} + Σveff_cycles={cyc:.0f} "
        f"+ other_in_kernel={other:.0f} ≈ total {setup + b['kernel_ms']:.0f} ms")
    log('=' * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mol', default='benzene', choices=['benzene', 'pentacene', 'PTCDA', 'H2O'])
    ap.add_argument('--basis', default='6-31g')
    ap.add_argument('--grid-level', type=int, default=2)
    ap.add_argument('--n-cycles', type=int, default=8)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--modes', nargs='+', default=['cpu_ref', 'cpu_small'])
    ap.add_argument('--no-prepare', action='store_true', help='leave DF lazy (CPU modes)')
    ap.add_argument('--df-storage', default='incore', choices=['auto', 'incore', 'outcore'],
                    help='DF.storage policy (default incore for deterministic benches)')
    ap.add_argument('--require-incore', action='store_true', default=True,
                    help='fail if _cderi is not an in-RAM ndarray (default on)')
    ap.add_argument('--allow-outcore', action='store_true',
                    help='disable require_incore guard')
    args = ap.parse_args()
    require_incore = args.require_incore and not args.allow_outcore
    if args.df_storage == 'outcore':
        require_incore = False

    log(f'host threads={args.threads} OPENBLAS_NUM_THREADS={os.environ.get("OPENBLAS_NUM_THREADS")} '
        f'df.storage={args.df_storage} require_incore={require_incore}')
    results = []
    for mode in args.modes:
        prepare = not args.no_prepare
        if mode.startswith('gpu'):
            prepare = True
        try:
            b = run_case(args.mol, mode, args.basis, args.grid_level, args.n_cycles,
                         args.threads, prepare, storage=args.df_storage,
                         require_incore=require_incore)
            print_budget(b)
            results.append(b)
        except Exception as e:
            log(f'FAIL {mode}: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()

    if len(results) >= 2:
        a, b = results[0], results[1]
        log('\nSPEEDUP SUMMARY (same n_cycles cap)')
        log(f"  cycle mean veff: {a['mode']} {a.get('veff_cycle_mean_ms', float('nan')):.1f} → "
            f"{b['mode']} {b.get('veff_cycle_mean_ms', float('nan')):.1f}  "
            f"×{a.get('veff_cycle_mean_ms', 0) / max(b.get('veff_cycle_mean_ms', 1e-9), 1e-9):.2f}")
        log(f"  total wall:      {a['mode']} {a['total_wall_ms']:.0f} → "
            f"{b['mode']} {b['total_wall_ms']:.0f}  "
            f"×{a['total_wall_ms'] / max(b['total_wall_ms'], 1e-9):.2f}")


if __name__ == '__main__':
    main()
