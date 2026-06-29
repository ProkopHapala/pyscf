#!/usr/bin/env python3
'''Full SCF convergence profile: CPU vs GPU XC (and optional GPU DF J/K).

Profiles: pyscf/OpenCL/gpu_profiles.py · doc/opencl_gpu_paths_cookbook.md

cProfile captures host Python; GPU kernels appear via mf._gpu_timing_acc when
_gpu_profile=True (queue.finish per gpu_* stage in xc_grid).

Usage:
  PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=15 python3 -u \\
    expamples_prokop/profile_gpu_scf.py --mode cpu gpu_otf gpu_full

  PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=15 python3 -u \\
    expamples_prokop/profile_gpu_scf.py --mode gpu_otf --profile
'''
import argparse
import cProfile
import io
import os
import pstats
import re
import sys
import time

import numpy as np
import pyscf
from pyscf import dft, gto, lib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_XYZ = os.path.join(_REPO, 'data', 'xyz', 'benzene.xyz')

_timers = {}


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


def _wrap(label, fn):
    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        rec = _timers.setdefault(label, {'calls': 0, 'wall': 0.0})
        rec['calls'] += 1
        rec['wall'] += dt
        return out
    wrapped.__name__ = getattr(fn, '__name__', label)
    return wrapped


def install_timers():
    from pyscf.scf import hf as hf_mod
    from pyscf.dft import rks as rks_mod
    hf_mod.kernel = _wrap('scf.kernel', hf_mod.kernel)
    rks_mod.get_veff = _wrap('rks.get_veff', rks_mod.get_veff)
    rks_mod.RKS.get_veff = rks_mod.get_veff
    _orig_get_jk = hf_mod.SCF.get_jk
    hf_mod.SCF.get_jk = lambda self, *a, **kw: _wrap('scf.get_jk', _orig_get_jk)(self, *a, **kw)
    from pyscf.df import df_jk as dfjk_mod
    dfjk_mod.get_jk = _wrap('df.get_jk', dfjk_mod.get_jk)
    _orig_eig = hf_mod.SCF.eig
    hf_mod.SCF.eig = lambda self, *a, **kw: _wrap('scf.eig', _orig_eig)(self, *a, **kw)
    _orig_get_fock = hf_mod.SCF.get_fock
    hf_mod.SCF.get_fock = lambda self, *a, **kw: _wrap('scf.get_fock', _orig_get_fock)(self, *a, **kw)


def print_timers():
    if not _timers:
        return
    log('\n--- Python wall timers (monkey-patch) ---')
    log(f'{"label":<22} {"calls":>6} {"wall_ms":>10} {"per_call_ms":>12}')
    for label, rec in sorted(_timers.items(), key=lambda x: -x[1]['wall']):
        pc = rec['wall'] / rec['calls'] * 1e3 if rec['calls'] else 0
        log(f'{label:<22} {rec["calls"]:>6} {rec["wall"]*1e3:>10.1f} {pc:>12.1f}')


def print_gpu_xc_acc(mf):
    acc = mf.__dict__.get('_gpu_timing_acc')
    if not acc:
        return
    from pyscf.OpenCL.xc_grid import TIMING_STAGE_ORDER
    n_cycles = max(1, mf.__dict__.get('_gpu_profile_cycles', 1))
    log(f'\n--- GPU XC accumulated ({n_cycles} get_veff calls) ---')
    for k in TIMING_STAGE_ORDER:
        if k not in acc or acc[k] <= 0:
            continue
        total_ms = acc[k] * 1e3
        per_ms = total_ms / n_cycles
        log(f'  {k:22s} total={total_ms:9.1f} ms  per_cycle={per_ms:7.1f} ms')
    gpu_total = sum(acc.get(k, 0) for k in ('gpu_rho', 'gpu_xc_pbe', 'gpu_vmat')) * 1e3
    host_total = sum(acc.get(k, 0) for k in ('host_h2d_dm', 'host_dm_cart', 'host_rho_d2h', 'host_xc_libxc', 'host_xc_reduce', 'host_vmat_d2h')) * 1e3
    log(f'  {"gpu_xc_sum":22s} total={gpu_total:9.1f} ms  per_cycle={gpu_total/n_cycles:7.1f} ms')
    log(f'  {"host_xc_pcie":22s} total={host_total:9.1f} ms  per_cycle={host_total/n_cycles:7.1f} ms')


def build_mf(basis, grid_level, use_df, conv_tol):
    mol = gto.M(atom=read_xyz(_XYZ), basis=basis, verbose=0)
    mf = dft.RKS(mol, xc='PBE')
    mf.grids.level = grid_level
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = 1e-5
    mf.max_cycle = 50
    if use_df:
        mf = mf.density_fit()
    return mf


_MODE_TO_PROFILE = {
    'cpu': 'cpu_reference',
    'gpu_otf': 'production_otf',
    'gpu_coalesced': 'production_coalesced',
    'gpu_full': 'fast_full_gpu',
}


def configure_mode(mf, mode):
    mf.__dict__.pop('_xc_gpu_plan', None)
    mf.__dict__.pop('_gpu_timing_acc', None)
    mf.__dict__.pop('_gpu_profile', None)
    if mode == 'cpu':
        mf.backend = 1
        if 'with_df' in mf.__dict__ and mf.with_df is not None:
            mf.with_df.backend = 1
        return mf
    from pyscf.OpenCL import init_device
    from pyscf.OpenCL.gpu_profiles import apply_gpu_profile
    init_device(quiet=True)
    prof_name = _MODE_TO_PROFILE[mode]
    apply_gpu_profile(mf, prof_name, setup=True)
    mf._gpu_profile = True
    return mf


def run_scf(mf, do_profile):
    _timers.clear()
    mf.__dict__['_gpu_timing_acc'] = {}
    t0 = time.perf_counter()
    if do_profile:
        pr = cProfile.Profile()
        pr.enable()
        e = mf.kernel()
        pr.disable()
        prof_out = io.StringIO()
        pstats.Stats(pr, stream=prof_out).sort_stats('tottime').print_stats(25)
        cprofile_text = prof_out.getvalue()
    else:
        e = mf.kernel()
        cprofile_text = None
    wall = time.perf_counter() - t0
    n_cycles = getattr(mf, 'cycles', 0) or 1
    mf.__dict__['_gpu_profile_cycles'] = n_cycles
    return dict(energy=e, wall_s=wall, converged=mf.converged, cycles=n_cycles, cprofile=cprofile_text)


def main():
    ap = argparse.ArgumentParser(description='Full SCF profile: CPU vs GPU')
    ap.add_argument('--mode', nargs='+', default=['cpu', 'gpu_otf', 'gpu_full'],
                    choices=['cpu', 'gpu_otf', 'gpu_coalesced', 'gpu_full'])
    ap.add_argument('--basis', default='ccpvdz')
    ap.add_argument('--grid-level', type=int, default=3)
    ap.add_argument('--conv-tol', type=float, default=1e-8)
    ap.add_argument('--no-df', action='store_true')
    ap.add_argument('--profile', action='store_true', help='cProfile last mode only')
    ap.add_argument('--threads', type=int, default=None)
    args = ap.parse_args()

    if args.threads is not None:
        lib.num_threads(args.threads)
        os.environ['OMP_NUM_THREADS'] = str(args.threads)

    log(f'PySCF {pyscf.__version__}  OMP_NUM_THREADS={lib.num_threads()}')
    install_timers()
    use_df = not args.no_df
    rows = []

    for mode in args.mode:
        log(f'\n{"="*72}')
        log(f'MODE: {mode}  basis={args.basis}  grid={args.grid_level}  DF={use_df}')
        mf = build_mf(args.basis, args.grid_level, use_df, args.conv_tol)
        nao = mf.mol.nao_nr()
        configure_mode(mf, mode)
        prof_name = _MODE_TO_PROFILE.get(mode)
        if prof_name:
            from pyscf.OpenCL.gpu_profiles import get_profile
            p = get_profile(prof_name)
            log(f'  profile={prof_name}  scf_tol={p.get("scf_kw")}  note={p.get("accuracy", {}).get("energy_note", "")}')
        log(f'  nao={nao}  backend={getattr(mf,"backend",1)}  df_backend={getattr(mf.with_df,"backend",1) if use_df else "n/a"}')

        do_prof = args.profile and (mode == args.mode[-1])
        out = run_scf(mf, do_prof)
        rows.append(dict(mode=mode, **out))
        log(f'  converged={out["converged"]}  cycles={out["cycles"]}  E={out["energy"]:.10f}')
        log(f'  total wall={out["wall_s"]:.2f} s  per_cycle={out["wall_s"]/max(out["cycles"],1)*1e3:.1f} ms')
        print_timers()
        print_gpu_xc_acc(mf)
        if out.get('cprofile'):
            log('\n--- cProfile top 25 (tottime; GPU kernels mostly invisible) ---')
            log(out['cprofile'])

    log(f'\n{"="*72}')
    log('SUMMARY')
    log(f'{"mode":<16} {"cycles":>6} {"wall_s":>8} {"ms/cyc":>8} {"conv":>5}')
    for r in rows:
        ms = r['wall_s'] / max(r['cycles'], 1) * 1e3
        log(f'{r["mode"]:<16} {r["cycles"]:>6} {r["wall_s"]:>8.2f} {ms:>8.1f} {str(r["converged"]):>5}')
    if len(rows) >= 2 and rows[0]['mode'] == 'cpu':
        cpu_ms = rows[0]['wall_s'] / max(rows[0]['cycles'], 1)
        for r in rows[1:]:
            gpu_ms = r['wall_s'] / max(r['cycles'], 1)
            log(f'  {r["mode"]} speedup vs cpu: {cpu_ms/gpu_ms:.2f}x per cycle')


if __name__ == '__main__':
    main()
