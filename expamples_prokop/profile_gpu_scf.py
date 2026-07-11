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
import json
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
_XYZ_DIR = os.path.join(_REPO, 'data', 'xyz')

_timers = {}
_timers_installed = False


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
        rec = _timers.setdefault(label, {'calls': 0, 'wall': 0.0, 'init': 0.0, 'cycle': 0.0, 'calls_ms': []})
        phase = 'init' if rec['calls'] == 0 else 'cycle'
        rec['calls'] += 1
        rec['wall'] += dt
        rec[phase] += dt
        rec['calls_ms'].append(dt * 1e3)
        return out
    wrapped.__name__ = getattr(fn, '__name__', label)
    return wrapped


def install_timers():
    global _timers_installed
    if _timers_installed:
        return
    from pyscf.scf import hf as hf_mod
    from pyscf.dft import rks as rks_mod
    from pyscf.dft import gen_grid
    from pyscf.df import df as df_mod
    from pyscf.df import df_jk as dfjk_mod
    hf_mod.kernel = _wrap('scf.kernel', hf_mod.kernel)
    rks_mod.get_veff = _wrap('rks.get_veff', rks_mod.get_veff)
    rks_mod.RKS.get_veff = rks_mod.get_veff
    for name in ('get_hcore', 'get_ovlp', 'get_init_guess', 'get_fock', 'eig',
                 'get_occ', 'make_rdm1', 'energy_tot', 'get_grad',
                 'check_linear_dependency', 'pre_kernel'):
        fn = getattr(hf_mod.SCF, name, None)
        if fn is not None:
            setattr(hf_mod.SCF, name, _wrap(f'scf.{name}', fn))
    _orig_get_jk = hf_mod.SCF.get_jk
    hf_mod.SCF.get_jk = _wrap('scf.get_jk', _orig_get_jk)
    dfjk_mod.get_jk = _wrap('df.get_jk', dfjk_mod.get_jk)
    dfjk_mod._get_jk_cpu = _wrap('df._get_jk_cpu', dfjk_mod._get_jk_cpu)
    df_mod.DF.build = _wrap('df.build', df_mod.DF.build)
    gen_grid.Grids.build = _wrap('grid.build', gen_grid.Grids.build)
    try:
        from pyscf.OpenCL import df_jk as gpu_dfjk
        gpu_dfjk.df_jk_gpu = _wrap('gpu_df_jk', gpu_dfjk.df_jk_gpu)
        gpu_dfjk.DFJKPlan.__init__ = _wrap('gpu_df_plan_init', gpu_dfjk.DFJKPlan.__init__)
        gpu_dfjk.DFJKPlan.get_jk = _wrap('gpu_df_jk_contract', gpu_dfjk.DFJKPlan.get_jk)
        from pyscf.OpenCL import xc_grid
        xc_grid.XCGridPlan.setup_onthefly = _wrap('gpu_xc_setup', xc_grid.XCGridPlan.setup_onthefly)
        xc_grid.XCGridPlan.nr_rks_hermite_onthefly = _wrap('gpu_xc_outer', xc_grid.XCGridPlan.nr_rks_hermite_onthefly)
    except ImportError:
        pass
    _timers_installed = True


def print_timers():
    if not _timers:
        return
    log('\n--- Python wall timers (monkey-patch) ---')
    log(f'{"label":<22} {"calls":>6} {"init_ms":>10} {"cycle_ms":>10} {"wall_ms":>10}')
    for label, rec in sorted(_timers.items(), key=lambda x: -x[1]['wall']):
        log(f'{label:<22} {rec["calls"]:>6} {rec["init"]*1e3:>10.1f} {rec["cycle"]*1e3:>10.1f} {rec["wall"]*1e3:>10.1f}')
    log('  call sequence (ms):')
    for label, rec in sorted(_timers.items(), key=lambda x: -x[1]['wall']):
        if rec['calls'] > 1:
            vals = ', '.join(f'{v:.1f}' for v in rec['calls_ms'])
            log(f'    {label:<20} {vals}')


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


def build_mf(mol_name, basis, grid_level, use_df, conv_tol, max_cycle, xyz_dir=_XYZ_DIR):
    xyz_path = os.path.join(xyz_dir, f'{mol_name}.xyz')
    mol = gto.M(atom=read_xyz(xyz_path), basis=basis, verbose=0)
    mf = dft.RKS(mol, xc='PBE')
    mf.grids.level = grid_level
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = 1e-5
    mf.max_cycle = max_cycle
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
        from pyscf.OpenCL.gpu_profiles import prepare_df_for_scf
        prepare_df_for_scf(mf)
        return mf
    from pyscf.OpenCL import init_device
    from pyscf.OpenCL.gpu_profiles import apply_gpu_profile
    init_device(quiet=True)
    prof_name = _MODE_TO_PROFILE[mode]
    apply_gpu_profile(mf, prof_name, setup=True)
    mf._gpu_profile = True
    return mf


def run_scf(mf, do_profile):
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
    def timer_ms(label, phase):
        return _timers.get(label, {}).get(phase, 0.0) * 1e3
    return dict(energy=e, wall_s=wall, converged=mf.converged, cycles=n_cycles,
                init_veff_ms=timer_ms('rks.get_veff', 'init'),
                cycle_veff_ms=timer_ms('rks.get_veff', 'cycle'),
                init_j_ms=timer_ms('df.get_jk', 'init'),
                cycle_j_ms=timer_ms('df.get_jk', 'cycle'),
                init_eig_ms=timer_ms('scf.eig', 'init'),
                cycle_eig_ms=timer_ms('scf.eig', 'cycle'),
                timers={k: list(v.get('calls_ms', [])) for k, v in _timers.items()},
                cprofile=cprofile_text)


def main():
    ap = argparse.ArgumentParser(description='Full SCF profile: CPU vs GPU')
    ap.add_argument('--mols', nargs='+', default=['benzene'], choices=['benzene', 'pentacene', 'PTCDA'])
    ap.add_argument('--mode', nargs='+', default=['cpu', 'gpu_otf', 'gpu_full'],
                    choices=['cpu', 'gpu_otf', 'gpu_coalesced', 'gpu_full'])
    ap.add_argument('--basis', default='ccpvdz')
    ap.add_argument('--grid-level', type=int, default=3)
    ap.add_argument('--conv-tol', type=float, default=1e-8)
    ap.add_argument('--max-cycle', type=int, default=50, help='SCF iterations; use 1 for one-cycle Amdahl profiling')
    ap.add_argument('--no-df', action='store_true')
    ap.add_argument('--profile', action='store_true', help='cProfile last mode only')
    ap.add_argument('--json', default=None, help='write detailed timer records to JSON')
    ap.add_argument('--threads', type=int, default=None)
    args = ap.parse_args()

    if args.threads is not None:
        lib.num_threads(args.threads)
        os.environ['OMP_NUM_THREADS'] = str(args.threads)

    log(f'PySCF {pyscf.__version__}  OMP_NUM_THREADS={lib.num_threads()}')
    install_timers()
    use_df = not args.no_df
    rows = []

    for mol_name in args.mols:
      for mode in args.mode:
        log(f'\n{"="*72}')
        log(f'MOLECULE: {mol_name}  MODE: {mode}  basis={args.basis}  grid={args.grid_level}  DF={use_df}')
        t_build = time.perf_counter()
        mf = build_mf(mol_name, args.basis, args.grid_level, use_df, args.conv_tol, args.max_cycle)
        build_ms = (time.perf_counter() - t_build) * 1e3
        nao = mf.mol.nao_nr()
        _timers.clear()
        t_setup = time.perf_counter()
        configure_mode(mf, mode)
        setup_ms = (time.perf_counter() - t_setup) * 1e3
        prof_name = _MODE_TO_PROFILE.get(mode)
        if prof_name:
            from pyscf.OpenCL.gpu_profiles import get_profile
            p = get_profile(prof_name)
            log(f'  profile={prof_name}  scf_tol={p.get("scf_kw")}  note={p.get("accuracy", {}).get("energy_note", "")}')
        log(f'  nao={nao}  backend={getattr(mf,"backend",1)}  df_backend={getattr(mf.with_df,"backend",1) if use_df else "n/a"}')

        do_prof = args.profile and (mode == args.mode[-1] and mol_name == args.mols[-1])
        out = run_scf(mf, do_prof)
        rows.append(dict(molecule=mol_name, mode=mode, build_ms=build_ms, setup_ms=setup_ms, **out))
        log(f'  molecular build={build_ms:.1f} ms  backend setup={setup_ms:.1f} ms')
        log(f'  converged={out["converged"]}  cycles={out["cycles"]}  E={out["energy"]:.10f}')
        log(f'  total wall={out["wall_s"]:.2f} s  per_cycle={out["wall_s"]/max(out["cycles"],1)*1e3:.1f} ms')
        print_timers()
        print_gpu_xc_acc(mf)
        if out.get('cprofile'):
            log('\n--- cProfile top 25 (tottime; GPU kernels mostly invisible) ---')
            log(out['cprofile'])

    log(f'\n{"="*72}')
    log('SUMMARY')
    log(f'{"molecule":<12} {"mode":<16} {"cycles":>6} {"wall_s":>8} {"ms/cyc":>8} {"conv":>5}')
    for r in rows:
        ms = r['wall_s'] / max(r['cycles'], 1) * 1e3
        log(f'{r["molecule"]:<12} {r["mode"]:<16} {r["cycles"]:>6} {r["wall_s"]:>8.2f} {ms:>8.1f} {str(r["converged"]):>5}')
    for mol_name in args.mols:
        cpu = next((r for r in rows if r['molecule'] == mol_name and r['mode'] == 'cpu'), None)
        if cpu is None:
            continue
        cpu_ms = cpu['wall_s'] / max(cpu['cycles'], 1) * 1e3
        for r in (x for x in rows if x['molecule'] == mol_name and x['mode'] != 'cpu'):
            gpu_ms = r['wall_s'] / max(r['cycles'], 1) * 1e3
            log(f'  {mol_name} {r["mode"]} speedup vs cpu: {cpu_ms/gpu_ms:.2f}x per cycle')
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(rows, f, indent=2)
        log(f'Wrote detailed profile: {args.json}')


if __name__ == '__main__':
    main()
