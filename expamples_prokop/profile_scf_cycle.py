#!/usr/bin/env python3
'''Profile one SCF cycle via real mf.kernel() + PySCF monkey-patch timers.

Default mode runs the actual SCF driver (max_cycle=1) and records wall time for
rks.get_veff, scf.get_jk, df.get_jk, NumInt.nr_rks, eval_ao, eig, get_fock, etc.
First get_veff / nr_rks / get_jk call = scf_init; second = scf_cycle.

Pre-SCF (outside kernel): Grids.build, optional GridWorkspace.eval_ao.

Usage:
  OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/prokop/git/pyscf \\
    python3 expamples_prokop/profile_scf_cycle.py --mol benzene --threads 8

  OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/prokop/git/pyscf \\
    python3 expamples_prokop/profile_scf_cycle.py --mol benzene --df --profile

  # legacy hand-rolled stage timing (no get_jk hooks):
  python3 expamples_prokop/profile_scf_cycle.py --mol benzene --manual
'''
import argparse
import cProfile
import io
import os
import pstats
import time

import numpy
from pyscf import dft, gto, lib

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

MOLS = {
    'H2O': 'O 0 0 0; H 0 0 0.96; H 0 0.96 0',
    'benzene': '''
C 0 1.396 0; C 1.209 0.698 0; C 1.209 -0.698 0; C 0 -1.396 0; C -1.209 -0.698 0; C -1.209 0.698 0
H 0 2.479 0; H 2.146 1.239 0; H 2.146 -1.239 0; H 0 -2.479 0; H -2.146 -1.239 0; H -2.146 1.239 0
''',
}

# First call → scf_init, later calls → scf_cycle (within one kernel()).
_SPLIT_LABELS = frozenset({
    'rks.get_veff', 'NumInt.nr_rks', 'scf.get_jk', 'df.get_jk', 'hf.get_jk',
})

_timers = {}       # label -> {calls, wall, init_wall, cycle_wall}
_call_n = {}       # label -> int
_phase = 'kernel'  # pre_scf | kernel
_patches_installed = False


def _record(label, dt):
    rec = _timers.setdefault(label, {'calls': 0, 'wall': 0.0, 'init': 0.0, 'cycle': 0.0, 'other': 0.0})
    rec['calls'] += 1
    rec['wall'] += dt
    if label in _SPLIT_LABELS:
        n = _call_n.get(label, 0)
        _call_n[label] = n + 1
        key = 'init' if n == 0 else 'cycle'
    elif _phase == 'pre_scf':
        key = 'other'
    else:
        key = 'other'
    rec[key] += dt


def _wrap(label, fn):
    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        _record(label, time.perf_counter() - t0)
        return out
    wrapped.__name__ = getattr(fn, '__name__', label)
    return wrapped


def _patch_nr_rks(fn):
    from pyscf.dft import numint as numint_mod
    numint_mod.NumInt.nr_rks = lambda self, *a, **kw: _wrap('NumInt.nr_rks', fn)(self, *a, **kw)


def install_timers(small_nr_rks=None):
    global _patches_installed
    if _patches_installed:
        if small_nr_rks is not None:
            _patch_nr_rks(small_nr_rks)
        return
    from pyscf.scf import hf as hf_mod
    from pyscf.dft import rks as rks_mod
    from pyscf.dft import numint as numint_mod
    from pyscf.dft import gen_grid as gen_grid_mod

    rks_mod.get_veff = _wrap('rks.get_veff', rks_mod.get_veff)
    rks_mod.RKS.get_veff = rks_mod.get_veff

    _orig_get_jk = hf_mod.SCF.get_jk
    hf_mod.SCF.get_jk = lambda self, *a, **kw: _wrap('scf.get_jk', _orig_get_jk)(self, *a, **kw)
    if hasattr(hf_mod, 'RHF') and 'get_jk' in hf_mod.RHF.__dict__:
        _orig_rhf_jk = hf_mod.RHF.get_jk
        hf_mod.RHF.get_jk = lambda self, *a, **kw: _wrap('scf.get_jk', _orig_rhf_jk)(self, *a, **kw)
    # get_j → get_jk; timer on get_jk only (avoid duplicate lines)
    hf_mod.get_jk = _wrap('hf.get_jk', hf_mod.get_jk)

    try:
        from pyscf.df import df_jk as dfjk_mod
        dfjk_mod.get_jk = _wrap('df.get_jk', dfjk_mod.get_jk)
    except ImportError:
        pass

    _orig_nr_rks = numint_mod.NumInt.nr_rks
    _patch_nr_rks(small_nr_rks or _orig_nr_rks)

    numint_mod.eval_ao = _wrap('numint.eval_ao', numint_mod.eval_ao)

    _orig_block = numint_mod.NumInt.block_loop
    def _timed_block(self, *a, **kw):
        t0 = time.perf_counter()
        for item in _orig_block(self, *a, **kw):
            yield item
        _record('NumInt.block_loop', time.perf_counter() - t0)
    numint_mod.NumInt.block_loop = _timed_block

    gen_grid_mod.Grids.build = _wrap('Grids.build', gen_grid_mod.Grids.build)

    _orig_eig = hf_mod.SCF.eig
    hf_mod.SCF.eig = lambda self, *a, **kw: _wrap('scf.eig', _orig_eig)(self, *a, **kw)
    _orig_fock = hf_mod.SCF.get_fock
    hf_mod.SCF.get_fock = lambda self, *a, **kw: _wrap('scf.get_fock', _orig_fock)(self, *a, **kw)
    _orig_grad = hf_mod.SCF.get_grad
    hf_mod.SCF.get_grad = lambda self, *a, **kw: _wrap('scf.get_grad', _orig_grad)(self, *a, **kw)
    _orig_et = hf_mod.SCF.energy_tot
    hf_mod.SCF.energy_tot = lambda self, *a, **kw: _wrap('scf.energy_tot', _orig_et)(self, *a, **kw)

    _patches_installed = True


def _reset_timers():
    _timers.clear()
    _call_n.clear()


def _small_nr_rks_factory(ws):
    from pyscf.smallDFT import nr_rks as small_nr_rks
    def _dispatch(self, mol, grids, xc_code, dms, *args, **kwargs):
        kwargs.setdefault('n_workers', lib.num_threads())
        if ws is not None:
            kwargs['ws'] = ws
        return small_nr_rks(self, mol, grids, xc_code, dms, *args, **kwargs)
    return _dispatch


def profile_kernel(mol_name, path='ref', nthreads=1, xc='PBE', grids_level=3,
                   basis='6-31g', use_df=False, do_cprofile=False, repeat=1):
    global _phase
    lib.num_threads(nthreads)
    _reset_timers()

    mol = gto.M(atom=MOLS[mol_name], basis=basis, verbose=0)
    mf = dft.RKS(mol, xc=xc)
    mf.grids.level = grids_level
    mf.verbose = 0
    mf.max_cycle = 1
    mf.conv_tol = 1e-10
    mf.direct_scf = True
    if use_df:
        mf = mf.density_fit()

    ws = None
    small_fn = None

    pre = {}
    _phase = 'pre_scf'
    t0 = time.perf_counter()
    mf.grids.build(with_non0tab=True)
    pre['Grids.build'] = (time.perf_counter() - t0) * 1000

    if path == 'smallDFT_ws':
        from pyscf.smallDFT import GridWorkspace
        ws = GridWorkspace(mol, mf.grids, deriv=1)
        t0 = time.perf_counter()
        ws.eval_ao(mol, mf.grids)
        pre['GridWorkspace.eval_ao'] = (time.perf_counter() - t0) * 1000
        mf._smallDFT_ws = ws
    if path in ('smallDFT', 'smallDFT_ws'):
        small_fn = _small_nr_rks_factory(ws)
    install_timers(small_nr_rks=small_fn)

    _phase = 'kernel'
    cprofile_text = None
    kernel_wall = []
    for _ in range(repeat):
        _reset_timers()
        # Re-seed split counters each repeat; keep grid built
        t0 = time.perf_counter()
        if do_cprofile:
            pr = cProfile.Profile()
            pr.enable()
            e = mf.kernel()
            pr.disable()
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(35)
            cprofile_text = s.getvalue()
        else:
            e = mf.kernel()
        kernel_wall.append(time.perf_counter() - t0)

    return {
        'mol': mol_name, 'path': path, 'nthreads': nthreads, 'use_df': use_df,
        'nao': mol.nao_nr(), 'ngrids': mf.grids.coords.shape[0],
        'energy': e, 'pre': pre,
        'kernel_wall_ms': min(kernel_wall) * 1000,
        'timers': {k: dict(v) for k, v in _timers.items()},
        'cprofile': cprofile_text,
    }


def print_kernel_report(res):
    pre_tot = sum(res['pre'].values())
    print(f"\n{'='*78}")
    print(f"{res['mol']}  path={res['path']}  threads={res['nthreads']}  "
          f"df={res['use_df']}  nao={res['nao']}  ngrids={res['ngrids']}")
    print(f"{'='*78}")
    print(f"  pre_scf (explicit, outside kernel):")
    for k, v in res['pre'].items():
        print(f"    {k:<28} {v:8.1f} ms")
    print(f"    {'TOTAL pre_scf':<28} {pre_tot:8.1f} ms")
    print(f"  mf.kernel(max_cycle=1) wall:     {res['kernel_wall_ms']:8.1f} ms")
    print()
    print(f"  {'label':<26} {'calls':>5} {'total':>9} {'init':>9} {'cycle':>9} {'per_call':>9}")
    print(f"  {'-'*26} {'-'*5} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    for label, rec in sorted(res['timers'].items(), key=lambda x: -x[1]['wall']):
        tot = rec['wall'] * 1000
        init = rec.get('init', 0) * 1000
        cyc = rec.get('cycle', 0) * 1000
        pc = tot / rec['calls'] if rec['calls'] else 0
        print(f"  {label:<26} {rec['calls']:5d} {tot:9.1f} {init:9.1f} {cyc:9.1f} {pc:9.1f}")

    t = res['timers']
    if 'rks.get_veff' in t and t['rks.get_veff']['calls'] >= 2:
        cyc_veff = t['rks.get_veff'].get('cycle', 0) * 1000
        cyc_xc = t.get('NumInt.nr_rks', {}).get('cycle', 0) * 1000
        cyc_jk = t.get('scf.get_jk', {}).get('cycle', 0) * 1000
        if cyc_jk == 0:
            cyc_jk = t.get('df.get_jk', {}).get('cycle', 0) * 1000
        print()
        print(f"  One SCF iteration (2nd call): get_veff {cyc_veff:.1f} ms "
              f"(nr_rks {cyc_xc:.1f} + J/get_jk {cyc_jk:.1f} ms)")

    if res.get('cprofile'):
        print(f"\n{'='*78}")
        print('cProfile top 35 (tottime)')
        print(res['cprofile'])


# --- legacy manual mode (kept for sub-step min timing) ---

def _min_veff(run_veff, repeat=5, warmup=2):
    best = None
    best_vhf = None
    for _ in range(repeat + warmup):
        t_xc, t_j, t_tot, vhf = run_veff()
        if best is None or t_tot < best[2]:
            best = (t_xc, t_j, t_tot)
            best_vhf = vhf
    return best[0], best[1], best[2], best_vhf


def _min_ms(fn, repeat=5, warmup=2):
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t0)
    return min(xs) * 1000


def _get_veff_parts(ks, mol, dm, dm_last, vhf_last, ni, use_small, ws):
    max_memory = ks.max_memory - lib.current_memory()[0]
    t0 = time.perf_counter()
    if use_small:
        from pyscf.smallDFT import nr_rks
        n, exc, vmat_xc = nr_rks(ni, mol, ks.grids, ks.xc, dm, ws=ws,
                                 n_workers=lib.num_threads())
    else:
        n, exc, vmat_xc = ni.nr_rks(mol, ks.grids, ks.xc, dm, max_memory=max_memory)
    t_xc = time.perf_counter() - t0
    incremental_jk = (ks._eri is None and ks.direct_scf and
                      getattr(vhf_last, 'vj', None) is not None)
    _dm = numpy.asarray(dm) - numpy.asarray(dm_last) if incremental_jk else dm
    t0 = time.perf_counter()
    vj = ks.get_j(mol, _dm, 1)
    if incremental_jk:
        vj = vj + vhf_last.vj
    t_j = time.perf_counter() - t0
    vxc = vmat_xc + vj
    ecoul = numpy.einsum('ij,ji', dm, vj).real * .5
    vhf = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=None)
    return t_xc * 1000, t_j * 1000, vhf


def _run_veff(ks, mol, dm, dm_last, vhf_last, ni, use_small, ws):
    t_xc, t_j, vhf = _get_veff_parts(ks, mol, dm, dm_last, vhf_last, ni, use_small, ws)
    return t_xc, t_j, t_xc + t_j, vhf


def profile_manual(mol_name, path='ref', nthreads=1, xc='PBE', grids_level=3,
                   basis='6-31g', repeat=5):
    lib.num_threads(nthreads)
    mol = gto.M(atom=MOLS[mol_name], basis=basis, verbose=0)
    mf = dft.RKS(mol, xc=xc)
    mf.grids.level = grids_level
    mf.verbose = 0
    mf.max_cycle = 1
    mf.direct_scf = True
    t0 = time.perf_counter()
    mf.grids.build(with_non0tab=True)
    pre_grid = (time.perf_counter() - t0) * 1000
    ni = mf._numint
    ws = None
    use_small = path in ('smallDFT', 'smallDFT_ws')
    if path == 'smallDFT_ws':
        from pyscf.smallDFT import GridWorkspace
        ws = GridWorkspace(mol, mf.grids, deriv=1)
        pre_ao = _min_ms(lambda: ws.eval_ao(mol, mf.grids), repeat=repeat)
    else:
        pre_ao = 0.0
    s1e = mf.get_ovlp(mol)
    dm = mf.get_init_guess(mol, mf.init_guess, s1e=s1e)
    h1e = mf.get_hcore(mol)
    x_orth = mf.check_linear_dependency(s1e)

    def _init_veff():
        return _run_veff(mf, mol, dm, 0, 0, ni, use_small, ws)
    tx, tj, tt, vhf = _min_veff(_init_veff, repeat=repeat)
    fock = mf.get_fock(h1e, s1e, vhf, dm)
    mo_energy, mo_coeff = mf.eig(fock, s1e, x=x_orth)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    dm_new = mf.make_rdm1(mo_coeff, mo_occ)

    def _cycle_veff():
        return _run_veff(mf, mol, dm_new, dm, vhf, ni, use_small, ws)
    tx2, tj2, tt2, _ = _min_veff(_cycle_veff, repeat=repeat)

    def _one_cycle_wall():
        f = mf.get_fock(h1e, s1e, vhf, dm)
        mo_e, mo_c = mf.eig(f, s1e, x=x_orth)
        dm1 = mf.make_rdm1(mo_c, mf.get_occ(mo_e, mo_c))
        _run_veff(mf, mol, dm1, dm, vhf, ni, use_small, ws)

    return {
        'mol': mol_name, 'path': path, 'nthreads': nthreads,
        'pre_scf': pre_grid + pre_ao, 'cycle_wall': _min_ms(_one_cycle_wall, repeat=repeat),
        'cycle_veff': tt2, 'cycle_xc': tx2, 'cycle_j': tj2,
    }


def main():
    parser = argparse.ArgumentParser(description='Profile one SCF cycle')
    parser.add_argument('--mol', nargs='+', default=['benzene'], choices=list(MOLS))
    parser.add_argument('--path', nargs='+', default=['ref'],
                        choices=['ref', 'smallDFT', 'smallDFT_ws'])
    parser.add_argument('--threads', nargs='+', type=int, default=[8])
    parser.add_argument('--df', action='store_true', help='density fitting (RI-J)')
    parser.add_argument('--profile', action='store_true', help='cProfile on kernel()')
    parser.add_argument('--manual', action='store_true', help='legacy hand-rolled timers (no get_jk)')
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()

    print(f'PySCF: {gto.__file__.split("pyscf")[0]}pyscf')
    print(f'OPENBLAS_NUM_THREADS={os.environ.get("OPENBLAS_NUM_THREADS", "(unset)")}')
    print(f'mode: {"manual" if args.manual else "kernel (mf.kernel + monkey-patch)"}')

    if args.manual:
        for mol in args.mol:
            for path in args.path:
                for nt in args.threads:
                    r = profile_manual(mol, path=path, nthreads=nt, repeat=max(3, args.repeat))
                    print(f"\n{mol} {path} @{nt}: cycle_wall={r['cycle_wall']:.1f} ms  "
                          f"veff={r['cycle_veff']:.1f} (xc={r['cycle_xc']:.1f} j={r['cycle_j']:.1f})")
        return

    results = []
    for mol in args.mol:
        for path in args.path:
            for nt in args.threads:
                results.append(profile_kernel(
                    mol, path=path, nthreads=nt, use_df=args.df,
                    do_cprofile=args.profile, repeat=max(1, args.repeat)))
    for res in results:
        print_kernel_report(res)


if __name__ == '__main__':
    main()
