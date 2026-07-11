#!/usr/bin/env python3
'''Non-overlapping, same-input CPU/GPU RKS-cycle Amdahl profile.

Static grid/DF/GPU setup is completed before timing.  The timed cycle mirrors
the PBE RKS SCF driver and validates the manually assembled veff against the
real mf.get_veff call.  GPU rho/vmat event timings are diagnostic only and are
not added to wall time.
'''
import argparse
import os
import re
import time

import numpy as np
from pyscf import dft, gto, lib
from pyscf.dft.numint import NBINS, _dot_ao_ao_sparse, _scale_ao_sparse

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_XYZ_DIR = os.path.join(_REPO, 'data', 'xyz')


def read_xyz(path):
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0])
    return '; '.join(' '.join(x.split()[:4]) for x in lines[2:2 + n] if re.match(r'^[A-Z][a-z]?\s', x))


def make_mf(name, mode, basis, grid_level):
    mol = gto.M(atom=read_xyz(os.path.join(_XYZ_DIR, f'{name}.xyz')), basis=basis, verbose=0)
    mf = dft.RKS(mol, xc='PBE').density_fit()
    mf.grids.level = grid_level
    mf.max_cycle = 1
    if mode == 'cpu':
        mf.backend = mf.with_df.backend = 1
        from pyscf.OpenCL.gpu_profiles import prepare_df_for_scf
        prepare_df_for_scf(mf)
    elif mode == 'gpu_full':
        from pyscf.OpenCL import init_device
        from pyscf.OpenCL.gpu_profiles import apply_gpu_profile
        init_device(quiet=True)
        apply_gpu_profile(mf, 'fast_full_gpu', setup=True)
    else:
        raise ValueError(mode)
    return mf


def cpu_nr_rks_staged(ni, mol, grids, xc_code, dm, max_memory):
    '''PBE/GGA nr_rks with additive CPU stage timings and identical algebra.'''
    xctype = ni._xc_type(xc_code)
    if xctype != 'GGA':
        raise NotImplementedError(xctype)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, 1, False, grids)
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(grids.cutoff))
    pair_mask = mol.get_overlap_cond() < -np.log(ni.cutoff)
    nelec = np.zeros(nset)
    excsum = np.zeros(nset)
    vmat = np.zeros((nset, nao, nao))
    aow = None
    ts = dict(ao=0.0, rho=0.0, libxc=0.0, bookkeeping=0.0, vmat=0.0)
    blocks = iter(ni.block_loop(mol, grids, nao, 1, max_memory=max_memory))
    while True:
        t0 = time.perf_counter()
        try:
            ao, mask, weight, _ = next(blocks)
        except StopIteration:
            break
        ts['ao'] += time.perf_counter() - t0
        for i in range(nset):
            t0 = time.perf_counter()
            rho = make_rho(i, ao, mask, xctype)
            ts['rho'] += time.perf_counter() - t0
            t0 = time.perf_counter()
            exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=0)[:2]
            ts['libxc'] += time.perf_counter() - t0
            t0 = time.perf_counter()
            den = rho[0] * weight
            nelec[i] += den.sum()
            excsum[i] += np.dot(den, exc)
            wv = weight * vxc
            ts['bookkeeping'] += time.perf_counter() - t0
            t0 = time.perf_counter()
            wv[0] *= .5
            aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc, hermi=0, out=vmat[i])
            ts['vmat'] += time.perf_counter() - t0
    t0 = time.perf_counter()
    vmat = lib.hermi_sum(vmat, axes=(0, 2, 1))
    ts['vmat'] += time.perf_counter() - t0
    return nelec[0], excsum[0], vmat[0], ts


def veff_parts(mf, dm, dm_last, vhf_last, gpu_profile=False):
    '''Exact non-hybrid RKS veff decomposition; parts do not overlap.'''
    mol, ni = mf.mol, mf._numint
    t0 = time.perf_counter()
    cpu_xc = {}
    if mf.backend & 2:
        plan = mf._xc_gpu_plan
        if mf._gpu_xc_path == 'precomputed':
            n, exc, vmat_xc = plan.nr_rks_precomputed_gto(dm, profile=gpu_profile)
        else:
            n, exc, vmat_xc = plan.nr_rks_hermite_onthefly(dm, profile=gpu_profile)
    else:
        mem = mf.max_memory - lib.current_memory()[0]
        n, exc, vmat_xc, cpu_xc = cpu_nr_rks_staged(ni, mol, mf.grids, mf.xc, dm, mem)
    t_xc = time.perf_counter() - t0
    incremental = mf._eri is None and mf.direct_scf and getattr(vhf_last, 'vj', None) is not None
    dmd = np.asarray(dm) - np.asarray(dm_last) if incremental else dm
    t0 = time.perf_counter()
    vj = mf.get_j(mol, dmd, 1)
    if incremental:
        vj += vhf_last.vj
    t_j = time.perf_counter() - t0
    t0 = time.perf_counter()
    vxc = vmat_xc + vj
    ecoul = np.einsum('ij,ji', dm, vj).real * .5
    vhf = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=None)
    t_assemble = time.perf_counter() - t0
    return vhf, {'xc': t_xc, 'j': t_j, 'assemble': t_assemble, 'cpu_xc': cpu_xc}, getattr(mf, '_xc_gpu_plan', None)


def one_cycle(mf, repeat):
    mol = mf.mol
    s = mf.get_ovlp(mol)
    h = mf.get_hcore(mol)
    x = mf.check_linear_dependency(s)
    dm0 = mf.get_init_guess(mol, mf.init_guess, s1e=s)
    vhf0, _, _ = veff_parts(mf, dm0, 0, 0)
    f0 = mf.get_fock(h, s, vhf0, dm0)
    e0, c0 = mf.eig(f0, s, x=x)
    dm1 = mf.make_rdm1(c0, mf.get_occ(e0, c0))
    # Warm the identical density-dependent path; setup is already excluded.
    veff_parts(mf, dm1, dm0, vhf0)
    rows = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        f = mf.get_fock(h, s, vhf0, dm0, cycle=0, diis=None)
        t_fock0 = time.perf_counter() - t0
        t0 = time.perf_counter()
        mo_e, mo_c = mf.eig(f, s, x=x)
        mo_o = mf.get_occ(mo_e, mo_c)
        dm = mf.make_rdm1(mo_c, mo_o)
        t_orb = time.perf_counter() - t0
        vhf, parts, plan = veff_parts(mf, dm, dm0, vhf0, gpu_profile=True)
        t0 = time.perf_counter()
        etot = mf.energy_tot(dm, h, vhf)
        f1 = mf.get_fock(h, s, vhf, dm)
        mf.get_grad(mo_c, mo_o, f1)
        t_tail = time.perf_counter() - t0
        total = t_fock0 + t_orb + parts['xc'] + parts['j'] + parts['assemble'] + t_tail
        actual0 = time.perf_counter()
        actual = mf.get_veff(mol, dm, dm0, vhf0)
        actual_wall = time.perf_counter() - actual0
        rows.append(dict(fock=t_fock0, orbitals=t_orb, **parts, tail=t_tail, cycle=total,
                         actual_veff=actual_wall, veff_error=float(np.abs(actual - vhf).max()), energy=float(etot),
                         gpu_timing=dict(getattr(plan, 'last_timing', {})) if plan is not None else {}))
    return min(rows, key=lambda r: r['cycle'])


def ms(x):
    return x * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mols', nargs='+', default=['pentacene', 'PTCDA'], choices=['pentacene', 'PTCDA'])
    ap.add_argument('--basis', default='6-31g')
    ap.add_argument('--grid-level', type=int, default=2)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--repeat', type=int, default=3)
    args = ap.parse_args()
    lib.num_threads(args.threads)
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    print(f'OMP={lib.num_threads()} OPENBLAS={os.environ.get("OPENBLAS_NUM_THREADS", "unset")}', flush=True)
    for name in args.mols:
        rows = {}
        for mode in ('cpu', 'gpu_full'):
            t0 = time.perf_counter()
            mf = make_mf(name, mode, args.basis, args.grid_level)
            setup = time.perf_counter() - t0
            rows[mode] = one_cycle(mf, args.repeat)
            rows[mode]['setup'] = setup
        cpu, gpu = rows['cpu'], rows['gpu_full']
        print(f'\n{name} PBE/{args.basis} grid={args.grid_level}; static setup excluded from cycle', flush=True)
        print(f'{"stage":<18} {"CPU ms":>10} {"GPU ms":>10} {"speedup":>10} {"CPU %":>9} {"GPU %":>9}', flush=True)
        for key, label in (('fock', 'pre-Fock/DIIS'), ('orbitals', 'eig+occ+DM'), ('xc', 'XC total'), ('j', 'DF-J'), ('assemble', 'veff assembly'), ('tail', 'energy+grad'), ('cycle', 'cycle total')):
            c, g = ms(cpu[key]), ms(gpu[key])
            print(f'{label:<18} {c:10.1f} {g:10.1f} {c/g:9.2f}x {c/ms(cpu["cycle"])*100:8.1f}% {g/ms(gpu["cycle"])*100:8.1f}%', flush=True)
        print(f'  real get_veff wall: CPU={ms(cpu["actual_veff"]):.1f} GPU={ms(gpu["actual_veff"]):.1f} ms; manual parity max={gpu["veff_error"]:.2e}', flush=True)
        if cpu['cpu_xc']:
            x = cpu['cpu_xc']
            print(f'  CPU XC additive: AO={ms(x["ao"]):.1f} rho={ms(x["rho"]):.1f} libxc={ms(x["libxc"]):.1f} vmat={ms(x["vmat"]):.1f} other={ms(x["bookkeeping"]):.1f} ms', flush=True)
        tim = gpu['gpu_timing']
        if tim:
            print(f'  GPU XC diagnostic event/wall (not additive): rho={ms(tim.get("gpu_rho", 0)):.1f} ms, vmat={ms(tim.get("gpu_vmat", 0)):.1f} ms, PBE={ms(tim.get("gpu_xc_pbe", 0)):.1f} ms', flush=True)
        print(f'  one-time setup: CPU={ms(cpu["setup"]):.1f} ms GPU={ms(gpu["setup"]):.1f} ms', flush=True)


if __name__ == '__main__':
    main()
