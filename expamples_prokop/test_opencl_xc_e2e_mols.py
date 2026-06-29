#!/usr/bin/env python3
"""End-to-end XC pipeline benchmark: speed + step-wise accuracy.

Tests CPU libxc reference vs GPU paths (OTF Hermite, precomp coalesced/radial/tiled).
Reports per-stage GPU timing (queue.finish) and step decomposition (rho/wv/vmat) for precomp paths.

Usage:
  PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 \\
    python3 -u expamples_prokop/test_opencl_xc_e2e_mols.py --mols pentacene PTCDA

  PYTHONPATH=/home/prokop/git/pyscf OMP_NUM_THREADS=1 \\
    python3 -u expamples_prokop/test_opencl_xc_e2e_mols.py --mols PTCDA --basis ccpvdz --grid-level 2 --profile
"""
import argparse
import cProfile
import io
import os
import pstats
import re
import sys
import time

import numpy as np
import pyopencl as cl
import pyscf
from pyscf import dft, gto, lib
from pyscf.dft.gen_grid import BLKSIZE
from pyscf.dft.numint import _dot_ao_ao, _rks_gga_wv0, _scale_ao

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
        if not re.match(r'^[A-Z][a-z]?$', parts[0]):
            continue
        atoms.append(f'{parts[0]} {parts[1]} {parts[2]} {parts[3]}')
    return '; '.join(atoms)


def err_max_rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    amax = float(np.abs(a - b).max())
    ref = max(float(np.abs(b).max()), 1e-10)
    return amax, amax / ref


def cpu_rho(ni, mol, grids, dm):
    dm = np.asarray(dm, order='C', dtype=np.float64)
    make_rho, _, _ = ni._gen_rho_evaluator(mol, dm, hermi=1, with_lapl=False, grids=grids)
    ngrids = grids.coords.shape[0]
    rho = np.zeros((4, ngrids), dtype=np.float64)
    for ip0 in range(0, ngrids, 8192):
        ip1 = min(ip0 + 8192, ngrids)
        ao = ni.eval_ao(mol, grids.coords[ip0:ip1], deriv=1)
        rho[:, ip0:ip1] = make_rho(0, ao, None, 'GGA')
    return rho


def cpu_wv_libxc(ni, grids, rho):
    weight = grids.weights
    exc_raw, vxc_raw = ni.eval_xc('PBE', rho, deriv=1, spin=0)[:2]
    return _rks_gga_wv0(rho, vxc_raw, weight)


def cpu_vmat_gga(ni, mol, grids, wv):
    nao = mol.nao_nr()
    ngrids = grids.coords.shape[0]
    vmat = np.zeros((nao, nao), dtype=np.float64)
    for ip0 in range(0, ngrids, 8192):
        ip1 = min(ip0 + 8192, ngrids)
        ao = ni.eval_ao(mol, grids.coords[ip0:ip1], deriv=1, non0tab=grids.non0tab[ip0 // BLKSIZE:])
        wva = wv[:, ip0:ip1].astype(np.float64)
        aow = _scale_ao(ao[:4], wva[:4])
        vmat += _dot_ao_ao(mol, ao[0], aow, grids.non0tab[ip0 // BLKSIZE:], (0, mol.nbas), mol.ao_loc_nr())
    return vmat + vmat.T


def print_stage_timing(tim, indent='    '):
    from pyscf.OpenCL.xc_grid import TIMING_STAGE_ORDER
    if not tim:
        return
    log(f'{indent}--- stages (ms) ---')
    for k in TIMING_STAGE_ORDER:
        if k not in tim or tim[k] <= 0:
            continue
        if k == 'n_blocks':
            log(f'{indent}  {k:22s} {int(tim[k]):8d}')
        else:
            log(f'{indent}  {k:22s} {tim[k]*1e3:8.1f}')


def gpu_precomp_rho_wv(plan, dm):
    pcg = plan.pcg
    ngrids = plan.ngrids
    dm32 = np.ascontiguousarray(dm, dtype=np.float32)
    cl.enqueue_copy(plan.queue, plan.bufDm, dm32)
    if pcg.get('radial_precomp'):
        np.matmul(pcg['c2s'], dm32, out=pcg['dm_tmp'])
        np.matmul(pcg['dm_tmp'], pcg['c2s'].T, out=pcg['dm_cart32'])
        cl.enqueue_copy(plan.queue, pcg['buf_dm_cart'], pcg['dm_cart32'])
    plan._precomp_rho_fused(pcg, 'GGA', plan.nao, ngrids)
    rho = np.empty(4 * ngrids, dtype=np.float32)
    cl.enqueue_copy(plan.queue, rho, pcg['buf_rho']).wait()
    rho = rho.reshape(4, ngrids).astype(np.float64)
    st = {**pcg, 'rho_host': pcg['rho32_host'], 'wv_host': pcg['wv32_host'],
          'weight32': pcg['weight'], 'weight64': pcg['weight64']}
    plan._xc_pbe_gpu(st, ngrids)
    wv = np.empty(4 * ngrids, dtype=np.float32)
    cl.enqueue_copy(plan.queue, wv, pcg['buf_wv']).wait()
    return rho, wv.reshape(4, ngrids).astype(np.float64)


def gpu_precomp_wv_from_rho(plan, rho64):
    pcg = plan.pcg
    ngrids = plan.ngrids
    rho32 = np.ascontiguousarray(rho64, dtype=np.float32).reshape(-1)
    cl.enqueue_copy(plan.queue, pcg['buf_rho'], rho32).wait()
    st = {**pcg, 'rho_host': pcg['rho32_host'], 'wv_host': pcg['wv32_host'],
          'weight32': pcg['weight'], 'weight64': pcg['weight64']}
    plan._xc_pbe_gpu(st, ngrids)
    wv = np.empty(4 * ngrids, dtype=np.float32)
    cl.enqueue_copy(plan.queue, wv, pcg['buf_wv']).wait()
    return wv.reshape(4, ngrids).astype(np.float64)


def audit_precomp_steps(plan, dm, rho_cpu, wv_cpu, vmat_cpu, vxc_cpu):
    rho_gpu, wv_gpu = gpu_precomp_rho_wv(plan, dm)
    steps = {}
    for key, a, b, names in (
        ('rho', rho_gpu, rho_cpu, ('rho0', 'gx', 'gy', 'gz')),
        ('wv2a', gpu_precomp_wv_from_rho(plan, rho_cpu), wv_cpu, ('wv0', 'wv1', 'wv2', 'wv3')),
        ('wv2b', wv_gpu, wv_cpu, ('wv0', 'wv1', 'wv2', 'wv3')),
    ):
        d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
        row = {'max_abs': float(np.abs(d).max())}
        row['comps'] = {}
        for c, n in enumerate(names):
            mc = float(np.abs(d[c]).max())
            rc = max(float(np.abs(b[c]).max()), 1e-10)
            row['comps'][n] = {'max_abs': mc, 'rel': mc / rc}
        steps[key] = row
    vmat_a = plan.nr_rks_precomputed_vmat_only(np.ascontiguousarray(wv_cpu, np.float32))
    vmat_b = plan.nr_rks_precomputed_vmat_only(np.ascontiguousarray(wv_gpu, np.float32))
    steps['vmat3a'] = dict(zip(('max_abs', 'rel'), err_max_rel(vmat_a, vmat_cpu)))
    steps['vmat3b'] = dict(zip(('max_abs', 'rel'), err_max_rel(vmat_b, vmat_cpu)))
    n, exc, vxc = plan.nr_rks_precomputed_gto(dm, projection='gpu')
    steps['full4'] = dict(zip(('max_abs', 'rel'), err_max_rel(vxc, vxc_cpu)))
    steps['nelec'] = float(n)
    steps['exc'] = float(exc)
    return steps


def bench_path(label, setup_fn, run_fn, ref_vxc, ref_exc, n_warm=1, n_timed=3):
    from pyscf.OpenCL import init_device, reset_opencl
    from pyscf.OpenCL.xc_grid import clear_xc_plan_cache

    clear_xc_plan_cache()
    reset_opencl()
    init_device(quiet=True)
    t0 = time.perf_counter()
    plan = setup_fn()
    setup_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(n_warm):
        run_fn(plan, profile=False)
    times, last_tim, last_out = [], {}, None
    for _ in range(n_timed):
        t0 = time.perf_counter()
        last_out = run_fn(plan, profile=True)
        times.append(time.perf_counter() - t0)
        last_tim = plan.last_timing or {}
    wall_ms = min(times) * 1e3
    n, exc, vxc = last_out
    vxc_err, vxc_rel = err_max_rel(vxc, ref_vxc)
    exc_err = abs(exc - ref_exc) / max(abs(ref_exc), 1e-10)
    return dict(label=label, plan=plan, setup_ms=setup_ms, wall_ms=wall_ms, timing=last_tim,
                nelec=n, exc=exc, vxc_err=vxc_err, vxc_rel=vxc_rel, exc_err=exc_err)


def build_cpu_reference(mol, grids, dm, ni, step_audit, do_profile):
    ref = {}

    def _nr_rks():
        return ni.nr_rks(mol, grids, 'PBE', dm, max_memory=4000)

    t0 = time.perf_counter()
    if do_profile:
        pr = cProfile.Profile()
        pr.enable()
        n, exc, vxc = _nr_rks()
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(20)
        ref['cprofile_nr_rks'] = s.getvalue()
    else:
        n, exc, vxc = _nr_rks()
    ref['nr_rks_ms'] = (time.perf_counter() - t0) * 1e3
    ref['n'], ref['exc'], ref['vxc'] = n, exc, vxc
    log(f'  cpu nr_rks: {ref["nr_rks_ms"]:.1f} ms  nelec={n:.6f} exc={exc:.8f}')

    if not step_audit:
        return ref

    t0 = time.perf_counter()
    ref['rho'] = cpu_rho(ni, mol, grids, dm)
    ref['rho_ms'] = (time.perf_counter() - t0) * 1e3
    log(f'  cpu rho:    {ref["rho_ms"]:.1f} ms')

    t0 = time.perf_counter()
    ref['wv'] = cpu_wv_libxc(ni, grids, ref['rho'])
    ref['wv_ms'] = (time.perf_counter() - t0) * 1e3
    log(f'  cpu wv:     {ref["wv_ms"]:.1f} ms')

    t0 = time.perf_counter()
    ref['vmat'] = cpu_vmat_gga(ni, mol, grids, ref['wv'])
    ref['vmat_ms'] = (time.perf_counter() - t0) * 1e3
    log(f'  cpu vmat:   {ref["vmat_ms"]:.1f} ms')
    ref['vmat_err'] = err_max_rel(ref['vmat'], vxc)[0]
    log(f'  cpu vmat vs nr_rks vxc: max_abs={ref["vmat_err"]:.3e}')
    return ref


def run_molecule(mol_name, xyz_path, basis, grid_level, n_timed, step_audit, skip_gto_ao, do_profile):
    from pyscf.OpenCL.xc_grid import get_xc_grid_plan

    log(f'\n{"="*80}')
    log(f'Molecule: {mol_name}  basis={basis}  grid={grid_level}')
    mol = gto.M(atom=read_xyz(xyz_path), basis=basis, verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.level = grid_level
    grids.build(with_non0tab=True)
    nao, ngrids, natm = mol.nao_nr(), grids.coords.shape[0], mol.natm
    dm = dft.RKS(mol, xc='PBE').density_fit().get_init_guess()
    ni = dft.numint.NumInt()
    chi_mb = 4 * 4 * nao * ngrids * 4 / 1e6  # GGA deriv1 f32, 4 components
    log(f'  natm={natm}  nao={nao}  ngrids={ngrids}  chi_GGA_f32~{chi_mb:.0f} MB')
    log(f'  OMP={lib.num_threads()}  PySCF={pyscf.__version__}')

    log('\n--- CPU reference ---')
    ref = build_cpu_reference(mol, grids, dm, ni, step_audit, do_profile)
    if do_profile and 'cprofile_nr_rks' in ref:
        log(ref['cprofile_nr_rks'])

    paths = [
        ('gpu_hermite_otf', 'otf', dict(xc_eval='gpu')),
        ('gpu_hermite_otf_libxc', 'otf', dict(xc_eval='cpu')),
        ('gpu_precomp_radial_hermite', 'precomp', dict(fused='radial_precomp', xc_eval='gpu', ao_proj='hermite_gpu', gpu_xc='pbe_f32')),
        ('gpu_precomp_coalesced_hermite', 'precomp', dict(fused='coalesced', xc_eval='gpu', ao_proj='hermite_gpu', gpu_xc='pbe_f32')),
        ('gpu_precomp_coalesced_auto', 'precomp', dict(fused='coalesced', xc_eval='gpu', gpu_xc='pbe_f32')),
        ('gpu_precomp_tiled_auto', 'precomp', dict(fused='tiled', xc_eval='gpu', gpu_xc='pbe_f32')),
    ]
    if not skip_gto_ao and chi_mb < 4000:
        paths.append(('gpu_precomp_coalesced_gto', 'precomp',
                      dict(fused='coalesced', xc_eval='gpu', ao_proj='cpu', gpu_xc='pbe_f32')))
    elif not skip_gto_ao:
        log(f'  skip gpu_precomp_coalesced_gto: chi ~{chi_mb:.0f} MB > 4000 MB budget')

    results = []
    cpu_ms = ref['nr_rks_ms']

    for label, kind, kw in paths:
        log(f'\n--- {label} ---')

        def setup(kw=kw):
            p = get_xc_grid_plan(mol, grids, 'PBE')
            if kind == 'otf':
                p.setup_onthefly(**{k: v for k, v in kw.items() if k in ('xc_eval', 'gpu_xc')})
            else:
                p.setup_precomputed_gto(gpu_only=True, **kw)
            return p

        def run(plan, profile):
            if kind == 'otf':
                return plan.nr_rks_hermite_onthefly(dm, profile=profile)
            return plan.nr_rks_precomputed_gto(dm, projection='gpu', profile=profile)

        try:
            row = bench_path(label, setup, run, ref['vxc'], ref['exc'], n_timed=n_timed)
        except Exception as e:
            log(f'  FAILED: {e}')
            results.append(dict(label=label, error=str(e)))
            continue

        pt = getattr(row['plan'], 'precalc_timing', None) or {}
        spd = cpu_ms / row['wall_ms'] if row['wall_ms'] > 0 else 0
        log(f'  setup={row["setup_ms"]:.1f} ms  wall={row["wall_ms"]:.1f} ms  vs_cpu={spd:.2f}x')
        log(f'  vxc max_abs={row["vxc_err"]:.3e} rel={row["vxc_rel"]:.3e}  exc_rel={row["exc_err"]:.3e}')
        if pt:
            ao = pt.get('eval_ao_hermite_gpu') or pt.get('eval_ao_cpu')
            if ao:
                log(f'  setup AO: {ao*1e3:.1f} ms ({pt.get("ao_proj", "?")})')
            if pt.get('radial_gpu'):
                log(f'  setup radial_gpu: {pt["radial_gpu"]*1e3:.1f} ms')
        print_stage_timing(row['timing'])

        if kind == 'precomp' and step_audit:
            log('  step audit:')
            steps = audit_precomp_steps(row['plan'], dm, ref['rho'], ref['wv'], ref['vmat'], ref['vxc'])
            row['steps'] = steps
            for sk in ('rho', 'wv2a', 'wv2b', 'vmat3a', 'vmat3b', 'full4'):
                s = steps[sk]
                if 'comps' in s:
                    worst = max(s['comps'].values(), key=lambda x: x['max_abs'])
                    log(f'    {sk:8s} max_abs={s["max_abs"]:.3e}  worst_comp rel={worst["rel"]:.3e}')
                else:
                    log(f'    {sk:8s} max_abs={s["max_abs"]:.3e} rel={s["rel"]:.3e}')
        row.pop('plan', None)
        results.append(row)

    log(f'\n{"="*80}')
    log(f'SUMMARY {mol_name}  (cpu={cpu_ms:.1f} ms)')
    log(f'{"path":<30} {"setup":>8} {"wall":>8} {"rho":>7} {"xc":>7} {"vmat":>7} {"vs_cpu":>7} {"vxc_err":>10}')
    log('-' * 95)
    for r in results:
        if 'error' in r:
            log(f'{r["label"]:<30} FAILED: {r["error"]}')
            continue
        tim = r.get('timing') or {}
        rho = tim.get('gpu_rho', 0) * 1e3
        xc = (tim.get('gpu_xc_pbe', 0) or tim.get('host_xc_libxc', 0)) * 1e3
        vmat = tim.get('gpu_vmat', 0) * 1e3
        spd = cpu_ms / r['wall_ms'] if r['wall_ms'] > 0 else 0
        log(f'{r["label"]:<30} {r["setup_ms"]:8.1f} {r["wall_ms"]:8.1f} {rho:7.1f} {xc:7.1f} {vmat:7.1f} {spd:7.2f}x {r["vxc_err"]:10.3e}')
    if step_audit:
        log('\nCPU step refs (ms):')
        log(f'  rho={ref.get("rho_ms",0):.1f}  wv={ref.get("wv_ms",0):.1f}  vmat={ref.get("vmat_ms",0):.1f}')
    return dict(mol=mol_name, nao=nao, ngrids=ngrids, cpu_ms=cpu_ms, ref=ref, results=results)


def parse_args():
    ap = argparse.ArgumentParser(description='E2E XC pipeline benchmark (speed + accuracy)')
    ap.add_argument('--mols', nargs='+', default=['pentacene', 'PTCDA'])
    ap.add_argument('--xyz-dir', default=_XYZ_DIR)
    ap.add_argument('--basis', default='6-31g')
    ap.add_argument('--grid-level', type=int, default=2)
    ap.add_argument('--n-timed', type=int, default=3)
    ap.add_argument('--no-step-audit', action='store_true', help='Skip per-step rho/wv/vmat parity (faster)')
    ap.add_argument('--skip-gto-ao', action='store_true', help='Skip ao_proj=cpu path')
    ap.add_argument('--profile', action='store_true', help='cProfile CPU nr_rks only')
    return ap.parse_args()


def main():
    args = parse_args()
    all_out = []
    for name in args.mols:
        xyz = os.path.join(args.xyz_dir, f'{name}.xyz')
        if not os.path.isfile(xyz):
            log(f'SKIP {name}: missing {xyz}')
            continue
        out = run_molecule(name, xyz, args.basis, args.grid_level, args.n_timed,
                           step_audit=not args.no_step_audit, skip_gto_ao=args.skip_gto_ao,
                           do_profile=args.profile)
        all_out.append(out)
    log('\nAll done.')
    return all_out


if __name__ == '__main__':
    main()
