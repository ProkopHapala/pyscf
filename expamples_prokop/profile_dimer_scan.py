#!/usr/bin/env python3
'''profile_dimer_scan.py — inter-fragment distance scan benchmarking all OpenCL XC paths vs CPU.

Motivation: single-point ΔE bar charts miss scan-dependent errors (grid screening, warm-start,
χ memory leaks). Running PBE/DF SCF along a rigid dimer separation curve is the acceptance
test for non-covalent GPU parity.

Design:
- **n0** splits the dimer: fragment 1 fixed [0,n0), fragment 2 mobile [n0,natom) — only input
  needed for arbitrary molecules.
- dm warm-start between frames mimics production geometry optimization; GPU plans released each
  frame to avoid χ cache OOM on precomp paths.
- Timers wrap scf.kernel / get_veff / get_jk / GPU ρ-PBE-vmat stages; CSV is SSOT for plots.

Outputs: debug/profile_<name>_scan/scan_scf_profile.csv, energy_profile_ez.png (primary).
See /home/prokophapala/git/pyscf/doc/dimer_scan_benchmarks.md.
'''
import argparse
import csv
import os
import re
import sys
import time

import numpy as np
import pyscf
from pyscf import dft, gto, lib

from xc_path_modes import XC_PATH_MODES, SCAN_MODE_KEYS, mode_label
from dimer_scan_frames import load_distances_file, frames_from_relaxed_xyz

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
HA_TO_KCAL = 627.5094740631
EV_TO_KCAL = HA_TO_KCAL / 27.211386245988

_timers = {}


def log(msg):
    print(msg, flush=True)


def default_outdir(geom=None, scan_xyz=None):
    if geom:
        name = os.path.splitext(os.path.basename(geom))[0]
        return os.path.join(_REPO, 'debug', f'profile_{name}_scan')
    if scan_xyz:
        name = os.path.splitext(os.path.basename(scan_xyz))[0]
        if 'scan' in name.lower():
            name = name.replace('_scan_out', '').replace('scan_out', 'dimer')
        return os.path.join(_REPO, 'debug', f'profile_{name}_scan')
    return os.path.join(_REPO, 'debug', 'profile_dimer_scan')


def iter_xyz_frames(path):
    with open(path) as f:
        lines = f.readlines()
    i = 0
    nframes = 0
    while i < len(lines):
        natom = int(lines[i].strip())
        comment = lines[i + 1].strip()
        atoms = []
        for j in range(natom):
            parts = lines[i + 2 + j].split()
            if not re.match(r'^[A-Z][a-z]?$', parts[0]):
                continue
            atoms.append(f'{parts[0]} {parts[1]} {parts[2]} {parts[3]}')
        if len(atoms) != natom:
            raise ValueError(f'frame {nframes}: expected {natom} atoms, parsed {len(atoms)}')
        yield comment, '; '.join(atoms)
        i += 2 + natom
        nframes += 1


def parse_r_from_comment(comment):
    m = re.search(r'r=([0-9.+-eE]+)', comment)
    return float(m.group(1)) if m else float('nan')


def select_frames_at_distances(all_frames, target_rs, tol=0.08):
    src_rs = np.array([parse_r_from_comment(c) for c, _ in all_frames])
    out = []
    for tr in target_rs:
        j = int(np.argmin(np.abs(src_rs - tr)))
        dr = abs(src_rs[j] - tr)
        if dr > tol:
            raise ValueError(f'target r={tr:.4f} Å: nearest frame r={src_rs[j]:.4f} Å (Δ={dr:.4f} > tol={tol})')
        comment, atom = all_frames[j]
        out.append((f'r={tr:.4f}', atom, float(tr), float(src_rs[j])))
    return out


def _wrap(label, fn):
    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        rec = _timers.setdefault(label, {'calls': 0, 'wall': 0.0})
        rec['calls'] += 1
        rec['wall'] += time.perf_counter() - t0
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
    _orig_build_grids = rks_mod.KohnShamDFT.initialize_grids
    rks_mod.KohnShamDFT.initialize_grids = lambda self, *a, **kw: _wrap('rks.init_grids', _orig_build_grids)(self, *a, **kw)


def timer_ms(label):
    rec = _timers.get(label)
    return rec['wall'] * 1e3 if rec else 0.0


def reset_timers():
    _timers.clear()


def gpu_xc_stage_ms(mf, key):
    acc = mf.__dict__.get('_gpu_timing_acc') or {}
    return acc.get(key, 0.0) * 1e3


def build_mf(atom, basis, grid_level, conv_tol, use_df):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = dft.RKS(mol, xc='PBE')
    mf.grids.level = grid_level
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = 1e-5
    mf.max_cycle = 50
    mf.chkfile = None
    if use_df:
        mf = mf.density_fit()
    return mf


def release_gpu_between_frames(mf=None):
    if mf is not None:
        plan = mf.__dict__.pop('_xc_gpu_plan', None)
        if plan is not None:
            try:
                plan.release()
            except Exception:
                pass
    try:
        from pyscf.OpenCL.xc_grid import clear_xc_plan_cache
        clear_xc_plan_cache()
        from pyscf.OpenCL import get_queue
        get_queue().finish()
    except Exception:
        pass


def configure_mode(mf, mode, dm0=None):
    mf.__dict__.pop('_xc_gpu_plan', None)
    mf.__dict__.pop('_gpu_timing_acc', None)
    mf.__dict__.pop('_gpu_profile', None)
    if mode == 'cpu':
        mf.backend = 1
        if 'with_df' in mf.__dict__ and mf.with_df is not None:
            mf.with_df.backend = 1
        return mf, 0.0
    release_gpu_between_frames()
    from pyscf.OpenCL.gpu_profiles import apply_gpu_profile
    t0 = time.perf_counter()
    prof_name = XC_PATH_MODES[mode]['profile']
    apply_gpu_profile(mf, prof_name, setup=True, dm=dm0)
    mf._gpu_profile = True
    setup_s = time.perf_counter() - t0
    return mf, setup_s


def run_scf(mf, dm0):
    reset_timers()
    mf.__dict__['_gpu_timing_acc'] = {}
    t0 = time.perf_counter()
    e = mf.kernel(dm0=dm0)
    wall_s = time.perf_counter() - t0
    n_cycles = getattr(mf, 'cycles', 0) or 0
    mf.__dict__['_gpu_profile_cycles'] = max(n_cycles, 1)
    dm = mf.make_rdm1() if mf.mo_coeff is not None else None
    return dict(energy=e, wall_s=wall_s, converged=bool(mf.converged), cycles=n_cycles, dm=dm)


def run_mode_scan(frames, mode, args):
    rows = []
    dm_prev = None
    log(f'\n{"="*72}\nMODE: {mode} ({mode_label(mode)})  ({len(frames)} frames)\n{"="*72}')
    for iframe, frame in enumerate(frames):
        comment, atom, r_A, r_src = frame
        warm = dm_prev is not None and not args.cold_each_frame
        log(f'\n--- frame {iframe}  r={r_A:.4f} Å  (xyz r={r_src:.4f})  warm_start={warm} ---')
        mf = build_mf(atom, args.basis, args.grid_level, args.conv_tol, not args.no_df)
        mf, setup_s = configure_mode(mf, mode, dm0=dm_prev if warm else None)
        out = run_scf(mf, dm_prev if warm else None)
        if out['dm'] is not None:
            dm_prev = out['dm']
        row = dict(
            iframe=iframe, r_A=r_A, mode=mode, warm_start=int(warm),
            E_Ha=out['energy'], converged=int(out['converged']), cycles=out['cycles'],
            setup_ms=setup_s * 1e3, scf_ms=out['wall_s'] * 1e3,
            wall_ms=(setup_s + out['wall_s']) * 1e3,
            scf_kernel_ms=timer_ms('scf.kernel'),
            get_veff_ms=timer_ms('rks.get_veff'),
            get_jk_ms=timer_ms('scf.get_jk') + timer_ms('df.get_jk'),
            eig_ms=timer_ms('scf.eig'),
            init_grids_ms=timer_ms('rks.init_grids'),
            gpu_rho_ms=gpu_xc_stage_ms(mf, 'gpu_rho'),
            gpu_xc_pbe_ms=gpu_xc_stage_ms(mf, 'gpu_xc_pbe'),
            gpu_vmat_ms=gpu_xc_stage_ms(mf, 'gpu_vmat'),
            nao=mf.mol.nao_nr(), ngrids=mf.grids.coords.shape[0] if mf.grids.coords is not None else 0,
        )
        rows.append(row)
        log(f'  conv={out["converged"]}  cycles={out["cycles"]}  E={out["energy"]*HA_TO_KCAL:.6f} kcal/mol')
        log(f'  setup={setup_s*1e3:.0f} ms  scf={out["wall_s"]*1e3:.0f} ms  total={(setup_s+out["wall_s"])*1e3:.0f} ms')
        log(f'  get_veff={row["get_veff_ms"]:.0f} ms  get_jk={row["get_jk_ms"]:.0f} ms  eig={row["eig_ms"]:.0f} ms')
        if mode != 'cpu':
            log(f'  gpu: rho={row["gpu_rho_ms"]:.0f}  pbe={row["gpu_xc_pbe_ms"]:.0f}  vmat={row["gpu_vmat_ms"]:.0f} ms')
        release_gpu_between_frames(mf)
    release_gpu_between_frames()
    return rows


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def print_summary(all_rows, modes):
    log(f'\n{"="*72}\nSUMMARY\n{"="*72}')
    by_mode = {m: [r for r in all_rows if r['mode'] == m] for m in modes}
    log(f'{"mode":<14} {"frames":>6} {"avg_cyc":>8} {"avg_scf_ms":>10} {"avg_setup_ms":>12} {"avg_total_ms":>11} {"conv%":>6}')
    for mode in modes:
        rs = by_mode[mode]
        if not rs:
            continue
        avg_cyc = np.mean([r['cycles'] for r in rs])
        avg_scf = np.mean([r['scf_ms'] for r in rs])
        avg_setup = np.mean([r['setup_ms'] for r in rs])
        avg_tot = np.mean([r['wall_ms'] for r in rs])
        conv_pct = 100.0 * np.mean([r['converged'] for r in rs])
        log(f'{mode:<14} {len(rs):>6} {avg_cyc:>8.1f} {avg_scf:>10.0f} {avg_setup:>12.0f} {avg_tot:>11.0f} {conv_pct:>5.0f}%')
    warm = [r for r in all_rows if r['warm_start']]
    cold = [r for r in all_rows if not r['warm_start']]
    if warm and cold:
        log(f'\nWarm-start effect (all modes): cold avg cycles={np.mean([r["cycles"] for r in cold]):.1f}  warm avg cycles={np.mean([r["cycles"] for r in warm]):.1f}')
    if 'cpu' in modes and len(modes) > 1:
        cpu_by_iframe = {r['iframe']: r for r in all_rows if r['mode'] == 'cpu'}
        for mode in modes:
            if mode == 'cpu':
                continue
            gpu_rows = [r for r in all_rows if r['mode'] == mode]
            scf_ratios = []
            tot_ratios = []
            for gr in gpu_rows:
                cr = cpu_by_iframe.get(gr['iframe'])
                if cr and cr['scf_ms'] > 0:
                    scf_ratios.append(cr['scf_ms'] / gr['scf_ms'])
                    tot_ratios.append(cr['wall_ms'] / gr['wall_ms'])
            if scf_ratios:
                log(f'  {mode} vs cpu: scf speedup {np.mean(scf_ratios):.2f}x  (setup+scf) {np.mean(tot_ratios):.2f}x')


def build_frames(args):
    if args.geom:
        if not os.path.isfile(args.geom):
            raise FileNotFoundError(args.geom)
        if args.n0 is None:
            raise ValueError('--n0 required with --geom (0-based index of first atom in fragment 2)')
        if not args.distances_file or not os.path.isfile(args.distances_file):
            raise FileNotFoundError(args.distances_file)
        target_rs = load_distances_file(args.distances_file)
        if args.max_frames is not None:
            target_rs = target_rs[:args.max_frames]
        raw, meta = frames_from_relaxed_xyz(args.geom, target_rs, args.n0, args.anchor_fixed, args.anchor_mobile, prefer_o=not args.no_prefer_o)
        frames = [(f'r={r:.4f}', atom, r, r) for r, atom, *_ in raw]
        dr = np.diff([f[2] for f in frames])
        log(f'geom={args.geom}  n0={meta["n0"]}  anchor={meta["anchor_pair"]}  r0={meta["r0"]:.4f} Å')
        log(f'distances={args.distances_file}  n={len(frames)}  dr=[{", ".join(f"{d:.2f}" for d in dr[:8])} … {", ".join(f"{d:.1f}" for d in dr[-3:])}] Å')
        return frames
    if not args.scan_xyz:
        raise ValueError('set --geom or --scan-xyz')
    if not os.path.isfile(args.scan_xyz):
        raise FileNotFoundError(args.scan_xyz)
    all_xyz = list(iter_xyz_frames(args.scan_xyz))
    if args.distances_file:
        if not os.path.isfile(args.distances_file):
            raise FileNotFoundError(args.distances_file)
        target_rs = load_distances_file(args.distances_file)
        if args.stride != 1 or args.start_frame != 0:
            log('note: --stride/--start-frame ignored when --distances-file is set')
        if args.max_frames is not None:
            target_rs = target_rs[:args.max_frames]
        frames = select_frames_at_distances(all_xyz, target_rs, tol=args.match_tol)
        dr = np.diff([f[2] for f in frames])
        log(f'scan-xyz={args.scan_xyz}  distances={args.distances_file}  n={len(frames)}  dr=[{", ".join(f"{d:.2f}" for d in dr[:8])} … {", ".join(f"{d:.1f}" for d in dr[-3:])}] Å')
    else:
        frames = [(c, a, parse_r_from_comment(c), parse_r_from_comment(c)) for c, a in all_xyz]
        frames = frames[args.start_frame::args.stride]
        if args.max_frames is not None:
            frames = frames[:args.max_frames]
        log(f'scan-xyz={args.scan_xyz}  n={len(frames)} (native frame spacing)')
    return frames


def parse_args():
    ap = argparse.ArgumentParser(description='General dimer distance scan — CPU + GPU XC paths, E(z) plot')
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--geom', default=None, help='Single relaxed dimer XYZ; rigid shift with --distances-file (requires --n0)')
    src.add_argument('--scan-xyz', default=None, help='Pre-built multi-frame XYZ scan')
    ap.add_argument('--n0', type=int, default=None, help='0-based index of first atom in fragment 2 (mobile monomer); required for --geom')
    ap.add_argument('--anchor-fixed', type=int, default=None, help='Anchor atom in fragment 1 [0, n0); default: auto closest cross-fragment pair')
    ap.add_argument('--anchor-mobile', type=int, default=None, help='Anchor atom in fragment 2 [n0, natom); default: auto')
    ap.add_argument('--no-prefer-o', action='store_true', help='Auto anchor: use closest heavy-atom pair, not O···O')
    ap.add_argument('--distances-file', default=None, help='Target distance grid (Å); resample scan-xyz or drive --geom rigid shift')
    ap.add_argument('--ref-scan', default=None, help='Reference scan.dat for E(z) overlay (column 0=z, col 2=E_bind eV)')
    ap.add_argument('--title', default=None, help='Plot title (default: from xyz basename)')
    ap.add_argument('--z-label', default=None, help='X-axis label (default: inter-fragment distance (Å))')
    ap.add_argument('--match-tol', type=float, default=0.08, help='Max |r_frame − r_target| when pairing scan-xyz to distances')
    ap.add_argument('--mode', nargs='+', default=SCAN_MODE_KEYS, choices=list(XC_PATH_MODES))
    ap.add_argument('--basis', default='6-31g')
    ap.add_argument('--grid-level', type=int, default=3)
    ap.add_argument('--conv-tol', type=float, default=1e-8)
    ap.add_argument('--no-df', action='store_true')
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--start-frame', type=int, default=0)
    ap.add_argument('--cold-each-frame', action='store_true')
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--no-plot-diag', action='store_true', help='Skip 4-panel diagnostic plot (H2O-style)')
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--out-csv', default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.threads is not None:
        lib.num_threads(args.threads)
        os.environ['OMP_NUM_THREADS'] = str(args.threads)

    frames = build_frames(args)
    if not frames:
        raise ValueError('No frames selected')

    if args.outdir is None:
        args.outdir = default_outdir(args.geom, args.scan_xyz)
    if args.title is None:
        src = args.geom or args.scan_xyz
        args.title = os.path.splitext(os.path.basename(src))[0].replace('_', ' ')
    z_label = args.z_label or 'inter-fragment distance (Å)'

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = args.out_csv or os.path.join(args.outdir, 'scan_scf_profile.csv')

    log(f'PySCF {pyscf.__version__}  OMP_NUM_THREADS={lib.num_threads()}')
    log(f'frames={len(frames)}  basis={args.basis}  grid={args.grid_level}  DF={not args.no_df}  n0={args.n0}')
    log(f'modes={args.mode}  warm_start={not args.cold_each_frame}  outdir={args.outdir}')

    install_timers()
    if any(m != 'cpu' for m in args.mode):
        from pyscf.OpenCL import init_device
        init_device(quiet=True)

    all_rows = []
    for mode in args.mode:
        all_rows.extend(run_mode_scan(frames, mode, args))

    write_csv(csv_path, all_rows)
    print_summary(all_rows, args.mode)
    log(f'\nWrote {csv_path}')
    if not args.no_plot:
        from plot_scan_ez import plot_ez, load_csv as _load_csv, group_by_mode as _group, load_ref_scan
        dft = _group(_load_csv(csv_path))
        plot_modes = [m for m in args.mode if m in dft]
        ref_z = ref_bind = None
        if args.ref_scan and os.path.isfile(args.ref_scan):
            ref_z, ref_ev = load_ref_scan(args.ref_scan)
            ref_bind = ref_ev * EV_TO_KCAL
        ez_png = os.path.join(args.outdir, 'energy_profile_ez.png')
        plot_ez(dft, plot_modes, ez_png, z_label=z_label, title=args.title, ref_z=ref_z, ref_bind_kcal=ref_bind)
        log(f'Wrote {ez_png}')
        if not args.no_plot_diag and args.ref_scan:
            from plot_h2o_dimer_scan_energy import main as plot_main
            import sys as _sys
            _argv = _sys.argv
            _sys.argv = ['plot_h2o_dimer_scan_energy.py', '--csv', csv_path, '--outdir', args.outdir, '--ref-scan', args.ref_scan, '--title', args.title, '--z-label', z_label, '--modes', *plot_modes]
            try:
                plot_main()
            finally:
                _sys.argv = _argv


if __name__ == '__main__':
    main()
