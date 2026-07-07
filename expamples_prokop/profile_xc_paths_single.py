#!/usr/bin/env python3
'''Single-geometry SCF: compare all OpenCL XC paths vs CPU reference.

Reports E_tot and ΔE in kcal/mol per path; timing breakdown; writes plot immediately.

Usage:
  PYTHONPATH=/home/prokophapala/git/pyscf OMP_NUM_THREADS=1 python3 -u \\
    expamples_prokop/profile_xc_paths_single.py --xyz data/xyz/formic_dimer.xyz
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

from xc_path_modes import XC_PATH_MODES, SCAN_MODE_KEYS, MODE_COLORS, mode_label, mode_plot_label, path_description, gpu_full_vs_otf_note

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
HA_TO_KCAL = 627.5094740631
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
        if not re.match(r'^[A-Z][a-z]?$', parts[0]):
            continue
        atoms.append(f'{parts[0]} {parts[1]} {parts[2]} {parts[3]}')
    return '; '.join(atoms)


def _wrap(label, fn):
    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        rec = _timers.setdefault(label, {'calls': 0, 'wall': 0.0})
        rec['calls'] += 1
        rec['wall'] += time.perf_counter() - t0
        return out
    return wrapped


def install_timers():
    from pyscf.scf import hf as hf_mod
    from pyscf.dft import rks as rks_mod
    hf_mod.kernel = _wrap('scf.kernel', hf_mod.kernel)
    rks_mod.get_veff = _wrap('rks.get_veff', rks_mod.get_veff)
    rks_mod.RKS.get_veff = rks_mod.get_veff
    _orig = hf_mod.SCF.get_jk
    hf_mod.SCF.get_jk = lambda self, *a, **kw: _wrap('scf.get_jk', _orig)(self, *a, **kw)
    from pyscf.df import df_jk as dfjk_mod
    dfjk_mod.get_jk = _wrap('df.get_jk', dfjk_mod.get_jk)


def timer_ms(k):
    rec = _timers.get(k)
    return rec['wall'] * 1e3 if rec else 0.0


def build_mf(atom, basis, grid_level, use_df):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = dft.RKS(mol, xc='PBE')
    mf.grids.level = grid_level
    mf.max_cycle = 50
    mf.chkfile = None
    if use_df:
        mf = mf.density_fit()
    return mf


def run_mode(atom, mode, basis, grid_level, use_df, dm0=None):
    _timers.clear()
    mf = build_mf(atom, basis, grid_level, use_df)
    setup_s = 0.0
    mf.__dict__.pop('_xc_gpu_plan', None)
    if mode == 'cpu':
        mf.backend = 1
        if mf.with_df is not None:
            mf.with_df.backend = 1
        from pyscf.OpenCL.gpu_profiles import apply_scf_kw, get_profile
        apply_scf_kw(mf, get_profile('cpu_reference')['scf_kw'])
    else:
        from pyscf.OpenCL.gpu_profiles import apply_gpu_profile
        t0 = time.perf_counter()
        apply_gpu_profile(mf, XC_PATH_MODES[mode]['profile'], setup=True, dm=dm0)
        mf._gpu_profile = True
        setup_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    e = mf.kernel(dm0=dm0)
    scf_s = time.perf_counter() - t0
    return dict(mf=mf, E_Ha=e, E_kcal=e * HA_TO_KCAL, converged=mf.converged, cycles=mf.cycles, setup_ms=setup_s * 1e3, scf_ms=scf_s * 1e3, get_veff_ms=timer_ms('rks.get_veff'), get_jk_ms=timer_ms('scf.get_jk') + timer_ms('df.get_jk'))


def plot_paths(rows, out_png, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    modes = [r['mode'] for r in rows]
    labels = [mode_plot_label(m) for m in modes]
    dE = [r['dE_kcal_vs_cpu'] for r in rows]
    scf = [r['scf_ms'] for r in rows]
    setup = [r['setup_ms'] for r in rows]
    colors = [MODE_COLORS.get(m, '#333333') for m in modes]

    fig, (ax_de, ax_time) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(modes))
    ax_de.bar(x, dE, color=colors, edgecolor='k', lw=0.4)
    ax_de.axhline(0, color='k', lw=0.6)
    ax_de.set_xticks(x)
    ax_de.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax_de.set_ylabel('ΔE_tot vs CPU (kcal/mol)')
    ax_de.set_title('Energy deviation by XC path')
    ax_de.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(dE):
        if modes[i] != 'cpu':
            ax_de.text(i, v, f'{v:+.4f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=7)

    w = 0.35
    ax_time.bar(x - w/2, scf, w, label='SCF', color=colors, alpha=0.85, edgecolor='k', lw=0.3)
    ax_time.bar(x + w/2, setup, w, label='GPU setup', color=colors, alpha=0.45, edgecolor='k', lw=0.3)
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax_time.set_ylabel('Time (ms)')
    ax_time.set_title('Wall time per path')
    ax_time.legend(fontsize=8)
    ax_time.grid(True, axis='y', alpha=0.3)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def write_analysis(rows, out_txt, mol_name):
    lines = [f'{mol_name} — XC path comparison (kcal/mol)', '']
    for k in [r['mode'] for r in rows]:
        if k in XC_PATH_MODES:
            lines.append(path_description(k))
            lines.append('')
    lines.append(gpu_full_vs_otf_note())
    lines.append('')
    lines.append(f'{"mode":<16} {"E kcal/mol":>14} {"ΔE vs CPU":>14} {"cycles":>7} {"scf_ms":>8}')
    for r in rows:
        lines.append(f'{r["mode"]:<16} {r["E_kcal"]:>14.6f} {r["dE_kcal_vs_cpu"]:>+14.6f} {r["cycles"]:>7} {r["scf_ms"]:>8.0f}')
    with open(out_txt, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xyz', default=os.path.join(_REPO, 'data', 'xyz', 'formic_dimer.xyz'))
    ap.add_argument('--mode', nargs='+', default=SCAN_MODE_KEYS, choices=list(XC_PATH_MODES))
    ap.add_argument('--basis', default='6-31g')
    ap.add_argument('--grid-level', type=int, default=3)
    ap.add_argument('--no-df', action='store_true')
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--outdir', default=os.path.join(_REPO, 'debug', 'profile_xc_paths'))
    args = ap.parse_args()

    if args.threads is not None:
        lib.num_threads(args.threads)
        os.environ['OMP_NUM_THREADS'] = str(args.threads)

    atom = read_xyz(args.xyz)
    os.makedirs(args.outdir, exist_ok=True)
    log(f'PySCF {pyscf.__version__}  OMP={lib.num_threads()}')
    log(f'xyz={args.xyz}  basis={args.basis}  grid={args.grid_level}  modes={args.mode}')
    log('')
    for k in args.mode:
        log(path_description(k))
        log('')

    install_timers()
    if any(m != 'cpu' for m in args.mode):
        from pyscf.OpenCL import init_device
        init_device(quiet=True)

    mol_name = os.path.splitext(os.path.basename(args.xyz))[0]
    out_png = os.path.join(args.outdir, f'{mol_name}_paths.png')
    out_txt = os.path.join(args.outdir, f'{mol_name}_paths_analysis.txt')

    rows = []
    e_cpu = None
    dm_cpu = None
    for mode in args.mode:
        log(f'--- {mode} ({mode_label(mode)}) ---')
        dm0 = dm_cpu if mode != 'cpu' else None
        out = run_mode(atom, mode, args.basis, args.grid_level, not args.no_df, dm0=dm0)
        mf = out['mf']
        if mode == 'cpu':
            e_cpu = out['E_Ha']
            dm_cpu = mf.make_rdm1()
        dE_kcal = (out['E_Ha'] - e_cpu) * HA_TO_KCAL if e_cpu is not None else 0.0
        row = dict(mode=mode, label=mode_label(mode), E_kcal=out['E_kcal'], dE_kcal_vs_cpu=dE_kcal, converged=int(out['converged']), cycles=out['cycles'], setup_ms=out['setup_ms'], scf_ms=out['scf_ms'], get_veff_ms=out['get_veff_ms'], get_jk_ms=out['get_jk_ms'], nao=mf.mol.nao_nr(), ngrids=mf.grids.coords.shape[0])
        rows.append(row)
        log(f'  E={out["E_kcal"]:.6f} kcal/mol  ΔE_vs_cpu={dE_kcal:+.6f} kcal/mol  conv={out["converged"]}  cycles={out["cycles"]}')
        log(f'  setup={out["setup_ms"]:.0f} ms  scf={out["scf_ms"]:.0f} ms  veff={out["get_veff_ms"]:.0f} ms  jk={out["get_jk_ms"]:.0f} ms')
        plot_paths(rows, out_png, f'{mol_name} — PBE/6-31g XC paths (kcal/mol)')
        write_analysis(rows, out_txt, mol_name)
        log(f'  → updated plot {out_png}')

    csv_path = os.path.join(args.outdir, f'{mol_name}_paths.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    log(f'\n{"="*72}')
    log(f'{"mode":<16} {"ΔE kcal/mol":>14} {"cycles":>7} {"scf_ms":>8} {"setup_ms":>9}')
    for r in rows:
        log(f'{r["mode"]:<16} {r["dE_kcal_vs_cpu"]:>+14.6f} {r["cycles"]:>7} {r["scf_ms"]:>8.0f} {r["setup_ms"]:>9.0f}')
    log('')
    log(gpu_full_vs_otf_note())
    log(f'\nWrote {csv_path}\nWrote {out_png}\nWrote {out_txt}')
    log('For E(z) profile run: expamples_prokop/profile_formic_dimer_scan.py --mode cpu gpu_otf …')


if __name__ == '__main__':
    main()
