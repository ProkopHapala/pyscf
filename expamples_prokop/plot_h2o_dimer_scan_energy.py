#!/usr/bin/env python3
'''plot_h2o_dimer_scan_energy.py — 4-panel dimer scan diagnostic (binding deviation, DFTB ref).

Secondary to plot_scan_ez.py. Adds total-energy deviation panel and CPU vs DFTB shape comparison.
All energies in kcal/mol; binding ylim from CPU minimum.
'''
import argparse
import csv
import os
import sys

import numpy as np

from xc_path_modes import XC_PATH_MODES, SCAN_MODE_KEYS, MODE_COLORS, mode_plot_label, path_description, gpu_full_vs_otf_note

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_DEFAULT_CSV = os.path.join(_REPO, 'debug', 'profile_h2o_dimer_scan', 'scan_scf_profile.csv')
_DEFAULT_XTB = os.path.join(os.path.normpath(os.path.join(_REPO, '..', 'CompChemUtils')), 'tmp', 'H2O_dimer_scan_dftb', 'scan.dat')
_DEFAULT_OUTDIR = os.path.join(_REPO, 'debug', 'profile_h2o_dimer_scan')

HA_TO_KCAL = 627.5094740631
EV_TO_KCAL = HA_TO_KCAL / 27.211386245988


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_xtb_scan(path):
    r, e_bind_ev = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                r.append(float(parts[0]))
                e_bind_ev.append(float(parts[2]))
    return np.asarray(r), np.asarray(e_bind_ev)


def group_by_mode(rows):
    out = {}
    for row in rows:
        mode = row['mode']
        out.setdefault(mode, {'r': [], 'E_Ha': []})
        out[mode]['r'].append(float(row['r_A']))
        out[mode]['E_Ha'].append(float(row['E_Ha']))
    for mode in out:
        o = np.argsort(out[mode]['r'])
        out[mode]['r'] = np.asarray(out[mode]['r'])[o]
        out[mode]['E_Ha'] = np.asarray(out[mode]['E_Ha'])[o]
    return out


def binding_kcal(r, E_Ha, r_ref=20.0):
    i_ref = int(np.argmin(np.abs(r - r_ref)))
    return (E_Ha - E_Ha[i_ref]) * HA_TO_KCAL, E_Ha[i_ref]


def r_at_min(r, y):
    i = int(np.argmin(y))
    return float(r[i]), float(y[i])


def binding_ylim(cpu_bind_kcal):
    e_min = float(np.min(cpu_bind_kcal))
    vmin = e_min * 1.2
    vmax = -2.0 * vmin
    return vmin, vmax


def analyze(dft, xtb_r, xtb_ebind_ev, modes, r_ref=20.0, title='dimer scan'):
    lines = []
    lines.append(f'{title} — energy profile (kcal/mol)')
    lines.append(f'E_bind(r) = [E_tot(r) − E_tot(r={r_ref} Å)] × {HA_TO_KCAL:.4f} kcal/mol/Ha')
    lines.append('')

    if 'cpu' not in dft:
        raise ValueError('CSV must include cpu mode for reference')
    cpu = dft['cpu']
    cpu_bind, _ = binding_kcal(cpu['r'], cpu['E_Ha'], r_ref)
    cpu_req, cpu_emin = r_at_min(cpu['r'], cpu_bind)

    lines.append('=== XC path definitions ===')
    for k in modes:
        if k in XC_PATH_MODES:
            lines.append(path_description(k))
            lines.append('')
    lines.append('=== ' + gpu_full_vs_otf_note() + ' ===')
    lines.append('')

    lines.append('=== Binding curves (kcal/mol, ref r=20 Å) ===')
    lines.append(f'{"mode":<16} {"r_min Å":>8} {"E_bind_min":>14} {"ΔE_bind vs CPU":>16}')
    for mode in modes:
        if mode not in dft:
            continue
        bind, _ = binding_kcal(dft[mode]['r'], dft[mode]['E_Ha'], r_ref)
        req, emin = r_at_min(dft[mode]['r'], bind)
        d_min = emin - cpu_emin if mode != 'cpu' else 0.0
        lines.append(f'{mode:<16} {req:>8.3f} {emin:>14.4f} {d_min:>+16.4f}')

    xtb_bind = xtb_ebind_ev * EV_TO_KCAL
    xtb_req, xtb_emin = r_at_min(xtb_r, xtb_bind)
    lines.append(f'{"ref DFTB":<16} {xtb_req:>8.3f} {xtb_emin:>14.4f} {"(ref only)":>16}')
    lines.append('')

    lines.append('=== Path deviation vs CPU (kcal/mol) ===')
    for mode in modes:
        if mode == 'cpu' or mode not in dft:
            continue
        bind, _ = binding_kcal(dft[mode]['r'], dft[mode]['E_Ha'], r_ref)
        d_bind = bind - cpu_bind
        d_tot = (dft[mode]['E_Ha'] - cpu['E_Ha']) * HA_TO_KCAL
        lines.append(f'{mode} ({XC_PATH_MODES[mode]["short"]}):')
        lines.append(f'  ΔE_bind: max|Δ|={np.max(np.abs(d_bind)):.4f}  RMS={np.sqrt(np.mean(d_bind**2)):.4f} kcal/mol')
        lines.append(f'  ΔE_tot:  max|Δ|={np.max(np.abs(d_tot)):.4f}  RMS={np.sqrt(np.mean(d_tot**2)):.4f} kcal/mol')
        lines.append(f'  at r_min(CPU): ΔE_bind={d_bind[int(np.argmin(cpu_bind))]:+.4f} kcal/mol')

    lines.append('')
    lines.append('=== DFT (CPU) vs DFTB reference binding shape ===')
    cpu_on_xtb = np.interp(xtb_r, cpu['r'], cpu_bind)
    d_xtb = cpu_on_xtb - xtb_bind
    lines.append(f'max|ΔE_bind|={np.max(np.abs(d_xtb)):.3f} kcal/mol  RMS={np.sqrt(np.mean(d_xtb**2)):.3f} kcal/mol')
    lines.append(f'r_min: CPU {cpu_req:.3f} Å vs DFTB {xtb_req:.3f} Å  (Δr={cpu_req-xtb_req:+.3f} Å)')
    lines.append(f'well depth: CPU {cpu_emin:.2f} vs DFTB {xtb_emin:.2f} kcal/mol  (Δ={cpu_emin-xtb_emin:+.2f})')
    return '\n'.join(lines), cpu_bind


def make_plot(dft, xtb_r, xtb_ebind_ev, modes, out_png, r_ref=20.0, title='dimer scan', z_label='inter-fragment distance (Å)'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cpu = dft['cpu']
    cpu_bind, _ = binding_kcal(cpu['r'], cpu['E_Ha'], r_ref)
    ymin, ymax = binding_ylim(cpu_bind)
    xtb_bind = xtb_ebind_ev * EV_TO_KCAL
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_bind, ax_dev_bind, ax_dev_tot, ax_xtb = axes.ravel()

    for mode in modes:
        if mode not in dft:
            continue
        bind, _ = binding_kcal(dft[mode]['r'], dft[mode]['E_Ha'], r_ref)
        c = MODE_COLORS.get(mode, None)
        lbl = mode_plot_label(mode)
        ax_bind.plot(dft[mode]['r'], bind, '.-', ms=4, lw=1.2, color=c, label=lbl)

    ax_bind.plot(xtb_r, xtb_bind, '.-', ms=3, lw=1.0, color='#888888', alpha=0.85, label='DFTB ref')
    ax_bind.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax_bind.set_xlabel(z_label)
    ax_bind.set_ylabel('E_bind (kcal/mol)')
    ax_bind.set_title('Binding curves (kcal/mol, ref r = 20 Å)')
    ax_bind.set_ylim(ymin, ymax)
    ax_bind.legend(fontsize=7, loc='upper right')
    ax_bind.grid(True, alpha=0.25)
    ax_bind.set_xlim(left=1.5)

    for mode in modes:
        if mode == 'cpu' or mode not in dft:
            continue
        bind, _ = binding_kcal(dft[mode]['r'], dft[mode]['E_Ha'], r_ref)
        ax_dev_bind.plot(dft[mode]['r'], bind - cpu_bind, '.-', ms=4, lw=1.2, color=MODE_COLORS.get(mode), label=mode_plot_label(mode))
    ax_dev_bind.axhline(0, color='k', lw=0.5)
    ax_dev_bind.set_xlabel(z_label)
    ax_dev_bind.set_ylabel('ΔE_bind vs CPU (kcal/mol)')
    ax_dev_bind.set_title('Binding deviation by XC path')
    ax_dev_bind.legend(fontsize=7)
    ax_dev_bind.grid(True, alpha=0.25)

    for mode in modes:
        if mode == 'cpu' or mode not in dft:
            continue
        d_tot = (dft[mode]['E_Ha'] - cpu['E_Ha']) * HA_TO_KCAL
        ax_dev_tot.plot(dft[mode]['r'], d_tot, '.-', ms=4, lw=1.2, color=MODE_COLORS.get(mode), label=mode_plot_label(mode))
    ax_dev_tot.axhline(0, color='k', lw=0.5)
    ax_dev_tot.set_xlabel(z_label)
    ax_dev_tot.set_ylabel('ΔE_tot vs CPU (kcal/mol)')
    ax_dev_tot.set_title('Total energy deviation (same geometry)')
    ax_dev_tot.legend(fontsize=7)
    ax_dev_tot.grid(True, alpha=0.25)

    cpu_on_xtb = np.interp(xtb_r, cpu['r'], cpu_bind)
    ax_xtb.plot(xtb_r, xtb_bind, '.-', ms=3, lw=1.2, color='#888888', label='DFTB ref')
    ax_xtb.plot(xtb_r, cpu_on_xtb, '.-', ms=4, lw=1.2, color=MODE_COLORS['cpu'], label='CPU libxc')
    ax_xtb.set_xlabel(z_label)
    ax_xtb.set_ylabel('E_bind (kcal/mol)')
    ax_xtb.set_title('CPU PBE vs DFTB ref')
    ax_xtb.set_ylim(ymin, ymax)
    ax_xtb.legend(fontsize=8)
    ax_xtb.grid(True, alpha=0.25)
    ax_xtb.set_xlim(left=1.5)

    fig.suptitle(f'{title} — XC paths in kcal/mol (PBE/6-31g DF)', fontsize=11, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=_DEFAULT_CSV)
    ap.add_argument('--ref-scan', default=_DEFAULT_XTB, help='Reference scan.dat (r, E_tot, E_bind in eV)')
    ap.add_argument('--outdir', default=_DEFAULT_OUTDIR)
    ap.add_argument('--r-ref', type=float, default=20.0)
    ap.add_argument('--modes', nargs='+', default=None)
    ap.add_argument('--title', default='H₂O dimer scan')
    ap.add_argument('--z-label', default='O···O distance (Å)')
    args = ap.parse_args()

    rows = load_csv(args.csv)
    dft = group_by_mode(rows)
    modes = args.modes or sorted(dft.keys(), key=lambda m: (m != 'cpu', m))
    xtb_r, xtb_ebind_ev = load_xtb_scan(args.ref_scan)

    report, _ = analyze(dft, xtb_r, xtb_ebind_ev, modes, r_ref=args.r_ref, title=args.title)
    print(report)

    out_png = os.path.join(args.outdir, 'energy_profile.png')
    out_txt = os.path.join(args.outdir, 'energy_analysis.txt')
    make_plot(dft, xtb_r, xtb_ebind_ev, modes, out_png, r_ref=args.r_ref, title=args.title, z_label=args.z_label)
    with open(out_txt, 'w') as f:
        f.write(report + f'\n\nPlot: {out_png}\n')
    print(f'\nWrote {out_png}\nWrote {out_txt}')


if __name__ == '__main__':
    main()
