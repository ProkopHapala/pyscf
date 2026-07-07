#!/usr/bin/env python3
'''plot_scan_ez.py — primary E_bind(z) figure for dimer scan CSVs (kcal/mol).

Two panels: all XC paths on binding curve (CPU sets ylim via vmin=E_min×1.2, vmax=−2×vmin),
then ΔE_bind vs CPU. DFTB ref optional overlay. Replaces bar charts for path comparison on scans.
'''
import argparse
import csv
import os
import sys

import numpy as np

from xc_path_modes import XC_PATH_MODES, SCAN_MODE_KEYS, MODE_COLORS, mode_plot_label

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

HA_TO_KCAL = 627.5094740631
EV_TO_KCAL = HA_TO_KCAL / 27.211386245988


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_ref_scan(path):
    z, e_bind_ev = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                z.append(float(parts[0]))
                e_bind_ev.append(float(parts[2]))
    return np.asarray(z), np.asarray(e_bind_ev)


def group_by_mode(rows):
    out = {}
    for row in rows:
        mode = row['mode']
        out.setdefault(mode, {'z': [], 'E_Ha': []})
        out[mode]['z'].append(float(row['r_A']))
        out[mode]['E_Ha'].append(float(row['E_Ha']))
    for mode in out:
        o = np.argsort(out[mode]['z'])
        out[mode]['z'] = np.asarray(out[mode]['z'])[o]
        out[mode]['E_Ha'] = np.asarray(out[mode]['E_Ha'])[o]
    return out


def binding_kcal(z, E_Ha, z_ref):
    i_ref = int(np.argmin(np.abs(z - z_ref)))
    return (E_Ha - E_Ha[i_ref]) * HA_TO_KCAL


def binding_ylim(cpu_bind):
    e_min = float(np.min(cpu_bind))
    vmin = e_min * 1.2
    return vmin, -2.0 * vmin


def plot_ez(dft, modes, out_png, z_label='z (Å)', title='scan', ref_z=None, ref_bind_kcal=None, z_ref=20.0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if 'cpu' not in dft:
        raise ValueError('CSV must include cpu rows for ylim reference')
    cpu_bind = binding_kcal(dft['cpu']['z'], dft['cpu']['E_Ha'], z_ref)
    ymin, ymax = binding_ylim(cpu_bind)

    fig, (ax_ez, ax_dev) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

    for mode in modes:
        if mode not in dft:
            continue
        bind = binding_kcal(dft[mode]['z'], dft[mode]['E_Ha'], z_ref)
        ax_ez.plot(dft[mode]['z'], bind, '.-', ms=5, lw=1.4, color=MODE_COLORS.get(mode, '#333'), label=mode_plot_label(mode))

    if ref_z is not None and ref_bind_kcal is not None:
        ax_ez.plot(ref_z, ref_bind_kcal, '.-', ms=4, lw=1.0, color='#888888', alpha=0.9, label='DFTB ref')

    ax_ez.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax_ez.set_xlabel(z_label)
    ax_ez.set_ylabel('E_bind (kcal/mol)')
    ax_ez.set_title(f'{title} — E(z) all XC paths')
    ax_ez.set_ylim(ymin, ymax)
    ax_ez.legend(fontsize=8, loc='upper right', ncol=2)
    ax_ez.grid(True, alpha=0.25)
    ax_ez.set_xlim(left=max(0, float(dft['cpu']['z'].min()) - 0.15))

    for mode in modes:
        if mode == 'cpu' or mode not in dft:
            continue
        bind = binding_kcal(dft[mode]['z'], dft[mode]['E_Ha'], z_ref)
        ax_dev.plot(dft[mode]['z'], bind - cpu_bind, '.-', ms=4, lw=1.2, color=MODE_COLORS.get(mode), label=mode_plot_label(mode))
    ax_dev.axhline(0, color='k', lw=0.5)
    ax_dev.set_xlabel(z_label)
    ax_dev.set_ylabel('ΔE_bind vs CPU (kcal/mol)')
    ax_dev.legend(fontsize=7, ncol=2)
    ax_dev.grid(True, alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Plot E(z) binding profile — all XC paths')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default=None, help='Output PNG (default: same dir as csv → energy_profile_ez.png)')
    ap.add_argument('--ref-scan', default=None, help='Optional reference scan.dat')
    ap.add_argument('--z-ref', type=float, default=20.0)
    ap.add_argument('--z-label', default='O···O distance (Å)')
    ap.add_argument('--title', default='dimer scan')
    ap.add_argument('--modes', nargs='+', default=None)
    args = ap.parse_args()

    rows = load_csv(args.csv)
    dft = group_by_mode(rows)
    modes = args.modes or [m for m in SCAN_MODE_KEYS if m in dft] or sorted(dft.keys(), key=lambda m: (m != 'cpu', m))
    out_png = args.out or os.path.join(os.path.dirname(args.csv), 'energy_profile_ez.png')
    ref_z = ref_bind = None
    if args.ref_scan and os.path.isfile(args.ref_scan):
        ref_z, ref_ev = load_ref_scan(args.ref_scan)
        ref_bind = ref_ev * EV_TO_KCAL
    plot_ez(dft, modes, out_png, z_label=args.z_label, title=args.title, ref_z=ref_z, ref_bind_kcal=ref_bind, z_ref=args.z_ref)
    print(f'Wrote {out_png}  modes={modes}  n_z={len(dft.get("cpu", {}).get("z", []))}')


if __name__ == '__main__':
    main()
