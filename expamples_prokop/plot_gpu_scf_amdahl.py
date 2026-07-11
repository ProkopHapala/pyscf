#!/usr/bin/env python3
'''Plot the measured one-iteration GPU SCF Amdahl profile.'''
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


ROWS = [
    # molecule, method, end-to-end wall, kernel wall, setup, cycle get_veff, cycle DF-J, eig
    ('pentacene', 'cpu', 5355.6, 4296.7, 1058.9, 1297.7, 24.6, 4.3),
    ('pentacene', 'gpu_otf', 2964.0, 757.6, 2206.2, 228.9, 25.2, 4.2),
    ('pentacene', 'gpu_full', 3035.1, 746.2, 2288.9, 213.9, 11.5, 4.5),
    ('PTCDA', 'cpu', 8940.4, 6857.7, 2082.7, 1984.3, 52.0, 8.0),
    ('PTCDA', 'gpu_otf', 4498.6, 1087.9, 3410.7, 334.0, 50.0, 7.6),
    ('PTCDA', 'gpu_full', 4686.9, 1066.0, 3620.9, 309.4, 21.9, 7.6),
]


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'debug', 'acceptance_2026-07-11')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'pentacene_ptcda_gpu_scf_amdahl.csv')
    png_path = os.path.join(out_dir, 'pentacene_ptcda_gpu_scf_amdahl.png')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['molecule', 'method', 'end_to_end_ms', 'kernel_wall_ms', 'setup_ms', 'cycle_veff_ms', 'cycle_dfj_ms', 'cycle_eig_ms', 'speedup'])
        for mol, method, wall, kernel, setup, veff, dfj, eig in ROWS:
            cpu_wall = next(x[2] for x in ROWS if x[0] == mol and x[1] == 'cpu')
            w.writerow([mol, method, wall, kernel, setup, veff, dfj, eig, cpu_wall / wall])

    colors = {'end_to_end_ms': '#4472c4', 'kernel_wall_ms': '#ed7d31', 'setup_ms': '#ffc000', 'cycle_veff_ms': '#70ad47', 'cycle_dfj_ms': '#a5a5a5'}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1.6, 1]})
    x = np.arange(len(ROWS))
    width = .19
    for idx, key in enumerate(('end_to_end_ms', 'kernel_wall_ms', 'setup_ms', 'cycle_veff_ms', 'cycle_dfj_ms')):
        vals = np.array([r[2 + idx] for r in ROWS])
        ax[0].bar(x + (idx - 1.5) * width, vals, width, color=colors[key], label=key.replace('_ms', '').replace('_', ' '))
    ax[0].set_xticks(x, [f'{m}\n{method}' for m, method, *_ in ROWS])
    ax[0].set_ylabel('time (ms)')
    ax[0].set_title('Prepared two-cycle end-to-end decomposition')
    ax[0].grid(axis='y', alpha=.25)
    ax[0].legend(frameon=False, fontsize=9)
    speedups = []
    for mol, method, wall, *_ in ROWS:
        cpu_wall = next(r[2] for r in ROWS if r[0] == mol and r[1] == 'cpu')
        speedups.append(cpu_wall / wall)
    ax[1].bar(x, speedups, color=['#7f7f7f' if method == 'cpu' else '#2f5597' for _, method, *_ in ROWS])
    ax[1].set_xticks(x, [f'{m}\n{method}' for m, method, *_ in ROWS])
    ax[1].set_ylabel('full one-iteration kernel speedup')
    ax[1].set_title('Including initialization and setup')
    ax[1].axhline(1, color='k', lw=.8)
    ax[1].grid(axis='y', alpha=.25)
    for i, v in enumerate(speedups):
        ax[1].text(i, v + .04, f'{v:.2f}×', ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=170)
    print(png_path)
    print(csv_path)


if __name__ == '__main__':
    main()
