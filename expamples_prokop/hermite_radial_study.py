#!/usr/bin/env python3
'''Hermite radial spline study — shared plots + unified CLI.

Subcommands:
  report   grid + beta table + error curves + per-shell u vs r  (run this)
  carbon   per-shell |ΔR| vs β for one interp mode
  grid     r_i vs node index
  compare  per-shell spatial u vs r overlay
  matrix   cubic/quintic plots: u/r × analytic/quadrature (β=1 power grid)
  f32      f64 vs OpenCL float32 spline error (same combos)

Grid power: fixed N; r_i = r0·(rmax/r0)^((i/(N-1))^(1/β)); β>1 packs toward origin.

  PYTHONPATH=/home/prokop/git/pyscf python3 -u expamples_prokop/hermite_radial_study.py report
'''
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto
from pyscf.data import nist
from pyscf.OpenCL.hermite_spline import (
    build_radial_tables_for_shell, contracted_radial_coeff, eval_radial, eval_radial_dr,
    eval_radial_spline, eval_radial_spline_dr, node_r_distribution, normalize_tangents, reference_node_count,
)
from pyscf.OpenCL.hermite_spline_cl import eval_radial_spline_cl, eval_radial_spline_f32_cpu, tables_to_f32

ANG = nist.BOHR
LW = 0.5
LOG_FLOOR = 1e-16
DEFAULT_ORDER = 'quintic'
DEFAULT_TANGENTS = None  # None → quadrature


def _resolve_tangents(order, tangents, fit=None):
    if tangents is not None or fit is not None:
        return normalize_tangents(tangents=tangents, fit=fit)
    return 'quadrature'


MATRIX_COMBOS = [(order, interp, tang) for order in ('cubic', 'quintic') for interp in ('u', 'r') for tang in ('analytic', 'quadrature')]
ORDER_COMBOS = [(interp, tang) for interp in ('u', 'r') for tang in ('analytic', 'quadrature')]
_COMBO_COLORS = {('analytic', 'u'): 'green', ('analytic', 'r'): 'blue', ('quadrature', 'u'): 'orange', ('quadrature', 'r'): 'purple'}
LW_REF = 1.5


def _combo_style(interp, tangents):
    tag = 'quad' if tangents == 'quadrature' else 'ana'
    return dict(ls='-', lw=LW, color=_COMBO_COLORS[(tangents, interp)], label=f'{tag}-{interp}')


def sliding_max(y, window):
    '''Centered sliding maximum of |y|; window<=0 returns |y| unchanged.'''
    a = np.abs(np.asarray(y, dtype=np.double))
    w = int(window)
    if w <= 1:
        return a
    half = w // 2
    out = np.empty_like(a)
    for i in range(a.size):
        out[i] = np.max(a[max(0, i - half):min(a.size, i + half + 1)])
    return out


_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_DEFAULT_OUT = os.path.join(_REPO, 'debug', 'plot_hermite_cubic_quintic')


def log(msg):
    print(msg, flush=True)


def make_mol(name='benzene'):
    if name == 'water':
        return gto.M(atom='O 0 0 0; H 0 0 0.957; H 0 0.957 0', basis='cc-pvdz', unit='Angstrom', cart=False, verbose=0)
    xyz = os.path.join(_REPO, 'data', 'xyz', 'benzene.xyz')
    with open(xyz) as f:
        lines = f.readlines()
    nat = int(lines[0].strip())
    atoms = [' '.join(line.split()[:4]) for line in lines[2:2 + nat]]
    return gto.M(atom='; '.join(atoms), basis='cc-pvdz', unit='Angstrom', cart=False, verbose=0)


def carbon_ref_atom(mol):
    return next(i for i in range(mol.natm) if mol.atom_symbol(i) == 'C')


def carbon_radial_classes(mol):
    ia = carbon_ref_atom(mol)
    classes, k_per_l = [], {0: 0, 1: 0, 2: 0, 3: 0}
    for ib in range(mol.nbas):
        if mol.bas_atom(ib) != ia:
            continue
        l = mol.bas_angular(ib)
        k_per_l[l] += 1
        lname = 'spdf'[l] if l < 4 else f'l{l}'
        nlabel = k_per_l[l] + l
        for ic in range(mol.bas_nctr(ib)):
            tag = f'{nlabel}{lname}' if ic == 0 else f"{nlabel}{lname}'"
            classes.append((tag, ib, ic))
    return classes


def _logy(y):
    return np.maximum(np.abs(y), LOG_FLOOR)


def _n_nodes(r0_ang, du, rmax_ang, beta_ref, grid, n_nodes):
    if n_nodes is not None:
        return int(n_nodes)
    if grid in ('power', 'uniform'):
        return reference_node_count(r0_ang / ANG, du, rmax_ang / ANG, beta_ref)
    return None


def _r_dense_grid(r0_ang, rmax_ang):
    '''Sample from first knot r0 — avoids extrapolation below grid (boundary artifact).'''
    return np.linspace(float(r0_ang) / ANG, float(rmax_ang) / ANG, 5000)


def profile_channel(mol, ib, ic, r_dense, r0_ang, du, rmax_ang, beta, *, grid='power', n_nodes=None, interp_space='u', order=DEFAULT_ORDER, tangents=DEFAULT_TANGENTS, fit=None, n_quad=5, origin_knot=True):
    tang = _resolve_tangents(order, tangents, fit)
    nn = _n_nodes(r0_ang, du, rmax_ang, beta, grid, n_nodes) if grid in ('power', 'uniform') else None
    tab = build_radial_tables_for_shell(mol, ib, r0_ang, du, rmax_ang, order=order, tangents=tang, n_quad=n_quad, map_b=beta, interp_space=interp_space, grid=grid, n_nodes=nn, origin_knot=origin_knot)
    expn, coeff = contracted_radial_coeff(mol, ib)
    ref = eval_radial(r_dense, expn, coeff)[:, ic]
    ref_dr = eval_radial_dr(r_dense, expn, coeff)[:, ic]
    got = eval_radial_spline(r_dense, tab, order=order)[:, ic]
    got_dr = eval_radial_spline_dr(r_dense, tab, order=order)[:, ic]
    err, err_dr = got - ref, got_dr - ref_dr
    i_max, i_max_dr = int(np.argmax(np.abs(err))), int(np.argmax(np.abs(err_dr)))
    return dict(ref=ref, ref_dr=ref_dr, err=err, err_dr=err_dr, n_nodes=tab['n_nodes'], max_abs=float(np.abs(err[i_max])), max_abs_dr=float(np.abs(err_dr[i_max_dr])), r_max_err_ang=float(r_dense[i_max] * ANG), r_max_err_dr_ang=float(r_dense[i_max_dr] * ANG))


def profile_channel_f32(mol, ib, ic, r_dense, r0_ang, du, rmax_ang, beta, *, grid='power', n_nodes=None, interp_space='u', order=DEFAULT_ORDER, tangents=DEFAULT_TANGENTS, fit=None, n_quad=5, origin_knot=True, backend='cl'):
    tang = _resolve_tangents(order, tangents, fit)
    nn = _n_nodes(r0_ang, du, rmax_ang, beta, grid, n_nodes) if grid in ('power', 'uniform') else None
    tab = build_radial_tables_for_shell(mol, ib, r0_ang, du, rmax_ang, order=order, tangents=tang, n_quad=n_quad, map_b=beta, interp_space=interp_space, grid=grid, n_nodes=nn, origin_knot=origin_knot)
    tab_f32 = tables_to_f32(tab)
    r_q = r_dense.astype(np.float32)
    eval_fn = eval_radial_spline_cl if backend == 'cl' else eval_radial_spline_f32_cpu
    got, got_dr = eval_fn(r_q, tab_f32, order=order, interp_space=interp_space, ic=ic)
    expn, coeff = contracted_radial_coeff(mol, ib)
    ref = eval_radial(r_dense, expn, coeff)[:, ic]
    ref_dr = eval_radial_dr(r_dense, expn, coeff)[:, ic]
    err, err_dr = got.astype(np.float64) - ref, got_dr.astype(np.float64) - ref_dr
    i_max, i_max_dr = int(np.argmax(np.abs(err))), int(np.argmax(np.abs(err_dr)))
    return dict(err=err, err_dr=err_dr, max_abs=float(np.abs(err[i_max])), max_abs_dr=float(np.abs(err_dr[i_max_dr])), r_max_err_dr_ang=float(r_dense[i_max_dr] * ANG))


def plot_shell_beta_sweep(mol, tag, ib, ic, r_dense, r0_ang, du, rmax_ang, beta_list, outdir, *, grid='power', n_nodes=None, interp_space='u', order=DEFAULT_ORDER, tangents=DEFAULT_TANGENTS, fit=None, origin_knot=True):
    tang = _resolve_tangents(order, tangents, fit)
    r_ang = r_dense * ANG
    expn, coeff = contracted_radial_coeff(mol, ib)
    ref = eval_radial(r_dense, expn, coeff)[:, ic]
    ref_dr = eval_radial_dr(r_dense, expn, coeff)[:, ic]
    cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(beta_list)))
    nn = _n_nodes(r0_ang, du, rmax_ang, beta_list[0], grid, n_nodes)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax_r, ax_d = axes
    ax_r.semilogy(r_ang, _logy(ref), color='k', ls='-', lw=LW, label='|R| exact', zorder=10)
    ax_d.semilogy(r_ang, _logy(ref_dr), color='k', ls='-', lw=LW, label='|dR/dr| exact', zorder=10)
    rows = []
    for j, beta in enumerate(beta_list):
        p = profile_channel(mol, ib, ic, r_dense, r0_ang, du, rmax_ang, beta, grid=grid, n_nodes=nn, interp_space=interp_space, order=order, tangents=tang, origin_knot=origin_knot)
        rows.append((beta, p['n_nodes'], p['max_abs'], p['max_abs_dr']))
        lbl = f'β={beta:g} |ΔR|={p["max_abs"]:.1e}'
        ax_r.semilogy(r_ang, _logy(p['err']), ls='-', lw=LW, color=cmap[j], label=lbl)
        ax_d.semilogy(r_ang, _logy(p['err_dr']), ls='-', lw=LW, color=cmap[j], label=f'β={beta:g} |ΔR\'|={p["max_abs_dr"]:.1e}')
    for ax, ylab in zip(axes, ('|R| exact & |ΔR|', '|dR/dr| exact & |Δ(dR/dr)|')):
        ax.set_ylabel(ylab)
        ax.set_ylim(bottom=LOG_FLOOR)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, which='both', alpha=0.3)
    ax_d.set_xlabel('r (Å)')
    fig.suptitle(f'C {tag}  {order}/{tang}  interp={interp_space}  {grid} N={nn}', fontsize=9)
    fig.tight_layout()
    fname = re.sub(r'[^\w]', '', tag.replace("'", 'prime'))
    path = os.path.join(outdir, f'shell_{fname}_{order}_{grid}_{interp_space}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, rows


def plot_node_distribution(r0_ang, du, rmax_ang, beta_list, outdir, *, grid='power', n_nodes=None, beta_ref=1.0):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _plot_node_panel(axes[0], r0_ang, du, rmax_ang, beta_list, grid=grid, n_nodes=n_nodes, beta_ref=beta_ref)
    _plot_dr_panel(axes[1], r0_ang, du, rmax_ang, beta_list, grid=grid, n_nodes=n_nodes, beta_ref=beta_ref)
    nn = _n_nodes(r0_ang, du, rmax_ang, beta_ref, grid, n_nodes)
    if grid == 'power':
        title = f'power grid N={nn}: β>1 packs toward r0'
    elif grid == 'uniform':
        title = f'uniform grid N={nn}: equal Δr in physical r'
    else:
        title = f'log grid du={du}: β changes N'
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = os.path.join(outdir, f'grid_{grid}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_node_panel(ax, r0_ang, du, rmax_ang, beta_list, *, grid, n_nodes, beta_ref):
    cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(beta_list)))
    nn = _n_nodes(r0_ang, du, rmax_ang, beta_ref, grid, n_nodes)
    for j, beta in enumerate(beta_list):
        g = node_r_distribution(r0_ang, du, rmax_ang, beta, grid=grid, n_nodes=nn, beta_ref=beta_ref)
        ax.plot(g['i'], g['r_ang'], ls='-', lw=LW, color=cmap[j], label=f'β={beta:g} n={g["n"]}')
    ax.set_xlabel('node index i')
    ax.set_ylabel('r_i (Å)')
    ax.set_title('r_i vs index')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)


def _plot_dr_panel(ax, r0_ang, du, rmax_ang, beta_list, *, grid, n_nodes, beta_ref):
    cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(beta_list)))
    nn = _n_nodes(r0_ang, du, rmax_ang, beta_ref, grid, n_nodes)
    for j, beta in enumerate(beta_list):
        g = node_r_distribution(r0_ang, du, rmax_ang, beta, grid=grid, n_nodes=nn, beta_ref=beta_ref)
        dr = np.diff(g['r_ang'])
        ax.semilogy(g['r_ang'][:-1], np.maximum(dr, LOG_FLOOR), ls='-', lw=LW, color=cmap[j], label=f'β={beta:g}')
    ax.set_xlabel('r (Å)')
    ax.set_ylabel('Δr to next node')
    ax.set_ylim(bottom=LOG_FLOOR)
    ax.set_title('local spacing Δr')
    ax.legend(fontsize=6)
    ax.grid(True, which='both', alpha=0.3)


def cmd_carbon(args):
    os.makedirs(args.outdir, exist_ok=True)
    mol = make_mol(args.mol)
    r_dense = _r_dense_grid(args.r0_ang, args.rmax_ang)
    nn = _n_nodes(args.r0_ang, args.du, args.rmax_ang, args.beta_ref, args.grid, args.n_nodes)
    log(f'{args.order}/{_resolve_tangents(args.order, args.tangents, args.fit)}  grid={args.grid}  interp={args.interp}  N={nn}  β∈{args.beta_list}')
    for tag, ib, ic in carbon_radial_classes(mol):
        path, rows = plot_shell_beta_sweep(mol, tag, ib, ic, r_dense, args.r0_ang, args.du, args.rmax_ang, args.beta_list, args.outdir, grid=args.grid, n_nodes=nn, interp_space=args.interp, order=args.order, tangents=args.tangents, fit=args.fit, origin_knot=args.origin_knot)
        log(f'{tag}: ' + '  '.join(f'β={b:g} |ΔR|={e:.1e} |ΔR\'|={ed:.1e}' for b, _, e, ed in rows))
        log(f'  {path}')


def collect_beta_sweep(mol, r_dense, r0_ang, du, rmax_ang, beta_list, *, grid='power', n_nodes=None, order=DEFAULT_ORDER, tangents=DEFAULT_TANGENTS, fit=None, origin_knot=True):
    tang = _resolve_tangents(order, tangents, fit)
    nn = _n_nodes(r0_ang, du, rmax_ang, beta_list[0], grid, n_nodes)
    rows = []
    for tag, ib, ic in carbon_radial_classes(mol):
        for beta in beta_list:
            pu = profile_channel(mol, ib, ic, r_dense, r0_ang, du, rmax_ang, beta, grid=grid, n_nodes=nn, interp_space='u', order=order, tangents=tang, origin_knot=origin_knot)
            pr = profile_channel(mol, ib, ic, r_dense, r0_ang, du, rmax_ang, beta, grid=grid, n_nodes=nn, interp_space='r', order=order, tangents=tang, origin_knot=origin_knot)
            rows.append(dict(shell=tag, beta=beta, n_nodes=nn, max_R_u=pu['max_abs'], max_R_r=pr['max_abs'], max_dR_u=pu['max_abs_dr'], max_dR_r=pr['max_abs_dr'], r_dR_u=pu['r_max_err_dr_ang'], r_dR_r=pr['r_max_err_dr_ang']))
    return rows


def format_sweep_table(rows, nn, grid, order, tangents):
    hdr = f'{order}/{tangents}  origin_knot  {grid} grid N={nn}\n'
    hdr += f'{"shell":<6} {"β":>5} {"|ΔR|_u":>11} {"|ΔR|_r":>11} {"|ΔR\'|_u":>11} {"|ΔR\'|_r":>11} {"r\'_u(Å)":>8} {"r\'_r(Å)":>8}\n'
    hdr += '-' * 80 + '\n'
    lines = [hdr]
    for row in rows:
        lines.append(f'{row["shell"]:<6} {row["beta"]:5g} {row["max_R_u"]:11.3e} {row["max_R_r"]:11.3e} {row["max_dR_u"]:11.3e} {row["max_dR_r"]:11.3e} {row["r_dR_u"]:8.4f} {row["r_dR_r"]:8.4f}\n')
    return ''.join(lines)


def plot_beta_sweep_by_mode(rows, beta_list, outdir, *, grid, nn, mode, order, tangents):
    '''Separate plot per interp mode (u or r).'''
    shells = [r['shell'] for r in rows if r['beta'] == beta_list[0]]
    cmap = plt.cm.tab10(np.linspace(0, 0.85, len(shells)))
    betas = np.asarray(beta_list, dtype=np.double)
    key_R = f'max_R_{mode}'
    key_dR = f'max_dR_{mode}'
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for ax, key, ylab in zip(axes, (key_R, key_dR), ('max |ΔR|', 'max |Δ(dR/dr)|')):
        for j, shell in enumerate(shells):
            sub = [r for r in rows if r['shell'] == shell]
            y = np.array([r[key] for r in sub])
            ax.loglog(betas, y, ls='-', lw=LW, color=cmap[j], marker='o', ms=4, label=shell)
        ax.set_ylabel(ylab)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=7, loc='best')
    axes[1].set_xlabel('β (origin clustering, fixed N)')
    fig.suptitle(f'interp={mode}-mode  {order}/{tangents}  {grid} N={nn}', fontsize=10)
    fig.tight_layout()
    path = os.path.join(outdir, f'beta_sweep_{order}_{grid}_{mode}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_beta_sweep_derivative(rows, beta_list, outdir, *, grid, nn, order, tangents):
    '''Derivative errors only — separate u and r figures.'''
    paths = []
    for mode in ('u', 'r'):
        shells = [r['shell'] for r in rows if r['beta'] == beta_list[0]]
        cmap = plt.cm.tab10(np.linspace(0, 0.85, len(shells)))
        betas = np.asarray(beta_list, dtype=np.double)
        key = f'max_dR_{mode}'
        fig, ax = plt.subplots(figsize=(9, 5))
        for j, shell in enumerate(shells):
            sub = [r for r in rows if r['shell'] == shell]
            y = np.array([r[key] for r in sub])
            ax.loglog(betas, y, ls='-', lw=LW, color=cmap[j], marker='o', ms=4, label=shell)
        ax.set_xlabel('β (origin clustering, fixed N)')
        ax.set_ylabel('max |Δ(dR/dr)|')
        ax.set_title(f'derivative error  interp={mode}-mode  {order}/{tangents}  {grid} N={nn}')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=7, loc='best')
        fig.tight_layout()
        path = os.path.join(outdir, f'beta_sweep_derivative_{order}_{grid}_{mode}.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def cmd_report(args):
    os.makedirs(args.outdir, exist_ok=True)
    mol = make_mol(args.mol)
    r_dense = _r_dense_grid(args.r0_ang, args.rmax_ang)
    nn = _n_nodes(args.r0_ang, args.du, args.rmax_ang, args.beta_ref, args.grid, args.n_nodes)
    tang = _resolve_tangents(args.order, args.tangents, args.fit)
    log(f'report  {args.order}/{tang}  origin_knot={args.origin_knot}  grid={args.grid}  N={nn}  β∈{args.beta_list}')

    gpath = plot_node_distribution(args.r0_ang, args.du, args.rmax_ang, args.beta_list, args.outdir, grid=args.grid, n_nodes=nn, beta_ref=args.beta_ref)
    log(f'grid: {gpath}')

    rows = collect_beta_sweep(mol, r_dense, args.r0_ang, args.du, args.rmax_ang, args.beta_list, grid=args.grid, n_nodes=nn, order=args.order, tangents=args.tangents, fit=args.fit, origin_knot=args.origin_knot)
    table = format_sweep_table(rows, nn, args.grid, args.order, tang)
    tpath = os.path.join(args.outdir, f'beta_sweep_report_{args.order}_{args.grid}.txt')
    with open(tpath, 'w') as f:
        f.write(table)
    log(f'\n{table}')
    log(f'table: {tpath}')

    for mode in ('u', 'r'):
        p = plot_beta_sweep_by_mode(rows, args.beta_list, args.outdir, grid=args.grid, nn=nn, mode=mode, order=args.order, tangents=tang)
        log(f'beta sweep {mode}-mode: {p}')
    for p in plot_beta_sweep_derivative(rows, args.beta_list, args.outdir, grid=args.grid, nn=nn, order=args.order, tangents=tang):
        log(f'derivative plot: {p}')

    for tag, ib, ic in carbon_radial_classes(mol):
        for mode in ('u', 'r'):
            path, _ = plot_shell_beta_sweep(mol, tag, ib, ic, r_dense, args.r0_ang, args.du, args.rmax_ang, args.beta_list, args.outdir, grid=args.grid, n_nodes=nn, interp_space=mode, order=args.order, tangents=args.tangents, fit=args.fit, origin_knot=args.origin_knot)
            log(f'shell {tag} {mode}-mode: {path}')


def plot_matrix_shell(mol, tag, ib, ic, r_dense, r0_ang, du, rmax_ang, beta, outdir, *, grid='power', n_nodes=None, origin_knot=True, smooth_window=5):
    r_ang = r_dense * ANG
    expn, coeff = contracted_radial_coeff(mol, ib)
    ref = eval_radial(r_dense, expn, coeff)[:, ic]
    ref_dr = eval_radial_dr(r_dense, expn, coeff)[:, ic]
    nn = _n_nodes(r0_ang, du, rmax_ang, beta, grid, n_nodes)
    fname = re.sub(r'[^\w]', '', tag.replace("'", 'prime'))
    smooth_note = f'  smooth={smooth_window}' if smooth_window > 1 else ''
    paths, rows = [], []
    for order in ('cubic', 'quintic'):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for interp, tang in ORDER_COMBOS:
            p = profile_channel(mol, ib, ic, r_dense, r0_ang, du, rmax_ang, beta, grid=grid, n_nodes=nn, interp_space=interp, order=order, tangents=tang, origin_knot=origin_knot)
            sty = _combo_style(interp, tang)
            rows.append((order, interp, tang, p['max_abs'], p['max_abs_dr'], p['r_max_err_dr_ang']))
            axes[0].semilogy(r_ang, _logy(sliding_max(p['err'], smooth_window)), **sty)
            axes[1].semilogy(r_ang, _logy(sliding_max(p['err_dr'], smooth_window)), **sty)
        axes[0].semilogy(r_ang, _logy(ref), color='k', ls='-', lw=LW_REF, label='|R| exact', zorder=10)
        axes[1].semilogy(r_ang, _logy(ref_dr), color='k', ls='-', lw=LW_REF, label='|dR/dr| exact', zorder=10)
        for ax, ylab in zip(axes, ('|ΔR|', '|Δ(dR/dr)|')):
            ax.set_ylabel(ylab)
            ax.set_ylim(bottom=LOG_FLOOR)
            ax.grid(True, which='both', alpha=0.3)
        axes[1].set_xlabel('r (Å)')
        h0, l0 = axes[0].get_legend_handles_labels()
        fig.legend(h0, l0, fontsize=7, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f'C {tag}  {order}  β={beta:g}  {grid} N={nn}{smooth_note}  green=ana-u blue=ana-r orange=quad-u purple=quad-r', fontsize=8, y=1.06)
        fig.tight_layout()
        path = os.path.join(outdir, f'matrix_{fname}_{order}_{grid}_b{beta:g}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths.append(path)
    return paths, rows


def plot_f32_shell(mol, tag, ib, ic, r_dense, r0_ang, du, rmax_ang, beta, outdir, *, grid='power', n_nodes=None, origin_knot=True, smooth_window=5, backend='cl'):
    r_ang = r_dense * ANG
    expn, coeff = contracted_radial_coeff(mol, ib)
    ref = eval_radial(r_dense, expn, coeff)[:, ic]
    ref_dr = eval_radial_dr(r_dense, expn, coeff)[:, ic]
    nn = _n_nodes(r0_ang, du, rmax_ang, beta, grid, n_nodes)
    fname = re.sub(r'[^\w]', '', tag.replace("'", 'prime'))
    smooth_note = f'  smooth={smooth_window}' if smooth_window > 1 else ''
    paths, rows = [], []
    for order in ('cubic', 'quintic'):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for interp, tang in ORDER_COMBOS:
            pf = profile_channel(mol, ib, ic, r_dense, r0_ang, du, rmax_ang, beta, grid=grid, n_nodes=nn, interp_space=interp, order=order, tangents=tang, origin_knot=origin_knot)
            p32 = profile_channel_f32(mol, ib, ic, r_dense, r0_ang, du, rmax_ang, beta, grid=grid, n_nodes=nn, interp_space=interp, order=order, tangents=tang, origin_knot=origin_knot, backend=backend)
            sty = _combo_style(interp, tang)
            lbl = sty['label']
            rows.append((order, interp, tang, pf['max_abs_dr'], p32['max_abs_dr'], p32['r_max_err_dr_ang']))
            axes[0].semilogy(r_ang, _logy(sliding_max(pf['err'], smooth_window)), ls=sty['ls'], lw=sty['lw'], color=sty['color'], label=f'{lbl} f64')
            axes[0].semilogy(r_ang, _logy(sliding_max(p32['err'], smooth_window)), ls='--', lw=sty['lw'], color=sty['color'], label=f'{lbl} f32')
            axes[1].semilogy(r_ang, _logy(sliding_max(pf['err_dr'], smooth_window)), ls=sty['ls'], lw=sty['lw'], color=sty['color'])
            axes[1].semilogy(r_ang, _logy(sliding_max(p32['err_dr'], smooth_window)), ls='--', lw=sty['lw'], color=sty['color'])
        axes[0].semilogy(r_ang, _logy(ref), color='k', ls='-', lw=LW_REF, label='|R| exact', zorder=10)
        axes[1].semilogy(r_ang, _logy(ref_dr), color='k', ls='-', lw=LW_REF, label='|dR/dr| exact', zorder=10)
        for ax, ylab in zip(axes, ('|ΔR|', '|Δ(dR/dr)|')):
            ax.set_ylabel(ylab)
            ax.set_ylim(bottom=LOG_FLOOR)
            ax.grid(True, which='both', alpha=0.3)
        axes[1].set_xlabel('r (Å)')
        h0, l0 = axes[0].get_legend_handles_labels()
        fig.legend(h0, l0, fontsize=6, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f'C {tag}  {order}  f64(solid) vs f32(dashed)  β={beta:g}  {grid} N={nn}{smooth_note}  backend={backend}', fontsize=8, y=1.06)
        fig.tight_layout()
        path = os.path.join(outdir, f'f32_{fname}_{order}_{grid}_b{beta:g}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths.append(path)
    return paths, rows


def format_f32_table(rows_by_shell, nn, grid, beta, backend):
    hdr = f'f32 vs f64  β={beta:g}  {grid} N={nn}  backend={backend}\n'
    hdr += f'{"shell":<6} {"order":<8} {"interp":<6} {"tang":<10} {"|ΔR\'|_f64":>11} {"|ΔR\'|_f32":>11} {"r\'(Å)":>8}\n'
    hdr += '-' * 72 + '\n'
    lines = [hdr]
    for shell, rows in rows_by_shell:
        for order, interp, tang, e64, e32, rD in rows:
            t = 'quad' if tang == 'quadrature' else 'ana'
            lines.append(f'{shell:<6} {order:<8} {interp:<6} {t:<10} {e64:11.3e} {e32:11.3e} {rD:8.4f}\n')
    return ''.join(lines)


def cmd_f32(args):
    os.makedirs(args.outdir, exist_ok=True)
    mol = make_mol(args.mol)
    r_dense = _r_dense_grid(args.r0_ang, args.rmax_ang)
    beta = float(args.beta_list[0])
    nn = _n_nodes(args.r0_ang, args.du, args.rmax_ang, beta, args.grid, args.n_nodes)
    backend = args.backend
    log(f'f32  β={beta:g}  {args.grid} N={nn}  backend={backend}  smooth={args.error_smooth}')
    if backend == 'cl':
        from pyscf.OpenCL import init_device
        init_device(quiet=False)
    all_rows = []
    for tag, ib, ic in carbon_radial_classes(mol):
        paths, rows = plot_f32_shell(mol, tag, ib, ic, r_dense, args.r0_ang, args.du, args.rmax_ang, beta, args.outdir, grid=args.grid, n_nodes=nn, origin_knot=args.origin_knot, smooth_window=args.error_smooth, backend=backend)
        all_rows.append((tag, rows))
        log(f'{tag}: ' + '  '.join(f'{r[0][0]}{r[1][0]}{"q" if r[2]=="quadrature" else "a"} f64={r[3]:.1e} f32={r[4]:.1e}' for r in rows))
        for path in paths:
            log(f'  {path}')
    table = format_f32_table(all_rows, nn, args.grid, beta, backend)
    tpath = os.path.join(args.outdir, f'f32_report_{args.grid}_b{beta:g}.txt')
    with open(tpath, 'w') as f:
        f.write(table)
    log(f'\n{table}')
    log(f'table: {tpath}')


def format_matrix_table(rows_by_shell, nn, grid, beta):
    hdr = f'matrix  β={beta:g}  {grid} N={nn}\n'
    hdr += f'{"shell":<6} {"order":<8} {"interp":<6} {"tang":<10} {"|ΔR|":>11} {"|ΔR\'|":>11} {"r\'(Å)":>8}\n'
    hdr += '-' * 72 + '\n'
    lines = [hdr]
    for shell, rows in rows_by_shell:
        for order, interp, tang, eR, eD, rD in rows:
            t = 'quad' if tang == 'quadrature' else 'ana'
            lines.append(f'{shell:<6} {order:<8} {interp:<6} {t:<10} {eR:11.3e} {eD:11.3e} {rD:8.4f}\n')
    return ''.join(lines)


def cmd_matrix(args):
    os.makedirs(args.outdir, exist_ok=True)
    mol = make_mol(args.mol)
    r_dense = _r_dense_grid(args.r0_ang, args.rmax_ang)
    beta = float(args.beta_list[0])
    nn = _n_nodes(args.r0_ang, args.du, args.rmax_ang, beta, args.grid, args.n_nodes)
    log(f'matrix  β={beta:g}  {args.grid} N={nn}  cubic+quintic plots per shell  smooth={args.error_smooth}')
    gpath = plot_node_distribution(args.r0_ang, args.du, args.rmax_ang, [beta], args.outdir, grid=args.grid, n_nodes=nn, beta_ref=args.beta_ref)
    log(f'grid: {gpath}')
    all_rows = []
    for tag, ib, ic in carbon_radial_classes(mol):
        paths, rows = plot_matrix_shell(mol, tag, ib, ic, r_dense, args.r0_ang, args.du, args.rmax_ang, beta, args.outdir, grid=args.grid, n_nodes=nn, origin_knot=args.origin_knot, smooth_window=args.error_smooth)
        all_rows.append((tag, rows))
        log(f'{tag}: ' + '  '.join(f'{r[0][0]}{r[1][0]}{"q" if r[2]=="quadrature" else "a"}={r[4]:.1e}' for r in rows))
        for path in paths:
            log(f'  {path}')
    table = format_matrix_table(all_rows, nn, args.grid, beta)
    tpath = os.path.join(args.outdir, f'matrix_report_{args.grid}_b{beta:g}.txt')
    with open(tpath, 'w') as f:
        f.write(table)
    log(f'\n{table}')
    log(f'table: {tpath}')


def cmd_compare(args):
    os.makedirs(args.outdir, exist_ok=True)
    mol = make_mol(args.mol)
    r_dense = _r_dense_grid(args.r0_ang, args.rmax_ang)
    nn = _n_nodes(args.r0_ang, args.du, args.rmax_ang, args.beta_ref, args.grid, args.n_nodes)
    tang = _resolve_tangents(args.order, args.tangents, args.fit)
    log(f'{args.order}/{tang}  separate u/r shell plots  N={nn}')
    for tag, ib, ic in carbon_radial_classes(mol):
        for mode in ('u', 'r'):
            path, rows = plot_shell_beta_sweep(mol, tag, ib, ic, r_dense, args.r0_ang, args.du, args.rmax_ang, args.beta_list, args.outdir, grid=args.grid, n_nodes=nn, interp_space=mode, order=args.order, tangents=args.tangents, fit=args.fit, origin_knot=args.origin_knot)
            log(f'{tag} {mode}: ' + '  '.join(f'β={b:g} |ΔR\'|={ed:.1e}' for b, _, _, ed in rows))
            log(f'  {path}')


def cmd_grid(args):
    os.makedirs(args.outdir, exist_ok=True)
    path = plot_node_distribution(args.r0_ang, args.du, args.rmax_ang, args.beta_list, args.outdir, grid=args.grid, n_nodes=args.n_nodes, beta_ref=args.beta_ref)
    log(f'wrote {path}')
    r0 = args.r0_ang / ANG
    nn = _n_nodes(args.r0_ang, args.du, args.rmax_ang, args.beta_ref, args.grid, args.n_nodes)
    for beta in args.beta_list:
        g = node_r_distribution(args.r0_ang, args.du, args.rmax_ang, beta, grid=args.grid, n_nodes=nn, beta_ref=args.beta_ref)
        log(f'  β={beta:g}  n_nodes={g["n"]}')


def _add_common(ap):
    ap.add_argument('--outdir', default=_DEFAULT_OUT)
    ap.add_argument('--r0-ang', type=float, default=0.002)
    ap.add_argument('--rmax-ang', type=float, default=8.0)
    ap.add_argument('--du', type=float, default=0.04, help='log-grid step; also sets reference N for power grid')
    ap.add_argument('--beta-ref', type=float, default=1.0, help='β at which reference node count is taken (power grid)')
    ap.add_argument('--beta-list', type=float, nargs='+', default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument('--grid', choices=['power', 'uniform', 'log'], default='power', help='power: β clusters origin; uniform: equal Δr; log: uniform du')
    ap.add_argument('--n-nodes', type=int, default=None, help='override node count (power grid)')
    ap.add_argument('--order', choices=['cubic', 'quintic'], default=DEFAULT_ORDER)
    ap.add_argument('--tangents', choices=['analytic', 'quadrature'], default=None, help='knot derivatives: analytic GTO or LSQ quadrature')
    ap.add_argument('--fit', choices=['analytic', 'quadrature', 'exact'], default=None, help='deprecated alias for --tangents (exact→analytic)')
    ap.add_argument('--origin-knot', action=argparse.BooleanOptionalAction, default=True, help='prepend analytic knot at r=0 (half-line boundary fix)')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Hermite radial spline study')
    sub = parser.add_subparsers(dest='cmd', required=True)

    ap_c = sub.add_parser('carbon', help='per-shell C radial error vs β')
    ap_c.add_argument('--mol', default='benzene', choices=['benzene', 'water'])
    ap_c.add_argument('--interp', choices=['u', 'r'], default='u')
    _add_common(ap_c)
    ap_c.set_defaults(func=cmd_carbon)

    ap_cmp = sub.add_parser('compare', help='u-mode vs r-mode on same figure')
    ap_cmp.add_argument('--mol', default='benzene', choices=['benzene', 'water'])
    _add_common(ap_cmp)
    ap_cmp.set_defaults(func=cmd_compare)

    ap_rep = sub.add_parser('report', help='grid + table + beta curves + per-shell compare')
    ap_rep.add_argument('--mol', default='benzene', choices=['benzene', 'water'])
    _add_common(ap_rep)
    ap_rep.set_defaults(func=cmd_report)

    ap_g = sub.add_parser('grid', help='plot r_i vs node index')
    _add_common(ap_g)
    ap_g.set_defaults(func=cmd_grid)

    ap_m = sub.add_parser('matrix', help='4-way overlay per order: interp × tangents')
    ap_m.add_argument('--mol', default='benzene', choices=['benzene', 'water'])
    ap_m.add_argument('--error-smooth', type=int, default=5, metavar='N', help='sliding max window for error curves (0=off)')
    _add_common(ap_m)
    ap_m.set_defaults(func=cmd_matrix, beta_list=[1.0])

    ap_f32 = sub.add_parser('f32', help='f64 vs OpenCL float32 spline error')
    ap_f32.add_argument('--mol', default='benzene', choices=['benzene', 'water'])
    ap_f32.add_argument('--backend', choices=['cl', 'cpu'], default='cl', help='cl=OpenCL GPU kernel; cpu=numpy f32 replay')
    ap_f32.add_argument('--error-smooth', type=int, default=5, metavar='N', help='sliding max window for error curves (0=off)')
    _add_common(ap_f32)
    ap_f32.set_defaults(func=cmd_f32, beta_list=[1.0])

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
