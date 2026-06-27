import argparse
import time
import numpy as np
from pyscf import gto
from pyscf.data import nist
from pyscf.OpenCL.radial_hermite import MappedHermiteRadialBasis
from pyscf.OpenCL.ao_hermite import OpenCLAOHermiteEvaluator


ANG = nist.BOHR


def make_points(mol, npts, seed):
    rng = np.random.default_rng(seed)
    centers = mol.atom_coords()
    aidx = rng.integers(0, mol.natm, size=npts)
    direction = rng.normal(size=(npts, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    radius = (4.0 * rng.random(npts) ** 0.7) / ANG
    return centers[aidx] + direction * radius[:, None]


def metrics(ref, val, mask_eps=1e-8):
    err = np.asarray(val) - np.asarray(ref)
    mask = np.abs(ref) > mask_eps
    rel = np.max(np.abs(err[mask]) / np.maximum(np.abs(ref[mask]), 1e-30)) if np.any(mask) else 0.0
    return np.max(np.abs(err)), np.sqrt(np.mean(err * err)), rel


def print_metrics(name, ref, val):
    ma, rms, rel = metrics(ref, val)
    print(f'{name:32s} max_abs={ma:.6e} rms={rms:.6e} max_rel_sig={rel:.6e}')
    return ma, rms, rel


def run(args):
    mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', basis=args.basis, unit='Angstrom', cart=False, verbose=0)
    coords = make_points(mol, args.npts, args.seed)
    centers = mol.atom_coords()
    rmax_ang = np.max(np.linalg.norm(coords[:, None, :] - centers[None, :, :], axis=2)) * ANG + 0.2
    rmax_ang = max(rmax_ang, args.rmax_ang)
    print(f'basis={args.basis} npts={args.npts} nao_sph={mol.nao_nr()} nao_cart={mol.nao_cart()} nbas={mol.nbas}')
    print(f'r0={args.r0_ang} Ang du={args.du} rmax={rmax_ang:.3f} Ang')

    t0 = time.perf_counter()
    ref_cart = mol.eval_gto('GTOval_cart', coords).astype(np.float32)
    ref_sph = mol.eval_gto('GTOval_sph', coords).astype(np.float32)
    ref_sph_d1 = mol.eval_gto('GTOval_sph_deriv1', coords).astype(np.float32)
    print(f'PySCF eval_gto time: {time.perf_counter() - t0:.4f}s')

    t0 = time.perf_counter()
    cpu_plan = MappedHermiteRadialBasis(mol, r0_ang=args.r0_ang, du=args.du, rmax_ang=rmax_ang, midpoint_fit=True)
    cpu_cart = cpu_plan.eval_cart(coords)
    cpu_sph = cpu_plan.eval_sph(coords)
    print(f'CPU Hermite build+eval time: {time.perf_counter() - t0:.4f}s nrad={cpu_plan.nrad}')
    print_metrics('CPU Hermite cart vs PySCF', ref_cart, cpu_cart)
    print_metrics('CPU Hermite sph vs PySCF', ref_sph, cpu_sph)

    t0 = time.perf_counter()
    gpu_eval = OpenCLAOHermiteEvaluator(mol, r0_ang=args.r0_ang, du=args.du, rmax_ang=rmax_ang, midpoint_fit=True)
    gpu_cart = gpu_eval.eval_cart(coords)
    gpu_sph = gpu_eval.eval_sph(coords)
    gpu_sph_d1 = gpu_eval.eval_sph_deriv1(coords)
    print(f'GPU Hermite build+eval time: {time.perf_counter() - t0:.4f}s')
    e_gc = print_metrics('GPU cart vs CPU Hermite', cpu_cart, gpu_cart)
    e_gs = print_metrics('GPU sph vs CPU Hermite', cpu_sph, gpu_sph)
    e_ps = print_metrics('GPU sph vs PySCF', ref_sph, gpu_sph)
    e_d1 = print_metrics('GPU sph deriv1 vs PySCF', ref_sph_d1, gpu_sph_d1)
    if e_gc[0] > args.gpu_tol:
        raise SystemExit(f'GPU cart interpolation mismatch: {e_gc[0]} > {args.gpu_tol}')
    if e_gs[0] > args.gpu_tol:
        raise SystemExit(f'GPU sph interpolation mismatch: {e_gs[0]} > {args.gpu_tol}')
    if e_ps[0] > args.pyscf_tol:
        raise SystemExit(f'GPU Hermite AO error vs PySCF too large: {e_ps[0]} > {args.pyscf_tol}')
    if e_d1[0] > args.deriv_tol:
        raise SystemExit(f'GPU Hermite AO derivative error vs PySCF too large: {e_d1[0]} > {args.deriv_tol}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--basis', default='cc-pvdz')
    parser.add_argument('--npts', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--r0-ang', type=float, default=0.002)
    parser.add_argument('--du', type=float, default=0.02)
    parser.add_argument('--rmax-ang', type=float, default=8.0)
    parser.add_argument('--gpu-tol', type=float, default=2e-4)
    parser.add_argument('--pyscf-tol', type=float, default=3e-3)
    parser.add_argument('--deriv-tol', type=float, default=2e-2)
    run(parser.parse_args())
