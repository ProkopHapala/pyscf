import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyscf import gto
from pyscf.data import nist
from pyscf.OpenCL.ao_hermite import OpenCLAOHermiteEvaluator

ANG = nist.BOHR

def main():
    mol = gto.M(atom='O 0 0 0; H 0 0 0.957; H 0 0.957 0', basis='cc-pvdz', unit='Angstrom', cart=False, verbose=0)
    nao = mol.nao_nr()
    r0_ang = 0.002
    du = 0.02
    rmax_ang = 8.0

    gpu = OpenCLAOHermiteEvaluator(mol, r0_ang=r0_ang, du=du, rmax_ang=rmax_ang, midpoint_fit=True)

    # Dense 1D scan along x
    N = 2000
    r_max_bohr = 4.0 / ANG
    xs = np.linspace(0.001, r_max_bohr, N)
    coords = np.zeros((N, 3))
    coords[:, 0] = xs

    ref_d1 = mol.eval_gto('GTOval_sph_deriv1', coords).astype(np.float64)
    gpu_d1 = gpu.eval_sph_deriv1(coords).astype(np.float64)
    r_ang = xs * ANG

    # For each AO, find where the max d/dx error occurs
    print('=== Per-AO d/dx error analysis (1D scan along x) ===')
    print(f'{"AO":>4s} {"ref_max":>12s} {"err_max":>12s} {"rel_err%":>10s} {"r_at_err":>10s} {"ref_at_err":>12s}')
    worst_ao = 0
    worst_err = 0
    for iao in range(nao):
        err = np.abs(gpu_d1[1, :, iao] - ref_d1[1, :, iao])
        ref = np.abs(ref_d1[1, :, iao])
        idx = np.argmax(err)
        rel = err[idx] / max(ref[idx], 1e-30) * 100
        if err[idx] > worst_err:
            worst_err = err[idx]
            worst_ao = iao
        if err[idx] > 1e-5 or rel > 1:
            print(f'{iao:4d} {ref.max():12.6e} {err[idx]:12.6e} {rel:10.2f}% {r_ang[idx]:10.4f} {ref_d1[1, idx, iao]:12.6e}')

    print(f'\nWorst AO: {worst_ao} with max_abs_err={worst_err:.6e}')

    # Plot the worst AO in detail
    iao = worst_ao
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # d/dx value comparison
    ax = axes[0, 0]
    ax.plot(r_ang, ref_d1[1, :, iao], 'b-', lw=2, label='PySCF')
    ax.plot(r_ang, gpu_d1[1, :, iao], 'r--', lw=1.5, label='GPU')
    ax.set_title(f'AO[{iao}] d/dx (worst case)')
    ax.set_xlabel('r (Ang)')
    ax.legend()

    # d/dx error
    ax = axes[0, 1]
    err = np.abs(gpu_d1[1, :, iao] - ref_d1[1, :, iao])
    ax.semilogy(r_ang, err, 'k-', lw=1)
    ax.set_title(f'AO[{iao}] |d/dx error|')
    ax.set_xlabel('r (Ang)')

    # Zoom near nucleus
    mask = r_ang < 1.0
    ax = axes[1, 0]
    ax.plot(r_ang[mask], ref_d1[1, mask, iao], 'b-', lw=2, label='PySCF')
    ax.plot(r_ang[mask], gpu_d1[1, mask, iao], 'r--', lw=1.5, label='GPU')
    ax.set_title(f'AO[{iao}] d/dx (zoom r<1 Ang)')
    ax.set_xlabel('r (Ang)')
    ax.legend()

    # Relative error
    ax = axes[1, 1]
    ref_abs = np.abs(ref_d1[1, :, iao])
    mask2 = ref_abs > 1e-10
    rel_err = np.full(N, np.nan)
    rel_err[mask2] = np.abs(gpu_d1[1, mask2, iao] - ref_d1[1, mask2, iao]) / ref_abs[mask2]
    ax.semilogy(r_ang, rel_err, 'g-', lw=1)
    ax.set_title(f'AO[{iao}] relative |d/dx error|')
    ax.set_xlabel('r (Ang)')
    ax.set_ylim(1e-8, 1e2)

    plt.tight_layout()
    plt.savefig('debug/ao_deriv_worst_case.png', dpi=150)
    print('Saved debug/ao_deriv_worst_case.png')

    # Also check: is the error worse for specific shell types?
    print('\n=== Error by angular momentum ===')
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        nctr = mol.bas_nctr(ib)
        nc = (l+1)*(l+2)//2
        # Find AO indices for this shell (spherical)
        # We'll just check all AOs and group by l
        pass

    # Simpler: check s-type (l=0) vs p-type (l=1) vs d-type (l=2)
    # by looking at which AOs have largest errors
    print('\n=== Top-10 worst d/dx errors across all AOs ===')
    all_errs = np.zeros(nao)
    all_rel = np.zeros(nao)
    all_r = np.zeros(nao)
    for iao in range(nao):
        err = np.abs(gpu_d1[1, :, iao] - ref_d1[1, :, iao])
        ref_abs = np.abs(ref_d1[1, :, iao])
        idx = np.argmax(err)
        all_errs[iao] = err[idx]
        all_rel[iao] = err[idx] / max(ref_abs[idx], 1e-30)
        all_r[iao] = r_ang[idx]

    top10 = np.argsort(all_errs)[::-1][:10]
    for rank, iao in enumerate(top10):
        print(f'  #{rank+1}: AO[{iao}] max_abs={all_errs[iao]:.6e} rel={all_rel[iao]:.4f} at r={all_r[iao]:.4f} Ang')

    # === Check if error correlates with r near nucleus ===
    # Aggregate error across all AOs as function of r
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    total_err = np.max(np.abs(gpu_d1[1] - ref_d1[1]), axis=1)  # max over AOs at each grid point
    ax.semilogy(r_ang, total_err, 'b-', lw=1)
    ax.set_title('Max d/dx error across all AOs vs radius')
    ax.set_xlabel('r (Ang)')
    ax.set_ylabel('max |error|')
    ax.axvline(x=r0_ang, color='r', ls='--', label=f'r0={r0_ang} Ang')
    ax.legend()
    plt.tight_layout()
    plt.savefig('debug/ao_deriv_error_vs_r.png', dpi=150)
    print('Saved debug/ao_deriv_error_vs_r.png')

    # Copy to expamples_prokop for viewing
    import shutil
    for f in ['ao_deriv_worst_case.png', 'ao_deriv_error_vs_r.png']:
        shutil.copy(f'debug/{f}', f'expamples_prokop/{f}')


if __name__ == '__main__':
    import os
    os.makedirs('debug', exist_ok=True)
    main()
