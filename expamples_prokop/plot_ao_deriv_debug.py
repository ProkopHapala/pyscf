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
    nao_cart = mol.nao_cart()
    r0_ang = 0.002
    du = 0.02
    rmax_ang = 8.0

    gpu = OpenCLAOHermiteEvaluator(mol, r0_ang=r0_ang, du=du, rmax_ang=rmax_ang, midpoint_fit=True)

    # === 1D radial scan along x from O nucleus ===
    N1D = 500
    r_max_bohr = 4.0 / ANG  # 4 Angstrom
    xs = np.linspace(0.001, r_max_bohr, N1D)
    coords_1d = np.zeros((N1D, 3))
    coords_1d[:, 0] = xs  # scan along x, y=z=0

    ref_val = mol.eval_gto('GTOval_sph', coords_1d).astype(np.float64)
    ref_d1 = mol.eval_gto('GTOval_sph_deriv1', coords_1d).astype(np.float64)  # (4, N, nao)
    gpu_val = gpu.eval_sph(coords_1d).astype(np.float64)
    gpu_d1 = gpu.eval_sph_deriv1(coords_1d).astype(np.float64)  # (4, N, nao)

    r_ang = xs * ANG

    # Pick a few representative AO indices to plot
    # s-type on O (index 0), p-type on O, d-type on O
    ao_indices = [0, 1, 2, 3, 4, min(5, nao-1)]
    ao_labels = [f'AO[{i}]' for i in ao_indices]

    fig, axes = plt.subplots(len(ao_indices), 4, figsize=(20, 3*len(ao_indices)), squeeze=False)
    for row, (iao, label) in enumerate(zip(ao_indices, ao_labels)):
        # Value
        ax = axes[row, 0]
        ax.plot(r_ang, ref_val[:, iao], 'b-', lw=1.5, label='PySCF')
        ax.plot(r_ang, gpu_val[:, iao], 'r--', lw=1, label='GPU Hermite')
        ax.set_title(f'{label} value')
        ax.set_xlabel('r (Ang)')
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

        # dx derivative
        ax = axes[row, 1]
        ax.plot(r_ang, ref_d1[1, :, iao], 'b-', lw=1.5, label='PySCF dx')
        ax.plot(r_ang, gpu_d1[1, :, iao], 'r--', lw=1, label='GPU dx')
        ax.set_title(f'{label} d/dx')
        ax.set_xlabel('r (Ang)')
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

        # dy derivative
        ax = axes[row, 2]
        ax.plot(r_ang, ref_d1[2, :, iao], 'b-', lw=1.5, label='PySCF dy')
        ax.plot(r_ang, gpu_d1[2, :, iao], 'r--', lw=1, label='GPU dy')
        ax.set_title(f'{label} d/dy')
        ax.set_xlabel('r (Ang)')
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

        # dz derivative
        ax = axes[row, 3]
        ax.plot(r_ang, ref_d1[3, :, iao], 'b-', lw=1.5, label='PySCF dz')
        ax.plot(r_ang, gpu_d1[3, :, iao], 'r--', lw=1, label='GPU dz')
        ax.set_title(f'{label} d/dz')
        ax.set_xlabel('r (Ang)')
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig('debug/ao_deriv_1d_scan.png', dpi=150)
    print('Saved debug/ao_deriv_1d_scan.png')

    # === Error plots for 1D scan ===
    fig, axes = plt.subplots(len(ao_indices), 4, figsize=(20, 3*len(ao_indices)), squeeze=False)
    for row, (iao, label) in enumerate(zip(ao_indices, ao_labels)):
        ax = axes[row, 0]
        err = np.abs(gpu_val[:, iao] - ref_val[:, iao])
        ax.semilogy(r_ang, err, 'k-', lw=1)
        ax.set_title(f'{label} |value error|')
        ax.set_xlabel('r (Ang)')
        ax.tick_params(labelsize=8)

        for c, name in enumerate(['dx', 'dy', 'dz'], start=1):
            ax = axes[row, c]
            err = np.abs(gpu_d1[c, :, iao] - ref_d1[c, :, iao])
            ax.semilogy(r_ang, err, 'k-', lw=1)
            ax.set_title(f'{label} |d/d{name} error|')
            ax.set_xlabel('r (Ang)')
            ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig('debug/ao_deriv_1d_error.png', dpi=150)
    print('Saved debug/ao_deriv_1d_error.png')

    # === 2D heatmap in z=0 plane ===
    N2D = 200
    extent_ang = 3.0  # +/- 3 Angstrom
    x2d = np.linspace(-extent_ang, extent_ang, N2D) / ANG
    y2d = np.linspace(-extent_ang, extent_ang, N2D) / ANG
    X, Y = np.meshgrid(x2d, y2d)
    coords_2d = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    r2d_ang = np.sqrt(X**2 + Y**2) * ANG

    ref2d_val = mol.eval_gto('GTOval_sph', coords_2d).astype(np.float64)
    ref2d_d1 = mol.eval_gto('GTOval_sph_deriv1', coords_2d).astype(np.float64)
    gpu2d_val = gpu.eval_sph(coords_2d).astype(np.float64)
    gpu2d_d1 = gpu.eval_sph_deriv1(coords_2d).astype(np.float64)

    # Pick one AO for 2D visualization
    iao_2d = 1  # p-type
    components = [('value', ref2d_val, gpu2d_val),
                  ('d/dx', ref2d_d1[1], gpu2d_d1[1]),
                  ('d/dy', ref2d_d1[2], gpu2d_d1[2]),
                  ('d/dz', ref2d_d1[3], gpu2d_d1[3])]

    fig, axes = plt.subplots(len(components), 3, figsize=(15, 4*len(components)))
    for row, (name, ref, gpu_arr) in enumerate(components):
        ref_map = ref[:, iao_2d].reshape(N2D, N2D)
        gpu_map = gpu_arr[:, iao_2d].reshape(N2D, N2D)
        err_map = np.abs(gpu_map - ref_map)

        vmax = max(np.abs(ref_map).max(), np.abs(gpu_map).max(), 1e-12)
        ax = axes[row, 0]
        im = ax.imshow(ref_map, extent=[-extent_ang, extent_ang, -extent_ang, extent_ang], origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'PySCF AO[{iao_2d}] {name}')
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[row, 1]
        im = ax.imshow(gpu_map, extent=[-extent_ang, extent_ang, -extent_ang, extent_ang], origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'GPU AO[{iao_2d}] {name}')
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[row, 2]
        emax = max(err_map.max(), 1e-16)
        im = ax.imshow(err_map, extent=[-extent_ang, extent_ang, -extent_ang, extent_ang], origin='lower', cmap='hot', vmin=0, vmax=emax)
        ax.set_title(f'|error| {name} (max={emax:.2e})')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig('debug/ao_deriv_2d_heatmap.png', dpi=150)
    print('Saved debug/ao_deriv_2d_heatmap.png')

    # === Radial-only derivative check ===
    # Scan along x, look at s-type AO (index 0) where angular part is constant
    # So dAO/dx = dR/dr * x/r, dAO/dy = 0, dAO/dz = 0
    # This isolates the radial derivative
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    iao_s = 0  # s-type on O

    ax = axes[0]
    ax.plot(r_ang, ref_d1[1, :, iao_s], 'b-', lw=2, label='PySCF d/dx')
    ax.plot(r_ang, gpu_d1[1, :, iao_s], 'r--', lw=1.5, label='GPU d/dx')
    ax.set_title(f'AO[{iao_s}] (s-type) d/dx\n(radial derivative * x/r)')
    ax.set_xlabel('r (Ang)')
    ax.legend()
    ax.tick_params(labelsize=8)

    ax = axes[1]
    ax.plot(r_ang, ref_d1[2, :, iao_s], 'b-', lw=2, label='PySCF d/dy')
    ax.plot(r_ang, gpu_d1[2, :, iao_s], 'r--', lw=1.5, label='GPU d/dy')
    ax.set_title(f'AO[{iao_s}] (s-type) d/dy\n(should be ~0 along x-axis)')
    ax.set_xlabel('r (Ang)')
    ax.legend()
    ax.tick_params(labelsize=8)

    # Also plot the radial derivative itself: dR/dr = dAO/dx / (x/r) = dAO/dx * r/x
    mask = np.abs(xs) > 1e-10
    dr_dr_ref = ref_d1[1, mask, iao_s] * (np.sqrt(xs[mask]**2) / xs[mask])
    dr_dr_gpu = gpu_d1[1, mask, iao_s] * (np.sqrt(xs[mask]**2) / xs[mask])
    ax = axes[2]
    ax.plot(r_ang[mask], dr_dr_ref, 'b-', lw=2, label='PySCF dR/dr')
    ax.plot(r_ang[mask], dr_dr_gpu, 'r--', lw=1.5, label='GPU dR/dr')
    ax.set_title('Radial derivative dR/dr\n(extracted from d/dx)')
    ax.set_xlabel('r (Ang)')
    ax.legend()
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig('debug/ao_deriv_radial_isolated.png', dpi=150)
    print('Saved debug/ao_deriv_radial_isolated.png')

    # === Print summary stats ===
    print('\n=== Error summary (1D scan, all AOs) ===')
    for c, name in enumerate(['value', 'd/dx', 'd/dy', 'd/dz']):
        if c == 0:
            err = np.abs(gpu_val - ref_val)
        else:
            err = np.abs(gpu_d1[c] - ref_d1[c])
        print(f'  {name:6s}: max_abs={err.max():.6e}  rms={np.sqrt(np.mean(err**2)):.6e}')

    print('\n=== Error summary (2D plane, all AOs) ===')
    for c, name in enumerate(['value', 'd/dx', 'd/dy', 'd/dz']):
        if c == 0:
            err = np.abs(gpu2d_val - ref2d_val)
        else:
            err = np.abs(gpu2d_d1[c] - ref2d_d1[c])
        print(f'  {name:6s}: max_abs={err.max():.6e}  rms={np.sqrt(np.mean(err**2)):.6e}')


if __name__ == '__main__':
    import os
    os.makedirs('debug', exist_ok=True)
    main()
