'''xc_path_modes.py — SSOT for OpenCL XC execution path labels used in benchmarks.

Maps scan CLI keys (cpu, gpu_otf, gpu_coalesced, gpu_radial, gpu_gto, gpu_full, gpu_otf_radial_vmat, gpu_otf_radial_vmat_splitk) to
gpu_profiles.py presets, kernel names, and plot colors. gpu_full shares ρ/vmat kernels with
gpu_otf; differs only in GPU J and conv_tol 1e-6 — document this when interpreting scan scatter.
'''
from pyscf.OpenCL.gpu_profiles import GPU_PROFILES, get_profile

XC_PATH_MODES = {
    'cpu': {
        'profile': None,
        'short': 'CPU libxc',
        'rho': 'CPU eval_ao → libxc',
        'vmat': 'CPU contract wv×AO',
        'j': 'CPU RI-J',
        'ao_setup': 'PySCF GTO each cycle',
        'scf_tol': '1e-8',
    },
    'gpu_otf': {
        'profile': 'production_otf',
        'short': 'Hermite OTF',
        'rho': 'rho_gga_pair — Hermite radial eval in-kernel, atom-pair gather',
        'vmat': 'vmat_gga_pair — same AO source as rho',
        'j': 'CPU RI-J',
        'ao_setup': 'Hermite tables only (~0.2 MB); no χ on GPU',
        'scf_tol': '1e-8',
    },
    'gpu_coalesced': {
        'profile': 'production_coalesced',
        'short': 'Precomp coalesced',
        'rho': 'rho_gga_precomp_coalesced_pair — gather χ[iAO,iG]',
        'vmat': 'vmat_gga_precomp_coalesced_pair',
        'j': 'CPU RI-J',
        'ao_setup': 'Hermite χ[iAO,iG] upload at setup',
        'scf_tol': '1e-8',
    },
    'gpu_radial': {
        'profile': 'production_radial',
        'short': 'Radial precomp',
        'rho': 'rho_gga_radial_precomp_pair — R,dR on grid, no full χ',
        'vmat': 'vmat_gga_radial_precomp_pair — R,dR gather + shell unfold',
        'j': 'CPU RI-J',
        'ao_setup': 'build_radial_on_grid_tiled at setup',
        'scf_tol': '1e-8',
    },
    'gpu_otf_radial_vmat': {
        'profile': 'production_otf_radial_vmat',
        'short': 'OTF ρ + radial vmat',
        'rho': 'rho_gga_tiled — OTF Hermite (same as gpu_otf)',
        'vmat': 'vmat_gga_radial_precomp_pair — R,dR gather (no Hermite in vmat)',
        'j': 'CPU RI-J',
        'ao_setup': 'Hermite tables + build_radial_on_grid_tiled at setup',
        'scf_tol': '1e-8',
    },
    'gpu_otf_radial_vmat_splitk': {
        'profile': 'production_otf_radial_vmat_splitk',
        'short': 'OTF ρ + split-K vmat',
        'rho': 'rho_gga_tiled — OTF Hermite',
        'vmat': 'vmat_gga_radial_precomp_pair_splitk + reduce_split_vmat',
        'j': 'CPU RI-J',
        'ao_setup': 'Hermite tables + radial R,dR; WGS=128 compile for split-K',
        'scf_tol': '1e-8',
    },
    'gpu_gto': {
        'profile': 'production_gto_exact',
        'short': 'Exact GTO χ',
        'rho': 'rho_gga_precomp_coalesced_pair — exact PySCF GTO χ',
        'vmat': 'vmat_gga_precomp_coalesced_pair',
        'j': 'CPU RI-J',
        'ao_setup': 'CPU eval_ao → upload exact χ (slow, reference)',
        'scf_tol': '1e-8',
    },
    'gpu_full': {
        'profile': 'fast_full_gpu',
        'short': 'OTF fast (GPU J)',
        'rho': 'same kernels as gpu_otf (rho_gga_pair)',
        'vmat': 'same kernels as gpu_otf (vmat_gga_pair)',
        'j': 'GPU RI-J (f32 tiled GEMM)',
        'ao_setup': 'same as gpu_otf',
        'scf_tol': '1e-6 (relaxed — main accuracy difference vs gpu_otf)',
    },
}

MODE_COLORS = {
    'cpu': '#1f77b4', 'gpu_otf': '#ff7f0e', 'gpu_coalesced': '#9467bd',
    'gpu_radial': '#d62728', 'gpu_gto': '#8c564b', 'gpu_full': '#2ca02c',
    'gpu_otf_radial_vmat': '#17becf',
    'gpu_otf_radial_vmat_splitk': '#bcbd22',
}

SCAN_MODE_KEYS = ['cpu', 'gpu_otf', 'gpu_coalesced', 'gpu_radial', 'gpu_gto', 'gpu_full']


def mode_label(key):
    m = XC_PATH_MODES[key]
    return m['short']


def mode_plot_label(key):
    m = XC_PATH_MODES[key]
    if key == 'cpu':
        return 'CPU libxc (ref)'
    return f"GPU {m['short']}"


def path_description(key):
    m = XC_PATH_MODES[key]
    lines = [f"{key} ({m['short']})", f"  ρ: {m['rho']}", f"  vmat: {m['vmat']}", f"  J: {m['j']}", f"  AO setup: {m['ao_setup']}", f"  SCF tol: {m['scf_tol']}"]
    if key != 'cpu':
        acc = get_profile(m['profile']).get('accuracy', {})
        if acc.get('vxc_max_vs_cpu'):
            lines.append(f"  expected |vxc| vs CPU: {acc['vxc_max_vs_cpu']}")
        if acc.get('energy_note'):
            lines.append(f"  note: {acc['energy_note']}")
    return '\n'.join(lines)


def gpu_full_vs_otf_note():
    return (
        'gpu_full vs gpu_otf: IDENTICAL ρ/vmat kernels (Hermite OTF pair-gather). '
        'Differences are (1) Coulomb J on GPU f32 vs CPU DF, (2) SCF conv_tol 1e-6 vs 1e-8. '
        'Energy scatter on scans comes mainly from incomplete SCF convergence, not different XC integration.'
    )
