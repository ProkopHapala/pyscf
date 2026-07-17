'''Named GPU/CPU execution profiles for OpenCL DFT (RKS + DF).

Cookbook: doc/opencl_gpu_paths_cookbook.md
DF storage / benchmark hygiene: doc/df_storage_and_benchmark_hygiene.md

Usage:
    mf = dft.RKS(mol, xc='PBE').density_fit()
    from pyscf.OpenCL.gpu_profiles import apply_gpu_profile, prepare_df_for_scf
    apply_gpu_profile(mf, 'production_otf')
    # or CPU-only explicit DF prep with forced RAM tensor:
    # prepare_df_for_scf(mf, storage='incore', require_incore=True)
    mf.kernel()

Open issues / caveats:
- DF.storage='auto' can silently spill to HDF5 when AO/GPU buffers eat max_memory;
  benchmarks should use storage='incore' + require_incore=True.
- prepare_df_for_scf only hoists lifetime; it does not change contraction math.
'''
from __future__ import annotations

from dataclasses import replace

# Profile record keys:
#   label, description, mf_backend, df_backend, xc_path, setup_kw, scf_kw, accuracy

GPU_PROFILES = {
    'cpu_reference': {
        'label': 'cpu_reference',
        'description': 'PySCF CPU libxc + CPU DF J; parity reference.',
        'mf_backend': 1,
        'df_backend': 1,
        'xc_path': None,
        'setup_kw': {},
        'scf_kw': {'conv_tol': 1e-8, 'conv_tol_grad': 1e-5},
        'accuracy': {
            'vxc_max_vs_cpu': 0.0,
            'energy_note': 'Reference.',
            'converges_default_scf': True,
        },
    },
    'debug_compare': {
        'label': 'debug_compare',
        'description': 'Run CPU and GPU XC each cycle; log nelec/exc/vxc errors (slow).',
        'mf_backend': 3,
        'df_backend': 3,
        'xc_path': 'onthefly',
        'setup_kw': {'xc_eval': 'gpu', 'gpu_xc': 'auto'},
        'scf_kw': {'conv_tol': 1e-8, 'conv_tol_grad': 1e-5, 'max_cycle': 5},
        'accuracy': {
            'vxc_max_vs_cpu': '~3e-6 typical',
            'energy_note': 'Not for production; max_cycle kept low.',
            'converges_default_scf': False,
        },
    },
    'debug_xc_libxc': {
        'label': 'debug_xc_libxc',
        'description': 'GPU rho/vmat + CPU libxc (rho D2H each cycle). XC parity debug.',
        'mf_backend': 2,
        'df_backend': 1,
        'xc_path': 'precomputed',
        'setup_kw': {'fused': 'coalesced', 'ao_proj': 'auto', 'xc_eval': 'cpu', 'gpu_xc': 'auto'},
        'scf_kw': {'conv_tol': 1e-8, 'conv_tol_grad': 1e-5},
        'accuracy': {
            'vxc_max_vs_cpu': '~3e-6 (f32 rho)',
            'energy_note': 'Matches cpu_reference when converged.',
            'converges_default_scf': True,
        },
    },
    'production_otf': {
        'label': 'production_otf',
        'description': 'Hermite OTF rho/vmat + GPU PBE; CPU DF J (f64), overlapped with GPU XC via mf.overlap_j_xc. Default for medium/large molecules. For faster veff XC use production_otf_radial_vmat.',
        'mf_backend': 2,
        'df_backend': 1,
        'xc_path': 'onthefly',
        'setup_kw': {'xc_eval': 'gpu', 'gpu_xc': 'auto'},
        'scf_kw': {'conv_tol': 1e-8, 'conv_tol_grad': 1e-5, 'overlap_j_xc': True},
        'accuracy': {
            'vxc_max_vs_cpu': '~3e-6',
            'energy_note': 'Converges; final E within ~1e-6 Ha of CPU.',
            'converges_default_scf': True,
        },
    },
    'production_otf_quintic': {
        'label': 'production_otf_quintic',
        'description': 'Hermite OTF quintic spline (2× coarser du, analytic GTO d²R); GPU PBE; CPU DF J.',
        'mf_backend': 2,
        'df_backend': 1,
        'xc_path': 'onthefly',
        'setup_kw': {'xc_eval': 'gpu', 'gpu_xc': 'auto', 'spline_order': 'quintic'},
        'scf_kw': {'conv_tol': 1e-8, 'conv_tol_grad': 1e-5},
        'accuracy': {
            'vxc_max_vs_cpu': '~3e-6 (compare vs cubic OTF)',
            'energy_note': 'Half radial table size vs cubic; parity test expamples_prokop/test_quintic_rho_otf.py',
            'converges_default_scf': True,
        },
    },
    'production_otf_radial_vmat': {
        'label': 'production_otf_radial_vmat',
        'description': 'OTF Hermite rho + radial-precomp vmat (R,dR gather); GPU PBE; CPU DF J.',
        'mf_backend': 2,
        'df_backend': 1,
        'xc_path': 'onthefly',
        'setup_kw': {'xc_eval': 'gpu', 'gpu_xc': 'auto', 'vmat_mode': 'radial_precomp'},
        'scf_kw': {'conv_tol': 1e-8, 'conv_tol_grad': 1e-5},
        'accuracy': {
            'vxc_max_vs_cpu': '~3e-6',
            'energy_note': 'Hybrid: OTF rho_gga_tiled + vmat_gga_radial_precomp_pair',
            'converges_default_scf': True,
        },
    },
    'production_otf_radial_vmat_splitk': {
        'label': 'production_otf_radial_vmat_splitk',
        'description': 'OTF Hermite rho + split-K radial-precomp vmat; GPU PBE; CPU DF J. WGS_VMAT=128 + splits=64 from benzene 1-neighborhood sweep.',
        'mf_backend': 2,
        'df_backend': 1,
        'xc_path': 'onthefly',
        'setup_kw': {'xc_eval': 'gpu', 'gpu_xc': 'auto', 'vmat_mode': 'radial_precomp', 'vmat_grid_splits': 64},
        'scf_kw': {'conv_tol': 1e-8, 'conv_tol_grad': 1e-5},
        'accuracy': {
            'vxc_max_vs_cpu': '~3e-5',
            'energy_note': 'Split grid dimension in vmat_gga_radial_precomp_pair_splitk; reduce partial vmat on GPU.',
            'converges_default_scf': True,
        },
    },
    'production_radial_screened': {
        'label': 'production_radial_screened',
        'description': 'NEW: radial-precomp R,dR + grid_screen active atoms/pairs for rho+vmat; GPU PBE; CPU DF J. Targets PTCDA-scale sparsity.',
        'mf_backend': 2,
        'df_backend': 1,
        'xc_path': 'onthefly',
        'setup_kw': {'xc_eval': 'gpu', 'gpu_xc': 'auto', 'rho_mode': 'radial_screened', 'vmat_mode': 'radial_screened'},
        'scf_kw': {'conv_tol': 1e-8, 'conv_tol_grad': 1e-5, 'overlap_j_xc': True},
        'accuracy': {
            'vxc_max_vs_cpu': 'verify vs OTF (~1e-5 expected; screen_eps dependent)',
            'energy_note': 'Kernels: rho_gga_radial_screened + vmat_gga_radial_screened_pair',
            'converges_default_scf': True,
        },
    },
    'production_coalesced': {
        'label': 'production_coalesced',
        'description': 'Precomp chi[iAO,iG] + coalesced rho/vmat + GPU PBE; CPU DF J. Best for small/fixed geometry.',
        'mf_backend': 2,
        'df_backend': 1,
        'xc_path': 'precomputed',
        'setup_kw': {'fused': 'coalesced', 'ao_proj': 'auto', 'xc_eval': 'gpu', 'gpu_xc': 'auto'},
        'scf_kw': {'conv_tol': 1e-8, 'conv_tol_grad': 1e-5},
        'accuracy': {
            'vxc_max_vs_cpu': '~3e-6 (Hermite chi) or ~2.6e-6 (GTO chi)',
            'energy_note': 'Hermite AO: same as OTF. Requires chi ~4*ncomp*nao*ngrids f32 bytes.',
            'converges_default_scf': True,
        },
    },
    'production_radial': {
        'label': 'production_radial',
        'description': 'Precomp R,dR only + radial rho; coalesced/Hermite chi for vmat; GPU PBE; CPU DF J.',
        'mf_backend': 2,
        'df_backend': 1,
        'xc_path': 'precomputed',
        'setup_kw': {'fused': 'radial_precomp', 'ao_proj': 'hermite_gpu', 'xc_eval': 'gpu', 'gpu_xc': 'auto'},
        'scf_kw': {'conv_tol': 1e-8, 'conv_tol_grad': 1e-5},
        'accuracy': {
            'vxc_max_vs_cpu': '~3e-6',
            'energy_note': 'Low chi memory; fast rho. vmat still uses Hermite chi.',
            'converges_default_scf': True,
        },
    },
    'production_gto_exact': {
        'label': 'production_gto_exact',
        'description': 'Exact PySCF GTO chi on grid (slow setup); coalesced + GPU PBE. Small molecules only.',
        'mf_backend': 2,
        'df_backend': 1,
        'xc_path': 'precomputed',
        'setup_kw': {'fused': 'coalesced', 'ao_proj': 'cpu', 'xc_eval': 'gpu', 'gpu_xc': 'auto'},
        'scf_kw': {'conv_tol': 1e-8, 'conv_tol_grad': 1e-5},
        'accuracy': {
            'vxc_max_vs_cpu': '~2.6e-6',
            'energy_note': 'Best rho/chi vs CPU GTO; setup ~CPU eval_ao.',
            'converges_default_scf': True,
        },
    },
    'fast_full_gpu': {
        'label': 'fast_full_gpu',
        'description': 'OTF + GPU PBE + GPU DF J. Fastest per cycle; relaxed SCF tolerances for f32.',
        'mf_backend': 2,
        'df_backend': 2,
        'xc_path': 'onthefly',
        'setup_kw': {'xc_eval': 'gpu', 'gpu_xc': 'auto'},
        'scf_kw': {'conv_tol': 1e-6, 'conv_tol_grad': 1e-4},
        'accuracy': {
            'vxc_max_vs_cpu': '~8e-6 per veff; ~7e-5 Ha energy offset possible',
            'energy_note': '~0.04 kcal/mol vs tight CPU; sufficient for many applications.',
            'converges_default_scf': True,
        },
    },
    'legacy_tiled_rowmajor': {
        'label': 'legacy_tiled_rowmajor',
        'description': 'Row-major chi[iG,iAO] + tiled precomp kernels. Prefer production_coalesced.',
        'mf_backend': 2,
        'df_backend': 1,
        'xc_path': 'precomputed',
        'setup_kw': {'fused': 'tiled', 'ao_proj': 'auto', 'xc_eval': 'gpu', 'gpu_xc': 'auto'},
        'scf_kw': {'conv_tol': 1e-8, 'conv_tol_grad': 1e-5},
        'accuracy': {
            'vxc_max_vs_cpu': '~3e-6',
            'energy_note': 'Correct but slow rho on GPU; use coalesced instead.',
            'converges_default_scf': True,
        },
    },
}

DEFAULT_PROFILE = 'production_otf'

# Synergy groups: subsets that should be chosen together
PATH_SYNERGY = {
    'full_gpu_xc_chain': {
        'rho': ('gpu',),
        'xc_eval': ('gpu',),
        'vmat': ('gpu',),
        'note': 'If rho on GPU with precomp path, keep wv on GPU (xc_eval=gpu).',
    },
    'hermite_otf': {
        'xc_path': ('onthefly',),
        'ao_setup': ('none',),
        'fused': (None,),
        'note': 'OTF evaluates AOs in rho/vmat kernels; no chi upload.',
    },
    'precomp_coalesced': {
        'xc_path': ('precomputed',),
        'fused': ('coalesced',),
        'ao_proj': ('auto', 'hermite_gpu', 'cpu'),
        'note': 'Needs chi on GPU at setup; coalesced layout matches vmat gather.',
    },
    'precomp_radial': {
        'xc_path': ('precomputed',),
        'fused': ('radial_precomp',),
        'ao_proj': ('hermite_gpu',),
        'note': 'R,dR on GPU for rho; Hermite chi still built for vmat.',
    },
}


def list_profiles():
    return {k: v['description'] for k, v in GPU_PROFILES.items()}


def get_profile(name):
    if name not in GPU_PROFILES:
        raise KeyError(f'Unknown GPU profile {name!r}; choose from {sorted(GPU_PROFILES)}')
    return GPU_PROFILES[name]


def apply_scf_kw(mf, scf_kw):
    for k, v in scf_kw.items():
        setattr(mf, k, v)


def prepare_df_for_scf(mf, storage=None, require_incore=False):
    '''Prepare invariant DF data and GPU buffers before ``mf.kernel()``.

    This does **not** invent a new DF algorithm. It only makes lifetime and
    storage policy explicit so the first SCF cycle is not polluted by a
    hidden ``df.build()`` (and so benchmarks do not silently flip to HDF5).

    Parameters
    ----------
    mf : SCF
        Must already have ``mf.with_df`` (``density_fit()``).
    storage : {None, 'auto', 'incore', 'outcore'}
        If set, assigns ``mf.with_df.storage`` before build. ``None`` leaves
        the current value (default ``'auto'``). See
        ``doc/df_storage_and_benchmark_hygiene.md``.
    require_incore : bool
        After build, raise ``RuntimeError`` unless ``_cderi`` is an in-RAM
        ndarray. Use in deterministic timing scripts.

    Returns
    -------
    mf
    '''
    dfobj = getattr(mf, 'with_df', None)
    if dfobj is None:
        return mf
    if storage is not None:
        dfobj.storage = storage
    if dfobj._cderi is None:
        dfobj.build()
    kind, detail, nbytes = dfobj.describe_cderi()
    if require_incore and kind != 'incore':
        raise RuntimeError(
            f'prepare_df_for_scf(require_incore=True) but _cderi is {kind} '
            f'({detail}). Raise max_memory, set storage="incore", or free '
            f'competing buffers. See doc/df_storage_and_benchmark_hygiene.md')
    if dfobj.backend & 2:
        from pyscf.OpenCL.df_jk import prepare_df_jk_plan
        mf._gpu_df_jk_plan = prepare_df_jk_plan(dfobj, mf.mol.nao_nr())
    mf._df_prepared = True
    mf._df_storage_kind = kind
    mf._df_cderi_detail = detail
    mf._df_cderi_nbytes = nbytes
    return mf


def assert_df_incore(mf, where='assert_df_incore'):
    '''Fail loud if DF tensor is missing or on disk (benchmark guard).'''
    dfobj = getattr(mf, 'with_df', None)
    if dfobj is None:
        raise RuntimeError(f'{where}: mf has no with_df')
    kind, detail, _ = dfobj.describe_cderi()
    if kind != 'incore':
        raise RuntimeError(
            f'{where}: expected incore _cderi, got {kind} ({detail}). '
            f'storage={getattr(dfobj, "storage", None)!r} '
            f'max_memory={dfobj.max_memory}. '
            f'See doc/df_storage_and_benchmark_hygiene.md')
    return kind, detail


def _ensure_splitk_tile_config(setup_kw, quiet=True):
    '''Recompile kernels with WGS_VMAT=128 for split-K pair vmat (benzene sweep winner).'''
    if setup_kw.get('vmat_grid_splits', 1) <= 1:
        return
    from pyscf.OpenCL import init_device, reset_opencl
    from pyscf.OpenCL.tile_config import get_active_tile_config, TileConfig
    tc = get_active_tile_config()
    target_wgs = 128
    if tc.WGS_VMAT == target_wgs:
        return
    reset_opencl()
    init_device(tile_config=replace(tc, WGS_VMAT=target_wgs), force_rebuild=True, quiet=quiet)


def apply_gpu_profile(mf, name=DEFAULT_PROFILE, setup=True, dm=None,
                      df_storage=None, require_df_incore=False):
    '''Configure mf and prepare invariant XC/DF state before SCF.

    With ``setup=True``, grids, GPU XC state, DF tensors, and GPU DF-J
    buffers are prepared before ``mf.kernel()``. Density-dependent XC/J/K
    contractions remain inside each SCF cycle.

    ``df_storage`` / ``require_df_incore`` are forwarded to
    :func:`prepare_df_for_scf` (see ``doc/df_storage_and_benchmark_hygiene.md``).
    '''
    prof = get_profile(name)
    mf.backend = prof['mf_backend']
    apply_scf_kw(mf, prof.get('scf_kw', {}))
    df_backend = prof.get('df_backend')
    if df_backend is not None and 'with_df' in mf.__dict__ and mf.with_df is not None:
        mf.with_df.backend = df_backend
    xc_path = prof.get('xc_path')
    if setup and xc_path and (prof['mf_backend'] & 2):
        mol = mf.mol
        mf.initialize_grids(mol, dm)
        setup_kw = dict(prof.get('setup_kw', {}))
        gpu_xc = setup_kw.pop('gpu_xc', 'auto')
        _ensure_splitk_tile_config(setup_kw)
        if xc_path == 'precomputed':
            from pyscf.OpenCL.xc_grid import setup_precomputed_gto
            mf._xc_gpu_plan = setup_precomputed_gto(
                mol, mf.grids, mf.xc, gpu_only=True, gpu_xc=gpu_xc, **setup_kw)
            mf._gpu_xc_path = 'precomputed'
        elif xc_path == 'onthefly':
            from pyscf.OpenCL.xc_grid import setup_xc_grid_gpu
            mf._xc_gpu_plan = setup_xc_grid_gpu(
                mol, mf.grids, mf.xc, gpu_xc=gpu_xc, **setup_kw)
            mf._gpu_xc_path = 'onthefly'
        else:
            raise ValueError(f'profile {name!r}: xc_path={xc_path!r}')
    if setup:
        prepare_df_for_scf(mf, storage=df_storage, require_incore=require_df_incore)
    mf._gpu_profile_name = name
    return mf


def profile_accuracy_note(name):
  return get_profile(name).get('accuracy', {})
