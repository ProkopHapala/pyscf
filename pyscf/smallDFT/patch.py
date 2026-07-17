'''Optional monkey-patch of NumInt.nr_rks + SCF preparation for smallDFT.

Essence: route RKS XC to grid-parallel smallDFT with AO cached once per geometry,
and hoist DF tensor build (prefer incore) before the SCF loop so timings are not
spoiled by lazy HDF5 outcore.

Design:
- prepare_smalldft_for_scf is the SSOT setup for CPU accelerated RKS+DF benches.
- Order is grids → DF.build (storage) → [optional AO cache] → patch nr_rks.
- DF before AO so storage='auto' is less likely to spill; prefer storage='incore'.
- ao_mode='cache' (default) materializes full χ; ao_mode='stream' never does
  (block_loop + stream_grid.c) — for RAM-limited hosts / PTCDA-class grids.

Open issues / caveats:
- nao_max default raised to 400 (PTCDA 6-31g ≈ 286). Above that, fall back to
  reference NumInt unless force=True.
- See doc/df_storage_and_benchmark_hygiene.md for DF storage policy.
'''
from pyscf.dft import numint as numint_mod
from pyscf import lib

from .nr_rks import nr_rks as small_nr_rks, NAO_MAX_DEFAULT

_ORIGINAL_NR_RKS = None


def enable(nao_max=NAO_MAX_DEFAULT, n_workers=None, tile_size=None, precompute_ao=False,
           workspace=None, ao_mode=None):
    '''Route NumInt.nr_rks to smallDFT when mol.nao_nr() <= nao_max.

    If ``workspace`` is set (or later ``mf._smallDFT_ws``), pass it as ``ws=``
    so χ is reused across SCF cycles (ao_mode='cache').

    ao_mode: 'cache' | 'stream' | None — stored on NumInt as ``_smallDFT_ao_mode``.
    '''
    global _ORIGINAL_NR_RKS
    if _ORIGINAL_NR_RKS is None:
        _ORIGINAL_NR_RKS = numint_mod.NumInt.nr_rks

    def _dispatch(self, mol, grids, xc_code, dms, *args, **kwargs):
        if mol.nao_nr() <= nao_max:
            kw = dict(n_workers=n_workers, precompute_ao=precompute_ao)
            if tile_size is not None:
                kw['tile_size'] = tile_size
            if ao_mode is not None:
                kw['ao_mode'] = ao_mode
            ws = workspace
            if ws is None:
                ws = getattr(self, '_smallDFT_ws', None)
            if ws is not None and kwargs.get('ao_mode', ao_mode) != 'stream':
                kw['ws'] = ws
            kw.update(kwargs)
            return small_nr_rks(self, mol, grids, xc_code, dms, *args, **kw)
        return _ORIGINAL_NR_RKS(self, mol, grids, xc_code, dms, *args, **kwargs)

    numint_mod.NumInt.nr_rks = _dispatch


def disable():
    '''Restore original NumInt.nr_rks.'''
    global _ORIGINAL_NR_RKS
    if _ORIGINAL_NR_RKS is not None:
        numint_mod.NumInt.nr_rks = _ORIGINAL_NR_RKS
        _ORIGINAL_NR_RKS = None


def prepare_smalldft_for_scf(mf, storage='incore', require_incore=True,
                             deriv=1, nao_max=None, n_workers=None,
                             max_memory_mb=None, force=False, ao_mode='cache'):
    '''Prepare invariant CPU-XC + DF state before ``mf.kernel()``.

    Steps (once per geometry):
      1. ``grids.build`` if needed
      2. ``prepare_df_for_scf`` with ``storage`` (default **incore**, fail-loud)
      3. if ao_mode='cache': ``GridWorkspace.eval_ao`` (χ for all SCF cycles)
         if ao_mode='stream': skip χ allocation (block AO each cycle)
      4. patch ``NumInt.nr_rks`` → smallDFT

    Parameters
    ----------
    storage : {'auto','incore','outcore'}
        DF tensor policy; see ``doc/df_storage_and_benchmark_hygiene.md``.
    require_incore : bool
        Raise if `_cderi` is not an in-RAM ndarray after build.
    ao_mode : {'cache','stream'}
        ``cache`` — full GGA χ once (~3.5 GB PTCDA); fastest multi-cycle.
        ``stream`` — no full χ; ``stream_grid.c`` on block_loop AO (RAM-safe).
    nao_max : int or None
        Dispatch cutoff (default ``NAO_MAX_DEFAULT``). PTCDA 6-31g needs ≥286.
    max_memory_mb : float or None
        If set, raise ``mol`` / ``mf`` / ``with_df`` max_memory so large
        `_cderi` + χ fit (e.g. 8000 for PTCDA on 16 GiB hosts).
    force : bool
        If True, enable smallDFT even when ``nao > nao_max``.

    Returns
    -------
    mf
    '''
    from .workspace import GridWorkspace
    from .nr_rks import NAO_MAX_DEFAULT as _NAO_MAX

    if ao_mode not in ('cache', 'stream'):
        raise ValueError(f"ao_mode must be 'cache' or 'stream', got {ao_mode!r}")
    if nao_max is None:
        nao_max = _NAO_MAX
    mol = mf.mol
    nao = mol.nao_nr()
    if not force and nao > nao_max:
        raise ValueError(
            f'prepare_smalldft_for_scf: nao={nao} > nao_max={nao_max}. '
            f'Pass nao_max={nao} or force=True if intentional.')

    if max_memory_mb is not None:
        mol.max_memory = max(mol.max_memory, float(max_memory_mb))
        mf.max_memory = mol.max_memory
        if getattr(mf, 'with_df', None) is not None:
            mf.with_df.max_memory = mol.max_memory

    if mf.grids.coords is None:
        mf.grids.build(with_non0tab=True)

    # DF before AO — claiming χ first can flip storage='auto' → HDF5.
    if getattr(mf, 'with_df', None) is not None:
        from pyscf.OpenCL.gpu_profiles import prepare_df_for_scf
        prepare_df_for_scf(mf, storage=storage, require_incore=require_incore)

    xctype = mf._numint._xc_type(mf.xc)
    if deriv is None:
        deriv = 0 if xctype == 'LDA' else 1

    ws = None
    if ao_mode == 'cache':
        ws = GridWorkspace(mol, mf.grids, deriv=deriv)
        ws.eval_ao(mol, mf.grids)
        mf._smallDFT_ws = ws
        mf._numint._smallDFT_ws = ws
    else:
        mf._smallDFT_ws = None
        mf._numint._smallDFT_ws = None

    mf._numint._smallDFT_ao_mode = ao_mode
    mf._smallDFT_ao_mode = ao_mode

    nw = n_workers if n_workers is not None else lib.num_threads()
    enable(nao_max=max(nao_max, nao) if force else nao_max,
           n_workers=nw, workspace=ws, ao_mode=ao_mode)
    mf._smallDFT_prepared = True
    return mf
