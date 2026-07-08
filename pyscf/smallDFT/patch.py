'''Optional monkey-patch of NumInt.nr_rks for small systems.'''
from pyscf.dft import numint as numint_mod

from .nr_rks import nr_rks as small_nr_rks, NAO_MAX_DEFAULT

_ORIGINAL_NR_RKS = None


def enable(nao_max=NAO_MAX_DEFAULT, n_workers=None, tile_size=None, precompute_ao=False):
    '''Route NumInt.nr_rks to smallDFT when mol.nao_nr() <= nao_max.'''
    global _ORIGINAL_NR_RKS
    if _ORIGINAL_NR_RKS is None:
        _ORIGINAL_NR_RKS = numint_mod.NumInt.nr_rks

    def _dispatch(self, mol, grids, xc_code, dms, *args, **kwargs):
        if mol.nao_nr() <= nao_max:
            kw = dict(n_workers=n_workers, precompute_ao=precompute_ao)
            if tile_size is not None:
                kw['tile_size'] = tile_size
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
