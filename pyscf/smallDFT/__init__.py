'''smallDFT — CPU grid-parallel RKS XC for small molecules (nao ≲ 200).

OpenMP ρ/vmat in libsmalldft replaces the hot loops of numint.nr_rks while
keeping libcint AO layout (F-contiguous ngrids×nao). Python is orchestration
only: eval_ao_native, libxc, ctypes dispatch. libcint already parallelizes
grid AO evaluation with OpenMP; GridWorkspace supplies its reusable raw buffer
to avoid allocating and copying χ at each geometry update.

See doc/smallDFT_cpu_path.md and doc/CPU_benchmark.md.

Usage::

    from pyscf import gto, dft, lib
    from pyscf.smallDFT import nr_rks, GridWorkspace

    lib.num_threads(4)
    mol = gto.M(atom='...', basis='6-31g')
    mf = dft.RKS(mol, xc='PBE'); mf.grids.build(); mf.kernel()
    dm = mf.make_rdm1()
    ws = GridWorkspace(mol, mf.grids, deriv=1)
    ws.eval_ao(mol, mf.grids)   # once per geometry
    nelec, exc, vmat = nr_rks(dft.numint.NumInt(), mol, mf.grids, 'PBE', dm,
                              n_workers=4, ws=ws)
'''
from .layout import eval_ao_native, to_chi_T, ensure_native
from .rho import rho_lda, rho_gga
from .vmat import vmat_lda, vmat_gga
from .nr_rks import nr_rks, NAO_MAX_DEFAULT
from .profile import profile_compare, profile_nr_rks_breakdown, profile_xc_bottleneck, PRIORITY_PLAN
from .patch import enable, disable
from .parallel import shutdown_pool, TILE_SIZE_DEFAULT
from .workspace import GridWorkspace, workspace_for
from ._ctypes import has_c_lib

__all__ = [
    'eval_ao_native', 'to_chi_T', 'ensure_native',
    'rho_lda', 'rho_gga', 'vmat_lda', 'vmat_gga',
    'nr_rks', 'NAO_MAX_DEFAULT', 'TILE_SIZE_DEFAULT',
    'profile_compare', 'profile_nr_rks_breakdown', 'profile_xc_bottleneck', 'PRIORITY_PLAN',
    'enable', 'disable', 'shutdown_pool',
    'GridWorkspace', 'workspace_for', 'has_c_lib',
]
