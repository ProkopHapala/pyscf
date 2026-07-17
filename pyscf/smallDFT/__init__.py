'''smallDFT — CPU grid-parallel RKS XC for molecules with nao ≲ 400.

OpenMP ρ/vmat in libsmalldft replaces the hot loops of numint.nr_rks while
keeping libcint AO layout (F-contiguous ngrids×nao). Python is orchestration
only: eval_ao_native, libxc, ctypes dispatch. libcint already parallelizes
grid AO evaluation with OpenMP; GridWorkspace supplies its reusable raw buffer
to avoid allocating and copying χ at each geometry update.

For RKS+DF production setup use ``prepare_smalldft_for_scf`` (DF incore before
AO cache). Use ``ao_mode='stream'`` to skip the full-χ buffer (RAM-safe GGA).
See doc/smallDFT_cpu_path.md, doc/CPU_benchmark.md,
doc/df_storage_and_benchmark_hygiene.md.

Usage::

    from pyscf import gto, dft, lib
    from pyscf.smallDFT import prepare_smalldft_for_scf

    lib.num_threads(4)
    mol = gto.M(atom='...', basis='6-31g')
    mf = dft.RKS(mol, xc='PBE').density_fit()
    prepare_smalldft_for_scf(mf, storage='incore', max_memory_mb=8000)  # or ao_mode='stream'
    mf.kernel()
'''
from .layout import eval_ao_native, to_chi_T, ensure_native
from .rho import rho_lda, rho_gga
from .vmat import vmat_lda, vmat_gga
from .nr_rks import nr_rks, NAO_MAX_DEFAULT
from .profile import profile_compare, profile_nr_rks_breakdown, profile_xc_bottleneck, PRIORITY_PLAN
from .patch import enable, disable, prepare_smalldft_for_scf
from .parallel import shutdown_pool, TILE_SIZE_DEFAULT
from .workspace import GridWorkspace, workspace_for
from ._ctypes import has_c_lib

__all__ = [
    'eval_ao_native', 'to_chi_T', 'ensure_native',
    'rho_lda', 'rho_gga', 'vmat_lda', 'vmat_gga',
    'nr_rks', 'NAO_MAX_DEFAULT', 'TILE_SIZE_DEFAULT',
    'profile_compare', 'profile_nr_rks_breakdown', 'profile_xc_bottleneck', 'PRIORITY_PLAN',
    'enable', 'disable', 'prepare_smalldft_for_scf', 'shutdown_pool',
    'GridWorkspace', 'workspace_for', 'has_c_lib',
]
