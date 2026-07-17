'''Preallocated buffers reused across SCF iterations (fixed geometry/grid).

Open issues / caveats:
- Full GGA χ still materializes (~GB on large grids); streaming ρ/vmat per block
  without full χ is a longer-term RAM tradeoff (see doc/smallDFT_cpu_path.md).
'''
import numpy

from pyscf import lib
from pyscf.dft.numint import NumInt, BLKSIZE, NBINS, _empty_aligned

from .layout import ensure_native


def _chi_view(ao_buf, ngrids, nao, deriv):
    '''libcint-native χ view into ``ao_buf`` (same as ``mol.eval_gto(..., out=buf)``).'''
    if deriv == 0:
        return numpy.ndarray((nao, ngrids), buffer=ao_buf, dtype=numpy.double).T
    comp = (deriv + 1) * (deriv + 2) * (deriv + 3) // 6
    return numpy.ndarray((comp, nao, ngrids), buffer=ao_buf, dtype=numpy.double).transpose(0, 2, 1)


class GridWorkspace:
    '''Scratch arrays for one (mol, grids) pair.

    chi is set by eval_ao() using libcint-native layout (F-contiguous on grids).
    ao_buf is raw storage matching eval_gto's (comp, nao, ngrids) packing.
    '''

    __slots__ = ('nao', 'ngrids', 'deriv', 'chi', 'ao_buf', 'rho_lda', 'rho_gga', 'vmat')

    def __init__(self, mol, grids, deriv=1, nao=None, ngrids=None):
        self.nao = int(nao if nao is not None else mol.nao_nr())
        self.ngrids = int(ngrids if ngrids is not None else grids.coords.shape[0])
        self.deriv = int(deriv)
        self.chi = None
        comp = (self.deriv + 1) * (self.deriv + 2) * (self.deriv + 3) // 6
        # Blob sized for eval_gto packing (comp, nao, ngrids); shape label is informational.
        shape = (self.ngrids, self.nao) if comp == 1 else (comp, self.ngrids, self.nao)
        self.ao_buf = numpy.empty(shape, dtype=numpy.double, order='C')
        self.rho_lda = numpy.empty(self.ngrids, dtype=numpy.double)
        self.rho_gga = numpy.empty((4, self.ngrids), dtype=numpy.double, order='C')
        self.vmat = numpy.zeros((self.nao, self.nao), dtype=numpy.double)

    def eval_ao(self, mol, grids, max_memory=2000, blksize=None, blocked=False):
        '''Fill χ once per geometry.

        Default ``blocked=False`` (one-shot into ``ao_buf``) is fastest when the
        full GGA χ must be kept. ``blocked=True`` uses the same screened
        ``block_loop`` path as ref ``nr_rks`` then ``copyto`` slices — parity OK,
        but ~10% slower on PTCDA because it still writes the full ~GB tensor.
        '''
        if grids.coords is None:
            grids.build(with_non0tab=True)
        ngrids, nao, deriv = self.ngrids, self.nao, self.deriv
        assert grids.coords.shape[0] == ngrids
        chi = _chi_view(self.ao_buf, ngrids, nao, deriv)

        if not blocked:
            from .layout import eval_ao_native
            non0tab = grids.non0tab if getattr(grids, 'mol', None) is mol else None
            self.chi = eval_ao_native(mol, grids.coords, deriv=deriv,
                                      non0tab=non0tab, cutoff=grids.cutoff,
                                      buf=self.ao_buf)
            return self.chi

        comp = (deriv + 1) * (deriv + 2) * (deriv + 3) // 6
        if blksize is None:
            blksize = int(max_memory * 1e6 / ((comp + 1) * nao * 8 * BLKSIZE))
            blksize = max(4, min(blksize, ngrids // BLKSIZE + 1, 1200)) * BLKSIZE
        assert blksize % BLKSIZE == 0

        non0tab = grids.non0tab if getattr(grids, 'mol', None) is mol else None
        if non0tab is None:
            non0tab = numpy.empty(((ngrids + BLKSIZE - 1) // BLKSIZE, mol.nbas),
                                  dtype=numpy.uint8)
            non0tab[:] = NBINS + 1

        ni = NumInt()
        buf = _empty_aligned(comp * blksize * nao)
        for ip0, ip1 in lib.prange(0, ngrids, blksize):
            mask = non0tab[ip0 // BLKSIZE:]
            ao = ni.eval_ao(mol, grids.coords[ip0:ip1], deriv=deriv,
                            non0tab=mask, cutoff=grids.cutoff, out=buf)
            if deriv == 0:
                numpy.copyto(chi[ip0:ip1], ao)
            else:
                numpy.copyto(chi[:, ip0:ip1, :], ao)

        self.chi = ensure_native(chi, deriv=deriv)
        return self.chi

    def matches(self, nao, ngrids, deriv):
        return self.nao == nao and self.ngrids == ngrids and self.deriv == deriv


def workspace_for(mol, grids, deriv=1):
    return GridWorkspace(mol, grids, deriv=deriv)
