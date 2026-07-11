'''Preallocated buffers reused across SCF iterations (fixed geometry/grid).'''
import numpy

from .layout import eval_ao_native


class GridWorkspace:
    '''Scratch arrays for one (mol, grids) pair.

    chi is set by eval_ao() using libcint-native layout (same as eval_ao_native).
    ao_buf is raw C storage passed directly to libcint, avoiding a transient
    AO allocation and copy at every geometry update.
    '''

    __slots__ = ('nao', 'ngrids', 'deriv', 'chi', 'ao_buf', 'rho_lda', 'rho_gga', 'vmat')

    def __init__(self, mol, grids, deriv=1, nao=None, ngrids=None):
        self.nao = int(nao if nao is not None else mol.nao_nr())
        self.ngrids = int(ngrids if ngrids is not None else grids.coords.shape[0])
        self.deriv = int(deriv)
        self.chi = None
        comp = (self.deriv + 1) * (self.deriv + 2) * (self.deriv + 3) // 6
        shape = (self.ngrids, self.nao) if comp == 1 else (comp, self.ngrids, self.nao)
        self.ao_buf = numpy.empty(shape, dtype=numpy.double, order='C')
        self.rho_lda = numpy.empty(self.ngrids, dtype=numpy.double)
        self.rho_gga = numpy.empty((4, self.ngrids), dtype=numpy.double, order='C')
        self.vmat = numpy.zeros((self.nao, self.nao), dtype=numpy.double)

    def eval_ao(self, mol, grids):
        non0tab = grids.non0tab if getattr(grids, 'mol', None) is mol else None
        self.chi = eval_ao_native(mol, grids.coords, deriv=self.deriv,
                                  non0tab=non0tab, cutoff=grids.cutoff, buf=self.ao_buf)
        return self.chi

    def matches(self, nao, ngrids, deriv):
        return self.nao == nao and self.ngrids == ngrids and self.deriv == deriv


def workspace_for(mol, grids, deriv=1):
    return GridWorkspace(mol, grids, deriv=deriv)
