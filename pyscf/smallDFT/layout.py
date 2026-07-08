'''AO layout: PySCF (ngrids, nao) F-contiguous — grid tile chi[g0:g1,:] is one memory block.'''
import numpy
from pyscf.dft import numint as numint_mod

eval_ao = numint_mod.eval_ao


def ensure_native(ao, deriv=0):
    '''Keep libcint layout; copy only if not F-contiguous.'''
    if deriv == 0:
        if ao.flags.f_contiguous:
            return ao
        return numpy.asfortranarray(ao)
    if ao[0].flags.f_contiguous:
        return ao
    return numpy.asfortranarray(ao)


def to_chi_T(ao, deriv=0, out=None):
    '''Optional χ_T[nao,ngrids] for column-wise access (explicit copy).'''
    if deriv == 0:
        if out is None:
            return numpy.ascontiguousarray(ao.T)
        numpy.copyto(out, ao.T)
        return out
    if out is None:
        return numpy.ascontiguousarray(ao.transpose(0, 2, 1))
    numpy.copyto(out, ao.transpose(0, 2, 1))
    return out


def eval_ao_native(mol, coords, deriv=0, non0tab=None, cutoff=None, out=None):
    ao = eval_ao(mol, coords, deriv=deriv, non0tab=non0tab, cutoff=cutoff)
    ao = ensure_native(ao, deriv=deriv)
    if out is None:
        return ao
    numpy.copyto(out, ao)
    return out


eval_ao_chi_T = eval_ao_native
to_grid_major = ensure_native
