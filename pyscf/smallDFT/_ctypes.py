'''ctypes bindings for libsmalldft (optional C acceleration).'''
import ctypes

import numpy
from pyscf import lib

_LIB = None
_HAS_C = False


def _init():
    global _LIB, _HAS_C
    if _LIB is not None:
        return _LIB
    try:
        _LIB = lib.load_library('libsmalldft')
    except OSError:
        _LIB = False
        _HAS_C = False
        return None
    fn = _LIB.SMALL_rho_lda
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    fn.restype = None
    fn = _LIB.SMALL_rho_gga
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    fn.restype = None
    fn = _LIB.SMALL_vmat_lda
    fn.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    fn.restype = None
    fn = _LIB.SMALL_vmat_gga
    fn.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    fn.restype = None
    for name, nargs in (
        ('SMALL_stream_rho_lda', 6),
        ('SMALL_stream_rho_gga', 7),
        ('SMALL_stream_vmat_lda_acc', 6),
        ('SMALL_stream_vmat_gga_acc', 6),
    ):
        fn = getattr(_LIB, name)
        fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * (nargs - 3)
        fn.restype = None
    fn = _LIB.SMALL_stream_vmat_hermi
    fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
    fn.restype = None
    _HAS_C = True
    return _LIB


def has_c_lib():
    _init()
    return _HAS_C


def libsmalldft():
    return _init()


def c_rho_lda(rho, chi, dm, nthreads=0):
    '''SMALL_rho_lda; chi F (ngrids,nao), dm C (nao,nao).'''
    libc = _init()
    if libc is None:
        raise RuntimeError('libsmalldft not found')
    ngrids, nao = chi.shape
    libc.SMALL_rho_lda(
        rho.ctypes.data_as(ctypes.c_void_p),
        chi.ctypes.data_as(ctypes.c_void_p),
        numpy.asarray(dm, order='C').ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao),
        ctypes.c_int(ngrids),
        ctypes.c_int(int(nthreads)),
    )
    return rho


def c_rho_gga(rho, chi, dm, nthreads=0, hermi=1):
    '''SMALL_rho_gga; chi F (4,ngrids,nao), rho C (4,ngrids).'''
    libc = _init()
    if libc is None:
        raise RuntimeError('libsmalldft not found')
    _, ngrids, nao = chi.shape
    libc.SMALL_rho_gga(
        rho.ctypes.data_as(ctypes.c_void_p),
        chi.ctypes.data_as(ctypes.c_void_p),
        numpy.asarray(dm, order='C').ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao),
        ctypes.c_int(ngrids),
        ctypes.c_int(int(nthreads)),
        ctypes.c_int(int(hermi)),
    )
    return rho


def c_vmat_lda(vmat, chi, wv, nthreads=0):
    libc = _init()
    if libc is None:
        raise RuntimeError('libsmalldft not found')
    ngrids, nao = chi.shape
    libc.SMALL_vmat_lda(
        vmat.ctypes.data_as(ctypes.c_void_p),
        chi.ctypes.data_as(ctypes.c_void_p),
        numpy.asarray(wv, dtype=numpy.double).ravel().ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao), ctypes.c_int(ngrids), ctypes.c_int(int(nthreads)),
    )
    return vmat


def c_vmat_gga(vmat, chi, wv, nthreads=0, hermi=1):
    libc = _init()
    if libc is None:
        raise RuntimeError('libsmalldft not found')
    _, ngrids, nao = chi.shape
    libc.SMALL_vmat_gga(
        vmat.ctypes.data_as(ctypes.c_void_p),
        chi.ctypes.data_as(ctypes.c_void_p),
        numpy.asarray(wv, order='C', dtype=numpy.double).ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao), ctypes.c_int(ngrids),
        ctypes.c_int(int(nthreads)), ctypes.c_int(int(hermi)),
    )
    return vmat


def c_stream_rho_lda(rho, chi, dm, nthreads=0):
    libc = _init()
    if libc is None:
        raise RuntimeError('libsmalldft not found')
    ngrids, nao = chi.shape
    libc.SMALL_stream_rho_lda(
        rho.ctypes.data_as(ctypes.c_void_p),
        chi.ctypes.data_as(ctypes.c_void_p),
        numpy.asarray(dm, order='C').ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao), ctypes.c_int(ngrids), ctypes.c_int(int(nthreads)))
    return rho


def c_stream_rho_gga(rho, chi, dm, nthreads=0, hermi=1):
    libc = _init()
    if libc is None:
        raise RuntimeError('libsmalldft not found')
    _, ngrids, nao = chi.shape
    libc.SMALL_stream_rho_gga(
        rho.ctypes.data_as(ctypes.c_void_p),
        chi.ctypes.data_as(ctypes.c_void_p),
        numpy.asarray(dm, order='C').ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao), ctypes.c_int(ngrids),
        ctypes.c_int(int(nthreads)), ctypes.c_int(int(hermi)))
    return rho


def c_stream_vmat_lda_acc(vmat, chi, wv, nthreads=0):
    libc = _init()
    if libc is None:
        raise RuntimeError('libsmalldft not found')
    ngrids, nao = chi.shape
    libc.SMALL_stream_vmat_lda_acc(
        vmat.ctypes.data_as(ctypes.c_void_p),
        chi.ctypes.data_as(ctypes.c_void_p),
        numpy.asarray(wv, dtype=numpy.double).ravel().ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao), ctypes.c_int(ngrids), ctypes.c_int(int(nthreads)))
    return vmat


def c_stream_vmat_gga_acc(vmat, chi, wv, nthreads=0):
    libc = _init()
    if libc is None:
        raise RuntimeError('libsmalldft not found')
    _, ngrids, nao = chi.shape
    libc.SMALL_stream_vmat_gga_acc(
        vmat.ctypes.data_as(ctypes.c_void_p),
        chi.ctypes.data_as(ctypes.c_void_p),
        numpy.asarray(wv, order='C', dtype=numpy.double).ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao), ctypes.c_int(ngrids), ctypes.c_int(int(nthreads)))
    return vmat


def c_stream_vmat_hermi(vmat):
    libc = _init()
    if libc is None:
        raise RuntimeError('libsmalldft not found')
    nao = vmat.shape[0]
    libc.SMALL_stream_vmat_hermi(
        numpy.asarray(vmat, order='C').ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao))
    return vmat
