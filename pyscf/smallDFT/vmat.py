'''Grid-parallel vmat: C/OpenMP primary path; Python threads legacy fallback.'''
import numpy

from .parallel import parallel_grid_reduce, default_n_workers
from ._ctypes import has_c_lib, c_vmat_lda, c_vmat_gga


def _tile_size(ngrids, n_workers, tile_size):
    if tile_size is not None and int(tile_size) > 0:
        return int(tile_size)
    nw = max(1, int(n_workers))
    return max(512, (int(ngrids) + nw - 1) // nw)


def vmat_lda(chi, wv, out=None, tile_size=None, n_workers=None, executor=None,
             use_c=None, nthreads=None):
    '''V = Σ_g wv(g) χ(g)ᵀ χ(g); chi (ngrids, nao).'''
    nao = chi.shape[1]
    ngrids = chi.shape[0]
    if out is None:
        out = numpy.zeros((nao, nao), dtype=numpy.double)
    else:
        out[:] = 0
    wv = numpy.asarray(wv, dtype=numpy.double).ravel()
    if use_c is None:
        use_c = has_c_lib() and executor is None and ngrids * nao >= 65536
    if use_c and has_c_lib():
        nt = n_workers if nthreads is None else int(nthreads)
        c_vmat_lda(out, chi, wv, nthreads=nt)
        return out

    nw = n_workers if n_workers is not None else default_n_workers()
    ts = _tile_size(ngrids, nw, tile_size)

    def _tile(g0, g1):
        blk = chi[g0:g1]
        wt = wv[g0:g1, numpy.newaxis]
        return blk.T @ (blk * wt)

    parallel_grid_reduce(ngrids, ts, nw, _tile, out, executor=executor)
    return out


def vmat_gga(chi, wv, out=None, tile_size=None, n_workers=None, hermi_sum=True,
             executor=None, use_c=None, nthreads=None):
    nao = chi.shape[2]
    ngrids = chi.shape[1]
    if out is None:
        out = numpy.zeros((nao, nao), dtype=numpy.double)
    else:
        out[:] = 0
    wv = numpy.asarray(wv, order='C', dtype=numpy.double)
    if use_c is None:
        use_c = has_c_lib() and executor is None and ngrids * nao >= 65536
    if use_c and has_c_lib():
        nt = n_workers if nthreads is None else int(nthreads)
        c_vmat_gga(out, chi, wv, nthreads=nt, hermi=1 if hermi_sum else 0)
        return out

    nw = n_workers if n_workers is not None else default_n_workers()
    ts = _tile_size(ngrids, nw, tile_size)

    def _tile(g0, g1):
        b0 = chi[0, g0:g1]
        aow = numpy.einsum('cg,cgi->gi', wv[:, g0:g1], chi[:, g0:g1, :], optimize=True)
        return b0.T @ aow

    parallel_grid_reduce(ngrids, ts, nw, _tile, out, executor=executor)
    if hermi_sum:
        out[:] = out + out.T
    return out
