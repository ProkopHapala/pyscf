'''Grid-parallel density: each worker owns [g0,g1), writes rho[g0:g1] — no atomics.

χ layout (ngrids, nao) F-contiguous: tile χ[g0:g1,:] is contiguous in memory.
'''
import numpy

from .parallel import parallel_grid_fill, default_n_workers
from ._ctypes import has_c_lib, c_rho_lda, c_rho_gga


def _tile_size(ngrids, n_workers, tile_size):
    if tile_size is not None and int(tile_size) > 0:
        return int(tile_size)
    nw = max(1, int(n_workers))
    return max(512, (int(ngrids) + nw - 1) // nw)


def rho_lda(dm, chi, tile_size=None, n_workers=None, rho_out=None, executor=None,
            use_c=None, nthreads=None, c0_buf=None):
    '''ρ(g) = χ(g)ᵀ DM χ(g); chi (ngrids, nao).

    use_c: None → C if libsmalldft loaded, no executor, and ngrids*nao large enough.
    '''
    dm = numpy.asarray(dm, order='C', dtype=numpy.double)
    ngrids, nao = chi.shape
    if rho_out is None:
        rho_out = numpy.empty(ngrids, dtype=numpy.double)
    nw = n_workers if n_workers is not None else default_n_workers()
    if use_c is None:
        use_c = (has_c_lib() and executor is None
                 and ngrids * nao >= 65536)  # C OpenMP path for larger grids
    if use_c and has_c_lib():
        nt = nw if nthreads is None else int(nthreads)
        c_rho_lda(rho_out, chi, dm, nthreads=nt)
        return rho_out

    ts = _tile_size(ngrids, nw, tile_size)

    def _fill(g0, g1):
        blk = chi[g0:g1]
        rho_out[g0:g1] = numpy.sum(blk * (blk @ dm.T), axis=1)

    parallel_grid_fill(ngrids, ts, nw, _fill, executor=executor)
    return rho_out


def rho_gga(dm, chi, tile_size=None, n_workers=None, hermi=1, rho_out=None, executor=None,
            c0_buf=None, use_c=None, nthreads=None):
    '''GGA ρ; chi (4, ngrids, nao) F-order; rho (4, ngrids) C-order.'''
    dm = numpy.asarray(dm, order='C', dtype=numpy.double)
    ngrids = chi.shape[1]
    nao = chi.shape[2]
    if rho_out is None:
        rho_out = numpy.empty((4, ngrids), dtype=numpy.double, order='C')
    nw = n_workers if n_workers is not None else default_n_workers()
    if use_c is None:
        use_c = has_c_lib() and executor is None and ngrids * nao >= 65536
    if use_c and has_c_lib():
        nt = nw if nthreads is None else int(nthreads)
        c_rho_gga(rho_out, chi, dm, nthreads=nt, hermi=hermi)
        return rho_out

    ts = _tile_size(ngrids, nw, tile_size)

    def _fill(g0, g1):
        b0 = chi[0, g0:g1]
        if c0_buf is not None and c0_buf.shape[1] >= b0.shape[0]:
            c0 = c0_buf[:, :b0.shape[0]]
            numpy.dot(dm.T, b0.T, out=c0)
            c0 = c0.T
        else:
            c0 = b0 @ dm.T
        rho_out[0, g0:g1] = numpy.sum(b0 * c0, axis=1)
        if hermi:
            for k in range(1, 4):
                rho_out[k, g0:g1] = 2.0 * numpy.sum(chi[k, g0:g1] * c0, axis=1)
        else:
            for k in range(1, 4):
                bk = chi[k, g0:g1]
                rho_out[k, g0:g1] = numpy.sum(b0 * (bk @ dm.T), axis=1) + numpy.sum(bk * c0, axis=1)

    parallel_grid_fill(ngrids, ts, nw, _fill, executor=executor)
    return rho_out
