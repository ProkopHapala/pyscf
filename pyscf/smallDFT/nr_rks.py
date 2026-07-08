'''Drop-in nr_rks with grid-parallel ρ/vmat (parallel axis = grid index).'''
import numpy
from pyscf import lib

from .layout import ensure_native
from .rho import rho_lda, rho_gga
from .vmat import vmat_lda, vmat_gga
from .parallel import default_n_workers, shutdown_pool, get_pool
from ._ctypes import has_c_lib

NAO_MAX_DEFAULT = 200


def nr_rks(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
           max_memory=2000, verbose=None, n_workers=None, tile_size=None,
           precompute_ao=False, blas_threads=None, ws=None):
    '''RKS XC for small systems.

    Grid-parallel ρ/vmat: each thread owns a contiguous grid slice χ[g0:g1,:].
    BLAS is pinned to 1 thread by default so grid workers do not oversubscribe.

    ws: optional GridWorkspace with preallocated buffers (call ws.eval_ao once
        per geometry before the SCF loop).

    blas_threads: if set, use this many OMP threads inside each tile GEMM instead
        of grid-level threading (try blas_threads=4, n_workers=1 for comparison).
    '''
    xctype = ni._xc_type(xc_code)
    if xctype not in ('LDA', 'GGA', 'HF'):
        raise NotImplementedError(f'smallDFT.nr_rks: xctype={xctype}')

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        dms = dms[numpy.newaxis]
    nset = len(dms)
    nao = dms[0].shape[0]
    nw = n_workers if n_workers is not None else default_n_workers()

    if hermi != 1 and dms[0].dtype == numpy.double:
        dms = lib.hermi_sum(numpy.asarray(dms, order='C'), axes=(0, 2, 1)) * .5
        hermi = 1

    if grids.coords is None:
        grids.build(with_non0tab=True)

    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    vmat = numpy.zeros((nset, nao, nao))

    if xctype == 'HF':
        return _cast_outputs(dms, nelec, excsum, vmat)

    deriv = 0 if xctype == 'LDA' else 1
    omp_saved = lib.num_threads()
    use_c = has_c_lib() and blas_threads is None
    pool = None  # C/OpenMP path only; Python thread pool deprecated

    try:
        if precompute_ao or ws is not None:
            if ws is not None and ws.matches(nao, grids.coords.shape[0], deriv):
                chi = ws.chi
                if precompute_ao:
                    ws.eval_ao(mol, grids)
            elif precompute_ao:
                from .layout import eval_ao_native
                non0tab = grids.non0tab if grids.mol is mol else None
                chi = eval_ao_native(mol, grids.coords, deriv=deriv,
                                     non0tab=non0tab, cutoff=grids.cutoff)
            else:
                from .workspace import GridWorkspace
                ws = GridWorkspace(mol, grids, deriv=deriv, nao=nao,
                                   ngrids=grids.coords.shape[0])
                chi = ws.eval_ao(mol, grids)
            weight = grids.weights
            for i in range(nset):
                dm = numpy.asarray(dms[i], order='C')
                rho, exc, v_i = _xc_vmat_for_dm(
                    ni, xc_code, xctype, dm, chi, weight, deriv,
                    nw, tile_size, pool, ws=ws, use_c=use_c,
                    omp_threads=omp_saved)
                nelec[i] = rho_nelec(rho, weight, xctype)
                excsum[i] = numpy.dot(nelec_weight(rho, weight, xctype), exc)
                vmat[i] = v_i
        else:
            for i in range(nset):
                dm = numpy.asarray(dms[i], order='C')
                for ao, mask, weight, _coords in ni.block_loop(
                        mol, grids, nao, deriv, max_memory=max_memory):
                    chi = ensure_native(ao, deriv=deriv)
                    rho, exc, v_part = _xc_vmat_for_dm(
                        ni, xc_code, xctype, dm, chi, weight, deriv,
                        nw, tile_size, pool, ws=ws, use_c=use_c,
                        omp_threads=omp_saved)
                    nelec[i] += rho_nelec(rho, weight, xctype)
                    excsum[i] += numpy.dot(nelec_weight(rho, weight, xctype), exc)
                    vmat[i] += v_part
    finally:
        lib.num_threads(omp_saved)

    if nset == 1:
        nelec, excsum, vmat = nelec[0], excsum[0], vmat[0]
    return _cast_outputs(dms, nelec, excsum, vmat)


def rho_nelec(rho, weight, xctype):
    if xctype == 'LDA':
        return (rho * weight).sum()
    return (rho[0] * weight).sum()


def nelec_weight(rho, weight, xctype):
    if xctype == 'LDA':
        return rho * weight
    return rho[0] * weight


def _xc_vmat_for_dm(ni, xc_code, xctype, dm, chi, weight, deriv, n_workers, tile_size,
                    executor, ws=None, use_c=False, omp_threads=None):
    kw = dict(tile_size=tile_size, n_workers=n_workers, executor=executor)
    nt = omp_threads or n_workers

    if use_c and deriv == 0:
        rho_out = ws.rho_lda if ws is not None else None
        rho = rho_lda(dm, chi, rho_out=rho_out, use_c=True, nthreads=nt, **kw)
    elif use_c and deriv == 1:
        rho_out = ws.rho_gga if ws is not None else None
        rho = rho_gga(dm, chi, rho_out=rho_out, use_c=True, hermi=1, nthreads=nt, **kw)
    else:
        rho_out = (ws.rho_lda if ws is not None else None) if deriv == 0 else (ws.rho_gga if ws is not None else None)
        if deriv == 0:
            rho = rho_lda(dm, chi, rho_out=rho_out, use_c=False, **kw)
        else:
            rho = rho_gga(dm, chi, hermi=1, rho_out=rho_out, use_c=False, **kw)

    exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=0)[:2]
    wv = weight * vxc

    vmat_out = ws.vmat if ws is not None else None
    if xctype == 'LDA':
        if use_c:
            lib.num_threads(nt)
        vmat = vmat_lda(chi, wv, out=vmat_out, use_c=use_c, nthreads=nt, **kw)
    else:
        wv = numpy.asarray(wv, order='C').copy()
        wv[0] *= .5
        if use_c:
            lib.num_threads(nt)
        vmat = vmat_gga(chi, wv, out=vmat_out, use_c=use_c, nthreads=nt, **kw)

    return rho, exc, vmat


def _cast_outputs(dms, nelec, excsum, vmat):
    if isinstance(dms, numpy.ndarray):
        dtype = dms.dtype
    else:
        dtype = numpy.result_type(*dms)
    if numpy.asarray(vmat).dtype != dtype:
        vmat = numpy.asarray(vmat, dtype=dtype)
    return nelec, excsum, vmat
