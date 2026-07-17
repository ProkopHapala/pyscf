'''Drop-in nr_rks with grid-parallel ρ/vmat (parallel axis = grid index).

ao_mode:
  'cache'  — full χ in GridWorkspace (default; fastest multi-cycle when RAM fits)
  'stream' — block_loop eval_ao + C stream kernels; no full χ (~GB saved on PTCDA)
'''
import numpy
from pyscf import lib

from .layout import ensure_native
from .rho import rho_lda, rho_gga
from .vmat import vmat_lda, vmat_gga
from .parallel import default_n_workers
from ._ctypes import (
    has_c_lib, c_stream_rho_lda, c_stream_rho_gga,
    c_stream_vmat_lda_acc, c_stream_vmat_gga_acc, c_stream_vmat_hermi,
)

NAO_MAX_DEFAULT = 400  # PTCDA 6-31g ≈ 286; was 200 (benzene-era policy)


def nr_rks(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
           max_memory=2000, verbose=None, n_workers=None, tile_size=None,
           precompute_ao=False, blas_threads=None, ws=None, ao_mode=None):
    '''RKS XC for small systems.

    Grid-parallel ρ/vmat: each thread owns a contiguous grid slice χ[g0:g1,:].
    BLAS is pinned to 1 thread by default so grid workers do not oversubscribe.

    ws: optional GridWorkspace with preallocated buffers (call ws.eval_ao once
        per geometry before the SCF loop). Used only for ao_mode='cache'.

    ao_mode: 'cache' | 'stream' | None
        None → 'stream' if ni._smallDFT_ao_mode == 'stream', else 'cache' when
        ws/precompute_ao set, else 'stream' (block_loop, no full χ).
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
    pool = None

    if ao_mode is None:
        ao_mode = getattr(ni, '_smallDFT_ao_mode', None)
    if ao_mode is None:
        if precompute_ao or ws is not None:
            ao_mode = 'cache'
        else:
            ao_mode = 'stream'
    if ao_mode not in ('cache', 'stream'):
        raise ValueError(f"ao_mode must be 'cache' or 'stream', got {ao_mode!r}")

    try:
        if ao_mode == 'cache':
            if ws is not None and ws.matches(nao, grids.coords.shape[0], deriv):
                if ws.chi is None or precompute_ao:
                    ws.eval_ao(mol, grids)
                chi = ws.chi
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
            # stream: never allocate full χ; C block kernels + hermi once (GGA)
            for i in range(nset):
                dm = numpy.asarray(dms[i], order='C')
                v_acc = numpy.zeros((nao, nao), dtype=numpy.double)
                for ao, mask, weight, _coords in ni.block_loop(
                        mol, grids, nao, deriv, max_memory=max_memory):
                    chi = ensure_native(ao, deriv=deriv)
                    rho, exc = _stream_block_xc(
                        ni, xc_code, xctype, dm, chi, weight, deriv,
                        v_acc, use_c=use_c, omp_threads=omp_saved)
                    nelec[i] += rho_nelec(rho, weight, xctype)
                    excsum[i] += numpy.dot(nelec_weight(rho, weight, xctype), exc)
                if xctype == 'GGA' and use_c:
                    c_stream_vmat_hermi(v_acc)
                elif xctype == 'GGA':
                    v_acc = v_acc + v_acc.T
                vmat[i] = v_acc
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


def _stream_block_xc(ni, xc_code, xctype, dm, chi, weight, deriv, v_acc,
                     use_c=False, omp_threads=None):
    '''ρ + libxc on one AO block; accumulate vmat (GGA hermi deferred).'''
    nt = omp_threads or default_n_workers()
    if use_c and deriv == 0:
        rho = numpy.empty(chi.shape[0], dtype=numpy.double)
        lib.num_threads(nt)
        c_stream_rho_lda(rho, chi, dm, nthreads=nt)
    elif use_c and deriv == 1:
        rho = numpy.empty((4, chi.shape[1]), dtype=numpy.double, order='C')
        lib.num_threads(nt)
        c_stream_rho_gga(rho, chi, dm, nthreads=nt, hermi=1)
    elif deriv == 0:
        rho = rho_lda(dm, chi, use_c=False)
    else:
        rho = rho_gga(dm, chi, hermi=1, use_c=False)

    exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=0)[:2]
    wv = weight * vxc

    if xctype == 'LDA':
        if use_c:
            lib.num_threads(nt)
            c_stream_vmat_lda_acc(v_acc, chi, wv, nthreads=nt)
        else:
            v_acc += vmat_lda(chi, wv, use_c=False)
    else:
        wv = numpy.asarray(wv, order='C').copy()
        wv[0] *= .5
        if use_c:
            lib.num_threads(nt)
            c_stream_vmat_gga_acc(v_acc, chi, wv, nthreads=nt)
        else:
            v_acc += vmat_gga(chi, wv, use_c=False, hermi_sum=False)

    return rho, exc


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
