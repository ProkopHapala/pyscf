import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

from . import get_ctx, get_queue, get_prg, round_up
from .xc_grid import matmul_gpu, _knl

TILE = 32

def df_jk_gpu(dfobj, dm, hermi=0, with_j=True, with_k=True):
    '''DF J/K contraction on GPU using tiled GEMM.

    All computation in float32. Returns vj, vk as float64 arrays.

    J: vj = unpack_tril( dmtril * cderi^T * cderi )
    K: vk = sum_P cderi_P[i,j] * dm[j,k] * cderi_P[k,i]
         = einsum('pij,jk->pki', cderi, dm) then einsum('pki,pkj->ij', ...)
    '''
    ctx = get_ctx()
    queue = get_queue()
    prg = get_prg()

    dms = np.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1, nao, nao)
    nset = dms.shape[0]

    # Get cderi (the full Cholesky-decomposed 3-center integral tensor)
    if dfobj._cderi is None:
        dfobj.build()
    from pyscf.df import addons
    with addons.load(dfobj._cderi, dfobj._dataname) as feri:
        if isinstance(feri, np.ndarray):
            cderi = np.asarray(feri, dtype=np.float32)
        else:
            cderi = np.asarray(feri[:], dtype=np.float32)

    nao_pair = cderi.shape[1]
    naux = cderi.shape[0]
    assert nao_pair == nao * (nao + 1) // 2, f'nao_pair mismatch: {nao_pair} vs {nao*(nao+1)//2}'

    vj = None
    vk = None

    if with_j:
        # dmtril: packed triangular of (dm + dm^T)
        idx = np.arange(nao)
        dmtril = np.zeros((nset, nao_pair), dtype=np.float32)
        for k in range(nset):
            dm_sym = dms[k] + dms[k].conj().T
            dmtril[k] = _pack_tril_cpu(dm_sym.astype(np.float32))
            dmtril[k, idx*(idx+1)//2+idx] *= 0.5

        # tmp = dmtril * cderi^T  -> [nset, naux]
        # vj_packed = tmp * cderi -> [nset, nao_pair]
        vj_packed = np.zeros((nset, nao_pair), dtype=np.float32)
        for k in range(nset):
            tmp = matmul_gpu(dmtril[k:k+1], cderi, transpose_B=True)  # [1, naux]
            vj_packed[k] = matmul_gpu(tmp, cderi)[0]  # [1, nao_pair] -> [nao_pair]

        # Unpack triangular to full
        vj = np.zeros((nset, nao, nao), dtype=np.float64)
        for k in range(nset):
            vj_full = _unpack_tril_gpu(prg, queue, ctx, vj_packed[k], nao)
            vj[k] = vj_full.astype(np.float64)

    if with_k:
        # Unpack cderi on GPU
        cderi_full = np.zeros((naux, nao, nao), dtype=np.float32)
        for p in range(naux):
            cderi_full[p] = _unpack_tril_gpu(prg, queue, ctx, cderi[p], nao)

        vk = np.zeros((nset, nao, nao), dtype=np.float64)
        for k in range(nset):
            dm32 = np.ascontiguousarray(dms[k], dtype=np.float32)

            # buf1[p, i, k] = sum_j cderi_full[p, i, j] * dm[j, k]
            # Reshape cderi_full [naux, nao, nao] -> [naux*nao, nao], matmul with dm
            cderi_2d = np.ascontiguousarray(cderi_full.reshape(naux * nao, nao))
            buf1_2d = matmul_gpu(cderi_2d, dm32)  # [naux*nao, nao]
            buf1 = buf1_2d.reshape(naux, nao, nao)  # [p, i, k]

            # vk = einsum('ipk,pkj->ij', buf1, cderi_full)
            # buf1_reshaped[i, p*nao+k] = buf1[i, p, k]
            # cderi_reshaped[p*nao+k, j] = cderi_full[p, k, j]
            buf1_r = np.ascontiguousarray(buf1.transpose(1, 0, 2).reshape(nao, naux * nao))
            cderi_r = np.ascontiguousarray(cderi_full.reshape(naux * nao, nao))
            vk[k] = matmul_gpu(buf1_r, cderi_r).astype(np.float64)

    if vj is not None:
        vj = vj.reshape(dm_shape)
    if vk is not None:
        vk = vk.reshape(dm_shape)

    return vj, vk

def _pack_tril_cpu(mat):
    '''Pack lower triangular of a symmetric matrix.'''
    nao = mat.shape[0]
    idx = np.tril_indices(nao)
    return mat[idx]

def _unpack_tril_gpu(prg, queue, ctx, tril, nao):
    '''Unpack triangular packed to full symmetric matrix on GPU.'''
    tril_f32 = np.ascontiguousarray(tril, dtype=np.float32)
    full = np.zeros((nao, nao), dtype=np.float32)
    bufTril = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, tril_f32.nbytes, tril_f32)
    bufFull = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, full.nbytes)
    _knl(prg, 'unpack_tril')(
        queue, (round_up(nao, TILE), round_up(nao, TILE)), (TILE, TILE),
        bufTril, bufFull,
        np.int32(nao)
    )
    cl.enqueue_copy(queue, full, bufFull)
    queue.finish()
    bufTril.release()
    bufFull.release()
    return full
