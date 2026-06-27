import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

from . import get_ctx, get_queue, get_prg, round_up
from .xc_grid import matmul_gpu, matmul_gpu_buf, _knl

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

    cderi = np.ascontiguousarray(cderi, dtype=np.float32)
    nao_pair = cderi.shape[1]
    naux = cderi.shape[0]
    assert nao_pair == nao * (nao + 1) // 2, f'nao_pair mismatch: {nao_pair} vs {nao*(nao+1)//2}'
    fbytes = np.dtype(np.float32).itemsize
    bufCderi = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, cderi.nbytes, cderi)

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
        bufDmtril = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, dmtril.nbytes, dmtril)
        bufTmp = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, nset * naux * fbytes)
        bufVjPacked = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, nset * nao_pair * fbytes)
        matmul_gpu_buf(bufDmtril, bufCderi, bufTmp, nset, naux, nao_pair, transpose_B=True)
        matmul_gpu_buf(bufTmp, bufCderi, bufVjPacked, nset, nao_pair, naux)

        # Unpack triangular to full
        vj = _unpack_tril_batched_from_buf_gpu(prg, queue, ctx, bufVjPacked, nset, nao, nao_pair).astype(np.float64)
        bufDmtril.release()
        bufTmp.release()
        bufVjPacked.release()

    if with_k:
        # Unpack cderi on GPU
        bufCderiFull = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, naux * nao * nao * fbytes)
        _unpack_tril_batched_to_buf_gpu(prg, queue, ctx, bufCderi, bufCderiFull, naux, nao, nao_pair)

        vk = np.zeros((nset, nao, nao), dtype=np.float64)
        vk_tmp = np.empty((nao, nao), dtype=np.float32)
        bufDm = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, nao * nao * fbytes)
        bufBuf1 = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, naux * nao * nao * fbytes)
        bufBuf1R = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, nao * naux * nao * fbytes)
        bufVk = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, nao * nao * fbytes)
        for k in range(nset):
            dm32 = np.ascontiguousarray(dms[k], dtype=np.float32)
            cl.enqueue_copy(queue, bufDm, dm32).wait()

            # buf1[p, i, k] = sum_j cderi_full[p, i, j] * dm[j, k]
            # Reshape cderi_full [naux, nao, nao] -> [naux*nao, nao], matmul with dm
            matmul_gpu_buf(bufCderiFull, bufDm, bufBuf1, naux * nao, nao, nao)

            # vk = einsum('ipk,pkj->ij', buf1, cderi_full)
            # buf1_reshaped[i, p*nao+k] = buf1[i, p, k]
            # cderi_reshaped[p*nao+k, j] = cderi_full[p, k, j]
            _knl(prg, 'transpose_k_buf1')(
                queue, (round_up(nao, TILE), round_up(naux * nao, TILE)), (TILE, TILE),
                bufBuf1, bufBuf1R,
                np.int32(naux), np.int32(nao)
            )
            matmul_gpu_buf(bufBuf1R, bufCderiFull, bufVk, nao, nao, naux * nao)
            cl.enqueue_copy(queue, vk_tmp, bufVk).wait()
            vk[k] = vk_tmp.astype(np.float64)
        bufCderiFull.release()
        bufDm.release()
        bufBuf1.release()
        bufBuf1R.release()
        bufVk.release()

    bufCderi.release()

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
    return _unpack_tril_batched_gpu(prg, queue, ctx, tril, nao)[0]

def _unpack_tril_batched_gpu(prg, queue, ctx, tril, nao):
    tril_f32 = np.ascontiguousarray(tril, dtype=np.float32)
    if tril_f32.ndim == 1:
        tril_f32 = tril_f32.reshape(1, -1)
    nbatch, nao_pair = tril_f32.shape
    bufTril = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, tril_f32.nbytes, tril_f32)
    full = _unpack_tril_batched_from_buf_gpu(prg, queue, ctx, bufTril, nbatch, nao, nao_pair)
    bufTril.release()
    return full

def _unpack_tril_batched_from_buf_gpu(prg, queue, ctx, bufTril, nbatch, nao, nao_pair):
    full = np.empty((nbatch, nao, nao), dtype=np.float32)
    bufFull = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, full.nbytes)
    _unpack_tril_batched_to_buf_gpu(prg, queue, ctx, bufTril, bufFull, nbatch, nao, nao_pair)
    cl.enqueue_copy(queue, full, bufFull).wait()
    bufFull.release()
    return full

def _unpack_tril_batched_to_buf_gpu(prg, queue, ctx, bufTril, bufFull, nbatch, nao, nao_pair):
    _knl(prg, 'unpack_tril_batched')(
        queue, (round_up(nao, TILE), round_up(nao, TILE), nbatch), (TILE, TILE, 1),
        bufTril, bufFull,
        np.int32(nbatch), np.int32(nao), np.int32(nao_pair)
    )
    return bufFull
