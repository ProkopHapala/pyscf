import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

from . import get_ctx, get_queue, get_prg, round_up
from .xc_grid import matmul_gpu, matmul_gpu_buf, _knl

TILE = 32

class DFJKPlan:
    def __init__(self, dfobj, nao):
        self.dfobj = dfobj
        self.nao = int(nao)
        self.ctx = get_ctx()
        self.queue = get_queue()
        self.prg = get_prg()
        self.fbytes = np.dtype(np.float32).itemsize
        if dfobj._cderi is None:
            dfobj.build()
        from pyscf.df import addons
        with addons.load(dfobj._cderi, dfobj._dataname) as feri:
            if isinstance(feri, np.ndarray):
                cderi = np.asarray(feri, dtype=np.float32)
            else:
                cderi = np.asarray(feri[:], dtype=np.float32)
        self.cderi = np.ascontiguousarray(cderi, dtype=np.float32)
        self.nao_pair = self.cderi.shape[1]
        self.naux = self.cderi.shape[0]
        assert self.nao_pair == self.nao * (self.nao + 1) // 2, f'nao_pair mismatch: {self.nao_pair} vs {self.nao*(self.nao+1)//2}'
        self.bufCderi = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, self.cderi.nbytes, self.cderi)
        self.tril_idx = np.tril_indices(self.nao)
        idx = np.arange(self.nao)
        self.diag_pack_idx = idx * (idx + 1) // 2 + idx
        self.nset_alloc = 0
        self.bufDmtril = None
        self.bufTmp = None
        self.bufVjPacked = None
        self.bufVjFull = None
        self.vj_full = None
        self.bufCderiFull = None
        self.bufDmAll = None
        self.bufBuf1All = None
        self.bufBuf1RAll = None
        self.bufVkAll = None
        self.vk_all = None
        self.nset_k_alloc = 0

    def ensure_nset(self, nset):
        if nset <= self.nset_alloc:
            return
        for name in ('bufDmtril', 'bufTmp', 'bufVjPacked', 'bufVjFull'):
            buf = getattr(self, name)
            if buf is not None:
                buf.release()
        self.nset_alloc = int(nset)
        self.bufDmtril = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, self.nset_alloc * self.nao_pair * self.fbytes)
        self.bufTmp = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, self.nset_alloc * self.naux * self.fbytes)
        self.bufVjPacked = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, self.nset_alloc * self.nao_pair * self.fbytes)
        self.bufVjFull = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, self.nset_alloc * self.nao * self.nao * self.fbytes)
        self.vj_full = np.empty((self.nset_alloc, self.nao, self.nao), dtype=np.float32)

    def ensure_k_buffers(self, nset=1):
        if self.bufCderiFull is None:
            self.bufCderiFull = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, self.naux * self.nao * self.nao * self.fbytes)
            _unpack_tril_batched_to_buf_gpu(self.prg, self.queue, self.ctx, self.bufCderi, self.bufCderiFull, self.naux, self.nao, self.nao_pair)
        if nset <= self.nset_k_alloc:
            return
        for name in ('bufDmAll', 'bufBuf1All', 'bufBuf1RAll', 'bufVkAll'):
            buf = getattr(self, name)
            if buf is not None:
                buf.release()
        self.nset_k_alloc = int(nset)
        ns = self.nset_k_alloc
        self.bufDmAll = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, ns * self.nao * self.nao * self.fbytes)
        self.bufBuf1All = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, self.naux * self.nao * ns * self.nao * self.fbytes)
        self.bufBuf1RAll = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, ns * self.nao * self.naux * self.nao * self.fbytes)
        self.bufVkAll = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, ns * self.nao * self.nao * self.fbytes)
        self.vk_all = np.empty((ns, self.nao, self.nao), dtype=np.float32)

    def get_jk(self, dm, hermi=0, with_j=True, with_k=True):
        dms = np.asarray(dm)
        dm_shape = dms.shape
        nao = dm_shape[-1]
        assert nao == self.nao, f'nao mismatch: {nao} vs {self.nao}'
        dms = dms.reshape(-1, nao, nao)
        nset = dms.shape[0]
        self.ensure_nset(nset)
        vj = None
        vk = None

        if with_j:
            dm_sym = dms + dms.conj().transpose(0, 2, 1)
            dmtril = np.ascontiguousarray(dm_sym[:, self.tril_idx[0], self.tril_idx[1]], dtype=np.float32)
            dmtril[:, self.diag_pack_idx] *= 0.5
            cl.enqueue_copy(self.queue, self.bufDmtril, dmtril).wait()
            matmul_gpu_buf(self.bufDmtril, self.bufCderi, self.bufTmp, nset, self.naux, self.nao_pair, transpose_B=True)
            matmul_gpu_buf(self.bufTmp, self.bufCderi, self.bufVjPacked, nset, self.nao_pair, self.naux)
            _unpack_tril_batched_to_buf_gpu(self.prg, self.queue, self.ctx, self.bufVjPacked, self.bufVjFull, nset, nao, self.nao_pair)
            cl.enqueue_copy(self.queue, self.vj_full[:nset], self.bufVjFull).wait()
            vj = self.vj_full[:nset].astype(np.float64)

        if with_k:
            self.ensure_k_buffers(nset)
            dm_all = np.ascontiguousarray(dms.transpose(1, 0, 2).reshape(nao, nset * nao), dtype=np.float32)
            cl.enqueue_copy(self.queue, self.bufDmAll, dm_all).wait()
            matmul_gpu_buf(self.bufCderiFull, self.bufDmAll, self.bufBuf1All, self.naux * nao, nset * nao, nao)
            _knl(self.prg, 'transpose_k_buf1_batched')(
                self.queue, (round_up(nao, TILE), round_up(self.naux * nao, TILE), nset), (TILE, TILE, 1),
                self.bufBuf1All, self.bufBuf1RAll,
                np.int32(self.naux), np.int32(nao), np.int32(nset)
            )
            matmul_gpu_buf(self.bufBuf1RAll, self.bufCderiFull, self.bufVkAll, nset * nao, nao, self.naux * nao)
            cl.enqueue_copy(self.queue, self.vk_all[:nset], self.bufVkAll).wait()
            vk = self.vk_all[:nset].astype(np.float64)

        if vj is not None:
            vj = vj.reshape(dm_shape)
        if vk is not None:
            vk = vk.reshape(dm_shape)
        return vj, vk

    def release(self):
        for name in ('bufCderi', 'bufDmtril', 'bufTmp', 'bufVjPacked', 'bufVjFull', 'bufCderiFull', 'bufDmAll', 'bufBuf1All', 'bufBuf1RAll', 'bufVkAll'):
            buf = getattr(self, name, None)
            if buf is not None:
                buf.release()


_df_plan_cache = {}


def get_df_jk_plan(dfobj, nao):
    key = (id(dfobj), int(nao))
    plan = _df_plan_cache.get(key)
    if plan is not None and plan.nao == int(nao):
        return plan
    plan = DFJKPlan(dfobj, nao)
    _df_plan_cache[key] = plan
    return plan


def df_jk_gpu(dfobj, dm, hermi=0, with_j=True, with_k=True):
    '''DF J/K contraction on GPU using tiled GEMM.

    All computation in float32. Returns vj, vk as float64 arrays.

    J: vj = unpack_tril( dmtril * cderi^T * cderi )
    K: vk = sum_P cderi_P[i,j] * dm[j,k] * cderi_P[k,i]
         = einsum('pij,jk->pki', cderi, dm) then einsum('pki,pkj->ij', ...)
    '''
    dm_arr = np.asarray(dm)
    return get_df_jk_plan(dfobj, dm_arr.shape[-1]).get_jk(dm_arr, hermi=hermi, with_j=with_j, with_k=with_k)

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
