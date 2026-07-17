import time as _time
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

from pyscf.dft.gen_grid import ALIGNMENT_UNIT
from pyscf.gto.eval_gto import BLKSIZE, NBINS, CUTOFF

from .gpu_timing import profile_kernel, profile_call, event_elapsed_s
from . import get_ctx, get_queue, get_prg, round_up, get_device_mem_info
from .tile_config import get_active_tile_config
from .buffers import CLBuffer

TILE = 32
FBYTES = np.dtype(np.float32).itemsize
DBYTES = np.dtype(np.float64).itemsize

# Cache kernel objects to avoid repeated retrieval
_kernels = {}


def clear_xc_plan_cache():
    '''Release cached XCGridPlan objects (required after OpenCL reinit / tile recompile).'''
    global _xc_plan_cache
    for plan in list(_xc_plan_cache.values()):
        try:
            plan.release()
        except Exception:
            pass
    _xc_plan_cache.clear()
    _kernels.clear()

def _knl(prg, name):
    if name not in _kernels:
        _kernels[name] = cl.Kernel(prg, name)
    return _kernels[name]

def _timing_record(timing, key, t0):
    if timing is not None:
        timing[key] = _time.perf_counter() - t0

def _gpu_sync(queue):
    queue.finish()

def _finalize_gpu_timing(timing):
    '''Sum stage timers into gpu_total / host_total / wall_profiled.'''
    if not timing:
        return timing
    gpu_keys = ('gpu_dm_cart', 'gpu_rho', 'gpu_xc_pbe', 'gpu_xc_reduce', 'gpu_vmat')
    host_keys = ('host_h2d_dm', 'host_dm_cart', 'host_rho_d2h', 'host_xc_libxc', 'host_xc_reduce', 'host_vmat_d2h', 'host_pair_mask', 'host_cpu_projection')
    timing['gpu_total'] = sum(timing.get(k, 0.0) for k in gpu_keys)
    timing['host_total'] = sum(timing.get(k, 0.0) for k in host_keys)
    timing['wall_profiled'] = timing['gpu_total'] + timing['host_total']
    cl_gpu = sum(timing.get(k + '_cl', 0.0) for k in gpu_keys if k + '_cl' in timing)
    if cl_gpu > 0:
        timing['gpu_total_cl'] = cl_gpu
    return timing

# Order for benchmark printouts (seconds on plan.last_timing).
TIMING_STAGE_ORDER = (
    'host_h2d_dm', 'gpu_dm_cart', 'host_dm_cart', 'gpu_rho', 'host_rho_d2h',
    'gpu_xc_pbe', 'host_xc_libxc', 'host_xc_reduce', 'gpu_xc_reduce',
    'gpu_vmat_split', 'gpu_vmat_reduce', 'gpu_vmat', 'host_vmat_d2h', 'host_pair_mask', 'host_cpu_projection',
    'gpu_total', 'gpu_total_cl', 'host_total', 'wall_profiled', 'n_blocks',
)

def _is_pbe_xc(xc_code):
    s = str(xc_code).upper().replace(' ', '')
    if s != 'PBE' and 'GGA_X_PBE' not in s:
        return False
    for bad in ('PBE_SOL', 'PBESOL', 'RPBE', 'PBE0', 'PBELOC', 'PBEINT', 'PBE_VWN', 'PBE_MOL'):
        if bad in s:
            return False
    return s == 'PBE' or ('GGA_X_PBE' in s and 'GGA_C_PBE' in s)

def _resolve_xc_eval(xc_eval, gpu_xc, xc_code, xctype):
    '''Resolve XC evaluation backend.

    xc_eval: 'gpu' (default, no rho/wv PCIe) | 'cpu' (libxc debug path with D2H/H2D).
    gpu_xc: legacy precision selector 'auto'|'pbe_f32'|'pbe_f64', or 'cpu'/'libxc' to force CPU.
    Returns (mode, precision) with mode in ('gpu','cpu') and precision in ('pbe_f32','pbe_f64') or None.
    '''
    if gpu_xc in ('cpu', 'libxc'):
        return 'cpu', None
    if xc_eval not in ('gpu', 'cpu'):
        raise ValueError(f"xc_eval must be 'gpu' or 'cpu'; got {xc_eval!r}")
    if xc_eval == 'cpu':
        return 'cpu', None
    prec = 'pbe_f32' if gpu_xc in (None, 'auto') else gpu_xc
    if prec not in ('pbe_f32', 'pbe_f64'):
        raise ValueError(f"gpu_xc={gpu_xc!r}; use auto, pbe_f32, pbe_f64, or cpu/libxc for debug")
    if xctype != 'GGA' or not _is_pbe_xc(xc_code):
        raise ValueError(f"xc_eval='gpu' requires unmodified PBE GGA (xc_code={xc_code!r}); use xc_eval='cpu'")
    return 'gpu', prec


def _resolve_gpu_xc(gpu_xc, xc_code, xctype):
    '''Legacy alias: returns pbe precision string or None for CPU path.'''
    mode, prec = _resolve_xc_eval('gpu' if gpu_xc not in ('cpu', 'libxc') else 'cpu', gpu_xc, xc_code, xctype)
    return prec if mode == 'gpu' else None


def _alloc_xc_gpu_bufs(ctx, ngrids, precision):
    mf = cl.mem_flags
    n_partial = round_up(ngrids, TILE) // TILE + 1
    reduce_bufs = {
        'buf_nelec_exc': cl.Buffer(ctx, mf.READ_WRITE, 2 * ngrids * FBYTES),
        'buf_reduce0': cl.Buffer(ctx, mf.READ_WRITE, n_partial * FBYTES),
        'buf_reduce1': cl.Buffer(ctx, mf.READ_WRITE, n_partial * FBYTES),
    }
    if precision == 'pbe_f32':
        return {
            **reduce_bufs,
            'buf_exc': cl.Buffer(ctx, mf.READ_WRITE, ngrids * FBYTES),
            'buf_vrho': cl.Buffer(ctx, mf.READ_WRITE, ngrids * FBYTES),
            'buf_vsigma': cl.Buffer(ctx, mf.READ_WRITE, ngrids * FBYTES),
            'exc_host': np.empty(ngrids, dtype=np.float32),
        }
    return {
        **reduce_bufs,
        'buf_exc': cl.Buffer(ctx, mf.READ_WRITE, ngrids * DBYTES),
        'buf_vrho': cl.Buffer(ctx, mf.READ_WRITE, ngrids * DBYTES),
        'buf_vsigma': cl.Buffer(ctx, mf.READ_WRITE, ngrids * DBYTES),
        'buf_rho64': cl.Buffer(ctx, mf.READ_WRITE, 4 * ngrids * DBYTES),
        'buf_wv64': cl.Buffer(ctx, mf.READ_WRITE, 4 * ngrids * DBYTES),
        'exc_host': np.empty(ngrids, dtype=np.float64),
        'rho64_host': np.empty(4 * ngrids, dtype=np.float64),
        'wv64_host': np.empty(4 * ngrids, dtype=np.float64),
    }

def _gpu_reduce_sum(queue, prg, buf_in, n, buf_partial, buf_level2, offset=0):
    '''Tree-reduce buf_in[offset:offset+n] on GPU; return scalar float.'''
    ng = round_up(n, TILE) // TILE
    if offset == 0:
        _knl(prg, 'reduce_sum')(queue, (round_up(n, TILE),), (TILE,), buf_in, buf_partial, np.int32(n))
    else:
        _knl(prg, 'reduce_sum_offset')(queue, (round_up(n, TILE),), (TILE,), buf_in, np.int32(offset), buf_partial, np.int32(n))
    if ng <= 1:
        out = np.empty(1, dtype=np.float32)
        cl.enqueue_copy(queue, out, buf_partial).wait()
        return float(out[0])
    n2 = ng
    _knl(prg, 'reduce_sum')(queue, (round_up(n2, TILE),), (TILE,), buf_partial, buf_level2, np.int32(n2))
    ng2 = round_up(n2, TILE) // TILE
    if ng2 <= 1:
        out = np.empty(1, dtype=np.float32)
        cl.enqueue_copy(queue, out, buf_level2).wait()
        return float(out[0])
    out = np.empty(ng2, dtype=np.float32)
    cl.enqueue_copy(queue, out, buf_level2).wait()
    return float(out.sum())


def _gpu_nelec_excsum(queue, prg, st, ngrids):
    '''nelec, excsum from on-device rho/weight/exc (no rho/exc D2H).'''
    _knl(prg, 'compute_nelec_exc')(
        queue, (round_up(ngrids, TILE),), (TILE,),
        st['buf_rho'], st['buf_weight'], st['buf_exc'], st['buf_nelec_exc'], np.int32(ngrids))
    nelec = _gpu_reduce_sum(queue, prg, st['buf_nelec_exc'], ngrids, st['buf_reduce0'], st['buf_reduce1'], offset=0)
    excsum = _gpu_reduce_sum(queue, prg, st['buf_nelec_exc'], ngrids, st['buf_reduce0'], st['buf_reduce1'], offset=ngrids)
    return nelec, excsum


def _gpu_dm_to_cart(buf_c2s, buf_dm_sph, buf_scratch, buf_dm_cart, nao, ncart):
    '''dm_cart = c2s @ dm_sph @ c2s.T on GPU.'''
    matmul_gpu_buf(buf_c2s, buf_dm_sph, buf_scratch, ncart, nao, nao)
    matmul_gpu_buf(buf_scratch, buf_c2s, buf_dm_cart, ncart, ncart, nao, transpose_B=True)


def _gpu_vmat_cart_to_sph(buf_c2s, buf_vmat_cart, buf_scratch, buf_vmat_sph, nao, ncart):
    '''vmat_sph = c2s.T @ vmat_cart @ c2s on GPU.'''
    matmul_gpu_buf(buf_c2s, buf_vmat_cart, buf_scratch, nao, ncart, ncart, transpose_A=True)
    matmul_gpu_buf(buf_scratch, buf_c2s, buf_vmat_sph, nao, nao, ncart)

def matmul_gpu_buf(bufA, bufB, bufC, M, N, K, transpose_A=False, transpose_B=False, a_row0=0, b_row0=0, row0=None):
    if row0 is not None:
        a_row0 = row0
    if transpose_A and transpose_B:
        raise NotImplementedError('Both transposed not supported')
    if transpose_A:
        knl_name = 'matmul_tiled_transpose_A'
    elif transpose_B:
        knl_name = 'matmul_tiled_transpose_B'
    else:
        knl_name = 'matmul_tiled'
    queue = get_queue()
    prg = get_prg()
    if knl_name == 'matmul_tiled_transpose_B':
        _knl(prg, knl_name)(
            queue, (round_up(M, TILE), round_up(N, TILE)), (TILE, TILE),
            bufA, bufB, bufC,
            np.int32(M), np.int32(N), np.int32(K))
    elif knl_name == 'matmul_tiled_transpose_A':
        _knl(prg, knl_name)(
            queue, (round_up(M, TILE), round_up(N, TILE)), (TILE, TILE),
            bufA, bufB, bufC,
            np.int32(M), np.int32(N), np.int32(K), np.int32(a_row0), np.int32(b_row0))
    else:
        _knl(prg, knl_name)(
            queue, (round_up(M, TILE), round_up(N, TILE)), (TILE, TILE),
            bufA, bufB, bufC,
            np.int32(M), np.int32(N), np.int32(K), np.int32(a_row0))
    return bufC

def matmul_gpu_buf_accum(bufA, bufB, bufC, M, N, K, transpose_A=False, a_row0=0, b_row0=0, row0=None):
    if row0 is not None:
        a_row0 = row0
    if transpose_A:
        knl_name = 'matmul_tiled_transpose_A_accum'
    else:
        raise NotImplementedError('accum only supported for transpose_A')
    queue = get_queue()
    _knl(get_prg(), knl_name)(
        queue, (round_up(M, TILE), round_up(N, TILE)), (TILE, TILE),
        bufA, bufB, bufC,
        np.int32(M), np.int32(N), np.int32(K), np.int32(a_row0), np.int32(b_row0))
    return bufC

def zero_buffer_gpu(buf, n):
    _knl(get_prg(), 'zero_buffer')(
        get_queue(), (round_up(n, TILE),), (TILE,),
        buf, np.int32(n)
    )

def estimate_precomputed_ao_memory(ngrids, nao, xctype='GGA', nbas=None, include_host=True):
    '''Bytes for precomputed GTO AO on grid (GPU float32 + optional host float64 for CPU sparse).'''
    ncomp = 4 if xctype == 'GGA' else 1
    gpu = ncomp * ngrids * nao * FBYTES
    host = ncomp * ngrids * nao * DBYTES if include_host else 0
    screen = 0 if nbas is None else ((ngrids + BLKSIZE - 1) // BLKSIZE) * nbas
    workspace = (4 * ngrids + nao * nao + 4 * BLKSIZE * nao) * FBYTES
    return {'gpu_ao': gpu, 'host_ao': host, 'screen_index': screen, 'workspace': workspace, 'total': gpu + host + screen + workspace}


class _AoGGABlock:
    '''GGA AO block with F-contiguous components (matches eval_ao layout for sparse C code).'''
    ndim = 3

    def __init__(self, comps):
        self._comps = comps
        self.shape = (len(comps),) + comps[0].shape

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._comps[key]
        return self._comps[key]


def _pack_ao_block(ao_host, ip0, nblk, nao, ncomp, blk_buf):
    '''Pack precomputed AO into eval_ao-compatible strides for sparse C kernels.'''
    for c in range(ncomp):
        ao_c = np.ndarray((nblk, nao), dtype=np.float64, order='F', buffer=blk_buf[c * nblk * nao:(c + 1) * nblk * nao])
        np.copyto(ao_c, ao_host[c][ip0:ip0 + nblk])
    return np.ndarray((ncomp, nblk, nao), dtype=np.float64, buffer=blk_buf[:ncomp * nblk * nao], strides=(nblk * nao * DBYTES, DBYTES, nblk * DBYTES))


def _atom_ao_layout_mol(mol):
    '''Per-atom AO ranges from mol.aoslice_by_atom() -> (sh0, sh1, ao0, ao1).'''
    aorange = mol.aoslice_by_atom()
    atom_ao0 = aorange[:, 2].astype(np.int32)
    atom_nao = (aorange[:, 3] - aorange[:, 2]).astype(np.int32)
    return atom_ao0, atom_nao


def pack_ao_grid_iao_ig_f32(ao_staging, ncomp=None):
    '''Transpose eval_ao blocks [ngrids, nao] -> chi [nao, ngrids] C-contiguous f32 per component.'''
    if ncomp is None:
        ncomp = len(ao_staging)
    return [np.ascontiguousarray(ao_staging[c].T, dtype=np.float32) for c in range(ncomp)]


def _atom_ao_layout(ao_eval):
    natoms = ao_eval.natoms
    atom_ao0 = np.zeros(natoms, dtype=np.int32)
    atom_nao = np.zeros(natoms, dtype=np.int32)
    for ia in range(natoms):
        off = ao_eval.plan.atom_radial_offset[ia]
        ns = ao_eval.plan.atom_radial_offset[ia + 1] - off
        nao_atom = 0
        for s in range(ns):
            ir = ao_eval.plan.atom_radial_list[off + s]
            l = ao_eval.plan.radial_l[ir]
            nao_atom += (l + 1) * (l + 2) // 2
        if ia > 0:
            atom_ao0[ia] = atom_ao0[ia - 1] + atom_nao[ia - 1]
        atom_nao[ia] = nao_atom
    return atom_ao0, atom_nao

def matmul_gpu(A, B, transpose_A=False, transpose_B=False,
               bufA=None, bufB=None, bufC=None):
    '''Tiled matrix multiply on GPU using local memory with preallocated buffers.

    A, B: numpy float32 arrays
    Returns: numpy float32 array C = A * B or A^T * B or A * B^T

    If bufA/bufB/bufC are provided (cl.Buffer), they are used instead of
    creating new ones. When bufC is provided, caller must download result.
    '''
    ctx = get_ctx()
    queue = get_queue()
    prg = get_prg()

    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    if transpose_A and transpose_B:
        raise NotImplementedError('Both transposed not supported')

    if transpose_A:
        K, M = A.shape
        K2, N = B.shape
        assert K == K2, f'K mismatch: {K} vs {K2}'
        knl_name = 'matmul_tiled_transpose_A'
    elif transpose_B:
        M, K = A.shape
        N, K2 = B.shape
        assert K == K2, f'K mismatch: {K} vs {K2}'
        knl_name = 'matmul_tiled_transpose_B'
    else:
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f'K mismatch: {K} vs {K2}'
        knl_name = 'matmul_tiled'

    if bufA is None:
        bufA = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, A.nbytes, A)
    if bufB is None:
        bufB = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, B.nbytes, B)
    if bufC is None:
        C = np.zeros((M, N), dtype=np.float32)
        bufC = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, C.nbytes)
    else:
        C = None

    _knl(prg, knl_name)(
        queue, (round_up(M, TILE), round_up(N, TILE)), (TILE, TILE),
        bufA, bufB, bufC,
        np.int32(M), np.int32(N), np.int32(K)
    )

    if C is not None:
        cl.enqueue_copy(queue, C, bufC).wait()
    return C if C is not None else bufC

class XCGridPlan:
    def __init__(self, mol, grids, xc_code, blk=8192):
        from pyscf.dft import numint
        self.mol = mol
        self.grids = grids
        self.xc_code = xc_code
        self.ni = numint.NumInt()
        self.xctype = self.ni._xc_type(xc_code)
        if self.xctype not in ('LDA', 'GGA'):
            raise NotImplementedError(f'xctype={self.xctype} not supported on GPU')
        self.nao = mol.nao_nr()
        self.ngrids = grids.coords.shape[0]
        self.blk = min(int(blk), max(1, self.ngrids))
        self.ctx = get_ctx()
        self.queue = get_queue()
        self.prg = get_prg()
        fbytes = np.dtype(np.float32).itemsize
        self.ao32_lda = np.empty((self.blk, self.nao), dtype=np.float32)
        self.ao32_gga = np.empty((4, self.blk, self.nao), dtype=np.float32)
        self.rho = np.empty((4, self.blk), dtype=np.float64)
        self.rho32_flat = np.empty(4 * self.blk, dtype=np.float32)
        self.wv_flat = np.empty(4 * self.blk, dtype=np.float32)
        self.vmat_blk = np.empty((self.nao, self.nao), dtype=np.float32)
        self.bufDm = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, self.nao * self.nao * fbytes)
        self.bufAo = [cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, self.blk * self.nao * fbytes) for _ in range(4)]
        self.bufAoDm = [cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, self.blk * self.nao * fbytes) for _ in range(4)]
        self.bufRho = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, 4 * self.blk * fbytes)
        self.bufWv = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, 4 * self.blk * fbytes)
        self.bufAow = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, self.blk * self.nao * fbytes)
        self.bufVmat = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, self.nao * self.nao * fbytes)

    def nr_rks(self, dm):
        nao = self.nao
        dm32 = np.ascontiguousarray(dm, dtype=np.float32)
        cl.enqueue_copy(self.queue, self.bufDm, dm32).wait()
        nelec = 0.0
        excsum = 0.0
        zero_buffer_gpu(self.bufVmat, nao * nao)

        for ip0 in range(0, self.ngrids, self.blk):
            ip1 = min(ip0 + self.blk, self.ngrids)
            nblk = ip1 - ip0
            coords_blk = self.grids.coords[ip0:ip1]
            weight_blk = self.grids.weights[ip0:ip1]

            if self.xctype == 'LDA':
                ao = self.ni.eval_ao(self.mol, coords_blk, deriv=0)  # [nblk, nao] CPU

                ao32 = self.ao32_lda[:nblk]
                ao32[:] = ao
                cl.enqueue_copy(self.queue, self.bufAo[0], ao32)
                matmul_gpu_buf(self.bufAo[0], self.bufDm, self.bufAoDm[0], nblk, nao, nao)
                _knl(self.prg, 'contract_rho_lda_from_aodm')(
                    self.queue, (round_up(nblk, TILE),), (TILE,),
                    self.bufAo[0], self.bufAoDm[0], self.bufRho,
                    np.int32(nao), np.int32(nblk), np.int32(0), np.int32(0)
                )
                cl.enqueue_copy(self.queue, self.rho32_flat[:nblk], self.bufRho).wait()
                rho0 = self.rho[0, :nblk]
                rho0[:] = self.rho32_flat[:nblk]

                exc, vxc = self.ni.eval_xc_eff(self.xc_code, rho0, deriv=1, xctype='LDA', spin=0)[:2]

                den = rho0 * weight_blk
                nelec += float(den.sum())
                excsum += float(np.dot(den, exc))

                self.wv_flat[:nblk] = np.ascontiguousarray(weight_blk * vxc, dtype=np.float32)
                cl.enqueue_copy(self.queue, self.bufWv, self.wv_flat[:nblk])
                _knl(self.prg, 'scale_aow_lda')(
                    self.queue, (round_up(nblk, TILE), round_up(nao, TILE)), (TILE, TILE),
                    self.bufAo[0], self.bufWv, self.bufAow,
                    np.int32(nao), np.int32(nblk), np.int32(0), np.int32(0)
                )
                matmul_gpu_buf_accum(self.bufAow, self.bufAo[0], self.bufVmat, nao, nao, nblk, transpose_A=True)

            elif self.xctype == 'GGA':
                ao = self.ni.eval_ao(self.mol, coords_blk, deriv=1)  # [4, nblk, nao] CPU

                ao32 = self.ao32_gga[:, :nblk]
                ao32[:] = ao
                for c in range(4):
                    cl.enqueue_copy(self.queue, self.bufAo[c], ao32[c])
                    matmul_gpu_buf(self.bufAo[c], self.bufDm, self.bufAoDm[c], nblk, nao, nao)
                _knl(self.prg, 'contract_rho_gga_from_aodm')(
                    self.queue, (round_up(nblk, TILE),), (TILE,),
                    self.bufAo[0], self.bufAo[1], self.bufAo[2], self.bufAo[3],
                    self.bufAoDm[0], self.bufAoDm[1], self.bufAoDm[2], self.bufAoDm[3], self.bufRho,
                    np.int32(nao), np.int32(nblk), np.int32(0), np.int32(nblk), np.int32(0)
                )
                cl.enqueue_copy(self.queue, self.rho32_flat[:4*nblk], self.bufRho).wait()
                rho_blk = self.rho[:, :nblk]
                rho_blk[:] = self.rho32_flat[:4*nblk].reshape(4, nblk)

                evfk = self.ni.eval_xc_eff(self.xc_code, rho_blk, deriv=1, xctype='GGA', spin=0)
                exc = evfk[0]
                vxc = evfk[1]  # (4, nblk): vxc[0]=vrho, vxc[1:4]=dE/d(rho_grad)

                den = rho_blk[0] * weight_blk
                nelec += float(den.sum())
                excsum += float(np.dot(den, exc))

                # wv[c] = weight * vxc[c] for c=0..3
                # wv[0] *= 0.5 for hermi_sum (vmat + vmat.T)
                wv_blk = self.wv_flat[:4*nblk].reshape(4, nblk)
                wv_blk[:] = weight_blk.astype(np.float32)[np.newaxis, :] * np.ascontiguousarray(vxc, dtype=np.float32)
                wv_blk[0] *= 0.5

                # vmat = ao[0]^T @ aow  (then hermi_sum at end)
                cl.enqueue_copy(self.queue, self.bufWv, self.wv_flat[:4*nblk])
                _knl(self.prg, 'scale_aow_gga_split')(
                    self.queue, (round_up(nblk, TILE), round_up(nao, TILE)), (TILE, TILE),
                    self.bufAo[0], self.bufAo[1], self.bufAo[2], self.bufAo[3], self.bufWv, self.bufAow,
                    np.int32(nao), np.int32(nblk), np.int32(0), np.int32(nblk), np.int32(0)
                )
                matmul_gpu_buf_accum(self.bufAow, self.bufAo[0], self.bufVmat, nao, nao, nblk, transpose_A=True)

        cl.enqueue_copy(self.queue, self.vmat_blk, self.bufVmat).wait()
        vmat = self.vmat_blk.astype(np.float64)
        if self.xctype == 'GGA':
            # hermi_sum: vmat + vmat.T (wv[0] was halved to compensate)
            vmat = vmat + vmat.T
        return nelec, excsum, vmat

    def _ensure_full_buffers(self):
        if hasattr(self, 'bufAoDmFull'):
            return True
        fbytes = np.dtype(np.float32).itemsize
        ncomp = 4 if self.xctype == 'GGA' else 1
        needed = (ncomp * self.ngrids * self.nao + 4 * self.ngrids + 4 * self.ngrids + self.ngrids * self.nao) * fbytes
        dev_mem = get_device_mem_info()
        if needed > 0.8 * dev_mem:
            return False
        self.bufAoDmFull = [cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, self.ngrids * self.nao * fbytes) for _ in range(4)]
        self.bufRhoFull = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, 4 * self.ngrids * fbytes)
        self.bufWvFull = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, 4 * self.ngrids * fbytes)
        self.bufAowFull = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, self.ngrids * self.nao * fbytes)
        self.rho32_full = np.empty(4 * self.ngrids, dtype=np.float32)
        self.rho_full = np.empty((4, self.ngrids), dtype=np.float64)
        self.wv_full = np.empty(4 * self.ngrids, dtype=np.float32)
        return True

    def _get_ao_hermite(self, r0_ang=0.002, du=0.02, rmax_ang=None, spline_order='cubic'):
        key = (float(r0_ang), float(du), None if rmax_ang is None else float(rmax_ang), str(spline_order))
        if getattr(self, '_ao_hermite_key', None) == key:
            return self.ao_hermite
        if rmax_ang is None:
            from pyscf.data import nist
            rmax_ang = float(np.max(np.linalg.norm(self.grids.coords[:, None, :] - self.mol.atom_coords()[None, :, :], axis=2)) * nist.BOHR + 0.2)
        from .ao_hermite import OpenCLAOHermiteEvaluator
        self.ao_hermite = OpenCLAOHermiteEvaluator(self.mol, r0_ang=r0_ang, du=du, rmax_ang=max(rmax_ang, 8.0), midpoint_fit=True, spline_order=spline_order)
        self._ao_hermite_key = key
        return self.ao_hermite

    def nr_rks_hermite_ao(self, dm, r0_ang=0.002, du=0.02, rmax_ang=None):
        nao = self.nao
        ngrids = self.ngrids
        if not self._ensure_full_buffers():
            return self.nr_rks(dm)
        dm32 = np.ascontiguousarray(dm, dtype=np.float32)
        cl.enqueue_copy(self.queue, self.bufDm, dm32).wait()
        weight = self.grids.weights
        vmat = np.zeros((nao, nao), dtype=np.float64)
        ao_eval = self._get_ao_hermite(r0_ang=r0_ang, du=du, rmax_ang=rmax_ang)

        if self.xctype == 'LDA':
            bufAo, _ = ao_eval.eval_sph_buf(self.grids.coords)
            matmul_gpu_buf(bufAo, self.bufDm, self.bufAoDmFull[0], ngrids, nao, nao)
            _knl(self.prg, 'contract_rho_lda_from_aodm')(
                self.queue, (round_up(ngrids, TILE),), (TILE,),
                bufAo, self.bufAoDmFull[0], self.bufRhoFull,
                np.int32(nao), np.int32(ngrids), np.int32(0), np.int32(0)
            )
            cl.enqueue_copy(self.queue, self.rho32_full[:ngrids], self.bufRhoFull).wait()
            rho0 = self.rho_full[0, :ngrids]
            rho0[:] = self.rho32_full[:ngrids]
            exc, vxc = self.ni.eval_xc_eff(self.xc_code, rho0, deriv=1, xctype='LDA', spin=0)[:2]
            den = rho0 * weight
            nelec = float(den.sum())
            excsum = float(np.dot(den, exc))
            self.wv_full[:ngrids] = np.ascontiguousarray(weight * vxc, dtype=np.float32)
            cl.enqueue_copy(self.queue, self.bufWvFull, self.wv_full[:ngrids])
            _knl(self.prg, 'scale_aow_lda')(
                self.queue, (round_up(ngrids, TILE), round_up(nao, TILE)), (TILE, TILE),
                bufAo, self.bufWvFull, self.bufAowFull,
                np.int32(nao), np.int32(ngrids), np.int32(0), np.int32(0)
            )
            matmul_gpu_buf(self.bufAowFull, bufAo, self.bufVmat, nao, nao, ngrids, transpose_A=True)
            cl.enqueue_copy(self.queue, self.vmat_blk, self.bufVmat).wait()
            vmat += self.vmat_blk.astype(np.float64)
            return nelec, excsum, vmat

        bufAo = ao_eval.eval_sph_deriv1_buf(self.grids.coords)[0]
        for c in range(4):
            matmul_gpu_buf(bufAo[c], self.bufDm, self.bufAoDmFull[c], ngrids, nao, nao)
        _knl(self.prg, 'contract_rho_gga_from_aodm')(
            self.queue, (round_up(ngrids, TILE),), (TILE,),
            bufAo[0], bufAo[1], bufAo[2], bufAo[3],
            self.bufAoDmFull[0], self.bufAoDmFull[1], self.bufAoDmFull[2], self.bufAoDmFull[3], self.bufRhoFull,
            np.int32(nao), np.int32(ngrids), np.int32(0), np.int32(ngrids), np.int32(0)
        )
        cl.enqueue_copy(self.queue, self.rho32_full[:4*ngrids], self.bufRhoFull).wait()
        rho = self.rho_full[:, :ngrids]
        rho[:] = self.rho32_full[:4*ngrids].reshape(4, ngrids)
        evfk = self.ni.eval_xc_eff(self.xc_code, rho, deriv=1, xctype='GGA', spin=0)
        exc = evfk[0]
        vxc = evfk[1]
        den = rho[0] * weight
        nelec = float(den.sum())
        excsum = float(np.dot(den, exc))
        wv = self.wv_full[:4*ngrids].reshape(4, ngrids)
        wv[:] = weight.astype(np.float32)[np.newaxis, :] * np.ascontiguousarray(vxc, dtype=np.float32)
        wv[0] *= 0.5
        cl.enqueue_copy(self.queue, self.bufWvFull, self.wv_full[:4*ngrids])
        _knl(self.prg, 'scale_aow_gga_split')(
            self.queue, (round_up(ngrids, TILE), round_up(nao, TILE)), (TILE, TILE),
            bufAo[0], bufAo[1], bufAo[2], bufAo[3], self.bufWvFull, self.bufAowFull,
            np.int32(nao), np.int32(ngrids), np.int32(0), np.int32(ngrids), np.int32(0)
        )
        matmul_gpu_buf(self.bufAowFull, bufAo[0], self.bufVmat, nao, nao, ngrids, transpose_A=True)
        cl.enqueue_copy(self.queue, self.vmat_blk, self.bufVmat).wait()
        vmat += self.vmat_blk.astype(np.float64)
        return nelec, excsum, vmat + vmat.T

    def setup_precomputed_gto(self, max_memory_frac=0.75, max_memory_mb=2000, gpu_only=True, gpu_xc='auto', fused='tiled', ao_proj='auto', xc_eval='gpu', spline_order='cubic'):
        '''One-time: project GTO AOs on grid, upload float32 AO to GPU (outside SCF iteration budget).

        gpu_only=True: skip host float64 AO cache (GPU projection path only).
        xc_eval: 'gpu' (default, PBE vxc on device, rho/wv stay on GPU) | 'cpu' (libxc debug + D2H/H2D).
        gpu_xc: 'auto'|'pbe_f32'|'pbe_f64' precision when xc_eval='gpu'; 'cpu'/'libxc' forces debug path.
        ao_proj: 'auto' | 'cpu' | 'hermite_gpu' — how to fill buf_ao/buf_chi at setup.
          auto: GPU Hermite tiled projection when lmax<=3, else CPU PySCF eval_ao.
          hermite_gpu: eval_ao_hermite_cart_deriv1_tiled + c2s (+ transpose for coalesced).
        fused: projection strategy for per-SCF rho/vmat (see below).
          precomp_gto_rowmajor ('tiled', default): row-major χ[iG,iAO] + rho_gga_precomp_pair + vmat_gga_precomp_pair.
          precomp_gto_coalesced ('coalesced'): χ[iAO,iG] + rho/vmat_gga_precomp_coalesced_pair.
          precomp_radial_hermite ('radial_precomp'): R,dR on grid + rho/vmat_gga_radial_precomp_pair (no AO upload).
          gemm: full-grid GEMM + contract (slow fallback).
          False: Python block loop + tiled matmul.
        '''
        if fused is True:
            fused = 'tiled'
        if fused in ('coalesced', 'radial_precomp') and self.xctype != 'GGA':
            raise NotImplementedError(f'fused={fused!r} requires GGA (LDA kernels not implemented)')
        from . import init_device
        init_device(quiet=getattr(self, '_pcg_ready', False))
        self.ctx = get_ctx()
        self.queue = get_queue()
        self.prg = get_prg()
        mol = self.mol
        grids = self.grids
        if grids.coords is None or grids.non0tab is None:
            grids.build(with_non0tab=True)
        nao = self.nao
        ngrids = self.ngrids
        nbas = mol.nbas
        mem = estimate_precomputed_ao_memory(ngrids, nao, self.xctype, nbas=nbas, include_host=not gpu_only)
        dev_mem = get_device_mem_info()
        if mem['gpu_ao'] + mem['workspace'] > max_memory_frac * dev_mem:
            raise MemoryError(
                f'Precomputed AO GPU need ~{mem["gpu_ao"]/1e6:.0f} MB + workspace; '
                f'device has {dev_mem/1e6:.0f} MB (limit {max_memory_frac*100:.0f}%)')
        t0 = _time.perf_counter()
        screen_index = grids.non0tab
        blksize = int(max_memory_mb * 1e6 / ((5 if self.xctype == 'GGA' else 2) * nao * 4 * BLKSIZE))
        blksize = max(BLKSIZE, min(blksize, ngrids // BLKSIZE + 1, 1200)) * BLKSIZE
        blksize = min(blksize, self.blk)
        ncomp = 4 if self.xctype == 'GGA' else 1
        ao_deriv = 1 if self.xctype == 'GGA' else 0
        ao_host = None if gpu_only else [np.empty((ngrids, nao), order='F', dtype=np.float64) for _ in range(ncomp)]
        need_gto_ao = fused not in ('radial_precomp',)
        use_hermite_ao = False
        if need_gto_ao:
            if ao_proj == 'hermite_gpu':
                use_hermite_ao = True
            elif ao_proj == 'auto':
                use_hermite_ao = self._get_ao_hermite(spline_order=spline_order).plan.lmax <= 3 and self.xctype == 'GGA'
            elif ao_proj != 'cpu':
                raise ValueError(f"ao_proj must be 'auto', 'cpu', or 'hermite_gpu'; got {ao_proj!r}")
            if use_hermite_ao and self.xctype != 'GGA':
                raise NotImplementedError('ao_proj=hermite_gpu requires GGA (deriv1 tiled kernel)')
        ao_staging = None if use_hermite_ao or not need_gto_ao else [np.empty((ngrids, nao), dtype=np.float32) for _ in range(ncomp)]
        chi_staging = None if use_hermite_ao or fused != 'coalesced' else [np.empty((nao, ngrids), dtype=np.float32) for _ in range(ncomp)]
        t_eval0 = _time.perf_counter()
        t_eval = 0.0
        if need_gto_ao and not use_hermite_ao:
            for ip0 in range(0, ngrids, blksize):
                ip1 = min(ip0 + blksize, ngrids)
                coords = grids.coords[ip0:ip1]
                mask = screen_index[ip0 // BLKSIZE:]
                ao = self.ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask, cutoff=grids.cutoff)
                if ao_deriv:
                    for c in range(ncomp):
                        blk = ao[c].astype(np.float32)
                        ao_staging[c][ip0:ip1] = blk
                        if chi_staging is not None:
                            chi_staging[c][:, ip0:ip1] = blk.T
                        if ao_host is not None:
                            ao_host[c][ip0:ip1] = ao[c]
                else:
                    blk = ao.astype(np.float32)
                    ao_staging[0][ip0:ip1] = blk
                    if chi_staging is not None:
                        chi_staging[0][:, ip0:ip1] = blk.T
                    if ao_host is not None:
                        ao_host[0][ip0:ip1] = ao
            t_eval = _time.perf_counter() - t_eval0
        elif not need_gto_ao:
            t_eval = 0.0
        t_rad = 0.0
        mf = cl.mem_flags
        buf_ao = []
        buf_chi_gpu = None
        t_up0 = _time.perf_counter()
        if need_gto_ao:
            for c in range(ncomp):
                if use_hermite_ao:
                    buf_ao.append(cl.Buffer(self.ctx, mf.READ_ONLY, ngrids * nao * FBYTES))
                else:
                    buf_ao.append(cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, ao_staging[c].nbytes, ao_staging[c]))
            if use_hermite_ao and fused == 'coalesced':
                buf_chi_gpu = [cl.Buffer(self.ctx, mf.READ_ONLY, nao * ngrids * FBYTES) for _ in range(ncomp)]
        if use_hermite_ao:
            self._get_ao_hermite(spline_order=spline_order).project_sph_deriv1_to_bufs(grids.coords, buf_ao, buf_chi_gpu)
            t_eval = _time.perf_counter() - t_eval0
        buf_rho = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * ngrids * FBYTES)
        buf_wv = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * ngrids * FBYTES)
        buf_vmat = None
        buf_aodm = [cl.Buffer(self.ctx, mf.READ_WRITE, blksize * nao * FBYTES) for _ in range(4)]
        buf_aow = cl.Buffer(self.ctx, mf.READ_WRITE, blksize * nao * FBYTES)
        buf_aodm_full = buf_aow_full = None
        precomp_knl = None
        atom_ao0 = atom_nao = buf_atom_ao0 = buf_atom_nao = None
        buf_chi = buf_rad_val = buf_rad_dr = buf_coords4 = buf_dm_cart = None
        buf_atom_ao0_cart = buf_atom_nao_cart = None
        buf_atom_coords_h = buf_radial_l_h = buf_atom_radial_offset_h = buf_atom_radial_list_h = None
        c2s = dm_cart32 = dm_tmp = vmat_cart32_host = buf_c2s = buf_vmat_sph = vmat_sph32_host = buf_c2s_scratch = None
        ncart = natoms = None
        if fused in ('tiled', 'coalesced', 'radial_precomp'):
            from .tile_config import get_active_tile_config
            tc = get_active_tile_config()
            NPTILE, NATILE, WGS_VMAT = tc.NPTILE, tc.NATILE, tc.WGS_VMAT
            rho_global = (round_up(ngrids, NPTILE), 1)
            rho_local = (NPTILE, 1)
            k_vmat = None
            if fused == 'radial_precomp':
                ao_eval = self._get_ao_hermite(spline_order=spline_order)
                plan = ao_eval.plan
                ncart = plan.ncart
                natoms = ao_eval.natoms
                atom_ao0_cart, atom_nao_cart = _atom_ao_layout(ao_eval)
                atom_ao0, atom_nao = atom_ao0_cart, atom_nao_cart
                buf_vmat = cl.Buffer(self.ctx, mf.READ_WRITE, ncart * ncart * FBYTES)
                vmat_cart32_host = np.empty((ncart, ncart), dtype=np.float32)
                if int(atom_nao_cart.max()) > tc.MAX_AO_ATOM:
                    raise NotImplementedError(
                        f'fused=radial_precomp requires max atom_nao<={tc.MAX_AO_ATOM}; got {int(atom_nao_cart.max())}')
                nradial = plan.nradial
                coords4 = np.zeros((ngrids, 4), dtype=np.float32)
                coords4[:, :3] = grids.coords
                buf_coords4 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, coords4.nbytes, coords4)
                buf_rad_val = cl.Buffer(self.ctx, mf.READ_ONLY, nradial * ngrids * FBYTES)
                buf_rad_dr = cl.Buffer(self.ctx, mf.READ_ONLY, nradial * ngrids * FBYTES)
                t_rad0 = _time.perf_counter()
                ao_eval.build_radial_on_grid_gpu(buf_coords4, buf_rad_val, buf_rad_dr, ngrids)
                t_rad = _time.perf_counter() - t_rad0
                buf_dm_cart = cl.Buffer(self.ctx, mf.READ_ONLY, ncart * ncart * FBYTES)
                buf_atom_ao0_cart = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, atom_ao0_cart.nbytes, atom_ao0_cart)
                buf_atom_nao_cart = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, atom_nao_cart.nbytes, atom_nao_cart)
                buf_atom_coords_h = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, ao_eval.atom_coords.nbytes, ao_eval.atom_coords)
                buf_radial_l_h = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, ao_eval.radial_l.nbytes, ao_eval.radial_l)
                buf_atom_radial_offset_h = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, ao_eval.atom_radial_offset.nbytes, ao_eval.atom_radial_offset)
                buf_atom_radial_list_h = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, ao_eval.atom_radial_list.nbytes, ao_eval.atom_radial_list)
                k_rho = cl.Kernel(self.prg, 'rho_gga_radial_precomp_pair')
                k_rho.set_args(buf_coords4, buf_atom_coords_h, buf_rad_val, buf_rad_dr,
                               buf_radial_l_h, buf_atom_radial_offset_h, buf_atom_radial_list_h,
                               buf_dm_cart, buf_atom_ao0_cart, buf_atom_nao_cart, buf_rho,
                               np.int32(ncart), np.int32(ngrids), np.int32(natoms))
                c2s = ao_eval.c2s
                buf_c2s = ao_eval.buf_c2s
                dm_cart32 = np.empty((ncart, ncart), dtype=np.float32)
                dm_tmp = np.empty((ncart, nao), dtype=np.float32)
                buf_c2s_scratch = cl.Buffer(self.ctx, mf.READ_WRITE, ncart * nao * FBYTES)
                buf_vmat_sph = cl.Buffer(self.ctx, mf.READ_WRITE, nao * nao * FBYTES)
                vmat_sph32_host = np.empty((nao, nao), dtype=np.float32)
                k_vmat = cl.Kernel(self.prg, 'vmat_gga_radial_precomp_pair')
                k_vmat.set_args(buf_coords4, buf_atom_coords_h, buf_rad_val, buf_rad_dr,
                                buf_radial_l_h, buf_atom_radial_offset_h, buf_atom_radial_list_h,
                                buf_atom_ao0_cart, buf_atom_nao_cart, buf_wv, buf_vmat,
                                np.int32(ncart), np.int32(ngrids), np.int32(natoms))
            else:
                atom_ao0, atom_nao = _atom_ao_layout_mol(mol)
                natoms = mol.natm
                buf_vmat = cl.Buffer(self.ctx, mf.READ_WRITE, nao * nao * FBYTES)
                t_rad = 0.0
                if int(atom_nao.max()) > tc.MAX_AO_ATOM:
                    raise NotImplementedError(
                        f'fused={fused!r} requires max atom_nao<={tc.MAX_AO_ATOM}; got {int(atom_nao.max())}')
                if fused == 'coalesced':
                    if buf_chi_gpu is not None:
                        buf_chi = buf_chi_gpu
                    else:
                        buf_chi = [cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, chi_staging[c].nbytes, chi_staging[c]) for c in range(ncomp)]
                    k_rho = cl.Kernel(self.prg, 'rho_gga_precomp_coalesced_pair')
                    k_rho.set_args(buf_chi[0], buf_chi[1], buf_chi[2], buf_chi[3],
                                   self.bufDm, None, None, buf_rho,
                                   np.int32(nao), np.int32(ngrids), np.int32(natoms))
                else:
                    rho_knl = 'rho_lda_precomp_pair' if self.xctype == 'LDA' else 'rho_gga_precomp_pair'
                    k_rho = cl.Kernel(self.prg, rho_knl)
                    if self.xctype == 'LDA':
                        k_rho.set_args(buf_ao[0], self.bufDm, None, None, buf_rho, np.int32(nao), np.int32(ngrids), np.int32(natoms))
                    else:
                        k_rho.set_args(buf_ao[0], buf_ao[1], buf_ao[2], buf_ao[3], self.bufDm, None, None, buf_rho, np.int32(nao), np.int32(ngrids), np.int32(natoms))
            buf_atom_ao0 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, atom_ao0.nbytes, atom_ao0)
            buf_atom_nao = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, atom_nao.nbytes, atom_nao)
            if fused == 'coalesced':
                k_rho.set_arg(5, buf_atom_ao0)
                k_rho.set_arg(6, buf_atom_nao)
                k_vmat = cl.Kernel(self.prg, 'vmat_gga_precomp_coalesced_pair')
                k_vmat.set_args(buf_chi[0], buf_chi[1], buf_chi[2], buf_chi[3], buf_wv,
                                buf_atom_ao0, buf_atom_nao, buf_vmat,
                                np.int32(nao), np.int32(ngrids), np.int32(natoms))
            elif fused != 'radial_precomp':
                k_rho.set_arg(5, buf_atom_ao0)
                k_rho.set_arg(6, buf_atom_nao)
                vmat_knl = 'vmat_lda_precomp_pair' if self.xctype == 'LDA' else 'vmat_gga_precomp_pair'
                k_vmat = cl.Kernel(self.prg, vmat_knl)
                if self.xctype == 'LDA':
                    k_vmat.set_args(buf_ao[0], buf_wv, buf_atom_ao0, buf_atom_nao, buf_vmat, np.int32(nao), np.int32(ngrids), np.int32(natoms))
                else:
                    k_vmat.set_args(buf_ao[0], buf_ao[1], buf_ao[2], buf_ao[3], buf_wv, buf_atom_ao0, buf_atom_nao, buf_vmat, np.int32(nao), np.int32(ngrids), np.int32(natoms))
            precomp_knl = {
                'k_rho': k_rho, 'k_vmat': k_vmat,
                'rho_global': rho_global,
                'rho_local': rho_local,
                'vmat_global': (natoms, natoms * WGS_VMAT),
                'vmat_local': (1, WGS_VMAT),
                'tiled': fused == 'tiled',
                'coalesced': fused == 'coalesced',
                'radial_precomp': fused == 'radial_precomp',
            }
        elif fused == 'gemm':
            buf_aodm_full = [cl.Buffer(self.ctx, mf.READ_WRITE, ngrids * nao * FBYTES) for _ in range(4)]
            buf_aow_full = cl.Buffer(self.ctx, mf.READ_WRITE, ngrids * nao * FBYTES)
        try:
            xc_eval_mode, xc_gpu_prec = _resolve_xc_eval(xc_eval, gpu_xc, self.xc_code, self.xctype)
        except ValueError:
            if xc_eval == 'gpu':
                xc_eval_mode, xc_gpu_prec = 'cpu', None
            else:
                raise
        buf_exc = buf_vrho = buf_vsigma = buf_rho64 = buf_wv64 = None
        exc_host = rho64_host = wv64_host = None
        xc_extra = {}
        if xc_eval_mode == 'gpu':
            xc_bufs = _alloc_xc_gpu_bufs(self.ctx, ngrids, xc_gpu_prec)
            buf_exc = xc_bufs['buf_exc']
            buf_vrho = xc_bufs['buf_vrho']
            buf_vsigma = xc_bufs['buf_vsigma']
            exc_host = xc_bufs['exc_host']
            buf_rho64 = xc_bufs.get('buf_rho64')
            buf_wv64 = xc_bufs.get('buf_wv64')
            rho64_host = xc_bufs.get('rho64_host')
            wv64_host = xc_bufs.get('wv64_host')
            xc_extra = {k: v for k, v in xc_bufs.items() if k.startswith('buf_') and k not in ('buf_exc', 'buf_vrho', 'buf_vsigma', 'buf_rho64', 'buf_wv64')}
        weight32 = grids.weights.astype(np.float32)
        weight64 = grids.weights
        buf_weight = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, ngrids * FBYTES, weight32)
        buf_weight64 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, ngrids * DBYTES, weight64)
        t_upload = _time.perf_counter() - t_up0
        from pyscf.dft.numint import SWITCH_SIZE
        cutoff = grids.cutoff * 1e2
        nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(grids.cutoff))
        self.pcg = {
            'ao_host': ao_host, 'buf_ao': buf_ao, 'buf_rho': buf_rho, 'buf_wv': buf_wv,
            'buf_vmat': buf_vmat, 'buf_aodm': buf_aodm, 'buf_aow': buf_aow,
            'buf_aodm_full': buf_aodm_full, 'buf_aow_full': buf_aow_full,
            'precomp_knl': precomp_knl,
            'atom_ao0': atom_ao0, 'atom_nao': atom_nao,
            'buf_atom_ao0': buf_atom_ao0, 'buf_atom_nao': buf_atom_nao,
            'screen_index': screen_index, 'ao_loc': mol.ao_loc_nr(), 'nbins': nbins,
            'blksize': blksize, 'weight': weight32, 'weight64': weight64,
            'buf_weight': buf_weight, 'buf_weight64': buf_weight64,
            'ncomp': ncomp, 'gpu_only': gpu_only,
            'allow_sparse': ngrids % ALIGNMENT_UNIT == 0 and nao > SWITCH_SIZE,
            'rho32_host': np.empty(4 * ngrids, dtype=np.float32),
            'wv32_host': np.empty(4 * ngrids, dtype=np.float32),
            'vmat32_host': np.empty((nao, nao), dtype=np.float32),
            'xc_eval_mode': xc_eval_mode, 'xc_gpu_prec': xc_gpu_prec,
            'gpu_xc': xc_gpu_prec,
            'buf_exc': buf_exc, 'buf_vrho': buf_vrho, 'buf_vsigma': buf_vsigma,
            'buf_rho64': buf_rho64, 'buf_wv64': buf_wv64,
            'exc_host': exc_host,
            'rho64_host': rho64_host, 'wv64_host': wv64_host,
            'mem': mem, 'fused': fused,
            'buf_chi': buf_chi, 'buf_rad_val': buf_rad_val, 'buf_rad_dr': buf_rad_dr,
            'buf_coords4': buf_coords4, 'buf_dm_cart': buf_dm_cart,
            'buf_atom_ao0_cart': buf_atom_ao0_cart, 'buf_atom_nao_cart': buf_atom_nao_cart,
            'buf_atom_coords_h': buf_atom_coords_h, 'buf_radial_l_h': buf_radial_l_h,
            'buf_atom_radial_offset_h': buf_atom_radial_offset_h, 'buf_atom_radial_list_h': buf_atom_radial_list_h,
            'c2s': c2s, 'buf_c2s': buf_c2s, 'dm_cart32': dm_cart32, 'dm_tmp': dm_tmp, 'buf_c2s_scratch': buf_c2s_scratch, 'ncart': ncart,
            'vmat_cart32_host': vmat_cart32_host,
            'buf_vmat_sph': buf_vmat_sph, 'vmat_sph32_host': vmat_sph32_host,
            'radial_precomp': fused == 'radial_precomp',
            'coalesced': fused == 'coalesced',
            **xc_extra,
        }
        self.precalc_timing = {
            'eval_ao_cpu': t_eval if not use_hermite_ao else 0.0,
            'eval_ao_hermite_gpu': t_eval if use_hermite_ao else 0.0,
            'upload_gpu': t_upload, 'radial_cpu': 0.0, 'radial_gpu': t_rad,
            'setup_total': _time.perf_counter() - t0,
            'ao_proj': 'hermite_gpu' if use_hermite_ao else 'cpu',
        }
        self.last_timing = {}
        self._pcg_ready = True
        return self

    def _precomputed_block_loop(self):
        pcg = self.pcg
        grids = self.grids
        ngrids = self.ngrids
        blksize = pcg['blksize']
        screen_index = pcg['screen_index']
        ncomp = pcg['ncomp']
        nao = self.nao
        for ip0 in range(0, ngrids, blksize):
            ip1 = min(ip0 + blksize, ngrids)
            nblk = ip1 - ip0
            mask = screen_index[ip0 // BLKSIZE:]
            weight = self.grids.weights[ip0:ip1]
            if pcg['ao_host'] is None:
                ao = None
            elif ncomp == 1:
                ao = pcg['ao_host'][0][ip0:ip1]
            else:
                ao = _AoGGABlock([pcg['ao_host'][c][ip0:ip1] for c in range(ncomp)])
            yield ip0, ip1, nblk, ao, mask, weight

    def nr_rks_precomputed_gto(self, dm, projection='gpu', profile=False):
        '''XC with precomputed grid AOs. setup_precomputed_gto() must run first (upload not timed here).

        projection: 'gpu' (default, float32 GPU), 'cpu_sparse' (float64 CPU sparse reference).
        Uses xc_eval='gpu' (default) for on-device PBE; xc_eval='cpu' for libxc parity debug.
        '''
        if not getattr(self, '_pcg_ready', False):
            raise RuntimeError('Call setup_precomputed_gto() before projection')
        if projection in ('gpu', 'gpu_dense'):
            return self._nr_rks_precomputed_gpu(dm, profile=profile)
        if projection == 'cpu_sparse':
            if self.pcg.get('gpu_only', True):
                raise RuntimeError('cpu_sparse requires setup_precomputed_gto(gpu_only=False)')
            return self._nr_rks_precomputed_cpu_sparse(dm, profile=profile)
        raise ValueError(f'projection={projection!r}; use gpu or cpu_sparse')

    def _xc_pbe_gpu(self, st, ngrids, timing=None):
        '''PBE vxc on GPU: buf_rho -> buf_wv, no rho/wv PCIe.'''
        prec = st['xc_gpu_prec']
        if prec == 'pbe_f32':
            def _pbe_f32():
                _knl(self.prg, 'pbe_xc_f32')(
                    self.queue, (round_up(ngrids, TILE),), (TILE,),
                    st['buf_rho'], st['buf_exc'], st['buf_vrho'], st['buf_vsigma'], np.int32(ngrids))
                _knl(self.prg, 'sanitize_pbe_xc_f32')(
                    self.queue, (round_up(ngrids, TILE),), (TILE,),
                    st['buf_exc'], st['buf_vrho'], st['buf_vsigma'], np.int32(ngrids))
                _knl(self.prg, 'compute_wv_gga_f32')(
                    self.queue, (round_up(ngrids, TILE),), (TILE,),
                    st['buf_weight'], st['buf_vrho'], st['buf_vsigma'], st['buf_rho'],
                    st['buf_wv'], np.int32(ngrids))
            profile_call(self.queue, _pbe_f32, timing, 'gpu_xc_pbe', 'gpu_xc_pbe_cl')
        else:
            def _pbe_f64():
                rho32 = np.empty(4 * ngrids, dtype=np.float32)
                cl.enqueue_copy(self.queue, rho32, st['buf_rho']).wait()
                st['rho64_host'][:] = rho32.astype(np.float64)
                cl.enqueue_copy(self.queue, st['buf_rho64'], st['rho64_host']).wait()
                _knl(self.prg, 'pbe_xc_f64')(
                    self.queue, (round_up(ngrids, TILE),), (TILE,),
                    st['buf_rho64'], st['buf_exc'], st['buf_vrho'], st['buf_vsigma'], np.int32(ngrids))
                _knl(self.prg, 'compute_wv_gga_f64')(
                    self.queue, (round_up(ngrids, TILE),), (TILE,),
                    st['buf_weight64'], st['buf_vrho'], st['buf_vsigma'], st['buf_rho64'],
                    st['buf_wv64'], np.int32(ngrids))
                cl.enqueue_copy(self.queue, st['wv64_host'], st['buf_wv64']).wait()
                st['wv_host'][:4 * ngrids] = st['wv64_host'].astype(np.float32)
                cl.enqueue_copy(self.queue, st['buf_wv'], st['wv_host'][:4 * ngrids]).wait()
            profile_call(self.queue, _pbe_f64, timing, 'gpu_xc_pbe', 'gpu_xc_pbe_cl')
        out = {}
        def _reduce():
            out['nelec'], out['excsum'] = _gpu_nelec_excsum(self.queue, self.prg, st, ngrids)
        profile_call(self.queue, _reduce, timing, 'gpu_xc_reduce', 'gpu_xc_reduce_cl')
        return out['nelec'], out['excsum']

    def _xc_libxc_cpu(self, st, ngrids, xctype, timing=None):
        '''Debug path: D2H rho, CPU libxc, H2D weighted vxc into buf_wv.'''
        ni = self.ni
        t0 = _time.perf_counter()
        cl.enqueue_copy(self.queue, st['rho_host'][:4 * ngrids], st['buf_rho']).wait()
        _timing_record(timing, 'host_rho_d2h', t0)
        t0 = _time.perf_counter()
        weight32 = st['weight32']
        weight64 = st['weight64']
        if xctype == 'LDA':
            rho64 = st['rho_host'][:ngrids].astype(np.float64)
            exc, vxc = ni.eval_xc_eff(self.xc_code, rho64, deriv=1, xctype='LDA', spin=0)[:2]
            den = rho64 * weight64
            nelec = float(den.sum())
            excsum = float(np.dot(den, exc))
            st['wv_host'][:ngrids] = weight32 * vxc.astype(np.float32)
            cl.enqueue_copy(self.queue, st['buf_wv'], st['wv_host'][:ngrids])
        else:
            rho64 = st['rho_host'][:4 * ngrids].reshape(4, ngrids).astype(np.float64)
            evfk = ni.eval_xc_eff(self.xc_code, rho64, deriv=1, xctype='GGA', spin=0)
            exc, vxc = evfk[0], evfk[1]
            den = rho64[0] * weight64
            nelec = float(den.sum())
            excsum = float(np.dot(den, exc))
            wv = st['wv_host'][:4 * ngrids].reshape(4, ngrids)
            wv[:] = weight32[np.newaxis, :] * np.ascontiguousarray(vxc, dtype=np.float32)
            wv[0] *= 0.5
            cl.enqueue_copy(self.queue, st['buf_wv'], st['wv_host'][:4 * ngrids])
        _timing_record(timing, 'host_xc_libxc', t0)
        return nelec, excsum

    def _xc_after_rho(self, st, ngrids, xctype, timing=None):
        if st.get('xc_eval_mode') == 'gpu' and xctype == 'GGA' and st.get('xc_gpu_prec'):
            return self._xc_pbe_gpu(st, ngrids, timing=timing)
        return self._xc_libxc_cpu(st, ngrids, xctype, timing=timing)

    def _nr_rks_precomputed_cpu_sparse(self, dm, profile=False):
        from pyscf.dft.numint import _dot_ao_ao_sparse, _scale_ao_sparse, _sparse_enough
        mol = self.mol
        pcg = self.pcg
        ni = self.ni
        xctype = self.xctype
        nao = self.nao
        ao_loc = pcg['ao_loc']
        nbins = pcg['nbins']
        timing = {} if profile else None
        dm = np.asarray(dm, order='C')
        if dm.dtype != np.float64:
            dm = dm.astype(np.float64)
        t0 = _time.perf_counter()
        make_rho, nset, _ = ni._gen_rho_evaluator(mol, dm, hermi=1, with_lapl=False, grids=self.grids)
        ovlp_cond = mol.get_overlap_cond()
        dm_cond = mol.condense_to_shell(dm, 'absmax')
        pair_mask = np.asarray(np.exp(-ovlp_cond) * dm_cond > ni.cutoff, dtype=np.uint8)
        nelec = np.zeros(nset)
        excsum = np.zeros(nset)
        vmat = np.zeros((nset, nao, nao))
        aow = None
        if timing is not None:
            timing['host_pair_mask'] = _time.perf_counter() - t0
        t0 = _time.perf_counter()
        if xctype == 'LDA':
            for ip0, ip1, nblk, ao, mask, weight in self._precomputed_block_loop():
                smask = mask if pcg['allow_sparse'] and _sparse_enough(mask) else None
                for i in range(nset):
                    rho = make_rho(i, ao, smask, xctype)
                    exc, vxc = ni.eval_xc_eff(self.xc_code, rho, deriv=1, xctype=xctype, spin=0)[:2]
                    den = rho * weight
                    nelec[i] += float(den.sum())
                    excsum[i] += float(np.dot(den, exc))
                    _dot_ao_ao_sparse(ao, ao, weight * vxc, nbins, smask, pair_mask, ao_loc, 1, vmat[i])
        elif xctype == 'GGA':
            from pyscf.dft.numint import _scale_ao, _empty_aligned
            ao3_buf = _empty_aligned(4 * pcg['blksize'] * nao)
            aow_buf = _empty_aligned(nao * pcg['blksize'])
            for ip0, ip1, nblk, ao, mask, weight in self._precomputed_block_loop():
                smask = mask if pcg['allow_sparse'] and _sparse_enough(mask) else None
                ao3 = _pack_ao_block(pcg['ao_host'], ip0, nblk, nao, 4, ao3_buf)
                for i in range(nset):
                    rho = make_rho(i, ao3, smask, xctype)
                    exc, vxc = ni.eval_xc_eff(self.xc_code, rho, deriv=1, xctype=xctype, spin=0)[:2]
                    den = rho[0] * weight
                    nelec[i] += float(den.sum())
                    excsum[i] += float(np.dot(den, exc))
                    wv = weight * vxc
                    wv[0] *= 0.5
                    aow_out = aow_buf[:nao * nblk].reshape(nao, nblk)
                    if smask is not None:
                        aow = _scale_ao_sparse(ao3, wv[:4], smask, ao_loc, out=aow_out)
                    else:
                        aow = _scale_ao(ao3, wv[:4], out=aow_out)
                    _dot_ao_ao_sparse(ao3[0], aow, None, nbins, smask, pair_mask, ao_loc, hermi=0, out=vmat[i])
            vmat = vmat + np.transpose(vmat, (0, 2, 1))
        else:
            raise NotImplementedError(f'xctype={xctype}')
        if timing is not None:
            timing['host_cpu_projection'] = _time.perf_counter() - t0
            _finalize_gpu_timing(timing)
            self.last_timing = timing
        return float(nelec[0]), float(excsum[0]), vmat[0]

    def _precomp_rho_fused(self, pcg, xctype, nao, ngrids, timing=None):
        pk = pcg.get('precomp_knl')
        t0 = _time.perf_counter()
        if pk and (pk.get('tiled') or pk.get('coalesced') or pk.get('radial_precomp')):
            profile_kernel(self.queue, pk['k_rho'], pk['rho_global'], pk['rho_local'], timing, 'gpu_rho', 'gpu_rho_cl')
            return
        elif pcg.get('buf_aodm_full') is not None:
            aodm = pcg['buf_aodm_full']
            if xctype == 'LDA':
                matmul_gpu_buf(pcg['buf_ao'][0], self.bufDm, aodm[0], ngrids, nao, nao)
                _knl(self.prg, 'contract_rho_lda_from_aodm')(
                    self.queue, (round_up(ngrids, TILE),), (TILE,),
                    pcg['buf_ao'][0], aodm[0], pcg['buf_rho'],
                    np.int32(nao), np.int32(ngrids), np.int32(0), np.int32(0))
            else:
                for c in range(4):
                    matmul_gpu_buf(pcg['buf_ao'][c], self.bufDm, aodm[c], ngrids, nao, nao)
                _knl(self.prg, 'contract_rho_gga_from_aodm')(
                    self.queue, (round_up(ngrids, TILE),), (TILE,),
                    pcg['buf_ao'][0], pcg['buf_ao'][1], pcg['buf_ao'][2], pcg['buf_ao'][3],
                    aodm[0], aodm[1], aodm[2], aodm[3], pcg['buf_rho'],
                    np.int32(nao), np.int32(ngrids), np.int32(0), np.int32(ngrids), np.int32(0))
        else:
            raise RuntimeError('precomputed fused path: missing precomp_knl or buf_aodm_full')
        _gpu_sync(self.queue)
        _timing_record(timing, 'gpu_rho', t0)

    def _precomp_vmat_fused(self, pcg, xctype, nao, ngrids, timing=None):
        pk = pcg.get('precomp_knl')
        t0 = _time.perf_counter()
        if pk and (pk.get('tiled') or pk.get('coalesced') or pk.get('radial_precomp')):
            profile_kernel(self.queue, pk['k_vmat'], pk['vmat_global'], pk['vmat_local'], timing, 'gpu_vmat', 'gpu_vmat_cl')
            return
        elif pcg.get('buf_aow_full') is not None:
            aow = pcg['buf_aow_full']
            if xctype == 'LDA':
                _knl(self.prg, 'scale_aow_lda')(
                    self.queue, (round_up(ngrids, TILE), round_up(nao, TILE)), (TILE, TILE),
                    pcg['buf_ao'][0], pcg['buf_wv'], aow,
                    np.int32(nao), np.int32(ngrids), np.int32(0), np.int32(0))
                matmul_gpu_buf_accum(aow, pcg['buf_ao'][0], pcg['buf_vmat'], nao, nao, ngrids, transpose_A=True)
            else:
                _knl(self.prg, 'scale_aow_gga_split')(
                    self.queue, (round_up(ngrids, TILE), round_up(nao, TILE)), (TILE, TILE),
                    pcg['buf_ao'][0], pcg['buf_ao'][1], pcg['buf_ao'][2], pcg['buf_ao'][3],
                    pcg['buf_wv'], aow,
                    np.int32(nao), np.int32(ngrids), np.int32(0), np.int32(ngrids), np.int32(0))
                matmul_gpu_buf_accum(aow, pcg['buf_ao'][0], pcg['buf_vmat'], nao, nao, ngrids, transpose_A=True)
        else:
            raise RuntimeError('precomputed fused path: missing precomp_knl or buf_aow_full')
        _gpu_sync(self.queue)
        _timing_record(timing, 'gpu_vmat', t0)

    def _precomp_rho_blocked(self, pcg, xctype, nao, ngrids, timing=None):
        n_blocks = 0
        t0 = _time.perf_counter()
        for ip0, ip1, nblk, _ao, _mask, _weight in self._precomputed_block_loop():
            n_blocks += 1
            if xctype == 'LDA':
                matmul_gpu_buf(pcg['buf_ao'][0], self.bufDm, pcg['buf_aodm'][0], nblk, nao, nao, row0=ip0)
                _knl(self.prg, 'contract_rho_lda_from_aodm')(
                    self.queue, (round_up(nblk, TILE),), (TILE,),
                    pcg['buf_ao'][0], pcg['buf_aodm'][0], pcg['buf_rho'],
                    np.int32(nao), np.int32(nblk), np.int32(ip0), np.int32(ip0))
            else:
                for c in range(4):
                    matmul_gpu_buf(pcg['buf_ao'][c], self.bufDm, pcg['buf_aodm'][c], nblk, nao, nao, row0=ip0)
                _knl(self.prg, 'contract_rho_gga_from_aodm')(
                    self.queue, (round_up(nblk, TILE),), (TILE,),
                    pcg['buf_ao'][0], pcg['buf_ao'][1], pcg['buf_ao'][2], pcg['buf_ao'][3],
                    pcg['buf_aodm'][0], pcg['buf_aodm'][1], pcg['buf_aodm'][2], pcg['buf_aodm'][3],
                    pcg['buf_rho'], np.int32(nao), np.int32(nblk), np.int32(ip0), np.int32(ngrids), np.int32(ip0))
        _gpu_sync(self.queue)
        _timing_record(timing, 'gpu_rho', t0)
        return n_blocks

    def _precomp_vmat_blocked(self, pcg, xctype, nao, ngrids, timing=None):
        t0 = _time.perf_counter()
        for ip0, ip1, nblk, _ao, _mask, _weight in self._precomputed_block_loop():
            if xctype == 'LDA':
                _knl(self.prg, 'scale_aow_lda')(
                    self.queue, (round_up(nblk, TILE), round_up(nao, TILE)), (TILE, TILE),
                    pcg['buf_ao'][0], pcg['buf_wv'], pcg['buf_aow'],
                    np.int32(nao), np.int32(nblk), np.int32(ip0), np.int32(ip0))
                matmul_gpu_buf_accum(pcg['buf_aow'], pcg['buf_ao'][0], pcg['buf_vmat'], nao, nao, nblk, transpose_A=True, b_row0=ip0)
            else:
                _knl(self.prg, 'scale_aow_gga_split')(
                    self.queue, (round_up(nblk, TILE), round_up(nao, TILE)), (TILE, TILE),
                    pcg['buf_ao'][0], pcg['buf_ao'][1], pcg['buf_ao'][2], pcg['buf_ao'][3],
                    pcg['buf_wv'], pcg['buf_aow'],
                    np.int32(nao), np.int32(nblk), np.int32(ip0), np.int32(ngrids), np.int32(ip0))
                matmul_gpu_buf_accum(pcg['buf_aow'], pcg['buf_ao'][0], pcg['buf_vmat'], nao, nao, nblk, transpose_A=True, b_row0=ip0)
        _gpu_sync(self.queue)
        _timing_record(timing, 'gpu_vmat', t0)

    def _nr_rks_precomputed_gpu(self, dm, profile=False):
        '''Float32 GPU projection using pre-uploaded buf_ao (no per-iteration AO copy/eval).'''
        pcg = self.pcg
        ni = self.ni
        xctype = self.xctype
        nao = self.nao
        ngrids = self.ngrids
        weight32 = pcg['weight']
        weight64 = pcg['weight64']
        timing = {} if profile else None
        dm32 = np.ascontiguousarray(dm, dtype=np.float32)
        t0 = _time.perf_counter()
        cl.enqueue_copy(self.queue, self.bufDm, dm32)
        if pcg.get('radial_precomp'):
            _gpu_dm_to_cart(pcg['buf_c2s'], self.bufDm, pcg['buf_c2s_scratch'], pcg['buf_dm_cart'], nao, pcg['ncart'])
        ncart_v = pcg.get('ncart')
        if pcg.get('radial_precomp'):
            zero_buffer_gpu(pcg['buf_vmat'], ncart_v * ncart_v)
        else:
            zero_buffer_gpu(pcg['buf_vmat'], nao * nao)
        _timing_record(timing, 'host_h2d_dm', t0)
        fused = pcg.get('fused', 'tiled')
        if fused:
            self._precomp_rho_fused(pcg, xctype, nao, ngrids, timing=timing)
            n_blocks = 0
        else:
            n_blocks = self._precomp_rho_blocked(pcg, xctype, nao, ngrids, timing=timing)
        pcg_xc = {**pcg, 'rho_host': pcg['rho32_host'], 'wv_host': pcg['wv32_host'], 'weight32': weight32, 'weight64': weight64}
        nelec, excsum = self._xc_after_rho(pcg_xc, ngrids, xctype, timing=timing)
        if fused:
            self._precomp_vmat_fused(pcg, xctype, nao, ngrids, timing=timing)
        else:
            self._precomp_vmat_blocked(pcg, xctype, nao, ngrids, timing=timing)
        t0 = _time.perf_counter()
        if pcg.get('radial_precomp'):
            _gpu_vmat_cart_to_sph(pcg['buf_c2s'], pcg['buf_vmat'], pcg['buf_c2s_scratch'], pcg['buf_vmat_sph'], nao, pcg['ncart'])
            cl.enqueue_copy(self.queue, pcg['vmat_sph32_host'], pcg['buf_vmat_sph']).wait()
            vmat = pcg['vmat_sph32_host'].astype(np.float64)
        else:
            cl.enqueue_copy(self.queue, pcg['vmat32_host'], pcg['buf_vmat']).wait()
            vmat = pcg['vmat32_host'].astype(np.float64)
        if xctype == 'GGA':
            vmat = vmat + vmat.T
        _timing_record(timing, 'host_vmat_d2h', t0)
        if timing is not None:
            timing['n_blocks'] = n_blocks
            timing['fused'] = {'gemm': 1.0, 'tiled': 2.0, 'coalesced': 3.0, 'radial_precomp': 4.0, False: 0.0}.get(fused, 1.0)
            _finalize_gpu_timing(timing)
            self.last_timing = timing
        return nelec, excsum, vmat

    def nr_rks_precomputed_rho_only(self, dm, profile=False):
        '''GPU rho projection only (parity vs CPU make_rho). Requires setup_precomputed_gto().'''
        if not getattr(self, '_pcg_ready', False):
            raise RuntimeError('Call setup_precomputed_gto() before projection')
        pcg = self.pcg
        ngrids = self.ngrids
        timing = {} if profile else None
        dm32 = np.ascontiguousarray(dm, dtype=np.float32)
        t0 = _time.perf_counter()
        cl.enqueue_copy(self.queue, self.bufDm, dm32)
        if pcg.get('radial_precomp'):
            _gpu_dm_to_cart(pcg['buf_c2s'], self.bufDm, pcg['buf_c2s_scratch'], pcg['buf_dm_cart'], self.nao, pcg['ncart'])
        _timing_record(timing, 'host_h2d_dm', t0)
        self._precomp_rho_fused(pcg, self.xctype, self.nao, ngrids, timing=timing)
        rho32 = pcg['rho32_host']
        t0 = _time.perf_counter()
        cl.enqueue_copy(self.queue, rho32[:4 * ngrids], pcg['buf_rho']).wait()
        _timing_record(timing, 'host_rho_d2h', t0)
        if timing is not None:
            _finalize_gpu_timing(timing)
            self.last_timing = timing
        ncomp = 4 if self.xctype == 'GGA' else 1
        return rho32[:ncomp * ngrids].reshape(ncomp, ngrids).astype(np.float64)

    def nr_rks_precomputed_vmat_only(self, wv, profile=False):
        '''GPU vmat projection only. wv: (4,ngrids) weighted vxc (GGA: wv[0] halved).'''
        if not getattr(self, '_pcg_ready', False):
            raise RuntimeError('Call setup_precomputed_gto() before projection')
        pcg = self.pcg
        nao = self.nao
        ngrids = self.ngrids
        timing = {} if profile else None
        wv = np.asarray(wv, dtype=np.float32)
        if wv.shape != (4, ngrids):
            raise ValueError(f'wv shape {wv.shape} != (4, {ngrids})')
        t0 = _time.perf_counter()
        pcg['wv32_host'][:4 * ngrids] = np.ascontiguousarray(wv.reshape(-1))
        cl.enqueue_copy(self.queue, pcg['buf_wv'], pcg['wv32_host'][:4 * ngrids])
        ncart_v = pcg.get('ncart')
        if pcg.get('radial_precomp'):
            zero_buffer_gpu(pcg['buf_vmat'], ncart_v * ncart_v)
        else:
            zero_buffer_gpu(pcg['buf_vmat'], nao * nao)
        _timing_record(timing, 'host_h2d_wv', t0)
        fused = pcg.get('fused', 'tiled')
        if fused:
            self._precomp_vmat_fused(pcg, self.xctype, nao, ngrids, timing=timing)
        else:
            self._precomp_vmat_blocked(pcg, self.xctype, nao, ngrids, timing=timing)
        t0 = _time.perf_counter()
        if pcg.get('radial_precomp'):
            _gpu_vmat_cart_to_sph(pcg['buf_c2s'], pcg['buf_vmat'], pcg['buf_c2s_scratch'], pcg['buf_vmat_sph'], nao, pcg['ncart'])
            cl.enqueue_copy(self.queue, pcg['vmat_sph32_host'], pcg['buf_vmat_sph']).wait()
            vmat = pcg['vmat_sph32_host'].astype(np.float64)
        else:
            cl.enqueue_copy(self.queue, pcg['vmat32_host'], pcg['buf_vmat']).wait()
            vmat = pcg['vmat32_host'].astype(np.float64)
        if self.xctype == 'GGA':
            vmat = vmat + vmat.T
        _timing_record(timing, 'host_vmat_d2h', t0)
        if timing is not None:
            _finalize_gpu_timing(timing)
            self.last_timing = timing
        return vmat

    def _nr_rks_precomputed_gpu_dense(self, dm, profile=False):
        return self._nr_rks_precomputed_gpu(dm, profile=profile)

    def setup_onthefly(self, r0_ang=0.002, du=0.02, rmax_ang=None, xc_eval='gpu', gpu_xc='auto', spline_order='cubic', vmat_mode='otf', vmat_grid_splits=1, rho_mode='hermite', screen_eps=1e-7):
        '''One-time prep before SCF: OpenCL compile, Hermite tables, GPU buffers, kernel args.

        xc_eval: 'gpu' (default, PBE on device) | 'cpu' (libxc debug with rho D2H).
        spline_order: 'cubic' | 'quintic' (analytic GTO tangents; quintic uses 2× coarser du).
        rho_mode: 'hermite' (OTF) | 'radial_screened' (NEW screened radial rho kernels).
        vmat_mode: 'otf' | 'radial_precomp' | 'radial_screened' (NEW screened radial vmat).
        vmat_grid_splits: split radial-precomp vmat over grid chunks; 1 keeps the original kernel.
        screen_eps: radial-tail cutoff for grid_screen (radial_screened only).
        '''
        from . import init_device
        from .grid_screen import compute_atom_rcut, build_gtile_atom_lists
        init_device(quiet=getattr(self, '_otf_ready', False))
        self.ctx = get_ctx()
        self.queue = get_queue()
        self.prg = get_prg()
        ao_eval = self._get_ao_hermite(r0_ang=r0_ang, du=du, rmax_ang=rmax_ang, spline_order=spline_order)
        ncart = ao_eval.plan.ncart
        natoms = ao_eval.natoms
        ngrids = self.ngrids
        if ncart > 1024:
            raise NotImplementedError(f'On-the-fly kernels support ncart<=1024; got ncart={ncart}')
        mf = cl.mem_flags
        fbytes = np.dtype(np.float32).itemsize
        dbytes = np.dtype(np.float64).itemsize
        c2s = ao_eval.c2s
        atom_ao0, atom_nao = _atom_ao_layout(ao_eval)
        tc = get_active_tile_config()
        NPTILE, NATILE, WGS_VMAT = tc.NPTILE, tc.NATILE, tc.WGS_VMAT
        rho_mode = str(rho_mode).lower()
        if rho_mode not in ('hermite', 'radial_screened'):
            raise ValueError(f"rho_mode must be 'hermite' or 'radial_screened'; got {rho_mode!r}")
        if rho_mode == 'radial_screened' and self.xctype != 'GGA':
            raise NotImplementedError('rho_mode=radial_screened requires GGA')
        use_pair = (NATILE == 1)
        n_iTiles = round_up(natoms, NATILE) // NATILE
        if rho_mode == 'radial_screened':
            rho_knl = 'rho_gga_radial_screened'
            rho_global = (round_up(ngrids, NPTILE),)
            rho_local = (NPTILE,)
            vmat_global_otf = (natoms, natoms * WGS_VMAT)
            vmat_local_otf = (1, WGS_VMAT)
            vmat_knl_otf = 'vmat_gga_pair'
        elif use_pair:
            rho_knl = 'rho_lda_pair' if self.xctype == 'LDA' else 'rho_gga_pair'
            rho_global = (round_up(ngrids, NPTILE), 1)
            rho_local = (NPTILE, 1)
            vmat_global_otf = (natoms, natoms * WGS_VMAT)
            vmat_local_otf = (1, WGS_VMAT)
            vmat_knl_otf = 'vmat_lda_pair' if self.xctype == 'LDA' else 'vmat_gga_pair'
        else:
            rho_knl = 'rho_lda_tiled' if self.xctype == 'LDA' else 'rho_gga_tiled'
            rho_global = (round_up(ngrids, NPTILE), NATILE)
            rho_local = (NPTILE, NATILE)
            n_jTiles = round_up(natoms, NATILE) // NATILE
            vmat_global_otf = (n_iTiles, n_jTiles * WGS_VMAT)
            vmat_local_otf = (1, WGS_VMAT)
            vmat_knl_otf = 'vmat_lda_tiled' if self.xctype == 'LDA' else 'vmat_gga_tiled'
        vmat_mode = str(vmat_mode).lower()
        if vmat_mode not in ('otf', 'radial_precomp', 'radial_screened'):
            raise ValueError(f"vmat_mode must be 'otf', 'radial_precomp', or 'radial_screened'; got {vmat_mode!r}")
        if vmat_mode in ('radial_precomp', 'radial_screened') and self.xctype != 'GGA':
            raise NotImplementedError(f'vmat_mode={vmat_mode} requires GGA')
        coords4 = np.zeros((ngrids, 4), dtype=np.float32)
        coords4[:, :3] = self.grids.coords
        buf_coords4 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, coords4.nbytes, coords4)
        vmat_grid_splits = int(vmat_grid_splits)
        if vmat_grid_splits < 1:
            raise ValueError(f'vmat_grid_splits must be >= 1; got {vmat_grid_splits}')
        if vmat_grid_splits > 1 and vmat_mode not in ('radial_precomp', 'radial_screened'):
            raise NotImplementedError('vmat_grid_splits>1 is implemented only for vmat_mode in (radial_precomp, radial_screened)')
        need_radial = (rho_mode == 'radial_screened') or (vmat_mode in ('radial_precomp', 'radial_screened'))
        need_screen = (rho_mode == 'radial_screened') or (vmat_mode == 'radial_screened')
        buf_rad_val = buf_rad_dr = buf_vmat_partial = None
        buf_atom_coords_h = buf_radial_l_h = buf_atom_radial_offset_h = buf_atom_radial_list_h = None
        buf_gtile_atom_off = buf_gtile_atom_list = None
        buf_pair_gtile_off = buf_pair_gtile_list = None
        screen_stats = None
        screen_cap = None
        t_rad = 0.0
        ibytes = np.dtype(np.int32).itemsize
        n_gtile = round_up(ngrids, NPTILE) // NPTILE
        n_pairs = natoms * (natoms + 1) // 2
        # Screen CSR: prealloc worst-case capacity (geometry refill = enqueue_copy only).
        if need_screen:
            cap_atom_list = max(n_gtile * natoms, 1)
            cap_pair_list = max(n_pairs * n_gtile, 1)
            screen_cap = {
                'n_gtile': n_gtile, 'n_pairs': n_pairs,
                'cap_atom_list': cap_atom_list, 'cap_pair_list': cap_pair_list,
            }
            buf_gtile_atom_off = cl.Buffer(self.ctx, mf.READ_ONLY, (n_gtile + 1) * ibytes)
            buf_gtile_atom_list = cl.Buffer(self.ctx, mf.READ_ONLY, cap_atom_list * ibytes)
            buf_pair_gtile_off = cl.Buffer(self.ctx, mf.READ_ONLY, (n_pairs + 1) * ibytes)
            buf_pair_gtile_list = cl.Buffer(self.ctx, mf.READ_ONLY, cap_pair_list * ibytes)
        if need_radial:
            nradial = ao_eval.plan.nradial
            buf_rad_val = cl.Buffer(self.ctx, mf.READ_ONLY, nradial * ngrids * fbytes)
            buf_rad_dr = cl.Buffer(self.ctx, mf.READ_ONLY, nradial * ngrids * fbytes)
            buf_atom_coords_h = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, ao_eval.atom_coords.nbytes, ao_eval.atom_coords)
            buf_radial_l_h = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, ao_eval.radial_l.nbytes, ao_eval.radial_l)
            buf_atom_radial_offset_h = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, ao_eval.atom_radial_offset.nbytes, ao_eval.atom_radial_offset)
            buf_atom_radial_list_h = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, ao_eval.atom_radial_list.nbytes, ao_eval.atom_radial_list)
            t_rad0 = _time.perf_counter()
            ao_eval.build_radial_on_grid_gpu(buf_coords4, buf_rad_val, buf_rad_dr, ngrids)
            t_rad = _time.perf_counter() - t_rad0
        if need_screen:
            atom_rcut = compute_atom_rcut(ao_eval.plan, eps=screen_eps, margin_bohr=0.5)
            screen = build_gtile_atom_lists(
                self.grids.coords, ao_eval.atom_coords[:, :3], atom_rcut, NPTILE, natoms, pair_screen=True)
            screen_stats = screen['stats']
            gao = np.ascontiguousarray(screen['gtile_atom_off'], dtype=np.int32)
            gal = np.ascontiguousarray(screen['gtile_atom_list'], dtype=np.int32)
            pgo = np.ascontiguousarray(screen['pair_gtile_off'], dtype=np.int32)
            pgl = np.ascontiguousarray(screen['pair_gtile_list'], dtype=np.int32)
            if gal.size > screen_cap['cap_atom_list'] or pgl.size > screen_cap['cap_pair_list']:
                raise RuntimeError(
                    f'screen CSR exceeds prealloc cap: atom_list {gal.size}/{screen_cap["cap_atom_list"]} '
                    f'pair_list {pgl.size}/{screen_cap["cap_pair_list"]}')
            cl.enqueue_copy(self.queue, buf_gtile_atom_off, gao)
            if gal.size:
                cl.enqueue_copy(self.queue, buf_gtile_atom_list, gal)
            cl.enqueue_copy(self.queue, buf_pair_gtile_off, pgo)
            if pgl.size:
                cl.enqueue_copy(self.queue, buf_pair_gtile_list, pgl)
            self.queue.finish()
            screen_cap['n_atom_list'] = int(gal.size)
            screen_cap['n_pair_list'] = int(pgl.size)
        if vmat_mode == 'radial_screened':
            if vmat_grid_splits > 1:
                vmat_knl = 'vmat_gga_radial_screened_pair_splitk'
                vmat_global = (natoms, natoms * WGS_VMAT, vmat_grid_splits)
                vmat_local = (1, WGS_VMAT, 1)
            else:
                vmat_knl = 'vmat_gga_radial_screened_pair'
                vmat_global = (natoms, natoms * WGS_VMAT)
                vmat_local = (1, WGS_VMAT)
        elif vmat_mode == 'radial_precomp':
            vmat_knl = 'vmat_gga_radial_precomp_pair_splitk' if vmat_grid_splits > 1 else 'vmat_gga_radial_precomp_pair'
            vmat_global = (natoms, natoms * WGS_VMAT, vmat_grid_splits) if vmat_grid_splits > 1 else (natoms, natoms * WGS_VMAT)
            vmat_local = (1, WGS_VMAT, 1) if vmat_grid_splits > 1 else (1, WGS_VMAT)
        else:
            vmat_knl = vmat_knl_otf
            vmat_global = vmat_global_otf
            vmat_local = vmat_local_otf
        hermite_bufs = [
            None, ao_eval.buf_atom_coords,
            ao_eval.buf_rad_node,
            ao_eval.buf_radial_l, ao_eval.buf_radial_cart0,
            ao_eval.buf_atom_radial_offset, ao_eval.buf_atom_radial_list,
        ]
        hermite_params = [
            np.float32(ao_eval.plan.r0), np.float32(ao_eval.plan.du),
            np.int32(ao_eval.plan.nrad), np.int32(ncart),
            np.int32(ngrids), np.int32(natoms), np.int32(ao_eval.plan.spline_order_code),
        ]
        buf_dm_cart = cl.Buffer(self.ctx, mf.READ_ONLY, ncart * ncart * fbytes)
        buf_atom_ao0 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, atom_ao0.nbytes, atom_ao0)
        buf_atom_nao = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, atom_nao.nbytes, atom_nao)
        buf_rho = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * ngrids * fbytes)
        buf_wv = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * ngrids * fbytes)
        buf_vmat = cl.Buffer(self.ctx, mf.READ_WRITE, ncart * ncart * fbytes)
        if vmat_mode in ('radial_precomp', 'radial_screened') and vmat_grid_splits > 1:
            buf_vmat_partial = cl.Buffer(self.ctx, mf.READ_WRITE, vmat_grid_splits * ncart * ncart * fbytes)
        buf_dm_sph = cl.Buffer(self.ctx, mf.READ_ONLY, self.nao * self.nao * fbytes)
        buf_vmat_sph = cl.Buffer(self.ctx, mf.READ_WRITE, self.nao * self.nao * fbytes)
        buf_c2s_scratch = cl.Buffer(self.ctx, mf.READ_WRITE, ncart * self.nao * fbytes)
        weight32 = self.grids.weights.astype(np.float32)
        buf_weight = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, ngrids * fbytes, weight32)
        buf_weight64 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, ngrids * dbytes, self.grids.weights)
        try:
            xc_eval_mode, xc_gpu_prec = _resolve_xc_eval(xc_eval, gpu_xc, self.xc_code, self.xctype)
        except ValueError:
            if xc_eval == 'gpu':
                xc_eval_mode, xc_gpu_prec = 'cpu', None
            else:
                raise
        xc_gpu_bufs = _alloc_xc_gpu_bufs(self.ctx, ngrids, xc_gpu_prec) if xc_eval_mode == 'gpu' else {}
        k_rho = cl.Kernel(self.prg, rho_knl)
        if rho_mode == 'radial_screened':
            k_rho.set_args(buf_coords4, buf_atom_coords_h, buf_rad_val, buf_rad_dr,
                           buf_radial_l_h, buf_atom_radial_offset_h, buf_atom_radial_list_h,
                           buf_dm_cart, buf_atom_ao0, buf_atom_nao,
                           buf_gtile_atom_off, buf_gtile_atom_list, buf_rho,
                           np.int32(ncart), np.int32(ngrids), np.int32(natoms))
        else:
            k_rho.set_args(buf_coords4, *hermite_bufs[1:], buf_dm_cart, buf_atom_ao0, buf_atom_nao, buf_rho, *hermite_params)
        k_vmat = cl.Kernel(self.prg, vmat_knl)
        k_vmat_reduce = None
        if vmat_mode == 'radial_screened':
            if vmat_grid_splits > 1:
                k_vmat.set_args(buf_coords4, buf_atom_coords_h, buf_rad_val, buf_rad_dr,
                                buf_radial_l_h, buf_atom_radial_offset_h, buf_atom_radial_list_h,
                                buf_atom_ao0, buf_atom_nao, buf_wv,
                                buf_pair_gtile_off, buf_pair_gtile_list, buf_vmat_partial,
                                np.int32(ncart), np.int32(ngrids), np.int32(natoms), np.int32(vmat_grid_splits))
                k_vmat_reduce = cl.Kernel(self.prg, 'reduce_split_vmat')
                k_vmat_reduce.set_args(buf_vmat_partial, buf_vmat, np.int32(ncart), np.int32(vmat_grid_splits))
            else:
                k_vmat.set_args(buf_coords4, buf_atom_coords_h, buf_rad_val, buf_rad_dr,
                                buf_radial_l_h, buf_atom_radial_offset_h, buf_atom_radial_list_h,
                                buf_atom_ao0, buf_atom_nao, buf_wv,
                                buf_pair_gtile_off, buf_pair_gtile_list, buf_vmat,
                                np.int32(ncart), np.int32(ngrids), np.int32(natoms))
        elif vmat_mode == 'radial_precomp':
            if vmat_grid_splits > 1:
                k_vmat.set_args(buf_coords4, buf_atom_coords_h, buf_rad_val, buf_rad_dr,
                                buf_radial_l_h, buf_atom_radial_offset_h, buf_atom_radial_list_h,
                                buf_atom_ao0, buf_atom_nao, buf_wv, buf_vmat_partial,
                                np.int32(ncart), np.int32(ngrids), np.int32(natoms), np.int32(vmat_grid_splits))
                k_vmat_reduce = cl.Kernel(self.prg, 'reduce_split_vmat')
                k_vmat_reduce.set_args(buf_vmat_partial, buf_vmat, np.int32(ncart), np.int32(vmat_grid_splits))
            else:
                k_vmat.set_args(buf_coords4, buf_atom_coords_h, buf_rad_val, buf_rad_dr,
                                buf_radial_l_h, buf_atom_radial_offset_h, buf_atom_radial_list_h,
                                buf_atom_ao0, buf_atom_nao, buf_wv, buf_vmat,
                                np.int32(ncart), np.int32(ngrids), np.int32(natoms))
        else:
            k_vmat.set_scalar_arg_dtypes([None] * 11 + [np.float32, np.float32, np.int32, np.int32, np.int32, np.int32, np.int32])
            k_vmat.set_args(buf_coords4, *hermite_bufs[1:], buf_atom_ao0, buf_atom_nao, buf_wv, buf_vmat, *hermite_params)
        self.otf = {
            'ao_eval': ao_eval, 'ncart': ncart, 'natoms': natoms,
            'rho_mode': rho_mode, 'vmat_mode': vmat_mode, 'vmat_grid_splits': vmat_grid_splits,
            'setup_radial_gpu': t_rad, 'screen_stats': screen_stats, 'screen_cap': screen_cap,
            'buf_rad_val': buf_rad_val, 'buf_rad_dr': buf_rad_dr,
            'buf_atom_coords_h': buf_atom_coords_h, 'buf_radial_l_h': buf_radial_l_h,
            'buf_atom_radial_offset_h': buf_atom_radial_offset_h, 'buf_atom_radial_list_h': buf_atom_radial_list_h,
            'buf_gtile_atom_off': buf_gtile_atom_off, 'buf_gtile_atom_list': buf_gtile_atom_list,
            'buf_pair_gtile_off': buf_pair_gtile_off, 'buf_pair_gtile_list': buf_pair_gtile_list,
            'c2s': c2s, 'weight': self.grids.weights, 'weight32': weight32,
            'buf_weight': buf_weight, 'buf_weight64': buf_weight64,
            'buf_coords4': buf_coords4, 'buf_dm_cart': buf_dm_cart,
            'buf_dm_sph': buf_dm_sph, 'buf_vmat_sph': buf_vmat_sph, 'buf_c2s_scratch': buf_c2s_scratch,
            'buf_atom_ao0': buf_atom_ao0, 'buf_atom_nao': buf_atom_nao,
            'buf_rho': buf_rho, 'buf_wv': buf_wv, 'buf_vmat': buf_vmat, 'buf_vmat_partial': buf_vmat_partial,
            'xc_eval_mode': xc_eval_mode, 'xc_gpu_prec': xc_gpu_prec,
            'k_rho': k_rho, 'k_vmat': k_vmat, 'k_vmat_reduce': k_vmat_reduce,
            'rho_global': rho_global, 'rho_local': rho_local,
            'vmat_global': vmat_global, 'vmat_local': vmat_local,
            'use_pair_kernels': use_pair,
            'rho32_full': np.empty(4 * ngrids, dtype=np.float32),
            'rho_full': np.empty((4, ngrids), dtype=np.float64),
            'wv_full': np.empty(4 * ngrids, dtype=np.float32),
            'vmat_cart32': np.empty((ncart, ncart), dtype=np.float32),
            'vmat_sph32': np.empty((self.nao, self.nao), dtype=np.float32),
            'dm_cart32': np.empty((ncart, ncart), dtype=np.float32),
            'dm_tmp': np.empty((ncart, self.nao), dtype=np.float32),
            **xc_gpu_bufs,
        }
        self.last_timing = {}
        self._otf_ready = True
        return self

    def nr_rks_hermite_onthefly(self, dm, profile=False):
        if not getattr(self, '_otf_ready', False):
            raise RuntimeError('Call setup_onthefly() or setup_xc_grid_gpu() before SCF')
        ot = self.otf
        ncart = ot['ncart']
        ngrids = self.ngrids
        nao = self.nao
        buf_c2s = ot['ao_eval'].buf_c2s
        timing = {} if profile else None
        dm32 = np.ascontiguousarray(dm, dtype=np.float32)

        t0 = _time.perf_counter()
        cl.enqueue_copy(self.queue, ot['buf_dm_sph'], dm32)
        if timing is not None:
            self.queue.finish()
        _timing_record(timing, 'host_h2d_dm', t0)

        profile_call(self.queue, lambda: _gpu_dm_to_cart(buf_c2s, ot['buf_dm_sph'], ot['buf_c2s_scratch'], ot['buf_dm_cart'], nao, ncart), timing, 'gpu_dm_cart', 'gpu_dm_cart_cl')

        profile_kernel(self.queue, ot['k_rho'], ot['rho_global'], ot['rho_local'], timing, 'gpu_rho', 'gpu_rho_cl')

        ot_xc = {**ot, 'rho_host': ot['rho32_full'], 'wv_host': ot['wv_full'], 'weight64': ot['weight']}
        nelec, excsum = self._xc_after_rho(ot_xc, ngrids, self.xctype, timing=timing)

        if ot.get('k_vmat_reduce') is not None:
            profile_kernel(self.queue, ot['k_vmat'], ot['vmat_global'], ot['vmat_local'], timing, 'gpu_vmat_split', 'gpu_vmat_split_cl')
            profile_kernel(self.queue, ot['k_vmat_reduce'], (round_up(ncart * ncart, TILE),), (TILE,), timing, 'gpu_vmat_reduce', 'gpu_vmat_reduce_cl')
            if timing is not None:
                timing['gpu_vmat'] = timing.get('gpu_vmat_split', 0.0) + timing.get('gpu_vmat_reduce', 0.0)
                timing['gpu_vmat_cl'] = timing.get('gpu_vmat_split_cl', 0.0) + timing.get('gpu_vmat_reduce_cl', 0.0)
        else:
            if timing is not None:
                zero_buffer_gpu(ot['buf_vmat'], ncart * ncart)
                self.queue.finish()
            else:
                zero_buffer_gpu(ot['buf_vmat'], ncart * ncart)
            profile_kernel(self.queue, ot['k_vmat'], ot['vmat_global'], ot['vmat_local'], timing, 'gpu_vmat', 'gpu_vmat_cl')

        t0 = _time.perf_counter()
        _gpu_vmat_cart_to_sph(buf_c2s, ot['buf_vmat'], ot['buf_c2s_scratch'], ot['buf_vmat_sph'], nao, ncart)
        cl.enqueue_copy(self.queue, ot['vmat_sph32'], ot['buf_vmat_sph']).wait()
        vmat = ot['vmat_sph32'].astype(np.float64)
        if self.xctype == 'GGA':
            vmat = vmat + vmat.T
        _timing_record(timing, 'host_vmat_d2h', t0)
        if timing is not None:
            _finalize_gpu_timing(timing)
            self.last_timing = timing
        return nelec, excsum, vmat

    def nr_rks_hermite_rho_only(self, dm, profile=False):
        '''GPU ρ projection only (OTF Hermite). Requires setup_onthefly().'''
        if not getattr(self, '_otf_ready', False):
            raise RuntimeError('Call setup_onthefly() or setup_xc_grid_gpu() before rho projection')
        ot = self.otf
        ncart = ot['ncart']
        ngrids = self.ngrids
        timing = {} if profile else None
        dm32 = np.ascontiguousarray(dm, dtype=np.float32)
        t0 = _time.perf_counter()
        cl.enqueue_copy(self.queue, ot['buf_dm_sph'], dm32)
        _gpu_dm_to_cart(ot['ao_eval'].buf_c2s, ot['buf_dm_sph'], ot['buf_c2s_scratch'], ot['buf_dm_cart'], self.nao, ncart)
        _timing_record(timing, 'gpu_dm_cart', t0)
        t0 = _time.perf_counter()
        cl.enqueue_nd_range_kernel(self.queue, ot['k_rho'], ot['rho_global'], ot['rho_local'])
        _gpu_sync(self.queue)
        _timing_record(timing, 'gpu_rho', t0)
        t0 = _time.perf_counter()
        cl.enqueue_copy(self.queue, ot['rho32_full'][:4 * ngrids], ot['buf_rho']).wait()
        rho = ot['rho32_full'][:4 * ngrids].reshape(4, ngrids).astype(np.float64)
        _timing_record(timing, 'host_rho_d2h', t0)
        if timing is not None:
            _finalize_gpu_timing(timing)
            self.last_timing = timing
        return rho

    def release(self):
        for buf in self.bufAo:
            buf.release()
        for buf in self.bufAoDm:
            buf.release()
        self.bufRho.release()
        self.bufWv.release()
        self.bufAow.release()
        self.bufVmat.release()
        for name in ('bufAoDmFull',):
            for buf in getattr(self, name, []):
                buf.release()
        for name in ('bufRhoFull', 'bufWvFull', 'bufAowFull'):
            buf = getattr(self, name, None)
            if buf is not None:
                buf.release()
        ot = getattr(self, 'otf', None)
        if ot is not None:
            for key in ('buf_coords4', 'buf_dm_cart', 'buf_dm_sph', 'buf_vmat_sph', 'buf_c2s_scratch', 'buf_rad_val', 'buf_rad_dr', 'buf_atom_ao0', 'buf_atom_nao', 'buf_rho', 'buf_wv', 'buf_vmat', 'buf_vmat_partial', 'buf_nelec_exc', 'buf_reduce0', 'buf_reduce1', 'buf_gtile_atom_off', 'buf_gtile_atom_list', 'buf_pair_gtile_off', 'buf_pair_gtile_list'):
                buf = ot.get(key)
                if buf is not None:
                    buf.release()
            self.otf = None
            self._otf_ready = False
        pcg = getattr(self, 'pcg', None)
        if pcg is not None:
            for key in ('buf_ao', 'buf_aodm', 'buf_rho', 'buf_wv', 'buf_vmat', 'buf_aow', 'buf_chi'):
                for buf in pcg.get(key, []) if isinstance(pcg.get(key), list) else [pcg.get(key)]:
                    if buf is not None:
                        buf.release()
            for key in ('buf_rad_val', 'buf_rad_dr', 'buf_coords4', 'buf_dm_cart', 'buf_vmat_sph', 'buf_c2s_scratch',
                        'buf_nelec_exc', 'buf_reduce0', 'buf_reduce1',
                        'buf_atom_ao0', 'buf_atom_nao', 'buf_atom_ao0_cart', 'buf_atom_nao_cart',
                        'buf_atom_coords_h', 'buf_radial_l_h', 'buf_atom_radial_offset_h', 'buf_atom_radial_list_h'):
                buf = pcg.get(key)
                if buf is not None:
                    buf.release()
            self.pcg = None
            self._pcg_ready = False


_xc_plan_cache = {}


def get_xc_grid_plan(mol, grids, xc_code, blk=8192):
    key = (id(mol), id(grids), str(xc_code), int(blk), mol.nao_nr(), grids.coords.shape[0])
    plan = _xc_plan_cache.get(key)
    if plan is not None and plan.nao == mol.nao_nr() and plan.ngrids == grids.coords.shape[0]:
        return plan
    plan = XCGridPlan(mol, grids, xc_code, blk=blk)
    _xc_plan_cache[key] = plan
    return plan


def setup_precomputed_gto(mol, grids, xc_code, blk=8192, max_memory_frac=0.75, max_memory_mb=2000, gpu_only=True, gpu_xc='auto', xc_eval='gpu', fused='tiled', ao_proj='auto', **kwargs):
    '''Pre-SCF: eval GTO AOs, upload float32 to GPU (timed separately from SCF iterations).'''
    plan = get_xc_grid_plan(mol, grids, xc_code, blk=blk)
    plan.setup_precomputed_gto(max_memory_frac=max_memory_frac, max_memory_mb=max_memory_mb, gpu_only=gpu_only, gpu_xc=gpu_xc, xc_eval=xc_eval, fused=fused, ao_proj=ao_proj, **kwargs)
    return plan


def nr_rks_precomputed_gto(mol, grids, xc_code, dm, projection='gpu', profile=False):
    '''XC with precomputed grid AOs. Requires setup_precomputed_gto() first.'''
    plan = get_xc_grid_plan(mol, grids, xc_code)
    if not getattr(plan, '_pcg_ready', False):
        raise RuntimeError('Call setup_precomputed_gto(mol, grids, xc) before projection')
    return plan.nr_rks_precomputed_gto(dm, projection=projection, profile=profile)


def setup_xc_grid_gpu(mol, grids, xc_code, blk=8192, r0_ang=0.002, du=0.02, rmax_ang=None, xc_eval='gpu', gpu_xc='auto', spline_order='cubic', vmat_mode='otf', vmat_grid_splits=1, **kwargs):
    '''Pre-SCF setup: compile OpenCL kernels, build Hermite tables, alloc/upload GPU buffers.

    Must be called once after grids.build() and before the SCF loop when using GPU XC.
  '''
    plan = get_xc_grid_plan(mol, grids, xc_code, blk=blk)
    plan.setup_onthefly(r0_ang=r0_ang, du=du, rmax_ang=rmax_ang, xc_eval=xc_eval, gpu_xc=gpu_xc, spline_order=spline_order, vmat_mode=vmat_mode, vmat_grid_splits=vmat_grid_splits, **kwargs)
    return plan


def nr_rks_gpu(mol, grids, xc_code, dm, max_memory=2000, profile=False):
    '''GPU on-the-fly Hermite XC for RKS.  Requires prior setup_xc_grid_gpu() call.'''
    plan = get_xc_grid_plan(mol, grids, xc_code)
    if not getattr(plan, '_otf_ready', False):
        raise RuntimeError('Call setup_xc_grid_gpu(mol, grids, xc) before SCF when using GPU XC')
    return plan.nr_rks_hermite_onthefly(dm, profile=profile)


def nr_rks_gpu_hermite_ao(mol, grids, xc_code, dm, max_memory=2000, r0_ang=0.002, du=0.02, rmax_ang=None):
    return get_xc_grid_plan(mol, grids, xc_code).nr_rks_hermite_ao(dm, r0_ang=r0_ang, du=du, rmax_ang=rmax_ang)


def nr_rks_gpu_hermite_onthefly(mol, grids, xc_code, dm, max_memory=2000, r0_ang=0.002, du=0.02, rmax_ang=None, profile=False):
    plan = get_xc_grid_plan(mol, grids, xc_code)
    if not getattr(plan, '_otf_ready', False):
        plan.setup_onthefly(r0_ang=r0_ang, du=du, rmax_ang=rmax_ang)
    return plan.nr_rks_hermite_onthefly(dm, profile=profile)
