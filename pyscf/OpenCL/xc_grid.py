import time as _time
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

from pyscf.dft.gen_grid import ALIGNMENT_UNIT
from pyscf.gto.eval_gto import BLKSIZE, NBINS, CUTOFF

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
    gpu_keys = ('gpu_rho', 'gpu_xc_pbe', 'gpu_vmat')
    host_keys = ('host_h2d_dm', 'host_dm_cart', 'host_rho_d2h', 'host_xc_libxc', 'host_xc_reduce', 'host_vmat_d2h', 'host_pair_mask', 'host_cpu_projection')
    timing['gpu_total'] = sum(timing.get(k, 0.0) for k in gpu_keys)
    timing['host_total'] = sum(timing.get(k, 0.0) for k in host_keys)
    timing['wall_profiled'] = timing['gpu_total'] + timing['host_total']
    return timing

# Order for benchmark printouts (seconds on plan.last_timing).
TIMING_STAGE_ORDER = (
    'host_h2d_dm', 'host_dm_cart', 'gpu_rho', 'host_rho_d2h',
    'gpu_xc_pbe', 'host_xc_libxc', 'host_xc_reduce',
    'gpu_vmat', 'host_vmat_d2h', 'host_pair_mask', 'host_cpu_projection',
    'gpu_total', 'host_total', 'wall_profiled', 'n_blocks',
)

def _is_pbe_xc(xc_code):
    s = str(xc_code).upper().replace(' ', '')
    if s != 'PBE' and 'GGA_X_PBE' not in s:
        return False
    for bad in ('PBE_SOL', 'PBESOL', 'RPBE', 'PBE0', 'PBELOC', 'PBEINT', 'PBE_VWN', 'PBE_MOL'):
        if bad in s:
            return False
    return s == 'PBE' or ('GGA_X_PBE' in s and 'GGA_C_PBE' in s)

def _resolve_gpu_xc(gpu_xc, xc_code, xctype):
    if gpu_xc in (None, 'cpu', 'libxc'):
        return None
    if gpu_xc == 'auto':
        gpu_xc = 'pbe_f32' if xctype == 'GGA' and _is_pbe_xc(xc_code) else None
        return gpu_xc
    if gpu_xc not in ('pbe_f32', 'pbe_f64'):
        raise ValueError(f'gpu_xc={gpu_xc!r}; use auto, pbe_f32, pbe_f64, or cpu')
    if xctype != 'GGA' or not _is_pbe_xc(xc_code):
        raise ValueError(f'gpu_xc={gpu_xc!r} requires unmodified PBE (xc_code={xc_code!r})')
    return gpu_xc

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

    def _get_ao_hermite(self, r0_ang=0.002, du=0.02, rmax_ang=None):
        key = (float(r0_ang), float(du), None if rmax_ang is None else float(rmax_ang))
        if getattr(self, '_ao_hermite_key', None) == key:
            return self.ao_hermite
        if rmax_ang is None:
            from pyscf.data import nist
            rmax_ang = float(np.max(np.linalg.norm(self.grids.coords[:, None, :] - self.mol.atom_coords()[None, :, :], axis=2)) * nist.BOHR + 0.2)
        from .ao_hermite import OpenCLAOHermiteEvaluator
        self.ao_hermite = OpenCLAOHermiteEvaluator(self.mol, r0_ang=r0_ang, du=du, rmax_ang=max(rmax_ang, 8.0), midpoint_fit=True)
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

    def setup_precomputed_gto(self, max_memory_frac=0.75, max_memory_mb=2000, gpu_only=True, gpu_xc='auto', fused='tiled'):
        '''One-time: eval GTO AOs on grid, upload float32 AO to GPU (outside SCF iteration budget).

        gpu_only=True: skip host float64 AO cache (GPU projection path only).
        gpu_xc: auto | pbe_f32 | pbe_f64 | cpu — GPU PBE vxc (GGA PBE only); auto uses pbe_f32 for PBE.
        fused: 'tiled' | 'gemm' | False — projection strategy.
          tiled (default): fused tiled GEMM+contract rho + pair vmat (1 launch each).
          gemm: full-grid GEMM + contract (slow fallback for huge atoms).
          False: Python block loop + tiled matmul.
        '''
        if fused is True:
            fused = 'tiled'
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
        ao_staging = [np.empty((ngrids, nao), dtype=np.float32) for _ in range(ncomp)]
        t_eval0 = _time.perf_counter()
        for ip0 in range(0, ngrids, blksize):
            ip1 = min(ip0 + blksize, ngrids)
            coords = grids.coords[ip0:ip1]
            mask = screen_index[ip0 // BLKSIZE:]
            ao = self.ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask, cutoff=grids.cutoff)
            if ao_deriv:
                for c in range(ncomp):
                    ao_staging[c][ip0:ip1] = ao[c].astype(np.float32)
                    if ao_host is not None:
                        ao_host[c][ip0:ip1] = ao[c]
            else:
                ao_staging[0][ip0:ip1] = ao.astype(np.float32)
                if ao_host is not None:
                    ao_host[0][ip0:ip1] = ao
        t_eval = _time.perf_counter() - t_eval0
        mf = cl.mem_flags
        buf_ao = []
        t_up0 = _time.perf_counter()
        for c in range(ncomp):
            buf_ao.append(cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, ao_staging[c].nbytes, ao_staging[c]))
        buf_rho = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * ngrids * FBYTES)
        buf_wv = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * ngrids * FBYTES)
        buf_vmat = cl.Buffer(self.ctx, mf.READ_WRITE, nao * nao * FBYTES)
        buf_aodm = [cl.Buffer(self.ctx, mf.READ_WRITE, blksize * nao * FBYTES) for _ in range(4)]
        buf_aow = cl.Buffer(self.ctx, mf.READ_WRITE, blksize * nao * FBYTES)
        buf_aodm_full = buf_aow_full = None
        precomp_knl = None
        atom_ao0 = atom_nao = buf_atom_ao0 = buf_atom_nao = None
        if fused == 'tiled':
            from .tile_config import get_active_tile_config
            tc = get_active_tile_config()
            atom_ao0, atom_nao = _atom_ao_layout_mol(mol)
            natoms = mol.natm
            if int(atom_nao.max()) > tc.MAX_AO_ATOM:
                raise NotImplementedError(
                    f'fused=tiled requires max atom_nao<={tc.MAX_AO_ATOM}; got {int(atom_nao.max())}')
            NPTILE, NATILE, WGS_VMAT = tc.NPTILE, tc.NATILE, tc.WGS_VMAT
            buf_atom_ao0 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, atom_ao0.nbytes, atom_ao0)
            buf_atom_nao = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, atom_nao.nbytes, atom_nao)
            rho_knl = 'rho_lda_precomp_pair' if self.xctype == 'LDA' else 'rho_gga_precomp_pair'
            rho_global = (round_up(ngrids, NPTILE), 1)
            rho_local = (NPTILE, 1)
            vmat_knl = 'vmat_lda_precomp_pair' if self.xctype == 'LDA' else 'vmat_gga_precomp_pair'
            k_rho = cl.Kernel(self.prg, rho_knl)
            k_vmat = cl.Kernel(self.prg, vmat_knl)
            if self.xctype == 'LDA':
                k_rho.set_args(buf_ao[0], self.bufDm, buf_atom_ao0, buf_atom_nao, buf_rho, np.int32(nao), np.int32(ngrids), np.int32(natoms))
                k_vmat.set_args(buf_ao[0], buf_wv, buf_atom_ao0, buf_atom_nao, buf_vmat, np.int32(nao), np.int32(ngrids), np.int32(natoms))
            else:
                k_rho.set_args(buf_ao[0], buf_ao[1], buf_ao[2], buf_ao[3], self.bufDm, buf_atom_ao0, buf_atom_nao, buf_rho, np.int32(nao), np.int32(ngrids), np.int32(natoms))
                k_vmat.set_args(buf_ao[0], buf_ao[1], buf_ao[2], buf_ao[3], buf_wv, buf_atom_ao0, buf_atom_nao, buf_vmat, np.int32(nao), np.int32(ngrids), np.int32(natoms))
            precomp_knl = {
                'k_rho': k_rho, 'k_vmat': k_vmat,
                'rho_global': rho_global,
                'rho_local': rho_local,
                'vmat_global': (natoms, natoms * WGS_VMAT),
                'vmat_local': (1, WGS_VMAT),
                'tiled': True,
            }
        elif fused == 'gemm':
            buf_aodm_full = [cl.Buffer(self.ctx, mf.READ_WRITE, ngrids * nao * FBYTES) for _ in range(4)]
            buf_aow_full = cl.Buffer(self.ctx, mf.READ_WRITE, ngrids * nao * FBYTES)
        gpu_xc = _resolve_gpu_xc(gpu_xc, self.xc_code, self.xctype)
        buf_exc = buf_vrho = buf_vsigma = buf_rho64 = buf_wv64 = None
        exc_host = vrho_host = vsigma_host = rho64_host = wv64_host = None
        if gpu_xc is not None:
            if gpu_xc == 'pbe_f32':
                buf_exc = cl.Buffer(self.ctx, mf.READ_WRITE, ngrids * FBYTES)
                buf_vrho = cl.Buffer(self.ctx, mf.READ_WRITE, ngrids * FBYTES)
                buf_vsigma = cl.Buffer(self.ctx, mf.READ_WRITE, ngrids * FBYTES)
                exc_host = np.empty(ngrids, dtype=np.float32)
                vrho_host = np.empty(ngrids, dtype=np.float32)
                vsigma_host = np.empty(ngrids, dtype=np.float32)
            else:
                buf_exc = cl.Buffer(self.ctx, mf.READ_WRITE, ngrids * DBYTES)
                buf_vrho = cl.Buffer(self.ctx, mf.READ_WRITE, ngrids * DBYTES)
                buf_vsigma = cl.Buffer(self.ctx, mf.READ_WRITE, ngrids * DBYTES)
                buf_rho64 = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * ngrids * DBYTES)
                buf_wv64 = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * ngrids * DBYTES)
                exc_host = np.empty(ngrids, dtype=np.float64)
                vrho_host = np.empty(ngrids, dtype=np.float64)
                vsigma_host = np.empty(ngrids, dtype=np.float64)
                rho64_host = np.empty(4 * ngrids, dtype=np.float64)
                wv64_host = np.empty(4 * ngrids, dtype=np.float64)
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
            'gpu_xc': gpu_xc,
            'buf_exc': buf_exc, 'buf_vrho': buf_vrho, 'buf_vsigma': buf_vsigma,
            'buf_rho64': buf_rho64, 'buf_wv64': buf_wv64,
            'exc_host': exc_host, 'vrho_host': vrho_host, 'vsigma_host': vsigma_host,
            'rho64_host': rho64_host, 'wv64_host': wv64_host,
            'mem': mem, 'fused': fused,
        }
        self.precalc_timing = {
            'eval_ao_cpu': t_eval, 'upload_gpu': t_upload, 'setup_total': _time.perf_counter() - t0,
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
        Uses GPU PBE kernels when setup_precomputed_gto(gpu_xc='auto'|'pbe_f32'|'pbe_f64') was used.
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

    def _gpu_pbe_xc(self, pcg, ngrids, rho32, weight64, timing=None):
        '''Evaluate PBE vxc on GPU; fill buf_wv for vmat contraction.'''
        gpu_xc = pcg['gpu_xc']
        t0 = _time.perf_counter()
        if gpu_xc == 'pbe_f32':
            _knl(self.prg, 'pbe_xc_f32')(
                self.queue, (round_up(ngrids, TILE),), (TILE,),
                pcg['buf_rho'], pcg['buf_exc'], pcg['buf_vrho'], pcg['buf_vsigma'], np.int32(ngrids))
            _knl(self.prg, 'compute_wv_gga_f32')(
                self.queue, (round_up(ngrids, TILE),), (TILE,),
                pcg['buf_weight'], pcg['buf_vrho'], pcg['buf_vsigma'], pcg['buf_rho'],
                pcg['buf_wv'], np.int32(ngrids))
        else:
            rho64 = rho32[:4 * ngrids].reshape(4, ngrids).astype(np.float64)
            pcg['rho64_host'][:] = rho64.ravel()
            cl.enqueue_copy(self.queue, pcg['buf_rho64'], pcg['rho64_host'])
            _knl(self.prg, 'pbe_xc_f64')(
                self.queue, (round_up(ngrids, TILE),), (TILE,),
                pcg['buf_rho64'], pcg['buf_exc'], pcg['buf_vrho'], pcg['buf_vsigma'], np.int32(ngrids))
            _knl(self.prg, 'compute_wv_gga_f64')(
                self.queue, (round_up(ngrids, TILE),), (TILE,),
                pcg['buf_weight64'], pcg['buf_vrho'], pcg['buf_vsigma'], pcg['buf_rho64'],
                pcg['buf_wv64'], np.int32(ngrids))
        _gpu_sync(self.queue)
        _timing_record(timing, 'gpu_xc_pbe', t0)
        t0 = _time.perf_counter()
        if gpu_xc == 'pbe_f32':
            cl.enqueue_copy(self.queue, pcg['exc_host'], pcg['buf_exc']).wait()
            rho0 = rho32[:ngrids]
            den = rho0.astype(np.float64) * weight64
            nelec = float(den.sum())
            excsum = float(np.dot(den, np.nan_to_num(pcg['exc_host'].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)))
            cl.enqueue_copy(self.queue, pcg['wv32_host'][:4 * ngrids], pcg['buf_wv']).wait()
            wv32 = pcg['wv32_host'][:4 * ngrids].reshape(4, ngrids)
            np.nan_to_num(wv32, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            wv32[0] *= 0.5
            cl.enqueue_copy(self.queue, pcg['buf_wv'], pcg['wv32_host'][:4 * ngrids])
        else:
            cl.enqueue_copy(self.queue, pcg['exc_host'], pcg['buf_exc']).wait()
            rho64 = rho32[:4 * ngrids].reshape(4, ngrids).astype(np.float64)
            den = rho64[0] * weight64
            nelec = float(den.sum())
            excsum = float(np.dot(den, pcg['exc_host']))
            cl.enqueue_copy(self.queue, pcg['wv64_host'], pcg['buf_wv64']).wait()
            wv32 = pcg['wv32_host'][:4 * ngrids].reshape(4, ngrids)
            wv32[:] = pcg['wv64_host'].reshape(4, ngrids).astype(np.float32)
            wv32[0] *= 0.5
            cl.enqueue_copy(self.queue, pcg['buf_wv'], pcg['wv32_host'][:4 * ngrids])
        _timing_record(timing, 'host_xc_reduce', t0)
        return nelec, excsum

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
        if pk and pk.get('tiled'):
            cl.enqueue_nd_range_kernel(self.queue, pk['k_rho'], pk['rho_global'], pk['rho_local'])
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
        if pk and pk.get('tiled'):
            cl.enqueue_nd_range_kernel(self.queue, pk['k_vmat'], pk['vmat_global'], pk['vmat_local'])
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
        zero_buffer_gpu(pcg['buf_vmat'], nao * nao)
        _timing_record(timing, 'host_h2d_dm', t0)
        fused = pcg.get('fused', 'tiled')
        if fused:
            self._precomp_rho_fused(pcg, xctype, nao, ngrids, timing=timing)
            n_blocks = 0
        else:
            n_blocks = self._precomp_rho_blocked(pcg, xctype, nao, ngrids, timing=timing)
        rho32 = pcg['rho32_host']
        t0 = _time.perf_counter()
        cl.enqueue_copy(self.queue, rho32[:4 * ngrids], pcg['buf_rho']).wait()
        _timing_record(timing, 'host_rho_d2h', t0)
        if xctype == 'LDA':
            t0 = _time.perf_counter()
            rho64 = rho32[:ngrids].astype(np.float64)
            exc, vxc = ni.eval_xc_eff(self.xc_code, rho64, deriv=1, xctype='LDA', spin=0)[:2]
            den = rho64 * weight64
            nelec = float(den.sum())
            excsum = float(np.dot(den, exc))
            wv32 = pcg['wv32_host']
            wv32[:ngrids] = weight32 * vxc.astype(np.float32)
            cl.enqueue_copy(self.queue, pcg['buf_wv'], wv32[:ngrids])
            _timing_record(timing, 'host_xc_libxc', t0)
            if fused:
                self._precomp_vmat_fused(pcg, xctype, nao, ngrids, timing=timing)
            else:
                self._precomp_vmat_blocked(pcg, xctype, nao, ngrids, timing=timing)
            t0 = _time.perf_counter()
            cl.enqueue_copy(self.queue, pcg['vmat32_host'], pcg['buf_vmat']).wait()
            vmat = pcg['vmat32_host'].astype(np.float64)
            _timing_record(timing, 'host_vmat_d2h', t0)
        else:
            if pcg.get('gpu_xc'):
                nelec, excsum = self._gpu_pbe_xc(pcg, ngrids, rho32, weight64, timing=timing)
            else:
                t0 = _time.perf_counter()
                rho64 = rho32[:4 * ngrids].reshape(4, ngrids).astype(np.float64)
                evfk = ni.eval_xc_eff(self.xc_code, rho64, deriv=1, xctype='GGA', spin=0)
                exc, vxc = evfk[0], evfk[1]
                den = rho64[0] * weight64
                nelec = float(den.sum())
                excsum = float(np.dot(den, exc))
                wv32 = pcg['wv32_host'][:4 * ngrids].reshape(4, ngrids)
                wv32[:] = weight32[np.newaxis, :] * np.ascontiguousarray(vxc, dtype=np.float32)
                wv32[0] *= 0.5
                cl.enqueue_copy(self.queue, pcg['buf_wv'], pcg['wv32_host'][:4 * ngrids])
                _timing_record(timing, 'host_xc_libxc', t0)
            if fused:
                self._precomp_vmat_fused(pcg, xctype, nao, ngrids, timing=timing)
            else:
                self._precomp_vmat_blocked(pcg, xctype, nao, ngrids, timing=timing)
            t0 = _time.perf_counter()
            cl.enqueue_copy(self.queue, pcg['vmat32_host'], pcg['buf_vmat']).wait()
            vmat = pcg['vmat32_host'].astype(np.float64)
            vmat = vmat + vmat.T
            _timing_record(timing, 'host_vmat_d2h', t0)
        if timing is not None:
            timing['n_blocks'] = n_blocks
            timing['fused'] = {'gemm': 1.0, 'tiled': 2.0, False: 0.0}.get(fused, 1.0)
            _finalize_gpu_timing(timing)
            self.last_timing = timing
        return nelec, excsum, vmat

    def _nr_rks_precomputed_gpu_dense(self, dm, profile=False):
        return self._nr_rks_precomputed_gpu(dm, profile=profile)

    def setup_onthefly(self, r0_ang=0.002, du=0.02, rmax_ang=None):
        '''One-time prep before SCF: OpenCL compile, Hermite tables, GPU buffers, kernel args.

        Call once after grids are built and before the SCF loop.  Keeps all
        allocation and static uploads out of the per-cycle hot path.
        '''
        from . import init_device
        init_device(quiet=getattr(self, '_otf_ready', False))
        self.ctx = get_ctx()
        self.queue = get_queue()
        self.prg = get_prg()
        ao_eval = self._get_ao_hermite(r0_ang=r0_ang, du=du, rmax_ang=rmax_ang)
        ncart = ao_eval.plan.ncart
        natoms = ao_eval.natoms
        ngrids = self.ngrids
        if ncart > 1024:
            raise NotImplementedError(f'On-the-fly kernels support ncart<=1024; got ncart={ncart}')
        mf = cl.mem_flags
        fbytes = np.dtype(np.float32).itemsize
        c2s = ao_eval.c2s
        atom_ao0, atom_nao = _atom_ao_layout(ao_eval)
        tc = get_active_tile_config()
        NPTILE, NATILE, WGS_VMAT = tc.NPTILE, tc.NATILE, tc.WGS_VMAT
        use_pair = (NATILE == 1)
        n_iTiles = round_up(natoms, NATILE) // NATILE
        if not use_pair and n_iTiles > tc.MAX_ITILE:
            raise NotImplementedError(
                f'rho prepass MAX_ITILE={tc.MAX_ITILE} too small for natoms={natoms} '
                f'(need {n_iTiles}); recompile with larger OPENCL_MAX_ITILE')
        if use_pair:
            rho_knl = 'rho_lda_pair' if self.xctype == 'LDA' else 'rho_gga_pair'
            vmat_knl = 'vmat_lda_pair' if self.xctype == 'LDA' else 'vmat_gga_pair'
            rho_global = (round_up(ngrids, NPTILE), 1)
            rho_local = (NPTILE, 1)
            vmat_global = (natoms, natoms * WGS_VMAT)
            vmat_local = (1, WGS_VMAT)
        else:
            rho_knl = 'rho_lda_tiled' if self.xctype == 'LDA' else 'rho_gga_tiled'
            vmat_knl = 'vmat_lda_tiled' if self.xctype == 'LDA' else 'vmat_gga_tiled'
            rho_global = (round_up(ngrids, NPTILE), NATILE)
            rho_local = (NPTILE, NATILE)
            n_jTiles = round_up(natoms, NATILE) // NATILE
            vmat_global = (n_iTiles, n_jTiles * WGS_VMAT)
            vmat_local = (1, WGS_VMAT)
        hermite_bufs = [
            None, ao_eval.buf_atom_coords,
            ao_eval.buf_rad_node,
            ao_eval.buf_radial_l, ao_eval.buf_radial_cart0,
            ao_eval.buf_atom_radial_offset, ao_eval.buf_atom_radial_list,
        ]
        hermite_params = [
            np.float32(ao_eval.plan.r0), np.float32(ao_eval.plan.du),
            np.int32(ao_eval.plan.nrad), np.int32(ncart),
            np.int32(ngrids), np.int32(natoms),
        ]
        coords4 = np.zeros((ngrids, 4), dtype=np.float32)
        coords4[:, :3] = self.grids.coords
        buf_coords4 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, coords4.nbytes, coords4)
        buf_dm_cart = cl.Buffer(self.ctx, mf.READ_ONLY, ncart * ncart * fbytes)
        buf_atom_ao0 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, atom_ao0.nbytes, atom_ao0)
        buf_atom_nao = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, atom_nao.nbytes, atom_nao)
        buf_rho = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * ngrids * fbytes)
        buf_wv = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * ngrids * fbytes)
        buf_vmat = cl.Buffer(self.ctx, mf.READ_WRITE, ncart * ncart * fbytes)
        k_rho = cl.Kernel(self.prg, rho_knl)
        k_rho.set_args(buf_coords4, *hermite_bufs[1:], buf_dm_cart, buf_atom_ao0, buf_atom_nao, buf_rho, *hermite_params)
        k_vmat = cl.Kernel(self.prg, vmat_knl)
        k_vmat.set_scalar_arg_dtypes([None] * 11 + [np.float32, np.float32, np.int32, np.int32, np.int32, np.int32])
        k_vmat.set_args(buf_coords4, *hermite_bufs[1:], buf_atom_ao0, buf_atom_nao, buf_wv, buf_vmat, *hermite_params)
        self.otf = {
            'ao_eval': ao_eval, 'ncart': ncart, 'natoms': natoms,
            'c2s': c2s, 'weight': self.grids.weights,
            'buf_coords4': buf_coords4, 'buf_dm_cart': buf_dm_cart,
            'buf_atom_ao0': buf_atom_ao0, 'buf_atom_nao': buf_atom_nao,
            'buf_rho': buf_rho, 'buf_wv': buf_wv, 'buf_vmat': buf_vmat,
            'k_rho': k_rho, 'k_vmat': k_vmat,
            'rho_global': rho_global, 'rho_local': rho_local,
            'vmat_global': vmat_global, 'vmat_local': vmat_local,
            'use_pair_kernels': use_pair,
            'rho32_full': np.empty(4 * ngrids, dtype=np.float32),
            'rho_full': np.empty((4, ngrids), dtype=np.float64),
            'wv_full': np.empty(4 * ngrids, dtype=np.float32),
            'vmat_cart32': np.empty((ncart, ncart), dtype=np.float32),
            'dm_cart32': np.empty((ncart, ncart), dtype=np.float32),
            'dm_tmp': np.empty((ncart, self.nao), dtype=np.float32),
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
        c2s = ot['c2s']
        weight = ot['weight']
        timing = {} if profile else None
        dm32 = np.ascontiguousarray(dm, dtype=np.float32)

        t0 = _time.perf_counter()
        np.matmul(c2s, dm32, out=ot['dm_tmp'])
        np.matmul(ot['dm_tmp'], c2s.T, out=ot['dm_cart32'])
        cl.enqueue_copy(self.queue, ot['buf_dm_cart'], ot['dm_cart32'])
        _timing_record(timing, 'host_dm_cart', t0)

        t0 = _time.perf_counter()
        cl.enqueue_nd_range_kernel(self.queue, ot['k_rho'], ot['rho_global'], ot['rho_local'])
        _gpu_sync(self.queue)
        _timing_record(timing, 'gpu_rho', t0)

        t0 = _time.perf_counter()
        cl.enqueue_copy(self.queue, ot['rho32_full'], ot['buf_rho']).wait()
        _timing_record(timing, 'host_rho_d2h', t0)

        if self.xctype == 'LDA':
            t0 = _time.perf_counter()
            rho0 = ot['rho_full'][0, :ngrids]
            rho0[:] = ot['rho32_full'][:ngrids]
            exc, vxc = self.ni.eval_xc_eff(self.xc_code, rho0, deriv=1, xctype='LDA', spin=0)[:2]
            den = rho0 * weight
            nelec = float(den.sum())
            excsum = float(np.dot(den, exc))
            ot['wv_full'][:ngrids] = np.ascontiguousarray(weight * vxc, dtype=np.float32)
            cl.enqueue_copy(self.queue, ot['buf_wv'], ot['wv_full'][:ngrids])
            _timing_record(timing, 'host_xc_libxc', t0)

            t0 = _time.perf_counter()
            zero_buffer_gpu(ot['buf_vmat'], ncart * ncart)
            cl.enqueue_nd_range_kernel(self.queue, ot['k_vmat'], ot['vmat_global'], ot['vmat_local'])
            _gpu_sync(self.queue)
            _timing_record(timing, 'gpu_vmat', t0)

            t0 = _time.perf_counter()
            cl.enqueue_copy(self.queue, ot['vmat_cart32'], ot['buf_vmat']).wait()
            vmat = c2s.T @ ot['vmat_cart32'].astype(np.float64) @ c2s
            _timing_record(timing, 'host_vmat_d2h', t0)
            if timing is not None:
                _finalize_gpu_timing(timing)
                self.last_timing = timing
            return nelec, excsum, vmat

        t0 = _time.perf_counter()
        rho = ot['rho_full'][:, :ngrids]
        rho[:] = ot['rho32_full'][:4 * ngrids].reshape(4, ngrids)
        evfk = self.ni.eval_xc_eff(self.xc_code, rho, deriv=1, xctype='GGA', spin=0)
        exc = evfk[0]
        vxc = evfk[1]
        den = rho[0] * weight
        nelec = float(den.sum())
        excsum = float(np.dot(den, exc))
        wv = ot['wv_full'][:4 * ngrids].reshape(4, ngrids)
        wv[:] = weight.astype(np.float32)[np.newaxis, :] * np.ascontiguousarray(vxc, dtype=np.float32)
        wv[0] *= 0.5
        cl.enqueue_copy(self.queue, ot['buf_wv'], ot['wv_full'][:4 * ngrids])
        _timing_record(timing, 'host_xc_libxc', t0)

        t0 = _time.perf_counter()
        zero_buffer_gpu(ot['buf_vmat'], ncart * ncart)
        cl.enqueue_nd_range_kernel(self.queue, ot['k_vmat'], ot['vmat_global'], ot['vmat_local'])
        _gpu_sync(self.queue)
        _timing_record(timing, 'gpu_vmat', t0)

        t0 = _time.perf_counter()
        cl.enqueue_copy(self.queue, ot['vmat_cart32'], ot['buf_vmat']).wait()
        vmat = c2s.T @ ot['vmat_cart32'].astype(np.float64) @ c2s
        vmat = vmat + vmat.T
        _timing_record(timing, 'host_vmat_d2h', t0)
        if timing is not None:
            _finalize_gpu_timing(timing)
            self.last_timing = timing
        return nelec, excsum, vmat

    def release(self):
        self.bufDm.release()
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
            for key in ('buf_coords4', 'buf_dm_cart', 'buf_atom_ao0', 'buf_atom_nao', 'buf_rho', 'buf_wv', 'buf_vmat'):
                buf = ot.get(key)
                if buf is not None:
                    buf.release()
            self.otf = None
            self._otf_ready = False
        pcg = getattr(self, 'pcg', None)
        if pcg is not None:
            for key in ('buf_ao', 'buf_aodm', 'buf_rho', 'buf_wv', 'buf_vmat', 'buf_aow'):
                for buf in pcg.get(key, []) if isinstance(pcg.get(key), list) else [pcg.get(key)]:
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


def setup_precomputed_gto(mol, grids, xc_code, blk=8192, max_memory_frac=0.75, max_memory_mb=2000, gpu_only=True, gpu_xc='auto'):
    '''Pre-SCF: eval GTO AOs, upload float32 to GPU (timed separately from SCF iterations).'''
    plan = get_xc_grid_plan(mol, grids, xc_code, blk=blk)
    plan.setup_precomputed_gto(max_memory_frac=max_memory_frac, max_memory_mb=max_memory_mb, gpu_only=gpu_only, gpu_xc=gpu_xc)
    return plan


def nr_rks_precomputed_gto(mol, grids, xc_code, dm, projection='gpu', profile=False):
    '''XC with precomputed grid AOs. Requires setup_precomputed_gto() first.'''
    plan = get_xc_grid_plan(mol, grids, xc_code)
    if not getattr(plan, '_pcg_ready', False):
        raise RuntimeError('Call setup_precomputed_gto(mol, grids, xc) before projection')
    return plan.nr_rks_precomputed_gto(dm, projection=projection, profile=profile)


def setup_xc_grid_gpu(mol, grids, xc_code, blk=8192, r0_ang=0.002, du=0.02, rmax_ang=None):
    '''Pre-SCF setup: compile OpenCL kernels, build Hermite tables, alloc/upload GPU buffers.

    Must be called once after grids.build() and before the SCF loop when using GPU XC.
  '''
    plan = get_xc_grid_plan(mol, grids, xc_code, blk=blk)
    plan.setup_onthefly(r0_ang=r0_ang, du=du, rmax_ang=rmax_ang)
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
