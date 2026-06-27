import time as _time
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

from . import get_ctx, get_queue, get_prg, round_up, get_device_mem_info
from .buffers import CLBuffer

TILE = 32

# Cache kernel objects to avoid repeated retrieval
_kernels = {}

def _knl(prg, name):
    if name not in _kernels:
        _kernels[name] = cl.Kernel(prg, name)
    return _kernels[name]

def matmul_gpu_buf(bufA, bufB, bufC, M, N, K, transpose_A=False, transpose_B=False):
    if transpose_A and transpose_B:
        raise NotImplementedError('Both transposed not supported')
    if transpose_A:
        knl_name = 'matmul_tiled_transpose_A'
    elif transpose_B:
        knl_name = 'matmul_tiled_transpose_B'
    else:
        knl_name = 'matmul_tiled'
    queue = get_queue()
    _knl(get_prg(), knl_name)(
        queue, (round_up(M, TILE), round_up(N, TILE)), (TILE, TILE),
        bufA, bufB, bufC,
        np.int32(M), np.int32(N), np.int32(K)
    )
    return bufC

def matmul_gpu_buf_accum(bufA, bufB, bufC, M, N, K, transpose_A=False):
    if transpose_A:
        knl_name = 'matmul_tiled_transpose_A_accum'
    else:
        raise NotImplementedError('accum only supported for transpose_A')
    queue = get_queue()
    _knl(get_prg(), knl_name)(
        queue, (round_up(M, TILE), round_up(N, TILE)), (TILE, TILE),
        bufA, bufB, bufC,
        np.int32(M), np.int32(N), np.int32(K)
    )
    return bufC

def zero_buffer_gpu(buf, n):
    _knl(get_prg(), 'zero_buffer')(
        get_queue(), (round_up(n, TILE),), (TILE,),
        buf, np.int32(n)
    )

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
                    np.int32(nao), np.int32(nblk)
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
                    np.int32(nao), np.int32(nblk)
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
                    np.int32(nao), np.int32(nblk)
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
                    np.int32(nao), np.int32(nblk)
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
                np.int32(nao), np.int32(ngrids)
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
                np.int32(nao), np.int32(ngrids)
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
            np.int32(nao), np.int32(ngrids)
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
            np.int32(nao), np.int32(ngrids)
        )
        matmul_gpu_buf(self.bufAowFull, bufAo[0], self.bufVmat, nao, nao, ngrids, transpose_A=True)
        cl.enqueue_copy(self.queue, self.vmat_blk, self.bufVmat).wait()
        vmat += self.vmat_blk.astype(np.float64)
        return nelec, excsum, vmat + vmat.T

    def nr_rks_hermite_onthefly(self, dm, r0_ang=0.002, du=0.02, rmax_ang=None):
        import time as _time
        nao = self.nao
        ngrids = self.ngrids
        ao_eval = self._get_ao_hermite(r0_ang=r0_ang, du=du, rmax_ang=rmax_ang)
        ncart = ao_eval.plan.ncart
        natoms = ao_eval.natoms
        if ncart > 1024:
            raise NotImplementedError(f'On-the-fly kernels support ncart<=1024; got ncart={ncart}')
        mf = cl.mem_flags
        fbytes = np.dtype(np.float32).itemsize
        NPTILE = 16
        NATILE = 4
        WGS = NPTILE * NATILE  # 64
        _t = {}
        def _tick(label):
            _t[label] = _time.perf_counter()
        def _tock(label):
            dt = _time.perf_counter() - _t[label]
            print(f'  [timing] {label}: {dt:.3f}s')
            return dt
        _tick('setup0')

        if not hasattr(self, 'rho32_full'):
            self.rho32_full = np.empty(4 * self.ngrids, dtype=np.float32)
            self.rho_full = np.empty((4, self.ngrids), dtype=np.float64)
            self.wv_full = np.empty(4 * self.ngrids, dtype=np.float32)

        # Upload coords as float4
        coords4 = np.zeros((ngrids, 4), dtype=np.float32)
        coords4[:, :3] = self.grids.coords
        buf_coords4 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, coords4.nbytes, coords4)

        # Precompute DM_cart = c2s @ DM @ c2s^T  (work in Cartesian basis)
        c2s = ao_eval.c2s  # [ncart, nao] float32
        dm32 = np.ascontiguousarray(dm, dtype=np.float32)
        dm_cart32 = np.ascontiguousarray(c2s @ dm32 @ c2s.T, dtype=np.float32)
        buf_dm_cart = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, dm_cart32.nbytes, dm_cart32)

        # Precompute atom_ao0 and atom_nao from radial plan
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
        buf_atom_ao0 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, atom_ao0.nbytes, atom_ao0)
        buf_atom_nao = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, atom_nao.nbytes, atom_nao)

        # Buffers for rho partials and wv
        n_iTile = round_up(natoms, NATILE) // NATILE
        buf_rho = cl.Buffer(self.ctx, mf.READ_WRITE, n_iTile * 4 * ngrids * fbytes)
        buf_wv  = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * ngrids * fbytes)
        rho_partial_host = np.empty(n_iTile * 4 * ngrids, dtype=np.float32)

        weight = self.grids.weights

        # Common kernel args
        hermite_args = [
            buf_coords4, ao_eval.buf_atom_coords,
            ao_eval.buf_rad_val, ao_eval.buf_rad_du, ao_eval.buf_rad_dy,
            ao_eval.buf_radial_l, ao_eval.buf_radial_cart0,
            ao_eval.buf_atom_radial_offset, ao_eval.buf_atom_radial_list,
        ]
        hermite_params = [
            np.float32(ao_eval.plan.r0), np.float32(ao_eval.plan.du),
            np.int32(ao_eval.plan.nrad), np.int32(ncart),
            np.int32(ngrids), np.int32(natoms),
        ]

        # 2D grid for rho: (round_up(ngrids, NPTILE), round_up(natoms, NATILE))
        rho_global = (round_up(ngrids, NPTILE), round_up(natoms, NATILE))
        rho_local = (NPTILE, NATILE)

        _tock('setup0')
        if self.xctype == 'LDA':
            _tick('rho')
            _knl(self.prg, 'rho_lda_tiled')(
                self.queue, rho_global, rho_local,
                *hermite_args, buf_dm_cart, buf_atom_ao0, buf_atom_nao, buf_rho,
                *hermite_params,
            )
            self.queue.finish()
            _tock('rho')
            _tick('rho_copy')
            cl.enqueue_copy(self.queue, rho_partial_host, buf_rho).wait()
            _tock('rho_copy')
            # Sum over iTile partials
            self.rho32_full[:ngrids] = rho_partial_host[:n_iTile*ngrids].reshape(n_iTile, ngrids).sum(axis=0)
            rho0 = self.rho_full[0, :ngrids]
            rho0[:] = self.rho32_full[:ngrids]
            exc, vxc = self.ni.eval_xc_eff(self.xc_code, rho0, deriv=1, xctype='LDA', spin=0)[:2]
            den = rho0 * weight
            nelec = float(den.sum())
            excsum = float(np.dot(den, exc))
            self.wv_full[:ngrids] = np.ascontiguousarray(weight * vxc, dtype=np.float32)
            _tick('wv_upload')
            cl.enqueue_copy(self.queue, buf_wv, self.wv_full[:ngrids])
            _tock('wv_upload')
            # vmat: one workgroup per (iTile,jTile), private acc[QPT], no partial buffer
            n_iTiles = round_up(natoms, NATILE) // NATILE
            n_jTiles = round_up(natoms, NATILE) // NATILE
            WGS_V = 256
            buf_vmat = cl.Buffer(self.ctx, mf.READ_WRITE, ncart * ncart * fbytes)
            cl.enqueue_copy(self.queue, buf_vmat, np.zeros(ncart * ncart, dtype=np.float32)).wait()
            k_vmat = cl.Kernel(self.prg, 'vmat_lda_tiled')
            k_vmat.set_scalar_arg_dtypes([None]*13 + [np.float32, np.float32, np.int32, np.int32, np.int32, np.int32])
            k_vmat.set_args(*hermite_args, buf_atom_ao0, buf_atom_nao, buf_wv, buf_vmat, *hermite_params)
            _tick('vmat')
            cl.enqueue_nd_range_kernel(self.queue, k_vmat, (n_iTiles, n_jTiles * WGS_V), (1, WGS_V))
            self.queue.finish()
            _tock('vmat')
            _tick('vmat_copy')
            vmat_cart32 = np.empty((ncart, ncart), dtype=np.float32)
            cl.enqueue_copy(self.queue, vmat_cart32, buf_vmat).wait()
            _tock('vmat_copy')
            vmat_cart = vmat_cart32.astype(np.float64)
            vmat = c2s.T @ vmat_cart @ c2s
            buf_coords4.release()
            buf_dm_cart.release()
            buf_atom_ao0.release()
            buf_atom_nao.release()
            buf_rho.release()
            buf_wv.release()
            buf_vmat.release()
            return nelec, excsum, vmat

        # GGA
        _tick('rho')
        _knl(self.prg, 'rho_gga_tiled')(
            self.queue, rho_global, rho_local,
            *hermite_args, buf_dm_cart, buf_atom_ao0, buf_atom_nao, buf_rho,
            *hermite_params,
        )
        self.queue.finish()
        _tock('rho')
        _tick('rho_copy')
        cl.enqueue_copy(self.queue, rho_partial_host, buf_rho).wait()
        _tock('rho_copy')
        # Sum over iTile partials: each iTile writes 4*ngrids
        rho = self.rho_full[:, :ngrids]
        for c in range(4):
            rho[c] = rho_partial_host[:n_iTile*4*ngrids].reshape(n_iTile, 4, ngrids)[:, c, :].sum(axis=0)
        evfk = self.ni.eval_xc_eff(self.xc_code, rho, deriv=1, xctype='GGA', spin=0)
        exc = evfk[0]
        vxc = evfk[1]
        den = rho[0] * weight
        nelec = float(den.sum())
        excsum = float(np.dot(den, exc))
        wv = self.wv_full[:4*ngrids].reshape(4, ngrids)
        wv[:] = weight.astype(np.float32)[np.newaxis, :] * np.ascontiguousarray(vxc, dtype=np.float32)
        wv[0] *= 0.5
        _tick('wv_upload')
        cl.enqueue_copy(self.queue, buf_wv, self.wv_full[:4*ngrids])
        _tock('wv_upload')
        n_iTiles = round_up(natoms, NATILE) // NATILE
        n_jTiles = round_up(natoms, NATILE) // NATILE
        WGS_V = 256
        buf_vmat = cl.Buffer(self.ctx, mf.READ_WRITE, ncart * ncart * fbytes)
        cl.enqueue_copy(self.queue, buf_vmat, np.zeros(ncart * ncart, dtype=np.float32)).wait()
        k_vmat = cl.Kernel(self.prg, 'vmat_gga_tiled')
        k_vmat.set_scalar_arg_dtypes([None]*13 + [np.float32, np.float32, np.int32, np.int32, np.int32, np.int32])
        k_vmat.set_args(*hermite_args, buf_atom_ao0, buf_atom_nao, buf_wv, buf_vmat, *hermite_params)
        _tick('vmat')
        cl.enqueue_nd_range_kernel(self.queue, k_vmat, (n_iTiles, n_jTiles * WGS_V), (1, WGS_V))
        self.queue.finish()
        _tock('vmat')
        _tick('vmat_copy')
        vmat_cart32 = np.empty((ncart, ncart), dtype=np.float32)
        cl.enqueue_copy(self.queue, vmat_cart32, buf_vmat).wait()
        _tock('vmat_copy')
        vmat_cart = vmat_cart32.astype(np.float64)
        vmat = c2s.T @ vmat_cart @ c2s
        vmat = vmat + vmat.T
        buf_coords4.release()
        buf_dm_cart.release()
        buf_atom_ao0.release()
        buf_atom_nao.release()
        buf_rho.release()
        buf_wv.release()
        buf_vmat.release()
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


_xc_plan_cache = {}


def get_xc_grid_plan(mol, grids, xc_code, blk=8192):
    key = (id(mol), id(grids), str(xc_code), int(blk), mol.nao_nr(), grids.coords.shape[0])
    plan = _xc_plan_cache.get(key)
    if plan is not None and plan.nao == mol.nao_nr() and plan.ngrids == grids.coords.shape[0]:
        return plan
    plan = XCGridPlan(mol, grids, xc_code, blk=blk)
    _xc_plan_cache[key] = plan
    return plan


def nr_rks_gpu(mol, grids, xc_code, dm, max_memory=2000, use_hermite_ao=True):
    '''GPU XC grid integration for RKS.

    Strategy: AO evaluation on CPU (PySCF's eval_gto), then offload
    the expensive matrix multiplications (dot_ao_dm, dot_ao_ao) to GPU
    using tiled GEMM with local memory. XC functional eval on CPU (libxc).

    All GPU computation in float32. Returns nelec, excsum, vmat (float64).

    If use_hermite_ao=True (default), uses GPU Hermite interpolation for AO
    evaluation (one-shot, no Python block loop). Falls back to CPU-AO block
    loop if device memory is insufficient.
    '''
    plan = get_xc_grid_plan(mol, grids, xc_code)
    if use_hermite_ao:
        return plan.nr_rks_hermite_ao(dm)
    return plan.nr_rks(dm)


def nr_rks_gpu_hermite_ao(mol, grids, xc_code, dm, max_memory=2000, r0_ang=0.002, du=0.02, rmax_ang=None):
    return get_xc_grid_plan(mol, grids, xc_code).nr_rks_hermite_ao(dm, r0_ang=r0_ang, du=du, rmax_ang=rmax_ang)


def nr_rks_gpu_hermite_onthefly(mol, grids, xc_code, dm, max_memory=2000, r0_ang=0.002, du=0.02, rmax_ang=None):
    return get_xc_grid_plan(mol, grids, xc_code).nr_rks_hermite_onthefly(dm, r0_ang=r0_ang, du=du, rmax_ang=rmax_ang)
