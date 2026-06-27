import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

from . import get_ctx, get_queue, get_prg, round_up
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

def nr_rks_gpu(mol, grids, xc_code, dm, max_memory=2000):
    '''GPU XC grid integration for RKS.

    Strategy: AO evaluation on CPU (PySCF's eval_gto), then offload
    the expensive matrix multiplications (dot_ao_dm, dot_ao_ao) to GPU
    using tiled GEMM with local memory. XC functional eval on CPU (libxc).

    All GPU computation in float32. Returns nelec, excsum, vmat (float64).
    '''
    from pyscf.dft import numint
    ni = numint.NumInt()
    xctype = ni._xc_type(xc_code)

    nao = mol.nao_nr()
    ngrids = grids.coords.shape[0]
    dm32 = np.ascontiguousarray(dm, dtype=np.float32)
    if xctype not in ('LDA', 'GGA'):
        raise NotImplementedError(f'xctype={xctype} not supported on GPU')

    nelec = 0.0
    excsum = 0.0
    vmat = np.zeros((nao, nao), dtype=np.float64)

    BLK = 8192
    ctx = get_ctx()
    queue = get_queue()
    fbytes = np.dtype(np.float32).itemsize
    prg = get_prg()
    ao32_lda = np.empty((BLK, nao), dtype=np.float32)
    ao32_gga = np.empty((4, BLK, nao), dtype=np.float32)
    rho = np.empty((4, BLK), dtype=np.float64)
    rho32_flat = np.empty(4 * BLK, dtype=np.float32)
    wv_flat = np.empty(4 * BLK, dtype=np.float32)
    vmat_blk = np.empty((nao, nao), dtype=np.float32)
    bufDm = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, dm32.nbytes, dm32)
    bufAo = [cl.Buffer(ctx, cl.mem_flags.READ_WRITE, BLK * nao * fbytes) for _ in range(4)]
    bufAoDm = [cl.Buffer(ctx, cl.mem_flags.READ_WRITE, BLK * nao * fbytes) for _ in range(4)]
    bufRho = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 4 * BLK * fbytes)
    bufWv = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 4 * BLK * fbytes)
    bufAow = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, BLK * nao * fbytes)
    bufVmat = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, nao * nao * fbytes)

    for ip0 in range(0, ngrids, BLK):
        ip1 = min(ip0 + BLK, ngrids)
        nblk = ip1 - ip0
        coords_blk = grids.coords[ip0:ip1]
        weight_blk = grids.weights[ip0:ip1]

        if xctype == 'LDA':
            ao = ni.eval_ao(mol, coords_blk, deriv=0)  # [nblk, nao] CPU

            ao32 = ao32_lda[:nblk]
            ao32[:] = ao
            cl.enqueue_copy(queue, bufAo[0], ao32)
            matmul_gpu_buf(bufAo[0], bufDm, bufAoDm[0], nblk, nao, nao)
            _knl(prg, 'contract_rho_lda_from_aodm')(
                queue, (round_up(nblk, TILE),), (TILE,),
                bufAo[0], bufAoDm[0], bufRho,
                np.int32(nao), np.int32(nblk)
            )
            cl.enqueue_copy(queue, rho32_flat[:nblk], bufRho).wait()
            rho0 = rho[0, :nblk]
            rho0[:] = rho32_flat[:nblk]

            exc, vxc = ni.eval_xc_eff(xc_code, rho0, deriv=1, xctype='LDA', spin=0)[:2]

            den = rho0 * weight_blk
            nelec += float(den.sum())
            excsum += float(np.dot(den, exc))

            wv_flat[:nblk] = np.ascontiguousarray(weight_blk * vxc, dtype=np.float32)
            cl.enqueue_copy(queue, bufWv, wv_flat[:nblk])
            _knl(prg, 'scale_aow_lda')(
                queue, (round_up(nblk, TILE), round_up(nao, TILE)), (TILE, TILE),
                bufAo[0], bufWv, bufAow,
                np.int32(nao), np.int32(nblk)
            )
            matmul_gpu_buf(bufAow, bufAo[0], bufVmat, nao, nao, nblk, transpose_A=True)
            cl.enqueue_copy(queue, vmat_blk, bufVmat).wait()
            vmat += vmat_blk.astype(np.float64)

        elif xctype == 'GGA':
            ao = ni.eval_ao(mol, coords_blk, deriv=1)  # [4, nblk, nao] CPU

            ao32 = ao32_gga[:, :nblk]
            ao32[:] = ao
            for c in range(4):
                cl.enqueue_copy(queue, bufAo[c], ao32[c])
                matmul_gpu_buf(bufAo[c], bufDm, bufAoDm[c], nblk, nao, nao)
            _knl(prg, 'contract_rho_gga_from_aodm')(
                queue, (round_up(nblk, TILE),), (TILE,),
                bufAo[0], bufAo[1], bufAo[2], bufAo[3],
                bufAoDm[0], bufAoDm[1], bufAoDm[2], bufAoDm[3], bufRho,
                np.int32(nao), np.int32(nblk)
            )
            cl.enqueue_copy(queue, rho32_flat[:4*nblk], bufRho).wait()
            rho_blk = rho[:, :nblk]
            rho_blk[:] = rho32_flat[:4*nblk].reshape(4, nblk)

            evfk = ni.eval_xc_eff(xc_code, rho_blk, deriv=1, xctype='GGA', spin=0)
            exc = evfk[0]
            vxc = evfk[1]  # (4, nblk): vxc[0]=vrho, vxc[1:4]=dE/d(rho_grad)

            den = rho_blk[0] * weight_blk
            nelec += float(den.sum())
            excsum += float(np.dot(den, exc))

            # wv[c] = weight * vxc[c] for c=0..3
            # wv[0] *= 0.5 for hermi_sum (vmat + vmat.T)
            wv_blk = wv_flat[:4*nblk].reshape(4, nblk)
            wv_blk[:] = weight_blk.astype(np.float32)[np.newaxis, :] * np.ascontiguousarray(vxc, dtype=np.float32)
            wv_blk[0] *= 0.5

            # vmat = ao[0]^T @ aow  (then hermi_sum at end)
            cl.enqueue_copy(queue, bufWv, wv_flat[:4*nblk])
            _knl(prg, 'scale_aow_gga_split')(
                queue, (round_up(nblk, TILE), round_up(nao, TILE)), (TILE, TILE),
                bufAo[0], bufAo[1], bufAo[2], bufAo[3], bufWv, bufAow,
                np.int32(nao), np.int32(nblk)
            )
            matmul_gpu_buf(bufAow, bufAo[0], bufVmat, nao, nao, nblk, transpose_A=True)
            cl.enqueue_copy(queue, vmat_blk, bufVmat).wait()
            vmat += vmat_blk.astype(np.float64)

    if xctype == 'GGA':
        # hermi_sum: vmat + vmat.T (wv[0] was halved to compensate)
        vmat = vmat + vmat.T

    bufDm.release()
    for buf in bufAo:
        buf.release()
    for buf in bufAoDm:
        buf.release()
    bufRho.release()
    bufWv.release()
    bufAow.release()
    bufVmat.release()
    return nelec, excsum, vmat
