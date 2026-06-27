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
        cl.enqueue_copy(queue, C, bufC)
    queue.finish()
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

    nelec = 0.0
    excsum = 0.0
    vmat = np.zeros((nao, nao), dtype=np.float64)

    BLK = 8192

    for ip0 in range(0, ngrids, BLK):
        ip1 = min(ip0 + BLK, ngrids)
        nblk = ip1 - ip0
        coords_blk = grids.coords[ip0:ip1]
        weight_blk = np.ascontiguousarray(grids.weights[ip0:ip1], dtype=np.float64)

        if xctype == 'LDA':
            ao = ni.eval_ao(mol, coords_blk, deriv=0)  # [nblk, nao] CPU

            ao32 = np.ascontiguousarray(ao, dtype=np.float32)
            ao_dm = matmul_gpu(ao32, dm32)  # [nblk, nao]
            rho = np.sum(ao_dm * ao32, axis=1).astype(np.float64)

            exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype='LDA', spin=0)[:2]

            den = rho * weight_blk
            nelec += float(den.sum())
            excsum += float(np.dot(den, exc))

            wv = np.ascontiguousarray(weight_blk * vxc, dtype=np.float32)
            aow = ao32 * wv[:, np.newaxis]  # [nblk, nao]
            vmat_blk = matmul_gpu(aow, ao32, transpose_A=True)
            vmat += vmat_blk.astype(np.float64)

        elif xctype == 'GGA':
            ao = ni.eval_ao(mol, coords_blk, deriv=1)  # [4, nblk, nao] CPU

            ao0_32 = np.ascontiguousarray(ao[0], dtype=np.float32)  # [nblk, nao]
            ao_dm0 = matmul_gpu(ao0_32, dm32)  # [nblk, nao]

            rho = np.zeros((4, nblk), dtype=np.float64)
            rho[0] = np.sum(ao_dm0 * ao0_32, axis=1).astype(np.float64)

            for c in range(1, 4):
                ao_c_32 = np.ascontiguousarray(ao[c], dtype=np.float32)
                ao_dm_c = matmul_gpu(ao_c_32, dm32)  # [nblk, nao]
                rho[c] = (np.sum(ao_dm0 * ao_c_32, axis=1) +
                          np.sum(ao_dm_c * ao0_32, axis=1)).astype(np.float64)

            evfk = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype='GGA', spin=0)
            exc = evfk[0]
            vxc = evfk[1]  # (4, nblk): vxc[0]=vrho, vxc[1:4]=dE/d(rho_grad)

            den = rho[0] * weight_blk
            nelec += float(den.sum())
            excsum += float(np.dot(den, exc))

            # wv[c] = weight * vxc[c] for c=0..3
            # wv[0] *= 0.5 for hermi_sum (vmat + vmat.T)
            wv = np.zeros((4, nblk), dtype=np.float32)
            w32 = weight_blk.astype(np.float32)
            for c in range(4):
                wv[c] = w32 * np.ascontiguousarray(vxc[c], dtype=np.float32)
            wv[0] *= 0.5

            aow = np.zeros((nblk, nao), dtype=np.float32)
            for c in range(4):
                ao_c_32 = np.ascontiguousarray(ao[c], dtype=np.float32)
                aow += wv[c:c+1].T * ao_c_32

            # vmat = ao[0]^T @ aow  (then hermi_sum at end)
            vmat_blk = matmul_gpu(aow, ao0_32, transpose_A=True)
            vmat += vmat_blk.astype(np.float64)

        else:
            raise NotImplementedError(f'xctype={xctype} not supported on GPU')

    if xctype == 'GGA':
        # hermi_sum: vmat + vmat.T (wv[0] was halved to compensate)
        vmat = vmat + vmat.T

    return nelec, excsum, vmat
