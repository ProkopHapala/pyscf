import numpy as np
import pyopencl as cl

from . import get_ctx, get_queue, get_prg, round_up
from .radial_hermite import MappedHermiteRadialBasis
from .tile_config import get_active_tile_config
from .xc_grid import _knl, matmul_gpu_buf


TILE = 16


class OpenCLAOHermiteEvaluator:
    def __init__(self, mol, r0_ang=0.01, du=0.02, rmax_ang=8.0, midpoint_fit=True, spline_order='cubic'):
        self.mol = mol
        self.plan = MappedHermiteRadialBasis(mol, r0_ang=r0_ang, du=du, rmax_ang=rmax_ang, midpoint_fit=midpoint_fit, spline_order=spline_order)
        self.ctx = get_ctx()
        self.queue = get_queue()
        self.prg = get_prg()
        mf = cl.mem_flags
        if self.plan.lmax > 3:
            raise NotImplementedError('OpenCL atom-block Hermite AO kernel currently supports angular momentum l<=3')
        self.rad_node = np.ascontiguousarray(self.plan.radial_nodes, dtype=np.float32)
        self.atom_coords = np.zeros((self.plan.atom_coords.shape[0], 4), dtype=np.float32)
        self.atom_coords[:, :3] = self.plan.atom_coords
        self.radial_l = np.ascontiguousarray(self.plan.radial_l, dtype=np.int32)
        self.radial_cart0 = np.ascontiguousarray(self.plan.radial_cart0, dtype=np.int32)
        self.radial_atom = np.ascontiguousarray(self.plan.radial_atom, dtype=np.int32)
        self.atom_radial_offset = np.ascontiguousarray(self.plan.atom_radial_offset, dtype=np.int32)
        self.atom_radial_list = np.ascontiguousarray(self.plan.atom_radial_list, dtype=np.int32)
        self.natoms = self.plan.natoms
        self.buf_rad_node = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, self.rad_node.nbytes, self.rad_node)
        self.buf_atom_coords = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, self.atom_coords.nbytes, self.atom_coords)
        self.buf_radial_l = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, self.radial_l.nbytes, self.radial_l)
        self.buf_radial_cart0 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, self.radial_cart0.nbytes, self.radial_cart0)
        self.buf_radial_atom = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, self.radial_atom.nbytes, self.radial_atom)
        self.buf_atom_radial_offset = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, self.atom_radial_offset.nbytes, self.atom_radial_offset)
        self.buf_atom_radial_list = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, self.atom_radial_list.nbytes, self.atom_radial_list)
        self.c2s = np.ascontiguousarray(mol.cart2sph_coeff(normalized='sp'), dtype=np.float32)
        self.buf_c2s = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, self.c2s.nbytes, self.c2s)
        self.ngrid_alloc = 0
        self.buf_coords = None
        self.buf_cart = None
        self.buf_sph = None
        self.buf_cart_deriv = [None, None, None, None]
        self.buf_sph_deriv = [None, None, None, None]

    def _ensure_grid_buffers(self, ngrids):
        if ngrids <= self.ngrid_alloc:
            return
        mf = cl.mem_flags
        for name in ('buf_coords', 'buf_cart', 'buf_sph'):
            buf = getattr(self, name)
            if buf is not None:
                buf.release()
        for bufs in (self.buf_cart_deriv, self.buf_sph_deriv):
            for buf in bufs:
                if buf is not None:
                    buf.release()
        self.ngrid_alloc = int(ngrids)
        fbytes = np.dtype(np.float32).itemsize
        self.buf_coords = cl.Buffer(self.ctx, mf.READ_ONLY, self.ngrid_alloc * 4 * fbytes)
        self.buf_cart = cl.Buffer(self.ctx, mf.READ_WRITE, self.ngrid_alloc * self.plan.ncart * fbytes)
        self.buf_sph = cl.Buffer(self.ctx, mf.READ_WRITE, self.ngrid_alloc * self.mol.nao_nr() * fbytes)
        self.buf_cart_deriv = [cl.Buffer(self.ctx, mf.READ_WRITE, self.ngrid_alloc * self.plan.ncart * fbytes) for _ in range(4)]
        self.buf_sph_deriv = [cl.Buffer(self.ctx, mf.READ_WRITE, self.ngrid_alloc * self.mol.nao_nr() * fbytes) for _ in range(4)]

    def eval_cart_buf(self, coords):
        coords = np.ascontiguousarray(coords, dtype=np.float32)
        ngrids = coords.shape[0]
        coords4 = np.zeros((ngrids, 4), dtype=np.float32)
        coords4[:, :3] = coords
        self._ensure_grid_buffers(ngrids)
        cl.enqueue_copy(self.queue, self.buf_coords, coords4).wait()
        _knl(self.prg, 'eval_ao_mapped_hermite_cart_atom')(
            self.queue, (round_up(ngrids, TILE), round_up(self.natoms, TILE)), (TILE, TILE),
            self.buf_coords, self.buf_atom_coords, self.buf_rad_node,
            self.buf_radial_l, self.buf_radial_cart0,
            self.buf_atom_radial_offset, self.buf_atom_radial_list,
            self.buf_cart,
            np.float32(self.plan.r0), np.float32(self.plan.du),
            np.int32(self.plan.nrad), np.int32(self.plan.ncart), np.int32(ngrids), np.int32(self.natoms), np.int32(self.plan.spline_order_code)
        )
        return self.buf_cart, ngrids

    def eval_cart(self, coords):
        _, ngrids = self.eval_cart_buf(coords)
        out = np.empty((ngrids, self.plan.ncart), dtype=np.float32)
        cl.enqueue_copy(self.queue, out, self.buf_cart).wait()
        return out

    def eval_sph_buf(self, coords):
        _, ngrids = self.eval_cart_buf(coords)
        matmul_gpu_buf(self.buf_cart, self.buf_c2s, self.buf_sph, ngrids, self.mol.nao_nr(), self.plan.ncart)
        return self.buf_sph, ngrids

    def eval_sph(self, coords):
        _, ngrids = self.eval_sph_buf(coords)
        out = np.empty((ngrids, self.mol.nao_nr()), dtype=np.float32)
        cl.enqueue_copy(self.queue, out, self.buf_sph).wait()
        return out

    def eval_cart_deriv1_buf(self, coords):
        coords = np.ascontiguousarray(coords, dtype=np.float32)
        ngrids = coords.shape[0]
        coords4 = np.zeros((ngrids, 4), dtype=np.float32)
        coords4[:, :3] = coords
        self._ensure_grid_buffers(ngrids)
        cl.enqueue_copy(self.queue, self.buf_coords, coords4).wait()
        _knl(self.prg, 'eval_ao_mapped_hermite_cart_deriv1_atom')(
            self.queue, (round_up(ngrids, TILE), round_up(self.natoms, TILE)), (TILE, TILE),
            self.buf_coords, self.buf_atom_coords, self.buf_rad_node,
            self.buf_radial_l, self.buf_radial_cart0,
            self.buf_atom_radial_offset, self.buf_atom_radial_list,
            self.buf_cart_deriv[0], self.buf_cart_deriv[1], self.buf_cart_deriv[2], self.buf_cart_deriv[3],
            np.float32(self.plan.r0), np.float32(self.plan.du),
            np.int32(self.plan.nrad), np.int32(self.plan.ncart), np.int32(ngrids), np.int32(self.natoms), np.int32(self.plan.spline_order_code)
        )
        return self.buf_cart_deriv, ngrids

    def eval_cart_deriv1(self, coords):
        _, ngrids = self.eval_cart_deriv1_buf(coords)
        out = np.empty((4, ngrids, self.plan.ncart), dtype=np.float32)
        for c in range(4):
            cl.enqueue_copy(self.queue, out[c], self.buf_cart_deriv[c]).wait()
        return out

    def eval_sph_deriv1_buf(self, coords):
        _, ngrids = self.eval_cart_deriv1_buf(coords)
        for c in range(4):
            matmul_gpu_buf(self.buf_cart_deriv[c], self.buf_c2s, self.buf_sph_deriv[c], ngrids, self.mol.nao_nr(), self.plan.ncart)
        return self.buf_sph_deriv, ngrids

    def eval_sph_deriv1(self, coords):
        _, ngrids = self.eval_sph_deriv1_buf(coords)
        out = np.empty((4, ngrids, self.mol.nao_nr()), dtype=np.float32)
        for c in range(4):
            cl.enqueue_copy(self.queue, out[c], self.buf_sph_deriv[c]).wait()
        return out

    def _tiled_global(self, ngrids):
        tc = get_active_tile_config()
        return (round_up(ngrids, tc.NPTILE),), (tc.NPTILE,)

    def eval_cart_deriv1_tiled_buf(self, coords):
        '''Tiled Hermite AO projection (GGA cart, 4 components). Stays on GPU.'''
        coords = np.ascontiguousarray(coords, dtype=np.float32)
        ngrids = coords.shape[0]
        coords4 = np.zeros((ngrids, 4), dtype=np.float32)
        coords4[:, :3] = coords
        self._ensure_grid_buffers(ngrids)
        cl.enqueue_copy(self.queue, self.buf_coords, coords4).wait()
        g, l = self._tiled_global(ngrids)
        _knl(self.prg, 'eval_ao_hermite_cart_deriv1_tiled')(
            self.queue, g, l,
            self.buf_coords, self.buf_atom_coords, self.buf_rad_node,
            self.buf_radial_l, self.buf_radial_cart0,
            self.buf_atom_radial_offset, self.buf_atom_radial_list,
            self.buf_cart_deriv[0], self.buf_cart_deriv[1], self.buf_cart_deriv[2], self.buf_cart_deriv[3],
            np.float32(self.plan.r0), np.float32(self.plan.du),
            np.int32(self.plan.nrad), np.int32(self.plan.ncart), np.int32(ngrids), np.int32(self.natoms), np.int32(self.plan.spline_order_code)
        )
        return self.buf_cart_deriv, ngrids

    def eval_sph_deriv1_tiled_buf(self, coords):
        '''Tiled Hermite AO -> spherical [ngrids,nao] per component on GPU.'''
        cart_bufs, ngrids = self.eval_cart_deriv1_tiled_buf(coords)
        nao = self.mol.nao_nr()
        for c in range(4):
            matmul_gpu_buf(cart_bufs[c], self.buf_c2s, self.buf_sph_deriv[c], ngrids, nao, self.plan.ncart)
        return self.buf_sph_deriv, ngrids

    def project_sph_deriv1_to_bufs(self, coords, buf_ao_row, buf_chi=None):
        '''Full-grid Hermite AO setup: tiled cart eval + c2s + optional chi transpose.

        buf_ao_row: list of 4 cl.Buffer, each ngrids*nao f32 row-major [g,iAO].
        buf_chi: optional list of 4 cl.Buffer, each nao*ngrids f32 [iAO,g].
        '''
        cart_bufs, ngrids = self.eval_cart_deriv1_tiled_buf(coords)
        nao = self.mol.nao_nr()
        queue = self.queue
        for c in range(4):
            matmul_gpu_buf(cart_bufs[c], self.buf_c2s, buf_ao_row[c], ngrids, nao, self.plan.ncart)
            if buf_chi is not None:
                _knl(self.prg, 'transpose_ao_to_chi')(
                    queue, (round_up(ngrids, TILE), round_up(nao, TILE)), (TILE, TILE),
                    buf_ao_row[c], buf_chi[c], np.int32(nao), np.int32(ngrids)
                )
        queue.finish()
        return ngrids

    def build_radial_on_grid_gpu(self, buf_coords4, buf_rad_val, buf_rad_dr, ngrids):
        '''Fill rad_val[ir*ngrids+g], rad_dr[ir*ngrids+g] on GPU (no CPU staging).'''
        tc = get_active_tile_config()
        nradial = self.plan.nradial
        _knl(self.prg, 'build_radial_on_grid_tiled')(
            self.queue, (round_up(ngrids, tc.NPTILE), nradial), (tc.NPTILE, 1),
            buf_coords4, self.buf_atom_coords, self.buf_radial_atom, self.buf_rad_node,
            buf_rad_val, buf_rad_dr,
            np.float32(self.plan.r0), np.float32(self.plan.du),
            np.int32(self.plan.nrad), np.int32(nradial), np.int32(ngrids), np.int32(self.plan.spline_order_code),
        )
        self.queue.finish()

    def __del__(self):
        for name in ('buf_rad_node', 'buf_atom_coords', 'buf_radial_l', 'buf_radial_cart0', 'buf_radial_atom', 'buf_atom_radial_offset', 'buf_atom_radial_list', 'buf_c2s', 'buf_coords', 'buf_cart', 'buf_sph'):
            buf = getattr(self, name, None)
            if buf is not None:
                try:
                    buf.release()
                except Exception:
                    pass
        for name in ('buf_cart_deriv', 'buf_sph_deriv'):
            for buf in getattr(self, name, []):
                if buf is not None:
                    try:
                        buf.release()
                    except Exception:
                        pass
