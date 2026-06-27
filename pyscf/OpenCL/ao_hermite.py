import numpy as np
import pyopencl as cl

from . import get_ctx, get_queue, get_prg, round_up
from .radial_hermite import MappedHermiteRadialBasis
from .xc_grid import _knl, matmul_gpu_buf


TILE = 16


class OpenCLAOHermiteEvaluator:
    def __init__(self, mol, r0_ang=0.01, du=0.02, rmax_ang=8.0, midpoint_fit=True):
        self.mol = mol
        self.plan = MappedHermiteRadialBasis(mol, r0_ang=r0_ang, du=du, rmax_ang=rmax_ang, midpoint_fit=midpoint_fit)
        self.ctx = get_ctx()
        self.queue = get_queue()
        self.prg = get_prg()
        mf = cl.mem_flags
        self.rad_val = np.ascontiguousarray(self.plan.values, dtype=np.float32)
        self.rad_du = np.ascontiguousarray(self.plan.du_values, dtype=np.float32)
        self.atom_coords = np.ascontiguousarray(self.plan.atom_coords, dtype=np.float32)
        self.cart_shell = np.ascontiguousarray(self.plan.cart_shell, dtype=np.int32)
        self.cart_ctr = np.ascontiguousarray(self.plan.cart_ctr, dtype=np.int32)
        self.cart_ixyz = np.ascontiguousarray(self.plan.cart_ixyz, dtype=np.int32)
        self.shell_atom = np.ascontiguousarray(self.plan.shell_atom, dtype=np.int32)
        self.buf_rad_val = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, self.rad_val.nbytes, self.rad_val)
        self.buf_rad_du = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, self.rad_du.nbytes, self.rad_du)
        self.buf_atom_coords = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, self.atom_coords.nbytes, self.atom_coords)
        self.buf_cart_shell = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, self.cart_shell.nbytes, self.cart_shell)
        self.buf_cart_ctr = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, self.cart_ctr.nbytes, self.cart_ctr)
        self.buf_cart_ixyz = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, self.cart_ixyz.nbytes, self.cart_ixyz)
        self.buf_shell_atom = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, self.shell_atom.nbytes, self.shell_atom)
        self.c2s = np.ascontiguousarray(mol.cart2sph_coeff(normalized='sp'), dtype=np.float32)
        self.buf_c2s = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, self.c2s.nbytes, self.c2s)
        self.ngrid_alloc = 0
        self.buf_coords = None
        self.buf_cart = None
        self.buf_sph = None

    def _ensure_grid_buffers(self, ngrids):
        if ngrids <= self.ngrid_alloc:
            return
        mf = cl.mem_flags
        for name in ('buf_coords', 'buf_cart', 'buf_sph'):
            buf = getattr(self, name)
            if buf is not None:
                buf.release()
        self.ngrid_alloc = int(ngrids)
        self.buf_coords = cl.Buffer(self.ctx, mf.READ_ONLY, self.ngrid_alloc * 3 * np.dtype(np.float32).itemsize)
        self.buf_cart = cl.Buffer(self.ctx, mf.READ_WRITE, self.ngrid_alloc * self.plan.ncart * np.dtype(np.float32).itemsize)
        self.buf_sph = cl.Buffer(self.ctx, mf.READ_WRITE, self.ngrid_alloc * self.mol.nao_nr() * np.dtype(np.float32).itemsize)

    def eval_cart_buf(self, coords):
        coords = np.ascontiguousarray(coords, dtype=np.float32)
        ngrids = coords.shape[0]
        self._ensure_grid_buffers(ngrids)
        cl.enqueue_copy(self.queue, self.buf_coords, coords).wait()
        _knl(self.prg, 'eval_ao_mapped_hermite_cart')(
            self.queue, (round_up(ngrids, TILE), round_up(self.plan.ncart, TILE)), (TILE, TILE),
            self.buf_coords, self.buf_atom_coords, self.buf_rad_val, self.buf_rad_du,
            self.buf_cart_shell, self.buf_cart_ctr, self.buf_cart_ixyz, self.buf_shell_atom,
            self.buf_cart,
            np.float32(self.plan.r0), np.float32(self.plan.du),
            np.int32(self.plan.nrad), np.int32(self.rad_val.shape[1]), np.int32(self.plan.ncart), np.int32(ngrids)
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

    def __del__(self):
        for name in ('buf_rad_val', 'buf_rad_du', 'buf_atom_coords', 'buf_cart_shell', 'buf_cart_ctr', 'buf_cart_ixyz', 'buf_shell_atom', 'buf_c2s', 'buf_coords', 'buf_cart', 'buf_sph'):
            buf = getattr(self, name, None)
            if buf is not None:
                try:
                    buf.release()
                except Exception:
                    pass
