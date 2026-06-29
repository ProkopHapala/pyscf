'''OpenCL f32 Hermite radial spline evaluation (study / parity vs hermite_spline.py f64).'''
from __future__ import annotations

import os

import numpy as np
import pyopencl as cl

from pyscf.OpenCL import get_ctx, get_queue, init_device
from pyscf.OpenCL.hermite_spline import cubic_eval, cubic_eval_du, quintic_eval, quintic_eval_du

_CL_FILE = os.path.join(os.path.dirname(__file__), 'hermite_spline_f32.cl')
_prg = None


def _get_prg():
    global _prg
    if _prg is None:
        init_device(quiet=True)
        with open(_CL_FILE) as f:
            _prg = cl.Program(get_ctx(), f.read()).build()
    return _prg


def tables_to_f32(tab):
    '''Cast knot tables to float32 (values built in f64, stored f32 like GPU).'''
    return dict(tab, u=tab['u'].astype(np.float32), r=tab['r'].astype(np.float32), y=tab['y'].astype(np.float32), d=tab['d'].astype(np.float32), c=tab['c'].astype(np.float32), r0=np.float32(tab['r0']), map_b=np.float32(tab.get('map_b', 1.0)))


def _order_code(order):
    return 0 if order == 'cubic' else 1


def _space_code(interp_space):
    return 0 if interp_space == 'u' else 1


def _seg_lower(xq, x_grid):
    ik = int(np.searchsorted(x_grid.astype(np.float64), float(xq), side='right') - 1)
    return max(0, min(ik, x_grid.size - 2))


def _eval_channel_f32(rq, tab_f32, order, interp_space, ic):
    r0, map_b = tab_f32['r0'], tab_f32.get('map_b', np.float32(1.0))
    x_grid = tab_f32['u'] if interp_space == 'u' else tab_f32['r']
    y = tab_f32['y'][:, ic]
    d = tab_f32['d'][:, ic]
    c = tab_f32['c'][:, ic]
    xq = map_b * np.log1p(rq / r0) if interp_space == 'u' else rq
    ik = _seg_lower(xq, x_grid)
    h = np.float32(x_grid[ik + 1] - x_grid[ik])
    t = np.float32(np.clip((float(xq) - float(x_grid[ik])) / float(h), 0.0, 1.0))
    y0, y1, d0, d1 = y[ik], y[ik + 1], d[ik], d[ik + 1]
    if order == 'cubic':
        R = np.float32(cubic_eval(t, h, y0, y1, d0, d1))
        dx = np.float32(cubic_eval_du(t, h, y0, y1, d0, d1))
    else:
        c0, c1 = c[ik], c[ik + 1]
        R = np.float32(quintic_eval(t, h, y0, y1, d0, d1, c0, c1))
        dx = np.float32(quintic_eval_du(t, h, y0, y1, d0, d1, c0, c1))
    dR = dx * map_b / (rq + r0) if interp_space == 'u' else dx
    return R, np.float32(dR)


def eval_radial_spline_f32_cpu(r, tab_f32, order='cubic', interp_space='u', ic=0):
    '''Numpy f32 replay of OpenCL kernel (same formulas, no OpenCL).'''
    r = np.asarray(r, dtype=np.float32)
    R = np.empty(r.size, dtype=np.float32)
    dR = np.empty(r.size, dtype=np.float32)
    for i in range(r.size):
        R[i], dR[i] = _eval_channel_f32(r[i], tab_f32, order, interp_space, ic)
    return R, dR


def eval_radial_spline_cl(r, tab_f32, order='cubic', interp_space='u', ic=0, queue=None):
    '''Evaluate R and dR/dr on GPU in float32. tab_f32 from tables_to_f32().'''
    queue = queue or get_queue()
    prg = _get_prg()
    r = np.ascontiguousarray(r, dtype=np.float32)
    n = r.size
    x_grid = tab_f32['u'] if interp_space == 'u' else tab_f32['r']
    y = np.ascontiguousarray(tab_f32['y'][:, ic], dtype=np.float32)
    d = np.ascontiguousarray(tab_f32['d'][:, ic], dtype=np.float32)
    c = np.ascontiguousarray(tab_f32['c'][:, ic], dtype=np.float32)
    mf = cl.mem_flags
    ctx = queue.context
    bufs = [cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr) for arr in (r, x_grid, y, d, c)]
    out_R = np.empty(n, dtype=np.float32)
    out_dR = np.empty(n, dtype=np.float32)
    bo_R = cl.Buffer(ctx, mf.WRITE_ONLY, out_R.nbytes)
    bo_dR = cl.Buffer(ctx, mf.WRITE_ONLY, out_dR.nbytes)
    knl = prg.eval_radial_spline_f32
    knl(queue, (n,), None, *bufs, bo_R, bo_dR, np.int32(n), np.int32(x_grid.size), np.float32(tab_f32['r0']), np.float32(tab_f32.get('map_b', 1.0)), np.int32(_order_code(order)), np.int32(_space_code(interp_space)))
    cl.enqueue_copy(queue, out_R, bo_R)
    cl.enqueue_copy(queue, out_dR, bo_dR)
    return out_R, out_dR
