#!/usr/bin/env python3
"""Step-by-step parity: full GPU path vs CPU (rho -> wv -> vmat)."""
import os
import sys

import numpy as np
import pyopencl as cl
from pyscf import gto, dft, lib
from pyscf.dft.gen_grid import BLKSIZE
from pyscf.dft.numint import _dot_ao_ao, _scale_ao

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_XYZ = os.path.join(_REPO, 'data', 'xyz', 'benzene.xyz')


def read_xyz(path):
    with open(path) as f:
        lines = f.readlines()
    natom = int(lines[0].strip())
    return '; '.join(' '.join(line.split()[:4]) for line in lines[2:2 + natom] if line.strip())


def cpu_rho(ni, mol, grids, dm):
    dm = np.asarray(dm, order='C', dtype=np.float64)
    make_rho, _, _ = ni._gen_rho_evaluator(mol, dm, hermi=1, with_lapl=False, grids=grids)
    ngrids = grids.coords.shape[0]
    rho = np.zeros((4, ngrids), dtype=np.float64)
    for ip0 in range(0, ngrids, 8192):
        ip1 = min(ip0 + 8192, ngrids)
        ao = ni.eval_ao(mol, grids.coords[ip0:ip1], deriv=1)
        rho[:, ip0:ip1] = make_rho(0, ao, None, 'GGA')
    return rho


def cpu_wv(ni, grids, rho):
    weight = grids.weights
    evfk = ni.eval_xc_eff('PBE', rho, deriv=1, xctype='GGA', spin=0)
    vxc = evfk[1]
    wv = weight[np.newaxis, :].astype(np.float64) * np.ascontiguousarray(vxc, dtype=np.float64)
    wv[0] *= 0.5
    return wv


def cpu_vmat_gga(ni, mol, grids, wv):
    nao = mol.nao_nr()
    ngrids = grids.coords.shape[0]
    vmat = np.zeros((nao, nao), dtype=np.float64)
    for ip0 in range(0, ngrids, 8192):
        ip1 = min(ip0 + 8192, ngrids)
        ao = ni.eval_ao(mol, grids.coords[ip0:ip1], deriv=1, non0tab=grids.non0tab[ip0 // BLKSIZE:])
        wva = wv[:, ip0:ip1].astype(np.float64)
        aow = _scale_ao(ao[:4], wva[:4])
        vmat += _dot_ao_ao(mol, ao[0], aow, grids.non0tab[ip0 // BLKSIZE:], (0, mol.nbas), mol.ao_loc_nr())
    return vmat + vmat.T


def err_stats(a, b, label, per_comp=False, comp_names=None):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    amax = float(np.abs(d).max())
    ref = max(float(np.abs(b).max()), 1e-10)
    print(f'  {label:28s} max_abs={amax:.3e}  rel={amax/ref:.3e}', flush=True)
    if per_comp and d.ndim == 2 and d.shape[0] == 4:
        names = comp_names or ('c0', 'c1', 'c2', 'c3')
        for c, n in enumerate(names):
            mc = float(np.abs(d[c]).max())
            rc = max(float(np.abs(b[c]).max()), 1e-10)
            print(f'    {n}: max_abs={mc:.3e}  rel={mc/rc:.3e}', flush=True)
    return amax, amax / ref


def gpu_rho_wv(plan, dm):
    pcg = plan.pcg
    ngrids = plan.ngrids
    dm32 = np.ascontiguousarray(dm, dtype=np.float32)
    cl.enqueue_copy(plan.queue, plan.bufDm, dm32)
    if pcg.get('radial_precomp'):
        np.matmul(pcg['c2s'], dm32, out=pcg['dm_tmp'])
        np.matmul(pcg['dm_tmp'], pcg['c2s'].T, out=pcg['dm_cart32'])
        cl.enqueue_copy(plan.queue, pcg['buf_dm_cart'], pcg['dm_cart32'])
    plan._precomp_rho_fused(pcg, 'GGA', plan.nao, ngrids)
    rho = np.empty(4 * ngrids, dtype=np.float32)
    cl.enqueue_copy(plan.queue, rho, pcg['buf_rho']).wait()
    rho = rho.reshape(4, ngrids).astype(np.float64)
    st = {**pcg, 'rho_host': pcg['rho32_host'], 'wv_host': pcg['wv32_host'],
          'weight32': pcg['weight'], 'weight64': pcg['weight64']}
    plan._xc_pbe_gpu(st, ngrids)
    wv = np.empty(4 * ngrids, dtype=np.float32)
    cl.enqueue_copy(plan.queue, wv, pcg['buf_wv']).wait()
    wv = wv.reshape(4, ngrids).astype(np.float64)
    return rho, wv


def gpu_vmat(plan, wv32):
    return plan.nr_rks_precomputed_vmat_only(wv32.astype(np.float32))


def upload_rho_gpu_pbe_wv(plan, rho64):
    pcg = plan.pcg
    ngrids = plan.ngrids
    rho32 = np.ascontiguousarray(rho64, dtype=np.float32).reshape(-1)
    cl.enqueue_copy(plan.queue, pcg['buf_rho'], rho32).wait()
    st = {**pcg, 'rho_host': pcg['rho32_host'], 'wv_host': pcg['wv32_host'],
          'weight32': pcg['weight'], 'weight64': pcg['weight64']}
    plan._xc_pbe_gpu(st, ngrids)
    wv = np.empty(4 * ngrids, dtype=np.float32)
    cl.enqueue_copy(plan.queue, wv, pcg['buf_wv']).wait()
    return wv.reshape(4, ngrids).astype(np.float64)


def audit_path(plan, dm, rho_cpu, wv_cpu, vmat_cpu, vxc_cpu, label):
    print(f'\n=== {label} ===', flush=True)
    rho_gpu, wv_gpu_rho = gpu_rho_wv(plan, dm)
    err_stats(rho_gpu, rho_cpu, '1 rho GPU vs CPU GTO',
              per_comp=True, comp_names=('rho0', 'grad_x', 'grad_y', 'grad_z'))
    wv_gpu_cpu_rho = upload_rho_gpu_pbe_wv(plan, rho_cpu)
    err_stats(wv_gpu_cpu_rho, wv_cpu, '2a wv GPU-PBE vs CPU-libxc (CPU rho)',
              per_comp=True, comp_names=('wv0', 'wv1', 'wv2', 'wv3'))
    err_stats(wv_gpu_rho, wv_cpu, '2b wv GPU-PBE vs CPU-libxc (GPU rho)',
              per_comp=True, comp_names=('wv0', 'wv1', 'wv2', 'wv3'))
    wv_cpu32 = np.ascontiguousarray(wv_cpu, dtype=np.float32)
    vmat_gpu_cpu_wv = gpu_vmat(plan, wv_cpu32)
    err_stats(vmat_gpu_cpu_wv, vmat_cpu, '3a vmat GPU vs CPU (CPU wv)')
    wv_gpu_rho32 = np.ascontiguousarray(wv_gpu_rho, dtype=np.float32)
    vmat_gpu = gpu_vmat(plan, wv_gpu_rho32)
    err_stats(vmat_gpu, vmat_cpu, '3b vmat GPU vs CPU (GPU wv chain)')
    n, exc, vxc_full = plan.nr_rks_precomputed_gto(dm, projection='gpu')
    err_stats(vxc_full, vxc_cpu, '4 full vxc GPU path vs CPU')
    print(f'  nelec gpu={n:.6f}  exc gpu={exc:.8f}', flush=True)


def main():
    from pyscf.OpenCL.xc_grid import get_xc_grid_plan, clear_xc_plan_cache
    from pyscf.OpenCL import init_device, reset_opencl

    mol = gto.M(atom=read_xyz(_XYZ), basis='ccpvdz', verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build(with_non0tab=True)
    ni = dft.numint.NumInt()
    dm = dft.RKS(mol, xc='PBE').density_fit().get_init_guess()
    print(f'benzene nao={mol.nao_nr()} ngrids={grids.coords.shape[0]}', flush=True)

    n_cpu, exc_cpu, vxc_cpu = ni.nr_rks(mol, grids, 'PBE', dm, max_memory=2000)
    rho_cpu = cpu_rho(ni, mol, grids, dm)
    wv_cpu = cpu_wv(ni, grids, rho_cpu)
    vmat_cpu = cpu_vmat_gga(ni, mol, grids, wv_cpu)
    print(f'CPU ref nelec={n_cpu:.6f} exc={exc_cpu:.8f} vxc_max={np.abs(vxc_cpu).max():.6f}', flush=True)

    configs = (
        ('coalesced + Hermite AO + GPU XC', 'coalesced', 'hermite_gpu'),
        ('coalesced + GTO AO + GPU XC', 'coalesced', 'cpu'),
        ('radial + GPU XC', 'radial_precomp', 'hermite_gpu'),
        ('tiled + GTO AO + GPU XC', 'tiled', 'cpu'),
    )
    for label, fused, ao_proj in configs:
        clear_xc_plan_cache()
        reset_opencl()
        init_device(quiet=True)
        plan = get_xc_grid_plan(mol, grids, 'PBE')
        plan.setup_precomputed_gto(gpu_only=True, fused=fused, xc_eval='gpu', ao_proj=ao_proj)
        audit_path(plan, dm, rho_cpu, wv_cpu, vmat_cpu, vxc_cpu, label)
    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
