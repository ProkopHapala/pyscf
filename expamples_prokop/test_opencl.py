#!/usr/bin/env python3
"""Test OpenCL GPU vs CPU for DFT XC integration and DF J/K contraction.

Usage:
    PYTHONPATH=/home/prokophapala/git/pyscf python3 expamples_prokop/test_opencl.py
"""
import sys
import time
import numpy as np
from pyscf import gto, dft

def test_xc_grid():
    """Test XC grid integration: CPU vs GPU."""
    print("=" * 60)
    print("Test 1: XC Grid Integration (PBE, GGA)")
    print("=" * 60)

    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz', verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build()

    nao = mol.nao_nr()
    np.random.seed(42)
    dm = np.random.rand(nao, nao)
    dm = 0.5 * (dm + dm.T)  # symmetrize

    # CPU reference
    print("\nRunning CPU...")
    ni = dft.numint.NumInt()
    t0 = time.perf_counter()
    n_cpu, exc_cpu, vxc_cpu = ni.nr_rks(mol, grids, 'pbe', dm, max_memory=2000)
    t_cpu = time.perf_counter() - t0
    print(f"  CPU time: {t_cpu:.3f}s")
    print(f"  nelec={n_cpu:.6f}, excsum={exc_cpu:.6f}")
    print(f"  vxc shape={vxc_cpu.shape}, max={np.abs(vxc_cpu).max():.6f}")

    # GPU
    print("\nRunning GPU (OpenCL)...")
    from pyscf.OpenCL.xc_grid import setup_xc_grid_gpu, nr_rks_gpu
    setup_xc_grid_gpu(mol, grids, 'pbe')
    t0 = time.perf_counter()
    n_gpu, exc_gpu, vxc_gpu = nr_rks_gpu(mol, grids, 'pbe', dm, max_memory=2000)
    t_gpu = time.perf_counter() - t0
    print(f"  GPU time: {t_gpu:.3f}s")
    print(f"  nelec={n_gpu:.6f}, excsum={exc_gpu:.6f}")
    print(f"  vxc shape={vxc_gpu.shape}, max={np.abs(vxc_gpu).max():.6f}")

    # Comparison
    print("\n--- Comparison ---")
    err_n = abs(n_cpu - n_gpu) / max(abs(n_cpu), 1e-10)
    err_exc = abs(exc_cpu - exc_gpu) / max(abs(exc_cpu), 1e-10)
    err_vxc = np.abs(vxc_cpu - vxc_gpu).max()
    err_vxc_rel = err_vxc / max(np.abs(vxc_cpu).max(), 1e-10)
    print(f"  nelec relative error:  {err_n:.2e}")
    print(f"  excsum relative error: {err_exc:.2e}")
    print(f"  vxc max abs error:     {err_vxc:.2e}")
    print(f"  vxc max relative error:{err_vxc_rel:.2e}")
    print(f"  Speedup: {t_cpu/t_gpu:.2f}x")

    return err_vxc

def test_df_jk():
    """Test DF J/K contraction: CPU vs GPU."""
    print("\n" + "=" * 60)
    print("Test 2: DF J/K Contraction")
    print("=" * 60)

    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz', verbose=0)
    nao = mol.nao_nr()
    np.random.seed(42)
    dm = np.random.rand(nao, nao)
    dm = 0.5 * (dm + dm.T)

    from pyscf import df as df_mod
    dfobj = df_mod.DF(mol)
    dfobj.verbose = 0

    # CPU
    print("\nBuilding DF tensor (CPU)...")
    dfobj.build()
    print(f"  cderi shape: {dfobj._cderi.shape if hasattr(dfobj._cderi, 'shape') else 'HDF5'}")

    print("\nRunning CPU DF J/K...")
    from pyscf.df import df_jk
    t0 = time.perf_counter()
    vj_cpu, vk_cpu = df_jk._get_jk_cpu(dfobj, dm, hermi=1, with_j=True, with_k=True)
    t_cpu = time.perf_counter() - t0
    print(f"  CPU time: {t_cpu:.3f}s")
    print(f"  vj max={np.abs(vj_cpu).max():.6f}, vk max={np.abs(vk_cpu).max():.6f}")

    # GPU
    print("\nRunning GPU DF J/K (OpenCL)...")
    from pyscf.OpenCL.df_jk import df_jk_gpu
    t0 = time.perf_counter()
    vj_gpu, vk_gpu = df_jk_gpu(dfobj, dm, hermi=1, with_j=True, with_k=True)
    t_gpu = time.perf_counter() - t0
    print(f"  GPU time: {t_gpu:.3f}s")
    print(f"  vj max={np.abs(vj_gpu).max():.6f}, vk max={np.abs(vk_gpu).max():.6f}")

    # Comparison
    print("\n--- Comparison ---")
    err_j = np.abs(vj_cpu - vj_gpu).max()
    err_k = np.abs(vk_cpu - vk_gpu).max()
    err_j_rel = err_j / max(np.abs(vj_cpu).max(), 1e-10)
    err_k_rel = err_k / max(np.abs(vk_cpu).max(), 1e-10)
    print(f"  J max abs error:      {err_j:.2e}")
    print(f"  J max relative error: {err_j_rel:.2e}")
    print(f"  K max abs error:      {err_k:.2e}")
    print(f"  K max relative error: {err_k_rel:.2e}")
    print(f"  Speedup: {t_cpu/t_gpu:.2f}x")

    return err_j, err_k

def test_full_dft():
    """Test full DFT single-point with backend=3 (both CPU+GPU)."""
    print("\n" + "=" * 60)
    print("Test 3: Full DFT single-point (backend=3, both CPU+GPU)")
    print("=" * 60)

    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz', verbose=4)
    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.grids.level = 3
    mf.backend = 3  # both CPU and GPU for XC
    mf.density_fit()
    # Set DF backend after density_fit creates the with_df object
    # Use __dict__ to avoid triggering __getattr__
    if 'with_df' in mf.__dict__ and mf.__dict__['with_df'] is not None:
        mf.__dict__['with_df'].backend = 3
    mf.max_cycle = 1

    print("\nRunning single SCF cycle with backend=3...")
    e = mf.kernel()
    print(f"\nTotal energy: {e:.6f}")

if __name__ == '__main__':
    err_vxc = test_xc_grid()
    err_j, err_k = test_df_jk()
    test_full_dft()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Vxc max abs error: {err_vxc:.2e}")
    print(f"  J max abs error:   {err_j:.2e}")
    print(f"  K max abs error:   {err_k:.2e}")
    print("\n  Expected: ~1e-5 to 1e-6 (float32 precision)")
    print("  If errors are ~1e-6, GPU implementation is correct.")
