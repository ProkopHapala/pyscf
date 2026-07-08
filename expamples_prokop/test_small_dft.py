#!/usr/bin/env python
'''Parity and grid-parallel scaling tests for pyscf.smallDFT.'''
import sys
import time

from pyscf import gto, dft, lib
from pyscf.smallDFT import nr_rks, profile_compare, shutdown_pool
from pyscf.dft import numint

MOL = 'O 0 0 0; H 0 0 0.96; H 0 0.96 0'
BASIS = '6-31g'
BENZENE = '''
C 0 1.396 0; C 1.209 0.698 0; C 1.209 -0.698 0; C 0 -1.396 0; C -1.209 -0.698 0; C -1.209 0.698 0
H 0 2.479 0; H 2.146 1.239 0; H 2.146 -1.239 0; H 0 -2.479 0; H -2.146 -1.239 0; H -2.146 1.239 0
'''


def _tms(fn, repeat=5):
    for _ in range(2):
        fn()
    xs = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t0)
    return min(xs) * 1000


def _parity(mol_atom, label):
    mol = gto.M(atom=mol_atom, basis=BASIS)
    mf = dft.RKS(mol, xc='PBE')
    mf.grids.level = 3
    mf.grids.build()
    mf.kernel()
    dm = mf.make_rdm1()
    ni = numint.NumInt()
    lib.num_threads(1)
    n0, e0, v0 = ni.nr_rks(mol, mf.grids, 'PBE', dm)
    n1, e1, v1 = nr_rks(ni, mol, mf.grids, 'PBE', dm, n_workers=4)
    assert abs(n0 - n1) < 1e-7
    assert abs(e0 - e1) < 1e-7
    assert abs(v0 - v1).max() < 1e-6
    print(f'{label} parity OK  vmat max diff={abs(v0-v1).max():.2e}')


def _bench(mol_atom, label):
    mol = gto.M(atom=mol_atom, basis=BASIS)
    mf = dft.RKS(mol, xc='PBE')
    mf.grids.level = 3
    mf.grids.build()
    mf.kernel()
    dm = mf.make_rdm1()
    ni = numint.NumInt()

    print(f'\n{label} nao={mol.nao_nr()} ngrids={mf.grids.coords.shape[0]}')
    for nt in [1, 4]:
        lib.num_threads(nt)
        print(f'  ref omp={nt:2d}     {_tms(lambda: ni.nr_rks(mol, mf.grids, "PBE", dm)):7.1f} ms')
    lib.num_threads(1)
    for nw in [1, 2, 4, 8]:
        print(f'  small nw={nw:2d}     {_tms(lambda nw=nw: nr_rks(ni, mol, mf.grids, "PBE", dm, n_workers=nw)):7.1f} ms')
    shutdown_pool()


def _bench_rho(mol_atom, label):
    from pyscf.smallDFT.layout import eval_ao_native
    from pyscf.smallDFT.rho import rho_gga

    mol = gto.M(atom=mol_atom, basis=BASIS)
    mf = dft.RKS(mol, xc='PBE')
    mf.grids.level = 3
    mf.grids.build()
    mf.kernel()
    dm = mf.make_rdm1()
    lib.num_threads(1)
    chi = eval_ao_native(mol, mf.grids.coords, deriv=1,
                         non0tab=mf.grids.non0tab, cutoff=mf.grids.cutoff)
    print(f'\n=== rho_gga grid-parallel only: {label} ===')
    for nw in [1, 2, 4, 8]:
        print(f'  n_workers={nw}: {_tms(lambda nw=nw: rho_gga(dm, chi, n_workers=nw)):6.1f} ms')
    lib.num_threads(4)
    print(f'  BLAS omp=4 (1 grid worker): {_tms(lambda: rho_gga(dm, chi, n_workers=1)):6.1f} ms')
    shutdown_pool()


def _parity_lda_c(mol_atom, label):
    from pyscf.smallDFT.rho import rho_lda
    from pyscf.smallDFT.layout import eval_ao_native
    from pyscf.smallDFT._ctypes import has_c_lib

    if not has_c_lib():
        print(f'{label} LDA C parity: skip (libsmalldft not built)')
        return
    mol = gto.M(atom=mol_atom, basis=BASIS)
    mf = dft.RKS(mol, xc='LDA,VWN')
    mf.grids.level = 3
    mf.grids.build()
    mf.kernel()
    dm = mf.make_rdm1()
    chi = eval_ao_native(mol, mf.grids.coords, deriv=0,
                         non0tab=mf.grids.non0tab, cutoff=mf.grids.cutoff)
    r_py = rho_lda(dm, chi, use_c=False, n_workers=1)
    r_c = rho_lda(dm, chi, use_c=True, nthreads=4)
    assert abs(r_py - r_c).max() < 1e-12
    print(f'{label} LDA C parity OK  max diff={abs(r_py-r_c).max():.2e}')


def _bench_rho_lda_c(mol_atom, label):
    from pyscf.smallDFT.layout import eval_ao_native
    from pyscf.smallDFT.rho import rho_lda
    from pyscf.smallDFT._ctypes import has_c_lib

    if not has_c_lib():
        return
    mol = gto.M(atom=mol_atom, basis=BASIS)
    mf = dft.RKS(mol, xc='LDA,VWN')
    mf.grids.level = 3
    mf.grids.build()
    mf.kernel()
    dm = mf.make_rdm1()
    chi = eval_ao_native(mol, mf.grids.coords, deriv=0,
                         non0tab=mf.grids.non0tab, cutoff=mf.grids.cutoff)
    print(f'\n=== rho_lda C OpenMP: {label} ===')
    for nt in [1, 2, 4]:
        print(f'  nthreads={nt}: {_tms(lambda nt=nt: rho_lda(dm, chi, use_c=True, nthreads=nt)):6.1f} ms')
    print(f'  python nw=4:      {_tms(lambda: rho_lda(dm, chi, use_c=False, n_workers=4)):6.1f} ms')


def _parity_gga_c(mol_atom, label):
    from pyscf.smallDFT.rho import rho_gga
    from pyscf.smallDFT.layout import eval_ao_native
    from pyscf.smallDFT._ctypes import has_c_lib

    if not has_c_lib():
        print(f'{label} GGA C parity: skip (libsmalldft not built)')
        return
    mol = gto.M(atom=mol_atom, basis=BASIS)
    mf = dft.RKS(mol, xc='PBE')
    mf.grids.level = 3
    mf.grids.build()
    mf.kernel()
    dm = mf.make_rdm1()
    chi = eval_ao_native(mol, mf.grids.coords, deriv=1,
                         non0tab=mf.grids.non0tab, cutoff=mf.grids.cutoff)
    r_py = rho_gga(dm, chi, use_c=False, n_workers=1)
    r_c = rho_gga(dm, chi, use_c=True, nthreads=4)
    assert abs(r_py - r_c).max() < 1e-12
    print(f'{label} PBE rho_gga C parity OK  max diff={abs(r_py-r_c).max():.2e}')


def _bench_rho_gga_c(mol_atom, label):
    from pyscf.smallDFT.layout import eval_ao_native
    from pyscf.smallDFT.rho import rho_gga
    from pyscf.smallDFT._ctypes import has_c_lib

    if not has_c_lib():
        return
    mol = gto.M(atom=mol_atom, basis=BASIS)
    mf = dft.RKS(mol, xc='PBE')
    mf.grids.level = 3
    mf.grids.build()
    mf.kernel()
    dm = mf.make_rdm1()
    chi = eval_ao_native(mol, mf.grids.coords, deriv=1,
                         non0tab=mf.grids.non0tab, cutoff=mf.grids.cutoff)
    print(f'\n=== rho_gga C OpenMP (PBE): {label} ===')
    for nt in [1, 2, 4]:
        print(f'  nthreads={nt}: {_tms(lambda nt=nt: rho_gga(dm, chi, use_c=True, nthreads=nt)):6.1f} ms')
    print(f'  python nw=4:      {_tms(lambda: rho_gga(dm, chi, use_c=False, n_workers=4)):6.1f} ms')


if __name__ == '__main__':
    _parity(MOL, 'H2O')
    _parity(BENZENE, 'benzene')
    _parity_lda_c(MOL, 'H2O')
    _parity_lda_c(BENZENE, 'benzene')
    _parity_gga_c(MOL, 'H2O')
    _parity_gga_c(BENZENE, 'benzene')
    _bench(MOL, 'H2O')
    _bench(BENZENE, 'benzene')
    if '--rho' in sys.argv:
        _bench_rho(MOL, 'H2O')
        _bench_rho(BENZENE, 'benzene')
        _bench_rho_lda_c(MOL, 'H2O')
        _bench_rho_lda_c(BENZENE, 'benzene')
        _bench_rho_gga_c(MOL, 'H2O')
        _bench_rho_gga_c(BENZENE, 'benzene')
    if '--profile' in sys.argv:
        profile_compare('H2O')
        profile_compare('benzene')
