'''Profiling and impact/effort priority plan for smallDFT.'''
import time
import cProfile
import pstats
import io

import numpy
from pyscf import gto, dft, lib

from pyscf.smallDFT.nr_rks import nr_rks as small_nr_rks
from .layout import eval_ao_native
from .rho import rho_lda, rho_gga
from .vmat import vmat_lda, vmat_gga

# Measured on 6-31g grid level 3, OMP=1, nr_rks per call (Jul 2026).
# Fractions are approximate shares of nr_rks wall time.
PRIORITY_PLAN = '''
Impact / effort priority (H2O + benzene profiling)

| P | Change | Effort | Impact | Notes |
|---|--------|--------|--------|-------|
| 1 | Grid-tile ρ: each worker owns [g0,g1), writes rho[g0:g1] | S (done) | **ρ only 1.7×** (132→76ms benzene @4w) | embarrassingly parallel, no atomics |
| 2 | Native χ[g0:g1,:] contiguous (PySCF layout, no transpose) | S (done) | avoids copy | F-contiguous (ngrids,nao) |
| 3 | lib.num_threads(1) + grid n_workers | S (done) | required | nested OMP+threads was killing scaling |
| 4 | Persistent ThreadPool, one tile per worker | S (done) | reduces overhead | tile_size = ngrids // n_workers |
| 5 | Full nr_rks faster | — | limited | eval_gto ~50%; use precompute_ao + nw=4 |
| 6 | eval_gto grid-parallel | M–L | biggest remaining win | C/Hermite gTile |
| 7 | Fuse ρ+vmat single χ pass | M | saves one χ read | after P1–4 plateau |
| 8 | C/OpenMP kernels (port OpenCL gTile) | L | best CPU ceiling | smallDFT/small_grid.c later |

Bottleneck summary (nr_rks, 1 thread):
  H2O:     eval_gto ~43%, libxc ~31%, dgemm ~13%
  benzene: eval_gto ~50%, dgemm ~34%, libxc ~6%
'''


MOLS = {
    'H2O': {
        'atom': 'O 0 0 0; H 0 0 0.96; H 0 0.96 0',
        'basis': '6-31g',
    },
    'benzene': {
        'atom': '''
C 0 1.396 0; C 1.209 0.698 0; C 1.209 -0.698 0; C 0 -1.396 0; C -1.209 -0.698 0; C -1.209 0.698 0
H 0 2.479 0; H 2.146 1.239 0; H 2.146 -1.239 0; H 0 -2.479 0; H -2.146 -1.239 0; H -2.146 1.239 0
''',
        'basis': '6-31g',
    },
}


def _build(name, xc='PBE', grids_level=3):
    mol = gto.M(atom=MOLS[name]['atom'], basis=MOLS[name]['basis'])
    mf = dft.RKS(mol, xc=xc)
    mf.grids.level = grids_level
    mf.grids.build()
    mf.kernel()
    return mol, mf, mf.make_rdm1()


def _timeit(fn, repeat=5):
    for _ in range(2):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def profile_xc_bottleneck(name='benzene', nthreads=8, xc='PBE'):
    '''Timed breakdown of smallDFT C path (AO cached).'''
    import time
    mol, mf, dm = _build(name, xc=xc)
    ni = dft.numint.NumInt()
    from .workspace import GridWorkspace
    from .rho import rho_gga
    from .vmat import vmat_gga
    ws = GridWorkspace(mol, mf.grids, deriv=1)
    ws.eval_ao(mol, mf.grids)
    chi = ws.chi
    weight = mf.grids.weights
    lib.num_threads(nthreads)

    def _ms(fn):
        for _ in range(2):
            fn()
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1000

    rho = rho_gga(dm, chi, use_c=True, nthreads=nthreads)
    t_rho = _ms(lambda: rho_gga(dm, chi, use_c=True, nthreads=nthreads))
    t_xc = _ms(lambda: ni.eval_xc_eff(xc, rho, deriv=1, xctype='GGA', spin=0))
    exc, vxc = ni.eval_xc_eff(xc, rho, deriv=1, xctype='GGA', spin=0)[:2]
    wv = numpy.asarray(weight * vxc, order='C').copy()
    wv[0] *= .5
    t_vmat = _ms(lambda: vmat_gga(chi, wv, use_c=True, nthreads=nthreads))
    t_ao = _ms(lambda: eval_ao_native(mol, mf.grids.coords, deriv=1,
                                        non0tab=mf.grids.non0tab, cutoff=mf.grids.cutoff))
    tot = t_rho + t_xc + t_vmat
    print(f'{name} XC bottleneck @ {nthreads} threads (ms, AO cached):')
    for label, t in [('rho_gga', t_rho), ('libxc', t_xc), ('vmat_gga', t_vmat)]:
        print(f'  {label:10s} {t:7.1f}  ({100*t/tot:4.1f}% of XC)')
    print(f'  {"XC total":10s} {tot:7.1f}')
    print(f'  {"eval_ao":10s} {t_ao:7.1f}  (per geometry, outside XC timer)')
    return dict(rho=t_rho, libxc=t_xc, vmat=t_vmat, xc=tot, eval_ao=t_ao)


def profile_nr_rks_breakdown(name='H2O', nrepeat=3):
    '''cProfile of reference numint.nr_rks.'''
    mol, mf, dm = _build(name)
    ni = dft.numint.NumInt()
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(nrepeat):
        ni.nr_rks(mol, mf.grids, 'PBE', dm)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(20)
    print(f'=== {name} nao={mol.nao_nr()} ngrids={mf.grids.coords.shape[0]} ref nr_rks ===')
    print(s.getvalue())
    return s.getvalue()


def profile_compare(name='H2O', n_workers_list=(1, 4), tile_sizes=(0, 4096, 8192)):
    '''Compare reference vs smallDFT; return timing dict.'''
    mol, mf, dm = _build(name)
    ni = dft.numint.NumInt()
    results = {}

    results['ref_numint'] = _timeit(lambda: ni.nr_rks(mol, mf.grids, 'PBE', dm))

    for nw in n_workers_list:
        for ts in tile_sizes:
            label = f'smallDFT_w{nw}_t{ts}'
            results[label] = _timeit(
                lambda nw=nw, ts=ts: small_nr_rks(
                    ni, mol, mf.grids, 'PBE', dm, n_workers=nw, tile_size=ts))

    # component breakdown (smallDFT path, single-thread)
    chi = eval_ao_native(mol, mf.grids.coords, deriv=1,
                         non0tab=mf.grids.non0tab, cutoff=mf.grids.cutoff)
    dm = numpy.asarray(dm, order='C')
    results['comp_eval_ao'] = _timeit(
        lambda: eval_ao_native(mol, mf.grids.coords, deriv=1,
                               non0tab=mf.grids.non0tab, cutoff=mf.grids.cutoff))
    results['comp_rho_w1'] = _timeit(lambda: rho_gga(dm, chi, n_workers=1))
    results['comp_rho_w4'] = _timeit(lambda: rho_gga(dm, chi, n_workers=4))
    exc, vxc = ni.eval_xc_eff('PBE', rho_gga(dm, chi), deriv=1, xctype='GGA', spin=0)[:2]
    wv = mf.grids.weights * vxc
    wv = wv.copy(); wv[0] *= .5
    results['comp_vmat'] = _timeit(lambda: vmat_gga(chi, wv, tile_size=0, n_workers=1))
    results['comp_libxc'] = _timeit(
        lambda: ni.eval_xc_eff('PBE', rho_gga(dm, chi), deriv=1, xctype='GGA', spin=0))

    print(f'\n=== profile_compare {name} nao={mol.nao_nr()} ngrids={mf.grids.coords.shape[0]} ===')
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        print(f'  {k:28s} {v*1000:8.2f} ms')

    # parity
    n0, e0, v0 = ni.nr_rks(mol, mf.grids, 'PBE', dm)
    n1, e1, v1 = small_nr_rks(ni, mol, mf.grids, 'PBE', dm)
    print(f'  parity nelec diff {abs(n0-n1):.3e}  exc diff {abs(e0-e1):.3e}  vmat max {abs(v0-v1).max():.3e}')
    return results
