#!/usr/bin/env python
"""
Profile DFT (PBE) — single SCF cycle timing + cProfile + density fitting option.

Usage:
  PYTHONPATH=/home/prokophapala/git/pyscf OMP_NUM_THREADS=1 python3 expamples_prokop/profile_dft.py --mols H2O benzene
  PYTHONPATH=/home/prokophapala/git/pyscf OMP_NUM_THREADS=1 python3 expamples_prokop/profile_dft.py --mols PTCDA --df
  PYTHONPATH=/home/prokophapala/git/pyscf OMP_NUM_THREADS=1 python3 expamples_prokop/profile_dft.py --mols benzene --profile
"""

import sys
import time
import os

# Ensure we use the repo version, not pip-installed
# (PYTHONPATH should be set, but let's verify)
import pyscf
print(f"PySCF path: {pyscf.__file__}")
print(f"PySCF version: {pyscf.__version__}")

from pyscf import gto, dft, lib

# ============================================================
# Molecule definitions
# ============================================================

def read_xyz(path):
    """Read xyz file, extract element + x,y,z only (ignore charges, lattice vectors)."""
    import re
    with open(path) as f:
        lines = f.readlines()
    natom = int(lines[0].strip())
    atoms = []
    for line in lines[2:2+natom]:
        parts = line.split()
        el = parts[0]
        if not re.match(r'^[A-Z][a-z]?$', el):
            continue
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append(f"{el} {x:.6f} {y:.6f} {z:.6f}")
    return '\n'.join(atoms)

_xyz_dir = '/home/prokophapala/git/pyscf/data/xyz'

molecules = {
    'H2O': {
        'atom': '''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161''',
        'basis': '6-31g',
    },
    'benzene': {
        'atom': '''
C        0.000000    1.396000    0.000000
C        1.209000    0.698000    0.000000
C        1.209000   -0.698000    0.000000
C        0.000000   -1.396000    0.000000
C       -1.209000   -0.698000    0.000000
C       -1.209000    0.698000    0.000000
H        0.000000    2.479000    0.000000
H        2.146000    1.239000    0.000000
H        2.146000   -1.239000    0.000000
H        0.000000   -2.479000    0.000000
H       -2.146000   -1.239000    0.000000
H       -2.146000    1.239000    0.000000''',
        'basis': '6-31g',
    },
    'pentacene': {'atom': read_xyz(f'{_xyz_dir}/pentacene.xyz'), 'basis': '6-31g'},
    'porphirin': {'atom': read_xyz(f'{_xyz_dir}/porphirin.xyz'), 'basis': '6-31g'},
    'PTCDA': {'atom': read_xyz(f'{_xyz_dir}/PTCDA.xyz'), 'basis': '6-31g'},
}

# ============================================================
# Timing decorator for key DFT functions
# ============================================================

# Global timing storage
_timing_data = {}

def _timed(label, func):
    """Wrap a function to record wall time and CPU time."""
    def wrapper(*args, **kwargs):
        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()
        result = func(*args, **kwargs)
        t1_wall = time.perf_counter()
        t1_cpu = time.process_time()
        dt_wall = t1_wall - t0_wall
        dt_cpu = t1_cpu - t0_cpu
        if label not in _timing_data:
            _timing_data[label] = {'calls': 0, 'wall': 0.0, 'cpu': 0.0}
        _timing_data[label]['calls'] += 1
        _timing_data[label]['wall'] += dt_wall
        _timing_data[label]['cpu'] += dt_cpu
        print(f"  [TIMER] {label}: {dt_wall*1000:.1f} ms (wall), {dt_cpu*1000:.1f} ms (cpu)")
        return result
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

def install_timers():
    """Monkey-patch key DFT/SCF functions with timing wrappers."""
    from pyscf.scf import hf as hf_mod
    from pyscf.dft import rks as rks_mod
    from pyscf.dft import numint as numint_mod
    from pyscf.dft import gen_grid as gen_grid_mod

    # --- SCF kernel (the main loop) ---
    hf_mod.kernel = _timed('scf.kernel', hf_mod.kernel)

    # --- get_veff (DFT: J + Vxc) ---
    rks_mod.get_veff = _timed('rks.get_veff', rks_mod.get_veff)

    # --- get_jk (Coulomb + exchange) ---
    # We patch the instance method via the class
    original_get_jk = hf_mod.SCF.get_jk
    def timed_get_jk(self, *args, **kwargs):
        t0 = time.perf_counter()
        result = original_get_jk(self, *args, **kwargs)
        dt = time.perf_counter() - t0
        label = 'SCF.get_jk'
        if label not in _timing_data:
            _timing_data[label] = {'calls': 0, 'wall': 0.0, 'cpu': 0.0}
        _timing_data[label]['calls'] += 1
        _timing_data[label]['wall'] += dt
        print(f"  [TIMER] {label}: {dt*1000:.1f} ms (wall)")
        return result
    hf_mod.SCF.get_jk = timed_get_jk

    # --- eig (diagonalization) ---
    original_eig = hf_mod.eig
    def timed_eig(*args, **kwargs):
        t0 = time.perf_counter()
        result = original_eig(*args, **kwargs)
        dt = time.perf_counter() - t0
        label = 'scf.eig'
        if label not in _timing_data:
            _timing_data[label] = {'calls': 0, 'wall': 0.0, 'cpu': 0.0}
        _timing_data[label]['calls'] += 1
        _timing_data[label]['wall'] += dt
        print(f"  [TIMER] {label}: {dt*1000:.1f} ms (wall)")
        return result
    hf_mod.eig = timed_eig

    # --- nr_rks (XC numerical integration) ---
    original_nr_rks = numint_mod.NumInt.nr_rks
    def timed_nr_rks(self, *args, **kwargs):
        t0 = time.perf_counter()
        result = original_nr_rks(self, *args, **kwargs)
        dt = time.perf_counter() - t0
        label = 'NumInt.nr_rks (XC integration)'
        if label not in _timing_data:
            _timing_data[label] = {'calls': 0, 'wall': 0.0, 'cpu': 0.0}
        _timing_data[label]['calls'] += 1
        _timing_data[label]['wall'] += dt
        print(f"  [TIMER] {label}: {dt*1000:.1f} ms (wall)")
        return result
    numint_mod.NumInt.nr_rks = timed_nr_rks

    # --- Grids.build ---
    original_build = gen_grid_mod.Grids.build
    def timed_build(self, *args, **kwargs):
        t0 = time.perf_counter()
        result = original_build(self, *args, **kwargs)
        dt = time.perf_counter() - t0
        label = 'Grids.build'
        if label not in _timing_data:
            _timing_data[label] = {'calls': 0, 'wall': 0.0, 'cpu': 0.0}
        _timing_data[label]['calls'] += 1
        _timing_data[label]['wall'] += dt
        print(f"  [TIMER] {label}: {dt*1000:.1f} ms (wall)")
        return result
    gen_grid_mod.Grids.build = timed_build

    # --- block_loop (inner grid loop: AO eval + rho + vxc contraction) ---
    original_block_loop = numint_mod.NumInt.block_loop
    def timed_block_loop(self, *args, **kwargs):
        t0 = time.perf_counter()
        count = 0
        for item in original_block_loop(self, *args, **kwargs):
            count += 1
            yield item
        dt = time.perf_counter() - t0
        label = 'NumInt.block_loop (AO eval + grid iter)'
        if label not in _timing_data:
            _timing_data[label] = {'calls': 0, 'wall': 0.0, 'cpu': 0.0}
        _timing_data[label]['calls'] += 1
        _timing_data[label]['wall'] += dt
        print(f"  [TIMER] {label}: {dt*1000:.1f} ms (wall, {count} blocks)")
    numint_mod.NumInt.block_loop = timed_block_loop

def print_summary():
    print("\n" + "="*70)
    print("TIMING SUMMARY")
    print("="*70)
    print(f"{'Function':<45} {'Calls':>6} {'Wall (ms)':>12} {'Per call':>12}")
    print("-"*70)
    for label, data in sorted(_timing_data.items(), key=lambda x: -x[1]['wall']):
        per_call = data['wall'] / data['calls'] * 1000 if data['calls'] > 0 else 0
        print(f"{label:<45} {data['calls']:>6} {data['wall']*1000:>12.1f} {per_call:>12.1f}")
    print("="*70)

def reset_timers():
    _timing_data.clear()

# ============================================================
# Main profiling runs
# ============================================================

def run_dft(name, mol_def, xc='pbe,pbe', grids_level=3, verbose=0, use_df=False, do_profile=False):
    print(f"\n{'='*70}")
    print(f"Molecule: {name}, XC={xc}, basis={mol_def['basis']}, grids_level={grids_level}, DF={use_df}")
    print(f"OMP threads: {lib.num_threads()}")
    print(f"{'='*70}")

    reset_timers()

    t0_total = time.perf_counter()

    mol = gto.M(atom=mol_def['atom'], basis=mol_def['basis'], verbose=verbose)
    print(f"NAO = {mol.nao_nr()}, natm = {mol.natm}, nelec = {mol.nelectron}")

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = grids_level
    mf.verbose = verbose
    mf.max_cycle = 1  # only one SCF cycle — we care about timing, not convergence

    if use_df:
        mf = mf.density_fit()
        print(f"DF auxiliary basis: {mf.auxbasis or 'default'}")

    if do_profile:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        t0_kernel = time.perf_counter()
        pr.enable()
        e = mf.kernel()
        pr.disable()
        t_kernel = time.perf_counter() - t0_kernel
        # Print top 30 to stdout
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats(30)
        print("\n--- cProfile (top 30 by tottime) ---")
        print(s.getvalue())
        # Save binary .prof + human-readable .txt to debug/
        import os
        debug_dir = '/home/prokophapala/git/pyscf/debug/profile_dft'
        os.makedirs(debug_dir, exist_ok=True)
        nthreads = lib.num_threads()
        tag = f"{name}_g{grids_level}{'_df' if use_df else ''}_t{nthreads}"
        prof_path = f'{debug_dir}/{tag}.prof'
        txt_path = f'{debug_dir}/{tag}.txt'
        pr.dump_stats(prof_path)
        with open(txt_path, 'w') as f:
            ps2 = pstats.Stats(pr, stream=f).sort_stats('tottime')
            ps2.print_stats(50)
        print(f"cProfile saved: {prof_path} + {txt_path}")
    else:
        t0_kernel = time.perf_counter()
        e = mf.kernel()
        t_kernel = time.perf_counter() - t0_kernel

    t_total = time.perf_counter() - t0_total

    print(f"\nEnergy (1 cycle, not converged) = {e:.12f}")
    print(f"Kernel time (1 cycle) = {t_kernel:.3f} s")
    print(f"Total time  = {t_total:.3f} s")
    print(f"Number of grids = {mf.grids.weights.size}")

    print_summary()
    return e, t_total

# ============================================================
# Run all tests
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mols', nargs='*', default=None, help='Molecules to run (default: all)')
    parser.add_argument('--df', action='store_true', help='Use density fitting (RI-JK) for Coulomb')
    parser.add_argument('--profile', action='store_true', help='Run cProfile on single SCF cycle')
    parser.add_argument('--grid-sweep', action='store_true', help='Sweep grid levels 0-3')
    args = parser.parse_args()

    print("\n" + "#"*70)
    print("# Installing timing hooks into PySCF DFT functions")
    print("#"*70)
    install_timers()

    mol_names = args.mols if args.mols else list(molecules.keys())
    results = {}

    for name in mol_names:
        results[name] = run_dft(name, molecules[name], use_df=args.df, do_profile=args.profile)

    # --- Summary comparison ---
    print("\n" + "#"*70)
    print("# OVERALL COMPARISON (1 SCF cycle)")
    print("#"*70)
    df_tag = ' [DF]' if args.df else ''
    print(f"{'Molecule':<15} {'Energy':>20} {'1-cycle time (s)':>15}")
    print("-"*50)
    for name, (e, t) in results.items():
        print(f"{name:<15} {e:>20.12f} {t:>15.3f}{df_tag}")

    # --- DF vs no-DF comparison ---
    if args.df:
        print("\n" + "#"*70)
        print("# DF vs NO-DF COMPARISON (1 SCF cycle)")
        print("#"*70)
        print(f"{'Molecule':<15} {'no-DF (s)':>12} {'DF (s)':>12} {'Speedup':>10}")
        print("-"*50)
        for name in mol_names:
            _, t_nodf = run_dft(name, molecules[name], use_df=False)
            _, t_df = run_dft(name, molecules[name], use_df=True)
            speedup = t_nodf / t_df if t_df > 0 else 0
            print(f"{name:<15} {t_nodf:>12.3f} {t_df:>12.3f} {speedup:>10.2f}x")

    # --- Grid level sweep ---
    if args.grid_sweep:
        print("\n" + "#"*70)
        print("# GRID LEVEL SWEEP (1 SCF cycle)")
        print("#"*70)
        for name in mol_names:
            print(f"\n--- {name} ---")
            print(f"{'Level':>5} {'Grids':>10} {'Time (s)':>10} {'Energy':>20}")
            for level in [0, 1, 2, 3]:
                mol = gto.M(atom=molecules[name]['atom'], basis=molecules[name]['basis'], verbose=0)
                mf = dft.RKS(mol)
                mf.xc = 'pbe,pbe'
                mf.grids.level = level
                mf.verbose = 0
                mf.max_cycle = 1
                t0 = time.perf_counter()
                e = mf.kernel()
                dt = time.perf_counter() - t0
                ngrids = mf.grids.weights.size
                print(f"{level:>5} {ngrids:>10} {dt:>10.3f} {e:>20.10f}")
