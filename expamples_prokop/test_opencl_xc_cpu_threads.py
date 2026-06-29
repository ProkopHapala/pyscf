#!/usr/bin/env python3
"""CPU thread scaling vs GPU for benzene, pentacene, PTCDA.

Usage:
  OMP_NUM_THREADS=1 PYTHONPATH=/home/prokop/git/pyscf python3 -u \\
    expamples_prokop/test_opencl_xc_cpu_threads.py

  PYTHONPATH=/home/prokop/git/pyscf python3 -u \\
    expamples_prokop/test_opencl_xc_cpu_threads.py --threads 1 15
"""
import argparse
import os
import re
import sys
import time

import numpy as np
import pyscf
from pyscf import dft, gto, lib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_XYZ_DIR = os.path.join(_REPO, 'data', 'xyz')

# (name, basis, grid_level) — match report Parts 7–8
SYSTEMS = (
    ('benzene', 'ccpvdz', 3),
    ('pentacene', '6-31g', 2),
    ('PTCDA', '6-31g', 2),
)

# Best GPU wall (ms) from Part 7/8 benchmarks on RTX 3090, hermite_otf + gpu PBE
GPU_BEST_MS = {
    'benzene': 28.1,
    'pentacene': 183.1,
    'PTCDA': 261.3,
}


def log(msg='', end='\n'):
    print(msg, end=end, flush=True)


def read_xyz(path):
    with open(path) as f:
        lines = f.readlines()
    natom = int(lines[0].strip())
    atoms = []
    for line in lines[2:2 + natom]:
        parts = line.split()
        if re.match(r'^[A-Z][a-z]?$', parts[0]):
            atoms.append(f'{parts[0]} {parts[1]} {parts[2]} {parts[3]}')
    return '; '.join(atoms)


def bench_nr_rks(mol, grids, dm, ni, n_warm=1, n_timed=3):
    for _ in range(n_warm):
        ni.nr_rks(mol, grids, 'PBE', dm, max_memory=4000)
    times, last = [], None
    for _ in range(n_timed):
        t0 = time.perf_counter()
        last = ni.nr_rks(mol, grids, 'PBE', dm, max_memory=4000)
        times.append(time.perf_counter() - t0)
    return min(times) * 1e3, last


def main():
    ap = argparse.ArgumentParser(description='CPU thread scaling vs GPU reference')
    ap.add_argument('--threads', nargs='+', type=int, default=[1, 15])
    ap.add_argument('--n-timed', type=int, default=3)
    args = ap.parse_args()

    log(f'PySCF {pyscf.__version__}  thread counts: {args.threads}')
    log(f'{"system":<12} {"basis":<8} {"gr":>2} {"nao":>4} {"nG":>7}', end='')
    for t in args.threads:
        log(f' {"cpu_"+str(t):>8}', end='')
    log(f' {"gpu_otf":>8} {"gpu/cpu1":>8} {"gpu/cpuN":>8}')
    log('-' * 95)

    for name, basis, gl in SYSTEMS:
        mol = gto.M(atom=read_xyz(os.path.join(_XYZ_DIR, f'{name}.xyz')), basis=basis, verbose=0)
        grids = dft.gen_grid.Grids(mol)
        grids.level = gl
        grids.build(with_non0tab=True)
        dm = dft.RKS(mol, xc='PBE').density_fit().get_init_guess()
        ni = dft.numint.NumInt()
        nao, ngrids = mol.nao_nr(), grids.coords.shape[0]
        log(f'{name:<12} {basis:<8} {gl:>2} {nao:>4} {ngrids:>7}', end='')
        cpu_times = {}
        vxc_ref = None
        for nth in args.threads:
            lib.num_threads(nth)
            os.environ['OMP_NUM_THREADS'] = str(nth)
            ms, out = bench_nr_rks(mol, grids, dm, ni, n_timed=args.n_timed)
            cpu_times[nth] = ms
            if vxc_ref is None:
                vxc_ref = out[2]
            else:
                err = float(np.abs(out[2] - vxc_ref).max())
                if err > 1e-10:
                    log(f'  WARNING {name} threads={nth} vxc drift {err:.3e}', file=sys.stderr)
            log(f' {ms:8.1f}', end='')
        gpu_ms = GPU_BEST_MS.get(name)
        if gpu_ms:
            log(f' {gpu_ms:8.1f} {cpu_times[args.threads[0]]/gpu_ms:8.2f}x {cpu_times[args.threads[-1]]/gpu_ms:8.2f}x')
        else:
            log('')
    log('\nGPU column: gpu_hermite_otf + GPU PBE (RTX 3090, from Part 7/8).')
    log('Set OMP_NUM_THREADS before launch; script also calls lib.num_threads().')


if __name__ == '__main__':
    main()
