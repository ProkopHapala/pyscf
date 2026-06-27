# DFT Profiling Results

## Setup

- Running repo version via `PYTHONPATH=/home/prokophapala/git/pyscf`
- C extensions (.so) loaded from pip-installed binary at `~/.local/lib/python3.10/site-packages/pyscf/lib/`
- Python code from repo (modifiable)
- Profiling script: `expamples_prokop/profile_dft.py`
- All runs: PBE/6-31g, grid level 3 (default), 1 thread unless noted

## How to run

```bash
# Single-threaded, all molecules
PYTHONPATH=/home/prokophapala/git/pyscf OMP_NUM_THREADS=1 python3 expamples_prokop/profile_dft.py

# Specific molecules only
PYTHONPATH=/home/prokophapala/git/pyscf OMP_NUM_THREADS=1 python3 expamples_prokop/profile_dft.py --mols pentacene porphirin PTCDA

# Grid level sweep
PYTHONPATH=/home/prokophapala/git/pyscf OMP_NUM_THREADS=1 python3 expamples_prokop/profile_dft.py --mols H2O benzene --grid-sweep

# Multi-threaded (4 threads)
PYTHONPATH=/home/prokophapala/git/pyscf OMP_NUM_THREADS=4 python3 expamples_prokop/profile_dft.py

# With cProfile
PYTHONPATH=/home/prokophapala/git/pyscf OMP_NUM_THREADS=1 python3 -m cProfile -s tottime -o /tmp/dft_profile.out expamples_prokop/profile_dft.py
python3 -c "import pstats; pstats.Stats('/tmp/dft_profile.out').sort_stats('tottime').print_stats(30)"
```

## Results: Timing breakdown (1 thread, grid level 3)

### H2O (3 atoms, 13 AOs, 33704 grids, 7 SCF cycles)

| Function | Calls | Wall time (ms) | % of total |
|---|---|---|---|
| scf.kernel (total) | 1 | 186 | 100% |
| NumInt.nr_rks (XC integration) | 9 | 118 | 63% |
| NumInt.block_loop (AO eval + grid iter) | 9 | 116 | 62% |
| Grids.build | 1 | 24 | 13% |
| SCF.get_jk (Coulomb) | ~9 | <5 | <3% |
| scf.eig (diagonalization) | ~9 | <1 | <1% |

Energy: -76.2980382986 Ha

### Benzene (6 atoms, 66 AOs, 143560 grids, 8 SCF cycles)

| Function | Calls | Wall time (ms) | % of total |
|---|---|---|---|
| scf.kernel (total) | 1 | 3536 | 100% |
| NumInt.nr_rks (XC integration) | 10 | 2915 | 82% |
| NumInt.block_loop (AO eval + grid iter) | 10 | 2879 | 81% |
| Grids.build | 1 | 131 | 4% |
| SCF.get_jk (Coulomb) | ~10 | ~473 | 13% |
| scf.eig (diagonalization) | ~10 | <10 | <1% |

Energy: -231.8907720844 Ha

### Pentacene (36 atoms, ~270 AOs, ~450k grids, ~12 SCF cycles)

| Function | Calls | Wall time (ms) | % of total |
|---|---|---|---|
| scf.kernel (total) | 1 | 87495 | 100% |
| NumInt.nr_rks (XC integration) | ~12 | ~62000 | ~71% |
| NumInt.block_loop (AO eval + grid iter) | ~12 | ~62000 | ~71% |
| Grids.build | 1 | ~1400 | ~2% |
| SCF.get_jk (Coulomb) | ~12 | ~15000 | ~17% |

Energy: -845.522440788 Ha

### Porphirin (38 atoms, 244 AOs, 475280 grids, 12 SCF cycles)

| Function | Calls | Wall time (ms) | % of total |
|---|---|---|---|
| scf.kernel (total) | 1 | 91028 | 100% |
| NumInt.nr_rks (XC integration) | 14 | 62280 | 68% |
| NumInt.block_loop (AO eval + grid iter) | 14 | 62188 | 68% |
| Grids.build | 1 | 1415 | 2% |
| SCF.get_jk (Coulomb) | 14 | ~24000 | ~26% |

Energy: -988.113251077 Ha

### PTCDA (38 atoms, 286 AOs, 501792 grids, 17 SCF cycles)

| Function | Calls | Wall time (ms) | % of total |
|---|---|---|---|
| scf.kernel (total) | 1 | 584381 | 100% |
| SCF.get_jk (Coulomb) | 19 | 458141 | **78%** |
| NumInt.nr_rks (XC integration) | 19 | 124288 | 21% |
| NumInt.block_loop (AO eval + grid iter) | 19 | 124230 | 21% |
| Grids.build | 1 | 1525 | 0.3% |

Energy: -1368.912284152 Ha

## Overall comparison (1 thread, grid level 3)

| Molecule | Atoms | AOs | Electrons | SCF cycles | Grids | Time (s) | Energy |
|---|---|---|---|---|---|---|---|
| H2O | 3 | 13 | 10 | 7 | 33704 | 0.19 | -76.298038299 |
| Benzene | 6 | 66 | 42 | 8 | 143560 | 3.5 | -231.890772084 |
| Pentacene | 36 | ~270 | 156 | ~12 | ~450k | 87.5 | -845.522440788 |
| Porphirin | 38 | 244 | 162 | 12 | 475k | 91.0 | -988.113251077 |
| PTCDA | 38 | 286 | 200 | 17 | 502k | 584.4 | -1368.912284152 |

## cProfile breakdown (H2O + benzene, 1 thread)

Top functions by `tottime` (time excluding sub-calls):

| Function | tottime (s) | What it does |
|---|---|---|
| `eval_gto` | 0.928 | Evaluate AO basis functions on grid points (C library call) |
| `_dgemm` | 0.649 | Dense matrix multiply (density eval, Vxc contraction) |
| `getints4c` | 0.445 | 4-center ERI computation (Coulomb J build) |
| `_scale_ao` | 0.363 | Scale AO values by XC weights for GGA gradient terms |
| `_eval_xc` | 0.194 | Libxc functional evaluation |
| `_contract_rho` | 0.111 | Contract density with grid weights |
| `gen_grid_partition` | 0.031 | Becke grid partitioning |
| `_vhf.incore` | 0.026 | J matrix build from ERIs |

## Parallelization effect (OMP threads)

| Molecule | 1 thread | 4 threads | Speedup |
|---|---|---|---|
| H2O | 0.188s | 0.184s | 1.0x (no benefit) |
| Benzene | 3.537s | 2.164s | 1.6x |

Parallelization helps for benzene but not for H2O (too small to overcome overhead). Larger molecules (pentacene, porphirin, PTCDA) expected to benefit more but were not tested multi-threaded due to long runtimes.

## Grid level effect (1 thread)

| Molecule | Level | Grids | Time (s) | Energy | Error vs level 3 |
|---|---|---|---|---|---|
| H2O | 0 | 2328 | 0.052 | -76.3124151587 | ~0.014 Ha |
| H2O | 1 | 10128 | 0.074 | -76.2980141075 | ~0.00002 Ha |
| H2O | 2 | 21952 | 0.121 | -76.2980370170 | ~0.000001 Ha |
| H2O | 3 | 33704 | 0.160 | -76.2980382986 | reference |
| Benzene | 0 | 10240 | 0.669 | -231.9599184925 | ~0.069 Ha |
| Benzene | 1 | 45912 | 1.257 | -231.8907051215 | ~0.00007 Ha |
| Benzene | 2 | 99480 | 2.124 | -231.8907721191 | ~0.000001 Ha |
| Benzene | 3 | 143560 | 2.780 | -231.8907720844 | reference |

Grid level sweep for larger molecules not completed due to long runtimes.

## Key conclusions

1. **Bottleneck shifts with system size**:
   - Small molecules (H2O, benzene, pentacene, porphirin): XC grid integration dominates (63-82%)
   - Large molecules with many electrons (PTCDA, 200 e-): Coulomb J build dominates (78%)
   - Crossover occurs around ~250 AOs / ~170 electrons

2. **XC grid integration** breakdown (when it dominates):
   - `eval_gto` (AO evaluation on grid) is the single most expensive operation
   - Followed by `_dgemm` (matrix contractions for density and Vxc)
   - Then `_scale_ao` (GGA gradient scaling)

3. **Coulomb J build** (`get_jk`) scales as O(N^4) with basis size, becoming dominant for PTCDA (~24s/call × 19 cycles = 458s)

4. **Diagonalization is negligible** (<1%) for all tested systems

5. **Grid level is the most effective lever for XC-dominated cases**: level 0 gives ~4x speedup, level 1 gives ~2x speedup vs default level 3

6. **Parallelization gives modest speedup** (1.6x/4 threads for benzene), none for H2O

7. **Optimization strategies by regime**:
   - XC-dominated (small/planar molecules): fewer grid points, faster AO evaluation, incremental XC updates, sparse AO evaluation
   - Coulomb-dominated (large/heavy molecules): density fitting (RI-JK), J-matrix from auxiliary basis, fewer SCF cycles (better initial guess, DIIS)
