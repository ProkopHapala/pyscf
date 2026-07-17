---
type: ArchitectureNote
title: DF storage control and benchmark hygiene
description: Explicit DF.storage (auto|incore|outcore), lazy build lifetime, hidden disk I/O that spoils timings, other PySCF offloads to watch
tags: [df, density-fitting, benchmark, amdahl, memory, hdf5, hygiene]
timestamp: 2026-07-17
---

# DF storage and benchmark hygiene

**Why this doc exists:** On 2026-07-17 (i5-12500H, 16 GiB, GTX 1650) we measured PTCDA PBE/6-31g where CPU smallDFT cut XC **3.4×** (3041→900 ms) but cycle speedup was only **2.25×** because `_cderi` silently moved to **HDF5 outcore** and DF-J jumped **54→478 ms**. Disk/HDF5 I/O is not a “small constant” — it can dominate the cycle and invalidate Amdahl claims. See canvas `amdahl-budget-breakdown` and `expamples_prokop/profile_amdahl_budget.py`.

---

## Three clocks (never mix them)

| Clock | Contents | When paid |
|-------|----------|-----------|
| **A Setup / geometry** | `Grids.build`, `DF.build` (`_cderi`), AO cache / GPU plan | once per geometry |
| **B SCF cycle** | XC + J (+ tiny eig) | every iteration |
| **C Full job** | ≈ A + init + N×B | wall you care about for PTCDA |

Isolated XC microbenchmarks are **neither** B nor C. Speeding up 90% of B by 10× gives ~5× on B only if the other 10% stays put **and** you are not measuring A+C.

---

## What PySCF DF does by default (not a bug — a library default)

```
mf.density_fit()          # creates with_df; does NOT build _cderi yet
mf.kernel()
  └─ first get_jk / df.loop()
       └─ if _cderi is None: df.build()   # LAZY
```

`DF.build()` then chooses storage with the **legacy auto rule**:

```
need_mb = nao_pair * naux * 8 / 1e6
if need_mb < 0.9 * (max_memory - already_used):
    _cderi = incore ndarray          # RAM
else:
    _cderi = NamedTemporaryFile HDF5 # disk — every J reads blocks
```

**Why lazy?** Huge-system API: don’t allocate hundreds of MB if J is never called; allow preloaded `_cderi` files; support outcore when RAM is insufficient. It is **not** tuned for small-molecule timing hygiene.

**Measured (this laptop, prepared DF, 2026-07-17):**

| Molecule | `_cderi` size | Steady J (incore) | Notes |
|----------|--------------:|------------------:|-------|
| benzene 6-31g | ~10 MB | ~1 ms | fine |
| PTCDA 6-31g | ~750 MB | ~54–60 ms | fine **if** stays incore |
| PTCDA + AO cache, `storage=auto` (bad order) | same | **~480 ms** | flipped to HDF5 → spoils XC win |

### PTCDA one SCF cycle (this machine, `storage=incore`, 4 OMP)

Driver: `profile_amdahl_budget.py --mol PTCDA --n-cycles 1 --df-storage incore`.

| Path | Setup | Cycle XC | Cycle J | Cycle veff | vs ref |
|------|------:|---------:|--------:|-----------:|-------:|
| CPU ref | 3.1 s | 3433 ms | 61 ms | **3494 ms** | — |
| `prepare_smalldft_for_scf` | 5.0 s (incl. AO+DF) | 1011 ms | 59 ms | **1070 ms** | **3.27×** cycle |

Energy matches to displayed digits (`E=-1368.111832`). J stays ~60 ms because `_cderi` remains an incore ndarray (750 MB).

Geometry scans **must** rebuild `_cderi` each geometry (3-center integrals depend on nuclear positions). “Persist across geometries” is wrong; “build once per geometry, before the SCF loop” is right.

---

## Explicit control (SSOT) — not a hack

### `DF.storage` — `'auto' | 'incore' | 'outcore'`

Defined on `pyscf.df.df.DF` (default `'auto'` = legacy behaviour).

| Value | Behaviour |
|-------|-----------|
| `auto` | Legacy threshold vs remaining `max_memory` |
| `incore` | Force RAM ndarray; **`MemoryError`** if it does not fit (fail loud — never silent disk) |
| `outcore` | Force HDF5 even if RAM would suffice (to measure I/O on purpose) |

```python
mf = dft.RKS(mol, xc='PBE').density_fit()
mf.with_df.storage = 'incore'          # deterministic policy
mf.with_df.max_memory = 8000           # headroom for ~750 MB tensor + AO
from pyscf.OpenCL.gpu_profiles import prepare_df_for_scf, assert_df_incore
prepare_df_for_scf(mf, storage='incore', require_incore=True)
assert_df_incore(mf)                   # optional second guard
kind, detail, nbytes = mf.with_df.describe_cderi()
# kind == 'incore'  → safe to time J as RAM contraction
mf.kernel()
```

Helpers (same module):

| API | Role |
|-----|------|
| `prepare_df_for_scf(mf, storage=..., require_incore=...)` | Build `_cderi` (+ GPU DF plan) **before** `kernel`; optional hard require |
| `assert_df_incore(mf)` | Raise if not in-RAM |
| `df.describe_cderi()` | `('incore'\|'outcore'\|'none', detail, nbytes)` |
| `apply_gpu_profile(..., df_storage=..., require_df_incore=...)` | Forwards to `prepare_df_for_scf` |

**Ordering rule:** build DF **before** allocating a large AO workspace (`GridWorkspace.eval_ao`). Otherwise `storage='auto'` sees less free `max_memory` and spills. With `storage='incore'`, raise `max_memory` so both fit.

**Benchmark driver default:** `profile_amdahl_budget.py --df-storage incore` (require_incore on). Use `--df-storage outcore` / `--allow-outcore` only when deliberately studying I/O.

---

## Other hidden / lifetime traps (checklist)

| Mechanism | Disk? | Spoils cycle timing? | Control |
|-----------|:-----:|:--------------------:|---------|
| **DF `_cderi` outcore HDF5** | yes | **yes — severe** | `DF.storage`, `prepare_df_for_scf`, `assert_df_incore` |
| **DF lazy `build()` inside first `get_jk`** | maybe | first cycle only (looks like huge “J”) | `prepare_df_for_scf` before `kernel` |
| **4-center `_eri` incore** | no | huge RAM / first build if `direct_scf=False` or `incore_anyway` | keep `direct_scf=True` (default) for DFT |
| **Direct SCF integral screening cache** | no | in-RAM; incremental Δdm J | OK; don’t confuse with DF |
| **`chkfile`** | yes | I/O each dump if set | leave `mf.chkfile=None` in benches |
| **NumInt `block_loop` / `max_memory`** | no | chunks AO in RAM (not disk) | raises block count, not HDF5 |
| **GPU DF plan uploads f32 copy** | no | setup cost once | `prepare_df_for_scf` / profile setup |
| **OpenCL kernel compile** | maybe cache | setup | once per TileConfig; not per cycle |
| **`lib.param.TMPDIR` NamedTemporaryFile** | yes | DF outcore path | avoid via `storage='incore'` |

If a new “J got 10× slower” mystery appears: **first** print `mf.with_df.describe_cderi()` and `mf.with_df.storage`.

---

## Reproduce (this machine)

```bash
# Deterministic: force incore, fail if spilled
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/prokophapala/git/pyscf \
  python3 -u expamples_prokop/profile_amdahl_budget.py \
  --mol PTCDA --grid-level 2 --n-cycles 4 --threads 4 \
  --modes cpu_ref cpu_small --df-storage incore

# Deliberate I/O path (expect slow J)
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/prokophapala/git/pyscf \
  python3 -u expamples_prokop/profile_amdahl_budget.py \
  --mol PTCDA --grid-level 2 --n-cycles 2 --threads 4 \
  --modes cpu_ref --df-storage outcore --allow-outcore
```

---

## Related

- `doc/CPU_benchmark.md` — benzene CPU XC; Amdahl cycle tables
- `doc/acceptance_2026-07-11.md` — large-mol DF prep / Amdahl
- `doc/opencl_gpu_paths_cookbook.md` — `prepare_df_for_scf` in GPU profiles
- `pyscf/df/df.py` — `storage`, `describe_cderi`, `build`
- `pyscf/OpenCL/gpu_profiles.py` — `prepare_df_for_scf`, `assert_df_incore`
