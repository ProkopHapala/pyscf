# smallDFT

CPU grid-parallel RKS XC for small molecules (`nao ≲ 200`): OpenMP ρ/vmat in `libsmalldft`, libcint AO layout, drop-in `nr_rks`. See `/home/prokop/git/pyscf/doc/smallDFT_cpu_path.md`.

- **nr_rks.py** — drop-in replacement for `numint.nr_rks`; dispatches to C when `libsmalldft` loaded
- **rho.py** / **vmat.py** — ρ and vmat drivers; `use_c=True` → ctypes → OpenMP kernels
- **_ctypes.py** — `libsmalldft` load + `SMALL_*` bindings
- **workspace.py** — `GridWorkspace`: preallocated ρ/vmat buffers; `eval_ao()` sets χ from `eval_ao_native`
- **layout.py** — keep libcint F-contiguous `(ngrids, nao)`; `eval_ao_native` entry point
- **patch.py** — `enable()` / `disable()` monkey-patch on `NumInt.nr_rks`
- **profile.py** — `profile_xc_bottleneck`, `profile_compare`, timing breakdowns
- **parallel.py** — legacy Python `ThreadPoolExecutor` tiles (fallback only; do not extend)
