# libsmalldft

OpenMP grid-tile kernels for LDA/GGA ρ and vmat on PySCF F-contiguous χ. Built as `libsmalldft.so` beside other PySCF libs. See `/home/prokop/git/pyscf/doc/smallDFT_cpu_path.md`.

- **small_grid.c** — `SMALL_rho_lda`, `SMALL_rho_gga`, `SMALL_vmat_lda`, `SMALL_vmat_gga`; TILE=512, strided BLAS, private vmat buffers
- **small_grid.h** — C API declarations
- **CMakeLists.txt** — cmake target linked to `np_helper` + BLAS + OpenMP
- **build.sh** — quick standalone gcc build without full PySCF cmake
