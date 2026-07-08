/* Copyright 2026 The PySCF Developers. All Rights Reserved.
 *
 * libsmalldft — grid-tile OpenMP kernels for LDA/GGA ρ and vmat on PySCF χ layout.
 *
 * Motivation: small-molecule XC is memory-bound over ngrids; parallelize on grid
 * index with disjoint ρ writes and private vmat buffers (no hot-loop atomics).
 * ctypes passes NumPy pointers directly; BLAS uses lda=ngrids on F-order χ tiles
 * so tiles need no pack buffer. vmat dgemm output is Fortran-layout → transpose
 * on reduce; GGA hermi uses out+out.T via temp (in-place += would double-count).
 */
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "vhf/fblas.h"
#include "smalldft/small_grid.h"

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#ifndef TILE
#define TILE 512
#endif

static int _nthreads(int nthreads)
{
#ifdef _OPENMP
        if (nthreads > 0) {
                return nthreads;
        }
        return omp_get_max_threads();
#else
        (void)nthreads;
        return 1;
#endif
}

static void _rho_tile_lda(double *rho, const double *chi0, const double *dm,
                          double *c0, int tile, int nao, int ig0, int ngrids)
{
        const double one = 1.;
        const double zero = 0.;
        int t, mu;

        dgemm_("N", "T", &tile, &nao, &nao, &one,
               chi0 + ig0, &ngrids, dm, &nao, &zero, c0, &tile);

        for (t = 0; t < tile; t++) {
                rho[ig0 + t] = 0.;
        }
        for (mu = 0; mu < nao; mu++) {
                const double *chi_mu = chi0 + ig0 + (size_t)mu * ngrids;
                const double *c0_mu = c0 + (size_t)mu * tile;
#ifdef _OPENMP
#pragma omp simd
#endif
                for (t = 0; t < tile; t++) {
                        rho[ig0 + t] += chi_mu[t] * c0_mu[t];
                }
        }
}

void SMALL_rho_lda(double *rho, const double *chi, const double *dm,
                   int nao, int ngrids, int nthreads)
{
        int nth = _nthreads(nthreads);
        int ig0;

#ifdef _OPENMP
#pragma omp parallel num_threads(nth) default(none) \
        shared(rho, chi, dm, nao, ngrids) private(ig0)
#endif
{
        double *c0 = NULL;
        size_t bufsz = 0;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (ig0 = 0; ig0 < ngrids; ig0 += TILE) {
                int tile = MIN(TILE, ngrids - ig0);
                size_t need = (size_t)tile * nao;
                if (bufsz < need) {
                        free(c0);
                        c0 = (double *)malloc(need * sizeof(double));
                        bufsz = need;
                }
                _rho_tile_lda(rho, chi, dm, c0, tile, nao, ig0, ngrids);
        }
        free(c0);
}
}

/* hermi=1: ρ_0 from _rho_tile_lda; ρ_k = 2 Σ (Dχ₀)_μ χ_kμ.
 * chi: base of χ₀ in F (4,ngrids,nao); rho: C (4,ngrids). */
void SMALL_rho_gga(double *rho, const double *chi, const double *dm,
                   int nao, int ngrids, int nthreads, int hermi)
{
        int nth = _nthreads(nthreads);
        const size_t ao_size = (size_t)ngrids * nao;
        int ig0, t, k;

        (void)hermi;

#ifdef _OPENMP
#pragma omp parallel num_threads(nth) default(none) \
        shared(rho, chi, dm, nao, ngrids, ao_size) private(ig0, t, k)
#endif
{
        const double two = 2.;
        double *c0 = NULL;
        size_t bufsz = 0;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (ig0 = 0; ig0 < ngrids; ig0 += TILE) {
                int tile = MIN(TILE, ngrids - ig0);
                size_t need = (size_t)tile * nao;

                if (bufsz < need) {
                        free(c0);
                        c0 = (double *)malloc(need * sizeof(double));
                        bufsz = need;
                }

                _rho_tile_lda(rho, chi, dm, c0, tile, nao, ig0, ngrids);

                for (k = 1; k < 4; k++) {
                        double *rho_k = rho + (size_t)k * ngrids + ig0;
                        for (t = 0; t < tile; t++) {
                                rho_k[t] = 0.;
                        }
                }
                for (k = 1; k < 4; k++) {
                        const double *chi_k = chi + (size_t)k * ao_size + ig0;
                        double *rho_k = rho + (size_t)k * ngrids + ig0;
                        for (int mu = 0; mu < nao; mu++) {
                                const double *chi_mu = chi_k + (size_t)mu * ngrids;
                                const double *c0_mu = c0 + (size_t)mu * tile;
#ifdef _OPENMP
#pragma omp simd
#endif
                                for (t = 0; t < tile; t++) {
                                        rho_k[t] += two * chi_mu[t] * c0_mu[t];
                                }
                        }
                }
        }
        free(c0);
}
}

static void _hermi_sum_inplace(double *vmat, int nao)
{
        int i, j, n2 = nao * nao;
        double *tmp = (double *)malloc((size_t)n2 * sizeof(double));
        for (i = 0; i < n2; i++) {
                tmp[i] = vmat[i];
        }
        for (i = 0; i < nao; i++) {
                for (j = 0; j < nao; j++) {
                        vmat[i*nao + j] = tmp[i*nao + j] + tmp[j*nao + i];
                }
        }
        free(tmp);
}

void SMALL_vmat_lda(double *vmat, const double *chi, const double *wv,
                    int nao, int ngrids, int nthreads)
{
        int nth = _nthreads(nthreads);
        int ig0, n2 = nao * nao;

#ifdef _OPENMP
#pragma omp parallel num_threads(nth) default(none) \
        shared(vmat, chi, wv, nao, ngrids, n2) private(ig0)
#endif
{
        const double one = 1.;
        const double zero = 0.;
        double *v_priv = (double *)calloc((size_t)n2, sizeof(double));
        double *chi_w = NULL;
        size_t bufsz = 0;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (ig0 = 0; ig0 < ngrids; ig0 += TILE) {
                int tile = MIN(TILE, ngrids - ig0);
                int t, mu;
                size_t need = (size_t)tile * nao;

                if (bufsz < need) {
                        free(chi_w);
                        chi_w = (double *)malloc(need * sizeof(double));
                        bufsz = need;
                }

                for (mu = 0; mu < nao; mu++) {
                        const double *chi_mu = chi + ig0 + (size_t)mu * ngrids;
                        double *chi_w_mu = chi_w + (size_t)mu * tile;
#ifdef _OPENMP
#pragma omp simd
#endif
                        for (t = 0; t < tile; t++) {
                                chi_w_mu[t] = chi_mu[t] * wv[ig0 + t];
                        }
                }

                dgemm_("T", "N", &nao, &nao, &tile, &one,
                       chi + ig0, &ngrids, chi_w, &tile, &one, v_priv, &nao);
        }
        free(chi_w);
#ifdef _OPENMP
#pragma omp critical(small_vmat_reduce)
#endif
        {
                int i, j;
                for (j = 0; j < nao; j++) {
                        for (i = 0; i < nao; i++) {
                                vmat[i*nao + j] += v_priv[i + j*nao];
                        }
                }
        }
        free(v_priv);
}
}

void SMALL_vmat_gga(double *vmat, const double *chi, const double *wv,
                    int nao, int ngrids, int nthreads, int hermi)
{
        int nth = _nthreads(nthreads);
        const size_t ao_size = (size_t)ngrids * nao;
        int ig0, n2 = nao * nao;

#ifdef _OPENMP
#pragma omp parallel num_threads(nth) default(none) \
        shared(vmat, chi, wv, nao, ngrids, ao_size, n2) private(ig0)
#endif
{
        const double one = 1.;
        double *v_priv = (double *)calloc((size_t)n2, sizeof(double));
        double *aow = NULL;
        size_t bufsz = 0;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (ig0 = 0; ig0 < ngrids; ig0 += TILE) {
                int tile = MIN(TILE, ngrids - ig0);
                int t, mu;
                size_t need = (size_t)tile * nao;

                if (bufsz < need) {
                        free(aow);
                        aow = (double *)malloc(need * sizeof(double));
                        bufsz = need;
                }

                for (mu = 0; mu < nao; mu++) {
                        double *aow_mu = aow + (size_t)mu * tile;
                        const double *chi0_mu = chi + ig0 + (size_t)mu * ngrids;
                        const double *chi1_mu = chi + ao_size + ig0 + (size_t)mu * ngrids;
                        const double *chi2_mu = chi + 2 * ao_size + ig0 + (size_t)mu * ngrids;
                        const double *chi3_mu = chi + 3 * ao_size + ig0 + (size_t)mu * ngrids;
#ifdef _OPENMP
#pragma omp simd
#endif
                        for (t = 0; t < tile; t++) {
                                int g = ig0 + t;
                                aow_mu[t] = wv[g] * chi0_mu[t]
                                          + wv[(size_t)ngrids + g] * chi1_mu[t]
                                          + wv[2 * (size_t)ngrids + g] * chi2_mu[t]
                                          + wv[3 * (size_t)ngrids + g] * chi3_mu[t];
                        }
                }

                dgemm_("T", "N", &nao, &nao, &tile, &one,
                       chi + ig0, &ngrids, aow, &tile, &one, v_priv, &nao);
        }
        free(aow);
#ifdef _OPENMP
#pragma omp critical(small_vmat_reduce)
#endif
        {
                int i, j;
                for (j = 0; j < nao; j++) {
                        for (i = 0; i < nao; i++) {
                                vmat[i*nao + j] += v_priv[i + j*nao];
                        }
                }
        }
        free(v_priv);
}
        if (hermi) {
                _hermi_sum_inplace(vmat, nao);
        }
}

