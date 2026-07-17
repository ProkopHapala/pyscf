/* Copyright 2026 The PySCF Developers. All Rights Reserved.
 *
 * Block-streaming ρ/vmat — separate from the full-χ cache path in small_grid.c.
 *
 * Use when χ is evaluated per grid block (PySCF block_loop / eval_ao on a
 * slice) and never materialized as (4,ngrids,nao). GGA is local: ρ and vxc on
 * the same block points; ∇ρ uses χ and ∇χ on that block only.
 *
 * vmat_*_acc: accumulate into caller-owned vmat (zero once). GGA hermi is NOT
 * applied here — call SMALL_stream_vmat_hermi once after all blocks.
 */
#include <stdlib.h>
#include "smalldft/small_grid.h"

void SMALL_stream_rho_lda(double *rho, const double *chi, const double *dm,
                          int nao, int nblk, int nthreads)
{
        SMALL_rho_lda(rho, chi, dm, nao, nblk, nthreads);
}

void SMALL_stream_rho_gga(double *rho, const double *chi, const double *dm,
                          int nao, int nblk, int nthreads, int hermi)
{
        SMALL_rho_gga(rho, chi, dm, nao, nblk, nthreads, hermi);
}

void SMALL_stream_vmat_lda_acc(double *vmat, const double *chi, const double *wv,
                               int nao, int nblk, int nthreads)
{
        SMALL_vmat_lda(vmat, chi, wv, nao, nblk, nthreads);
}

void SMALL_stream_vmat_gga_acc(double *vmat, const double *chi, const double *wv,
                               int nao, int nblk, int nthreads)
{
        /* hermi=0: accumulate raw; finalize with SMALL_stream_vmat_hermi */
        SMALL_vmat_gga(vmat, chi, wv, nao, nblk, nthreads, 0);
}

void SMALL_stream_vmat_hermi(double *vmat, int nao)
{
        int i, j, n2 = nao * nao;
        double *tmp = (double *)malloc((size_t)n2 * sizeof(double));
        if (tmp == NULL) {
                return;
        }
        for (i = 0; i < n2; i++) {
                tmp[i] = vmat[i];
        }
        for (i = 0; i < nao; i++) {
                for (j = 0; j < nao; j++) {
                        vmat[i * nao + j] = tmp[i * nao + j] + tmp[j * nao + i];
                }
        }
        free(tmp);
}
