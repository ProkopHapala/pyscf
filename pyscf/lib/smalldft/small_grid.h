/* Copyright 2026 The PySCF Developers. All Rights Reserved.
 *
 * Grid-parallel LDA/GGA kernels for small molecules (nao << ngrids).
 * χ layout: F-contiguous (ngrids, nao), chi[g + mu*ngrids].
 * GGA χ: F (4, ngrids, nao); component c at chi + c*ngrids*nao.
 * ρ GGA: C (4, ngrids), rho[c*ngrids + g].
 * DM layout: C-contiguous (nao, nao), dm[mu*nao + nu].
 */
#ifndef SMALL_GRID_H
#define SMALL_GRID_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void SMALL_rho_lda(double *rho, const double *chi, const double *dm,
                   int nao, int ngrids, int nthreads);

void SMALL_rho_gga(double *rho, const double *chi, const double *dm,
                   int nao, int ngrids, int nthreads, int hermi);

/* wv: C (4,ngrids) or (ngrids) for LDA; chi F (4,ngrids,nao) or (ngrids,nao).
 * vmat: C (nao,nao), zeroed on entry; hermi=1 → vmat += vmat^T at end (GGA). */
void SMALL_vmat_lda(double *vmat, const double *chi, const double *wv,
                    int nao, int ngrids, int nthreads);
void SMALL_vmat_gga(double *vmat, const double *chi, const double *wv,
                    int nao, int ngrids, int nthreads, int hermi);

/* --- stream path (block χ; do not use for full-grid cache) --- */
void SMALL_stream_rho_lda(double *rho, const double *chi, const double *dm,
                          int nao, int nblk, int nthreads);
void SMALL_stream_rho_gga(double *rho, const double *chi, const double *dm,
                          int nao, int nblk, int nthreads, int hermi);
void SMALL_stream_vmat_lda_acc(double *vmat, const double *chi, const double *wv,
                               int nao, int nblk, int nthreads);
void SMALL_stream_vmat_gga_acc(double *vmat, const double *chi, const double *wv,
                               int nao, int nblk, int nthreads);
void SMALL_stream_vmat_hermi(double *vmat, int nao);

#ifdef __cplusplus
}
#endif

#endif
