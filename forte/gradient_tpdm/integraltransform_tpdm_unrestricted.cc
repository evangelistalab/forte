/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2016 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include <math.h>
#include <ctype.h>
#include <stdio.h>

#include <psi4/libtrans/integraltransform.h>
#include <psi4/libtrans/integraltransform_functors.h>
#include <psi4/libpsio/psio.hpp>
#include <psi4/libciomr/libciomr.h>
#include <psi4/libiwl/iwl.hpp>
#include <psi4/libmints/matrix.h>
#include <psi4/libqt/qt.h>
#include <psi4/psifiles.h>
#include <psi4/libtrans/mospace.h>
#include <psi4/libpsi4util/PsiOutStream.h>

#include "backtransform_tpdm.h"

#define EXTERN
#include <psi4/libdpd/dpd.h>

using namespace psi;

void TPDMBackTransform::backtransform_tpdm_unrestricted() {
    check_initialized();

    // This can be safely called - it returns immediately if the MO TPDM is already sorted
    presort_mo_tpdm_unrestricted();

    // Grab the transformation coefficients
    SharedMatrix ca = aMOCoefficients_[MOSPACE_ALL];
    SharedMatrix cb = bMOCoefficients_[MOSPACE_ALL];

    dpdbuf4 J1, J2, K;

    // Grab control of DPD for now, but store the active number to restore it later
    int currentActiveDPD = psi::dpd_default;
    dpd_set_default(myDPDNum_);

    int nBuckets;
    int thisBucketRows;
    size_t rowsPerBucket;
    size_t rowsLeft;
    size_t memFree;

    double** TMP = block_matrix(nso_, nso_);

    /*** first half transformation ***/

    if (print_) {
        outfile->Printf("\n    Starting first half-transformation.");
    }

    psio_->open(PSIF_TPDM_PRESORT, PSIO_OPEN_OLD);
    psio_->open(PSIF_TPDM_HALFTRANS, PSIO_OPEN_NEW);

    /*
     * (AA|AA) & (AA|aa) -> (AA|nn)
     */
    global_dpd_->buf4_init(&J1, PSIF_TPDM_PRESORT, 0, DPD_ID("[A>=A]+"), DPD_ID("[A,A]"),
                           DPD_ID("[A>=A]+"), DPD_ID("[A>=A]+"), 0, "MO TPDM (AA|AA)");
    //    global_dpd_->buf4_print(&J1, "outfile", 1);

    global_dpd_->buf4_init(&J2, PSIF_TPDM_PRESORT, 0, DPD_ID("[A>=A]+"), DPD_ID("[a,a]"),
                           DPD_ID("[A>=A]+"), DPD_ID("[a>=a]+"), 0, "MO TPDM (AA|aa)");
    //    global_dpd_->buf4_print(&J2, "outfile", 1);

    global_dpd_->buf4_init(&K, PSIF_TPDM_HALFTRANS, 0, DPD_ID("[A>=A]+"), DPD_ID("[n,n]"),
                           DPD_ID("[A>=A]+"), DPD_ID("[n>=n]+"), 0,
                           "Half-Transformed TPDM (AA|nn)");

    for (int h = 0; h < nirreps_; h++) {
        if (J1.params->coltot[h] && J1.params->rowtot[h]) {
            memFree =
                static_cast<size_t>(dpd_memfree() - J1.params->coltot[h] - K.params->coltot[h]);
            rowsPerBucket = memFree / (2 * J1.params->coltot[h]);
            if (rowsPerBucket > (size_t)J1.params->rowtot[h])
                rowsPerBucket = (size_t)J1.params->rowtot[h];
            nBuckets = static_cast<int>(ceil(static_cast<double>(J1.params->rowtot[h]) /
                                             static_cast<double>(rowsPerBucket)));
            rowsLeft = static_cast<size_t>(J1.params->rowtot[h] % rowsPerBucket);
        } else {
            nBuckets = 0;
            rowsPerBucket = 0;
            rowsLeft = 0;
        }

        if (print_ > 1) {
            outfile->Printf("\n    h = %d; memfree         = %lu", h, memFree);
            outfile->Printf("\n    h = %d; rows_per_bucket = %lu", h, rowsPerBucket);
            outfile->Printf("\n    h = %d; rows_left       = %lu", h, rowsLeft);
            outfile->Printf("\n    h = %d; nbuckets        = %d", h, nBuckets);
        }

        global_dpd_->buf4_mat_irrep_init_block(&K, h, rowsPerBucket);
        for (int n = 0; n < nBuckets; n++) {

            if (nBuckets == 1)
                thisBucketRows = rowsPerBucket;
            else
                thisBucketRows = (n < nBuckets - 1) ? rowsPerBucket : rowsLeft;

            global_dpd_->buf4_mat_irrep_init_block(&J1, h, rowsPerBucket);
            global_dpd_->buf4_mat_irrep_rd_block(&J1, h, n * rowsPerBucket, thisBucketRows);
            for (int pq = 0; pq < thisBucketRows; pq++) {
                for (int Gr = 0; Gr < nirreps_; Gr++) {
                    // Transform ( A A | A A ) -> ( A A | A n )
                    int Gs = h ^ Gr;
                    int nrows = sopi_[Gr];
                    int ncols = mopi_[Gs];
                    int nlinks = mopi_[Gs];
                    int rs = J1.col_offset[h][Gr];
                    double** pca = ca->pointer(Gs);
                    if (nrows && ncols && nlinks)
                        C_DGEMM('n', 't', nrows, ncols, nlinks, 1.0, &J1.matrix[h][pq][rs], nlinks,
                                pca[0], ncols, 0.0, TMP[0], nso_);

                    // Transform ( A A | A n ) -> ( A A | n n )
                    nrows = sopi_[Gr];
                    ncols = sopi_[Gs];
                    nlinks = mopi_[Gr];
                    rs = K.col_offset[h][Gr];
                    pca = ca->pointer(Gr);
                    if (nrows && ncols && nlinks)
                        C_DGEMM('n', 'n', nrows, ncols, nlinks, 1.0, pca[0], nrows, TMP[0], nso_,
                                0.0, &K.matrix[h][pq][rs], ncols);
                } /* Gr */
            }     /* pq */
            global_dpd_->buf4_mat_irrep_close_block(&J1, h, rowsPerBucket);

            global_dpd_->buf4_mat_irrep_init_block(&J2, h, rowsPerBucket);
            global_dpd_->buf4_mat_irrep_rd_block(&J2, h, n * rowsPerBucket, thisBucketRows);
            for (int pq = 0; pq < thisBucketRows; pq++) {
                for (int Gr = 0; Gr < nirreps_; Gr++) {
                    // Transform ( A A | a a ) -> ( A A | a n )
                    int Gs = h ^ Gr;
                    int nrows = sopi_[Gr];
                    int ncols = mopi_[Gs];
                    int nlinks = mopi_[Gs];
                    int rs = J2.col_offset[h][Gr];
                    double** pcb = cb->pointer(Gs);
                    if (nrows && ncols && nlinks)
                        C_DGEMM('n', 't', nrows, ncols, nlinks, 1.0, &J2.matrix[h][pq][rs], nlinks,
                                pcb[0], ncols, 0.0, TMP[0], nso_);

                    // Transform ( A A | a n ) -> ( A A | n n )
                    nrows = sopi_[Gr];
                    ncols = sopi_[Gs];
                    nlinks = mopi_[Gr];
                    rs = K.col_offset[h][Gr];
                    pcb = cb->pointer(Gr);
                    if (nrows && ncols && nlinks)
                        C_DGEMM('n', 'n', nrows, ncols, nlinks, 1.0, pcb[0], nrows, TMP[0], nso_,
                                1.0, &K.matrix[h][pq][rs], ncols);
                } /* Gr */
            }     /* pq */
            global_dpd_->buf4_mat_irrep_wrt_block(&K, h, n * rowsPerBucket, thisBucketRows);
            global_dpd_->buf4_mat_irrep_close_block(&J2, h, rowsPerBucket);
        }
        global_dpd_->buf4_mat_irrep_close_block(&K, h, rowsPerBucket);
    }
    global_dpd_->buf4_close(&K);
    global_dpd_->buf4_close(&J1);
    global_dpd_->buf4_close(&J2);

    /*
     * (aa|aa) -> (aa|nn)
     */
    global_dpd_->buf4_init(&J1, PSIF_TPDM_PRESORT, 0, DPD_ID("[a>=a]+"), DPD_ID("[a,a]"),
                           DPD_ID("[a>=a]+"), DPD_ID("[a>=a]+"), 0, "MO TPDM (aa|aa)");
    global_dpd_->buf4_init(&K, PSIF_TPDM_HALFTRANS, 0, DPD_ID("[a>=a]+"), DPD_ID("[n,n]"),
                           DPD_ID("[a>=a]+"), DPD_ID("[n>=n]+"), 0,
                           "Half-Transformed TPDM (aa|nn)");

    for (int h = 0; h < nirreps_; h++) {
        if (J1.params->coltot[h] && J1.params->rowtot[h]) {
            memFree =
                static_cast<size_t>(dpd_memfree() - J1.params->coltot[h] - K.params->coltot[h]);
            rowsPerBucket = memFree / (2 * J1.params->coltot[h]);
            if (rowsPerBucket > (size_t)J1.params->rowtot[h])
                rowsPerBucket = (size_t)J1.params->rowtot[h];
            nBuckets = static_cast<int>(ceil(static_cast<double>(J1.params->rowtot[h]) /
                                             static_cast<double>(rowsPerBucket)));
            rowsLeft = static_cast<size_t>(J1.params->rowtot[h] % rowsPerBucket);
        } else {
            nBuckets = 0;
            rowsPerBucket = 0;
            rowsLeft = 0;
        }

        if (print_ > 1) {
            outfile->Printf("\n    h = %d; memfree         = %lu", h, memFree);
            outfile->Printf("\n    h = %d; rows_per_bucket = %lu", h, rowsPerBucket);
            outfile->Printf("\n    h = %d; rows_left       = %lu", h, rowsLeft);
            outfile->Printf("\n    h = %d; nbuckets        = %d", h, nBuckets);
        }

        global_dpd_->buf4_mat_irrep_init_block(&K, h, rowsPerBucket);
        for (int n = 0; n < nBuckets; n++) {

            if (nBuckets == 1)
                thisBucketRows = rowsPerBucket;
            else
                thisBucketRows = (n < nBuckets - 1) ? rowsPerBucket : rowsLeft;

            global_dpd_->buf4_mat_irrep_init_block(&J1, h, rowsPerBucket);
            global_dpd_->buf4_mat_irrep_rd_block(&J1, h, n * rowsPerBucket, thisBucketRows);
            for (int pq = 0; pq < thisBucketRows; pq++) {
                for (int Gr = 0; Gr < nirreps_; Gr++) {
                    // Transform ( a a | a a ) -> ( a a | a n )
                    int Gs = h ^ Gr;
                    int nrows = sopi_[Gr];
                    int ncols = mopi_[Gs];
                    int nlinks = mopi_[Gs];
                    int rs = J1.col_offset[h][Gr];
                    double** pcb = cb->pointer(Gs);
                    if (nrows && ncols && nlinks)
                        C_DGEMM('n', 't', nrows, ncols, nlinks, 1.0, &J1.matrix[h][pq][rs], nlinks,
                                pcb[0], ncols, 0.0, TMP[0], nso_);

                    // Transform ( a a | a n ) -> ( a a | n n )
                    nrows = sopi_[Gr];
                    ncols = sopi_[Gs];
                    nlinks = mopi_[Gr];
                    rs = K.col_offset[h][Gr];
                    pcb = cb->pointer(Gr);
                    if (nrows && ncols && nlinks)
                        C_DGEMM('n', 'n', nrows, ncols, nlinks, 1.0, pcb[0], nrows, TMP[0], nso_,
                                0.0, &K.matrix[h][pq][rs], ncols);
                } /* Gr */
            }     /* pq */
            global_dpd_->buf4_mat_irrep_wrt_block(&K, h, n * rowsPerBucket, thisBucketRows);
            global_dpd_->buf4_mat_irrep_close_block(&J1, h, rowsPerBucket);
        }
        global_dpd_->buf4_mat_irrep_close_block(&K, h, rowsPerBucket);
    }
    global_dpd_->buf4_close(&K);
    global_dpd_->buf4_close(&J1);

    psio_->close(PSIF_TPDM_PRESORT, keepDpdMoTpdm_);

    if (print_) {
        outfile->Printf("\n    Sorting half-transformed TPDMs.");
    }

    global_dpd_->buf4_init(&K, PSIF_TPDM_HALFTRANS, 0, DPD_ID("[A>=A]+"), DPD_ID("[n>=n]+"),
                           DPD_ID("[A>=A]+"), DPD_ID("[n>=n]+"), 0,
                           "Half-Transformed TPDM (AA|nn)");
    //    global_dpd_->buf4_print(&K, "outfile", 1);
    global_dpd_->buf4_sort(&K, PSIF_TPDM_HALFTRANS, rspq, DPD_ID("[n>=n]+"), DPD_ID("[A>=A]+"),
                           "Half-Transformed TPDM (nn|AA)");
    global_dpd_->buf4_close(&K);

    global_dpd_->buf4_init(&K, PSIF_TPDM_HALFTRANS, 0, DPD_ID("[a>=a]+"), DPD_ID("[n>=n]+"),
                           DPD_ID("[a>=a]+"), DPD_ID("[n>=n]+"), 0,
                           "Half-Transformed TPDM (aa|nn)");
    global_dpd_->buf4_sort(&K, PSIF_TPDM_HALFTRANS, rspq, DPD_ID("[n>=n]+"), DPD_ID("[a>=a]+"),
                           "Half-Transformed TPDM (nn|aa)");
    global_dpd_->buf4_close(&K);

    if (print_) {
        outfile->Printf("\n    First half integral transformation complete.");
        outfile->Printf("\n    Starting second half-transformation.");
    }

    psio_->open(PSIF_AO_TPDM, PSIO_OPEN_NEW);

    /*
     * (nn|AA) & (nn|aa) -> (nn|nn)
     */
    global_dpd_->buf4_init(&J1, PSIF_TPDM_HALFTRANS, 0, DPD_ID("[n>=n]+"), DPD_ID("[A,A]"),
                           DPD_ID("[n>=n]+"), DPD_ID("[A>=A]+"), 0,
                           "Half-Transformed TPDM (nn|AA)");
    global_dpd_->buf4_init(&J2, PSIF_TPDM_HALFTRANS, 0, DPD_ID("[n>=n]+"), DPD_ID("[a,a]"),
                           DPD_ID("[n>=n]+"), DPD_ID("[a>=a]+"), 0,
                           "Half-Transformed TPDM (nn|aa)");
    global_dpd_->buf4_init(&K, PSIF_AO_TPDM, 0, DPD_ID("[n>=n]+"), DPD_ID("[n,n]"),
                           DPD_ID("[n>=n]+"), DPD_ID("[n>=n]+"), 0, "SO Basis TPDM (nn|nn)");

    for (int h = 0; h < nirreps_; h++) {
        if (J1.params->coltot[h] && J1.params->rowtot[h]) {
            memFree =
                static_cast<size_t>(dpd_memfree() - J1.params->coltot[h] - K.params->coltot[h]);
            rowsPerBucket = memFree / (2 * J1.params->coltot[h]);
            if (rowsPerBucket > (size_t)J1.params->rowtot[h])
                rowsPerBucket = static_cast<size_t>(J1.params->rowtot[h]);
            nBuckets = static_cast<int>(ceil(static_cast<double>(J1.params->rowtot[h]) /
                                             static_cast<double>(rowsPerBucket)));
            rowsLeft = static_cast<size_t>(J1.params->rowtot[h] % rowsPerBucket);
        } else {
            nBuckets = 0;
            rowsPerBucket = 0;
            rowsLeft = 0;
        }

        if (print_ > 1) {
            outfile->Printf("\n    h = %d; memfree         = %lu", h, memFree);
            outfile->Printf("\n    h = %d; rows_per_bucket = %lu", h, rowsPerBucket);
            outfile->Printf("\n    h = %d; rows_left       = %lu", h, rowsLeft);
            outfile->Printf("\n    h = %d; nbuckets        = %d", h, nBuckets);
        }

        global_dpd_->buf4_mat_irrep_init_block(&K, h, rowsPerBucket);

        for (int n = 0; n < nBuckets; n++) {
            if (nBuckets == 1)
                thisBucketRows = rowsPerBucket;
            else
                thisBucketRows = (n < nBuckets - 1) ? rowsPerBucket : rowsLeft;

            global_dpd_->buf4_mat_irrep_init_block(&J1, h, rowsPerBucket);
            global_dpd_->buf4_mat_irrep_rd_block(&J1, h, n * rowsPerBucket, thisBucketRows);
            for (int pq = 0; pq < thisBucketRows; pq++) {
                for (int Gr = 0; Gr < nirreps_; Gr++) {
                    // Transform ( n n | A A ) -> ( n n | A n )
                    int Gs = h ^ Gr;
                    int nrows = sopi_[Gr];
                    int ncols = mopi_[Gs];
                    int nlinks = mopi_[Gs];
                    int rs = J1.col_offset[h][Gr];
                    double** pca = ca->pointer(Gs);
                    if (nrows && ncols && nlinks)
                        C_DGEMM('n', 't', nrows, ncols, nlinks, 1.0, &J1.matrix[h][pq][rs], nlinks,
                                pca[0], ncols, 0.0, TMP[0], nso_);

                    // Transform ( n n | n A ) -> ( n n | n n )
                    nrows = sopi_[Gr];
                    ncols = sopi_[Gs];
                    nlinks = mopi_[Gr];
                    rs = K.col_offset[h][Gr];
                    pca = ca->pointer(Gr);
                    if (nrows && ncols && nlinks)
                        C_DGEMM('n', 'n', nrows, ncols, nlinks, 1.0, pca[0], nrows, TMP[0], nso_,
                                0.0, &K.matrix[h][pq][rs], ncols);
                } /* Gr */
            }     /* pq */
            global_dpd_->buf4_mat_irrep_close_block(&J1, h, rowsPerBucket);

            global_dpd_->buf4_mat_irrep_init_block(&J2, h, rowsPerBucket);
            global_dpd_->buf4_mat_irrep_rd_block(&J2, h, n * rowsPerBucket, thisBucketRows);
            for (int pq = 0; pq < thisBucketRows; pq++) {
                //                int PQ = n * rowsPerBucket + pq; // The absolute pq value
                for (int Gr = 0; Gr < nirreps_; Gr++) {
                    // Transform ( n n | a a ) -> ( n n | a n )
                    int Gs = h ^ Gr;
                    int nrows = sopi_[Gr];
                    int ncols = mopi_[Gs];
                    int nlinks = mopi_[Gs];
                    int rs = J2.col_offset[h][Gr];
                    double** pcb = cb->pointer(Gs);
                    if (nrows && ncols && nlinks)
                        C_DGEMM('n', 't', nrows, ncols, nlinks, 1.0, &J2.matrix[h][pq][rs], nlinks,
                                pcb[0], ncols, 0.0, TMP[0], nso_);

                    // Transform ( n n | n a ) -> ( n n | n n )
                    nrows = sopi_[Gr];
                    ncols = sopi_[Gs];
                    nlinks = mopi_[Gr];
                    rs = K.col_offset[h][Gr];
                    pcb = cb->pointer(Gr);
                    if (nrows && ncols && nlinks)
                        C_DGEMM('n', 'n', nrows, ncols, nlinks, 1.0, pcb[0], nrows, TMP[0], nso_,
                                1.0, &K.matrix[h][pq][rs], ncols);
                } /* Gr */
            }     /* pq */
            global_dpd_->buf4_mat_irrep_close_block(&J2, h, rowsPerBucket);
            sort_so_tpdm(&K, h, n * rowsPerBucket, thisBucketRows, (h == 0 && n == 0));
            if (write_dpd_so_tpdm_)
                global_dpd_->buf4_mat_irrep_wrt_block(&K, h, n * rowsPerBucket, thisBucketRows);
        }
        global_dpd_->buf4_mat_irrep_close_block(&K, h, rowsPerBucket);
    }

    if (print_) {
        outfile->Printf("\n    Second half integral transformation complete.\n");
    }

    global_dpd_->buf4_close(&K);
    global_dpd_->buf4_close(&J1);
    global_dpd_->buf4_close(&J2);

    free_block(TMP);

    psio_->close(PSIF_TPDM_HALFTRANS, keepHtTpdm_);
    psio_->close(PSIF_AO_TPDM, 1);

    // Hand DPD control back to the user
    dpd_set_default(currentActiveDPD);
}

// repack the unrestricted MO TPDMs into DPD buffers
// to prepare them for the back transformation in deriv()
// this function is a modifield version of
// IntegralTransform::presort_mo_tpdm_unrestricted()
// from libtrans/integraltransform_sort_mo_tpdm.cc
void TPDMBackTransform::presort_mo_tpdm_unrestricted() {

    check_initialized();

    int currentActiveDPD = psi::dpd_default;
    dpd_set_default(myDPDNum_);

    if (print_) {
        outfile->Printf("\n    Presorting MO-basis TPDM.");
    }

    dpdfile4 I;
    psio_->open(PSIF_TPDM_PRESORT, PSIO_OPEN_NEW);
    global_dpd_->file4_init(&I, PSIF_TPDM_PRESORT, 0, DPD_ID("[A>=A]+"), DPD_ID("[A>=A]+"),
                            "MO TPDM (AA|AA)");

    size_t memoryd = memory_ / sizeof(double);

    int nump = 0, numq = 0;
    for (int h = 0; h < nirreps_; ++h) {
        nump += I.params->ppi[h];
        numq += I.params->qpi[h];
    }
    int** bucketMap = init_int_matrix(nump, numq);

    /* Room for one bucket to begin with */
    int** bucketOffset = (int**)malloc(sizeof(int*));
    bucketOffset[0] = init_int_array(nirreps_);
    int** bucketRowDim = (int**)malloc(sizeof(int*));
    bucketRowDim[0] = init_int_array(nirreps_);
    int** bucketSize = (int**)malloc(sizeof(int*));
    bucketSize[0] = init_int_array(nirreps_);

    /* Figure out how many passes we need and where each p,q goes */
    int nBuckets = 1;
    size_t coreLeft = memoryd;
    psio_address next;
    for (int h = 0; h < nirreps_; ++h) {
        size_t rowLength = (size_t)I.params->coltot[h ^ (I.my_irrep)];

        for (int row = 0; row < I.params->rowtot[h]; ++row) {
            if (coreLeft >= rowLength) {
                coreLeft -= rowLength;
                bucketRowDim[nBuckets - 1][h]++;
                bucketSize[nBuckets - 1][h] += rowLength;
            } else {
                nBuckets++;
                coreLeft = memoryd - rowLength;
                /* Make room for another bucket */
                bucketOffset = (int**)realloc((void*)bucketOffset, nBuckets * sizeof(int*));
                bucketOffset[nBuckets - 1] = init_int_array(nirreps_);
                bucketOffset[nBuckets - 1][h] = row;

                bucketRowDim = (int**)realloc((void*)bucketRowDim, nBuckets * sizeof(int*));
                bucketRowDim[nBuckets - 1] = init_int_array(nirreps_);
                bucketRowDim[nBuckets - 1][h] = 1;

                bucketSize = (int**)realloc((void*)bucketSize, nBuckets * sizeof(int*));
                bucketSize[nBuckets - 1] = init_int_array(nirreps_);
                bucketSize[nBuckets - 1][h] = rowLength;
            }
            int p = I.params->roworb[h][row][0];
            int q = I.params->roworb[h][row][1];
            bucketMap[p][q] = nBuckets - 1;
        }
    }

    if (print_) {
        outfile->Printf("\n    Sorting File: %s nbuckets = %d", I.label, nBuckets);
    }

    // The alpha - alpha spin case
    next = PSIO_ZERO;
    for (int n = 0; n < nBuckets; ++n) { /* nbuckets = number of passes */
        /* Prepare target matrix */

        for (int h = 0; h < nirreps_; h++) {
            I.matrix[h] = block_matrix(bucketRowDim[n][h], I.params->coltot[h]);
        }
        IWL* iwl = new IWL(psio_.get(), PSIF_MO_AA_TPDM, tolerance_, 1, 0);
        DPDFillerFunctor aaDpdFiller(&I, n, bucketMap, bucketOffset, true, true);

        Label* lblptr = iwl->labels();
        Value* valptr = iwl->values();
        int lastbuf;

        /* Now run through the IWL buffers */
        do {
            iwl->fetch();
            lastbuf = iwl->last_buffer();
            for (int index = 0; index < iwl->buffer_count(); ++index) {
                int labelIndex = 4 * index;
                int p =
                    (int)lblptr[labelIndex++]; // aCorrToPitzer_[abs((int) lblptr[labelIndex++])];
                int q = (int)lblptr[labelIndex++]; // aCorrToPitzer_[(int) lblptr[labelIndex++]];
                int r = (int)lblptr[labelIndex++]; // aCorrToPitzer_[(int) lblptr[labelIndex++]];
                int s = (int)lblptr[labelIndex++]; // aCorrToPitzer_[(int) lblptr[labelIndex++]];
                double value = (double)valptr[index];
                aaDpdFiller(p, q, r, s, value);
            }               /* end loop through current buffer */
        } while (!lastbuf); /* end loop over reading buffers */

        iwl->set_keep_flag(1);
        delete iwl;

        for (int h = 0; h < nirreps_; ++h) {
            if (bucketSize[n][h])
                psio_->write(I.filenum, I.label, (char*)I.matrix[h][0],
                             bucketSize[n][h] * ((long int)sizeof(double)), next, &next);
            free_block(I.matrix[h]);
        }

    } /* end loop over buckets/passes */

    /* Get rid of the input integral file */
    psio_->open(PSIF_MO_AA_TPDM, PSIO_OPEN_OLD);
    psio_->close(PSIF_MO_AA_TPDM, keepIwlMoTpdm_);

    // The alpha - beta spin case
    global_dpd_->file4_init(&I, PSIF_TPDM_PRESORT, 0, DPD_ID("[A>=A]+"), DPD_ID("[a>=a]+"),
                            "MO TPDM (AA|aa)");

    if (print_) {
        outfile->Printf("\n    Sorting File: %s nbuckets = %d", I.label, nBuckets);
    }

    next = PSIO_ZERO;
    for (int n = 0; n < nBuckets; ++n) { /* nbuckets = number of passes */
        /* Prepare target matrix */
        for (int h = 0; h < nirreps_; h++) {
            I.matrix[h] = block_matrix(bucketRowDim[n][h], I.params->coltot[h]);
        }
        IWL* iwl = new IWL(psio_.get(), PSIF_MO_AB_TPDM, tolerance_, 1, 0);
        DPDFillerFunctor abDpdFiller(&I, n, bucketMap, bucketOffset, true, false);

        Label* lblptr = iwl->labels();
        Value* valptr = iwl->values();
        int lastbuf;
        /* Now run through the IWL buffers */
        do {
            iwl->fetch();
            lastbuf = iwl->last_buffer();
            for (int index = 0; index < iwl->buffer_count(); ++index) {
                int labelIndex = 4 * index;
                int p =
                    (int)lblptr[labelIndex++]; // aCorrToPitzer_[abs((int) lblptr[labelIndex++])];
                int q = (int)lblptr[labelIndex++]; // aCorrToPitzer_[(int) lblptr[labelIndex++]];
                int r = (int)lblptr[labelIndex++]; // bCorrToPitzer_[(int) lblptr[labelIndex++]];
                int s = (int)lblptr[labelIndex++]; // bCorrToPitzer_[(int) lblptr[labelIndex++]];
                double value = (double)valptr[index];
                abDpdFiller(p, q, r, s, value);
            }               /* end loop through current buffer */
        } while (!lastbuf); /* end loop over reading buffers */
        iwl->set_keep_flag(1);
        delete iwl;

        for (int h = 0; h < nirreps_; ++h) {
            if (bucketSize[n][h])
                psio_->write(I.filenum, I.label, (char*)I.matrix[h][0],
                             bucketSize[n][h] * ((long int)sizeof(double)), next, &next);
            free_block(I.matrix[h]);
        }
    } /* end loop over buckets/passes */

    /* Get rid of the input integral file */
    psio_->open(PSIF_MO_AB_TPDM, PSIO_OPEN_OLD);
    psio_->close(PSIF_MO_AB_TPDM, keepIwlMoTpdm_);

    // The beta - beta spin case
    global_dpd_->file4_init(&I, PSIF_TPDM_PRESORT, 0, DPD_ID("[a>=a]+"), DPD_ID("[a>=a]+"),
                            "MO TPDM (aa|aa)");

    if (print_) {
        outfile->Printf("\n    Sorting File: %s nbuckets = %d", I.label, nBuckets);
    }

    next = PSIO_ZERO;
    for (int n = 0; n < nBuckets; ++n) { /* nbuckets = number of passes */
        /* Prepare target matrix */
        for (int h = 0; h < nirreps_; h++) {
            I.matrix[h] = block_matrix(bucketRowDim[n][h], I.params->coltot[h]);
        }
        IWL* iwl = new IWL(psio_.get(), PSIF_MO_BB_TPDM, tolerance_, 1, 0);
        DPDFillerFunctor bbDpdFiller(&I, n, bucketMap, bucketOffset, true, true);

        Label* lblptr = iwl->labels();
        Value* valptr = iwl->values();
        int lastbuf;
        /* Now run through the IWL buffers */
        do {
            iwl->fetch();
            lastbuf = iwl->last_buffer();
            for (int index = 0; index < iwl->buffer_count(); ++index) {
                int labelIndex = 4 * index;
                int p =
                    (int)lblptr[labelIndex++]; // bCorrToPitzer_[abs((int) lblptr[labelIndex++])];
                int q = (int)lblptr[labelIndex++]; // bCorrToPitzer_[(int) lblptr[labelIndex++]];
                int r = (int)lblptr[labelIndex++]; // bCorrToPitzer_[(int) lblptr[labelIndex++]];
                int s = (int)lblptr[labelIndex++]; // bCorrToPitzer_[(int) lblptr[labelIndex++]];
                double value = (double)valptr[index];
                bbDpdFiller(p, q, r, s, value);
            }               /* end loop through current buffer */
        } while (!lastbuf); /* end loop over reading buffers */
        iwl->set_keep_flag(1);
        delete iwl;

        for (int h = 0; h < nirreps_; ++h) {
            if (bucketSize[n][h])
                psio_->write(I.filenum, I.label, (char*)I.matrix[h][0],
                             bucketSize[n][h] * ((long int)sizeof(double)), next, &next);
            free_block(I.matrix[h]);
        }
    } /* end loop over buckets/passes */

    /* Get rid of the input integral file */
    psio_->open(PSIF_MO_BB_TPDM, PSIO_OPEN_OLD);
    psio_->close(PSIF_MO_BB_TPDM, keepIwlMoTpdm_);

    free_int_matrix(bucketMap);

    for (int n = 0; n < nBuckets; ++n) {
        free(bucketOffset[n]);
        free(bucketRowDim[n]);
        free(bucketSize[n]);
    }
    free(bucketOffset);
    free(bucketRowDim);
    free(bucketSize);

    dpd_set_default(currentActiveDPD);

    tpdmAlreadyPresorted_ = true;

    global_dpd_->file4_close(&I);
    psio_->close(PSIF_TPDM_PRESORT, 1);
}
