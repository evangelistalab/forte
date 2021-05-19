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

void TPDMBackTransform::backtransform_tpdm_restricted() {
    check_initialized();

    // This can be safely called - it returns immediately if the MO TPDM is already sorted
    presort_mo_tpdm_restricted();

    // Grab the transformation coefficients
    SharedMatrix ca = aMOCoefficients_[MOSPACE_ALL];

    dpdbuf4 J, K, J2;

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
        outfile->Printf("\n    Starting first half-transformation (MCSCF).");
    }

    psio_->open(PSIF_FORTE_TPDM_PRESORT, PSIO_OPEN_OLD);
    psio_->open(PSIF_FORTE_TPDM_HALFTRANS, PSIO_OPEN_NEW);

    /*** (AA|aa) -> (AA|nn) ***/

    global_dpd_->buf4_init(&J, PSIF_FORTE_TPDM_PRESORT, 0, DPD_ID("[A>=A]+"), DPD_ID("[A,A]"),
                           DPD_ID("[A>=A]+"), DPD_ID("[A>=A]+"), 0, "MO TPDM (AA|AA)");
    if (print_ > 2) {
        global_dpd_->buf4_print(&J, "outfile", 1);
    }

    global_dpd_->buf4_init(&K, PSIF_FORTE_TPDM_HALFTRANS, 0, DPD_ID("[A>=A]+"), DPD_ID("[n,n]"),
                           DPD_ID("[A>=A]+"), DPD_ID("[n>=n]+"), 0,
                           "Half-Transformed TPDM (AA|nn)");

    for (int h = 0; h < nirreps_; h++) {
        size_t Jcol = J.params->coltot[h];
        size_t Jrow = J.params->rowtot[h];

        if (Jcol && Jrow) {
            memFree = static_cast<size_t>(dpd_memfree() - Jcol - K.params->coltot[h]);
            rowsPerBucket = memFree / (2 * Jcol);
            rowsPerBucket = rowsPerBucket > Jrow ? Jrow : rowsPerBucket;
            nBuckets = static_cast<int>(
                ceil(static_cast<double>(Jrow) / static_cast<double>(rowsPerBucket)));
            rowsLeft = static_cast<size_t>(Jrow % rowsPerBucket);
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

            global_dpd_->buf4_mat_irrep_init_block(&J, h, rowsPerBucket);
            global_dpd_->buf4_mat_irrep_rd_block(&J, h, n * rowsPerBucket, thisBucketRows);
            for (int pq = 0; pq < thisBucketRows; pq++) {
                for (int Gr = 0; Gr < nirreps_; Gr++) {
                    // Transform ( A A | a a ) -> ( A A | a n )
                    int Gs = h ^ Gr;
                    int nrows = sopi_[Gr];
                    int ncols = mopi_[Gs];
                    int nlinks = mopi_[Gs];
                    int rs = J.col_offset[h][Gr];
                    double** pca = ca->pointer(Gs);
                    if (nrows && ncols && nlinks)
                        C_DGEMM('n', 't', nrows, ncols, nlinks, 1.0, &J.matrix[h][pq][rs], nlinks,
                                pca[0], ncols, 0.0, TMP[0], nso_);

                    // Transform ( A A | a n ) -> ( A A | n n )
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
            global_dpd_->buf4_mat_irrep_wrt_block(&K, h, n * rowsPerBucket, thisBucketRows);
            global_dpd_->buf4_mat_irrep_close_block(&J, h, rowsPerBucket);
        }
        global_dpd_->buf4_mat_irrep_close_block(&K, h, rowsPerBucket);
    }
    global_dpd_->buf4_close(&K);
    global_dpd_->buf4_close(&J);

    psio_->close(PSIF_FORTE_TPDM_PRESORT, keepDpdMoTpdm_);

    // if there is additional set of orbitals
    if (Ca_forte_ != nullptr) {
        psio_->open(PSIF_FORTE_TPDM_PRESORT2, PSIO_OPEN_OLD);
        psio_->open(PSIF_FORTE_TPDM_HALFTRANS2, PSIO_OPEN_NEW);

        if (print_) {
            outfile->Printf("\n    Starting first half-transformation (SCF).");
        }

        global_dpd_->buf4_init(&J, PSIF_FORTE_TPDM_PRESORT2, 0, DPD_ID("[A>=A]+"), DPD_ID("[A,A]"),
                               DPD_ID("[A>=A]+"), DPD_ID("[A>=A]+"), 0, "MO TPDM (AA|AA)");
        global_dpd_->buf4_init(&K, PSIF_FORTE_TPDM_HALFTRANS2, 0, DPD_ID("[A>=A]+"),
                               DPD_ID("[n,n]"), DPD_ID("[A>=A]+"), DPD_ID("[n>=n]+"), 0,
                               "Half-Transformed TPDM (AA|nn)");

        for (int h = 0; h < nirreps_; h++) {
            size_t Jcol = J.params->coltot[h];
            size_t Jrow = J.params->rowtot[h];

            if (Jcol && Jrow) {
                memFree = static_cast<size_t>(dpd_memfree() - Jcol - K.params->coltot[h]);
                rowsPerBucket = memFree / (2 * Jcol);
                rowsPerBucket = rowsPerBucket > Jrow ? Jrow : rowsPerBucket;
                nBuckets = static_cast<int>(
                    ceil(static_cast<double>(Jrow) / static_cast<double>(rowsPerBucket)));
                rowsLeft = static_cast<size_t>(Jrow % rowsPerBucket);
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

                global_dpd_->buf4_mat_irrep_init_block(&J, h, rowsPerBucket);
                global_dpd_->buf4_mat_irrep_rd_block(&J, h, n * rowsPerBucket, thisBucketRows);
                for (int pq = 0; pq < thisBucketRows; pq++) {
                    for (int Gr = 0; Gr < nirreps_; Gr++) {
                        // Transform ( A A | a a ) -> ( A A | a n )
                        int Gs = h ^ Gr;
                        int nrows = sopi_[Gr];
                        int ncols = mopi_[Gs];
                        int nlinks = mopi_[Gs];
                        int rs = J.col_offset[h][Gr];
                        double** pca = Ca_forte_->pointer(Gs);
                        if (nrows && ncols && nlinks)
                            C_DGEMM('n', 't', nrows, ncols, nlinks, 1.0, &J.matrix[h][pq][rs],
                                    nlinks, pca[0], ncols, 0.0, TMP[0], nso_);

                        // Transform ( A A | a n ) -> ( A A | n n )
                        nrows = sopi_[Gr];
                        ncols = sopi_[Gs];
                        nlinks = mopi_[Gr];
                        rs = K.col_offset[h][Gr];
                        pca = Ca_forte_->pointer(Gr);
                        if (nrows && ncols && nlinks)
                            C_DGEMM('n', 'n', nrows, ncols, nlinks, 1.0, pca[0], nrows, TMP[0],
                                    nso_, 0.0, &K.matrix[h][pq][rs], ncols);
                    } /* Gr */
                }     /* pq */
                global_dpd_->buf4_mat_irrep_wrt_block(&K, h, n * rowsPerBucket, thisBucketRows);
                global_dpd_->buf4_mat_irrep_close_block(&J, h, rowsPerBucket);
            }
            global_dpd_->buf4_mat_irrep_close_block(&K, h, rowsPerBucket);
        }
        global_dpd_->buf4_close(&K);
        global_dpd_->buf4_close(&J);

        psio_->close(PSIF_FORTE_TPDM_PRESORT2, keepDpdMoTpdm_);
    }

    if (print_) {
        outfile->Printf("\n    Sorting half-transformed TPDMs.");
    }

    global_dpd_->buf4_init(&K, PSIF_FORTE_TPDM_HALFTRANS, 0, DPD_ID("[A>=A]+"), DPD_ID("[n>=n]+"),
                           DPD_ID("[A>=A]+"), DPD_ID("[n>=n]+"), 0,
                           "Half-Transformed TPDM (AA|nn)");
    global_dpd_->buf4_sort(&K, PSIF_FORTE_TPDM_HALFTRANS, rspq, DPD_ID("[n>=n]+"),
                           DPD_ID("[A>=A]+"), "Half-Transformed TPDM (nn|AA)");
    global_dpd_->buf4_close(&K);

    if (Ca_forte_ != nullptr) {
        global_dpd_->buf4_init(&K, PSIF_FORTE_TPDM_HALFTRANS2, 0, DPD_ID("[A>=A]+"),
                               DPD_ID("[n>=n]+"), DPD_ID("[A>=A]+"), DPD_ID("[n>=n]+"), 0,
                               "Half-Transformed TPDM (AA|nn)");
        global_dpd_->buf4_sort(&K, PSIF_FORTE_TPDM_HALFTRANS2, rspq, DPD_ID("[n>=n]+"),
                               DPD_ID("[A>=A]+"), "Half-Transformed TPDM (nn|AA)");
        global_dpd_->buf4_close(&K);
    }

    if (print_) {
        outfile->Printf("\n    First half integral transformation complete.");
        outfile->Printf("\n    Starting second half-transformation.");
    }

    psio_->open(PSIF_AO_TPDM, PSIO_OPEN_NEW);

    /*** (nn|AA) -> (nn|nn) ***/

    global_dpd_->buf4_init(&J, PSIF_FORTE_TPDM_HALFTRANS, 0, DPD_ID("[n>=n]+"), DPD_ID("[A,A]"),
                           DPD_ID("[n>=n]+"), DPD_ID("[A>=A]+"), 0,
                           "Half-Transformed TPDM (nn|AA)");
    global_dpd_->buf4_init(&K, PSIF_AO_TPDM, 0, DPD_ID("[n>=n]+"), DPD_ID("[n,n]"),
                           DPD_ID("[n>=n]+"), DPD_ID("[n>=n]+"), 0, "SO Basis TPDM (nn|nn)");

    if (Ca_forte_ != nullptr) {
        global_dpd_->buf4_init(&J2, PSIF_FORTE_TPDM_HALFTRANS2, 0, DPD_ID("[n>=n]+"),
                               DPD_ID("[A,A]"), DPD_ID("[n>=n]+"), DPD_ID("[A>=A]+"), 0,
                               "Half-Transformed TPDM (nn|AA)");
    }

    for (int h = 0; h < nirreps_; h++) {
        size_t Jcol = J.params->coltot[h];
        size_t Jrow = J.params->rowtot[h];

        if (Jcol && Jrow) {
            memFree = static_cast<size_t>(dpd_memfree() - Jcol - K.params->coltot[h]);
            rowsPerBucket = memFree / (2 * Jcol);
            rowsPerBucket = rowsPerBucket > Jrow ? Jrow : rowsPerBucket;
            nBuckets = static_cast<int>(
                ceil(static_cast<double>(Jrow) / static_cast<double>(rowsPerBucket)));
            rowsLeft = static_cast<size_t>(Jrow % rowsPerBucket);
        } else {
            nBuckets = 0;
            rowsPerBucket = 0;
            rowsLeft = 0;
        }

        if (print_ > 1) {
            outfile->Printf("\th = %d; memfree         = %lu\n", h, memFree);
            outfile->Printf("\th = %d; rows_per_bucket = %lu\n", h, rowsPerBucket);
            outfile->Printf("\th = %d; rows_left       = %lu\n", h, rowsLeft);
            outfile->Printf("\th = %d; nbuckets        = %d\n", h, nBuckets);
        }

        global_dpd_->buf4_mat_irrep_init_block(&K, h, rowsPerBucket);

        for (int n = 0; n < nBuckets; n++) {
            if (nBuckets == 1)
                thisBucketRows = rowsPerBucket;
            else
                thisBucketRows = (n < nBuckets - 1) ? rowsPerBucket : rowsLeft;

            global_dpd_->buf4_mat_irrep_init_block(&J, h, rowsPerBucket);
            global_dpd_->buf4_mat_irrep_rd_block(&J, h, n * rowsPerBucket, thisBucketRows);
            for (int pq = 0; pq < thisBucketRows; pq++) {
                for (int Gr = 0; Gr < nirreps_; Gr++) {
                    // Transform ( n n | A A ) -> ( n n | A n )
                    int Gs = h ^ Gr;
                    int nrows = sopi_[Gr];
                    int ncols = mopi_[Gs];
                    int nlinks = mopi_[Gs];
                    int rs = J.col_offset[h][Gr];
                    double** pca = ca->pointer(Gs);
                    if (nrows && ncols && nlinks)
                        C_DGEMM('n', 't', nrows, ncols, nlinks, 1.0, &J.matrix[h][pq][rs], nlinks,
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
            global_dpd_->buf4_mat_irrep_close_block(&J, h, rowsPerBucket);

            if (Ca_forte_ != nullptr) {
                global_dpd_->buf4_mat_irrep_init_block(&J2, h, rowsPerBucket);
                global_dpd_->buf4_mat_irrep_rd_block(&J2, h, n * rowsPerBucket, thisBucketRows);
                for (int pq = 0; pq < thisBucketRows; pq++) {
                    for (int Gr = 0; Gr < nirreps_; Gr++) {
                        // Transform ( n n | a a ) -> ( n n | a n )
                        int Gs = h ^ Gr;
                        int nrows = sopi_[Gr];
                        int ncols = mopi_[Gs];
                        int nlinks = mopi_[Gs];
                        int rs = J2.col_offset[h][Gr];
                        double** pca = Ca_forte_->pointer(Gs);
                        if (nrows && ncols && nlinks)
                            C_DGEMM('n', 't', nrows, ncols, nlinks, 1.0, &J2.matrix[h][pq][rs],
                                    nlinks, pca[0], ncols, 0.0, TMP[0], nso_);

                        // Transform ( n n | n a ) -> ( n n | n n )
                        nrows = sopi_[Gr];
                        ncols = sopi_[Gs];
                        nlinks = mopi_[Gr];
                        rs = K.col_offset[h][Gr];
                        pca = Ca_forte_->pointer(Gr);
                        if (nrows && ncols && nlinks)
                            C_DGEMM('n', 'n', nrows, ncols, nlinks, 1.0, pca[0], nrows, TMP[0],
                                    nso_, 1.0, &K.matrix[h][pq][rs], ncols);
                    } /* Gr */
                }     /* pq */
                global_dpd_->buf4_mat_irrep_close_block(&J2, h, rowsPerBucket);
            }

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
    global_dpd_->buf4_close(&J);
    if (Ca_forte_ != nullptr) {
        global_dpd_->buf4_close(&J2);
        psio_->close(PSIF_FORTE_TPDM_HALFTRANS2, keepHtTpdm_);
    }

    free_block(TMP);

    psio_->close(PSIF_FORTE_TPDM_HALFTRANS, keepHtTpdm_);
    psio_->close(PSIF_AO_TPDM, 1);

    // Hand DPD control back to the user
    dpd_set_default(currentActiveDPD);
}

void TPDMBackTransform::presort_mo_tpdm_restricted() {
    check_initialized();

    int currentActiveDPD = psi::dpd_default;
    dpd_set_default(myDPDNum_);

    if (print_) {
        outfile->Printf("\n    Presorting MO-basis TPDM.");
    }

    dpdfile4 I;
    psio_->open(PSIF_FORTE_TPDM_PRESORT, PSIO_OPEN_NEW);
    global_dpd_->file4_init(&I, PSIF_FORTE_TPDM_PRESORT, 0, DPD_ID("[A>=A]+"), DPD_ID("[A>=A]+"),
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
        size_t rowLength = I.params->coltot[h ^ (I.my_irrep)];

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

    next = PSIO_ZERO;
    for (int n = 0; n < nBuckets; ++n) { /* nbuckets = number of passes */
        /* Prepare target matrix */
        for (int h = 0; h < nirreps_; h++) {
            I.matrix[h] = block_matrix(bucketRowDim[n][h], I.params->coltot[h]);
        }
        IWL* iwl = new IWL(psio_.get(), PSIF_FORTE_MO_TPDM, tolerance_, 1, 0);

        // symmetrize (1st Boolean): scale by 0.5 when p != q and by another 0.5 when r != s
        // bra-ket (2nd Boolean): add value when p != r or q != s
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
                int p = lblptr[labelIndex++];
                int q = lblptr[labelIndex++];
                int r = lblptr[labelIndex++];
                int s = lblptr[labelIndex++];
                double value = (double)valptr[index];
                if (print_ > 1) {
                    outfile->Printf("\tp%4d q%4d r%4d s%4d = %20.10f\n", p, q, r, s, value);
                }
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
    psio_->open(PSIF_FORTE_MO_TPDM, PSIO_OPEN_OLD);
    psio_->close(PSIF_FORTE_MO_TPDM, keepIwlMoTpdm_);

    // if there are additional set of orbitals
    if (Ca_forte_ != nullptr) {
        psio_->open(PSIF_FORTE_TPDM_PRESORT2, PSIO_OPEN_NEW);
        global_dpd_->file4_init(&I, PSIF_FORTE_TPDM_PRESORT2, 0, DPD_ID("[A>=A]+"),
                                DPD_ID("[A>=A]+"), "MO TPDM (AA|AA)");

        if (print_) {
            outfile->Printf("\n    Sorting File: %s nbuckets = %d", I.label, nBuckets);
        }

        next = PSIO_ZERO;
        for (int n = 0; n < nBuckets; ++n) { /* nbuckets = number of passes */
            /* Prepare target matrix */
            for (int h = 0; h < nirreps_; h++) {
                I.matrix[h] = block_matrix(bucketRowDim[n][h], I.params->coltot[h]);
            }
            IWL* iwl = new IWL(psio_.get(), PSIF_FORTE_MO_TPDM2, tolerance_, 1, 0);

            // symmetrize (1st Boolean): scale by 0.5 when p != q and by another 0.5 when r != s
            // bra-ket (2nd Boolean): add value when p != r or q != s
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
                    int p = lblptr[labelIndex++];
                    int q = lblptr[labelIndex++];
                    int r = lblptr[labelIndex++];
                    int s = lblptr[labelIndex++];
                    double value = (double)valptr[index];
                    if (print_ > 1) {
                        outfile->Printf("\tp%4d q%4d r%4d s%4d = %20.10f\n", p, q, r, s, value);
                    }
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
        psio_->open(PSIF_FORTE_MO_TPDM2, PSIO_OPEN_OLD);
        psio_->close(PSIF_FORTE_MO_TPDM2, keepIwlMoTpdm_);

        psio_->close(PSIF_FORTE_TPDM_PRESORT2, 1);
    }

    // finalize and clean up
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
    psio_->close(PSIF_FORTE_TPDM_PRESORT, 1);
}
