/*
 *@BEGIN LICENSE
 *
 * v2RDM-CASSCF, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
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
 * Copyright (c) 2014, The Florida State University. All rights reserved.
 *
 *@END LICENSE
 *
 */

#include "backtransform_tpdm.h"
#include <psi4/libtrans/integraltransform.h>
#include <psi4/libpsio/psio.hpp>
#include <psi4/libciomr/libciomr.h>
#include <psi4/libqt/qt.h>
#include <psi4/libiwl/iwl.hpp>
#include <psi4/libtrans/integraltransform_functors.h>
#include <psi4/psifiles.h>
#include <psi4/libtrans/mospace.h>
#define EXTERN
//#include <libdpd/dpd.gbl>

using namespace psi;

TPDMBackTransform::TPDMBackTransform(SharedWavefunction wfn, SpaceVec spaces,
                                     TransformationType transformationType, OutputType outputType,
                                     MOOrdering moOrdering, FrozenOrbitals frozenOrbitals,
                                     bool init)
    : IntegralTransform(wfn, spaces, transformationType, outputType, moOrdering, frozenOrbitals,
                        init) {}

TPDMBackTransform::~TPDMBackTransform() {}

// repack the unrestricted MO TPDMs into DPD buffers
// to prepare them for the back transformation in deriv()
// this function is a modifield version of
// IntegralTransform::presort_mo_tpdm_unrestricted()
// from libtrans/integraltransform_sort_mo_tpdm.cc
void TPDMBackTransform::presort_mo_tpdm_unrestricted() {

    check_initialized();

    int currentActiveDPD = psi::dpd_default;
    dpd_set_default(myDPDNum_);

    outfile->Printf("\n");
    outfile->Printf("  ==> Back-transforming MO-basis TPDMs <==\n");
    outfile->Printf("\n");

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
    outfile->Printf("\n p = %zu, q = %zu", nump, numq);
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

        outfile->Printf("\n h = %d, rowlength = %zu,  rowtot = %zu", h, rowLength,
                        I.params->rowtot[h]);
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

    outfile->Printf("        Sorting File: %s nbuckets = %d\n", I.label, nBuckets);

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

    outfile->Printf("        Sorting File: %s nbuckets = %d\n", I.label, nBuckets);

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
                // Check:
                //                outfile->Printf("\t%4d %4d %4d %4d = %20.10f\n", p, q, r, s,
                //                value);
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

    outfile->Printf("        Sorting File: %s nbuckets = %d\n", I.label, nBuckets);
    outfile->Printf("\n");

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

void TPDMBackTransform::backtransform_density() {

    check_initialized();

    // This limitation can be remedied by accounting for the fact that Pitzer orbital numbering is
    // not dense, so certain quantities must be alloc'd for the full MO space.  It's no limitation,
    // though
    if (frozenOrbitals_ != FrozenOrbitals::None)
        throw SanityCheckError("No orbitals can be frozen in density matrix transformations\n",
                               __FILE__, __LINE__);
    // The full MO space must be in the list of spaces used, let's check
    bool allFound = false;
    for (int i = 0; i < spacesUsed_.size(); ++i)
        if (spacesUsed_[i] == MOSPACE_ALL)
            allFound = true;
    if (!allFound)
        throw PSIEXCEPTION("MOSpace::all must be amongst the spaces passed "
                           "to the integral object's constructor");

    backtransform_tpdm_unrestricted();
}
