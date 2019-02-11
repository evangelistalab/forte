/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#ifndef _fci_compressor_h_
#define _fci_compressor_h_

#include "base_classes/active_space_method.h"
#include "psi4/libmints/dimension.h"

namespace forte {

class FCIVector;
class StringLists;

/**
 * @brief The FCICompression class
 * This class compresses Full CI tensors as an analysis tool.
 */
class FCICompressor /* : public ActiveSpaceMethod */ {
    public:
        // ==> Class Constructor and Destructor <==

        /**
         * @brief FCICompressor A class that performs a compression of the FCI wfn
         * @param C_fci the FCI wfn
         * @param as_ints the FCI integrals
         */
        FCICompressor(std::shared_ptr<FCIVector> C_fci, FCIVector HC_fci,
                      std::shared_ptr<ActiveSpaceIntegrals> as_ints,
                      double nuclear_repulsion_energy,
                      double fci_energy);

        ~FCICompressor() = default;

        // ==> Class Interface <==

        /// Set the options
        void set_options(std::shared_ptr<ForteOptions> options) /* override */;

        void set_do_rr(bool value) { do_rr_ = value; }

        void set_do_st(bool value) { do_st_ = value; }

        void set_do_ds(bool value) { do_ds_ = value; }

        void set_do_mp(bool value) { do_mp_ = value; }

        void set_tau(double value) { tau_ = value; }

        void set_delta_tao(double value) { delta_tau_ = value; }

        void set_num_compressions(double value) { num_compressions_ = value; }

        void set_print_cume(double value) { print_cume_ = value; }

        /// Do various compressions and compute analysis data;
        void compress_and_analyze();

        // /// Comutes the squared Frobenius norm of the 2-body cumulant error matrix
        // double two_rcm_diff(std::shared_ptr<FCIVector> C_compressed);
        //
        // /// Comutes the squared Frobenius norm of the wfn error matrix
        // double C_diff(std::shared_ptr<FCIVector> C_compressed);
        //
        /// Compresses C_fci via SVD based rank reduction
        void rank_reduce();

        // /// Comoresses C_fci by removing dets with unimportant alfa or beta strings
        // void string_trim();
        //
        // /// Comoresses C_fci by removing dets with low weight
        // void det_screen();
        //
        // /// Comoresses C_fci via reduced rank matrix product states
        // void mps_ify();

    private:

        /// The origional (non compressed) FCIVector
        std::shared_ptr<FCIVector> C_fci_;

        /// The origional (non compressed) FCI Sigma Vector
        std::shared_ptr<FCIVector> HC_fci_;

        /// The active space integrals
        std::shared_ptr<ActiveSpaceIntegrals> fci_ints_;

        /// The nuclear repulsion energy
        double nuclear_repulsion_energy_;

        /// The FCI energy
        double fci_energy_;

        /// The FCI cumulant norm
        double fci_cume_norm_;

        // Options Vars
        /// Do a rank reduction compression?
        bool do_rr_ = true;

        /// Do a string trimming compression?
        bool do_st_ = true;

        /// Do a determinat screen compression?
        bool do_ds_ = true;

        /// Do a mps transform compression?
        bool do_mp_ = true;

        /// The cumulative compression threshold
        double tau_ = 0.0;

        /// The difference in cumulative compression threshold from kth compression
        /// to k+1th compression
        // double delta_tau_ = 0.000025;
        double delta_tau_ = 0.0001;

        /// The number of compressions to be performed for each compression type
        int num_compressions_ = 5;

        /// Print the cumulant matricies to a file for all compressions?
        bool print_cume_ = true;
};
} // namespace forte

#endif // _fci_compressor_h_
