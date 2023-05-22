/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "ambit/tensor.h"

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"

#include "helpers/printing.h"
#include "base_classes/rdms.h"

#include "localize.h"

using namespace psi;

namespace forte {

Localize::Localize(std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                   std::shared_ptr<MOSpaceInfo> mo_space_info)
    : OrbitalTransform(ints, mo_space_info) {

    if (ints_->nirrep() > 1) {
        throw psi::PSIEXCEPTION("\n\n ERROR: Localizer only implemented for C1 symmetry!");
    }

    orbital_spaces_ = options->get_int_list("LOCALIZE_SPACE");
    local_method_ = options->get_str("LOCALIZE");

    print_h2("Orbital Localizer");

    outfile->Printf("\n  Localize method: %s", local_method_.c_str());
}

void Localize::set_orbital_space(std::vector<int>& orbital_spaces) {
    orbital_spaces_ = orbital_spaces;
}

void Localize::set_orbital_space(std::vector<std::string>& labels) {

    for (const auto& label : labels) {
        std::vector<size_t> mos = mo_space_info_->corr_absolute_mo(label);
        orbital_spaces_.push_back(mos[0]);
        orbital_spaces_.push_back(mos.back());
    }
}

void Localize::compute_transformation() {

    if (orbital_spaces_.size() == 0) {
        outfile->Printf("\n  Error: Orbital space for localization is not set!");
        exit(1);
    } else if ((orbital_spaces_.size() % 2) != 0) {
        outfile->Printf("\n  Error: Orbital space for localization not properly set!");
        exit(1);
    }

    auto Ca = ints_->Ca();

    Ua_ = std::make_shared<psi::Matrix>("U", Ca->coldim(), Ca->coldim());
    Ub_ = std::make_shared<psi::Matrix>("U", Ca->coldim(), Ca->coldim());

    Ua_->identity();
    Ub_->identity();

    // loop through each space
    for (size_t f_idx = 0, max = orbital_spaces_.size(); f_idx < max - 1; f_idx += 2) {

        // indices are INCLUSIVE
        size_t first = orbital_spaces_[f_idx];
        size_t last = orbital_spaces_[f_idx + 1];

        // print
        outfile->Printf("\n  Localizing orbitals: ");
        for (size_t orb = first; orb <= last; ++orb) {
            outfile->Printf(" %d", orb);
        }
        outfile->Printf("\n");

        if (last < first) {
            outfile->Printf("\n  Error: Orbital space for localization not properly set!");
            exit(1);
        }

        // number of orbitals to localize
        size_t orb_dim = last - first + 1;

        // Build C matrix to localize
        auto Ca_loc = std::make_shared<psi::Matrix>("Caact", Ca->rowdim(), orb_dim);

        for (size_t i = 0; i < orb_dim; ++i) {
            auto col = Ca->get_column(0, first + i);
            Ca_loc->set_column(0, i, col);
        }

        // localize
        std::shared_ptr<psi::BasisSet> primary = ints_->wfn()->basisset();
        std::shared_ptr<psi::Localizer> loc_a =
            psi::Localizer::build(local_method_, primary, Ca_loc);
        loc_a->localize();

        // Grab the transformation and localized matrices
        auto Ua_loc = loc_a->U();

        // Set Ua, Ub
        for (size_t i = 0; i < orb_dim; ++i) {
            for (size_t j = 0; j < orb_dim; ++j) {
                Ua_->set(i + first, j + first, Ua_loc->get(i, j));
                Ub_->set(i + first, j + first, Ua_loc->get(i, j));
            }
        }
    }
}

} // namespace forte
