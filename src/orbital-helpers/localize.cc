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

#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libmints/local.h"
#include "psi4/liboptions/liboptions.h"
#include "base_classes/reference.h"

#include "localize.h"

using namespace psi;

namespace forte {

LOCALIZE::LOCALIZE(StateInfo state, std::shared_ptr<SCFInfo> scf_info,
                   std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                   std::shared_ptr<MOSpaceInfo> mo_space_info)
    : OrbitalTransform(scf_info, options, ints, mo_space_info), scf_info_(scf_info),
      options_(options), ints_(ints) {
    nfrz_ = mo_space_info->size("FROZEN_DOCC");
    nrst_ = mo_space_info->size("RESTRICTED_DOCC");
    namo_ = mo_space_info->size("ACTIVE");

    if (ints_->nirrep() > 1) {
        throw psi::PSIEXCEPTION("\n\n ERROR: Localizer only implemented for C1 symmetry!");
    }

    // The wavefunction multiplicity
    multiplicity_ = options->get_int("MULTIPLICITY");

    // double occupied active
    naocc_ = scf_info_->doccpi().n() - nfrz_ - nrst_;

    // virtual active
    navir_ = namo_ - naocc_;
    abs_act_ = mo_space_info->get_absolute_mo("ACTIVE");

    local_type_ = options->get_str("LOCALIZE_TYPE");
}

void LOCALIZE::localize() {

    if ((local_type_ == "FULL_BOYS") or (local_type_ == "SPLIT_BOYS")) {
        local_method_ = "BOYS";
    }
    if ((local_type_ == "FULL_PM") or (local_type_ == "SPLIT_PM")) {
        local_method_ = "PIPEK_MEZEY";
    }

    if ((local_type_ == "FULL_BOYS") or (local_type_ == "FULL_PM")) {
        full_localize();
    } else if ((local_type_ == "SPLIT_BOYS") or (local_type_ == "SPLIT_PM")) {
        split_localize();
    }
}

void LOCALIZE::compute_transformation() {
    std::string loc = options_->get_str("LOCALIZE");
    if (loc == "SPLIT") {
        split_localize();
    } else {
        full_localize();
    }
}

void LOCALIZE::split_localize() {
    psi::Dimension nsopi = scf_info_->nsopi();
    int nirrep = ints_->nirrep();
    int off = 0;
    if (multiplicity_ == 3) {
        naocc_ -= 1;
        navir_ -= 1;
        off = 2;
    }
    psi::SharedMatrix Ca = ints_->Ca();
    psi::SharedMatrix Cb = ints_->Cb();

    SharedMatrix Caocc = std::make_shared<Matrix>("Caocc", nsopi[0], naocc_);
    SharedMatrix Cavir = std::make_shared<Matrix>("Cavir", nsopi[0], navir_);
    SharedMatrix Caact = std::make_shared<Matrix>("Caact", nsopi[0], off);

    for (int h = 0; h < nirrep; h++) {
        for (int mu = 0; mu < nsopi[h]; mu++) {
            for (int i = 0; i < naocc_; i++) {
                Caocc->set(h, mu, i, Ca->get(h, mu, abs_act_[i]));
            }
            for (int i = 0; i < navir_; ++i) {
                Cavir->set(h, mu, i, Ca->get(h, mu, abs_act_[i + naocc_ + off]));
            }
            for (int i = 0; i < off; ++i) {
                Caact->set(h, mu, i, Ca->get(h, mu, abs_act_[i + naocc_]));
            }
        }
    }

    std::shared_ptr<psi::BasisSet> primary = ints_->basisset();
    std::shared_ptr<Localizer> loc_a = Localizer::build(local_type_, primary, Caocc);
    loc_a->localize();

    std::shared_ptr<psi::Localizer> loc_v = psi::Localizer::build(local_type_, primary, Cavir);
    loc_v->localize();

    psi::SharedMatrix Lact;
    psi::SharedMatrix Uact;
    if (multiplicity_ == 3) {
        std::shared_ptr<Localizer> loc_c = Localizer::build(local_method_, primary, Caact);
        loc_c->localize();
        Lact = loc_c->L();
        Uact = loc_c->U();
    }
    psi::SharedMatrix Uocc = loc_a->U();
    psi::SharedMatrix Uvir = loc_v->U();

    Ua_.reset(new psi::Matrix("Ua",nsopi[0], nsopi[0]));
    Ub_.reset(new psi::Matrix("Ua",nsopi[0], nsopi[0]));

    Ua_->identity();
    Ub_->identity();

    for (int h = 0; h < nirrep; ++h) {
        for (int i = 0; i < naocc_; ++i) {
            for (int j = 0; j < naocc_; ++j) {
                Ua_->set(h,i + nfrz_ + nrst_,j + nfrz_ + nrst_, Uocc->get(h,i,j));
                Ub_->set(h,i + nfrz_ + nrst_,j + nfrz_ + nrst_, Uocc->get(h,i,j));
            }
        }
        for (int i = 0; i < navir_; ++i) {
            for (int j = 0; j < navir_; ++j) {
                Ua_->set(h,i + nfrz_ + nrst_,j + nfrz_ + nrst_, Uvir->get(h,i,j));
                Ub_->set(h,i + nfrz_ + nrst_,j + nfrz_ + nrst_, Uvir->get(h,i,j));
            }
        }

        for (int i = 0; i < off; ++i) {
            for (int j = 0; j < off; ++j) {
                Ua_->set(h,i + nfrz_ + nrst_,j + nfrz_ + nrst_, Uact->get(h,i,j));
                Ub_->set(h,i + nfrz_ + nrst_,j + nfrz_ + nrst_, Uact->get(h,i,j));
            }
        }
    }
}

void LOCALIZE::full_localize() {

    psi::Dimension nsopi = scf_info_->nsopi();
    int nirrep = ints_->nirrep();
    size_t nact = abs_act_.size();

    psi::SharedMatrix Ca = ints_->Ca();
    psi::SharedMatrix Cb = ints_->Cb();

    psi::SharedMatrix Caact(new psi::Matrix("Caact", nsopi[0], nact));
    for (int h = 0; h < nirrep; h++) {
        for (int mu = 0; mu < nsopi[h]; mu++) {
            for (size_t i = 0; i < nact; i++) {
                Caact->set(h, mu, i, Ca->get(h, mu, abs_act_[i]));
            }
        }
    }

    // Localize all active together
    std::shared_ptr<psi::BasisSet> primary = ints_->basisset();

    std::shared_ptr<Localizer> loc_a = Localizer::build(local_method_, primary, Caact);
    loc_a->localize();

    psi::SharedMatrix Laocc = loc_a->L();
    psi::SharedMatrix Ua = loc_a->U();

    Ua_.reset(new psi::Matrix("Ua",Ca->rowdim(), Ca->coldim()));
    Ub_.reset(new psi::Matrix("Ua",Ca->rowdim(), Ca->coldim()));

    Ua_->identity();
    Ub_->identity();

    for (int h = 0; h < nirrep; ++h) {
        for (size_t i = 0; i < nact; ++i) {
            for (size_t j = 0; j < nact; ++j) {
                Ua_->set(h,i + nfrz_ + nrst_,j + nfrz_ + nrst_, Ua->get(h,i,j));
                Ub_->set(h,i + nfrz_ + nrst_,j + nfrz_ + nrst_, Ua->get(h,i,j));
            }
        }
    }
}

psi::SharedMatrix LOCALIZE::get_Ua() { return Ua_; }
psi::SharedMatrix LOCALIZE::get_Ub() { return Ub_; }

} // namespace forte
