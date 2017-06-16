/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "../helpers.h"

#include "fci_vector.h"

namespace psi {
namespace forte {

FCIIntegrals::FCIIntegrals(std::shared_ptr<ForteIntegrals> ints, std::vector<size_t> active_mo,
                           std::vector<size_t> restricted_docc_mo)
    : ints_(ints), active_mo_(active_mo), restricted_docc_mo_(restricted_docc_mo) {
    nmo_ = active_mo_.size();
    startup();
}
void FCIIntegrals::RestrictedOneBodyOperator(std::vector<double>& oei_a,
                                             std::vector<double>& oei_b) {

    std::vector<double> tei_rdocc_aa;
    std::vector<double> tei_rdocc_ab;
    std::vector<double> tei_rdocc_bb;

    std::vector<double> tei_gh_aa;
    std::vector<double> tei_gh_ab;
    std::vector<double> tei_gh_bb;
    std::vector<double> tei_gh2_ab;

    std::vector<size_t> fomo_to_mo(restricted_docc_mo_);
    std::vector<size_t> cmo_to_mo(active_mo_);
    size_t nfomo = fomo_to_mo.size();

    ambit::Tensor rdocc_aa = ints_->aptei_aa_block(fomo_to_mo, fomo_to_mo, fomo_to_mo, fomo_to_mo);
    ambit::Tensor rdocc_ab = ints_->aptei_ab_block(fomo_to_mo, fomo_to_mo, fomo_to_mo, fomo_to_mo);
    ambit::Tensor rdocc_bb = ints_->aptei_bb_block(fomo_to_mo, fomo_to_mo, fomo_to_mo, fomo_to_mo);
    tei_rdocc_aa = rdocc_aa.data();
    tei_rdocc_ab = rdocc_ab.data();
    tei_rdocc_bb = rdocc_bb.data();

    ambit::Tensor gh_aa = ints_->aptei_aa_block(cmo_to_mo, fomo_to_mo, cmo_to_mo, fomo_to_mo);
    ambit::Tensor gh_ab = ints_->aptei_ab_block(cmo_to_mo, fomo_to_mo, cmo_to_mo, fomo_to_mo);
    ambit::Tensor gh_bb = ints_->aptei_bb_block(cmo_to_mo, fomo_to_mo, cmo_to_mo, fomo_to_mo);
    ambit::Tensor gh2_ab = ints_->aptei_ab_block(fomo_to_mo, cmo_to_mo, fomo_to_mo, cmo_to_mo);

    tei_gh_aa = gh_aa.data();
    tei_gh_ab = gh_ab.data();
    tei_gh_bb = gh_bb.data();
    tei_gh2_ab = gh2_ab.data();

    // Compute the scalar contribution to the energy that comes from
    // the restricted occupied orbitals
    scalar_energy_ = ints_->scalar();
    for (size_t i = 0; i < nfomo; ++i) {
        size_t ii = fomo_to_mo[i];
        scalar_energy_ += ints_->oei_a(ii, ii);
        scalar_energy_ += ints_->oei_b(ii, ii);
        for (size_t j = 0; j < nfomo; ++j) {
            size_t index = nfomo * nfomo * nfomo * i + nfomo * nfomo * j + nfomo * i + j;
            scalar_energy_ += 0.5 * tei_rdocc_aa[index];
            scalar_energy_ += 1.0 * tei_rdocc_ab[index];
            scalar_energy_ += 0.5 * tei_rdocc_bb[index];
        }
    }

    for (size_t p = 0; p < nmo_; ++p) {
        size_t pp = cmo_to_mo[p];
        for (size_t q = 0; q < nmo_; ++q) {
            size_t qq = cmo_to_mo[q];
            size_t idx = nmo_ * p + q;
            oei_a[idx] = ints_->oei_a(pp, qq);
            oei_b[idx] = ints_->oei_b(pp, qq);
            // Compute the one-body contribution to the energy that comes from
            // the restricted occupied orbitals
            for (size_t f = 0; f < nfomo; ++f) {
                size_t index = nfomo * nmo_ * nfomo * p + nmo_ * nfomo * f + nfomo * q + f;
                oei_a[idx] += tei_gh_aa[index];
                oei_a[idx] += tei_gh_ab[index];
                oei_b[idx] += tei_gh_bb[index];
                oei_b[idx] += tei_gh_ab[index]; // TODO check these factors 0.5
            }
        }
    }
}

FCIIntegrals::FCIIntegrals(std::shared_ptr<ForteIntegrals> ints,
                           std::shared_ptr<MOSpaceInfo> mospace_info, FCIIntegralsType type)
    : ints_(ints) {
    std::vector<size_t> cmo_to_mo;
    std::vector<size_t> fomo_to_mo;

    nmo_ = mospace_info->size("ACTIVE");
    cmo_to_mo = mospace_info->get_corr_abs_mo("ACTIVE");
    fomo_to_mo = mospace_info->get_corr_abs_mo("RESTRICTED_DOCC");
    std::vector<size_t> ncmo_to_mo = mospace_info->get_corr_abs_mo("GENERALIZED HOLE");

    nmo2_ = nmo_ * nmo_;
    nmo3_ = nmo_ * nmo_ * nmo_;
    nmo4_ = nmo_ * nmo_ * nmo_ * nmo_;

    // size_t nfomo =  mospace_info->size("RESTRICTED_DOCC");

    oei_a_.resize(nmo2_);
    oei_b_.resize(nmo2_);
    tei_aa_.resize(nmo4_);
    tei_ab_.resize(nmo4_);
    tei_bb_.resize(nmo4_);
    diag_tei_aa_.resize(nmo2_);
    diag_tei_ab_.resize(nmo2_);
    diag_tei_bb_.resize(nmo2_);

    frozen_core_energy_ = ints->frozen_core_energy();
    // Initialize tei_rdocc vectors, but don't store them as global variable

    // Grab all integrals in blocks
    ambit::Tensor act_aa = ints->aptei_aa_block(cmo_to_mo, cmo_to_mo, cmo_to_mo, cmo_to_mo);
    ambit::Tensor act_ab = ints->aptei_ab_block(cmo_to_mo, cmo_to_mo, cmo_to_mo, cmo_to_mo);
    ambit::Tensor act_bb = ints->aptei_bb_block(cmo_to_mo, cmo_to_mo, cmo_to_mo, cmo_to_mo);
    tei_aa_ = act_aa.data();
    tei_ab_ = act_ab.data();
    tei_bb_ = act_bb.data();

    scalar_energy_ = ints->scalar();

    RestrictedOneBodyOperator(oei_a_, oei_b_);
}
void FCIIntegrals::startup() {

    nmo2_ = nmo_ * nmo_;
    nmo3_ = nmo_ * nmo_ * nmo_;
    nmo4_ = nmo_ * nmo_ * nmo_ * nmo_;

    oei_a_.resize(nmo2_);
    oei_b_.resize(nmo2_);
    tei_aa_.resize(nmo4_);
    tei_ab_.resize(nmo4_);
    tei_bb_.resize(nmo4_);
    diag_tei_aa_.resize(nmo2_);
    diag_tei_ab_.resize(nmo2_);
    diag_tei_bb_.resize(nmo2_);
    frozen_core_energy_ = ints_->frozen_core_energy();
}
void FCIIntegrals::set_active_integrals(const ambit::Tensor& act_aa, const ambit::Tensor& act_ab,
                                        const ambit::Tensor& act_bb) {
    tei_aa_ = act_aa.data();
    tei_ab_ = act_ab.data();
    tei_bb_ = act_bb.data();
}
void FCIIntegrals::compute_restricted_one_body_operator() {
    nmo2_ = nmo_ * nmo_;
    oei_a_.resize(nmo2_);
    oei_b_.resize(nmo2_);
    RestrictedOneBodyOperator(oei_a_, oei_b_);
}

void FCIIntegrals::set_active_integrals_and_restricted_docc() {
    ambit::Tensor act_aa = ints_->aptei_aa_block(active_mo_, active_mo_, active_mo_, active_mo_);
    ambit::Tensor act_ab = ints_->aptei_ab_block(active_mo_, active_mo_, active_mo_, active_mo_);
    ambit::Tensor act_bb = ints_->aptei_bb_block(active_mo_, active_mo_, active_mo_, active_mo_);

    tei_aa_ = act_aa.data();
    tei_ab_ = act_ab.data();
    tei_bb_ = act_bb.data();
    RestrictedOneBodyOperator(oei_a_, oei_b_);
}
}
}
