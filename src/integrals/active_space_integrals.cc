/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <cmath>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "base_classes/mo_space_info.h"
#include "integrals/active_space_integrals.h"

#define FAST_SLATER_RULES 0

namespace forte {

ActiveSpaceIntegrals::ActiveSpaceIntegrals(std::shared_ptr<ForteIntegrals> ints,
                                           const std::vector<size_t>& active_mo,
                                           const std::vector<int>& active_mo_symmetry,
                                           const std::vector<size_t>& restricted_docc_mo)
    : nmo_(active_mo.size()), ints_(ints), active_mo_(active_mo),
      active_mo_symmetry_(active_mo_symmetry), restricted_docc_mo_(restricted_docc_mo) {
    startup();
}

void ActiveSpaceIntegrals::RestrictedOneBodyOperator(std::vector<double>& oei_a,
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

void ActiveSpaceIntegrals::startup() {

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

void ActiveSpaceIntegrals::set_active_integrals(const ambit::Tensor& act_aa,
                                                const ambit::Tensor& act_ab,
                                                const ambit::Tensor& act_bb) {
    tei_aa_ = act_aa.data();
    tei_ab_ = act_ab.data();
    tei_bb_ = act_bb.data();
}

void ActiveSpaceIntegrals::compute_restricted_one_body_operator() {
    nmo2_ = nmo_ * nmo_;
    oei_a_.resize(nmo2_);
    oei_b_.resize(nmo2_);
    RestrictedOneBodyOperator(oei_a_, oei_b_);
}

void ActiveSpaceIntegrals::set_active_integrals_and_restricted_docc() {
    ambit::Tensor act_aa = ints_->aptei_aa_block(active_mo_, active_mo_, active_mo_, active_mo_);
    ambit::Tensor act_ab = ints_->aptei_ab_block(active_mo_, active_mo_, active_mo_, active_mo_);
    ambit::Tensor act_bb = ints_->aptei_bb_block(active_mo_, active_mo_, active_mo_, active_mo_);

    tei_aa_ = act_aa.data();
    tei_ab_ = act_ab.data();
    tei_bb_ = act_bb.data();
    RestrictedOneBodyOperator(oei_a_, oei_b_);
}

std::vector<size_t> ActiveSpaceIntegrals::active_mo() const { return active_mo_; }

std::vector<int> ActiveSpaceIntegrals::active_mo_symmetry() const { return active_mo_symmetry_; }

std::vector<size_t> ActiveSpaceIntegrals::restricted_docc_mo() const { return restricted_docc_mo_; }

double ActiveSpaceIntegrals::energy(const Determinant& det) const {
    double energy = frozen_core_energy_;

    String Ia = det.get_alfa_bits();
    String Ib = det.get_beta_bits();
    String Iac;
    String Ibc;

    int naocc = Ia.count();
    int nbocc = Ib.count();

    for (int A = 0; A < naocc; ++A) {
        int p = Ia.find_and_clear_first_one();
        energy += oei_a_[p * nmo_ + p];

        Iac = Ia;
        for (int AA = A + 1; AA < naocc; ++AA) {
            int q = Iac.find_and_clear_first_one();
            energy += tei_aa_[p * nmo3_ + q * nmo2_ + p * nmo_ + q];
        }

        Ibc = Ib;
        for (int B = 0; B < nbocc; ++B) {
            int q = Ibc.find_and_clear_first_one();
            energy += tei_ab_[p * nmo3_ + q * nmo2_ + p * nmo_ + q];
        }
    }

    for (int B = 0; B < nbocc; ++B) {
        int p = Ib.find_and_clear_first_one();
        energy += oei_b_[p * nmo_ + p];
        Ibc = Ib;
        for (int BB = B + 1; BB < nbocc; ++BB) {
            int q = Ibc.find_and_clear_first_one();
            energy += tei_bb_[p * nmo3_ + q * nmo2_ + p * nmo_ + q];
        }
    }

    return energy;
}

double ActiveSpaceIntegrals::slater_rules(const Determinant& lhs, const Determinant& rhs) const {
    // we first check that the two determinants have equal Ms
    if ((lhs.count_alfa() != rhs.count_alfa()) or (lhs.count_beta() != rhs.count_beta()))
        return 0.0;

    int nadiff = 0;
    int nbdiff = 0;

#if FAST_SLATER_RULES
#else
    // Count how many differences in mos are there
    for (size_t n = 0; n < nmo_; ++n) {
        if (lhs.get_alfa_bit(n) != rhs.get_alfa_bit(n))
            nadiff++;
        if (lhs.get_beta_bit(n) != rhs.get_beta_bit(n))
            nbdiff++;
        if (nadiff + nbdiff > 4)
            return 0.0; // Get out of this as soon as possible
    }
    nadiff /= 2;
    nbdiff /= 2;

    double matrix_element = 0.0;
    // Slater rule 1 PhiI = PhiJ
    if ((nadiff == 0) and (nbdiff == 0)) {
        // matrix_element += frozen_core_energy_ + this->energy(rhs);
        matrix_element = frozen_core_energy_;
        for (size_t p = 0; p < nmo_; ++p) {
            if (lhs.get_alfa_bit(p))
                matrix_element += oei_a_[p * nmo_ + p];
            if (lhs.get_beta_bit(p))
                matrix_element += oei_b_[p * nmo_ + p];
            for (size_t q = 0; q < nmo_; ++q) {
                if (lhs.get_alfa_bit(p) and lhs.get_alfa_bit(q))
                    matrix_element += 0.5 * tei_aa_[p * nmo3_ + q * nmo2_ + p * nmo_ + q];
                if (lhs.get_beta_bit(p) and lhs.get_beta_bit(q))
                    matrix_element += 0.5 * tei_bb_[p * nmo3_ + q * nmo2_ + p * nmo_ + q];
                if (lhs.get_alfa_bit(p) and lhs.get_beta_bit(q))
                    matrix_element += tei_ab_[p * nmo3_ + q * nmo2_ + p * nmo_ + q];
            }
        }
    }

    // Slater rule 2 PhiI = j_a^+ i_a PhiJ
    if ((nadiff == 1) and (nbdiff == 0)) {
        // Diagonal contribution
        size_t i = 0;
        size_t j = 0;
        for (size_t p = 0; p < nmo_; ++p) {
            if ((lhs.get_alfa_bit(p) != rhs.get_alfa_bit(p)) and lhs.get_alfa_bit(p))
                i = p;
            if ((lhs.get_alfa_bit(p) != rhs.get_alfa_bit(p)) and rhs.get_alfa_bit(p))
                j = p;
        }
        // double sign = SlaterSign(I, i, j);
        double sign = lhs.slater_sign_aa(i, j);
        matrix_element = sign * oei_a_[i * nmo_ + j];
        for (size_t p = 0; p < nmo_; ++p) {
            if (lhs.get_alfa_bit(p) and rhs.get_alfa_bit(p)) {
                matrix_element += sign * tei_aa_[i * nmo3_ + p * nmo2_ + j * nmo_ + p];
            }
            if (lhs.get_beta_bit(p) and rhs.get_beta_bit(p)) {
                matrix_element += sign * tei_ab_[i * nmo3_ + p * nmo2_ + j * nmo_ + p];
            }
        }
    }
    // Slater rule 2 PhiI = j_b^+ i_b PhiJ
    if ((nadiff == 0) and (nbdiff == 1)) {
        // Diagonal contribution
        size_t i = 0;
        size_t j = 0;
        for (size_t p = 0; p < nmo_; ++p) {
            if ((lhs.get_beta_bit(p) != rhs.get_beta_bit(p)) and lhs.get_beta_bit(p))
                i = p;
            if ((lhs.get_beta_bit(p) != rhs.get_beta_bit(p)) and rhs.get_beta_bit(p))
                j = p;
        }
        // double sign = SlaterSign(I, nmo_ + i, nmo_ + j);
        double sign = lhs.slater_sign_bb(i, j);
        matrix_element = sign * oei_b_[i * nmo_ + j];
        for (size_t p = 0; p < nmo_; ++p) {
            if (lhs.get_alfa_bit(p) and rhs.get_alfa_bit(p)) {
                matrix_element += sign * tei_ab_[p * nmo3_ + i * nmo2_ + p * nmo_ + j];
            }
            if (lhs.get_beta_bit(p) and rhs.get_beta_bit(p)) {
                matrix_element += sign * tei_bb_[i * nmo3_ + p * nmo2_ + j * nmo_ + p];
            }
        }
    }

    // Slater rule 3 PhiI = k_a^+ l_a^+ j_a i_a PhiJ
    if ((nadiff == 2) and (nbdiff == 0)) {
        // Diagonal contribution
        int i = -1;
        int j = 0;
        int k = -1;
        int l = 0;
        for (size_t p = 0; p < nmo_; ++p) {
            if ((lhs.get_alfa_bit(p) != rhs.get_alfa_bit(p)) and lhs.get_alfa_bit(p)) {
                if (i == -1) {
                    i = p;
                } else {
                    j = p;
                }
            }
            if ((lhs.get_alfa_bit(p) != rhs.get_alfa_bit(p)) and rhs.get_alfa_bit(p)) {
                if (k == -1) {
                    k = p;
                } else {
                    l = p;
                }
            }
        }
        // double sign = SlaterSign(I, i, j, k, l);
        double sign = lhs.slater_sign_aaaa(i, j, k, l);
        matrix_element = sign * tei_aa_[i * nmo3_ + j * nmo2_ + k * nmo_ + l];
    }

    // Slater rule 3 PhiI = k_a^+ l_a^+ j_a i_a PhiJ
    if ((nadiff == 0) and (nbdiff == 2)) {
        // Diagonal contribution
        int i, j, k, l;
        i = -1;
        j = -1;
        k = -1;
        l = -1;
        for (size_t p = 0; p < nmo_; ++p) {
            if ((lhs.get_beta_bit(p) != rhs.get_beta_bit(p)) and lhs.get_beta_bit(p)) {
                if (i == -1) {
                    i = p;
                } else {
                    j = p;
                }
            }
            if ((lhs.get_beta_bit(p) != rhs.get_beta_bit(p)) and rhs.get_beta_bit(p)) {
                if (k == -1) {
                    k = p;
                } else {
                    l = p;
                }
            }
        }
        // double sign = SlaterSign(I, nmo_ + i, nmo_ + j, nmo_ + k, nmo_ + l);
        double sign = lhs.slater_sign_bbbb(i, j, k, l);
        matrix_element = sign * tei_bb_[i * nmo3_ + j * nmo2_ + k * nmo_ + l];
    }

    // Slater rule 3 PhiI = j_a^+ i_a PhiJ
    if ((nadiff == 1) and (nbdiff == 1)) {
        // Diagonal contribution
        int i, j, k, l;
        i = j = k = l = -1;
        for (size_t p = 0; p < nmo_; ++p) {
            if ((lhs.get_alfa_bit(p) != rhs.get_alfa_bit(p)) and lhs.get_alfa_bit(p))
                i = p;
            if ((lhs.get_beta_bit(p) != rhs.get_beta_bit(p)) and lhs.get_beta_bit(p))
                j = p;
            if ((lhs.get_alfa_bit(p) != rhs.get_alfa_bit(p)) and rhs.get_alfa_bit(p))
                k = p;
            if ((lhs.get_beta_bit(p) != rhs.get_beta_bit(p)) and rhs.get_beta_bit(p))
                l = p;
        }
        //  double sign = SlaterSign(I, i, nmo_ + j, k, nmo_ + l);
        // double sign = lhs.slater_sign(i, nmo_ + j, k, nmo_ + l);
        double sign = lhs.slater_sign_aa(i, k) * lhs.slater_sign_bb(j, l);
        matrix_element = sign * tei_ab_[i * nmo3_ + j * nmo2_ + k * nmo_ + l];
    }
#endif
    return (matrix_element);
}

double ActiveSpaceIntegrals::slater_rules_single_alpha(const Determinant& det, int i, int a) const {
    // Slater rule 2 PhiI = j_a^+ i_a PhiJ
    double sign = det.slater_sign_aa(i, a);
    double matrix_element = oei_a_[i * nmo_ + a];
    for (size_t p = 0; p < nmo_; ++p) {
        if (det.get_alfa_bit(p)) {
            matrix_element += tei_aa_[i * nmo3_ + p * nmo2_ + a * nmo_ + p];
        }
        if (det.get_beta_bit(p)) {
            matrix_element += tei_ab_[i * nmo3_ + p * nmo2_ + a * nmo_ + p];
        }
    }
    return sign * matrix_element;
}

double ActiveSpaceIntegrals::slater_rules_single_alpha_abs(const Determinant& det, int i,
                                                           int a) const {
    // Slater rule 2 PhiI = j_a^+ i_a PhiJ
    double matrix_element = oei_a_[i * nmo_ + a];
    for (size_t p = 0; p < nmo_; ++p) {
        if (det.get_alfa_bit(p)) {
            matrix_element += tei_aa_[i * nmo3_ + p * nmo2_ + a * nmo_ + p];
        }
        if (det.get_beta_bit(p)) {
            matrix_element += tei_ab_[i * nmo3_ + p * nmo2_ + a * nmo_ + p];
        }
    }
    return matrix_element;
}

double ActiveSpaceIntegrals::slater_rules_single_beta(const Determinant& det, int i, int a) const {
    // Slater rule 2 PhiI = j_a^+ i_a PhiJ
    double sign = det.slater_sign_bb(i, a);
    double matrix_element = oei_b_[i * nmo_ + a];
    for (size_t p = 0; p < nmo_; ++p) {
        if (det.get_alfa_bit(p)) {
            matrix_element += tei_ab_[p * nmo3_ + i * nmo2_ + p * nmo_ + a];
        }
        if (det.get_beta_bit(p)) {
            matrix_element += tei_bb_[i * nmo3_ + p * nmo2_ + a * nmo_ + p];
        }
    }
    return sign * matrix_element;
}

double ActiveSpaceIntegrals::slater_rules_single_beta_abs(const Determinant& det, int i,
                                                          int a) const {
    // Slater rule 2 PhiI = j_a^+ i_a PhiJ
    double matrix_element = oei_b_[i * nmo_ + a];
    for (size_t p = 0; p < nmo_; ++p) {
        if (det.get_alfa_bit(p)) {
            matrix_element += tei_ab_[p * nmo3_ + i * nmo2_ + p * nmo_ + a];
        }
        if (det.get_beta_bit(p)) {
            matrix_element += tei_bb_[i * nmo3_ + p * nmo2_ + a * nmo_ + p];
        }
    }
    return matrix_element;
}

void ActiveSpaceIntegrals::print() {
    psi::outfile->Printf("\n\n  ==> Active Space Integrals <==\n");
    psi::outfile->Printf("\n  Nuclear repulsion energy:   %20.12f\n", nuclear_repulsion_energy());
    psi::outfile->Printf("  Frozen core energy:         %20.12f\n", frozen_core_energy());
    psi::outfile->Printf("  Scalar energy:              %20.12f\n", scalar_energy());

    psi::outfile->Printf("\nOne-electron integrals (alpha) <p|h|q> (includes restricted docc)\n");
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            if (std::fabs(oei_a(p, q)) > 1e-12)
                psi::outfile->Printf("  <%2d|h|%2d> = %20.12f\n", p, q, oei_a(p, q));
        }
    }
    psi::outfile->Printf("\nOne-electron integrals (beta) <p|h|q> (includes restricted docc)\n");
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            if (std::fabs(oei_b(p, q)) > 1e-12)
                psi::outfile->Printf("  <%2d|h|%2d> = %20.12f\n", p, q, oei_b(p, q));
        }
    }

    psi::outfile->Printf("\nAntisymmetrized two-electron integrals (alpha-alpha) <pq||rs>\n");
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            for (size_t r = 0; r < nmo_; ++r) {
                for (size_t s = 0; s < nmo_; ++s) {
                    if (std::fabs(tei_aa(p, q, r, s)) > 1e-12)
                        psi::outfile->Printf("  <%2d %2d|%2d %2d> = %20.12f\n", p, q, r, s,
                                             tei_aa(p, q, r, s));
                }
            }
        }
    }
    psi::outfile->Printf("\nAntisymmetrized two-electron integrals (beta-beta) <pq||rs>\n");
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            for (size_t r = 0; r < nmo_; ++r) {
                for (size_t s = 0; s < nmo_; ++s) {
                    if (std::fabs(tei_bb(p, q, r, s)) > 1e-12)
                        psi::outfile->Printf("  <%2d %2d|%2d %2d> = %20.12f\n", p, q, r, s,
                                             tei_bb(p, q, r, s));
                }
            }
        }
    }
    psi::outfile->Printf("\nTwo-electron integrals (alpha-beta) <pq|rs>\n");
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            for (size_t r = 0; r < nmo_; ++r) {
                for (size_t s = 0; s < nmo_; ++s) {
                    if (std::fabs(tei_ab(p, q, r, s)) > 1e-12)
                        psi::outfile->Printf("  <%2d %2d|%2d %2d> = %20.12f\n", p, q, r, s,
                                             tei_ab(p, q, r, s));
                }
            }
        }
    }
}

std::shared_ptr<ActiveSpaceIntegrals>
make_active_space_ints(std::shared_ptr<MOSpaceInfo> mo_space_info,
                       std::shared_ptr<ForteIntegrals> ints, const std::string& active_space,
                       const std::vector<std::string>& core_spaces) {

    // get the active/core vectors
    auto active_mo = mo_space_info->corr_absolute_mo(active_space);
    auto active_mo_symmetry = mo_space_info->symmetry(active_space);
    std::vector<size_t> core_mo;
    for (const auto space : core_spaces) {
        auto mos = mo_space_info->corr_absolute_mo(space);
        core_mo.insert(core_mo.end(), mos.begin(), mos.end());
    }

    // allocate the active space integral object
    auto as_ints =
        std::make_shared<ActiveSpaceIntegrals>(ints, active_mo, active_mo_symmetry, core_mo);

    // grab the integrals from the ForteIntegrals object
    ambit::Tensor tei_active_aa = ints->aptei_aa_block(active_mo, active_mo, active_mo, active_mo);
    ambit::Tensor tei_active_ab = ints->aptei_ab_block(active_mo, active_mo, active_mo, active_mo);
    ambit::Tensor tei_active_bb = ints->aptei_bb_block(active_mo, active_mo, active_mo, active_mo);
    as_ints->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);
    as_ints->compute_restricted_one_body_operator();
    return as_ints;
}

} // namespace forte
