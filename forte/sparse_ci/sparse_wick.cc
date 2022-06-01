/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2022 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

// psi4 includes
#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"

// forte includes
#include "sparse_ci/determinant.h"
#include "sparse_ci/sparse_wick.h"

extern std::vector<std::vector<std::vector<std::vector<int>>>> partitions;

#define SPARSE_WICK_DEBUG(msg) ;

namespace forte {

SparseWick::SparseWick(std::shared_ptr<MOSpaceInfo> mo_space_info) : mo_space_info_(mo_space_info) {
    for (auto i : mo_space_info->corr_absolute_mo("RESTRICTED_DOCC")) {
        docc_mask_.set_alfa_bit(i, true);
        docc_mask_.set_beta_bit(i, true);
    }
    for (auto i : mo_space_info->corr_absolute_mo("RESTRICTED_UOCC")) {
        uocc_mask_.set_alfa_bit(i, true);
        uocc_mask_.set_beta_bit(i, true);
    }
    for (auto i : mo_space_info->corr_absolute_mo("ACTIVE")) {
        actv_mask_.set_alfa_bit(i, true);
        actv_mask_.set_beta_bit(i, true);
    }
}

SparseOperator SparseWick::contract(const SparseOperator& lop, const SparseOperator& rop,
                                    int min_nops, int max_nops) {
    SparseOperator result;
    for (const auto& l : lop) {
        for (const auto& r : rop) {
            SPARSE_WICK_DEBUG(
                psi::outfile->Printf("Contracting %s with %s", l.str().c_str(), r.str().c_str());)
            contract_pair(l, r, result, min_nops, max_nops);
        }
    }
    result.simplify();
    return result;
}

void SparseWick::contract_pair(const SQOperator& l, const SQOperator& r, SparseOperator& result,
                               int min_nops, int max_nops) {
    // Check if the two operators have no terms that can be contracted
    if (l.cre().fast_a_and_b_and_c_eq_zero(r.ann(), docc_mask_) and
        l.ann().fast_a_and_b_and_c_eq_zero(r.cre(), uocc_mask_)) {
        // if not then compute only the uncontracted term
        // Check that
        // 1. the uncontracted product contains no repeated operators
        // 2. the number of operators is less than max_ops
        int nops = l.count() + r.count();
        if ((l.cre().fast_a_and_b_eq_zero(r.cre())) and (l.ann().fast_a_and_b_eq_zero(r.ann())) and
            (nops >= min_nops) and (nops <= max_nops)) {
            SPARSE_WICK_DEBUG(
                psi::outfile->Printf("\n  -> leads to a product of normal ordered terms");)
            double msign = merge_sign(l, r);
            double coeff = msign * l.coefficient() * r.coefficient();
            result.add_term(SQOperator(coeff, l.cre() | r.cre(), l.ann() | r.ann()));
        }
    } else {
        make_contractions(l, r, result, 1.0, min_nops, max_nops);
    }
}

SparseOperator SparseWick::commutator(const SparseOperator& lop, const SparseOperator& rop,
                                      int min_nops, int max_nops) {
    SparseOperator result;
    for (const auto& l : lop) {
        for (const auto& r : rop) {
            SPARSE_WICK_DEBUG(
                psi::outfile->Printf("Contracting %s with %s", l.str().c_str(), r.str().c_str());)
            if ((not l.cre().fast_a_and_b_and_c_eq_zero(r.ann(), docc_mask_)) or
                (not l.ann().fast_a_and_b_and_c_eq_zero(r.cre(), uocc_mask_))) {
                int nops = l.count() + r.count();
                // skip the uncontracted term (enforce that max_nops is never equal to nops)
                max_nops = std::min(nops - 1, max_nops);
                make_contractions(l, r, result, 1.0, min_nops, max_nops);
            }
            if ((not r.cre().fast_a_and_b_and_c_eq_zero(l.ann(), docc_mask_)) or
                (not r.ann().fast_a_and_b_and_c_eq_zero(l.cre(), uocc_mask_))) {
                int nops = l.count() + r.count();
                // skip the uncontracted term (enforce that max_nops is never equal to nops)
                max_nops = std::min(nops - 1, max_nops);
                make_contractions(r, l, result, -1.0, min_nops, max_nops);
            }
        }
    }
    result.simplify();
    return result;
}

void SparseWick::make_contractions(const SQOperator& l, const SQOperator& r, SparseOperator& result,
                                   double sign, int min_nops, int max_nops) {
    // find all the contractions
    const int nops = l.count() + r.count();
    const Determinant ca_matches = l.cre() & r.ann() & docc_mask_;
    const Determinant ac_matches = l.ann() & r.cre() & uocc_mask_;
    const int tot_nops_contr = 2 * (ca_matches.count() + ac_matches.count());
    // if the uncontracted product has too few operators, leave
    // if the maximally contracted product has too many operators, leave
    if ((nops < min_nops) or (nops - tot_nops_contr > max_nops))
        return;

    // make arrays that list all the operators to contract
    // cre/ann contractions
    std::array<size_t, Determinant::nbits> ca_matches_vec;
    const int nca = ca_matches.find_all_set(ca_matches_vec);
    // ann/cre contractions
    std::array<size_t, Determinant::nbits> ac_matches_vec;
    const int nac = ac_matches.find_all_set(ac_matches_vec);
    // this is a naive algorithm: generate all contractions and screen them
    for (int nconca = 0; nconca <= nca; nconca++) {
        for (int nconac = 0; nconac <= nac; nconac++) {
            // skip this term if the number of resulting operators falls outside of the bounds
            const int ncon_ops = 2 * (nconca + nconac);
            if ((nops - ncon_ops < min_nops) or (nops - ncon_ops > max_nops))
                continue;

            const auto& part_ca = partitions[nca][nconca];
            const auto& part_ac = partitions[nac][nconac];
            for (int ca = 0, camax = part_ca.size(); ca < camax; ca++) {
                for (int ac = 0, acmax = part_ac.size(); ac < acmax; ac++) {
                    process_contraction(l, r, result, ca_matches_vec, part_ca[ca], ac_matches_vec,
                                        part_ac[ac], sign);
                }
            }
        }
    }
    // // loop over cre/ann contractions
    // for (int ca = 0, camax = part_ca.size(); ca < camax; ca++) {
    //     // loop over ann/cre contractions
    //     for (int ac = 0, acmax = part_ac.size(); ac < acmax; ac++) {
    //         process_contraction(l, r, result, ca_matches_vec, part_ca[ca], ac_matches_vec,
    //                             part_ac[ac], sign);
    //     }
    // }
}

void SparseWick::process_contraction(SQOperator l, SQOperator r, SparseOperator& result,
                                     const std::array<size_t, Determinant::nbits>& ca_matches_vec,
                                     const std::vector<int>& part_ca,
                                     const std::array<size_t, Determinant::nbits>& ac_matches_vec,
                                     const std::vector<int>& part_ac, double sign) {
    // sign from the contraction
    double csign = sign;
    for (auto& i : part_ca) {
        csign *= contract_cre_ann(l, r, ca_matches_vec[i]);
    }
    for (auto& i : part_ac) {
        csign *= contract_ann_cre(l, r, ac_matches_vec[i]);
    }
    // this contraction is valid only if there are no repeated operators
    if ((l.cre().fast_a_and_b_eq_zero(r.cre())) and (l.ann().fast_a_and_b_eq_zero(r.ann()))) {
        double msign = merge_sign(l, r);
        double coeff = csign * msign * l.coefficient() * r.coefficient();
        result.add_term(SQOperator(coeff, l.cre() | r.cre(), l.ann() | r.ann()));
    }
}

double SparseWick::contract_cre_ann(SQOperator& l, SQOperator& r, size_t i) {
    l.cre().set_bit(i, false);
    r.ann().set_bit(i, false);
    // sign to pull the creation operator to the left and annihilation to the right
    double pull_out_sign = l.cre().slater_sign(i) * r.ann().slater_sign(i);
    // sign that counts the number of operators in between the contracted cre/ann pair
    double in_between_sign =
        1.0 - 2.0 * ((l.cre().count() + l.ann().count() + r.cre().count() + r.ann().count()) % 2);
    return pull_out_sign * in_between_sign;
}

double SparseWick::contract_ann_cre(SQOperator& l, SQOperator& r, size_t i) {
    l.ann().set_bit(i, false);
    r.cre().set_bit(i, false);
    // sign to pull the left ann and the right cre operators to the center
    double pull_in_sign = l.ann().slater_sign(i) * r.cre().slater_sign(i);
    return pull_in_sign;
}

double SparseWick::merge_sign(const SQOperator& l, const SQOperator& r) {
    // Computes the sign to merge two SQOperators
    // sign to move the right cre operators to the left of the left ann operators
    double r_c_l_a_permutation_sign = 1. - 2. * (l.ann().count() * r.cre().count() % 2);
    // sign to rearrange the creation operators into decreasing order
    double cre_sign = permutation_sign(l.cre(), r.cre());
    // sign to rearrange the annihilation operators into increasing order
    double ann_sign = permutation_sign(r.ann(), l.ann());
    return r_c_l_a_permutation_sign * cre_sign * ann_sign;
}

double SparseWick::permutation_sign(const Determinant& l, const Determinant& r) {
    double sign = 1.0;
    size_t pos = r.find_first_one();
    double l_parity = 1.0 - 2.0 * (l.count() % 2);
    while (pos != Determinant::end) {
        sign *= l_parity * l.slater_sign(pos);
        pos = r.find_next_one(pos + 1);
    }
    return sign;
}

} // namespace forte

// void SparseWick::process_cre_ann_contraction(
//     SQOperator l, SQOperator r, SparseOperator& result,
//     const std::array<size_t, Determinant::nbits>& matches_vec, const std::vector<size_t>& idx) {
//     // sign from the contraction
//     double csign = 1.0;
//     for (auto& i : idx) {
//         csign *= contract_cre_ann(l, r, matches_vec[i]);
//     }
//     double msign = merge_sign(l, r);
//     double coeff = csign * msign * l.coefficient() * r.coefficient();
//     result.add_term(SQOperator(coeff, l.cre() | r.cre(), l.ann() | r.ann()));
// }

// void SparseWick::process_ann_cre_contraction(
//     SQOperator l, SQOperator r, SparseOperator& result,
//     const std::array<size_t, Determinant::nbits>& matches_vec, const std::vector<size_t>& idx) {
//     // sign from the contraction
//     double csign = 1.0;
//     for (auto& i : idx) {
//         csign *= contract_ann_cre(l, r, matches_vec[i]);
//     }
//     double msign = merge_sign(l, r);
//     double coeff = csign * msign * l.coefficient() * r.coefficient();
//     result.add_term(SQOperator(coeff, l.cre() | r.cre(), l.ann() | r.ann()));
// }