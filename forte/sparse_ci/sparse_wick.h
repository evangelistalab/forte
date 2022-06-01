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

#ifndef _sparse_wick_h_
#define _sparse_wick_h_

#include "base_classes/mo_space_info.h"

#include "sparse_ci/sparse_operator.h"

namespace forte {

class ActiveSpaceIntegrals;

class SparseWick {
  public:
    /// Constructor
    SparseWick(std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Contract a pair of SparseOperator objects
    SparseOperator contract(const SparseOperator& lop, const SparseOperator& rop, int min_nops = 0,
                            int max_nops = 20);

    /// Contract a pair of SparseOperator objects
    SparseOperator commutator(const SparseOperator& lop, const SparseOperator& rop, int min_nops,
                              int max_nops);

  private:
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    Determinant docc_mask_;
    Determinant actv_mask_;
    Determinant uocc_mask_;

    void contract_pair(const SQOperator& l, const SQOperator& r, SparseOperator& result,
                       int min_nops, int max_nops);
    void make_contractions(const SQOperator& l, const SQOperator& r, SparseOperator& result,
                           double sign, int min_nops, int max_nops);
    void process_contraction(SQOperator l, SQOperator r, SparseOperator& result,
                             const std::array<size_t, Determinant::nbits>& ca_matches_vec,
                             const std::vector<int>& part_ca,
                             const std::array<size_t, Determinant::nbits>& ac_matches_vec,
                             const std::vector<int>& part_ac, double sign);
    double contract_cre_ann(SQOperator& l, SQOperator& r, size_t i);
    double contract_ann_cre(SQOperator& l, SQOperator& r, size_t i);
    double merge_sign(const SQOperator& l, const SQOperator& r);
    double permutation_sign(const Determinant& l, const Determinant& r);
};

} // namespace forte

#endif // _sparse_wick_h_
