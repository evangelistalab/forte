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

#ifndef _determinant_substitution_lists_h_
#define _determinant_substitution_lists_h_

#include "integrals/active_space_integrals.h"
#include "sparse_ci/determinant_hashvector.h"
#include "sparse_ci/determinant.h"
#include "sparse_ci/sorted_string_list.h"

namespace forte {

/**
 * @brief A class to compute various expectation values, projections,
 * and matrix elements of quantum mechanical operators on wavefunction objects.
 */

using wfn_hash = det_hash<double>;

class DeterminantSubstitutionLists {
  public:
    /// Default constructor
    DeterminantSubstitutionLists(const std::vector<int>& mo_symmetry);

    /// Set print level
    void set_quiet_mode(bool mode);

    /// Build the coupling lists for one-particle operators
    void op_s_lists(const DeterminantHashVec& wfn);

    /// Build the coupling lists for two-particle operators
    void tp_s_lists(const DeterminantHashVec& wfn);

    /// Build the coupling lists for three-particle operators
    void three_s_lists(const DeterminantHashVec& wfn);

    /// Build the coupling lists for a 1-body operators
    void lists_1a(const DeterminantHashVec& wfn);
    /// Build the coupling lists for b 1-body operators
    void lists_1b(const DeterminantHashVec& wfn);

    /// Build the coupling lists for aa 2-body operators
    void lists_2aa(const DeterminantHashVec& wfn);
    /// Build the coupling lists for ab 2-body operators
    void lists_2ab(const DeterminantHashVec& wfn);
    /// Build the coupling lists for bb 2-body operators
    void lists_2bb(const DeterminantHashVec& wfn);

    /// Build the coupling lists for aaa 3-body operators
    void lists_3aaa(const DeterminantHashVec& wfn);
    /// Build the coupling lists for aab 3-body operators
    void lists_3aab(const DeterminantHashVec& wfn);
    /// Build the coupling lists for abb 3-body operators
    void lists_3abb(const DeterminantHashVec& wfn);
    /// Build the coupling lists for bbb 3-body operators
    void lists_3bbb(const DeterminantHashVec& wfn);

    // Clear coupling lists for 1-body operators
    void clear_op_s_lists();
    // Clear coupling lists for 2-body operators
    void clear_tp_s_lists();
    // Clear coupling lists for 3-body operators
    void clear_3p_s_lists();

    /*- Operators -*/

    void build_strings(const DeterminantHashVec& wfn);

    std::vector<std::vector<std::pair<size_t, short>>> a_list_;
    std::vector<std::vector<std::pair<size_t, short>>> b_list_;

    std::vector<std::vector<std::tuple<size_t, short, short>>> aa_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>> bb_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>> ab_list_;

    /// Three particle lists
    std::vector<std::vector<std::tuple<size_t, short, short, short>>> aaa_list_;
    std::vector<std::vector<std::tuple<size_t, short, short, short>>> aab_list_;
    std::vector<std::vector<std::tuple<size_t, short, short, short>>> abb_list_;
    std::vector<std::vector<std::tuple<size_t, short, short, short>>> bbb_list_;

  protected:
    /// Initialize important variables on construction
    void startup();

    std::vector<std::vector<size_t>> beta_strings_;
    std::vector<std::vector<size_t>> alpha_strings_;
    std::vector<std::vector<std::pair<int, size_t>>> alpha_a_strings_;
    std::vector<std::vector<std::pair<int, size_t>>> beta_a_strings_;

    /// Number of active space orbitals
    size_t ncmo_;
    /// Active space symmetry
    std::vector<int> mo_symmetry_;

    /// Print level
    bool quiet_ = false;
};
} // namespace forte

#endif // _determinant_substitution_lists_h_
