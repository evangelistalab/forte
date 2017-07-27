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

#ifndef _ci_rdms_h_
#define _ci_rdms_h_

#include <cmath>
#include <numeric>

#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"

#include "determinant_hashvector.h"
#include "helpers.h"
#include "operator.h"
#include "reference.h"
#include "stl_bitset_determinant.h"
#include "fci/string_lists.h"
#include "fci/fci_integrals.h"

namespace psi {
namespace forte {

class CI_RDMS {
  public:
    using det_hash = std::unordered_map<STLBitsetDeterminant, size_t, STLBitsetDeterminant::Hash>;
    using det_hash_it = det_hash::iterator;

    // Class constructor and destructor
    CI_RDMS(Options& options, std::shared_ptr<FCIIntegrals>& fci_ints,
            const std::vector<STLBitsetDeterminant>& det_space, SharedMatrix evecs, int root1,
            int root2);

    CI_RDMS(Options& options, DeterminantHashVec& wfn, std::shared_ptr<FCIIntegrals> fci_ints,
            SharedMatrix evecs, int root1, int root2);

    ~CI_RDMS();

    // Return a reference object
    Reference reference(std::vector<double>& oprdm_a, std::vector<double>& oprdm_b,
                        std::vector<double>& tprdm_aa, std::vector<double>& tprdm_bb,
                        std::vector<double>& tprdm_ab, std::vector<double>& tprdm_aaa,
                        std::vector<double>& tprdm_aab, std::vector<double>& tprdm_abb,
                        std::vector<double>& tprdm_bbb);

    // Compute rdms
    void compute_1rdm(std::vector<double>& oprdm_a, std::vector<double>& oprdm_b);

    void compute_1rdm(std::vector<double>& oprdm_a, std::vector<double>& oprdm_b, WFNOperator& op);
    //    void compute_1rdm_str(std::vector<double>& oprdm_a,
    //                          std::vector<double>& oprdm_b);
    void compute_2rdm(std::vector<double>& tprdm_aa, std::vector<double>& tprdm_ab,
                      std::vector<double>& tprdm_bb);

    void compute_2rdm(std::vector<double>& tprdm_aa, std::vector<double>& tprdm_ab,
                      std::vector<double>& tprdm_bb, WFNOperator& op);
    //    void compute_2rdm_str(std::vector<double>& tprdm_aa,
    //                          std::vector<double>& tprdm_ab,
    //                          std::vector<double>& tprdm_bb);
    void compute_3rdm(std::vector<double>& tprdm_aaa, std::vector<double>& tprdm_aab,
                      std::vector<double>& tprdm_abb, std::vector<double>& tprdm_bbb);

    void compute_3rdm(std::vector<double>& tprdm_aaa, std::vector<double>& tprdm_aab,
                      std::vector<double>& tprdm_abb, std::vector<double>& tprdm_bbb,
                      WFNOperator& op);
    //    void compute_3rdm_str(std::vector<double>& tprdm_aaa,
    //                          std::vector<double>& tprdm_aab,
    //                          std::vector<double>& tprdm_abb,
    //                          std::vector<double>& tprdm_bbb);

    double get_energy(std::vector<double>& oprdm_a, std::vector<double>& oprdm_b,
                      std::vector<double>& tprdm_aa, std::vector<double>& tprdm_bb,
                      std::vector<double>& tprdm_ab);

    void rdm_test(std::vector<double>& oprdm_a, std::vector<double>& oprdm_b,
                  std::vector<double>& tprdm_aa, std::vector<double>& tprdm_bb,
                  std::vector<double>& tprdm_ab, std::vector<double>& tprdm_aaa,
                  std::vector<double>& tprdm_aab, std::vector<double>& tprdm_abb,
                  std::vector<double>& tprdm_bbb);

    void set_print(bool print) { print_ = print; }

    void set_max_rdm(int rdm);

    void set_symmetry(int sym) { symmetry_ = sym; }

    // Convert to strings
    void convert_to_string(std::vector<STLBitsetDeterminant>& space);

  private:
    /* Class Variables*/

    // The options object
    Options& options_;
    // The FCI integrals
    std::shared_ptr<FCIIntegrals> fci_ints_;
    // The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    // The Wavefunction
    DeterminantHashVec wfn_;

    // The Determinant Space
    const std::vector<STLBitsetDeterminant> det_space_;

    // The CI coefficients
    SharedMatrix evecs_;

    // Buffer to access cre_list
    std::vector<std::vector<size_t>> cre_list_buffer_;

    // The wavefunction symmetry
    int symmetry_;

    // The number of alpha electrons
    size_t na_;

    // The number of beta electrons
    size_t nb_;

    // The  roots
    int root1_;
    int root2_;

    // The number of correlated mos
    size_t ncmo_;
    size_t ncmo2_;
    size_t ncmo3_;
    size_t ncmo4_;
    size_t ncmo5_;

    // The correlated mos per irrep
    Dimension active_dim_;

    std::vector<size_t> active_mo_;

    // The dimension of the vector space
    size_t dim_space_;

    // Has the one-map been constructed?
    bool one_map_done_;

    bool print_;

    int max_rdm_;

    // The list of a_p |N>
    std::vector<std::vector<std::pair<size_t, short>>> a_ann_list_;
    std::vector<std::vector<std::pair<size_t, short>>> b_ann_list_;

    // The list of a_p |N>
    std::vector<std::pair<size_t, short>> a_ann_list_s_;
    std::vector<std::pair<size_t, short>> b_ann_list_s_;

    // The list of a^(+)_q |N-1>
    std::vector<std::vector<std::pair<size_t, short>>> a_cre_list_;
    std::vector<std::vector<std::pair<size_t, short>>> b_cre_list_;

    // The list of a^(+)_q |N-1>
    //    std::vector<std::pair<size_t, short>> a_cre_list_s_;
    //    std::vector<std::pair<size_t, short>> b_cre_list_s_;

    // The list of a_q a_p|N>
    std::vector<std::vector<std::tuple<size_t, short, short>>> aa_ann_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>> ab_ann_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>> bb_ann_list_;

    // The list of a_q a_p|N>
    std::vector<std::tuple<size_t, short, short>> aa_ann_list_s_;
    std::vector<std::tuple<size_t, short, short>> ab_ann_list_s_;
    std::vector<std::tuple<size_t, short, short>> bb_ann_list_s_;

    // The list of a_q^(+) a_p^(+)|N-2>
    std::vector<std::vector<std::tuple<size_t, short, short>>> aa_cre_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>> ab_cre_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>> bb_cre_list_;

    // The list of a_q^(+) a_p^(+)|N-2>
    //    std::vector<std::tuple<size_t, short, short>> aa_cre_list_s_;
    //    std::vector<std::tuple<size_t, size_t, short, short>> ab_cre_list_s_;
    //    std::vector<std::tuple<size_t, short, short>> bb_cre_list_s_;

    // The list of a_r a_q a_p |N>
    std::vector<std::vector<std::tuple<size_t, short, short, short>>> aaa_ann_list_;
    std::vector<std::vector<std::tuple<size_t, short, short, short>>> aab_ann_list_;
    std::vector<std::vector<std::tuple<size_t, short, short, short>>> abb_ann_list_;
    std::vector<std::vector<std::tuple<size_t, short, short, short>>> bbb_ann_list_;

    // The list of a^(+)_r a^(+)_q a^(+)_p |N-3>
    std::vector<std::vector<std::tuple<size_t, short, short, short>>> aaa_cre_list_;
    std::vector<std::vector<std::tuple<size_t, short, short, short>>> aab_cre_list_;
    std::vector<std::vector<std::tuple<size_t, short, short, short>>> abb_cre_list_;
    std::vector<std::vector<std::tuple<size_t, short, short, short>>> bbb_cre_list_;

    // The list of a_r a_q a_p |N>
    std::vector<std::tuple<size_t, short, short, short>> aaa_ann_list_s_;
    std::vector<std::tuple<size_t, short, short, short>> aab_ann_list_s_;
    std::vector<std::tuple<size_t, short, short, short>> abb_ann_list_s_;
    std::vector<std::tuple<size_t, short, short, short>> bbb_ann_list_s_;

    // The list of a^(+)_r a^(+)_q a^(+)_p |N-3>
    //    std::vector<std::tuple<size_t, short, short, short>> aaa_cre_list_s_;
    //    std::vector<std::tuple<size_t, size_t, short, short, short>>
    //        aab_cre_list_s_;
    //    std::vector<std::tuple<size_t, size_t, short, short, short>>
    //        abb_cre_list_s_;
    //    std::vector<std::tuple<size_t, short, short, short>> bbb_cre_list_s_;

    /* Class functions*/

    // Startup function, mostly just gathering all variables
    void startup();

    // Generate one-particle map
    void get_one_map();

    // Generate two-particle map
    void get_two_map();

    // Generate three-particle map
    void get_three_map();
};
}
} // End namepaces

#endif // _ci_rdms_h_
