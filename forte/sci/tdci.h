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

#ifndef _tdci_h_
#define _tdci_h_

#include <fstream>
#include <iomanip>

#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/libpsi4util.h"
#include "psi4/physconst.h"

#include "base_classes/forte_options.h"
#include "ci_rdm/ci_rdms.h"
#include "sparse_ci/ci_reference.h"
#include "integrals/active_space_integrals.h"
#include "sparse_ci/sparse_ci_solver.h"
#include "sparse_ci/determinant.h"
#include "sparse_ci/determinant_hashvector.h"
#include "orbital-helpers/iao_builder.h"
#include "orbital-helpers/localize.h"

#include "sparse_ci/determinant_substitution_lists.h"
#include "ci_ex_states/excited_state_solver.h"
#include "sci/aci.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

namespace forte {

class Reference;
class ActiveSpaceMethod;

/// Set the TDCI options
void set_TDCI_options(ForteOptions& foptions);

/**
 * @brief The TDCI class
 * This class implements time propagation for CI states
 */
class TDCI {
  public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info A pointer to the MOSpaceInfo object
     */
    TDCI(std::shared_ptr<ActiveSpaceMethod> active_space_method, std::shared_ptr<SCFInfo> scf_info,
         std::shared_ptr<ForteOptions> options, std::shared_ptr<MOSpaceInfo> mo_space_info,
         std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    /// Destructor
    ~TDCI();

    // ==> Class Interface <==

    /// Compute the energy
    double compute_energy();

  private:
    std::shared_ptr<SCFInfo> scf_info_;
    std::shared_ptr<ActiveSpaceIntegrals> as_ints_;
    std::shared_ptr<ForteOptions> options_;
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    std::shared_ptr<ActiveSpaceMethod> active_space_method_;

    std::vector<std::vector<double>> occupations_;

    void annihilate_wfn(DeterminantHashVec& olddets, DeterminantHashVec& adets, int frz_orb);

    void renormalize_wfn(std::vector<double>& acoeffs);

    void save_matrix(psi::SharedMatrix mat, std::string str);
    void save_vector(psi::SharedVector vec, std::string str);
    void save_vector(std::vector<double>& vec, std::string str);
    void save_vector(std::vector<size_t>& vec, std::string str);
    void save_vector(std::vector<std::string>& vec, std::string str);

    void propagate_exact(psi::SharedVector C0, psi::SharedMatrix H);
    void propagate_cn(psi::SharedVector C0, psi::SharedMatrix H);
    void propagate_taylor1(psi::SharedVector C0, psi::SharedMatrix H);
    void propagate_taylor2(psi::SharedVector C0, psi::SharedMatrix H);
    void propagate_RK4(psi::SharedVector C0, psi::SharedMatrix H);
    void propagate_QCN(psi::SharedVector C0, psi::SharedMatrix H);
    void propagate_lanczos(psi::SharedVector C0, psi::SharedMatrix H);

    void compute_tdci_select(psi::SharedVector C0);

    void propagate_list(psi::SharedVector C0);

    void propagate_exact_select(std::vector<double>& PQ_coeffs_r, std::vector<double>& PQ_coeffs_i,
                                DeterminantHashVec& PQ_space, double dt);

    void propagate_RK4_select(std::vector<double>& PQ_coeffs_r, std::vector<double>& PQ_coeffs_i,
                              DeterminantHashVec& PQ_space, double dt);

    void propagate_RK4_list(std::vector<double>& PQ_coeffs_r, std::vector<double>& PQ_coeffs_i,
                            DeterminantHashVec& PQ_space, DeterminantSubstitutionLists& op,
                            double dt);
    // The core state determinant space
    DeterminantHashVec core_dets_;
    DeterminantHashVec ann_dets_;
    std::vector<double> compute_occupation(psi::SharedVector Cr, psi::SharedVector Ci,
                                           std::vector<int>& orb);
    std::vector<double> compute_occupation(DeterminantHashVec& dets, std::vector<double>& Cr,
                                           std::vector<double>& Ci, std::vector<int>& orb);

    void get_PQ_space(DeterminantHashVec& P_space, std::vector<double>& P_coeffs_r,
                      std::vector<double>& P_coeffs_i, DeterminantHashVec& PQ_space,
                      std::vector<double>& PQ_coeffs_r, std::vector<double>& PQ_coeffs_i);

    void update_P_space(DeterminantHashVec& P_space, std::vector<double>& P_coeffs_r,
                        std::vector<double>& P_coeffs_i, DeterminantHashVec& PQ_space,
                        std::vector<double>& PQ_coeffs_r, std::vector<double>& PQ_coeffs_i);

    // Compute Hc using coupling lists
    void complex_sigma_build(std::vector<double>& sigma_r, std::vector<double>& sigma_i,
                             std::vector<double>& c_r, std::vector<double>& c_i,
                             DeterminantHashVec& dethash, DeterminantSubstitutionLists& op);

    // Test occupation vectors using ref_occ_n.txt file
    double test_occ();
};

} // namespace forte

#endif // _tdci_h_
