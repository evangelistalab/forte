/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#ifndef _ewci_sci_h_
#define _ewci_sci_h_

#include "sci/sci.h"

namespace forte {
using det_hashvec = HashVector<Determinant, Determinant::Hash>;

class EWCI_SCI : public SelectedCIMethod {
  public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info A pointer to the MOSpaceInfo object
     */
    EWCI_SCI(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
                   std::shared_ptr<MOSpaceInfo> mo_space_info,
                   std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    // ==> Class Interface <==

    /// Set options from an option object
    /// @param options the options passed in
    void set_options(std::shared_ptr<ForteOptions> options) override;

    // Interfaces of SCI algorithm
    /// Print the banner and starting information.
    void print_info() override;
    /// Pre-iter preparation, usually includes preparing an initial reference
    void pre_iter_preparation() override;
    /// Step 1. Diagonalize the Hamiltonian in the P space
    void diagonalize_P_space() override;
    /// Step 2. Find determinants in the Q space
    void find_q_space() override;
    /// Step 3. Diagonalize the Hamiltonian in the P + Q space
    void diagonalize_PQ_space() override;
    /// Step 4. Check convergence
    bool check_convergence() override;
    /// Step 5. Prune the P + Q space to get an updated P space
    void prune_PQ_to_P() override;
    /// Post-iter process
    void post_iter_process() override;

    // Temporarily added interface to ExcitedStateSolver
    /// Set the class variable
    virtual void set_method_variables(
        std::string ex_alg, size_t nroot_method, size_t root,
        std::vector<std::vector<std::pair<Determinant, double>>> old_roots) override;
    /// Getters
    DeterminantHashVec get_PQ_space() override;
    psi::SharedMatrix get_PQ_evecs() override;
    psi::SharedVector get_PQ_evals() override;
    WFNOperator get_op() override;
    size_t get_ref_root() override;
    std::vector<double> get_multistate_pt2_energy_correction() override;

  private:
    // ==> Class data <==

};
}

#endif // _ewci_sci_h_
