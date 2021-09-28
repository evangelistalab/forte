/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _sci_h_
#define _sci_h_

#include <memory>
#include <vector>

#include "base_classes/active_space_method.h"
#include "base_classes/state_info.h"
#include "sparse_ci/determinant_hashvector.h"
#include "sparse_ci/determinant_substitution_lists.h"
#include "sparse_ci/sigma_vector.h"

namespace psi {
class Vector;
class Matrix;
} // namespace psi

namespace forte {

class ActiveSpaceIntegrals;
class ForteIntegrals;
class ForteOptions;
class MOSpaceInfo;
class Reference;
class SCFInfo;
class SparseCISolver;

class SelectedCIMethod {
  public:
    SelectedCIMethod(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
                     std::shared_ptr<ForteOptions> options,
                     std::shared_ptr<MOSpaceInfo> mo_space_info,
                     std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    /// Virtual destructor to enable deletion of a Derived* through a Base*
    virtual ~SelectedCIMethod() = default;

    // ==> Class Interface <==

    /// Set options from an option object
    /// @param options the options passed in
    virtual void set_options(std::shared_ptr<ForteOptions> options) = 0;

    /// Compute the energy and return it
    virtual double compute_energy();

    // Interfaces of SCI algorithm
    /// Print the banner and starting information.
    virtual void print_info() = 0;
    /// Pre-iter preparation, usually includes preparing an initial reference
    virtual void pre_iter_preparation() = 0;
    /// Step 1. Diagonalize the Hamiltonian in the P space
    virtual void diagonalize_P_space() = 0;
    /// Step 2. Find determinants in the Q space
    virtual void find_q_space() = 0;
    /// Step 3. Diagonalize the Hamiltonian in the P + Q space
    virtual void diagonalize_PQ_space() = 0;
    /// Step 4. Check convergence
    virtual bool check_convergence() = 0;
    /// Step 5. Prune the P + Q space to get an updated P space
    virtual void prune_PQ_to_P() = 0;
    /// Post-iter process
    virtual void post_iter_process() = 0;

    // Temporarily added interface to ExcitedStateSolver
    /// Set the class variable
    virtual void set_method_variables(
        std::string ex_alg, size_t nroot_method, size_t root,
        const std::vector<std::vector<std::pair<Determinant, double>>>& old_roots) = 0;
    /// Getters
    virtual DeterminantHashVec get_PQ_space() = 0;
    virtual std::shared_ptr<psi::Matrix> get_PQ_evecs() = 0;
    virtual std::shared_ptr<psi::Vector> get_PQ_evals() = 0;
    virtual std::vector<double> get_PQ_spin2();
    virtual size_t get_ref_root() = 0;
    virtual std::vector<double> get_multistate_pt2_energy_correction() = 0;
    virtual size_t get_cycle();

    void base_startup();
    void print_wfn(DeterminantHashVec& space, std::shared_ptr<psi::Matrix> evecs, int nroot,
                   size_t max_dets_to_print = 20);

    SigmaVectorType sigma_vector_type() const;
    /// Return the maximum amount of memory allowed
    size_t max_memory() const;

  protected:
    /// The state to calculate
    StateInfo state_;

    /// The number of roots (default = 1)
    size_t nroot_ = 1;

    /// The nuclear repulsion energy
    double nuclear_repulsion_energy_;

    /// The sigma vector type
    SigmaVectorType sigma_vector_type_ = SigmaVectorType::Dynamic;

    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// The molecular integrals for the active space
    /// This object holds only the integrals for the orbital contained in the active_mo_ vector.
    /// The one-electron integrals and scalar energy contains contributions from the
    /// doubly occupied orbitals specified by the core_mo_ vector.
    std::shared_ptr<ActiveSpaceIntegrals> as_ints_;

    /// HF info
    std::shared_ptr<SCFInfo> scf_info_;

    /// Options
    std::shared_ptr<ForteOptions> options_;

    /// The sparse CI solver (allocated at creation by the base class)
    std::shared_ptr<SparseCISolver> sparse_solver_;

    /// The current iteration
    size_t cycle_;

    /// TODO needs documentation
    size_t pre_iter_;

    /// Maximum number of SCI iterations
    size_t max_cycle_;

    /// Maximum memory size
    size_t max_memory_ = 0;

    /// Control amount of printing
    bool quiet_mode_;

    /// Add missing degenerate determinants excluded from the aimed selection?
    bool project_out_spin_contaminants_;

    /// The wave function symmetry
    int wavefunction_symmetry_;
    /// The symmetry of each orbital in Pitzer ordering
    std::vector<int> mo_symmetry_;
    /// The number of correlated molecular orbitals
    int ncmo_;
    /// The multiplicity of the reference
    int multiplicity_;
    /// M_s of the reference
    int twice_ms_;
    /// The number of active electrons
    int nactel_;
    /// The number of correlated alpha electrons
    int nalpha_;
    /// The number of correlated beta electrons
    int nbeta_;
    /// The number of frozen core orbitals
    int nfrzc_;
    /// The number of irreps
    size_t nirrep_;
    psi::Dimension frzcpi_;
    /// The number of correlated molecular orbitals per irrep
    psi::Dimension ncmopi_;
    /// The number of restricted docc orbitals per irrep
    psi::Dimension rdoccpi_;
    /// The number of active orbitals per irrep
    psi::Dimension nactpi_;
    /// The number of restricted docc
    size_t rdocc_;
    /// The number of restricted virtual
    size_t rvir_;
    /// The number of frozen virtual
    size_t fvir_;

    /// The number of active orbitals
    size_t nact_;

    /// Do single calculation instead of selective ci
    bool one_cycle_ = false;

    /// Enforce spin completeness of the P and P + Q spaces?
    bool spin_complete_;
    /// Enforce spin completeness of the P space?
    bool spin_complete_P_;

    // The RDMS
    ambit::Tensor ordm_a_;
    ambit::Tensor ordm_b_;
    ambit::Tensor trdm_aa_;
    ambit::Tensor trdm_ab_;
    ambit::Tensor trdm_bb_;
    ambit::Tensor trdm_aaa_;
    ambit::Tensor trdm_aab_;
    ambit::Tensor trdm_abb_;
    ambit::Tensor trdm_bbb_;
};
} // namespace forte

#endif // _sci_h_
