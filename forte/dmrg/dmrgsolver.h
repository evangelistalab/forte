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

#ifndef _dmrgsolver_h_
#define _dmrgsolver_h_

#include "psi4/libmints/wavefunction.h"
#include "psi4/libfock/jk.h"

#include "chemps2/Irreps.h"
#include "chemps2/Problem.h"
#include "chemps2/CASSCF.h"
#include "chemps2/Initialize.h"
#include "chemps2/EdmistonRuedenberg.h"

#include "base_classes/active_space_method.h"
#include "base_classes/rdms.h"
#include "integrals/integrals.h"
#include "base_classes/mo_space_info.h"
#include "base_classes/state_info.h"
#include "base_classes/scf_info.h"
#include "base_classes/forte_options.h"

namespace forte {

class DMRGSolver : public ActiveSpaceMethod {
  public:
    /**
     * @brief DMRGSolver Constructor
     * @param state The state info (symmetry, multiplicity, na, nb, etc.)
     * @param nroot Number of roots of interests
     * @param scf_info SCF information
     * @param options Forte options
     * @param mo_space_info MOSpaceInfo
     * @param as_ints Active space integrals
     */
    DMRGSolver(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
               std::shared_ptr<ForteOptions> options, std::shared_ptr<MOSpaceInfo> mo_space_info,
               std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    /// Compute the energy
    double compute_energy() override;

    /// RDMs override
    std::vector<RDMs> rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                           int max_rdm_level) override;

    /// Transition RDMs override
    std::vector<RDMs> transition_rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                      std::shared_ptr<ActiveSpaceMethod> method2,
                                      int max_rdm_level) override;

    /// Set options override
    void set_options(std::shared_ptr<ForteOptions> options) override;

    /// Return the CI wave functions for current state symmetry
    //    psi::SharedMatrix ci_wave_functions() override { return evecs_; }

    /// Dump wave function to disk
    //    void dump_wave_function(const std::string& filename) override;

    /// Read wave function from disk
    /// Return the number of active orbitals, set of determinants, CI coefficients
    //    std::tuple<size_t, std::vector<Determinant>, psi::SharedMatrix>
    //    read_wave_function(const std::string& filename) override;

    //    RDMs rdms() { return dmrg_rdms_; }
    //    void set_max_rdm(int max_rdm) { max_rdm_ = max_rdm; }
    //    void spin_free_rdm(bool spin_free) { spin_free_rdm_ = spin_free; }
    //    void disk_3_rdm(bool use_disk_for_3rdm) { disk_3_rdm_ = use_disk_for_3rdm; }
    //    void set_up_integrals(const ambit::Tensor& active_integrals,
    //                          const std::vector<double>& one_body) {
    //        active_integrals_ = active_integrals;
    //        one_body_integrals_ = one_body;
    //        use_user_integrals_ = true;
    //    }
    //    void set_scalar(double energy) { scalar_energy_ = energy; }

  private:
    /// SCFInfo object
    std::shared_ptr<SCFInfo> scf_info_;
    /// ForteOptions
    std::shared_ptr<ForteOptions> options_;

    /// Point group number in CheMPS2
    int pg_number_;
    /// Number of irreps
    int nirrep_;
    /// Wave function symmetry
    int wfn_irrep_;
    /// Multiplicity
    int multiplicity_;
    /// Number of active electrons
    int nelecs_actv_;
    /// Number of active orbitals
    int nactv_;
    /// State label
    std::string state_label_;

    /// Vector of spin-free 1RDMs
    std::vector<ambit::Tensor> opdms_;
    /// Vector of spin-free 2RDMs
    std::vector<ambit::Tensor> tpdms_;
    /// Fill the spin-free 1- and 2-RDMs to the vector of ambit Tensor
    void push_back_rdms();

    /// Number of reduced renormalized basis states kept during successive DMRG instructions
    std::vector<int> dmrg_sweep_states_;
    /// Energy convergence to stop an instruction during successive DMRG instructions
    std::vector<double> dmrg_sweep_e_convergence_;
    /// Max number of sweeps to stop an instruction during successive DMRG instructions
    std::vector<int> dmrg_sweep_max_sweeps_;
    /// The noise prefactors for successive DMRG instructions
    std::vector<double> dmrg_noise_prefactors_;
    /// The residual tolerances for the Davidson diagonalization during DMRG instructions
    std::vector<double> dmrg_davidson_rtol_;
    /// Whether or not to print the correlation functions after the DMRG calculation
    bool dmrg_print_corr_;

    /// The DMRG solver
    std::unique_ptr<CheMPS2::DMRG> solver_;

    /// Vector of file names
    std::vector<std::string> mps_files_;

    /// Setup some internal variable
    void startup();

    /// Return the RDMs for the current state
    RDMs fill_current_rdms(const bool do_3rdm);

    //    RDMs dmrg_rdms_;
    //    bool disk_3_rdm_ = false;
    //    void compute_reference(double* one_rdm, double* two_rdm, double* three_rdm,
    //                           CheMPS2::DMRGSCFindices* iHandler);
    //    /// By default, compute the second rdm.  If you are doing MRPT2, may need to
    //    /// change this.
    //    int max_rdm_ = 3;
    //    bool spin_free_rdm_ = false;
    //    int chemps2_groupnumber(const string SymmLabel);
    //    ambit::Tensor active_integrals_;
    //    std::vector<double> one_body_integrals_;
    //    double scalar_energy_ = 0.0;
    //    std::vector<double> one_body_operator();
    //    bool use_user_integrals_ = false;
    //    void print_natural_orbitals(double* one_rdm);
};
} // namespace forte
#endif // _dmrgsolver_h_