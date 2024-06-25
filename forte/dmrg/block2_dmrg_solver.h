/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#pragma once

#include "base_classes/active_space_method.h"
#include "base_classes/rdms.h"
#include "integrals/integrals.h"
#include "base_classes/mo_space_info.h"
#include "base_classes/state_info.h"
#include "base_classes/scf_info.h"
#include "base_classes/forte_options.h"

namespace forte {

struct Block2DMRGSolverImpl;

class Block2DMRGSolver : public ActiveSpaceMethod {
  public:
    /**
     * @brief Block2DMRGSolver Constructor
     * @param state The state info (symmetry, multiplicity, na, nb, etc.)
     * @param nroot Number of roots of interests
     * @param scf_info SCF information
     * @param options Forte options
     * @param mo_space_info MOSpaceInfo
     * @param as_ints Active space integrals
     */
    Block2DMRGSolver(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
                     std::shared_ptr<ForteOptions> options,
                     std::shared_ptr<MOSpaceInfo> mo_space_info,
                     std::shared_ptr<ActiveSpaceIntegrals> as_ints);

    /// Block2DMRGSolver Destructor
    ~Block2DMRGSolver();

    /// Compute the energy
    double compute_energy() override;

    /**
     * @brief Compute the reduced density matrices up to a given particle rank (max_rdm_level)
     *
     *        This function can be used to compute transition density matrices between
     *        states of difference symmetry,
     *
     *        D^{p}_{q} = <I, symmetry_l| a+_p1 ... a_qn |J, symmetry_r>
     *
     *        where |I, symmetry_l> is the I-th state of symmetry = symmetry_l
     *              |J, symmetry_r> is the J-th state of symmetry = symmetry_r
     *
     * @param root_list     a list of pairs of roots to compute [(I_1, J_1), (I_2, J_2), ...]
     * @param method2       a second ActiveSpaceMethod object that holds the states for symmetry_r
     * @param max_rdm_level the maximum RDM rank
     * @return
     */
    std::vector<std::shared_ptr<RDMs>> rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                            int max_rdm_level, RDMsType type) override;

    std::vector<std::shared_ptr<RDMs>>
    transition_rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                    std::shared_ptr<ActiveSpaceMethod> method2, int max_rdm_level,
                    RDMsType type) override;

    /// Set options from an option object
    /// @param options the options passed in
    void set_options(std::shared_ptr<ForteOptions> options) override;

    /// Print the natural_orbitals from DMRG WFN
    /// Assume user specified active space
    void print_natural_orbitals(std::shared_ptr<MOSpaceInfo> mo_space_info,
                                std::shared_ptr<RDMs> rdms);
    
    /// Dump MPS to the current working directory
    void dump_wave_function(const std::string&) override;

  private:
    /// SCFInfo object
    std::shared_ptr<SCFInfo> scf_info_;
    /// ForteOptions
    std::shared_ptr<ForteOptions> options_;
    /// block2 related data
    std::shared_ptr<Block2DMRGSolverImpl> impl_;
    // block2 DMRG sweep options
    std::shared_ptr<ForteOptions> dmrg_options_;
    /// The number of alpha electrons
    int na_;
    /// The number of beta electrons
    int nb_;
};

} // namespace forte
