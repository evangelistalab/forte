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

#ifndef _ci_no_h_
#define _ci_no_h_

#include "psi4/psi4-dec.h"
#include "psi4/libmints/molecule.h"
#include "helpers/timer.h"
#include "ci_rdm/ci_rdms.h"
#include "integrals/active_space_integrals.h"
#include "base_classes/forte_options.h"
#include "sparse_ci/sparse_ci_solver.h"
#include "sparse_ci/determinant.h"

#include "base_classes/orbital_transform.h"

namespace forte {

class ForteOptions;

/**
 * @brief The CINO class
 * This class implements natural orbitals for CI wave functions
 */
class CINO : public OrbitalTransform {
  public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info A pointer to the MOSpaceInfo object
     */
    CINO(std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
         std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~CINO();

    // ==> Class Interface <==

    /// Compute the energy
    void compute_transformation();

    psi::SharedMatrix get_Ua();
    psi::SharedMatrix get_Ub();

  private:
    // ==> Class data <==

    /// ForteOptions
    std::shared_ptr<ForteOptions> options_;
    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    /// Pointer to FCI integrals
    std::shared_ptr<ActiveSpaceIntegrals> fci_ints_;
    /// The number of orbitals
    size_t nmo_;
    /// The number of active orbitals
    size_t nactv_;
    /// The number of irreps
    int nirrep_;
    /// The number of active orbitals per irrep
    psi::Dimension actvpi_;
    /// The number of restricted doubly occupied orbitals per irrep
    psi::Dimension rdoccpi_;
    /// The number of frozen doubly occupied orbitals per irrep
    psi::Dimension fdoccpi_;
    /// The number of restricted virtual orbitals per irrep
    psi::Dimension ruoccpi_;
    /// The number of frozen virtual orbitals per irrep
    psi::Dimension fuoccpi_;
    /// The number of alpha occupied active orbitals per irrep
    psi::Dimension aoccpi_;
    //    /// The number of alpha unoccupied active orbitals per irrep
    //    psi::Dimension avirpi_;
    /// The number of beta occupied active orbitals per irrep
    psi::Dimension boccpi_;
    //    /// The number of beta unoccupied active orbitals per irrep
    //    psi::Dimension bvirpi_;

    // The transformation matrices
    psi::SharedMatrix Ua_;
    psi::SharedMatrix Ub_;

    // ==> CINO Options <==
    /// Add missing degenerate determinants excluded from the aimed selection?
    bool project_out_spin_contaminants_ = true;
    /// The multiplicity of the reference
    int wavefunction_multiplicity_ = 0;
    // The number of correlated mos
    size_t ncmo2_;
    /// Pass MoSpaceInfo
    bool cino_auto;

    // The RDMS
    std::vector<double> ordm_a_;
    std::vector<double> ordm_b_;
    /// Order of RDM to compute
    int rdm_level_;

    // ==> Class functions <==
    /// All that happens before we compute the energy
    void startup();

    // std::vector<std::vector<Determinant> > build_dets_cas();

    std::vector<Determinant> build_dets(int irrep);

    std::pair<psi::SharedVector, psi::SharedMatrix>
    diagonalize_hamiltonian(const std::vector<Determinant>& dets, int nsolutions);

    std::pair<psi::SharedMatrix, psi::SharedMatrix>
    build_density_matrix(const std::vector<Determinant>& dets, psi::SharedMatrix evecs, int nroot_);

    /// Diagonalize the density matrix
    std::tuple<psi::SharedVector, psi::SharedMatrix, psi::SharedVector, psi::SharedMatrix>
    diagonalize_density_matrix(std::pair<psi::SharedMatrix, psi::SharedMatrix> gamma);

    /// Find optimal active space and transform the orbitals
    void find_active_space_and_transform(
        std::tuple<psi::SharedVector, psi::SharedMatrix, psi::SharedVector, psi::SharedMatrix>
            no_U);
};
std::string dimension_to_string(psi::Dimension dim);
} // namespace forte

#endif // _ci_no_h_
