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
#ifndef _mrci_no_h_
#define _mrci_no_h_

#include "base_classes/orbital_transform.h"

namespace forte {

class ForteOptions;

/**
 * @brief The MRCINO class
 * This class implements natural orbitals for CI wave functions
 */
class MRCINO : public OrbitalTransform {
  public:
    // ==> Class Constructor and Destructor <==

    MRCINO(std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
           std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~MRCINO();

    // ==> Class Interface <==

    /// Compute the energy
    void compute_transformation() override;

  private:
    // ==> Class data <==

    /// SCFInfo
    std::shared_ptr<SCFInfo> scf_info_;
    /// Options
    std::shared_ptr<ForteOptions> options_;
    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    /// Pointer to FCI integrals
    std::shared_ptr<ActiveSpaceIntegrals> fci_ints_;
    /// The number of correlated orbitals
    size_t corr_;
    /// The number of active orbitals
    size_t nactv_;

    /// Number of irreps
    int nirrep_;
    /// The number of correlated orbitals per irrep
    psi::Dimension corrpi_;
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

    /// The transformation matrices
    std::shared_ptr<psi::Matrix> Ua_;
    std::shared_ptr<psi::Matrix> Ub_;

    // ==> MRCINO Options <==
    /// Add missing degenerate determinants excluded from the aimed selection?
    bool project_out_spin_contaminants_ = true;
    /// The multiplicity of the reference
    int wavefunction_multiplicity_ = 0;
    // The number of correlated mos
    size_t ncmo2_;
    /// Pass MoSpaceInfo
    bool mrcino_auto;

    // The RDMS
    std::vector<double> ordm_a_;
    std::vector<double> ordm_b_;
    /// Order of RDM to compute
    int rdm_level_;

    // ==> Class functions <==
    /// All that happens before we compute the energy
    void startup();

    std::vector<std::vector<Determinant>> build_dets_cas();

    std::vector<Determinant> build_dets(int irrep,
                                        const std::vector<std::vector<Determinant>>& dets_cas);

    std::pair<std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>>
    diagonalize_hamiltonian(const std::vector<Determinant>& dets, int nsolutions);

    std::pair<std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Matrix>>
    build_density_matrix(const std::vector<Determinant>& dets, std::shared_ptr<psi::Matrix> evecs,
                         int nroot_);

    /// Diagonalize the density matrix
    std::tuple<std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>,
               std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>>
    diagonalize_density_matrix(
        std::pair<std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Matrix>> gamma);

    /// Find optimal active space and transform the orbitals
    void find_active_space_and_transform(
        std::tuple<std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>,
                   std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Matrix>>
            no_U);
};
} // namespace forte

#endif // MRCISNO_H
