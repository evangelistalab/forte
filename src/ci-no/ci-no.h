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

#ifndef _ci_no_h_
#define _ci_no_h_

//#include <fstream>
//#include <iomanip>

#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"
//#include "psi4/physconst.h"

#include "../forte_options.h"
#include "../helpers.h"
#include "../integrals/integrals.h"

#include "../ci_rdms.h"
//#include "../determinant_map.h"
//#include "../fci/fci_integrals.h"
//#include "../operator.h"
#include "../sparse_ci_solver.h"
#include "../stl_bitset_determinant.h"

namespace psi {
namespace forte {

/// Set the CI-NO options
void set_CINO_options(ForteOptions& foptions);

/**
 * @brief The CINO class
 * This class implements natural orbitals for CI wave functions
 */
class CINO : public Wavefunction {
  public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info A pointer to the MOSpaceInfo object
     */
    CINO(SharedWavefunction ref_wfn, Options& options,
         std::shared_ptr<ForteIntegrals> ints,
         std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~CINO();

    // ==> Class Interface <==

    /// Compute the energy
    double compute_energy();

  private:
    // ==> Class data <==

    /// The molecular integrals
    std::shared_ptr<ForteIntegrals> ints_;
    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    /// Pointer to FCI integrals
    std::shared_ptr<FCIIntegrals> fci_ints_;
    /// The number of active orbitals
    size_t nactv_;
    /// The number of active orbitals per irrep
    Dimension actvpi_;
    /// The number of restricted doubly occupied orbitals per irrep
    Dimension rdoccpi_;
    /// The number of frozen doubly occupied orbitals per irrep
    Dimension fdoccpi_;
    /// The number of alpha occupied active orbitals per irrep
    Dimension aoccpi_;
//    /// The number of alpha unoccupied active orbitals per irrep
//    Dimension avirpi_;
//    /// The number of beta occupied active orbitals per irrep
//    Dimension boccpi_;
//    /// The number of beta unoccupied active orbitals per irrep
//    Dimension bvirpi_;

    // ==> CINO Options <==
    /// Add missing degenerate determinants excluded from the aimed selection?
    bool project_out_spin_contaminants_ = true;
    /// The eigensolver type
    DiagonalizationMethod diag_method_;
    /// The multiplicity of the reference
    int wavefunction_multiplicity_ = 0;
    // The number of correlated mos
    size_t ncmo2_;

    //The RDMS
    std::vector<double> ordm_a_;
    std::vector<double> ordm_b_;
    /// Order of RDM to compute
    int rdm_level_;

    // ==> Class functions <==
    /// All that happens before we compute the energy
    void startup();

    std::vector<Determinant> build_dets(int irrep);

    std::pair<SharedVector, SharedMatrix>
    diagonalize_hamiltonian(const std::vector<Determinant>& dets, int nsolutions);

    std::pair<SharedMatrix, SharedMatrix> build_density_matrix(const std::vector<Determinant>& dets,
                                      SharedMatrix evecs, int nroot_);

    /// Diagonalize the density matrix
    std::tuple<SharedVector, SharedMatrix, SharedVector, SharedMatrix> diagonalize_density_matrix(std::pair<SharedMatrix, SharedMatrix> gamma);

    /// Find optimal active space and transform the orbitals
    void
    find_active_space_and_transform(std::tuple<SharedVector, SharedMatrix, SharedVector, SharedMatrix> no_U);
};
}
} // End Namespaces

#endif // _ci_no_h_
