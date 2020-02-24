/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _ci_reference_h_
#define _ci_reference_h_



#include "integrals/active_space_integrals.h"
#include "sparse_ci/determinant.h"
#include "base_classes/mo_space_info.h"
#include "base_classes/scf_info.h"

namespace forte {

class CI_Reference
{
  protected:
    // The wavefunction object
    std::shared_ptr<SCFInfo> scf_info_;

    // Multiplicity of the reference
    int multiplicity_;

    // Twice the Ms
    double twice_ms_;

    // Number of active alpha electrons
    int nalpha_;

    // Number of active beta electrons
    int nbeta_;

    // Symmetry of the reference
    int root_sym_;

    // Number of irreps
    int nirrep_;

    // Maximum number of determinants
    size_t subspace_size_;

    // Pointer to the MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    // Number of active MOs
    int nact_;

    // Symmetry of each active MO
    psi::Dimension mo_symmetry_;

    // Number of active MOs per irrep
    psi::Dimension nactpi_;

    // Number of frozen_docc + restriced_docc MOs
    psi::Dimension frzcpi_;

    // Returns MO energies, symmetries, and indicies, sorted
    std::vector<std::tuple<double, int, int>> sym_labeled_orbitals(std::string type);

    std::string ref_type_;

    void build_ci_reference(std::vector<Determinant>& ref_space);
    void build_cas_reference(std::vector<Determinant>& ref_space);

    Determinant get_occupation();

    std::shared_ptr<ActiveSpaceIntegrals> fci_ints_;

  public:
    /// Default constructor
    CI_Reference(std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
                 std::shared_ptr<MOSpaceInfo> mo_space_info,
                 std::shared_ptr<ActiveSpaceIntegrals> fci_ints, int multiplicity, double ms,
                 int symmetry);

    /// Destructor
    ~CI_Reference();

    /// Build a reference
    void build_reference(std::vector<Determinant>& ref_space);

    /// Set the reference type
    void set_ref_type(const std::string& ref_type) { ref_type_ = ref_type; }
};
}

#endif // _ci_reference_h_
