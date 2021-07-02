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

#ifndef _ci_reference_h_
#define _ci_reference_h_

#include "integrals/active_space_integrals.h"
#include "sparse_ci/determinant.h"
#include "base_classes/mo_space_info.h"
#include "base_classes/scf_info.h"
#include "base_classes/state_info.h"

namespace forte {

class CI_Reference {
  protected:
    // The wavefunction object
    std::shared_ptr<SCFInfo> scf_info_;

    // Pointer to the state info object
    StateInfo state_info_;

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

    // Pointer to the options object
    std::shared_ptr<ForteOptions> options_;

    // Number of active MOs
    int nact_;

    // Symmetry of each active MO
    std::vector<int> mo_symmetry_;

    // Number of active MOs per irrep
    psi::Dimension nactpi_;

    // Number of frozen_docc + restriced_docc MOs
    psi::Dimension frzcpi_;

    // Returns MO energies, symmetries, and indicies, sorted
    std::vector<std::tuple<double, int, int>> sym_labeled_orbitals(std::string type);

    std::string ref_type_;

    void build_ci_reference(std::vector<Determinant>& ref_space);

    Determinant get_occupation();

    void get_gas_occupation();

    std::pair<std::vector<int>, std::vector<int>> get_gas_max_min();

    std::shared_ptr<ActiveSpaceIntegrals> fci_ints_;

    /// The number of used GAS
    size_t gas_num_;

    /// All the possible alpha and beta electron occupations of each GAS
    /// GAS1_A, GAS1_B, GAS2_A, .... GAS6_B (12 elements)
    std::vector<std::vector<int>> gas_electrons_;

    /// Print SCF orbital energies for GAS
    void print_gas_scf_epsilon();

    /// Compute the occupation string for a given number of electrons and orbitals
    /// @return nirrep of vector of occupation
    std::vector<std::vector<std::vector<bool>>> build_occ_string(size_t norb, size_t nele,
                                                                 const std::vector<int>& symmetry);

    /// Compute the cartesian product of occupation strings
    /// @arg vector of vector of occupation
    /// @return nirrep of vector of occupation
    std::vector<std::vector<bool>>
    build_gas_occ_string(const std::vector<std::vector<std::vector<bool>>>& gas_strings,
                         const std::vector<std::vector<size_t>>& rel_mos);

  public:
    /// Default constructor
    CI_Reference(std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
                 std::shared_ptr<MOSpaceInfo> mo_space_info,
                 std::shared_ptr<ActiveSpaceIntegrals> fci_ints, int multiplicity, double twice_ms,
                 int symmetry, StateInfo state_info);

    /// Destructor
    ~CI_Reference();

    /// Build a reference
    void build_reference(std::vector<Determinant>& ref_space);

    /// Build the CAS reference (with a high limit of basis functions)
    void build_cas_reference(std::vector<Determinant>& ref_space);

    /// Build the complete CAS reference
    void build_cas_reference_full(std::vector<Determinant>& ref_space);

    /// Build the doubly occupied CI reference
    void build_doci_reference(std::vector<Determinant>& ref_space);

    /// Build the complete GAS reference
    void build_gas_reference(std::vector<Determinant>& ref_space);

    /// Build single lowest energy state
    void build_gas_single(std::vector<Determinant>& ref_space);

    /// Set the reference type
    void set_ref_type(const std::string& ref_type) { ref_type_ = ref_type; }

    /// Return a pair for single excitation between different GAS
    /// alpha, beta
    std::pair<std::map<std::vector<int>, std::vector<std::pair<size_t, size_t>>>,
              std::map<std::vector<int>, std::vector<std::pair<size_t, size_t>>>>
    gas_single_criterion();

    /// Return a tuple of maps for double excitation between different GAS
    /// alphaalpha, betabeta, alphabeta
    std::tuple<std::map<std::vector<int>, std::vector<std::tuple<size_t, size_t, size_t, size_t>>>,
               std::map<std::vector<int>, std::vector<std::tuple<size_t, size_t, size_t, size_t>>>,
               std::map<std::vector<int>, std::vector<std::tuple<size_t, size_t, size_t, size_t>>>>
    gas_double_criterion();

    /// return gas_electrons_
    std::vector<std::vector<int>> gas_electrons();
};
} // namespace forte

#endif // _ci_reference_h_
