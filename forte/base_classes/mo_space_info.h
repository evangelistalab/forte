/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER,
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

#ifndef _mo_space_info_h_
#define _mo_space_info_h_

#include "psi4/libmints/dimension.h"

#include "helpers/symmetry.h"
#include "base_classes/forte_options.h"

namespace psi {
class Wavefunction;
class Options;
} // namespace psi

namespace forte {

/// MOInfo stores information about an orbital: (absolute index,irrep,relative
/// index in irrep)
using MOInfo = std::tuple<size_t, size_t, size_t>;

/// SpaceInfo stores information about a MO space: (psi::Dimension,vector of MOInfo)
using SpaceInfo = std::pair<psi::Dimension, std::vector<MOInfo>>;

/**
 * @brief The MOSpaceInfo class
 *
 * This class reads and holds information about orbital spaces
 * The class defines two type of orbital spaces: elementary and composite.
 *
 * ==> ELEMENTARY SPACES <===
 * Within each elementary space, the orbitals are grouped into irreps according to
 * Pitzer ordering, e.g.,
 *
 * RESTRICTED_DOCC = [A1 A1 A2 B1 B2]
 *
 * By default the following elementary spaces are defined
 *
 * ================================================================================
 * Type             Occupation  Occupation      Description
 *                  in CAS/GAS  in correlated
 * --------------------------------------------------------------------------------
 * FROZEN_DOCC          2           2           Frozen doubly occupied orbitals
 * RESTRICTED_DOCC      2          0-2          Restricted doubly occupied orbitals
 * GAS1, GAS2, ...     0-2         0-2          Generalized active spaces
 * RESTRICTED_UOCC      0          0-2          Restricted unoccupied orbitals
 * FROZEN_UOCC          0           0           Frozen unoccupied orbitals
 * ================================================================================
 *
 *
 * ==> COMPOSITE SPACES <===
 *
 * Composite spaces are formed by combining elementary spaces. The following table
 * defines the most important composite spaces used in the MOSpaceInfo class
 *
 * ========================================================================================
 *                   ALL  FROZEN CORRELATED  ACTIVE  GENERALIZED GENERALIZED  CORE  VIRTUAL
 *                                                      HOLE       PARTICLE
 *
 * FROZEN_DOCC        *      *
 * RESTRICTED_DOCC    *              *                   *                      *
 * GAS1 - GAS6        *              *          *        *             *
 * RESTRICTED_UOCC    *              *                                 *               *
 * FROZEN_UOCC        *      *
 * ========================================================================================
 *
 * By convention, orbitals within a composite space are blocked first by symmetry and then
 * by elementary space (Pitzer ordering). For example if the restricted docc and active orbitals are
 *
 * RESTRICTED_DOCC = [A1 A1 | A2 | B1 | B2]
 * ACTIVE = [A1 | B2]
 *
 * then in the composite space HOLE = RESTRICTED_DOCC + ACTIVE the orbitals are arranged as
 *
 * HOLE = [A1 A1 A1 | A2 | B1 | B2 B2]
 * SPACE   R  R  A    R    R    R  A    // R = RESTRICTED_DOCC, A = ACTIVE
 *
 *
 * ==> EXAMPLE <===
 *
 * The following is an example of how the orbitals are assigned when the user specifies
 * the orbitals spaces as below
 *
 * Irrep                 A1(0)       A2(1)    B1(2)   B2(3)
 *
 * Indexing:
 *
 * Absolute index in the full orbital space
 * ALL              | 0 1 2 3 4 | 5 6 7 8 9 | 10 11 | 12 13 |
 *
 * Absolute index in the space of non-frozen orbitals
 * CORRELATED       | - 0 1 2 3 | - 4 5 6 7 |  8  - |  9 10 |
 *
 * Index relative to the irrep in the full orbital space
 * RELATIVE         | 0 1 2 3 4 | 0 1 2 3 4 |  0  1 |  0  1 |
 *
 * FROZEN_DOCC        *           *
 * RESTRICTED_DOCC      *           * *        *       *
 * GAS1                   *             *
 * GAS2                     *
 * RESTRICED_UOCC             *           *               *
 * FROZEN_UOCC                                    *
 *
 * This returns:
 *
 * size("FROZEN_DOCC")     -> 2
 * size("RESTRICTED_DOCC") -> 5
 * size("ACTIVE")          -> 3
 * size("RESTRICTED_UOCC") -> 3
 * size("FROZEN_UOCC")     -> 1
 *
 * dimension("FROZEN_DOCC")     -> [1,1,0,0]
 * dimension("RESTRICTED_DOCC") -> [1,2,1,1]
 * dimension("ACTIVE")          -> [2,1,0,0]
 * dimension("RESTRICTED_UOCC") -> [1,1,0,1]
 * dimension("FROZEN_UOCC")     -> [0,0,1,0]
 *
 * absolute_mo("FROZEN_DOCC")     -> [0,5]
 * absolute_mo("RESTRICTED_DOCC") -> [1,6,7,10,12]
 * absolute_mo("GAS1")            -> [2,8]
 * absolute_mo("GAS2")            -> [3]
 * absolute_mo("ACTIVE")          -> [2,3,8]
 * absolute_mo("RESTRICTED_UOCC") -> [4,9,13]
 * absolute_mo("FROZEN_UOCC")     -> [11]
 *
 * corr_abs_mo("FROZEN_DOCC")     -> []
 * corr_abs_mo("RESTRICTED_DOCC") -> [0,4,5,8,9]
 * corr_abs_mo("GAS1")            -> [1,6]
 * corr_abs_mo("GAS2")            -> [2]
 * corr_abs_mo("ACTIVE")          -> [1,2,6]
 * corr_abs_mo("RESTRICTED_UOCC") -> [3,7,10]
 * corr_abs_mo("FROZEN_UOCC")     -> []
 *
 * relative_mo("FROZEN_DOCC")     -> [(0,0),(1,0)]
 * relative_mo("RESTRICTED_DOCC") -> [(0,1),(1,1),(1,2),(2,0),(3,0)]
 * relative_mo("GAS1")            -> [(0,2),(1,3)]
 * relative_mo("GAS2")            -> [(0,3)]
 * relative_mo("ACTIVE")          -> [(0,2),(0,3),(1,3)]
 * relative_mo("RESTRICTED_UOCC") -> [(0,4),(1,4),(3,1)]
 * relative_mo("FROZEN_UOCC")     -> [(2,1)]
 */

class MOSpaceInfo {
  public:
    // ==> Class Constructor <==
    MOSpaceInfo(const psi::Dimension& nmopi, const std::string& point_group);

    // ==> Class Interface <==

    /// @return A vector of labels of each irrep (e.g. ["A1","A2"])
    const std::vector<std::string>& irrep_labels() const { return symmetry_.irrep_labels(); }
    /// @return The label of each irrep h (e.g. h = 0 -> "A1")
    const std::string& irrep_label(size_t h) const { return symmetry_.irrep_label(h); }
    /// @return The label of the molecular point groupo (e.g. "C2V")
    std::string point_group_label() const { return symmetry_.point_group_label(); }
    /// @return The names of the elementary orbital spaces
    std::vector<std::string> space_names() const { return elementary_spaces_; }
    /// @return The names of the composite orbital spaces
    std::map<std::string, std::vector<std::string>> composite_space_names() const {
        return composite_spaces_;
    }
    /// @return The number of orbitals in a space
    size_t size(const std::string& space);
    /// @return The psi::Dimension object for space
    psi::Dimension dimension(const std::string& space);
    /// @return The symmetry of each orbital
    std::vector<int> symmetry(const std::string& space);
    /// @return The list of the absolute index of the molecular orbitals in a space
    std::vector<size_t> absolute_mo(const std::string& space);
    /// @return The list of the absolute index of the molecular orbitals in a
    /// space excluding the frozen core/virtual orbitals
    std::vector<size_t> corr_absolute_mo(const std::string& space);
    /// @return The list of the relative index (h,p_rel) of the molecular
    /// orbitals in space
    std::vector<std::pair<size_t, size_t>> relative_mo(const std::string& space);
    /// @return The list of the relative index (h,p_rel) of the molecular
    /// orbitals in space
    [[deprecated("this function will soon be removed")]] std::vector<std::pair<size_t, size_t>>
    get_relative_mo(const std::string& space);
    /// @return The position of the orbitals in a space in a larger composite space
    std::vector<size_t> pos_in_space(const std::string& space, const std::string& composite_space);
    /// @return The psi::Slice for a space counting started at absolute zero
    psi::Slice range(const std::string& space);

    /// Read the space info from forte options(inputs)
    void read_options(std::shared_ptr<ForteOptions> options);

    /// Read the space info from a map of space name-dimension_vector
    void read_from_map(const std::map<std::string, std::vector<size_t>>& mo_space_map);

    /// Reorder MOs according to the input indexing vector
    void set_reorder(const std::vector<size_t>& reorder);

    /// Process current MOSpaceInfo: calculate frozen core, count, and assign orbitals
    void compute_space_info();

    /// @return The number of irreps
    size_t nirrep() { return nirrep_; }

  private:
    // ==> Class Data <==
    Symmetry symmetry_;
    /// The number of irreducible representations
    size_t nirrep_;
    /// The number of molecular orbitals per irrep
    psi::Dimension nmopi_;
    /// Information about each elementary space stored in a map
    std::map<std::string, SpaceInfo> mo_spaces_;
    /// The list of elementary spaces
    std::vector<std::string> elementary_spaces_{
        "FROZEN_DOCC", "RESTRICTED_DOCC", "GAS1",       "GAS2", "GAS3", "GAS4", "GAS5",
        "GAS6",        "RESTRICTED_UOCC", "FROZEN_UOCC"};
    /// The priority used to assign orbitals to elementary spaces
    std::vector<std::string> elementary_spaces_priority_{"GAS1",
                                                         "RESTRICTED_UOCC",
                                                         "RESTRICTED_DOCC",
                                                         "FROZEN_DOCC",
                                                         "FROZEN_UOCC",
                                                         "GAS2",
                                                         "GAS3",
                                                         "GAS4",
                                                         "GAS5",
                                                         "GAS6"};

    /// Defines composite orbital spaces
    std::map<std::string, std::vector<std::string>> composite_spaces_{
        {"ALL",
         {"FROZEN_DOCC", "RESTRICTED_DOCC", "GAS1", "GAS2", "GAS3", "GAS4", "GAS5", "GAS6",
          "RESTRICTED_UOCC", "FROZEN_UOCC"}},
        {"FROZEN", {"FROZEN_DOCC", "FROZEN_UOCC"}},
        {"CORRELATED",
         {"RESTRICTED_DOCC", "GAS1", "GAS2", "GAS3", "GAS4", "GAS5", "GAS6", "RESTRICTED_UOCC"}},
        {"ACTIVE", {"GAS1", "GAS2", "GAS3", "GAS4", "GAS5", "GAS6"}},
        {"INACTIVE_DOCC", {"FROZEN_DOCC", "RESTRICTED_DOCC"}},
        {"INACTIVE_UOCC", {"RESTRICTED_UOCC", "FROZEN_UOCC"}},
        // Spaces for multireference calculations
        {"GENERALIZED HOLE", {"RESTRICTED_DOCC", "GAS1", "GAS2", "GAS3", "GAS4", "GAS5", "GAS6"}},
        {"GENERALIZED PARTICLE",
         {"GAS1", "GAS2", "GAS3", "GAS4", "GAS5", "GAS6", "RESTRICTED_UOCC"}},
        {"CORE", {"RESTRICTED_DOCC"}},
        {"VIRTUAL", {"RESTRICTED_UOCC"}}};

    /// The map from all MO to the correlated MOs (excludes frozen core/virtual)
    std::vector<size_t> mo_to_cmo_;

    /// The index vector used to reorder the orbitals
    std::vector<size_t> reorder_;

    // ==> Class functions <==

    /// Read information about each elementary space from the psi Options object
    std::pair<SpaceInfo, bool> read_mo_space(const std::string& space,
                                             std::shared_ptr<ForteOptions> options);

    /// Read information about each elementary space from a map
    std::pair<SpaceInfo, bool>
    read_mo_space_from_map(const std::string& space,
                           const std::map<std::string, std::vector<size_t>>& mo_space_map);
};

/// Make MOSpaceInfo from input (options)
std::shared_ptr<MOSpaceInfo> make_mo_space_info(const psi::Dimension& nmopi,
                                                const std::string& point_group,
                                                std::shared_ptr<ForteOptions> options);

/// Make MOSpaceInfo from a map of spacename-dimension_vector ("ACTIVE", [size_t, size_t, ...])
std::shared_ptr<MOSpaceInfo>
make_mo_space_info_from_map(const psi::Dimension& nmopi, const std::string& point_group,
                            const std::map<std::string, std::vector<size_t>>& mo_space_map,
                            std::vector<size_t> reorder);

} // namespace forte

#endif // _mo_space_info_h_
