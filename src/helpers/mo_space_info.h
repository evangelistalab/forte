/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER,
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

#include <algorithm>
#include <chrono>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "ambit/blocked_tensor.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libqt/qt.h"

namespace psi {

class Options;

namespace forte {

/// MOInfo stores information about an orbital: (absolute index,irrep,relative
/// index in irrep)
using MOInfo = std::tuple<size_t, size_t, size_t>;

/// SpaceInfo stores information about a MO space: (Dimension,vector of MOInfo)
using SpaceInfo = std::pair<Dimension, std::vector<MOInfo>>;

/**
 * @brief The MOSpaceInfo class
 *
 * This class reads and holds information about orbital spaces
 *
 * Irrep:                A1(0)       A2(1)    B1(2)   B2(3)
 * ALL:             | 0 1 2 3 4 | 5 6 7 8 9 | 10 11 | 12 13 |
 * CORRELATED:      | - 0 1 2 3 | - 4 5 6 7 |  8  - |  9 10 |
 * RELATIVE:        | 0 1 2 3 4 | 0 1 2 3 4 |  0  1 |  0  1 |
 * FROZEN_DOCC        *           *
 * RESTRICTED_DOCC      *           * *        *       *
 * ACTIVE                 * *           *
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
 * absolute_mo("ACTIVE")          -> [2,3,8]
 * absolute_mo("RESTRICTED_UOCC") -> [4,9]
 * absolute_mo("FROZEN_UOCC")     -> [11]
 *
 * corr_abs_mo("FROZEN_DOCC")     -> []
 * corr_abs_mo("RESTRICTED_DOCC") -> [0,4,5,8,9]
 * corr_abs_mo("ACTIVE")          -> [1,2,6]
 * corr_abs_mo("RESTRICTED_UOCC") -> [3,7,10]
 * corr_abs_mo("FROZEN_UOCC")     -> []
 *
 * get_relative_mo("FROZEN_DOCC")     -> [(0,0),(1,0)]
 * get_relative_mo("RESTRICTED_DOCC") -> [(0,1),(1,1),(1,2),(2,0),(3,0)]
 * get_relative_mo("ACTIVE")          -> [(0,2),(0,3),(1,3)]
 * get_relative_mo("RESTRICTED_UOCC") -> [(0,4),(1,4),(3,1)]
 * get_relative_mo("FROZEN_UOCC")     -> [(2,1)]

 */
class MOSpaceInfo {
  public:
    MOSpaceInfo(Dimension& nmopi);
    ~MOSpaceInfo();

    /// @return The names of orbital spaces
    std::vector<std::string> space_names() const { return space_names_; }
    /// @return The number of orbitals in space
    size_t size(const std::string& space);
    /// @return The Dimension object for space
    Dimension get_dimension(const std::string& space);
    /// @return The Slice object for space in a given composite space
    Slice get_slice(const std::string& space, const std::string& comp_space);
    /// @return The symmetry of each orbital
    std::vector<int> symmetry(const std::string& space);
    /// @return The list of the absolute index of the molecular orbitals in a
    /// space
    std::vector<size_t> get_absolute_mo(const std::string& space);
    /// @return The list of the absolute index of the molecular orbitals in a
    /// space
    ///         excluding the frozen core/virtual orbitals
    std::vector<size_t> get_corr_abs_mo(const std::string& space);
    /// @return The list of the relative index (h,p_rel) of the molecular
    /// orbitals in space
    std::vector<std::pair<size_t, size_t>> get_relative_mo(const std::string& space);
    void read_options(Options& options);
    /// @return The number of irreps
    size_t nirrep() { return nirrep_; }

  private:
    std::pair<SpaceInfo, bool> read_mo_space(const std::string& space, Options& options);

    /// The number of irreducible representations
    size_t nirrep_;
    /// The number of molecular orbitals per irrep
    Dimension nmopi_;
    /// The mo space info
    std::map<std::string, SpaceInfo> mo_spaces_;

    std::vector<std::string> elementary_spaces_{"FROZEN_DOCC", "RESTRICTED_DOCC", "ACTIVE",
                                                "RESTRICTED_UOCC", "FROZEN_UOCC"};
    std::vector<std::string> elementary_spaces_priority_{
        "ACTIVE", "RESTRICTED_UOCC", "RESTRICTED_DOCC", "FROZEN_DOCC", "FROZEN_UOCC"};

    /// Defines composite orbital spaces
    std::map<std::string, std::vector<std::string>> composite_spaces_{
        {"ALL", {"FROZEN_DOCC", "RESTRICTED_DOCC", "ACTIVE", "RESTRICTED_UOCC", "FROZEN_UOCC"}},
        {"FROZEN", {"FROZEN_DOCC", "FROZEN_UOCC"}},
        {"CORRELATED", {"RESTRICTED_DOCC", "ACTIVE", "RESTRICTED_UOCC"}},
        {"INACTIVE_DOCC", {"FROZEN_DOCC", "RESTRICTED_DOCC"}},
        {"INACTIVE_UOCC", {"RESTRICTED_UOCC", "FROZEN_UOCC"}},
        // Spaces for multireference calculations
        {"GENERALIZED HOLE", {"RESTRICTED_DOCC", "ACTIVE"}},
        {"GENERALIZED PARTICLE", {"ACTIVE", "RESTRICTED_UOCC"}},
        {"CORE", {"RESTRICTED_DOCC"}},
        {"VIRTUAL", {"RESTRICTED_UOCC"}}};
    /// The names of the orbital spaces
    std::vector<std::string> space_names_;
    /// The map from all MO to the correlated MOs (excludes frozen core/virtual)
    std::vector<size_t> mo_to_cmo_;
};

} // namespace forte
} // namespace psi

#endif // _mo_space_info_h_
