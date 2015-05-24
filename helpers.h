/*
 *@BEGIN LICENSE
 *
 * Libadaptive: an ab initio quantum chemistry software package
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *@END LICENSE
 */

#ifndef _helpers_h_
#define _helpers_h_

#include <map>
#include <vector>
#include <string>

#include "ambit/blocked_tensor.h"
#include <libmints/matrix.h>
#include <libmints/vector.h>

namespace psi{ namespace libadaptive{

/// MOInfo stores information about an orbital: (absolute index,irrep,relative index in irrep)
using MOInfo = std::tuple<size_t,size_t,size_t>;

/// SpaceInfo stores information about a MO space: (Dimension,vector of MOInfo)
using SpaceInfo = std::pair<Dimension,std::vector<MOInfo>>;

/**
 * @brief The MOSpaceInfo class
 *
 * This class reads and holds information about orbital spaces
 *
 * Irrep:                 A1         A2        B1      B2
 * ALL:             | 0 1 2 3 4 | 5 6 7 8 9 |10 11 | 12 13 |
 * FROZEN_DOCC        *           *
 * RESTRICTED_DOCC      *           *         *       *
 * ACTIVE                 * *         * *
 * FROZEN_UOCC                *           *      *       *
 */
class MOSpaceInfo
{
public:
    MOSpaceInfo();
    ~MOSpaceInfo();

    /// @return The names of orbital spaces
    std::vector<std::string> space_names() const {return space_names_;}
    /// @return The number of orbitals in space
    size_t size(const std::string& space);
    /// @return The Dimension object for space
    Dimension get_dimension(const std::string& space);
    /// @return The list of the absolute index of the molecular orbitals in a space
    std::vector<size_t> get_absolute_mo(const std::string& space);
    /// @return The list of the absolute index of the molecular orbitals in a space
    ///         excluding the frozen core/virtual orbitals
    std::vector<size_t> get_corr_abs_mo(const std::string& space);
    /// @return The list of the relative index (h,p_rel) of the molecular orbitals in space
    std::vector<std::pair<size_t,size_t>> get_relative_mo(const std::string& space);

    void  read_options(Options& options);
private:

    std::pair<SpaceInfo,bool> read_mo_space(const std::string& space,Options& options);

    /// The number of irreducible representations
    size_t nirrep_;
    /// The number of molecular orbitals per irrep
    Dimension nmopi_;
    /// The mo space info
    std::map<std::string,SpaceInfo> mo_spaces_;

    std::vector<std::string> elementary_spaces_{"FROZEN_DOCC","RESTRICTED_DOCC","ACTIVE","RESTRICTED_UOCC","FROZEN_UOCC"};
    std::vector<std::string> elementary_spaces_priority_{"ACTIVE","RESTRICTED_UOCC","RESTRICTED_DOCC","FROZEN_DOCC","FROZEN_UOCC"};

    std::vector<std::pair<std::string,std::vector<std::string>>> composite_spaces_{{"INACTIVE_DOCC",{"FROZEN_DOCC","RESTRICTED_DOCC"}},
                                                                                   {"INACTIVE_UOCC",{"RESTRICTED_UOCC","FROZEN_UOCC"}}};
    /// The names of the orbital spaces
    std::vector<std::string> space_names_;
    /// The map from all MO to the correlated MOs (excludes frozen core/virtual)
    std::vector<size_t> mo_to_cmo_;
};

/**
 * @brief tensor_to_matrix
 * @param t The input tensor
 * @param dims Dimensions of the matrix extracted from the tensor
 * @return A copy of the tensor data in symmetry blocked form
 */
Matrix tensor_to_matrix(ambit::Tensor t,Dimension dims);

/**
 * @brief print_method_banner Print a banner
 * @param text A vector of strings to print in the banner. Each string is a line.
 * @param separator A string The separator used in the banner (defalut = "-").
 */
void print_method_banner(const std::vector<std::string>& text, const std::string& separator = "-");

/**
 * @brief print_h2 Print a header
 * @param text The string to print in the header.
 * @param separator
 */
void print_h2(const std::string& text, const std::string& left_separator = "==>", const std::string& right_separator  = "<==");

/**
 * @brief Compute the memory (in GB) required to store arrays
 * @typename T The data typename
 * @param num_el The number of elements to store
 * @return The size in GB
 */
template <typename T>
double to_gb(T num_el){
    return static_cast<double>(num_el) * static_cast<double>(sizeof(T)) / 1073741824.0;
}

}} // End Namespaces

#endif // _helpers_h_
