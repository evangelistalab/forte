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

namespace psi{ namespace libadaptive{

/// MOInfo stores information about an orbital: (absolute index,irrep,relative index in irrep)
using MOInfo = std::tuple<size_t,size_t,size_t>;
/// SpaceInfo stores information about a space: (Dimension,vector of MOInfo)
using SpaceInfo = std::pair<Dimension,std::vector<MOInfo>>;

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
    /// @return The list of the absolute index of the molecular orbitals in space
    std::vector<size_t> get_absolute_mo(const std::string& space);
    /// @return The list of the absolute index of the molecular orbitals in space
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

void print_method_banner(const std::vector<std::string>& text, const std::string& separator = "-");

template <typename T>
double to_gb(T num_el){
    return static_cast<double>(num_el) * static_cast<double>(sizeof(T)) / 1073741824.0;
}

}} // End Namespaces

#endif // _helpers_h_
