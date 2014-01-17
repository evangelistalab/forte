/*
 *@BEGIN LICENSE
 *
 * Basic Tensor Library: a library to perform common tensor operations
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


#ifndef _tensor_blocked_h_
#define _tensor_blocked_h_

#include <string>
#include <vector>

class Tensor;
class LabeledTensor;
class LabeledTensorProduct;

/// Represent a tensor aware of spin and MO spaces
/// This class holds several tensors, blocked according to spin and MO spaces.
///
/// Sample usage:
///    BlockedTensor::add_mo_space("occupied","ijkl",{0,1,2,3,4});
///    BlockedTensor::add_mo_space("virtual" ,"abcd",{7,8,9});
///    BlockedTensor::add_mo_space("all"     ,"pqrst",{0,1,2,3,4,5,6,7,8,9});
///    BlockedTensor::add_mo_space("active"  ,"uvwxyz",{5,6});
///    BlockedTensor::add_mo_space("gen. occ"  ,"mn",{"occupied","active"});
///    BlockedTensor::add_mo_space("gen. vir"  ,"ef",{"active","virtual"});
///
///    BlockedTensor T("T","abij");
///    BlockedTensor V("V","abij");
///    E = 0.25 * T("abij") * V("abij")
class BlockedTensor
{
public:
    static void add_mo_space(std::string label,std::vector<std::string> symbols,std::vector<size_t> mo);
private:
    /// Stores the MO space information as the pair ("label",[MO indices])
    static std::vector<std::pair<std::string,std::vector<size_t>>> mo_spaces;
};

#endif // _tensor_blocked_h_

