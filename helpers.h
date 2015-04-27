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

#include <vector>
#include <string>

namespace psi{ namespace libadaptive{

void print_method_banner(const std::vector<std::string>& text, const std::string& separator = "-");

template <typename T>
double to_gb(T num_el){
    return static_cast<double>(num_el) * static_cast<double>(sizeof(T)) / 1073741824.0;
}

}} // End Namespaces

#endif // _helpers_h_
