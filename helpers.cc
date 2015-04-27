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

#include "psi4-dec.h"

#include "helpers.h"

namespace psi{ namespace libadaptive{

void print_method_banner(const std::vector<std::string>& text, const std::string &separator)
{
    size_t max_width = 80;

    size_t width = 0;
    for (auto& line : text){
        width = std::max(width,line.size());
    }

    std::string tab((max_width - width - 4)/2,' ');
    std::string header(width + 4,char(separator[0]));

    *outfile << "\n\n" << tab << header << std::endl;
    for (auto& line : text){
        size_t padding = 2 + (width - line.size()) / 2;
        *outfile << tab << std::string(padding,' ') << line << std::endl;
    }
    *outfile << tab << header << std::endl;

    outfile->Flush();
}

}} // End Namespaces
