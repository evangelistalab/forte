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

#ifndef _cube_file_h_
#define _cube_file_h_

#include <string>
#include <vector>

namespace forte {

class CubeFile {
  public:
    CubeFile(const std::string& filename);

    const std::vector<double>& data() const;

  private:
    void load(std::string filename);

    std::string filename_;
    std::string title_;
    std::string comments_;
    std::vector<double> levels_;

    std::vector<double> num_;
    std::vector<double> min_;
    std::vector<double> max_;
    std::vector<double> inc_;
    int natoms_;
    std::vector<double> atom_numbers_;
    std::vector<std::tuple<double, double, double>> atom_coords_;
    std::vector<double> data_;
};

} // namespace forte

#endif // _cube_file_h_
