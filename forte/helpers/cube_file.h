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

#ifndef _cube_file_h_
#define _cube_file_h_

#include <string>
#include <vector>
#include <utility>

namespace forte {

/**
 * @class CubeFile
 *
 * @brief A class for loading, maniputlating, and storing cube files.
 *
 * This class provides some basic functionality to handle cube files. It can read/write
 * cube files and since it uses low-level C++ functions is faster than a Pyhton based
 * implementation. This class also provides some basic operations (adding, scaling, multiplying)
 * on cube files.
 */
class CubeFile {
  public:
    // ==> Class Constructor and Destructor <==
    /**
     * @brief Build a CubeFile object from a file stored on disk
     * @param filename the path to the cube file
     */
    CubeFile(const std::string& filename);
    CubeFile(const CubeFile& cube) = default;

    /// @return the number of atoms
    int natoms() const;
    /// @return the atomic numbers of the atoms
    const std::vector<int>& atom_numbers() const;
    /// @return the (x,y,z) atomic coordinates (in Angstrom)
    const std::vector<std::tuple<double, double, double>>& atom_coords() const;
    /// @return the number of grid points in each direction
    const std::vector<int>& num() const;
    /// @return the minimum value of a grid point
    const std::vector<double>& min() const;
    /// @return the maximum value of a grid point
    const std::vector<double>& max() const;
    /// @return the grid increment
    const std::vector<double>& inc() const;
    /// @return the grid points stored as a vector
    const std::vector<double>& data() const;

    // ==> Class Functions <==
    /// load a cube file
    /// @param filename the cube file name
    void load(std::string filename);
    /// save a cube file
    /// @param filename the cube file name
    void save(std::string filename) const;

    /**
     * @brief Compute the isolevel that encompasses a give fraction of the total density
     * @param type the type of cube file ("mo" or "density")
     * @param fraction the fraction of the density
     */
    std::pair<double, double> compute_levels(std::string type, double fraction) const;

    /// zero this cube file
    void zero();

    /**
     * @brief scale the value of the cube file at each point by a factor
     * @param factor the scaling factor
     */
    void scale(double factor);

    /**
     * @brief add to each grid point the value of another cube file times a scaling factor
     * @param other the cube file to add
     * @param factor the scaling factor
     */
    void add(const CubeFile& other, double factor = 1.0);

    /**
     * @brief multiply each grid point by the value of another cube file
     * @param other the cube file to add
     * @param factor the scaling factor
     */
    void pointwise_product(const CubeFile& other);

  private:
    /// the title
    std::string title_;
    /// a comment
    std::string comments_;
    /// the number of atoms
    int natoms_;
    /// the atomic numbers of the atoms
    std::vector<int> atom_numbers_;
    /// the (x,y,z) atomic coordinates (in Angstrom)
    std::vector<std::tuple<double, double, double>> atom_coords_;
    /// the number of grid points in each direction
    std::vector<int> num_;
    /// the minimum value of a grid point in each direction
    std::vector<double> min_;
    /// the maximum value of a grid point in each direction
    std::vector<double> max_;
    /// the grid increment in each direction
    std::vector<double> inc_;
    /// the grid points stored as a vector
    std::vector<double> data_;
};

} // namespace forte

#endif // _cube_file_h_
