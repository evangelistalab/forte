/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _helpers_h_
#define _helpers_h_

#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>

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

/**
 * @brief tensor_to_matrix
 * @param t The input tensor
 * @param dims Dimensions of the matrix extracted from the tensor
 * @return A copy of the tensor data in symmetry blocked form
 */
Matrix tensor_to_matrix(ambit::Tensor t, Dimension dims);

SharedMatrix tensor_to_matrix(ambit::Tensor t);

/// Save a vector of double to file
void write_disk_vector_double(const std::string& filename, const std::vector<double>& data);

/// Read a vector of double from file
void read_disk_vector_double(const std::string& filename, std::vector<double>& data);

/**
 * @brief view_modified_orbitals Write orbitals using molden
 * @param Ca  The Ca matrix to be viewed with MOLDEN
 * @param diag_F -> The Orbital energies (diagonal elements of Fock operator)
 * @param occupation -> occupation vector
 */
void view_modified_orbitals(SharedWavefunction wfn, const std::shared_ptr<Matrix>& Ca,
                            const std::shared_ptr<Vector>& diag_F,
                            const std::shared_ptr<Vector>& occupation);

/**
 * @brief print_h2 Print a header
 * @param text The string to print in the header.
 * @param left_separator The left separator (default = "==>")
 * @param right_separator The right separator (default = "<==")
 */
void print_h2(const std::string& text, const std::string& left_separator = "==>",
              const std::string& right_separator = "<==");

/**
 * Returns the Ms as a string, using
 * fractions if needed
 */
std::string get_ms_string(double twice_ms);

std::string to_string(const std::vector<std::string>& vec_str, const std::string& sep = ",");

/**
 * @brief Compute the memory (in GB) required to store arrays
 * @typename T The data typename
 * @param num_el The number of elements to store
 * @return The size in GB
 */
template <typename T> double to_gb(T num_el) {
    return static_cast<double>(num_el) * static_cast<double>(sizeof(T)) / 1073741824.0;
}

/**
 * @brief Compute the memory requirement
 * @param nele The number of elements for storage
 * @param type_size The size of the data type
 * @return A pair of size in appropriate unit (B, KB, MB, GB, TB, PB)
 */
std::pair<double, std::string> to_xb(size_t nele, size_t type_size);

/**
 * @brief split up a vector into different processors
 * @param size_t size_of_tasks
 * @param nproc (the global number of processors)
 * @return a pair of vectors -> pair.0 -> start for each processor
 *                           -> pair.1 -> end or each processor
 */
std::pair<std::vector<int>, std::vector<int>> split_up_tasks(size_t size_of_tasks, int nproc);

template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(
    const std::vector<T>& vec,
    Compare& compare)
{
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
        [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
    return p;
}

template <typename T>
void apply_permutation_in_place(
    std::vector<T>& vec,
    const std::vector<std::size_t>& p)
{
    std::vector<bool> done(vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i)
    {
        if (done[i])
        {
            continue;
        }
        done[i] = true;
        std::size_t prev_j = i;
        std::size_t j = p[i];
        while (i != j)
        {
            std::swap(vec[prev_j], vec[j]);
            done[j] = true;
            prev_j = j;
            j = p[j];
        }
    }
}

/**
  * @brief A timer class
  */
class timer {
  public:
    timer(const std::string& name) : name_(name) { timer_on(name_); }
    ~timer() { stop(); }

    /// Return the elapsed time in seconds
    void stop() {
        if (running_) {
            running_ = false;
            timer_off(name_);
        }
    }

  private:
    std::string name_;
    bool running_ = true;
};

/**
  * @brief A timer class
  */
class parallel_timer {
  public:
    parallel_timer(const std::string& name, int rank) : name_(name), rank_(rank) {
        parallel_timer_on(name_, rank_);
    }
    ~parallel_timer() { stop(); }

    /// Return the elapsed time in seconds
    void stop() {
        if (running_) {
            running_ = false;
            parallel_timer_off(name_, rank_);
        }
    }

  private:
    std::string name_;
    int rank_;
    bool running_ = true;
};
}
} // End Namespaces

#endif // _helpers_h_
