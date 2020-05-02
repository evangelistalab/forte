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

#ifndef _dsrg_time_h_
#define _dsrg_time_h_

#include <vector>
#include <string>
#include <map>

#include "psi4/libtrans/integraltransform.h"

namespace forte {

class DSRG_TIME {
  public:
    /// Constructor
    DSRG_TIME();

    /// Accumulate timings
    void add(const std::string& code, const double& t);

    /// Subtract timings
    void subtract(const std::string& code, const double& t);

    /// Reset timings
    void reset();                        // reset all timings to zero
    void reset(const std::string& code); // reset timing of the code

    /// Create info of a code
    void create_code(const std::string& code);

    /// Delete info of a code
    void delete_code(const std::string& code);

    /// Print summary for with default code
    void print_comm_time();

    /// Print the timing in a generic way
    void print();
    void print(const std::string& code);

    /// Clear all the private variables
    void clear() {
        code_.clear();
        code_to_tidx_.clear();
        timing_.clear();
    }

  private:
    /**
     * commutator: [ H, T ] = C
     * time code: 1st digit: rank of H
     *            2nd digit: rank of T
     *            3rd digit: rank of C
     */
    std::vector<std::string> code_;

    /// Map from code to values
    std::map<std::string, int> code_to_tidx_;

    /// Timings for commutators
    std::vector<double> timing_;

    /// Test code
    bool test_code(const std::string& code);
};
}

#endif // DSRG_TIME_H
