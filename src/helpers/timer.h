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

#ifndef _helpers_timer_h_
#define _helpers_timer_h_

#include <chrono>

namespace psi {
namespace forte {

/**
 * @brief A timer class that returns the elapsed time
 */
class local_timer {
  public:
    local_timer() : start_(std::chrono::high_resolution_clock::now()) {}

    /// return the elapsed time in seconds
    double get() {
        auto duration = std::chrono::high_resolution_clock::now() - start_;
        return std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
    }

  private:
    /// stores the time when this object is created
    std::chrono::high_resolution_clock::time_point start_;
};
}
} // End Namespaces

#endif // _helpers_timer_h_
