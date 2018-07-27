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

#ifndef _test_ambit_h_
#define _test_ambit_h_

#include <cmath>
#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/wavefunction.h"
#include "ambit/blocked_tensor.h"

using namespace ambit;
namespace psi {
namespace forte {

class AMBIT_TEST {
  public:
    /**
     * AMBIT_TEST Constructor
     */
    AMBIT_TEST();

    /// Destructor
    ~AMBIT_TEST();

    /// Compute the ambit tests
    double compute_energy();

  private:
    static const size_t MAXTWO = 1024;
    static const size_t MAXFOUR = 10;
    constexpr static const double zero = 1e-10;
    double a1[MAXTWO];
    double a2[MAXTWO][MAXTWO];
    double b2[MAXTWO][MAXTWO];
    double c2[MAXTWO][MAXTWO];
    double d2[MAXTWO][MAXTWO];
    double e2[MAXTWO][MAXTWO];
    double a4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];
    double b4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];
    double c4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];
    double d4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];
    double e4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];

    Tensor build_and_fill(const std::string& name, const ambit::Dimension& dims, double matrix[MAXTWO]);

    Tensor build_and_fill(const std::string& name, const ambit::Dimension& dims,
                          double matrix[MAXTWO][MAXTWO]);

    Tensor build_and_fill(const std::string& name, const ambit::Dimension& dims,
                          double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR]);

    void initialize_random(Tensor& tensor, double matrix[MAXTWO]);
    std::pair<double, double> difference(Tensor& tensor, double matrix[MAXTWO]);

    void initialize_random(Tensor& tensor, double matrix[MAXTWO][MAXTWO]);
    std::pair<double, double> difference(Tensor& tensor, double matrix[MAXTWO][MAXTWO]);

    void initialize_random(Tensor& tensor, double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR]);
    std::pair<double, double> difference(Tensor& tensor,
                                         double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR]);
    double test_Cij_equal_Aik_Bkj(size_t ni, size_t nj, size_t nk);
};
}
}
#endif // _test_ambit_h_
