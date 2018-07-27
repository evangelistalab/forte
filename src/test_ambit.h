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
    enum TestResult { kPass, kFail, kException };

    static const size_t MAXTWO = 1024;
    static const size_t MAXFOUR = 10;
    constexpr static const double zero = 1e-10;
    static double a1[MAXTWO];
    static double a2[MAXTWO][MAXTWO];
    static double b2[MAXTWO][MAXTWO];
    static double c2[MAXTWO][MAXTWO];
    static double d2[MAXTWO][MAXTWO];
    static double e2[MAXTWO][MAXTWO];
    static double a4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];
    static double b4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];
    static double c4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];
    static double d4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];
    static double e4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];

    static Tensor build_and_fill(const std::string& name, const ambit::Dimension& dims,
                          double matrix[MAXTWO]);

    static Tensor build_and_fill(const std::string& name, const ambit::Dimension& dims,
                          double matrix[MAXTWO][MAXTWO]);

    static Tensor build_and_fill(const std::string& name, const ambit::Dimension& dims,
                          double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR]);

    static void initialize_random(Tensor& tensor, double matrix[MAXTWO]);
    static std::pair<double, double> difference(Tensor& tensor, double matrix[MAXTWO]);

    static void initialize_random(Tensor& tensor, double matrix[MAXTWO][MAXTWO]);
    static std::pair<double, double> difference(Tensor& tensor, double matrix[MAXTWO][MAXTWO]);

    static void initialize_random(Tensor& tensor, double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR]);
    static std::pair<double, double> difference(Tensor& tensor,
                                         double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR]);
    //    double test_Cij_equal_Aik_Bkj(size_t ni, size_t nj, size_t nk);
    static double test_C_equal_A_B(std::string c_ind, std::string a_ind, std::string b_ind,
                            std::vector<int> c_dim, std::vector<int> a_dim, std::vector<int> b_dim);

    static double test_wrapper();

    static double test_Cij_plus_equal_Aik_Bkj();

    static double test_Cij_minus_equal_Aik_Bkj();

    static double test_Cij_equal_Aik_Bkj();

    static double test_Cij_equal_Aik_Bjk();

    static double test_Cijkl_equal_Aijab_Bklab();

    static double test_Cikjl_equal_Aijab_Bklab();

    static double test_Cij_equal_Aiabc_Bjabc();

    static double test_Cij_minus_equal_Aij_Bij();

    static double test_Dij_plus_equal_Aij_Bij_Cij();

    static double test_E_abcd_equal_Aijab_Bklcd_C_jl_D_ik();

    static double test_C_equal_2_A();

    static double test_C_plus_equal_2_A();

    static double test_C_minus_equal_2_A();

    static double test_C_times_equal_2();

    static double test_C_divide_equal_2();

    static double test_Cij_equal_Aji();

    static double test_Cijkl_equal_Akijl();

    static double test_Cijkl_equal_Akilj();

    static double test_Cij_equal_Cij();

    static double test_syev();

    static double test_geev();

    static double test_Cilkj_equal_Aibaj_Bblak();

    static double test_Cljik_equal_Abija_Blbak();

    static double test_Cij_equal_Aij_plus_Bij();

    static double test_Dij_equal_Aij_plus_Bij_plus_Cij();

    static double test_Cij_equal_Aij_minus_Bij();

    static double test_Dij_equal_Aij_minus_Bij_plus_Cij();

    static double test_Dij_equal_Aij_times_Bij_plus_Cij();

    static double test_Dij_equal_Bij_plus_Cij_times_Aij();

    static double test_F_equal_D_times_2g_minus_g();

    static double test_Dij_equal_2_times_Aij_plus_Bij();

    static double test_Dij_equal_negate_Aij_plus_Bij();

    static double test_power();

    static double test_dot_product();

    static double test_dot_product2();

    static double test_dot_product3();

    static double test_dot_product4();

    static double test_dot_product5();

    static double test_dot_product6();

    static double test_chain_multiply();

    static double test_chain_multiply2();

    static double test_chain_multiply3();

    static double test_chain_multiply4();
    static double test_slice2();

    static double test_Cijkl_equal_Aijab_Bklab_batched();

    static double test_chain_multiply2_batched();

    static double test_chain_multiply3_batched();

    static double test_chain_multiply4_batched();

    static double test_batched();

    static double test_batched_with_factor();

    static double test_batched_with_factor_permute();
};
}
}
#endif // _test_ambit_h_
