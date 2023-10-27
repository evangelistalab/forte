/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _cpscf_h_
#define _cpscf_h_

#include "psi4/libfock/jk.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"

#include "base_classes/forte_options.h"

namespace forte {

class CPSCF {
  public:
    /**
     * @brief Constructor of CPSCF_FUNCTOR
     * @param JK: The pointer to a Psi4 JK object
     * @param C: The Hartree-Fock orbital coefficients
     * @param b: The R.H.S. of CPSCF equation Ax = b (nuocc x ndocc)
     * @param edocc: The occupied orbital energies
     * @param euocc: The unoccupied orbital energies
     */
    CPSCF(std::shared_ptr<psi::JK> JK, std::shared_ptr<psi::Matrix> C,
          std::shared_ptr<psi::Matrix> b, std::shared_ptr<psi::Vector> edocc,
          std::shared_ptr<psi::Vector> euocc);

    /// Evaluate the scalar and gradient
    double evaluate(std::shared_ptr<psi::Vector> x, std::shared_ptr<psi::Vector> g,
                    bool do_g = true);

    /// Evaluate the diagonal Hessian
    void hess_diag(std::shared_ptr<psi::Vector> x, std::shared_ptr<psi::Vector> h0);

    /// Return the vector dimension
    psi::Dimension vdimpi() { return vdims_; }

    /// Reshape matrix to vector
    std::shared_ptr<psi::Vector> mat_to_vec(std::shared_ptr<psi::Matrix> M);
    /// Reshape vector to matrix
    std::shared_ptr<psi::Matrix> vec_to_mat(std::shared_ptr<psi::Vector> v);

  private:
    /// The JK object of Psi4
    std::shared_ptr<psi::JK> JK_;

    /// The MO coefficients
    std::shared_ptr<psi::Matrix> C_;
    /// The DOCC part of MO coefficients
    std::shared_ptr<psi::Matrix> Cdocc_;
    /// The UOCC part of MO coefficients
    std::shared_ptr<psi::Matrix> Cuocc_;

    /// The b vector
    std::shared_ptr<psi::Vector> b_;

    /// Canonial orbital energies for occupied orbitals
    std::shared_ptr<psi::Vector> edocc_;
    /// Canonial orbital energies for unoccupied orbitals
    std::shared_ptr<psi::Vector> euocc_;

    /// The number of irrep
    int nirrep_;
    /// The number of DOCC per irrep
    psi::Dimension ndoccpi_;
    /// The number of UOCC per irrep
    psi::Dimension nuoccpi_;

    /// The dimension of vector
    psi::Dimension vdims_;
    /// Test Matrix dimension
    void test_mat_dim(std::shared_ptr<psi::Matrix> M);
    /// Test Vector dimension
    void test_vec_dim(std::shared_ptr<psi::Vector> v);
};

class CPSCF_SOLVER {
  public:
    /**
     * @brief Constructor of CPSCF_SOLVER
     * @param options: The ForteOptions pointer
     * @param JK: The pointer to a Psi4 JK object
     * @param C: The Hartree-Fock orbital coefficients
     * @param b: The R.H.S. of CPSCF equation Ax = b (nuocc x ndocc)
     * @param edocc: The occupied orbital energies
     * @param euocc: The unoccupied orbital energies
     */
    CPSCF_SOLVER(std::shared_ptr<ForteOptions> options, std::shared_ptr<psi::JK> JK,
                 std::shared_ptr<psi::Matrix> C, std::shared_ptr<psi::Matrix> b,
                 std::shared_ptr<psi::Vector> edocc, std::shared_ptr<psi::Vector> euocc);

    /// Solve CPSCF equation and return if the equations are converged or not
    bool solve();

    /// Return the unknown in matrix form
    std::shared_ptr<psi::Matrix> x();

  private:
    /// The unknown in vector form
    std::shared_ptr<psi::Vector> x_;

    /// The Forte options
    std::shared_ptr<ForteOptions> options_;

    /// The CPSCF object
    CPSCF cpscf_;
};
} // namespace forte

#endif // _cpscf_h_
