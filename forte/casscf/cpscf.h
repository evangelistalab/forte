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
    CPSCF(std::shared_ptr<psi::JK> JK, psi::SharedMatrix C, psi::SharedMatrix b,
          psi::SharedVector edocc, psi::SharedVector euocc);

    /// Evaluate the scalar and gradient
    double evaluate(psi::SharedVector x, psi::SharedVector g, bool do_g = true);

    /// Evaluate the diagonal Hessian
    void hess_diag(psi::SharedVector x, psi::SharedVector h0);

    /// Return the vector dimension
    psi::Dimension vdimpi() { return vdims_; }

    /// Reshape matrix to vector
    psi::SharedVector mat_to_vec(psi::SharedMatrix M);
    /// Reshape vector to matrix
    psi::SharedMatrix vec_to_mat(psi::SharedVector v);

  private:
    /// The JK object of Psi4
    std::shared_ptr<psi::JK> JK_;

    /// The MO coefficients
    psi::SharedMatrix C_;
    /// The DOCC part of MO coefficients
    psi::SharedMatrix Cdocc_;
    /// The UOCC part of MO coefficients
    psi::SharedMatrix Cuocc_;

    /// The b vector
    psi::SharedVector b_;

    /// Canonial orbital energies for occupied orbitals
    psi::SharedVector edocc_;
    /// Canonial orbital energies for unoccupied orbitals
    psi::SharedVector euocc_;

    /// The number of irrep
    int nirrep_;
    /// The number of DOCC per irrep
    psi::Dimension ndoccpi_;
    /// The number of UOCC per irrep
    psi::Dimension nuoccpi_;

    /// The dimension of vector
    psi::Dimension vdims_;
    /// Test Matrix dimension
    void test_mat_dim(psi::SharedMatrix M);
    /// Test Vector dimension
    void test_vec_dim(psi::SharedVector v);
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
                 psi::SharedMatrix C, psi::SharedMatrix b, psi::SharedVector edocc,
                 psi::SharedVector euocc);

    /// Solve CPSCF equation and return if the equations are converged or not
    bool solve();

    /// Return the unknown in matrix form
    psi::SharedMatrix x();

  private:
    /// The unknown in vector form
    psi::SharedVector x_;

    /// The Forte options
    std::shared_ptr<ForteOptions> options_;

    /// The CPSCF object
    CPSCF cpscf_;
};
} // namespace forte

#endif // _cpscf_h_
